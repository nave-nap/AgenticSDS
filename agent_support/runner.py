"""Run baseline episodes with one model, manifest, and durable per-task logging."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .benchmarks import make_episode
from .common import canonical_digest, read_json, validate_config, validate_manifest, write_json_atomic
from .model import QwenModel
from .policy import alfworld_messages, parse_alfworld, parse_textcraft, textcraft_messages
from .memory import ActionMemory, SentenceEmbedder, support_messages


def package_versions() -> dict:
    names = ("torch", "transformers", "accelerate", "bitsandbytes", "alfworld", "textworld", "textcraft")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_episode(benchmark: str, task: dict, config: dict, model, episode_factory=make_episode,
                support: ActionMemory | None = None) -> dict:
    episode = episode_factory(benchmark, task, config)
    started = time.perf_counter()
    history: list[dict] = []
    trace: list[dict] = []
    status = "max_decisions"
    actions_taken = 0
    parser_errors = 0
    failed_actions = 0
    input_tokens = output_tokens = 0
    inference_seconds = 0.0
    support_calls = support_model_calls = support_input_tokens = support_output_tokens = support_parse_errors = 0
    retrieval_seconds = 0.0
    success = False
    try:
        episode.reset()
        for decision in range(1, config["max_decisions"] + 1):
            if actions_taken >= config["max_actions"]:
                status = "max_actions"
                break
            if benchmark == "alfworld":
                if not episode.actions:
                    status = "no_actions"
                    break
                available = list(episode.actions)
                messages = alfworld_messages(episode.task_prompt, episode.observation, available, history, config["history_window"])
            else:
                available = None
                messages = textcraft_messages(episode.initial_observation, history, config["history_window"])
            generation = model.generate(messages)
            input_tokens += generation["input_tokens"]
            output_tokens += generation["output_tokens"]
            inference_seconds += generation["inference_seconds"]
            raw = generation["text"]
            action = parse_alfworld(raw, available) if benchmark == "alfworld" else parse_textcraft(raw)
            record = {
                "decision": decision,
                "input_observation": episode.observation,
                "available_actions": available,
                "raw_model_output": raw,
                "action": action,
                "input_tokens": generation["input_tokens"],
                "output_tokens": generation["output_tokens"],
                "inference_seconds": generation["inference_seconds"],
            }
            if action is not None and support is not None:
                proposed_action = action
                hits, retrieval_time = support.retrieve(benchmark, episode.task_prompt, episode.observation, action)
                support_calls += 1
                retrieval_seconds += retrieval_time
                record["proposed_action"] = proposed_action
                record["proposal_raw_model_output"] = raw
                record["support"] = {
                    "mechanism": "m2_action_memory", "retrieval_seconds": retrieval_time,
                    "retrieved": [{"memory_id": hit["memory_id"], "similarity": hit["similarity"],
                                   "content": {key: hit[key] for key in ("task_prompt", "observation", "action",
                                                                            "feedback", "failed_action", "episode_success")}}
                                  for hit in hits],
                }
                if hits:
                    revision = model.generate(support_messages(messages, proposed_action, hits, benchmark))
                    support_model_calls += 1
                    support_input_tokens += revision["input_tokens"]
                    support_output_tokens += revision["output_tokens"]
                    input_tokens += revision["input_tokens"]
                    output_tokens += revision["output_tokens"]
                    inference_seconds += revision["inference_seconds"]
                    record["input_tokens"] += revision["input_tokens"]
                    record["output_tokens"] += revision["output_tokens"]
                    record["inference_seconds"] = round(record["inference_seconds"] + revision["inference_seconds"], 3)
                    record["support"]["raw_model_output"] = revision["text"]
                    record["support"]["input_tokens"] = revision["input_tokens"]
                    record["support"]["output_tokens"] = revision["output_tokens"]
                    record["support"]["inference_seconds"] = revision["inference_seconds"]
                    revised = (parse_alfworld(revision["text"], available) if benchmark == "alfworld"
                               else parse_textcraft(revision["text"]))
                    if revised is not None:
                        action = revised
                        record["action"] = action
                        record["raw_model_output"] = revision["text"]
                    else:
                        record["support"]["parse_error"] = True
                        support_parse_errors += 1
                    record["support"]["action_changed"] = action != proposed_action
            if action is None:
                parser_errors += 1
                record["error"] = "parse_error"
                record["output_observation"] = "Invalid action format. Reply with exactly one valid command."
                trace.append(record)
                if benchmark == "alfworld":
                    status = "parse_error"
                    break
                history.append({"action": "<PARSE_ERROR>", "observation": record["output_observation"]})
                continue
            try:
                step = episode.step(action)
            except Exception as exc:
                record["error"] = "environment_error"
                record["details"] = repr(exc)
                trace.append(record)
                status = "environment_error"
                break
            actions_taken += 1
            failed_actions += int(step.get("failed_action", False))
            record.update({
                "environment_action_number": actions_taken,
                "output_observation": step["observation"],
                "outcome": step,
            })
            trace.append(record)
            history.append({"action": action, "observation": step["observation"]})
            if step["done"]:
                success = bool(step["success"])
                status = "success" if success else "task_failed"
                break
            if benchmark == "alfworld" and len(history) >= 6:
                recent_actions = [item["action"] for item in history[-6:]]
                if recent_actions[:2] == recent_actions[2:4] == recent_actions[4:6]:
                    status = "repeated_action_loop"
                    break
        else:
            if actions_taken >= config["max_actions"]:
                status = "max_actions"
    finally:
        episode.close()
    result = {
        "task_id": task["task_id"],
        "benchmark": benchmark,
        "initial_observation": episode.initial_observation,
        "task_prompt": episode.task_prompt,
        "success": success,
        "status": status,
        "environment_actions": actions_taken,
        "model_decisions": len(trace),
        "parser_errors": parser_errors,
        "failed_actions": failed_actions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "inference_seconds": round(inference_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "trace": trace,
    }
    if support is not None:
        result.update({"model_calls": len(trace) + support_model_calls,
                       "support_calls": support_calls, "support_model_calls": support_model_calls,
                       "support_input_tokens": support_input_tokens, "support_output_tokens": support_output_tokens,
                       "support_parse_errors": support_parse_errors,
                       "retrieval_seconds": round(retrieval_seconds, 3),
                       "actions_changed_by_support": sum(bool(item.get("support", {}).get("action_changed")) for item in trace)})
    return result


def summarize(episodes: list[dict]) -> dict:
    count = len(episodes)
    summary = {
        "tasks": count,
        "successes": sum(int(item["success"]) for item in episodes),
        "success_rate": sum(int(item["success"]) for item in episodes) / count if count else None,
        "environment_actions": sum(item["environment_actions"] for item in episodes),
        "model_decisions": sum(item["model_decisions"] for item in episodes),
        "parser_errors": sum(item["parser_errors"] for item in episodes),
        "failed_actions": sum(item["failed_actions"] for item in episodes),
        "input_tokens": sum(item["input_tokens"] for item in episodes),
        "output_tokens": sum(item["output_tokens"] for item in episodes),
        "inference_seconds": round(sum(item["inference_seconds"] for item in episodes), 3),
    }
    if episodes and "support_calls" in episodes[0]:
        for key in ("model_calls", "support_calls", "support_model_calls", "support_input_tokens",
                    "support_output_tokens", "support_parse_errors", "retrieval_seconds", "actions_changed_by_support"):
            summary[key] = round(sum(item[key] for item in episodes), 3) if key == "retrieval_seconds" else sum(item[key] for item in episodes)
    return summary


def run(config: dict, manifest: dict, output_root: str | Path, limit: int | None = None, model=None, episode_factory=make_episode) -> Path:
    validate_manifest(manifest)
    validate_config(config, manifest["benchmark"])
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected = manifest["tasks"][:limit] if limit is not None else manifest["tasks"]
    corpus = index_data = None
    if config["condition"] == "m2_always":
        corpus = read_json(config["memory_corpus"])
        index_data = read_json(config["memory_index"])
        if canonical_digest(manifest) not in corpus.get("excluded_manifest_sha256", {}).values():
            raise ValueError("Memory corpus was not built with this evaluation manifest excluded")
        selected_ids = {task["task_id"] for task in selected}
        if any(record["benchmark"] == manifest["benchmark"] and record["task_id"] in selected_ids
               for record in corpus["records"]):
            raise ValueError("Memory corpus contains evaluation task trajectories")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = Path(output_root) / f"{manifest['benchmark']}_{run_id}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "episodes").mkdir()
    selected_manifest = dict(manifest)
    selected_manifest["tasks"] = selected
    write_json_atomic(directory / "config.json", config)
    write_json_atomic(directory / "manifest.json", selected_manifest)
    alfworld_config_sha256 = None
    textcraft_recipe_order_sha256 = None
    if manifest["benchmark"] == "alfworld":
        source = Path(config["alfworld_config"])
        config_bytes = source.read_bytes()
        alfworld_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        (directory / "alfworld_config.yaml").write_bytes(config_bytes)
    else:
        source = Path(__file__).with_name("textcraft_recipe_order.json")
        order_bytes = source.read_bytes()
        textcraft_recipe_order_sha256 = hashlib.sha256(order_bytes).hexdigest()
        (directory / "textcraft_recipe_order.json").write_bytes(order_bytes)
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "benchmark": manifest["benchmark"],
        "condition": config["condition"],
        "partition": manifest.get("partition"),
        "manifest_sha256": canonical_digest(manifest),
        "selected_manifest_sha256": canonical_digest(selected_manifest),
        "selected_task_count": len(selected),
        "total_manifest_task_count": len(manifest["tasks"]),
        "limit": limit,
        "config_sha256": canonical_digest(config),
        "alfworld_config_sha256": alfworld_config_sha256,
        "textcraft_recipe_order_sha256": textcraft_recipe_order_sha256,
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if config["condition"] == "m2_always":
        metadata["memory_corpus_sha256"] = canonical_digest(corpus)
        metadata["memory_index_sha256"] = canonical_digest(index_data)
    write_json_atomic(directory / "run.json", metadata)
    episodes = []
    try:
        actual_model = model if model is not None else QwenModel(config)
        support = (ActionMemory(corpus, index_data,
                                SentenceEmbedder(index_data["embedding_model"],
                                                 revision=index_data.get("embedding_revision")),
                                config["memory_top_k"])
                   if config["condition"] == "m2_always" else None)
        for index, task in enumerate(selected):
            result = run_episode(manifest["benchmark"], task, config, actual_model, episode_factory, support=support)
            filename = f"{index:04d}_{canonical_digest(task['task_id'])[:12]}.json"
            write_json_atomic(directory / "episodes" / filename, result)
            episodes.append(result)
            write_json_atomic(directory / "summary.json", summarize(episodes))
            print(f"{index + 1}/{len(selected)} {task['task_id']} {result['status']} actions={result['environment_actions']}", flush=True)
        metadata["status"] = "complete"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        raise
    finally:
        metadata["completed_task_count"] = len(episodes)
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(directory / "run.json", metadata)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    manifest = read_json(args.manifest)
    output = run(config, manifest, args.output_root, args.limit)
    print(f"Run complete: {output}")


if __name__ == "__main__":
    main()
