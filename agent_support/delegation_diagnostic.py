"""Build, predict, and replay a development-only TextCraft delegation diagnostic.

The prediction path reads inputs.jsonl only. Labels and replay prefixes are kept
in separate files and are never passed to a model adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from .benchmarks import TextCraftEpisode
from .common import canonical_digest, read_json, write_json_new
from .components import load_components
from .model import make_model
from .policy import PROMPT_VERSIONS, parse_textcraft, textcraft_messages
from .run_store import episode_filename, git_state


SOURCE_PREFIX = "runs/never/"
SOURCE_COMMIT = "2cc1a41b6d55288f3fdb50fc06e92d6ac52b2d75"
CATEGORY_COUNTS = {
    "unproductive_or_irrelevant": 8,
    "repeated_failed_commands": 8,
    "missing_prerequisites": 8,
    "unavailable_items": 8,
    "productive_controls": 8,
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _new_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                            for row in rows), encoding="utf-8")


def _verify_dataset_files(inputs_path: Path, labels_path: Path | None = None,
                          replay_path: Path | None = None) -> dict:
    information = read_json(inputs_path.parent / "dataset.json")
    if information.get("partition") != "development" or information.get("schema_version") != 1:
        raise ValueError("Expected a development diagnostic dataset")
    for path in (inputs_path, labels_path, replay_path):
        if path is None:
            continue
        if path.parent != inputs_path.parent:
            raise ValueError("Diagnostic input, label, and replay files must share a dataset directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != information["files_sha256"].get(path.name):
            raise ValueError(f"Diagnostic dataset hash mismatch: {path}")
    return information


def _recipes_and_goal(initial: str) -> tuple[list[str], str, set[str]]:
    prefix = "Crafting commands:\n"
    separator = "\n\nGoal: craft "
    if not initial.startswith(prefix) or separator not in initial:
        raise ValueError("Unrecognized TextCraft prompt")
    command_text, goal = initial[len(prefix):].rsplit(separator, 1)
    if not goal.endswith("."):
        raise ValueError("Unrecognized TextCraft goal")
    goal = goal[:-1]
    recipes = command_text.splitlines()
    graph: dict[str, list[list[str]]] = defaultdict(list)
    for command in recipes:
        match = re.fullmatch(r"craft [1-9]\d* (.+?) using (.+)", command)
        if match:
            ingredients = [re.sub(r"^[1-9]\d* ", "", part)
                           for part in match.group(2).split(", ")]
            graph[match.group(1)].append(ingredients)
    relevant: set[str] = set()
    pending = [goal]
    while pending:
        item = pending.pop()
        if item in relevant:
            continue
        relevant.add(item)
        if item in graph:
            for ingredients in graph[item]:
                pending.extend(ingredients)
        else:
            pending.extend(product for product in graph if product.endswith(" " + item))
    return recipes, goal, relevant


def _category(step: dict, prior: list[dict], recipes: list[str], relevant: set[str]) -> str | None:
    action = step.get("action") or ""
    feedback = step.get("output_observation", "")
    failed = step.get("outcome", {}).get("failed_action")
    if prior and action == prior[-1]["action"] and prior[-1].get("outcome", {}).get("failed_action"):
        return "repeated_failed_commands"
    if action.startswith("craft ") and feedback.startswith("Could not find enough items"):
        return "missing_prerequisites"
    if action.startswith("get ") and feedback.startswith("Could not find"):
        return "unavailable_items"
    get_item = re.sub(r"^get [1-9]\d* ", "", action) if action.startswith("get ") else None
    if action.startswith("craft ") and action not in recipes and failed:
        return "unproductive_or_irrelevant"
    crafted = re.fullmatch(r"craft [1-9]\d* (.+?) using .+", action)
    if ((feedback.startswith("Got ") and get_item in relevant)
            or (feedback.startswith("Crafted ") and crafted and crafted.group(1) in relevant)):
        return "productive_controls"
    return None


def _select(candidates: dict[str, list[dict]]) -> list[dict]:
    selected: list[dict] = []
    task_counts: Counter[str] = Counter()
    chosen: set[str] = set()
    for category, target in CATEGORY_COUNTS.items():
        pool = candidates[category]
        groups = (("get", 4), ("craft", 4)) if category == "productive_controls" else ((None, target),)
        for action_kind, group_target in groups:
            count = 0
            while count < group_target:
                available = [item for item in pool if item["sample_id"] not in chosen
                             and task_counts[item["task_id"]] < 3
                             and (action_kind is None or item["label"]["qwen_action"].startswith(action_kind + " "))]
                if not available:
                    raise ValueError(f"Only {count} candidates for {category}/{action_kind}")
                # Prefer distinct tasks, then decisions with more visible failed actions.
                available.sort(key=lambda item: (task_counts[item["task_id"]],
                           -item["prior_failed_count"], item["task_id"], item["decision"]))
                item = available[0]
                selected.append(item)
                chosen.add(item["sample_id"])
                task_counts[item["task_id"]] += 1
                count += 1
    return sorted(selected, key=lambda item: (item["task_index"], item["decision"]))


def build(archive: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    policy_path = Path(__file__).with_name("policy.py")
    source_policy = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:agent_support/policy.py"],
        cwd=policy_path.parent.parent, capture_output=True, check=True).stdout
    if policy_path.read_bytes() != source_policy:
        raise ValueError("TextCraft prompt/parser code differs from the source baseline commit")
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Source archive failed ZIP integrity check")
        def load(name: str) -> dict:
            return json.loads(bundle.read(SOURCE_PREFIX + name))
        metadata = load("run.json")
        config = load("config.json")
        manifest = load("manifest.json")
        checkpoint = load("checkpoint.json")
        summary = load("summary.json")
        if (metadata.get("status") != "complete" or metadata.get("git_commit") != SOURCE_COMMIT
                or metadata.get("git_dirty") or metadata.get("partition") != "development"
                or manifest.get("partition") != "development" or len(manifest.get("tasks", [])) != 20
                or config.get("selection_policy") != {"id": "never_v1"}
                or config.get("model_type") != "qwen"
                or metadata.get("prompt_versions", {}).get("textcraft") != PROMPT_VERSIONS["textcraft"]
                or metadata.get("config_sha256") != canonical_digest(config)
                or metadata.get("selected_manifest_sha256") != canonical_digest(manifest)
                or metadata.get("completed_task_count") != 20
                or summary.get("successes") != 4 or summary.get("model_calls") != 545
                or summary.get("failed_actions") != 460
                or summary.get("input_tokens", 0) + summary.get("output_tokens", 0) != 391407):
            raise ValueError("Archive is not the complete, pinned Experiment 1 Qwen baseline")
        expected_files = {episode_filename(i, task) for i, task in enumerate(manifest["tasks"])}
        if set(checkpoint["episodes"]) != expected_files:
            raise ValueError("Baseline checkpoint does not cover the ordered manifest")
        candidates: dict[str, list[dict]] = defaultdict(list)
        for task_index, task in enumerate(manifest["tasks"]):
            filename = episode_filename(task_index, task)
            episode = load("episodes/" + filename)
            if (canonical_digest(episode) != checkpoint["episodes"][filename]
                    or episode.get("task_id") != task["task_id"]
                    or episode.get("task_sha256") != canonical_digest(task)
                    or episode.get("config_sha256") != metadata["config_sha256"]
                    or episode.get("initial_observation") != task["initial_observation"]
                    or episode.get("parser_errors") != 0
                    or episode.get("support_calls") != 0):
                raise ValueError(f"Baseline episode is incompatible: {filename}")
            recipes, goal, relevant = _recipes_and_goal(episode["initial_observation"])
            history: list[dict] = []
            for decision, step in enumerate(episode["trace"], 1):
                if (step.get("decision") != decision or step.get("action") != parse_textcraft(step["raw_model_output"])
                        or step.get("input_observation") != (history[-1]["observation"] if history else episode["initial_observation"])
                        or step.get("output_observation") != step.get("outcome", {}).get("observation")):
                    raise ValueError(f"Trace continuity or parser mismatch: {filename} decision {decision}")
                category = _category(step, episode["trace"][:decision - 1], recipes, relevant)
                if category:
                    sample_id = hashlib.sha256(f"{task['task_id']}\0{decision}".encode()).hexdigest()[:16]
                    visible = history[-config["history_window"]:]
                    messages = textcraft_messages(episode["initial_observation"], history,
                                                  config["history_window"])
                    candidate = {
                        "sample_id": sample_id, "task_id": task["task_id"], "task_index": task_index,
                        "decision": decision, "category": category,
                        "prior_failed_count": sum(bool(item["failed_action"]) for item in visible),
                        "input": {"sample_id": sample_id, "task_id": task["task_id"],
                                  "decision": decision, "task_goal": goal,
                                  "current_observation": step["input_observation"],
                                  "visible_history": [{"action": item["action"],
                                                       "observation": item["observation"]}
                                                      for item in visible],
                                  "messages": messages},
                        "label": {"sample_id": sample_id, "category": category,
                                  "qwen_action": step["action"],
                                  "qwen_outcome": step["outcome"]},
                        "replay": {"sample_id": sample_id, "task": task,
                                   "prior_steps": [{"action": item["action"],
                                                    "observation": item["observation"],
                                                    "failed_action": item["failed_action"]}
                                                   for item in history]},
                    }
                    candidates[category].append(candidate)
                history.append({"action": step["action"], "observation": step["output_observation"],
                                "failed_action": step["outcome"]["failed_action"]})
        chosen = _select(candidates)
    output_dir.mkdir(parents=True, exist_ok=False)
    _new_jsonl(output_dir / "inputs.jsonl", [item["input"] for item in chosen])
    _new_jsonl(output_dir / "labels.jsonl", [item["label"] for item in chosen])
    _new_jsonl(output_dir / "replay.jsonl", [item["replay"] for item in chosen])
    information = {"schema_version": 1, "partition": "development", "count": len(chosen),
                   "category_counts": dict(Counter(item["category"] for item in chosen)),
                   "task_count": len({item["task_id"] for item in chosen}),
                   "selection_rule": "delegation_diagnostic_v1", "selection_uses_outcome_labels": True,
                   "source_archive_sha256": archive_sha256, "source_run_id": metadata["run_id"],
                   "source_git_commit": SOURCE_COMMIT,
                   "source_model_revision": config["model_revision"],
                   "source_manifest_sha256": metadata["selected_manifest_sha256"],
                   "source_config_sha256": metadata["config_sha256"],
                   "prompt_version": PROMPT_VERSIONS["textcraft"],
                   "prompt_source_sha256": hashlib.sha256(source_policy).hexdigest(),
                   "history_window": config["history_window"],
                   "files_sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                                    for name in ("inputs.jsonl", "labels.jsonl", "replay.jsonl")}}
    write_json_new(output_dir / "dataset.json", information)
    return information


def predict(config_path: Path, inputs_path: Path, output_dir: Path, limit: int | None = None) -> None:
    """Invoke a registered adapter with only the saved pre-generation messages."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    load_components(config)
    _verify_dataset_files(inputs_path)
    inputs = _jsonl(inputs_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        inputs = inputs[:limit]
    if not inputs or len({item["sample_id"] for item in inputs}) != len(inputs):
        raise ValueError("Diagnostic inputs must contain distinct sample IDs")
    model = make_model(config)
    model_metadata = model.metadata()
    if (model_metadata.get("model_id") != config.get("model_id")
            or model_metadata.get("model_revision") != config.get("model_revision")):
        raise ValueError("Loaded model identity differs from the config")
    output_dir.mkdir(parents=True, exist_ok=False)
    run_metadata = {"schema_version": 1, "status": "running", "config": config,
                    "config_sha256": canonical_digest(config),
                    "inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
                    "selected_sample_ids": [item["sample_id"] for item in inputs],
                    "model_metadata": model_metadata, **git_state()}
    write_json_new(output_dir / "run.json", run_metadata)
    try:
        prediction_hashes = {}
        for index, item in enumerate(inputs):
            result = model.generate(item["messages"])
            prediction = {"sample_id": item["sample_id"], "task_id": item["task_id"],
                          "decision": item["decision"], "action": parse_textcraft(result["text"]),
                          "raw_output": result["text"],
                          "input_tokens": result["input_tokens"],
                          "output_tokens": result["output_tokens"],
                          "inference_seconds": result["inference_seconds"]}
            for key in ("reasoning_tokens", "final_tokens", "generation_status",
                        "generation_error", "raw_completion",
                        "peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes"):
                if key in result:
                    prediction[key] = result[key]
            path = output_dir / f"{index:03d}_{item['sample_id']}.json"
            write_json_new(path, prediction)
            prediction_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        run_metadata["prediction_file_sha256"] = prediction_hashes
        run_metadata["status"] = "complete"
    except BaseException:
        run_metadata["status"] = "interrupted"
        raise
    finally:
        from .common import write_json_atomic
        write_json_atomic(output_dir / "run.json", run_metadata)


def replay(predictions_dir: Path, inputs_path: Path, labels_path: Path,
           replay_path: Path, output_path: Path) -> dict:
    """Compare immediate actions from two verified resets of the same state."""
    run = read_json(predictions_dir / "run.json")
    _verify_dataset_files(inputs_path, labels_path, replay_path)
    if run.get("status") != "complete" or run.get("inputs_sha256") != hashlib.sha256(inputs_path.read_bytes()).hexdigest():
        raise ValueError("Prediction run is incomplete or uses different inputs")
    inputs = {item["sample_id"]: item for item in _jsonl(inputs_path)}
    labels = {item["sample_id"]: item for item in _jsonl(labels_path)}
    prefixes = {item["sample_id"]: item for item in _jsonl(replay_path)}
    selected = run["selected_sample_ids"]
    if not set(selected) <= set(inputs) & set(labels) & set(prefixes):
        raise ValueError("Prediction IDs do not match diagnostic source files")
    expected_prediction_files = {f"{index:03d}_{sample_id}.json"
                                 for index, sample_id in enumerate(selected)}
    actual_prediction_files = {path.name for path in predictions_dir.glob("*.json")} - {"run.json"}
    if (actual_prediction_files != expected_prediction_files or
            set(run.get("prediction_file_sha256", {})) != expected_prediction_files):
        raise ValueError("Prediction files differ from the completed run record")
    observations = []
    for index, sample_id in enumerate(selected):
        path = predictions_dir / f"{index:03d}_{sample_id}.json"
        if (hashlib.sha256(path.read_bytes()).hexdigest() !=
                run.get("prediction_file_sha256", {}).get(path.name)):
            raise ValueError(f"Prediction file hash mismatch: {path}")
        prediction = read_json(path)
        if prediction.get("sample_id") != sample_id:
            raise ValueError("Prediction file mismatch")
        prefix = prefixes[sample_id]
        original = labels[sample_id]
        task = prefix["task"]
        def branch(action: str | None) -> dict | None:
            if action is None:
                return None
            episode = TextCraftEpisode(task)
            try:
                episode.reset()
                for earlier in prefix["prior_steps"]:
                    actual = episode.step(earlier["action"])
                    if (actual["observation"] != earlier["observation"]
                            or actual["failed_action"] != earlier["failed_action"]
                            or actual["done"]):
                        raise ValueError(f"TextCraft replay diverged before {sample_id}")
                if episode.observation != inputs[sample_id]["current_observation"]:
                    raise ValueError(f"TextCraft replay state diverged at {sample_id}")
                return episode.step(action)
            finally:
                episode.close()
        qwen = branch(original["qwen_action"])
        if qwen != original["qwen_outcome"]:
            raise ValueError(f"Original action did not replay exactly: {sample_id}")
        alternative = branch(prediction["action"])
        observations.append({"sample_id": sample_id, "category": original["category"],
                             "qwen_action": original["qwen_action"], "alternative_action": prediction["action"],
                             "action_changed": prediction["action"] != original["qwen_action"],
                             "qwen_immediate_outcome": qwen,
                             "alternative_immediate_outcome": alternative})
    by_category = {}
    for category in CATEGORY_COUNTS:
        subset = [item for item in observations if item["category"] == category]
        by_category[category] = {
            "count": len(subset),
            "valid_alternative_actions": sum(item["alternative_action"] is not None for item in subset),
            "actions_changed": sum(item["action_changed"] for item in subset),
            "original_failed": sum(item["qwen_immediate_outcome"]["failed_action"] for item in subset),
            "alternative_failed": sum(item["alternative_immediate_outcome"] is not None and
                                      item["alternative_immediate_outcome"]["failed_action"] for item in subset),
            "immediate_failure_avoided": sum(item["qwen_immediate_outcome"]["failed_action"] and
                                             item["alternative_immediate_outcome"] is not None and
                                             not item["alternative_immediate_outcome"]["failed_action"]
                                             for item in subset),
            "new_immediate_failure": sum(not item["qwen_immediate_outcome"]["failed_action"] and
                                         item["alternative_immediate_outcome"] is not None and
                                         item["alternative_immediate_outcome"]["failed_action"]
                                         for item in subset)}
    result = {"schema_version": 1, "scope": "immediate_action_only", "count": len(observations),
              "predictions_git_commit": run["git_commit"], "observations": observations,
              "by_category": by_category,
              "caution": "Immediate feedback cannot establish downstream task success or delegation benefit."}
    write_json_new(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--archive", type=Path, required=True)
    builder.add_argument("--output-dir", type=Path, required=True)
    predictor = commands.add_parser("predict")
    predictor.add_argument("--config", type=Path, required=True)
    predictor.add_argument("--inputs", type=Path, required=True)
    predictor.add_argument("--output-dir", type=Path, required=True)
    predictor.add_argument("--limit", type=int)
    replayer = commands.add_parser("replay")
    replayer.add_argument("--predictions", type=Path, required=True)
    replayer.add_argument("--inputs", type=Path, required=True)
    replayer.add_argument("--labels", type=Path, required=True)
    replayer.add_argument("--replay-prefixes", type=Path, required=True)
    replayer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.archive, args.output_dir), indent=2))
    elif args.command == "predict":
        predict(args.config, args.inputs, args.output_dir, args.limit)
    else:
        print(json.dumps(replay(args.predictions, args.inputs, args.labels,
                                args.replay_prefixes, args.output), indent=2))


if __name__ == "__main__":
    main()
