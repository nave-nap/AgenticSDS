"""Training-only, one-decision comparison of M2 and sequence memory.

Run in a fresh Python process with PYTHONHASHSEED=0. This diagnostic does not
change the frozen M2 condition or run a development/held-out task.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .analysis import load_run
from .benchmarks import make_episode
from .common import canonical_digest, read_json, validate_config, write_json_atomic
from .memory import ActionMemory, SentenceEmbedder, support_messages
from .model import QwenModel
from .policy import parse_textcraft, textcraft_messages
from .runner import git_commit


DEFAULT_SEEDS = (1009, 1018, 1007, 1003, 1027, 1023)


def recipes(initial_observation: str) -> set[str]:
    prefix = "Crafting commands:\n"
    separator = "\n\nGoal: craft "
    if not initial_observation.startswith(prefix) or separator not in initial_observation:
        raise ValueError("Unrecognized TextCraft task format")
    block = initial_observation[len(prefix):].rsplit(separator, 1)[0]
    return set(block.splitlines())


def successful_actions(episode: dict) -> list[dict]:
    return [step for step in episode["trace"]
            if step.get("outcome") and not step["outcome"].get("failed_action")]


def select_sequence(query: dict, episodes: dict[str, dict]) -> tuple[dict, int, list[dict]] | None:
    """Select another successful task by goal material and shared recipe lines."""
    query_recipes = recipes(query["initial_observation"])
    goal_word = query["initial_observation"].rsplit("Goal: craft ", 1)[1].split()[0].lower()
    goal_pattern = re.compile(rf"\b{re.escape(goal_word)}\b")
    candidates = []
    for source in episodes.values():
        if not source["success"] or source["task_id"] == query["task_id"]:
            continue
        if source["initial_observation"] == query["initial_observation"]:
            continue
        overlap = len(query_recipes & recipes(source["initial_observation"]))
        actions = successful_actions(source)
        if overlap and any(goal_pattern.search(step["action"].lower()) for step in actions):
            candidates.append((-overlap, source["task_id"], source, actions))
    if not candidates:
        return None
    negative_overlap, _, source, actions = min(candidates, key=lambda item: (item[0], item[1]))
    return source, -negative_overlap, actions[-4:]


def sequence_messages(base_messages: list[dict], source: dict, actions: list[dict]) -> list[dict]:
    goal = source["initial_observation"].rsplit("Goal: craft ", 1)[1].strip()
    lines = ["A different training task was completed successfully.",
             f"Past goal: craft {goal}", "Actions and observed feedback from that task:"]
    for step in actions:
        lines.append(f"Action: {step['action']}")
        lines.append(f"Feedback: {step['output_observation'][:240]}")
    lines.append("Choose the next command for the current task. Use its own recipes and current state.")
    messages = [dict(message) for message in base_messages]
    messages[-1]["content"] += "\n\n" + "\n".join(lines)
    return messages


def without_task(corpus: dict, index: dict, task_id: str) -> tuple[dict, dict]:
    """Create an ephemeral leave-one-task-out view of the frozen M2 index."""
    if len(corpus["records"]) != len(index["vectors"]):
        raise ValueError("M2 index and corpus lengths differ")
    positions = [i for i, record in enumerate(corpus["records"])
                 if record["task_id"] != task_id]
    filtered = {**corpus, "records": [corpus["records"][i] for i in positions]}
    filtered_index = {**index,
                      "corpus_sha256": canonical_digest(filtered),
                      "memory_ids": [index["memory_ids"][i] for i in positions],
                      "vectors": [index["vectors"][i] for i in positions]}
    return filtered, filtered_index


def check_action(task: dict, config: dict, action: str | None) -> dict:
    if action is None:
        return {"accepted": False, "solved": False, "feedback": "unparseable"}
    episode = make_episode("textcraft", task, config)
    try:
        episode.reset()
        outcome = episode.step(action)
        return {"accepted": not outcome["failed_action"],
                "solved": outcome["success"], "feedback": outcome["observation"]}
    finally:
        episode.close()


def generate_choice(model: QwenModel, messages: list[dict], task: dict, config: dict) -> dict:
    generation = model.generate(messages)
    action = parse_textcraft(generation["text"])
    return {"raw_output": generation["text"], "action": action,
            "check": check_action(task, config, action),
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
            "inference_seconds": generation["inference_seconds"]}


def run_probe(config: dict, training_run: str | Path, corpus: dict, index: dict,
              seeds: tuple[int, ...] = DEFAULT_SEEDS) -> dict:
    if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization != 0:
        raise RuntimeError("Start a fresh Python process with PYTHONHASHSEED=0")
    metadata, training_config, manifest, episodes = load_run(training_run)
    validate_config(config, "textcraft")
    if config["condition"] != "baseline" or config != training_config:
        raise ValueError("Probe must use the unchanged baseline model settings")
    if (metadata.get("partition") != "training" or manifest.get("partition") != "training"
            or metadata.get("condition") != "baseline" or training_config.get("condition") != "baseline"
            or manifest.get("benchmark") != "textcraft"
            or metadata.get("limit") is not None
            or metadata.get("selected_task_count") != metadata.get("total_manifest_task_count")):
        raise ValueError("Expected one complete TextCraft baseline training run")
    if metadata["run_id"] not in {item["run_id"] for item in corpus["source_runs"]}:
        raise ValueError("M2 corpus does not contain this training run")
    if index["corpus_sha256"] != canonical_digest(corpus):
        raise ValueError("M2 index does not match corpus")
    task_map = {task["task_id"]: task for task in manifest["tasks"]}
    probes = []
    for seed in seeds:
        task_id = f"textcraft:seed:{seed}"
        if task_id not in task_map:
            raise ValueError(f"Missing training task: {task_id}")
        task = task_map[task_id]
        selected = select_sequence(episodes[task_id], episodes)
        source, overlap, steps = selected if selected is not None else (None, 0, [])
        probes.append((task, source, overlap, steps))
    embedder = SentenceEmbedder(index["embedding_model"], revision=index.get("embedding_revision"))
    model = QwenModel(config)
    rows = []
    for task, source, overlap, steps in probes:
        base_messages = textcraft_messages(task["initial_observation"], [], config["history_window"])
        baseline = generate_choice(model, base_messages, task, config)
        filtered_corpus, filtered_index = without_task(corpus, index, task["task_id"])
        memory = ActionMemory(filtered_corpus, filtered_index, embedder, top_k=3)
        hits = []
        if baseline["action"] is not None:
            hits, _ = memory.retrieve("textcraft", task["initial_observation"],
                                      task["initial_observation"], baseline["action"])
        m2 = (generate_choice(model, support_messages(base_messages, baseline["action"], hits,
                                                       "textcraft"), task, config)
              if hits else {**baseline, "fallback_reason": "no_relevant_memory"})
        sequence = (generate_choice(model, sequence_messages(base_messages, source, steps), task, config)
                    if source is not None else {**baseline, "fallback_reason": "no_relevant_sequence"})
        row = {"task_id": task["task_id"], "baseline": baseline,
               "m2": m2, "m2_memory_ids": [hit["memory_id"] for hit in hits],
               "sequence": sequence,
               "sequence_source_task_id": source["task_id"] if source is not None else None,
               "shared_recipe_count": overlap,
               "sequence_actions": [{"action": step["action"],
                                     "feedback": step["output_observation"][:240]} for step in steps]}
        rows.append(row)
        print(f"{task['task_id']}: baseline={baseline['action']} "
              f"accepted={baseline['check']['accepted']}; M2={m2['action']} "
              f"accepted={m2['check']['accepted']} hits={len(hits)}; "
              f"sequence={sequence['action']} accepted={sequence['check']['accepted']} "
              f"source={source['task_id'] if source is not None else 'none'} "
              f"shared_recipes={overlap}", flush=True)
    return {"schema_version": 1, "kind": "training_only_first_action_memory_probe",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(), "training_run_id": metadata["run_id"],
            "training_manifest_sha256": metadata["selected_manifest_sha256"],
            "m2_corpus_sha256": canonical_digest(corpus),
            "m2_index_sha256": canonical_digest(index),
            "config_sha256": canonical_digest(config),
            "selection": "other_successful_task_goal_word_in_valid_actions_most_shared_recipes_last_four_valid_actions",
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    report = run_probe(read_json(args.config), args.training_run, read_json(args.corpus),
                       read_json(args.index), tuple(args.seed) if args.seed else DEFAULT_SEEDS)
    write_json_atomic(args.output, report)
    print(f"Saved training-only probe: {args.output}", flush=True)


if __name__ == "__main__":
    main()
