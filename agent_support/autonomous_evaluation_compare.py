"""Strict paired TextCraft comparison for the new matched evaluation cohort."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .analysis import load_run
from .autonomous_evaluation import ARMS, ORIGINAL_COMPARISON_ARMS, FROZEN_MANIFEST_SHA256
from .common import read_json, write_json_atomic
from .support_sequence_analysis import analyze_episodes, decision_sequences


SHARED_AGENT_KEYS = ("benchmark", "model_id", "model_revision", "quantization",
                     "max_actions", "max_decisions", "history_window", "max_new_tokens", "seed")
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def _episode_metrics(episode: dict) -> dict:
    rows = decision_sequences(episode)
    gptoss_support_calls = sum(
        event.get("result", {}).get("cost", {}).get("model_calls", 0)
        for record in episode.get("trace", [])
        for event in record.get("support_trace", {}).get("events", [])
        if event.get("request", {}).get("operation") == "reason_over_artifacts_v1")
    return {
        "success": bool(episode["success"]), "status": episode["status"],
        "environment_actions": episode["environment_actions"],
        "failed_actions": episode["failed_actions"],
        "model_calls": episode.get("model_calls", episode["model_decisions"]),
        "input_tokens": episode["input_tokens"], "output_tokens": episode["output_tokens"],
        "total_tokens": episode["input_tokens"] + episode["output_tokens"],
        "inference_seconds": episode["inference_seconds"],
        "controller_model_calls": episode.get("controller_model_calls", 0),
        "controller_tokens": (episode.get("controller_input_tokens", 0) +
                              episode.get("controller_output_tokens", 0)),
        "controller_inference_seconds": episode.get("controller_inference_seconds", 0),
        "support_model_calls": episode.get("support_model_calls", 0),
        "gptoss_calls": episode.get("controller_model_calls", 0) + gptoss_support_calls,
        "support_operations": sum(row["length"] for row in rows),
        "composed_decisions": sum(row["length"] > 1 for row in rows),
        "no_support_decisions": sum(row["length"] == 0 for row in rows),
        "support_decisions": len(rows),
        "action_replacements": episode.get("actions_changed_by_support", 0),
        "controller_wall_seconds": round(sum(row["controller_wall_seconds"] for row in rows), 3),
        "support_wall_seconds": (None if any(row["support_wall_seconds"] is None for row in rows)
                                 else round(sum(row["support_wall_seconds"] for row in rows), 3)),
        "support_inference_seconds": episode.get("support_inference_seconds",
                                                  sum(event.get("inference_seconds", 0)
                                                      for record in episode.get("trace", [])
                                                      for event in record.get("support_events", []))),
        "retrieval_seconds": episode.get("retrieval_seconds", 0),
        "budget_exhausted_decisions": sum(row["budget_exhausted"] for row in rows),
        "parser_errors": episode["parser_errors"],
        "generation_failures": episode.get("generation_failures", 0) +
                               sum(row["generation_failures"] for row in rows),
        "controller_parse_errors": sum(row["controller_parse_errors"] for row in rows),
        "support_parse_errors": max(episode.get("support_parse_errors", 0),
                                    sum(row["support_parse_errors"] for row in rows)),
    }


def _aggregate(episodes: dict[str, dict]) -> dict:
    rows = [_episode_metrics(item) for item in episodes.values()]
    fields = ("environment_actions", "failed_actions", "model_calls", "input_tokens", "output_tokens",
              "total_tokens", "inference_seconds", "controller_model_calls", "controller_tokens",
              "controller_inference_seconds", "support_model_calls", "gptoss_calls",
              "support_operations", "composed_decisions", "no_support_decisions", "support_decisions",
              "action_replacements", "controller_wall_seconds", "support_wall_seconds",
              "support_inference_seconds", "retrieval_seconds", "budget_exhausted_decisions",
              "parser_errors", "generation_failures", "controller_parse_errors",
              "support_parse_errors")
    totals = {key: (None if any(row[key] is None for row in rows) else
                    round(sum(row[key] for row in rows), 3)) for key in fields}
    successes = sum(row["success"] for row in rows)
    n = len(rows)
    if n:
        z = 1.96
        rate = successes / n
        denominator = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
        interval = [round(max(0.0, center - radius), 4), round(min(1.0, center + radius), 4)]
    else:
        interval = None
    return {"tasks": len(rows), "successes": successes,
            "success_rate": successes / n if n else None,
            "wilson_95_percent": interval, **totals}


def _mcnemar_exact(helped: int, hurt: int) -> float:
    discordant = helped + hurt
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(helped, hurt) + 1)) / 2 ** discordant
    return min(1.0, 2 * tail)


def _paired_bootstrap(left: list[bool], right: list[bool], seed: int = 20261118,
                      replicates: int = 10000) -> list[float]:
    rng = random.Random(seed)
    n = len(left)
    # Sample task indices jointly. The explicit loop keeps the pairing visible.
    values = []
    for _ in range(replicates):
        difference = 0
        for _ in range(n):
            index = rng.randrange(n)
            difference += int(right[index]) - int(left[index])
        values.append(difference / n)
    values.sort()
    return [values[int(0.025 * (replicates - 1))], values[int(0.975 * (replicates - 1))]]


def _pair(left_name: str, right_name: str, task_ids: list[str],
          episodes: dict[str, dict[str, dict]]) -> dict:
    left = [episodes[left_name][task_id] for task_id in task_ids]
    right = [episodes[right_name][task_id] for task_id in task_ids]
    helped = [task_ids[index] for index, (a, b) in enumerate(zip(left, right))
              if not a["success"] and b["success"]]
    hurt = [task_ids[index] for index, (a, b) in enumerate(zip(left, right))
            if a["success"] and not b["success"]]
    n = len(task_ids)
    return {"left": left_name, "right": right_name,
            "success_rate_difference": (len(helped) - len(hurt)) / n,
            "helped_task_ids": helped, "hurt_task_ids": hurt,
            "mcnemar_exact_two_sided_p": _mcnemar_exact(len(helped), len(hurt)),
            "paired_task_bootstrap_95_percent": _paired_bootstrap(
                [bool(item["success"]) for item in left],
                [bool(item["success"]) for item in right]),
            "mean_model_call_difference": sum(_episode_metrics(b)["model_calls"] -
                                              _episode_metrics(a)["model_calls"]
                                              for a, b in zip(left, right)) / n,
            "mean_token_difference": sum(_episode_metrics(b)["total_tokens"] -
                                         _episode_metrics(a)["total_tokens"]
                                         for a, b in zip(left, right)) / n}


def compare_cohort(paths: dict[str, str | Path], require_full: bool = True) -> dict:
    if require_full and set(paths) != set(ORIGINAL_COMPARISON_ARMS):
        raise ValueError("Full comparison requires exactly the six predeclared arms")
    if not paths or any(name not in ARMS for name in paths):
        raise ValueError("Unknown or empty evaluation arm set")
    loaded = {name: load_run(path) for name, path in paths.items()}
    first_meta, first_config, first_manifest, _ = next(iter(loaded.values()))
    task_ids = [task["task_id"] for task in first_manifest["tasks"]]
    if require_full and (len(task_ids) != 20 or first_meta.get("manifest_sha256") != FROZEN_MANIFEST_SHA256):
        raise ValueError("Full comparison requires the frozen 20-task development manifest")
    shared = {key: first_config.get(key) for key in SHARED_AGENT_KEYS}
    if shared["model_id"] != QWEN_MODEL_ID or shared["model_revision"] != QWEN_REVISION:
        raise ValueError("Primary cohort must use the pinned Qwen task agent")
    model_spec = None
    memory_digests = None
    for name, (meta, config, manifest, results) in loaded.items():
        if ({key: config.get(key) for key in SHARED_AGENT_KEYS} != shared or
                manifest != first_manifest or
                meta.get("manifest_sha256") != first_meta.get("manifest_sha256") or
                meta.get("package_versions") != first_meta.get("package_versions") or
                meta.get("python_version") != first_meta.get("python_version") or
                meta.get("git_commit") != first_meta.get("git_commit") or
                meta.get("textcraft_recipe_order_sha256") != first_meta.get("textcraft_recipe_order_sha256") or
                set(results) != set(task_ids)):
            raise ValueError(f"Incompatible primary cohort run: {name}")
        if require_full and (meta.get("limit") is not None or
                             meta.get("selected_manifest_sha256") != FROZEN_MANIFEST_SHA256):
            raise ValueError(f"Full run is truncated: {name}")
        support = config.get("autonomous_support", {})
        if name in {"random_support", "autonomous", "fixed_reasoning_2"}:
            current = support.get("model")
            if model_spec is None:
                model_spec = current
            elif current != model_spec:
                raise ValueError("Controller model configuration differs across support arms")
        if name in {"qwen_m2_200", "random_support", "autonomous"}:
            current = ((meta.get("trajectory_corpus_sha256"), meta.get("trajectory_index_sha256"))
                       if name == "qwen_m2_200" else
                       (meta.get("autonomous_memory_corpus_sha256"),
                        meta.get("autonomous_memory_index_sha256")))
            if None in current or (memory_digests is not None and current != memory_digests):
                raise ValueError("M2-200 corpus or index differs across memory arms")
            memory_digests = current
    episodes = {name: value[3] for name, value in loaded.items()}
    aggregates = {name: _aggregate(values) for name, values in episodes.items()}
    rows = [{"task_id": task_id, "arms": {name: _episode_metrics(values[task_id])
                                            for name, values in episodes.items()}}
            for task_id in task_ids]
    pair_names = [("qwen_standalone", name) for name in paths if name != "qwen_standalone"]
    pair_names += [(name, "autonomous") for name in paths
                   if name not in {"qwen_standalone", "autonomous"}]
    pairs = [_pair(left, right, task_ids, episodes) for left, right in pair_names
             if left in paths and right in paths]
    primary = [pair for pair in pairs if pair["right"] == "autonomous"]
    previous = 0.0
    for rank, pair in enumerate(sorted(primary, key=lambda item: item["mcnemar_exact_two_sided_p"])):
        previous = min(1.0, max(previous,
                                (len(primary) - rank) * pair["mcnemar_exact_two_sided_p"]))
        pair["holm_adjusted_p_across_autonomous_contrasts"] = previous
    sequence_summaries = {}
    for name, values in episodes.items():
        summary = analyze_episodes(values)
        summary.pop("decision_rows")
        sequence_summaries[name] = summary
    return {"schema_version": 1, "cohort": "matched_qwen_task_agent_textcraft_development",
            "full_20_task_comparison": require_full,
            "manifest_sha256": first_meta.get("manifest_sha256"),
            "code_commit": first_meta.get("git_commit"),
            "task_agent_settings": shared,
            "controller_model": model_spec,
            "memory_digests": {"corpus": memory_digests[0], "index": memory_digests[1]}
                              if memory_digests else None,
            "run_ids": {name: value[0]["run_id"] for name, value in loaded.items()},
            "aggregates": aggregates, "task_rows": rows, "paired_comparisons": pairs,
            "sequence_summaries": sequence_summaries,
            "interpretation": ("Paired task contrasts are exploratory on a fixed 20-task development set. "
                               "Support-use correlations are not per-call causal effects. "
                               "Fixed reasoning matches a call schedule, not realized tokens or decisions.")}


def historical_reference(path: str | Path) -> dict:
    meta, config, _, episodes = load_run(path)
    return {"run_id": meta["run_id"], "source": str(Path(path).resolve()),
            "condition": meta["condition"], "model_id": config.get("model_id"),
            "model_revision": config.get("model_revision"),
            "memory_corpus_sha256": meta.get("trajectory_corpus_sha256"),
            "manifest_sha256": meta.get("manifest_sha256"),
            "code_commit": meta.get("git_commit"),
            "aggregate": read_json(Path(path) / "summary.json"),
            "comparison_role": "historical_descriptive_reference_only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=PATH")
    parser.add_argument("--historical", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    def parse_pairs(items):
        pairs = [item.split("=", 1) for item in items]
        if any(len(pair) != 2 or not all(pair) for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
            raise ValueError("Run arguments must be unique LABEL=PATH pairs")
        return dict(pairs)
    report = compare_cohort(parse_pairs(args.run), require_full=not args.allow_partial)
    report["historical_references"] = {name: historical_reference(path)
                                        for name, path in parse_pairs(args.historical).items()}
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"Evaluation report already exists: {destination}")
    write_json_atomic(destination, report)
    print(json.dumps({"output": str(destination), "run_ids": report["run_ids"]}, indent=2))


if __name__ == "__main__":
    main()
