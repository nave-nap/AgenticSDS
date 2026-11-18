"""Paired outcome and cost comparison of prompted and learned-gated controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import load_run
from .autonomous_evaluation import FROZEN_MANIFEST_SHA256
from .autonomous_evaluation_compare import _aggregate, _episode_metrics, _pair
from .common import canonical_digest, read_json, write_json_atomic
from .learned_support_gate import validate_artifact
from .support_sequence_analysis import analyze_episodes


def compare_prompted_and_learned(prompted_path: str | Path,
                                 learned_path: str | Path) -> dict:
    prompted_meta, prompted_config, prompted_manifest, prompted_episodes = load_run(prompted_path)
    learned_meta, learned_config, learned_manifest, learned_episodes = load_run(learned_path)
    if (prompted_config.get("condition") != "autonomous_support_v1" or
            learned_config.get("condition") != "autonomous_support_v1" or
            prompted_config["autonomous_support"]["controller"] != "prompted_gpt_oss_v1" or
            learned_config["autonomous_support"]["controller"] != "learned_gate_v1"):
        raise ValueError("Expected prompted and learned autonomous-support runs")
    prompted_base = {key: value for key, value in prompted_config.items() if key != "autonomous_support"}
    learned_base = {key: value for key, value in learned_config.items() if key != "autonomous_support"}
    prompted_support = {key: value for key, value in prompted_config["autonomous_support"].items()
                        if key != "controller"}
    learned_support = {key: value for key, value in learned_config["autonomous_support"].items()
                       if key not in {"controller", "learned_gate"}}
    if (prompted_base != learned_base or prompted_support != learned_support or
            prompted_manifest != learned_manifest or
            prompted_meta.get("manifest_sha256") != learned_meta.get("manifest_sha256") or
            prompted_meta.get("git_commit") != learned_meta.get("git_commit") or
            prompted_meta.get("python_version") != learned_meta.get("python_version") or
            prompted_meta.get("package_versions") != learned_meta.get("package_versions") or
            prompted_meta.get("textcraft_recipe_order_sha256") != learned_meta.get("textcraft_recipe_order_sha256") or
            prompted_meta.get("autonomous_memory_corpus_sha256") != learned_meta.get("autonomous_memory_corpus_sha256") or
            prompted_meta.get("autonomous_memory_index_sha256") != learned_meta.get("autonomous_memory_index_sha256") or
            not learned_meta.get("learned_gate_artifact_sha256") or
            learned_meta["learned_gate_artifact_sha256"] !=
            learned_config["autonomous_support"]["learned_gate"]["artifact_sha256"]):
        raise ValueError("Prompted and learned runs differ beyond the controller gate")
    if (len(prompted_manifest["tasks"]) != 20 or
            prompted_meta.get("manifest_sha256") != FROZEN_MANIFEST_SHA256 or
            prompted_meta.get("limit") is not None or learned_meta.get("limit") is not None or
            prompted_meta.get("selected_manifest_sha256") != FROZEN_MANIFEST_SHA256 or
            learned_meta.get("selected_manifest_sha256") != FROZEN_MANIFEST_SHA256):
        raise ValueError("Learned comparison requires full frozen 20-task development runs")
    artifact = read_json(Path(learned_path) / "learned_gate_artifact.json")
    validate_artifact(artifact)
    if (canonical_digest(artifact) != learned_meta["learned_gate_artifact_sha256"] or
            artifact.get("development_manifest_sha256") != FROZEN_MANIFEST_SHA256 or
            artifact.get("task_agent_model_revision") != learned_config.get("model_revision") or
            artifact.get("controller_model_revision") !=
            learned_config["autonomous_support"]["model"]["model"]["model_revision"] or
            artifact.get("memory_corpus_sha256") != learned_meta["autonomous_memory_corpus_sha256"]):
        raise ValueError("Learned artifact snapshot or training provenance differs")
    task_ids = [task["task_id"] for task in prompted_manifest["tasks"]]
    paired = _pair("prompted", "learned", task_ids,
                   {"prompted": prompted_episodes, "learned": learned_episodes})
    sequences = {}
    for label, episodes in (("prompted", prompted_episodes), ("learned", learned_episodes)):
        value = analyze_episodes(episodes)
        value.pop("decision_rows")
        sequences[label] = value
    return {"schema_version": 1, "comparison": "learned_gate_vs_prompted_controller",
            "manifest_sha256": prompted_meta["manifest_sha256"],
            "code_commit": prompted_meta["git_commit"],
            "run_ids": {"prompted": prompted_meta["run_id"],
                        "learned": learned_meta["run_id"]},
            "learned_artifact_sha256": learned_meta["learned_gate_artifact_sha256"],
            "aggregates": {"prompted": _aggregate(prompted_episodes),
                           "learned": _aggregate(learned_episodes)},
            "paired": paired, "sequence_summaries": sequences,
            "task_rows": [{"task_id": task_id,
                           "prompted": _episode_metrics(prompted_episodes[task_id]),
                           "learned": _episode_metrics(learned_episodes[task_id])}
                          for task_id in task_ids],
            "interpretation": ("Paired tasks and matched settings isolate the gate configuration at the run level, "
                               "subject to stochastic model generation. Per-support causal effects require "
                               "separate paired state interventions.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompted-run", required=True)
    parser.add_argument("--learned-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_prompted_and_learned(args.prompted_run, args.learned_run)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(f"Learned comparison report already exists: {path}")
    write_json_atomic(path, report)
    print(json.dumps({"output": str(path), "run_ids": report["run_ids"]}, indent=2))


if __name__ == "__main__":
    main()
