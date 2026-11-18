"""CPU-only ridge baseline for paired support-versus-execute outcomes."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .common import canonical_digest, read_json, validate_manifest, write_json_atomic
from .learned_support_gate import FEATURE_NAMES, FEATURE_VERSION, feature_vector


PAIRING_PROTOCOL = "replayed_identical_textcraft_state_v1"
BRANCHES = {"no_support", "prompted_support"}


def _solve_ridge(features: list[list[float]], targets: list[float],
                 sample_weights: list[float], regularization: float) -> list[float]:
    width = len(FEATURE_NAMES)
    matrix = [[0.0] * (width + 1) for _ in range(width)]
    for row, target, weight in zip(features, targets, sample_weights):
        for i in range(width):
            for j in range(width):
                matrix[i][j] += weight * row[i] * row[j]
            matrix[i][-1] += weight * row[i] * target
    for i in range(width):
        matrix[i][i] += regularization * (0.1 if i == 0 else 1.0)
    for column in range(width):
        pivot = max(range(column, width), key=lambda index: abs(matrix[index][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        if abs(matrix[column][column]) < 1e-12:
            raise ValueError("Learned gate fit is singular")
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for row in range(width):
            if row == column:
                continue
            factor = matrix[row][column]
            matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[column])]
    return [row[-1] for row in matrix]


def train_gate(dataset: dict, training_manifest: dict, development_manifest: dict,
               heldout_manifest: dict, memory_corpus: dict, *,
               regularization: float = 1.0, call_penalty: float = 0.01,
               token_penalty_per_1000: float = 0.001,
               minimum_tasks: int = 10) -> dict:
    for manifest, partition in ((training_manifest, "training"),
                                (development_manifest, "development"),
                                (heldout_manifest, "heldout")):
        validate_manifest(manifest)
        if manifest.get("partition") != partition or manifest["benchmark"] != "textcraft":
            raise ValueError(f"Expected a frozen TextCraft {partition} manifest")
    if any(not math.isfinite(value) or value < 0 for value in
           (regularization, call_penalty, token_penalty_per_1000)) or regularization <= 0:
        raise ValueError("Training penalties and regularization are invalid")
    if type(minimum_tasks) is not int or minimum_tasks < 10:
        raise ValueError("minimum_tasks must be at least ten")
    if (dataset.get("schema_version") != 1 or dataset.get("partition") != "training" or
            dataset.get("pairing_protocol") != PAIRING_PROTOCOL or
            dataset.get("manifest_sha256") != canonical_digest(training_manifest) or
            dataset.get("memory_corpus_sha256") != canonical_digest(memory_corpus)):
        raise ValueError("Paired intervention data provenance differs")
    for key in ("task_agent_model_revision", "controller_model_revision"):
        if not isinstance(dataset.get(key), str) or not dataset[key]:
            raise ValueError(f"Paired intervention data lacks {key}")
    training_tasks = {item["task_id"]: item for item in training_manifest["tasks"]}
    excluded_ids = {item["task_id"] for manifest in (development_manifest, heldout_manifest)
                    for item in manifest["tasks"]}
    excluded_hashes = {item["observation_sha256"] for manifest in (development_manifest, heldout_manifest)
                       for item in manifest["tasks"]}
    memory_ids = {item["task_id"] for item in memory_corpus.get("records", [])}
    memory_hashes = {item.get("task_fingerprint") for item in memory_corpus.get("records", [])}
    records = dataset.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Paired intervention data has no records")
    pairs = []
    seen = set()
    for record in records:
        task_id = record.get("task_id")
        task = training_tasks.get(task_id)
        if (task is None or task_id in excluded_ids or task_id in memory_ids or
                task["observation_sha256"] in excluded_hashes or
                task["observation_sha256"] in memory_hashes or
                record.get("task_fingerprint") != task["observation_sha256"]):
            raise ValueError("Intervention record leaks across task, memory, or evaluation partitions")
        pair_id = record.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise ValueError("Intervention pair ID is missing or duplicated")
        seen.add(pair_id)
        state = record.get("decision_state")
        if (not isinstance(state, dict) or record.get("state_sha256") != canonical_digest(state) or
                record.get("replay_verified") is not True or
                not isinstance(record.get("rollout_seed"), int) or
                not isinstance(record.get("followup_policy_id"), str)):
            raise ValueError("Intervention pair lacks a verified shared decision state")
        vector = feature_vector(state)
        branches = record.get("branches")
        if not isinstance(branches, dict) or set(branches) != BRANCHES:
            raise ValueError("Intervention pair requires no-support and prompted-support branches")
        for name, branch in branches.items():
            if (branch.get("complete") is not True or type(branch.get("task_success")) is not bool or
                    type(branch.get("total_model_calls")) is not int or branch["total_model_calls"] < 0 or
                    type(branch.get("total_tokens")) is not int or branch["total_tokens"] < 0 or
                    branch.get("rollout_seed") != record["rollout_seed"] or
                    branch.get("followup_policy_id") != record["followup_policy_id"]):
                raise ValueError(f"Incomplete or incompatible intervention branch: {name}")
        no_support = branches["no_support"]
        prompted = branches["prompted_support"]
        target = (int(prompted["task_success"]) - int(no_support["task_success"]) -
                  call_penalty * (prompted["total_model_calls"] - no_support["total_model_calls"]) -
                  token_penalty_per_1000 * (prompted["total_tokens"] - no_support["total_tokens"]) / 1000)
        pairs.append((task_id, vector, target,
                      int(prompted["task_success"]) - int(no_support["task_success"])))
    task_counts = Counter(task_id for task_id, _, _, _ in pairs)
    if len(task_counts) < minimum_tasks:
        raise ValueError("Too few distinct training tasks for the learned gate")
    weights = _solve_ridge([item[1] for item in pairs], [item[2] for item in pairs],
                           [1 / task_counts[item[0]] for item in pairs], regularization)
    return {"schema_version": 1, "policy_type": "cost_sensitive_ridge_gate",
            "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
            "weights": [round(value, 10) for value in weights],
            "regularization": regularization, "call_penalty": call_penalty,
            "token_penalty_per_1000": token_penalty_per_1000,
            "training_pairs": len(pairs), "training_tasks": len(task_counts),
            "observed_prompted_win_pairs": sum(item[3] == 1 for item in pairs),
            "observed_no_support_win_pairs": sum(item[3] == -1 for item in pairs),
            "training_data_sha256": canonical_digest(dataset),
            "training_manifest_sha256": canonical_digest(training_manifest),
            "development_manifest_sha256": canonical_digest(development_manifest),
            "heldout_manifest_sha256": canonical_digest(heldout_manifest),
            "memory_corpus_sha256": canonical_digest(memory_corpus),
            "task_agent_model_revision": dataset["task_agent_model_revision"],
            "controller_model_revision": dataset["controller_model_revision"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-data", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--memory-corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = train_gate(read_json(args.paired_data), read_json(args.training_manifest),
                          read_json(args.development_manifest), read_json(args.heldout_manifest),
                          read_json(args.memory_corpus))
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"Learned gate artifact already exists: {destination}")
    write_json_atomic(destination, artifact)
    print(json.dumps({"artifact_path": str(destination),
                      "artifact_sha256": canonical_digest(artifact),
                      "training_pairs": artifact["training_pairs"],
                      "training_tasks": artifact["training_tasks"]}, indent=2))


if __name__ == "__main__":
    main()
