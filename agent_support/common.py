"""Small dependency-free helpers used by manifest, runner, and analysis."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_text(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def canonical_digest(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    if manifest.get("benchmark") not in {"alfworld", "textcraft"}:
        raise ValueError("Unknown benchmark")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Manifest must contain tasks")
    ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Task IDs must be distinct nonempty strings")
    if manifest["benchmark"] == "textcraft":
        for task in tasks:
            if not isinstance(task.get("seed"), int) or not isinstance(task.get("initial_observation"), str):
                raise ValueError("TextCraft tasks require seed and initial_observation")
            if digest_text(task["initial_observation"]) != task.get("observation_sha256"):
                raise ValueError("TextCraft observation hash mismatch")
    else:
        for task in tasks:
            if not all(isinstance(task.get(key), str) and task[key] for key in ("split", "game_relpath", "game_sha256")):
                raise ValueError("ALFWorld tasks require split, game_relpath, and game_sha256")


TRAJECTORY_CONDITIONS = {"expel_retrieval_only_v1", "m2_trajectory_v1"}


def validate_config(config: dict, benchmark: str) -> None:
    if config.get("benchmark") != benchmark or config.get("condition") not in {"baseline", "m2_always", *TRAJECTORY_CONDITIONS}:
        raise ValueError("Config benchmark or condition is invalid")
    if config.get("quantization") != "nf4_4bit":
        raise ValueError("Baseline requires NF4 4-bit quantization")
    for key in ("max_actions", "max_decisions", "history_window", "max_new_tokens"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config.get("model_id"), str) or not config["model_id"]:
        raise ValueError("model_id is required")
    if "support_policy" in config and (not isinstance(config["support_policy"], str)
                                       or not config["support_policy"]):
        raise ValueError("support_policy must be a nonempty policy ID")
    if config["condition"] == "m2_always" or any(key in config for key in
                                                 ("memory_corpus", "memory_index", "memory_top_k")):
        if not all(isinstance(config.get(key), str) and config[key] for key in ("memory_corpus", "memory_index")):
            raise ValueError("M2 requires memory_corpus and memory_index paths")
        if not isinstance(config.get("memory_top_k"), int) or config["memory_top_k"] <= 0:
            raise ValueError("M2 requires a positive memory_top_k")
    if config["condition"] in TRAJECTORY_CONDITIONS or any(key in config for key in
                                                           ("trajectory_corpus", "trajectory_index",
                                                            "trajectory_top_k", "trajectory_token_budget")):
        if not all(isinstance(config.get(key), str) and config[key]
                   for key in ("trajectory_corpus", "trajectory_index")):
            raise ValueError("Trajectory support requires corpus and index paths")
        for key in ("trajectory_top_k", "trajectory_token_budget"):
            if not isinstance(config.get(key), int) or config[key] <= 0:
                raise ValueError(f"Trajectory support requires a positive {key}")
