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
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json_new(path: str | Path, value: Any) -> None:
    """Publish a complete JSON file exactly once, even across process races."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)  # EEXIST protects completed episodes.
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported manifest schema")
    from .benchmarks import BENCHMARK_REGISTRY, MANIFEST_VALIDATORS
    if manifest.get("benchmark") not in BENCHMARK_REGISTRY:
        raise ValueError("Unknown benchmark")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Manifest must contain tasks")
    ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Task IDs must be distinct nonempty strings")
    if manifest["benchmark"] not in {"alfworld", "textcraft"}:
        if manifest["benchmark"] not in MANIFEST_VALIDATORS:
            raise ValueError("Custom benchmark requires a manifest validator")
        MANIFEST_VALIDATORS[manifest["benchmark"]](manifest)
        return
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
    if config.get("schema_version") == 2:
        validate_experiment_config(config, benchmark)
        return
    if config.get("benchmark") != benchmark or config.get("condition") not in {"baseline", "m2_always", "reconsideration_only_v1", "adapt_dynamic_replanning_v1", *TRAJECTORY_CONDITIONS}:
        raise ValueError("Config benchmark or condition is invalid")
    if config.get("quantization") != "nf4_4bit":
        raise ValueError("Baseline requires NF4 4-bit quantization")
    for key in ("max_actions", "max_decisions", "history_window", "max_new_tokens"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config.get("model_id"), str) or not config["model_id"]:
        raise ValueError("model_id is required")
    if "model_revision" in config and (not isinstance(config["model_revision"], str)
                                       or len(config["model_revision"]) != 40
                                       or any(char not in "0123456789abcdef" for char in config["model_revision"])):
        raise ValueError("model_revision must be a full 40-character lowercase commit hash")
    if "support_policy" in config and (not isinstance(config["support_policy"], str)
                                       or not config["support_policy"]):
        raise ValueError("support_policy must be a nonempty policy ID")
    if config["condition"] == "adapt_dynamic_replanning_v1":
        if benchmark != "textcraft":
            raise ValueError("ADaPT development condition currently supports TextCraft only")
        for key in ("adapt_max_depth", "adapt_max_executor_steps", "adapt_max_environment_actions",
                    "adapt_max_planner_calls", "adapt_max_model_calls",
                    "adapt_executor_max_new_tokens", "adapt_planner_max_new_tokens"):
            if not isinstance(config.get(key), int) or config[key] <= 0:
                raise ValueError(f"ADaPT requires a positive {key}")
        if (config["adapt_max_environment_actions"] != config["max_actions"]
                or config["adapt_max_model_calls"] != config["max_decisions"]
                or config["adapt_executor_max_new_tokens"] != config["max_new_tokens"]):
            raise ValueError("ADaPT shared action, decision, and executor-token limits must match their explicit budgets")
        if any(key in config for key in ("memory_corpus", "memory_index", "memory_top_k",
                                      "trajectory_corpus", "trajectory_index", "trajectory_top_k",
                                      "trajectory_token_budget")):
            raise ValueError("ADaPT cannot combine with memory or retrieval in v1")
    if config["condition"] == "reconsideration_only_v1" and any(
        key in config for key in ("memory_corpus", "memory_index", "memory_top_k",
                              "trajectory_corpus", "trajectory_index", "trajectory_top_k",
                              "trajectory_token_budget")
    ):
        raise ValueError("Reconsideration-only cannot configure memory or retrieval")
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


def validate_experiment_config(config: dict, benchmark: str) -> None:
    """Strict v2 configuration; v1 conditions retain their historical contract."""
    from .model import MODEL_REGISTRY
    from .support_policy import POLICY_REGISTRY
    from .supports import SUPPORT_REGISTRY
    from .controllers import CONTROLLER_REGISTRY, CONTROLLER_VALIDATORS
    from .benchmarks import BENCHMARK_DRIVERS, BENCHMARK_REGISTRY

    if (config.get("benchmark") != benchmark or benchmark not in BENCHMARK_REGISTRY
            or benchmark not in BENCHMARK_DRIVERS):
        raise ValueError("Unknown or mismatched benchmark")
    for key in ("manifest", "output_root", "model_id", "model_revision"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} is required")
    if not isinstance(config.get("partition_manifests"), dict) or not config["partition_manifests"]:
        raise ValueError("partition_manifests must identify the active and comparison partitions")
    if config.get("model_type") not in MODEL_REGISTRY:
        raise ValueError("Unknown model type")
    if config["model_type"] == "qwen":
        revision = config["model_revision"]
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("Qwen model_revision must be an exact 40-character commit")
        if config.get("quantization") != "nf4_4bit":
            raise ValueError("Qwen requires NF4 4-bit quantization")
    if config["model_type"] == "gpt_oss_20b":
        revision = config["model_revision"]
        if (config["model_id"] != "openai/gpt-oss-20b" or len(revision) != 40
                or any(char not in "0123456789abcdef" for char in revision)):
            raise ValueError("GPT-OSS requires its exact model ID and a pinned 40-character revision")
        if config.get("quantization") != "mxfp4_packed":
            raise ValueError("GPT-OSS requires packed MXFP4 quantization")
        if config.get("reasoning_effort") not in {"low", "medium", "high"}:
            raise ValueError("GPT-OSS reasoning_effort must be low, medium, or high")
        if not isinstance(config.get("backend_version"), str) or not config["backend_version"]:
            raise ValueError("GPT-OSS backend_version must be pinned")
        import datetime
        try:
            if datetime.date.fromisoformat(config["prompt_date"]).isoformat() != config["prompt_date"]:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("GPT-OSS prompt_date must be a fixed ISO date") from exc
        for key in ("temperature", "top_p"):
            if type(config.get(key)) not in (int, float) or not 0 < config[key] <= 1:
                raise ValueError(f"GPT-OSS {key} must be in (0, 1]")
        if type(config.get("top_k")) is not int or config["top_k"] < 0:
            raise ValueError("GPT-OSS top_k must be nonnegative")
    if type(config.get("do_sample")) is not bool:
        raise ValueError("do_sample must be explicit and Boolean")
    if config["model_type"] == "qwen" and config["do_sample"]:
        raise ValueError("The reference Qwen adapter preserves greedy decoding")
    if config["model_type"] == "gpt_oss_20b" and not config["do_sample"]:
        raise ValueError("GPT-OSS adapter requires sampling")
    for key in ("max_actions", "max_model_calls", "max_decisions", "history_window", "max_new_tokens"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "max_support_calls", "max_support_tokens"):
        if type(config.get(key)) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    supports = config.get("supports")
    if not isinstance(supports, list) or len(supports) != len(set(supports)) or any(
            item not in SUPPORT_REGISTRY for item in supports):
        raise ValueError("supports must be a list of distinct registered IDs")
    versions = config.get("support_versions")
    if not isinstance(versions, dict) or set(versions) != set(supports) or any(
            not isinstance(version, str) or not version for version in versions.values()):
        raise ValueError("support_versions must pin every configured support")
    if any(item in supports for item in ("task_trajectory_examples", "action_trajectory_reconsideration")):
        for key in ("trajectory_corpus", "trajectory_index"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"Trajectory support requires {key}")
        for key in ("trajectory_top_k", "trajectory_token_budget"):
            if type(config.get(key)) is not int or config[key] <= 0:
                raise ValueError(f"Trajectory support requires positive {key}")
    if "legacy_action_memory_reconsideration" in supports:
        for key in ("memory_corpus", "memory_index"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"Legacy M2 requires {key}")
        if type(config.get("memory_top_k")) is not int or config["memory_top_k"] <= 0:
            raise ValueError("Legacy M2 requires positive memory_top_k")
    policy = config.get("selection_policy")
    if not isinstance(policy, dict) or policy.get("id") not in POLICY_REGISTRY:
        raise ValueError("selection_policy must name a registered policy")
    if policy["id"] in {"always_v1", "random_v1", "failure_triggered_v1"}:
        if policy.get("support_id") not in supports or policy["support_id"] == "none":
            raise ValueError("Policy support_id must name an available support")
    if policy["id"] == "random_v1":
        if type(policy.get("probability")) not in (float, int) or not 0 <= policy["probability"] <= 1:
            raise ValueError("Random policy probability must be in [0, 1]")
        if type(policy.get("seed")) is not int:
            raise ValueError("Random policy seed must be an integer")
    if policy["id"] == "failure_triggered_v1":
        if type(policy.get("threshold")) is not int or policy["threshold"] <= 0:
            raise ValueError("Failure threshold must be positive")
        if type(policy.get("cooldown")) is not int or policy["cooldown"] < 0:
            raise ValueError("Failure cooldown must be nonnegative")
    controller = config.get("controller")
    if controller is not None and controller not in CONTROLLER_REGISTRY:
        raise ValueError("Unknown controller")
    if controller is not None:
        if controller not in CONTROLLER_VALIDATORS:
            raise ValueError("Schema-v2 controller needs its own budget and compatibility validator")
        CONTROLLER_VALIDATORS[controller](config)
    if config.get("condition") != "modular_v1" and controller is None:
        raise ValueError("v2 condition must be modular_v1")
