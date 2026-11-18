"""Small dependency-free helpers used by manifest, runner, and analysis."""

from __future__ import annotations

import hashlib
import json
import math
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
AUTONOMOUS_CONDITIONS = {"autonomous_support_v1", "autonomous_support_v3"}


def validate_config(config: dict, benchmark: str) -> None:
    if config.get("benchmark") != benchmark or config.get("condition") not in {"baseline", "m2_always", "reconsideration_only_v1", *AUTONOMOUS_CONDITIONS, *TRAJECTORY_CONDITIONS}:
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
    if config["condition"] in AUTONOMOUS_CONDITIONS:
        spec = config.get("autonomous_support")
        if not isinstance(spec, dict) or not isinstance(spec.get("controller"), str) or not spec["controller"]:
            raise ValueError("Autonomous support requires a controller ID")
        if config["condition"] == "autonomous_support_v3" and spec["controller"] != "prompted_gpt_oss_v3":
            raise ValueError("Autonomous support v3 requires the v3 controller")
        names = spec.get("operations", [])
        if not isinstance(names, list) or any(
                not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
            raise ValueError("Autonomous support operations must be distinct names")
        memory_spec = spec.get("memory")
        if "retrieve_episodic_memory_v1" in names:
            if benchmark != "textcraft" or not isinstance(memory_spec, dict):
                raise ValueError("TextCraft episodic retrieval requires a frozen memory configuration")
            if any(not isinstance(memory_spec.get(key), str) or not memory_spec[key]
                   for key in ("corpus_path", "index_path")):
                raise ValueError("Episodic retrieval requires corpus_path and index_path")
            for key in ("corpus_sha256", "index_sha256"):
                digest = memory_spec.get(key)
                if not isinstance(digest, str) or len(digest) != 64 or any(
                        char not in "0123456789abcdef" for char in digest):
                    raise ValueError(f"Episodic retrieval requires pinned {key}")
            for key in ("expected_trajectories", "max_top_k", "max_result_tokens"):
                if type(memory_spec.get(key)) is not int or memory_spec[key] <= 0:
                    raise ValueError(f"Episodic retrieval requires positive {key}")
        elif memory_spec is not None:
            raise ValueError("Memory configuration requires the episodic retrieval capability")
        from .autonomous_support import SupportBudget
        if not isinstance(spec.get("budget"), dict):
            raise ValueError("Autonomous support requires a budget")
        budget = SupportBudget.from_config(spec["budget"])
        if spec["controller"] == "random_single_support_v1":
            selection = spec.get("random_selection")
            if not isinstance(selection, dict):
                raise ValueError("Random support baseline requires random_selection")
            probability = selection.get("probability")
            eligible = selection.get("operations")
            if (type(probability) not in (int, float) or not 0 <= probability <= 1 or
                    type(selection.get("seed")) is not int or
                    not isinstance(eligible, list) or not eligible or
                    any(not isinstance(name, str) for name in eligible) or
                    len(eligible) != len(set(eligible)) or
                    not set(eligible) <= set(names) or
                    not set(eligible) <= {"retrieve_episodic_memory_v1", "inspect_history_v1",
                                          "inspect_state_v1", "reason_over_artifacts_v1"} or
                    budget.max_depth < 2 or budget.max_operations_per_decision < 1):
                raise ValueError("Random support baseline selection is invalid")
            if "retrieve_episodic_memory_v1" in eligible and memory_spec["max_top_k"] < 2:
                raise ValueError("Random retrieval requires max_top_k at least 2")
        if spec["controller"] == "fixed_reasoning_v1":
            fixed = spec.get("fixed_reasoning")
            if (not isinstance(fixed, dict) or type(fixed.get("calls_per_decision")) is not int or
                    fixed["calls_per_decision"] < 1 or
                    "reason_over_artifacts_v1" not in names or
                    budget.max_depth <= fixed["calls_per_decision"] or
                    budget.max_operations_per_decision < fixed["calls_per_decision"]):
                raise ValueError("Fixed reasoning baseline configuration is invalid")
        if spec["controller"] == "learned_gate_v1":
            gate = spec.get("learned_gate")
            if (not isinstance(gate, dict) or
                    not isinstance(gate.get("artifact_path"), str) or not gate["artifact_path"] or
                    not isinstance(gate.get("artifact_sha256"), str) or
                    len(gate["artifact_sha256"]) != 64 or
                    any(char not in "0123456789abcdef" for char in gate["artifact_sha256"]) or
                    type(gate.get("minimum_effect")) not in (int, float) or
                    not math.isfinite(gate["minimum_effect"])):
                raise ValueError("Learned gate requires a pinned artifact and finite threshold")
        if (spec["controller"] in {"prompted_gpt_oss_v1", "prompted_gpt_oss_v2",
                                   "prompted_gpt_oss_v3", "fixed_reasoning_v1"} or
                "reason_over_artifacts_v1" in names):
            model_spec = spec.get("model")
            if not isinstance(model_spec, dict) or model_spec.get("type") != "gpt_oss_20b_worker_v1":
                raise ValueError("Prompted GPT-OSS controller requires a GPT-OSS worker model")
            worker_model = model_spec.get("model")
            if not isinstance(worker_model, dict) or worker_model.get("model_id") != "openai/gpt-oss-20b":
                raise ValueError("GPT-OSS controller requires its exact model ID")
            revision = worker_model.get("model_revision")
            if not isinstance(revision, str) or len(revision) != 40 or any(
                    char not in "0123456789abcdef" for char in revision):
                raise ValueError("GPT-OSS controller requires a pinned model revision")
            for key in ("max_new_tokens", "seed"):
                if type(worker_model.get(key)) is not int or worker_model[key] <= 0:
                    raise ValueError(f"GPT-OSS controller requires positive {key}")
            for key in ("backend_version", "reasoning_effort", "prompt_date", "python_env"):
                value = worker_model.get(key) if key != "python_env" else model_spec.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"GPT-OSS controller requires {key}")
            if worker_model["reasoning_effort"] not in {"low", "medium", "high"}:
                raise ValueError("GPT-OSS controller reasoning effort is invalid")
            for key in ("temperature", "top_p"):
                if type(worker_model.get(key)) not in (int, float) or not 0 < worker_model[key] <= 1:
                    raise ValueError(f"GPT-OSS controller requires valid {key}")
            if type(worker_model.get("top_k")) is not int or worker_model["top_k"] < 0:
                raise ValueError("GPT-OSS controller requires nonnegative top_k")
            if config["condition"] == "autonomous_support_v3":
                if (spec["controller"] != "prompted_gpt_oss_v3" or
                        worker_model.get("completion_parser") != "harmony_final_v3"):
                    raise ValueError("Autonomous support v3 requires the v3 controller and Harmony parser")
            elif (spec["controller"] == "prompted_gpt_oss_v3" or
                  worker_model.get("completion_parser") == "harmony_final_v3"):
                raise ValueError("V3 controller and Harmony parser require autonomous_support_v3")
            import datetime
            try:
                if datetime.date.fromisoformat(worker_model["prompt_date"]).isoformat() != worker_model["prompt_date"]:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("GPT-OSS controller requires an ISO prompt_date") from exc
            ceiling = spec.get("support_max_new_tokens", 256)
            if type(ceiling) is not int or not 0 < ceiling <= worker_model["max_new_tokens"]:
                raise ValueError("Support output ceiling must fit the controller model ceiling")
        if any(key in config for key in ("memory_corpus", "memory_index", "memory_top_k",
                                         "trajectory_corpus", "trajectory_index", "trajectory_top_k",
                                         "trajectory_token_budget", "support_policy")):
            raise ValueError("Autonomous support v1 uses its own isolated support configuration")
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
