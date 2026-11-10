"""Provenance and verified checkpoint recovery for schema-v2 runs."""

from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .common import canonical_digest, read_json, validate_manifest, write_json_atomic


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state() -> dict:
    root = Path(__file__).resolve().parent.parent
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, check=True).stdout
    status = git("status", "--porcelain", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD")
    return {"git_commit": git("rev-parse", "HEAD").strip(), "git_dirty": bool(status.strip()),
            "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def audit_partitions(config: dict, manifest: dict) -> dict:
    paths = config.get("partition_manifests")
    if not isinstance(paths, dict) or manifest.get("partition") not in paths:
        raise ValueError("partition_manifests must include the active partition")
    loaded = {}
    for partition, path in paths.items():
        if not isinstance(path, str) or not path:
            raise ValueError("Invalid partition manifest path")
        item = read_json(path)
        validate_manifest(item)
        if item.get("partition") != partition or item["benchmark"] != manifest["benchmark"]:
            raise ValueError(f"Partition manifest identity mismatch: {partition}")
        loaded[partition] = item
    if loaded[manifest["partition"]] != manifest:
        raise ValueError("Active manifest differs from partition snapshot")
    identities = {}
    for partition, item in loaded.items():
        fingerprint = ("game_sha256" if item["benchmark"] == "alfworld" else
                       "observation_sha256" if item["benchmark"] == "textcraft" else "task_fingerprint")
        if any(fingerprint not in task for task in item["tasks"]):
            raise ValueError(f"Partition tasks require {fingerprint}")
        identities[partition] = ({task["task_id"] for task in item["tasks"]},
                                 {task[fingerprint] for task in item["tasks"]})
    names = list(loaded)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if identities[left][0] & identities[right][0] or identities[left][1] & identities[right][1]:
                raise ValueError(f"Task overlap between {left} and {right}")
    if manifest["partition"] == "heldout" and not {"training", "development", "heldout"} <= set(loaded):
        raise ValueError("Held-out runs require training, development, and held-out manifests")
    return {name: canonical_digest(item) for name, item in loaded.items()}


def checked_run_key(run_key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", run_key):
        raise ValueError("run_key must be a short alphanumeric identifier")
    return run_key


def episode_filename(index: int, task: dict) -> str:
    return f"{index:04d}_{canonical_digest(task['task_id'])[:12]}.json"


def checkpoint(directory: Path, selected: list[dict], episodes: dict[int, dict]) -> None:
    files = {episode_filename(index, selected[index]): canonical_digest(result)
             for index, result in episodes.items()}
    write_json_atomic(directory / "checkpoint.json", {"schema_version": 1,
                      "episodes": files, "completed_task_count": len(files), "updated_utc": now_utc()})


def verify_resume(directory: Path, config: dict, selected_manifest: dict,
                  expected_provenance: dict, expected_artifacts: dict) -> tuple[dict, dict[int, dict], list[str]]:
    metadata = read_json(directory / "run.json")
    if metadata.get("schema_version") != 2 or metadata.get("status") == "complete":
        raise ValueError("Run is complete or does not support verified resumption")
    if metadata.get("status") not in {"running", "failed", "interrupted"}:
        raise ValueError("Run has an unknown checkpoint status")
    if read_json(directory / "config.json") != config or read_json(directory / "manifest.json") != selected_manifest:
        raise ValueError("Resume config or manifest snapshot differs")
    if metadata.get("config_sha256") != canonical_digest(config) or metadata.get(
            "selected_manifest_sha256") != canonical_digest(selected_manifest):
        raise ValueError("Resume configuration or manifest hash differs")
    for key, value in expected_provenance.items():
        if metadata.get(key) != value:
            raise ValueError(f"Resume provenance differs: {key}")
    for key, value in expected_artifacts.items():
        if metadata.get(key) != value:
            raise ValueError(f"Resume artifact differs: {key}")
    if metadata.get("textcraft_recipe_order_sha256"):
        path = directory / "textcraft_recipe_order.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["textcraft_recipe_order_sha256"]:
            raise ValueError("TextCraft recipe snapshot differs")
    if metadata.get("alfworld_config_sha256"):
        path = directory / "alfworld_config.yaml"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["alfworld_config_sha256"]:
            raise ValueError("ALFWorld source configuration differs")
    tasks = selected_manifest["tasks"]
    expected = {episode_filename(index, task): index for index, task in enumerate(tasks)}
    paths = list((directory / "episodes").glob("*.json"))
    if any(path.name not in expected for path in paths):
        raise ValueError("Run contains an unexpected episode file")
    saved = read_json(directory / "checkpoint.json") if (directory / "checkpoint.json").exists() else {"episodes": {}}
    if any(name not in expected for name in saved["episodes"]):
        raise ValueError("Checkpoint contains an unexpected episode")
    episodes = {}
    recovered = []
    for path in paths:
        index = expected[path.name]
        episode = read_json(path)
        task = tasks[index]
        if (episode.get("task_id") != task["task_id"] or
                episode.get("benchmark") != selected_manifest["benchmark"] or
                episode.get("task_sha256") != canonical_digest(task) or
                episode.get("config_sha256") != canonical_digest(config) or
                not isinstance(episode.get("success"), bool) or
                not isinstance(episode.get("trace"), list)):
            raise ValueError(f"Saved episode is incompatible: {path}")
        digest = canonical_digest(episode)
        if path.name in saved["episodes"] and saved["episodes"][path.name] != digest:
            raise ValueError(f"Saved episode hash differs from checkpoint: {path}")
        if path.name not in saved["episodes"]:
            recovered.append(path.name)
        episodes[index] = episode
    if any(name not in {path.name for path in paths} for name in saved["episodes"]):
        raise ValueError("Checkpoint refers to a missing episode")
    return metadata, episodes, recovered
