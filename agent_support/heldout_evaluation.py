"""Frozen, three-condition TextCraft held-out evaluation on Locust."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import subprocess
from pathlib import Path

from .analysis import compare_trajectory, load_run
from .common import canonical_digest, read_json, validate_manifest, write_json_atomic
from .runner import package_versions, run


REPO = Path(__file__).resolve().parent.parent
EXPECTED_HELDOUT_SHA256 = "5c8dbc6caa2db221cc6ed4ce3208d9c53157af515563895a927957c3ac8349a0"
CONDITIONS = ("baseline", "m2_trajectory_v1", "reconsideration_only_v1")
BOOTSTRAP_SEED = 20261109
BOOTSTRAP_REPLICATES = 10000


def _manifest(root: Path, partition: str) -> dict:
    manifest = read_json(root / "manifests" / "textcraft" / f"{partition}.json")
    validate_manifest(manifest)
    if manifest.get("partition") != partition:
        raise ValueError(f"TextCraft {partition} manifest has the wrong partition")
    return manifest


def audit_pool(root: Path) -> dict:
    """Audit the existing finite held-out set; never generate fresh seeds here."""
    training, development, heldout = (_manifest(root, name) for name in
                                       ("training", "development", "heldout"))
    if canonical_digest(heldout) != EXPECTED_HELDOUT_SHA256:
        raise ValueError("Held-out manifest differs from the originally frozen 100-task set")
    for name, manifest in (("development", development), ("heldout", heldout)):
        if manifest != read_json(REPO / "manifests" / "textcraft" / f"{name}.json"):
            raise ValueError(f"Cluster {name} manifest differs from the repository snapshot")
    for name, manifest in (("training", training), ("development", development), ("heldout", heldout)):
        ids = [task["task_id"] for task in manifest["tasks"]]
        hashes = [task["observation_sha256"] for task in manifest["tasks"]]
        if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
            raise ValueError(f"{name} manifest repeats a task identity or observation")
        if any(task["task_id"] != f"textcraft:seed:{task['seed']}" for task in manifest["tasks"]):
            raise ValueError(f"{name} task ID does not match its seed")
    partitions = {"training": training, "development": development, "heldout": heldout}
    identities = {name: ({task["task_id"] for task in manifest["tasks"]},
                         {task["observation_sha256"] for task in manifest["tasks"]})
                  for name, manifest in partitions.items()}
    for left, right in (("training", "development"), ("training", "heldout"),
                        ("development", "heldout")):
        if identities[left][0] & identities[right][0] or identities[left][1] & identities[right][1]:
            raise ValueError(f"{left} and {right} tasks overlap by identity or content")
    heldout_ids, heldout_hashes = identities["heldout"]
    touched = []
    for path in sorted((root / "results").glob("textcraft_*/manifest.json")):
        old = read_json(path)
        if old.get("benchmark") != "textcraft":
            continue
        if any(task.get("task_id") in heldout_ids or task.get("observation_sha256") in heldout_hashes
               for task in old.get("tasks", [])):
            touched.append(path.parent.name)
    return {"eligible_count": len(heldout["tasks"]), "training_count": len(training["tasks"]),
            "pilot_count": len(development["tasks"]),
            "training_manifest_sha256": canonical_digest(training),
            "development_manifest_sha256": canonical_digest(development),
            "heldout_manifest_sha256": canonical_digest(heldout),
            "heldout_file_sha256": hashlib.sha256((root / "manifests" / "textcraft" / "heldout.json").read_bytes()).hexdigest(),
            "task_ids_sha256": canonical_digest([task["task_id"] for task in heldout["tasks"]]),
            "observation_hashes_sha256": canonical_digest([task["observation_sha256"] for task in heldout["tasks"]]),
            "prior_heldout_runs": touched}


def cached_model_revision(model_id: str) -> str:
    """Resolve an already cached Hugging Face snapshot without downloading data."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=model_id, local_files_only=True))
    revision = snapshot.name
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not (snapshot / "config.json").is_file():
        raise RuntimeError(f"No complete-looking cached model snapshot found: {snapshot}")
    return revision


def additional_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name)
            for name in ("huggingface-hub", "tokenizers", "safetensors")}


def _clean_commit() -> str:
    result = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError("Commit the code and use a clean cluster checkout before freezing the evaluation")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return commit


def lock_payload(root: Path) -> dict:
    from .cluster_workflow import resolved_config

    pool = audit_pool(root)
    corpus_path = root / "trajectory_memory" / "textcraft" / "corpus.json"
    index_path = root / "trajectory_memory" / "textcraft" / "index.json"
    corpus, index = read_json(corpus_path), read_json(index_path)
    corpus_hash = canonical_digest(corpus)
    index_hash = canonical_digest(index)
    if index.get("corpus_sha256") != corpus_hash:
        raise ValueError("Trajectory index does not match the frozen corpus")
    exclusions = set(corpus.get("excluded_manifest_sha256", {}).values())
    if not {pool["development_manifest_sha256"], pool["heldout_manifest_sha256"]} <= exclusions:
        raise ValueError("Trajectory corpus did not exclude both exact evaluation manifests")
    training = _manifest(root, "training")
    training_hashes = {task["task_id"]: task["observation_sha256"] for task in training["tasks"]}
    for record in corpus.get("records", []):
        if (record.get("benchmark") != "textcraft"
                or training_hashes.get(record.get("task_id")) != record.get("task_fingerprint")):
            raise ValueError("Trajectory corpus contains a task outside the frozen training manifest")
    if not corpus.get("source_runs") or any(source.get("benchmark") != "textcraft"
           or source.get("manifest_sha256") != pool["training_manifest_sha256"]
           for source in corpus.get("source_runs", [])):
        raise ValueError("Trajectory corpus source differs from the frozen training manifest")
    configs = {condition: resolved_config(root, "textcraft", condition) for condition in CONDITIONS}
    shared = {key: configs["baseline"].get(key) for key in
              ("model_id", "quantization", "seed", "history_window", "max_actions",
               "max_decisions", "max_new_tokens")}
    if any(any(config.get(key) != value for key, value in shared.items()) for config in configs.values()):
        raise ValueError("The three conditions do not share agent settings and budgets")
    return {"schema_version": 1, "protocol": "textcraft_heldout_abc_v1",
            "pool": {key: value for key, value in pool.items() if key != "prior_heldout_runs"},
            "conditions": list(CONDITIONS), "config_sha256": {key: canonical_digest(value)
                                                        for key, value in configs.items()},
            "shared_agent_settings": shared, "git_commit": _clean_commit(),
            "python_version": platform.python_version(), "package_versions": package_versions(),
            "additional_package_versions": additional_versions(),
            "model_revision": cached_model_revision(shared["model_id"]),
            "trajectory_corpus_sha256": corpus_hash, "trajectory_index_sha256": index_hash,
            "textcraft_recipe_order_sha256": hashlib.sha256(
                (REPO / "agent_support" / "textcraft_recipe_order.json").read_bytes()).hexdigest(),
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES}


def lock_path(root: Path) -> Path:
    return root / "heldout_evaluation" / "lock.json"


def check_task_replay(root: Path) -> int:
    """Check every frozen task against the installed recipe catalog without acting."""
    from .benchmarks import TextCraftEpisode

    tasks = _manifest(root, "heldout")["tasks"]
    for task in tasks:
        episode = TextCraftEpisode(task)
        try:
            episode.reset()
        finally:
            episode.close()
    return len(tasks)


def freeze(root: Path) -> dict:
    path = lock_path(root)
    snapshot = path.parent / "manifest.json"
    if path.exists() or snapshot.exists():
        raise FileExistsError(f"Evaluation snapshot already exists; it will not be overwritten: {path.parent}")
    pool = audit_pool(root)
    if pool["prior_heldout_runs"]:
        raise ValueError(f"Held-out tasks were already exposed in runs: {pool['prior_heldout_runs']}")
    payload = lock_payload(root)
    replay_checked = check_task_replay(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open("xb") as stream:
        stream.write((root / "manifests" / "textcraft" / "heldout.json").read_bytes())
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"lock": str(path), "manifest_snapshot": str(snapshot),
            "lock_sha256": canonical_digest(payload), **pool,
            "replay_checked_task_count": replay_checked,
            "model_revision": payload["model_revision"], "git_commit": payload["git_commit"]}


def checked_lock(root: Path) -> dict:
    path = lock_path(root)
    frozen = read_json(path)
    snapshot = path.parent / "manifest.json"
    if snapshot.read_bytes() != (root / "manifests" / "textcraft" / "heldout.json").read_bytes():
        raise ValueError("Frozen held-out manifest snapshot differs from the input manifest")
    current = lock_payload(root)
    if frozen != current:
        raise ValueError("Evaluation environment, manifests, code, model, or memory changed since freeze")
    return frozen


def run_condition(root: Path, condition: str) -> dict:
    from .cluster_workflow import resolved_config

    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported held-out condition: {condition}")
    frozen = checked_lock(root)
    from .cluster_workflow import preflight
    preflight(root, "textcraft", require_gpu=True,
              require_training=condition == "m2_trajectory_v1")
    manifest = read_json(lock_path(root).parent / "manifest.json")
    for path in (root / "results").glob("textcraft_*/run.json"):
        metadata = read_json(path)
        if (metadata.get("condition") == condition
                and metadata.get("selected_manifest_sha256") == frozen["pool"]["heldout_manifest_sha256"]):
            raise FileExistsError(f"This condition already has a held-out run: {path.parent}")
    config = resolved_config(root, "textcraft", condition)
    config["model_revision"] = frozen["model_revision"]
    config["evaluation_lock_sha256"] = canonical_digest(frozen)
    output = run(config, manifest, root / "results")
    return {"run_dir": str(output), "condition": condition,
            "manifest_sha256": frozen["pool"]["heldout_manifest_sha256"],
            "evaluation_lock_sha256": canonical_digest(frozen)}


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def paired_interval(left: list[bool], right: list[bool]) -> list[float]:
    n = len(left)
    differences = [int(b) - int(a) for a, b in zip(left, right)]
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(BOOTSTRAP_REPLICATES))
    return [round(samples[int(0.025 * BOOTSTRAP_REPLICATES)], 6),
            round(samples[int(0.975 * BOOTSTRAP_REPLICATES) - 1], 6)]


def find_evaluation_runs(root: Path, frozen: dict) -> dict[str, Path]:
    selected = {}
    for path in sorted((root / "results").glob("textcraft_*/run.json")):
        metadata = read_json(path)
        condition = metadata.get("condition")
        if (condition in CONDITIONS
                and metadata.get("selected_manifest_sha256") == frozen["pool"]["heldout_manifest_sha256"]):
            if condition in selected:
                raise ValueError(f"Multiple held-out {condition} runs found; select runs explicitly")
            if metadata.get("status") != "complete":
                raise ValueError(f"Held-out {condition} run is not complete: {path.parent}")
            selected[condition] = path.parent
    if set(selected) != set(CONDITIONS):
        raise ValueError(f"Expected one complete run per condition; found {sorted(selected)}")
    return selected


def compare_three(root: Path, runs: dict[str, str | Path] | None = None) -> dict:
    frozen = checked_lock(root)
    if runs is None:
        runs = find_evaluation_runs(root, frozen)
    loaded = {condition: load_run(directory) for condition, directory in runs.items()}
    if set(loaded) != set(CONDITIONS):
        raise ValueError("Provide exactly baseline, M2, and reconsideration runs")
    manifest = read_json(lock_path(root).parent / "manifest.json")
    for condition, (meta, config, snapshot, _) in loaded.items():
        if (meta.get("condition") != condition or config.get("condition") != condition
                or snapshot != manifest or config.get("evaluation_lock_sha256") != canonical_digest(frozen)
                or meta.get("evaluation_lock_sha256") != canonical_digest(frozen)):
            raise ValueError(f"{condition} run differs from the frozen held-out evaluation")
        unpinned_config = {key: value for key, value in config.items()
                           if key not in {"model_revision", "evaluation_lock_sha256"}}
        if (canonical_digest(unpinned_config) != frozen["config_sha256"][condition]
                or config.get("model_revision") != frozen["model_revision"]
                or meta.get("model_revision") != frozen["model_revision"]
                or meta.get("git_commit") != frozen["git_commit"]
                or meta.get("python_version") != frozen["python_version"]
                or meta.get("package_versions") != frozen["package_versions"]
                or meta.get("textcraft_recipe_order_sha256") != frozen["textcraft_recipe_order_sha256"]):
            raise ValueError(f"{condition} run environment or config differs from the evaluation lock")
    if (loaded["m2_trajectory_v1"][0].get("trajectory_corpus_sha256") != frozen["trajectory_corpus_sha256"]
            or loaded["m2_trajectory_v1"][0].get("trajectory_index_sha256") != frozen["trajectory_index_sha256"]):
        raise ValueError("M2 run used different trajectory memory than the evaluation lock")
    pair_specs = (("baseline_vs_m2", "baseline", "m2_trajectory_v1"),
                  ("baseline_vs_reconsideration", "baseline", "reconsideration_only_v1"),
                  ("m2_vs_reconsideration", "m2_trajectory_v1", "reconsideration_only_v1"))
    reports = {}
    for label, left, right in pair_specs:
        report = compare_trajectory(runs[left], runs[right])
        left_values = [bool(loaded[left][3][task["task_id"]]["success"]) for task in manifest["tasks"]]
        right_values = [bool(loaded[right][3][task["task_id"]]["success"]) for task in manifest["tasks"]]
        report["paired"]["success_rate_delta_95pct_paired_bootstrap_ci"] = paired_interval(left_values, right_values)
        for side, values in (("left", left_values), ("right", right_values)):
            report[side]["success_rate_95pct_wilson_ci"] = wilson(sum(values), len(values))
        reports[label] = report
    rows = []
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        row = {"task_id": task_id, "observation_sha256": task["observation_sha256"]}
        for condition in CONDITIONS:
            episode = loaded[condition][3][task_id]
            row[condition] = {key: episode[key] for key in
                              ("success", "status", "environment_actions", "model_decisions",
                               "input_tokens", "output_tokens", "inference_seconds", "parser_errors",
                               "failed_actions")}
            row[condition]["model_calls"] = episode.get("model_calls", episode["model_decisions"])
            row[condition]["retrieval_seconds"] = episode.get("retrieval_seconds", 0)
            row[condition]["support_calls"] = episode.get("support_calls", 0)
            row[condition]["support_parse_errors"] = episode.get("support_parse_errors", 0)
            row[condition]["actions_changed_by_support"] = episode.get("actions_changed_by_support", 0)
        rows.append(row)
    result = {"schema_version": 1, "protocol": frozen["protocol"],
              "manifest_sha256": frozen["pool"]["heldout_manifest_sha256"],
              "evaluation_lock_sha256": canonical_digest(frozen),
              "uncertainty": "95% Wilson success intervals; 10,000 paired task bootstrap resamples, fixed seed 20261109",
              "pairs": reports, "tasks": rows}
    destination = root / "heldout_evaluation" / "analysis.json"
    if destination.exists() and read_json(destination) != result:
        raise FileExistsError(f"Existing analysis differs: {destination}")
    if not destination.exists():
        write_json_atomic(destination, result)
    return {"analysis": str(destination), "manifest_sha256": result["manifest_sha256"],
            "successes": {condition: sum(row[condition]["success"] for row in rows) for condition in CONDITIONS},
            "pairs": {label: report["paired"] for label, report in reports.items()}}
