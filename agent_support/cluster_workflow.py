"""Run the shared trajectory-memory experiment from a Slurm allocation.

Set AGENT_SUPPORT_DATA to the transferred data directory and PYTHONHASHSEED=0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

from .analysis import compare_trajectory, load_run
from .common import canonical_digest, read_json, validate_config, validate_manifest, write_json_atomic
from .runner import git_commit, package_versions, run
from .trajectory_memory import MPNetEmbedder, build_trajectory_corpus, build_trajectory_index


REPO = Path(__file__).resolve().parent.parent
CONDITION_CONFIGS = {"baseline": "baseline", "expel_retrieval_only_v1": "expel_retrieval",
                     "m2_trajectory_v1": "m2_trajectory"}


def training_run(root: Path, benchmark: str) -> Path:
    manifest = read_json(root / "manifests" / benchmark / "training.json")
    expected = canonical_digest(manifest)
    candidates = []
    for directory in (root / "training_runs").glob(f"{benchmark}_*"):
        metadata, config, snapshot, _ = load_run(directory)
        if (metadata.get("condition") == config.get("condition") == "baseline"
                and metadata.get("limit") is None and snapshot == manifest
                and metadata.get("selected_manifest_sha256") == expected):
            candidates.append(directory)
    if len(candidates) != 1:
        raise ValueError(f"Expected one matching full {benchmark} training run, found {len(candidates)}")
    return candidates[0]


def manifest_path(root: Path, benchmark: str, partition: str) -> Path:
    path = root / "manifests" / benchmark / f"{partition}.json"
    validate_manifest(read_json(path))
    return path


def frozen_baseline(root: Path, benchmark: str) -> Path:
    frozen = read_json(REPO / "docs" / "baseline_pilot.json")["runs"][benchmark]
    directory = root / "results" / f"{benchmark}_{frozen['run_id']}"
    metadata, config, manifest, _ = load_run(directory)
    if (metadata.get("condition") != config.get("condition") or config["condition"] != "baseline"
            or canonical_digest(manifest) != frozen["manifest_sha256"]):
        raise ValueError(f"Frozen {benchmark} baseline differs from its pilot record")
    return directory


def frozen_compatibility(root: Path, benchmark: str) -> list[str]:
    baseline = frozen_baseline(root, benchmark)
    metadata = read_json(baseline / "run.json")
    problems = []
    if metadata["python_version"] != platform.python_version():
        problems.append(f"Python {platform.python_version()} vs baseline {metadata['python_version']}")
    current = package_versions()
    for package, old_version in metadata["package_versions"].items():
        if current.get(package) != old_version:
            problems.append(f"{package} {current.get(package)} vs baseline {old_version}")
    if benchmark == "alfworld":
        source = root / "alfworld_config.yaml"
        if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["alfworld_config_sha256"]:
            problems.append("ALFWorld YAML differs from frozen baseline")
    return problems


def preflight(root: Path, benchmark: str, require_gpu: bool = False) -> dict:
    for partition in ("training", "development", "heldout"):
        manifest_path(root, benchmark, partition)
    training = training_run(root, benchmark)
    frozen_baseline(root, benchmark)
    if benchmark == "alfworld":
        data = Path(os.environ.get("ALFWORLD_DATA", str(root / "alfworld")))
        for relative in ("json_2.1.1/train", "json_2.1.1/valid_seen", "logic/alfred.pddl"):
            if not (data / relative).exists():
                raise FileNotFoundError(f"ALFWorld benchmark data missing: {data / relative}")
    if require_gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is unavailable in this job; request a Slurm GPU allocation")
    mismatches = frozen_compatibility(root, benchmark)
    home = Path.home().resolve()
    storage = {"data_root": root, "alfworld_data": Path(os.environ.get("ALFWORLD_DATA", str(root / "alfworld"))),
               "model_cache": Path(os.environ.get("HF_HOME", str(root / "hf_cache")))}
    home_paths = [name for name, path in storage.items() if path.expanduser().resolve().is_relative_to(home)]
    return {"benchmark": benchmark, "training_run": training.name,
            "frozen_baseline_compatible": not mismatches,
            "frozen_baseline_mismatches": mismatches,
            "home_resident_storage": home_paths}


def build_memory(root: Path, benchmark: str) -> dict:
    paths = [manifest_path(root, benchmark, partition) for partition in ("development", "heldout")]
    corpus = build_trajectory_corpus([training_run(root, benchmark)], paths)
    directory = root / "trajectory_memory" / benchmark
    directory.mkdir(parents=True, exist_ok=True)
    corpus_path = directory / "corpus.json"
    if corpus_path.exists() and read_json(corpus_path) != corpus:
        raise FileExistsError(f"Existing corpus differs: {corpus_path}")
    if not corpus_path.exists():
        write_json_atomic(corpus_path, corpus)
    index = build_trajectory_index(corpus, MPNetEmbedder())
    index_path = directory / "index.json"
    if index_path.exists() and read_json(index_path) != index:
        raise FileExistsError(f"Existing index differs: {index_path}")
    if not index_path.exists():
        write_json_atomic(index_path, index)
    return {"benchmark": benchmark, "successful_trajectories": len(corpus["records"]),
            "corpus_sha256": canonical_digest(corpus), "index_sha256": canonical_digest(index),
            "directory": str(directory)}


def resolved_config(root: Path, benchmark: str, condition: str) -> dict:
    if condition not in CONDITION_CONFIGS:
        raise ValueError(f"Unknown condition: {condition}")
    config = read_json(REPO / "configs" / f"{benchmark}_{CONDITION_CONFIGS[condition]}.json")
    if benchmark == "alfworld":
        config["alfworld_config"] = str((root / "alfworld_config.yaml").resolve())
    if condition != "baseline":
        directory = (root / "trajectory_memory" / benchmark).resolve()
        config["trajectory_corpus"] = str(directory / "corpus.json")
        config["trajectory_index"] = str(directory / "index.json")
        if not (directory / "corpus.json").exists() or not (directory / "index.json").exists():
            raise FileNotFoundError(f"Build trajectory memory first: {directory}")
    validate_config(config, benchmark)
    config_path = root / "configs" / f"{benchmark}_{condition}.json"
    if config_path.exists() and read_json(config_path) != config:
        raise FileExistsError(f"Existing cluster config differs: {config_path}")
    if not config_path.exists():
        write_json_atomic(config_path, config)
    return config


def run_condition(root: Path, benchmark: str, condition: str, smoke: bool) -> dict:
    checks = preflight(root, benchmark, require_gpu=True)
    if condition != "baseline" and not checks["frozen_baseline_compatible"]:
        print("Frozen baseline incompatible on this cluster; run a new cluster baseline before comparison:",
              checks["frozen_baseline_mismatches"], flush=True)
    config = resolved_config(root, benchmark, condition)
    manifest = read_json(manifest_path(root, benchmark, "development"))
    output = run(config, manifest, root / "results", limit=3 if smoke else None)
    metadata = read_json(output / "run.json")
    return {"condition": condition, "benchmark": benchmark, "run_dir": str(output),
            "status": metadata["status"], "selected_task_count": metadata["selected_task_count"]}


def latest_pilot(root: Path, benchmark: str, condition: str) -> Path:
    manifest_hash = canonical_digest(read_json(manifest_path(root, benchmark, "development")))
    candidates = []
    for directory in (root / "results").glob(f"{benchmark}_*"):
        metadata_path = directory / "run.json"
        if not metadata_path.exists():
            continue
        metadata = read_json(metadata_path)
        if (metadata.get("condition") == condition and metadata.get("status") == "complete"
                and metadata.get("limit") is None
                and metadata.get("selected_manifest_sha256") == manifest_hash
                and metadata.get("git_commit") == git_commit()):
            candidates.append(directory)
    if not candidates:
        raise ValueError(f"No full current-commit {benchmark} {condition} pilot found")
    return max(candidates, key=lambda path: path.name)


def compare_pilots(root: Path, benchmark: str, baseline_dir: str | Path | None = None) -> dict:
    baseline = Path(baseline_dir) if baseline_dir else frozen_baseline(root, benchmark)
    retrieval = latest_pilot(root, benchmark, "expel_retrieval_only_v1")
    m2 = latest_pilot(root, benchmark, "m2_trajectory_v1")
    output = root / "comparisons"
    output.mkdir(exist_ok=True)
    results = {}
    for label, left, right in (("A_B", baseline, retrieval), ("A_C", baseline, m2),
                               ("B_C", retrieval, m2)):
        report = compare_trajectory(left, right)
        destination = output / f"{benchmark}_{label}_{report['left']['run_id']}_{report['right']['run_id']}.json"
        if destination.exists() and read_json(destination) != report:
            raise FileExistsError(f"Existing comparison differs: {destination}")
        if not destination.exists():
            write_json_atomic(destination, report)
        results[label] = {"left_successes": report["left"]["successes"],
                          "right_successes": report["right"]["successes"],
                          "helped": report["paired"]["helped"],
                          "hurt": report["paired"]["hurt"], "file": str(destination)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "build-memory", "smoke", "pilot", "compare"))
    parser.add_argument("--data-root", default=os.environ.get("AGENT_SUPPORT_DATA"))
    parser.add_argument("--benchmark", choices=("alfworld", "textcraft"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITION_CONFIGS))
    parser.add_argument("--baseline-run", help="Full cluster baseline directory when frozen baseline versions differ")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set AGENT_SUPPORT_DATA or pass --data-root")
    root = Path(args.data_root).expanduser().resolve()
    if args.stage in {"smoke", "pilot"} and not args.condition:
        parser.error("--condition is required for smoke and pilot")
    if args.stage == "preflight":
        result = preflight(root, args.benchmark)
    elif args.stage == "build-memory":
        result = build_memory(root, args.benchmark)
    elif args.stage in {"smoke", "pilot"}:
        result = run_condition(root, args.benchmark, args.condition, args.stage == "smoke")
    else:
        result = compare_pilots(root, args.benchmark, args.baseline_run)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
