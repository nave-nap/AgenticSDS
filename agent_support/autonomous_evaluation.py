"""CPU preflight and gated TextCraft autonomous-support evaluation runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

from .analysis import load_run
from .common import canonical_digest, read_json, validate_config, validate_manifest, write_json_atomic
from .runner import git_commit, package_versions, run


REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "manifests/textcraft/development.json"
CONFIGS = REPO / "configs/autonomous_evaluation"
ARMS = ("qwen_standalone", "qwen_reconsideration", "qwen_m2_200",
        "random_support", "autonomous", "fixed_reasoning_2")
STAGE_SIZES = {"smoke": 1, "pilot": 5, "full": 20}
FROZEN_MANIFEST_SHA256 = "4ec57f36416ffe7f97b004b67d107e94750fa20b889d95f580690f4dfca4900e"


def locked_manifest() -> dict:
    manifest = read_json(MANIFEST)
    validate_manifest(manifest)
    if (manifest.get("benchmark") != "textcraft" or manifest.get("partition") != "development" or
            len(manifest["tasks"]) != 20 or canonical_digest(manifest) != FROZEN_MANIFEST_SHA256):
        raise ValueError("Frozen 20-task TextCraft development manifest differs")
    return manifest


def resolved_config(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown evaluation arm: {arm}")
    config = read_json(CONFIGS / f"{arm}.json")
    artifact_root = os.environ.get("AGENT_SUPPORT_M2_200_ARTIFACT_DIR")
    if artifact_root:
        root = Path(artifact_root).expanduser()
        if not root.is_absolute():
            raise ValueError("AGENT_SUPPORT_M2_200_ARTIFACT_DIR must be absolute")
        if "trajectory_corpus" in config:
            config["trajectory_corpus"] = str(root / "corpus_200.json")
            config["trajectory_index"] = str(root / "index_200.json")
        memory = config.get("autonomous_support", {}).get("memory")
        if memory is not None:
            memory["corpus_path"] = str(root / "corpus_200.json")
            memory["index_path"] = str(root / "index_200.json")
    if "trajectory_corpus" in config:
        config["trajectory_corpus"] = str(Path(config["trajectory_corpus"]).expanduser().resolve())
        config["trajectory_index"] = str(Path(config["trajectory_index"]).expanduser().resolve())
    validate_config(config, "textcraft")
    return config


def preflight(arm: str) -> dict:
    manifest = locked_manifest()
    config = resolved_config(arm)
    selected_ids = {task["task_id"] for task in manifest["tasks"]}
    selected_hashes = {task["observation_sha256"] for task in manifest["tasks"]}

    def check_memory_structure(corpus: dict, index: dict) -> None:
        from .trajectory_memory import TrajectoryMemory
        if (corpus.get("kind") != "successful_trajectory_retrieval_only" or
                any(record.get("benchmark") != "textcraft" or
                    record.get("task_id") in selected_ids or
                    record.get("task_fingerprint") in selected_hashes
                    for record in corpus["records"])):
            raise ValueError("Frozen memory contains a non-TextCraft or development trajectory")
        descriptor = SimpleNamespace(model_id=index["embedding_model"],
                                     revision=index.get("embedding_revision"))
        TrajectoryMemory(corpus, index, descriptor, top_k=1,
                         token_budget=1, token_counter=lambda value: 1)

    memory = config.get("autonomous_support", {}).get("memory")
    if memory is not None:
        corpus = read_json(Path(memory["corpus_path"]).expanduser())
        index = read_json(Path(memory["index_path"]).expanduser())
        if (canonical_digest(corpus) != memory["corpus_sha256"] or
                canonical_digest(index) != memory["index_sha256"] or
                len(corpus.get("records", [])) != memory["expected_trajectories"] or
                FROZEN_MANIFEST_SHA256 not in corpus.get("excluded_manifest_sha256", {}).values()):
            raise ValueError("Autonomous frozen memory artifacts or exclusions differ")
        check_memory_structure(corpus, index)
    elif "trajectory_corpus" in config:
        corpus = read_json(config["trajectory_corpus"])
        index = read_json(config["trajectory_index"])
        if (index.get("corpus_sha256") != canonical_digest(corpus) or len(corpus.get("records", [])) != 200 or
                FROZEN_MANIFEST_SHA256 not in corpus.get("excluded_manifest_sha256", {}).values()):
            raise ValueError("M2-200 frozen memory artifacts or exclusions differ")
        check_memory_structure(corpus, index)
    return {"arm": arm, "manifest_sha256": canonical_digest(manifest),
            "config_sha256": canonical_digest(config), "task_count": len(manifest["tasks"]),
            "code_commit": git_commit(), "memory_checked": bool(memory or "trajectory_corpus" in config)}


def checked_stage_run(path: str | Path, arm: str, stage: str) -> dict:
    metadata, config, manifest, results = load_run(path)
    expected = locked_manifest()["tasks"][:STAGE_SIZES[stage]]
    if (metadata.get("condition") != resolved_config(arm)["condition"] or
            config != resolved_config(arm) or
            metadata.get("manifest_sha256") != FROZEN_MANIFEST_SHA256 or
            [task["task_id"] for task in manifest["tasks"]] != [task["task_id"] for task in expected] or
            metadata.get("limit") != (None if stage == "full" else STAGE_SIZES[stage]) or
            len(results) != STAGE_SIZES[stage]):
        raise ValueError(f"{stage} run does not match the frozen {arm} protocol")
    return {"run_id": metadata["run_id"], "run_path": str(Path(path).resolve()),
            "git_commit": metadata.get("git_commit"),
            "package_versions": metadata.get("package_versions"),
            "python_version": metadata.get("python_version"),
            "recipe_order_sha256": metadata.get("textcraft_recipe_order_sha256"),
            "successes": sum(bool(item["success"]) for item in results.values()),
            "parser_errors": sum(item["parser_errors"] for item in results.values()),
            "budget_stops": sum(
                event.get("termination_reason", "").endswith("budget")
                for item in results.values() for record in item["trace"]
                for event in [record.get("support_trace", {})])}


def require_clean_commit() -> str:
    commit = git_commit()
    if not commit:
        raise ValueError("GPU evaluation requires a Git commit")
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise ValueError("GPU evaluation requires a clean checkout; commit the protocol first")
    return commit


def make_gate(arm: str, smoke: str, pilot: str, output: str | Path,
              acknowledge_traces_reviewed: bool) -> dict:
    if not acknowledge_traces_reviewed:
        raise ValueError("Pilot traces must be reviewed before a full-run gate is written")
    smoke_record = checked_stage_run(smoke, arm, "smoke")
    pilot_record = checked_stage_run(pilot, arm, "pilot")
    if any(smoke_record[key] != pilot_record[key] for key in
           ("git_commit", "package_versions", "python_version", "recipe_order_sha256")):
        raise ValueError("Smoke and pilot code or runtime differ")
    if (smoke_record["git_commit"] != require_clean_commit() or
            smoke_record["package_versions"] != package_versions() or
            smoke_record["python_version"] != platform.python_version()):
        raise ValueError("Gate code or runtime differs from the completed pilot")
    gate = {"schema_version": 1, "arm": arm,
            "manifest_sha256": FROZEN_MANIFEST_SHA256,
            "config_sha256": preflight(arm)["config_sha256"],
            "git_commit": smoke_record["git_commit"],
            "package_versions": smoke_record["package_versions"],
            "python_version": smoke_record["python_version"],
            "recipe_order_sha256": smoke_record["recipe_order_sha256"],
            "smoke": smoke_record, "pilot": pilot_record,
            "traces_reviewed": True}
    destination = Path(output)
    if destination.name != f"{arm}.json":
        raise ValueError("Gate filename must match the evaluation arm")
    if destination.exists():
        raise FileExistsError(f"Evaluation gate already exists: {destination}")
    write_json_atomic(destination, gate)
    return gate


def run_stage(arm: str, stage: str, output_root: str | Path,
              smoke_run: str | None = None, gate_file: str | None = None) -> Path:
    commit = require_clean_commit()
    checks = preflight(arm)
    current_runtime = package_versions()
    recipe_hash = hashlib.sha256((REPO / "agent_support/textcraft_recipe_order.json").read_bytes()).hexdigest()
    if stage == "pilot":
        if smoke_run is None:
            raise ValueError("Pilot requires --smoke-run")
        smoke = checked_stage_run(smoke_run, arm, "smoke")
        if (smoke["git_commit"] != commit or smoke["package_versions"] != current_runtime or
                smoke["python_version"] != platform.python_version() or
                smoke["recipe_order_sha256"] != recipe_hash):
            raise ValueError("Smoke run code or runtime differs")
    elif stage == "full":
        if gate_file is None:
            raise ValueError("Full run requires a reviewed --gate-file")
        gate = read_json(gate_file)
        if (gate.get("arm") != arm or gate.get("manifest_sha256") != FROZEN_MANIFEST_SHA256 or
                gate.get("config_sha256") != checks["config_sha256"] or
                gate.get("git_commit") != commit or gate.get("traces_reviewed") is not True or
                gate.get("package_versions") != current_runtime or
                gate.get("python_version") != platform.python_version() or
                gate.get("recipe_order_sha256") != recipe_hash):
            raise ValueError("Full-run gate differs from the current frozen protocol")
        checked_stage_run(gate["smoke"]["run_path"], arm, "smoke")
        checked_stage_run(gate["pilot"]["run_path"], arm, "pilot")
        for companion_arm in ARMS:
            companion_path = Path(gate_file).parent / f"{companion_arm}.json"
            companion = read_json(companion_path)
            if (companion.get("arm") != companion_arm or
                    companion.get("manifest_sha256") != FROZEN_MANIFEST_SHA256 or
                    companion.get("config_sha256") != preflight(companion_arm)["config_sha256"] or
                    companion.get("git_commit") != commit or
                    companion.get("package_versions") != current_runtime or
                    companion.get("python_version") != platform.python_version() or
                    companion.get("recipe_order_sha256") != recipe_hash or
                    companion.get("traces_reviewed") is not True):
                raise ValueError(f"Companion arm gate is missing or differs: {companion_arm}")
    directory = Path(output_root) / stage / arm
    directory.mkdir(parents=True, exist_ok=True)
    return run(resolved_config(arm), locked_manifest(), directory,
               limit=None if stage == "full" else STAGE_SIZES[stage])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--arm", choices=ARMS, required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--arm", choices=ARMS, required=True)
    launch.add_argument("--stage", choices=STAGE_SIZES, required=True)
    launch.add_argument("--output-root", required=True)
    launch.add_argument("--smoke-run")
    launch.add_argument("--gate-file")
    gate = sub.add_parser("gate")
    gate.add_argument("--arm", choices=ARMS, required=True)
    gate.add_argument("--smoke-run", required=True)
    gate.add_argument("--pilot-run", required=True)
    gate.add_argument("--output", required=True)
    gate.add_argument("--acknowledge-traces-reviewed", action="store_true")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.arm)
    elif args.command == "run":
        result = {"run_dir": str(run_stage(args.arm, args.stage, args.output_root,
                                           args.smoke_run, args.gate_file))}
    else:
        result = make_gate(args.arm, args.smoke_run, args.pilot_run, args.output,
                           args.acknowledge_traces_reviewed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
