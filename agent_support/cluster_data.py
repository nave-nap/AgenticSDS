"""Stage the frozen Colab inputs for transfer to a Slurm cluster.

This command runs on the user's computer. It never rewrites the source archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .analysis import load_run
from .common import canonical_digest, read_json, validate_manifest
from .trajectory_memory import build_trajectory_corpus


REPO = Path(__file__).resolve().parent.parent
# Recorded by the original training-only corpus in the user's archived
# memory/alfworld/corpus.json. This identifies the exact held-out set.
ALFWORLD_HELDOUT_SHA256 = "b412a39c42566e1dbb2132f4499064affc8762de8578c22f70c7925a7cff13b0"


def _extract_prefix(archive: str | Path, prefix: str, output: Path) -> None:
    found = False
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            if not entry.filename.startswith(prefix + "/") or entry.is_dir():
                continue
            destination = output.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry) as reader, destination.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
            found = True
    if not found:
        raise ValueError(f"Archive has no {prefix}/ directory: {archive}")


def prepare(training_archive: str | Path, baseline_archive: str | Path,
            alfworld_heldout: str | Path, output: str | Path) -> dict:
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing data bundle: {output}")
    frozen = read_json(REPO / "docs" / "baseline_pilot.json")["runs"]
    heldout = read_json(alfworld_heldout)
    validate_manifest(heldout)
    if heldout["benchmark"] != "alfworld" or heldout.get("partition") != "heldout":
        raise ValueError("Expected the original ALFWorld held-out manifest")
    if canonical_digest(heldout) != ALFWORLD_HELDOUT_SHA256:
        raise ValueError("ALFWorld held-out manifest differs from the original frozen manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agent-support-cluster-", dir=output.parent) as temporary:
        root = Path(temporary) / "agent_support_data"
        root.mkdir()
        _extract_prefix(training_archive, "training_runs", root)
        for benchmark in ("alfworld", "textcraft"):
            run_id = frozen[benchmark]["run_id"]
            _extract_prefix(baseline_archive, f"results/{benchmark}_{run_id}", root)
            source_dir = root / "results" / f"{benchmark}_{run_id}"
            metadata, _, baseline_manifest, _ = load_run(source_dir)
            if (metadata["run_id"] != run_id or metadata["condition"] != "baseline"
                    or canonical_digest(baseline_manifest) != frozen[benchmark]["manifest_sha256"]):
                raise ValueError(f"Frozen {benchmark} baseline archive does not match docs/baseline_pilot.json")
            manifest_dir = root / "manifests" / benchmark
            manifest_dir.mkdir(parents=True)
            development = REPO / "manifests" / benchmark / "development.json"
            if canonical_digest(read_json(development)) != frozen[benchmark]["manifest_sha256"]:
                raise ValueError(f"Repository {benchmark} development manifest differs from frozen pilot")
            shutil.copy2(development, manifest_dir / "development.json")
            if benchmark == "alfworld":
                shutil.copy2(alfworld_heldout, manifest_dir / "heldout.json")
                snapshot = source_dir / "alfworld_config.yaml"
                if hashlib.sha256(snapshot.read_bytes()).hexdigest() != metadata["alfworld_config_sha256"]:
                    raise ValueError("Frozen ALFWorld YAML snapshot changed")
                shutil.copy2(snapshot, root / "alfworld_config.yaml")
            else:
                shutil.copy2(REPO / "manifests" / "textcraft" / "heldout.json",
                             manifest_dir / "heldout.json")
        training = {}
        for benchmark in ("alfworld", "textcraft"):
            matches = []
            for directory in (root / "training_runs").glob(f"{benchmark}_*"):
                metadata, config, manifest, _ = load_run(directory)
                if (metadata.get("condition") == "baseline" and config.get("condition") == "baseline"
                        and manifest.get("partition") == "training"
                        and metadata.get("limit") is None):
                    matches.append((directory, manifest))
            if len(matches) != 1:
                raise ValueError(f"Expected one complete baseline training run for {benchmark}; found {len(matches)}")
            directory, manifest = matches[0]
            (root / "manifests" / benchmark / "training.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            # The same production leakage check validates the transferred inputs.
            corpus = build_trajectory_corpus(
                [directory], [root / "manifests" / benchmark / "development.json",
                              root / "manifests" / benchmark / "heldout.json"])
            training[benchmark] = {"run_id": directory.name, "successful_trajectories": len(corpus["records"])}
        root.rename(output)
        os.chmod(output, 0o700)
    return {"data_root": str(output), "training": training,
            "baseline_runs": {key: frozen[key]["run_id"] for key in frozen}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-archive", required=True)
    parser.add_argument("--baseline-archive", required=True)
    parser.add_argument("--alfworld-heldout", required=True,
                        help="Original manifests/alfworld/heldout.json from Google Drive")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.training_archive, args.baseline_archive,
                             args.alfworld_heldout, args.output), indent=2))


if __name__ == "__main__":
    main()
