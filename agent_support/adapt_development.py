"""Freeze and verify paired TextCraft development configs for ADaPT.

This touches only a separate development directory. It never generates tasks,
loads a model, or accesses held-out results.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .analysis import SUPPORT_ONLY_CONFIG_KEYS
from .common import canonical_digest, read_json, validate_config, validate_manifest, write_json_atomic
from .heldout_evaluation import cached_model_revision
from .runner import git_commit


REPO = Path(__file__).resolve().parent.parent
CONDITIONS = {"baseline": "textcraft_adapt_control_baseline.json",
              "adapt_dynamic_replanning_v1": "textcraft_adapt_dynamic_replanning.json"}


def _clean_commit() -> str:
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True)
    if dirty.stdout.strip():
        raise RuntimeError("Commit tracked ADaPT code before freezing development configs")
    return git_commit()


def _manifest(root: Path) -> dict:
    manifest = read_json(root / "manifests" / "textcraft" / "development.json")
    validate_manifest(manifest)
    if manifest.get("partition") != "development" or manifest != read_json(
            REPO / "manifests" / "textcraft" / "development.json"):
        raise ValueError("Development manifest differs from the tracked snapshot")
    return manifest


def _configs(revision: str) -> dict[str, dict]:
    configs = {}
    for condition, filename in CONDITIONS.items():
        config = read_json(REPO / "configs" / filename)
        config["model_revision"] = revision
        validate_config(config, "textcraft")
        configs[condition] = config
    shared = [{key: value for key, value in item.items() if key not in SUPPORT_ONLY_CONFIG_KEYS}
              for item in configs.values()]
    if shared[0] != shared[1]:
        raise ValueError("Development control and ADaPT shared settings differ")
    return configs


def _payload(root: Path) -> tuple[dict, dict[str, dict]]:
    manifest = _manifest(root)
    revision = cached_model_revision("Qwen/Qwen2.5-7B-Instruct")
    configs = _configs(revision)
    payload = {"protocol": "adapt_development_v1", "git_commit": _clean_commit(),
               "manifest_sha256": canonical_digest(manifest),
               "task_count": len(manifest["tasks"]), "model_revision": revision,
               "config_sha256": {name: canonical_digest(config) for name, config in configs.items()}}
    return payload, configs


def prepare(root: Path) -> dict:
    payload, configs = _payload(root)
    directory = root / "adapt_development"
    paths = {directory / "protocol.json": payload}
    paths.update({directory / "configs" / f"{name}.json": config for name, config in configs.items()})
    for path, value in paths.items():
        if path.exists() and read_json(path) != value:
            raise FileExistsError(f"Existing development artifact differs: {path}")
    for path, value in paths.items():
        if not path.exists():
            write_json_atomic(path, value)
    return payload


def preflight(root: Path) -> dict:
    payload, configs = _payload(root)
    directory = root / "adapt_development"
    if read_json(directory / "protocol.json") != payload:
        raise ValueError("ADaPT development protocol differs from current code, manifest, or cached model")
    for name, expected in configs.items():
        if read_json(directory / "configs" / f"{name}.json") != expected:
            raise ValueError(f"ADaPT development config changed: {name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight"))
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    print((prepare if args.command == "prepare" else preflight)(args.data_root))


if __name__ == "__main__":
    main()
