"""Build fixed development and held-out task manifests on a benchmark installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

from .common import digest_text, validate_manifest, write_json_atomic


SPLIT_KEYS = {
    "train": "data_path",
    "eval_in_distribution": "eval_id_data_path",
    "eval_out_of_distribution": "eval_ood_data_path",
}


@lru_cache(maxsize=1)
def _textcraft_recipe_order(recipe_dir: Path) -> tuple[str, ...]:
    from importlib.metadata import version

    frozen = json.loads(Path(__file__).with_name("textcraft_recipe_order.json").read_text(encoding="utf-8"))
    if version("textcraft") != frozen["textcraft_version"]:
        raise RuntimeError("TextCraft package version differs from the frozen recipe catalog")
    names = tuple(frozen["order"])
    if len(names) != len(set(names)) or set(names) != set(os.listdir(recipe_dir)):
        raise RuntimeError("TextCraft recipe files differ from the frozen catalog")
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((recipe_dir / name).read_bytes())
    if digest.hexdigest() != frozen["recipe_data_sha256"]:
        raise RuntimeError("TextCraft recipe contents differ from the frozen catalog")
    return names


def textcraft_environment():
    if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization != 0:
        raise RuntimeError("TextCraft requires a fresh Python process started with PYTHONHASHSEED=0")
    from importlib import resources
    from textcraft import TextCraft

    data_dir = Path(str(resources.files("textcraft").joinpath("data")))
    recipe_dir = (data_dir / "recipes").resolve()
    if not recipe_dir.is_dir():
        raise RuntimeError(f"TextCraft recipes missing at {data_dir}")
    order = _textcraft_recipe_order(recipe_dir)
    original_listdir = os.listdir

    def ordered_listdir(path):
        if isinstance(path, (str, os.PathLike)) and Path(path).resolve() == recipe_dir:
            return list(order)
        return original_listdir(path)

    # TextCraft's constructor reads recipes in os.listdir order. Freeze that
    # brief read to reproduce the catalog used by the committed task manifests.
    with patch.object(os, "listdir", side_effect=ordered_listdir):
        return TextCraft(minecraft_dir=str(data_dir))


def textcraft_observation(seed: int) -> str:
    env = textcraft_environment()
    try:
        observation, _ = env.reset(seed=seed)
        return observation
    finally:
        env.close()


def build_textcraft(start_seed: int, count: int) -> list[dict]:
    tasks = []
    for seed in range(start_seed, start_seed + count):
        first = textcraft_observation(seed)
        second = textcraft_observation(seed)
        if first != second:
            raise RuntimeError(f"TextCraft seed {seed} generated different tasks in fresh environments")
        tasks.append({
            "task_id": f"textcraft:seed:{seed}",
            "seed": seed,
            "initial_observation": first,
            "observation_sha256": digest_text(first),
        })
    return tasks


def load_alfworld_config(path: str | Path) -> dict:
    import yaml

    os.environ.setdefault("ALFWORLD_DATA", str(Path.home() / ".cache" / "alfworld"))
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["dataset"]["num_train_games"] = -1
    config["dataset"]["num_eval_games"] = -1
    config["env"]["domain_randomization"] = False
    return config


def alfworld_data_root(config: dict, split: str) -> Path:
    return Path(os.path.expandvars(config["dataset"][SPLIT_KEYS[split]])).expanduser().resolve()


def alfworld_game_records(config_path: str | Path, split: str) -> list[dict]:
    from alfworld.agents.environment import get_environment

    config = load_alfworld_config(config_path)
    root = alfworld_data_root(config, split)
    factory = get_environment("AlfredTWEnv")(config, train_eval=split)
    records = []
    for path_text in sorted(factory.game_files):
        path = Path(path_text).resolve()
        relative = path.relative_to(root).as_posix()
        records.append({
            "task_id": f"alfworld:{split}:{relative}",
            "split": split,
            "game_relpath": relative,
            "game_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return records


def write_splits(benchmark: str, tasks: list[dict], dev_count: int, heldout_count: int, output_dir: str | Path, seed: int) -> None:
    if dev_count < 1 or heldout_count < 1:
        raise ValueError("Development and held-out counts must be positive")
    if len(tasks) < dev_count + heldout_count:
        raise ValueError(f"Only {len(tasks)} tasks available; requested {dev_count + heldout_count}")
    # ALFWorld records arrive in filesystem order; shuffle reproducibly before slicing.
    selected = list(tasks)
    random.Random(seed).shuffle(selected)
    destination = Path(output_dir)
    for name, subset in (("development", selected[:dev_count]), ("heldout", selected[dev_count:dev_count + heldout_count])):
        manifest = {"schema_version": 1, "benchmark": benchmark, "partition": name, "selection_seed": seed, "tasks": subset}
        validate_manifest(manifest)
        output = destination / f"{name}.json"
        if output.exists():
            old = json.loads(output.read_text(encoding="utf-8"))
            if old != manifest:
                raise FileExistsError(f"Existing manifest differs: {output}. Move it deliberately before regenerating")
        else:
            write_json_atomic(output, manifest)
        print(f"{output}: {len(subset)} tasks")


def write_training_manifest(benchmark: str, count: int, output: str | Path,
                            excluded_paths: list[str | Path], seed: int = 43,
                            config_path: str | Path | None = None, start_seed: int = 1000) -> dict:
    """Choose distinct training tasks and reject overlap with evaluation manifests."""
    from .common import read_json

    if count < 1 or not excluded_paths:
        raise ValueError("Positive count and development/held-out exclusions are required")
    excluded_ids = set()
    excluded_hashes = set()
    partitions = set()
    key = "game_sha256" if benchmark == "alfworld" else "observation_sha256"
    for path in excluded_paths:
        manifest = read_json(path)
        validate_manifest(manifest)
        if manifest["benchmark"] != benchmark or manifest.get("partition") not in {"development", "heldout"}:
            raise ValueError(f"Wrong exclusion manifest: {path}")
        partitions.add(manifest["partition"])
        excluded_ids.update(task["task_id"] for task in manifest["tasks"])
        excluded_hashes.update(task[key] for task in manifest["tasks"])
    if partitions != {"development", "heldout"}:
        raise ValueError("Both development and held-out manifests are required")
    if benchmark == "alfworld":
        if config_path is None:
            raise ValueError("ALFWorld training manifest requires --config")
        candidates = alfworld_game_records(config_path, "train")
        random.Random(seed).shuffle(candidates)
    elif benchmark == "textcraft":
        # Generate enough candidates to skip accidental duplicate observations.
        candidates = build_textcraft(start_seed, count * 3)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    tasks = []
    chosen_hashes = set()
    for task in candidates:
        fingerprint = task[key]
        if task["task_id"] in excluded_ids or fingerprint in excluded_hashes or fingerprint in chosen_hashes:
            continue
        tasks.append(task)
        chosen_hashes.add(fingerprint)
        if len(tasks) == count:
            break
    if len(tasks) != count:
        raise ValueError(f"Only {len(tasks)} distinct training tasks available; requested {count}")
    manifest = {"schema_version": 1, "benchmark": benchmark, "partition": "training",
                "selection_seed": seed, "tasks": tasks}
    validate_manifest(manifest)
    target = Path(output)
    if target.exists():
        if read_json(target) != manifest:
            raise FileExistsError(f"Existing training manifest differs: {target}")
    else:
        write_json_atomic(target, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("alfworld", "textcraft"))
    parser.add_argument("--config", help="ALFWorld base_config.yaml")
    parser.add_argument("--split", choices=tuple(SPLIT_KEYS), default="eval_in_distribution")
    parser.add_argument("--dev-count", type=int, default=20)
    parser.add_argument("--heldout-count", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    count = args.dev_count + args.heldout_count
    if args.benchmark == "alfworld":
        if not args.config:
            parser.error("--config is required for ALFWorld")
        tasks = alfworld_game_records(args.config, args.split)
    else:
        tasks = build_textcraft(args.start_seed, count)
    write_splits(args.benchmark, tasks, args.dev_count, args.heldout_count, args.output_dir, args.selection_seed)


if __name__ == "__main__":
    main()
