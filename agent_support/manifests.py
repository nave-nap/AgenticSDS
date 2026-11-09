"""Build fixed development and held-out task manifests on a benchmark installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from .common import digest_text, validate_manifest, write_json_atomic


SPLIT_KEYS = {
    "train": "data_path",
    "eval_in_distribution": "eval_id_data_path",
    "eval_out_of_distribution": "eval_ood_data_path",
}


def textcraft_environment():
    if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization != 0:
        raise RuntimeError("TextCraft requires a fresh Python process started with PYTHONHASHSEED=0")
    from importlib import resources
    from textcraft import TextCraft

    data_dir = Path(str(resources.files("textcraft").joinpath("data")))
    if not (data_dir / "recipes").is_dir():
        raise RuntimeError(f"TextCraft recipes missing at {data_dir}")
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
