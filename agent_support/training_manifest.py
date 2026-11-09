"""Create a disjoint training manifest for memory collection."""

from __future__ import annotations

import argparse

from .manifests import write_training_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("alfworld", "textcraft"))
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-manifest", action="append", required=True)
    parser.add_argument("--config", help="ALFWorld base_config.yaml")
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--selection-seed", type=int, default=43)
    args = parser.parse_args()
    manifest = write_training_manifest(args.benchmark, args.count, args.output,
                                       args.exclude_manifest, seed=args.selection_seed,
                                       config_path=args.config, start_seed=args.start_seed)
    print(f"{args.output}: {len(manifest['tasks'])} distinct training tasks")


if __name__ == "__main__":
    main()
