"""Read saved hybrid episodes to audit delegation budget decisions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from agent_support.common import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    metadata = read_json(run_dir / "run.json")
    config = read_json(run_dir / "config.json")
    episodes = sorted((run_dir / "episodes").glob("*.json"))
    print(f"run={run_dir} status={metadata['status']} saved_episodes={len(episodes)} "
          f"max_support_tokens={config['max_support_tokens']}")
    total = Counter()
    for path in episodes:
        episode = read_json(path)
        blocked = Counter()
        reservations = []
        for step in episode["trace"]:
            for entry in step["support_policy"].get("blocked", []):
                if entry["support_id"] == "stronger_model_delegation":
                    blocked[entry["reason"]] += 1
            for event in step.get("support_events", []):
                if "reserved_tokens_before_generation" in event:
                    reservations.append((event["reserved_tokens_before_generation"],
                                         event["actual_tokens_after_generation"]))
        total.update(blocked)
        used = episode["budgets"]["used"]["support_tokens"]
        print(f"{episode['task_id']}: selections={episode['delegation_selections']} "
              f"attempts={episode['delegation_attempts']} blocked={dict(blocked)} "
              f"support_tokens={used} remaining={config['max_support_tokens'] - used} "
              f"reservations={reservations}")
        if episode["delegation_selections"] != episode["delegation_attempts"] + sum(blocked.values()):
            raise RuntimeError(f"Selection accounting mismatch: {path}")
        if len(reservations) != episode["delegation_attempts"]:
            raise RuntimeError(f"Reservation count mismatch: {path}")
        if any(actual > reserved for reserved, actual in reservations):
            raise RuntimeError(f"Saved reservation overrun: {path}")
    print(f"total_blocked={dict(total)}")
    if total["support_token_budget"]:
        print("Earlier support-token budget blocks are present; inspect affected decisions.")
    else:
        print("No support-token budget blocks in saved episodes; selection was unaffected there.")
    if metadata["status"] != "complete":
        print("The unsaved interrupted episode cannot be audited from completed episode traces.")


if __name__ == "__main__":
    main()
