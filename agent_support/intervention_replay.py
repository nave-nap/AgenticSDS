"""CPU TextCraft replay checks for paired support-intervention studies."""

from __future__ import annotations

import argparse
import contextlib
import io
import json

from .analysis import load_run
from .benchmarks import TextCraftEpisode
from .common import canonical_digest


OUTCOME_FIELDS = ("observation", "reward", "done", "success", "failed_action")


def replay_prefix(task: dict, transitions: list[dict], episode_factory=TextCraftEpisode):
    """Return a fresh live episode only after every recorded transition matches."""
    episode = episode_factory(task)
    try:
        episode.reset()
        for position, transition in enumerate(transitions):
            if episode.observation != transition["input_observation"]:
                raise ValueError(f"Replay input observation differs at transition {position}")
            with contextlib.redirect_stdout(io.StringIO()):
                observed = episode.step(transition["action"])
            expected = transition["outcome"]
            if any(observed.get(field) != expected.get(field) for field in OUTCOME_FIELDS):
                raise ValueError(f"Replay outcome differs at transition {position}")
            if observed["observation"] != transition["output_observation"]:
                raise ValueError(f"Replay output observation differs at transition {position}")
        return episode
    except BaseException:
        episode.close()
        raise


def branch_first_actions(task: dict, transitions: list[dict], actions: dict[str, str],
                         episode_factory=TextCraftEpisode) -> dict:
    """Test validated candidate actions from separately replayed identical prefixes."""
    if "no_support" not in actions or not actions or any(not isinstance(value, str) or not value
                                                         for value in actions.values()):
        raise ValueError("Branch actions must include a nonempty no_support action")
    results = {}
    state_digest = None
    for label, action in actions.items():
        episode = replay_prefix(task, transitions, episode_factory)
        try:
            simulator = getattr(episode, "env", None)
            if simulator is not None:
                current_digest = canonical_digest({"goal": simulator.goal,
                                                   "inventory": simulator.inventory})
                if state_digest is not None and current_digest != state_digest:
                    raise ValueError("Replayed branches do not share an identical simulator state")
                state_digest = current_digest
            from .policy import parse_textcraft
            if parse_textcraft(action) != action:
                raise ValueError(f"Invalid TextCraft branch action: {label}")
            with contextlib.redirect_stdout(io.StringIO()):
                results[label] = episode.step(action)
        finally:
            episode.close()
    return results


def verify_run(directory: str) -> dict:
    metadata, _, manifest, episodes = load_run(directory)
    if manifest.get("benchmark") != "textcraft":
        raise ValueError("Replay audit currently supports TextCraft only")
    checked = 0
    for task in manifest["tasks"]:
        transitions = [record for record in episodes[task["task_id"]]["trace"]
                       if "outcome" in record]
        episode = replay_prefix(task, transitions)
        checked += len(transitions)
        episode.close()
    return {"run_id": metadata["run_id"], "tasks": len(manifest["tasks"]),
            "executed_actions_verified": checked, "mismatches": 0,
            "scope": "environment replay only; model sampling not replayed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run), indent=2))


if __name__ == "__main__":
    main()
