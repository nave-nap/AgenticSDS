"""Thin adapters for the two original benchmark APIs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .common import digest_text
from .manifests import alfworld_data_root, load_alfworld_config, textcraft_environment


class TextCraftEpisode:
    def __init__(self, task: dict):
        self.task = task
        self.env = textcraft_environment()
        self.initial_observation = ""
        self.observation = ""
        self.actions: list[str] = []
        self.task_prompt = ""

    def reset(self) -> None:
        observed, _ = self.env.reset(seed=self.task["seed"])
        if observed != self.task["initial_observation"] or digest_text(observed) != self.task["observation_sha256"]:
            raise RuntimeError(f"TextCraft task identity changed: {self.task['task_id']}")
        self.initial_observation = self.observation = self.task_prompt = observed

    def step(self, action: str) -> dict:
        observed, reward, terminated, truncated, info = self.env.step(action)
        self.observation = observed
        failed = observed.startswith(("Could not", "Wrong item"))
        return {
            "observation": observed,
            "reward": float(reward),
            "done": bool(terminated or truncated),
            "success": bool(terminated and reward == 1),
            "truncated": bool(truncated),
            "failed_action": failed,
        }

    def close(self) -> None:
        self.env.close()


class ALFWorldEpisode:
    def __init__(self, task: dict, config_path: str):
        from alfworld.agents.environment import get_environment

        self.task = task
        self.config = load_alfworld_config(config_path)
        root = alfworld_data_root(self.config, task["split"])
        self.game_path = (root / task["game_relpath"]).resolve()
        if not self.game_path.is_relative_to(root):
            raise ValueError("Game path escapes the ALFWorld split")
        if hashlib.sha256(self.game_path.read_bytes()).hexdigest() != task["game_sha256"]:
            raise RuntimeError(f"ALFWorld game content changed: {task['task_id']}")
        factory = get_environment("AlfredTWEnv")(self.config, train_eval=task["split"])
        if str(self.game_path) not in {str(Path(path).resolve()) for path in factory.game_files}:
            raise RuntimeError(f"ALFWorld game is no longer in the selected split: {self.game_path}")
        factory.game_files = [str(self.game_path)]
        factory.num_games = 1
        self.env = factory.init_env(batch_size=1)
        self.observation = ""
        self.initial_observation = ""
        self.task_prompt = ""
        self.actions: list[str] = []

    def reset(self) -> None:
        observations, info = self.env.reset()
        actual = Path(info["extra.gamefile"][0]).resolve()
        if actual != self.game_path:
            raise RuntimeError(f"ALFWorld reset returned wrong game: {actual}")
        self.initial_observation = self.observation = observations[0]
        self.actions = list(info["admissible_commands"][0])
        match = re.search(r"Your task is to:\s*([^\n]+)", self.observation, flags=re.IGNORECASE)
        if not match:
            raise ValueError("Could not extract ALFWorld task")
        self.task_prompt = match.group(1).strip()

    def step(self, action: str) -> dict:
        observations, scores, dones, info = self.env.step([action])
        self.observation = observations[0]
        self.actions = list(info["admissible_commands"][0])
        return {
            "observation": self.observation,
            "score": float(scores[0]),
            "done": bool(dones[0]),
            "success": bool(info["won"][0]),
            "failed_action": self.observation.strip().lower() == "nothing happens.",
        }

    def close(self) -> None:
        self.env.close()


def make_episode(benchmark: str, task: dict, config: dict):
    if benchmark == "textcraft":
        return TextCraftEpisode(task)
    if benchmark == "alfworld":
        return ALFWorldEpisode(task, config["alfworld_config"])
    raise ValueError(f"Unknown benchmark: {benchmark}")
