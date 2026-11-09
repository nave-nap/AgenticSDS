from types import SimpleNamespace

import pytest

from agent_support.benchmarks import TextCraftEpisode
from agent_support.common import digest_text


def task():
    observation = "Crafting commands:\ncraft 1 plank using 1 wood\n\nGoal: craft plank."
    return {"task_id": "textcraft:seed:51", "seed": 51, "initial_observation": observation,
            "observation_sha256": digest_text(observation)}


class FakeTextCraft:
    def __init__(self, recipe="craft 1 plank using 1 wood"):
        self.crafting_tree = SimpleNamespace(
            itemid_recipes={"minecraft:plank": [SimpleNamespace(recipe_str=recipe)]},
            tag_recipes={},
        )
        self.inventory = {"stale": 1}
        self.goal = "minecraft:wrong_goal"

    def reset(self, seed):
        raise AssertionError("Replay must not regenerate a seed-based task")

    def close(self):
        pass


def test_manifest_replay_sets_exact_task_without_seed_generation(monkeypatch):
    env = FakeTextCraft()
    monkeypatch.setattr("agent_support.benchmarks.textcraft_environment", lambda: env)
    episode = TextCraftEpisode(task())
    episode.reset()
    assert episode.initial_observation == task()["initial_observation"]
    assert env.goal == "minecraft:plank"
    assert env.inventory == {}


def test_manifest_replay_rejects_recipe_catalog_drift(monkeypatch):
    monkeypatch.setattr("agent_support.benchmarks.textcraft_environment", lambda: FakeTextCraft("craft 1 plank using 2 wood"))
    episode = TextCraftEpisode(task())
    with pytest.raises(RuntimeError, match="incompatible with installed recipes"):
        episode.reset()
