"""CPU-only delegation, accounting, policy, and checkpoint tests."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from agent_support.common import read_json, validate_config
from agent_support.components import load_components
from agent_support.runner import run, run_episode
from agent_support.support_policy import TurnContext, make_support_policy, selection_decision


class Episode:
    def __init__(self, failures=()):
        self.task_prompt = self.initial_observation = self.observation = "Goal: craft plank."
        self.failures = failures
        self.actions = []
        self.steps = 0

    def reset(self):
        pass

    def step(self, action):
        self.actions.append(action)
        failed = self.failures[self.steps] if self.steps < len(self.failures) else False
        self.steps += 1
        self.observation = "Could not craft" if failed else "Crafted successfully"
        return {"observation": self.observation, "done": False, "success": False,
                "failed_action": failed}

    def close(self):
        pass


class Models:
    primary_model_type = "qwen"
    delegated_model_type = "gpt_oss_20b"

    def __init__(self, outputs=None, fail_on=None):
        self.calls = []
        self.outputs = outputs or {}
        self.fail_on = fail_on
        self.seeds = []
        self.closed = False

    def metadata(self):
        return {"model_id": "Qwen/Qwen2.5-7B-Instruct",
                "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                "workers": {"primary": {"metadata": {"model_revision": "a09a35458c702b33eeacc393d103063234e8bc28"}},
                            "delegated": {"metadata": {"model_revision": "6cee5e81ee83917806bbde320786a8fb61efebee"}}}}

    def generate_for(self, kind, messages):
        self.calls.append((kind, copy.deepcopy(messages)))
        if self.fail_on == len(self.calls):
            raise RuntimeError("simulated interruption")
        output = self.outputs.get(len(self.calls), "Action: inventory")
        if kind == "gpt_oss_20b":
            return {"text": output, "input_tokens": 10, "output_tokens": 4,
                    "reasoning_tokens": 2, "final_tokens": 2, "inference_seconds": 1.5,
                    "generation_status": "complete", "generation_error": None,
                    "raw_completion": "reasoning and final",
                    "peak_gpu_memory_allocated_bytes": 100,
                    "peak_gpu_memory_reserved_bytes": 150,
                    "worker_cuda_memory": {"peak_allocated_bytes": 100,
                                           "peak_reserved_bytes": 150},
                    "combined_worker_cuda_allocated_bytes": 120,
                    "combined_worker_cuda_reserved_bytes": 180}
        return {"text": output, "input_tokens": 3, "output_tokens": 2,
                "inference_seconds": 0.1,
                "worker_cuda_memory": {"peak_allocated_bytes": 20,
                                       "peak_reserved_bytes": 30},
                "combined_worker_cuda_allocated_bytes": 120,
                "combined_worker_cuda_reserved_bytes": 180}

    def count_messages_tokens_for(self, kind, messages):
        return 10

    def max_new_tokens_for(self, kind):
        return 4

    def begin_episode(self, task_id):
        self.seeds.append(task_id)

    def close(self):
        self.closed = True


def config(policy=None, cap=3, fallback=False, **extra):
    base = read_json("configs/hybrid_textcraft_dev_smoke_always.json")
    base.update(max_actions=3, max_decisions=3, max_model_calls=3,
                max_support_calls=cap, max_support_tokens=100,
                delegation_fallback_to_primary=fallback,
                selection_policy=policy or {"id": "always_v1",
                                            "support_id": "stronger_model_delegation",
                                            "phase": "before_proposal"})
    base.update(extra)
    base["worker_models"]["delegated"]["model"]["max_new_tokens"] = 4
    return base


def episode_run(config_value, model=None, failures=()):
    episode = Episode(failures)
    result = run_episode("textcraft", {"task_id": "task-a"}, config_value,
                         model or Models(), episode_factory=lambda *args: episode)
    return result, episode


def test_never_uses_only_qwen_and_returns_to_policy():
    model = Models()
    result, episode = episode_run(config({"id": "never_v1"}, cap=0), model)
    assert len(model.calls) == 3 and all(kind == "qwen" for kind, _ in model.calls)
    assert result["support_calls"] == result["delegation_attempts"] == 0
    assert len(episode.actions) == 3
    assert all(not item["support_policy"]["before_decision"]["invoke"] for item in result["trace"])


def test_delegation_replaces_qwen_generation_and_logs_budgets():
    model = Models()
    result, episode = episode_run(config(cap=1), model)
    assert [kind for kind, _ in model.calls] == ["gpt_oss_20b", "qwen", "qwen"]
    assert len(episode.actions) == 3
    assert result["delegation_selections"] == 3
    assert result["delegation_attempts"] == result["support_calls"] == 1
    assert result["blocked_delegations"] == 2
    assert result["valid_delegated_actions"] == 1
    assert result["delegated_actions_without_failure"] == 1
    assert result["model_calls"] == 3
    assert result["model_usage"]["gpt_oss_20b"]["model_calls"] == 1
    assert result["peak_combined_worker_cuda_reserved_bytes"] == 180
    assert result["trace"][1]["support_policy"]["blocked"][0]["reason"] == "support_call_budget"
    assert result["trace"][0]["generation_attempts"][0]["reasoning_tokens"] == 2
    assert result["trace"][0]["remaining_delegations"] == 0


def test_failed_actions_trigger_only_after_two_executed_failures():
    policy = {"id": "failure_triggered_v1", "support_id": "stronger_model_delegation",
              "phase": "before_proposal", "threshold": 2, "cooldown": 0}
    model = Models()
    result, _ = episode_run(config(policy, max_actions=4, max_decisions=4,
                                   max_model_calls=4), model, failures=(True, True, False, False))
    assert [kind for kind, _ in model.calls] == ["qwen", "qwen", "gpt_oss_20b", "qwen"]
    assert [step["support_policy"]["before_decision"]["reason"] for step in result["trace"]] == [
        "failure_threshold_not_met", "failure_threshold_not_met", "failure_threshold_met",
        "failure_threshold_not_met"]
    assert result["failed_actions"] == 2


def test_fallback_charges_both_calls_and_executes_at_most_one_action():
    model = Models(outputs={1: "not an action"})
    result, episode = episode_run(config(cap=1, fallback=True), model)
    assert len(model.calls) == result["model_calls"] == 3
    assert [kind for kind, _ in model.calls] == ["gpt_oss_20b", "qwen", "qwen"]
    assert result["fallback_calls"] == 1 and result["support_calls"] == 1
    assert result["input_tokens"] == 16 and result["output_tokens"] == 8
    assert result["reasoning_tokens"] == 2 and result["final_tokens"] == 2
    assert result["inference_seconds"] == 1.7
    assert len(episode.actions) == 2
    first = result["trace"][0]
    assert first["fallback_occurred"] and len(first["generation_attempts"]) == 2
    assert first["input_tokens"] == 13 and first["output_tokens"] == 6
    assert first["generation_attempts"][0]["parse_valid"] is False
    assert first["generation_attempts"][1]["parse_valid"] is True
    assert first["actual_generation_model"] == "qwen"


def test_fallback_blocked_by_total_call_budget():
    model = Models(outputs={1: "not an action"})
    result, episode = episode_run(config(cap=1, fallback=True, max_model_calls=1), model)
    assert len(model.calls) == 1 and not episode.actions
    assert result["trace"][0]["fallback_blocked"] == "model_call_budget"
    assert result["delegation_attempts"] == 1 and result["fallback_calls"] == 0
    assert result["failed_actions"] == 0 and result["parser_errors"] == 1


def test_delegation_token_budget_blocks_before_generation():
    model = Models()
    result, _ = episode_run(config(cap=3, max_support_tokens=11), model)
    assert all(kind == "qwen" for kind, _ in model.calls)
    assert result["blocked_delegations"] == 3 and result["delegation_attempts"] == 0
    assert result["trace"][0]["support_policy"]["blocked"][0]["reason"] == "support_token_budget"


def test_actual_tokens_above_reservation_fail_visibly():
    class Undercounting(Models):
        def count_messages_tokens_for(self, kind, messages):
            return 8
    with pytest.raises(RuntimeError, match="Actual delegated tokens exceeded"):
        episode_run(config(cap=1), Undercounting())


def test_generation_failure_is_not_environment_failure():
    class Malformed(Models):
        def generate_for(self, kind, messages):
            result = super().generate_for(kind, messages)
            if kind == "gpt_oss_20b":
                result.update(text="", generation_status="missing_final",
                              generation_error="missing_final")
            return result
    result, episode = episode_run(config(cap=1, max_model_calls=1), Malformed())
    assert result["generation_failures"] == 1
    assert result["parser_errors"] == 1 and result["failed_actions"] == 0
    assert not episode.actions
    assert result["trace"][0]["generation_attempts"][0]["generation_status"] == "missing_final"


def test_random_delegation_is_task_seeded_and_schedule_independent():
    policy = {"id": "random_v1", "support_id": "stronger_model_delegation",
              "phase": "before_proposal", "probability": 0.25, "seed": 987}
    a, _ = episode_run(config(policy), Models())
    b, _ = episode_run(config(policy), Models())
    assert [step["support_policy"]["before_decision"]["reason"] for step in a["trace"]] == [
        step["support_policy"]["before_decision"]["reason"] for step in b["trace"]]
    context = TurnContext("before_proposal", "textcraft", "modular_v1", 1,
                          "task", "observation", (), (), 0, 30, 30, 0,
                          available_supports=("stronger_model_delegation",))
    first = make_support_policy({"selection_policy": policy}, "task-a")
    expected = [selection_decision(first, replace(context, decision=i)).reason for i in range(1, 15)]
    other = make_support_policy({"selection_policy": policy}, "task-b")
    _ = [selection_decision(other, replace(context, decision=i)) for i in range(1, 15)]
    resumed = make_support_policy({"selection_policy": policy}, "task-a")
    assert expected == [selection_decision(resumed, replace(context, decision=i)).reason
                        for i in range(1, 15)]


def test_delegated_action_enters_qwen_history():
    model = Models(outputs={1: "Action: get 1 wood", 2: "Action: inventory"})
    result, _ = episode_run(config(cap=1), model)
    assert "get 1 wood" in str(model.calls[1][1])
    assert "Crafted successfully" in str(model.calls[1][1])
    assert result["trace"][0]["actual_generation_model"] == "gpt_oss_20b"


def test_configs_validate_and_match_worker_settings():
    for path in Path("configs").glob("hybrid_textcraft_dev_*.json"):
        item = read_json(path)
        load_components(item)
        validate_config(item, "textcraft")
        changed = copy.deepcopy(item)
        changed["worker_models"]["delegated"]["model"]["model_revision"] = "bad"
        with pytest.raises(ValueError):
            validate_config(changed, "textcraft")


def test_resume_keeps_completed_hybrid_episode_immutable(tmp_path):
    item = read_json("configs/hybrid_textcraft_dev_smoke_always.json")
    class DoneEpisode(Episode):
        def step(self, action):
            result = super().step(action)
            result.update(done=True, success=True)
            return result
    factory = lambda *args: DoneEpisode()
    first = Models(fail_on=2)
    root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(item, read_json(item["manifest"]), root, limit=2, model=first,
            episode_factory=factory, run_key="hybrid_test")
    assert first.closed
    directory = root / "textcraft_hybrid_test"
    saved = next((directory / "episodes").glob("*.json"))
    original = saved.read_bytes()
    changed = copy.deepcopy(item)
    changed["worker_models"]["delegated"]["model"]["reasoning_effort"] = "medium"
    with pytest.raises(ValueError, match="snapshot"):
        run(changed, read_json(item["manifest"]), root, limit=2, model=Models(),
            episode_factory=factory, run_key="hybrid_test", resume=True)
    resumed = Models()
    run(item, read_json(item["manifest"]), root, limit=2, model=resumed,
        episode_factory=factory, run_key="hybrid_test", resume=True)
    assert resumed.closed and resumed.seeds == [read_json(item["manifest"])["tasks"][1]["task_id"]]
    assert saved.read_bytes() == original
    meta = read_json(directory / "run.json")
    assert meta["status"] == "complete" and meta["model_metadata"]["workers"]["delegated"]["metadata"]["model_revision"]
    with pytest.raises(ValueError, match="complete"):
        run(changed, read_json(item["manifest"]), root, limit=2, model=Models(),
            episode_factory=factory, run_key="hybrid_test", resume=True)
