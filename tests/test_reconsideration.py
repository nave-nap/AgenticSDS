import json

import pytest

from agent_support.analysis import compare_trajectory
from agent_support.cluster_workflow import compare_pilots, resolved_config
from agent_support.common import digest_text, read_json, validate_config, write_json_atomic
from agent_support.policy import alfworld_messages, textcraft_messages
from agent_support.reconsideration import reconsideration_messages
from agent_support.runner import run, run_episode
from agent_support.support_policy import FixedConditionPolicy, RECONSIDERATION, TurnContext


class Episode:
    task_prompt = initial_observation = observation = "Goal: craft plank."
    actions = []

    def reset(self):
        pass

    def step(self, action):
        return {"observation": "Done", "done": True, "success": action == "get 1 log",
                "failed_action": False}

    def close(self):
        pass


class Model:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def generate(self, messages):
        self.messages.append(messages)
        return {"text": next(self.outputs), "input_tokens": 11, "output_tokens": 3,
                "inference_seconds": 0.2}


def config(condition="reconsideration_only_v1"):
    return {"benchmark": "textcraft", "condition": condition,
            "support_policy": "fixed_by_condition_v1", "model_id": "fake",
            "quantization": "nf4_4bit", "max_actions": 1, "max_decisions": 1,
            "history_window": 2, "max_new_tokens": 10, "seed": 42}


def manifest():
    observation = Episode.initial_observation
    return {"schema_version": 1, "benchmark": "textcraft", "partition": "development",
            "tasks": [{"task_id": "textcraft:seed:42", "seed": 42,
                       "initial_observation": observation,
                       "observation_sha256": digest_text(observation)}]}


def test_policy_and_prompt_use_current_state_only():
    context = TurnContext("after_proposal", "textcraft", "reconsideration_only_v1", 1,
                          "Goal: craft plank.", "Start", (), (), 0, 1, 1, 0,
                          proposed_action="get 1 plank")
    assert FixedConditionPolicy().select(context) == (RECONSIDERATION,)
    assert FixedConditionPolicy().select(TurnContext(**{**context.__dict__, "proposed_action": None})) == ()
    base = textcraft_messages("Goal: craft plank.", [], 2)
    revised = reconsideration_messages(base, "get 1 plank", "textcraft")
    assert base[-1]["content"].endswith("\nAction:")
    assert revised[-1]["content"].endswith("\nAction:")
    assert "Your proposed action was: get 1 plank" in revised[-1]["content"]
    assert "Past successful" not in revised[-1]["content"]
    alf = alfworld_messages("put apple", "Kitchen", ["take apple"], [], 2)
    assert reconsideration_messages(alf, "take apple", "alfworld")[-1]["content"].endswith(
        "Choose the best next action.")


def test_one_call_after_valid_proposal_and_metrics():
    model = Model(["Action: get 1 plank", "Action: get 1 log"])
    result = run_episode("textcraft", manifest()["tasks"][0], config(), model,
                         episode_factory=lambda *_: Episode())
    assert result["success"] and len(model.messages) == 2
    assert model.messages[0] == textcraft_messages(Episode.initial_observation, [], 2)
    assert (result["model_calls"], result["support_calls"], result["support_model_calls"]) == (2, 1, 1)
    assert (result["input_tokens"], result["output_tokens"], result["support_input_tokens"],
            result["support_output_tokens"]) == (22, 6, 11, 3)
    assert result["inference_seconds"] == 0.4
    assert result["actions_changed_by_support"] == 1
    event = result["trace"][0]["support"]
    assert event["support_id"] == RECONSIDERATION
    assert event["retrieval_performed"] is False and event["retrieval_event"] is False
    assert event["retrieved"] == [] and event["retrieval_seconds"] == 0
    assert event["action_changed"] is True


def test_invalid_revision_falls_back_and_invalid_proposal_skips_support():
    model = Model(["Action: get 1 log", "I cannot decide"])
    result = run_episode("textcraft", manifest()["tasks"][0], config(), model,
                         episode_factory=lambda *_: Episode())
    assert result["success"] and result["trace"][0]["action"] == "get 1 log"
    assert result["support_parse_errors"] == 1 and result["parser_errors"] == 0
    assert result["actions_changed_by_support"] == 0
    assert result["trace"][0]["support"]["parse_error"] is True
    invalid = Model(["not a command"])
    result = run_episode("textcraft", manifest()["tasks"][0], config(), invalid,
                         episode_factory=lambda *_: Episode())
    assert len(invalid.messages) == 1 and result["support_calls"] == 0
    assert result["parser_errors"] == 1 and result["environment_actions"] == 0


def test_alfworld_invalid_revision_executes_original_numbered_action():
    class AlfEpisode(Episode):
        task_prompt = "put apple on table"
        initial_observation = observation = "Kitchen"
        actions = ["take apple", "go to table"]

        def step(self, action):
            return {"observation": "Done", "done": True, "success": action == "take apple",
                    "failed_action": False}

    model = Model(["1", "unparseable"])
    result = run_episode("alfworld", {"task_id": "alfworld:one"},
                         {**config(), "benchmark": "alfworld"}, model,
                         episode_factory=lambda *_: AlfEpisode())
    assert result["success"] and result["trace"][0]["action"] == "take apple"
    assert result["support_parse_errors"] == 1 and result["parser_errors"] == 0
    assert len(model.messages) == 2


def test_run_never_loads_memory_and_cluster_config_needs_no_index(monkeypatch, tmp_path):
    monkeypatch.setattr("agent_support.runner.MPNetEmbedder", lambda *_args, **_kw: pytest.fail("memory loaded"))
    monkeypatch.setattr("agent_support.runner.TrajectoryMemory", lambda *_args, **_kw: pytest.fail("memory loaded"))
    model = Model(["Action: get 1 plank", "Action: get 1 log"])
    directory = run(config(), manifest(), tmp_path / "results", model=model,
                    episode_factory=lambda *_: Episode())
    metadata = json.loads((directory / "run.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert metadata["status"] == "complete" and metadata["support_policy_id"] == "fixed_by_condition_v1"
    assert not any("memory" in key or "trajectory" in key for key in metadata)
    assert summary["support_calls"] == summary["support_model_calls"] == 1
    assert summary["retrieval_seconds"] == 0
    resolved = resolved_config(tmp_path, "textcraft", "reconsideration_only_v1")
    assert resolved["condition"] == "reconsideration_only_v1"
    assert not any("memory" in key or "trajectory" in key for key in resolved)
    with pytest.raises(ValueError, match="cannot configure memory"):
        validate_config({**config(), "trajectory_corpus": "hidden"}, "textcraft")


@pytest.mark.parametrize("benchmark", ["alfworld", "textcraft"])
def test_condition_configs_match_m2_agent_settings(benchmark):
    m2 = read_json(f"configs/{benchmark}_m2_trajectory.json")
    d = read_json(f"configs/{benchmark}_reconsideration_only.json")
    for key in ("benchmark", "support_policy", "history_window", "max_actions",
                "max_decisions", "max_new_tokens", "model_id", "quantization", "seed"):
        assert d[key] == m2[key]
    assert not any(key.startswith("trajectory_") or key.startswith("memory_") for key in d)


def test_pair_comparison_baseline_and_m2_vs_d(monkeypatch, tmp_path):
    monkeypatch.setattr("agent_support.runner.git_commit", lambda: "new-commit")
    source = manifest()
    d = run(config(), source, tmp_path / "d", model=Model(["Action: get 1 plank", "Action: get 1 log"]),
            episode_factory=lambda *_: Episode())
    baseline = run(config("baseline"), source, tmp_path / "baseline",
                   model=Model(["Action: get 1 plank"]), episode_factory=lambda *_: Episode())
    comparison = compare_trajectory(baseline, d)
    assert comparison["paired"]["helped"] == 1
    assert comparison["right"]["support_input_tokens"] == 11
    assert comparison["right"]["retrieval_events"] == 0
    saved = compare_pilots(tmp_path, "textcraft", left_dir=baseline, right_dir=d)
    assert saved["helped"] == 1 and read_json(saved["file"])["right"]["condition"] == "reconsideration_only_v1"
    from tests.test_trajectory_memory import make_run
    m2 = make_run(tmp_path / "m2", "m2_trajectory_v1", source, [False],
                  {**config("m2_trajectory_v1"), "trajectory_corpus": "corpus",
                   "trajectory_index": "index", "trajectory_top_k": 2,
                   "trajectory_token_budget": 1800},
                  {"trajectory_corpus_sha256": "abc", "trajectory_index_sha256": "def",
                   "memory_selection_policy": "v1", "git_commit": "older-commit"})
    m2_meta = json.loads((m2 / "run.json").read_text())
    d_meta = json.loads((d / "run.json").read_text())
    m2_meta["python_version"] = d_meta["python_version"]
    m2_meta["package_versions"] = d_meta["package_versions"]
    m2_meta["textcraft_recipe_order_sha256"] = d_meta["textcraft_recipe_order_sha256"]
    (m2 / "textcraft_recipe_order.json").write_bytes((d / "textcraft_recipe_order.json").read_bytes())
    write_json_atomic(m2 / "run.json", m2_meta)
    paired = compare_trajectory(m2, d)
    assert paired["left"]["git_commit"] == "older-commit"
    assert paired["different_git_commits"] is True
    assert paired["right"]["condition"] == "reconsideration_only_v1"
    d_meta["trajectory_corpus_sha256"] = "unexpected"
    write_json_atomic(d / "run.json", d_meta)
    with pytest.raises(ValueError, match="memory provenance"):
        compare_trajectory(m2, d)
    d_meta.pop("trajectory_corpus_sha256")
    write_json_atomic(d / "run.json", d_meta)
    m2_meta["package_versions"]["torch"] = "different"
    write_json_atomic(m2 / "run.json", m2_meta)
    with pytest.raises(ValueError, match="Package version differs"):
        compare_trajectory(m2, d)
