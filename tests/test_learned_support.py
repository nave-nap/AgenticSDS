"""CPU tests for replay, paired-data validation, fitting, and learned gating."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_support.autonomous_support import (SupportBudget, SupportContext,
                                              SupportExecutor, SupportResult, make_support_controller,
                                              run_support_loop)
from agent_support.common import canonical_digest, digest_text, validate_config
from agent_support.intervention_replay import branch_first_actions, replay_prefix
from agent_support.learned_support_compare import compare_prompted_and_learned
from agent_support.learned_support_gate import (FEATURE_NAMES, FEATURE_VERSION, feature_vector,
                                                validate_artifact, visible_state)
from agent_support.learned_support_training import PAIRING_PROTOCOL, train_gate


ROOT = Path(__file__).resolve().parents[1]


def manifest(partition, start, count):
    tasks = []
    for seed in range(start, start + count):
        observation = f"Crafting commands:\ncraft 1 item{seed} using 1 log\n\nGoal: craft item{seed}."
        tasks.append({"task_id": f"textcraft:seed:{seed}", "seed": seed,
                      "initial_observation": observation,
                      "observation_sha256": digest_text(observation)})
    return {"schema_version": 1, "benchmark": "textcraft", "partition": partition,
            "tasks": tasks}


def decision_state(failure=False):
    return {"benchmark": "textcraft", "task_prompt": "Crafting commands:\ncraft 1 item using 1 log\n\nGoal: craft item.",
            "observation": "Could not craft item" if failure else "Got 1 log",
            "available_actions": [],
            "history": [{"action": "craft 1 item using 1 log", "observation": "Could not craft item"}]
                       if failure else [],
            "decision": 2 if failure else 1,
            "original_action": "craft 1 item using 1 log",
            "current_action": "craft 1 item using 1 log",
            "results": [],
            "remaining_budget": {"operations_this_decision": 2, "operations_task": 6,
                                 "additional_model_calls": 12},
            "capabilities": [], "remaining_actions": 28, "remaining_decisions": 29}


def paired_data(train, memory):
    records = []
    for index, task in enumerate(train["tasks"]):
        failure = index % 2 == 0
        state = decision_state(failure)
        no_success = not failure
        yes_success = failure
        branches = {
            "no_support": {"complete": True, "task_success": no_success,
                           "total_model_calls": 4, "total_tokens": 1000,
                           "rollout_seed": 7, "followup_policy_id": "frozen_followup_v1"},
            "prompted_support": {"complete": True, "task_success": yes_success,
                                 "total_model_calls": 5, "total_tokens": 1400,
                                 "rollout_seed": 7, "followup_policy_id": "frozen_followup_v1"},
        }
        records.append({"pair_id": f"pair-{index}", "task_id": task["task_id"],
                        "task_fingerprint": task["observation_sha256"],
                        "decision_state": state, "state_sha256": canonical_digest(state),
                        "replay_verified": True, "rollout_seed": 7,
                        "followup_policy_id": "frozen_followup_v1",
                        "branches": branches})
    return {"schema_version": 1, "partition": "training",
            "pairing_protocol": PAIRING_PROTOCOL,
            "manifest_sha256": canonical_digest(train),
            "memory_corpus_sha256": canonical_digest(memory),
            "task_agent_model_revision": "qwen-revision",
            "controller_model_revision": "gptoss-revision", "records": records}


def test_train_gate_learns_from_paired_outcomes_and_never_reads_hidden_fields():
    training = manifest("training", 1000, 12)
    development = manifest("development", 2000, 2)
    heldout = manifest("heldout", 3000, 2)
    memory = {"records": []}
    data = paired_data(training, memory)
    artifact = train_gate(data, training, development, heldout, memory)
    validate_artifact(artifact)
    assert artifact["training_tasks"] == artifact["training_pairs"] == 12
    assert artifact["observed_prompted_win_pairs"] == 6
    from agent_support.learned_support_gate import predict_effect
    assert predict_effect(artifact, decision_state(True)) > predict_effect(artifact, decision_state(False))
    with pytest.raises(ValueError, match="allowed visible"):
        feature_vector({**decision_state(True), "hidden_inventory": {"log": 1}})


def test_training_rejects_memory_overlap_and_unverified_or_unpaired_records():
    training = manifest("training", 1000, 12)
    development = manifest("development", 2000, 2)
    heldout = manifest("heldout", 3000, 2)
    memory = {"records": [{"task_id": training["tasks"][0]["task_id"],
                           "task_fingerprint": training["tasks"][0]["observation_sha256"]}]}
    data = paired_data(training, memory)
    with pytest.raises(ValueError, match="leaks"):
        train_gate(data, training, development, heldout, memory)
    memory = {"records": []}
    data = paired_data(training, memory)
    data["records"][0]["replay_verified"] = False
    with pytest.raises(ValueError, match="verified shared"):
        train_gate(data, training, development, heldout, memory)
    data = paired_data(training, memory)
    data["records"][0]["branches"].pop("no_support")
    with pytest.raises(ValueError, match="requires no-support"):
        train_gate(data, training, development, heldout, memory)


def test_learned_gate_stops_or_delegates_and_logs_score(tmp_path):
    config = json.loads((ROOT / "configs/autonomous_evaluation/autonomous.json").read_text())
    config["autonomous_support"]["controller"] = "learned_gate_v1"
    task_revision = config["model_revision"]
    controller_revision = config["autonomous_support"]["model"]["model"]["model_revision"]
    config["autonomous_support"]["memory"]["corpus_sha256"] = "a" * 64
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("bias")] = -0.5
    weights[FEATURE_NAMES.index("latest_visible_failure")] = 1.0
    weights[FEATURE_NAMES.index("prior_support_fraction")] = -4.0
    artifact = {"schema_version": 1, "policy_type": "cost_sensitive_ridge_gate",
                "training_tasks": 12, "training_pairs": 12,
                "training_data_sha256": "1" * 64,
                "training_manifest_sha256": "2" * 64,
                "development_manifest_sha256": "3" * 64,
                "heldout_manifest_sha256": "4" * 64,
                "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                "weights": weights, "task_agent_model_revision": task_revision,
                "controller_model_revision": controller_revision,
                "memory_corpus_sha256": "a" * 64}
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(artifact))
    config["autonomous_support"]["learned_gate"] = {
        "artifact_path": str(path), "artifact_sha256": canonical_digest(artifact),
        "minimum_effect": 0.0}
    validate_config(config, "textcraft")
    class CountedModel:
        def __init__(self):
            self.calls = 0
        def count_messages_tokens(self, messages):
            return 10
        def generate(self, messages, max_new_tokens=None):
            self.calls += 1
            return {"text": json.dumps({"kind": "execute", "action": None, "reason": "proceed"}),
                    "input_tokens": 10, "output_tokens": 20,
                    "inference_seconds": 0.01, "generation_status": "complete"}
    model = CountedModel()
    controller = make_support_controller(config, model)
    def context(failed):
        return SupportContext("textcraft", "textcraft:seed:130", decision_state()["task_prompt"],
                              "Could not craft item" if failed else "Got 1 log", (), (), 1,
                              "get 1 log", "get 1 log",
                              remaining_budget={"operations_this_decision": 2,
                                                "operations_task": 6,
                                                "additional_model_calls": 12})
    budget = lambda: SupportBudget(2, 6, 10, 2000, 3)
    stopped = run_support_loop(controller, SupportExecutor(), budget(), context(False), lambda action: action)
    delegated = run_support_loop(controller, SupportExecutor(), budget(), context(True), lambda action: action)
    no_calls = run_support_loop(controller, SupportExecutor(), SupportBudget(2, 6, 0, 2000, 3),
                                context(True), lambda action: action)
    assert stopped.termination_reason == delegated.termination_reason == "execute"
    assert model.calls == 1
    assert no_calls.termination_reason == "execute" and model.calls == 1
    assert no_calls.events[0]["controller_metadata"]["learned_gate"]["delegated"] is False
    assert stopped.events[0]["controller_metadata"]["learned_gate"]["delegated"] is False
    assert delegated.events[0]["controller_metadata"]["learned_gate"]["delegated"] is True
    after_support = controller.decide(replace(context(True), results=(SupportResult("ok", {"note": "checked"}),)))
    assert after_support.decision.kind == "execute"
    assert after_support.metadata["learned_gate"]["delegated"] is False
    assert model.calls == 1
    assert "task_id" not in visible_state(context(True))


def test_replay_prefix_checks_state_and_branches_without_using_hidden_state():
    class Episode:
        def __init__(self, task):
            self.observation = ""
            self.count = 0
        def reset(self):
            self.observation = "start"
        def step(self, action):
            self.count += 1
            self.observation = f"observed {action}"
            return {"observation": self.observation, "reward": 0.0,
                    "done": False, "success": False, "failed_action": False}
        def close(self):
            pass
    prefix = [{"input_observation": "start", "action": "get 1 log",
               "output_observation": "observed get 1 log",
               "outcome": {"observation": "observed get 1 log", "reward": 0.0,
                           "done": False, "success": False, "failed_action": False}}]
    episode = replay_prefix({}, prefix, Episode)
    assert episode.count == 1
    episode.close()
    results = branch_first_actions({}, prefix, {"no_support": "get 1 log",
                                                      "prompted_support": "get 1 oak logs"}, Episode)
    assert results["no_support"]["observation"] != results["prompted_support"]["observation"]
    bad = [{**prefix[0], "output_observation": "different"}]
    with pytest.raises(ValueError, match="output observation differs"):
        replay_prefix({}, bad, Episode)


def test_strict_learned_comparison_checks_artifact_and_matched_settings(tmp_path):
    import hashlib
    manifest_data = json.loads((ROOT / "manifests/textcraft/development.json").read_text())
    recipe = (ROOT / "agent_support/textcraft_recipe_order.json").read_bytes()
    base = json.loads((ROOT / "configs/autonomous_evaluation/autonomous.json").read_text())
    learned = json.loads(json.dumps(base))
    learned["autonomous_support"]["controller"] = "learned_gate_v1"
    artifact = {"schema_version": 1, "policy_type": "cost_sensitive_ridge_gate",
                "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                "weights": [0.0] * len(FEATURE_NAMES),
                "training_tasks": 12, "training_pairs": 12,
                "training_data_sha256": "1" * 64,
                "training_manifest_sha256": "2" * 64,
                "development_manifest_sha256": canonical_digest(manifest_data),
                "heldout_manifest_sha256": "4" * 64,
                "task_agent_model_revision": base["model_revision"],
                "controller_model_revision": base["autonomous_support"]["model"]["model"]["model_revision"],
                "memory_corpus_sha256": base["autonomous_support"]["memory"]["corpus_sha256"]}
    learned["autonomous_support"]["learned_gate"] = {
        "artifact_path": "/tmp/fixture-gate.json", "artifact_sha256": canonical_digest(artifact),
        "minimum_effect": 0.0}
    paths = {}
    for label, config in (("prompted", base), ("learned", learned)):
        directory = tmp_path / label
        (directory / "episodes").mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "manifest.json").write_text(json.dumps(manifest_data))
        (directory / "textcraft_recipe_order.json").write_bytes(recipe)
        metadata = {"status": "complete", "run_id": label,
                    "manifest_sha256": canonical_digest(manifest_data),
                    "selected_manifest_sha256": canonical_digest(manifest_data),
                    "selected_task_count": 20, "limit": None,
                    "config_sha256": canonical_digest(config), "git_commit": "fixture",
                    "python_version": "3.14", "package_versions": {"textcraft": "0.0.3"},
                    "textcraft_recipe_order_sha256": hashlib.sha256(recipe).hexdigest(),
                    "autonomous_memory_corpus_sha256": artifact["memory_corpus_sha256"],
                    "autonomous_memory_index_sha256": base["autonomous_support"]["memory"]["index_sha256"]}
        if label == "learned":
            metadata["learned_gate_artifact_sha256"] = canonical_digest(artifact)
            (directory / "learned_gate_artifact.json").write_text(json.dumps(artifact))
        (directory / "run.json").write_text(json.dumps(metadata))
        for index, task in enumerate(manifest_data["tasks"]):
            episode = {"task_id": task["task_id"], "success": index < (11 if label == "learned" else 10),
                       "status": "success" if index < (11 if label == "learned" else 10) else "max_actions",
                       "environment_actions": 1, "failed_actions": 0, "model_decisions": 1,
                       "model_calls": 1, "input_tokens": 10, "output_tokens": 5,
                       "inference_seconds": 0.1, "parser_errors": 0, "trace": []}
            (directory / "episodes" / f"{index:04d}.json").write_text(json.dumps(episode))
        paths[label] = directory
    report = compare_prompted_and_learned(paths["prompted"], paths["learned"])
    assert report["paired"]["success_rate_difference"] == 0.05
    assert report["aggregates"]["learned"]["successes"] == 11
    (paths["learned"] / "learned_gate_artifact.json").write_text(json.dumps({**artifact, "tampered": True}))
    with pytest.raises(ValueError, match="artifact snapshot"):
        compare_prompted_and_learned(paths["prompted"], paths["learned"])
