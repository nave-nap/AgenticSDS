from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_support import recovery_policy_eval as evaluation
from agent_support.common import canonical_digest
from agent_support.recovery_policy_registry import POLICIES, policy_digest

SOURCE = "701c25d"
REPO = Path(__file__).resolve().parents[1]


def archived(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{SOURCE}:{path}"], cwd=REPO)


def test_policy_sources_and_controller_prompt_match_frozen_commit():
    for name in ("recovery_accuracy_policy.py", "recovery_policy_registry.py",
                 "recovery_policy_telemetry.py"):
        path = f"agent_support/{name}"
        assert (REPO / path).read_bytes() == archived(path)
    source = ast.parse(archived("agent_support/recovery_v4.py"))
    current = ast.parse((REPO / "agent_support/recovery_v4.py").read_text())
    for name in ("SYSTEM", "parse_decision", "_messages", "adaptive_loop",
                 "run_recovery_episode"):
        def selected(tree):
            return next(node for node in tree.body if
                        (isinstance(node, ast.FunctionDef) and node.name == name) or
                        (isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == name
                            for target in node.targets)))
        assert ast.dump(selected(current)) == ast.dump(selected(source))


def test_policy_and_configuration_digests_are_frozen():
    assert POLICIES == ("original_v4", "state_aware_v1", "accuracy_first_v1",
                        "accuracy_first_terminal_v1")
    expected = ("3cbd31d336be037890248d921527c043c829f6fafbd469b14447d8de0a35894d",
                "b0f4a2a0bb4daf77b74a4464616107b5e3e69d8170c3602f16e38f8877bcb314",
                "14a19273a3afb2b04b589610014d7981a9fb355d2071dbef964b1150c7ad4733",
                "0fe2a50209cbaa8f32e2271264035431fd5eb5f892b638388162bae76134d3ae")
    assert tuple(policy_digest(policy) for policy in POLICIES) == expected
    assert canonical_digest(evaluation.config_for(evaluation.BASE_ARM)) == (
        "a8e57f46a11646aaaac5b50b3cc386c6770e77ff1ad557f7115de03ff4885d1e")
    assert evaluation.MANIFEST_SHA256 == (
        "150c7bbd973936e0f3ef8fe1266060e88de0181f0068ce98bc7bce0989a2d562")
    assert evaluation.STAGES == {"smoke": (0,), "pilot": (0, 1, 2)}


def test_full_feasibility_gate_rejects_incomplete_output(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "feasibility_preflight", lambda: {"protocol": "three_gpu_chunked_eager_v2"})
    monkeypatch.setattr(evaluation, "require_clean_commit", lambda: "port-commit")
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    value = {"protocol": "three_gpu_chunked_eager_v2", "status": "infrastructure_valid",
             "git_commit": "feasibility-commit", "generation": {"input_tokens": 8887,
             "output_tokens": 1, "generation_status": "complete"},
             "forward_execution": {"prefill_layers_completed": 24},
             "qwen_hf_device_map": {"": "0"}}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(value))
    assert evaluation.feasibility_gate(path)["status"] == "gate_passed"
    value["generation"]["output_tokens"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="invalid or incomplete"):
        evaluation.feasibility_gate(path)


def test_matched_smoke_reuses_one_model_load_and_matches_task(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "require_clean_commit", lambda: "port-commit")
    monkeypatch.setattr(evaluation, "feasibility_gate", lambda *_: {"status": "gate_passed"})
    monkeypatch.setenv("POLICY_FEASIBILITY_RESULT", "fake-result")
    plan = evaluation._plan(require_runtime=False)
    monkeypatch.setattr(evaluation, "_plan", lambda require_runtime: plan)
    loads = {"qwen": 0, "gpt": 0, "memory": 0, "closed": 0}
    class Worker:
        def __init__(self, *_):
            loads["gpt"] += 1
        def metadata(self):
            return {"gpu_name": "NVIDIA L40S", "attention_implementation": evaluation.REQUESTED_ATTENTION,
                    "sharding": "explicit_decoder_layers_12_12",
                    "verified_expert_modules_by_device": {"cuda:0": 12, "cuda:1": 12},
                    "gpu_names": ["NVIDIA L40S", "NVIDIA L40S"]}
        def begin_episode(self, *_):
            pass
        def close(self):
            loads["closed"] += 1
    class Qwen:
        model = SimpleNamespace(hf_device_map={"": 0})
    def qwen_factory(_):
        loads["qwen"] += 1
        return Qwen()
    def memory_factory(*_):
        loads["memory"] += 1
        return object()
    def episode(task, config, qwen, gpt, *, memory, checkpoint, policy):
        return {"task_id": task["task_id"], "arm": config["arm"],
                "initial_observation": task["initial_observation"], "success": False,
                "status": "max_actions", "environment_actions": 0, "failed_actions": 0,
                "model_decisions": 0, "support_calls": 0, "trace": []}
    def runner(*args, **kwargs):
        return evaluation._run_tasks(*args, episode_runner=episode, **kwargs)
    report = evaluation.run_matched_stage(tmp_path, "smoke", POLICIES,
        qwen_factory=qwen_factory, gpt_factory=Worker,
        memory_factory=memory_factory, runner=runner)
    assert report["status"] == "complete"
    assert loads == {"qwen": 1, "gpt": 1, "memory": 1, "closed": 1}
    assert evaluation.stage_report(tmp_path, "smoke")["ready"] is True
