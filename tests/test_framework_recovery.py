"""Checkpoint recovery verifies identity and never replaces saved episodes."""

import json
import copy
from pathlib import Path

import pytest

from agent_support.common import digest_text, read_json, write_json_atomic, write_json_new
from agent_support.model import MODEL_REGISTRY
from agent_support.run_store import audit_partitions, compare_model_metadata
from agent_support.runner import PreemptionRequested, run


class Episode:
    actions = []
    def __init__(self, task):
        self.task_prompt = self.initial_observation = self.observation = task["initial_observation"]
    def reset(self):
        pass
    def step(self, action):
        return {"observation": "Done", "done": True, "success": True, "failed_action": False}
    def close(self):
        pass


class Model:
    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on
    def metadata(self):
        return {"model_id": "mock", "model_revision": "mock-r1"}
    def generate(self, messages):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("simulated preemption before episode completion")
        return {"text": "Action: inventory", "input_tokens": 3, "output_tokens": 2,
                "inference_seconds": 0.01}


def runtime_metadata():
    return {"model_id": "mock", "model_revision": "mock-r1", "tokenizer_revision": "mock-r1",
            "quantization": "mxfp4_packed", "backend_version": "5.22.2",
            "chat_template_sha256": "pinned-template", "decoding": {"max_new_tokens": 1024},
            "gpu_name": "NVIDIA L40S", "gpu_compute_capability": (8, 9),
            "gpu_total_memory_bytes": 47_676_129_280, "gpu_uuid": "GPU-node9",
            "process_id": 101, "verified_expert_modules": 24}


@pytest.mark.parametrize("field,value", [
    ("gpu_compute_capability", [9, 0]), ("gpu_name", "NVIDIA H100"),
    ("model_revision", "mock-r2"), ("tokenizer_revision", "mock-r2"),
    ("quantization", "dequantized"), ("backend_version", "5.23.0"),
    ("chat_template_sha256", "changed"), ("decoding", {"max_new_tokens": 2048}),
    ("verified_expert_modules", 23), ("gpu_total_memory_bytes", 40_000_000_000),
])
def test_runtime_compatibility_rejects_execution_changes(field, value):
    previous = runtime_metadata()
    current = copy.deepcopy(previous)
    current[field] = value
    check = compare_model_metadata(previous, current)
    assert check["compatible"] is False
    assert any(item["field"] == field or item["field"].startswith(field + ".")
               for item in check["differences"])


def test_runtime_compatibility_accepts_json_round_trip_and_allocation_changes():
    previous = runtime_metadata()
    assert compare_model_metadata(previous, json.loads(json.dumps(previous))) == {
        "compatible": True, "differences": []}
    current = json.loads(json.dumps(previous))
    current.update(gpu_total_memory_bytes=47_600_000_000,
                   gpu_uuid="GPU-node10", process_id=202)
    check = compare_model_metadata(previous, current)
    assert check["compatible"]
    assert {item["field"] for item in check["differences"]} == {
        "gpu_total_memory_bytes", "gpu_uuid", "process_id"}
    assert all(item["compatible"] for item in check["differences"])
    assert compare_model_metadata(previous, previous)["differences"] == []


def test_hybrid_worker_runtime_compatibility_is_recursive():
    worker = runtime_metadata()
    previous = {"model_id": "hybrid", "model_revision": "pair-r1",
                "workers": {"delegated": {"python": "/shared/gpt/bin/python", "metadata": worker},
                            "primary": {"python": "/shared/qwen/bin/python",
                                        "metadata": {"model_revision": "qwen-r1"}}}}
    current = copy.deepcopy(previous)
    current["workers"]["delegated"]["metadata"].update(
        gpu_total_memory_bytes=47_600_000_000, process_id=202)
    assert compare_model_metadata(previous, current)["compatible"]
    current["workers"]["delegated"]["metadata"]["backend_version"] = "5.23.0"
    check = compare_model_metadata(previous, current)
    assert not check["compatible"]
    assert any(item["field"] == "workers.delegated.metadata.backend_version"
               for item in check["differences"])


def task(number):
    observation = f"Crafting commands:\nget 1 wood\n\nGoal: craft {number} plank."
    return {"task_id": f"textcraft:seed:{number}", "seed": number,
            "initial_observation": observation, "observation_sha256": digest_text(observation)}


def setup(tmp_path):
    manifest = {"schema_version": 1, "benchmark": "textcraft", "partition": "development",
                "tasks": [task(1), task(2)]}
    heldout = {"schema_version": 1, "benchmark": "textcraft", "partition": "heldout", "tasks": [task(3)]}
    dev_path, heldout_path = tmp_path / "dev.json", tmp_path / "heldout.json"
    write_json_atomic(dev_path, manifest)
    write_json_atomic(heldout_path, heldout)
    config = {"schema_version": 2, "benchmark": "textcraft", "condition": "modular_v1",
              "manifest": str(dev_path), "partition_manifests": {"development": str(dev_path),
                                                                  "heldout": str(heldout_path)},
              "model_type": "mock", "model_id": "mock", "model_revision": "mock-r1", "seed": 1,
              "max_actions": 2, "max_decisions": 2, "max_model_calls": 2,
              "history_window": 1, "max_new_tokens": 2, "max_support_calls": 0,
              "do_sample": False,
              "max_support_tokens": 0, "supports": [], "support_versions": {},
              "selection_policy": {"id": "never_v1"}, "output_root": str(tmp_path / "runs")}
    return config, manifest


def test_resume_verifies_and_skips_completed_episode(monkeypatch, tmp_path):
    monkeypatch.setitem(MODEL_REGISTRY, "mock", lambda config: Model())
    config, manifest = setup(tmp_path)
    first_model = Model(fail_on=2)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        run(config, manifest, tmp_path / "runs", model=first_model,
            episode_factory=lambda _, task, config: Episode(task), run_key="job_11")
    directory = tmp_path / "runs" / "textcraft_job_11"
    first = next((directory / "episodes").glob("*.json"))
    original = first.read_bytes()
    assert read_json(directory / "run.json")["status"] == "failed"

    resumed = Model()
    output = run(config, manifest, tmp_path / "runs", model=resumed,
                 episode_factory=lambda _, task, config: Episode(task), run_key="job_11", resume=True)
    assert output == directory and resumed.calls == 1
    assert first.read_bytes() == original
    metadata = read_json(directory / "run.json")
    assert metadata["status"] == "complete"
    assert metadata["git_commit"] and "git_dirty" in metadata
    assert metadata["config_file_sha256"] and metadata["manifest_file_sha256"]
    assert metadata["model_revision"] == "mock-r1"
    assert metadata["checkpoint_history"][-1]["verified_episodes"] == 1
    assert len(list((directory / "episodes").glob("*.json"))) == 2
    assert len(metadata["runtime_attempts"]) == 2
    assert all(item["compatibility_check"]["compatible"] for item in metadata["runtime_attempts"])
    with pytest.raises(ValueError, match="complete"):
        run(config, manifest, tmp_path / "runs", model=Model(),
            episode_factory=lambda _, task, config: Episode(task), run_key="job_11", resume=True)


def test_resume_accepts_new_l40s_allocation_and_preserves_episode_hash(monkeypatch, tmp_path):
    monkeypatch.setitem(MODEL_REGISTRY, "mock", lambda config: Model())
    config, manifest = setup(tmp_path)

    class RuntimeModel(Model):
        def __init__(self, details, fail_on=None):
            super().__init__(fail_on)
            self.details = details

        def metadata(self):
            return self.details

    previous = runtime_metadata()
    with pytest.raises(RuntimeError, match="simulated preemption"):
        run(config, manifest, tmp_path / "runs", model=RuntimeModel(previous, fail_on=2),
            episode_factory=lambda _, task, config: Episode(task), run_key="job_memory")
    directory = tmp_path / "runs" / "textcraft_job_memory"
    episode_path = next((directory / "episodes").glob("*.json"))
    original_bytes = episode_path.read_bytes()
    original_hashes = read_json(directory / "checkpoint.json")["episodes"]
    current = json.loads(json.dumps(previous))
    current.update(gpu_total_memory_bytes=47_600_000_000,
                   gpu_uuid="GPU-node10", process_id=202)
    run(config, manifest, tmp_path / "runs", model=RuntimeModel(current),
        episode_factory=lambda _, task, config: Episode(task), run_key="job_memory", resume=True)
    metadata = read_json(directory / "run.json")
    assert metadata["status"] == "complete"
    assert metadata["model_metadata"]["gpu_uuid"] == "GPU-node9"
    assert metadata["runtime_attempts"][1]["model_metadata"] == json.loads(json.dumps(current))
    assert metadata["runtime_attempts"][1]["compatibility_check"]["compatible"]
    assert episode_path.read_bytes() == original_bytes
    for name, digest in original_hashes.items():
        assert read_json(directory / "checkpoint.json")["episodes"][name] == digest


def test_rejected_resume_persists_current_runtime_and_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setitem(MODEL_REGISTRY, "mock", lambda config: Model())
    config, manifest = setup(tmp_path)

    class RuntimeModel(Model):
        def __init__(self, details, fail_on=None):
            super().__init__(fail_on)
            self.details = details

        def metadata(self):
            return self.details

    previous = runtime_metadata()
    with pytest.raises(RuntimeError, match="simulated preemption"):
        run(config, manifest, tmp_path / "runs", model=RuntimeModel(previous, fail_on=2),
            episode_factory=lambda _, task, config: Episode(task), run_key="job_reject")
    directory = tmp_path / "runs" / "textcraft_job_reject"
    saved = next((directory / "episodes").glob("*.json"))
    saved_bytes = saved.read_bytes()
    saved_checkpoint = read_json(directory / "checkpoint.json")["episodes"]
    current = copy.deepcopy(previous)
    current["backend_version"] = "5.23.0"
    with pytest.raises(ValueError, match="backend_version: previous='5.22.2', current='5.23.0'"):
        run(config, manifest, tmp_path / "runs", model=RuntimeModel(current),
            episode_factory=lambda _, task, config: Episode(task), run_key="job_reject", resume=True)
    metadata = read_json(directory / "run.json")
    assert metadata["status"] == "failed"
    assert metadata["model_metadata"] == json.loads(json.dumps(previous))
    assert metadata["runtime_attempts"][1]["model_metadata"] == json.loads(json.dumps(current))
    assert metadata["runtime_attempts"][1]["compatibility_check"]["compatible"] is False
    assert saved.read_bytes() == saved_bytes
    assert read_json(directory / "checkpoint.json")["episodes"] == saved_checkpoint
    run(config, manifest, tmp_path / "runs", model=RuntimeModel(previous),
        episode_factory=lambda _, task, config: Episode(task), run_key="job_reject", resume=True)
    recovered = read_json(directory / "run.json")
    assert recovered["status"] == "complete"
    assert [item["compatibility_check"]["compatible"] for item in recovered["runtime_attempts"]] == [
        True, False, True]
    assert saved.read_bytes() == saved_bytes


def test_resume_rejects_tampered_episode_and_changed_config(monkeypatch, tmp_path):
    monkeypatch.setitem(MODEL_REGISTRY, "mock", lambda config: Model())
    config, manifest = setup(tmp_path)
    with pytest.raises(RuntimeError):
        run(config, manifest, tmp_path / "runs", model=Model(fail_on=2),
            episode_factory=lambda _, task, config: Episode(task), run_key="job_12")
    changed = {**config, "seed": 2}
    with pytest.raises(ValueError, match="snapshot"):
        run(changed, manifest, tmp_path / "runs", model=Model(),
            episode_factory=lambda _, task, config: Episode(task), run_key="job_12", resume=True)
    path = next((tmp_path / "runs" / "textcraft_job_12" / "episodes").glob("*.json"))
    episode = read_json(path)
    episode["success"] = False
    write_json_atomic(path, episode)
    with pytest.raises(ValueError, match="hash differs"):
        run(config, manifest, tmp_path / "runs", model=Model(),
            episode_factory=lambda _, task, config: Episode(task), run_key="job_12", resume=True)


def test_partition_overlap_detects_identity_and_content(tmp_path):
    config, manifest = setup(tmp_path)
    assert set(audit_partitions(config, manifest)) == {"development", "heldout"}
    heldout_path = Path(config["partition_manifests"]["heldout"])
    heldout = read_json(heldout_path)
    heldout["tasks"][0]["initial_observation"] = manifest["tasks"][0]["initial_observation"]
    heldout["tasks"][0]["observation_sha256"] = manifest["tasks"][0]["observation_sha256"]
    write_json_atomic(heldout_path, heldout)
    with pytest.raises(ValueError, match="overlap"):
        audit_partitions(config, manifest)


def test_orphan_episode_is_verified_after_checkpoint_interruption(monkeypatch, tmp_path):
    from agent_support import runner
    monkeypatch.setitem(MODEL_REGISTRY, "mock", lambda config: Model())
    config, manifest = setup(tmp_path)
    original_checkpoint = runner.checkpoint
    calls = 0

    def fail_after_first_episode(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("checkpoint interrupted")
        return original_checkpoint(*args)

    with monkeypatch.context() as patcher:
        patcher.setattr(runner, "checkpoint", fail_after_first_episode)
        with pytest.raises(RuntimeError, match="checkpoint interrupted"):
            run(config, manifest, tmp_path / "runs", model=Model(),
                episode_factory=lambda _, task, config: Episode(task), run_key="job_13")
    resumed = Model()
    directory = run(config, manifest, tmp_path / "runs", model=resumed,
                    episode_factory=lambda _, task, config: Episode(task), run_key="job_13", resume=True)
    assert resumed.calls == 1
    history = read_json(directory / "run.json")["checkpoint_history"]
    assert len(history[-1]["uncheckpointed_episodes"]) == 1


def test_preemption_status_and_exclusive_episode_publication(monkeypatch, tmp_path):
    monkeypatch.setitem(MODEL_REGISTRY, "mock", lambda config: Model())
    config, manifest = setup(tmp_path)
    class PreemptingModel(Model):
        def generate(self, messages):
            self.calls += 1
            if self.calls == 2:
                raise PreemptionRequested()
            return {"text": "Action: inventory", "input_tokens": 3,
                    "output_tokens": 2, "inference_seconds": 0.01}

    with pytest.raises(PreemptionRequested):
        run(config, manifest, tmp_path / "runs", model=PreemptingModel(),
            episode_factory=lambda _, task, config: Episode(task), run_key="job_14")
    directory = tmp_path / "runs" / "textcraft_job_14"
    assert read_json(directory / "run.json")["status"] == "interrupted"
    saved = next((directory / "episodes").glob("*.json"))
    with pytest.raises(FileExistsError):
        write_json_new(saved, {"task_id": "replacement"})
