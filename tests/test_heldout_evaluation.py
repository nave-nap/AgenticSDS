import json
import shutil
from pathlib import Path

import pytest

from agent_support.analysis import compare_trajectory
from agent_support.common import canonical_digest, digest_text, read_json, write_json_atomic
from agent_support.heldout_evaluation import (EXPECTED_HELDOUT_SHA256, audit_pool,
                                               checked_lock, compare_three, freeze,
                                               paired_interval, wilson)
from tests.test_trajectory_memory import make_run


REPO = Path(__file__).resolve().parents[1]
REVISION = "a" * 40


def data_root(tmp_path):
    manifests = tmp_path / "manifests" / "textcraft"
    manifests.mkdir(parents=True)
    for partition in ("development", "heldout"):
        shutil.copy2(REPO / "manifests" / "textcraft" / f"{partition}.json",
                     manifests / f"{partition}.json")
    training = {"schema_version": 1, "benchmark": "textcraft", "partition": "training",
                "tasks": [{"task_id": f"textcraft:seed:{seed}", "seed": seed,
                           "initial_observation": f"Training task {seed}",
                           "observation_sha256": digest_text(f"Training task {seed}")}
                          for seed in range(1000, 1040)]}
    write_json_atomic(manifests / "training.json", training)
    return tmp_path


def test_existing_heldout_pool_has_100_distinct_unexposed_tasks(tmp_path):
    root = data_root(tmp_path)
    audit = audit_pool(root)
    assert audit["eligible_count"] == 100
    assert audit["training_count"] == 40 and audit["pilot_count"] == 20
    assert audit["heldout_manifest_sha256"] == EXPECTED_HELDOUT_SHA256
    assert audit["prior_heldout_runs"] == []
    prior = root / "results" / "textcraft_old"
    prior.mkdir(parents=True)
    write_json_atomic(prior / "manifest.json", {**read_json(root / "manifests" / "textcraft" / "heldout.json"),
                                                  "tasks": [read_json(root / "manifests" / "textcraft" / "heldout.json")["tasks"][0]]})
    assert audit_pool(root)["prior_heldout_runs"] == ["textcraft_old"]
    with pytest.raises(ValueError, match="already exposed"):
        freeze(root)


def test_freeze_is_immutable_and_checks_corpus_exclusion(monkeypatch, tmp_path):
    root = data_root(tmp_path)
    memory = root / "trajectory_memory" / "textcraft"
    memory.mkdir(parents=True)
    heldout_hash = canonical_digest(read_json(root / "manifests" / "textcraft" / "heldout.json"))
    development_hash = canonical_digest(read_json(root / "manifests" / "textcraft" / "development.json"))
    training = read_json(root / "manifests" / "textcraft" / "training.json")
    corpus = {"excluded_manifest_sha256": {"heldout": heldout_hash, "development": development_hash},
              "source_runs": [{"benchmark": "textcraft", "manifest_sha256": canonical_digest(training)}],
              "records": [{"benchmark": "textcraft", "task_id": training["tasks"][0]["task_id"],
                           "task_fingerprint": training["tasks"][0]["observation_sha256"]}]}
    index = {"corpus_sha256": canonical_digest(corpus)}
    write_json_atomic(memory / "corpus.json", corpus)
    write_json_atomic(memory / "index.json", index)
    monkeypatch.setattr("agent_support.heldout_evaluation.cached_model_revision", lambda _: REVISION)
    monkeypatch.setattr("agent_support.heldout_evaluation.check_task_replay", lambda _: 100)
    monkeypatch.setattr("agent_support.heldout_evaluation._clean_commit", lambda: "commit-one")
    monkeypatch.setattr("agent_support.heldout_evaluation.package_versions", lambda: {"torch": "working"})
    monkeypatch.setattr("agent_support.heldout_evaluation.additional_versions", lambda: {"tokenizers": "fixed"})
    leaked = {**corpus, "records": [{**corpus["records"][0],
                                     "task_fingerprint": read_json(root / "manifests" / "textcraft" / "heldout.json")["tasks"][0]["observation_sha256"]}]}
    write_json_atomic(memory / "corpus.json", leaked)
    write_json_atomic(memory / "index.json", {"corpus_sha256": canonical_digest(leaked)})
    with pytest.raises(ValueError, match="outside the frozen training manifest"):
        freeze(root)
    write_json_atomic(memory / "corpus.json", corpus)
    write_json_atomic(memory / "index.json", index)
    frozen = freeze(root)
    assert frozen["eligible_count"] == 100
    assert read_json(frozen["manifest_snapshot"]) == read_json(root / "manifests" / "textcraft" / "heldout.json")
    assert checked_lock(root)["model_revision"] == REVISION
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        freeze(root)
    monkeypatch.setattr("agent_support.heldout_evaluation.package_versions", lambda: {"torch": "changed"})
    with pytest.raises(ValueError, match="changed since freeze"):
        checked_lock(root)


def test_heldout_pair_and_three_way_intervals(monkeypatch, tmp_path):
    root = data_root(tmp_path)
    full = read_json(root / "manifests" / "textcraft" / "heldout.json")
    # Use a small full held-out fixture to exercise the comparison contract.
    small = {**full, "tasks": full["tasks"][:4]}
    common = {"benchmark": "textcraft", "model_id": "fake", "quantization": "nf4_4bit",
              "max_actions": 30, "max_decisions": 30, "max_new_tokens": 100,
              "history_window": 8, "seed": 42}
    configs = {"baseline": {**common, "condition": "baseline"},
               "m2_trajectory_v1": {**common, "condition": "m2_trajectory_v1",
                                    "trajectory_corpus": "c", "trajectory_index": "i",
                                    "trajectory_top_k": 2, "trajectory_token_budget": 1800},
               "reconsideration_only_v1": {**common, "condition": "reconsideration_only_v1"}}
    lock = {"protocol": "textcraft_heldout_abc_v1",
            "pool": {"heldout_manifest_sha256": canonical_digest(small)},
            "config_sha256": {key: canonical_digest(value) for key, value in configs.items()},
            "model_revision": REVISION, "git_commit": "one", "python_version": "3.13",
            "package_versions": {"textcraft": "0.0.3"},
            "textcraft_recipe_order_sha256": None,
            "trajectory_corpus_sha256": "corpus", "trajectory_index_sha256": "index"}
    (root / "heldout_evaluation").mkdir()
    write_json_atomic(root / "heldout_evaluation" / "manifest.json", small)
    monkeypatch.setattr("agent_support.heldout_evaluation.checked_lock", lambda _: lock)
    pinned = {"model_revision": REVISION, "evaluation_lock_sha256": canonical_digest(lock)}
    provenance = {"trajectory_corpus_sha256": "corpus", "trajectory_index_sha256": "index",
                  "memory_selection_policy": "policy", "git_commit": "one",
                  "evaluation_lock_sha256": canonical_digest(lock), "model_revision": REVISION}
    base = make_run(root / "base", "baseline", small, [False, True, False, False],
                    {**configs["baseline"], **pinned},
                    {"git_commit": "one", "evaluation_lock_sha256": canonical_digest(lock),
                     "model_revision": REVISION})
    m2 = make_run(root / "m2", "m2_trajectory_v1", small, [True, True, False, False],
                  {**configs["m2_trajectory_v1"], **pinned}, provenance)
    d = make_run(root / "d", "reconsideration_only_v1", small, [False, True, True, False],
                 {**configs["reconsideration_only_v1"], **pinned},
                 {"git_commit": "one", "evaluation_lock_sha256": canonical_digest(lock),
                  "model_revision": REVISION, "support_policy_id": "fixed_by_condition_v1"})
    assert compare_trajectory(base, m2)["partition"] == "heldout"
    assert compare_trajectory(m2, d)["paired"]["helped"] == 1
    base_meta = read_json(base / "run.json")
    write_json_atomic(base / "run.json", {**base_meta, "limit": 3})
    with pytest.raises(ValueError, match="full heldout manifest"):
        compare_trajectory(base, m2)
    write_json_atomic(base / "run.json", base_meta)
    report = compare_three(root, {"baseline": base, "m2_trajectory_v1": m2,
                                  "reconsideration_only_v1": d})
    assert report["successes"] == {"baseline": 1, "m2_trajectory_v1": 2,
                                   "reconsideration_only_v1": 2}
    saved = read_json(report["analysis"])
    assert len(saved["tasks"]) == 4
    assert saved["pairs"]["baseline_vs_m2"]["paired"]["helped"] == 1
    assert wilson(0, 4)[0] == 0
    assert paired_interval([False] * 4, [True] * 4) == [1.0, 1.0]
    results_dir = root / "results"
    results_dir.mkdir()
    for directory, name in ((base, "textcraft_base"), (m2, "textcraft_m2"), (d, "textcraft_d")):
        target = results_dir / name
        target.mkdir()
        shutil.copy2(directory / "run.json", target / "run.json")
    from agent_support.heldout_evaluation import find_evaluation_runs
    assert set(find_evaluation_runs(root, lock)) == {"baseline", "m2_trajectory_v1", "reconsideration_only_v1"}
    meta = read_json(d / "run.json")
    meta["git_commit"] = "different"
    write_json_atomic(d / "run.json", meta)
    with pytest.raises(ValueError, match="different code commits"):
        compare_trajectory(m2, d)


def test_heldout_run_uses_exact_frozen_manifest_and_pinned_model(monkeypatch, tmp_path):
    from agent_support.heldout_evaluation import run_condition

    root = data_root(tmp_path)
    snapshot = root / "heldout_evaluation" / "manifest.json"
    snapshot.parent.mkdir()
    shutil.copy2(root / "manifests" / "textcraft" / "heldout.json", snapshot)
    frozen = {"pool": {"heldout_manifest_sha256": EXPECTED_HELDOUT_SHA256},
              "model_revision": REVISION}
    monkeypatch.setattr("agent_support.heldout_evaluation.checked_lock", lambda _: frozen)
    monkeypatch.setattr("agent_support.cluster_workflow.preflight", lambda *_a, **_kw: {})
    seen = {}

    def fake_run(config, manifest, output_root):
        seen.update(config=config, manifest=manifest, output_root=output_root)
        return root / "results" / "new-run"

    monkeypatch.setattr("agent_support.heldout_evaluation.run", fake_run)
    result = run_condition(root, "baseline")
    assert result["manifest_sha256"] == EXPECTED_HELDOUT_SHA256
    assert seen["config"]["model_revision"] == REVISION
    assert seen["config"]["evaluation_lock_sha256"] == canonical_digest(frozen)
    assert seen["manifest"] == read_json(snapshot)
    assert len(seen["manifest"]["tasks"]) == 100
    prior = root / "results" / "textcraft_previous"
    prior.mkdir(parents=True)
    write_json_atomic(prior / "run.json", {"condition": "baseline",
                                           "selected_manifest_sha256": EXPECTED_HELDOUT_SHA256})
    with pytest.raises(FileExistsError, match="already has a held-out run"):
        run_condition(root, "baseline")
