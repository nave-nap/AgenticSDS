import json
import zipfile

import pytest

from agent_support.cluster_data import _extract_prefix, prepare
from agent_support.cluster_workflow import resolved_config


def test_cluster_config_uses_transferred_paths(tmp_path):
    (tmp_path / "alfworld_config.yaml").write_text("dataset: {}\n")
    memory_dir = tmp_path / "trajectory_memory" / "alfworld"
    memory_dir.mkdir(parents=True)
    (memory_dir / "corpus.json").write_text("{}")
    (memory_dir / "index.json").write_text("{}")
    config = resolved_config(tmp_path, "alfworld", "m2_trajectory_v1")
    assert config["alfworld_config"] == str(tmp_path / "alfworld_config.yaml")
    assert config["trajectory_corpus"] == str(memory_dir / "corpus.json")
    assert config["trajectory_index"] == str(memory_dir / "index.json")
    assert json.loads((tmp_path / "configs" / "alfworld_m2_trajectory_v1.json").read_text()) == config
    assert resolved_config(tmp_path, "alfworld", "m2_trajectory_v1") == config


def test_cluster_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("../outside.txt", "x")
        writer.writestr("training_runs/example/run.json", "{}")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        _extract_prefix(archive, "training_runs", tmp_path / "out")


def test_cluster_bundle_requires_original_alfworld_heldout(tmp_path):
    wrong = tmp_path / "heldout.json"
    wrong.write_text(json.dumps({"schema_version": 1, "benchmark": "alfworld",
                                 "partition": "heldout", "tasks": [{"task_id": "different",
                                 "split": "eval_in_distribution", "game_relpath": "x",
                                 "game_sha256": "different"}]}))
    with pytest.raises(ValueError, match="differs from the original frozen manifest"):
        prepare(tmp_path / "training.zip", tmp_path / "baseline.zip", wrong, tmp_path / "out")
