"""Manually run alternating Qwen/GPT-OSS worker requests on one GPU."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .common import read_json, validate_config, write_json_new
from .components import load_components
from .hybrid_model import HybridModel
from .policy import parse_textcraft, textcraft_messages


def gpu_memory() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    load_components(config)
    validate_config(config, "textcraft")
    task = read_json(config["manifest"])["tasks"][0]
    messages = textcraft_messages(task["initial_observation"], [], config["history_window"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["AGENT_SUPPORT_WORKER_LOG_DIR"] = str(args.output.parent / "workers")
    model = HybridModel(config)
    try:
        model.begin_episode(task["task_id"])
        samples = []
        for kind in (model.primary_model_type, model.delegated_model_type,
                     model.primary_model_type, model.delegated_model_type):
            generation = model.generate_for(kind, messages)
            action = parse_textcraft(generation["text"])
            samples.append({"model_type": kind, "generation": generation, "parsed_action": action,
                            "memory_after": model.memory_snapshot(), "gpu_memory_mib": gpu_memory()})
            if action is None or generation.get("generation_status", "complete") != "complete":
                raise RuntimeError(f"{kind} did not produce a valid action on smoke request {len(samples)}")
        result = {"model_metadata": model.metadata(),
                  "worker_pids": {kind: worker.process.pid for kind, worker in model.workers.items()},
                  "task_id": task["task_id"], "samples": samples,
                  "final_memory": model.memory_snapshot(), "gpu_memory_mib": gpu_memory()}
        write_json_new(args.output, result)
        print(args.output)
    finally:
        model.close()


if __name__ == "__main__":
    main()
