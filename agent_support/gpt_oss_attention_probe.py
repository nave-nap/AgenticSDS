"""Explicit, inference-only GPT-OSS attention comparison on Locust.

The normal recovery runner never imports or invokes this module. Fixture
preparation is CPU-only; `run` deliberately loads the two co-resident models
and performs one GPT-OSS generation, so call it only on an authorized GPU job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

from .common import read_json, write_json_atomic
from .gpu_memory_diagnostics import CACHE_RELEASE_ENV, DIRECTORY_ENV, ENABLE_ENV
from .input_token_fingerprint import input_ids_sha256, single_input_token_ids

SCHEMA_VERSION = 1
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs/autonomous_recovery_v4/autonomous_recovery_v4.json"
SYNTHETIC_SYSTEM = "Synthetic TextCraft controller memory probe. Return a short JSON decision."
HISTORICAL_EVALUATOR_TRANSFORMERS = "4.57.6"
HISTORICAL_WORKER_TORCH = "2.8.0+cu128"
HISTORICAL_WORKER_CUDA = "12.8"


def config_for_probe() -> dict:
    # The evaluator-side command retains the frozen config verification. The
    # tokenizer-only command works in the separate, minimal GPT-OSS venv.
    from .recovery_v4_eval import config_for
    return config_for("autonomous_recovery_v4")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":")).encode()).hexdigest()


def validate_fixture(value: dict, max_output_tokens: int) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Probe fixture requires schema_version 1")
    messages = value.get("messages")
    if (not isinstance(messages, list) or not messages or
            any(not isinstance(item, dict) or set(item) != {"role", "content"} or
                item["role"] not in {"system", "developer", "user", "assistant"} or
                not isinstance(item["content"], str) for item in messages)):
        raise ValueError("Probe fixture requires plain-text chat messages")
    if (type(value.get("max_new_tokens")) is not int or
            not 1 <= value["max_new_tokens"] <= max_output_tokens):
        raise ValueError("Probe output ceiling exceeds the pinned controller ceiling")
    if (not isinstance(value.get("task_id"), str) or not value["task_id"] or
            type(value.get("seed")) is not int):
        raise ValueError("Probe fixture requires task_id and integer seed")
    count = value.get("expected_input_tokens")
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("Expected input token count is invalid")
    digest = value.get("expected_input_ids_sha256")
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or
                               any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("Expected input token digest is invalid")
    return value


def export_episode(episode_path: Path, decision: int, event_index: int,
                   config: dict) -> dict:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    records = [item for item in episode["trace"] if item["decision"] == decision]
    if len(records) != 1:
        raise ValueError("Episode has no unique requested decision")
    events = records[0]["support_trace"]["events"]
    if not 0 <= event_index < len(events):
        raise ValueError("Controller event index is out of range")
    event = events[event_index]
    messages = event.get("controller_messages")
    if not messages:
        raise ValueError("Archived controller event has no input messages")
    cost = event.get("controller_cost") or {}
    input_count = cost.get("input_tokens")
    if input_count is None:
        input_count = (event.get("controller_generation") or {}).get("input_tokens")
    if type(input_count) is not int or input_count < 1:
        raise ValueError("Archived event has no counted input-token usage")
    remaining = event["state"]["remaining_budget"]["additional_tokens"]
    limit = min(config["autonomous_support"]["model"]["model"]["max_new_tokens"],
                remaining - input_count)
    return validate_fixture({"schema_version": SCHEMA_VERSION,
        "source": {"kind": "archived_episode", "path": str(episode_path),
                   "decision": decision, "controller_event_index": event_index},
        "task_id": episode["task_id"], "seed": config["seed"],
        "messages": messages, "max_new_tokens": limit,
        "expected_input_tokens": input_count},
        config["autonomous_support"]["model"]["model"]["max_new_tokens"])


def list_episode_inputs(episode_path: Path) -> list[dict]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    rows = []
    for record in episode.get("trace", []):
        for index, event in enumerate(record.get("support_trace", {}).get("events", [])):
            if "controller_messages" not in event:
                continue
            cost = event.get("controller_cost") or {}
            generation = event.get("controller_generation") or {}
            rows.append({"decision": record["decision"], "event_index": index,
                         "input_tokens": cost.get("input_tokens", generation.get("input_tokens")),
                         "generation_status": generation.get("generation_status"),
                         "error": event.get("error")})
    return rows


def _pinned_tokenizer(config: dict):
    from transformers import AutoTokenizer
    spec = config["autonomous_support"]["model"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["model_revision"])
    date_expression = 'strftime_now("%Y-%m-%d")'
    template = tokenizer.chat_template
    if not isinstance(template, str) or template.count(date_expression) != 1:
        raise RuntimeError("Pinned GPT-OSS chat template changed")
    template = template.replace(date_expression, f'"{spec["prompt_date"]}"')
    return tokenizer, template


def _token_ids(tokenizer, template: str, config: dict, messages: list[dict]) -> list[int]:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        reasoning_effort=config["autonomous_support"]["model"]["model"]["reasoning_effort"],
        chat_template=template, return_tensors="pt", return_dict=True)
    return single_input_token_ids(encoded)


def synthetic_fixture(config: dict, target_tokens: int) -> dict:
    if not 128 <= target_tokens <= 16000:
        raise ValueError("Synthetic target must be between 128 and 16000 tokens")
    tokenizer, template = _pinned_tokenizer(config)
    line = "Visible recipe: craft 1 oak planks using 1 oak log. Prior action: get 1 oak log. Feedback: Got 1 oak log.\n"
    def material(repeats, suffix=""):
        return [{"role": "system", "content": SYNTHETIC_SYSTEM},
                {"role": "user", "content": json.dumps({"benchmark": "textcraft",
                    "observation": "Synthetic memory-stress fixture; no environment state is implied.",
                    "visible_history": line * repeats + suffix, "current_action": "inventory"},
                    separators=(",", ":"))}]
    low, high = 0, target_tokens
    while low < high:
        mid = (low + high) // 2
        count = len(_token_ids(tokenizer, template, config, material(mid)))
        if count < target_tokens:
            low = mid + 1
        else:
            high = mid
    repeats = max(0, low - 1)
    messages = None
    # Ordinary text padding fills the gap left by whole recipe lines. Check
    # actual token IDs at every step; BPE length need not equal character count.
    for unit in (" x", " a", "\n"):
        def candidate(padding):
            return material(repeats, unit * padding)
        start, end = 0, max(1, target_tokens - len(_token_ids(
            tokenizer, template, config, candidate(0))))
        while len(_token_ids(tokenizer, template, config, candidate(end))) < target_tokens:
            end *= 2
        while start < end:
            mid = (start + end) // 2
            if len(_token_ids(tokenizer, template, config, candidate(mid))) < target_tokens:
                start = mid + 1
            else:
                end = mid
        for padding in range(max(0, start - 8), start + 9):
            trial = candidate(padding)
            if len(_token_ids(tokenizer, template, config, trial)) == target_tokens:
                messages = trial
                break
        if messages is not None:
            break
    if messages is None:
        raise RuntimeError(f"Could not construct exactly {target_tokens} input tokens")
    ids = _token_ids(tokenizer, template, config, messages)
    if len(ids) != target_tokens:
        raise RuntimeError("Synthetic fixture token count changed")
    return validate_fixture({"schema_version": SCHEMA_VERSION,
        "source": {"kind": "synthetic", "target_input_tokens": target_tokens},
        "task_id": "textcraft:synthetic:attention_probe", "seed": config["seed"],
        "messages": messages,
        "max_new_tokens": config["autonomous_support"]["model"]["model"]["max_new_tokens"],
        "expected_input_tokens": len(ids),
        "expected_input_ids_sha256": input_ids_sha256(ids)},
        config["autonomous_support"]["model"]["model"]["max_new_tokens"])


def _require_single_cuda_qwen(qwen) -> dict:
    device_map = {str(key): str(value) for key, value in qwen.model.hf_device_map.items()}
    if not device_map or any(value not in {"0", "cuda:0"} for value in device_map.values()):
        raise RuntimeError(f"Qwen must reside entirely on CUDA 0: {device_map}")
    if qwen.torch.cuda.device_count() != 1:
        raise RuntimeError("Probe requires one visible CUDA device")
    return device_map


def _generation_telemetry(directory: Path) -> dict:
    files = sorted(directory.glob("gpt_oss_cuda_memory_*.jsonl"))
    if len(files) != 1:
        raise RuntimeError("Expected exactly one GPT-OSS diagnostic telemetry file")
    events = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    starts = [item for item in events if item.get("event") == "generation_start"]
    ends = [item for item in events if item.get("event") == "generation_end"]
    if len(starts) != 1 or len(ends) != 1 or starts[0]["request_id"] != ends[0]["request_id"]:
        raise RuntimeError("Expected one complete diagnostic generation")
    return {"file": str(files[0]), "start": starts[0], "end": ends[0]}


def run_probe(fixture: dict, backend: str, output_dir: Path, *,
              worker_factory=None, qwen_factory=None) -> dict:
    if backend not in {"eager", "flex_attention"}:
        raise ValueError("Backend must be eager or flex_attention")
    if os.environ.get(CACHE_RELEASE_ENV) == "1":
        raise ValueError("Disable GPT-OSS CUDA cache release for the initial paired comparison")
    config = config_for_probe()
    spec = config["autonomous_support"]["model"]
    fixture = validate_fixture(fixture, spec["model"]["max_new_tokens"])
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if worker_factory is None:
        from .controller_backend import GptOssControllerWorker
        worker_factory = GptOssControllerWorker
    if qwen_factory is None:
        from .model import QwenModel
        qwen_factory = QwenModel
    evaluator_transformers = importlib.metadata.version("transformers")
    if evaluator_transformers != HISTORICAL_EVALUATOR_TRANSFORMERS:
        raise RuntimeError("Evaluator Transformers version differs from historical paired runs")
    output_dir.mkdir(parents=True)
    previous = {name: os.environ.get(name) for name in
                ("AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION", ENABLE_ENV, DIRECTORY_ENV)}
    os.environ["AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION"] = "1" if backend == "flex_attention" else "0"
    os.environ[ENABLE_ENV] = "1"
    gpt = None
    report = {"schema_version": SCHEMA_VERSION, "backend_requested": backend,
              "fixture_sha256": _digest(fixture), "fixture_source": fixture.get("source"),
              "max_new_tokens": fixture["max_new_tokens"],
              "evaluator_python": sys.executable,
              "evaluator_transformers_version": evaluator_transformers}
    try:
        # Match the evaluation loader order: GPT-OSS worker, then Qwen in the
        # parent process. Keep both resident during the sole GPT generation.
        gpt = worker_factory(spec, output_dir / "workers")
        report["worker_metadata"] = gpt.metadata()
        metadata = report["worker_metadata"]
        expected_decoding = {"do_sample": True, "temperature": spec["model"]["temperature"],
                             "top_p": spec["model"]["top_p"], "top_k": spec["model"]["top_k"],
                             "max_new_tokens": spec["model"]["max_new_tokens"]}
        if (metadata.get("backend_version") != spec["model"]["backend_version"] or
                metadata.get("torch_version") != HISTORICAL_WORKER_TORCH or
                metadata.get("torch_cuda_version") != HISTORICAL_WORKER_CUDA or
                "L40S" not in metadata.get("gpu_name", "") or
                metadata.get("model_id") != spec["model"]["model_id"] or
                metadata.get("model_revision") != spec["model"]["model_revision"] or
                metadata.get("quantization") != "mxfp4_packed" or
                metadata.get("decoding") != expected_decoding):
            raise RuntimeError("Worker model, runtime, GPU, quantization, or decoding differs from historical runs")
        qwen = qwen_factory(config)
        qwen.enable_gpu_memory_diagnostics(output_dir / "diagnostics")
        report["qwen_hf_device_map"] = _require_single_cuda_qwen(qwen)
        # Both conditions leave Qwen's allocator in the same post-generation
        # state, close to the co-resident evaluation workload.
        from .policy import textcraft_messages
        warmup = qwen.generate(textcraft_messages(
            "Crafting commands:\ncraft 1 oak planks using 1 oak log\n\nGoal: craft oak planks.",
            [], config["history_window"]))
        report["qwen_warmup"] = {"input_tokens": warmup["input_tokens"],
                                 "output_tokens": warmup["output_tokens"],
                                 "inference_seconds": warmup["inference_seconds"]}
        probe = gpt.probe_input(fixture["messages"])
        report["input_probe"] = probe
        if (probe["attention_implementation"] != backend or
                (fixture.get("expected_input_tokens") is not None and
                 probe["input_tokens"] != fixture["expected_input_tokens"]) or
                (fixture.get("expected_input_ids_sha256") is not None and
                 probe["input_ids_sha256"] != fixture["expected_input_ids_sha256"])):
            raise RuntimeError("Attention backend or exact input tokens differ from fixture")
        gpt.begin_episode(fixture["task_id"], fixture["seed"])
        started = time.perf_counter()
        try:
            generation = gpt.generate(fixture["messages"], max_new_tokens=fixture["max_new_tokens"])
            report["generation"] = {"status": (
                "complete" if generation["generation_status"] == "complete" else "incomplete"),
                "generation_status": generation["generation_status"],
                "input_tokens": generation["input_tokens"], "output_tokens": generation["output_tokens"],
                "inference_seconds": generation["inference_seconds"],
                "completion_sha256": hashlib.sha256(generation["raw_completion"].encode()).hexdigest()}
        except Exception as exc:
            from .controller_backend import ControllerWorkerOOM
            report["generation"] = {"status": "error", "error_type": (
                "OutOfMemoryError" if isinstance(exc, ControllerWorkerOOM) else type(exc).__name__),
                                    "error_detail": getattr(exc, "detail", str(exc))}
        report["generation"]["round_trip_seconds"] = round(time.perf_counter() - started, 3)
    finally:
        try:
            if gpt is not None:
                gpt.close()
        finally:
            for name, original in previous.items():
                if original is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = original
    report["telemetry"] = _generation_telemetry(output_dir / "diagnostics")
    write_json_atomic(output_dir / "probe_result.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    archived = sub.add_parser("export-episode", help="Extract an exact saved controller event; CPU-only")
    archived.add_argument("--episode", type=Path, required=True)
    archived.add_argument("--decision", type=int, required=True)
    archived.add_argument("--event-index", type=int, required=True)
    archived.add_argument("--output", type=Path, required=True)
    listed = sub.add_parser("list-episode", help="List saved controller inputs; CPU-only")
    listed.add_argument("--episode", type=Path, required=True)
    synthetic = sub.add_parser("synthetic", help="Create a deterministic long-context fixture; tokenizer only")
    synthetic.add_argument("--target-tokens", type=int, required=True)
    synthetic.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="Load both models and perform one GPT-OSS generation")
    run.add_argument("--fixture", type=Path, required=True)
    run.add_argument("--backend", choices=("eager", "flex_attention"), required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in {"export-episode", "synthetic"} and args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "export-episode":
        result = export_episode(args.episode, args.decision, args.event_index,
                                config_for_probe())
        write_json_atomic(args.output, result)
    elif args.command == "list-episode":
        result = list_episode_inputs(args.episode)
    elif args.command == "synthetic":
        result = synthetic_fixture(read_json(CONFIG_PATH), args.target_tokens)
        write_json_atomic(args.output, result)
    else:
        result = run_probe(json.loads(args.fixture.read_text(encoding="utf-8")),
                           args.backend, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
