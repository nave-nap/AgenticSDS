"""Persistent GPT-OSS controller worker in its pinned Python environment."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback

from .gpu_memory_diagnostics import cache_release_enabled
from .gpt_oss_two_gpu import enabled as two_gpu_enabled


def serve(source=None, sink=None, model_factory=None) -> None:
    source, sink = source or sys.stdin, sink or sys.stdout
    model = None
    for line in source:
        release_after_generation = False
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "init":
                if model is not None:
                    raise ValueError("Controller model is already initialized")
                if model_factory is None:
                    from .gpt_oss_model import GptOss20BModel
                    model_factory = GptOss20BModel
                with contextlib.redirect_stdout(sys.stderr):
                    model = model_factory(request["config"])
                    metadata = model.metadata()
                response = {"ok": True, "metadata": metadata}
            elif op == "count_messages_tokens":
                if model is None:
                    raise ValueError("Controller model is not initialized")
                response = {"ok": True, "count": model.count_messages_tokens(request["messages"])}
            elif op == "generate":
                if model is None:
                    raise ValueError("Controller model is not initialized")
                messages = request["messages"]
                max_new_tokens = request.get("max_new_tokens")
                release_after_generation = True
                with contextlib.redirect_stdout(sys.stderr):
                    generated = model.generate(messages, max_new_tokens=max_new_tokens)
                response = {"ok": True, "generation": generated}
            elif op == "seed":
                if model is None:
                    raise ValueError("Controller model is not initialized")
                import random
                import torch
                random.seed(request["seed"])
                torch.manual_seed(request["seed"])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(request["seed"])
                response = {"ok": True}
            elif op == "shutdown":
                response = {"ok": True}
                sink.write(json.dumps(response) + "\n")
                sink.flush()
                break
            else:
                raise ValueError(f"Unknown controller worker operation: {op}")
        except Exception as exc:
            if two_gpu_enabled():
                traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        if release_after_generation and cache_release_enabled():
            # The generation frame and exception handler have both unwound here.
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    model.release_cuda_cache()
            except Exception:
                pass  # Cache maintenance must not replace the generation response.
        sink.write(json.dumps(response, ensure_ascii=False) + "\n")
        sink.flush()


if __name__ == "__main__":
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        serve(sink=protocol)
    finally:
        protocol.close()
