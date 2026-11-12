"""One persistent model per process; JSON lines on stdout, diagnostics on stderr."""

from __future__ import annotations

import contextlib
import json
import os
import sys

from .components import load_components
from .model import make_model


def cuda_memory() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"allocated_bytes": torch.cuda.memory_allocated(0),
                    "reserved_bytes": torch.cuda.memory_reserved(0),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    "total_bytes": torch.cuda.get_device_properties(0).total_memory}
    except ImportError:
        pass
    return {}


def serve(input_stream=None, output_stream=None, model_factory=make_model) -> None:
    source = input_stream or sys.stdin
    sink = output_stream or sys.stdout
    model = None
    try:
        for line in source:
            try:
                request = json.loads(line)
                operation = request["op"]
                if operation == "init":
                    if model is not None:
                        raise ValueError("Model worker is already initialized")
                    config = request["config"]
                    with contextlib.redirect_stdout(sys.stderr):
                        load_components(config)
                        model = model_factory(config)
                        metadata = model.metadata()
                        memory = cuda_memory()
                    response = {"ok": True, "metadata": metadata, "cuda_memory": memory}
                elif operation == "generate":
                    if model is None:
                        raise ValueError("Model worker is not initialized")
                    with contextlib.redirect_stdout(sys.stderr):
                        generation = model.generate(request["messages"])
                        memory = cuda_memory()
                    response = {"ok": True, "generation": generation, "cuda_memory": memory}
                elif operation == "count_messages_tokens":
                    if model is None:
                        raise ValueError("Model worker is not initialized")
                    with contextlib.redirect_stdout(sys.stderr):
                        response = {"ok": True, "count": model.count_messages_tokens(request["messages"])}
                elif operation == "seed":
                    if model is None:
                        raise ValueError("Model worker is not initialized")
                    import random
                    random.seed(request["seed"])
                    with contextlib.redirect_stdout(sys.stderr):
                        import torch
                        torch.manual_seed(request["seed"])
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(request["seed"])
                    response = {"ok": True}
                elif operation == "memory":
                    with contextlib.redirect_stdout(sys.stderr):
                        response = {"ok": True, "cuda_memory": cuda_memory()}
                elif operation == "shutdown":
                    response = {"ok": True}
                    sink.write(json.dumps(response) + "\n")
                    sink.flush()
                    break
                else:
                    raise ValueError(f"Unknown worker operation: {operation}")
            except Exception as exc:
                response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            sink.write(json.dumps(response, ensure_ascii=False) + "\n")
            sink.flush()
    finally:
        if model is not None and hasattr(model, "close"):
            model.close()


if __name__ == "__main__":
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        serve(output_stream=protocol)
    finally:
        protocol.close()
