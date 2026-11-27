"""Small subprocess entry point for timed CUDA and worker-import probes."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def _event(path: Path, stage: str, started: float, **details) -> None:
    now = time.monotonic()
    row = {"stage": stage, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "elapsed_seconds": round(now - started, 3),
           "monotonic_seconds": now, **details}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("cuda_identity", "worker_import"))
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    _event(args.events, "python_started", started, pid=os.getpid(), phase=args.phase)
    if args.phase == "cuda_identity":
        import torch
        _event(args.events, "torch_imported", started, torch_version=torch.__version__)
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(index) for index in range(count)]
        _event(args.events, "cuda_enumerated", started, count=count, names=names)
        print(json.dumps({"count": count, "names": names}), flush=True)
    else:
        import agent_support.gpt_oss_controller_worker
        _event(args.events, "worker_module_imported", started)
        print(json.dumps({"worker_module_imported": True}), flush=True)


if __name__ == "__main__":
    main()
