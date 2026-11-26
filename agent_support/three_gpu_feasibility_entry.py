"""Stdlib-only entry point that records import failures before feasibility.run."""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    try:
        importlib.import_module("agent_support.three_gpu_feasibility").main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            return
        try:
            position = sys.argv.index("--output-dir")
            directory = Path(sys.argv[position + 1])
            result_path = directory / "feasibility_result.json"
            if directory.exists() and not result_path.exists():
                raise FileExistsError(f"Refusing to write into an existing attempt: {directory}")
            if not result_path.exists():
                directory.mkdir(parents=True, exist_ok=True)
                with result_path.open("x", encoding="utf-8") as stream:
                    json.dump({"status": "failed", "startup_stage": "entry_import_or_dispatch",
                               "error_type": type(exc).__name__, "error_detail": str(exc),
                               "traceback": traceback.format_exc()}, stream, indent=2)
                    stream.write("\n")
        except (OSError, ValueError, IndexError) as record_error:
            print(f"Could not record feasibility entry failure: {record_error!r}",
                  file=sys.stderr, flush=True)
        if not isinstance(exc, SystemExit):
            traceback.print_exc(file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
