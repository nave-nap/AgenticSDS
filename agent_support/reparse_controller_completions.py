"""Read-only offline audit of saved GPT-OSS controller completions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from .autonomous_support import make_support_executor
from .gpt_oss_model import extract_decoded_final_v3
from .llm_support_controller import parse_controller_decision


CATEGORIES = ("already_accepted", "recoverable_final_variant",
              "valid_json_schema_invalid", "malformed_or_unsafe")


def reparse_run(run_dir: str | Path) -> dict:
    """Classify saved text; never replay actions or mutate run artifacts."""
    root = Path(run_dir)
    config = json.loads((root / "config.json").read_text())
    memory_spec = config["autonomous_support"].get("memory")
    memory = (SimpleNamespace(max_top_k=memory_spec["max_top_k"],
                              max_example_tokens=memory_spec["max_result_tokens"])
              if memory_spec else None)
    executor = make_support_executor(config, controller_model=object(), memory=memory)
    rows = []
    for episode_path in sorted((root / "episodes").glob("*.json")):
        episode = json.loads(episode_path.read_text())
        for record in episode["trace"]:
            for step, event in enumerate(record.get("support_trace", {}).get("events", []), 1):
                metadata = event.get("controller_metadata")
                if metadata is None:
                    continue
                extracted = extract_decoded_final_v3(metadata.get("raw_completion"))
                row = {"task_id": episode["task_id"], "decision": record["decision"],
                       "controller_step": step, "original_generation_status": metadata.get("generation_status"),
                       "final_channel_format": extracted["final_channel_format"] if extracted else None}
                if extracted is None:
                    row.update(category="malformed_or_unsafe", detail="No fully bounded final message")
                else:
                    try:
                        json.loads(extracted["text"])
                    except (TypeError, ValueError) as exc:
                        row.update(category="malformed_or_unsafe", detail=str(exc))
                    else:
                        available = {item["operation"] for item in event["state"]["capabilities"]}
                        try:
                            decision = parse_controller_decision(extracted["text"], available)
                            if decision.request is not None:
                                operation = executor.operations[decision.request.operation]
                                operation.validate_arguments(
                                    decision.request.arguments,
                                    SimpleNamespace(benchmark=event["state"]["benchmark"]))
                        except (ValueError, KeyError, TypeError) as exc:
                            row.update(category="valid_json_schema_invalid", detail=str(exc))
                        else:
                            row["decision_kind"] = decision.kind
                            row["operation"] = decision.request.operation if decision.request else None
                            if (metadata.get("generation_status") == "complete" and
                                    not metadata.get("parse_error") and
                                    extracted["final_channel_format"] == "canonical"):
                                row["category"] = "already_accepted"
                            elif extracted["final_channel_format"] != "canonical":
                                row["category"] = "recoverable_final_variant"
                            else:
                                row.update(category="malformed_or_unsafe",
                                           detail="Original parser rejected a canonical final message")
                rows.append(row)
    counts = Counter(row["category"] for row in rows)
    return {"run": str(root), "controller_completions": len(rows),
            "counts": {key: counts[key] for key in CATEGORIES},
            "rows": rows,
            "interpretation": "Offline parsing only; saved environment actions and outcomes are unchanged."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(json.dumps(reparse_run(args.run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
