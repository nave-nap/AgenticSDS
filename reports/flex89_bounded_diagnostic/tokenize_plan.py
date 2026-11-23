"""CPU-only pinned-tokenizer check and token-ID manifest for the archived replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agent_support.gpt_oss_attention_probe import _pinned_tokenizer, _token_ids
from agent_support.input_token_fingerprint import input_ids_sha256
from reports.flex89_bounded_diagnostic.replay import validate_plan


def tokenize(plan: dict, config: dict) -> dict:
    requests = validate_plan(plan, config)
    tokenizer, template = _pinned_tokenizer(config)
    rows = []
    for request in requests:
        ids = _token_ids(tokenizer, template, config, request["messages"])
        if len(ids) != request["expected_input_tokens"]:
            raise ValueError(f"Pinned tokenizer count differs at request {request['request_id']}: "
                             f"{len(ids)} != {request['expected_input_tokens']}")
        rows.append({"request_id": request["request_id"],
                     "input_tokens": len(ids), "input_ids_sha256": input_ids_sha256(ids),
                     "messages_sha256": request["messages_sha256"]})
    return {"schema_version": 1, "model_revision": config["autonomous_support"]["model"]["model"]["model_revision"],
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    config = json.loads(args.config.read_text())
    result = tokenize(plan, config)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "verified", "requests": len(result["rows"]) }))


if __name__ == "__main__":
    main()
