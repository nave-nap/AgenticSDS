"""Verify the pinned GPT-OSS tokenizer's reservation without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from importlib.metadata import version

from agent_support.common import read_json
from agent_support.gpt_oss_model import GptOss20BModel
from agent_support.policy import textcraft_messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/standalone_textcraft_dev_gptoss20b_low_never.json")
    args = parser.parse_args()
    config = read_json(args.config)
    if version("transformers") != "5.22.2" or config["backend_version"] != "5.22.2":
        raise RuntimeError("This check requires the pinned Transformers 5.22.2 environment")
    from transformers import AutoTokenizer

    adapter = object.__new__(GptOss20BModel)
    adapter.tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], local_files_only=True)
    adapter.reasoning_effort = config["reasoning_effort"]
    template = adapter.tokenizer.chat_template
    date_expression = 'strftime_now("%Y-%m-%d")'
    if not isinstance(template, str) or template.count(date_expression) != 1:
        raise RuntimeError("Pinned tokenizer has an unexpected Harmony date expression")
    adapter.chat_template = template.replace(date_expression, f'"{config["prompt_date"]}"')
    template_sha256 = hashlib.sha256(adapter.chat_template.encode()).hexdigest()
    if template_sha256 != "517c99708cfdbfe89e2554d167442df5702de865bd63d54eef3ff379ed83d6e4":
        raise RuntimeError(f"Pinned Harmony template hash changed: {template_sha256}")

    initial = read_json(config["manifest"])["tasks"][0]["initial_observation"]
    counts = []
    for history_length in (0, 8):
        history = [{"action": "get 1 wood", "observation": "Could not obtain the item."}
                   for _ in range(history_length)]
        messages = textcraft_messages(initial, history, config["history_window"])
        old_encoded = adapter._encode_messages(messages, tensors=False)
        if not isinstance(old_encoded, Mapping) or "input_ids" not in old_encoded:
            raise RuntimeError(f"Expected Transformers v5 BatchEncoding, got {type(old_encoded).__name__}")
        old_count = len(old_encoded)
        counted = adapter.count_messages_tokens(messages)
        generation_input = adapter._encode_messages(messages, tensors=True)
        actual_input = int(generation_input["input_ids"].shape[1])
        if counted != actual_input or old_count >= actual_input:
            raise RuntimeError(f"Token-count mismatch: old={old_count}, counted={counted}, actual={actual_input}")
        counts.append(counted)
        print(f"history={history_length} BatchEncoding_fields={old_count} "
              f"input_tokens={counted} old_reservation={old_count + config['max_new_tokens']} "
              f"correct_reservation={counted + config['max_new_tokens']}")
    if counts[1] <= counts[0]:
        raise RuntimeError("Longer history did not increase the pinned token count")
    print(f"model_revision={config['model_revision']} template_sha256={template_sha256} "
          f"reasoning_effort={adapter.reasoning_effort} prompt_date={config['prompt_date']}")


if __name__ == "__main__":
    main()
