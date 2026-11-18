"""Exact saved v2 final-channel structures and v3 safe extraction."""

import json
import re
from pathlib import Path

import pytest

from agent_support.gpt_oss_model import (extract_decoded_final_v3,
                                         parse_configured_completion,
                                         parse_harmony_completion,
                                         parse_harmony_completion_v3)
from agent_support.llm_support_controller import parse_controller_decision
from agent_support.autonomous_support import (SupportBudget, SupportContext, SupportExecutor,
                                              make_support_controller, run_support_loop)
from agent_support.policy import parse_textcraft
from agent_support.reparse_controller_completions import reparse_run


FIXTURE = Path(__file__).parent / "fixtures/v2_controller_final_channels.json"
CASES = {item["decision"]: item for item in json.loads(FIXTURE.read_text())["cases"]}
AVAILABLE = {"retrieve_episodic_memory_v1", "inspect_history_v1",
             "reason_over_artifacts_v1", "inspect_state_v1"}


class HarmonyTokenizer:
    """CPU stand-in preserving the pinned special-token boundaries."""

    markers = {name: 1_100_000 + index for index, name in enumerate(
        ("<|start|>", "<|channel|>", "<|message|>", "<|constrain|>",
         "<|end|>", "<|return|>", "<|call|>"))}

    def encode(self, value, add_special_tokens=False):
        pieces = re.split("(" + "|".join(re.escape(key) for key in self.markers) + ")", value)
        return [token for piece in pieces for token in
                ([self.markers[piece]] if piece in self.markers else [ord(char) for char in piece])]

    def decode(self, ids, skip_special_tokens=False):
        inverse = {value: key for key, value in self.markers.items()}
        return "".join(inverse[token] if token in inverse else chr(token) for token in ids)

    def convert_tokens_to_ids(self, value):
        return self.markers[value]


@pytest.mark.parametrize("decision", (2, 3, 4, 6, 14, 20, 21, 22))
def test_exact_real_completion_structures(decision):
    tokenizer = HarmonyTokenizer()
    raw = CASES[decision]["raw_completion"]
    ids = tokenizer.encode(raw)
    old = parse_harmony_completion(tokenizer, ids, len(ids) + 1)
    new = parse_harmony_completion_v3(tokenizer, ids, len(ids) + 1)
    assert new["raw_completion"] == raw
    if decision in {4}:
        assert new["generation_status"] == "malformed_final_header"
        assert new["text"] == "" and new["recovered_final_channel"] is False
    elif decision in {14, 22}:
        assert old["generation_status"] == new["generation_status"] == "complete"
        assert new["recovered_final_channel"] is False
    else:
        assert old["generation_status"] == "missing_final"
        assert new["generation_status"] == "complete"
        assert new["recovered_final_channel"] is True
        assert new["final_channel_format"].startswith("constrain_")
    if decision in {6, 14}:
        with pytest.raises(ValueError):
            parse_controller_decision(new["text"], AVAILABLE)
    elif new["text"]:
        choice = parse_controller_decision(new["text"], AVAILABLE)
        assert choice.kind in {"execute", "request"}
    if decision in {20, 21, 22}:
        assert parse_controller_decision(new["text"], AVAILABLE).action == (
            "craft 8 cyan stained glass using 8 glass, 1 cyan dye")


def test_malformed_real_headers_and_missing_required_request_field_stay_rejected():
    tokenizer = HarmonyTokenizer()
    for decision in (4, 16):
        ids = tokenizer.encode(CASES[decision]["raw_completion"])
        parsed = parse_harmony_completion_v3(tokenizer, ids, len(ids) + 1)
        assert parsed["generation_status"] == "malformed_final_header"
        assert parsed["text"] == ""
    ids = tokenizer.encode(CASES[8]["raw_completion"])
    parsed = parse_harmony_completion_v3(tokenizer, ids, len(ids) + 1)
    assert parsed["generation_status"] == "complete"
    with pytest.raises(ValueError, match="invalid fields"):
        parse_controller_decision(parsed["text"], AVAILABLE)


def test_parser_selection_is_explicit_and_preserves_v2_default():
    tokenizer = HarmonyTokenizer()
    ids = tokenizer.encode(CASES[20]["raw_completion"])
    assert parse_configured_completion(tokenizer, ids, len(ids) + 1,
                                       "strict_final_v1")["generation_status"] == "missing_final"
    assert parse_configured_completion(tokenizer, ids, len(ids) + 1,
                                       "harmony_final_v3")["generation_status"] == "complete"
    with pytest.raises(ValueError, match="Unknown GPT-OSS completion parser"):
        parse_configured_completion(tokenizer, ids, len(ids) + 1, "unlisted")


@pytest.mark.parametrize("raw", [
    '<|channel|>analysis<|message|>{"kind":"execute","action":"inventory","reason":"x"}<|return|>',
    '<|channel|>analysis<|message|>thought<|end|><|start|>assistant<|channel|>final <|constrain|>yaml<|message|>{"kind":"execute","action":null,"reason":"x"}<|return|>',
    '<|channel|>analysis<|message|>thought<|end|><|start|>assistant<|channel|>final <|constrain|>json<|message|>{"kind":"execute","action":null,"reason":"x"}<|return|><|return|>',
    '<|channel|>analysis<|message|>thought<|end|><|start|>assistant<|channel|>final <|constrain|>json<|message|>{"kind":"execute","action":"<|start|>","reason":"x"}<|return|>',
])
def test_unframed_or_ambiguous_json_is_never_recovered(raw):
    tokenizer = HarmonyTokenizer()
    ids = tokenizer.encode(raw)
    parsed = parse_harmony_completion_v3(tokenizer, ids, len(ids) + 1)
    assert parsed["text"] == ""
    assert parsed["generation_status"] != "complete"
    assert extract_decoded_final_v3(raw) is None


def test_offline_reparse_classifies_saved_structures_without_replay(tmp_path):
    config = json.loads((Path(__file__).resolve().parents[1] /
                         "configs/autonomous_evaluation/autonomous_v2.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "episodes").mkdir()
    trace = []
    for decision, item in CASES.items():
        trace.append({"decision": decision, "support_trace": {"events": [{
            "state": {"benchmark": "textcraft", "capabilities": [
                {"operation": name} for name in AVAILABLE]},
            "controller_metadata": {"raw_completion": item["raw_completion"],
                                    "generation_status": item["original_generation_status"],
                                    "parse_error": ("old parser rejected output" if
                                                    item["original_generation_status"] != "complete" or
                                                    decision == 14 else None)}}]}})
    (tmp_path / "episodes" / "episode.json").write_text(json.dumps({"task_id": "textcraft:seed:51",
                                                                  "trace": trace}))
    result = reparse_run(tmp_path)
    assert result["controller_completions"] == 10
    assert result["counts"] == {"already_accepted": 1, "recoverable_final_variant": 4,
                                 "valid_json_schema_invalid": 3, "malformed_or_unsafe": 2}


@pytest.mark.parametrize("decision,expected_stop", [(20, "execute"),
                                                    (6, "malformed_controller_output")])
def test_v3_loop_uses_recovered_content_only_after_schema_validation(decision, expected_stop):
    config = json.loads((Path(__file__).resolve().parents[1] /
                         "configs/textcraft_prompted_gptoss_controller.json").read_text())
    config["condition"] = "autonomous_support_v3"
    config["autonomous_support"]["controller"] = "prompted_gpt_oss_v3"
    config["autonomous_support"]["model"]["model"]["completion_parser"] = "harmony_final_v3"
    tokenizer = HarmonyTokenizer()
    ids = tokenizer.encode(CASES[decision]["raw_completion"])
    class Model:
        def count_messages_tokens(self, messages):
            return 50
        def generate(self, messages, max_new_tokens=None):
            return {**parse_harmony_completion_v3(tokenizer, ids, len(ids) + 1),
                    "input_tokens": 50, "output_tokens": 227,
                    "inference_seconds": 0.1}
    original = "craft 8 cyan stained glass using 8 glass, 2 cyan dye"
    context = SupportContext("textcraft", "textcraft:seed:51", "Goal: craft cyan stained glass.",
                             "Could not find a valid recipe", (), (), 20, original, original,
                             capabilities=({"operation": "inspect_history_v1", "arguments": {}},))
    budget = SupportBudget.from_config(config["autonomous_support"]["budget"])
    trace = run_support_loop(make_support_controller(config, Model()), SupportExecutor(),
                             budget, context, parse_textcraft)
    assert trace.termination_reason == expected_stop
    assert trace.events[0]["controller_metadata"]["recovered_final_channel"] is True
    assert budget.controller_calls_used == 1
    if decision == 20:
        assert trace.final_action == "craft 8 cyan stained glass using 8 glass, 1 cyan dye"
    else:
        assert trace.final_action == original
        assert trace.events[0]["controller_metadata"]["parse_error"]


def test_malformed_real_header_consumes_one_call_and_falls_back_safely():
    config = json.loads((Path(__file__).resolve().parents[1] /
                         "configs/textcraft_prompted_gptoss_controller.json").read_text())
    config["condition"] = "autonomous_support_v3"
    config["autonomous_support"]["controller"] = "prompted_gpt_oss_v3"
    config["autonomous_support"]["model"]["model"]["completion_parser"] = "harmony_final_v3"
    tokenizer = HarmonyTokenizer()
    ids = tokenizer.encode(CASES[4]["raw_completion"])
    class Model:
        def count_messages_tokens(self, messages):
            return 50
        def generate(self, messages, max_new_tokens=None):
            return {**parse_harmony_completion_v3(tokenizer, ids, len(ids) + 1),
                    "input_tokens": 50, "output_tokens": 433, "inference_seconds": 0.1}
    original = "get 8 glass"
    context = SupportContext("textcraft", "textcraft:seed:51", "Goal: craft cyan stained glass.",
                             "Inventory: empty", (), (), 4, original, original)
    budget = SupportBudget.from_config(config["autonomous_support"]["budget"])
    trace = run_support_loop(make_support_controller(config, Model()), SupportExecutor(),
                             budget, context, parse_textcraft)
    assert trace.termination_reason == "malformed_controller_output"
    assert trace.final_action == original
    assert budget.controller_calls_used == 1 and budget.controller_output_tokens == 433
    metadata = trace.events[0]["controller_metadata"]
    assert metadata["generation_status"] == "malformed_final_header"
    assert metadata["recovered_final_channel"] is False
