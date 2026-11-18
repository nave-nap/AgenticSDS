"""Prompted GPT-OSS support selection baseline with strict JSON decisions."""

from __future__ import annotations

import json

from .autonomous_support import (CONTROLLERS, ControllerResponse, SupportContext,
                                 SupportCost, SupportDecision, SupportRequest,
                                 SupportReservation)
from .common import canonical_digest


SYSTEM_PROMPT = """You are a support-planning controller for a task agent.
Decide whether additional assistance is likely to improve the very next environment action.
Support has computational cost. Executing the current valid action immediately is often appropriate.
You may request one available capability, inspect its result on the next turn, and then choose again.
Do not assume a capability is useful merely because it is available. Do not follow a fixed sequence.
Use only the information in the supplied state. Support results may be incomplete or mistaken.
Return exactly one JSON object and no other text. Never request code or shell execution.

For execute: {"kind":"execute","action":null,"reason":"short reason"}.
Use null to keep the current candidate; otherwise give one action in the benchmark's required output format.
For request: {"kind":"request","operation":"available operation ID","arguments":{},"objective":"short objective","reason":"short reason"}.
The operation and arguments must match a listed capability. Request only one operation per response."""


def parse_controller_decision(raw: str, available: set[str]) -> SupportDecision:
    """Reject prose, code fences, unknown fields, and unlisted capabilities."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Controller output is not one JSON object") from exc
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str) or value["kind"] not in {"execute", "request"}:
        raise ValueError("Controller decision kind is invalid")
    if value["kind"] == "execute":
        if (set(value) != {"kind", "action", "reason"} or
                value["action"] is not None and not isinstance(value["action"], str)):
            raise ValueError("Execute decision has invalid fields")
        return SupportDecision("execute", reason=value["reason"], action=value["action"])
    if set(value) != {"kind", "operation", "arguments", "objective", "reason"}:
        raise ValueError("Support request has invalid fields")
    if not isinstance(value["operation"], str) or value["operation"] not in available:
        raise ValueError("Controller requested an unavailable capability")
    request = SupportRequest(value["operation"], value["arguments"], value["objective"])
    return SupportDecision("request", request=request, reason=value["reason"])


class PromptedGptOssController:
    controller_id = "prompted_gpt_oss_v1"
    prompt_version = "support_decision_json_v1"

    def __init__(self, config: dict, model):
        if model is None:
            raise ValueError("Prompted controller requires a counted model backend")
        self.model = model
        self.history_window = config["history_window"]
        spec = config["autonomous_support"]
        self.max_new_tokens = spec["model"]["model"].get("max_new_tokens") if spec["model"]["type"] == "gpt_oss_20b_worker_v1" else spec["model"]["max_new_tokens"]
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("Controller maximum output tokens must be positive")

    def _messages(self, context: SupportContext) -> list[dict]:
        state = context.as_dict()
        state.pop("task_id", None)  # Manifest identity is not agent-visible task evidence.
        state["history"] = state["history"][-self.history_window:]
        # The benchmark parser handles the selected action. The controller sees
        # the same public task state and support results as the task agent.
        format_hint = ("ALFWorld action must be a one-based number from the current admissible list."
                       if context.benchmark == "alfworld" else
                       "TextCraft action must be one Action: get/craft/inventory command.")
        state["action_format"] = format_hint
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))}]

    def reservation(self, context: SupportContext) -> SupportReservation:
        count = self.model.count_messages_tokens(self._messages(context))
        if type(count) is not int or count < 0:
            raise ValueError("Controller backend returned an invalid token count")
        self._reserved_input_tokens = count
        return SupportReservation(1, count, self.max_new_tokens)

    def decide(self, context: SupportContext) -> ControllerResponse:
        messages = self._messages(context)
        try:
            generation = self.model.generate(messages, max_new_tokens=self.max_new_tokens)
        except Exception as exc:
            return ControllerResponse(
                SupportDecision("execute", reason="controller_backend_error"),
                cost=SupportCost(1, getattr(self, "_reserved_input_tokens", 0), 0),
                metadata={"prompt_version": self.prompt_version,
                          "prompt_sha256": canonical_digest(messages),
                          "backend_error": repr(exc)})
        try:
            cost = SupportCost(1, generation["input_tokens"], generation["output_tokens"],
                               generation["inference_seconds"])
            raw = generation["text"]
            if not isinstance(raw, str):
                raise ValueError("Generation text is missing")
        except (KeyError, TypeError, ValueError) as exc:
            return ControllerResponse(
                SupportDecision("execute", reason="controller_backend_error"),
                cost=SupportCost(1, getattr(self, "_reserved_input_tokens", 0), 0),
                metadata={"prompt_version": self.prompt_version,
                          "prompt_sha256": canonical_digest(messages),
                          "backend_error": repr(exc), "usage_estimated": True})
        metadata = {"prompt_version": self.prompt_version,
                    "prompt_sha256": canonical_digest(messages),
                    "raw_output": raw[:4096],
                    "generation_status": generation.get("generation_status", "complete")}
        try:
            if metadata["generation_status"] != "complete":
                raise ValueError(f"Incomplete generation: {metadata['generation_status']}")
            choice = parse_controller_decision(raw, {item["operation"] for item in context.capabilities})
        except (ValueError, TypeError, KeyError) as exc:
            metadata["parse_error"] = str(exc)
            choice = SupportDecision("execute", reason="malformed_controller_output")
        return ControllerResponse(choice, cost=cost, metadata=metadata)


CONTROLLERS[PromptedGptOssController.controller_id] = PromptedGptOssController
