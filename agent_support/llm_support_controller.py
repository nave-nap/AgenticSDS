"""Prompted GPT-OSS support selection baseline with strict JSON decisions."""

from __future__ import annotations

import json

from .autonomous_support import (CONTROLLERS, ControllerResponse, SupportContext,
                                 SupportCost, SupportDecision, SupportRequest,
                                 SupportReservation, visible_state_digest)
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
        if not getattr(self, "include_support_history", False):
            state.pop("support_history", None)
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
        if getattr(self, "record_raw_completion", False):
            metadata.update(raw_completion=generation.get("raw_completion"),
                            reasoning_tokens=generation.get("reasoning_tokens"),
                            final_tokens=generation.get("final_tokens"),
                            final_channel_format=generation.get("final_channel_format"),
                            recovered_final_channel=generation.get("recovered_final_channel"),
                            generation_prompt_tail=generation.get("generation_prompt_tail"),
                            last_generated_token_id=generation.get("last_generated_token_id"),
                            stopped_by_final_return=generation.get("stopped_by_final_return"),
                            hit_token_limit=generation.get("hit_token_limit"),
                            peak_gpu_memory_allocated_bytes=generation.get("peak_gpu_memory_allocated_bytes"),
                            peak_gpu_memory_reserved_bytes=generation.get("peak_gpu_memory_reserved_bytes"))
        try:
            if metadata["generation_status"] != "complete":
                raise ValueError(f"Incomplete generation: {metadata['generation_status']}")
            choice = parse_controller_decision(raw, {item["operation"] for item in context.capabilities})
        except (ValueError, TypeError, KeyError) as exc:
            metadata["parse_error"] = str(exc)
            choice = SupportDecision("execute", reason="malformed_controller_output")
        return ControllerResponse(choice, cost=cost, metadata=metadata)


SYSTEM_PROMPT_V2 = """You decide whether a task agent should seek assistance before its next environment action.
Use only visible task evidence and the listed capabilities. Each support request costs an additional controller decision after its result. Execute the current candidate immediately when support is unlikely to improve it. You may also execute a different valid action if visible evidence justifies the change.
Each capability returns information; none performs an environment action. A request with the same arguments after an empty result and unchanged visible evidence cannot reveal new information. An inventory-only state inspection also remains empty until an inventory environment action has been observed, regardless of its limit or query. Inspecting visible state cannot query the simulator or discover hidden items. If fresh inventory is needed, you may execute the benchmark's inventory action as the final environment action when valid.
An "ok" state inspection can still have "inventory_observed":false; read the returned fields and status rather than assuming its objective was fulfilled.
After every support result, decide again using its status and information. No support sequence or capability is required. Consider the remaining budget and the value of another call.
Return exactly one JSON object in the final channel, without markdown, prose, or extra keys. Never request code or shell execution.
Execute schema: {"kind":"execute","action":null,"reason":"brief evidence-based reason"}. Use null to keep the current candidate; otherwise provide one valid benchmark action.
Request schema: {"kind":"request","operation":"exact listed operation ID","arguments":{},"objective":"specific question for this operation","reason":"brief evidence-based reason"}. Use exactly the argument keys and values described for the selected capability. Request one operation only."""


class PromptedGptOssControllerV2(PromptedGptOssController):
    """Versioned reliability prompt; v1 run and configuration remain frozen."""

    controller_id = "prompted_gpt_oss_v2"
    prompt_version = "support_decision_json_v2"
    include_support_history = True
    record_raw_completion = True

    def _messages(self, context: SupportContext) -> list[dict]:
        messages = super()._messages(context)
        messages[0]["content"] = SYSTEM_PROMPT_V2
        state = json.loads(messages[1]["content"])
        state["visible_state_sha256"] = visible_state_digest(context)
        for capability in state["capabilities"]:
            name = capability["operation"]
            if name == "inspect_state_v1":
                inventory_observed = any(action == "inventory" for action, _ in context.history)
                capability["availability"] = ("inventory observed" if inventory_observed else
                    "inventory not observed; inventory scope will return empty until an inventory environment action")
                capability["limitations"] = ("Recipes are already in the task prompt; craft feedback is prior "
                    "episode feedback. No live or hidden state query.")
                capability["arguments"] = {"type": "object", "required": ["scope", "limit"],
                    "properties": {"scope": {"type": "string", "enum": ["inventory", "recipes",
                                                                  "craft_feedback", "all"]},
                                   "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                                   "query": {"type": "string", "maxLength": 128}},
                    "additionalProperties": False}
                capability["argument_example"] = {"scope": "recipes", "limit": 5}
            elif name == "inspect_history_v1":
                capability["arguments"] = {"type": "object", "required": ["limit"],
                    "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 8},
                                   "scope": {"type": "string", "enum": ["recent", "failures", "repeated"]}},
                    "additionalProperties": False}
                capability["argument_example"] = {"scope": "failures", "limit": 5}
            elif name == "retrieve_episodic_memory_v1":
                capability["arguments"] = {"type": "object", "required": ["query", "scope", "top_k"],
                    "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 1000},
                                   "scope": {"type": "string", "enum": ["task", "steps", "both"]},
                                   "top_k": {"type": "integer", "minimum": 1, "maximum": self.max_top_k}},
                    "additionalProperties": False}
                capability["argument_example"] = {"query": "specific evidence question", "scope": "both",
                                                   "top_k": min(2, self.max_top_k)}
            elif name == "reason_over_artifacts_v1":
                capability["arguments"] = {"type": "object", "required": ["objective", "output_type"],
                    "properties": {"objective": {"type": "string", "minLength": 1, "maxLength": 512},
                                   "output_type": {"type": "string",
                                                   "enum": ["note", "plan", "candidate_action"]}},
                    "additionalProperties": False}
                capability["argument_example"] = {"objective": "specific evidence to assess",
                                                   "output_type": "candidate_action"}
        messages[1]["content"] = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        return messages

    def __init__(self, config: dict, model):
        super().__init__(config, model)
        self.max_top_k = config["autonomous_support"].get("memory", {}).get("max_top_k", 1)

    def decide(self, context: SupportContext) -> ControllerResponse:
        response = super().decide(context)
        request = response.decision.request
        no_inventory_observation = not any(action == "inventory" for action, _ in context.history)
        if request and any(
                item.get("operation") == request.operation and
                item.get("status") == "empty" and
                ((item.get("arguments") == request.arguments and
                  item.get("visible_state_sha256") == visible_state_digest(context)) or
                 (request.operation == "inspect_state_v1" and
                  item.get("arguments", {}).get("scope") == "inventory" and
                  request.arguments.get("scope") == "inventory" and
                  type(request.arguments.get("limit")) is int and
                  1 <= request.arguments["limit"] <= 20 and
                  set(request.arguments) <= {"scope", "limit", "query"} and
                  isinstance(request.arguments.get("query", ""), str) and
                  no_inventory_observation))
                for item in context.support_history):
            return ControllerResponse(
                SupportDecision("execute", reason="repeated_empty_request_with_unchanged_visible_state"),
                cost=response.cost,
                metadata={**response.metadata, "blocked_repeated_empty_request": request.as_dict()})
        return response


class PromptedGptOssControllerV3(PromptedGptOssControllerV2):
    """Same v2 decision prompt with a versioned Harmony extraction backend."""

    controller_id = "prompted_gpt_oss_v3"


CONTROLLERS[PromptedGptOssController.controller_id] = PromptedGptOssController
CONTROLLERS[PromptedGptOssControllerV2.controller_id] = PromptedGptOssControllerV2
CONTROLLERS[PromptedGptOssControllerV3.controller_id] = PromptedGptOssControllerV3
