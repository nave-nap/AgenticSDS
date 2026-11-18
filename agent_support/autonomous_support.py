"""Bounded, sequential support execution for a new opt-in runner condition.

Controllers choose one registered operation at a time. Operations cannot step
the benchmark environment; the caller alone validates and executes an action.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Protocol


@dataclass(frozen=True)
class SupportCost:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    inference_seconds: float = 0.0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in
               (self.model_calls, self.input_tokens, self.output_tokens)):
            raise ValueError("Support cost counts must be nonnegative integers")
        if not isinstance(self.inference_seconds, (int, float)) or not math.isfinite(
                self.inference_seconds) or self.inference_seconds < 0:
            raise ValueError("Support inference time must be finite and nonnegative")

    def as_dict(self) -> dict:
        return {"model_calls": self.model_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "inference_seconds": self.inference_seconds}


@dataclass(frozen=True)
class SupportReservation:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in
               (self.model_calls, self.input_tokens, self.output_tokens)):
            raise ValueError("Support reservations must be nonnegative integers")

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def covers(self, cost: SupportCost) -> bool:
        return (cost.model_calls <= self.model_calls and
                cost.input_tokens <= self.input_tokens and
                cost.output_tokens <= self.output_tokens)


@dataclass(frozen=True)
class SupportRequest:
    operation: str
    arguments: dict = field(default_factory=dict)
    objective: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("Support operation must be a nonempty name")
        if not isinstance(self.arguments, dict) or any(not isinstance(key, str)
                                                       for key in self.arguments):
            raise ValueError("Support arguments must be a JSON object")
        try:
            encoded = json.dumps(self.arguments, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Support arguments must be JSON data") from exc
        if len(encoded) > 8192:
            raise ValueError("Support arguments are too large")
        if self.objective is not None and (not isinstance(self.objective, str) or
                                           len(self.objective) > 512):
            raise ValueError("Support objective is too large or invalid")

    def as_dict(self) -> dict:
        return {"operation": self.operation, "arguments": copy.deepcopy(self.arguments),
                "objective": self.objective}


@dataclass(frozen=True)
class SupportDecision:
    kind: Literal["execute", "request"]
    request: SupportRequest | None = None
    reason: str = ""
    action: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("execute", "request") or (self.kind == "request") != (
                isinstance(self.request, SupportRequest)):
            raise ValueError("Decision must be execute or a request for one operation")
        if not isinstance(self.reason, str) or len(self.reason) > 512:
            raise ValueError("Decision reason is too large or invalid")
        if self.action is not None and (self.kind != "execute" or not isinstance(self.action, str)
                                        or len(self.action) > 4096):
            raise ValueError("Only execute decisions may specify a bounded action")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "reason": self.reason, "action": self.action,
                "request": self.request.as_dict() if self.request else None}


@dataclass(frozen=True)
class ControllerResponse:
    decision: SupportDecision
    cost: SupportCost = field(default_factory=SupportCost)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, SupportDecision) or not isinstance(self.cost, SupportCost):
            raise ValueError("Controller response requires a decision and typed cost")
        if not isinstance(self.metadata, dict):
            raise ValueError("Controller metadata must be an object")


@dataclass(frozen=True)
class SupportResult:
    status: Literal["ok", "empty", "error", "invalid_action"]
    information: object = None
    proposed_action: str | None = None
    cost: SupportCost = field(default_factory=SupportCost)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ("ok", "empty", "error", "invalid_action"):
            raise ValueError("Unknown support result status")
        if self.proposed_action is not None and not isinstance(self.proposed_action, str):
            raise ValueError("Proposed action must be text")
        if not isinstance(self.metadata, dict):
            raise ValueError("Support result metadata must be an object")
        if not isinstance(self.cost, SupportCost):
            raise ValueError("Support result requires typed cost")
        try:
            encoded = json.dumps({"information": self.information, "metadata": self.metadata}, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Support result must be JSON data") from exc
        if len(encoded) > 131072:
            raise ValueError("Support result is too large")

    def as_dict(self) -> dict:
        return {"status": self.status, "information": copy.deepcopy(self.information),
                "proposed_action": self.proposed_action, "cost": self.cost.as_dict(),
                "metadata": copy.deepcopy(self.metadata)}


@dataclass(frozen=True)
class SupportContext:
    benchmark: str
    task_id: str
    task_prompt: str
    observation: str
    available_actions: tuple[str, ...]
    history: tuple[tuple[str, str], ...]
    decision: int
    original_action: str | None
    current_action: str | None
    results: tuple[SupportResult, ...] = ()
    remaining_budget: dict = field(default_factory=dict)
    capabilities: tuple[dict, ...] = ()
    remaining_actions: int | None = None
    remaining_decisions: int | None = None
    support_history: tuple[dict, ...] = ()

    def as_dict(self) -> dict:
        return {"benchmark": self.benchmark, "task_id": self.task_id,
                "task_prompt": self.task_prompt, "observation": self.observation,
                "available_actions": list(self.available_actions),
                "history": [{"action": action, "observation": observation}
                            for action, observation in self.history],
                "decision": self.decision, "original_action": self.original_action,
                "current_action": self.current_action,
                "results": [item.as_dict() for item in self.results],
                "remaining_budget": copy.deepcopy(self.remaining_budget),
                "capabilities": copy.deepcopy(list(self.capabilities)),
                "remaining_actions": self.remaining_actions,
                "remaining_decisions": self.remaining_decisions,
                "support_history": copy.deepcopy(list(self.support_history))}


def visible_state_digest(context: SupportContext) -> str:
    """Fingerprint only evidence that an inspection capability can already see."""
    payload = (context.task_prompt, context.observation, context.available_actions,
               context.history)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class SupportController(Protocol):
    controller_id: str

    def reservation(self, context: SupportContext) -> SupportReservation: ...
    def decide(self, context: SupportContext) -> ControllerResponse: ...


class SupportOperation(Protocol):
    operation_id: str

    def validate_arguments(self, arguments: dict, context: SupportContext) -> None: ...
    def reservation(self, arguments: dict, context: SupportContext) -> SupportReservation: ...
    def execute(self, arguments: dict, context: SupportContext) -> SupportResult: ...


class SupportExecutor:
    """Explicit allowlist of operations; never dispatch model text as code."""

    def __init__(self, operations: dict[str, SupportOperation] | None = None):
        self.operations = dict(operations or {})
        if any(name != operation.operation_id for name, operation in self.operations.items()):
            raise ValueError("Operation registry keys must match operation IDs")

    def catalog(self) -> tuple[dict, ...]:
        return tuple({"operation": name,
                      "description": getattr(operation, "description", "Registered support operation"),
                      "arguments": copy.deepcopy(getattr(operation, "argument_schema", {})),
                      "model_calls_per_use": getattr(operation, "model_calls_per_use", 0),
                      "max_output_tokens_per_use": getattr(operation, "max_output_tokens_per_use", 0),
                      "embedding_calls_per_use": getattr(operation, "embedding_calls_per_use", 0),
                      "max_example_tokens_per_use": getattr(operation, "max_example_tokens_per_use", 0)}
                     for name, operation in self.operations.items())

    def prepare(self, request: SupportRequest, context: SupportContext) -> tuple[SupportOperation, SupportReservation]:
        if not isinstance(request, SupportRequest):
            raise ValueError("Malformed support request")
        try:
            operation = self.operations[request.operation]
        except KeyError as exc:
            raise ValueError(f"Unknown support operation: {request.operation}") from exc
        arguments = copy.deepcopy(request.arguments)
        operation.validate_arguments(arguments, copy.deepcopy(context))
        reservation = operation.reservation(arguments, copy.deepcopy(context))
        if not isinstance(reservation, SupportReservation):
            raise ValueError("Operation returned an invalid reservation")
        return operation, reservation

    def execute(self, operation: SupportOperation, request: SupportRequest,
                context: SupportContext) -> SupportResult:
        try:
            result = operation.execute(copy.deepcopy(request.arguments), copy.deepcopy(context))
            if not isinstance(result, SupportResult):
                raise ValueError("Operation returned an invalid result")
            return result
        except Exception as exc:
            return SupportResult("error", metadata={"error": repr(exc)})


@dataclass
class SupportBudget:
    max_operations_per_decision: int
    max_operations_per_task: int
    max_additional_model_calls: int
    max_additional_tokens: int
    max_depth: int
    operations_used: int = 0
    model_calls_used: int = 0
    tokens_used: int = 0
    controller_calls_used: int = 0
    controller_input_tokens: int = 0
    controller_output_tokens: int = 0
    controller_inference_seconds: float = 0.0
    support_model_calls_used: int = 0
    support_input_tokens: int = 0
    support_output_tokens: int = 0
    support_inference_seconds: float = 0.0

    def __post_init__(self) -> None:
        for key in ("max_operations_per_decision", "max_operations_per_task",
                    "max_additional_model_calls", "max_additional_tokens", "max_depth"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.max_depth == 0:
            raise ValueError("max_depth must allow an execute decision")

    @classmethod
    def from_config(cls, config: dict) -> SupportBudget:
        keys = ("max_operations_per_decision", "max_operations_per_task",
                "max_additional_model_calls", "max_additional_tokens", "max_depth")
        if any(key not in config for key in keys):
            raise ValueError("Autonomous support budget is missing a required limit")
        return cls(**{key: config[key] for key in keys})

    def remaining(self, operations_this_decision: int, depth: int) -> dict:
        return {"operations_this_decision": max(0, self.max_operations_per_decision - operations_this_decision),
                "operations_task": max(0, self.max_operations_per_task - self.operations_used),
                "additional_model_calls": max(0, self.max_additional_model_calls - self.model_calls_used),
                "additional_tokens": max(0, self.max_additional_tokens - self.tokens_used),
                "depth": max(0, self.max_depth - depth)}

    def block_reason(self, reservation: SupportReservation) -> str | None:
        if self.model_calls_used + reservation.model_calls > self.max_additional_model_calls:
            return "model_call_budget"
        if self.tokens_used + reservation.tokens > self.max_additional_tokens:
            return "token_budget"
        return None

    def charge(self, role: Literal["controller", "support"], cost: SupportCost,
               reservation: SupportReservation) -> None:
        if not reservation.covers(cost):
            raise RuntimeError(f"{role} exceeded its reserved model calls or tokens")
        if reason := self.block_reason(SupportReservation(cost.model_calls, cost.input_tokens,
                                                          cost.output_tokens)):
            raise RuntimeError(f"{role} exceeded {reason}")
        self.model_calls_used += cost.model_calls
        self.tokens_used += cost.input_tokens + cost.output_tokens
        if role == "controller":
            self.controller_calls_used += cost.model_calls
            self.controller_input_tokens += cost.input_tokens
            self.controller_output_tokens += cost.output_tokens
            self.controller_inference_seconds += cost.inference_seconds
        else:
            self.support_model_calls_used += cost.model_calls
            self.support_input_tokens += cost.input_tokens
            self.support_output_tokens += cost.output_tokens
            self.support_inference_seconds += cost.inference_seconds

    def as_dict(self) -> dict:
        return {"limits": {key: getattr(self, key) for key in
                           ("max_operations_per_decision", "max_operations_per_task",
                            "max_additional_model_calls", "max_additional_tokens", "max_depth")},
                "used": {"operations": self.operations_used, "additional_model_calls": self.model_calls_used,
                         "additional_tokens": self.tokens_used,
                         "controller": SupportCost(self.controller_calls_used, self.controller_input_tokens,
                                                   self.controller_output_tokens,
                                                   self.controller_inference_seconds).as_dict(),
                         "support": SupportCost(self.support_model_calls_used, self.support_input_tokens,
                                                self.support_output_tokens,
                                                self.support_inference_seconds).as_dict()}}


@dataclass
class SupportTrace:
    original_action: str | None
    final_action: str | None
    termination_reason: str = ""
    events: list[dict] = field(default_factory=list)
    operation_count: int = 0
    controller_decisions: int = 0
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"original_action": self.original_action, "final_action": self.final_action,
                "termination_reason": self.termination_reason, "events": self.events,
                "operation_count": self.operation_count,
                "controller_decisions": self.controller_decisions,
                "elapsed_seconds": round(self.elapsed_seconds, 3)}


class ExecuteImmediately:
    """Framework smoke controller. No autonomous selection algorithm is implied."""

    controller_id = "execute_immediately_v1"

    def reservation(self, context: SupportContext) -> SupportReservation:
        return SupportReservation()

    def decide(self, context: SupportContext) -> ControllerResponse:
        return ControllerResponse(SupportDecision("execute", reason="no_support_requested"))


CONTROLLERS = {ExecuteImmediately.controller_id: ExecuteImmediately}
OPERATIONS: dict[str, Callable[[dict, object, object], SupportOperation]] = {}


def register_support_operation(operation_id: str, factory: Callable[[dict, object, object], SupportOperation]) -> None:
    if not operation_id or operation_id in OPERATIONS:
        raise ValueError(f"Support operation is empty or already registered: {operation_id}")
    OPERATIONS[operation_id] = factory


def make_support_controller(config: dict, controller_model=None) -> SupportController:
    name = config["autonomous_support"]["controller"]
    if name in {"prompted_gpt_oss_v1", "prompted_gpt_oss_v2", "prompted_gpt_oss_v3"}:
        from . import llm_support_controller  # Register the isolated baseline.
    if name in {"random_single_support_v1", "fixed_reasoning_v1"}:
        from . import autonomous_evaluation_baselines
    if name == "learned_gate_v1":
        from . import learned_support_gate
    try:
        if name == ExecuteImmediately.controller_id:
            return CONTROLLERS[name]()
        return CONTROLLERS[name](config, controller_model)
    except KeyError as exc:
        raise ValueError(f"Unknown autonomous support controller: {name}") from exc


def make_support_executor(config: dict, controller_model=None, memory=None) -> SupportExecutor:
    names = config["autonomous_support"].get("operations", [])
    if any(name in {"inspect_history_v1", "inspect_candidate_v1", "reason_over_artifacts_v1",
                    "retrieve_episodic_memory_v1", "inspect_state_v1"}
           for name in names):
        from . import autonomous_operations  # Register built-in safe capabilities.
    operations = {}
    for name in names:
        try:
            operations[name] = OPERATIONS[name](config, controller_model, memory)
        except KeyError as exc:
            raise ValueError(f"Unknown configured support operation: {name}") from exc
    return SupportExecutor(operations)


def run_support_loop(controller: SupportController, executor: SupportExecutor,
                     budget: SupportBudget, context: SupportContext,
                     validate_action: Callable[[str], str | None]) -> SupportTrace:
    """One decision's support loop. The caller alone can step the environment."""
    started = time.perf_counter()
    trace = SupportTrace(context.original_action, context.original_action)
    operations_this_decision = depth = 0
    current = replace(context, remaining_budget=budget.remaining(0, 0))
    try:
        while depth < budget.max_depth:
            # A decision to execute is still possible when operation limits are zero.
            try:
                reservation = controller.reservation(copy.deepcopy(current))
                if not isinstance(reservation, SupportReservation):
                    raise ValueError("Controller returned an invalid reservation")
            except Exception as exc:
                trace.events.append({"state": current.as_dict(), "controller_id": controller.controller_id,
                                     "error": repr(exc)})
                trace.termination_reason = "controller_error"
                break
            if reason := budget.block_reason(reservation):
                trace.termination_reason = reason
                break
            event = {"state": current.as_dict(), "original_action": context.original_action,
                     "controller_id": controller.controller_id}
            depth += 1
            controller_started = time.perf_counter()
            try:
                response = controller.decide(copy.deepcopy(current))
                if not isinstance(response, ControllerResponse) or not isinstance(
                        response.decision, SupportDecision):
                    raise ValueError("Controller returned a malformed response")
                budget.charge("controller", response.cost, reservation)
                event["controller_cost"] = response.cost.as_dict()
                event["controller_decision"] = response.decision.as_dict()
                if response.metadata:
                    event["controller_metadata"] = copy.deepcopy(response.metadata)
            except Exception as exc:
                event["error"] = repr(exc)
                event["controller_elapsed_seconds"] = round(time.perf_counter() - controller_started, 3)
                trace.events.append(event)
                trace.termination_reason = "controller_error"
                break
            event["controller_elapsed_seconds"] = round(time.perf_counter() - controller_started, 3)
            trace.controller_decisions += 1
            if response.decision.kind == "execute":
                if response.decision.action is not None:
                    parsed = validate_action(response.decision.action)
                    if parsed is None:
                        event["error"] = "execute_action_parse_error"
                        trace.final_action = context.original_action
                        trace.termination_reason = "invalid_execute_action"
                        trace.events.append(event)
                        break
                    event["action_revision"] = {"from": current.current_action, "to": parsed}
                    trace.final_action = parsed
                trace.events.append(event)
                trace.termination_reason = ("redundant_empty_request"
                                            if response.metadata.get("blocked_repeated_empty_request") else
                                            "controller_backend_error" if response.metadata.get("backend_error")
                                            else "malformed_controller_output"
                                            if response.metadata.get("parse_error") else "execute")
                break
            request = response.decision.request
            event["request"] = request.as_dict()
            if (operations_this_decision >= budget.max_operations_per_decision or
                    budget.operations_used >= budget.max_operations_per_task):
                event["blocked"] = "operation_budget"
                trace.events.append(event)
                trace.termination_reason = "operation_budget"
                break
            try:
                operation, operation_reservation = executor.prepare(request, current)
            except Exception as exc:
                event["error"] = repr(exc)
                trace.events.append(event)
                trace.termination_reason = "invalid_request"
                break
            if reason := budget.block_reason(operation_reservation):
                event["blocked"] = reason
                trace.events.append(event)
                trace.termination_reason = reason
                break
            operation_started = time.perf_counter()
            result = executor.execute(operation, request, current)
            event["operation_elapsed_seconds"] = round(time.perf_counter() - operation_started, 3)
            try:
                budget.charge("support", result.cost, operation_reservation)
            except RuntimeError as exc:
                event["error"] = repr(exc)
                trace.events.append(event)
                trace.termination_reason = "accounting_error"
                break
            operations_this_decision += 1
            budget.operations_used += 1
            trace.operation_count += 1
            if result.proposed_action is not None and result.status == "ok":
                parsed = validate_action(result.proposed_action)
                if parsed is None:
                    result = replace(result, status="invalid_action",
                                     metadata={**result.metadata, "action_parse_error": True})
                else:
                    trace.final_action = parsed
                    event["action_revision"] = {"from": current.current_action, "to": parsed}
            event["result"] = result.as_dict()
            trace.events.append(event)
            history_entry = {"decision": context.decision, "operation": request.operation,
                             "arguments": copy.deepcopy(request.arguments),
                             "status": result.status,
                             "visible_state_sha256": visible_state_digest(current)}
            current = replace(current, current_action=trace.final_action,
                              results=current.results + (copy.deepcopy(result),),
                              support_history=(current.support_history + (history_entry,))[-16:],
                              remaining_budget=budget.remaining(operations_this_decision, depth))
            if result.status in ("error", "invalid_action"):
                trace.termination_reason = "support_failure"
                break
        else:
            trace.termination_reason = "depth_budget"
        if trace.termination_reason != "execute" and trace.termination_reason in {
                "controller_error", "invalid_request", "support_failure", "accounting_error",
                "malformed_controller_output", "controller_backend_error"}:
            trace.final_action = context.original_action
        return trace
    finally:
        trace.elapsed_seconds = time.perf_counter() - started
