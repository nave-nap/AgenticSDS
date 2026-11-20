"""Adaptive within-task TextCraft recovery, isolated from the frozen v3 runner."""
from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field, replace

from .autonomous_support import (ControllerResponse, SupportBudget, SupportContext,
                                 SupportCost, SupportDecision, SupportExecutor,
                                 SupportRequest, SupportReservation, SupportResult,
                                 SupportTrace, make_support_controller,
                                 make_support_executor, run_support_loop,
                                 visible_state_digest)
from .benchmarks import make_episode
from .policy import parse_textcraft, textcraft_messages

ARMS = ("expanded_v3_v1", "autonomous_recovery_v4")
REPAIR_VERSION = "autonomous_recovery_v4_protocol_v2"
STATE_FIELDS = ("subgoal", "strategy", "plan", "hypotheses", "uncertainties",
                "attempted_approaches", "expected_outcome")
ASSESSMENTS = {"progressed", "stalled", "contradicted", "uncertain", "not_applicable"}
MAX_TEXT = 400
MAX_LIST = 8
SYSTEM = """You are the adaptive TextCraft recovery controller. Decide autonomously whether to act, revise Qwen's action, investigate with a listed support, update your strategy, or stop. No support is mandatory. Use only visible task evidence and listed supports; no simulator or hidden state access. The environment alone verifies success. Treat your hypotheses, plans, and predictions as interpretations, not verified facts. Return exactly one JSON object in the final channel, no markdown or extra keys.
Required key: kind. Include reason, state_update, progress_assessment, progress_evidence, and support_assessment when you have evidence to report; omitted fields keep their current state.
kind is execute, request, reflect, or stop.
For execute: include action (null keeps the current candidate; otherwise one native environment command beginning with get or craft, or inventory). Include expected_outcome when you can predict an observable result.
For request: include operation (an exact registered support name from the capabilities catalog) and arguments (matching that support's schema). Request is only for supports; get, craft, and inventory are environment actions, never support operations. Include uncertainty and expected_information to explain the investigation.
For reflect: include focus (brief reason for further internal deliberation). This consumes another controller call.
For stop: include stop_reason (no_viable_strategy or budget_exhausted). Never claim task success.
state_update may contain subgoal, strategy, plan, hypotheses, uncertainties, attempted_approaches, expected_outcome. Text fields must be concise; lists contain short strings. Empty object keeps state. When observed feedback changes your intended approach, update strategy or plan in state_update; do not repeat unchanged text. Compare a pending predicted_outcome with observed_feedback and, when visible evidence supports a judgment, set progress_assessment (progressed, stalled, contradicted, uncertain, or not_applicable) and a brief progress_evidence string. Otherwise use null or omit it. After a support result, assess whether it was useful and resolved the uncertainty using {"useful":true|false,"resolved":true|false,"reason":"brief evidence"} when the result permits a judgment; otherwise use null or omit support_assessment."""


def _unique_json(raw: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("Controller response must be an object")
    return value


def _short(value, limit=MAX_TEXT) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit


def _state_update(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("state_update must be an object")
    if extra := set(value) - set(STATE_FIELDS):
        raise ValueError(f"state_update has unknown fields: {sorted(extra)}")
    for key, item in value.items():
        if key in {"hypotheses", "uncertainties", "attempted_approaches"}:
            if not isinstance(item, list) or len(item) > MAX_LIST or any(not _short(x, 240) for x in item):
                raise ValueError(f"state_update.{key} must be a list of at most {MAX_LIST} short strings")
        elif not _short(item):
            raise ValueError(f"state_update.{key} must be a short nonempty string")
    return value


def parse_decision(raw: str, allowed: set[str]) -> dict:
    value = _unique_json(raw)
    kind = value.get("kind")
    required = {"execute": {"action"}, "request": {"operation", "arguments"},
                "reflect": {"focus"}, "stop": {"stop_reason"}}
    optional = {"execute": {"expected_outcome"},
                "request": {"uncertainty", "expected_information"},
                "reflect": set(), "stop": set()}
    if kind not in required:
        raise ValueError(f"Invalid decision kind: {kind!r}")
    common = {"kind", "reason", "state_update", "progress_assessment",
              "progress_evidence", "support_assessment"}
    if missing := ({"kind"} | required[kind]) - set(value):
        raise ValueError(f"{kind} decision missing fields: {sorted(missing)}")
    if extra := set(value) - common - required[kind] - optional[kind]:
        raise ValueError(f"{kind} decision has unknown fields: {sorted(extra)}")
    if "reason" in value and not _short(value["reason"], 512):
        raise ValueError("reason must be a short nonempty string")
    defaults = {"reason": "", "state_update": {}, "progress_assessment": None,
                "progress_evidence": "", "support_assessment": None}
    if kind == "execute":
        defaults["expected_outcome"] = ""
    elif kind == "request":
        defaults.update(uncertainty="", expected_information="")
    value = {**defaults, **value}
    _state_update(value["state_update"])
    assessment = value["progress_assessment"]
    if assessment is not None and assessment not in ASSESSMENTS:
        raise ValueError(f"Invalid progress assessment: {assessment!r}")
    if value["progress_evidence"] == []:
        value["progress_evidence"] = ""
    if not isinstance(value["progress_evidence"], str) or len(value["progress_evidence"]) > 400:
        raise ValueError("progress_evidence must be a string of at most 400 characters")
    support = value["support_assessment"]
    if support is not None and (not isinstance(support, dict) or
            set(support) != {"useful", "resolved", "reason"} or
            type(support["useful"]) is not bool or type(support["resolved"]) is not bool or
            not _short(support["reason"], 300)):
        raise ValueError("support_assessment requires useful and resolved booleans and a short reason")
    if kind == "execute":
        if value["action"] is not None and (not isinstance(value["action"], str) or
                                              parse_textcraft(value["action"]) != value["action"]):
            raise ValueError(f"Invalid native TextCraft action: {value['action']!r}")
        if not isinstance(value["expected_outcome"], str) or len(value["expected_outcome"]) > MAX_TEXT:
            raise ValueError("expected_outcome must be a string of at most 400 characters")
    elif kind == "request":
        if not isinstance(value["operation"], str) or value["operation"] not in allowed:
            raise ValueError(f"Unknown support operation: {value['operation']!r}")
        if not isinstance(value["arguments"], dict):
            raise ValueError("support arguments must be an object")
        for name in ("uncertainty", "expected_information"):
            if not isinstance(value[name], str) or len(value[name]) > MAX_TEXT:
                raise ValueError(f"{name} must be a string of at most 400 characters")
    elif kind == "reflect":
        if not _short(value["focus"]):
            raise ValueError("reflect focus must be a short nonempty string")
    elif value["stop_reason"] not in {"no_viable_strategy", "budget_exhausted"}:
        raise ValueError(f"Invalid stop reason: {value['stop_reason']!r}")
    return value


@dataclass
class RecoveryState:
    objective: str
    subgoal: str = ""
    strategy: str = ""
    plan: str = ""
    hypotheses: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    attempted_approaches: list[str] = field(default_factory=list)
    expected_outcome: str = ""
    last_progress: dict | None = None
    support_findings: list[dict] = field(default_factory=list)
    version: int = 0

    def snapshot(self) -> dict:
        return {"objective": self.objective, **{key: copy.deepcopy(getattr(self, key)) for key in STATE_FIELDS},
                "last_progress": copy.deepcopy(self.last_progress),
                "support_findings": copy.deepcopy(self.support_findings[-MAX_LIST:]),
                "version": self.version,
                "epistemic_status": "Controller plans and hypotheses are interpretations; only task/history/support outputs are evidence."}

    def apply(self, patch: dict) -> dict:
        clean = _state_update(patch)
        changes = {}
        for key, value in clean.items():
            if getattr(self, key) != value:
                changes[key] = {"before": copy.deepcopy(getattr(self, key)), "after": copy.deepcopy(value)}
                setattr(self, key, copy.deepcopy(value))
        if changes:
            self.version += 1
        return changes

    def add_support(self, request: SupportRequest, result: SupportResult, uncertainty: str,
                    expected: str) -> None:
        self.support_findings.append({"operation": request.operation,
            "uncertainty": uncertainty, "expected_information": expected,
            "status": result.status,
            "result_excerpt": json.dumps(result.information, ensure_ascii=False)[:1800],
            "assessment": None})
        self.support_findings = self.support_findings[-MAX_LIST:]

    def assess_support(self, value: dict | None) -> dict | None:
        if value is None or not self.support_findings:
            return None
        if self.support_findings[-1]["assessment"] is not None:
            return None
        self.support_findings[-1]["assessment"] = copy.deepcopy(value)
        return copy.deepcopy(value)


class AnalyzeFailures:
    operation_id = "analyze_failures_v1"
    description = "Summarize failed and repeated visible action/feedback pairs; no simulator access."
    argument_schema = {"limit": "integer 1..20", "action": "optional exact action filter"}
    return_schema = {"failed_attempts": "list of visible failures", "repeat_counts": "action counts"}
    expected_information = "Evidence of repeated unsuccessful attempts and their feedback"
    model_calls_per_use = 0

    def validate_arguments(self, arguments, context):
        if (not isinstance(arguments, dict) or not {"limit"} <= set(arguments) <= {"limit", "action"} or
                type(arguments["limit"]) is not int or not 1 <= arguments["limit"] <= 20 or
                ("action" in arguments and not _short(arguments["action"], 256))):
            raise ValueError("Invalid failure analysis arguments")

    def reservation(self, arguments, context):
        return SupportReservation()

    def execute(self, arguments, context):
        counts = {}
        failures = []
        for position, (action, feedback) in enumerate(context.history, 1):
            counts[action] = counts.get(action, 0) + 1
            if feedback.startswith(("Could not", "Wrong item")) and (
                    "action" not in arguments or arguments["action"] == action):
                failures.append({"position": position, "action": action,
                                 "feedback": feedback, "attempt": counts[action]})
        selected = failures[-arguments["limit"]:]
        return SupportResult("ok" if selected else "empty",
                             {"failed_attempts": selected, "repeat_counts": {
                                 action: count for action, count in counts.items() if count > 1}},
                             metadata={"source": "visible_episode_history", "no_environment_query": True})


class VerifyPrerequisites:
    operation_id = "verify_prerequisites_v1"
    description = "Locate a displayed recipe and latest observed inventory feedback for a proposed craft."
    argument_schema = {"action": "one native craft action"}
    return_schema = {"displayed_recipe": "exact visible recipe or null",
                     "ingredients": "required quantity and last observed inventory quantity per item",
                     "inventory_current": "whether inventory was the latest environment action",
                     "recent_craft_feedback": "visible feedback for that action"}
    expected_information = "Whether the craft is displayed and whether a fresh observed inventory satisfies its ingredients"
    model_calls_per_use = 0

    def validate_arguments(self, arguments, context):
        if (not isinstance(arguments, dict) or set(arguments) != {"action"} or
                not isinstance(arguments["action"], str) or
                not arguments["action"].startswith("craft ") or
                parse_textcraft(arguments["action"]) != arguments["action"]):
            raise ValueError("Verification requires one native craft action")

    def reservation(self, arguments, context):
        return SupportReservation()

    def execute(self, arguments, context):
        action = arguments["action"]
        recipes = [line for line in context.task_prompt.splitlines() if line.startswith("craft ")]
        displayed = action if action in recipes else None
        inventory_index = next((index for index in range(len(context.history) - 1, -1, -1)
                                if context.history[index][0] == "inventory"), None)
        inventory = context.history[inventory_index][1] if inventory_index is not None else None
        current_inventory = inventory_index == len(context.history) - 1 if inventory_index is not None else False
        observed = ({name: int(count) for name, count in re.findall(r"\[([^\]]+)\] \((\d+)\)", inventory)}
                    if inventory is not None else {})
        requirements = {}
        if displayed is not None:
            for ingredient in action.split(" using ", 1)[1].split(", "):
                amount, item = ingredient.split(" ", 1)
                requirements[item] = requirements.get(item, 0) + int(amount)
        ingredients = [{"item": item, "required": amount,
                        "observed": observed.get(item, 0) if current_inventory else None,
                        "sufficient": (observed.get(item, 0) >= amount if current_inventory else None)}
                       for item, amount in requirements.items()]
        feedback = [observed for previous, observed in context.history if previous == action][-5:]
        return SupportResult("ok" if displayed else "empty",
                             {"displayed_recipe": displayed,
                              "last_inventory_feedback": inventory,
                              "inventory_observed": inventory is not None,
                              "inventory_current": current_inventory,
                              "ingredients": ingredients,
                              "all_prerequisites_observed_sufficient": (
                                  all(item["sufficient"] for item in ingredients)
                                  if current_inventory and displayed is not None else None),
                              "recent_craft_feedback": feedback},
                             metadata={"source": "visible_task_and_history", "no_environment_query": True,
                                       "availability_not_proven": True})


def recovery_executor(config: dict, controller_model, memory=None) -> SupportExecutor:
    names = config["autonomous_support"]["operations"]
    custom = {"analyze_failures_v1": AnalyzeFailures(),
              "verify_prerequisites_v1": VerifyPrerequisites()}
    builtins = [name for name in names if name not in custom]
    copy_config = copy.deepcopy(config)
    copy_config["autonomous_support"]["operations"] = builtins
    base = make_support_executor(copy_config, controller_model, memory)
    return SupportExecutor({**base.operations, **{name: custom[name] for name in names if name in custom}})


def capability_catalog(executor: SupportExecutor) -> tuple[dict, ...]:
    items = []
    for item in executor.catalog():
        operation = executor.operations[item["operation"]]
        items.append({**item,
                      "return_schema": copy.deepcopy(getattr(operation, "return_schema", {
                          "status": "ok, empty, error, or invalid_action",
                          "information": "bounded result described by operation"})),
                      "expected_information": getattr(operation, "expected_information",
                                                      operation.description),
                      "estimated_cost": {"model_calls": item.get("model_calls_per_use", 0),
                                         "max_output_tokens": item.get("max_output_tokens_per_use", 0)}})
    return tuple(items)


def _messages(context: SupportContext, state: RecoveryState, pending: dict | None) -> list[dict]:
    visible = context.as_dict()
    visible.pop("task_id", None)
    visible["history"] = visible["history"][-8:]
    visible["results"] = [{"status": item["status"],
                           "information_excerpt": json.dumps(item["information"], ensure_ascii=False)[:2400],
                           "proposed_action": item["proposed_action"]}
                          for item in visible["results"][-2:]]
    visible["support_history"] = visible["support_history"][-8:]
    visible["recovery_state"] = state.snapshot()
    visible["pending_support_assessment"] = pending
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(visible, ensure_ascii=False, separators=(",", ":"))}]


def adaptive_loop(model, executor: SupportExecutor, budget: SupportBudget,
                  context: SupportContext, state: RecoveryState, *, deadline: float,
                  inference_limit: float, inference_used: float) -> tuple[SupportTrace, str | None]:
    trace = SupportTrace(context.original_action, context.original_action)
    current = replace(context, capabilities=capability_catalog(executor))
    pending = None
    operations_this_decision = 0
    for depth in range(budget.max_depth):
        if time.perf_counter() >= deadline:
            trace.termination_reason = "wall_time_budget"
            return trace, "wall_time_budget"
        if inference_used + budget.controller_inference_seconds + budget.support_inference_seconds >= inference_limit:
            trace.termination_reason = "inference_time_budget"
            return trace, "inference_time_budget"
        current = replace(current, remaining_budget=budget.remaining(operations_this_decision, depth))
        messages = _messages(current, state, pending)
        try:
            input_count = model.count_messages_tokens(messages)
            output_limit = min(model.max_new_tokens,
                               current.remaining_budget["additional_tokens"] - input_count)
            if output_limit < 1:
                trace.termination_reason = "token_budget"
                return trace, "token_budget"
            reservation = SupportReservation(1, input_count, output_limit)
            if reason := budget.block_reason(reservation):
                trace.termination_reason = reason
                return trace, reason
        except Exception as exc:
            trace.events.append({"error": repr(exc), "stage": "reservation"})
            trace.termination_reason = "controller_error"
            return trace, "controller_error"
        event = {"state": current.as_dict(), "recovery_state_before": state.snapshot(),
                 "controller_messages": messages}
        trace.controller_decisions += 1
        try:
            generation = model.generate(messages, max_new_tokens=reservation.output_tokens)
        except Exception as exc:
            # The attempt is visible even when backend usage cannot be recovered.
            budget.charge("controller", SupportCost(1, input_count, 0), reservation)
            event.update(error=repr(exc), usage_estimated=True)
            trace.events.append(event)
            trace.termination_reason = "controller_backend_error"
            return trace, "controller_backend_error"
        try:
            cost = SupportCost(1, generation["input_tokens"], generation["output_tokens"],
                               generation["inference_seconds"])
            budget.charge("controller", cost, reservation)
            event.update(controller_generation=generation, controller_cost=cost.as_dict())
        except Exception as exc:
            event.update(controller_generation=generation, error=repr(exc))
            trace.events.append(event)
            trace.termination_reason = "accounting_error"
            return trace, "accounting_error"
        if generation.get("generation_status") != "complete":
            event["error"] = generation.get("generation_status", "incomplete_generation")
            trace.events.append(event)
            trace.termination_reason = "controller_generation_failure"
            return trace, "controller_generation_failure"
        try:
            choice = parse_decision(generation["text"], set(executor.operations))
        except (ValueError, KeyError, TypeError) as exc:
            event["parse_error"] = str(exc)
            trace.events.append(event)
            trace.termination_reason = "controller_parse_error"
            return trace, "controller_parse_error"
        event["controller_decision"] = choice
        if pending is None and choice["support_assessment"] is not None:
            event["parse_error"] = "support_assessment has no pending support result"
            trace.events.append(event)
            trace.termination_reason = "controller_parse_error"
            return trace, "controller_parse_error"
        if choice["progress_assessment"] is not None and (
                state.last_progress is None or state.last_progress.get("assessment") is not None):
            event["parse_error"] = "progress_assessment has no pending environment outcome"
            trace.events.append(event)
            trace.termination_reason = "controller_parse_error"
            return trace, "controller_parse_error"
        kind = choice["kind"]
        if kind == "request":
            request = SupportRequest(choice["operation"], choice["arguments"], choice["uncertainty"])
            try:
                operation, operation_reservation = executor.prepare(request, current)
            except (ValueError, KeyError, TypeError) as exc:
                event["parse_error"] = f"support arguments for {request.operation}: {exc}"
                trace.events.append(event)
                trace.termination_reason = "controller_parse_error"
                return trace, "controller_parse_error"
        patch = dict(choice["state_update"])
        if kind == "execute":
            prediction = choice["expected_outcome"] or patch.get("expected_outcome", "")
            if prediction:
                patch["expected_outcome"] = prediction
        event["state_changes"] = state.apply(patch)
        if kind == "execute" and not prediction and state.expected_outcome:
            if not event["state_changes"]:
                state.version += 1
            event["state_changes"]["expected_outcome"] = {
                "before": state.expected_outcome, "after": ""}
            state.expected_outcome = ""
        if state.last_progress and state.last_progress.get("assessment") is None and choice["progress_assessment"]:
            state.last_progress["assessment"] = choice["progress_assessment"]
            state.last_progress["assessment_evidence"] = choice["progress_evidence"]
            event["progress_assessment"] = copy.deepcopy(state.last_progress)
        if pending is not None:
            event["support_assessment"] = state.assess_support(choice["support_assessment"])
            pending = None
        if kind == "execute":
            trace.final_action = choice["action"] if choice["action"] is not None else current.current_action
            event["action_revision"] = {"from": current.current_action, "to": trace.final_action}
            event["recovery_state_after"] = state.snapshot()
            trace.events.append(event)
            trace.termination_reason = "execute"
            return trace, None
        if kind == "stop":
            event["recovery_state_after"] = state.snapshot()
            trace.events.append(event)
            trace.final_action = None
            trace.termination_reason = choice["stop_reason"]
            return trace, choice["stop_reason"]
        if kind == "reflect":
            event["recovery_state_after"] = state.snapshot()
            trace.events.append(event)
            continue
        if (operations_this_decision >= budget.max_operations_per_decision or
                budget.operations_used >= budget.max_operations_per_task):
            event["blocked"] = "operation_budget"
            trace.events.append(event)
            trace.termination_reason = "operation_budget"
            return trace, "operation_budget"
        event["request"] = request.as_dict()
        event["expected_information"] = choice["expected_information"]
        try:
            if reason := budget.block_reason(operation_reservation):
                event["blocked"] = reason
                trace.events.append(event)
                trace.termination_reason = reason
                return trace, reason
            result = executor.execute(operation, request, current)
            budget.charge("support", result.cost, operation_reservation)
        except Exception as exc:
            event["error"] = repr(exc)
            trace.events.append(event)
            trace.termination_reason = "support_error"
            return trace, "support_error"
        operations_this_decision += 1
        budget.operations_used += 1
        trace.operation_count += 1
        if result.proposed_action is not None and result.status == "ok":
            candidate = parse_textcraft(result.proposed_action)
            if candidate is not None:
                trace.final_action = candidate
                current = replace(current, current_action=candidate)
                event["action_revision"] = {"from": context.original_action, "to": candidate}
            else:
                event["support_action_parse_error"] = True
        event["result"] = result.as_dict()
        state.add_support(request, result, choice["uncertainty"], choice["expected_information"])
        pending = state.support_findings[-1]
        event["recovery_state_after"] = state.snapshot()
        trace.events.append(event)
        current = replace(current, results=current.results + (result,),
                          support_history=(current.support_history + ({
                              "decision": context.decision, "operation": request.operation,
                              "arguments": request.arguments, "status": result.status,
                              "visible_state_sha256": visible_state_digest(current)},))[-16:])
    trace.termination_reason = "depth_budget"
    return trace, "depth_budget"


def run_recovery_episode(task: dict, config: dict, qwen, gpt, *, memory=None,
                         episode_factory=make_episode, controller=None,
                         executor=None, clock=time.perf_counter, checkpoint=None) -> dict:
    arm = config["arm"]
    if arm not in ARMS:
        raise ValueError("Unknown recovery arm")
    budget = SupportBudget.from_config(config["autonomous_support"]["budget"])
    if executor is None:
        executor = (make_support_executor(config, gpt, memory) if arm == ARMS[0]
                    else recovery_executor(config, gpt, memory))
    if arm == ARMS[0] and controller is None:
        controller = make_support_controller(config, gpt)
    episode = episode_factory("textcraft", task, config)
    started = clock()
    deadline = started + config["limits"]["wall_seconds"]
    state = RecoveryState(objective=task["initial_observation"].rsplit("Goal: ", 1)[-1].strip())
    history, trace, support_history = [], [], []
    status, success = "max_decisions", False
    actions_taken = failed_actions = parser_errors = 0
    qwen_calls = qwen_input = qwen_output = 0
    qwen_seconds = environment_seconds = 0.0

    def append_record(record: dict) -> None:
        trace.append(record)
        if checkpoint is not None:
            checkpoint({"task_id": task["task_id"], "arm": arm,
                        "status": "running", "decisions_recorded": len(trace),
                        "environment_actions": actions_taken,
                        "qwen_model_calls": qwen_calls,
                        "support_budget": budget.as_dict(),
                        "recovery_state": state.snapshot() if arm == ARMS[1] else None,
                        "trace": copy.deepcopy(trace)})
    try:
        episode.reset()
        for decision in range(1, config["max_decisions"] + 1):
            used_inference = qwen_seconds + budget.controller_inference_seconds + budget.support_inference_seconds
            if clock() >= deadline:
                status = "wall_time_budget"
                break
            if used_inference >= config["limits"]["inference_seconds"]:
                status = "inference_time_budget"
                break
            if actions_taken >= config["max_actions"]:
                status = "max_actions"
                break
            if qwen_calls + budget.model_calls_used >= config["limits"]["max_model_calls_total"]:
                status = "model_call_budget"
                break
            messages = textcraft_messages(episode.initial_observation, history,
                                          config["history_window"])
            try:
                generation = qwen.generate(messages)
            except Exception as exc:
                generation = {"text": "", "input_tokens": 0, "output_tokens": 0,
                              "inference_seconds": 0.0, "error": repr(exc), "usage_unavailable": True}
            qwen_calls += 1
            qwen_input += generation["input_tokens"]
            qwen_output += generation["output_tokens"]
            qwen_seconds += generation["inference_seconds"]
            proposal = parse_textcraft(generation["text"])
            if (clock() >= deadline or
                    qwen_seconds + budget.controller_inference_seconds + budget.support_inference_seconds
                    >= config["limits"]["inference_seconds"]):
                status = "wall_time_budget" if clock() >= deadline else "inference_time_budget"
                append_record({"decision": decision, "input_observation": episode.observation,
                              "qwen_messages": messages, "qwen_generation": generation,
                              "proposed_action": proposal, "action": None,
                              "termination_reason": status})
                break
            context = SupportContext("textcraft", task["task_id"], episode.task_prompt,
                episode.observation, (), tuple((item["action"], item["observation"]) for item in history),
                decision, proposal, proposal,
                capabilities=(executor.catalog() if arm == ARMS[0] else capability_catalog(executor)),
                remaining_actions=config["max_actions"] - actions_taken,
                remaining_decisions=config["max_decisions"] - decision + 1,
                support_history=tuple(support_history[-16:]))
            before_budget = budget.as_dict()["used"]
            if arm == ARMS[0]:
                support_trace = (SupportTrace(None, None, termination_reason="invalid_task_proposal")
                                 if proposal is None else
                                 run_support_loop(controller, executor, budget, context, parse_textcraft))
                stop = None
            else:
                loop_started = clock()
                support_trace, stop = adaptive_loop(gpt, executor, budget, context, state,
                    deadline=deadline, inference_limit=config["limits"]["inference_seconds"],
                    inference_used=qwen_seconds)
                support_trace.elapsed_seconds = clock() - loop_started
            support_history.extend({"decision": decision, "operation": event["request"]["operation"],
                "arguments": event["request"]["arguments"],
                "status": event.get("result", {}).get("status", event.get("blocked", "error")),
                "visible_state_sha256": visible_state_digest(context)}
                for event in support_trace.events if "request" in event)
            action = support_trace.final_action
            if action is not None and parse_textcraft(action) != action:
                action = None
            record = {"decision": decision, "input_observation": episode.observation,
                      "qwen_messages": messages, "qwen_generation": generation,
                      "proposed_action": proposal, "action": action,
                      "support_trace": support_trace.as_dict(),
                      "support_budget_before": before_budget,
                      "support_budget_after": budget.as_dict()["used"],
                      "recovery_state": state.snapshot() if arm == ARMS[1] else None}
            if stop and stop not in {"wall_time_budget", "inference_time_budget",
                                     "no_viable_strategy", "budget_exhausted"}:
                record["controller_fallback_reason"] = stop
                record["controller_fallback_detail"] = next((
                    event.get("parse_error") or event.get("error")
                    for event in reversed(support_trace.events)
                    if event.get("parse_error") or event.get("error")), None)
                action = proposal
                record["action"] = action
                stop = None
            if stop:
                status = stop
                append_record(record)
                break
            used_inference = qwen_seconds + budget.controller_inference_seconds + budget.support_inference_seconds
            if clock() >= deadline or used_inference >= config["limits"]["inference_seconds"]:
                status = "wall_time_budget" if clock() >= deadline else "inference_time_budget"
                append_record(record)
                break
            if action is None:
                parser_errors += 1
                record.update(error="parse_error", output_observation="Invalid action format. Reply with exactly one valid command.")
                append_record(record)
                history.append({"action": "<PARSE_ERROR>", "observation": record["output_observation"]})
                continue
            try:
                environment_started = clock()
                step = episode.step(action)
                elapsed = round(clock() - environment_started, 3)
            except Exception as exc:
                record.update(error="environment_error", details=repr(exc))
                append_record(record)
                status = "environment_error"
                break
            environment_seconds += elapsed
            actions_taken += 1
            failed_actions += int(step.get("failed_action", False))
            record.update(environment_seconds=elapsed, environment_action_number=actions_taken,
                          output_observation=step["observation"], outcome=step)
            append_record(record)
            history.append({"action": action, "observation": step["observation"]})
            if arm == ARMS[1]:
                occurrences = sum(item["action"] == action for item in history)
                state.last_progress = {"action": action,
                    "predicted_outcome": state.expected_outcome,
                    "observed_feedback": step["observation"][:600],
                    "failed_action": bool(step.get("failed_action", False)),
                    "repeated_attempts": occurrences,
                    "observation_changed": step["observation"] != record["input_observation"],
                    "environment_success": bool(step["success"]), "assessment": None}
                record["progress_evidence"] = copy.deepcopy(state.last_progress)
            if step["done"]:
                success = bool(step["success"])
                status = "success" if success else "task_failed"
                break
        else:
            status = "max_actions" if actions_taken >= config["max_actions"] else "max_decisions"
    finally:
        episode.close()
    retrieval_events = [event for record in trace
        for event in record.get("support_trace", {}).get("events", [])
        if event.get("request", {}).get("operation") == "retrieve_episodic_memory_v1"
        and event.get("result")]
    embedding_calls = sum(event["result"]["cost"]["model_calls"] for event in retrieval_events)
    embedding_seconds = sum(event["result"]["cost"]["inference_seconds"] for event in retrieval_events)
    return {"task_id": task["task_id"], "benchmark": "textcraft", "arm": arm,
        **({"recovery_integration_version": REPAIR_VERSION,
            "controller_fallbacks": sum("controller_fallback_reason" in record for record in trace),
            "support_assessments": sum(event.get("support_assessment") is not None
                for record in trace for event in record.get("support_trace", {}).get("events", []))}
           if arm == ARMS[1] else {}),
        "initial_observation": episode.initial_observation, "task_prompt": episode.task_prompt,
        "success": success, "status": status, "termination_reason": status,
        "environment_actions": actions_taken, "failed_actions": failed_actions,
        "model_decisions": len(trace), "parser_errors": parser_errors,
        "model_calls": qwen_calls + budget.model_calls_used,
        "qwen_model_calls": qwen_calls, "qwen_input_tokens": qwen_input,
        "qwen_output_tokens": qwen_output, "qwen_inference_seconds": round(qwen_seconds, 3),
        "gptoss_model_calls": budget.model_calls_used - embedding_calls,
        "embedding_model_calls": embedding_calls,
        "embedding_inference_seconds": round(embedding_seconds, 3),
        "gptoss_inference_seconds": round(budget.controller_inference_seconds +
                                          budget.support_inference_seconds - embedding_seconds, 3),
        "controller_model_calls": budget.controller_calls_used,
        "support_model_calls": budget.support_model_calls_used,
        "input_tokens": qwen_input + budget.controller_input_tokens + budget.support_input_tokens,
        "output_tokens": qwen_output + budget.controller_output_tokens + budget.support_output_tokens,
        "inference_seconds": round(qwen_seconds + budget.controller_inference_seconds + budget.support_inference_seconds, 3),
        "controller_input_tokens": budget.controller_input_tokens,
        "controller_output_tokens": budget.controller_output_tokens,
        "controller_inference_seconds": round(budget.controller_inference_seconds, 3),
        "support_input_tokens": budget.support_input_tokens,
        "support_output_tokens": budget.support_output_tokens,
        "support_inference_seconds": round(budget.support_inference_seconds, 3),
        "support_calls": budget.operations_used, "support_budget": budget.as_dict(),
        "controller_errors": sum(event.get("error") is not None or
            event.get("parse_error") is not None or
            bool(event.get("controller_metadata", {}).get("parse_error"))
            for record in trace for event in record.get("support_trace", {}).get("events", [])),
        "support_useful_judgments": sum(bool(event.get("support_assessment", {}).get("useful"))
            for record in trace for event in record.get("support_trace", {}).get("events", [])
            if event.get("support_assessment") is not None),
        "qwen_generation_failures": sum(bool(record.get("qwen_generation", {}).get("error"))
            for record in trace),
        "environment_seconds": round(environment_seconds, 3),
        "elapsed_seconds": round(clock() - started, 3),
        "strategy_changes": sum(bool(event.get("state_changes", {}).get("strategy"))
            for record in trace for event in record["support_trace"]["events"]),
        "replanning_events": sum(bool(event.get("state_changes", {}).get("plan")) and
            (arm != ARMS[1] or bool(event["state_changes"]["plan"]["before"]))
            for record in trace for event in record["support_trace"]["events"]),
        "progress_assessments": sum(bool(event.get("progress_assessment"))
            for record in trace for event in record["support_trace"]["events"]),
        "recovery_state_final": state.snapshot() if arm == ARMS[1] else None,
        "trace": trace}
