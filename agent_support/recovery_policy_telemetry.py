"""Normalize v4 traces for policy attribution without changing the frozen runner."""
from __future__ import annotations

import copy
import json

from .recovery_v4 import TERMINAL_WORKER_STATUSES

VERIFICATION_OPERATIONS = {"inspect_candidate_v1", "verify_prerequisites_v1"}
FAILURE_PREFIXES = ("Could not", "Wrong item")


def decision_telemetry(record: dict, policy_id: str) -> dict:
    support_trace = record.get("support_trace", {})
    events = support_trace.get("events", [])
    first = events[0] if events else {}
    state = first.get("state", {})
    history = state.get("history", [])[-8:]
    recovery_before = first.get("recovery_state_before") or {}
    prior_progress = recovery_before.get("last_progress") or {}
    guidance = {}
    messages = first.get("controller_messages", [])
    if messages:
        try:
            guidance = json.loads(messages[-1]["content"]).get("policy_guidance", {})
        except (ValueError, KeyError, TypeError):
            pass
    supports = []
    candidates = []
    if record.get("proposed_action") is not None:
        candidates.append({"source": "qwen", "action": record["proposed_action"]})
    evaluations = []
    interventions = []
    terminal_reviews = []
    controller_errors = []
    progress_judgments = []
    worker_failures = []
    for depth, event in enumerate(events, 1):
        choice = event.get("controller_decision") or {}
        if choice.get("kind") == "execute" and choice.get("action") is not None:
            candidates.append({"source": "controller", "depth": depth,
                               "action": choice["action"]})
        if event.get("policy_intervention"):
            interventions.append(copy.deepcopy(event["policy_intervention"]))
        if event.get("policy_action_selection"):
            interventions.append({"policy_action_selection":
                                  copy.deepcopy(event["policy_action_selection"])})
            selection = event["policy_action_selection"]
            if selection.get("source") == "terminal_review" and selection.get("to"):
                candidates.append({"source": "terminal_review", "depth": depth,
                                   "action": selection["to"]})
        if event.get("terminal_review"):
            terminal_reviews.append(copy.deepcopy(event["terminal_review"]))
        if event.get("progress_assessment"):
            progress_judgments.append(copy.deepcopy(event["progress_assessment"]))
        if ((event.get("error") and not event.get("request")) or
                event.get("parse_error") or event.get("support_action_parse_error")):
            controller_errors.append({"depth": depth, "error": event.get("error"),
                "parse_error": event.get("parse_error"),
                "support_action_parse_error": bool(event.get("support_action_parse_error")),
                "stage": event.get("stage")})
        request = event.get("request")
        if request:
            result = event.get("result")
            supports.append({"depth": depth, "request": copy.deepcopy(request),
                "expected_information": event.get("expected_information"),
                "result": copy.deepcopy(result), "blocked": event.get("blocked"),
                "error": event.get("error"),
                "selected_by_policy": bool(event.get("policy_intervention"))})
            if result:
                candidate = result.get("proposed_action")
                if candidate is not None:
                    candidates.append({"source": request["operation"],
                                       "depth": depth, "action": candidate,
                                       "status": result.get("status")})
                if request.get("operation") in VERIFICATION_OPERATIONS:
                    evaluations.append({"depth": depth, "operation": request["operation"],
                                        "action": request.get("arguments", {}).get("action"),
                                        "status": result.get("status"),
                                        "information": copy.deepcopy(result.get("information"))})
                failure = (result.get("metadata") or {}).get("worker_failure")
                if failure:
                    worker_failures.append(copy.deepcopy(failure))
    fallback = record.get("controller_fallback_reason")
    if fallback:
        controller_errors.append({"fallback_reason": fallback,
                                  "detail": record.get("controller_fallback_detail")})
    recent_failures = [item for item in history if
                       item.get("observation", "").startswith(FAILURE_PREFIXES)]
    infrastructure_failure = (
        record.get("termination_reason") in TERMINAL_WORKER_STATUSES or
        fallback in {"controller_backend_error", "controller_worker_unavailable",
                     "controller_cuda_oom", "controller_error",
                     "controller_generation_failure", "support_error",
                     "accounting_error", "policy_error"} or bool(worker_failures) or
        any((item.get("result") or {}).get("metadata", {}).get("backend_error")
            for item in supports) or
        bool(record.get("qwen_generation", {}).get("error")) or
        record.get("error") == "environment_error")
    return {"decision": record.get("decision"), "policy_id": policy_id,
        "observed_state": {"observation": record.get("input_observation"),
                           "history_tail": copy.deepcopy(history),
                           "recovery_state_before": copy.deepcopy(recovery_before)},
        "failure_signals": {"last_progress": copy.deepcopy(prior_progress),
                            "recent_failed_pairs": copy.deepcopy(recent_failures),
                            "policy_guidance": guidance},
        "supports": supports, "candidate_actions": candidates,
        "candidate_evaluations": evaluations,
        "policy_interventions": interventions,
        "terminal_reviews": terminal_reviews,
        "final_selected_action": record.get("action"),
        "environment_outcome": copy.deepcopy(record.get("outcome")),
        "progress_evidence": copy.deepcopy(record.get("progress_evidence")),
        "progress_judgments": progress_judgments,
        "controller_errors": controller_errors,
        "controller_fallback_reason": fallback,
        "fatal_infrastructure_failure": (record.get("termination_reason")
                                         in TERMINAL_WORKER_STATUSES),
        "infrastructure_failure_signal": bool(infrastructure_failure),
        "worker_failures": worker_failures}


def episode_telemetry(trace: list[dict], policy_id: str) -> list[dict]:
    return [decision_telemetry(record, policy_id) for record in trace]


def annotate_episode(result: dict, policy_id: str, commit: str,
                     protocol: str) -> dict:
    result = copy.deepcopy(result)
    result["support_policy_id"] = policy_id
    result["implementation_commit"] = commit
    result["experiment_protocol"] = protocol
    result["decision_telemetry"] = episode_telemetry(result["trace"], policy_id)
    result["infrastructure_invalid"] = (result.get("status") in TERMINAL_WORKER_STATUSES or
        result.get("status") == "environment_error" or any(
            item["infrastructure_failure_signal"] for item in result["decision_telemetry"]))
    return result
