"""Opt-in, replaceable accuracy-first selector for the v4 recovery loop.

The selector only examines visible task evidence and completed support results.
The v4 loop remains responsible for validation, dispatch, accounting, and limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .autonomous_support import SupportContext, SupportRequest, visible_state_digest
from .policy import parse_textcraft


@dataclass(frozen=True)
class PolicyDirective:
    request: SupportRequest
    expected_information: str
    reason: str


@dataclass(frozen=True)
class TerminalReview:
    """One bounded step in checking a controller's impossibility claim."""

    reason: str
    evidence: dict
    directive: PolicyDirective | None = None
    action: str | None = None


class RecoveryPolicy(Protocol):
    policy_id: str

    def guidance(self, context: SupportContext, state) -> dict: ...
    def select(self, context: SupportContext, state,
               available: set[str]) -> PolicyDirective | None: ...
    def choose_action(self, context: SupportContext,
                      proposed: str | None) -> str | None: ...


class AccuracyFirstPolicy:
    policy_id = "accuracy_first_v1"
    failure_feedback_prefixes = ("Could not", "Wrong item")
    failure_reset_feedback_prefixes = ("Got ", "Crafted ")
    failure_analysis_limit = 8
    memory_scope = "both"
    memory_top_k = 3
    generated_candidates_on_stall = 2

    def _failed(self, feedback: str) -> bool:
        return feedback.startswith(self.failure_feedback_prefixes)

    def _active_history(self, context: SupportContext) -> tuple[tuple[str, str], ...]:
        """A successful get/craft can make a previously failed attempt viable."""
        last_change = max((index for index, (_, feedback) in enumerate(context.history)
                           if feedback.startswith(self.failure_reset_feedback_prefixes)),
                          default=-1)
        return context.history[last_change + 1:]

    def _stalled(self, context: SupportContext, state) -> bool:
        active = self._active_history(context)
        recent = active[-3:]
        repeated_failure = (context.current_action is not None and
            any(action == context.current_action and self._failed(feedback)
                for action, feedback in active))
        return (repeated_failure or
                sum(self._failed(feedback) for _, feedback in recent) >= 2 or
                bool(state.last_progress and (
                    state.last_progress.get("failed_action") or
                    state.last_progress.get("assessment") in
                    {"stalled", "contradicted"})))

    def guidance(self, context: SupportContext, state) -> dict:
        candidates = [result.proposed_action for result in context.results
                      if result.status == "ok" and result.proposed_action is not None]
        failed = [action for action, feedback in self._active_history(context)
                  if self._failed(feedback)]
        return {"policy_id": self.policy_id,
                "stalled": self._stalled(context, state),
                "candidate_actions": candidates,
                "failed_actions": list(dict.fromkeys(failed))[-8:],
                "instruction": ("Compare distinct candidates against visible recipes, inventory, "
                    "and prior feedback. Prefer an evidenced alternative to repeating an action "
                    "that failed without a relevant state change. The environment alone verifies success.")}

    def select(self, context: SupportContext, state, available: set[str]) -> PolicyDirective | None:
        """Choose one useful next support before an intended execute decision."""
        used = [(item["operation"], item["arguments"]) for item in context.support_history
                if item["decision"] == context.decision]
        action = context.current_action
        stalled = self._stalled(context, state)

        def request(operation: str, arguments: dict, reason: str) -> PolicyDirective | None:
            if operation not in available or (operation, arguments) in used:
                return None
            return PolicyDirective(SupportRequest(operation, arguments, reason), reason, reason)

        if stalled:
            selected = request("analyze_failures_v1", {"limit": self.failure_analysis_limit},
                               "Identify repeated failures and their observed feedback")
            if selected:
                return selected
        if action and action.startswith("craft "):
            selected = request("verify_prerequisites_v1", {"action": action},
                               "Check the displayed recipe and fresh observed prerequisites")
            if selected:
                return selected
        if stalled and "retrieve_episodic_memory_v1" in available:
            recent_failure = next((feedback for _, feedback in reversed(context.history)
                                   if self._failed(feedback)), "")
            query = (f"Goal: {state.objective}; candidate: {action or 'none'}; "
                     f"failed feedback: {recent_failure}")[:1000]
            selected = request("retrieve_episodic_memory_v1",
                               {"query": query, "scope": self.memory_scope,
                                "top_k": self.memory_top_k},
                               "Find relevant successful prior steps for the current failure")
            if selected:
                return selected
        if stalled and self.generated_candidates_on_stall and "reason_over_artifacts_v1" in available:
            prior_searches = sum(operation == "reason_over_artifacts_v1" and
                                 arguments.get("output_type") == "candidate_action"
                                 for operation, arguments in used)
            if prior_searches < self.generated_candidates_on_stall:
                first = next((result.proposed_action for result in context.results
                              if result.status == "ok" and result.proposed_action), "none")
                objective = (
                    "Propose one valid next action that repairs a missing prerequisite. "
                    "Use visible evidence and avoid repeating failed actions."
                    if prior_searches == 0 else
                    f"Propose a meaningfully different valid next action or route. "
                    f"Differ from the first candidate ({first[:120]}). Exclude previously "
                    "failed actions unless new evidence changes their prerequisites.")
                selected = request("reason_over_artifacts_v1",
                                   {"objective": objective, "output_type": "candidate_action"},
                                   "Generate a competing action using a different strategy")
                if selected:
                    return selected
        candidates = [result.proposed_action for result in context.results
                      if result.status == "ok" and result.proposed_action is not None]
        for candidate in dict.fromkeys(candidates):
            if candidate.startswith("craft ") and parse_textcraft(candidate) == candidate:
                selected = request("verify_prerequisites_v1", {"action": candidate},
                                   "Check generated craft prerequisites before selection")
                if selected:
                    return selected
        for candidate in dict.fromkeys([*candidates, action] if action else candidates):
            selected = request("inspect_candidate_v1", {"action": candidate},
                               "Check candidate syntax, displayed recipe, and prior use")
            if selected:
                return selected
        return None

    def choose_action(self, context: SupportContext, proposed: str | None) -> str | None:
        """Prefer a checked alternative only when the proposed action already failed."""
        active = self._active_history(context)
        if proposed is None or not any(action == proposed and self._failed(feedback)
                                           for action, feedback in active):
            return proposed
        checked = {item.information.get("parsed_action"): item.information
                   for item in context.results
                   if item.status == "ok" and isinstance(item.information, dict)
                   and "format_valid" in item.information}
        prerequisites = {item.information.get("displayed_recipe"): item.information
                         for item in context.results
                         if item.status == "ok" and isinstance(item.information, dict)
                         and "all_prerequisites_observed_sufficient" in item.information}
        for item in context.results:
            candidate = item.proposed_action
            if (item.status != "ok" or candidate is None or candidate == proposed or
                    parse_textcraft(candidate) != candidate or
                    (candidate == "inventory" and context.history and
                     context.history[-1][0] == "inventory") or
                    any(action == candidate and self._failed(feedback)
                        for action, feedback in active)):
                continue
            evidence = checked.get(candidate, {})
            if not evidence.get("admissible") or (candidate.startswith("craft ") and
                    evidence.get("displayed_recipe") is not True):
                continue
            if (candidate.startswith("craft ") and
                    prerequisites.get(candidate, {}).get(
                        "all_prerequisites_observed_sufficient") is not True):
                continue
            return candidate
        if proposed.startswith("craft ") and (not context.history or
                context.history[-1][0] != "inventory"):
            return "inventory"
        return proposed


class StateAwarePolicy(AccuracyFirstPolicy):
    """Matched selector with state and verification but no candidate search."""

    policy_id = "state_aware_v1"
    generated_candidates_on_stall = 0


class AccuracyFirstTerminalPolicy(AccuracyFirstPolicy):
    """Check ordinary impossibility claims before ending the episode."""

    policy_id = "accuracy_first_terminal_v1"
    terminal_candidate_limit = 2
    terminal_state_limit = 20

    def __init__(self):
        # An action is investigated only for the visible state that produced it.
        self._investigated: dict[str, set[str]] = {}

    def review_terminal(self, context: SupportContext, state,
                        available: set[str], claim_reason: str) -> TerminalReview:
        used = [(item["operation"], item["arguments"])
                for item in context.support_history if item["decision"] == context.decision]
        records = list(zip((item for item in context.support_history
                            if item["decision"] == context.decision), context.results))
        digest = visible_state_digest(context)
        investigated = self._investigated.setdefault(f"{context.task_id}:{digest}", set())
        candidate_results = [result for entry, result in records
                             if entry["operation"] == "reason_over_artifacts_v1"
                             and entry["arguments"].get("output_type") == "candidate_action"]
        candidates = list(dict.fromkeys(result.proposed_action for result in candidate_results
            if result.status == "ok" and result.proposed_action and
            parse_textcraft(result.proposed_action) == result.proposed_action))
        inspected = {result.information.get("parsed_action"): result.information
                     for entry, result in records if entry["operation"] == "inspect_candidate_v1"
                     and result.status == "ok" and isinstance(result.information, dict)}
        prerequisites = {result.information.get("displayed_recipe"): result.information
                         for entry, result in records if entry["operation"] == "verify_prerequisites_v1"
                         and result.status == "ok" and isinstance(result.information, dict)}
        inventory_seen = any(action == "inventory" for action, _ in context.history)
        failed = {action for action, feedback in self._active_history(context)
                  if self._failed(feedback)}
        evidence = {"claim_reason": claim_reason, "visible_state_sha256": digest,
                    "inventory_observed": inventory_seen,
                    "attempted_approaches": list(state.attempted_approaches),
                    "failed_actions": sorted(failed), "candidate_actions": candidates,
                    "investigated_actions": sorted(investigated),
                    "support_operations": [operation for operation, _ in used]}
        if not inventory_seen and not context.history:
            evidence["claim_analysis"] = "No inventory observation or environment attempt supports impossibility"
        elif not failed and not state.attempted_approaches:
            evidence["claim_analysis"] = "No failed approach supports impossibility"
        else:
            evidence["claim_analysis"] = "Compare observed failures with distinct alternatives"

        def request(operation: str, arguments: dict, reason: str) -> TerminalReview | None:
            if operation not in available or (operation, arguments) in used:
                return None
            directive = PolicyDirective(SupportRequest(operation, arguments, reason), reason, reason)
            return TerminalReview(reason, evidence, directive=directive)

        selected = request("inspect_state_v1", {"scope": "all", "limit": self.terminal_state_limit},
                           "Inspect visible goal, recipes, and inventory evidence for the stop claim")
        if selected:
            return selected
        query = (f"Goal: {state.objective}; controller proposed no viable strategy because: "
                 f"{claim_reason}; observation: {context.observation[:300]}")[:1000]
        selected = request("retrieve_episodic_memory_v1",
                           {"query": query, "scope": self.memory_scope, "top_k": self.memory_top_k},
                           "Find prior successful approaches relevant to the stop claim")
        if selected:
            return selected
        if len(candidate_results) < self.terminal_candidate_limit:
            previous = candidates[0] if candidates else "none"
            objective = ("Challenge the proposed no_viable_strategy stop. "
                         f"Alternative {len(candidate_results) + 1}. "
                         f"Goal: {state.objective[:120]}. Claim: {claim_reason[:160]}. "
                         f"Suggest one valid next action using visible evidence. "
                         f"Differ from prior candidate: {previous[:100]}. "
                         "Prefer inventory if it has never been observed. "
                         "Avoid previously failed actions without new evidence.")[:512]
            selected = request("reason_over_artifacts_v1",
                               {"objective": objective, "output_type": "candidate_action"},
                               "Generate a distinct alternative to the terminal claim")
            if selected:
                return selected

        for candidate in candidates:
            if candidate not in inspected and candidate not in investigated:
                selected = request("inspect_candidate_v1", {"action": candidate},
                                   "Check alternative action format and visible admissibility")
                if selected:
                    return selected
        for candidate in candidates:
            if candidate.startswith("craft ") and candidate in inspected and candidate not in prerequisites:
                selected = request("verify_prerequisites_v1", {"action": candidate},
                                   "Check displayed craft and observed ingredient sufficiency")
                if selected:
                    return selected

        for candidate in candidates:
            checked = inspected.get(candidate, {})
            if candidate in investigated or candidate in failed or not checked.get("admissible"):
                continue
            if any(action == candidate and self._failed(feedback) for action, feedback in context.history):
                continue
            if candidate.startswith("craft ") and (checked.get("displayed_recipe") is not True or
                    prerequisites.get(candidate, {}).get(
                        "all_prerequisites_observed_sufficient") is not True):
                continue
            if candidate == "inventory" and inventory_seen:
                continue
            investigated.update(candidates)
            return TerminalReview("A checked alternative remains available", evidence, action=candidate)

        if not inventory_seen and "inventory" not in investigated:
            if "inventory" not in inspected:
                selected = request("inspect_candidate_v1", {"action": "inventory"},
                                   "Check inventory as an evidence-gathering action")
                if selected:
                    return selected
            if inspected.get("inventory", {}).get("admissible"):
                investigated.update(candidates)
                investigated.add("inventory")
                return TerminalReview("Inventory has not been observed", evidence, action="inventory")

        investigated.update(candidates)
        return TerminalReview("No supported alternative remained after bounded review", evidence)
