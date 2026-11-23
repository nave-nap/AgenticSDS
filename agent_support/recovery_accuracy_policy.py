"""Opt-in, replaceable accuracy-first selector for the v4 recovery loop.

The selector only examines visible task evidence and completed support results.
The v4 loop remains responsible for validation, dispatch, accounting, and limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .autonomous_support import SupportContext, SupportRequest
from .policy import parse_textcraft


@dataclass(frozen=True)
class PolicyDirective:
    request: SupportRequest
    expected_information: str
    reason: str


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
