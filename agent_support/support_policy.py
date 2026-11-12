"""Replaceable per-turn selector for support mechanisms.

The selector decides *which* support calls to make. Retrieval and model
reconsideration remain in the runner, so a new selector can be evaluated
without changing the support implementations or the base agent prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import random
from typing import Literal, Protocol


Phase = Literal["before_proposal", "after_proposal"]
TASK_TRAJECTORY = "task_trajectory_examples"
ACTION_TRAJECTORY = "action_trajectory_reconsideration"
LEGACY_ACTION_MEMORY = "legacy_action_memory_reconsideration"
RECONSIDERATION = "proposal_reconsideration_without_retrieval"
KNOWN_SUPPORTS = {TASK_TRAJECTORY, ACTION_TRAJECTORY, LEGACY_ACTION_MEMORY, RECONSIDERATION}


@dataclass(frozen=True)
class TurnContext:
    phase: Phase
    benchmark: str
    condition: str
    decision: int
    task_prompt: str
    observation: str
    available_actions: tuple[str, ...]
    history: tuple[tuple[str, str], ...]
    actions_taken: int
    remaining_actions: int
    remaining_decisions: int
    support_calls_so_far: int
    proposed_action: str | None = None
    proposal_output: str | None = None
    available_supports: tuple[str, ...] = ()
    remaining_model_calls: int | None = None
    remaining_support_calls: int | None = None
    remaining_support_tokens: int | None = None
    recent_failures: tuple[bool, ...] = ()


@dataclass(frozen=True)
class SelectionDecision:
    support_ids: tuple[str, ...] = ()
    reason: str = ""
    trigger: str | None = None
    budget: dict = field(default_factory=dict)
    cooldown_remaining: int = 0

    def as_dict(self) -> dict:
        return {"invoke": bool(self.support_ids), "support_ids": list(self.support_ids),
                "reason": self.reason, "trigger": self.trigger,
                "budget": self.budget, "cooldown_remaining": self.cooldown_remaining}


class SupportPolicy(Protocol):
    policy_id: str

    def select(self, context: TurnContext) -> tuple[str, ...] | SelectionDecision:
        """Return support IDs to execute, in order, at this phase of the turn."""


class FixedConditionPolicy:
    """Reproduce the frozen condition-specific support schedule."""

    policy_id = "fixed_by_condition_v1"

    def __init__(self, spec: dict | None = None, task_id: str = ""):
        pass

    def select(self, context: TurnContext) -> tuple[str, ...]:
        if context.phase == "before_proposal":
            return (TASK_TRAJECTORY,) if context.condition == "expel_retrieval_only_v1" else ()
        if context.phase == "after_proposal" and context.proposed_action is not None:
            if context.condition == "m2_trajectory_v1":
                return (ACTION_TRAJECTORY,)
            if context.condition == "m2_always":
                return (LEGACY_ACTION_MEMORY,)
            if context.condition == "reconsideration_only_v1":
                return (RECONSIDERATION,)
        return ()


POLICY_REGISTRY = {FixedConditionPolicy.policy_id: FixedConditionPolicy}


def register_policy(policy_id: str, factory) -> None:
    if not policy_id or policy_id in POLICY_REGISTRY:
        raise ValueError(f"Policy is empty or already registered: {policy_id}")
    POLICY_REGISTRY[policy_id] = factory


class NeverPolicy:
    policy_id = "never_v1"

    def __init__(self, spec: dict | None = None, task_id: str = ""):
        pass

    def select(self, context: TurnContext) -> SelectionDecision:
        return SelectionDecision(reason="never")


class AlwaysPolicy:
    policy_id = "always_v1"

    def __init__(self, spec: dict, task_id: str = ""):
        self.support_id = spec["support_id"]
        self.phase = spec.get("phase", "after_proposal")

    def select(self, context: TurnContext) -> SelectionDecision:
        if context.phase != self.phase or (self.phase == "after_proposal" and context.proposed_action is None):
            return SelectionDecision(reason="ineligible_phase_or_proposal")
        return SelectionDecision((self.support_id,), "always", "configured_always")


class RandomPolicy:
    policy_id = "random_v1"

    def __init__(self, spec: dict, task_id: str = ""):
        self.support_id = spec["support_id"]
        self.phase = spec.get("phase", "after_proposal")
        self.probability = spec["probability"]
        seed = f"{spec['seed']}\0{task_id}\0{self.policy_id}".encode()
        self.rng = random.Random(int.from_bytes(hashlib.sha256(seed).digest(), "big"))

    def select(self, context: TurnContext) -> SelectionDecision:
        if context.phase != self.phase or (self.phase == "after_proposal" and context.proposed_action is None):
            return SelectionDecision(reason="ineligible_phase_or_proposal")
        chosen = self.rng.random() < self.probability
        return SelectionDecision((self.support_id,) if chosen else (),
                                 "random_draw_selected" if chosen else "random_draw_skipped",
                                 "seeded_probability")


class FailureTriggeredPolicy:
    policy_id = "failure_triggered_v1"

    def __init__(self, spec: dict, task_id: str = ""):
        self.support_id = spec["support_id"]
        self.phase = spec.get("phase", "after_proposal")
        self.threshold = spec["threshold"]
        self.cooldown = spec["cooldown"]
        self.last_selected_decision = -10**9

    def select(self, context: TurnContext) -> SelectionDecision:
        if context.phase != self.phase or (self.phase == "after_proposal" and context.proposed_action is None):
            return SelectionDecision(reason="ineligible_phase_or_proposal")
        remaining = max(0, self.cooldown - (context.decision - self.last_selected_decision - 1))
        if remaining:
            return SelectionDecision(reason="cooldown", cooldown_remaining=remaining)
        failures = 0
        for failed in reversed(context.recent_failures):
            if not failed:
                break
            failures += 1
        if failures < self.threshold:
            return SelectionDecision(reason="failure_threshold_not_met", trigger="consecutive_failed_actions")
        return SelectionDecision((self.support_id,), "failure_threshold_met", "consecutive_failed_actions")

    def on_executed(self, decision: int) -> None:
        self.last_selected_decision = decision


for _policy in (NeverPolicy, AlwaysPolicy, RandomPolicy, FailureTriggeredPolicy):
    register_policy(_policy.policy_id, _policy)


def make_support_policy(config: dict, task_id: str = "") -> SupportPolicy:
    spec = config.get("selection_policy")
    policy_id = spec["id"] if spec else config.get("support_policy", FixedConditionPolicy.policy_id)
    try:
        factory = POLICY_REGISTRY[policy_id]
        return factory(spec, task_id) if spec else factory()
    except KeyError as exc:
        raise ValueError(f"Unknown support policy: {policy_id}") from exc


def selected_supports(policy: SupportPolicy, context: TurnContext) -> tuple[str, ...]:
    return selection_decision(policy, context).support_ids


def selection_decision(policy: SupportPolicy, context: TurnContext) -> SelectionDecision:
    selection = policy.select(context)
    decision = selection if isinstance(selection, SelectionDecision) else SelectionDecision(selection, "fixed_condition")
    choices = decision.support_ids
    from .supports import SUPPORT_REGISTRY
    if not isinstance(choices, tuple) or any(choice not in SUPPORT_REGISTRY for choice in choices):
        raise ValueError("Support policy returned an unknown support or a non-tuple selection")
    if len(choices) != len(set(choices)):
        raise ValueError("Support policy selected a support more than once in one phase")
    allowed = ({TASK_TRAJECTORY} if context.phase == "before_proposal"
               else {ACTION_TRAJECTORY, LEGACY_ACTION_MEMORY, RECONSIDERATION, "none"})
    allowed = allowed | (set(context.available_supports) - KNOWN_SUPPORTS)
    if any(choice not in allowed for choice in choices):
        raise ValueError(f"Support policy selected a support in the wrong phase: {choices}")
    if context.phase == "after_proposal" and context.proposed_action is None and choices:
        raise ValueError("Support policy selected post-proposal support without a valid proposal")
    if context.available_supports and any(choice not in context.available_supports for choice in choices):
        raise ValueError(f"Support policy selected unavailable support: {choices}")
    if not decision.budget:
        decision = replace(decision, budget={
            "remaining_model_calls": context.remaining_model_calls,
            "remaining_support_calls": context.remaining_support_calls,
            "remaining_support_tokens": context.remaining_support_tokens})
    return decision
