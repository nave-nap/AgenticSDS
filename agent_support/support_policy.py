"""Replaceable per-turn selector for support mechanisms.

The selector decides *which* support calls to make. Retrieval and model
reconsideration remain in the runner, so a new selector can be evaluated
without changing the support implementations or the base agent prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


Phase = Literal["before_proposal", "after_proposal"]
TASK_TRAJECTORY = "task_trajectory_examples"
ACTION_TRAJECTORY = "action_trajectory_reconsideration"
LEGACY_ACTION_MEMORY = "legacy_action_memory_reconsideration"
KNOWN_SUPPORTS = {TASK_TRAJECTORY, ACTION_TRAJECTORY, LEGACY_ACTION_MEMORY}


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


class SupportPolicy(Protocol):
    policy_id: str

    def select(self, context: TurnContext) -> tuple[str, ...]:
        """Return support IDs to execute, in order, at this phase of the turn."""


class FixedConditionPolicy:
    """Reproduce the frozen condition-specific support schedule."""

    policy_id = "fixed_by_condition_v1"

    def select(self, context: TurnContext) -> tuple[str, ...]:
        if context.phase == "before_proposal":
            return (TASK_TRAJECTORY,) if context.condition == "expel_retrieval_only_v1" else ()
        if context.phase == "after_proposal" and context.proposed_action is not None:
            if context.condition == "m2_trajectory_v1":
                return (ACTION_TRAJECTORY,)
            if context.condition == "m2_always":
                return (LEGACY_ACTION_MEMORY,)
        return ()


POLICY_REGISTRY = {FixedConditionPolicy.policy_id: FixedConditionPolicy}


def make_support_policy(config: dict) -> SupportPolicy:
    policy_id = config.get("support_policy", FixedConditionPolicy.policy_id)
    try:
        return POLICY_REGISTRY[policy_id]()
    except KeyError as exc:
        raise ValueError(f"Unknown support policy: {policy_id}") from exc


def selected_supports(policy: SupportPolicy, context: TurnContext) -> tuple[str, ...]:
    choices = policy.select(context)
    if not isinstance(choices, tuple) or any(choice not in KNOWN_SUPPORTS for choice in choices):
        raise ValueError("Support policy returned an unknown support or a non-tuple selection")
    if len(choices) != len(set(choices)):
        raise ValueError("Support policy selected a support more than once in one phase")
    allowed = ({TASK_TRAJECTORY} if context.phase == "before_proposal"
               else {ACTION_TRAJECTORY, LEGACY_ACTION_MEMORY})
    if any(choice not in allowed for choice in choices):
        raise ValueError(f"Support policy selected a support in the wrong phase: {choices}")
    if context.phase == "after_proposal" and context.proposed_action is None and choices:
        raise ValueError("Support policy selected post-proposal support without a valid proposal")
    return choices
