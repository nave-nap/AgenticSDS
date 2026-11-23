"""Versioned, explicit policy choices for the matched recovery experiment."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

from .common import canonical_digest
from .recovery_accuracy_policy import AccuracyFirstPolicy, RecoveryPolicy, StateAwarePolicy


@dataclass(frozen=True)
class PolicyEntry:
    factory: Callable[[], RecoveryPolicy | None]
    configuration: dict


POLICY_REGISTRY: dict[str, PolicyEntry] = {
    "original_v4": PolicyEntry(lambda: None, {
        "policy_id": "original_v4", "selector": "unmodified_v4_controller",
        "candidate_generation": "controller_discretion"}),
    "state_aware_v1": PolicyEntry(StateAwarePolicy, {
        "policy_id": "state_aware_v1", "selector": "visible_failure_and_support_state_v1",
        "failure_feedback_prefixes": list(StateAwarePolicy.failure_feedback_prefixes),
        "failure_reset_feedback_prefixes": list(StateAwarePolicy.failure_reset_feedback_prefixes),
        "failure_analysis_limit": StateAwarePolicy.failure_analysis_limit,
        "memory_scope": StateAwarePolicy.memory_scope,
        "memory_top_k": StateAwarePolicy.memory_top_k,
        "verify_craft": True, "inspect_candidates": True,
        "generated_candidates_on_stall": StateAwarePolicy.generated_candidates_on_stall,
        "alternative_craft_requires_fresh_sufficient_inventory": True}),
    "accuracy_first_v1": PolicyEntry(AccuracyFirstPolicy, {
        "policy_id": "accuracy_first_v1", "selector": "visible_failure_and_support_state_v1",
        "failure_feedback_prefixes": list(AccuracyFirstPolicy.failure_feedback_prefixes),
        "failure_reset_feedback_prefixes": list(AccuracyFirstPolicy.failure_reset_feedback_prefixes),
        "failure_analysis_limit": AccuracyFirstPolicy.failure_analysis_limit,
        "memory_scope": AccuracyFirstPolicy.memory_scope,
        "memory_top_k": AccuracyFirstPolicy.memory_top_k,
        "verify_craft": True, "inspect_candidates": True,
        "generated_candidates_on_stall": AccuracyFirstPolicy.generated_candidates_on_stall,
        "candidate_objectives": ["repair_prerequisite", "different_route_excluding_failed_action"],
        "alternative_craft_requires_fresh_sufficient_inventory": True}),
}
POLICIES = tuple(POLICY_REGISTRY)


def policy_configuration(policy_id: str) -> dict:
    try:
        value = POLICY_REGISTRY[policy_id].configuration
    except KeyError as exc:
        raise ValueError(f"Unknown recovery policy: {policy_id}") from exc
    return copy.deepcopy(value)


def policy_digest(policy_id: str) -> str:
    return canonical_digest(policy_configuration(policy_id))


def make_policy(policy_id: str) -> RecoveryPolicy | None:
    if policy_id not in POLICY_REGISTRY:
        raise ValueError(f"Unknown recovery policy: {policy_id}")
    policy = POLICY_REGISTRY[policy_id].factory()
    if policy is not None and policy.policy_id != policy_id:
        raise ValueError("Policy factory returned a mismatched policy identifier")
    return policy
