"""Per-decision support adapters and registration.

Adapters own retrieval, prompt construction, and episode-local state. The
runner only accounts for their results and invokes the shared model/parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .memory import RETRIEVAL_POLICY, support_messages
from .reconsideration import reconsideration_messages
from .support_policy import ACTION_TRAJECTORY, LEGACY_ACTION_MEMORY, RECONSIDERATION, TASK_TRAJECTORY, TurnContext
from .trajectory_memory import POLICY as TRAJECTORY_POLICY, memory_messages


@dataclass
class SupportResult:
    messages: list[dict]
    event: dict | None = None
    calls: int = 0
    retrieval_seconds: float = 0.0
    revision_messages: list[dict] | None = None
    fallback_reason: str | None = None


class SupportAdapter(Protocol):
    support_id: str
    phase: str
    version: str
    requirements: tuple[str, ...]

    def invoke(self, context: TurnContext, messages: list[dict], action: str | None) -> SupportResult: ...


SUPPORT_REGISTRY: dict[str, Callable[[dict, object, dict], SupportAdapter]] = {}


def register_support(support_id: str, factory: Callable[[dict, object, dict], SupportAdapter]) -> None:
    if not support_id or support_id in SUPPORT_REGISTRY:
        raise ValueError(f"Support is empty or already registered: {support_id}")
    SUPPORT_REGISTRY[support_id] = factory


def make_supports(config: dict, model, resources: dict) -> dict[str, SupportAdapter]:
    requested = config.get("supports")
    if requested is None:
        requested = [RECONSIDERATION]
        if resources.get("legacy") is not None:
            requested.append(LEGACY_ACTION_MEMORY)
        if resources.get("trajectory") is not None:
            requested.extend((TASK_TRAJECTORY, ACTION_TRAJECTORY))
    result = {}
    for support_id in requested:
        try:
            result[support_id] = SUPPORT_REGISTRY[support_id](config, model, resources)
        except KeyError as exc:
            raise ValueError(f"Unknown support: {support_id}") from exc
    return result


class NoSupport:
    support_id = "none"
    phase = "after_proposal"
    version = "none_v1"
    requirements = ()

    def invoke(self, context, messages, action):
        return SupportResult(messages)


class ReconsiderationSupport:
    support_id = RECONSIDERATION
    phase = "after_proposal"
    version = "reconsideration_only_v1"
    requirements = ("valid_proposal",)

    def invoke(self, context, messages, action):
        event = {"mechanism": "reconsideration_only_v1", "support_id": RECONSIDERATION,
                 "selection_policy": None, "retrieval_seconds": 0.0,
                 "retrieval_event": False, "retrieval_performed": False,
                 "retrieved": [], "proposed_action": action, "action_changed": False}
        return SupportResult(messages, event, calls=1,
                             revision_messages=reconsideration_messages(messages, action, context.benchmark))


class LegacyActionMemorySupport:
    support_id = LEGACY_ACTION_MEMORY
    phase = "after_proposal"
    version = "m2_action_memory_v1"
    requirements = ("memory_corpus", "memory_index", "valid_proposal")

    def __init__(self, memory):
        self.memory = memory

    def invoke(self, context, messages, action):
        hits, retrieval_time = self.memory.retrieve(
            context.benchmark, context.task_prompt, context.observation, action)
        event = {"mechanism": "m2_action_memory", "selection_policy": RETRIEVAL_POLICY,
                 "retrieval_seconds": retrieval_time, "retrieval_event": True,
                 "proposed_action": action, "action_changed": False,
                 "retrieved": [{"memory_id": hit["memory_id"], "similarity": hit["similarity"],
                                "content": {key: hit[key] for key in ("task_prompt", "observation", "action",
                                                                         "feedback", "failed_action", "episode_success")}}
                               for hit in hits]}
        return SupportResult(messages, event, calls=1, retrieval_seconds=retrieval_time,
                             revision_messages=support_messages(messages, action, hits, context.benchmark) if hits else None,
                             fallback_reason="no_relevant_memory")


class TaskTrajectorySupport:
    support_id = TASK_TRAJECTORY
    phase = "before_proposal"
    version = "expel_retrieval_only_adaptation_v1"
    requirements = ("trajectory_corpus", "trajectory_index")

    def __init__(self, memory):
        self.memory = memory
        self.hits = []
        self.retrieved = False

    def invoke(self, context, messages, action):
        retrieval_time = 0.0
        retrieval_event = False
        if not self.retrieved:
            self.hits, retrieval_time = self.memory.retrieve(context.benchmark, context.task_prompt)
            self.retrieved = retrieval_event = True
        event = {"mechanism": "expel_retrieval_only_adaptation", "selection_policy": TRAJECTORY_POLICY,
                 "retrieval_seconds": retrieval_time, "retrieval_event": retrieval_event,
                 "retrieved": self.hits, "action_changed": False}
        if not self.hits:
            event["fallback_reason"] = "no_trajectory_within_budget"
        return SupportResult(memory_messages(messages, self.hits), event,
                             calls=int(retrieval_event), retrieval_seconds=retrieval_time)


class ActionTrajectorySupport:
    support_id = ACTION_TRAJECTORY
    phase = "after_proposal"
    version = "m2_trajectory_memory_v1"
    requirements = ("trajectory_corpus", "trajectory_index", "valid_proposal")

    def __init__(self, memory):
        self.memory = memory

    def invoke(self, context, messages, action):
        hits, retrieval_time = self.memory.retrieve(
            context.benchmark, context.task_prompt, context.observation, action)
        event = {"mechanism": "m2_trajectory_memory", "selection_policy": TRAJECTORY_POLICY,
                 "retrieval_seconds": retrieval_time, "retrieval_event": True,
                 "proposed_action": action, "retrieved": hits, "action_changed": False}
        return SupportResult(messages, event, calls=1, retrieval_seconds=retrieval_time,
                             revision_messages=memory_messages(messages, hits, action) if hits else None,
                             fallback_reason="no_trajectory_within_budget")


register_support("none", lambda config, model, resources: NoSupport())
register_support(RECONSIDERATION, lambda config, model, resources: ReconsiderationSupport())
register_support(LEGACY_ACTION_MEMORY, lambda config, model, resources: LegacyActionMemorySupport(resources["legacy"]))
register_support(TASK_TRAJECTORY, lambda config, model, resources: TaskTrajectorySupport(resources["trajectory"]))
register_support(ACTION_TRAJECTORY, lambda config, model, resources: ActionTrajectorySupport(resources["trajectory"]))
