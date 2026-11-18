"""Predeclared, nonadaptive selectors for autonomous-support evaluation."""

from __future__ import annotations

import hashlib
import random

from .autonomous_support import (CONTROLLERS, ControllerResponse, SupportContext,
                                 SupportDecision, SupportRequest, SupportReservation)


class RandomSingleSupportController:
    controller_id = "random_single_support_v1"

    def __init__(self, config: dict, model=None):
        spec = config["autonomous_support"]["random_selection"]
        self.probability = spec["probability"]
        self.seed = spec["seed"]
        self.eligible = tuple(spec["operations"])
        self._rng = None

    def reservation(self, context: SupportContext) -> SupportReservation:
        return SupportReservation()

    def _request(self, name: str, context: SupportContext) -> SupportRequest:
        if name == "retrieve_episodic_memory_v1":
            goal = context.task_prompt.rsplit("Goal: ", 1)[-1].splitlines()[0]
            query = f"Task: {goal}\nObservation: {context.observation[:300]}\nProposed action: {context.current_action or ''}"
            return SupportRequest(name, {"query": query[:1000], "scope": "both", "top_k": 2},
                                  "retrieve a comparable experience")
        if name == "inspect_history_v1":
            return SupportRequest(name, {"scope": "recent", "limit": 4}, "review recent interactions")
        if name == "inspect_state_v1":
            return SupportRequest(name, {"scope": "all", "limit": 8}, "review visible state")
        if name == "reason_over_artifacts_v1":
            return SupportRequest(name, {"objective": "Assess the next environment action from the visible evidence",
                                         "output_type": "candidate_action"}, "reconsider the candidate")
        raise ValueError(f"Random baseline includes unknown operation: {name}")

    def decide(self, context: SupportContext) -> ControllerResponse:
        if context.results:
            return ControllerResponse(SupportDecision("execute", reason="one_support_limit"))
        if self._rng is None:
            material = f"{self.seed}\0{context.task_id}\0random_single_support_v1".encode()
            self._rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
        if self._rng.random() >= self.probability:
            return ControllerResponse(SupportDecision("execute", reason="random_no_support"))
        name = self._rng.choice(self.eligible)
        return ControllerResponse(SupportDecision("request", self._request(name, context),
                                                  reason="random_selection"))


class FixedReasoningController:
    controller_id = "fixed_reasoning_v1"

    def __init__(self, config: dict, model=None):
        self.calls_per_decision = config["autonomous_support"]["fixed_reasoning"]["calls_per_decision"]

    def reservation(self, context: SupportContext) -> SupportReservation:
        return SupportReservation()

    def decide(self, context: SupportContext) -> ControllerResponse:
        if context.original_action is None:
            return ControllerResponse(SupportDecision("execute", reason="invalid_task_proposal"))
        if len(context.results) >= self.calls_per_decision:
            return ControllerResponse(SupportDecision("execute", reason="fixed_reasoning_complete"))
        request = SupportRequest(
            "reason_over_artifacts_v1",
            {"objective": "Reconsider the next environment action using the visible task state",
             "output_type": "candidate_action"},
            "fixed extra reasoning control")
        return ControllerResponse(SupportDecision("request", request, reason="fixed_reasoning_schedule"))


CONTROLLERS[RandomSingleSupportController.controller_id] = RandomSingleSupportController
CONTROLLERS[FixedReasoningController.controller_id] = FixedReasoningController
