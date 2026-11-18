"""Cost-sensitive learned gate around the existing prompted support controller.

The gate learns whether to spend another controller call. The prompted model
still constructs any requested support operation and arguments.
"""

from __future__ import annotations

import math
from pathlib import Path

from .autonomous_support import (CONTROLLERS, ControllerResponse, SupportContext,
                                 SupportDecision, SupportReservation)
from .common import canonical_digest, read_json
from .llm_support_controller import PromptedGptOssController


FEATURE_VERSION = "visible_support_gate_v1"
FEATURE_NAMES = (
    "bias", "decision_fraction", "remaining_action_fraction", "candidate_get",
    "candidate_craft", "candidate_missing", "latest_visible_failure",
    "recent_visible_failures", "candidate_repeat_after_failure",
    "candidate_displayed_recipe", "prior_support_fraction",
    "latest_support_empty_or_error", "remaining_model_calls_fraction",
)
ARTIFACT_DIGEST_FIELDS = ("training_data_sha256", "training_manifest_sha256",
                          "development_manifest_sha256", "heldout_manifest_sha256",
                          "memory_corpus_sha256")
ALLOWED_STATE_KEYS = {"benchmark", "task_prompt", "observation", "available_actions",
                      "history", "decision", "original_action", "current_action",
                      "results", "remaining_budget", "capabilities", "remaining_actions",
                      "remaining_decisions"}


def visible_state(context: SupportContext) -> dict:
    value = context.as_dict()
    value.pop("task_id", None)
    return value


def feature_vector(state: dict | SupportContext) -> list[float]:
    if isinstance(state, SupportContext):
        state = visible_state(state)
    if not isinstance(state, dict) or set(state) - ALLOWED_STATE_KEYS:
        raise ValueError("Learning features must use only allowed visible decision fields")
    if state.get("benchmark") != "textcraft":
        raise ValueError("First learned gate supports TextCraft only")
    history = state.get("history", [])
    results = state.get("results", [])
    if (not isinstance(history, list) or not isinstance(results, list) or
            any(not isinstance(item, dict) for item in history + results) or
            not isinstance(state.get("task_prompt"), str) or
            not isinstance(state.get("observation"), str) or
            type(state.get("decision")) is not int):
        raise ValueError("Malformed visible decision history or support results")
    action = state.get("current_action") or ""
    if not isinstance(action, str):
        raise ValueError("Malformed candidate action")
    def failed(feedback):
        return isinstance(feedback, str) and feedback.startswith(("Could not", "Wrong item"))
    recent = history[-4:]
    failures = sum(failed(item.get("observation")) for item in recent)
    latest = history[-1] if history else {}
    latest_result = results[-1] if results else {}
    budget = state.get("remaining_budget") or {}
    if (not isinstance(budget, dict) or
            any(type(state.get(key)) not in (int, type(None)) for key in
                ("remaining_actions", "remaining_decisions")) or
            any(type(budget.get(key, 0)) is not int for key in
                ("operations_this_decision", "operations_task", "additional_model_calls"))):
        raise ValueError("Malformed visible support budget")
    recipe_match = action.startswith("craft ") and action in state.get("task_prompt", "").splitlines()
    values = (
        1.0,
        min(max(state.get("decision", 1), 1), 30) / 30,
        min(max(state.get("remaining_actions") or 0, 0), 30) / 30,
        float(action.startswith("get ")),
        float(action.startswith("craft ")),
        float(not action),
        float(failed(state.get("observation"))),
        failures / 4,
        float(bool(action and latest.get("action") == action and
                   failed(latest.get("observation")))),
        float(recipe_match),
        min(len(results), 4) / 4,
        float(latest_result.get("status") in {"empty", "error", "invalid_action"}),
        min(max(budget.get("additional_model_calls", 0), 0), 30) / 30,
    )
    return list(values)


def predict_effect(artifact: dict, state: dict | SupportContext) -> float:
    if artifact.get("feature_version") != FEATURE_VERSION or artifact.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Learned gate feature schema differs")
    weights = artifact.get("weights")
    if (not isinstance(weights, list) or len(weights) != len(FEATURE_NAMES) or
            any(type(weight) not in (int, float) or not math.isfinite(weight)
                for weight in weights)):
        raise ValueError("Learned gate weights are malformed")
    return sum(float(weight) * value for weight, value in zip(weights, feature_vector(state)))


def validate_artifact(artifact: dict) -> None:
    if (artifact.get("schema_version") != 1 or
            artifact.get("policy_type") != "cost_sensitive_ridge_gate" or
            artifact.get("feature_version") != FEATURE_VERSION or
            artifact.get("feature_names") != list(FEATURE_NAMES) or
            type(artifact.get("training_tasks")) is not int or artifact["training_tasks"] < 10 or
            type(artifact.get("training_pairs")) is not int or
            artifact["training_pairs"] < artifact["training_tasks"] or
            any(not isinstance(artifact.get(key), str) or len(artifact[key]) != 64 or
                any(char not in "0123456789abcdef" for char in artifact[key])
                for key in ARTIFACT_DIGEST_FIELDS) or
            len({artifact[key] for key in ARTIFACT_DIGEST_FIELDS[1:4]}) != 3):
        raise ValueError("Learned gate artifact lacks valid training provenance")
    weights = artifact.get("weights")
    if (not isinstance(weights, list) or len(weights) != len(FEATURE_NAMES) or
            any(type(weight) not in (int, float) or not math.isfinite(weight)
                for weight in weights)):
        raise ValueError("Learned gate artifact weights are invalid")


class LearnedGateController:
    controller_id = "learned_gate_v1"

    def __init__(self, config: dict, model):
        spec = config["autonomous_support"]["learned_gate"]
        path = Path(spec["artifact_path"]).expanduser()
        artifact = read_json(path)
        if canonical_digest(artifact) != spec["artifact_sha256"]:
            raise ValueError("Learned gate artifact digest differs from configuration")
        validate_artifact(artifact)
        memory_spec = config["autonomous_support"].get("memory", {})
        if (artifact.get("feature_version") != FEATURE_VERSION or
                artifact.get("feature_names") != list(FEATURE_NAMES) or
                artifact.get("task_agent_model_revision") != config.get("model_revision") or
                artifact.get("controller_model_revision") != config["autonomous_support"]["model"]["model"]["model_revision"] or
                artifact.get("memory_corpus_sha256") != memory_spec.get("corpus_sha256")):
            raise ValueError("Learned gate artifact was trained for a different policy or model setup")
        self.artifact = artifact
        self.minimum_effect = spec["minimum_effect"]
        self.delegate = PromptedGptOssController(config, model)

    def _score(self, context: SupportContext) -> float:
        return predict_effect(self.artifact, context)

    def _delegate(self, context: SupportContext) -> bool:
        budget = context.remaining_budget
        return (self._score(context) > self.minimum_effect and
                budget.get("operations_this_decision", 0) > 0 and
                budget.get("operations_task", 0) > 0 and
                budget.get("additional_model_calls", 0) > 0 and
                budget.get("additional_tokens", 0) > 0)

    def reservation(self, context: SupportContext) -> SupportReservation:
        return self.delegate.reservation(context) if self._delegate(context) else SupportReservation()

    def decide(self, context: SupportContext) -> ControllerResponse:
        score = self._score(context)
        metadata = {"learned_gate": {"feature_version": FEATURE_VERSION,
                                     "predicted_net_effect": round(score, 6),
                                     "minimum_effect": self.minimum_effect,
                                     "delegated": self._delegate(context),
                                     "artifact_sha256": canonical_digest(self.artifact)}}
        if not metadata["learned_gate"]["delegated"]:
            return ControllerResponse(SupportDecision("execute", reason="learned_gate_stop"),
                                      metadata=metadata)
        response = self.delegate.decide(context)
        return ControllerResponse(response.decision, response.cost,
                                  {**response.metadata, **metadata})


CONTROLLERS[LearnedGateController.controller_id] = LearnedGateController
