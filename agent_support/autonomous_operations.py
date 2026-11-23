"""Small, explicitly registered support capabilities for the first controller."""

from __future__ import annotations

import json
import time

from .autonomous_support import (SupportContext, SupportCost, SupportReservation,
                                 SupportResult, register_support_operation)
from .common import canonical_digest
from .controller_backend import (ControllerWorkerFatalCUDA, ControllerWorkerOOM,
                                 ControllerWorkerUnavailable,
                                 worker_failure_marker)
from .policy import parse_alfworld, parse_textcraft


class InspectHistory:
    operation_id = "inspect_history_v1"
    description = "Inspect recent, visibly failed, or repeated action/feedback history; no environment action."
    argument_schema = {"limit": "integer from 1 to 8",
                       "scope": "optional: recent (default), failures, or repeated"}
    model_calls_per_use = 0

    def validate_arguments(self, arguments: dict, context: SupportContext) -> None:
        if (set(arguments) not in ({"limit"}, {"limit", "scope"}) or
                type(arguments["limit"]) is not int or not 1 <= arguments["limit"] <= 8 or
                arguments.get("scope", "recent") not in {"recent", "failures", "repeated"}):
            raise ValueError("inspect_history requires limit 1..8 and a valid scope")

    def reservation(self, arguments: dict, context: SupportContext) -> SupportReservation:
        return SupportReservation()

    def execute(self, arguments: dict, context: SupportContext) -> SupportResult:
        scope = arguments.get("scope", "recent")
        counts = {}
        items = []
        for position, (action, feedback) in enumerate(context.history, 1):
            counts[action] = counts.get(action, 0) + 1
            failed = (feedback.startswith(("Could not", "Wrong item"))
                      if context.benchmark == "textcraft" else
                      feedback.strip().lower() == "nothing happens.")
            items.append({"position": position, "action": action, "feedback": feedback,
                          "failed_action": failed, "action_occurrence": counts[action]})
        if scope == "failures":
            items = [item for item in items if item["failed_action"] is True]
        elif scope == "repeated":
            repeated = {action for action, count in counts.items() if count > 1}
            items = [item for item in items if item["action"] in repeated]
        selected = items[-arguments["limit"]:]
        return SupportResult("ok" if selected else "empty",
                             {"scope": scope, "interactions": selected,
                              "count": len(selected), "total_matches": len(items)},
                             metadata={"source": "episode_history",
                                       "failure_basis": "visible_feedback_text"})


class InspectState:
    operation_id = "inspect_state_v1"
    description = ("Inspect only visible TextCraft goal, displayed recipes, observed inventory feedback, "
                   "and crafting feedback; no simulator query or environment action.")
    argument_schema = {"scope": "inventory, recipes, craft_feedback, or all",
                       "limit": "integer from 1 to 20",
                       "query": "optional substring filter for recipes or craft feedback"}
    model_calls_per_use = 0

    def validate_arguments(self, arguments: dict, context: SupportContext) -> None:
        if (context.benchmark != "textcraft" or not {"scope", "limit"} <= set(arguments) or
                set(arguments) - {"scope", "limit", "query"} or
                arguments["scope"] not in {"inventory", "recipes", "craft_feedback", "all"} or
                type(arguments["limit"]) is not int or not 1 <= arguments["limit"] <= 20 or
                not isinstance(arguments.get("query", ""), str) or
                len(arguments.get("query", "")) > 128):
            raise ValueError("inspect_state requires a TextCraft scope, limit 1..20, and optional query")

    def reservation(self, arguments: dict, context: SupportContext) -> SupportReservation:
        return SupportReservation()

    def execute(self, arguments: dict, context: SupportContext) -> SupportResult:
        scope = arguments["scope"]
        limit = arguments["limit"]
        query = arguments.get("query", "").lower()
        lines = context.task_prompt.splitlines()
        goal = next((line for line in reversed(lines) if line.startswith("Goal: ")), None)
        information = {"goal": goal, "current_observation": context.observation}
        if scope in {"inventory", "all"}:
            observed = next((feedback for action, feedback in reversed(context.history)
                             if action == "inventory"), None)
            information["inventory_feedback"] = observed
            information["inventory_observed"] = observed is not None
        if scope in {"recipes", "all"}:
            matching = [line for line in lines if line.startswith("craft ") and query in line.lower()]
            information["displayed_recipes"] = matching[:limit]
            information["matching_recipe_count"] = len(matching)
        if scope in {"craft_feedback", "all"}:
            matching = [{"action": action, "feedback": feedback}
                        for action, feedback in context.history
                        if action.startswith("craft ") and
                        (query in action.lower() or query in feedback.lower())]
            information["craft_feedback"] = matching[-limit:]
            information["matching_feedback_count"] = len(matching)
        useful = (information.get("inventory_observed", False) or
                  bool(information.get("displayed_recipes")) or
                  bool(information.get("craft_feedback")))
        return SupportResult("ok" if useful else "empty", information,
                             metadata={"source": "visible_task_prompt_and_episode_history",
                                       "no_environment_query": True})


class RetrieveEpisodicMemory:
    operation_id = "retrieve_episodic_memory_v1"
    description = ("Search the frozen successful TextCraft trajectory index with your own query; "
                   "return bounded examples with provenance and similarities; no environment action.")
    argument_schema = {"query": "context-specific text, 1..1000 characters",
                       "scope": "task, steps, or both",
                       "top_k": "integer from 1 to configured maximum"}
    model_calls_per_use = 1

    def __init__(self, memory):
        if memory is None:
            raise ValueError("Episodic retrieval requires a validated frozen memory index")
        self.memory = memory
        self.embedding_calls_per_use = 1
        self.max_example_tokens_per_use = memory.max_example_tokens

    def validate_arguments(self, arguments: dict, context: SupportContext) -> None:
        if (context.benchmark != "textcraft" or set(arguments) != {"query", "scope", "top_k"} or
                not isinstance(arguments["query"], str) or
                not 0 < len(arguments["query"]) <= 1000 or
                arguments["scope"] not in {"task", "steps", "both"} or
                type(arguments["top_k"]) is not int or
                not 1 <= arguments["top_k"] <= self.memory.max_top_k):
            raise ValueError("retrieve_episodic_memory requires a bounded query, scope, and top_k")

    def reservation(self, arguments: dict, context: SupportContext) -> SupportReservation:
        return SupportReservation(model_calls=1)

    def execute(self, arguments: dict, context: SupportContext) -> SupportResult:
        started = time.perf_counter()
        try:
            hits, elapsed = self.memory.search(context.benchmark, arguments["query"],
                                               arguments["scope"], arguments["top_k"])
        except (ControllerWorkerOOM, ControllerWorkerFatalCUDA, ControllerWorkerUnavailable) as exc:
            elapsed = time.perf_counter() - started
            return SupportResult("error", cost=SupportCost(model_calls=1, inference_seconds=elapsed),
                                 metadata={"retrieval_seconds": elapsed, "error": repr(exc),
                                           "worker_failure": worker_failure_marker(exc)})
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return SupportResult("error", cost=SupportCost(model_calls=1, inference_seconds=elapsed),
                                 metadata={"retrieval_seconds": elapsed, "error": repr(exc)})
        return SupportResult("ok" if hits else "empty",
                             {"query": arguments["query"], "scope": arguments["scope"],
                              "hits": hits, "count": len(hits)},
                             cost=SupportCost(model_calls=1, inference_seconds=elapsed),
                             metadata={"retrieval_seconds": elapsed, "embedding_calls": 1,
                                       "corpus_sha256": self.memory.corpus_sha256,
                                       "index_sha256": self.memory.index_sha256,
                                       "retrieval_policy": "controller_query_frozen_m2_vectors_v1"})


class InspectCandidate:
    operation_id = "inspect_candidate_v1"
    description = "Check action format, current admissibility or displayed recipe, and repeated use; no environment action."
    argument_schema = {"action": "candidate action string"}
    model_calls_per_use = 0

    def validate_arguments(self, arguments: dict, context: SupportContext) -> None:
        if set(arguments) != {"action"} or not isinstance(arguments["action"], str) or not 0 < len(arguments["action"]) <= 4096:
            raise ValueError("inspect_candidate requires a bounded action string")

    def reservation(self, arguments: dict, context: SupportContext) -> SupportReservation:
        return SupportReservation()

    def execute(self, arguments: dict, context: SupportContext) -> SupportResult:
        raw = arguments["action"]
        if context.benchmark == "alfworld":
            action = parse_alfworld(raw, list(context.available_actions))
            admissible = action is not None
            displayed_recipe = None
        else:
            action = parse_textcraft(raw)
            admissible = action is not None
            displayed_recipe = (action in context.task_prompt.splitlines()
                                if action and action.startswith("craft ") else None)
        return SupportResult("ok", {"parsed_action": action, "format_valid": action is not None,
                              "admissible": admissible, "displayed_recipe": displayed_recipe,
                              "prior_same_action_count": sum(previous == action for previous, _ in context.history)
                              if action else 0},
                             metadata={"scope": "syntax_and_visible_state_only"})


class ReasonOverArtifacts:
    operation_id = "reason_over_artifacts_v1"
    description = "Use the shared controller model to assess supplied evidence or recommend one next action; no environment action."
    argument_schema = {"objective": "short text describing the analysis needed",
                       "output_type": "note, plan, or candidate_action"}
    model_calls_per_use = 1

    def __init__(self, config: dict, model):
        if model is None:
            raise ValueError("reason_over_artifacts requires a counted model backend")
        self.model = model
        self.include_support_history = config["autonomous_support"]["controller"] in {
            "prompted_gpt_oss_v2", "prompted_gpt_oss_v3"}
        self.max_new_tokens = config["autonomous_support"].get("support_max_new_tokens", 256)
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("support_max_new_tokens must be positive")
        if getattr(model, "max_new_tokens", self.max_new_tokens) < self.max_new_tokens:
            raise ValueError("Support output ceiling exceeds the shared model ceiling")
        self.max_output_tokens_per_use = self.max_new_tokens

    def validate_arguments(self, arguments: dict, context: SupportContext) -> None:
        if (set(arguments) != {"objective", "output_type"} or
                not isinstance(arguments["objective"], str) or
                not 0 < len(arguments["objective"]) <= 512 or
                arguments["output_type"] not in {"note", "plan", "candidate_action"}):
            raise ValueError("reason_over_artifacts requires objective and output_type")

    def _messages(self, arguments: dict, context: SupportContext) -> list[dict]:
        state = context.as_dict()
        state.pop("task_id", None)
        if not self.include_support_history:
            state.pop("support_history", None)
        state["history"] = state["history"][-8:]
        state["requested_analysis"] = arguments
        system = ("Analyze the supplied task state and evidence for the stated objective. "
                  "You cannot act in the environment or call tools. Support may be unnecessary. "
                  "Treat previous support information as evidence, not as authority. "
                  "Return exactly one JSON object and no prose. "
                  "Return exactly the requested output_type and content fields. "
                  "For note or plan, content is bounded text. "
                  "For candidate_action, content is one action in the benchmark output format.")
        return [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))}]

    def reservation(self, arguments: dict, context: SupportContext) -> SupportReservation:
        count = self.model.count_messages_tokens(self._messages(arguments, context))
        if type(count) is not int or count < 0:
            raise ValueError("Support model returned an invalid token count")
        self._reserved_input_tokens = count
        return SupportReservation(1, count, self.max_new_tokens)

    def execute(self, arguments: dict, context: SupportContext) -> SupportResult:
        messages = self._messages(arguments, context)
        try:
            generation = self.model.generate(messages, max_new_tokens=self.max_new_tokens)
        except (ControllerWorkerOOM, ControllerWorkerFatalCUDA, ControllerWorkerUnavailable) as exc:
            return SupportResult("error", cost=SupportCost(1, getattr(self, "_reserved_input_tokens", 0), 0),
                                 metadata={"prompt_sha256": canonical_digest(messages),
                                           "backend_error": repr(exc), "usage_estimated": True,
                                           "worker_failure": worker_failure_marker(exc)})
        except Exception as exc:
            return SupportResult("error", cost=SupportCost(1, getattr(self, "_reserved_input_tokens", 0), 0),
                                 metadata={"prompt_sha256": canonical_digest(messages),
                                           "backend_error": repr(exc), "usage_estimated": True})
        try:
            cost = SupportCost(1, generation["input_tokens"], generation["output_tokens"],
                               generation["inference_seconds"])
            raw = generation["text"]
            if not isinstance(raw, str):
                raise ValueError("Generation text is missing")
        except (KeyError, TypeError, ValueError) as exc:
            return SupportResult("error", cost=SupportCost(1, getattr(self, "_reserved_input_tokens", 0), 0),
                                 metadata={"prompt_sha256": canonical_digest(messages),
                                           "backend_error": repr(exc), "usage_estimated": True})
        metadata = {"prompt_sha256": canonical_digest(messages),
                    "generation_status": generation.get("generation_status", "complete"),
                    "raw_output": raw[:4096]}
        if generation.get("final_channel_format") is not None:
            metadata["final_channel_format"] = generation["final_channel_format"]
            metadata["recovered_final_channel"] = generation.get("recovered_final_channel", False)
        try:
            if metadata["generation_status"] != "complete":
                raise ValueError("Incomplete model generation")
            value = json.loads(raw)
            if (not isinstance(value, dict) or set(value) != {"output_type", "content"} or
                    value["output_type"] != arguments["output_type"] or
                    not isinstance(value["content"], str) or not 0 < len(value["content"]) <= 4096):
                raise ValueError("Support model returned an invalid artifact")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            metadata["parse_error"] = str(exc)
            return SupportResult("error", cost=cost, metadata=metadata)
        action = value["content"] if value["output_type"] == "candidate_action" else None
        return SupportResult("ok", {"output_type": value["output_type"], "content": value["content"]},
                             proposed_action=action, cost=cost, metadata=metadata)


register_support_operation(InspectHistory.operation_id, lambda config, model, memory: InspectHistory())
register_support_operation(InspectState.operation_id, lambda config, model, memory: InspectState())
register_support_operation(RetrieveEpisodicMemory.operation_id,
                           lambda config, model, memory: RetrieveEpisodicMemory(memory))
register_support_operation(InspectCandidate.operation_id, lambda config, model, memory: InspectCandidate())
register_support_operation(ReasonOverArtifacts.operation_id,
                           lambda config, model, memory: ReasonOverArtifacts(config, model))
