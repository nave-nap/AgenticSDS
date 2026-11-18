"""Opt-in benchmark runner for the sequential support framework."""

from __future__ import annotations

import time

from .autonomous_support import (SupportBudget, SupportContext, SupportController,
                                 SupportExecutor, make_support_controller,
                                 make_support_executor, run_support_loop)
from .benchmarks import make_episode
from .policy import alfworld_messages, parse_alfworld, parse_textcraft, textcraft_messages


def run_autonomous_episode(benchmark: str, task: dict, config: dict, model,
                           episode_factory=make_episode,
                           controller: SupportController | None = None,
                           executor: SupportExecutor | None = None,
                           controller_model=None, memory=None) -> dict:
    """Keep the old runner untouched; only this condition enters the new loop."""
    episode = episode_factory(benchmark, task, config)
    controller = controller if controller is not None else make_support_controller(config, controller_model)
    executor = executor if executor is not None else make_support_executor(config, controller_model, memory)
    budget = SupportBudget.from_config(config["autonomous_support"]["budget"])
    started = time.perf_counter()
    history: list[dict] = []
    trace: list[dict] = []
    actions_taken = parser_errors = failed_actions = 0
    proposal_input_tokens = proposal_output_tokens = 0
    proposal_inference_seconds = 0.0
    environment_seconds = 0.0
    success = False
    status = "max_decisions"

    try:
        episode.reset()
        for decision in range(1, config["max_decisions"] + 1):
            if actions_taken >= config["max_actions"]:
                status = "max_actions"
                break
            if benchmark == "alfworld":
                if not episode.actions:
                    status = "no_actions"
                    break
                available = list(episode.actions)
                messages = alfworld_messages(episode.task_prompt, episode.observation,
                                             available, history, config["history_window"])
                parse = lambda value: parse_alfworld(value, available)
            else:
                available = None
                messages = textcraft_messages(episode.initial_observation, history,
                                              config["history_window"])
                parse = parse_textcraft
            generation = model.generate(messages)
            proposal_input_tokens += generation["input_tokens"]
            proposal_output_tokens += generation["output_tokens"]
            proposal_inference_seconds += generation["inference_seconds"]
            raw = generation["text"]
            proposal = parse(raw)
            context = SupportContext(
                benchmark=benchmark, task_id=task["task_id"], task_prompt=episode.task_prompt,
                observation=episode.observation, available_actions=tuple(available or ()),
                history=tuple((item["action"], item["observation"]) for item in history),
                decision=decision, original_action=proposal, current_action=proposal,
                capabilities=executor.catalog(), remaining_actions=config["max_actions"] - actions_taken,
                remaining_decisions=config["max_decisions"] - decision + 1)
            before_budget = budget.as_dict()["used"]
            support_trace = run_support_loop(controller, executor, budget, context, parse)
            action = support_trace.final_action
            # No capability may step the environment. Recheck the selected action
            # against the same action list and parser immediately before stepping.
            if benchmark == "alfworld":
                if action not in available:
                    action = None
            elif action is not None and parse_textcraft(action) != action:
                action = None
            record = {
                "decision": decision,
                "input_observation": episode.observation,
                "available_actions": available,
                "proposal_raw_model_output": raw,
                "proposed_action": proposal,
                "raw_model_output": raw,
                "action": action,
                "input_tokens": generation["input_tokens"],
                "output_tokens": generation["output_tokens"],
                "inference_seconds": generation["inference_seconds"],
                "support_trace": support_trace.as_dict(),
                "support_budget_before": before_budget,
                "support_budget_after": budget.as_dict()["used"],
            }
            if action is None:
                parser_errors += 1
                record["error"] = "parse_error"
                record["output_observation"] = "Invalid action format. Reply with exactly one valid command."
                record["support_trace"]["environment_outcome"] = {"executed": False,
                                                                      "reason": "parse_error"}
                trace.append(record)
                if benchmark == "alfworld":
                    status = "parse_error"
                    break
                history.append({"action": "<PARSE_ERROR>", "observation": record["output_observation"]})
                continue
            try:
                environment_started = time.perf_counter()
                step = episode.step(action)
            except Exception as exc:
                record["environment_seconds"] = round(time.perf_counter() - environment_started, 3)
                environment_seconds += record["environment_seconds"]
                record["error"] = "environment_error"
                record["details"] = repr(exc)
                record["support_trace"]["environment_outcome"] = {"executed": False,
                                                                      "reason": "environment_error"}
                trace.append(record)
                status = "environment_error"
                break
            record["environment_seconds"] = round(time.perf_counter() - environment_started, 3)
            environment_seconds += record["environment_seconds"]
            actions_taken += 1
            failed_actions += int(step.get("failed_action", False))
            record.update({"environment_action_number": actions_taken,
                           "output_observation": step["observation"], "outcome": step})
            record["support_trace"]["environment_outcome"] = {
                "executed": True, "failed_action": bool(step.get("failed_action", False)),
                "done": bool(step["done"]), "success": bool(step["success"])}
            trace.append(record)
            history.append({"action": action, "observation": step["observation"]})
            if step["done"]:
                success = bool(step["success"])
                status = "success" if success else "task_failed"
                break
            if benchmark == "alfworld" and len(history) >= 6:
                recent = [item["action"] for item in history[-6:]]
                if recent[:2] == recent[2:4] == recent[4:6]:
                    status = "repeated_action_loop"
                    break
        else:
            if actions_taken >= config["max_actions"]:
                status = "max_actions"
    finally:
        episode.close()

    return {
        "task_id": task["task_id"], "benchmark": benchmark,
        "initial_observation": episode.initial_observation, "task_prompt": episode.task_prompt,
        "success": success, "status": status, "environment_actions": actions_taken,
        "model_decisions": len(trace), "parser_errors": parser_errors,
        "failed_actions": failed_actions,
        "model_calls": len(trace) + budget.model_calls_used,
        "input_tokens": proposal_input_tokens + budget.controller_input_tokens + budget.support_input_tokens,
        "output_tokens": proposal_output_tokens + budget.controller_output_tokens + budget.support_output_tokens,
        "inference_seconds": round(proposal_inference_seconds + budget.controller_inference_seconds +
                                   budget.support_inference_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "support_calls": budget.operations_used,
        "support_model_calls": budget.support_model_calls_used,
        "support_input_tokens": budget.support_input_tokens,
        "support_output_tokens": budget.support_output_tokens,
        "support_parse_errors": sum(
            event.get("result", {}).get("status") == "invalid_action"
            for record in trace for event in record["support_trace"]["events"]),
        "retrieval_seconds": round(sum(
            event.get("result", {}).get("metadata", {}).get("retrieval_seconds", 0.0)
            for record in trace for event in record["support_trace"]["events"]), 3),
        "actions_changed_by_support": sum(record["action"] is not None and
                                          record["action"] != record["proposed_action"]
                                          for record in trace),
        "controller_model_calls": budget.controller_calls_used,
        "controller_input_tokens": budget.controller_input_tokens,
        "controller_output_tokens": budget.controller_output_tokens,
        "controller_inference_seconds": round(budget.controller_inference_seconds, 3),
        "support_inference_seconds": round(budget.support_inference_seconds, 3),
        "proposal_model_calls": len(trace),
        "proposal_input_tokens": proposal_input_tokens,
        "proposal_output_tokens": proposal_output_tokens,
        "proposal_inference_seconds": round(proposal_inference_seconds, 3),
        "environment_seconds": round(environment_seconds, 3),
        "support_budget": budget.as_dict(),
        "trace": trace,
    }
