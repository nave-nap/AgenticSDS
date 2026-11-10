"""TextCraft ADaPT controller adapted from Prasad et al. (NAACL 2024).

Upstream: https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/run_textcraft.py
The prompt examples are retained in adapt_prompts.py under the upstream MIT
license. This controller is an independent, development-only condition.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol

from . import adapt_prompts
from .benchmarks import make_episode
from .policy import parse_textcraft


class PlanParseError(ValueError):
    pass


@dataclass(frozen=True)
class PlanNode:
    logic: str
    steps: tuple["PlanNode | str", ...]

    def as_dict(self) -> dict:
        return {"logic": self.logic,
                "steps": [step.as_dict() if isinstance(step, PlanNode) else step
                          for step in self.steps]}


def parse_plan(raw: str) -> PlanNode:
    """Parse the authors' Step N / Execution Order format without AND fallback."""
    steps: dict[int, str] = {}
    order = None
    for line in raw.splitlines():
        step = re.fullmatch(r"\s*Step\s+(\d+)\s*:\s*(.*?)\s*", line, flags=re.I)
        if step:
            number, text = int(step.group(1)), step.group(2).strip()
            if number in steps or not text:
                raise PlanParseError("Duplicate or empty Step definition")
            steps[number] = text
        elif re.match(r"\s*Execution\s+Order\s*:", line, flags=re.I):
            if order is not None:
                raise PlanParseError("Multiple Execution Order expressions")
            order = line.split(":", 1)[1].strip()
    if not steps or not order or len(steps) > 12:
        raise PlanParseError("Plan needs Step definitions and one Execution Order")
    token_pattern = re.compile(r"Step\s+\d+|AND|OR|\(|\)", flags=re.I)
    tokens = []
    position = 0
    while position < len(order):
        if order[position].isspace():
            position += 1
            continue
        match = token_pattern.match(order, position)
        if match is None:
            raise PlanParseError("Unexpected text in Execution Order")
        tokens.append(re.sub(r"\s+", " ", match.group().upper()))
        position = match.end()
    cursor = 0
    used: list[int] = []

    def atom() -> PlanNode | str:
        nonlocal cursor
        if cursor >= len(tokens):
            raise PlanParseError("Incomplete Execution Order")
        token = tokens[cursor]
        cursor += 1
        if token == "(":
            nested = expression()
            if cursor >= len(tokens) or tokens[cursor] != ")":
                raise PlanParseError("Unclosed parenthesis")
            cursor += 1
            return nested
        if token.startswith("STEP "):
            number = int(token.split()[1])
            if number not in steps or number in used:
                raise PlanParseError("Unknown or repeated Step reference")
            used.append(number)
            return steps[number]
        raise PlanParseError("Expected a Step reference or parenthesized group")

    def expression() -> PlanNode | str:
        nonlocal cursor
        parts = [atom()]
        logic = None
        while cursor < len(tokens) and tokens[cursor] in {"AND", "OR"}:
            operator = tokens[cursor]
            cursor += 1
            if logic is not None and logic != operator:
                raise PlanParseError("Mixed AND/OR needs parentheses")
            logic = operator
            parts.append(atom())
        return PlanNode(logic, tuple(parts)) if logic else parts[0]

    parsed = expression()
    if cursor != len(tokens) or set(used) != set(steps):
        raise PlanParseError("Execution Order has extra tokens or unused Steps")
    return parsed if isinstance(parsed, PlanNode) else PlanNode("AND", (parsed,))


class PlannerRole(Protocol):
    def plan(self, commands: str, task: str) -> dict: ...


class ExecutorRole(Protocol):
    def next_action(self, commands: str, task: str, inventory_hint: str,
                    history: list[tuple[str, str]]) -> dict: ...


class QwenPlanner:
    def __init__(self, model, max_new_tokens: int):
        self.model, self.max_new_tokens = model, max_new_tokens

    def plan(self, commands: str, task: str) -> dict:
        return self.model.generate([
            {"role": "system", "content": "You are a helpful assistant helping me play a simple version of Minecraft."},
            {"role": "user", "content": adapt_prompts.plan_prompt + "\n" + commands + "\nGoal: " + task},
        ], max_new_tokens=self.max_new_tokens)


class QwenExecutor:
    def __init__(self, model, max_new_tokens: int):
        self.model, self.max_new_tokens = model, max_new_tokens

    def next_action(self, commands: str, task: str, inventory_hint: str,
                    history: list[tuple[str, str]]) -> dict:
        prompt = adapt_prompts.atomic_exec_prompt + "\n" + commands + "\nGoal: " + task + "\n"
        prompt += ("\nUse the exact available crafting command with its printed quantities. "
                   "For gathering, use get N ITEM; for inventory, use inventory. "
                   "Output exactly one command or one think: line first.\n")
        if inventory_hint:
            prompt += "\n" + inventory_hint + "\n"
        for action, feedback in history:
            prompt += f"> {action}\n{feedback}\n"
        prompt += ">"
        return self.model.generate([
            {"role": "system", "content": "You are a helpful assistant playing Minecraft. Produce one next action or think line."},
            {"role": "user", "content": prompt},
        ], max_new_tokens=self.max_new_tokens)


def _goal_item(task: str) -> tuple[int, str] | None:
    match = re.fullmatch(r"\s*(?:fetch|get|craft)\s+(?:(\d+)\s+)?([a-z0-9][a-z0-9 _-]*?)(?:\s+using\s+.+)?\.?\s*",
                         task.lower())
    if not match:
        return None
    return int(match.group(1) or 1), match.group(2).replace("_", " ").strip()


def _inventory_count(observation: str, item: str) -> int:
    expected = {item}
    if item.endswith("s") and not item.endswith("ss"):
        expected.add(item[:-1])
    for name, count in re.findall(r"\[([^\]]+)\]\s*\((\d+)\)", observation.lower()):
        if name.replace("_", " ").strip() in expected:
            return int(count)
    return 0


def _next_line(raw: str) -> str:
    line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    line = re.sub(r"^>\s*", "", line)
    line = re.sub(r"^Action\s*:\s*", "", line, flags=re.I).strip().rstrip(".")
    return "inventory" if line.lower() == "inventory:" else line


def _parse_adapt_action(action: str) -> str | None:
    """Accept TextCraft's optional craft output count used by upstream demos."""
    parsed = parse_textcraft(action)
    if parsed is not None:
        return parsed
    normalized = re.sub(r"(?<=\w)_(?=\w)", " ", action)
    if re.fullmatch(r"craft [a-z0-9 ]+ using (?:[1-9][0-9]* [a-z0-9 ]+)(?:, [1-9][0-9]* [a-z0-9 ]+)*",
                    normalized):
        return normalized
    return None


class ADaPTController:
    """Execution-first recursive AND/OR controller with a shared TextCraft state."""

    def __init__(self, episode, config: dict, planner: PlannerRole, executor: ExecutorRole):
        self.episode, self.config = episode, config
        self.planner, self.executor = planner, executor
        self.trace: list[dict] = []
        self.controller_events: list[dict] = []
        self.environment_actions = self.failed_actions = self.parser_errors = 0
        self.planner_calls = self.executor_calls = 0
        self.input_tokens = self.output_tokens = 0
        self.inference_seconds = 0.0
        self.goal_done = False
        self.environment_terminated = False
        self.max_depth_reached = 0
        self.decompositions = self.plan_parse_errors = 0
        self.false_completion_claims = 0
        self.direct_execution_succeeded = False
        self.decomposition_trigger = None
        self.root_task = ""
        self.commands = ""
        self.patience = 8  # Final TextCraft executor in official run_textcraft.py.

    @property
    def model_calls(self) -> int:
        return self.planner_calls + self.executor_calls

    def _record_generation(self, role: str, generation: dict, task: str, depth: int) -> dict:
        self.input_tokens += generation["input_tokens"]
        self.output_tokens += generation["output_tokens"]
        self.inference_seconds += generation["inference_seconds"]
        record = {"decision": len(self.trace) + 1, "role": role, "task": task,
                  "depth": depth, "input_observation": self.episode.observation,
                  "raw_model_output": generation["text"],
                  "input_tokens": generation["input_tokens"],
                  "output_tokens": generation["output_tokens"],
                  "inference_seconds": generation["inference_seconds"]}
        self.trace.append(record)
        return record

    def _step(self, action: str, task: str, depth: int, source: str, record: dict | None = None) -> dict:
        if self.environment_actions >= self.config["adapt_max_environment_actions"]:
            raise RuntimeError("global_action_budget")
        outcome = self.episode.step(action)
        self.environment_actions += 1
        self.failed_actions += int(outcome.get("failed_action", False))
        self.goal_done |= bool(outcome.get("done") and outcome.get("success"))
        self.environment_terminated |= bool(outcome.get("done"))
        event = {"source": source, "task": task, "depth": depth, "action": action,
                 "observation": outcome["observation"], "outcome": outcome,
                 "environment_action_number": self.environment_actions}
        self.controller_events.append(event)
        if record is not None:
            record.update({"action": action, "output_observation": outcome["observation"],
                           "outcome": outcome, "environment_action_number": self.environment_actions})
        return outcome

    def _executor_attempt(self, task: str, depth: int, inventory_hint: str) -> tuple[bool, str, str]:
        local_history: list[tuple[str, str]] = []
        patience = 0
        for _ in range(self.config["adapt_max_executor_steps"]):
            if self.environment_terminated:
                return self.goal_done, "environment_terminated", inventory_hint
            if self.model_calls >= self.config["adapt_max_model_calls"]:
                return False, "global_model_call_budget", inventory_hint
            if self.environment_actions >= self.config["adapt_max_environment_actions"]:
                return False, "global_action_budget", inventory_hint
            generation = self.executor.next_action(self.commands, task, inventory_hint, local_history)
            self.executor_calls += 1
            record = self._record_generation("executor", generation, task, depth)
            action = _next_line(generation["text"])
            record["proposed_line"] = action
            if action.lower().startswith("think"):
                record["action_type"] = "thought"
                if "task completed" in action.lower():
                    record["self_reported_completion"] = True
                    if self.goal_done:
                        record["verified_completion"] = True
                        return True, "verified_original_goal", inventory_hint
                    goal = _goal_item(task)
                    if goal is None:
                        record["verified_completion"] = False
                        self.false_completion_claims += 1
                        return False, "unverifiable_completion_claim", inventory_hint
                    if self.environment_actions >= self.config["adapt_max_environment_actions"]:
                        return False, "global_action_budget", inventory_hint
                    inventory = self._step("inventory", task, depth, "completion_verification")
                    inventory_hint = inventory["observation"]
                    verified = _inventory_count(inventory_hint, goal[1]) >= goal[0]
                    record["verified_completion"] = verified
                    record["inventory_observation"] = inventory_hint
                    if verified:
                        return True, "verified_inventory_subtask", inventory_hint
                    self.false_completion_claims += 1
                    return False, "false_completion_claim", inventory_hint
                if "task failed" in action.lower():
                    record["self_reported_failure"] = True
                    return False, "executor_reported_failure", inventory_hint
                feedback = "OK."
                patience += 1
            else:
                parsed = _parse_adapt_action(action)
                if parsed is None:
                    self.parser_errors += 1
                    record["error"] = "invalid_action_format"
                    feedback = "Could not parse action"
                    patience += 1
                else:
                    try:
                        outcome = self._step(parsed, task, depth, "executor", record)
                    except RuntimeError as exc:
                        if str(exc) == "global_action_budget":
                            return False, "global_action_budget", inventory_hint
                        raise
                    except Exception as exc:
                        record["error"] = "environment_error"
                        record["details"] = repr(exc)
                        return False, "environment_error", inventory_hint
                    feedback = outcome["observation"]
                    if parsed == "inventory":
                        inventory_hint = feedback
                    patience = patience + 1 if ("Could not" in feedback or feedback == "OK.") else 0
                    if self.goal_done:
                        return True, "verified_original_goal", inventory_hint
            local_history.append((action, feedback))
            record["output_observation"] = feedback
            if patience >= self.patience:
                return False, "executor_patience_exhausted", inventory_hint
        return False, "executor_step_limit", inventory_hint

    def _plan(self, task: str, depth: int) -> tuple[PlanNode | None, dict]:
        if self.model_calls >= self.config["adapt_max_model_calls"]:
            return None, {"reason": "global_model_call_budget"}
        if self.planner_calls >= self.config["adapt_max_planner_calls"]:
            return None, {"reason": "planner_call_budget"}
        generation = self.planner.plan(self.commands, task)
        self.planner_calls += 1
        record = self._record_generation("planner", generation, task, depth)
        try:
            plan = parse_plan(generation["text"])
        except PlanParseError as exc:
            self.plan_parse_errors += 1
            record["plan_parse_error"] = str(exc)
            return None, {"reason": "malformed_plan", "raw_plan": generation["text"],
                          "parse_error": str(exc)}
        record["parsed_plan"] = plan.as_dict()
        self.decompositions += 1
        return plan, {"raw_plan": generation["text"], "parsed_plan": plan.as_dict()}

    def _run_plan(self, plan: PlanNode, depth: int, ancestors: tuple[str, ...],
                  inventory_hint: str) -> tuple[bool, list[dict], str]:
        children = []
        for subtask in plan.steps:
            if self.goal_done:
                return True, children, inventory_hint
            if self.environment_terminated:
                return False, children, inventory_hint
            if isinstance(subtask, PlanNode):
                success, nested, updated = self._run_plan(subtask, depth, ancestors, inventory_hint)
                child = {"logic": subtask.logic, "children": nested, "success": success}
            else:
                child, updated = self._execute(subtask, depth + 1, ancestors, inventory_hint)
                success = child["success"]
            children.append(child)
            inventory_hint = updated
            if success:
                if plan.logic == "OR":
                    return True, children, inventory_hint
            elif plan.logic == "AND":
                return False, children, inventory_hint
        return plan.logic == "AND", children, inventory_hint

    def _execute(self, task: str, depth: int, ancestors: tuple[str, ...],
                 inventory_hint: str) -> tuple[dict, str]:
        self.max_depth_reached = max(self.max_depth_reached, depth)
        node = {"task": task, "depth": depth, "direct_success": False,
                "success": False, "decomposition_trigger": None, "children": []}
        if self.goal_done:
            node.update(success=True, reason="original_goal_already_satisfied")
            return node, inventory_hint
        if self.environment_terminated:
            node["reason"] = "environment_terminated"
            return node, inventory_hint
        if depth > self.config["adapt_max_depth"]:
            node["reason"] = "max_depth"
            return node, inventory_hint
        normalized = " ".join(task.lower().split())
        if normalized in ancestors:
            node["reason"] = "recursive_task_loop"
            return node, inventory_hint
        direct, reason, inventory_hint = self._executor_attempt(task, depth, inventory_hint)
        node.update(direct_success=direct, direct_reason=reason)
        if depth == 1:
            self.direct_execution_succeeded = direct
            self.decomposition_trigger = None if direct else reason
        if direct:
            node.update(success=True, reason=reason)
            return node, inventory_hint
        node["decomposition_trigger"] = reason
        if depth >= self.config["adapt_max_depth"] or reason in {
                "global_action_budget", "global_model_call_budget", "environment_terminated"}:
            node["reason"] = "max_depth" if depth >= self.config["adapt_max_depth"] else reason
            return node, inventory_hint
        plan, plan_info = self._plan(task, depth)
        node.update(plan_info)
        if plan is None:
            return node, inventory_hint
        composed, children, inventory_hint = self._run_plan(plan, depth, ancestors + (normalized,), inventory_hint)
        node.update(children=children, success=composed, reason="plan_composed" if composed else "plan_failed")
        return node, inventory_hint

    def run(self, task: dict) -> dict:
        started = time.perf_counter()
        try:
            self.episode.reset()
            original = self.episode.initial_observation
            if "\n\nGoal: " not in original:
                raise ValueError("TextCraft initial observation lacks a Goal section")
            self.commands, self.root_task = original.rsplit("\n\nGoal: ", 1)
            tree, _ = self._execute(self.root_task, 1, (), "")
        finally:
            self.episode.close()
        success = self.goal_done
        if success:
            status = "success"
        elif self.environment_actions >= self.config["adapt_max_environment_actions"]:
            status = "max_actions"
        elif self.model_calls >= self.config["adapt_max_model_calls"]:
            status = "max_decisions"
        elif self.environment_terminated:
            status = "environment_terminated"
        else:
            status = tree.get("reason", "task_failed")
        return {"task_id": task["task_id"], "benchmark": "textcraft",
                "initial_observation": original, "task_prompt": original,
                "success": success, "status": status,
                "environment_actions": self.environment_actions,
                "model_decisions": len(self.trace), "model_calls": self.model_calls,
                "parser_errors": self.parser_errors, "failed_actions": self.failed_actions,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "inference_seconds": round(self.inference_seconds, 3),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "trace": self.trace, "controller_events": self.controller_events,
                "original_task": self.root_task, "direct_execution_succeeded": self.direct_execution_succeeded,
                "decomposition_trigger": self.decomposition_trigger,
                "decomposition_tree": tree, "planner_calls": self.planner_calls,
                "executor_model_calls": self.executor_calls,
                "decompositions": self.decompositions, "max_depth_reached": self.max_depth_reached,
                "plan_parse_errors": self.plan_parse_errors,
                "false_completion_claims": self.false_completion_claims,
                "plan_heuristic_success": tree["success"],
                "support_calls": self.planner_calls, "support_model_calls": self.planner_calls,
                "support_input_tokens": sum(step["input_tokens"] for step in self.trace if step["role"] == "planner"),
                "support_output_tokens": sum(step["output_tokens"] for step in self.trace if step["role"] == "planner"),
                "support_parse_errors": self.plan_parse_errors,
                "retrieval_seconds": 0.0, "actions_changed_by_support": 0}


def run_adapt_episode(task: dict, config: dict, model, episode_factory=make_episode) -> dict:
    episode = episode_factory("textcraft", task, config)
    controller = ADaPTController(episode, config,
                                 QwenPlanner(model, config["adapt_planner_max_new_tokens"]),
                                 QwenExecutor(model, config["adapt_executor_max_new_tokens"]))
    return controller.run(task)
