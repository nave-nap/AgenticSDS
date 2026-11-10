"""Fixed no-support prompts and parsers derived from the exploratory notebooks."""

from __future__ import annotations

import re

PROMPT_VERSIONS = {"alfworld": "numbered_actions_v1", "textcraft": "single_command_v1"}


ALFWORLD_SYSTEM = (
    "You are an intelligent agent operating in ALFWorld. "
    "Your objective is to complete the assigned household task. "
    "Before choosing an action, consider the sequence of subgoals required to complete the entire task. "
    "If a task requires modifying an object, such as heating, cooling, or cleaning, complete that "
    "modification before placing it at its destination. "
    "Use previous observations to remember object locations and which steps have already been completed. "
    "Avoid repeating actions that have not produced progress. Choose exactly one action from the numbered list. "
    "Respond ONLY with its number."
)

TEXTCRAFT_SYSTEM = """You are an agent solving a text-based Minecraft crafting task.
You are given a list of available crafting commands and a target item.
Your job is to craft the target item in as few actions as possible.

Allowed commands:
- get N ITEM          (gather N of a basic/non-craftable item)
- craft N ITEM using N INGREDIENT, N INGREDIENT  (copy an exact available crafting command)
- inventory           (inspect your inventory)

Important rules:
- First plan the dependency order of recipes, then execute one command at a time.
- Gather base ingredients before crafting the items that depend on them.
- Do not try to 'get' any item that has to be crafted.
- Use only crafting commands from the task's displayed recipe list.
- Use exact names and explicit integer quantities for get/craft commands.
- Never assume a previous command succeeded unless the environment confirmed it.
- Respond with exactly one line: Action: <one command>
- Do not add explanations, markdown, or additional actions.
"""


def alfworld_messages(task: str, observation: str, actions: list[str], history: list[dict], window: int) -> list[dict]:
    recent = history[-window:]
    history_text = "\n".join(f"Action: {item['action']}\nObservation: {item['observation']}" for item in recent)
    action_list = "\n".join(f"{index}. {action}" for index, action in enumerate(actions, 1))
    return [
        {"role": "system", "content": ALFWORLD_SYSTEM},
        {"role": "user", "content": f"Task: {task}\n\nPrevious interactions:\n{history_text}\n\nCurrent observation:\n{observation}\n\nAvailable actions:\n{action_list}\n\nChoose the best next action."},
    ]


def textcraft_messages(initial_task: str, history: list[dict], window: int) -> list[dict]:
    lines = ["TASK AND ALLOWED CRAFTING RECIPES:", initial_task, "", "RECENT ACTIONS AND ENVIRONMENT FEEDBACK:"]
    recent = history[-window:]
    if recent:
        for index, step in enumerate(recent, 1):
            lines.append(f"{index}. Action: {step['action']}")
            lines.append(f"   Feedback: {step['observation']}")
    else:
        lines.append("(No actions taken yet.)")
    lines.extend(["", "Choose the next command.", "Action:"])
    return [{"role": "system", "content": TEXTCRAFT_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def parse_alfworld(raw: str, actions: list[str]) -> str | None:
    if not re.fullmatch(r"\d+", raw.strip()):
        return None
    index = int(raw.strip()) - 1
    return actions[index] if 0 <= index < len(actions) else None


def parse_textcraft(raw: str) -> str | None:
    cleaned = re.sub(r"(?<=\w)_(?=\w)", " ", raw)
    matches = re.findall(r"(?im)^\s*Action\s*:\s*(.*)$", cleaned)
    if len(matches) == 1:
        command = matches[0].strip().strip("`")
    elif len(matches) == 0 and len(cleaned.splitlines()) == 1:
        command = cleaned.strip().strip("`")
    else:
        return None
    if re.fullmatch(r"get [1-9][0-9]* [a-z0-9 ]+", command) and " using " not in command:
        return command
    if re.fullmatch(r"craft [1-9][0-9]* [a-z0-9 ]+ using (?:[1-9][0-9]* [a-z0-9 ]+)(?:, [1-9][0-9]* [a-z0-9 ]+)*", command):
        return command
    return "inventory" if command == "inventory" else None
