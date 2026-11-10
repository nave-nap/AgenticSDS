"""Post-proposal reconsideration using only the current task state."""

from __future__ import annotations


def reconsideration_messages(base_messages: list[dict], proposed_action: str, benchmark: str) -> list[dict]:
    """Keep the baseline state and answer cue; add no training-task content."""
    messages = [dict(message) for message in base_messages]
    content = messages[-1]["content"]
    if benchmark == "textcraft":
        cue = "\nAction:"
        if not content.endswith(cue):
            raise ValueError("TextCraft baseline answer cue is missing")
        guidance = (f"Your proposed action was: {proposed_action}\n"
                    "Reconsider it using only the current task, recipes, and feedback above. "
                    "Select one valid next action. Reply in the required format.")
        messages[-1]["content"] = content[:-len(cue)] + "\n\n" + guidance + "\n" + cue.lstrip("\n")
    elif benchmark == "alfworld":
        cue = "Choose the best next action."
        if not content.endswith(cue):
            raise ValueError("ALFWorld baseline answer cue is missing")
        guidance = (f"Your proposed action was: {proposed_action}\n"
                    "Reconsider it using only the current task, observations, and available actions above. "
                    "Select one valid next action from the numbered list. Reply with its number.")
        messages[-1]["content"] = content[:-len(cue)] + guidance + "\n\n" + cue
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return messages
