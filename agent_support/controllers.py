"""Episode-level controllers, separate from per-decision support adapters."""

from __future__ import annotations

from typing import Callable


CONTROLLER_REGISTRY: dict[str, Callable] = {}


def register_controller(controller_id: str, runner: Callable) -> None:
    if not controller_id or controller_id in CONTROLLER_REGISTRY:
        raise ValueError(f"Controller is empty or already registered: {controller_id}")
    CONTROLLER_REGISTRY[controller_id] = runner


def run_controller(controller_id: str, task: dict, config: dict, model, episode_factory) -> dict:
    try:
        return CONTROLLER_REGISTRY[controller_id](task, config, model, episode_factory)
    except KeyError as exc:
        raise ValueError(f"Unknown controller: {controller_id}") from exc


def _adapt(task: dict, config: dict, model, episode_factory) -> dict:
    from .adapt import run_adapt_episode
    return run_adapt_episode(task, config, model, episode_factory)


register_controller("adapt_dynamic_replanning_v1", _adapt)
