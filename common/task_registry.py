"""Helpers for loading the ASCENT-G task registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_REGISTRY_PATH = REPO_ROOT / "config" / "task_registry.json"


def load_task_registry() -> dict[str, Any]:
    with TASK_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_task_config(task_name: str) -> dict[str, Any]:
    registry = load_task_registry()
    tasks = registry.get("tasks", {})
    if task_name not in tasks:
        raise KeyError(f"Unknown task '{task_name}'. Known tasks: {sorted(tasks)}")
    return tasks[task_name]
