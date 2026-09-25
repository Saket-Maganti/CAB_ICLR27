"""Exact structured-answer scoring with explicit task-level aliases."""
from __future__ import annotations

from typing import Any

from .tasks import Task


def normalize_answer(answer: dict[str, Any], aliases: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Map only explicitly declared aliases to their canonical field values."""
    normalized = dict(answer)
    for field, field_aliases in (aliases or {}).items():
        value = normalized.get(field)
        if isinstance(value, str) and value in field_aliases:
            normalized[field] = field_aliases[value]
    return normalized


def score_answer(task: Task, answer: dict[str, Any] | None) -> bool:
    if answer is None or set(answer) != set(task.answer_schema):
        return False
    aliases = task.raw.get("aliases", {})
    return normalize_answer(answer, aliases) == task.gold
