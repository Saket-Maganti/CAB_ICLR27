"""Load task panels for the portable CAB reference runner."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    answer_schema: dict[str, str]
    gold: dict[str, Any]
    source_ids: tuple[str, ...]
    contents: dict[str, dict[str, str]]
    raw: dict[str, Any]

    def sources_for(self, condition: str) -> dict[str, str]:
        try:
            return self.contents[condition]
        except KeyError as exc:
            raise ValueError(f"task {self.task_id!r} has no condition {condition!r}") from exc

    def source_ids_for(self, condition: str) -> tuple[str, ...]:
        available = self.sources_for(condition)
        return tuple(source_id for source_id in self.source_ids if source_id in available)


def task_from_mapping(value: dict[str, Any]) -> Task:
    required = {"task_id", "question", "answer_schema", "gold", "source_ids", "contents"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"task is missing required fields: {sorted(missing)}")
    if not isinstance(value["task_id"], str) or not value["task_id"]:
        raise ValueError("task_id must be a non-empty string")
    if not isinstance(value["question"], str) or not value["question"]:
        raise ValueError("question must be a non-empty string")
    schema = value["answer_schema"]
    if not isinstance(schema, dict) or not schema or any(t not in {"string", "integer"} for t in schema.values()):
        raise ValueError("answer_schema must map answer fields to 'string' or 'integer'")
    if not isinstance(value["gold"], dict) or set(value["gold"]) != set(schema):
        raise ValueError("gold must provide exactly the answer_schema fields")
    for field, field_type in schema.items():
        expected_type = str if field_type == "string" else int
        if type(value["gold"][field]) is not expected_type:
            raise ValueError(f"gold[{field!r}] does not match answer_schema type {field_type!r}")
    if not isinstance(value["source_ids"], list) or not value["source_ids"]:
        raise ValueError("source_ids must be a non-empty list")
    source_ids = tuple(value["source_ids"])
    if any(not isinstance(x, str) for x in source_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("source_ids must contain unique strings")
    if not isinstance(value["contents"], dict) or not value["contents"]:
        raise ValueError("contents must map conditions to source content")
    contents: dict[str, dict[str, str]] = {}
    for condition, condition_sources in value["contents"].items():
        if not isinstance(condition_sources, dict):
            raise ValueError(f"contents[{condition!r}] must be an object")
        if set(condition_sources) - set(source_ids):
            raise ValueError(f"contents[{condition!r}] contains undeclared source IDs")
        if any(not isinstance(text, str) for text in condition_sources.values()):
            raise ValueError(f"contents[{condition!r}] values must be strings")
        contents[condition] = dict(condition_sources)
    aliases = value.get("aliases", {})
    if not isinstance(aliases, dict) or set(aliases) - set(schema):
        raise ValueError("aliases must map declared answer fields to alias objects")
    for field, field_aliases in aliases.items():
        if not isinstance(field_aliases, dict) or any(not isinstance(k, str) for k in field_aliases):
            raise ValueError(f"aliases[{field!r}] must map strings to canonical values")
    return Task(value["task_id"], value["question"], dict(schema), dict(value["gold"]),
                source_ids, contents, dict(value))


def load_tasks(path: str | Path) -> list[Task]:
    """Read one task object, a `{tasks: [...]}` file, or a JSON Lines panel."""
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        decoded = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(decoded, dict) and "tasks" in decoded:
        decoded = decoded["tasks"]
    if isinstance(decoded, dict):
        decoded = [decoded]
    if not isinstance(decoded, list):
        raise ValueError("task file must contain an object, a tasks array, or JSON Lines")
    tasks = [task_from_mapping(item) for item in decoded]
    ids = [task.task_id for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task IDs must be unique within a panel")
    return tasks
