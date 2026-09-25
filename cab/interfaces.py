"""CAB source-read action contracts."""
from __future__ import annotations

from typing import Any


def action_kind(action: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return action kind and protocol key; mixed or missing keys are invalid."""
    has_action = "action" in action
    has_type = "type" in action
    if has_action == has_type:
        return None, None
    key = "action" if has_action else "type"
    return action.get(key), key


def _matches_type(value: Any, field_type: str) -> bool:
    if field_type == "string":
        return type(value) is str
    if field_type == "integer":
        return type(value) is int
    return False


def validate_action(action: dict[str, Any], answer_schema: dict[str, str],
                    declared_source_ids: tuple[str, ...] | list[str], interface: str) -> tuple[bool, str | None]:
    """Validate E01/E05 `action` and E04 `type` protocols with shared cardinality."""
    if interface not in {"SERIAL", "PARALLEL"}:
        return False, "unknown_interface"
    kind, protocol_key = action_kind(action)
    if kind is None or protocol_key is None:
        return False, "action_shape"
    if kind == "read":
        if set(action) != {protocol_key, "source_ids"}:
            return False, "read_keys"
        ids = action.get("source_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
            return False, "read_ids"
        if len(ids) != len(set(ids)):
            return False, "duplicates"
        if any(source_id not in declared_source_ids for source_id in ids):
            return False, "undeclared_source"
        if interface == "SERIAL" and len(ids) != 1:
            return False, "serial_cardinality"
        return True, None
    if kind == "final":
        if set(action) != {protocol_key, "answer"} or not isinstance(action.get("answer"), dict):
            return False, "final_shape"
        answer = action["answer"]
        if set(answer) != set(answer_schema):
            return False, "final_keys"
        if any(not _matches_type(answer[name], field_type) for name, field_type in answer_schema.items()):
            return False, "final_type"
        return True, None
    return False, "unknown_action"
