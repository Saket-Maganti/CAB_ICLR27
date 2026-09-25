"""Portable CAB episode runner."""
from __future__ import annotations

import json
from typing import Any, Protocol

from .interfaces import action_kind, validate_action
from .parsing import parse_response
from .prompts import build_prompt
from .scoring import score_answer
from .tasks import Task


class Model(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


def _retry_feedback(policy: dict[str, Any], count: int) -> bool:
    return policy.get("mode", "stop") == "retry" and count <= int(policy.get("max_retries", 0))


def run_episode(task: Task, model: Model, *, condition: str, arm: str, interface: str,
                settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or {}
    max_steps = int(settings.get("max_steps", task.raw.get("max_steps", 8)))
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    parse_mode = settings.get("parser", "e01_e05")
    invalid_policy = settings.get("invalid_action", {"mode": "retry", "max_retries": 1})
    tool_policy = settings.get("tool_failure", {"mode": "return_error", "max_retries": 1})
    messages = [{"role": "user", "content": build_prompt(task, condition, arm, interface)}]
    failures_remaining = {source_id: int(spec.get("failures", 1))
                         for source_id, spec in task.raw.get("tool_failures", {}).get(condition, {}).items()}
    trace: list[dict[str, Any]] = []
    parser_rejections = action_rejections = tool_calls = turn_count = 0
    final_answer = None
    error_class = None
    source_acquired = False
    tool_failure_count = 0

    for step in range(max_steps):
        turn_count += 1
        raw = model.complete(messages)
        try:
            action = parse_response(raw, allow_e04_formatting=parse_mode == "e04")
        except (json.JSONDecodeError, ValueError):
            parser_rejections += 1
            trace.append({"step": step, "raw": raw, "error": "parser_rejection"})
            if _retry_feedback(settings.get("parser_failure", invalid_policy), parser_rejections):
                messages.extend([{"role": "assistant", "content": raw},
                                 {"role": "user", "content": "Invalid JSON object. Return one valid action object."}])
                continue
            error_class = "parser_rejection"
            break
        valid, reason = validate_action(action, task.answer_schema, task.source_ids_for(condition), interface)
        if not valid:
            action_rejections += 1
            trace.append({"step": step, "action": action, "error": reason})
            if _retry_feedback(invalid_policy, action_rejections):
                messages.extend([{"role": "assistant", "content": raw},
                                 {"role": "user", "content": f"Invalid action ({reason}). Return a valid action."}])
                continue
            error_class = f"action_{reason}"
            break
        kind, _ = action_kind(action)
        if kind == "final":
            final_answer = action["answer"]
            trace.append({"step": step, "action": action})
            break

        source_ids = action["source_ids"]
        tool_calls += len(source_ids)
        results: dict[str, str] = {}
        failed_sources: list[str] = []
        available = task.sources_for(condition)
        for source_id in source_ids:
            if source_id not in available:
                failed_sources.append(source_id)
                results[source_id] = "Source unavailable in this condition."
                continue
            remaining = failures_remaining.get(source_id, 0)
            if remaining > 0:
                failures_remaining[source_id] = remaining - 1
                failed_sources.append(source_id)
                spec = task.raw.get("tool_failures", {}).get(condition, {}).get(source_id, {})
                results[source_id] = str(spec.get("message", "Tool read failed."))
                continue
            results[source_id] = available[source_id]
        source_acquired = source_acquired or any(source_id in available and source_id not in failed_sources for source_id in source_ids)
        trace.append({"step": step, "action": action, "tool_result": results,
                      "failed_source_ids": failed_sources})
        messages.extend([{"role": "assistant", "content": raw},
                         {"role": "user", "content": "TOOL_RESULT\n" + json.dumps(results, ensure_ascii=False, separators=(",", ":"))}])
        if failed_sources and tool_policy.get("mode") == "stop":
            error_class = "tool_failure"
            break
        if failed_sources and tool_policy.get("mode") == "retry":
            tool_failure_count += 1
            if tool_failure_count > int(tool_policy.get("max_retries", 0)):
                error_class = "tool_failure"
                break
    else:
        error_class = "max_steps"

    return {"schema_version": 1, "task_id": task.task_id, "condition": condition, "arm": arm,
            "interface": interface, "final_answer": final_answer, "correct": score_answer(task, final_answer),
            "valid_terminal_final": final_answer is not None, "runtime_complete": error_class is None,
            "parser_rejections": parser_rejections, "action_rejections": action_rejections,
            "tool_calls": tool_calls, "turn_count": turn_count, "source_acquired": source_acquired,
            "error_class": error_class, "trace": trace}
