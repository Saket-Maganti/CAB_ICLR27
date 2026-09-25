"""Prompt construction for supplied CAB task panels."""
from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from .tasks import Task

_ARM_GUIDANCE = {
    "BASE": "Use the declared sources to answer the question.",
    "CONTROL": "Inspect only evidence that can help answer the question.",
    "NEUTRAL": "Use the declared sources as needed, then answer the question.",
    "PROCEDURAL": "Check source relevance, compare supporting evidence, then answer the question.",
}
_E01_PROTOCOLS = {
    "SERIAL": ('You interact with declared repository evidence by returning exactly one JSON action per turn. '
               'Read action: {"action":"read","source_ids":["ONE_DECLARED_ID"]}. '
               'Final action: {"action":"final","answer":{...}}. '
               "SERIAL requires exactly one source ID in every read. Do not invent source IDs."),
    "PARALLEL": ('You interact with declared repository evidence by returning exactly one JSON action per turn. '
                 'Read action: {"action":"read","source_ids":["ID1","ID2",...]}. '
                 'Final action: {"action":"final","answer":{...}}. '
                 "PARALLEL may read one or more declared source IDs in one action. Do not invent source IDs."),
}
_E01_ARM_TEXT = {
    "BASE": "",
    "NEUTRAL": "CONTEXT_NOTE: The declared sources use stable identifiers assigned at panel construction and retained consistently for this task.",
    "PROCEDURAL": "LOCALIZATION_STEP: Before answering, determine which of the declared sources are relevant to this task, then consult them.",
}


def attach_prompt_manifest(tasks: list[Task], path: str | Path) -> list[Task]:
    """Attach exact prompts from a paper manifest without re-rendering them."""
    value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    index: dict[str, dict[str, dict[str, str]]] = {}

    def add(task_id: str, interface: str, arm: str, prompt: str) -> None:
        arms = index.setdefault(task_id, {}).setdefault(interface, {})
        previous = arms.get(arm)
        if previous is not None and previous != prompt:
            raise ValueError(f"conflicting exact prompt records for {task_id}/{interface}/{arm}")
        arms[arm] = prompt

    if isinstance(value, dict) and value.get("schema_version") == "cab-paper-prompts-v1":
        definitions = value.get("arm_definitions", {})
        observed_counts: dict[str, int] = {}
        for base in value.get("base_prompts", []):
            task_id, interface, prompt = base.get("task_id"), base.get("interface"), base.get("text")
            if not all(isinstance(item, str) for item in (task_id, interface, prompt)):
                raise ValueError("paper prompt base needs task_id, interface, and text strings")
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if digest != base.get("sha256"):
                raise ValueError(f"paper prompt base hash mismatch for {task_id}/{interface}")
            for evaluation, arms in base.get("prompt_hashes", {}).items():
                if evaluation not in definitions or not isinstance(arms, dict):
                    raise ValueError(f"unknown prompt evaluation {evaluation!r}")
                for arm, expected_hash in arms.items():
                    definition = definitions[evaluation].get(arm)
                    if not isinstance(definition, dict) or not isinstance(definition.get("append_text"), str):
                        raise ValueError(f"missing exact arm addition for {evaluation}/{arm}")
                    exact = prompt + definition["append_text"]
                    exact_hash = hashlib.sha256(exact.encode("utf-8")).hexdigest()
                    if exact_hash != expected_hash:
                        raise ValueError(f"paper prompt hash mismatch for {evaluation}/{task_id}/{interface}/{arm}")
                    add(task_id, interface, arm, exact)
                    observed_counts[evaluation] = observed_counts.get(evaluation, 0) + 1
        if observed_counts != value.get("prompt_record_counts", observed_counts):
            raise ValueError("paper prompt record counts do not match the manifest index")
    else:
        rows = value if isinstance(value, list) else value.get("prompts", [])
        if not isinstance(rows, list):
            raise ValueError("prompt manifest must be an array or an object with a prompts array")
        for row in rows:
            task_id = row.get("task_id", row.get("case_id"))
            interface = row.get("interface")
            arm = row.get("arm", row.get("arm_id"))
            prompt = row.get("prompt")
            if not all(isinstance(item, str) for item in (task_id, interface, arm, prompt)):
                raise ValueError("each prompt record needs task_id/case_id, interface, arm/arm_id, and prompt strings")
            add(task_id, interface, arm, prompt)
    output = []
    for task in tasks:
        raw = dict(task.raw)
        attached = {iface: dict(arms) for iface, arms in index.get(task.task_id, {}).items()}
        existing = raw.get("prompts", {})
        for iface, arms in existing.items():
            attached.setdefault(iface, {}).update(arms)
        raw["prompts"] = attached
        output.append(replace(task, raw=raw))
    return output


def build_prompt(task: Task, condition: str, arm: str, interface: str) -> str:
    """Use a supplied frozen prompt verbatim when present; otherwise use the portable template."""
    prompts = task.raw.get("prompts", {})
    exact = prompts.get(interface, {}).get(arm) if isinstance(prompts, dict) else None
    if isinstance(exact, str):
        return exact
    descriptions = task.raw.get("source_descriptions", {})
    source_lines = [f"- {source_id}: {descriptions.get(source_id, 'source')}" for source_id in task.source_ids_for(condition)]
    protocol = task.raw.get("protocol", "e01_e05")
    if protocol != "e04" and interface in _E01_PROTOCOLS:
        lines = [
            _E01_PROTOCOLS[interface], "", "Declared source catalog:", *source_lines,
            "", "Task:", task.question, "",
            "Required answer keys and types: " + json.dumps(task.answer_schema, sort_keys=True),
            "Use repository evidence rather than guessing hidden values.",
        ]
        prompt = "\n".join(lines)
        suffix = _E01_ARM_TEXT.get(arm)
        if suffix is None:
            suffix = _ARM_GUIDANCE.get(arm)
            if suffix is None:
                raise ValueError(f"no portable prompt template for arm {arm!r}; supply an exact prompt in the manifest")
        return prompt if not suffix else prompt + "\n\n" + suffix
    if protocol == "e04":
        actions = '{"type":"read","source_ids":[...]} or {"type":"final","answer":{...}}'
    else:
        actions = '{"action":"read","source_ids":[...]} or {"action":"final","answer":{...}}'
    return "\n".join([
        "You are a CAB repository-evidence agent. Use only declared sources.",
        _ARM_GUIDANCE.get(arm, _ARM_GUIDANCE["BASE"]),
        f"Interface: {interface}. In SERIAL, read exactly one source per action; in PARALLEL, read one or more.",
        f"Question: {task.question}",
        f"Required answer fields and types: {json.dumps(task.answer_schema, separators=(',', ':'))}",
        "Declared sources:", *source_lines,
        f"Return exactly one JSON object per turn: {actions}.",
    ])
