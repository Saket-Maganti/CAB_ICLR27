"""Strict response parsing for the released CAB action shapes."""
from __future__ import annotations

import json
import re
from typing import Any

_UNQUOTED_SOURCE_ID = re.compile(r'(?<!["\w])(SRC_[A-F0-9]+)(?!["\w])')


def parse_response(raw: str, *, allow_e04_formatting: bool = False) -> dict[str, Any]:
    """Parse one JSON object; E04 mode also accepts fenced JSON and fixed SRC IDs."""
    text = raw.strip()
    if allow_e04_formatting and text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError("unterminated JSON code fence")
        text = "\n".join(lines[1:-1]).strip()
    if allow_e04_formatting:
        text = _UNQUOTED_SOURCE_ID.sub(r'"\1"', text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value
