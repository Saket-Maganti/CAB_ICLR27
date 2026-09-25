#!/usr/bin/env python3
"""Run the portable CAB plumbing demo without a model server or GPU."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cab.runner import run_episode
from cab.tasks import load_tasks


class ScriptedModel:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return next(self.responses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSONL destination")
    args = parser.parse_args()
    task = load_tasks(ROOT / "data" / "smoke_task.json")[0]
    runs = [
        run_episode(task, ScriptedModel([
            '{"type":"read","source_ids":["SRC_A1C3"]}',
            '{"type":"read","source_ids":["SRC_C3E5"]}',
            '{"type":"final","answer":{"config_path":"build/runner.yml","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}}',
        ]), condition="DISRUPT", arm="BASE", interface="SERIAL",
            settings={"parser": "e04", "max_steps": 8}),
        run_episode(task, ScriptedModel([
            '{"type":"read","source_ids":["SRC_A1C3","SRC_C3E5"]}',
            '{"type":"final","answer":{"config_path":"./build/runner.yml","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}}',
        ]), condition="DISRUPT", arm="PROCEDURAL", interface="PARALLEL",
            settings={"parser": "e04", "max_steps": 8}),
    ]
    if not all(row["correct"] for row in runs):
        raise SystemExit("smoke run failed")
    output = args.output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in runs), encoding="utf-8")
    else:
        with tempfile.TemporaryDirectory(prefix="cab-smoke-") as tmp:
            output = Path(tmp) / "results.jsonl"
            output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in runs), encoding="utf-8")
            print(f"wrote {len(runs)} result records to {output}")
            print(f"SERIAL={runs[0]['correct']} PARALLEL={runs[1]['correct']} (mock model; runtime plumbing only)")
            return
    print(f"wrote {len(runs)} result records to {output}")
    print(f"SERIAL={runs[0]['correct']} PARALLEL={runs[1]['correct']} (mock model; runtime plumbing only)")


if __name__ == "__main__":
    main()
