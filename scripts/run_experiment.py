#!/usr/bin/env python3
"""Run CAB tasks with a configured local model adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cab.models import create_model
from cab.prompts import attach_prompt_manifest
from cab.runner import run_episode
from cab.tasks import load_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="JSON task, panel, or JSONL file")
    parser.add_argument("--config", type=Path, required=True, help="model and run settings JSON")
    parser.add_argument("--output", type=Path, required=True, help="result JSONL path")
    parser.add_argument("--limit", type=int, help="run only the first N tasks")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = create_model(config["model"])
    tasks = load_tasks(args.tasks)
    if config.get("prompt_manifest"):
        prompt_path = Path(config["prompt_manifest"])
        if not prompt_path.is_absolute():
            prompt_path = args.config.parent / prompt_path
        tasks = attach_prompt_manifest(tasks, prompt_path)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        tasks = tasks[:args.limit]
    run_settings = config.get("run", {})
    output_rows = []
    for task in tasks:
        output_rows.append(run_episode(task, model, condition=run_settings["condition"],
                                       arm=run_settings["arm"], interface=run_settings["interface"],
                                       settings=run_settings))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(output_rows)} task results to {args.output}")


if __name__ == "__main__":
    main()
