import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cab.interfaces import validate_action
from cab.parsing import parse_response
from cab.prompts import attach_prompt_manifest, build_prompt
from cab.runner import run_episode
from cab.scoring import score_answer
from cab.tasks import load_tasks


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, messages):
        return next(self.responses)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_tasks(ROOT / "data" / "smoke_task.json")[0]

    def test_task_loading_and_prompt(self):
        self.assertEqual(self.task.task_id, "SMOKE-CONFIG-001")
        self.assertEqual(self.task.source_ids_for("DISRUPT"), self.task.source_ids)
        self.assertIn("Question:", build_prompt(self.task, "DISRUPT", "BASE", "SERIAL"))

    def test_exact_prompt_manifest_is_used_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.json"
            path.write_text(json.dumps([{"task_id": self.task.task_id, "interface": "SERIAL",
                                         "arm": "CONTROL", "prompt": "frozen prompt bytes\n"}]),
                            encoding="utf-8")
            task = attach_prompt_manifest([self.task], path)[0]
            self.assertEqual(build_prompt(task, "DISRUPT", "CONTROL", "SERIAL"), "frozen prompt bytes\n")

    def test_serial_requires_one_declared_source(self):
        ok, error = validate_action({"type": "read", "source_ids": ["SRC_A1C3", "SRC_C3E5"]},
                                    self.task.answer_schema, self.task.source_ids, "SERIAL")
        self.assertFalse(ok)
        self.assertEqual(error, "serial_cardinality")

    def test_parallel_accepts_multiple_declared_sources(self):
        ok, error = validate_action({"type": "read", "source_ids": ["SRC_A1C3", "SRC_C3E5"]},
                                    self.task.answer_schema, self.task.source_ids, "PARALLEL")
        self.assertTrue(ok)
        self.assertIsNone(error)

    def test_parser_protocols(self):
        self.assertEqual(parse_response('{"action":"read","source_ids":["SRC_A1C3"]}')['action'], "read")
        self.assertEqual(parse_response('```json\n{"type":"read","source_ids":[SRC_A1C3]}\n```',
                                        allow_e04_formatting=True)['source_ids'], ["SRC_A1C3"])
        with self.assertRaises(ValueError):
            parse_response('```json\n{}\n```')

    def test_scorer_uses_only_declared_aliases(self):
        self.assertTrue(score_answer(self.task, {
            "config_path": "./build/runner.yml",
            "config_sha256": self.task.gold["config_sha256"],
        }))
        self.assertFalse(score_answer(self.task, {
            "config_path": "docs/example.yml",
            "config_sha256": self.task.gold["config_sha256"],
        }))

    def test_end_to_end_smoke_and_result_serialization(self):
        serial = run_episode(self.task, ScriptedModel([
            '{"type":"read","source_ids":["SRC_A1C3"]}',
            '{"type":"read","source_ids":["SRC_C3E5"]}',
            '{"type":"final","answer":{"config_path":"build/runner.yml","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}}',
        ]), condition="DISRUPT", arm="BASE", interface="SERIAL", settings={"parser": "e04"})
        parallel = run_episode(self.task, ScriptedModel([
            '{"type":"read","source_ids":["SRC_A1C3","SRC_C3E5"]}',
            '{"type":"final","answer":{"config_path":"./build/runner.yml","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}}',
        ]), condition="DISRUPT", arm="PROCEDURAL", interface="PARALLEL", settings={"parser": "e04"})
        self.assertTrue(serial["correct"] and parallel["correct"])
        self.assertEqual(serial["tool_calls"], 2)
        self.assertEqual(parallel["tool_calls"], 2)
        self.assertEqual(parallel["trace"][0]["action"]["source_ids"], ["SRC_A1C3", "SRC_C3E5"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.jsonl"
            path.write_text(json.dumps(parallel) + "\n", encoding="utf-8")
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["correct"])


if __name__ == "__main__":
    unittest.main()
