import csv
import hashlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cab.prompts import attach_prompt_manifest, build_prompt
from cab.tasks import load_tasks


class PaperManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt_path = ROOT / "data" / "prompt_manifest.json"
        cls.prompts = json.loads(cls.prompt_path.read_text(encoding="utf-8"))
        cls.config = json.loads((ROOT / "configs" / "paper_experiments.json").read_text(encoding="utf-8"))
        cls.coverage = json.loads((ROOT / "configs" / "prompt_coverage.json").read_text(encoding="utf-8"))
        smoke = load_tasks(ROOT / "data" / "smoke_task.json")[0]
        task_ids = {
            next(b["task_id"] for b in cls.prompts["base_prompts"] if "E01" in b["prompt_hashes"]),
            next(b["task_id"] for b in cls.prompts["base_prompts"] if "E04" in b["prompt_hashes"]),
            next(b["task_id"] for b in cls.prompts["base_prompts"] if "E21" in b["prompt_hashes"]),
        }
        tasks = [replace(smoke, task_id=task_id) for task_id in sorted(task_ids)]
        cls.attached = {
            task.task_id: task
            for task in attach_prompt_manifest(tasks, cls.prompt_path)
        }

    def test_exact_e01_e04_e05_prompts_load_unchanged(self):
        bases = {(b["task_id"], b["interface"]): b for b in self.prompts["base_prompts"]}
        checks = [
            ("E01", "U_01_A", "SERIAL", "NEUTRAL"),
            ("E01", "U_01_A", "PARALLEL", "PROCEDURAL"),
            ("E04", next(b["task_id"] for b in self.prompts["base_prompts"] if "E04" in b["prompt_hashes"]), "SERIAL", "BASE"),
            ("E04", next(b["task_id"] for b in self.prompts["base_prompts"] if "E04" in b["prompt_hashes"]), "SERIAL", "N_SHORT"),
            ("E04", next(b["task_id"] for b in self.prompts["base_prompts"] if "E04" in b["prompt_hashes"]), "PARALLEL", "P_MEDIUM"),
            ("E05", "U_01_A", "SERIAL", "P_LONG_2"),
        ]
        for evaluation, task_id, interface, arm in checks:
            with self.subTest(evaluation=evaluation, task_id=task_id, interface=interface, arm=arm):
                base = bases[(task_id, interface)]
                suffix = self.prompts["arm_definitions"][evaluation][arm]["append_text"]
                expected = base["text"] + suffix
                actual = build_prompt(self.attached[task_id], "DISRUPT", arm, interface)
                expected_hash = base["prompt_hashes"][evaluation][arm]
                self.assertEqual(actual.encode("utf-8"), expected.encode("utf-8"))
                self.assertEqual(hashlib.sha256(actual.encode("utf-8")).hexdigest(), expected_hash)

    def test_six_e05_matched_pairs_are_complete(self):
        self.assertEqual(
            {pair["prompt_pair_id"] for pair in self.prompts["prompt_pairs"]},
            {"S1", "S2", "M1", "M3", "L1", "L2"},
        )
        definitions = self.prompts["arm_definitions"]["E05"]
        for pair in self.prompts["prompt_pairs"]:
            neutral = definitions[pair["neutral_arm"]]
            procedural = definitions[pair["procedural_arm"]]
            self.assertEqual(neutral["prompt_pair_id"], pair["prompt_pair_id"])
            self.assertEqual(procedural["prompt_pair_id"], pair["prompt_pair_id"])
            self.assertEqual(neutral["content_class"], "NEUTRAL")
            self.assertEqual(procedural["content_class"], "PROCEDURAL")

    def test_e04_short_medium_long_variants_are_present(self):
        expected = {
            "BASE", "N_SHORT", "P_SHORT", "N_MEDIUM", "P_MEDIUM", "N_LONG", "P_LONG"
        }
        self.assertTrue(expected.issubset(self.prompts["arm_definitions"]["E04"]))
        for base in self.prompts["base_prompts"]:
            if "E04" in base["prompt_hashes"]:
                self.assertTrue(expected.issubset(base["prompt_hashes"]["E04"]))

    def test_config_prompt_references_resolve(self):
        self.assertTrue((ROOT / self.config["prompt_manifest"]).is_file())
        local_config = json.loads((ROOT / "configs" / "local_openai.json").read_text(encoding="utf-8"))
        self.assertTrue((ROOT / "configs" / local_config["prompt_manifest"]).resolve().is_file())
        by_evaluation = {
            evaluation["evaluation_id"]: evaluation
            for evaluation in self.config["evaluations"]
        }
        for evaluation in ("same_run_prompt_comparison", "six_pair_prompt_library", "large_repository_evaluation"):
            ref = by_evaluation[evaluation]["prompt_reference"]
            self.assertEqual(ref["manifest"], "data/prompt_manifest.json")
            self.assertTrue(any(ref["evaluation"] in b["prompt_hashes"] for b in self.prompts["base_prompts"]))
        fragment_ids = {item["fragment_id"] for item in self.prompts["exact_fragments"]}
        required_source = by_evaluation["required_source_intervention"]["prompt_reference"]
        self.assertIn(required_source["fragment_id"], fragment_ids)
        self.assertTrue((ROOT / self.config["incomplete_experiment_statuses"]).is_file())
        self.assertTrue((ROOT / self.config["prompt_coverage"]).is_file())

    def test_prompt_coverage_is_truthful_and_resolves(self):
        self.assertEqual(self.coverage["prompt_manifest"], "data/prompt_manifest.json")
        self.assertTrue((ROOT / self.coverage["prompt_manifest"]).is_file())
        by_evaluation = {entry["evaluation"]: entry for entry in self.coverage["evaluations"]}
        self.assertEqual(
            {key: by_evaluation[key]["status"] for key in by_evaluation},
            {
                "E01": "EXACT_RELEASED",
                "E04": "EXACT_RELEASED",
                "E05": "EXACT_RELEASED",
                "E21": "EXACT_RELEASED",
                "required_source_intervention": "EXACT_FRAGMENT_RELEASED",
                "source_information_intervention": "PORTABLE_RECONSTRUCTION",
                "other_legacy_comparisons": "NOT_AVAILABLE",
            },
        )
        for evaluation in ("E01", "E04", "E05", "E21"):
            self.assertEqual(by_evaluation[evaluation]["records"], self.prompts["prompt_record_counts"][evaluation])
        self.assertEqual(set(by_evaluation["E05"]["pairs"]), {"S1", "S2", "M1", "M3", "L1", "L2"})
        fragment_ids = {item["fragment_id"] for item in self.prompts["exact_fragments"]}
        required = by_evaluation["required_source_intervention"]
        self.assertIn(required["fragment_id"], fragment_ids)
        self.assertFalse(required["complete_task_prompts_available"])
        source_info = by_evaluation["source_information_intervention"]
        self.assertFalse(source_info["complete_task_prompts_available"])
        self.assertEqual(by_evaluation["E21"]["experiment_status"], "INVALID_FOR_PARALLEL_COMPARISON")
        with (ROOT / "data" / "additional_checks.csv").open(encoding="utf-8", newline="") as handle:
            checks = {row["evaluation"]: row["status"] for row in csv.DictReader(handle)}
        self.assertEqual(checks["Held-out repository comparison (E21)"], "INVALID_FOR_PARALLEL_COMPARISON")


if __name__ == "__main__":
    unittest.main()
