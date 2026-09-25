import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cab.analysis import analyze, exact_p, load_rows


class AnalysisReleaseTests(unittest.TestCase):
    def test_exact_paired_test(self):
        self.assertEqual(exact_p(13, 0), 0.000244140625)
        self.assertEqual(exact_p(0, 0), 1.0)

    def test_compact_outcome_schema_and_row_total(self):
        rows = load_rows()
        self.assertEqual(len(rows), 44832)
        required = {"experiment", "model", "interface", "condition", "arm", "task_id", "repository_id", "correct"}
        self.assertTrue(required.issubset(rows[0]))
        self.assertTrue(all(row["correct"] in {"0", "1"} for row in rows))

    def test_numerical_release_checks(self):
        report = analyze(ROOT / "results-test")
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(any(item["status"] == "FAIL" for item in report["checks"]))
        self.assertFalse(any(item["status"] == "WARN" for item in report["checks"]))
        self.assertTrue(report["reproduction_limitations"])
        e04 = next(item for item in report["checks"] if item["name"] == "180-repository submitted effects, intervals, and Holm values")
        self.assertEqual(e04["status"], "PASS")
        e05 = next(item for item in report["checks"] if item["name"] == "six-pair submitted value and recomputed Monte Carlo estimate")
        self.assertEqual(e05["submitted_exceedances"], 6)
        self.assertEqual(e05["recomputed_exceedances"], 5)
        self.assertNotEqual(e05["submitted_p"], e05["recomputed_p"])

    def test_analysis_works_outside_repository_cwd(self):
        from cab.analysis import ROOT as module_root
        self.assertEqual(module_root, ROOT)
        self.assertTrue((module_root / "data" / "results.csv").is_file())


if __name__ == "__main__":
    unittest.main()
