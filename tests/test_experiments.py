"""Regression tests for the new paired evaluation and exploration protocols."""

import tempfile
import unittest
from pathlib import Path

from rsi_agent.study import study
from vlm_posttraining.compare import compare
from vlm_posttraining.fixtures import generate
from vlm_posttraining.pipeline import read_jsonl, write_jsonl


class ExperimentTests(unittest.TestCase):
    def test_agent_ablation_uses_matched_tasks_and_improves_call_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            result = study(output, per_family=2, workers=2)
            variants = result["variants"]
            self.assertEqual([variants[name]["tool_calls"] for name in
                              ("baseline", "brs", "brs_drs")], [14, 8, 6])
            self.assertTrue(all(item["success_rate"] == 1.0 for item in variants.values()))
            baseline_ids = [row["id"] for row in read_jsonl(output / "baseline.jsonl")]
            self.assertEqual(baseline_ids, [row["id"] for row in read_jsonl(output / "brs_drs.jsonl")])
            self.assertEqual([row["task"] for row in read_jsonl(output / "deep_exploration.jsonl")
                              if row["stage"] == "practice"], ["words"])

    def test_paired_fixtures_measure_image_sensitivity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = root / "fixtures"
            prepared = generate(fixtures)
            self.assertEqual(prepared["input"], 72)
            baseline = fixtures / "fixture-repeated_answer.jsonl"
            oracle = fixtures / "fixture-oracle.jsonl"
            test = fixtures / "prepared/test.jsonl"
            report = compare(test, {"base": baseline, "candidate": oracle}, root / "report", rounds=50)
            self.assertEqual(report["paired"]["candidate"]["delta_grounded_accuracy"], 0.5)
            self.assertEqual(report["paired"]["candidate"]["groups"], 9)
            self.assertEqual(report["variants"]["base"]["counterfactual"]["both_answers_correct"], 0)
            self.assertEqual(report["variants"]["candidate"]["counterfactual"]["both_answers_correct"], 1)
            self.assertEqual(len(list(read_jsonl(root / "report/review.jsonl"))), 9)
            broken = root / "missing.jsonl"
            write_jsonl(broken, list(read_jsonl(oracle))[:-1])
            with self.assertRaisesRegex(ValueError, "match"):
                compare(test, {"base": baseline, "candidate": broken}, root / "rejected")
            self.assertFalse((root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
