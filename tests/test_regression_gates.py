"""Exercise same-sample comparisons and release exit statuses."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_practice.gates import evaluate_pairs, gate_agent, gate_vision
from ai_practice.report import render
from rsi_agent.study import study
from vlm_posttraining.fixtures import generate
from vlm_posttraining.pipeline import read_jsonl, write_jsonl


class RegressionGateTests(unittest.TestCase):
    def test_agent_expected_improvement_passes_and_other_task_snapshots_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "agent"
            study(base, per_family=1)
            report = gate_agent(base, base, root / "agent_gate", max_regressions=0)
            self.assertTrue(report["passed"])
            self.assertEqual(report["checks"]["tool_calls"]["delta"], -2.0)
            other = root / "other"
            study(other, seed=43, per_family=1)
            with self.assertRaisesRegex(ValueError, "identical task snapshots"):
                gate_agent(base, other, root / "rejected")

    def test_vision_reversal_returns_nonzero_and_keeps_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data"
            generate(dataset)
            ref = dataset / "prepared/test.jsonl"
            oracle = dataset / "fixture-oracle.jsonl"
            repeated = dataset / "fixture-repeated_answer.jsonl"
            output = root / "gate"
            proc = subprocess.run([sys.executable, "-m", "ai_practice.gates", "vision",
                                   str(ref), str(oracle), str(repeated), str(output),
                                   "--max-grounded-drop", "0"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(len((output / "changes.jsonl").read_text().splitlines()), 9)
            self.assertIn("结果  失败", render(output))

    def test_strict_gate_detects_sample_regressions_hidden_by_unchanged_mean(self):
        rows = [{"id": "x", "accuracy": 1}, {"id": "y", "accuracy": 0}]
        other = [{"id": "y", "accuracy": 1}, {"id": "x", "accuracy": 0}]
        result, changes = evaluate_pairs(rows, other, {"accuracy": (lambda row: row["accuracy"], 0.0)},
                                         max_regressions=0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"]["accuracy"]["delta"], 0)
        self.assertEqual(len(changes), 2)


    def test_task_slices_flag_regression_cancelled_by_another_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            generate(data)
            manifest = data / "prepared/test.jsonl"
            samples = list(read_jsonl(manifest))

            def predictions(wrong_task):
                return [
                    {"id": row["id"], "completion": json.dumps({
                        "answer": "wrong" if row["task"] == wrong_task else row["answer"],
                        "evidence": row["evidence"],
                    })}
                    for row in reversed(samples)
                ]

            baseline, candidate = root / "base.jsonl", root / "candidate.jsonl"
            write_jsonl(baseline, predictions("ocr"))
            write_jsonl(candidate, predictions("docvqa"))
            output = root / "gate"
            result = gate_vision(manifest, baseline, candidate, output, slice_by=["task"], min_slice_samples=2)
            self.assertTrue(result["overall_passed"])
            self.assertFalse(result["passed"])
            groups = {row["value"]: row for row in result["by_slice"]["task"]["slices"]}
            self.assertEqual(groups["docvqa"]["checks"]["answer_accuracy"]["regressed"], 6)
            self.assertEqual(groups["ocr"]["status"], "passed")
            changes = list(read_jsonl(output / "changes.jsonl"))
            self.assertEqual({row["slices"]["task"] for row in changes}, {"docvqa", "ocr"})
            self.assertIn("分组验收", render(output))
            unavailable = gate_vision(manifest, baseline, candidate, root / "no_labels", slice_by=["dataset"])
            self.assertFalse(unavailable["passed"])
            self.assertEqual(unavailable["by_slice"]["dataset"]["missing_labels"], 18)


if __name__ == "__main__":
    unittest.main()
