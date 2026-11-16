"""Exercise same-sample comparisons and release exit statuses."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_practice.gates import evaluate_pairs, gate_agent
from ai_practice.report import render
from rsi_agent.study import study
from vlm_posttraining.fixtures import generate


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


if __name__ == "__main__":
    unittest.main()
