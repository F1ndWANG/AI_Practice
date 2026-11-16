"""Verify policy provenance and enforce required slice support."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_practice.artifacts import sha256
from ai_practice.gates import gate_vision, load_policy
from ai_practice.report import render
from vlm_posttraining.fixtures import generate


class GatePolicyTests(unittest.TestCase):
    def test_coverage_fails_for_absent_and_underpowered_required_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            generate(data)
            manifest = data / "prepared/test.jsonl"
            baseline = data / "fixture-repeated_answer.jsonl"
            candidate = data / "fixture-oracle.jsonl"
            policy = root / "policy.json"
            spec = {"schema": 1, "domain": "vision", "settings": {
                "slice_by": ["task"], "min_slice_samples": 7, "required_slices": {
                    "task": {"ocr": ["answer_accuracy"], "missing_task": ["answer_accuracy"]},
                },
            }}
            policy.write_text(json.dumps(spec), encoding="utf-8")
            report = gate_vision(manifest, baseline, candidate, root / "rejected",
                                 **load_policy(policy, "vision"))
            self.assertTrue(report["overall_passed"])
            self.assertFalse(report["passed"])
            self.assertEqual([row["status"] for row in report["required_slice_checks"]],
                             ["insufficient", "missing"])
            self.assertEqual(report["policy"]["sha256"], sha256(policy))
            self.assertIn("缺少分组", render(root / "rejected"))

            spec["settings"]["required_slices"]["task"].pop("missing_task")
            spec["settings"]["min_slice_samples"] = 2
            policy.write_text(json.dumps(spec), encoding="utf-8")
            args = [sys.executable, "-m", "ai_practice.gates", "vision", str(manifest),
                    str(baseline), str(candidate), str(root / "accepted"), "--policy", str(policy)]
            run = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            mixed = subprocess.run([*args, "--max-regressions", "0"], capture_output=True, text=True)
            self.assertEqual(mixed.returncode, 2)
            self.assertIn("cannot be combined", mixed.stderr)

    def test_duplicate_policy_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text('{"schema":1,"domain":"agent","settings":{"max_regressions":0,"max_regressions":1}}')
            with self.assertRaisesRegex(ValueError, "Repeated JSON key"):
                load_policy(path, "agent")


if __name__ == "__main__":
    unittest.main()
