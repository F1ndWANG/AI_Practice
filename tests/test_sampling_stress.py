"""Regression tests for sampling receipts and false-positive memory validation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_practice.artifacts import sha256
from ai_practice.report import render
from rsi_agent.memory_stress import stress
from rsi_agent.study import study
from vlm_posttraining.blind import build
from vlm_posttraining.fixtures import generate
from vlm_posttraining.pipeline import read_jsonl, write_jsonl
from vlm_posttraining.reward_audit import audit
from vlm_posttraining.sample_candidates import collect


class SamplingStressTests(unittest.TestCase):
    def test_sampling_uses_blind_prompts_and_binds_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            generate(data)
            refs = data / "prepared/validation.jsonl"
            dispatch = root / "inference.jsonl"
            build(refs, dispatch, root / "binding.json")
            calls = []

            def sample(item, **options):
                self.assertNotIn("answer", item)
                self.assertNotIn("evidence", item)
                calls.append(options["seed"])
                return [{"completion": '{"answer":"test","evidence":[]}', "generated_tokens": 3,
                         "generation_seconds": 0.1}] * options["count"]

            output = root / "candidate"
            with patch("vlm_posttraining.sample_candidates.VisualModel") as visual:
                visual.return_value.sample.side_effect = sample
                metadata = collect(dispatch, output, count=3, limit=2)
                self.assertNotEqual(*calls)
                self.assertEqual(metadata["groups"], 2)
            predictions = output / "candidates.jsonl"
            receipt = output / "sampling.json"
            result = audit(refs, root / "audit", candidates=predictions, sampling_receipt=receipt)
            self.assertTrue(result["sampling_receipt_verified"])
            self.assertEqual(result["overall"]["zero_signal_groups"], 2)
            rows = list(read_jsonl(predictions))
            rows[0]["completions"][0] = "tampered"
            write_jsonl(predictions, rows)
            with self.assertRaisesRegex(ValueError, "Sampling receipt"):
                audit(refs, root / "invalid", candidates=predictions, sampling_receipt=receipt)

    def test_wrong_memory_can_be_detected_without_mutating_the_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "study"
            study(source, per_family=1)
            before = sha256(source / "experience.sqlite")
            output = root / "stress"
            result = stress(source, output)
            variants = result["variants"]
            self.assertLess(variants["polluted"]["success_rate"], variants["clean"]["success_rate"])
            self.assertEqual(variants["guarded"]["success_rate"], variants["clean"]["success_rate"])
            self.assertEqual(result["calibration_rejected"], 2)
            self.assertEqual(sha256(source / "experience.sqlite"), before)
            self.assertIn("记忆压力测试", render(output))


if __name__ == "__main__":
    unittest.main()
