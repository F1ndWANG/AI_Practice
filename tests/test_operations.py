"""Cover optional model telemetry, manifest preflight and report output."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_practice.report import render
from rsi_agent.study import study
from vlm_posttraining.compare import compare
from vlm_posttraining.doctor import inspect_dataset
from vlm_posttraining.fixtures import generate
from vlm_posttraining.pipeline import read_jsonl, write_jsonl


class OperationsTests(unittest.TestCase):
    def test_agent_usage_and_fallback_report(self):
        def reply(request, timeout):
            self.assertEqual(timeout, 20)
            body = json.loads(request.data)
            user = json.loads(body["messages"][1]["content"])
            self.assertNotIn("expected", user)
            # A tool outside the registry is rejected while its reported usage is retained.
            return io.BytesIO(json.dumps({
                "choices": [{"message": {"content": '{"recipe":["unregistered_tool"]}'}}],
                "usage": {"total_tokens": 37},
            }).encode())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "agent"
            with patch("rsi_agent.runtime.urlopen", reply):
                result = study(output, per_family=1, workers=1, model="test-model")
            no_memory = result["variants"]["baseline"]
            self.assertGreater(no_memory["model_fallbacks"], 0)
            self.assertEqual(no_memory["model_requests"], no_memory["model_requests_with_usage"])
            self.assertEqual(no_memory["model_request_errors"], no_memory["model_requests"])
            self.assertEqual(no_memory["model_tokens_reported"], 37 * no_memory["model_requests"])
            self.assertIn("已报告 Tokens", render(output))

    def test_preflight_rejects_incomplete_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixtures"
            generate(output)
            source = output / "manifest.jsonl"
            report = inspect_dataset(source, 42, decode=False)
            self.assertEqual(report["unique_image_paths"], 72)
            self.assertEqual(report["splits"]["test"]["counterfactual_pairs"], 9)
            last_removed = list(read_jsonl(source))[:-1]
            write_jsonl(source, last_removed)
            with self.assertRaisesRegex(ValueError, "exactly two samples"):
                inspect_dataset(source, 42, decode=False)

    def test_reports_generation_metrics_only_when_provided(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "fixtures"
            generate(output)
            candidate = root / "measured.jsonl"
            write_jsonl(candidate, (
                {**row, "generation_seconds": 0.25, "generated_tokens": 25}
                for row in read_jsonl(output / "fixture-oracle.jsonl")
            ))
            report = root / "report"
            metrics = compare(output / "prepared/test.jsonl", {
                "base": output / "fixture-repeated_answer.jsonl", "candidate": candidate,
            }, report, rounds=10)
            self.assertIsNone(metrics["variants"]["base"]["overall"]["generation_performance"])
            speed = metrics["variants"]["candidate"]["overall"]["generation_performance"]
            self.assertEqual(speed["generated_tokens_per_second"], 100.0)
            self.assertEqual(speed["p95_seconds"], 0.25)
            self.assertIn("生成 Tokens/秒", render(report))


if __name__ == "__main__":
    unittest.main()
