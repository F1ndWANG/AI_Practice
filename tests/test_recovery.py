"""Exercise recoverable endpoint failures and interrupted sampling."""

import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from rsi_agent.runtime import Task, run_task
from rsi_agent.study import summarize
from vlm_posttraining.blind import build
from vlm_posttraining.fixtures import generate
from vlm_posttraining.sample_candidates import collect


class RecoveryTests(unittest.TestCase):
    def test_transient_failure_retries_and_records_actual_http_cost(self):
        headers = Message()
        headers["Retry-After"] = "0"
        error = HTTPError("https://example.com", 429, "busy", headers, io.BytesIO())
        success = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": '{"recipe":["sum"]}'}}],
        }).encode())
        with patch("rsi_agent.runtime.urlopen", side_effect=[error, success]) as endpoint:
            result = run_task(Task("sum", "8\n11\n", "19\n"), [], "brs", 1, model="fixture")
        self.assertTrue(result["success"])
        self.assertEqual(endpoint.call_count, 2)
        self.assertEqual(result["planner_calls"][0]["decision"], "model")
        stats = summarize([result])
        self.assertEqual(stats["model_http_retries"], 1)
        self.assertEqual(stats["model_http_errors"], 1)
        self.assertEqual(stats["model_request_errors"], 0)

    def test_permanent_denial_falls_back_without_retry(self):
        error = HTTPError("https://example.com", 401, "unauthorized", None, io.BytesIO())
        with patch("rsi_agent.runtime.urlopen", side_effect=error) as endpoint:
            result = run_task(Task("sum", "8\n11\n", "19\n"), [], "brs", 1, model="fixture")
        self.assertEqual(endpoint.call_count, 1)
        self.assertEqual(result["planner_calls"][0]["decision"], "fallback")

    def test_sampling_resumes_after_the_last_complete_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            generate(data)
            dispatch = root / "dispatch.jsonl"
            build(data / "prepared/validation.jsonl", dispatch, root / "binding.json")
            output = root / "sample"
            candidate = [{"completion": "text", "generated_tokens": 1, "generation_seconds": 0.1}] * 2
            with patch("vlm_posttraining.sample_candidates.VisualModel") as model:
                model.return_value.sample.side_effect = [candidate, RuntimeError("device stopped")]
                with self.assertRaisesRegex(RuntimeError, "device stopped"):
                    collect(dispatch, output, count=2, limit=2)
                self.assertFalse(output.exists())
                self.assertTrue((root / ".sample.partial/groups/00000000.json").is_file())
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    collect(dispatch, output, count=2, limit=2, seed=7, resume=True)
                model.return_value.sample.reset_mock()
                model.return_value.sample.side_effect = [candidate]
                receipt = collect(dispatch, output, count=2, limit=2, resume=True)
                self.assertEqual(model.return_value.sample.call_count, 1)
                self.assertEqual(receipt["groups"], 2)
                self.assertEqual(collect(dispatch, output, count=2, limit=2, resume=True), receipt)
                self.assertEqual(model.return_value.sample.call_count, 1)


if __name__ == "__main__":
    unittest.main()
