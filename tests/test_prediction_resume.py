"""Verify atomic batch output and sample-level prediction recovery."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_posttraining.blind import build
from vlm_posttraining.fixtures import generate
from vlm_posttraining.pipeline import read_jsonl, score_predictions, write_jsonl
from vlm_posttraining.predict import predict


class PredictionResumeTests(unittest.TestCase):
    def test_prediction_resumes_without_repeating_completed_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "fixtures"
            generate(reference)
            blind = root / "blind.jsonl"
            build(reference / "prepared/test.jsonl", blind, root / "binding.json")
            rows = list(read_jsonl(blind))[:3]
            write_jsonl(blind, rows)
            output = root / "predictions.jsonl"
            response = {"completion": '{"answer":"42","evidence":[]}', "generated_tokens": 3,
                        "generation_seconds": 0.1}
            with patch("vlm_posttraining.predict.VisualModel") as model:
                ask = model.return_value.ask
                ask.side_effect = [response, RuntimeError("inference interrupted")]
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    predict(blind, output, "fixture", resume=True)
                self.assertFalse(output.exists())
                ask.reset_mock()
                ask.side_effect = [response, response]
                predict(blind, output, "fixture", resume=True)
                self.assertEqual(ask.call_count, 2)
                self.assertEqual([item["id"] for item in read_jsonl(output)], [item["id"] for item in rows])
                subset = root / "reference.jsonl"
                write_jsonl(subset, list(read_jsonl(reference / "prepared/test.jsonl"))[:3])
                self.assertEqual(len(score_predictions(subset, output)), 3)
                predict(blind, output, "fixture", resume=True)
                self.assertEqual(ask.call_count, 2)
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    predict(blind, output, "different-model", resume=True)
                self.assertEqual(ask.call_count, 2)


if __name__ == "__main__":
    unittest.main()
