"""Exercise model workers and validate shard coverage, provenance and order."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_posttraining.blind import build
from vlm_posttraining.fixtures import generate
from vlm_posttraining.pipeline import read_jsonl, score_predictions
from vlm_posttraining.shards import merge, prepare, run_shard


class PredictionShardTests(unittest.TestCase):
    def test_merge_checks_coverage_configuration_and_original_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixtures = root / "data"
            generate(fixtures)
            reference = fixtures / "prepared/test.jsonl"
            blind = root / "blind.jsonl"
            build(reference, blind, root / "binding.json")
            shards = root / "shards"
            output = root / "predictions.jsonl"
            self.assertEqual(prepare(blind, shards, 3), {"shards": 3, "samples": 18})
            response = {"completion": '{"answer":"42","evidence":[]}', "generation_seconds": 0.01,
                        "generated_tokens": 3}
            with patch("vlm_posttraining.predict.VisualModel") as model:
                model.return_value.ask.return_value = response
                run_shard(shards, 2, "fixture")
                with self.assertRaisesRegex(ValueError, "Missing completed shard 0"):
                    merge(shards, output)
                self.assertFalse(output.exists())
                run_shard(shards, 0, "fixture")
                run_shard(shards, 1, "fixture")
                run_shard(shards, 1, "fixture")
                self.assertEqual(model.return_value.ask.call_count, 18)
            receipt = shards / "receipt-1.json"
            original = receipt.read_bytes()
            invalid = json.loads(original)
            invalid["model_settings"]["model"] = "other"
            receipt.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "Model settings differ"):
                merge(shards, output)
            receipt.write_bytes(original)
            self.assertEqual(merge(shards, output)["samples"], 18)
            self.assertEqual([row["id"] for row in read_jsonl(output)],
                             [row["id"] for row in read_jsonl(reference)])
            self.assertEqual(len(score_predictions(reference, output)), 18)
            prediction = shards / "predictions-0.jsonl"
            prediction.write_bytes(prediction.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Invalid receipt or predictions"):
                merge(shards, root / "rejected.jsonl")


if __name__ == "__main__":
    unittest.main()
