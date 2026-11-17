"""Check repeated scheduling, paired statistics and optional scoring."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_posttraining.blind import build
from vlm_posttraining.fixtures import generate
from vlm_posttraining.shard_matrix import execute
from vlm_posttraining.shards import run_shard


class ShardMatrixTests(unittest.TestCase):
    def test_matrix_rotates_order_and_pairs_fresh_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "fixtures"
            generate(data)
            reference = data / "prepared/test.jsonl"
            manifest = root / "blind.jsonl"
            build(reference, manifest, root / "binding.json")

            def worker(shards, index, gpu, model, adapter):
                run_shard(shards, index, model, adapter)
                return {"index": index, "gpu": gpu, "exit_code": 0, "seconds": 0.01}

            with patch("vlm_posttraining.shard_experiment.worker", worker), patch(
                "vlm_posttraining.predict.VisualModel"
            ) as visual:
                visual.return_value.ask.return_value = {
                    "completion": '{"answer":"wrong","evidence":[]}',
                    "generation_seconds": 0.01, "generated_tokens": 3,
                }
                output = root / "matrix"
                result = execute(manifest, output, [["0"], ["0", "1"]], repeats=2,
                                 model="fixture", reference=reference)
                pilot = output / "repeat-1/group-0/predictions.jsonl"
                comparison = root / "strategies"
                strategies = execute(manifest, comparison, [["0", "1"]], repeats=2,
                                     model="fixture", reference=reference, profile=pilot)
            self.assertTrue(result["passed"])
            rounds = json.loads((output / "rounds.json").read_text())
            self.assertEqual([row["order"] for row in rounds], [[0, 1], [1, 0]])
            parallel = result["groups"][1]
            self.assertEqual(parallel["paired_speedup"]["runs"], 2)
            self.assertEqual(parallel["changed_completions"]["median"], 0)
            self.assertEqual(parallel["answer_accuracy"]["median"], 0)
            self.assertIn("配对加速比", (output / "report.md").read_text())
            self.assertEqual(visual.return_value.ask.call_count, 18 * 8)
            self.assertTrue(strategies["passed"])
            self.assertTrue(strategies["profile_sha256"])
            strategy_rounds = json.loads((comparison / "rounds.json").read_text())
            self.assertEqual([row["order"] for row in strategy_rounds], [[0, 1], [1, 0]])
            baseline, candidate = strategies["groups"]
            self.assertEqual(baseline["gpus"], candidate["gpus"])
            self.assertEqual([baseline["strategy"], candidate["strategy"]], ["round_robin", "profile"])
            self.assertEqual(candidate["paired_speedup"]["runs"], 2)
            self.assertEqual(candidate["paired_slowest_worker_delta_seconds"]["median"], 0)
            self.assertEqual(candidate["changed_completions"]["median"], 0)
            sample = json.loads((comparison / "profile.jsonl").read_text().splitlines()[0])
            self.assertEqual(set(sample), {"id", "generation_seconds"})


if __name__ == "__main__":
    unittest.main()
