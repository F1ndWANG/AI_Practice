"""Exercise worker scheduling and prove resumed runs cannot inflate benchmark speed."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_posttraining.blind import build
from vlm_posttraining.fixtures import generate
from vlm_posttraining.shard_experiment import pair, run
from vlm_posttraining.shards import run_shard


class ShardExperimentTests(unittest.TestCase):
    def test_failed_worker_is_retried_and_full_runs_are_comparable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            generate(data)
            dispatch = root / "blind.jsonl"
            build(data / "prepared/test.jsonl", dispatch, root / "binding.json")
            fail = {1}
            calls = []

            def worker(path, index, gpu, model, adapter):
                calls.append((index, gpu))
                if index in fail:
                    return {"index": index, "gpu": gpu, "exit_code": 1, "seconds": 0.01}
                run_shard(path, index, model, adapter)
                return {"index": index, "gpu": gpu, "exit_code": 0, "seconds": 0.01}

            with patch("vlm_posttraining.shard_experiment.worker", worker), patch(
                "vlm_posttraining.predict.VisualModel"
            ) as model:
                model.return_value.ask.return_value = {
                    "completion": "fixture", "generation_seconds": 0.01, "generated_tokens": 1,
                }
                retries = root / "retry"
                result = run(dispatch, retries, ["1", "3"], model="fixture")
                self.assertEqual(result["failed_shards"], [1])
                fail.clear()
                calls.clear()
                result = run(dispatch, retries, ["1", "3"], model="fixture", resume=True)
                self.assertEqual(calls, [(1, "3")])
                self.assertTrue(result["passed"])
                self.assertIsNone(result["samples_per_second"])

                baseline, parallel = root / "single", root / "parallel"
                run(dispatch, baseline, ["1"], model="fixture")
                run(dispatch, parallel, ["1", "3"], model="fixture")
                comparison = pair(baseline, parallel)
                self.assertGreater(comparison["speedup"], 0)
                self.assertEqual(comparison["changed_completions"], 0)
                with self.assertRaisesRegex(ValueError, "without reused shards"):
                    pair(baseline, retries)


if __name__ == "__main__":
    unittest.main()
