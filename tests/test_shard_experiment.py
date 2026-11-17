"""Exercise worker scheduling and prove resumed runs cannot inflate benchmark speed."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_posttraining.blind import build
from vlm_posttraining.fixtures import generate
from vlm_posttraining.shard_experiment import pair, run
from vlm_posttraining.pipeline import read_jsonl, write_jsonl
from vlm_posttraining.shards import balanced_assignments, profile_weights, run_shard


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
                profile = baseline / "predictions.jsonl"
                balanced = root / "balanced"
                result = run(dispatch, balanced, ["1", "3"], model="fixture", profile=profile)
                self.assertEqual(result["assigned_samples"], result["samples"])
                strategy = pair(parallel, balanced)
                self.assertEqual(strategy["changed_completions"], 0)
                self.assertEqual(strategy["candidate_strategy"], "profile")
                self.assertEqual(strategy["baseline_strategy"], "round_robin")
                self.assertEqual(strategy["candidate_slowest_worker_seconds"], 0.01)
                rows = list(read_jsonl(profile))
                rows[0]["generation_seconds"] *= 2
                changed_profile = root / "changed.jsonl"
                write_jsonl(changed_profile, rows)
                with self.assertRaisesRegex(ValueError, "changed"):
                    run(dispatch, balanced, ["1", "3"], model="fixture", resume=True,
                        profile=changed_profile)
                comparison = pair(baseline, parallel)
                self.assertGreater(comparison["speedup"], 0)
                self.assertEqual(comparison["changed_completions"], 0)
                with self.assertRaisesRegex(ValueError, "without reused shards"):
                    pair(baseline, retries)

    def test_profile_validation_and_deterministic_balance(self):
        weights = {"a": 12, "b": 2, "c": 10, "d": 1}
        ids = list(weights)
        assignments = balanced_assignments(ids, weights, 2)
        self.assertEqual(assignments, [["a", "d"], ["c", "b"]])
        self.assertEqual(max(sum(weights[key] for key in group) for group in assignments), 13)
        self.assertEqual(balanced_assignments(ids, dict.fromkeys(ids, 1), 2), [ids[::2], ids[1::2]])
        with tempfile.TemporaryDirectory() as folder:
            profile = Path(folder) / "pilot.jsonl"
            rows = [{"id": key, "generation_seconds": value} for key, value in weights.items()]
            write_jsonl(profile, rows)
            self.assertEqual(profile_weights(profile, ids), weights)
            for bad in (rows[:-1], rows + rows[:1], rows[1:] + [{"id": "a", "generation_seconds": float("nan")}],
                        rows[1:] + [{"id": "a", "generation_seconds": True}]):
                with self.subTest(bad=bad):
                    write_jsonl(profile, bad)
                    with self.assertRaises(ValueError):
                        profile_weights(profile, ids)


if __name__ == "__main__":
    unittest.main()
