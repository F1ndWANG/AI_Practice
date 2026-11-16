"""Regression checks for frozen memory, split isolation and scoring."""

import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rsi_agent.runtime import Memory, run
from vlm_posttraining.demo import smoke
from vlm_posttraining.pipeline import prepare, read_jsonl, score, write_jsonl
from vlm_posttraining.train import reward


class AgentTests(unittest.TestCase):
    def test_frozen_memory_reduces_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            baseline = run("test", None, "brs", 6)
            with closing(Memory(path, frozen=False)) as memory:
                training = run("train", memory, "brs", 6)
            before = path.read_bytes()
            with closing(Memory(path, frozen=True)) as memory:
                result = run("test", memory, "brs", 6)
            self.assertEqual(training["success_rate"], 1)
            self.assertEqual(result["success_rate"], baseline["success_rate"])
            self.assertLess(result["tool_calls"], baseline["tool_calls"])
            self.assertEqual(path.read_bytes(), before)

    def test_budget_limits_tool_calls(self):
        result = run("test", None, "brs", 1)
        self.assertTrue(all(task["tool_calls"] <= 1 for task in result["tasks"]))
        self.assertEqual(result["success_rate"], 0)


class VLMTests(unittest.TestCase):
    def test_smoke_accepts_only_grounded_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result = smoke(Path(directory))
        self.assertEqual(result["prepare"]["duplicates_removed"], 1)
        self.assertEqual(result["evaluate"]["overall"]["accuracy"], 1)
        self.assertEqual(result["evaluate"]["overall"]["unsupported_evidence_rate_among_valid"], 0.5)

    def test_duplicate_image_cannot_cross_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            smoke(path)
            manifest = path / "manifest.jsonl"
            rows = list(read_jsonl(manifest))
            rows[-1]["split"] = "test"
            rows[-1]["source_group"] = "different-group"
            write_jsonl(manifest, rows)
            with self.assertRaisesRegex(ValueError, "leakage"):
                prepare(manifest, path / "rejected", 42)

    def test_reward_handles_conversational_results(self):
        valid = '[{"role":"assistant"}]'
        self.assertEqual(score(valid, "red", ["red square"])["format"], 0)
        completions = [[{"content": '{"answer":"RED","evidence":["red square"]}'}], valid]
        self.assertEqual(reward(completions, ["red"] * 2, [["red square"]] * 2), [1.7, 0.0])


if __name__ == "__main__":
    unittest.main()
