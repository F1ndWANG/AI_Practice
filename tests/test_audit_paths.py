"""Verify replay against tampered state and isolate blind inference labels."""

import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rsi_agent.replay import replay
from rsi_agent.runtime import Memory
from rsi_agent.study import study
from vlm_posttraining.blind import PROMPT_FIELDS, build, verify
from vlm_posttraining.fixtures import generate
from vlm_posttraining.pipeline import read_jsonl, write_jsonl


class AuditPathTests(unittest.TestCase):
    def test_replay_checks_allowed_tools_and_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "agent"
            study(root, per_family=1)
            report = replay(root)
            self.assertEqual(report["memory_snapshots_verified"], 2)
            self.assertEqual(report["replayed_attempts"]["baseline"], 6)
            trace = root / "baseline.jsonl"
            original = trace.read_bytes()
            raw = [json.loads(line) for line in original.splitlines()]
            raw[0]["attempts"][0]["recipe"] = ["unregistered"]
            write_jsonl(trace, raw)
            with self.assertRaisesRegex(ValueError, "Invalid recipe"):
                replay(root)
            trace.write_bytes(original)
            with closing(Memory(root / "experience.sqlite", frozen=False)) as memory:
                memory.update("unknown", ("copy",), True)
            with self.assertRaisesRegex(ValueError, "Memory does not match"):
                replay(root)

    def test_blind_manifest_detects_changes_and_never_contains_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixtures = root / "fixtures"
            generate(fixtures)
            reference = fixtures / "prepared/test.jsonl"
            dispatch = root / "predict.jsonl"
            receipt = root / "receipt.json"
            build(reference, dispatch, receipt)
            self.assertEqual(verify(reference, dispatch, receipt)["verified_samples"], 18)
            rows = list(read_jsonl(dispatch))
            self.assertTrue(all(set(row) == set(PROMPT_FIELDS) for row in rows))
            rows[0]["answer"] = "inserted"
            write_jsonl(dispatch, rows)
            with self.assertRaisesRegex(ValueError, "changed since dispatch"):
                verify(reference, dispatch, receipt)
            build(reference, dispatch, receipt)
            Path(rows[0]["image"]).write_bytes(b"changed image")
            with self.assertRaisesRegex(ValueError, "image changed"):
                verify(reference, dispatch, receipt)


if __name__ == "__main__":
    unittest.main()
