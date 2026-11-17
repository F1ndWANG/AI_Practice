"""Check equivalent protocols, detected regressions and incompatible runs."""

import tempfile
import unittest
from pathlib import Path

from ai_practice.compare_releases import compare, markdown
from ai_practice.gates import gate_agent, gate_vision
from ai_practice.release import build
from rsi_agent.study import study
from vlm_posttraining.fixtures import generate
from vlm_posttraining.pipeline import read_jsonl, write_jsonl


class ReleaseComparisonTests(unittest.TestCase):
    def test_release_comparison_uses_protocol_identity_and_detects_new_regressions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            agent = root / "agent"
            study(agent, per_family=1)
            gate_agent(agent, agent, root / "a1", slice_by=["task"])
            trace = agent / "baseline.jsonl"
            rows = list(read_jsonl(trace))
            for row in rows:
                row["attempts"][0]["calls"][0]["seconds"] += 1.0
            write_jsonl(trace, rows)
            gate_agent(agent, agent, root / "a2", slice_by=["task"])

            data = root / "data"
            generate(data)
            manifest = data / "prepared/test.jsonl"
            base = data / "fixture-repeated_answer.jsonl"
            oracle = data / "fixture-oracle.jsonl"
            gate_vision(manifest, base, oracle, root / "v1", slice_by=["task"])
            relocated = root / "relocated.jsonl"
            rows = [{**row, "image": "/different/root/" + Path(row["image"]).name}
                    for row in read_jsonl(manifest)]
            write_jsonl(relocated, reversed(rows))
            baseline_reordered = root / "baseline.jsonl"
            write_jsonl(baseline_reordered, reversed(list(read_jsonl(base))))
            gate_vision(relocated, baseline_reordered, base, root / "v2", slice_by=["task"])
            build(root / "a1", root / "v1", root / "release1", revision="1" * 40)
            build(root / "a2", root / "v2", root / "release2", revision="2" * 40)
            result = compare(root / "release1", root / "release2")
            self.assertTrue(result["comparable"])
            self.assertTrue(result["regressed"])
            self.assertTrue(result["domains"]["vision"]["current_passed"])
            overall = next(row for row in result["domains"]["vision"]["checks"]
                           if row["scope"] == "overall" and row["metric"] == "answer_accuracy")
            self.assertEqual(overall["delta"], -0.5)
            self.assertIn("退化", markdown(result))

            gate_vision(manifest, base, oracle, root / "v3", max_accuracy_drop=0.2, slice_by=["task"])
            build(root / "a2", root / "v3", root / "release3")
            incompatible = compare(root / "release1", root / "release3")
            self.assertFalse(incompatible["comparable"])
            self.assertEqual(incompatible["domains"]["vision"]["reasons"], ["policy_changed"])


if __name__ == "__main__":
    unittest.main()
