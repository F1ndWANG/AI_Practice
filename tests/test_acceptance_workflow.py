"""Run the integrated workflow against passing, failing and invalid inputs."""

import tempfile
import unittest
from pathlib import Path

from ai_practice.acceptance import run
from ai_practice.release import verify
from rsi_agent.study import study
from vlm_posttraining.fixtures import generate


class AcceptanceWorkflowTests(unittest.TestCase):
    def test_acceptance_preserves_review_evidence_and_applies_previous_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            agent, vision = root / "agent", root / "vision"
            study(agent, per_family=2)
            generate(vision)
            policies = Path(__file__).resolve().parents[1] / "examples/policies"
            weak = vision / "fixture-repeated_answer.jsonl"
            oracle = vision / "fixture-oracle.jsonl"
            settings = dict(agent_base=agent, agent_candidate=agent, agent_policy=policies / "agent.json",
                            vision_manifest=vision / "prepared/test.jsonl",
                            vision_base=weak, vision_candidate=oracle, vision_policy=policies / "vision.json")
            good = root / "good"
            self.assertEqual(run(good, **settings)["exit_code"], 0)
            self.assertTrue(verify(good / "bundle")["passed"])
            regression = root / "regression"
            result = run(regression, **{**settings, "vision_candidate": weak}, previous=good / "bundle")
            self.assertEqual(result["exit_code"], 1)
            self.assertTrue(verify(regression / "bundle")["passed"])
            self.assertTrue((regression / "comparison/report.md").is_file())

            failed = root / "failed"
            result = run(failed, **{**settings, "vision_base": oracle, "vision_candidate": weak})
            self.assertEqual(result["exit_code"], 1)
            self.assertFalse(verify(failed / "bundle")["passed"])

            error = root / "error"
            result = run(error, **{**settings, "vision_policy": root / "missing.json"})
            self.assertEqual(result["exit_code"], 2)
            self.assertEqual(result["stages"]["agent"]["state"], "passed")
            self.assertEqual(result["stages"]["vision"]["state"], "error")
            self.assertFalse((error / "bundle").exists())


if __name__ == "__main__":
    unittest.main()
