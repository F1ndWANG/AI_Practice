"""Pin runnable acceptance data and reject silent input changes."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_practice.acceptance_plan import checked_options
from rsi_agent.study import study
from vlm_posttraining.fixtures import generate


class AcceptancePlanTests(unittest.TestCase):
    def test_frozen_plan_is_relocatable_and_rejects_changed_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            study(source / "agent", per_family=2)
            generate(source / "vision")
            policies = Path(__file__).resolve().parents[1] / "examples/policies"
            shutil.copytree(policies, source / "policies")
            options = {
                "agent_base": source / "agent", "agent_candidate": source / "agent",
                "agent_base_variant": "baseline", "agent_candidate_variant": "brs_drs",
                "agent_policy": source / "policies/agent.json", "vision_policy": source / "policies/vision.json",
                "vision_manifest": source / "vision/prepared/test.jsonl",
                "vision_base": source / "vision/fixture-repeated_answer.jsonl",
                "vision_candidate": source / "vision/fixture-oracle.jsonl",
                "include_changes": False, "revision": "f" * 40, "previous": None,
            }
            plan = source / "plan.json"
            create = [sys.executable, "-m", "ai_practice.acceptance_plan", "freeze", str(plan)]
            for name, value in options.items():
                if value is not None and value is not False:
                    create.extend([f"--{name.replace('_', '-')}", str(value)])
            proc = subprocess.run(create, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            relocated = root / "relocated"
            source.rename(relocated)
            plan = relocated / "plan.json"
            _, receipt = checked_options(plan)
            self.assertEqual(receipt["verified_inputs"], 11)
            output = root / "result"
            command = [sys.executable, "-m", "ai_practice.acceptance_plan", "run", str(plan)]
            proc = subprocess.run([*command, str(output)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            bundle = json.loads((output / "bundle/summary.json").read_text())
            self.assertEqual(bundle["plan_sha256"], receipt["plan_sha256"])

            candidate = relocated / "vision/fixture-oracle.jsonl"
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            rejected = root / "rejected"
            proc = subprocess.run([*command, str(rejected)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("Acceptance input changed: vision_candidate", proc.stderr)
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
