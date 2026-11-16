"""Exercise passing and failing release bundles and verify copied artifacts."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_practice.artifacts import artifact_index, write_json
from ai_practice.release import build, verify


def write_gate(folder, domain, passed):
    folder.mkdir()
    checks = {"answer_accuracy": {"baseline": 1.0, "candidate": float(passed),
                                 "delta": float(passed) - 1.0, "regressed": int(not passed),
                                 "samples": 1, "passed": passed}}
    write_json(folder / "summary.json", {"kind": "regression_gate", "domain": domain,
                                        "passed": passed, "checks": checks})
    (folder / "changes.jsonl").write_text('{"id":"private-test-id"}\n', encoding="utf-8")
    write_json(folder / "receipt.json", {"schema": 1, "artifacts": artifact_index(
        folder, ["summary.json", "changes.jsonl"]
    )})


class ReleaseBundleTests(unittest.TestCase):
    def test_default_bundle_excludes_sample_ids_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent, vision, output = root / "agent", root / "vision", root / "bundle"
            write_gate(agent, "Agent", True)
            write_gate(vision, "VLM", True)
            self.assertTrue(build(agent, vision, output, revision="1" * 40)["passed"])
            self.assertEqual(verify(output)["verified_files"], 4)
            self.assertFalse((output / "agent/changes.jsonl").exists())
            extra = output / "unexpected.txt"
            extra.touch()
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                verify(output)
            extra.unlink()
            (output / "report.md").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "changed artifact"):
                verify(output)
            (agent / "changes.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "changed artifact"):
                build(agent, vision, root / "rejected")

    def test_failing_bundle_returns_one_and_still_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent, vision, output = root / "agent", root / "vision", root / "bundle"
            write_gate(agent, "Agent", True)
            write_gate(vision, "VLM", False)
            run = subprocess.run([sys.executable, "-m", "ai_practice.release", "build",
                                  str(agent), str(vision), str(output), "--include-changes"],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            result = verify(output)
            self.assertFalse(result["passed"])
            self.assertEqual(result["verified_files"], 6)
            self.assertTrue((output / "vision/changes.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
