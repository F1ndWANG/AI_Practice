"""Crash recovery, permissions, memory decisions and independent metric denominators."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent.governance import GovernedMemory, verify
from rsi_agent.runtime_metrics import summarize


class DurableTests(unittest.TestCase):
    def test_resume_and_idempotency(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            rule = [{"kind": "text", "expected": "apple\npear\n"}]
            key = runtime.submit("one", "Pear\napple\nPear\n",
                                 ["normalize", "sort_normalized"], rule)
            first = runtime.run(key, interrupt_after=1)
            self.assertEqual((first["state"], first["next_step"], first["calls"]),
                             ("interrupted", 1, 1))
            runtime.close()
            runtime = Runtime(directory)
            result = runtime.run(key)
            self.assertEqual((result["state"], result["calls"], result["recoveries"]),
                             ("succeeded", 2, 1))
            self.assertEqual(runtime.run(key)["calls"], 2)
            self.assertEqual(runtime.submit("one", "Pear\napple\nPear\n",
                                           ["normalize", "sort_normalized"], rule), key)
            with self.assertRaises(ValueError):
                runtime.submit("one", "changed", ["copy"], rule)
            (Path(directory) / key / "checkpoint-2/output.txt").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "Checkpoint"):
                runtime.run(key)
            runtime.close()

    def test_budget_permissions_timeout_and_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            with self.assertRaises(ValueError):
                runtime.submit("bad", "", ["arbitrary_shell"], [{"kind": "file"}])
            key = runtime.submit("budget", "hi", ["copy", "copy"],
                                 [{"kind": "file"}], limits={"max_calls": 1})
            self.assertIn("budget", runtime.run(key)["error"])
            with runtime.tool_slot(1):
                with self.assertRaisesRegex(RuntimeError, "concurrency"):
                    with runtime.tool_slot(1):
                        pass
            with patch.dict("rsi_agent.durable.TOOLS", {
                "sleep": (sys.executable, "-c", "import time; time.sleep(10)")
            }):
                runtime.permissions["tools"].append("sleep")
                key = runtime.submit("timeout", "", ["sleep"], [{"kind": "file"}],
                                     limits={"timeout_seconds": 0.05})
                self.assertEqual(runtime.run(key)["state"], "failed")
            runtime.close()

    def test_memory_dedup_expiration_conflict_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = GovernedMemory(Path(directory) / "memory.sqlite")
            args = {"confidence": 0.8, "ttl_seconds": 100, "provenance": {"source": "task-a"}}
            memory.remember("fact", "a", **args)
            memory.remember("fact", "a", **args)
            self.assertEqual(memory.db.execute("SELECT count(*) FROM governed_memory").fetchone()[0], 1)
            memory.remember("fact", "b", **args)
            self.assertEqual(memory.recall("fact")["state"], "conflict")
            memory.remember("fact", "a", confidence=0.9, ttl_seconds=100,
                            provenance={"source": "task-b"})
            self.assertEqual(memory.recall("fact")["selected"]["value"], "a")
            self.assertEqual(memory.recall("fact", now=time.time()+200)["state"], "missing")
            memory.close()

    def test_verification_and_system_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output.txt").write_text('[{"id":1},{"id":1}]')
            self.assertFalse(verify(root, {"kind": "data", "required": ["id"], "unique": ["id"]})["passed"])
            with self.assertRaises(ValueError):
                verify(root, {"kind": "file", "path": "../missing"})
            (root / "paper.txt").write_text("Methods: measured on 100 held-out tasks.")
            self.assertTrue(verify(root, {"kind": "paper", "path": "paper.txt",
                                        "quote": "100 held-out tasks"})["passed"])
            (root / "page.html").write_text('<button id="done">Done</button>')
            self.assertTrue(verify(root, {"kind": "page", "path": "page.html",
                                        "element_id": "done"})["passed"])
            suite = root / "suite"
            suite.mkdir()
            (suite / "test_result.py").write_text(
                "import unittest\nclass Result(unittest.TestCase):\n"
                " def test_result(self):\n  self.assertEqual(2+2,4)\n")
            self.assertTrue(verify(root, {"kind": "tests", "suite": str(suite)})["passed"])
            (suite / "test_result.py").write_text(
                "import unittest\nclass Result(unittest.TestCase):\n"
                " def test_result(self):\n  self.fail('independent failure')\n")
            self.assertFalse(verify(root, {"kind": "tests", "suite": str(suite)})["passed"])
        report = summarize([
            {"success": True, "verifier_passed": True, "tool_calls": 2, "seconds": 1,
             "recoveries": 1, "tokens": None, "cost": 0.1},
            {"success": False, "verifier_passed": True, "tool_calls": 4, "seconds": 5,
             "recoveries": 0, "tokens": None, "cost": 0.2}])
        self.assertEqual(report["recovery_success_rate"], 1)
        self.assertEqual(report["verifier_false_accept_rate"], 1)
        self.assertAlmostEqual(report["cost_per_success"], 0.3)
        self.assertIsNone(report["mean_tokens"])

    def test_killed_process_resumes_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            key = runtime.submit("killed", "Pear\napple\nPear\n",
                                 ["normalize", "sort_normalized"],
                                 [{"kind": "text", "expected": "apple\npear\n"}])
            runtime.close()
            script = """
import sys, time
from rsi_agent.durable import Runtime
r = Runtime(sys.argv[1])
original = r.event
def event(key, kind, payload):
    original(key, kind, payload)
    if kind == "tool_started" and payload["step"] == 1:
        # First checkpoint is committed. Pause in the next transaction.
        time.sleep(30)
r.event = event
r.run(sys.argv[2])
"""
            process = subprocess.Popen([sys.executable, "-c", script, directory, key])
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    runtime = Runtime(directory)
                    row = runtime.db.execute("SELECT next_step FROM tasks WHERE id=?", (key,)).fetchone()
                    runtime.close()
                    if row["next_step"] == 1:
                        break
                    time.sleep(0.02)
                else:
                    self.fail("No committed checkpoint")
                process.kill()
                process.wait()
                runtime = Runtime(directory)
                result = runtime.run(key)
                self.assertEqual(result["state"], "succeeded")
                self.assertEqual(result["next_step"], 2)
                runtime.close()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    unittest.main()
