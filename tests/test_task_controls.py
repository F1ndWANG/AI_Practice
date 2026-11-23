"""Cross-process cancellation, persistent wall deadlines and bounded queue retries."""

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent.runtime_planning import export_record
from rsi_agent.task_worker import drain


class TaskControlTests(unittest.TestCase):
    def test_pending_cancel_and_status_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            key = runtime.submit("cancel", "hi", ["copy"], [{"kind": "file"}])
            self.assertEqual(runtime.status(key)["total_steps"], 1)
            self.assertNotIn("spec", runtime.status(key))
            self.assertEqual(runtime.cancel(key)["state"], "cancelled")
            self.assertEqual(runtime.cancel(key)["state"], "cancelled")
            self.assertEqual(runtime.run(key)["state"], "cancelled")
            self.assertEqual(runtime.status(key)["calls"], 0)
            self.assertIsNone(export_record(runtime, key, success=False)["verifier_passed"])
            runtime.close()

    def test_total_deadline_is_not_reset_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            key = runtime.submit("deadline", "Pear\napple\n", ["normalize", "sort_normalized"],
                                 [{"kind": "text", "expected": "apple\npear\n"}],
                                 limits={"total_seconds": 1})
            first = runtime.run(key, interrupt_after=1)
            self.assertEqual(first["state"], "interrupted")
            deadline = runtime.status(key)["deadline_at"]
            runtime.close()
            time.sleep(1.05)
            runtime = Runtime(directory)
            result = runtime.run(key)
            self.assertEqual(result["state"], "timed_out")
            self.assertEqual(result["calls"], 1)
            self.assertEqual(runtime.status(key)["deadline_at"], deadline)
            self.assertEqual(runtime.run(key)["state"], "timed_out")
            runtime.close()

    def test_running_subprocess_is_stopped_by_another_process(self):
        command = (sys.executable, "-I", "-c",
                   "from pathlib import Path; import time; "
                   "Path('started').write_text('yes'); time.sleep(30)")
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("rsi_agent.durable.TOOLS", {"sleep_fixture": command}):
                runtime = Runtime(directory, permissions={"tools": ["sleep_fixture"], "origins": []})
                key = runtime.submit("running", "", ["sleep_fixture"], [{"kind": "file"}],
                                     limits={"timeout_seconds": 35, "total_seconds": 40})
                runtime.close()
            script = """
import json,sys
from rsi_agent.durable import Runtime, TOOLS
TOOLS["sleep_fixture"] = tuple(json.loads(sys.argv[3]))
r = Runtime(sys.argv[1])
try:
 print(json.dumps(r.run(sys.argv[2])))
finally:
 r.close()
"""
            process = subprocess.Popen([sys.executable, "-c", script, directory, key, json.dumps(command)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                until = time.monotonic() + 5
                while not list((Path(directory) / key).glob("staging-*/started")):
                    if time.monotonic() >= until:
                        self.fail("Tool subprocess did not start")
                    time.sleep(0.02)
                runtime = Runtime(directory)
                request = runtime.cancel(key)
                self.assertTrue(request["cancel_requested"])
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, stderr)
                result = json.loads(stdout)
                self.assertEqual(result["state"], "cancelled")
                self.assertEqual(result["next_step"], 0)
                self.assertEqual(runtime.status(key)["calls"], 1)
                runtime.close()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_verifier_subprocess_can_be_cancelled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            suite.mkdir()
            marker = root / "verifier_started"
            (suite / "test_slow.py").write_text(
                "import unittest,time\nfrom pathlib import Path\n"
                "class Slow(unittest.TestCase):\n"
                " def test_slow(self):\n"
                f"  Path({str(marker)!r}).write_text('started')\n"
                "  time.sleep(30)\n  self.assertTrue(True)\n")
            runtime = Runtime(root / "runtime")
            key = runtime.submit("verifier", "input", ["copy"],
                                 [{"kind": "tests", "suite": str(suite)}],
                                 limits={"timeout_seconds": 35, "total_seconds": 40})
            runtime.close()
            script = ("import sys,json; from rsi_agent.durable import Runtime; "
                      "r=Runtime(sys.argv[1]); print(json.dumps(r.run(sys.argv[2]))); r.close()")
            process = subprocess.Popen([sys.executable, "-c", script, str(root / "runtime"), key],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                until = time.monotonic() + 5
                while not marker.exists():
                    if time.monotonic() >= until:
                        self.fail("Verifier subprocess did not start")
                    time.sleep(0.02)
                runtime = Runtime(root / "runtime")
                self.assertEqual(runtime.status(key)["state"], "verifying")
                runtime.cancel(key)
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, stderr)
                result = json.loads(stdout)
                self.assertEqual(result["state"], "cancelled")
                self.assertIsNone(result["verification"])
                runtime.close()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_queue_retry_limit_persists_and_permanent_failures_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            with patch.dict("rsi_agent.durable.TOOLS", {
                "sleep_fixture": (sys.executable, "-c", "import time; time.sleep(1)"),
                "bad_fixture": (sys.executable, "-c", "raise SystemExit(1)"),
            }):
                runtime.permissions["tools"].extend(["sleep_fixture", "bad_fixture"])
                retry = runtime.submit("retry", "", ["sleep_fixture"], [{"kind": "file"}],
                                       limits={"timeout_seconds": 0.05})
                bad = runtime.submit("bad", "", ["bad_fixture"], [{"kind": "file"}])
                good = runtime.submit("good", "hello", ["copy"], [{"kind": "file"}])
                result = drain(runtime, max_attempts=2, retry_delay=0)
                self.assertFalse(result["passed"])
                self.assertEqual(runtime.status(retry)["attempts"], 2)
                self.assertEqual(runtime.status(retry)["calls"], 2)
                self.assertEqual(runtime.status(bad)["attempts"], 1)
                self.assertEqual(runtime.status(good)["state"], "succeeded")
                drain(runtime, max_attempts=2, retry_delay=0)
                self.assertEqual(runtime.status(retry)["calls"], 2)
                self.assertEqual(runtime.status(bad)["attempts"], 1)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
