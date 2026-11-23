"""Verify live state, governed planning, billing coverage and failed-task metrics."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent.governance import GovernedMemory
from rsi_agent.runtime_metrics import summarize
from rsi_agent.runtime_planning import export_record, submit


class RuntimeIntegrationTests(unittest.TestCase):
    def test_live_page_and_persisted_origin_permissions(self):
        state = {"status": "done"}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                raw = ('<button id="task" data-status="' + state["status"] + '">Task</button>').encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            self.skipTest("Host sandbox forbids binding a local HTTP server")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                url = "http://127.0.0.1:" + str(server.server_port)
                runtime = Runtime(directory, permissions={"tools": ["copy"], "origins": [url]})
                rule = [{"kind": "page_live", "url": url, "element_id": "task",
                         "attributes": {"data-status": "done"}}]
                key = runtime.submit("live-1", "input", ["copy"], rule)
                self.assertEqual(runtime.run(key)["state"], "succeeded")
                state["status"] = "pending"
                key = runtime.submit("live-2", "input", ["copy"], rule)
                self.assertEqual(runtime.run(key)["state"], "rejected")
                with self.assertRaises(ValueError):
                    runtime.submit("denied-tool", "", ["sum"], [{"kind": "file"}])
                with self.assertRaises(ValueError):
                    runtime.submit("denied-origin", "", ["copy"], [
                        {**rule[0], "url": "http://127.0.0.1:1"}])
                runtime.close()
                runtime = Runtime(directory)
                self.assertEqual(runtime.permissions["tools"], ["copy"])
                runtime.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_permissions_are_enforced_without_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory, permissions={"tools": ["copy"], "origins": []})
            with self.assertRaises(ValueError):
                runtime.submit("denied-tool", "", ["sum"], [{"kind": "file"}])
            with self.assertRaises(ValueError):
                runtime.submit("denied-origin", "", ["copy"], [
                    {"kind": "page_live", "url": "http://127.0.0.1", "element_id": "task"}])
            runtime.close()
            runtime = Runtime(directory)
            self.assertEqual(runtime.permissions["tools"], ["copy"])
            runtime.close()

    def test_model_usage_is_persistent_and_memory_participates_in_planning(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            source = "Pear\napple\nPear\n"
            rules = [{"kind": "text", "expected": "apple\npear\n"}]

            def planner(*args):
                args[-1].append({"attempt": 1})
                return ("normalize", "sort_normalized"), 120

            with patch("rsi_agent.runtime_planning.model_recipe", planner):
                key = submit(runtime, "planned", source, "words", rules,
                             model="fixture", token_price=0.001)
            result = runtime.run(key)
            self.assertEqual(result["state"], "succeeded")
            memory = GovernedMemory(Path(directory) / "memory.sqlite")
            try:
                for index in range(5):
                    memory.observe("family:words", {"recipe": ["normalize", "sort_normalized"], "passed": True},
                                   observation_id=f"independent-{index}", success=True,
                                   provenance={"source": "held_out_fixture"})
            finally:
                memory.close()
            with patch("rsi_agent.runtime_planning.model_recipe") as model:
                self.assertEqual(submit(runtime, "planned", source, "words", rules,
                                        model="fixture", token_price=0.001), key)
                second = submit(runtime, "from-memory", source, "words", rules, model="fixture")
                model.assert_not_called()
            self.assertEqual(runtime.run(second)["state"], "succeeded")
            record = export_record(runtime, key, success=True)
            self.assertEqual(record["tokens"], 120)
            self.assertAlmostEqual(record["cost"], 0.12)
            self.assertGreaterEqual(record["seconds"], 0)
            memory_usage = json.loads(runtime.db.execute(
                "SELECT evidence FROM usage WHERE task_id=?", (second,)).fetchone()[0])
            self.assertEqual(memory_usage["method"], "memory")
            runtime.close()

    def test_changed_registered_command_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            key = runtime.submit("pinned", "Pear\napple\nPear\n",
                                 ["normalize", "sort_normalized"],
                                 [{"kind": "text", "expected": "apple\npear\n"}])
            self.assertEqual(runtime.run(key, interrupt_after=1)["state"], "interrupted")
            with patch.dict("rsi_agent.durable.TOOLS", {"sort_normalized": ("false",)}):
                result = runtime.run(key)
            self.assertEqual(result["state"], "failed")
            self.assertIn("implementation changed", result["error"])
            self.assertEqual(result["calls"], 1)
            runtime.close()

    def test_interrupted_planning_uses_fallback_and_retains_unknown_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            source = "1\n2\n"
            rules = [{"kind": "text", "expected": "3\n"}]

            def interrupted(*args):
                raise KeyboardInterrupt()

            with patch("rsi_agent.runtime_planning.model_recipe", interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    submit(runtime, "request", source, "sum", rules, model="fixture")
            with patch("rsi_agent.runtime_planning.model_recipe") as planner:
                key = submit(runtime, "request", source, "sum", rules, model="fixture")
                planner.assert_not_called()
            self.assertEqual(runtime.run(key)["state"], "succeeded")
            self.assertIsNone(export_record(runtime, key, success=True)["tokens"])
            with patch("rsi_agent.durable.TOOLS", {"bad": ("false",)}):
                # This fixture explicitly extends the operator policy for failure injection.
                runtime.permissions["tools"].append("bad")
                failed = runtime.submit("failed", "", ["bad"], [{"kind": "file"}])
                self.assertEqual(runtime.run(failed)["state"], "failed")
            record = export_record(runtime, failed, success=False)
            self.assertIsNone(record["verifier_passed"])
            report = summarize([record])
            self.assertEqual(report["task_success_rate"], 0)
            self.assertEqual(report["verifier_coverage"], 0)
            self.assertIsNone(report["verifier_error_rate"])
            runtime.close()


if __name__ == "__main__":
    unittest.main()
