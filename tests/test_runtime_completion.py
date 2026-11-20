"""Exercise external reconciliation, empirical memory scores and browser-rendered state."""

import hashlib
import importlib.util
import json
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent.external_effects import encode
from rsi_agent.governance import GovernedMemory
from rsi_agent.memory_calibration import evaluate


class ReferenceServiceTests(unittest.TestCase):
    def test_concurrent_retries_apply_once_and_survive_reopening(self):
        from rsi_agent.effect_service import initialize, apply
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "service.sqlite")
            initialize(database)
            body = encode({"name": "counter", "amount": 1})
            operation = "a" * 64
            with ThreadPoolExecutor(max_workers=8) as pool:
                receipts = list(pool.map(lambda _: apply(database, operation, body), range(16)))
            self.assertTrue(all(item["result"]["value"] == 1 for item in receipts))
            initialize(database)
            self.assertEqual(apply(database, operation, body)["result"]["value"], 1)
            with self.assertRaises(ValueError):
                apply(database, operation, encode({"name": "counter", "amount": 2}))


class CalibrationTests(unittest.TestCase):
    def test_exported_labels_calibrate_memory_and_include_tool_cost(self):
        from rsi_agent.runtime_export import export
        from rsi_agent.runtime_planning import submit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = Runtime(root / "runtime")
            labels = {}
            for index in range(5):
                key = submit(runtime, f"sample-{index}", f"{index}\n2\n", "sum",
                             [{"kind": "text", "expected": f"{index + 2}\n"}])
                self.assertEqual(runtime.run(key)["state"], "succeeded")
                labels[key] = True
            runtime.close()
            report = export(root / "runtime", labels, root / "report",
                            calibrate_memory=True, tool_cost=0.2)
            self.assertAlmostEqual(report["cost_per_success"], 0.2)
            export(root / "runtime", labels, root / "repeated", calibrate_memory=True)
            memory = GovernedMemory(root / "runtime/memory.sqlite")
            try:
                selected = memory.recall("family:sum", require_calibrated=True)["selected"]
                self.assertEqual(selected["observations"], 5)
            finally:
                memory.close()

    def test_independent_evidence_is_deduplicated_and_holdout_is_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = GovernedMemory(Path(directory) / "memory.sqlite")
            value = {"recipe": ["sum"], "passed": True}
            memory.remember("family:sum", value, confidence=0.9, ttl_seconds=100,
                            provenance={"source": "verification"})
            self.assertIsNone(memory.recall("family:sum", require_calibrated=True)["selected"])
            for index in range(5):
                args = {"observation_id": str(index), "success": True,
                        "provenance": {"source": "independent_annotation"}}
                memory.observe("family:sum", value, **args)
                memory.observe("family:sum", value, **args)
            selected = memory.recall("family:sum", require_calibrated=True)["selected"]
            self.assertEqual(selected["observations"], 5)
            self.assertAlmostEqual(selected["confidence"], 6/7)
            self.assertEqual(selected["confidence_method"], "beta_posterior")
            heldout = [{"id": "heldout", "subject": "family:sum", "value": value, "success": True}]
            report = evaluate(memory, heldout)
            self.assertAlmostEqual(report["brier_score"], (1/7)**2)
            self.assertEqual(report["covered"], 1)
            with self.assertRaises(ValueError):
                evaluate(memory, [{**heldout[0], "id": "0"}])
            with self.assertRaises(ValueError):
                memory.observe("family:sum", value, observation_id="0", success=False,
                               provenance={"source": "independent_annotation"})
            memory.close()


class ExternalServiceTests(unittest.TestCase):
    def setUp(self):
        self.receipts = {}
        self.puts = 0
        self.lose_response = True
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                operation = self.path.rsplit("/", 1)[-1]
                if operation not in fixture.receipts:
                    self.send_error(404)
                else:
                    self.reply(fixture.receipts[operation])

            def do_PUT(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                operation = self.path.rsplit("/", 1)[-1]
                digest = hashlib.sha256(raw).hexdigest()
                if operation in fixture.receipts:
                    if fixture.receipts[operation]["request_sha256"] != digest:
                        self.send_error(409)
                        return
                else:
                    fixture.puts += 1
                    fixture.receipts[operation] = {
                        "idempotency_key": operation, "request_sha256": digest,
                        "status": "succeeded", "result": json.loads(raw)}
                if fixture.lose_response:
                    fixture.lose_response = False
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                self.reply(fixture.receipts[operation])

            def reply(self, value):
                raw = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            self.skipTest("Host sandbox forbids local HTTP servers")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        if hasattr(self, "thread"):
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()

    def test_applied_write_with_lost_response_is_reconciled_without_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            permissions = {"tools": ["http_write"], "origins": [self.url]}
            runtime = Runtime(directory, permissions=permissions)
            source = json.dumps({"url": self.url + "/operations", "payload": {"value": 42}})
            key = runtime.submit("external", source, ["http_write"], [{"kind": "file"}])
            first = runtime.run(key)
            self.assertEqual(first["state"], "failed")
            self.assertEqual(first["next_step"], 0)
            self.assertEqual(self.puts, 1)
            self.assertEqual(runtime.db.execute("SELECT state FROM external_effects").fetchone()[0], "uncertain")
            runtime.close()
            runtime = Runtime(directory)
            result = runtime.run(key)
            self.assertEqual(result["state"], "succeeded", result["error"])
            self.assertEqual(self.puts, 1)
            self.assertEqual(result["recoveries"], 1)
            self.assertEqual(runtime.db.execute("SELECT state FROM external_effects").fetchone()[0], "confirmed")
            runtime.close()

    def test_queue_recovers_lost_service_response_with_a_bounded_retry(self):
        from rsi_agent.task_worker import drain
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory, permissions={"tools": ["http_write"], "origins": [self.url]})
            key = runtime.submit("queued", json.dumps({
                "url": self.url + "/operations", "payload": {"value": 42}}),
                ["http_write"], [{"kind": "file"}])
            result = drain(runtime, max_attempts=2, retry_delay=0)
            self.assertTrue(result["passed"], result)
            self.assertEqual(runtime.status(key)["attempts"], 2)
            self.assertEqual(runtime.status(key)["external_operations"][0]["state"], "confirmed")
            self.assertEqual(self.puts, 1)
            runtime.close()

    def test_interruption_after_confirmed_receipt_reuses_committed_service_result(self):
        self.lose_response = False
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory, permissions={"tools": ["http_write"], "origins": [self.url]})
            key = runtime.submit("confirmed", json.dumps({
                "url": self.url + "/operations", "payload": {"value": 42}}),
                ["http_write"], [{"kind": "file"}])
            from rsi_agent.external_effects import execute

            def interrupted(*args):
                execute(*args)
                raise KeyboardInterrupt()

            with patch("rsi_agent.durable.execute_external", interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    runtime.run(key)
            runtime.close()
            runtime = Runtime(directory)
            self.assertEqual(runtime.run(key)["state"], "succeeded")
            self.assertEqual(self.puts, 1)
            runtime.close()


@unittest.skipUnless(importlib.util.find_spec("playwright"), "Requires the optional browser extra")
class BrowserTests(unittest.TestCase):
    def test_dynamic_dom_is_verified_after_javascript_runs(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                raw = b"""<div id="status" data-state="loading">Loading</div>
                <script>setTimeout(() => {
                document.querySelector('#status').setAttribute('data-state', 'ready');
                document.querySelector('#status').textContent = 'Ready';
                }, 100);</script>"""
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            self.skipTest("Host sandbox forbids local HTTP servers")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                url = f"http://127.0.0.1:{server.server_port}"
                runtime = Runtime(directory, permissions={"tools": ["copy"], "origins": [url]})
                key = runtime.submit("browser", "input", ["copy"], [
                    {"kind": "browser", "url": url, "selector": "#status",
                     "attributes": {"data-state": "ready"}, "text": "Ready"}],
                    limits={"timeout_seconds": 10})
                result = runtime.run(key)
                self.assertEqual(result["state"], "succeeded", result["error"])
                evidence = json.loads(result["verification"])[0]["evidence"]
                self.assertEqual(evidence["scope"], "rendered_chromium_dom")
                self.assertEqual(evidence["text"], "Ready")
                runtime.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
