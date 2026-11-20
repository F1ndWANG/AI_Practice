"""Stable span identity, delivery receipts and real OTLP protobuf transport."""

import importlib.util
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent.runtime_traces import export, identifiers


def sdk_available():
    try:
        return importlib.util.find_spec("opentelemetry.sdk") is not None
    except ModuleNotFoundError:
        return False


class TraceTests(unittest.TestCase):
    def test_trace_tree_is_stable_and_delivery_receipt_follows_acknowledgement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = Runtime(root / "runtime")
            key = runtime.submit("trace", "private-input-marker", ["copy"], [{"kind": "file"}])
            runtime.run(key)
            output = root / "trace.json"
            self.assertEqual(export(runtime, output=output)["delivered"], 1)
            self.assertEqual(export(runtime, output=output)["delivered"], 0)
            raw = output.read_text()
            self.assertNotIn("private-input-marker", raw)
            trace = json.loads(raw)["traces"][0]
            self.assertEqual(len(trace["spans"]), 4)
            ids = {span["span_id"] for span in trace["spans"]}
            for span in trace["spans"]:
                self.assertTrue(span["parent_span_id"] is None or span["parent_span_id"] in ids)
                self.assertEqual(len(span["trace_id"]), 32)
                self.assertGreaterEqual(span["end_unix_nano"], span["start_unix_nano"])
            endpoint = "http://127.0.0.1:4318/v1/traces"
            with patch("rsi_agent.runtime_traces.sdk_export", side_effect=RuntimeError("unavailable")):
                with self.assertRaises(RuntimeError):
                    export(runtime, endpoint=endpoint)
            self.assertEqual(runtime.db.execute(
                "SELECT count(*) FROM trace_deliveries WHERE target=?", (endpoint,)).fetchone()[0], 0)
            with patch("rsi_agent.runtime_traces.sdk_export") as sender:
                self.assertEqual(export(runtime, endpoint=endpoint)["delivered"], 1)
                self.assertEqual(export(runtime, endpoint=endpoint)["delivered"], 0)
                sender.assert_called_once()
            runtime.close()


@unittest.skipUnless(sdk_available(), "Requires the telemetry extra")
class OtlpTests(unittest.TestCase):
    def test_sdk_posts_valid_protobuf_with_the_persisted_span_ids(self):
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                message = ExportTraceServiceRequest()
                message.ParseFromString(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(message)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

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
                runtime = Runtime(directory)
                key = runtime.submit("otlp", "input", ["copy"], [{"kind": "file"}])
                runtime.run(key)
                endpoint = f"http://127.0.0.1:{server.server_port}/v1/traces"
                self.assertEqual(export(runtime, endpoint=endpoint)["spans"], 4)
                self.assertEqual(len(received), 1)
                spans = [span for resource in received[0].resource_spans
                         for scope in resource.scope_spans for span in scope.spans]
                self.assertEqual(len(spans), 4)
                root = next(span for span in spans if span.name == "agent.task")
                trace_id, span_id = identifiers(key, "task")
                self.assertEqual(root.trace_id.hex(), trace_id)
                self.assertEqual(root.span_id.hex(), span_id)
                runtime.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
