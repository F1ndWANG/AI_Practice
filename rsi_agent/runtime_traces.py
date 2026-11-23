"""Reconstruct OpenTelemetry spans from committed events and acknowledge delivery."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


TERMINAL = {"succeeded", "rejected", "failed", "cancelled", "timed_out"}


def identifiers(task_id, label):
    trace = hashlib.sha256(("trace:" + task_id).encode()).hexdigest()[:32]
    span = hashlib.sha256((task_id + ":" + label).encode()).hexdigest()[:16]
    return trace, span


def traceparent(task_id, label):
    trace, span = identifiers(task_id, label)
    return f"00-{trace}-{span}-01"


def build(runtime, task):
    key = task["id"]
    events = [dict(row) for row in runtime.db.execute(
        "SELECT * FROM events WHERE task_id=? ORDER BY seq", (key,))]
    if task["state"] not in TERMINAL:
        return None
    if not events:
        raise ValueError("Task event log is missing")
    trace, root_span = identifiers(key, "task")
    spans = []

    def span(label, name, start, end, parent, attributes, outcome):
        _, span_id = identifiers(key, label)
        spans.append({"trace_id": trace, "span_id": span_id, "parent_span_id": parent,
                      "traceparent": traceparent(key, label), "name": name,
                      "start_unix_nano": int(start * 1_000_000_000),
                      "end_unix_nano": int(max(start, end) * 1_000_000_000),
                      "status": "OK" if outcome == "succeeded" else "ERROR",
                      "attributes": {"agent.task.id": key, "agent.outcome": outcome, **attributes}})

    attempt = None
    tool = None
    verification = None
    for event in events:
        data, timestamp = json.loads(event["payload"]), event["timestamp"]
        if event["kind"] == "state" and data["to"] == "running":
            attempt = (event["seq"], timestamp)
        elif event["kind"] == "tool_started":
            tool = (event["seq"], timestamp, data["tool"], data["step"])
        if tool and (event["kind"] == "checkpoint" or
                     event["kind"] == "state" and data["to"] in TERMINAL | {"interrupted"}):
            outcome = "succeeded" if event["kind"] == "checkpoint" else data["to"]
            parent = identifiers(key, f"attempt:{attempt[0]}")[1] if attempt else root_span
            span(f"tool:{tool[0]}", "agent.tool." + tool[2], tool[1], timestamp, parent,
                 {"agent.step": tool[3], "agent.timing": "observed_event_bounds"}, outcome)
            tool = None
        if event["kind"] == "state" and data["to"] == "verifying":
            verification = (event["seq"], timestamp)
        if event["kind"] == "state" and data["to"] in TERMINAL | {"interrupted"}:
            outcome = data["to"]
            if verification:
                parent = identifiers(key, f"attempt:{attempt[0]}")[1] if attempt else root_span
                span(f"verification:{verification[0]}", "agent.verification",
                     verification[1], timestamp, parent, {}, outcome)
                verification = None
            if attempt:
                span(f"attempt:{attempt[0]}", "agent.attempt", attempt[1], timestamp,
                     root_span, {}, outcome)
                attempt = None
    first = events[0]["timestamp"]
    usage = runtime.db.execute("SELECT evidence FROM usage WHERE task_id=?", (key,)).fetchone()
    if usage:
        first = min(first, json.loads(usage["evidence"]).get("started_at", first))
    span("task", "agent.task", first, events[-1]["timestamp"], None,
         {"agent.tool_calls": task["calls"], "agent.recoveries": task["recoveries"]}, task["state"])
    revision = hashlib.sha256(canonical(spans).encode()).hexdigest()
    return {"task_id": key, "revision": revision, "spans": spans}


def sdk_export(spans, endpoint):
    try:
        from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
        from opentelemetry.sdk.trace.id_generator import IdGenerator
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import SpanExportResult
        from opentelemetry.context import Context
        from opentelemetry.trace import (
            SpanContext, TraceFlags, SpanKind, Status, StatusCode, NonRecordingSpan, set_span_in_context)
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as error:
        raise RuntimeError("Install the telemetry extra for OTLP export") from error

    class ReplayIds(IdGenerator):
        def generate_trace_id(self):
            return self.trace_id

        def generate_span_id(self):
            return self.span_id

    class Capture(SpanProcessor):
        def __init__(self):
            self.spans = []

        def on_end(self, span):
            self.spans.append(span)

    ids = ReplayIds()
    provider = TracerProvider(id_generator=ids, resource=Resource.create(
        {"service.name": "ai-practice-agent"}), shutdown_on_exit=False)
    capture = Capture()
    provider.add_span_processor(capture)
    tracer = provider.get_tracer("ai_practice.runtime")
    exporter = OTLPSpanExporter(endpoint=endpoint, timeout=10)
    try:
        for span in spans:
            ids.trace_id, ids.span_id = int(span["trace_id"], 16), int(span["span_id"], 16)
            parent = Context()
            if span["parent_span_id"]:
                parent = set_span_in_context(NonRecordingSpan(SpanContext(
                    ids.trace_id, int(span["parent_span_id"], 16), True, TraceFlags(1))), parent)
            recording = tracer.start_span(
                span["name"], context=parent, kind=SpanKind.INTERNAL,
                attributes=span["attributes"], start_time=span["start_unix_nano"])
            recording.set_status(Status(StatusCode.OK if span["status"] == "OK" else StatusCode.ERROR))
            recording.end(end_time=span["end_unix_nano"])
        if exporter.export(capture.spans) != SpanExportResult.SUCCESS:
            raise RuntimeError("OTLP collector did not acknowledge trace delivery")
    finally:
        exporter.shutdown()
        provider.shutdown()


def export(runtime, *, output=None, endpoint=None):
    if (output is None) == (endpoint is None):
        raise ValueError("Choose a JSON output or an OTLP endpoint")
    if endpoint is not None:
        from .live_verifier import origin
        origin(endpoint)
    target = str(Path(output).resolve()) if output is not None else endpoint
    lock_key = "trace-export-" + hashlib.sha256(target.encode()).hexdigest()
    with runtime.task_lock(lock_key):
        runtime.db.execute("""CREATE TABLE IF NOT EXISTS trace_deliveries (
            task_id TEXT, target TEXT, revision TEXT, PRIMARY KEY(task_id,target))""")
        runtime.db.commit()
        with runtime.db:
            runtime.db.execute("BEGIN")
            traces = [build(runtime, dict(task)) for task in runtime.db.execute(
                "SELECT * FROM tasks ORDER BY rowid") if task["state"] in TERMINAL]
            pending = [trace for trace in traces if not runtime.db.execute(
                "SELECT 1 FROM trace_deliveries WHERE task_id=? AND target=? AND revision=?",
                (trace["task_id"], target, trace["revision"])).fetchone()]
        if output is not None:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = (json.dumps({"traces": traces}, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
            if path.exists():
                if path.read_bytes() != raw:
                    raise FileExistsError("Trace output already contains a different snapshot")
            else:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
                    name = staged.name
                    staged.write(raw)
                    staged.flush()
                    os.fsync(staged.fileno())
                try:
                    os.link(name, path)
                finally:
                    Path(name).unlink(missing_ok=True)
        elif pending:
            sdk_export([span for trace in pending for span in trace["spans"]], endpoint)
        with runtime.db:
            runtime.db.executemany("INSERT OR REPLACE INTO trace_deliveries VALUES (?,?,?)",
                                  [(trace["task_id"], target, trace["revision"]) for trace in pending])
        return {"tasks": len(traces), "delivered": len(pending), "target": target,
                "spans": sum(len(trace["spans"]) for trace in pending)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--otlp-endpoint")
    args = parser.parse_args()
    from .durable import Runtime
    runtime = Runtime(args.root)
    try:
        print(json.dumps(export(runtime, output=args.output, endpoint=args.otlp_endpoint), indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
