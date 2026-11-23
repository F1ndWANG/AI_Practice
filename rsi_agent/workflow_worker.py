"""Poll persisted workflow queues with fair rotation and graceful draining."""

import argparse
import json
import math
import signal
import sqlite3
import threading
from pathlib import Path

from .durable import Runtime
from .workflows import Workflows


QUEUED = ("pending", "running", "incomplete", "cancelling", "timing_out")


class WorkflowWorker:
    def __init__(self, workflows):
        self.workflows = workflows
        self.db = workflows.runtime.db

    def tick(self, *, max_workflows=10, node_workers=2, max_attempts=3, stop_event=None):
        for value in (max_workflows, node_workers, max_attempts):
            if type(value) is not int or value < 1:
                raise ValueError("Batch, node worker and attempt limits must be positive integers")
        results, busy, errors = [], [], []
        if stop_event is not None and stop_event.is_set():
            return {"kind": "workflow_worker_cycle", "workflows": results, "busy": busy, "errors": errors}
        # Persistent sequence numbers avoid starvation by an incomplete/busy head
        # and preserve fair rotation across restarts and wall-clock adjustments.
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            rows = self.db.execute(
                "SELECT w.id FROM workflows w LEFT JOIN workflow_dispatch d ON d.workflow_id=w.id "
                "WHERE w.state IN (?,?,?,?,?) ORDER BY coalesce(d.last_visit,0),w.rowid LIMIT ?",
                (*QUEUED, max_workflows)).fetchall()
            sequence = self.db.execute("SELECT coalesce(max(last_visit),0) FROM workflow_dispatch").fetchone()[0]
            for offset, row in enumerate(rows, 1):
                self.db.execute("INSERT INTO workflow_dispatch VALUES (?,?) "
                                "ON CONFLICT(workflow_id) DO UPDATE SET last_visit=excluded.last_visit",
                                (row["id"], sequence + offset))
        for row in rows:
            if stop_event is not None and stop_event.is_set():
                break
            key = row["id"]
            try:
                result = self.workflows.run(key, workers=node_workers, max_attempts=max_attempts,
                                            stop_event=stop_event)
                results.append({"id": key, "state": result["state"], "passed": result["passed"]})
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                if isinstance(error, RuntimeError) and str(error) == "Task is already executing":
                    busy.append(key)
                else:
                    errors.append({"id": key, "type": type(error).__name__, "message": str(error)})
        return {"kind": "workflow_worker_cycle", "workflows": results, "busy": busy, "errors": errors}

    def serve(self, *, stop_event, poll_interval=0.5, on_cycle=None, **options):
        if (type(poll_interval) not in (int, float) or not math.isfinite(poll_interval)
                or poll_interval <= 0):
            raise ValueError("Poll interval must be finite and positive")
        while not stop_event.is_set():
            result = self.tick(stop_event=stop_event, **options)
            if on_cycle is not None:
                on_cycle(result)
            stop_event.wait(poll_interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--once", action="store_true", help="Process one bounded queue snapshot")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--max-workflows", type=int, default=10)
    parser.add_argument("--node-workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    runtime = None
    try:
        runtime = Runtime(args.root)
        worker = WorkflowWorker(Workflows(runtime))
        options = {"max_workflows": args.max_workflows, "node_workers": args.node_workers,
                   "max_attempts": args.max_attempts}

        def report(value):
            print(json.dumps(value), flush=True)

        if args.once:
            result = worker.tick(stop_event=stop, **options)
            report(result)
            if result["errors"] or any(row["state"] in {"failed", "timed_out"} for row in result["workflows"]):
                raise SystemExit(1)
        else:
            worker.serve(stop_event=stop, poll_interval=args.poll_interval, on_cycle=report, **options)
    finally:
        if runtime is not None:
            runtime.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
