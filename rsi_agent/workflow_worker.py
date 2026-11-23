"""Poll persisted workflow queues with fair rotation and graceful draining."""

import argparse
import json
import math
import signal
import sqlite3
import threading
import time
from pathlib import Path

from .durable import Runtime, canonical
from .workflows import Workflows


QUEUED = ("pending", "running", "incomplete", "cancelling", "timing_out")


class WorkflowWorker:
    def __init__(self, workflows, *, max_errors=None, retry_base=None, retry_max=None):
        self.workflows = workflows
        self.db = workflows.runtime.db
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS workflow_faults (
                workflow_id TEXT PRIMARY KEY, failures INTEGER NOT NULL,
                next_retry_at REAL NOT NULL, quarantined INTEGER NOT NULL,
                last_error TEXT, updated_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS workflow_fault_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, workflow_id TEXT NOT NULL,
                timestamp REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
        """)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            saved = self.db.execute(
                "SELECT value FROM runtime_settings WHERE name='workflow_worker_policy'").fetchone()
            policy = json.loads(saved[0]) if saved else {"max_errors": 3, "retry_base": 1, "retry_max": 60}
            candidate = {**policy, **{name: value for name, value in
                         (("max_errors", max_errors), ("retry_base", retry_base), ("retry_max", retry_max))
                         if value is not None}}
            if (type(candidate["max_errors"]) is not int or candidate["max_errors"] < 1
                    or any(type(candidate[name]) not in (int, float)
                           or not math.isfinite(candidate[name]) or candidate[name] <= 0
                           for name in ("retry_base", "retry_max"))
                    or candidate["retry_max"] < candidate["retry_base"]):
                raise ValueError("Controller error limit and backoff intervals must be positive and finite")
            if saved and candidate != policy:
                raise ValueError("Workflow worker policy is already bound to this runtime")
            self.db.execute("INSERT OR IGNORE INTO runtime_settings VALUES ('workflow_worker_policy',?)",
                            (canonical(candidate),))
            self.policy = candidate

    def fault_status(self):
        with self.db:
            self.db.execute("BEGIN")
            faults = [dict(row) for row in self.db.execute(
                "SELECT f.*,w.state AS workflow_state FROM workflow_faults f JOIN workflows w "
                "ON w.id=f.workflow_id WHERE f.failures>0 AND w.state IN (?,?,?,?,?) "
                "ORDER BY f.updated_at,f.workflow_id", QUEUED)]
            events = [dict(row) for row in self.db.execute(
                "SELECT * FROM workflow_fault_events ORDER BY seq DESC LIMIT 100")]
        for row in faults:
            row["last_error"] = json.loads(row["last_error"]) if row["last_error"] else None
            row["quarantined"] = bool(row["quarantined"])
        for row in events:
            row["payload"] = json.loads(row["payload"])
        return {"kind": "workflow_worker_faults", "policy": self.policy, "faults": faults, "events": events}

    def release(self, key, *, reason):
        if (not isinstance(key, str) or len(key) != 64
                or any(character not in "0123456789abcdef" for character in key)):
            raise ValueError("Invalid workflow ID")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Releasing controller faults requires an operator reason")
        runtime = self.workflows.runtime
        with runtime.task_lock("dispatch-" + key), runtime.task_lock("workflow-" + key), self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT * FROM workflow_faults WHERE workflow_id=?", (key,)).fetchone()
            state = self.db.execute("SELECT state FROM workflows WHERE id=?", (key,)).fetchone()
            if row is None or state is None or state[0] not in QUEUED:
                raise ValueError("Only queued workflows with controller fault records can be released")
            now = time.time()
            self.db.execute("UPDATE workflow_faults SET failures=0,next_retry_at=0,quarantined=0,"
                            "last_error=NULL,updated_at=? WHERE workflow_id=?", (now, key))
            self.db.execute("INSERT INTO workflow_fault_events(workflow_id,timestamp,kind,payload) "
                            "VALUES (?,?,'released',?)", (key, now, canonical(
                                {"reason": reason.strip(), "previous_failures": row["failures"]})))
        return {"kind": "workflow_worker_released", "id": key}

    def eligible(self, key):
        return self.db.execute(
            "SELECT 1 FROM workflows w LEFT JOIN workflow_faults f ON f.workflow_id=w.id "
            "WHERE w.id=? AND w.state IN (?,?,?,?,?) "
            "AND (f.workflow_id IS NULL OR (f.quarantined=0 AND f.next_retry_at<=?))",
            (key, *QUEUED, time.time())).fetchone() is not None

    def record_failure(self, key, error):
        # A manual run can take over after the failed run releases its graph lock.
        # Do not let its in-flight or terminal result receive a stale fault record.
        try:
            with self.workflows.runtime.task_lock("workflow-" + key), self.db:
                self.db.execute("BEGIN IMMEDIATE")
                state = self.db.execute("SELECT state FROM workflows WHERE id=?", (key,)).fetchone()
                if state is None or state[0] not in QUEUED:
                    return None
                old = self.db.execute("SELECT failures FROM workflow_faults WHERE workflow_id=?", (key,)).fetchone()
                failures = (old[0] if old else 0) + 1
                now = time.time()
                retry_at = now + min(self.policy["retry_max"],
                                     self.policy["retry_base"] * 2 ** min(failures - 1, 30))
                quarantined = failures >= self.policy["max_errors"]
                detail = {"type": type(error).__name__, "message": str(error)}
                self.db.execute("INSERT INTO workflow_faults VALUES (?,?,?,?,?,?) "
                                "ON CONFLICT(workflow_id) DO UPDATE SET failures=excluded.failures,"
                                "next_retry_at=excluded.next_retry_at,quarantined=excluded.quarantined,"
                                "last_error=excluded.last_error,updated_at=excluded.updated_at",
                                (key, failures, retry_at, int(quarantined), canonical(detail), now))
                payload = {**detail, "failures": failures, "next_retry_at": retry_at, "quarantined": quarantined}
                self.db.execute("INSERT INTO workflow_fault_events(workflow_id,timestamp,kind,payload) "
                                "VALUES (?,?,'failure',?)", (key, now, canonical(payload)))
                return payload
        except RuntimeError as problem:
            if str(problem) != "Task is already executing":
                raise
            return None

    def record_recovery(self, key):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT failures FROM workflow_faults WHERE workflow_id=?", (key,)).fetchone()
            if row and row[0]:
                now = time.time()
                self.db.execute("UPDATE workflow_faults SET failures=0,next_retry_at=0,quarantined=0,"
                                "last_error=NULL,updated_at=? WHERE workflow_id=?", (now, key))
                self.db.execute("INSERT INTO workflow_fault_events(workflow_id,timestamp,kind,payload) "
                                "VALUES (?,?,'recovered',?)", (key, now, canonical({"previous_failures": row[0]})))

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
                "LEFT JOIN workflow_faults f ON f.workflow_id=w.id WHERE w.state IN (?,?,?,?,?) "
                "AND (f.workflow_id IS NULL OR (f.quarantined=0 AND f.next_retry_at<=?)) "
                "ORDER BY coalesce(d.last_visit,0),w.rowid LIMIT ?",
                (*QUEUED, time.time(), max_workflows)).fetchall()
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
                with self.workflows.runtime.task_lock("dispatch-" + key):
                    if not self.eligible(key):
                        continue
                    try:
                        result = self.workflows.run(key, workers=node_workers, max_attempts=max_attempts,
                                                    stop_event=stop_event)
                    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                        if isinstance(error, RuntimeError) and str(error) == "Task is already executing":
                            busy.append(key)
                        else:
                            fault = self.record_failure(key, error)
                            errors.append({"id": key, "type": type(error).__name__, "message": str(error),
                                           "fault": fault})
                    else:
                        self.record_recovery(key)
                        results.append({"id": key, "state": result["state"], "passed": result["passed"]})
            except RuntimeError as error:
                if str(error) != "Task is already executing":
                    raise
                busy.append(key)
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Process one bounded queue snapshot")
    mode.add_argument("--faults", action="store_true", help="Inspect controller faults and recent audit events")
    mode.add_argument("--release", metavar="WORKFLOW_ID", help="Release a repaired controller fault")
    parser.add_argument("--reason", help="Required operator reason for --release")
    parser.add_argument("--max-controller-errors", type=int)
    parser.add_argument("--retry-base", type=float)
    parser.add_argument("--retry-max", type=float)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--max-workflows", type=int, default=10)
    parser.add_argument("--node-workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if bool(args.release) != bool(args.reason and args.reason.strip()):
        parser.error("--release and a nonempty --reason must be supplied together")
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    runtime = None
    try:
        runtime = Runtime(args.root)
        worker = WorkflowWorker(Workflows(runtime), max_errors=args.max_controller_errors,
                                retry_base=args.retry_base, retry_max=args.retry_max)
        options = {"max_workflows": args.max_workflows, "node_workers": args.node_workers,
                   "max_attempts": args.max_attempts}

        def report(value):
            print(json.dumps(value), flush=True)

        if args.faults:
            report(worker.fault_status())
        elif args.release:
            report(worker.release(args.release, reason=args.reason))
        elif args.once:
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
