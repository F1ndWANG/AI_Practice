"""Inspect persisted runtime health using one read-only SQLite snapshot."""

import argparse
import json
from pathlib import Path
import sqlite3
import time

from .workflow_worker import QUEUED

QUEUED_SQL = "(" + ",".join("'" + state + "'" for state in QUEUED) + ")"


def _counts(db, table):
    # Table names are internal constants, never CLI input.
    return {row[0]: row[1] for row in db.execute(f"SELECT state,count(*) FROM {table} GROUP BY state")}


def _snapshot(db, now):
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "tasks" not in tables:
        raise ValueError("Database is not an initialized durable runtime")
    states = _counts(db, "tasks")
    totals = dict(db.execute(
        "SELECT count(*) AS count,coalesce(sum(calls),0) AS tool_calls,"
        "coalesce(sum(seconds),0) AS active_execution_seconds,coalesce(sum(recoveries),0) AS recoveries "
        "FROM tasks").fetchone())
    if "usage" in tables:
        usage = dict(db.execute(
            "SELECT count(u.tokens) AS token_records,sum(u.tokens) AS known_tokens,"
            "count(u.cost) AS cost_records,sum(u.cost) AS known_cost "
            "FROM tasks t LEFT JOIN usage u ON u.task_id=t.id").fetchone())
    else:
        usage = {"token_records": 0, "known_tokens": None, "cost_records": 0, "known_cost": None}
    usage["token_coverage"] = usage["token_records"] / totals["count"] if totals["count"] else None
    usage["cost_coverage"] = usage["cost_records"] / totals["count"] if totals["count"] else None
    workflow_states, queue, budgets = {}, None, None
    if "workflows" in tables:
        workflow_states = _counts(db, "workflows")
        joins = []
        quarantined, retry, deadline = "0", "0", "NULL"
        if "workflow_faults" in tables:
            joins.append("LEFT JOIN workflow_faults f ON f.workflow_id=w.id")
            quarantined, retry = "coalesce(f.quarantined,0)", "coalesce(f.next_retry_at,0)"
        if "workflow_deadlines" in tables:
            joins.append("LEFT JOIN workflow_deadlines d ON d.workflow_id=w.id")
            deadline = "d.deadline_at"
        queue = dict(db.execute(
            "SELECT count(*) AS total,"
            f"coalesce(sum(CASE WHEN {quarantined}=1 THEN 1 ELSE 0 END),0) AS quarantined,"
            f"coalesce(sum(CASE WHEN {quarantined}=0 AND {retry}>:now THEN 1 ELSE 0 END),0) AS backoff,"
            f"coalesce(sum(CASE WHEN {quarantined}=0 AND {retry}<=:now THEN 1 ELSE 0 END),0) AS eligible,"
            f"coalesce(sum(CASE WHEN {deadline}<=:now THEN 1 ELSE 0 END),0) AS expired "
            "FROM workflows w " + " ".join(joins)
            + f" WHERE w.state IN {QUEUED_SQL}",
            {"now": now}).fetchone())
        if "workflow_budgets" in tables:
            budgets = dict(db.execute(
                "SELECT count(*) AS configured_workflows,coalesce(sum(b.max_calls),0) AS tool_limit,"
                "coalesce(sum(b.calls),0) AS tool_used,"
                f"coalesce(sum(CASE WHEN w.state IN {QUEUED_SQL} "
                "AND b.calls>=b.max_calls THEN 1 ELSE 0 END),0) AS unfinished_at_limit "
                "FROM workflow_budgets b JOIN workflows w ON w.id=b.workflow_id").fetchone())
    issues = []
    if queue and queue["quarantined"]:
        issues.append({"kind": "quarantined_workflows", "count": queue["quarantined"]})
    if queue and queue["expired"]:
        issues.append({"kind": "expired_unfinished_workflows", "count": queue["expired"]})
    return {"kind": "agent_runtime_health", "observed_at": now,
            "scope": "persisted_runtime_state", "worker_liveness": "unknown",
            "attention_required": bool(issues), "issues": issues,
            "tasks": {**totals, "states": states}, "usage": usage,
            "workflows": {"available": "workflows" in tables, "states": workflow_states,
                          "queue": queue, "budgets": budgets},
            "capabilities": {name: name in tables for name in
                             ("usage", "workflow_faults", "workflow_deadlines", "workflow_budgets")}}


def inspect(root):
    path = Path(root).resolve() / "runtime.sqlite"
    if not path.is_file():
        raise ValueError("Runtime database does not exist")
    # Do not instantiate Runtime/Workflows/Worker: inspection must not create
    # schema, bind worker policy or advance a task/queue state.
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        return _snapshot(db, time.time())
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true", help="Exit 1 when persisted state needs attention")
    args = parser.parse_args()
    try:
        report = inspect(args.root)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"Runtime inspection failed: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.check and report["attention_required"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
