"""Persist and resume dependency graphs over the existing task Runtime."""

import argparse
import hashlib
import json
import math
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from graphlib import TopologicalSorter
from pathlib import Path

from .durable import Runtime, TOOLS, DEFAULTS, VERIFIERS, canonical
from .governance import artifact
from .task_worker import run_one
from .execution_control import TaskCancelled, TaskDeadlineExceeded


FAILED = {"failed", "rejected", "cancelled", "timed_out", "error", "blocked"}


class Workflows:
    def __init__(self, runtime):
        self.runtime = runtime
        runtime.db.executescript("""
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY, plan TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS workflow_controls (
                workflow_id TEXT PRIMARY KEY, cancel_requested INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS workflow_nodes (
                workflow_id TEXT, node_id TEXT, task_id TEXT, state TEXT,
                error TEXT, PRIMARY KEY(workflow_id,node_id));
        """)
        runtime.db.execute(
            "INSERT OR IGNORE INTO workflow_controls(workflow_id) SELECT id FROM workflows")
        runtime.db.commit()

    def submit(self, key, plan):
        if (not isinstance(key, str) or not key or not isinstance(plan, dict)
                or set(plan) - {"nodes", "limits"} or not isinstance(plan.get("nodes"), list)
                or not plan["nodes"]):
            raise ValueError("Workflow requires an idempotency key and nonempty nodes")
        limits = plan.get("limits", {})
        if (not isinstance(limits, dict) or set(limits) - {"max_tool_calls", "total_seconds"}
                or ("max_tool_calls" in limits and
                    (type(limits["max_tool_calls"]) is not int or limits["max_tool_calls"] < 1))):
            raise ValueError("Workflow max_tool_calls must be a positive integer")
        if ("total_seconds" in limits and
                (type(limits["total_seconds"]) not in (int, float)
                 or not math.isfinite(limits["total_seconds"]) or limits["total_seconds"] <= 0)):
            raise ValueError("Workflow total_seconds must be finite and positive")
        nodes = plan["nodes"]
        graph = {}
        for node in nodes:
            if (not isinstance(node, dict)
                    or set(node) - {"id", "source", "inputs", "separator", "depends_on", "recipe", "rules", "limits"}
                    or not isinstance(node.get("id"), str) or not node["id"] or node["id"] in graph
                    or ("source" in node) == ("inputs" in node)
                    or not isinstance(node.get("recipe"), list) or not node["recipe"]
                    or not isinstance(node.get("rules"), list) or not node["rules"]
                    or any(not isinstance(rule, dict) or rule.get("kind") not in VERIFIERS for rule in node["rules"])
                    or not isinstance(node.get("limits", {}), dict)
                    or any(not isinstance(tool, str) or tool not in TOOLS or tool not in self.runtime.permissions["tools"]
                           for tool in node["recipe"])):
                raise ValueError("Invalid workflow node or tool permissions")
            dependencies = node.get("depends_on", [])
            if (not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies)
                    or len(set(dependencies)) != len(dependencies)):
                raise ValueError("Node dependencies must be unique string IDs")
            dependencies = set(dependencies)
            if "source" in node and not isinstance(node["source"], str):
                raise ValueError("Node source must be text")
            if "inputs" in node:
                if not isinstance(node["inputs"], list) or not node["inputs"]:
                    raise ValueError("Node inputs must be nonempty")
                for item in node["inputs"]:
                    if (not isinstance(item, dict) or set(item) - {"node", "path"}
                            or not isinstance(item.get("node"), str)
                            or not isinstance(item.get("path", "output.txt"), str)):
                        raise ValueError("Invalid node output reference")
                    dependencies.add(item["node"])
            if not isinstance(node.get("separator", ""), str):
                raise ValueError("Node separator must be text")
            graph[node["id"]] = dependencies
        if any(dependency not in graph for dependencies in graph.values() for dependency in dependencies):
            raise ValueError("Unknown workflow dependency")
        list(TopologicalSorter(graph).static_order())
        encoded = canonical(plan)
        workflow_id = hashlib.sha256(key.encode()).hexdigest()
        with self.runtime.db:
            self.runtime.db.execute("BEGIN IMMEDIATE")
            old = self.runtime.db.execute("SELECT plan FROM workflows WHERE id=?", (workflow_id,)).fetchone()
            if old and old["plan"] != encoded:
                raise ValueError("Workflow key is bound to a different plan")
            if not old:
                self.runtime.db.execute("INSERT INTO workflows VALUES (?,?,'pending')", (workflow_id, encoded))
                self.runtime.db.execute("INSERT INTO workflow_controls(workflow_id) VALUES (?)", (workflow_id,))
                if "total_seconds" in limits:
                    self.runtime.db.execute("INSERT INTO workflow_deadlines VALUES (?,?)",
                                            (workflow_id, time.time() + limits["total_seconds"]))
                if "max_tool_calls" in limits:
                    self.runtime.db.execute("INSERT INTO workflow_budgets(workflow_id,max_calls) VALUES (?,?)",
                                            (workflow_id, limits["max_tool_calls"]))
                self.runtime.db.executemany("INSERT INTO workflow_nodes VALUES (?,?,NULL,'pending',NULL)",
                                            [(workflow_id, node["id"]) for node in nodes])
        return workflow_id

    def status(self, workflow_id):
        workflow = self.runtime.db.execute(
            "SELECT w.*,c.cancel_requested FROM workflows w JOIN workflow_controls c ON c.workflow_id=w.id "
            "WHERE w.id=?", (workflow_id,)).fetchone()
        if workflow is None:
            raise ValueError("Unknown workflow")
        nodes = [dict(row) for row in self.runtime.db.execute(
            "SELECT n.node_id,n.task_id,n.state,n.error,t.state AS task_state,t.next_step AS completed_steps "
            "FROM workflow_nodes n LEFT JOIN tasks t ON t.id=n.task_id WHERE n.workflow_id=? ORDER BY n.rowid",
            (workflow_id,))]
        state = workflow["state"]
        if workflow["cancel_requested"]:
            state = "cancelling" if any(node["task_state"] in {"running", "verifying"} for node in nodes) else "cancelled"
        budget = self.runtime.db.execute(
            "SELECT max_calls,calls FROM workflow_budgets WHERE workflow_id=?", (workflow_id,)).fetchone()
        deadline = self.deadline(workflow_id)
        return {"kind": "agent_workflow", "id": workflow_id, "state": state,
                "passed": state == "succeeded", "cancel_requested": bool(workflow["cancel_requested"]),
                "deadline_at": deadline, "deadline_expired": deadline is not None and time.time() >= deadline,
                "tool_budget": ({"limit": budget["max_calls"], "used": budget["calls"],
                                 "remaining": budget["max_calls"] - budget["calls"]} if budget else None),
                "nodes": nodes}

    def cancel(self, workflow_id):
        with self.runtime.db:
            self.runtime.db.execute("BEGIN IMMEDIATE")
            row = self.runtime.db.execute("SELECT state FROM workflows WHERE id=?", (workflow_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown workflow")
            if row["state"] in {"succeeded", "cancelled", "timed_out"}:
                return self.status(workflow_id)
            self.runtime.db.execute("UPDATE workflow_controls SET cancel_requested=1 WHERE workflow_id=?",
                                    (workflow_id,))
            for node in self.runtime.db.execute(
                "SELECT task_id FROM workflow_nodes WHERE workflow_id=? AND task_id IS NOT NULL",
                (workflow_id,)).fetchall():
                self.runtime.cancel_locked(node["task_id"], context={"workflow_id": workflow_id})
            self.runtime.db.execute(
                "UPDATE workflow_nodes SET state='cancelled',error='Workflow cancellation requested' "
                "WHERE workflow_id=? AND (task_id IS NULL OR task_id IN "
                "(SELECT id FROM tasks WHERE state='cancelled'))", (workflow_id,))
            active = self.runtime.db.execute(
                "SELECT 1 FROM workflow_nodes n JOIN tasks t ON t.id=n.task_id "
                "WHERE n.workflow_id=? AND t.state IN ('running','verifying') LIMIT 1",
                (workflow_id,)).fetchone()
            self.runtime.db.execute("UPDATE workflows SET state=? WHERE id=?",
                                    ("cancelling" if active else "cancelled", workflow_id))
        return self.status(workflow_id)

    def cancellation_requested(self, workflow_id):
        return bool(self.runtime.db.execute(
            "SELECT cancel_requested FROM workflow_controls WHERE workflow_id=?", (workflow_id,)).fetchone()[0])

    def deadline(self, workflow_id):
        row = self.runtime.db.execute(
            "SELECT deadline_at FROM workflow_deadlines WHERE workflow_id=?", (workflow_id,)).fetchone()
        return row[0] if row else None

    def expired(self, workflow_id):
        deadline = self.deadline(workflow_id)
        return deadline is not None and time.time() >= deadline

    def source(self, node, completed):
        if "source" in node:
            return node["source"]
        parts = []
        budget = {**DEFAULTS, **node.get("limits", {})}["workspace_bytes"]
        size = 0
        for reference in node["inputs"]:
            task = completed[reference["node"]]["task_id"]
            row = self.runtime.db.execute("SELECT next_step FROM tasks WHERE id=?", (task,)).fetchone()
            checkpoint = self.runtime.check_checkpoint(task, self.runtime.root / task, row["next_step"])
            path = artifact(checkpoint, reference.get("path", "output.txt"))
            if path.stat().st_size > budget - size:
                raise ValueError("Workflow input exceeds workspace budget")
            raw = path.read_bytes()
            manifest = json.loads(self.runtime.db.execute(
                "SELECT manifest FROM checkpoints WHERE task_id=? AND step=?", (task, row["next_step"])).fetchone()[0])
            if hashlib.sha256(raw).hexdigest() != manifest[str(path.relative_to(checkpoint))]:
                raise ValueError("Dependency artifact changed while reading")
            size += len(raw)
            parts.append(raw.decode("utf-8"))
        source = node.get("separator", "").join(parts)
        if len(source.encode()) > budget:
            raise ValueError("Workflow input exceeds workspace budget")
        return source

    def execute_node(self, workflow_id, node, source, max_attempts):
        runtime = Runtime(self.runtime.root)
        state, error = "error", "Node execution interrupted"
        try:
            key = f"workflow:{workflow_id}:node:{node['id']}"
            task = runtime.submit(key, source, node["recipe"], node["rules"], limits=node.get("limits"),
                                  workflow=(workflow_id, node["id"]))
            result = run_one(runtime, task, max_attempts=max_attempts)
            if result["state"] == "succeeded":
                runtime.run(task)
            state, error = result["state"], result["error"]
        except TaskCancelled as problem:
            state, error = "cancelled", str(problem)
        except TaskDeadlineExceeded as problem:
            state, error = "timed_out", str(problem)
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as problem:
            state, error = "error", str(problem)
        finally:
            try:
                with runtime.db:
                    runtime.db.execute(
                        "UPDATE workflow_nodes SET state=?,error=? WHERE workflow_id=? AND node_id=?",
                        (state, error, workflow_id, node["id"]))
            finally:
                runtime.close()
        return state, error

    def run(self, workflow_id, *, workers=2, max_attempts=3, stop_event=None):
        if type(workers) is not int or workers < 1 or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("Workers and attempt limits must be positive integers")
        if len(workflow_id) != 64 or any(character not in "0123456789abcdef" for character in workflow_id):
            raise ValueError("Invalid workflow ID")
        with self.runtime.task_lock("workflow-" + workflow_id):
            status = self.status(workflow_id)
            if stop_event is not None and stop_event.is_set():
                return status
            if self.expired(workflow_id) and not status["cancel_requested"]:
                # A crashed orchestrator can leave bound tasks without an active worker.
                # Their inherited deadlines also apply when resumed by a queue worker.
                for node in status["nodes"]:
                    if node["task_state"] in {"pending", "failed", "interrupted", "running", "verifying"}:
                        try:
                            self.runtime.run(node["task_id"])
                        except RuntimeError as error:
                            if str(error) != "Task is already executing":
                                raise
                status = self.status(workflow_id)
            if status["cancel_requested"]:
                for node in status["nodes"]:
                    if node["task_state"] in {"running", "verifying"}:
                        try:
                            self.runtime.run(node["task_id"])
                        except RuntimeError as error:
                            if str(error) != "Task is already executing":
                                raise
                return self.cancel(workflow_id)
            plan = json.loads(self.runtime.db.execute(
                "SELECT plan FROM workflows WHERE id=?", (workflow_id,)).fetchone()[0])
            nodes = {node["id"]: node for node in plan["nodes"]}
            dependencies = {key: set(node.get("depends_on", [])) |
                            {item["node"] for item in node.get("inputs", [])} for key, node in nodes.items()}
            completed = {row["node_id"]: row for row in status["nodes"]}
            visited = set()
            # Revalidate completed checkpoints before reusing any graph output.
            for node_id, row in completed.items():
                if row["state"] == "succeeded":
                    try:
                        self.runtime.run(row["task_id"])
                    except (OSError, ValueError, RuntimeError) as error:
                        row["state"], row["error"] = "error", str(error)
                    visited.add(node_id)
            with self.runtime.db:
                self.runtime.db.execute(
                    "UPDATE workflows SET state='running' WHERE id=? AND id IN "
                    "(SELECT workflow_id FROM workflow_controls WHERE cancel_requested=0)", (workflow_id,))
            with ThreadPoolExecutor(max_workers=min(workers, len(nodes))) as pool:
                active = {}
                try:
                    while True:
                        cancelled = self.cancellation_requested(workflow_id)
                        expired = self.expired(workflow_id)
                        stopping = stop_event is not None and stop_event.is_set()
                        if not cancelled and not expired and not stopping:
                            ready = [key for key in nodes if key not in visited and all(
                                completed[dependency]["state"] == "succeeded" for dependency in dependencies[key])]
                            for key in ready[:max(0, workers - len(active))]:
                                if stop_event is not None and stop_event.is_set():
                                    break
                                visited.add(key)
                                try:
                                    source = self.source(nodes[key], completed)
                                    if stop_event is not None and stop_event.is_set():
                                        break
                                    job = pool.submit(self.execute_node, workflow_id, nodes[key], source, max_attempts)
                                    active[job] = key
                                except (OSError, ValueError) as error:
                                    completed[key]["state"], completed[key]["error"] = "error", str(error)
                                    with self.runtime.db:
                                        self.runtime.db.execute(
                                            "UPDATE workflow_nodes SET state='error',error=? "
                                            "WHERE workflow_id=? AND node_id=?", (str(error), workflow_id, key))
                        if not active:
                            remaining = any(key not in visited and all(
                                completed[parent]["state"] == "succeeded" for parent in dependencies[key])
                                for key in nodes)
                            if (not cancelled and not expired and remaining
                                    and not (stop_event is not None and stop_event.is_set())):
                                continue
                            break
                        finished, _ = wait(active, timeout=0.1, return_when=FIRST_COMPLETED)
                        for job in finished:
                            key = active.pop(job)
                            try:
                                completed[key]["state"], completed[key]["error"] = job.result()
                            except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                                completed[key]["state"], completed[key]["error"] = "error", str(error)
                            completed[key]["task_id"] = self.runtime.db.execute(
                                "SELECT task_id FROM workflow_nodes WHERE workflow_id=? AND node_id=?",
                                (workflow_id, key)).fetchone()[0]
                            with self.runtime.db:
                                self.runtime.db.execute(
                                    "UPDATE workflow_nodes SET state=?,error=? WHERE workflow_id=? AND node_id=?",
                                    (completed[key]["state"], completed[key]["error"], workflow_id, key))
                except BaseException:
                    self.cancel(workflow_id)
                    raise
            for key in TopologicalSorter(dependencies).static_order():
                if key not in visited and any(completed[parent]["state"] in FAILED for parent in dependencies[key]):
                    completed[key]["state"], completed[key]["error"] = "blocked", "Dependency did not succeed"
            if self.cancellation_requested(workflow_id):
                for key in nodes:
                    if key not in visited:
                        completed[key]["state"], completed[key]["error"] = "cancelled", "Workflow cancellation requested"
            state = ("succeeded" if all(row["state"] == "succeeded" for row in completed.values()) else
                     "failed" if any(row["state"] in FAILED for row in completed.values()) else "incomplete")
            if self.cancellation_requested(workflow_id):
                state = self.status(workflow_id)["state"]
            with self.runtime.db:
                self.runtime.db.execute("BEGIN IMMEDIATE")
                if self.expired(workflow_id) and state != "succeeded":
                    for key in nodes:
                        if key not in visited:
                            completed[key]["state"], completed[key]["error"] = (
                                "timed_out", "Workflow total wall-time budget exhausted")
                    active = self.runtime.db.execute(
                        "SELECT 1 FROM workflow_nodes n JOIN tasks t ON t.id=n.task_id "
                        "WHERE n.workflow_id=? AND t.state IN ('running','verifying') LIMIT 1",
                        (workflow_id,)).fetchone()
                    state = "timing_out" if active else "timed_out"
                if self.cancellation_requested(workflow_id):
                    state = self.status(workflow_id)["state"]
                    for key in nodes:
                        if key not in visited:
                            completed[key]["state"], completed[key]["error"] = "cancelled", "Workflow cancellation requested"
                self.runtime.db.executemany(
                    "UPDATE workflow_nodes SET state=?,error=? WHERE workflow_id=? AND node_id=?",
                    [(row["state"], row["error"], workflow_id, key) for key, row in completed.items()])
                self.runtime.db.execute("UPDATE workflows SET state=? WHERE id=?", (state, workflow_id))
            return self.status(workflow_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("key")
    submit.add_argument("plan", type=Path)
    for command in ("run", "status", "cancel"):
        child = sub.add_parser(command)
        child.add_argument("workflow_id")
        if command == "run":
            child.add_argument("--workers", type=int, default=2)
            child.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    runtime = Runtime(args.root)
    try:
        workflows = Workflows(runtime)
        if args.command == "submit":
            print(json.dumps({"workflow_id": workflows.submit(args.key, json.loads(args.plan.read_text()))}))
        elif args.command == "cancel":
            print(json.dumps(workflows.cancel(args.workflow_id), indent=2))
        elif args.command == "status":
            print(json.dumps(workflows.status(args.workflow_id), indent=2))
        else:
            result = workflows.run(args.workflow_id, workers=args.workers, max_attempts=args.max_attempts)
            print(json.dumps(result, indent=2))
            if not result["passed"]:
                raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
