"""Persistent, checkpointed execution of the registered local Agent tools."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .governance import GovernedMemory, verify
from .runtime import TOOLS as LOCAL_TOOLS
from .external_effects import execute as execute_external, GatewayError
from .execution_control import (TaskCancelled, TaskDeadlineExceeded, ToolCapacityError, wait_process)

from .live_verifier import origin
from .sandbox_worker import isolate, container_image, remove_container

VERIFIERS = {"file", "text", "data", "tests", "page", "page_live", "browser", "paper"}
TOOLS = {**LOCAL_TOOLS, "http_write": ("runtime-gateway", "http_write")}

DEFAULTS = {"max_calls": 6, "timeout_seconds": 3, "cpu_seconds": 3,
            "file_bytes": 1048576, "workspace_bytes": 4194304,
            "memory_bytes": 536870912, "concurrency": 2, "isolation": "process",
            "max_processes": 64, "cpu_quota": 1, "total_seconds": 300}
TRANSITIONS = {
    "pending": {"running", "cancelled"}, "running": {"interrupted", "failed", "verifying", "cancelled", "timed_out"},
    "interrupted": {"running", "cancelled"}, "failed": {"running", "cancelled"},
    "verifying": {"running", "failed", "succeeded", "rejected", "cancelled", "timed_out"},
    "rejected": set(), "succeeded": set(), "cancelled": set(), "timed_out": set(),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Runtime:
    def __init__(self, root, *, permissions=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "permissions.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / "permissions.json"
            policy = permissions or {"tools": sorted(TOOLS), "origins": []}
            if (set(policy) != {"tools", "origins"} or not isinstance(policy["tools"], list)
                    or any(tool not in TOOLS for tool in policy["tools"])):
                raise ValueError("Permission policy contains unknown tools")
            for url in policy["origins"]:
                origin(url)
            if path.exists():
                self.permissions = json.loads(path.read_text())
                if permissions is not None and self.permissions != permissions:
                    raise ValueError("Runtime permission policy already exists with different permissions")
            else:
                self.permissions = policy
                path.write_text(canonical(policy) + "\n")
        self.db = sqlite3.connect(self.root / "runtime.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runtime_settings (
                name TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL,
                next_step INTEGER NOT NULL DEFAULT 0, calls INTEGER NOT NULL DEFAULT 0,
                seconds REAL NOT NULL DEFAULT 0, recoveries INTEGER NOT NULL DEFAULT 0,
                verification TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, timestamp REAL,
                kind TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS task_controls (
                task_id TEXT PRIMARY KEY, cancel_requested INTEGER NOT NULL DEFAULT 0,
                deadline_at REAL, attempts INTEGER NOT NULL DEFAULT 0,
                retryable INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS usage (
                task_id TEXT PRIMARY KEY, tokens INTEGER, cost REAL, evidence TEXT);
            CREATE TABLE IF NOT EXISTS checkpoints (
                task_id TEXT, step INTEGER, manifest TEXT, PRIMARY KEY(task_id,step));
        """)
        legacy = self.root / "concurrency.json"
        if legacy.is_file():
            self.db.execute("INSERT OR IGNORE INTO runtime_settings VALUES ('concurrency',?)",
                            (canonical(json.loads(legacy.read_text())),))
        self.db.commit()

    def close(self):
        self.db.close()

    def event(self, key, kind, payload):
        self.db.execute("INSERT INTO events(task_id,timestamp,kind,payload) VALUES (?,?,?,?)",
                        (key, time.time(), kind, canonical(payload)))

    def transition(self, key, state):
        current = self.db.execute("SELECT state FROM tasks WHERE id=?", (key,)).fetchone()["state"]
        if state not in TRANSITIONS[current]:
            raise ValueError(f"Invalid task transition {current} -> {state}")
        self.db.execute("UPDATE tasks SET state=? WHERE id=?", (state, key))
        self.event(key, "state", {"from": current, "to": state})

    def submit(self, key, source, recipe, rules, *, limits=None, memory_subject=None, usage=None):
        if not isinstance(key, str) or not key:
            raise ValueError("A nonempty idempotency key is required")
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError("Usage must be a record")
            tokens, cost = usage.get("tokens"), usage.get("cost")
            if (tokens is not None and (type(tokens) is not int or tokens < 0)
                    or cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0)
                    or not isinstance(usage.get("evidence"), dict)):
                raise ValueError("Usage requires valid token, cost and evidence records")
        policy = {**DEFAULTS, **(limits or {})}
        if set(policy) != set(DEFAULTS) or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for name, value in policy.items() if name != "isolation"
        ) or any(type(policy[name]) is not int for name in policy if name not in {"timeout_seconds", "isolation", "cpu_quota", "total_seconds"}):
            raise ValueError("Invalid resource limits")
        if policy["isolation"] not in {"process", "bubblewrap", "docker"}:
            raise ValueError("Unknown isolation mode")
        isolate(["true"], self.root, policy["isolation"], policy)
        if (not isinstance(recipe, (list, tuple)) or not recipe
                or any(not isinstance(tool, str) or tool not in TOOLS
                       or tool not in self.permissions["tools"] for tool in recipe)):
            raise ValueError("Recipe must contain registered tools")
        if (not isinstance(rules, (list, tuple)) or not rules
                or any(not isinstance(rule, dict) or rule.get("kind") not in VERIFIERS for rule in rules)):
            raise ValueError("Provide objective verification rules")
        if "http_write" in recipe:
            external = json.loads(source)
            if (not isinstance(external, dict) or set(external) != {"url", "payload"}
                    or origin(external["url"]) not in {origin(url) for url in self.permissions["origins"]}):
                raise ValueError("External write requires an approved origin and a payload")
        if len(source.encode()) > policy["workspace_bytes"]:
            raise ValueError("Input exceeds workspace budget")
        rules = json.loads(canonical(rules))
        for rule in rules:
            if rule["kind"] in {"page_live", "browser"}:
                if origin(rule["url"]) not in {origin(url) for url in self.permissions["origins"]}:
                    raise ValueError("Page origin is outside runtime permissions")
            if rule["kind"] == "tests":
                suite = Path(rule["suite"]).resolve()
                if not suite.is_dir():
                    raise ValueError("Independent test suite directory is missing")
                rule["suite"] = str(suite)
                rule["suite_manifest"] = self.snapshot(suite)
        spec = canonical({"source": source, "recipe": list(recipe), "rules": rules, "limits": policy,
                          "tool_commands": {tool: list(TOOLS[tool]) for tool in recipe}})
        if policy["isolation"] == "docker":
            specification = json.loads(spec)
            specification["container_image_id"] = container_image()
            spec = canonical(specification)
        if memory_subject is not None:
            specification = json.loads(spec)
            specification["memory_subject"] = memory_subject
            spec = canonical(specification)
        task_id = hashlib.sha256(key.encode()).hexdigest()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            configured = self.db.execute(
                "SELECT value FROM runtime_settings WHERE name='concurrency'").fetchone()
            if configured and json.loads(configured["value"]) != policy["concurrency"]:
                raise ValueError("Concurrency must match this runtime's persisted limit")
            self.db.execute("INSERT OR IGNORE INTO runtime_settings VALUES ('concurrency',?)",
                            (canonical(policy["concurrency"]),))
            old = self.db.execute("SELECT spec FROM tasks WHERE id=?", (task_id,)).fetchone()
            comparison = json.loads(spec)
            if old:
                old_specification = json.loads(old["spec"])
                old_specification["limits"] = {**DEFAULTS, **old_specification["limits"]}
            if old and "tool_commands" not in json.loads(old["spec"]):
                comparison.pop("tool_commands", None)
            if old and old_specification != comparison:
                raise ValueError("Idempotency key already bound to a different task")
            if not old:
                self.db.execute("INSERT INTO tasks(id,spec,state) VALUES (?,?,'pending')", (task_id, spec))
                self.db.execute("INSERT INTO task_controls(task_id,deadline_at) VALUES (?,?)",
                                (task_id, time.time() + policy["total_seconds"]))
                self.event(task_id, "submitted", {"recipe": list(recipe)})
            if usage is not None:
                ledger = (usage.get("tokens"), usage.get("cost"), canonical(usage["evidence"]))
                old_usage = self.db.execute(
                    "SELECT tokens,cost,evidence FROM usage WHERE task_id=?", (task_id,)).fetchone()
                if old_usage and tuple(old_usage) != ledger:
                    raise ValueError("Task usage is already bound to different records")
                self.db.execute("INSERT OR IGNORE INTO usage VALUES (?,?,?,?)", (task_id, *ledger))
        return task_id

    @contextmanager
    def task_lock(self, key):
        with (self.root / (key + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Task is already executing") from error
            yield

    @contextmanager
    def tool_slot(self, count):
        slot = None
        for index in range(count):
            candidate = (self.root / f"slot-{index}.lock").open("a")
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                slot = candidate
                break
            except BlockingIOError:
                candidate.close()
        if slot is None:
            raise ToolCapacityError("Tool concurrency limit reached")
        try:
            yield
        finally:
            slot.close()

    def snapshot(self, folder):
        if not folder.is_dir():
            raise ValueError("Checkpoint directory is missing")
        result = {}
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError("Checkpoint cannot contain symlinks")
            if path.is_file():
                result[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def status(self, key=None):
        query = ("SELECT t.id,t.state,t.next_step,t.calls,t.seconds,t.recoveries,t.error,"
                 "c.cancel_requested,c.deadline_at,c.attempts,c.retryable,t.spec FROM tasks t "
                 "LEFT JOIN task_controls c ON c.task_id=t.id")
        has_effects = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='external_effects'").fetchone()
        def public(row):
            result = dict(row)
            result["total_steps"] = len(json.loads(result.pop("spec"))["recipe"])
            result["external_operations"] = (
                [dict(effect) for effect in self.db.execute(
                    "SELECT step,operation,state FROM external_effects WHERE task_id=? ORDER BY step",
                    (result["id"],))] if has_effects else [])
            return result
        if key is None:
            return [public(row) for row in self.db.execute(query + " ORDER BY t.rowid")]
        row = self.db.execute(query + " WHERE t.id=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Unknown task")
        return public(row)

    def cancel(self, key):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT state FROM tasks WHERE id=?", (key,)).fetchone()
            if row is None:
                raise ValueError("Unknown task")
            if row["state"] in {"succeeded", "rejected", "cancelled", "timed_out"}:
                return self.status(key)
            self.db.execute("INSERT OR IGNORE INTO task_controls(task_id) VALUES (?)", (key,))
            self.db.execute("UPDATE task_controls SET cancel_requested=1 WHERE task_id=?", (key,))
            self.event(key, "cancel_requested", {})
            if row["state"] in {"pending", "failed", "interrupted"}:
                self.transition(key, "cancelled")
        return self.status(key)

    def guard(self, key):
        control = self.db.execute("SELECT * FROM task_controls WHERE task_id=?", (key,)).fetchone()
        if control is not None:
            if control["cancel_requested"]:
                raise TaskCancelled("Task cancellation requested")
            if control["deadline_at"] is not None and time.time() >= control["deadline_at"]:
                raise TaskDeadlineExceeded("Task total wall-time budget exhausted")

    def run(self, key, *, interrupt_after=None, max_attempts=None):
        if max_attempts is not None and (type(max_attempts) is not int or max_attempts < 1):
            raise ValueError("Attempt limit must be a positive integer")
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("Invalid task ID")
        with self.task_lock(key):
            row = self.db.execute("SELECT * FROM tasks WHERE id=?", (key,)).fetchone()
            if row is None:
                raise ValueError("Unknown task")
            spec = json.loads(row["spec"])
            if spec["limits"]["isolation"] == "docker":
                for event in self.db.execute(
                    "SELECT payload FROM events WHERE task_id=? AND kind='container_reserved'", (key,)):
                    remove_container([shutil.which("docker") or "docker", "--name",
                                      json.loads(event["payload"])["name"]])
            root = self.root / key
            root.mkdir(exist_ok=True)
            if row["state"] in {"cancelled", "timed_out"}:
                return dict(row)
            if row["state"] in {"succeeded", "rejected"}:
                self.check_checkpoint(key, root, row["next_step"])
                return dict(row)
            recovering = row["state"] != "pending"
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                current = self.db.execute("SELECT state FROM tasks WHERE id=?", (key,)).fetchone()[0]
                if current == "cancelled":
                    return dict(self.db.execute("SELECT * FROM tasks WHERE id=?", (key,)).fetchone())
                self.db.execute("INSERT OR IGNORE INTO task_controls(task_id) VALUES (?)", (key,))
                attempts = self.db.execute(
                    "SELECT attempts FROM task_controls WHERE task_id=?", (key,)).fetchone()[0]
                if max_attempts is not None and attempts >= max_attempts:
                    return dict(self.db.execute("SELECT * FROM tasks WHERE id=?", (key,)).fetchone())
                self.db.execute("UPDATE task_controls SET attempts=attempts+1 WHERE task_id=?", (key,))
                if current == "running":
                    self.transition(key, "interrupted")
                self.transition(key, "running")
                if recovering:
                    self.db.execute("UPDATE tasks SET recoveries=recoveries+1 WHERE id=?", (key,))
                    self.event(key, "resume", {"step": row["next_step"]})
            started = time.monotonic()
            try:
                self.guard(key)
                self.execute_steps(key, root, row, spec, interrupt_after)
            except Exception as error:
                with self.db:
                    state = self.db.execute("SELECT state FROM tasks WHERE id=?", (key,)).fetchone()[0]
                    if state in {"running", "verifying"}:
                        target = ("cancelled" if isinstance(error, TaskCancelled) else
                                  "timed_out" if isinstance(error, TaskDeadlineExceeded) else "failed")
                        self.transition(key, target)
                    retryable = isinstance(error, (OSError, subprocess.TimeoutExpired,
                                                   ToolCapacityError, GatewayError))
                    self.db.execute("UPDATE task_controls SET retryable=? WHERE task_id=?",
                                    (int(retryable), key))
                    self.db.execute("UPDATE tasks SET error=? WHERE id=?", (str(error), key))
                    self.event(key, "error", {"message": str(error), "type": type(error).__name__,
                                              "retryable": retryable})
            finally:
                with self.db:
                    self.db.execute("UPDATE tasks SET seconds=seconds+? WHERE id=?",
                                    (time.monotonic() - started, key))
            return dict(self.db.execute("SELECT * FROM tasks WHERE id=?", (key,)).fetchone())

    def check_checkpoint(self, key, root, step):
        record = self.db.execute("SELECT manifest FROM checkpoints WHERE task_id=? AND step=?",
                                 (key, step)).fetchone()
        folder = root / f"checkpoint-{step}"
        if record is None or canonical(self.snapshot(folder)) != record["manifest"]:
            raise ValueError("Checkpoint content changed")
        return folder

    def execute_steps(self, key, root, row, spec, interrupt_after):
        limits = {**DEFAULTS, **spec["limits"]}
        if "container_image_id" in spec:
            limits["_image_id"] = spec["container_image_id"]
        step = row["next_step"]
        if step == 0 and not self.db.execute(
            "SELECT 1 FROM checkpoints WHERE task_id=? AND step=0", (key,)
        ).fetchone():
            folder = root / "checkpoint-0"
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir()
            (folder / "input.txt").write_text(spec["source"], encoding="utf-8")
            with self.db:
                self.db.execute("INSERT INTO checkpoints VALUES (?,?,?)",
                                (key, 0, canonical(self.snapshot(folder))))
        checkpoint = self.check_checkpoint(key, root, step)
        for index in range(step, len(spec["recipe"])):
            self.guard(key)
            if interrupt_after is not None and index == interrupt_after:
                with self.db:
                    self.transition(key, "interrupted")
                    self.event(key, "injected_interrupt", {"step": index})
                return
            staging = Path(tempfile.mkdtemp(prefix="staging-", dir=root))
            shutil.copytree(checkpoint, staging, dirs_exist_ok=True)
            tool = spec["recipe"][index]
            if tool not in self.permissions["tools"]:
                raise ValueError("Tool permission was revoked")
            if "tool_commands" in spec and list(TOOLS[tool]) != spec["tool_commands"][tool]:
                raise ValueError("Registered tool implementation changed since submission")
            with self.tool_slot(limits["concurrency"]):
                with self.db:
                    calls = self.db.execute("SELECT calls FROM tasks WHERE id=?", (key,)).fetchone()[0]
                    if calls >= limits["max_calls"]:
                        raise RuntimeError("Tool call budget exhausted")
                    self.db.execute("UPDATE tasks SET calls=calls+1 WHERE id=?", (key,))
                    self.event(key, "tool_started", {"step": index, "tool": tool})
                if tool == "http_write":
                    receipt = execute_external(self, key, index,
                                               json.loads((staging / "input.txt").read_text()), limits, lambda: self.guard(key))
                    (staging / "output.txt").write_text(canonical(receipt) + "\n")
                else:
                    command = [sys.executable, "-I", str(Path(__file__).with_name("sandbox_worker.py")),
                               canonical(limits), *TOOLS[tool]]
                    command = isolate(command, staging, limits["isolation"], limits)
                    if limits["isolation"] == "docker":
                        with self.db:
                            self.event(key, "container_reserved",
                                       {"name": command[command.index("--name") + 1]})
                    with (root / "tool.log").open("wb") as log:
                        process = subprocess.Popen(command, cwd=staging, stdout=log, stderr=log,
                                                   env={"PATH": os.defpath, "LC_ALL": "C"},
                                                   start_new_session=True)
                        try:
                            code = wait_process(process, limits["timeout_seconds"],
                                                check=lambda: self.guard(key))
                        finally:
                            remove_container(command)
                    if code:
                        raise RuntimeError(f"Tool {tool} exited with {code}")
            size = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
            if size > limits["workspace_bytes"]:
                raise RuntimeError("Workspace byte budget exceeded")
            manifest = canonical(self.snapshot(staging))
            final = root / f"checkpoint-{index + 1}"
            if final.exists():
                shutil.rmtree(final)
            staging.rename(final)
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                self.guard(key)
                self.db.execute("INSERT INTO checkpoints VALUES (?,?,?)", (key, index + 1, manifest))
                self.db.execute("UPDATE tasks SET next_step=?,error=NULL WHERE id=?", (index + 1, key))
                self.event(key, "checkpoint", {"step": index + 1, "manifest": json.loads(manifest)})
            checkpoint = final
        with self.db:
            self.guard(key)
            self.transition(key, "verifying")
        for rule in spec["rules"]:
            if rule["kind"] == "tests" and self.snapshot(Path(rule["suite"])) != rule["suite_manifest"]:
                raise ValueError("Independent test suite changed")
        before = self.snapshot(checkpoint)
        with self.tool_slot(limits["concurrency"]):
            checks = [verify(checkpoint, {**rule, "_limits": limits,
                                         "_allowed_origins": self.permissions["origins"],
                                         "_control_check": lambda: self.guard(key)})
                      for rule in spec["rules"]]
        for rule in spec["rules"]:
            if rule["kind"] == "tests" and self.snapshot(Path(rule["suite"])) != rule["suite_manifest"]:
                raise ValueError("Independent test suite changed during verification")
        if self.snapshot(checkpoint) != before:
            raise ValueError("Verifier modified checkpoint artifacts")
        passed = all(check["passed"] for check in checks)
        memory = GovernedMemory(self.root / "memory.sqlite")
        try:
            memory.remember(spec.get("memory_subject", "recipe:" + ",".join(spec["recipe"])), {"recipe": spec["recipe"], "passed": passed},
                            confidence=0.9, ttl_seconds=86400 * 30,
                            provenance={"source": key, "checkpoint": json.loads(canonical(self.snapshot(checkpoint)))})
        finally:
            memory.close()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.guard(key)
            self.db.execute("UPDATE tasks SET verification=? WHERE id=?", (canonical(checks), key))
            self.transition(key, "succeeded" if passed else "rejected")
            self.db.execute("UPDATE tasks SET error=NULL WHERE id=?", (key,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    initialize = sub.add_parser("init")
    initialize.add_argument("--allow-tool", action="append", default=[])
    initialize.add_argument("--allow-origin", action="append", default=[])
    plan = sub.add_parser("plan")
    plan.add_argument("plan", type=Path)
    plan.add_argument("--model")
    plan.add_argument("--token-price", type=float)
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("plan", type=Path)
    submit = sub.add_parser("submit")
    submit.add_argument("plan", type=Path)
    memory = sub.add_parser("memory")
    memory.add_argument("subject")
    memory.add_argument("--require-calibrated", action="store_true")
    memory.add_argument("--min-confidence", type=float, default=0.5)
    status = sub.add_parser("status")
    status.add_argument("task_id", nargs="?")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("task_id")
    events = sub.add_parser("events")
    events.add_argument("task_id")
    resume = sub.add_parser("resume")
    resume.add_argument("task_id")
    args = parser.parse_args()
    permissions = ({"tools": args.allow_tool, "origins": args.allow_origin}
                   if args.action == "init" else None)
    runtime = Runtime(args.root, permissions=permissions)
    try:
        if args.action == "init":
            print(canonical(runtime.permissions))
            return
        if args.action in {"status", "cancel"}:
            result = runtime.status(args.task_id) if args.action == "status" else runtime.cancel(args.task_id)
            print(json.dumps(result, indent=2))
            return
        if args.action == "memory":
            memory = GovernedMemory(runtime.root / "memory.sqlite")
            try:
                print(canonical(memory.recall(args.subject, min_confidence=args.min_confidence,
                                                     require_calibrated=args.require_calibrated)))
            finally:
                memory.close()
            return
        if args.action == "plan":
            from .runtime_planning import submit as submit_planned
            plan = json.loads(args.plan.read_text())
            key = submit_planned(runtime, plan["idempotency_key"], plan["source"], plan["family"],
                                 plan["rules"], model=args.model, limits=plan.get("limits"),
                                 token_price=args.token_price)
            result = runtime.run(key)
        elif args.action in {"submit", "enqueue"}:
            plan = json.loads(args.plan.read_text())
            key = runtime.submit(plan["idempotency_key"], plan["source"], plan["recipe"],
                                 plan["rules"], limits=plan.get("limits"))
            if args.action == "enqueue":
                print(json.dumps(runtime.status(key), indent=2))
                return
            result = runtime.run(key)
        elif args.action == "events":
            print(json.dumps([dict(row) for row in runtime.db.execute(
                "SELECT * FROM events WHERE task_id=? ORDER BY seq", (args.task_id,)
            )], indent=2))
            return
        else:
            result = runtime.run(args.task_id)
        print(json.dumps(result, indent=2))
        if result["state"] != "succeeded":
            raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
