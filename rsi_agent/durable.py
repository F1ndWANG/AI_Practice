"""Persistent, checkpointed execution of the registered local Agent tools."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .governance import GovernedMemory, verify
from .runtime import TOOLS
from .live_verifier import origin
from .sandbox_worker import isolate

DEFAULTS = {"max_calls": 6, "timeout_seconds": 3, "cpu_seconds": 3,
            "file_bytes": 1048576, "workspace_bytes": 4194304,
            "memory_bytes": 536870912, "concurrency": 2, "isolation": "process"}
TRANSITIONS = {
    "pending": {"running"}, "running": {"interrupted", "failed", "verifying"},
    "interrupted": {"running"}, "failed": {"running"},
    "verifying": {"running", "succeeded", "rejected"}, "rejected": set(), "succeeded": set(),
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
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL,
                next_step INTEGER NOT NULL DEFAULT 0, calls INTEGER NOT NULL DEFAULT 0,
                seconds REAL NOT NULL DEFAULT 0, recoveries INTEGER NOT NULL DEFAULT 0,
                verification TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, timestamp REAL,
                kind TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS usage (
                task_id TEXT PRIMARY KEY, tokens INTEGER, cost REAL, evidence TEXT);
            CREATE TABLE IF NOT EXISTS checkpoints (
                task_id TEXT, step INTEGER, manifest TEXT, PRIMARY KEY(task_id,step));
        """)
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

    def submit(self, key, source, recipe, rules, *, limits=None, memory_subject=None):
        if not isinstance(key, str) or not key:
            raise ValueError("A nonempty idempotency key is required")
        policy = {**DEFAULTS, **(limits or {})}
        if set(policy) != set(DEFAULTS) or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for name, value in policy.items() if name != "isolation"
        ) or any(type(policy[name]) is not int for name in policy if name not in {"timeout_seconds", "isolation"}):
            raise ValueError("Invalid resource limits")
        if policy["isolation"] not in {"process", "bubblewrap"}:
            raise ValueError("Unknown isolation mode")
        isolate(["true"], self.root, policy["isolation"])
        if not recipe or any(tool not in TOOLS or tool not in self.permissions["tools"] for tool in recipe):
            raise ValueError("Recipe must contain registered tools")
        if not rules or any(rule.get("kind") not in
                            {"file", "text", "data", "tests", "page", "page_live", "paper"} for rule in rules):
            raise ValueError("Provide objective verification rules")
        if len(source.encode()) > policy["workspace_bytes"]:
            raise ValueError("Input exceeds workspace budget")
        with (self.root / "policy.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            policy_file = self.root / "concurrency.json"
            if policy_file.exists():
                if json.loads(policy_file.read_text()) != policy["concurrency"]:
                    raise ValueError("Concurrency must match this runtime's persisted limit")
            else:
                policy_file.write_text(json.dumps(policy["concurrency"]))
        rules = json.loads(canonical(rules))
        for rule in rules:
            if rule["kind"] == "page_live":
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
        if memory_subject is not None:
            specification = json.loads(spec)
            specification["memory_subject"] = memory_subject
            spec = canonical(specification)
        task_id = hashlib.sha256(key.encode()).hexdigest()
        with self.db:
            old = self.db.execute("SELECT spec FROM tasks WHERE id=?", (task_id,)).fetchone()
            comparison = json.loads(spec)
            if old and "tool_commands" not in json.loads(old["spec"]):
                comparison.pop("tool_commands", None)
            if old and json.loads(old["spec"]) != comparison:
                raise ValueError("Idempotency key already bound to a different task")
            if not old:
                self.db.execute("INSERT INTO tasks(id,spec,state) VALUES (?,?,'pending')", (task_id, spec))
                self.event(task_id, "submitted", {"recipe": list(recipe)})
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
            raise RuntimeError("Tool concurrency limit reached")
        try:
            yield
        finally:
            slot.close()

    def snapshot(self, folder):
        result = {}
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError("Checkpoint cannot contain symlinks")
            if path.is_file():
                result[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def run(self, key, *, interrupt_after=None):
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("Invalid task ID")
        with self.task_lock(key):
            row = self.db.execute("SELECT * FROM tasks WHERE id=?", (key,)).fetchone()
            if row is None:
                raise ValueError("Unknown task")
            spec = json.loads(row["spec"])
            root = self.root / key
            root.mkdir(exist_ok=True)
            if row["state"] in {"succeeded", "rejected"}:
                self.check_checkpoint(key, root, row["next_step"])
                return dict(row)
            recovering = row["state"] != "pending"
            with self.db:
                if row["state"] == "running":
                    self.transition(key, "interrupted")
                self.transition(key, "running")
                if recovering:
                    self.db.execute("UPDATE tasks SET recoveries=recoveries+1 WHERE id=?", (key,))
                    self.event(key, "resume", {"step": row["next_step"]})
            started = time.monotonic()
            try:
                self.execute_steps(key, root, row, spec, interrupt_after)
            except Exception as error:
                with self.db:
                    state = self.db.execute("SELECT state FROM tasks WHERE id=?", (key,)).fetchone()[0]
                    if state in {"running", "verifying"}:
                        if state == "verifying":
                            self.transition(key, "running")
                        self.transition(key, "failed")
                    self.db.execute("UPDATE tasks SET error=? WHERE id=?", (str(error), key))
                    self.event(key, "error", {"message": str(error)})
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
        limits = spec["limits"]
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
                command = [sys.executable, "-I", str(Path(__file__).with_name("sandbox_worker.py")),
                           canonical(limits), *TOOLS[tool]]
                command = isolate(command, staging, limits["isolation"])
                with (root / "tool.log").open("wb") as log:
                    process = subprocess.Popen(command, cwd=staging, stdout=log, stderr=log,
                                               env={"PATH": os.defpath, "LC_ALL": "C"},
                                               start_new_session=True)
                    try:
                        code = process.wait(timeout=limits["timeout_seconds"])
                    except BaseException:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
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
                self.db.execute("INSERT INTO checkpoints VALUES (?,?,?)", (key, index + 1, manifest))
                self.db.execute("UPDATE tasks SET next_step=?,error=NULL WHERE id=?", (index + 1, key))
                self.event(key, "checkpoint", {"step": index + 1, "manifest": json.loads(manifest)})
            checkpoint = final
        with self.db:
            self.transition(key, "verifying")
        for rule in spec["rules"]:
            if rule["kind"] == "tests" and self.snapshot(Path(rule["suite"])) != rule["suite_manifest"]:
                raise ValueError("Independent test suite changed")
        before = self.snapshot(checkpoint)
        with self.tool_slot(limits["concurrency"]):
            checks = [verify(checkpoint, {**rule, "_limits": limits,
                                         "_allowed_origins": self.permissions["origins"]})
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
    submit = sub.add_parser("submit")
    submit.add_argument("plan", type=Path)
    memory = sub.add_parser("memory")
    memory.add_argument("subject")
    memory.add_argument("--min-confidence", type=float, default=0.5)
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
        if args.action == "memory":
            memory = GovernedMemory(runtime.root / "memory.sqlite")
            try:
                print(canonical(memory.recall(args.subject, min_confidence=args.min_confidence)))
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
        elif args.action == "submit":
            plan = json.loads(args.plan.read_text())
            key = runtime.submit(plan["idempotency_key"], plan["source"], plan["recipe"],
                                 plan["rules"], limits=plan.get("limits"))
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
