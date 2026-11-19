"""Persist planning decisions, governed memory selection and model usage."""

import hashlib
import json
import math
import time

from .durable import canonical
from .governance import GovernedMemory
from .runtime import HINTS, POLICIES, Task, model_recipe


def submit(runtime, key, source, family, rules, *, model=None, limits=None, token_price=None):
    if family not in HINTS:
        raise ValueError("Unknown task family")
    if token_price is not None and (type(token_price) not in (int, float)
                                    or not math.isfinite(token_price) or token_price < 0):
        raise ValueError("Token price must be finite and nonnegative")
    binding = canonical({"source": source, "family": family, "rules": rules,
                         "model": model, "limits": limits, "token_price": token_price})
    task_id = hashlib.sha256(key.encode()).hexdigest()
    subject = "family:" + family
    runtime.db.execute("""CREATE TABLE IF NOT EXISTS planning (
        task_id TEXT PRIMARY KEY, binding TEXT, state TEXT, decision TEXT)""")
    runtime.db.commit()
    with runtime.task_lock(task_id + ".planner"):
        previous = runtime.db.execute("SELECT * FROM planning WHERE task_id=?", (task_id,)).fetchone()
        if previous is not None and previous["binding"] != binding:
            raise ValueError("Planning idempotency key is bound to different inputs")
        if previous is not None and previous["state"] == "resolved":
            decision = json.loads(previous["decision"])
        else:
            memory = GovernedMemory(runtime.root / "memory.sqlite")
            try:
                recalled = memory.recall(subject)
            finally:
                memory.close()
            recipe = None
            selection = recalled["selected"]
            if (selection and isinstance(selection["value"], dict)
                    and selection["value"].get("passed") is True):
                recipe = selection["value"].get("recipe")
            transport = []
            tokens = 0
            evidence = {"memory": recalled, "method": "memory" if recipe else "local",
                        "model": model, "started_at": time.time()}
            if previous is not None:
                # An interrupted HTTP request has unknown billing. Avoid issuing it twice.
                recipe = None
                if previous["decision"]:
                    evidence["started_at"] = json.loads(previous["decision"])["started_at"]
                tokens = None
                evidence["method"] = "interrupted_request_fallback"
            elif recipe is None and model:
                with runtime.db:
                    runtime.db.execute("INSERT INTO planning VALUES (?,?,'requesting',?)",
                                       (task_id, binding, canonical({"started_at": evidence["started_at"]})))
                evidence["method"] = "model"
                try:
                    recipe, tokens = model_recipe(Task(family, source, ""), [], [], model, transport)
                except (OSError, ValueError, RuntimeError) as error:
                    evidence["error"] = str(error)
                    tokens = None
                if len(transport) > 1:
                    tokens = None
                evidence["transport"] = transport
            if (not recipe or any(tool not in runtime.permissions["tools"] for tool in recipe)):
                recipe = POLICIES[family][-1]
                evidence["fallback"] = True
            cost = tokens * token_price if tokens is not None and token_price is not None else None
            decision = {"recipe": list(recipe), "tokens": tokens, "cost": cost, "evidence": evidence}
            with runtime.db:
                runtime.db.execute("INSERT OR REPLACE INTO planning VALUES (?,?,'resolved',?)",
                                   (task_id, binding, canonical(decision)))
        submitted = runtime.submit(key, source, decision["recipe"], rules,
                                   limits=limits, memory_subject=subject)
        with runtime.db:
            runtime.db.execute("INSERT OR IGNORE INTO usage VALUES (?,?,?,?)",
                               (submitted, decision["tokens"], decision["cost"], canonical(decision["evidence"])))
        return submitted


def export_record(runtime, task_id, *, success):
    if type(success) is not bool:
        raise ValueError("An independent boolean success label is required")
    task = runtime.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if task is None or task["state"] not in {"succeeded", "rejected", "failed"}:
        raise ValueError("Only terminal task attempts can be exported")
    first = runtime.db.execute(
        "SELECT min(timestamp) FROM events WHERE task_id=? AND kind='submitted'", (task_id,)).fetchone()[0]
    last = runtime.db.execute(
        "SELECT max(timestamp) FROM events WHERE task_id=?", (task_id,)).fetchone()[0]
    usage = runtime.db.execute("SELECT * FROM usage WHERE task_id=?", (task_id,)).fetchone()
    if usage:
        first = min(first, json.loads(usage["evidence"]).get("started_at", first))
    if task["verification"] is None:
        verdict = None
    else:
        verdict = all(check["passed"] for check in json.loads(task["verification"]))
    return {"task_id": task_id, "success": success, "verifier_passed": verdict,
            "tool_calls": task["calls"], "seconds": max(0, last - first),
            "active_seconds": task["seconds"], "recoveries": task["recoveries"],
            "tokens": usage["tokens"] if usage else 0, "cost": usage["cost"] if usage else None,
            "latency_scope": "submission_to_last_terminal_event_including_recovery_downtime"}
