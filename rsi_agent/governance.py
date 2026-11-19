"""Provenance-aware memory and objective artifact verifiers."""

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import sqlite3
import time
from html.parser import HTMLParser
from pathlib import Path

from .sandbox_worker import isolate
from .live_verifier import fetch_page


class GovernedMemory:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("""CREATE TABLE IF NOT EXISTS governed_memory (
            subject TEXT, digest TEXT, value TEXT, confidence REAL, expires REAL,
            provenance TEXT, created REAL, PRIMARY KEY(subject, digest, provenance))""")
        self.db.commit()

    def close(self):
        self.db.close()

    def remember(self, subject, value, *, confidence, ttl_seconds, provenance):
        if (not subject or not isinstance(provenance, dict) or not provenance.get("source")
                or type(confidence) not in (float, int) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1 or not math.isfinite(ttl_seconds) or ttl_seconds <= 0):
            raise ValueError("Memory requires source provenance, confidence in [0,1] and positive TTL")
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        source = json.dumps(provenance, sort_keys=True)
        now = time.time()
        with self.db:
            # Repeated evidence does not gain confidence or extend its expiration.
            self.db.execute("INSERT OR IGNORE INTO governed_memory VALUES (?,?,?,?,?,?,?)",
                            (subject, digest, encoded, confidence, now + ttl_seconds, source, now))
        return digest

    def recall(self, subject, *, min_confidence=0.5, now=None):
        now = time.time() if now is None else now
        rows = self.db.execute(
            "SELECT digest,value,confidence,expires,provenance FROM governed_memory "
            "WHERE subject=? AND expires>? AND confidence>=? ORDER BY confidence DESC,digest",
            (subject, now, min_confidence)).fetchall()
        candidates = {}
        for digest, value, confidence, expires, source in rows:
            candidate = candidates.setdefault(digest, {
                "value": json.loads(value), "confidence": confidence, "expires": expires,
                "provenance": []})
            candidate["provenance"].append(json.loads(source))
        values = list(candidates.values())
        if not values:
            return {"state": "missing", "selected": None, "candidates": []}
        tied = sum(item["confidence"] == values[0]["confidence"] for item in values) > 1
        return {"state": "conflict" if tied else "resolved",
                "selected": None if tied else values[0], "candidates": values}


def artifact(root, relative):
    path = (Path(root) / relative).resolve()
    if Path(root).resolve() not in path.parents or not path.is_file():
        raise ValueError("Verification artifact must be a file within the checkpoint")
    return path


class PageState(HTMLParser):
    def __init__(self, element_id):
        super().__init__()
        self.element_id = element_id
        self.found = False
        self.attributes = {}

    def handle_starttag(self, tag, attrs):
        if dict(attrs).get("id") == self.element_id:
            self.found = True
            self.attributes = dict(attrs)


def verify(root, rule):
    """Evaluate evidence without consulting the actor or its self-assessment."""
    kind = rule["kind"]
    if kind == "page_live":
        limits = rule.get("_limits", {})
        text, evidence = fetch_page(
            rule["url"], rule.get("_allowed_origins", []),
            timeout=limits.get("timeout_seconds", 3),
            max_bytes=limits.get("file_bytes", 1048576))
        parser = PageState(rule["element_id"])
        parser.feed(text)
        passed = parser.found and all(
            parser.attributes.get(name) == value for name, value in rule.get("attributes", {}).items())
        return {"passed": passed, "evidence": {"kind": kind, **evidence}}
    if kind == "tests":
        # The caller supplies an independent unittest suite. Output never decides its own verdict.
        suite = Path(rule["suite"]).resolve()
        if not suite.is_dir():
            raise ValueError("Independent test suite directory is missing")
        limits = rule.get("_limits", {})
        timeout = limits.get("timeout_seconds", 3)
        command = [sys.executable, "-I", str(Path(__file__).with_name("sandbox_worker.py")),
                   json.dumps({"cpu_seconds": limits.get("cpu_seconds", 3),
                               "file_bytes": limits.get("file_bytes", 1048576),
                               "memory_bytes": limits.get("memory_bytes", 536870912)}),
                   sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", str(suite)]
        command = isolate(command, root, limits.get("isolation", "process"))
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=log,
                                       env={"PATH": os.defpath, "LC_ALL": "C"}, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
            log.seek(0)
            output = log.read().decode("utf-8", errors="replace")
        counts = re.findall(r"Ran (\d+) tests?", output)
        passed = code == 0 and bool(counts) and int(counts[-1]) > 0 and "skipped=" not in output
        return {"passed": passed, "evidence": {
            "kind": kind, "suite": str(suite), "returncode": code,
            "tests_run": int(counts[-1]) if counts else 0, "output": output[-4000:],
            "suite_sha256": hashlib.sha256(b"".join(
                str(path.relative_to(suite)).encode() + path.read_bytes()
                for path in sorted(suite.rglob("*.py")))).hexdigest()}}
    path = artifact(root, rule.get("path", "output.txt"))
    raw = path.read_bytes()
    evidence = {"kind": kind, "path": rule.get("path", "output.txt"),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if kind == "file":
        passed = bool(raw) if "sha256" not in rule else evidence["sha256"] == rule["sha256"]
    elif kind == "text":
        passed = raw.decode("utf-8") == rule["expected"]
    elif kind == "data":
        rows = json.loads(raw)
        passed = isinstance(rows, list) and len(rows) >= rule.get("min_rows", 1)
        if passed:
            passed = all(isinstance(row, dict) and all(
                key in row and row[key] is not None for key in rule["required"]) for row in rows)
        if passed and rule.get("unique"):
            keys = [json.dumps([row[key] for key in rule["unique"]], sort_keys=True) for row in rows]
            passed = len(keys) == len(set(keys))
    elif kind == "page":
        parser = PageState(rule["element_id"])
        parser.feed(raw.decode("utf-8"))
        passed = parser.found
        evidence["scope"] = "saved_page_snapshot"
    elif kind == "paper":
        quote = rule["quote"]
        passed = bool(quote.strip()) and quote in raw.decode("utf-8")
        evidence["quote"] = quote
        evidence["scope"] = "literal_source_evidence"
    else:
        raise ValueError("Unknown verifier kind")
    return {"passed": bool(passed), "evidence": evidence}
