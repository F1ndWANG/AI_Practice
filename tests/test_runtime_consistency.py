"""Regression cases for atomic submission, retry races and consistent export snapshots."""

import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent.governance import GovernedMemory
from rsi_agent.runtime_export import export
from rsi_agent.runtime_planning import export_record
from rsi_agent.task_worker import drain


class ConsistencyTests(unittest.TestCase):
    def test_usage_failure_rolls_back_task_controls_and_events(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            runtime.db.execute("""CREATE TRIGGER fail_usage BEFORE INSERT ON usage
                BEGIN SELECT RAISE(ABORT,'injected ledger failure'); END""")
            runtime.db.commit()
            usage = {"tokens": 10, "cost": 0.01, "evidence": {"method": "fixture"}}
            with self.assertRaises(sqlite3.IntegrityError):
                runtime.submit("atomic", "input", ["copy"], [{"kind": "file"}], usage=usage)
            for table in ("tasks", "task_controls", "events", "usage", "runtime_settings"):
                self.assertEqual(runtime.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
            runtime.db.execute("DROP TRIGGER fail_usage")
            runtime.db.commit()
            key = runtime.submit("atomic", "input", ["copy"], [{"kind": "file"}], usage=usage)
            with self.assertRaises(ValueError):
                runtime.submit("atomic", "input", ["copy"], [{"kind": "file"}],
                               usage={**usage, "tokens": 11})
            self.assertEqual(runtime.db.execute("SELECT tokens FROM usage WHERE task_id=?", (key,)).fetchone()[0], 10)
            runtime.close()

    def test_missing_checkpoint_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            with self.assertRaisesRegex(ValueError, "directory is missing"):
                runtime.snapshot(Path(directory) / "absent")
            runtime.close()

    def test_concurrent_same_key_submission_creates_one_task(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            runtime.close()
            barrier = threading.Barrier(6)

            def submit(_):
                runtime = Runtime(directory)
                try:
                    barrier.wait(timeout=10)
                    return runtime.submit("shared", "input", ["copy"], [{"kind": "file"}])
                finally:
                    runtime.close()

            with ThreadPoolExecutor(max_workers=6) as pool:
                keys = list(pool.map(submit, range(6)))
            self.assertEqual(len(set(keys)), 1)
            runtime = Runtime(directory)
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 1)
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM events WHERE kind='submitted'").fetchone()[0], 1)
            runtime.close()

    def test_stale_worker_snapshot_cannot_exceed_attempt_limit(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "rsi_agent.durable.TOOLS", {"bad_fixture": (sys.executable, "-c", "raise SystemExit(1)")}
        ):
            runtime = Runtime(directory)
            key = runtime.submit("race", "", ["bad_fixture"], [{"kind": "file"}])
            runtime.close()
            original = Runtime.run
            barrier = threading.Barrier(2)
            completed = threading.Event()
            counter_lock = threading.Lock()
            order = []

            def scheduled(self, task, **kwargs):
                with counter_lock:
                    position = len(order)
                    order.append(position)
                barrier.wait(timeout=10)
                if position == 0:
                    try:
                        return original(self, task, **kwargs)
                    finally:
                        completed.set()
                if not completed.wait(timeout=10):
                    raise RuntimeError("First worker did not finish")
                return original(self, task, **kwargs)

            def worker():
                runtime = Runtime(directory)
                try:
                    return drain(runtime, max_attempts=1, retry_delay=0)
                finally:
                    runtime.close()

            with patch.object(Runtime, "run", scheduled), ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker) for _ in range(2)]
                for future in futures:
                    future.result()
            runtime = Runtime(directory)
            self.assertEqual(runtime.status(key)["attempts"], 1)
            self.assertEqual(runtime.status(key)["calls"], 1)
            runtime.close()

    def test_report_uses_one_snapshot_while_another_worker_retries(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "rsi_agent.durable.TOOLS", {"bad_fixture": (sys.executable, "-c", "raise SystemExit(1)")}
        ):
            root = Path(directory)
            runtime = Runtime(root / "runtime")
            first = runtime.submit("first", "ok", ["copy"], [{"kind": "file"}])
            second = runtime.submit("second", "", ["bad_fixture"], [{"kind": "file"}])
            runtime.run(first)
            runtime.run(second)
            runtime.close()

            def concurrent_export(runtime, key, *, success):
                result = export_record(runtime, key, success=success)
                if key == first:
                    writer = Runtime(root / "runtime")
                    writer.run(second)
                    writer.close()
                return result

            with patch("rsi_agent.runtime_export.export_record", concurrent_export):
                export(root / "runtime", {first: True, second: False}, root / "report")
            records = [json.loads(line) for line in (root / "report/records.jsonl").read_text().splitlines()]
            self.assertEqual(records[1]["tool_calls"], 1)
            runtime = Runtime(root / "runtime")
            self.assertEqual(runtime.status(second)["calls"], 2)
            runtime.close()

    def test_conflicting_calibration_batch_rolls_back_every_new_label(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = GovernedMemory(Path(directory) / "memory.sqlite")
            value = {"recipe": ["copy"], "passed": True}
            memory.remember("test", value, confidence=0.9, ttl_seconds=100, provenance={"source": "fixture"})
            memory.observe("test", value, observation_id="old", success=True, provenance={"source": "fixture"})
            with self.assertRaises(ValueError):
                memory.observe_many([
                    {"subject": "test", "value": value, "observation_id": "new", "success": True,
                     "provenance": {"source": "fixture"}},
                    {"subject": "test", "value": value, "observation_id": "old", "success": False,
                     "provenance": {"source": "fixture"}},
                ])
            self.assertEqual(memory.db.execute("SELECT count(*) FROM memory_observations").fetchone()[0], 1)
            memory.close()
