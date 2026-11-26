"""Read-only operational snapshots, coverage and actionable queue health."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent import runtime_health
from rsi_agent.workflow_worker import WorkflowWorker
from rsi_agent.workflows import Workflows


def graph():
    return {"limits": {"max_tool_calls": 1, "total_seconds": 60}, "nodes": [
        {"id": "one", "source": "private-source", "recipe": ["copy"], "rules": [{"kind": "file"}]}]}


class RuntimeHealthTests(unittest.TestCase):
    def test_inspection_does_not_initialize_policy_or_mutate_schema_and_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            workflows.submit("pending", graph())
            before = list(runtime.db.iterdump())
            report = runtime_health.inspect(directory)
            self.assertEqual(list(runtime.db.iterdump()), before)
            self.assertEqual(report["workflows"]["queue"]["eligible"], 1)
            self.assertFalse(report["capabilities"]["workflow_faults"])
            self.assertEqual(report["worker_liveness"], "unknown")
            self.assertNotIn("private-source", json.dumps(report))
            runtime.close()

    def test_queue_buckets_are_disjoint_and_expiry_is_a_separate_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            keys = {name: workflows.submit(name, graph()) for name in ("ready", "backoff", "isolated", "expired")}
            worker = WorkflowWorker(workflows)
            worker.record_failure(keys["backoff"], ValueError("not exported"))
            worker.record_failure(keys["isolated"], ValueError("not exported"))
            runtime.db.execute("UPDATE workflow_faults SET next_retry_at=?", (time.time() + 60,))
            runtime.db.execute("UPDATE workflow_faults SET quarantined=1 WHERE workflow_id=?", (keys["isolated"],))
            runtime.db.execute("UPDATE workflow_deadlines SET deadline_at=? WHERE workflow_id=?",
                               (time.time() - 1, keys["expired"]))
            runtime.db.commit()
            report = runtime_health.inspect(directory)
            self.assertEqual(report["workflows"]["queue"],
                             {"total": 4, "quarantined": 1, "backoff": 1, "eligible": 2, "expired": 1})
            self.assertTrue(report["attention_required"])
            self.assertEqual(len(report["issues"]), 2)
            self.assertNotIn("not exported", json.dumps(report))
            runtime.close()

    def test_usage_coverage_preserves_unknown_values(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            for key, usage in (("known", {"tokens": 10, "cost": 0.5, "evidence": {}}),
                               ("unknown", {"tokens": None, "cost": None, "evidence": {}}), ("absent", None)):
                runtime.submit(key, "private-source", ["copy"], [{"kind": "file"}], usage=usage)
            report = runtime_health.inspect(directory)
            self.assertEqual(report["usage"]["known_tokens"], 10)
            self.assertEqual(report["usage"]["known_cost"], 0.5)
            self.assertEqual(report["usage"]["token_coverage"], 1 / 3)
            self.assertEqual(report["usage"]["cost_coverage"], 1 / 3)
            runtime.close()

    def test_one_snapshot_prevents_mixed_task_and_workflow_states(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            workflows.submit("snapshot", graph())
            runtime.submit("task", "input", ["copy"], [{"kind": "file"}])
            writer = Runtime(directory)
            original = runtime_health._counts

            def update_after_task_read(db, table):
                result = original(db, table)
                if table == "tasks":
                    with writer.db:
                        writer.db.execute("UPDATE tasks SET state='failed'")
                        writer.db.execute("UPDATE workflows SET state='failed'")
                return result

            with patch.object(runtime_health, "_counts", update_after_task_read):
                report = runtime_health.inspect(directory)
            self.assertEqual(report["tasks"]["states"], {"pending": 1})
            self.assertEqual(report["workflows"]["states"], {"pending": 1})
            later = runtime_health.inspect(directory)
            self.assertEqual(later["tasks"]["states"], {"failed": 1})
            self.assertEqual(later["workflows"]["states"], {"failed": 1})
            writer.close()
            runtime.close()

    def test_completed_graph_at_exact_budget_limit_is_not_an_alert(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            self.assertTrue(workflows.run(workflows.submit("complete", graph()))["passed"])
            report = runtime_health.inspect(directory)
            self.assertEqual(report["workflows"]["budgets"]["tool_used"], 1)
            self.assertEqual(report["workflows"]["budgets"]["unfinished_at_limit"], 0)
            self.assertFalse(report["attention_required"])
            runtime.close()

    def test_legacy_database_without_optional_tables_is_not_migrated(self):
        with tempfile.TemporaryDirectory() as directory:
            db = sqlite3.connect(Path(directory) / "runtime.sqlite")
            db.execute("CREATE TABLE tasks (id TEXT,state TEXT,calls INTEGER,seconds REAL,recoveries INTEGER)")
            db.commit()
            db.close()
            report = runtime_health.inspect(directory)
            self.assertFalse(report["workflows"]["available"])
            self.assertIsNone(report["workflows"]["queue"])
            self.assertIsNone(report["usage"]["known_tokens"])
            self.assertIsNone(report["usage"]["token_coverage"])
            db = sqlite3.connect(Path(directory) / "runtime.sqlite")
            self.assertEqual(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("tasks",)])
            db.close()

    def test_cli_check_exit_codes_and_missing_root_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-m", "rsi_agent.runtime_health"]
            missing = Path(directory) / "missing"
            result = subprocess.run(command + [str(missing), "--check"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(missing.exists())
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("isolated", graph())
            worker = WorkflowWorker(workflows, max_errors=1)
            worker.record_failure(key, ValueError("poison"))
            result = subprocess.run(command + [directory, "--check"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(json.loads(result.stdout)["attention_required"])
            result = subprocess.run(command + [directory], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
