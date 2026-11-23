"""Persistent backoff, quarantine and concurrent stale-snapshot protection."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent.workflow_worker import WorkflowWorker
from rsi_agent.workflows import Workflows


def graph():
    return {"limits": {"max_tool_calls": 1}, "nodes": [
        {"id": "one", "source": "input", "recipe": ["copy"],
         "rules": [{"kind": "text", "expected": "input"}]}]}


class WorkflowFaultTests(unittest.TestCase):
    def test_cli_inspection_and_explicit_release_reuse_persisted_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("cli-release", graph())
            worker = WorkflowWorker(workflows, max_errors=1)
            with patch.object(Workflows, "run", side_effect=ValueError("controller fault")):
                worker.tick()
            command = [sys.executable, "-m", "rsi_agent.workflow_worker", directory]
            status = subprocess.run(command + ["--faults"], capture_output=True, text=True, timeout=10)
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertTrue(json.loads(status.stdout)["faults"][0]["quarantined"])
            invalid = subprocess.run(command + ["--release", key], capture_output=True, text=True, timeout=10)
            self.assertEqual(invalid.returncode, 2)
            released = subprocess.run(command + ["--release", key, "--reason", "Repaired controller"],
                                      capture_output=True, text=True, timeout=10)
            self.assertEqual(released.returncode, 0, released.stderr)
            self.assertEqual(json.loads(released.stdout)["id"], key)
            result = subprocess.run(command + ["--once"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["workflows"][0]["passed"])
            runtime.close()

    def test_backoff_and_quarantine_survive_restart_with_shared_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("poison-controller", graph())
            worker = WorkflowWorker(workflows, max_errors=3, retry_base=2, retry_max=3)
            with patch.object(Workflows, "run", side_effect=ValueError("poison")) as run:
                with patch("rsi_agent.workflow_worker.time.time", return_value=1000):
                    first = worker.tick()["errors"][0]["fault"]
                self.assertEqual(first["failures"], 1)
                self.assertEqual(first["next_retry_at"], 1002)
                with patch("rsi_agent.workflow_worker.time.time", return_value=1001):
                    self.assertEqual(worker.tick()["errors"], [])
                runtime.close()
                runtime = Runtime(directory)
                workflows = Workflows(runtime)
                worker = WorkflowWorker(workflows)
                self.assertEqual(worker.policy, {"max_errors": 3, "retry_base": 2, "retry_max": 3})
                with patch("rsi_agent.workflow_worker.time.time", return_value=1002):
                    second = worker.tick()["errors"][0]["fault"]
                self.assertEqual(second["next_retry_at"], 1005)
                with patch("rsi_agent.workflow_worker.time.time", return_value=1005):
                    third = worker.tick()["errors"][0]["fault"]
                self.assertTrue(third["quarantined"])
                with patch("rsi_agent.workflow_worker.time.time", return_value=10000):
                    self.assertEqual(worker.tick()["errors"], [])
                self.assertEqual(run.call_count, 3)
            self.assertEqual(worker.fault_status()["faults"][0]["workflow_id"], key)
            self.assertEqual(len(worker.fault_status()["events"]), 3)
            good = workflows.submit("healthy", graph())
            self.assertEqual(worker.tick()["workflows"][0]["id"], good)
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 1)
            runtime.close()

    def test_release_is_audited_and_does_not_reset_task_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("release", graph())
            task = runtime.submit(f"workflow:{key}:node:one", "input", ["copy"],
                                  [{"kind": "text", "expected": "input"}], workflow=(key, "one"))
            runtime.run(task, interrupt_after=0)
            attempts = runtime.status(task)["attempts"]
            worker = WorkflowWorker(workflows, max_errors=1)
            with patch.object(Workflows, "run", side_effect=ValueError("repair me")):
                worker.tick()
            self.assertTrue(worker.fault_status()["faults"][0]["quarantined"])
            with self.assertRaises(ValueError):
                worker.release(key, reason=" ")
            with self.assertRaises(ValueError):
                worker.release("../bad", reason="invalid")
            worker.release(key, reason="Repaired controller data")
            self.assertEqual(worker.fault_status()["faults"], [])
            event = worker.fault_status()["events"][0]
            self.assertEqual(event["kind"], "released")
            self.assertEqual(event["payload"]["reason"], "Repaired controller data")
            self.assertEqual(runtime.status(task)["attempts"], attempts)
            self.assertTrue(worker.tick()["workflows"][0]["passed"])
            self.assertEqual(workflows.status(key)["tool_budget"]["used"], 1)
            self.assertEqual(runtime.status(task)["attempts"], attempts + 1)
            runtime.close()

    def test_busy_lock_does_not_count_as_failure_or_clear_existing_fault(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("busy", graph())
            worker = WorkflowWorker(workflows)
            with patch.object(Workflows, "run", side_effect=ValueError("temporary")):
                worker.tick()
            runtime.db.execute("UPDATE workflow_faults SET next_retry_at=0")
            runtime.db.commit()
            with runtime.task_lock("workflow-" + key):
                result = worker.tick()
                with self.assertRaises(RuntimeError):
                    worker.release(key, reason="Cannot release an active workflow")
            self.assertEqual(result["busy"], [key])
            self.assertEqual(result["errors"], [])
            self.assertEqual(worker.fault_status()["faults"][0]["failures"], 1)
            self.assertTrue(worker.tick()["workflows"][0]["passed"])
            self.assertEqual(worker.fault_status()["faults"], [])
            self.assertEqual(worker.fault_status()["events"][0]["kind"], "recovered")
            runtime.close()

    def test_stale_snapshot_rechecks_quarantine_after_acquiring_dispatch_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("stale", graph())
            stale = WorkflowWorker(workflows, max_errors=1)
            other_runtime = Runtime(directory)
            other = WorkflowWorker(Workflows(other_runtime))
            original_lock = runtime.task_lock
            interleaved = []

            def quarantine_before_lock(name):
                if name == "dispatch-" + key and not interleaved:
                    interleaved.append(other.tick())
                return original_lock(name)

            with patch.object(Workflows, "run", side_effect=ValueError("poison")) as run:
                with patch.object(runtime, "task_lock", quarantine_before_lock):
                    result = stale.tick()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(result["workflows"], [])
            self.assertEqual(result["errors"], [])
            self.assertEqual(stale.fault_status()["faults"][0]["failures"], 1)
            self.assertTrue(stale.fault_status()["faults"][0]["quarantined"])
            other_runtime.close()
            runtime.close()

    def test_stale_error_cannot_quarantine_a_manually_completed_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            workflows.submit("manual-takeover", graph())
            worker = WorkflowWorker(workflows, max_errors=1)
            original = Workflows.run

            def completed_before_error(self, *args, **kwargs):
                original(self, *args, **kwargs)
                raise ValueError("late error")

            with patch.object(Workflows, "run", completed_before_error):
                result = worker.tick()
            self.assertIsNone(result["errors"][0]["fault"])
            self.assertEqual(worker.fault_status()["faults"], [])
            runtime.close()

    def test_fault_and_release_transactions_roll_back_when_audit_write_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("atomic-audit", graph())
            worker = WorkflowWorker(workflows, max_errors=1)
            runtime.db.execute("CREATE TRIGGER fail_audit BEFORE INSERT ON workflow_fault_events "
                               "BEGIN SELECT RAISE(ABORT,'audit unavailable'); END")
            runtime.db.commit()
            with patch.object(Workflows, "run", side_effect=ValueError("poison")):
                with self.assertRaises(sqlite3.IntegrityError):
                    worker.tick()
            self.assertEqual(worker.fault_status()["faults"], [])
            runtime.db.execute("DROP TRIGGER fail_audit")
            runtime.db.commit()
            with patch.object(Workflows, "run", side_effect=ValueError("poison")):
                worker.tick()
            runtime.db.execute("CREATE TRIGGER fail_release BEFORE INSERT ON workflow_fault_events "
                               "WHEN NEW.kind='released' BEGIN SELECT RAISE(ABORT,'audit unavailable'); END")
            runtime.db.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                worker.release(key, reason="fixed")
            self.assertTrue(worker.fault_status()["faults"][0]["quarantined"])
            self.assertEqual(len(worker.fault_status()["events"]), 1)
            runtime.close()

    def test_policy_validation_and_shared_root_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            for options in ({"max_errors": True}, {"max_errors": 0}, {"retry_base": -1},
                            {"retry_max": float("inf")}, {"retry_base": float("nan")},
                            {"retry_base": 2, "retry_max": 1}):
                with self.assertRaises(ValueError):
                    WorkflowWorker(workflows, **options)
            worker = WorkflowWorker(workflows, max_errors=2)
            self.assertEqual(WorkflowWorker(workflows).policy, worker.policy)
            with self.assertRaises(ValueError):
                WorkflowWorker(workflows, max_errors=3)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
