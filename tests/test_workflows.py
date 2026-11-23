"""Parallel DAG fan-in, resumability, structural validation and artifact integrity."""

import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import unittest
from graphlib import CycleError
from pathlib import Path
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent.workflows import Workflows


def plan():
    return {"nodes": [
        {"id": "a", "source": "1\n2\n", "recipe": ["sum"],
         "rules": [{"kind": "text", "expected": "3\n"}]},
        {"id": "b", "source": "4\n5\n", "recipe": ["copy", "sum"],
         "rules": [{"kind": "text", "expected": "9\n"}]},
        {"id": "join", "inputs": [{"node": "a"}, {"node": "b"}], "recipe": ["sum"],
         "rules": [{"kind": "text", "expected": "12\n"}]},
    ]}


class WorkflowTests(unittest.TestCase):
    def test_parallel_branches_join_and_completed_graph_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("parallel", plan())
            barrier = threading.Barrier(2)
            original = Runtime.run

            def synchronize(self, task_id, **kwargs):
                spec = json.loads(self.db.execute("SELECT spec FROM tasks WHERE id=?", (task_id,)).fetchone()[0])
                if spec["source"] in {"1\n2\n", "4\n5\n"} and self.status(task_id)["state"] == "pending":
                    barrier.wait(timeout=10)
                return original(self, task_id, **kwargs)

            with patch.object(Runtime, "run", synchronize):
                result = workflows.run(key, workers=2)
            self.assertTrue(result["passed"], result)
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 4)
            self.assertTrue(workflows.run(key)["passed"])
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 4)
            self.assertEqual(workflows.submit("parallel", plan()), key)
            changed = plan()
            changed["nodes"][0]["source"] = "2\n2\n"
            with self.assertRaises(ValueError):
                workflows.submit("parallel", changed)
            runtime.close()

    def test_graph_resume_keeps_committed_steps_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("resume", plan())
            original = Runtime.run
            stopped = set()

            def interrupt(self, task_id, **kwargs):
                spec = json.loads(self.db.execute("SELECT spec FROM tasks WHERE id=?", (task_id,)).fetchone()[0])
                if spec["source"] == "4\n5\n" and not stopped:
                    stopped.add(task_id)
                    return original(self, task_id, interrupt_after=1, **kwargs)
                return original(self, task_id, **kwargs)

            with patch.object(Runtime, "run", interrupt):
                result = workflows.run(key)
            self.assertEqual(result["state"], "incomplete", result)
            runtime.close()
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            result = workflows.run(key)
            self.assertTrue(result["passed"], result)
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 4)
            first = next(node for node in result["nodes"] if node["node_id"] == "a")
            (Path(directory) / first["task_id"] / "checkpoint-1/output.txt").write_text("changed")
            result = workflows.run(key)
            self.assertFalse(result["passed"])
            self.assertEqual(result["nodes"][0]["state"], "error")
            self.assertFalse(workflows.run(key)["passed"])
            (Path(directory) / first["task_id"] / "checkpoint-1/output.txt").write_text("3\n")
            self.assertTrue(workflows.run(key)["passed"])
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 4)
            runtime.close()

    def test_node_binding_failure_rolls_back_the_entire_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            graph = {"nodes": [{"id": "one", "source": "input", "recipe": ["copy"],
                                "rules": [{"kind": "file"}]}]}
            workflow_id = workflows.submit("atomic-binding", graph)
            runtime.db.execute("""CREATE TRIGGER fail_binding BEFORE UPDATE ON workflow_nodes
                BEGIN SELECT RAISE(ABORT,'injected binding failure'); END""")
            runtime.db.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                runtime.submit(f"workflow:{workflow_id}:node:one", "input", ["copy"], [{"kind": "file"}],
                               workflow=(workflow_id, "one"))
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 0)
            self.assertIsNone(workflows.status(workflow_id)["nodes"][0]["task_id"])
            runtime.db.execute("DROP TRIGGER fail_binding")
            runtime.db.commit()
            self.assertTrue(workflows.run(workflow_id)["passed"])
            runtime.close()

    def test_orchestrator_interrupt_is_cancelled_and_acknowledged_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            graph = {"nodes": [{"id": "one", "source": "input", "recipe": ["copy"],
                                "rules": [{"kind": "file"}]}]}
            workflow_id = workflows.submit("controller-interrupt", graph)
            original = Runtime.event

            def interrupt(self, key, kind, payload):
                original(self, key, kind, payload)
                if kind == "tool_started":
                    raise KeyboardInterrupt()

            with patch.object(Runtime, "event", interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    workflows.run(workflow_id)
            self.assertTrue(workflows.status(workflow_id)["cancel_requested"])
            self.assertEqual(workflows.run(workflow_id)["state"], "cancelled")
            self.assertEqual(runtime.status()[0]["calls"], 0)
            runtime.close()

    def test_ready_child_starts_before_an_unrelated_slow_root_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            graph = {"nodes": [
                {"id": "fast", "source": "1\n", "recipe": ["sum"],
                 "rules": [{"kind": "text", "expected": "1\n"}]},
                {"id": "slow", "source": "2\n", "recipe": ["sum"],
                 "rules": [{"kind": "text", "expected": "2\n"}]},
                {"id": "child", "inputs": [{"node": "fast"}], "recipe": ["sum"],
                 "rules": [{"kind": "text", "expected": "1\n"}]},
            ]}
            released = threading.Event()
            original = Workflows.execute_node

            def gate(self, workflow, node, source, max_attempts):
                if node["id"] == "slow" and not released.wait(timeout=10):
                    raise RuntimeError("Ready child was delayed by the unrelated branch")
                if node["id"] == "child":
                    released.set()
                return original(self, workflow, node, source, max_attempts)

            with patch.object(Workflows, "execute_node", gate):
                result = workflows.run(workflows.submit("streaming", graph), workers=2)
            self.assertTrue(result["passed"], result)
            self.assertTrue(released.is_set())
            runtime.close()

    def test_cancel_racing_with_node_admission_leaves_no_task(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            graph = {"nodes": [{"id": "one", "source": "hello", "recipe": ["copy"],
                                "rules": [{"kind": "file"}]}]}
            workflow_id = workflows.submit("admission", graph)
            entered, release = threading.Event(), threading.Event()
            original = Runtime.submit

            def gate(self, *args, **kwargs):
                if kwargs.get("workflow") == (workflow_id, "one"):
                    entered.set()
                    if not release.wait(timeout=10):
                        raise RuntimeError("Admission fixture was not released")
                return original(self, *args, **kwargs)

            def runner():
                runtime = Runtime(directory)
                try:
                    return Workflows(runtime).run(workflow_id)
                finally:
                    runtime.close()

            with patch.object(Runtime, "submit", gate), ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(runner)
                try:
                    self.assertTrue(entered.wait(timeout=10))
                    self.assertEqual(workflows.cancel(workflow_id)["state"], "cancelled")
                finally:
                    release.set()
                result = future.result(timeout=10)
            self.assertEqual(result["state"], "cancelled")
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 0)
            self.assertEqual(workflows.run(workflow_id)["state"], "cancelled")
            runtime.close()

    def test_cancel_stops_running_nodes_and_prevents_downstream_admission(self):
        command = (sys.executable, "-I", "-c",
                   "from pathlib import Path; import time; "
                   "Path('started').write_text('yes'); time.sleep(30)")
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "rsi_agent.durable.TOOLS", {"slow_fixture": command}
        ):
            runtime = Runtime(directory, permissions={"tools": ["slow_fixture", "copy"], "origins": []})
            workflows = Workflows(runtime)
            graph = {"nodes": [
                {"id": "long", "source": "input", "recipe": ["slow_fixture"],
                 "rules": [{"kind": "file"}],
                 "limits": {"timeout_seconds": 35, "total_seconds": 40}},
                {"id": "after", "inputs": [{"node": "long"}], "recipe": ["copy"],
                 "rules": [{"kind": "file"}]},
            ]}
            workflow_id = workflows.submit("running-cancel", graph)

            def runner():
                runtime = Runtime(directory)
                try:
                    return Workflows(runtime).run(workflow_id)
                finally:
                    runtime.close()

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(runner)
                try:
                    until = time.monotonic() + 10
                    while not list(Path(directory).glob("*/staging-*/started")):
                        if time.monotonic() >= until:
                            self.fail("Workflow tool did not start")
                        time.sleep(0.02)
                    self.assertTrue(workflows.cancel(workflow_id)["cancel_requested"])
                finally:
                    workflows.cancel(workflow_id)
                result = future.result(timeout=10)
            self.assertEqual(result["state"], "cancelled", result)
            self.assertTrue(all(node["state"] == "cancelled" for node in result["nodes"]))
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 1)
            self.assertEqual(runtime.status()[0]["state"], "cancelled")
            runtime.close()

    def test_invalid_input_artifact_does_not_starve_an_independent_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            graph = {"nodes": [
                {"id": "a", "source": "1", "recipe": ["copy"], "rules": [{"kind": "file"}]},
                {"id": "bad", "inputs": [{"node": "a", "path": "missing"}], "recipe": ["copy"],
                 "rules": [{"kind": "file"}]},
                {"id": "b", "source": "2", "recipe": ["copy"], "rules": [{"kind": "file"}]},
                {"id": "good", "inputs": [{"node": "b"}], "recipe": ["copy"], "rules": [{"kind": "file"}]},
            ]}
            result = workflows.run(workflows.submit("source-error", graph), workers=1)
            states = {node["node_id"]: node["state"] for node in result["nodes"]}
            self.assertEqual(states, {"a": "succeeded", "bad": "error", "b": "succeeded", "good": "succeeded"})
            runtime.close()

    def test_cycles_and_unknown_dependencies_are_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            cyclic = plan()
            cyclic["nodes"][0]["depends_on"] = ["join"]
            with self.assertRaises(CycleError):
                workflows.submit("cycle", cyclic)
            unknown = plan()
            unknown["nodes"][0]["depends_on"] = ["missing"]
            with self.assertRaises(ValueError):
                workflows.submit("unknown", unknown)
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 0)
            runtime.close()

    def test_failed_dependency_blocks_descendants_in_reverse_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            graph = {"nodes": [
                {"id": "last", "source": "unused", "depends_on": ["middle"], "recipe": ["copy"],
                 "rules": [{"kind": "file"}]},
                {"id": "middle", "source": "unused", "depends_on": ["first"], "recipe": ["copy"],
                 "rules": [{"kind": "file"}]},
                {"id": "first", "source": "bad", "recipe": ["copy"],
                 "rules": [{"kind": "text", "expected": "good"}]},
            ]}
            result = workflows.run(workflows.submit("blocked", graph))
            states = {node["node_id"]: node["state"] for node in result["nodes"]}
            self.assertEqual(states, {"first": "rejected", "middle": "blocked", "last": "blocked"})
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 1)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
