"""Parallel DAG fan-in, resumability, structural validation and artifact integrity."""

import json
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
