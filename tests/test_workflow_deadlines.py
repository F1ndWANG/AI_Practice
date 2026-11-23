"""Graph deadlines survive downtime and constrain actual node subprocesses."""

import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent.execution_control import TaskDeadlineExceeded
from rsi_agent.workflows import Workflows


def graph(seconds=60):
    return {"limits": {"total_seconds": seconds}, "nodes": [
        {"id": "first", "source": "input", "recipe": ["copy"],
         "rules": [{"kind": "text", "expected": "input"}]},
        {"id": "after", "inputs": [{"node": "first"}], "recipe": ["copy"],
         "rules": [{"kind": "text", "expected": "input"}]},
    ]}


class WorkflowDeadlineTests(unittest.TestCase):
    def test_expiration_racing_node_admission_rolls_back_without_orphans(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("admission-race", graph())
            original = Workflows.execute_node

            def expire_then_admit(self, workflow_id, node, source, max_attempts):
                connection = Runtime(directory)
                try:
                    with connection.db:
                        connection.db.execute("UPDATE workflow_deadlines SET deadline_at=? WHERE workflow_id=?",
                                              (time.time() - 1, workflow_id))
                finally:
                    connection.close()
                return original(self, workflow_id, node, source, max_attempts)

            with patch.object(Workflows, "execute_node", expire_then_admit):
                result = workflows.run(key, workers=1)
            self.assertEqual(result["state"], "timed_out", result)
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 0)
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM workflow_budget_tasks").fetchone()[0], 0)
            runtime.close()

    def test_expired_queued_graph_does_not_admit_tasks_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            plan = graph()
            key = workflows.submit("queued", plan)
            deadline = workflows.status(key)["deadline_at"]
            self.assertEqual(workflows.submit("queued", plan), key)
            self.assertEqual(workflows.status(key)["deadline_at"], deadline)
            runtime.db.execute("UPDATE workflow_deadlines SET deadline_at=?", (time.time() - 1,))
            runtime.db.commit()
            runtime.close()
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            result = workflows.run(key)
            self.assertEqual(result["state"], "timed_out", result)
            self.assertTrue(result["deadline_expired"])
            self.assertTrue(all(n["state"] == "timed_out" for n in result["nodes"]))
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 0)
            with self.assertRaises(TaskDeadlineExceeded):
                runtime.submit("late", "input", ["copy"], [{"kind": "file"}], workflow=(key, "first"))
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM workflow_budget_tasks").fetchone()[0], 0)
            self.assertEqual(workflows.run(key)["state"], "timed_out")
            runtime.close()

    def test_node_deadlines_are_clamped_and_do_not_extend_on_resubmission(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("clamp", graph(60))
            task = runtime.submit("long-node", "input", ["copy"], [{"kind": "file"}],
                                  workflow=(key, "first"))
            short = runtime.submit("short-node", "input", ["copy"], [{"kind": "file"}],
                                   limits={"total_seconds": 1}, workflow=(key, "after"))
            deadlines = dict(runtime.db.execute("SELECT task_id,deadline_at FROM task_controls"))
            self.assertEqual(deadlines[task], workflows.deadline(key))
            self.assertLess(deadlines[short], deadlines[task])
            runtime.submit("long-node", "input", ["copy"], [{"kind": "file"}], workflow=(key, "first"))
            self.assertEqual(dict(runtime.db.execute("SELECT task_id,deadline_at FROM task_controls")), deadlines)
            runtime.close()

    def test_deadline_kills_real_tool_and_prevents_descendant_launch(self):
        command = (sys.executable, "-I", "-c", "import time; time.sleep(30)")
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "rsi_agent.durable.TOOLS", {"slow_fixture": command}):
            runtime = Runtime(directory, permissions={"tools": ["slow_fixture", "copy"], "origins": []})
            workflows = Workflows(runtime)
            plan = graph(1)
            plan["nodes"][0]["recipe"] = ["slow_fixture"]
            plan["nodes"][0]["limits"] = {"timeout_seconds": 35, "total_seconds": 40}
            key = workflows.submit("running-timeout", plan)
            started = time.monotonic()
            result = workflows.run(key)
            self.assertLess(time.monotonic() - started, 10)
            self.assertEqual(result["state"], "timed_out", result)
            self.assertTrue(all(n["state"] == "timed_out" for n in result["nodes"]))
            self.assertEqual(runtime.status()[0]["state"], "timed_out")
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 1)
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 1)
            runtime.close()

    def test_crashed_bound_task_expires_without_starting_any_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("crashed-admission", graph())
            task = runtime.submit(f"workflow:{key}:node:first", "input", ["copy"],
                                  [{"kind": "text", "expected": "input"}], workflow=(key, "first"))
            past = time.time() - 1
            runtime.db.execute("UPDATE workflow_deadlines SET deadline_at=?", (past,))
            runtime.db.execute("UPDATE task_controls SET deadline_at=?", (past,))
            runtime.db.commit()
            result = workflows.run(key)
            self.assertEqual(result["state"], "timed_out", result)
            self.assertEqual(runtime.status(task)["state"], "timed_out")
            self.assertEqual(runtime.status(task)["calls"], 0)
            runtime.close()

    def test_successful_graph_remains_reusable_after_its_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("completed", graph())
            self.assertTrue(workflows.run(key)["passed"])
            runtime.db.execute("UPDATE workflow_deadlines SET deadline_at=?", (time.time() - 1,))
            runtime.db.commit()
            result = workflows.run(key)
            self.assertTrue(result["passed"], result)
            self.assertTrue(result["deadline_expired"])
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 2)
            runtime.close()

    def test_deadline_validation_does_not_write_partial_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            for seconds in (0, -1, True, "1", None, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    workflows.submit("invalid", graph(seconds))
            for nodes in ("bad", {"id": "bad"}, None):
                with self.assertRaises(ValueError):
                    workflows.submit("invalid", {"nodes": nodes})
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM workflows").fetchone()[0], 0)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
