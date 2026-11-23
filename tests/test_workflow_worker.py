"""Queue fairness, restart recovery and actual worker process shutdown."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from rsi_agent.durable import Runtime
from rsi_agent.workflow_worker import WorkflowWorker
from rsi_agent.workflows import Workflows


def graph():
    return {"limits": {"max_tool_calls": 2, "total_seconds": 60}, "nodes": [
        {"id": "first", "source": "input", "recipe": ["copy"],
         "rules": [{"kind": "text", "expected": "input"}]},
        {"id": "after", "inputs": [{"node": "first"}], "recipe": ["copy"],
         "rules": [{"kind": "text", "expected": "input"}]},
    ]}


class WorkflowWorkerTests(unittest.TestCase):
    def test_new_arrivals_join_the_tail_instead_of_starving_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            older = workflows.submit("older", graph())
            worker = WorkflowWorker(workflows)
            with runtime.task_lock("workflow-" + older):
                self.assertEqual(worker.tick(max_workflows=1)["busy"], [older])
            newer = workflows.submit("newer", graph())
            result = worker.tick(max_workflows=1)
            self.assertEqual(result["workflows"][0]["id"], older)
            self.assertEqual(workflows.status(newer)["state"], "pending")
            self.assertEqual(worker.tick(max_workflows=1)["workflows"][0]["id"], newer)
            runtime.close()

    def test_polling_acknowledges_expired_graph_without_launching_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("expired-queue", graph())
            runtime.db.execute("UPDATE workflow_deadlines SET deadline_at=?", (time.time() - 1,))
            runtime.db.commit()
            worker = WorkflowWorker(workflows)
            result = worker.tick()
            self.assertEqual(result["workflows"], [{"id": key, "state": "timed_out", "passed": False}])
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 0)
            self.assertEqual(worker.tick()["workflows"], [])
            runtime.close()

    def test_tick_drains_pending_and_does_not_automatically_retry_failed_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            failed_plan = graph()
            failed_plan["nodes"][0]["rules"][0]["expected"] = "wrong"
            bad = workflows.submit("bad", failed_plan)
            good = workflows.submit("good", graph())
            worker = WorkflowWorker(workflows)
            result = worker.tick()
            self.assertEqual({r["id"]: r["state"] for r in result["workflows"]},
                             {bad: "failed", good: "succeeded"})
            self.assertEqual(result["errors"], [])
            calls = runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0]
            self.assertEqual(worker.tick()["workflows"], [])
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], calls)
            runtime.close()

    def test_busy_head_does_not_starve_later_work_across_worker_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            first = workflows.submit("busy", graph())
            second = workflows.submit("ready", graph())
            worker = WorkflowWorker(workflows)
            with runtime.task_lock("workflow-" + first):
                self.assertEqual(worker.tick(max_workflows=1)["busy"], [first])
                other = Runtime(directory)
                try:
                    restarted = WorkflowWorker(Workflows(other))
                    result = restarted.tick(max_workflows=1)
                    self.assertEqual(result["busy"], [])
                    self.assertEqual(result["workflows"], [{"id": second, "state": "succeeded", "passed": True}])
                finally:
                    other.close()
            self.assertEqual(worker.tick(max_workflows=1)["workflows"][0]["id"], first)
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 4)
            runtime.close()

    def test_graceful_stop_drains_active_node_then_resumes_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            key = workflows.submit("draining", graph())
            stop = threading.Event()
            original = Workflows.execute_node

            def finish_then_stop(self, *args):
                result = original(self, *args)
                stop.set()
                return result

            worker = WorkflowWorker(workflows)
            with patch.object(Workflows, "execute_node", finish_then_stop):
                result = worker.tick(stop_event=stop)
            self.assertEqual(result["workflows"][0]["state"], "incomplete")
            self.assertFalse(workflows.status(key)["cancel_requested"])
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 1)
            self.assertEqual(worker.tick()["workflows"][0]["state"], "succeeded")
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 2)
            runtime.close()

    def test_stop_while_reading_source_does_not_admit_a_node(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            workflows.submit("source-stop", graph())
            stop = threading.Event()
            original = Workflows.source

            def stop_after_read(self, *args):
                source = original(self, *args)
                stop.set()
                return source

            worker = WorkflowWorker(workflows)
            with patch.object(Workflows, "source", stop_after_read):
                result = worker.tick(stop_event=stop)
            self.assertEqual(result["workflows"][0]["state"], "incomplete")
            self.assertEqual(runtime.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 0)
            self.assertEqual(worker.tick()["workflows"][0]["state"], "succeeded")
            runtime.close()

    def test_one_orchestrator_error_does_not_prevent_other_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            bad = workflows.submit("bad-controller", graph())
            good = workflows.submit("good-controller", graph())
            original = Workflows.run

            def fail_one(self, key, **kwargs):
                if key == bad:
                    raise ValueError("injected controller failure")
                return original(self, key, **kwargs)

            worker = WorkflowWorker(workflows)
            with patch.object(Workflows, "run", fail_one):
                result = worker.tick()
            self.assertEqual(result["errors"][0]["id"], bad)
            self.assertEqual(result["workflows"][0]["id"], good)
            self.assertEqual(worker.tick()["workflows"], [])
            worker.release(bad, reason="Controller fixture repaired")
            self.assertEqual(worker.tick()["workflows"][0]["state"], "succeeded")
            runtime.close()

    def test_serve_discovers_new_submissions_and_stops_interruptible_idle_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            worker = WorkflowWorker(workflows)
            stop = threading.Event()
            cycles = []

            def on_cycle(result):
                cycles.append(result)
                if len(cycles) == 1:
                    workflows.submit("arrived-after-start", graph())
                else:
                    stop.set()

            worker.serve(stop_event=stop, poll_interval=0.001, on_cycle=on_cycle)
            self.assertEqual(len(cycles), 2)
            self.assertTrue(cycles[-1]["workflows"][0]["passed"])
            for interval in (0, -1, True, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    worker.serve(stop_event=stop, poll_interval=interval)
            for options in ({"max_workflows": 0}, {"node_workers": True}, {"max_attempts": -1}):
                with self.assertRaises(ValueError):
                    worker.tick(**options)
            self.assertEqual(worker.tick(stop_event=stop)["workflows"], [])
            runtime.close()

    def test_hard_process_exit_releases_lock_and_preserves_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            workflows = Workflows(runtime)
            plan = graph()
            plan["nodes"] = [plan["nodes"][0]]
            plan["nodes"][0]["recipe"] = ["copy", "copy"]
            key = workflows.submit("crashed-worker", plan)
            script = ("import os,sys; from rsi_agent.durable import Runtime; "
                      "r=Runtime(sys.argv[1]); w=sys.argv[2]; "
                      "lock=r.task_lock('workflow-'+w); lock.__enter__(); "
                      "t=r.submit('workflow:'+w+':node:first','input',['copy','copy'],"
                      "[{'kind':'text','expected':'input'}],workflow=(w,'first')); "
                      "r.run(t,interrupt_after=1); os._exit(17)")
            process = subprocess.run([sys.executable, "-c", script, directory, key],
                                     capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 17, process.stderr)
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 1)
            result = WorkflowWorker(workflows).tick()
            self.assertTrue(result["workflows"][0]["passed"], result)
            self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 2)
            runtime.close()

    def test_actual_sigterm_drains_process_and_restart_resumes(self):
        command = (sys.executable, "-I", "-c",
                   "from pathlib import Path; import time; Path('started').write_text('yes'); "
                   "time.sleep(1); Path('output.txt').write_text('input')")
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "rsi_agent.durable.TOOLS", {"slow_fixture": command}):
            runtime = Runtime(directory, permissions={"tools": ["slow_fixture", "copy"], "origins": []})
            workflows = Workflows(runtime)
            plan = graph()
            plan["nodes"][0]["recipe"] = ["slow_fixture"]
            key = workflows.submit("signal-worker", plan)
            script = ("from rsi_agent import durable; durable.TOOLS['slow_fixture']=" + repr(command)
                      + "; from rsi_agent.workflow_worker import main; main()")
            process = subprocess.Popen([sys.executable, "-c", script, directory, "--poll-interval", "30"],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                until = time.monotonic() + 10
                while not list(Path(directory).glob("*/staging-*/started")):
                    if time.monotonic() >= until or process.poll() is not None:
                        self.fail("Worker subprocess did not launch its tool")
                    time.sleep(0.02)
                os.kill(process.pid, signal.SIGTERM)
                output, error = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, error)
                self.assertEqual(json.loads(output.splitlines()[-1])["workflows"][0]["state"], "incomplete")
                self.assertFalse(workflows.status(key)["cancel_requested"])
                self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 1)
                self.assertTrue(WorkflowWorker(workflows).tick()["workflows"][0]["passed"])
                self.assertEqual(runtime.db.execute("SELECT sum(calls) FROM tasks").fetchone()[0], 2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)
                runtime.close()


if __name__ == "__main__":
    unittest.main()
