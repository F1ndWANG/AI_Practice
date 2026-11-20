"""Run real cgroup quota and host-file visibility checks on a prepared Docker daemon."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import unittest
from pathlib import Path

from rsi_agent.durable import Runtime
from rsi_agent.sandbox_worker import isolate, remove_container


@unittest.skipUnless(os.environ.get("AGENT_DOCKER_TESTS") == "1", "Requires explicitly prepared Docker test image")
class DockerTests(unittest.TestCase):
    def test_cancelling_a_running_task_removes_its_container(self):
        command = (sys.executable, "-I", "-c",
                   "from pathlib import Path; import time; "
                   "Path('started').write_text('yes'); time.sleep(30)")
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            with patch.dict("rsi_agent.durable.TOOLS", {"sleep_fixture": command}):
                runtime = Runtime(root, permissions={"tools": ["sleep_fixture"], "origins": []})
                key = runtime.submit("cancel-docker", "", ["sleep_fixture"], [{"kind": "file"}],
                                     limits={"isolation": "docker", "timeout_seconds": 35,
                                             "total_seconds": 50})
                runtime.close()
            script = """
import sys,json
from rsi_agent.durable import Runtime,TOOLS
TOOLS["sleep_fixture"] = tuple(json.loads(sys.argv[3]))
r=Runtime(sys.argv[1])
try:
 print(json.dumps(r.run(sys.argv[2])))
finally:
 r.close()
"""
            process = subprocess.Popen([sys.executable, "-c", script, str(root), key, json.dumps(command)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                until = time.monotonic() + 15
                while not list((root / key).glob("staging-*/started")):
                    if time.monotonic() >= until:
                        self.fail("Container tool did not start")
                    time.sleep(0.05)
                runtime = Runtime(root)
                runtime.cancel(key)
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, stderr)
                self.assertEqual(json.loads(stdout)["state"], "cancelled")
                names = [json.loads(row[0])["name"] for row in runtime.db.execute(
                    "SELECT payload FROM events WHERE task_id=? AND kind='container_reserved'", (key,))]
                for name in names:
                    inspect = subprocess.run(["docker", "container", "inspect", name],
                                             capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(inspect.returncode, 0, name)
                runtime.close()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_runtime_container_and_cgroup_quotas(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            secret = root / "host-secret"
            secret.write_text("host-only")
            runtime = Runtime(root / "runtime")
            key = runtime.submit("docker", "1\n2\n", ["sum"],
                                 [{"kind": "text", "expected": "3\n"}],
                                 limits={"isolation": "docker", "timeout_seconds": 15,
                                         "memory_bytes": 268435456, "max_processes": 16})
            result = runtime.run(key)
            log = root / "runtime" / key / "tool.log"
            self.assertEqual(result["state"], "succeeded",
                             log.read_text() if log.is_file() else result["error"])
            runtime.close()
            workspace = root / "workspace"
            workspace.mkdir()
            script = (
                "import json; from pathlib import Path; "
                f"assert not Path({str(secret)!r}).exists(); "
                "print(json.dumps({'memory':Path('/sys/fs/cgroup/memory.max').read_text().strip(),"
                "'pids':Path('/sys/fs/cgroup/pids.max').read_text().strip()}))"
            )
            command = isolate([sys.executable, "-c", script], workspace, "docker",
                              {"memory_bytes": 268435456, "max_processes": 16})
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            finally:
                remove_container(command)
            self.assertEqual(result.returncode, 0, result.stderr)
            quotas = json.loads(result.stdout)
            self.assertEqual(quotas, {"memory": "268435456", "pids": "16"})


if __name__ == "__main__":
    unittest.main()
