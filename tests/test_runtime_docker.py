"""Run real cgroup quota and host-file visibility checks on a prepared Docker daemon."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rsi_agent.durable import Runtime
from rsi_agent.sandbox_worker import isolate, remove_container


@unittest.skipUnless(os.environ.get("AGENT_DOCKER_TESTS") == "1", "Requires explicitly prepared Docker test image")
class DockerTests(unittest.TestCase):
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
