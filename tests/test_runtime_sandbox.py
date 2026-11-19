"""Exercise actual Linux namespace isolation when the host supports Bubblewrap."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rsi_agent.durable import Runtime
from rsi_agent.sandbox_worker import isolate


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("bwrap"),
                     "Requires Linux with Bubblewrap installed")
class SandboxTests(unittest.TestCase):
    def test_real_tool_execution_and_readonly_host_filesystem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = Runtime(root / "runtime")
            key = runtime.submit("isolated", "1\n2\n", ["sum"],
                                 [{"kind": "text", "expected": "3\n"}],
                                 limits={"isolation": "bubblewrap"})
            result = runtime.run(key)
            log = root / "runtime" / key / "tool.log"
            diagnostic = log.read_text(errors="replace") if log.is_file() else result["error"]
            self.assertEqual(result["state"], "succeeded", diagnostic)
            runtime.close()
            workspace = root / "workspace"
            workspace.mkdir()
            forbidden = root / "host-file"
            forbidden.write_text("preserved")
            script = (
                "from pathlib import Path; "
                "Path(" + repr(str(forbidden)) + ").write_text('changed')"
            )
            process = subprocess.run(isolate([sys.executable, "-I", "-c", script],
                                             workspace, "bubblewrap"),
                                     capture_output=True, timeout=10)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(forbidden.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
