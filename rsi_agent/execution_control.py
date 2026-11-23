"""Cooperative task cancellation and bounded subprocess supervision."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time


class TaskCancelled(Exception):
    pass


class TaskDeadlineExceeded(Exception):
    pass


class ToolCapacityError(RuntimeError):
    pass


def wait_process(process, timeout, *, check=None, input_text=None):
    deadline = time.monotonic() + timeout
    first = True
    try:
        while True:
            if check:
                check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            try:
                if process.stdin is not None:
                    process.communicate(input_text if first else None, timeout=min(0.05, remaining))
                    return process.returncode
                return process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                first = False
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise


def json_process(module, payload, timeout, *, check=None):
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [sys.executable, "-m", module], stdin=subprocess.PIPE, stdout=output, stderr=errors,
            text=True, start_new_session=True,
            env={name: value for name, value in os.environ.items()
                 if name in {"PATH", "HOME", "PLAYWRIGHT_BROWSERS_PATH", "LD_LIBRARY_PATH"}})
        code = wait_process(process, timeout, check=check, input_text=json.dumps(payload))
        if code:
            errors.seek(0)
            raise RuntimeError(f"{module} exited with {code}: " + errors.read(2000).decode(errors="replace"))
        output.seek(0)
        return json.load(output)
