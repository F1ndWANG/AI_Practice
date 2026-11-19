"""POSIX resource limits for a registered tool subprocess."""

import json
import os
import resource
import shutil
import sys



def isolate(command, root, mode):
    if mode == "process":
        return command
    executable = shutil.which("bwrap")
    if mode != "bubblewrap" or not executable or not sys.platform.startswith("linux"):
        raise RuntimeError("bubblewrap isolation requires Linux and an installed bwrap executable")
    return [executable, "--unshare-all", "--new-session", "--die-with-parent",
            "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--bind", str(root), "/work", "--chdir", "/work", *command]


def main():
    limits = json.loads(sys.argv[1])
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits["file_bytes"], limits["file_bytes"]))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if limits.get("memory_bytes") and sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (limits["memory_bytes"], limits["memory_bytes"]))
    os.execvpe(sys.argv[2], sys.argv[2:], {"PATH": os.defpath, "LC_ALL": "C"})


if __name__ == "__main__":
    main()
