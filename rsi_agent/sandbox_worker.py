"""POSIX resource limits for a registered tool subprocess."""

import json
import os
import resource
import shutil
import sys
import subprocess
import uuid
from pathlib import Path



def container_image():
    executable = shutil.which("docker")
    if not executable:
        raise RuntimeError("Docker isolation requires an installed Docker client and running daemon")
    result = subprocess.run(
        [executable, "image", "inspect", "--format", "{{.Id}}", "ai-practice-runtime:local"],
        capture_output=True, text=True, timeout=15, check=False)
    if result.returncode or not result.stdout.strip().startswith("sha256:"):
        raise RuntimeError("Build the ai-practice-runtime:local image before selecting Docker isolation")
    return result.stdout.strip()


def remove_container(command):
    if "--name" in command and Path(command[0]).name == "docker":
        name = command[command.index("--name") + 1]
        subprocess.run([command[0], "rm", "-f", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=15, check=False)


def isolate(command, root, mode, limits=None, readonly_paths=()):
    limits = limits or {}
    if mode == "docker":
        image = limits.get("_image_id") or container_image()
        executable = shutil.which("docker")
        source = Path(__file__).parent.resolve()
        rewritten = []
        for argument in command:
            if argument == sys.executable:
                rewritten.append("/usr/local/bin/python")
            elif argument.startswith(str(source) + "/"):
                rewritten.append("/opt/agent/rsi_agent/" + str(Path(argument).relative_to(source)))
            else:
                rewritten.append(argument)
        result = [executable, "run", "--rm", "--pull=never", "--name", "agent-" + uuid.uuid4().hex,
                  "--network=none", "--read-only", "--cap-drop=ALL",
                  "--security-opt=no-new-privileges", "--memory", str(limits.get("memory_bytes", 536870912)),
                  "--memory-swap", str(limits.get("memory_bytes", 536870912)),
                  "--cpus", str(limits.get("cpu_quota", 1)),
                  "--pids-limit", str(limits.get("max_processes", 64)),
                  "--user", f"{os.getuid()}:{os.getgid()}",
                  "--tmpfs", "/tmp:rw,nosuid,nodev,size=16777216",
                  "--mount", f"type=bind,src={source},dst=/opt/agent/rsi_agent,readonly",
                  "--mount", f"type=bind,src={root},dst={root}", "--workdir", str(root)]
        for path in readonly_paths:
            result.extend(["--mount", f"type=bind,src={path},dst={path},readonly"])
        return [*result, image, *rewritten]
    if mode == "process":
        return command
    executable = shutil.which("bwrap")
    if mode != "bubblewrap" or not executable or not sys.platform.startswith("linux"):
        raise RuntimeError("bubblewrap isolation requires Linux and an installed bwrap executable")
    return [executable, "--unshare-all", "--new-session", "--die-with-parent",
            "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--bind", str(root), str(root), "--chdir", str(root), *command]


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
