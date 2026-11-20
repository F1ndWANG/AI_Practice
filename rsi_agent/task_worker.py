"""Drain queued tasks with persisted retry limits and interruptible backoff."""

import argparse
import json
import math
import time
from pathlib import Path

from .durable import Runtime
from .execution_control import TaskCancelled, TaskDeadlineExceeded


def drain(runtime, *, max_attempts=3, retry_delay=0.25):
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("Retry limit must be a positive integer")
    if type(retry_delay) not in (int, float) or not math.isfinite(retry_delay) or retry_delay < 0:
        raise ValueError("Retry delay must be finite and nonnegative")
    results = []
    # One snapshot of the queue, so the command always completes.
    for item in runtime.status():
        key = item["id"]
        status = item
        if status["state"] in {"succeeded", "rejected", "cancelled", "timed_out"}:
            continue
        while (status["attempts"] or 0) < max_attempts:
            if status["state"] == "failed" and not status["retryable"]:
                break
            try:
                runtime.run(key)
            except RuntimeError as error:
                if str(error) != "Task is already executing":
                    raise
                break
            status = runtime.status(key)
            if status["state"] != "failed" or not status["retryable"]:
                break
            if status["attempts"] >= max_attempts:
                break
            until = time.monotonic() + retry_delay * 2 ** (status["attempts"] - 1)
            while time.monotonic() < until:
                try:
                    runtime.guard(key)
                except (TaskCancelled, TaskDeadlineExceeded):
                    # run applies the terminal state under the task lock.
                    break
                time.sleep(min(0.05, max(0, until - time.monotonic())))
        results.append(runtime.status(key))
    return {"tasks": results, "passed": all(row["state"] == "succeeded" for row in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=0.25)
    args = parser.parse_args()
    runtime = Runtime(args.root)
    try:
        result = drain(runtime, max_attempts=args.max_attempts, retry_delay=args.retry_delay)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
