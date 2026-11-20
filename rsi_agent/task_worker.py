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
    for item in runtime.status():
        if item["state"] not in {"succeeded", "rejected", "cancelled", "timed_out"}:
            results.append(run_one(runtime, item["id"], max_attempts=max_attempts, retry_delay=retry_delay))
    return {"tasks": results, "passed": all(row["state"] == "succeeded" for row in results)}


def run_one(runtime, key, *, max_attempts=3, retry_delay=0.25):
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("Retry limit must be a positive integer")
    if type(retry_delay) not in (int, float) or not math.isfinite(retry_delay) or retry_delay < 0:
        raise ValueError("Retry delay must be finite and nonnegative")
    status = runtime.status(key)
    if status["state"] in {"succeeded", "rejected", "cancelled", "timed_out"}:
        return status
    while (status["attempts"] or 0) < max_attempts:
        if status["state"] == "failed" and not status["retryable"]:
            break
        try:
            runtime.run(key, max_attempts=max_attempts)
        except RuntimeError as error:
            if str(error) != "Task is already executing":
                raise
            break
        status = runtime.status(key)
        if status["state"] != "failed" or not status["retryable"] or status["attempts"] >= max_attempts:
            break
        until = time.monotonic() + retry_delay * 2 ** min(status["attempts"] - 1, 10)
        while time.monotonic() < until:
            try:
                runtime.guard(key)
            except (TaskCancelled, TaskDeadlineExceeded):
                break
            time.sleep(min(0.05, max(0, until - time.monotonic())))
    return runtime.status(key)


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
