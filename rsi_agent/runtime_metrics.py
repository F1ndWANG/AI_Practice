"""System metrics with explicit independent labels and usage coverage."""

import argparse
import json
import math
from pathlib import Path


def summarize(rows):
    if not rows:
        raise ValueError("Evaluation requires task records")
    for row in rows:
        if (type(row.get("success")) is not bool
                or row.get("verifier_passed") is not None and type(row["verifier_passed"]) is not bool):
            raise ValueError("Independent success labels and verifier verdicts must be booleans")
        if row.get("attempts") is not None and (type(row["attempts"]) is not int or row["attempts"] < 0):
            raise ValueError("Attempt counts must be nonnegative integers or null")
        if row.get("state") is not None and row["state"] not in {
            "succeeded", "rejected", "failed", "cancelled", "timed_out"
        }:
            raise ValueError("Lifecycle metrics require terminal task states")
        for field in ("tool_calls", "seconds", "recoveries"):
            value = row[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Metrics must be finite nonnegative numbers")
        for field in ("tokens", "cost"):
            value = row.get(field)
            if value is not None and (type(value) not in (int, float)
                                      or not math.isfinite(value) or value < 0):
                raise ValueError("Usage must be finite nonnegative numbers or null")
    count = len(rows)
    successful = sum(row["success"] for row in rows)
    recovered = [row for row in rows if row["recoveries"] > 0]
    verified = [row for row in rows if row.get("verifier_passed") is not None]
    actual_failures = [row for row in verified if not row["success"]]
    actual_successes = [row for row in verified if row["success"]]
    latencies = sorted(row["seconds"] for row in rows)
    token_rows = [row for row in rows if row.get("tokens") is not None]
    attempt_rows = [row for row in rows if row.get("attempts") is not None]
    costs = [row["cost"] for row in rows if row.get("cost") is not None]
    return {
        "tasks": count,
        "lifecycle_coverage": sum(row.get("state") is not None for row in rows) / count,
        "cancelled_tasks": sum(row.get("state") == "cancelled" for row in rows),
        "timed_out_tasks": sum(row.get("state") == "timed_out" for row in rows),
        "attempt_coverage": len(attempt_rows) / count,
        "mean_attempts": sum(row["attempts"] for row in attempt_rows) / len(attempt_rows) if attempt_rows else None, "task_success_rate": successful / count,
        "mean_tool_calls": sum(row["tool_calls"] for row in rows) / count,
        "mean_tokens": sum(row["tokens"] for row in token_rows) / len(token_rows) if token_rows else None,
        "token_coverage": len(token_rows) / count,
        "p95_end_to_end_seconds": latencies[math.ceil(count * 0.95) - 1],
        "recovery_tasks": len(recovered),
        "recovery_success_rate": sum(row["success"] for row in recovered) / len(recovered) if recovered else None,
        "verifier_coverage": len(verified) / count,
        "verifier_error_rate": sum(row["success"] != row["verifier_passed"] for row in verified) / len(verified) if verified else None,
        "verifier_false_accept_rate": sum(row["verifier_passed"] for row in actual_failures) / len(actual_failures) if actual_failures else None,
        "verifier_false_reject_rate": sum(not row["verifier_passed"] for row in actual_successes) / len(actual_successes) if actual_successes else None,
        "cost_coverage": len(costs) / count,
        "cost_per_success": sum(costs) / successful if successful and len(costs) == count else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
