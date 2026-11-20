"""Export task telemetry against independently supplied success labels."""

import argparse
import json
import math
from pathlib import Path

from .durable import Runtime
from .governance import GovernedMemory
from .runtime_metrics import summarize
from .runtime_planning import export_record


def export(root, labels, output, *, calibrate_memory=False, tool_cost=None):
    if tool_cost is not None and (type(tool_cost) not in (int, float)
                                  or not math.isfinite(tool_cost) or tool_cost < 0):
        raise ValueError("Tool price must be finite and nonnegative")
    if not (Path(root) / "runtime.sqlite").is_file():
        raise ValueError("Runtime database does not exist")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("Provide a task-ID to independent success-label mapping")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    runtime = Runtime(root)
    observations = []
    try:
        with runtime.db:
            runtime.db.execute("BEGIN")
            rows = [export_record(runtime, key, success=value) for key, value in labels.items()]
            if calibrate_memory:
                for key, label in labels.items():
                    task = runtime.db.execute("SELECT spec,state FROM tasks WHERE id=?", (key,)).fetchone()
                    spec = json.loads(task["spec"])
                    if "memory_subject" in spec and task["state"] in {"succeeded", "rejected"}:
                        observations.append({
                            "subject": spec["memory_subject"],
                            "value": {"recipe": spec["recipe"], "passed": task["state"] == "succeeded"},
                            "observation_id": key, "success": label,
                            "provenance": {"source": "independent_runtime_labels", "task_id": key}})
    finally:
        runtime.close()
    if tool_cost is not None:
        for row in rows:
            row["model_cost"] = row["cost"] if row["tokens"] != 0 else 0
            row["tool_cost"] = row["tool_calls"] * tool_cost
            row["cost"] = (row["model_cost"] + row["tool_cost"]
                           if row["model_cost"] is not None else None)
            row["cost_scope"] = "observed_model_usage_plus_declared_per_attempt_tool_price"
    if observations:
        memory = GovernedMemory(Path(root) / "memory.sqlite")
        try:
            memory.observe_many(observations)
        finally:
            memory.close()
    summary = summarize(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("labels", type=Path, help="JSON object mapping task IDs to independent boolean labels")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tool-cost", type=float, help="Declared price per tool attempt")
    parser.add_argument("--calibrate-memory", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.root, json.loads(args.labels.read_text()), args.output,
                            calibrate_memory=args.calibrate_memory, tool_cost=args.tool_cost), indent=2))


if __name__ == "__main__":
    main()
