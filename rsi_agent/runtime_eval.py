"""Run local runtime controls, recovery cases and verifier calibration."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .durable import Runtime
from .runtime_metrics import summarize
from .runtime_planning import export_record


def evaluate(output, tool_cost=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = Runtime(output / "runtime")
    rows = []
    try:
        for index in range(8):
            expected = "apple\npear\n"
            # Include deliberately incorrect tools and permissive verifier controls.
            correct = index % 4 < 2
            recipe = ["normalize", "sort_normalized"] if correct else ["copy"]
            rules = [{"kind": "text", "expected": expected}] if index % 4 != 3 else [{"kind": "file"}]
            key = runtime.submit(f"task-{index}", "Pear\napple\nPear\n", recipe, rules)
            if correct and index >= 4:
                runtime.run(key, interrupt_after=1)
            result = runtime.run(key)
            artifact = output / "runtime" / key / f"checkpoint-{result['next_step']}" / "output.txt"
            success = artifact.is_file() and hashlib.sha256(artifact.read_bytes()).hexdigest() == hashlib.sha256(expected.encode()).hexdigest()
            record = export_record(runtime, key, success=success)
            record["cost"] = result["calls"] * tool_cost if tool_cost is not None else None
            rows.append(record)
    finally:
        runtime.close()
    report = summarize(rows)
    report["scope"] = "local_allowlisted_tools_with_intentional_verifier_controls"
    report["latency_scope"] = "submission_to_terminal_event_including_recovery_downtime"
    (output / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tool-cost", type=float, help="Declared price per tool attempt, in your chosen currency")
    args = parser.parse_args()
    if args.tool_cost is not None and (args.tool_cost < 0 or not math.isfinite(args.tool_cost)):
        parser.error("Tool cost must be finite and nonnegative")
    print(json.dumps(evaluate(args.output, args.tool_cost), indent=2))


if __name__ == "__main__":
    main()
