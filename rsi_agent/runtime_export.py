"""Export task telemetry against independently supplied success labels."""

import argparse
import json
from pathlib import Path

from .durable import Runtime
from .runtime_metrics import summarize
from .runtime_planning import export_record


def export(root, labels, output):
    if not (Path(root) / "runtime.sqlite").is_file():
        raise ValueError("Runtime database does not exist")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("Provide a task-ID to independent success-label mapping")
    runtime = Runtime(root)
    try:
        rows = [export_record(runtime, key, success=value) for key, value in labels.items()]
    finally:
        runtime.close()
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
    args = parser.parse_args()
    print(json.dumps(export(args.root, json.loads(args.labels.read_text()), args.output), indent=2))


if __name__ == "__main__":
    main()
