"""Deterministic, allowlisted operations on local task inputs."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def canonical_json(source, sort_keys):
    return json.dumps(json.loads(source), ensure_ascii=False, sort_keys=sort_keys,
                      separators=(",", ":")) + "\n"


def csv_totals(source, *, robust):
    totals = defaultdict(int)
    if robust:
        records = ((row["category"], row["amount"]) for row in csv.DictReader(source.splitlines()))
    else:
        # An exploratory shortcut that works for the clean two-column exercise.
        records = (line.split(",") for line in source.splitlines()[1:])
    for category, amount in records:
        totals[category] += int(amount)
    return "".join(f"{category}\t{totals[category]}\n" for category in sorted(totals))


def main():
    operation = sys.argv[1]
    source = Path("input.txt").read_text(encoding="utf-8")
    tools = {
        "json_compact": lambda: canonical_json(source, False),
        "json_canonical": lambda: canonical_json(source, True),
        "csv_simple_totals": lambda: csv_totals(source, robust=False),
        "csv_totals": lambda: csv_totals(source, robust=True),
    }
    Path("output.txt").write_text(tools[operation](), encoding="utf-8")


if __name__ == "__main__":
    main()
