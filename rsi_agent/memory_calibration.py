"""Audit memory probabilities on independent labels excluded from calibration."""

import argparse
import hashlib
import json
from pathlib import Path

from .governance import GovernedMemory


def evaluate(memory, rows):
    errors = []
    bins = [[] for _ in range(10)]
    covered = 0
    for row in rows:
        if type(row["success"]) is not bool:
            raise ValueError("Independent labels must be boolean")
        digest = hashlib.sha256(json.dumps(
            row["value"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if memory.db.execute(
            "SELECT 1 FROM memory_observations WHERE subject=? AND digest=? AND observation_id=?",
            (row["subject"], digest, row["id"])).fetchone():
            raise ValueError("Calibration observations cannot also be evaluation observations")
        result = memory.recall(row["subject"], min_confidence=0,
                               require_calibrated=True, min_observations=5)
        candidate = next((item for item in result["candidates"] if item["value"] == row["value"]), None)
        if candidate is None:
            continue
        covered += 1
        probability = candidate["confidence"]
        label = int(row["success"])
        errors.append((probability - label) ** 2)
        bins[min(int(probability * 10), 9)].append((probability, label))
    ece = sum(abs(sum(p for p, _ in group) - sum(y for _, y in group))
              for group in bins) / covered if covered else None
    return {"samples": len(rows), "covered": covered,
            "coverage": covered / len(rows) if rows else 0,
            "brier_score": sum(errors) / covered if covered else None, "ece_10_bins": ece}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("memory", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args()
    memory = GovernedMemory(args.memory)
    try:
        rows = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
        print(json.dumps(evaluate(memory, rows), indent=2))
    finally:
        memory.close()


if __name__ == "__main__":
    main()
