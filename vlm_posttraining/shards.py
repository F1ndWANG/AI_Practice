"""Split blind inference across workers and merge verified predictions."""

import argparse
import heapq
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_practice.artifacts import sha256, write_json

from .blind import fingerprint
from .inference import DEFAULT_MODEL
from .pipeline import read_jsonl, write_jsonl
from .predict import model_settings, prediction_inputs, predict, publish


def profile_weights(profile, sample_ids):
    weights = {}
    for row in read_jsonl(profile):
        if not isinstance(row, dict):
            raise ValueError("Profile rows must be JSON objects")
        key, seconds = row.get("id"), row.get("generation_seconds")
        if (not isinstance(key, str) or key in weights or type(seconds) not in (int, float)
                or not 0 < seconds <= sys.float_info.max):
            raise ValueError("Profile requires unique IDs and finite positive generation_seconds")
        weights[key] = seconds
    if weights.keys() != set(sample_ids) or not math.isfinite(sum(weights.values(), 0.0)):
        raise ValueError("Profile must cover exactly the input IDs with a finite total duration")
    return weights


def balanced_assignments(sample_ids, weights, count):
    assignments = [[] for _ in range(count)]
    loads = [(0.0, index) for index in range(count)]
    for key in sorted(sample_ids, key=weights.__getitem__, reverse=True):
        load, index = loads[0]
        assignments[index].append(key)
        heapq.heapreplace(loads, (load + weights[key], index))
    return assignments


def shard_ids(plan, index):
    if "assignments" in plan:
        return plan["assignments"][index]
    return plan["sample_ids"][index::len(plan["shard_sha256"])]


def prepare(manifest, output, count, *, weights=None):
    if output.exists():
        raise FileExistsError(output)
    rows = prediction_inputs(manifest)
    if not 1 <= count <= len(rows):
        raise ValueError("Shard count must be between one and the number of samples")
    result = {"schema": 1, "input_sha256": fingerprint(rows),
              "sample_ids": [row["id"] for row in rows], "shard_sha256": []}
    if weights is not None:
        if weights.keys() != set(result["sample_ids"]):
            raise ValueError("Profile must cover exactly the input IDs")
        result["assignments"] = balanced_assignments(result["sample_ids"], weights, count)
        result["profile_sha256"] = fingerprint(weights)
        rows_by_id = {row["id"]: row for row in rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "shards"
        staged.mkdir()
        for index in range(count):
            path = staged / f"input-{index}.jsonl"
            shard_rows = rows[index::count]
            if weights is not None:
                shard_rows = (rows_by_id[key] for key in result["assignments"][index])
            write_jsonl(path, shard_rows)
            result["shard_sha256"].append(sha256(path))
        write_json(staged / "plan.json", result)
        staged.rename(output)
    return {"shards": count, "samples": len(rows)}


def load_plan(root):
    path = root / "plan.json"
    raw = path.read_bytes()
    plan = json.loads(raw)
    if (plan.get("schema") != 1 or not plan["shard_sha256"]
            or len(plan["shard_sha256"]) > len(plan["sample_ids"])
            or len(set(plan["sample_ids"])) != len(plan["sample_ids"])):
        raise ValueError("Invalid shard plan")
    if "assignments" in plan:
        assignments = plan["assignments"]
        if (not isinstance(assignments, list) or len(assignments) != len(plan["shard_sha256"])
                or any(not isinstance(group, list) or not group for group in assignments)):
            raise ValueError("Invalid shard assignments")
        assigned = [key for group in assignments for key in group]
        if (any(not isinstance(key, str) for key in assigned)
                or len(assigned) != len(plan["sample_ids"])
                or set(assigned) != set(plan["sample_ids"])):
            raise ValueError("Shard assignments must partition the input IDs")
    return plan, fingerprint(plan)


def planned_input(root, plan, index):
    path = root / f"input-{index}.jsonl"
    expected = shard_ids(plan, index)
    if sha256(path) != plan["shard_sha256"][index]:
        raise ValueError(f"Shard {index} input changed")
    if [row["id"] for row in read_jsonl(path)] != expected:
        raise ValueError(f"Shard {index} input IDs differ from plan")
    return path, expected


def checked_predictions(path, expected):
    rows = list(read_jsonl(path))
    if ([row["id"] for row in rows] != expected
            or any(not isinstance(row.get("completion"), str) for row in rows)):
        raise ValueError(f"Prediction IDs or completion fields do not match {path.name}")
    return rows


def run_shard(root, index, model=DEFAULT_MODEL, adapter=None, device="cuda"):
    plan, digest = load_plan(root)
    if not 0 <= index < len(plan["shard_sha256"]):
        raise ValueError("Shard index is out of range")
    manifest, expected = planned_input(root, plan, index)
    settings = model_settings(model, adapter, device)
    predictions = root / f"predictions-{index}.jsonl"
    predict(manifest, predictions, model, adapter, device, resume=True)
    if settings != model_settings(model, adapter, device):
        raise ValueError("Model settings changed during prediction")
    checked_predictions(predictions, expected)
    result = {"schema": 1, "index": index, "plan_sha256": digest, "model_settings": settings,
              "samples": len(expected), "predictions_sha256": sha256(predictions)}
    write_json(root / f"receipt-{index}.json", result)
    return result


def completed_shard(root, plan, digest, index):
    _, expected = planned_input(root, plan, index)
    receipt = root / f"receipt-{index}.json"
    predictions = root / f"predictions-{index}.jsonl"
    if not receipt.is_file() or not predictions.is_file():
        raise ValueError(f"Missing completed shard {index}")
    completed = json.loads(receipt.read_text(encoding="utf-8"))
    if (completed.get("schema") != 1 or completed.get("index") != index
            or completed.get("plan_sha256") != digest or completed.get("samples") != len(expected)
            or completed.get("predictions_sha256") != sha256(predictions)):
        raise ValueError(f"Invalid receipt or predictions for shard {index}")
    return completed, checked_predictions(predictions, expected)


def merge(root, output):
    plan, digest = load_plan(root)
    count = len(plan["shard_sha256"])
    merged = {}
    settings = None
    for index in range(count):
        completed, rows = completed_shard(root, plan, digest, index)
        if index == 0:
            settings = completed["model_settings"]
        if completed["model_settings"] != settings:
            raise ValueError(f"Model settings differ for shard {index}")
        merged.update((row["id"], row) for row in rows)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    publish(output, (merged[key] for key in plan["sample_ids"]))
    return {"shards": count, "samples": len(merged), "predictions_sha256": sha256(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("prepare")
    split.add_argument("manifest", type=Path)
    split.add_argument("root", type=Path)
    split.add_argument("--shards", type=int, required=True)
    split.add_argument("--profile", type=Path, help="Pilot predictions JSONL")
    run = sub.add_parser("run")
    run.add_argument("root", type=Path)
    run.add_argument("index", type=int)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    join = sub.add_parser("merge")
    join.add_argument("root", type=Path)
    join.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        weights = (profile_weights(args.profile, (row["id"] for row in prediction_inputs(args.manifest)))
                   if args.profile else None)
        result = prepare(args.manifest, args.root, args.shards, weights=weights)
    elif args.command == "run":
        result = run_shard(args.root, args.index, args.model, args.adapter, args.device)
    else:
        result = merge(args.root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
