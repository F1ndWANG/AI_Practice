"""Execute and measure a single-host multi-GPU inference experiment."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_practice.artifacts import sha256, write_json

from .blind import fingerprint
from .inference import DEFAULT_MODEL
from .pipeline import read_jsonl
from .predict import model_settings, prediction_inputs
from .shards import completed_shard, load_plan, merge, prepare


def worker(root, index, gpu, model, adapter):
    command = [sys.executable, "-m", "vlm_posttraining.shards", "run", str(root), str(index),
               "--model", model]
    if adapter:
        command.extend(["--adapter", str(adapter)])
    started = time.perf_counter()
    with (root.parent / f"worker-{index}.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, check=False)
    return {"index": index, "gpu": gpu, "exit_code": result.returncode,
            "seconds": time.perf_counter() - started}


def run(manifest, output, gpus, *, model=DEFAULT_MODEL, adapter=None, resume=False):
    if not gpus or len(set(gpus)) != len(gpus) or any(not gpu or not gpu.isascii() or not gpu.isdecimal()
                                                     for gpu in gpus):
        raise ValueError("Use distinct nonnegative GPU IDs")
    settings = model_settings(model, adapter, "cuda")
    digest = fingerprint(prediction_inputs(manifest))
    spec = {"schema": 1, "gpus": gpus, "model_settings": settings, "input_sha256": digest}
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Experiment already running") from error
        return execute_experiment(manifest, output, spec, model, adapter)


def execute_experiment(manifest, output, spec, model, adapter):
    settings = output / "experiment.json"
    if settings.exists() and json.loads(settings.read_text(encoding="utf-8")) != spec:
        raise ValueError("Experiment input, model or GPU assignment changed")
    if not settings.exists():
        write_json(settings, spec)

    shards = output / "shards"
    if not shards.exists():
        prepare(manifest, shards, len(spec["gpus"]))
    plan, plan_digest = load_plan(shards)
    if (plan["input_sha256"] != spec["input_sha256"]
            or len(plan["shard_sha256"]) != len(spec["gpus"])):
        raise ValueError("Shard plan does not match the experiment")

    pending = []
    reused = []
    for index, gpu in enumerate(spec["gpus"]):
        if (shards / f"receipt-{index}.json").is_file():
            receipt, _ = completed_shard(shards, plan, plan_digest, index)
            if receipt["model_settings"] != spec["model_settings"]:
                raise ValueError(f"Model settings differ for completed shard {index}")
            reused.append(index)
        else:
            pending.append((index, gpu))

    merged = output / "predictions.jsonl"
    summary_file = output / "summary.json"
    if merged.exists() and summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if (pending or summary.get("plan_sha256") != plan_digest
                or summary.get("predictions_sha256") != sha256(merged)):
            raise ValueError("Existing merged results differ from the experiment")
        return summary

    attempt_path = output / "last_attempt.json"
    full_run = not reused and not attempt_path.exists() and not any(shards.glob(".*.prediction.sqlite"))
    write_json(attempt_path, {"state": "running", "pending": [index for index, _ in pending]})
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(spec["gpus"])) as pool:
        jobs = [pool.submit(worker, shards, index, gpu, model, adapter) for index, gpu in pending]
        workers = [job.result() for job in jobs]
    elapsed = time.perf_counter() - started
    failures = [row["index"] for row in workers if row["exit_code"]]
    summary = {**spec, "kind": "sharded_inference_run", "plan_sha256": plan_digest,
               "reused_shards": reused, "workers": workers, "worker_wall_seconds": elapsed,
               "failed_shards": failures, "passed": not failures}
    write_json(attempt_path, summary)
    if failures:
        return summary

    merge_started = time.perf_counter()
    if merged.exists():
        # Recover if the process stopped after publishing results and before writing the summary.
        recovery = output / ".verify-merge.jsonl"
        recovery.unlink(missing_ok=True)
        try:
            result = merge(shards, recovery)
            if sha256(merged) != result["predictions_sha256"]:
                raise ValueError("Existing merged predictions differ from verified shard results")
        finally:
            recovery.unlink(missing_ok=True)
    else:
        result = merge(shards, merged)
    merge_seconds = time.perf_counter() - merge_started
    processed = sum(len(plan["sample_ids"][index::len(spec["gpus"])]) for index, _ in pending)
    total_seconds = elapsed + merge_seconds
    summary.update({**result, "merge_seconds": merge_seconds, "assigned_samples": processed,
                    "end_to_end_seconds": total_seconds,
                    "samples_per_second": processed / total_seconds if full_run and processed else None,
                    "full_run_timing": full_run})
    write_json(summary_file, summary)
    return summary


def pair(left_dir, right_dir):
    left, right = [json.loads((path / "summary.json").read_text(encoding="utf-8"))
                   for path in (left_dir, right_dir)]
    for folder, item in ((left_dir, left), (right_dir, right)):
        if not item["passed"] or not item["full_run_timing"]:
            raise ValueError("Comparison requires complete runs without reused shards")
        if sha256(folder / "predictions.jsonl") != item["predictions_sha256"]:
            raise ValueError("Prediction result changed")
    if any(left[key] != right[key] for key in ("input_sha256", "model_settings", "samples")):
        raise ValueError("Comparison requires matching model settings and input samples")
    baseline = list(read_jsonl(left_dir / "predictions.jsonl"))
    candidate = list(read_jsonl(right_dir / "predictions.jsonl"))
    if [row["id"] for row in baseline] != [row["id"] for row in candidate]:
        raise ValueError("Comparison requires matching output IDs")
    changed = sum(a["completion"] != b["completion"] for a, b in zip(baseline, candidate))
    return {"changed_completions": changed, "samples": left["samples"],
            "baseline_gpus": left["gpus"], "candidate_gpus": right["gpus"],
            "baseline_samples_per_second": left["samples_per_second"],
            "candidate_samples_per_second": right["samples_per_second"],
            "speedup": right["samples_per_second"] / left["samples_per_second"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    single = commands.add_parser("run")
    single.add_argument("manifest", type=Path)
    single.add_argument("output", type=Path)
    single.add_argument("--gpus", required=True, help="Comma-separated physical GPU IDs")
    single.add_argument("--model", default=DEFAULT_MODEL)
    single.add_argument("--adapter", type=Path)
    single.add_argument("--resume", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        result = run(args.manifest, args.output, args.gpus.split(","), model=args.model,
                     adapter=args.adapter, resume=args.resume)
    else:
        result = pair(args.baseline, args.candidate)
    print(json.dumps(result, indent=2))
    if args.action == "run" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
