"""Compare GPU counts or shard strategies with repeated, paired inference runs."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

from ai_practice.artifacts import sha256, write_json
from ai_practice.report import table

from .blind import fingerprint
from .inference import DEFAULT_MODEL
from .pipeline import metrics, score_predictions, write_jsonl
from .predict import model_settings, prediction_inputs
from .shard_experiment import pair, run
from .shards import profile_weights


def spread(values):
    if not values:
        return {"runs": 0, "median": None, "min": None, "max": None}
    return {"runs": len(values), "median": median(values), "min": min(values), "max": max(values)}


def execute(manifest, output, groups, *, repeats=3, model=DEFAULT_MODEL, adapter=None, reference=None,
            profile=None):
    if profile is not None:
        if len(groups) != 1:
            raise ValueError("A strategy comparison requires exactly one GPU group")
    elif len(groups) < 2 or len(groups[0]) != 1:
        raise ValueError("Provide a single-GPU baseline, at least one alternative")
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    for group in groups:
        if (not group or len(set(group)) != len(group)
                or any(not gpu.isascii() or not gpu.isdecimal() or len(gpu) > 1 and gpu.startswith("0")
                       for gpu in group)):
            raise ValueError("GPU groups must contain unique, canonical numeric IDs")
    if len({tuple(sorted(group)) for group in groups}) != len(groups):
        raise ValueError("GPU groups must be distinct")
    inputs = prediction_inputs(manifest)
    input_digest = fingerprint(inputs)
    profile_digest = None
    if profile is not None:
        weights = profile_weights(profile, (row["id"] for row in inputs))
        profile_digest = fingerprint(weights)
        groups = [groups[0], groups[0]]
    strategies = ["round_robin"] * len(groups)
    if profile is not None:
        strategies[1] = "profile"
    settings = model_settings(model, adapter, "cuda")
    reference_digest = sha256(reference) if reference else None
    output.mkdir(parents=True, exist_ok=False)
    if profile is not None:
        profile = output / "profile.jsonl"
        write_jsonl(profile, ({"id": key, "generation_seconds": seconds}
                              for key, seconds in sorted(weights.items())))
    rounds = []
    for repetition in range(repeats):
        offset = repetition % len(groups)
        order = list(range(offset, len(groups))) + list(range(offset))
        results = {}
        comparisons = {}
        for index in order:
            folder = output / f"repeat-{repetition + 1}" / f"group-{index}"
            try:
                candidate_profile = profile if strategies[index] == "profile" else None
                summary = run(manifest, folder, groups[index], model=model, adapter=adapter,
                              profile=candidate_profile)
                if not summary["passed"]:
                    results[index] = {"state": "failed", "failed_shards": summary["failed_shards"]}
                    continue
                if summary["input_sha256"] != input_digest or summary["model_settings"] != settings:
                    raise ValueError("Inference inputs or model changed across rounds")
                expected_profile = profile_digest if candidate_profile else None
                if summary.get("profile_sha256") != expected_profile:
                    raise ValueError("Timing profile changed across rounds")
                throughput = summary["samples_per_second"]
                if not summary["full_run_timing"] or throughput is None or not math.isfinite(throughput) or throughput <= 0:
                    raise ValueError("Missing full-run timing")
                result = {"state": "passed", "samples_per_second": throughput,
                          "end_to_end_seconds": summary["end_to_end_seconds"],
                          "slowest_worker_seconds": max(worker["seconds"] for worker in summary["workers"])}
                if reference:
                    if sha256(reference) != reference_digest:
                        raise ValueError("Reference labels changed across rounds")
                    scored = score_predictions(reference, folder / "predictions.jsonl")
                    result["quality"] = metrics(scored)
                results[index] = result
            except (OSError, ValueError, RuntimeError) as error:
                results[index] = {"state": "error", "message": str(error)}
        if results[0]["state"] == "passed":
            for index in range(1, len(groups)):
                if results[index]["state"] != "passed":
                    continue
                folder = output / f"repeat-{repetition + 1}"
                try:
                    comparisons[index] = pair(folder / "group-0", folder / f"group-{index}")
                except (OSError, ValueError) as error:
                    comparisons[index] = {"error": str(error)}
        rounds.append({"repeat": repetition + 1, "order": order, "runs": results, "comparisons": comparisons})
        write_json(output / "rounds.json", rounds)

    report_groups = []
    for index, gpus in enumerate(groups):
        successes = [row["runs"][index] for row in rounds if row["runs"][index]["state"] == "passed"]
        matched = [row["comparisons"][index] for row in rounds
                   if index in row["comparisons"] and "speedup" in row["comparisons"][index]]
        report_groups.append({
            "gpus": gpus, "strategy": strategies[index],
            "samples_per_second": spread([row["samples_per_second"] for row in successes]),
            "end_to_end_seconds": spread([row["end_to_end_seconds"] for row in successes]),
            "slowest_worker_seconds": spread([row["slowest_worker_seconds"] for row in successes]),
            "paired_slowest_worker_delta_seconds": spread([
                row["candidate_slowest_worker_seconds"] - row["baseline_slowest_worker_seconds"]
                for row in matched
            ]),
            "paired_speedup": spread([row["speedup"] for row in matched]),
            "changed_completions": spread([row["changed_completions"] for row in matched]),
            "answer_accuracy": spread([row["quality"]["accuracy"] for row in successes if "quality" in row]),
        })
    passed = all(len(row["runs"]) == len(groups) and all(item["state"] == "passed" for item in row["runs"].values())
                 and len(row["comparisons"]) == len(groups) - 1
                 and all("speedup" in item for item in row["comparisons"].values()) for row in rounds)
    result = {"kind": "sharded_inference_matrix", "repeats": repeats, "passed": passed,
              "input_sha256": input_digest, "model_settings": settings, "reference_sha256": reference_digest,
              "profile_sha256": profile_digest, "groups": report_groups}
    write_json(output / "summary.json", result)

    def interval(item):
        return "—" if item["median"] is None else f"{item['median']:.3f} [{item['min']:.3f}, {item['max']:.3f}]"

    title = "VLM 同卡数分片策略对照" if profile else "VLM 单卡与多卡对照"
    lines = [f"# {title}", "", f"计划重复 {repeats} 次，{'全部完成' if passed else '部分未完成'}", "",
             *table(("GPU", "策略", "完成次数", "样本/秒", "耗时（秒）", "最慢 worker（秒）",
                     "最慢 worker 配对差值（秒）", "配对加速比", "输出变化数", "答案准确率"),
                    [(",".join(item["gpus"]), item["strategy"], item["samples_per_second"]["runs"],
                      interval(item["samples_per_second"]), interval(item["end_to_end_seconds"]),
                      interval(item["slowest_worker_seconds"]),
                      interval(item["paired_slowest_worker_delta_seconds"]),
                      interval(item["paired_speedup"]), interval(item["changed_completions"]),
                      interval(item["answer_accuracy"])) for item in report_groups]), "",
             "各值为中位数及 [最小值, 最大值]。加速比使用同轮基线。最慢 worker 差值按候选减基线计算。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--group", action="append", required=True,
                        help="Comma-separated GPU IDs; use one group with --profile")
    parser.add_argument("--profile", type=Path, help="Pilot predictions for a same-GPU strategy comparison")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--reference", type=Path, help="Optional labeled manifest for post-inference scoring")
    args = parser.parse_args()
    result = execute(args.manifest, args.output, [group.split(",") for group in args.group],
                     repeats=args.repeats, model=args.model, adapter=args.adapter, reference=args.reference,
                     profile=args.profile)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
