"""Run a matched Agent ablation over several seeds and summarize its costs."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from ai_practice.artifacts import write_json

from .study import study


def spread(values):
    return {"mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def aggregate(results):
    variants = ("baseline", "brs", "brs_drs")
    summary = {}
    for variant in variants:
        samples = []
        for run in results:
            row = run["variants"][variant]
            baseline = run["variants"]["baseline"]
            broad_cost = run["exploration_tool_calls"]["brs"] if variant != "baseline" else 0
            deep_cost = run["exploration_tool_calls"]["drs"] if variant == "brs_drs" else 0
            samples.append({"seed": run["seed"], "success_rate": row["success_rate"],
                            "success_delta": row["success_rate"] - baseline["success_rate"],
                            "tool_calls_saved": baseline["tool_calls"] - row["tool_calls"],
                            "tool_calls_saved_including_exploration": (
                                baseline["tool_calls"] - row["tool_calls"] - broad_cost - deep_cost
                            )})
        keys = ("success_rate", "success_delta", "tool_calls_saved", "tool_calls_saved_including_exploration")
        summary[variant] = {"seeds": samples, **{key: spread([row[key] for row in samples]) for key in keys}}
    return summary


def matrix(output, seeds, per_family=8, budget=6, workers=4, extended=False):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("At least one seed is required; seed values must be unique")
    if min(per_family, budget, workers) < 1:
        raise ValueError("per-family, budget and workers must be positive")
    output.mkdir(parents=True, exist_ok=False)
    results = [study(output / f"seed-{seed}", seed, per_family, budget, workers, extended=extended)
               for seed in seeds]
    payload = {"kind": "agent_matrix", "seeds": seeds, "per_family": per_family,
               "budget": budget, "workers": workers, "extended": extended,
               "task_families": results[0]["test_tasks_by_family"], "variants": aggregate(results)}
    write_json(output / "summary.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--per-family", type=int, default=8)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    print(json.dumps(matrix(args.output, args.seeds, args.per_family, args.budget,
                            args.workers, args.extended), indent=2))


if __name__ == "__main__":
    main()
