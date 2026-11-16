"""Run and resume a single-GPU base/SFT/GRPO experiment."""

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

from ai_practice.artifacts import artifact_index, sha256, verify_index, write_json

from .blind import build as build_blind, verify as verify_blind
from .compare import compare
from .pipeline import prepare, validate_images
from .train import DEFAULT_MODEL


def adapter_paths(folder):
    return [folder / name for name in ("adapter_config.json", "adapter_model.safetensors",
                                       "run.json", "trainer_state.json")]


def latest_checkpoint(folder):
    candidates = []
    for path in folder.glob("checkpoint-*"):
        number = path.name.removeprefix("checkpoint-")
        required = ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                    "optimizer.pt", "scheduler.pt", "rng_state.pth")
        if number.isdecimal() and all((path / name).is_file() for name in required):
            candidates.append((int(number), path))
    return max(candidates)[1] if candidates else None


def launch(arguments, log):
    command = [sys.executable, "-m", "vlm_posttraining.train", *map(str, arguments)]
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(command, ensure_ascii=False) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def execute(output, manifest, config, evaluate_test):
    status_path = output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"stages": {}}
    stages = status["stages"]
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    prepared = output / "prepared"
    training = prepared / "train.jsonl"

    def stage(name, outputs, action, inputs=()):
        earlier = stages.get(name, {})
        dependencies = artifact_index(output, inputs)
        if earlier.get("state") == "completed" and earlier.get("inputs", {}) == dependencies:
            verify_index(output, earlier["artifacts"])
            return
        stages[name] = {"state": "running"}
        write_json(status_path, status)
        try:
            action()
            stages[name] = {"state": "completed", "artifacts": artifact_index(output, outputs),
                            "inputs": dependencies}
        except Exception as error:
            stages[name] = {"state": "failed", "error": str(error)}
            raise
        finally:
            write_json(status_path, status)

    def prepare_data():
        stats = prepare(manifest, prepared, config["seed"])
        write_json(prepared / "statistics.json", stats)

    stage("prepare", [prepared / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl",
                                                   "statistics.json")], prepare_data)
    validate_images(prepared)
    common = ["--model", config["model"], "--seed", config["seed"]]
    adapters = {"base": None}
    variants = [("sft", "sft", None), ("grpo", "grpo", 0.5)]
    if config["ablate_evidence"]:
        variants.append(("grpo_no_evidence", "grpo", 0.0))
    for name, mode, evidence_weight in variants:
        folder = output / name
        args = [mode, training, folder, *common, "--steps", config[f"{mode}_steps"],
                "--save-steps", config["save_steps"]]
        if mode == "grpo":
            args.extend(["--adapter", output / "sft", "--generations", config["generations"],
                         "--evidence-weight", evidence_weight])
            if config["vllm"]:
                args.append("--vllm")

        def train_variant():
            checkpoint = latest_checkpoint(folder)
            resume = ["--resume", checkpoint] if checkpoint else []
            launch([*args, *resume], logs / f"{name}.log")

        stage(f"train_{name}", adapter_paths(folder), train_variant)
        adapters[name] = folder

    splits = ("validation", "test") if evaluate_test else ("validation",)
    for split in splits:
        evaluation = prepared / f"{split}.jsonl"
        dispatch = output / "inference" / f"{split}.jsonl"
        binding = prepared / f"{split}-binding.json"
        stage(f"dispatch_{split}", [dispatch, binding],
              lambda: build_blind(evaluation, dispatch, binding), [evaluation])
        verify_blind(evaluation, dispatch, binding)
        predictions = {}
        for name, adapter in adapters.items():
            pred = output / f"{split}-{name}.jsonl"
            args = ["predict", dispatch, pred, *common]
            if adapter:
                args.extend(["--adapter", adapter])
            inputs = [dispatch] + (adapter_paths(adapter) if adapter else [])
            stage(f"predict_{split}_{name}", [pred],
                  lambda: launch(args, logs / f"predict_{split}_{name}.log"), inputs)
            predictions[name] = pred

        report_path = output / f"comparison-{split}"

        def compare_split():
            # compare reserves a new folder. A partial report can only contain
            # these two generated files, so retry without touching other content.
            if report_path.is_dir():
                for name in ("summary.json", "review.jsonl"):
                    (report_path / name).unlink(missing_ok=True)
                report_path.rmdir()
            compare(evaluation, predictions, report_path, seed=config["seed"])

        stage(f"compare_{split}", [report_path / "summary.json", report_path / "review.jsonl"],
              compare_split, [evaluation, binding, *predictions.values()])
    return status


def run(output, manifest, config, resume, evaluate_test=False):
    output = output.resolve()
    manifest = manifest.resolve()
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is already using this run directory") from error
        spec = {"schema": 1, "manifest": str(manifest), "manifest_sha256": sha256(manifest), **config}
        spec_path = output / "experiment.json"
        if spec_path.exists():
            if json.loads(spec_path.read_text()) != spec:
                raise ValueError("Run settings or manifest changed. Choose a new output directory.")
        else:
            if resume:
                raise ValueError("Resume requires a previously initialized run directory")
            write_json(spec_path, spec)
        return execute(output, manifest, config, evaluate_test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-steps", type=int, default=50)
    parser.add_argument("--grpo-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--ablate-evidence", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Also evaluate the holdout after validation")
    args = parser.parse_args()
    if min(args.sft_steps, args.grpo_steps, args.save_steps) < 1 or args.generations < 2:
        parser.error("Step counts must be positive and GRPO needs at least two generations")
    config = {key: value for key, value in vars(args).items() if key not in {"output", "manifest", "resume", "evaluate_test"}}
    print(json.dumps(run(args.output, args.manifest, config, args.resume, args.evaluate_test), indent=2))


if __name__ == "__main__":
    main()
