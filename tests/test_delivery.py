"""Exercise the acceptance receipt and an interrupted experiment without CUDA."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_practice.__main__ import smoke, verify
from rsi_agent.study import load_suite
from vlm_posttraining.experiment import adapter_paths, run
from vlm_posttraining.fixtures import generate
from vlm_posttraining.pipeline import read_jsonl, write_jsonl


class DeliveryTests(unittest.TestCase):
    def test_acceptance_receipt_detects_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            smoke(output, 42)
            self.assertGreater(verify(output)["verified"], 75)
            (output / "summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "summary.json"):
                verify(output)

    def test_custom_suite_rejects_cross_split_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "suite.jsonl"
            tasks = [{"id": phase, "split": phase, "family": "sum",
                      "source": f"{i + 1}\n", "expected": f"{i + 1}\n"}
                     for i, phase in enumerate(("broad", "probe", "practice", "test"))]
            write_jsonl(suite, tasks)
            self.assertEqual(len(load_suite(suite)["test"]), 1)
            tasks[-1]["source"] = tasks[0]["source"]
            write_jsonl(suite, tasks)
            with self.assertRaisesRegex(ValueError, "reused across splits"):
                load_suite(suite)

    def test_experiment_resumes_failed_training_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            generate(data)
            manifest = data / "manifest.jsonl"
            output = root / "run"
            config = {"model": "fixture", "seed": 42, "sft_steps": 11, "grpo_steps": 11,
                      "save_steps": 10, "generations": 2, "ablate_evidence": True, "vllm": False}
            calls = []

            def fake_launch(arguments, log):
                stage, input_path, destination, *flags = arguments
                calls.append((stage, destination.name))
                if stage == "predict":
                    self.assertIn("--resume-prediction", flags)
                    self.assertTrue(all("answer" not in row and "evidence" not in row
                                        for row in read_jsonl(input_path)))
                    reference = output / "prepared" / input_path.name
                    write_jsonl(destination, (
                        {"id": row["id"], "completion": json.dumps({
                            key: row[key] for key in ("answer", "evidence")
                        })}
                        for row in read_jsonl(reference)
                    ))
                    return
                paths = adapter_paths(destination)
                if stage == "grpo" and destination.name == "grpo" and "--resume" not in flags:
                    checkpoint = destination / "checkpoint-10"
                    checkpoint.mkdir(parents=True)
                    for file in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                                 "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                        (checkpoint / file).touch()
                    raise subprocess.CalledProcessError(1, "fixture")
                destination.mkdir(exist_ok=True)
                for file in paths:
                    file.write_text(stage)

            with patch("vlm_posttraining.experiment.launch", fake_launch):
                with self.assertRaises(subprocess.CalledProcessError):
                    run(output, manifest, config, resume=False)
                status = json.loads((output / "status.json").read_text())
                self.assertEqual(status["stages"]["train_grpo"]["state"], "failed")
                result = run(output, manifest, config, resume=True)
                self.assertEqual(calls.count(("sft", "sft")), 1)
                self.assertEqual(calls.count(("grpo", "grpo")), 2)
                self.assertNotIn("compare_test", result["stages"])
                self.assertIn("compare_validation", result["stages"])
                result = run(output, manifest, config, resume=True, evaluate_test=True)
                self.assertIn("compare_test", result["stages"])
                count_before = len(calls)
                run(output, manifest, config, resume=True)
                self.assertEqual(len(calls), count_before)
                with self.assertRaisesRegex(ValueError, "settings"):
                    run(output, manifest, {**config, "seed": 3}, resume=True)
                (output / "sft/adapter_model.safetensors").write_text("modified")
                with self.assertRaisesRegex(ValueError, "changed artifact"):
                    run(output, manifest, config, resume=True)


if __name__ == "__main__":
    unittest.main()
