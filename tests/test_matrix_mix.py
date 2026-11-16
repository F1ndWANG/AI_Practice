"""Match seed-wise deltas and audit cross-dataset data integrity."""

import tempfile
import unittest
from pathlib import Path

from ai_practice.report import render
from rsi_agent.matrix import matrix
from vlm_posttraining.demo import png
from vlm_posttraining.mix import mix, verify
from vlm_posttraining.pipeline import read_jsonl, write_jsonl


class MatrixMixTests(unittest.TestCase):
    def test_seed_matching_and_exploration_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "matrix"
            seeds = [3, 4]
            result = matrix(out, seeds, per_family=1, workers=1)
            drs = result["variants"]["brs_drs"]
            self.assertEqual([item["seed"] for item in drs["seeds"]], seeds)
            self.assertEqual(drs["tool_calls_saved"]["mean"], 4)
            self.assertLess(drs["tool_calls_saved_including_exploration"]["mean"], 0)
            self.assertTrue((out / "seed-4/baseline.jsonl").is_file())
            self.assertIn("跨种子对照", render(out))

    def test_mix_namespaces_ids_detects_cross_source_leakage_and_checks_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alpha, beta = root / "alpha", root / "beta"
            alpha.mkdir()
            beta.mkdir()
            paths = [alpha / "train.png", alpha / "val.png", alpha / "test.png", beta / "test.png"]
            for index, path in enumerate(paths):
                png(path, (25, 10, 30), index + 15)
            split_names = ("train", "validation", "test")

            def reference(sample_id, split, image):
                return {"id": sample_id, "split": split, "source_group": sample_id,
                        "image": image, "task": "ocr", "difficulty": "easy",
                        "question": "What is the number?", "answer": "12", "evidence": []}

            left = alpha / "source.jsonl"
            write_jsonl(left, [reference(str(i), split, paths[i].name) for i, split in enumerate(split_names)])
            right = beta / "source.jsonl"
            independent = reference("0", "test", "test.png")
            twin = reference("duplicate", "train", "../alpha/train.png")
            write_jsonl(right, [independent, twin])
            output = root / "mixed"
            summary = mix({"alpha": left, "beta": right}, output)
            self.assertEqual(summary["sources"]["beta"]["duplicates_removed"], 1)
            self.assertEqual(summary["splits"]["test"]["by_dataset"], {"alpha": 1, "beta": 1})
            self.assertEqual({row["id"] for row in read_jsonl(output / "prepared/test.jsonl")},
                             {"alpha:2", "beta:0"})
            self.assertEqual(verify(output)["verified_manifests"], 4)
            self.assertIn("多数据源审计", render(output))
            png(beta / "test.png", (10, 220, 100), 21)
            with self.assertRaisesRegex(ValueError, "Image changed"):
                verify(output)
            write_jsonl(right, [independent, {**twin, "question": "New prompt?", "split": "test"}])
            rejected = root / "rejected"
            with self.assertRaisesRegex(ValueError, "leakage"):
                mix({"alpha": left, "beta": right}, rejected)
            self.assertFalse(rejected.exists())

            original = list(read_jsonl(left))
            original[0]["shared_source_group"] = "external-doc-31"
            write_jsonl(left, original)
            write_jsonl(right, [{**independent, "shared_source_group": "external-doc-31"}])
            with self.assertRaisesRegex(ValueError, "leakage"):
                mix({"alpha": left, "beta": right}, rejected)
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
