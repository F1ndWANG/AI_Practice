"""Exercise document import, unknown evidence, expanded tasks and export staging."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_practice.report import render
from rsi_agent.study import study
from vlm_posttraining.compare import compare
from vlm_posttraining.demo import png
from vlm_posttraining.export import export
from vlm_posttraining.import_docvqa import convert
from vlm_posttraining.pipeline import prepare, read_jsonl, write_jsonl
from vlm_posttraining.train import reward


class RealWorkflowTests(unittest.TestCase):
    def test_docvqa_labels_and_aliases_flow_through_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for i, split in enumerate(("train", "validation", "test")):
                image = root / f"document-{i}.png"
                png(image, (30, 50, 75), i + 10)
                annotations = root / f"{split}.json"
                annotations.write_text(json.dumps({"data": [{
                    "questionId": i, "image": image.name, "ucsf_document_id": f"doc-{i}",
                    "answers": ["Main Street", "main street", "Main St."], "question": "What street?",
                }]}), encoding="utf-8")
                sources.append((split, annotations))
            source = root / "manifest.jsonl"
            self.assertEqual(convert(sources, root, source)["with_aliases"], 3)
            prepared = root / "prepared"
            prepare(source, prepared, 42)
            test = prepared / "test.jsonl"
            row = next(read_jsonl(test))
            self.assertFalse(row["evidence_observed"])
            self.assertEqual(row["answer_aliases"], ["Main St."])

            wrong, correct = root / "base.jsonl", root / "candidate.jsonl"
            for path, response in ((wrong, "Side Street"), (correct, "Main St.")):
                write_jsonl(path, [{"id": row["id"], "completion": json.dumps({
                    "answer": response, "evidence": ["transcription"]
                })}])
            output = root / "report"
            result = compare(test, {"base": wrong, "candidate": correct}, output)
            self.assertEqual(result["paired"]["candidate"]["answer_accuracy"]["delta_accuracy"], 1)
            self.assertIsNone(result["paired"]["candidate"]["delta_grounded_accuracy"])
            self.assertIsNone(result["variants"]["candidate"]["overall"]["evidence"])
            review = next(read_jsonl(output / "review.jsonl"))
            self.assertEqual(review["evidence"], [])
            self.assertEqual(review["candidate_errors"], [])
            self.assertIn("答案准确率变化", render(output))
            self.assertEqual(reward(['{"answer":"Main St.","evidence":[] }'],
                                    ["Main Street"], [[]], answer_aliases=[["Main St."]],
                                    evidence_observed=[False]), [1.2])

    def test_extended_agent_exercises_json_and_quoted_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            result = study(output, per_family=1, extended=True)
            self.assertEqual(set(result["families"]), {"json", "csv", "words", "sum"})
            for name in ("csv", "json"):
                self.assertEqual(result["families"][name]["brs_drs"]["success_rate"], 1)
                self.assertLess(result["families"][name]["brs_drs"]["tool_calls"],
                                result["families"][name]["baseline"]["tool_calls"])

    def test_export_publishes_only_a_complete_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture weights")
            failed = Mock()
            failed.merge_and_unload.side_effect = ValueError("malformed adapter")
            peft = Mock()
            peft.from_pretrained.return_value = failed
            base = Mock()
            processor = Mock()
            dependencies = {
                "torch": SimpleNamespace(bfloat16="bf16", float32="fp32"),
                "peft": SimpleNamespace(PeftModel=peft),
                "transformers": SimpleNamespace(AutoModelForImageTextToText=base, AutoProcessor=processor),
            }

            def save(path, **options):
                path.mkdir()
                (path / "config.json").write_text("{}")
                (path / "model.safetensors").write_bytes(b"merged fixture")

            with patch.dict(sys.modules, dependencies):
                out = root / "output"
                with self.assertRaisesRegex(ValueError, "malformed"):
                    export("base", adapter, out)
                self.assertFalse(out.exists())
                failed.merge_and_unload.side_effect = None
                failed.merge_and_unload.return_value.save_pretrained.side_effect = save
                result = export("base", adapter, out)
                self.assertEqual(result["format"], "transformers-merged-lora")
                self.assertTrue((out / "export.json").is_file())


if __name__ == "__main__":
    unittest.main()
