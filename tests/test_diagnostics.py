"""Check actual budget reruns and risk signals in reward diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path

from ai_practice.report import render
from rsi_agent.budgets import sweep
from rsi_agent.study import study
from vlm_posttraining.fixtures import generate
from vlm_posttraining.pipeline import read_jsonl, write_jsonl
from vlm_posttraining.reward_audit import audit
from vlm_posttraining.train import reward


class DiagnosticsTests(unittest.TestCase):
    def test_budget_reruns_match_the_original_frozen_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "original"
            baseline = study(parent, per_family=1)
            output = root / "budgets"
            report = sweep(parent, output, [1, 6])
            self.assertEqual(report["budgets"]["1"]["baseline"]["success_rate"], 0)
            self.assertEqual(report["budgets"]["6"]["brs_drs"]["tool_calls"],
                             baseline["variants"]["brs_drs"]["tool_calls"])
            self.assertEqual(report["budgets"]["6"]["brs_drs"]["estimated_break_even_tasks"], 8)
            self.assertIn("估算回本任务数", render(output))

    def test_reward_audit_identifies_inversion_and_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = root / "fixtures"
            generate(fixtures)
            manifest = fixtures / "prepared/test.jsonl"
            one, two = list(read_jsonl(manifest))[:2]

            def completion(answer, evidence):
                return json.dumps({"answer": answer, "evidence": evidence})

            candidates = root / "candidates.jsonl"
            first_group = [completion(one["answer"], []), completion("WRONG", one["evidence"])]
            second_group = [completion(two["answer"], two["evidence"])] * 2
            write_jsonl(candidates, [{"id": one["id"], "completions": first_group},
                                     {"id": two["id"], "completions": second_group}])
            output = root / "audit"
            result = audit(manifest, output, 2.0, candidates)
            self.assertEqual(result["overall"]["answer_preference_inversions"], 1)
            self.assertEqual(result["overall"]["zero_signal_groups"], 1)
            self.assertEqual(result["reference_samples"], 18)
            details = next(read_jsonl(output / "groups.jsonl"))
            self.assertEqual([item["reward"] for item in details["scores"]],
                             reward(first_group, [one["answer"]] * 2, [one["evidence"]] * 2,
                                    evidence_weight=2.0))
            self.assertIn("正确答案排序反转", render(output))
            fixture_result = audit(manifest, root / "fixture_audit")
            self.assertEqual(fixture_result["overall"]["answer_preference_inversions"], 0)
            self.assertEqual(fixture_result["overall"]["malformed_candidates"], 18)


if __name__ == "__main__":
    unittest.main()
