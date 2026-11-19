"""CPU export and held-out panel checks for the non-reportable BF16 rehearsal."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3VLForConditionalGeneration

from fine_tuning.error_note import v18b_rehearsal_eval as rehearsal
from fine_tuning.error_note.prep_v16b import read_sealed
from fine_tuning.notebook_rl.checkpoint import capture_rng, make_optimizer, save
from fine_tuning.notebook_rl.export import FORMAT_LORA, validate_export_identity
from fine_tuning.tests.test_notebook_rl_checkpoint import fixture


class RehearsalEvaluationTests(unittest.TestCase):
    def test_scoped_fixture_memory_override_keeps_the_normal_default(self):
        from mmdl.evaluation.backends import vllm_backend
        from fine_tuning.error_note.v17_eval_engine import engine_execution

        calls = []
        def construct(**kwargs):
            calls.append(kwargs)
        with patch.object(vllm_backend, "_vllm", return_value=(construct, None, "0.11.0")):
            with engine_execution("eager", 2, gpu_memory_utilization=.6):
                vllm_backend._vllm()[0](gpu_memory_utilization=.9)
            with engine_execution("cuda_graph", 9):
                vllm_backend._vllm()[0](gpu_memory_utilization=.9)
        self.assertEqual([call["gpu_memory_utilization"] for call in calls], [.6, .9])
        for fraction in (0., -1., 1.1, float("nan"), True):
            with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, "memory fraction"):
                with engine_execution("eager", 2, gpu_memory_utilization=fraction):
                    self.fail("invalid GPU fraction entered the context")

    def checker_fixture(self, *, eligible):
        from mmdl.evaluation.parsers import score_with_parser
        from mmdl.runtime.contracts import sample_seed

        items, records = [], []
        for index in range(9):
            pid = f"external-{index}"
            is_open = eligible and index == 0
            spec = {"answer_type": "open" if is_open else "mcq", "gold": "1/4" if is_open else "A",
                    "question": "Select the requested value.", "choices": [] if is_open else ["right", "wrong"]}
            raw = "The value is 0.26." if is_open else "Answer: A"
            score = score_with_parser(raw, "open" if is_open else "multiple-choice", spec["choices"],
                                      spec["gold"], "team-final-answer-v8", "stop", question=spec["question"])
            items.append({"problem_id": pid, "source": "authored-external", "evaluation": spec})
            records.append({"sample_id": pid, "seed": sample_seed(3407, pid), "raw_response": raw,
                            "finish_reason": "stop", "generated_tokens": 8, "v8": score})
        return {"records": records, "reportable": False}, items

    def test_real_eligible_saved_response_is_selected_and_v8_is_preserved(self):
        from mmdl.evaluation.CompassVerifier_checker import backend, selection

        external, items = self.checker_fixture(eligible=True)
        before = json.dumps(external, sort_keys=True)
        rows = rehearsal._checker_rows(external, items)
        calls = []
        def decision(prompt):
            calls.append(prompt)
            return {"p_a": .95, "p_b": .02, "p_c": .01, "top_token_id": 0, "top_token": "A"}
        result = rehearsal._supplementary_check(rows, backend.FakeJudge(decision), selection.load_config())
        self.assertEqual(result["real_eligible_ids"], ["external-0"])
        self.assertEqual(result["real_eligible_count"], 1)
        self.assertEqual(result["judge_call_count"], 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("0.26", calls[0])
        self.assertIsNone(result["synthetic_control"])
        self.assertIs(result["reportable"], False)
        self.assertEqual(before, json.dumps(external, sort_keys=True))
        self.assertFalse(result["merged"][0]["v8_correct"])
        self.assertTrue(result["merged"][0]["final_correct"])
        external["records"][1]["v8"]["correct"] = False
        with self.assertRaisesRegex(ValueError, "V8 replay"):
            rehearsal._checker_rows(external, items)

    def test_empty_eligible_set_calls_only_the_synthetic_control(self):
        from mmdl.evaluation.CompassVerifier_checker import backend, selection

        external, items = self.checker_fixture(eligible=False)
        judge = backend.FakeJudge(lambda prompt: {
            "p_a": .95, "p_b": .02, "p_c": .01, "top_token_id": 0, "top_token": "A"})
        result = rehearsal._supplementary_check(rehearsal._checker_rows(external, items), judge, selection.load_config())
        self.assertEqual(result["real_eligible_count"], 0)
        self.assertEqual(result["real_eligible_ids"], [])
        self.assertEqual(result["judge_call_count"], 1)
        self.assertEqual(result["synthetic_control"]["case_id"], "fraction-decimal")
        self.assertEqual(result["synthetic_control"]["verdict"], "ACCEPT")
        self.assertTrue(all(row["v8_correct"] and row["final_correct"] for row in result["merged"]))

    def test_panel_requires_nine_distinct_external_problems(self):
        items = [{"problem_id": str(i), "split": "performance", "training_allowed": False,
                  "benchmark_derived": False} for i in range(9)]
        with patch.object(rehearsal, "load_items", return_value=items):
            self.assertEqual(len(rehearsal._external_panel(Path("/unused"))), 9)
        for bad in (items[:8], items[:-1] + [items[0]],
                    items[:-1] + [dict(items[-1], training_allowed=True)],
                    items[:-1] + [dict(items[-1], benchmark_derived=True)]):
            with self.subTest(last=bad[-1]), patch.object(rehearsal, "load_items", return_value=bad):
                with self.assertRaisesRegex(ValueError, "held-out external"):
                    rehearsal._external_panel(Path("/unused"))

    def test_actual_tiny_checkpoint_exports_unmerged_on_cpu(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory(prefix="v18b-rehearsal-export-") as temporary:
            root = Path(temporary)
            model_path, _, protocol, binding = fixture(root)
            config = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, dtype=torch.bfloat16, local_files_only=True).config
            config.text_config.num_hidden_layers = 2
            Qwen3VLForConditionalGeneration(config).bfloat16().save_pretrained(model_path, safe_serialization=True)
            bundle = rehearsal._tiny_factory(model_path, "MOD-RV", protocol)
            self.assertEqual(next(bundle.model.parameters()).device.type, "cpu")
            optimizer, scheduler = make_optimizer(bundle, protocol)
            checkpoint = root / "checkpoint"
            save(bundle, optimizer, scheduler, 0,
                {"epoch": 0, "position": 0, "discarded_problem_ids": [], "order_sha256": "d" * 64},
                capture_rng(), checkpoint, protocol=protocol, binding=binding)
            output = root / "evaluation"
            output.mkdir()
            args = SimpleNamespace(mode="tiny-bf16", model_path=model_path, output=output)
            with patch.object(rehearsal, "_boundary", return_value=(checkpoint, binding)):
                rehearsal._export(args)
            receipt = read_sealed(output / "rehearsal-export.json")
            self.assertIs(receipt["reportable"], False)
            identity = validate_export_identity(output / "model", "MOD-RV")
            self.assertEqual(identity, receipt["identity"])
            manifest = json.loads((output / "model/export_manifest.json").read_text())
            self.assertEqual(manifest["format"], FORMAT_LORA)
            self.assertEqual(manifest["precision_policy"]["fusion"], "none: no adapter is merged into any weight")


if __name__ == "__main__":
    unittest.main()
