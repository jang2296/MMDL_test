"""Authored controlled engine + actual verifier worker; no GPU/model execution."""
import hashlib
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl.engines import Completion, RolloutResult, SamplingSettings
from fine_tuning.notebook_rl.phase_a import _cap, _measure_candidates, measure_candidates
from fine_tuning.notebook_rl.schema import load_protocol
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier_service import VerifierService
from fine_tuning.tests.test_notebook_rl_eval_external import Characters
from fine_tuning.tests.test_notebook_rl_schema import records


class ControlledEngine:
    policy_version = 0
    supported_sampling = {"temperature_1"}
    tokenizer = Characters()

    def __init__(self, behavior="normal", after_sample=lambda: None):
        self.behavior, self.after_sample, self.calls = behavior, after_sample, []

    def prepare(self, request):
        ids = request.prompt_token_ids
        if ids is None:
            ids = self.tokenizer.encode(json.dumps(request.messages, ensure_ascii=False))
        return SimpleNamespace(token_ids=list(ids))

    def sample_with_settings(self, requests, sampling):
        assert sampling == SamplingSettings()
        request = requests[0]
        assert request.max_new_tokens == 8192
        self.calls.append({"request": request, "gray": all(im.getextrema() == ((128, 128),) * 3 for im in request.images)})
        if self.behavior == "error":
            raise RuntimeError("authored engine failure")
        if self.behavior == "interrupt":
            raise KeyboardInterrupt()
        is_retry = request.prompt_token_ids is not None
        third = ":T3:" in request.request_id
        second = is_retry and not third
        if self.behavior == "t3_interrupt" and third:
            raise KeyboardInterrupt()
        blind = request.request_id.endswith(":blind")
        completions = []
        for index in range(request.n):
            answer = "7" if not blind and (is_retry or index == 0) else "8"
            if self.behavior in ("three_turn", "t3_context", "t3_interrupt") and is_retry:
                answer = "7" if index == 0 and (third or self.behavior == "three_turn") else "8"
            text = "Observed facts:\ncell[a][b] = " + answer + "\nSolution:\nFinal answer: " + answer
            if self.behavior == "no_final":
                text = "unfinished"
            if self.behavior == "t2_no_final" and second:
                text = "Error check: an answer is still missing"
            if self.behavior == "three_turn" and is_retry:
                length = 3300 if third else 2200
                text = "x" * (length - len(self.tokenizer.encode(text)) - 2) + "\n" + text
            if self.behavior == "t3_context" and second:
                text = "x" * (7600 - len(self.tokenizer.encode(text)) - 2) + "\n" + text
            long = self.behavior == "long_parent" and not is_retry and not blind and index > 0
            if long:
                text = "x" * (8192 - len(self.tokenizer.encode(text)) - 1) + text
            ids = self.tokenizer.encode(text) + [151645]
            lps = [-.1] * len(ids)
            if self.behavior == "nan":
                lps[0] = float("nan")
            if self.behavior == "text_mismatch":
                text += " changed"
            completions.append(Completion(ids, lps, "length" if long else "stop", text))
        if self.behavior == "extra":
            completions.append(completions[0])
        if self.behavior == "missing":
            completions.pop()
        actual = self.prepare(request).token_ids
        if self.behavior == "prompt":
            actual = actual + [123]
        if self.behavior == "version":
            self.policy_version = 1
        if self.behavior == "seed_mutation":
            request.seeds[0] += 1
        self.after_sample()
        return [RolloutResult(request.request_id, actual, completions)]


class PhaseAMeasurementTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="phase-a-measurements-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.build = self.root / "build"
        self.build.mkdir()
        self.images = self.build / "images"
        self.images.mkdir()
        Image.new("RGB", (9, 7), (10, 20, 30)).save(self.images / "a.png")
        digest = hashlib.sha256((self.images / "a.png").read_bytes()).hexdigest()
        actor, verifier, lineage, *_ = records()
        self.actors = [replace(actor, problem_id=f"fixture-{i}", image_paths=("a.png",), image_sha256=(digest,)) for i in range(2)]
        verifiers = [replace(verifier, problem_id=a.problem_id,
            oracle_fact_lines=("PRIVATE-ORACLE-TEXT-83145",)) for a in self.actors]
        write_split(self.build / "train", self.actors, verifiers,
            [replace(lineage, problem_id=a.problem_id) for a in self.actors])
        self.build_manifest = {"format": "notebook-rl-candidate-build-v1", "training_allowed": False,
            "counts": {"train": 2}, "split_manifests": {"train": hashlib.sha256(
                (self.build / "train/manifest.yaml").read_bytes()).hexdigest()}, "image_sha256": {"a.png": digest}}
        (self.build / "manifest.yaml").write_text(json.dumps(self.build_manifest))
        self.protocol = load_protocol(Path(__file__).resolve().parents[1] / "notebook_rl/protocol_v1.yaml")
        self.binding = {"protocol_sha256": self.protocol["sha256"], "prompts_sha256": self.protocol["prompts_sha256"],
            "data_sha256": "d" * 64, "code_sha256": "c" * 64}

    def measure(self, engine=None, name="output", **kwargs):
        engine = engine or ControlledEngine()
        summary = _measure_candidates(self.build, self.root / name, engine=engine, protocol=self.protocol,
            input_binding=self.binding, deadline_timestamp=time.time() + 3600,
            expected_candidates=2, retry_subset_size=1, **kwargs)
        return summary, engine

    def read(self, name="records.jsonl", output="output"):
        return [json.loads(line) for line in (self.root / output / name).read_text().splitlines()]

    def test_actual_worker_sampling_retry_and_cap_counts(self):
        summary, engine = self.measure()
        self.assertEqual(summary["status"], "MEASUREMENTS_COMPLETED_NOT_PHASE_A_APPROVAL")
        self.assertFalse(summary["training_allowed"])
        self.assertFalse(summary["gpu_execution_certified"])
        self.assertTrue(summary["controlled_fixture_scope"])
        self.assertEqual(summary["max_new_tokens"], {"1": 2048, "2": 2048})
        self.assertEqual(summary["t3_cap_recommendation"]["status"], "NOT_APPLICABLE_NO_ELIGIBLE_PARENT")
        self.assertIsNone(summary["t3_cap_recommendation"]["cap"])
        self.assertEqual(len(engine.calls), 6)
        self.assertEqual(summary["sampling_cost_counts"], {"attempted_calls": 6, "returned_calls": 6,
            "returned_completions": 32, "returned_tokens": sum(len(r["sample"]["completion_token_ids"]) for r in self.read() if "sample" in r),
            "verifier_calls": 32})
        self.assertEqual(summary["retry"]["n"], 8)
        self.assertEqual(summary["retry"]["success_rate"], 1.)
        self.assertEqual(summary["retry"]["parent_coverage"][0]["failed_parents"], 2)
        self.assertTrue(all(row["correct"] == 1 and row["blind_correct"] == 0 for row in summary["rows"]))
        verifier = json.loads((self.root / "output/verifier_manifest.json").read_text())
        self.assertTrue(verifier["python_guards_installed"])
        self.assertEqual(verifier["item_count"], 2)
        for name, digest in summary["files"].items():
            self.assertEqual(hashlib.sha256((self.root / "output" / name).read_bytes()).hexdigest(), digest)

    def test_literal_history_distinct_parents_gray_images_and_gold_independence(self):
        summary, engine = self.measure()
        rows = [row for row in self.read() if "sample" in row]
        by_id = {r["sample"]["sample_id"]: r["sample"] for r in rows}
        parents = set()
        for row in rows:
            sample = row["sample"]
            if row["kind"] == "T2":
                parent = by_id[sample["parent_sample_id"]]
                parents.add(parent["sample_id"])
                actual_prefix = parent["prompt_token_ids"] + parent["completion_token_ids"]
                self.assertEqual(sample["prompt_token_ids"][:len(actual_prefix)], actual_prefix)
                suffix = engine.tokenizer.decode(sample["prompt_token_ids"][len(actual_prefix):])
                self.assertIn("INCORRECT", suffix)
                self.assertNotIn("Final answer: 7", suffix)
        self.assertEqual(len(parents), 2)
        self.assertEqual(sum(call["gray"] for call in engine.calls), 2)
        self.assertNotIn("PRIVATE-ORACLE-TEXT-83145", (self.root / "output/sampler_calls.jsonl").read_text())
        self.assertNotIn('"gold"', (self.root / "output/sampler_calls.jsonl").read_text())
        original = VerifierService.grade
        def change_grade(service, pid, text, finish, mode):
            result = original(service, pid, text, finish, mode)
            return replace(result, status="INCORRECT", reward_answer=0.)
        # Only first-turn requests depend on actors, never on answer correctness.
        with patch.object(VerifierService, "grade", change_grade), patch.object(VerifierService, "cause", return_value=None):
            _, changed = self.measure(name="other")
        for first, second in zip(engine.calls[:4], changed.calls[:4], strict=True):
            self.assertEqual(first["request"].messages, second["request"].messages)
            self.assertEqual(first["request"].seeds, second["request"].seeds)
        self.assertEqual(summary["retry_subset_ids"], json.loads((self.root / "other/manifest.json").read_text())["retry_subset_ids"])

    def test_raw_group_is_durable_before_any_interpretation(self):
        for behavior in ("extra", "missing", "prompt", "nan", "text_mismatch", "version", "seed_mutation", "error"):
            with self.subTest(behavior=behavior):
                summary, engine = self.measure(ControlledEngine(behavior), name=behavior)
                self.assertEqual(summary["status"], "INCOMPLETE")
                self.assertEqual(summary["max_new_tokens"], {})
                self.assertIsNone(summary["retry"])
                raw = self.read("sampler_calls.jsonl", behavior)
                self.assertEqual(raw[0]["event"], "request")
                self.assertEqual(raw[-1]["event"], "error")
                if behavior != "error":
                    self.assertEqual(raw[1]["event"], "return")
                    self.assertEqual(len(raw[1]["results"][0]["completions"]), 9 if behavior == "extra" else 7 if behavior == "missing" else 8)
                self.assertEqual(len(engine.calls), 1)
                self.assertEqual(summary["sampling_cost_counts"]["verifier_calls"], 0)

    def test_budget_before_start_and_after_actual_return_is_not_completion(self):
        summary, engine = self.measure(clock=lambda: time.time() + 7200)
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(summary["errors"], ["BUDGET_STOP"])
        self.assertFalse(engine.calls)
        state = [0.]
        engine = ControlledEngine(after_sample=lambda: state.__setitem__(0, time.time() + 7200))
        summary, _ = self.measure(engine, name="partial", clock=lambda: state[0])
        self.assertEqual(summary["errors"], ["BUDGET_STOP"])
        self.assertEqual(self.read("sampler_calls.jsonl", "partial")[1]["event"], "return")
        self.assertEqual(summary["sampling_cost_counts"]["verifier_calls"], 0)

    def test_output_and_engine_are_single_use_and_public_api_requires_full_pool(self):
        _, engine = self.measure()
        before = (self.root / "output/sampler_calls.jsonl").read_bytes()
        with self.assertRaises(ValueError):
            self.measure(engine, name="new-output")
        with self.assertRaises(FileExistsError):
            self.measure()
        self.assertEqual(before, (self.root / "output/sampler_calls.jsonl").read_bytes())
        with self.assertRaisesRegex(ValueError, "full train"):
            measure_candidates(self.build, self.root / "public", engine=ControlledEngine(), protocol=self.protocol,
                input_binding=self.binding, deadline_timestamp=time.time() + 3600)

    def test_no_final_answers_and_mutated_images_never_yield_manual_metrics(self):
        summary, _ = self.measure(ControlledEngine("no_final"))
        self.assertIn("T1_NO_COMPLETED_FINAL_ANSWERS_FOR_CAP", summary["errors"])
        self.assertEqual(summary["t1_cap_recommendation"]["status"], "NO_COMPLETED_FINAL_ANSWERS")
        self.assertEqual(len(summary["rows"]), 2)
        self.assertTrue(all(row["truncated"] is None for row in summary["rows"]))
        self.assertEqual(summary["max_new_tokens"], {})
        altered = [False]
        def mutate():
            if not altered[0]:
                Image.new("RGB", (9, 7), (30, 40, 50)).save(self.images / "a.png")
                altered[0] = True
        summary, _ = self.measure(ControlledEngine(after_sample=mutate), name="changed-image")
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(summary["rows"], [])

    def test_cap_uses_completed_response_nearest_rank_and_bounded_rounding(self):
        self.assertEqual(_cap([]), (None, None))
        self.assertEqual(_cap([1] * 19 + [8192]), (2048, 1))
        self.assertEqual(_cap([2049]), (2304, 2049))
        self.assertEqual(_cap([8000]), (4096, 8000))

    def test_t2_context_blocks_preserve_t1_metrics_and_full_literal_history(self):
        summary, engine = self.measure(ControlledEngine("long_parent"))
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(summary["errors"], ["T2_CONTEXT_BLOCKED"])
        self.assertEqual(summary["max_new_tokens"], {"1": 2048})
        self.assertEqual(len(summary["rows"]), 2)
        self.assertEqual(len(engine.calls), 4)
        retry = summary["retry"]
        self.assertEqual(retry["eligible_samples"], 8)
        self.assertEqual(retry["context_blocked_samples"], 8)
        self.assertEqual(retry["n"], 0)
        self.assertIsNone(retry["success_rate"])
        self.assertIsNone(retry["success_rate_conditional_on_executed"])
        rows = self.read()
        parents = {r["sample"]["sample_id"]: r["sample"] for r in rows if "sample" in r}
        blocked = [r for r in rows if r.get("status") == "CONTEXT_BLOCKED"]
        self.assertEqual(len(blocked), 2)
        for row in blocked:
            parent = parents[row["parent_sample_id"]]
            expected = parent["prompt_token_ids"] + parent["completion_token_ids"]
            self.assertEqual(row["prompt_token_ids"][:len(expected)], expected)
            self.assertGreater(row["prompt_tokens"] + row["reserved_output"], self.protocol["context_limit"])
        self.assertTrue(all(r["truncated_at_8192"] == 7 for r in summary["rows"]))

    def test_interrupt_closes_incomplete_summary_then_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            self.measure(ControlledEngine("interrupt"))
        summary = json.loads((self.root / "output/summary.json").read_text())
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(summary["errors"], ["KeyboardInterrupt"])
        self.assertEqual(summary["sampling_cost_counts"]["attempted_calls"], 1)
        self.assertEqual(self.read("sampler_calls.jsonl")[-1]["error"], "KeyboardInterrupt")

    def test_t3_literal_history_separate_caps_and_observed_attribution(self):
        summary, engine = self.measure(ControlledEngine("three_turn"))
        self.assertEqual(summary["status"], "MEASUREMENTS_COMPLETED_NOT_PHASE_A_APPROVAL")
        self.assertEqual(summary["max_new_tokens"], {"1": 2048, "2": 2304, "3": 3328})
        self.assertEqual(summary["t2_cap_recommendation"]["complete_final_p95"], 2200)
        self.assertEqual(summary["t3_cap_recommendation"]["complete_final_p95"], 3300)
        self.assertEqual(summary["t3_cap_recommendation"]["families"]["table"]["truncation_rate_at_recommended_cap"], 0.)
        self.assertEqual(len(engine.calls), 8)
        self.assertEqual(summary["sampling_cost_counts"]["returned_completions"], 40)
        self.assertEqual(summary["third_retry"]["requested_samples"], 8)
        self.assertEqual(summary["third_retry"]["n"], 8)
        self.assertEqual(summary["third_retry"]["success_rate"], .25)
        rows = [row for row in self.read() if "sample" in row]
        by_id = {row["sample"]["sample_id"]: row["sample"] for row in rows}
        third = [row["sample"] for row in rows if row["kind"] == "T3"]
        self.assertEqual(len({sample["group_id"] for sample in third}), 2)
        for sample in third:
            self.assertEqual(sample["turn"], 3)
            self.assertEqual(sample["kind"], "T3")
            parent = by_id[sample["parent_sample_id"]]
            first = by_id[parent["parent_sample_id"]]
            expected = parent["prompt_token_ids"] + parent["completion_token_ids"]
            self.assertEqual(sample["prompt_token_ids"][:len(expected)], expected)
            first_history = first["prompt_token_ids"] + first["completion_token_ids"]
            self.assertEqual(sample["prompt_token_ids"][:len(first_history)], first_history)
        selections = [row for row in self.read() if row["kind"] == "T3_parent_selection"]
        self.assertEqual(len(selections), 2)
        self.assertTrue(all(len(row["selected_sample_ids"]) == 1 and row["unselected_failures"] == 2 for row in selections))
        for turn, failures in (("T1", 14), ("T2", 6), ("T3", 6)):
            attribution = summary["attribution"][turn]
            self.assertEqual(attribution["cause_counts"]["VISUAL_MISREAD"], failures)
            self.assertEqual(attribution["attributable_failed_observations"], failures)
            self.assertEqual(attribution["required_slot_parse_rate"], 1.)
            self.assertFalse(attribution["module_causality_certified"])
        self.assertFalse(summary["gpu_execution_certified"])
        self.assertFalse(summary["training_allowed"])
        self.assertNotIn("T3", summary["unmeasured"])
        self.assertIn("oracle_counterfactual", summary["unmeasured"])

    def test_t2_no_final_does_not_borrow_t1_or_t3_cap(self):
        summary, _ = self.measure(ControlledEngine("t2_no_final"))
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(summary["max_new_tokens"], {"1": 2048, "3": 2048})
        self.assertEqual(summary["t2_cap_recommendation"]["status"], "NO_COMPLETED_FINAL_ANSWERS")
        self.assertEqual(summary["t2_cap_recommendation"]["n"], 8)
        self.assertIsNone(summary["t2_cap_recommendation"]["cap"])
        self.assertIsNone(summary["t2_cap_recommendation"]["complete_final_p95"])
        self.assertEqual(summary["t3_cap_recommendation"]["n"], 8)
        self.assertEqual(summary["errors"], ["T2_NO_COMPLETED_FINAL_ANSWERS_FOR_CAP"])

    def test_t3_context_block_keeps_t1_t2_measurements_and_denominators(self):
        summary, engine = self.measure(ControlledEngine("t3_context"))
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(summary["errors"], ["T3_CONTEXT_BLOCKED"])
        self.assertEqual(summary["max_new_tokens"], {"1": 2048, "2": 4096})
        self.assertEqual(len(engine.calls), 6)
        self.assertEqual(summary["retry"]["n"], 8)
        self.assertEqual(summary["third_retry"]["context_blocked_samples"], 8)
        self.assertEqual(summary["third_retry"]["n"], 0)
        self.assertIsNone(summary["third_retry"]["success_rate"])
        self.assertEqual(summary["t2_cap_recommendation"]["families"]["table"]["truncation_rate_at_recommended_cap"], 1.)
        self.assertEqual(summary["t3_cap_recommendation"]["status"], "CONTEXT_BLOCKED")
        self.assertIsNone(summary["t3_cap_recommendation"]["cap"])
        self.assertEqual(summary["measurement_status"]["T2"], "COMPLETED")
        self.assertEqual(summary["measurement_status"]["T3"], "CONTEXT_BLOCKED")
        rows = self.read()
        parents = {row["sample"]["sample_id"]: row["sample"] for row in rows if "sample" in row}
        for row in (row for row in rows if row.get("status") == "CONTEXT_BLOCKED"):
            self.assertEqual(row["kind"], "T3")
            parent = parents[row["parent_sample_id"]]
            expected = parent["prompt_token_ids"] + parent["completion_token_ids"]
            self.assertEqual(row["prompt_token_ids"][:len(expected)], expected)
            self.assertGreater(row["prompt_tokens"] + row["reserved_output"], 16384)
            self.assertEqual(row["reserved_output"], 8192)
        self.assertEqual(self.protocol["rollout"]["max_new_tokens"], {1: 3072, 2: 3072, 3: 3072})

    def test_t3_interrupt_records_partial_costs_and_raw_history(self):
        with self.assertRaises(KeyboardInterrupt):
            self.measure(ControlledEngine("t3_interrupt"))
        summary = json.loads((self.root / "output/summary.json").read_text())
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(summary["errors"], ["KeyboardInterrupt"])
        self.assertEqual(summary["sampling_cost_counts"]["attempted_calls"], 6)
        self.assertEqual(summary["sampling_cost_counts"]["returned_calls"], 5)
        raw = self.read("sampler_calls.jsonl")
        self.assertEqual(raw[-2]["event"], "request")
        self.assertIn(":T3:", raw[-2]["request"]["request_id"])
        self.assertTrue(raw[-2]["request"]["prompt_token_ids"])
        self.assertEqual(raw[-1]["error"], "KeyboardInterrupt")


if __name__ == "__main__":
    unittest.main()
