"""CPU checks; injected completions never count as actual model inference."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from experiments.bottleneck_diagnostic.input_builder import (
    build_request,
    canonical_hash,
    perception_schema,
    stage_seed,
)
from experiments.bottleneck_diagnostic.oracle_validation import oracle_hash, validate_oracle
from experiments.bottleneck_diagnostic.runner import BudgetStop, remaining_window, run_diagnostics
from experiments.bottleneck_diagnostic.scoring import extract_diagnostic, score_record
from experiments.bottleneck_diagnostic.cli import selected, validated_run


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        Image.new("RGB", (4, 3), "white").save(self.root / "image.png")
        self.sha = hashlib.sha256((self.root / "image.png").read_bytes()).hexdigest()
        self.sample = {
            "sample_id": "synthetic_Music_1", "subject": "Music", "question_type": "multiple-choice",
            "question": "What relation is shown in <image 1>?", "options": ["Alpha", "Beta"],
            "hint": None, "images": [{"field": "image_1", "marker": "<image 1>",
                "role": "question", "order": 0, "path": "image.png", "sha256": self.sha,
                "original_size": [4, 3]}],
            "family_id": "family", "skill_tags": ["structure"],
            "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
        }
        self.oracle = {
            "sample_id": self.sample["sample_id"], "level": "O0", "annotation_version": "test-v1",
            "facts": {"symbols": ["two black marks"]}, "source_image_hashes": [self.sha],
            "evidence_locations": [{"image": "<image 1>", "region": [0, 0, 1, 1]}],
            "completeness_status": "COMPLETE", "target_leakage_status": "VALID",
            "annotator_exposure_status": "GOLD_BLIND", "reviewer_status": "INDEPENDENT_REVIEWED",
            "annotator_id": "synthetic-annotator", "reviewer_id": "synthetic-reviewer",
            "approved": True, "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
        }
        self.oracle["oracle_hash"] = oracle_hash(self.oracle)
        self.calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def backend(self, request):
        self.calls.append(request)
        return {"raw_response": '{"unverified": "unchanged P"}' if request["mode"] == "P"
                else "A might be wrong.\nFINAL_ANSWER: B", "generated_tokens": 12,
                "input_tokens": 35, "finish_reason": "stop", "image_grid_thw": [[1, 2, 2]]}

    def run_cells(self, modes, oracles=(), backend=None, **kwargs):
        return run_diagnostics(self.root, self.root / "output", {"modes": modes},
                               [self.sample], oracles, backend or self.backend, **kwargs)

    def test_gold_fields_are_rejected_not_silently_scrubbed(self):
        for forbidden in ("gold", "answer", "gold_explanation", "correct", "raw_response"):
            sample = self.sample | {forbidden: "secret"}
            with self.assertRaisesRegex(ValueError, "forbidden"):
                build_request(sample, "I", 3407)
        text = json.dumps(build_request(self.sample, "I", 3407))
        self.assertNotIn("synthetic-annotator", text)

    def test_expansion_remainder_preserves_order_without_repeating_prior_suites(self):
        selection = {"pilot_ids": ["pilot"], "expansion_ids": ["extra2", "pilot", "covered", "extra1"]}
        (self.root / "sample_selection.json").write_text(json.dumps(selection))
        rows = [self.sample | {"sample_id": sid} for sid in selection["expansion_ids"]]
        (self.root / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with mock.patch("experiments.bottleneck_diagnostic.cli.all_subject_first_pass", return_value=["covered"]):
            result = selected(self.root, "expansion-remainder")
        self.assertEqual([row["sample_id"] for row in result], ["extra2", "extra1"])

    def test_option_image_order_and_role_guard(self):
        sample = copy.deepcopy(self.sample)
        sample["options"][1] += " <image 2>"
        second = sample["images"][0] | {"field": "image_2", "marker": "<image 2>", "order": 1, "role": "option"}
        sample["images"].append(second)
        request = build_request(sample, "I", 3407)
        self.assertEqual([item["marker"] for item in request["images"]], ["<image 1>", "<image 2>"])
        sample["images"][1]["role"] = "explanation"
        with self.assertRaisesRegex(ValueError, "Explanation-only"):
            build_request(sample, "I", 3407)

    def test_hint_references_and_source_bytes_metadata(self):
        sample = copy.deepcopy(self.sample)
        sample["question"] = "What relation?"
        sample["hint"] = "Observe <image 1>."
        sample["images"][0]["source_bytes_sha256"] = "d" * 64
        request = build_request(sample, "I", 3407)
        self.assertIn("Hint: Observe", request["messages"][0]["content"][-1]["text"])

    def test_no_question_options_or_hint_in_perception(self):
        sample = self.sample | {"hint": "UNIQUEHINT", "question": "UNIQUEQUESTION <image 1>",
                               "options": ["UNIQUEOPTION", "OTHEROPTION"]}
        text = json.dumps(build_request(sample, "P", 3407)["messages"])
        for forbidden in ("UNIQUEHINT", "UNIQUEQUESTION", "UNIQUEOPTION", "OTHEROPTION"):
            self.assertNotIn(forbidden, text)
        self.assertIn("Do not solve", text)

    def test_oracle_review_target_and_image_gates(self):
        self.assertEqual(validate_oracle(self.oracle, self.sample)["status"], "VALID")
        self.assertEqual(validate_oracle(None, self.sample)["status"], "BLOCKED")
        for patch, expected in (({"approved": False}, "BLOCKED"),
                                ({"target_leakage_status": "TARGET_REVEALED"}, "EXCLUDED"),
                                ({"target_leakage_status": "NOT_APPLICABLE"}, "EXCLUDED"),
                                ({"source_image_hashes": []}, "BLOCKED"),
                                ({"reviewer_id": "synthetic-annotator"}, "BLOCKED"),
                                ({"facts": "FINAL_ANSWER: A"}, "EXCLUDED")):
            changed = self.oracle | patch
            changed["oracle_hash"] = oracle_hash(changed)
            self.assertEqual(validate_oracle(changed, self.sample)["status"], expected)
            with self.assertRaises(ValueError):
                build_request(self.sample, "O0", 3407, changed)

    def test_ai_review_requires_explicit_policy_and_label(self):
        oracle = self.oracle | {"reviewer_status": "AI_INDEPENDENT_REVIEWED"}
        oracle["oracle_hash"] = oracle_hash(oracle)
        self.assertEqual(validate_oracle(oracle, self.sample)["status"], "BLOCKED")
        request = build_request(self.sample, "O0", 3407, oracle,
                                oracle_review_policy="ALLOW_AI_REVIEWED_DIAGNOSTIC")
        self.assertEqual(request["verification_kind"], "AI_REVIEWED_NOT_HUMAN_VERIFIED")

    def test_seed_and_stateless_cache_identity(self):
        expected = int.from_bytes(hashlib.sha256(b"3407:synthetic_Music_1").digest()[:8], "big") % 2**31
        self.assertEqual(stage_seed(3407, self.sample["sample_id"], "I"), expected)
        self.assertEqual(stage_seed(3407, self.sample["sample_id"], "I"), stage_seed(3407, self.sample["sample_id"], "O0"))
        self.assertNotEqual(expected, stage_seed(3407, self.sample["sample_id"], "P"))
        first = build_request(self.sample, "I", 3407)
        second = build_request(self.sample, "T0", 3407)
        self.assertNotEqual(first["cache_key"], second["cache_key"])
        self.assertNotEqual(canonical_hash(["p1", first["cache_key"]]), canonical_hash(["p2", first["cache_key"]]))
        self.assertEqual(len(first["messages"]), 1)
        self.assertEqual(second["images"], [])
        self.assertEqual(first["max_new_tokens"], 4096)
        self.assertEqual(build_request(self.sample, "P", 3407)["max_new_tokens"], 1536)
        self.assertEqual(build_request(self.sample, "LEGACY", 3407)["max_new_tokens"], 32768)

    def test_p_output_is_shared_exactly_and_resume_is_append_only(self):
        records = self.run_cells(["P", "M", "MI"])
        self.assertEqual([record["runtime_status"] for record in records], ["COMPLETED"] * 3)
        self.assertTrue(all(record["run_kind"] == "TEST_ONLY" for record in records))
        m, mi = self.calls[1:]
        self.assertEqual(m["perception_response_hash"], mi["perception_response_hash"])
        self.assertEqual(m["messages"][0]["content"][-1]["text"], mi["messages"][0]["content"][-1]["text"])
        self.assertIn('{"unverified": "unchanged P"}', m["messages"][0]["content"][-1]["text"])
        before = (self.root / "output/responses.jsonl").read_bytes()
        self.run_cells(["P", "M", "MI"])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual((self.root / "output/responses.jsonl").read_bytes(), before)

    def test_changed_data_protocol_oracle_resume_refused(self):
        self.run_cells(["I"])
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_cells(["T0"])
        self.sample["options"] = ["changed", "Beta"]
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_cells(["I"])

    def test_test_only_results_cannot_resume_as_actual_inference(self):
        self.run_cells(["I"])
        class ClaimedActual:
            actual_inference = True

            def __call__(self, request):
                raise AssertionError("Must refuse before inference")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_cells(["I"], backend=ClaimedActual())

    def test_response_tampering_is_rejected(self):
        self.run_cells(["I"])
        path = self.root / "output/responses.jsonl"
        row = json.loads(path.read_text())
        row["raw_response"] = "FINAL_ANSWER: A"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "record hash mismatch"):
            self.run_cells(["I"])

    def test_scoring_validates_ledger_and_keeps_missing_cells_pending(self):
        self.run_cells(["I", "T0"])
        output = self.root / "output"
        path = output / "responses.jsonl"
        first = path.read_text().splitlines()[0]
        path.write_text(first + "\n")
        _, observed, matrix, missing = validated_run(output, [self.sample])
        self.assertEqual(len(observed), 1)
        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(missing), 1)
        self.assertEqual(matrix[-1]["runtime_status"], "PENDING")
        row = json.loads(first)
        row["raw_response"] = "forged"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "integrity failed"):
            validated_run(output, [self.sample])

    def test_crashed_uncompleted_request_reserves_budget_on_resume(self):
        self.run_cells(["I"])
        output = self.root / "output"
        # Simulate a process killed between the durable input audit and completion append.
        (output / "responses.jsonl").write_text("")
        audits = json.loads((output / "input_audit.jsonl").read_text())
        self.assertEqual(audits["max_new_tokens"], 4096)
        self.run_cells(["I"])
        self.assertEqual(len(self.calls), 2)
        starts = [json.loads(line) for line in (output / "budget_events.jsonl").read_text().splitlines()
                  if json.loads(line)["event"] == "SESSION_START"]
        self.assertEqual(starts[-1]["previous_output_tokens"], 4096)

    def test_image_mutation_and_path_traversal_rejected(self):
        (self.root / "image.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_cells(["I"])
        sample = copy.deepcopy(self.sample)
        sample["images"][0]["path"] = "../image.png"
        with self.assertRaisesRegex(ValueError, "unsafe"):
            build_request(sample, "I", 3407)

    def test_missing_or_excluded_oracles_never_call_backend(self):
        records = self.run_cells(["O0", "O1", "IO0"])
        self.assertTrue(all(record["runtime_status"] == "BLOCKED" for record in records))
        self.assertEqual(self.calls, [])

    def test_legacy_without_original_image_role_equivalence_is_blocked(self):
        records = self.run_cells(["LEGACY"])
        self.assertEqual(self.calls, [])
        self.assertEqual(records[0]["runtime_status"], "BLOCKED")

    def test_token_budget_stops_before_call(self):
        records = run_diagnostics(self.root, self.root / "output", {
            "modes": ["I", "T0"], "total_output_token_budget": 1,
        }, [self.sample], [], self.backend)
        self.assertEqual(self.calls, [])
        self.assertEqual([record["runtime_status"] for record in records], ["BUDGET_STOP"] * 2)

    def test_time_budget_aborts_active_and_keeps_remaining_pending(self):
        def limited(request):
            raise BudgetStop("CPU_TEST_DEADLINE")
        records = self.run_cells(["I", "T0"], backend=limited)
        self.assertEqual([record["runtime_status"] for record in records], ["BUDGET_STOP"] * 2)
        self.assertEqual(records[0]["charged_output_tokens"], 4096)

    def test_absolute_window_stops_even_with_unused_run_budget(self):
        protocol = {"modes": ["I", "T0"], "overall_deadline_utc": "2000-01-01T00:00:00Z"}
        records = run_diagnostics(self.root, self.root / "output", protocol,
                                  [self.sample], [], self.backend)
        self.assertEqual(self.calls, [])
        self.assertEqual([r["runtime_status"] for r in records], ["BUDGET_STOP"] * 2)
        with self.assertRaises(ValueError):
            remaining_window({"overall_deadline_utc": "2099-01-01T00:00:00"})
        with mock.patch("experiments.bottleneck_diagnostic.runner.time.time", return_value=946684790):
            self.assertEqual(remaining_window(protocol), 10)

    def test_backend_errors_are_not_completed_wrong_answers(self):
        def broken(request):
            return {"raw_response": "FINAL_ANSWER: A", "generated_tokens": 4097,
                    "input_tokens": 2, "finish_reason": "length"}
        records = self.run_cells(["I"], backend=broken)
        self.assertEqual(records[0]["runtime_status"], "SYSTEM_ERROR")
        scored = score_record(records[0], self.sample, "A")
        self.assertIsNone(scored["diagnostic_score"])

    def test_diagnostic_extracts_before_gold_and_never_rescues_body(self):
        raw = "The correct answer is A.\nFINAL_ANSWER: B"
        record = {"sample_id": self.sample["sample_id"], "mode": "I", "repetition": 3407,
                  "runtime_status": "COMPLETED", "raw_response": raw, "finish_reason": "stop"}
        scored = score_record(record, self.sample, "A")
        self.assertEqual(scored["diagnostic_parsed_answer"], "B")
        self.assertFalse(scored["diagnostic_score"])
        self.assertEqual(extract_diagnostic(raw, "mcq", ["Alpha", "Beta"])["answer"], "B")
        self.assertEqual(extract_diagnostic("FINAL_ANSWER: A\nFINAL_ANSWER: B", "mcq", ["x", "y"])["status"], "NO_PARSE")
        self.assertEqual(extract_diagnostic("A is correct", "mcq", ["x", "y"])["status"], "NO_PARSE")

    def test_terminal_exact_option_forms_without_gold_or_candidate_guessing(self):
        options = ["red", "blue", "green"]
        for text, answer in [("B", "B"), ("B. blue", "B"), ("blue", "B"),
                             ("B. red", None), ("B. not blue", None), ("B or C", None),
                             ("B. blue or C. green", None), ("B. Blue", None), ("B. blue.", None)]:
            with self.subTest(text=text):
                result = extract_diagnostic("Intermediate answer is A.\nFINAL_ANSWER: " + text, "mcq", options)
                self.assertEqual(result["answer"], answer)
        self.assertIsNone(extract_diagnostic("FINAL_ANSWER: blue", "mcq", ["blue", "blue"])["answer"])
        self.assertIsNone(extract_diagnostic("FINAL_ANSWER: A", "mcq", ["red", "A"])["answer"])
        for text, options, expected in [("A. L sin θ", ["L sin θ", "other"], "A"),
                                        ("C. 2π×10^-5 A", ["one", "two", "2π×10^-5 A"], "C"),
                                        ("B. Not enough information", ["yes", "Not enough information"], "B")]:
            self.assertEqual(extract_diagnostic("FINAL_ANSWER: " + text, "mcq", options)["answer"], expected)
        sample = self.sample | {"options": ["yes", "Not enough information"]}
        record = {"sample_id": sample["sample_id"], "mode": "I", "repetition": 3407,
                  "runtime_status": "COMPLETED", "raw_response": "FINAL_ANSWER: B. Not enough information",
                  "finish_reason": "stop"}
        right, wrong = score_record(record, sample, "B"), score_record(record, sample, "A")
        self.assertEqual(right["diagnostic_parsed_answer"], wrong["diagnostic_parsed_answer"])
        self.assertTrue(right["diagnostic_score"])
        self.assertFalse(wrong["diagnostic_score"])

    def test_structure_scoring_needs_approved_o0_and_keeps_answer_scores_empty(self):
        fields = perception_schema(self.sample)["properties"]["images"]["items"]["properties"]
        facts = {"images": [{field: "<image 1>" if field == "marker" else [] for field in fields}]}
        record = {"sample_id": self.sample["sample_id"], "mode": "P", "repetition": 3407,
                  "runtime_status": "COMPLETED", "raw_response": json.dumps(facts)}
        pending = score_record(record, self.sample, "A")
        self.assertEqual(pending["structure_status"], "PENDING_REVIEW")
        self.assertIsNone(pending["diagnostic_score"])
        oracle = self.oracle | {"facts": facts}
        oracle["oracle_hash"] = oracle_hash(oracle)
        scored = score_record(record, self.sample, "A", oracle)
        self.assertTrue(scored["structure_exact"])
        self.assertEqual(scored["structure_status"], "SCORED_EXACT_JSON")


if __name__ == "__main__":
    unittest.main()
