"""Authored CPU-only cases for independent, exact-row solution review receipts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fine_tuning.error_note.audit import (FILE_VERSION, SOLUTION_FILE_VERSION, SOLUTION_REVIEW_VERSION,
                                          digest, solution_quality_reasons, solution_review_bindings, text_sha256)
from fine_tuning.error_note.contract import build_rows_v3, parse_note_v2, resolve_prompt
from fine_tuning.tests.test_error_note_v3_audit import audited, example, review_for, tokenize


class SolutionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.note_path = self.root / "note-audits.json"
        self.solution_path = self.root / "solution-audits.json"
        self.rows_path = self.root / "rows.jsonl"
        first = audited()
        second = example(correct=False)
        second.update(problem_id="fixture-2", p0_text="P0 second question", image_paths=["second.png"])
        second = audited(second)
        lesson = parse_note_v2(second["note"]["text"]).lesson
        second["resolve"] = {"text": " ".join(f"resolved{i}" for i in range(38)) + " Final answer: B",
                             "finish_reason": "stop", "status": "CORRECT",
                             "prompt": resolve_prompt(second["p0_text"], lesson),
                             "lesson_sha256": text_sha256(lesson)}
        self.records = [first, second]
        self.rows = [row for record in self.records for row in build_rows_v3(record, tokenize)[0]]
        self.notes = {record["problem_id"]: record["audit"] for record in self.records}
        self.write_inputs()
        self.document = {"version": SOLUTION_FILE_VERSION,
                         "bindings": {"rows_sha256": self.file_sha(self.rows_path),
                                      "note_audit_sha256": self.file_sha(self.note_path)},
                         "items": {row["row_id"]: self.review(row) for row in self.rows if row["kind"] == "A"}}
        self.flush()

    @staticmethod
    def file_sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_inputs(self):
        self.rows_path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.note_path.write_text(json.dumps({"version": FILE_VERSION, "items": self.notes}))

    def review(self, row):
        return {"version": SOLUTION_REVIEW_VERSION, "row_id": row["row_id"], "problem_id": row["problem_id"],
                "target_source": row["target_source"],
                "bindings": solution_review_bindings(row, self.notes[row["problem_id"]]),
                "reviewer_kind": "ai_review", "human_verified": False, "method": "source_review",
                "content": {"result": "PASS", "scope": "entire_solution",
                            "evidence": ["authored fixture: complete solution checked against source"]}}

    def flush(self):
        self.solution_path.write_text(json.dumps(self.document))

    def check(self):
        return solution_quality_reasons(self.rows, self.note_path, self.solution_path,
                                        rows_sha256=self.file_sha(self.rows_path))

    def first_review(self):
        return self.document["items"]["fixture-1:A"]

    def test_full_source_review_admits_t1_and_resolve_without_changing_rows(self):
        before = copy.deepcopy(self.rows)
        self.assertEqual({row["target_source"] for row in self.rows if row["kind"] == "A"}, {"T1", "RESOLVE"})
        self.assertEqual(self.check(), {})
        self.assertEqual(self.rows, before)
        self.assertFalse(self.first_review()["human_verified"])
        self.first_review().update(reviewer_kind="human", human_verified=True)
        self.flush()
        self.assertEqual(self.check(), {})

    def test_solution_bindings_cover_exact_row_target_prompt_and_note_source_input(self):
        row = next(row for row in self.rows if row["row_id"] == "fixture-1:A")
        audit = self.notes[row["problem_id"]]
        self.assertEqual(solution_review_bindings(row, audit), {
            "row_sha256": digest(row), "target_sha256": text_sha256(row["target"]),
            "prompt_sha256": text_sha256(row["prompt"]), "source_sha256": audit["bindings"]["source_sha256"],
            "input_sha256": audit["bindings"]["input_sha256"], "note_audit_sha256": audit["sha256"]})
        with self.assertRaises(ValueError):
            solution_review_bindings(next(row for row in self.rows if row["kind"] == "B"), audit)

    def test_failed_solution_only_rejects_A_and_keeps_independent_note_rows(self):
        self.first_review()["content"]["result"] = "FAIL"
        self.flush()
        self.assertEqual(self.check(), {"fixture-1:A": "SOLUTION_CONTENT_FAIL"})
        self.assertEqual(sum(row["kind"] == "B" for row in self.rows), 2)

    def test_note_failure_does_not_stand_in_for_solution_review(self):
        record = example()
        record = audited(record, review=review_for(record, result="FAIL"))
        self.rows = build_rows_v3(record, tokenize)[0]
        self.notes = {record["problem_id"]: record["audit"]}
        self.write_inputs()
        self.document["bindings"] = {"rows_sha256": self.file_sha(self.rows_path),
                                     "note_audit_sha256": self.file_sha(self.note_path)}
        self.document["items"] = {row["row_id"]: self.review(row) for row in self.rows}
        self.flush()
        self.assertEqual([row["kind"] for row in self.rows], ["A"])
        self.assertEqual(self.check(), {})

    def test_missing_reviews_drop_all_A_even_when_notes_pass(self):
        self.document["items"] = {}
        self.flush()
        self.assertEqual(self.check(), {row["row_id"]: "SOLUTION_CONTENT_MISSING_REVIEW"
                                       for row in self.rows if row["kind"] == "A"})

    def test_zero_A_accepts_empty_review_items_but_still_requires_valid_files(self):
        self.rows = [row for row in self.rows if row["kind"] == "B"]
        self.write_inputs()
        self.document["bindings"]["rows_sha256"] = self.file_sha(self.rows_path)
        self.document["items"] = {}
        self.flush()
        self.assertEqual(self.check(), {})
        self.solution_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_fail_and_unknown_keep_their_negative_reason(self):
        for result in ("FAIL", "UNKNOWN"):
            with self.subTest(result=result):
                self.first_review()["content"]["result"] = result
                self.first_review()["content"]["evidence"] = [] if result == "UNKNOWN" else ["fixture contradiction"]
                self.flush()
                self.assertEqual(self.check(), {"fixture-1:A": f"SOLUTION_CONTENT_{result}"})

    def test_partial_program_and_bounded_checks_cannot_admit_entire_solution(self):
        original = copy.deepcopy(self.first_review())
        for changes in ({"content": original["content"] | {"scope": "partial"}}, {"reviewer_kind": "program"},
                        *({"method": method} for method in ("bounded_check", "format_only", "no_contradiction", "regex"))):
            with self.subTest(changes=changes):
                self.document["items"]["fixture-1:A"] = original | changes
                self.flush()
                self.assertEqual(self.check(), {"fixture-1:A": "SOLUTION_CONTENT_BOUNDED_REVIEW_ONLY"})

    def test_stale_target_route_and_target_source_are_errors(self):
        original = copy.deepcopy(self.rows)
        for field, value in (("target", "a different complete solution"), ("route", "FULL"), ("target_source", "T1")):
            with self.subTest(field=field):
                self.rows = copy.deepcopy(original)
                row = next(row for row in self.rows if row["target_source"] == "RESOLVE")
                row[field] = value
                row["target_sha256"] = text_sha256(row["target"])
                self.write_inputs()
                self.document["bindings"]["rows_sha256"] = self.file_sha(self.rows_path)
                self.flush()
                with self.assertRaises(ValueError):
                    self.check()

    def test_rebinding_changed_image_subject_or_source_cannot_escape_note_input(self):
        original = copy.deepcopy(self.rows)
        for field, value in (("image_paths", ["swapped.png"]), ("subject", "Changed"), ("source", "other-source")):
            with self.subTest(field=field):
                self.rows = copy.deepcopy(original)
                row = next(row for row in self.rows if row["row_id"] == "fixture-1:A")
                row[field] = value
                self.write_inputs()
                self.document["bindings"]["rows_sha256"] = self.file_sha(self.rows_path)
                self.document["items"][row["row_id"]] = self.review(row)
                self.flush()
                with self.assertRaisesRegex(ValueError, "source/image input"):
                    self.check()

    def test_file_hashes_pin_rows_and_raw_note_bytes(self):
        self.rows_path.write_text(self.rows_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "file bindings"):
            self.check()
        self.document["bindings"]["rows_sha256"] = self.file_sha(self.rows_path)
        self.flush()
        self.note_path.write_text(self.note_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "file bindings"):
            self.check()

    def test_missing_files_are_errors(self):
        self.note_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_unknown_and_non_A_review_ids_are_errors(self):
        for row_id in ("unknown:A", "fixture-1:B"):
            with self.subTest(row_id=row_id):
                self.document["items"][row_id] = self.first_review()
                self.flush()
                with self.assertRaisesRegex(ValueError, "unknown or non-A"):
                    self.check()
                del self.document["items"][row_id]

    def test_duplicate_row_ids_are_errors(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate row_id"):
            self.check()

    def test_review_fields_identity_human_flag_and_binding_are_strict(self):
        original = copy.deepcopy(self.first_review())
        changes = ({"extra": "replacement target forbidden"}, {"version": "wrong"}, {"row_id": "wrong"},
                   {"problem_id": "wrong"}, {"target_source": "RESOLVE"}, {"bindings": {}},
                   {"reviewer_kind": "teacher"}, {"human_verified": True}, {"human_verified": 0},
                   {"method": "unsupported"}, {"content": None})
        for change in changes:
            with self.subTest(change=change):
                self.document["items"]["fixture-1:A"] = original | change
                self.flush()
                with self.assertRaises(ValueError):
                    self.check()
        review = copy.deepcopy(original)
        del review["content"]
        self.document["items"]["fixture-1:A"] = review
        self.flush()
        with self.assertRaises(ValueError):
            self.check()

    def test_content_schema_and_evidence_are_strict(self):
        original = copy.deepcopy(self.first_review()["content"])
        for change in ({"result": "OK"}, {"scope": "entire_note"}, {"extra": "field"},
                       {"evidence": []}, {"evidence": [""]}, {"evidence": [1]}, {"evidence": "reference"}):
            with self.subTest(change=change):
                self.first_review()["content"] = original | change
                self.flush()
                with self.assertRaises(ValueError):
                    self.check()
        self.first_review()["content"] = original | {"result": "FAIL", "evidence": []}
        self.flush()
        with self.assertRaises(ValueError):
            self.check()

    def test_wrapper_schema_and_explicit_rows_sha_are_strict(self):
        original = copy.deepcopy(self.document)
        for change in ({"version": "wrong"}, {"bindings": {}}, {"items": []}, {"extra": 1}):
            with self.subTest(change=change):
                self.document = original | change
                self.flush()
                with self.assertRaises(ValueError):
                    self.check()
        self.document = original
        self.flush()
        for invalid in (None, "x" * 64, "a" * 63):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                solution_quality_reasons(self.rows, self.note_path, self.solution_path, rows_sha256=invalid)

    def test_duplicate_JSON_keys_in_either_receipt_are_errors(self):
        solution_raw = self.solution_path.read_text()
        self.solution_path.write_text(solution_raw.replace('{"version":', '{"version":"ignored","version":', 1))
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.check()
        self.flush()
        note_raw = self.note_path.read_text()
        self.note_path.write_text(note_raw.replace('{"version":', '{"version":"ignored","version":', 1))
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.check()

    def test_malformed_json_and_changed_note_audit_are_errors(self):
        self.solution_path.write_text("{")
        with self.assertRaises(ValueError):
            self.check()
        self.flush()
        self.notes["fixture-1"]["content"]["result"] = "FAIL"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "sha256"):
            self.check()


if __name__ == "__main__":
    unittest.main()
