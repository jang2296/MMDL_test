"""Authored CPU-only checks of single-response source reviews, never real model/source evidence."""
from __future__ import annotations

import copy
import unittest

from fine_tuning.error_note import audit as a
from fine_tuning.error_note.contract import parse_note_v2, resolve_prompt
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE, example, review_for as legacy_review_for


def record_for(*, correct=False, lesson="Read each label before comparing the curves."):
    record = example(correct=correct, lesson=lesson)
    record["image_sha256"] = ["a" * 64]
    record["t1"].update(prompt=record["p0_text"], token_ids=[11, 12], processor_input_sha256="b" * 64)
    record["note"].update(token_ids=[13, 14], processor_input_sha256="c" * 64)
    parsed = parse_note_v2(record["note"]["text"])
    record["resolve"] = {"text": "Final answer: B", "finish_reason": "stop", "status": "CORRECT",
                         "prompt": resolve_prompt(record["p0_text"], parsed.lesson),
                         "lesson_sha256": a.text_sha256(parsed.lesson), "token_ids": [15, 16],
                         "processor_input_sha256": "d" * 64}
    return record


def review_for(record, **kwargs):
    return legacy_review_for(record, **kwargs) | {"single_bindings": a.single_note_review_bindings(record, PROVENANCE)}


def note_for(record, *, review=None, **kwargs):
    return a.audit_single_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE,
                               review=review_for(record) if review is None else review, **kwargs)


def row_for(record, kind):
    return {"row_id": record["problem_id"] + ":" + kind, "problem_id": record["problem_id"], "kind": kind,
            "target_source": "T1" if kind == "A1" else "RESOLVE", "prompt": record["p0_text"],
            "target": record["t1" if kind == "A1" else "resolve"]["text"],
            **{key: copy.deepcopy(record[key]) for key in ("image_paths", "image_sha256", "subject", "source")}}


def solution_review(row, record, note_audit=None):
    return {"version": a.SINGLE_SOLUTION_REVIEW_VERSION, "row_id": row["row_id"], "problem_id": row["problem_id"],
            "target_source": row["target_source"],
            "bindings": a.single_solution_review_bindings(row, record, PROVENANCE, note_audit),
            "reviewer_kind": "ai_review", "human_verified": False, "method": "source_review",
            "content": {"result": "PASS", "scope": "entire_solution",
                        "evidence": ["authored fixture: every solution assertion checked against source"]}}


class SingleNoteAuditTests(unittest.TestCase):
    def test_new_wrapper_preserves_base_and_accepts_only_full_review(self):
        record = record_for()
        audit = note_for(record)
        self.assertEqual(audit["version"], a.SINGLE_NOTE_AUDIT_VERSION)
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["base_audit"], a.audit_note(record, gold="B", choices=["red", "blue"],
                                                          provenance=PROVENANCE, review=review_for(record)))
        self.assertFalse(audit["base_audit"]["review"]["human_verified"])
        self.assertIs(a.validate_single_note_audit(audit, record, PROVENANCE), audit)
        with self.assertRaises(ValueError):
            a.validate_single_note_audit(audit["base_audit"], record, PROVENANCE)
        with self.assertRaises(ValueError):
            a.validate_audit(audit, record, PROVENANCE)

    def test_new_lowercase_and_position_leaks_do_not_change_old_v3_decisions(self):
        for lesson in ("Pick b.", "Choose the second choice."):
            with self.subTest(lesson=lesson):
                record = record_for(lesson=lesson)
                audit = note_for(record)
                self.assertEqual(audit["base_audit"]["leak"]["result"], "CLEAR")
                self.assertIsNotNone(audit["single_hint_leak"])
                self.assertFalse(audit["accepted"])
        alias = note_for(record_for(lesson="Check the crimson label."), aliases=["crimson"])
        self.assertFalse(alias["accepted"])
        self.assertEqual(alias["single_hint_leak"], "SOURCE_ALIAS")
        uncertain = note_for(record_for(lesson="Check TYLCV symptoms."), acronym_candidates=["TYLCV"])
        self.assertFalse(uncertain["accepted"])

    def test_missing_partial_negative_and_program_reviews_never_admit(self):
        record = record_for()
        missing = a.audit_single_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE)
        self.assertFalse(missing["accepted"])
        for kwargs in ({"result": "FAIL"}, {"result": "UNKNOWN"}, {"scope": "partial"},
                       {"kind": "program"}, {"method": "bounded_check"}, {"leak": "UNCERTAIN"}, {"leak": "LEAK"}):
            with self.subTest(kwargs=kwargs):
                audit = note_for(record, review=review_for(record, **kwargs))
                self.assertFalse(audit["accepted"])
        record.pop("note")
        self.assertFalse(a.audit_single_note(record, gold="B", choices=["red", "blue"],
                                             provenance=PROVENANCE)["accepted"])

    def test_stale_note_generation_t1_image_and_provenance_are_rejected(self):
        record = record_for()
        audit = note_for(record)
        mutations = (lambda r: r["note"].update(text=r["note"]["text"].replace("fact0", "fake0")),
                     lambda r: r["note"].update(token_ids=[99]),
                     lambda r: r["note"].update(processor_input_sha256="e" * 64),
                     lambda r: r["t1"].update(token_ids=[98]),
                     lambda r: r["t1"].update(processor_input_sha256="e" * 64),
                     lambda r: r.update(image_sha256=["e" * 64]))
        for mutate in mutations:
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.subTest(record=changed), self.assertRaises(ValueError):
                a.validate_single_note_audit(audit, changed, PROVENANCE)
        with self.assertRaises(ValueError):
            a.validate_single_note_audit(audit, record, PROVENANCE | {"revision": "changed"})

    def test_rehashed_forgery_cannot_turn_missing_review_into_acceptance(self):
        record = record_for()
        audit = a.audit_single_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE)
        audit["accepted"] = True
        audit["sha256"] = a.digest({k: v for k, v in audit.items() if k != "sha256"})
        with self.assertRaises(ValueError):
            a.validate_single_note_audit(audit, record, PROVENANCE)

    def test_rewrapping_a_stale_review_after_original_image_hash_change_is_rejected(self):
        record = record_for()
        review = review_for(record)
        self.assertTrue(note_for(record, review=review)["accepted"])
        record["image_sha256"] = ["e" * 64]
        with self.assertRaisesRegex(ValueError, "reviewer source/image/generation"):
            note_for(record, review=review)

    def test_single_reviewer_binding_is_required_and_covers_generated_records(self):
        record = record_for()
        legacy = legacy_review_for(record)
        self.assertEqual(a.audit_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE,
                                      review=legacy)["content"]["result"], "PASS")
        with self.assertRaisesRegex(ValueError, "reviewer source/image/generation"):
            note_for(record, review=legacy)
        original = a.single_note_review_bindings(record, PROVENANCE)
        self.assertEqual(original["note_record_sha256"], a.digest(record["note"]))
        for mutate in (lambda r: r["t1"].update(processor_input_sha256="e" * 64),
                       lambda r: r["note"].update(token_ids=[99])):
            changed = copy.deepcopy(record)
            mutate(changed)
            self.assertNotEqual(original, a.single_note_review_bindings(changed, PROVENANCE))
            with self.assertRaisesRegex(ValueError, "reviewer source/image/generation"):
                note_for(changed, review=review_for(record))

    def test_original_file_hashes_are_required_and_ai_cannot_claim_human_review(self):
        original = record_for()
        for hashes in (None, [], ["not-a-sha"], ["a" * 64, "b" * 64]):
            record = copy.deepcopy(original)
            record["image_sha256"] = hashes
            with self.subTest(hashes=hashes), self.assertRaises(ValueError):
                note_for(record)
        review = review_for(original)
        review["human_verified"] = True
        with self.assertRaises(ValueError):
            note_for(original, review=review)


class SingleSolutionAuditTests(unittest.TestCase):
    def test_A1_needs_no_note_and_A2_binds_accepted_note_and_generated_target(self):
        first = record_for(correct=True)
        first.pop("note")
        first.pop("resolve")
        row = row_for(first, "A1")
        review = solution_review(row, first)
        result = a.validate_single_solution_review(review, row, first, PROVENANCE)
        self.assertTrue(result["accepted"])
        self.assertIsNone(result["bindings"]["note_audit_sha256"])
        record = record_for()
        note = note_for(record)
        row = row_for(record, "A2")
        review = solution_review(row, record, note)
        result = a.validate_single_solution_review(review, row, record, PROVENANCE, note)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["bindings"]["generation_sha256"], a.digest(record["resolve"]))
        self.assertEqual(result["bindings"]["note_audit_sha256"], note["sha256"])
        self.assertEqual(row["target"], "Final answer: B")

    def test_wrong_kind_or_target_source_and_unaccepted_A2_note_are_refused(self):
        record = record_for()
        note = note_for(record)
        original = row_for(record, "A2")
        for changes in ({"kind": "B"}, {"kind": "A"}, {"kind": "A1"}, {"target_source": "T1"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                a.single_solution_review_bindings(original | changes, record, PROVENANCE, note)
        for missing in (None, a.audit_single_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE),
                        note_for(record, review=review_for(record, result="FAIL"))):
            with self.subTest(note=missing), self.assertRaises(ValueError):
                a.single_solution_review_bindings(original, record, PROVENANCE, missing)

    def test_stale_generation_and_note_cannot_reuse_whole_solution_review(self):
        record = record_for()
        note = note_for(record)
        row = row_for(record, "A2")
        review = solution_review(row, record, note)
        for mutate in (lambda r: r["resolve"].update(token_ids=[99]),
                       lambda r: r["resolve"].update(processor_input_sha256="e" * 64),
                       lambda r: r["resolve"].update(text="Different generated answer"),
                       lambda r: r["note"].update(token_ids=[99])):
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.subTest(record=changed), self.assertRaises(ValueError):
                a.validate_single_solution_review(review, row, changed, PROVENANCE, note)

    def test_wrong_A2_training_prompt_target_resolve_prompt_lesson_and_input_are_refused(self):
        record = record_for()
        note = note_for(record)
        row = row_for(record, "A2")
        for changes in ({"prompt": record["resolve"]["prompt"]}, {"target": "Reviewer-authored explanation"},
                        {"source": "changed"}, {"image_paths": ["changed.png"]},
                        {"image_sha256": ["e" * 64]}, {"lesson_sha256": "e" * 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                a.single_solution_review_bindings(row | changes, record, PROVENANCE, note)
        for changes in ({"prompt": "gold and note supplied"}, {"lesson_sha256": "e" * 64},
                        {"finish_reason": "length"}, {"status": "INCORRECT"}):
            changed = copy.deepcopy(record)
            changed["resolve"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                a.single_solution_review_bindings(row, changed, PROVENANCE, note)

    def test_missing_partial_FAIL_UNKNOWN_and_bounded_reviews_never_pass(self):
        record = record_for(correct=True)
        row = row_for(record, "A1")
        review = solution_review(row, record)
        missing = a.validate_single_solution_review(None, row, record, PROVENANCE)
        self.assertEqual((missing["accepted"], missing["reason"]), (False, "SOLUTION_CONTENT_MISSING_REVIEW"))
        for changes in ({"result": "FAIL"}, {"result": "UNKNOWN", "evidence": []}, {"scope": "partial"}):
            changed = copy.deepcopy(review)
            changed["content"].update(changes)
            with self.subTest(changes=changes):
                self.assertFalse(a.validate_single_solution_review(changed, row, record, PROVENANCE)["accepted"])
        for changes in ({"reviewer_kind": "program"}, {"method": "bounded_check"}, {"method": "regex"}):
            with self.subTest(changes=changes):
                self.assertFalse(a.validate_single_solution_review(review | changes, row, record, PROVENANCE)["accepted"])

    def test_review_identity_human_flag_and_evidence_are_strict(self):
        record = record_for(correct=True)
        row = row_for(record, "A1")
        review = solution_review(row, record)
        for changes in ({"version": a.SOLUTION_REVIEW_VERSION}, {"bindings": {}}, {"row_id": "changed"},
                        {"human_verified": True}, {"human_verified": 0}, {"reviewer_kind": "teacher"},
                        {"extra": "replacement target"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                a.validate_single_solution_review(review | changes, row, record, PROVENANCE)
        for evidence in ([], [""], [1], "reference"):
            changed = copy.deepcopy(review)
            changed["content"]["evidence"] = evidence
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                a.validate_single_solution_review(changed, row, record, PROVENANCE)
        human = review | {"reviewer_kind": "human", "human_verified": True}
        self.assertTrue(a.validate_single_solution_review(human, row, record, PROVENANCE)["accepted"])


if __name__ == "__main__":
    unittest.main()
