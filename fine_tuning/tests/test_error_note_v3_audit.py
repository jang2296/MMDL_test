"""Authored CPU-only fixtures for source-bound note review and v3 row admission."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fine_tuning.error_note.audit import (FILE_VERSION, REVIEW_VERSION, audit_note, digest, read_audits,
                                          review_bindings, text_sha256, validate_audit, validate_row_audits)
from fine_tuning.error_note.contract import (ROUTE_FULL, build_rows_v2, build_rows_v3, resolve_prompt,
                                             select_rows_v3)


def tokenize(text):
    return list(range(len(text.split())))


def example(*, correct=True, misread=False, lesson="Read each label before comparing the curves."):
    # Public invented fixture only; neither the item nor review is training evidence.
    note = ("What the image shows: " + " ".join(f"fact{i}" for i in range(35)) +
            f"\nImage reading: {'MISREAD' if misread else 'CORRECT'}\nCause: {'IMAGE' if misread else 'NONE'}" +
            "\nWhat went wrong: The earlier response confused two labels." + f"\nLesson: {lesson}")
    solution = " ".join(f"step{i}" for i in range(38)) + " Final answer: B"
    return {"problem_id": "fixture-1", "subject": "Biology", "source": "authored-test",
            "p0_text": "P0 question", "image_paths": ["fixture.png"],
            "t1": {"text": solution, "finish_reason": "stop", "status": "CORRECT" if correct else "INCORRECT"},
            "note": {"text": note, "prompt": "Exact note input", "finish_reason": "stop"}}


PROVENANCE = {"source": "authored-test", "revision": "fixture-v1", "split": "train", "native_id": "1",
              "annotations": {"field": "fixture.legend", "value": "two curves"}, "human_verified": False}


def review_for(record, *, result="PASS", scope="entire_note", leak="CLEAR", kind="ai_review",
               method="source_review"):
    return {"version": REVIEW_VERSION, "problem_id": record["problem_id"],
            "bindings": review_bindings(record, PROVENANCE), "reviewer_kind": kind,
            "human_verified": kind == "human", "method": method,
            "content": {"result": result, "scope": scope, "evidence": ["fixture.legend: full note reviewed"]},
            "leak": {"result": leak, "scope": "entire_lesson", "evidence": ["fixture: whole lesson reviewed"]}}


def audited(record=None, *, review=None, **options):
    record = example() if record is None else record
    record["source_provenance"] = copy.deepcopy(PROVENANCE)
    record["audit"] = audit_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE,
                                 review=review_for(record) if review is None else review, **options)
    return record


class NoteAuditTests(unittest.TestCase):
    def test_format_and_no_detected_contradiction_cannot_certify_content_or_lesson(self):
        record = example()
        audit = audit_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE)
        self.assertEqual(audit["format"]["result"], "PASS")
        self.assertEqual((audit["content"]["result"], audit["leak"]["result"]), ("UNKNOWN", "UNCERTAIN"))
        self.assertIs(validate_audit(audit, record, PROVENANCE), audit)

    def test_entire_source_review_is_bound_and_ai_is_not_human_verified(self):
        record = audited()
        audit = record["audit"]
        self.assertEqual((audit["content"]["result"], audit["content"]["scope"]), ("PASS", "entire_note"))
        self.assertEqual((audit["leak"]["result"], audit["review"]["human_verified"]), ("CLEAR", False))
        self.assertEqual(audit, audit_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE,
                                           review=review_for(record)))

    def test_format_regex_program_and_partial_checks_remain_bounded(self):
        record = example()
        for changes in ({"method": "format_only"}, {"method": "no_contradiction"}, {"method": "regex"},
                        {"kind": "program"}, {"scope": "partial"}):
            with self.subTest(changes=changes):
                audit = audit_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE,
                                   review=review_for(record, **changes))
                self.assertEqual(audit["content"]["result"], "UNKNOWN")
                if changes != {"scope": "partial"}:
                    self.assertEqual(audit["leak"]["result"], "UNCERTAIN")

    def test_explicit_negative_source_review_is_preserved(self):
        record = example()
        audit = audited(record, review=review_for(record, result="FAIL"))["audit"]
        self.assertEqual(audit["content"]["result"], "FAIL")

    def test_self_misread_and_no_error_is_an_explicit_contradiction(self):
        record = example(misread=True)
        record["note"]["text"] = record["note"]["text"].replace("Cause: IMAGE", "Cause: NONE")
        audit = audited(record)["audit"]
        self.assertEqual((audit["content"]["result"], audit["content"]["reason"]),
                         ("FAIL", "MISREAD_WITH_CAUSE_NONE"))

    def test_known_leak_and_source_alias_override_review_clear(self):
        cases = [("Use the blue curve.", {}, "GOLD_OPTION_TEXT"),
                 ("Check TYLCV symptoms first.", {"aliases": ["TYLCV"]}, "SOURCE_ALIAS"),
                 ("Check the crimson label.", {"aliases": ["crimson"]}, "SOURCE_ALIAS")]
        for lesson, options, reason in cases:
            with self.subTest(lesson=lesson):
                audit = audited(example(lesson=lesson), **options)["audit"]
                self.assertEqual((audit["leak"]["result"], audit["leak"]["reason"]), ("LEAK", reason))

    def test_unconfirmed_acronym_blocks_clear_and_general_words_do_not(self):
        audit = audited(example(lesson="Check TYLCV symptoms."), acronym_candidates=["TYLCV"])["audit"]
        self.assertEqual((audit["leak"]["result"], audit["leak"]["reason"]), ("UNCERTAIN", "ACRONYM_CANDIDATE"))
        safe = audited(example(lesson="A careful reading checks Fe symbols and F# before comparing curves."),
                       aliases=["TYLCV"])["audit"]
        self.assertEqual(safe["leak"]["result"], "CLEAR")

    def test_review_rejects_wrong_binding_fake_human_flag_and_empty_evidence(self):
        record = example()
        receipts = []
        for field in ("problem_id", "bindings", "human_verified"):
            receipt = review_for(record)
            receipt[field] = {"bad": "binding"} if field == "bindings" else True if field == "human_verified" else "wrong"
            receipts.append(receipt)
        receipt = review_for(record)
        receipt["content"]["evidence"] = []
        receipts.append(receipt)
        for receipt in receipts:
            with self.subTest(receipt=receipt):
                with self.assertRaises(ValueError):
                    audit_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE, review=receipt)

    def test_raw_note_t1_input_note_prompt_and_source_tamper_are_rejected(self):
        record = audited()
        mutations = [lambda r: r["note"].update(text=r["note"]["text"].replace("fact0", "fake0")),
                     lambda r: r["note"].update(prompt="X" + r["note"]["prompt"][1:]),
                     lambda r: r["note"].update(finish_reason="length"),
                     lambda r: r["t1"].update(text="X" + r["t1"]["text"][1:]),
                     lambda r: r["t1"].update(status="INCORRECT"),
                     lambda r: r.update(p0_text="X" + r["p0_text"][1:])]
        for mutate in mutations:
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_audit(record["audit"], changed, PROVENANCE)
        source = copy.deepcopy(PROVENANCE)
        source["revision"] = "fixture-v2"
        with self.assertRaises(ValueError):
            validate_audit(record["audit"], record, source)

    def test_rehashed_forged_audit_decision_is_recomputed(self):
        record = example()
        audit = audit_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE)
        audit["content"].update(result="PASS", scope="entire_note")
        audit["sha256"] = digest({k: v for k, v in audit.items() if k != "sha256"})
        with self.assertRaises(ValueError):
            validate_audit(audit, record, PROVENANCE)

    def test_file_wrapper_and_exact_byte_pin(self):
        record = audited()
        pid = record["problem_id"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            path.write_text(json.dumps({"version": FILE_VERSION, "items": {pid: record["audit"]}}))
            pinned = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(read_audits(path, [record], {pid: PROVENANCE}, expected_sha256=pinned),
                             {pid: record["audit"]})
            with self.assertRaises(ValueError):
                read_audits(path, [record], {pid: PROVENANCE}, expected_sha256="0" * 64)
            with self.assertRaises(ValueError):
                read_audits(path, [], {pid: PROVENANCE})


class RowsV3Tests(unittest.TestCase):
    def test_admitted_targets_and_routes_equal_v2_and_no_note_preserves_a(self):
        record = audited(example(correct=False, misread=True))
        lesson = "Read each label before comparing the curves."
        record["resolve"] = {**record["t1"], "status": "CORRECT", "prompt": resolve_prompt(record["p0_text"], lesson),
                             "lesson_sha256": text_sha256(lesson)}
        rows, drops = build_rows_v3(record, tokenize)
        v2_rows, v2_drops = build_rows_v2(record, tokenize)
        metadata = {"prompt_sha256", "target_sha256", "audit_sha256", "source_sha256", "lesson_sha256"}
        self.assertEqual(([{k: v for k, v in row.items() if k not in metadata} for row in rows], drops),
                         (v2_rows, v2_drops))
        self.assertEqual(rows[0]["prompt_sha256"], text_sha256(record["p0_text"]))
        self.assertEqual(rows[0]["target_sha256"], text_sha256(record["resolve"]["text"]))
        self.assertEqual(rows[0]["audit_sha256"], record["audit"]["sha256"])
        self.assertEqual([row["route"] for row in rows], [ROUTE_FULL, ROUTE_FULL])
        self.assertEqual(select_rows_v3(rows, "note"), rows)
        self.assertEqual(select_rows_v3(rows, "no-note"), rows[:1])
        with self.assertRaises(ValueError):
            select_rows_v3(rows, "solution")

    def test_missing_audit_drops_b_only_and_preserves_t1(self):
        rows, drops = build_rows_v3(example(), tokenize)
        self.assertEqual([(row["kind"], row["target_source"]) for row in rows], [("A", "T1")])
        self.assertEqual(drops, [{"problem_id": "fixture-1", "kind": "B", "reason": "NOTE_CONTENT_UNKNOWN"}])
        record = example()
        del record["note"]
        rows, drops = build_rows_v3(record, tokenize)
        self.assertEqual([row["kind"] for row in rows], ["A"])
        self.assertEqual(drops[0]["reason"], "NO_NOTE")

    def test_failed_or_partial_content_drops_b_and_resolve_even_when_lesson_clear(self):
        for options in ({"result": "FAIL"}, {"scope": "partial"}):
            record = example(correct=False)
            record["resolve"] = {**record["t1"], "status": "CORRECT"}
            record = audited(record, review=review_for(record, **options))
            rows, drops = build_rows_v3(record, tokenize)
            self.assertEqual(rows, [])
            self.assertEqual({drop["kind"] for drop in drops}, {"A", "B"})
            self.assertIn("RESOLVE_LESSON_NOT_AUTHORIZED", [drop["reason"] for drop in drops])

    def test_leaking_lesson_drops_resolve_but_gold_conditioned_b_is_allowed(self):
        record = audited(example(correct=False, lesson="Check TYLCV symptoms."), aliases=["TYLCV"])
        record["resolve"] = {**record["t1"], "status": "CORRECT"}
        rows, drops = build_rows_v3(record, tokenize)
        self.assertEqual([row["kind"] for row in rows], ["B"])
        self.assertEqual(drops[0]["reason"], "RESOLVE_LESSON_NOT_AUTHORIZED")

    def test_unbound_resolve_prompt_or_lesson_is_an_error(self):
        for changes in ({}, {"prompt": "forged", "lesson_sha256": "0" * 64}):
            record = audited(example(correct=False))
            record["resolve"] = {**record["t1"], "status": "CORRECT", **changes}
            with self.assertRaisesRegex(ValueError, "resolve prompt/lesson"):
                build_rows_v3(record, tokenize)

    def test_legacy_misread_gate_and_length_policy_are_unchanged(self):
        record = audited(example(misread=True))
        rows, drops = build_rows_v3(record, tokenize)
        self.assertEqual([row["kind"] for row in rows], ["B"])
        self.assertEqual(drops[0]["reason"], "T1_SELF_MISREAD")
        record = example()
        record["t1"]["text"] = "Final answer: B"
        record = audited(record)
        rows, drops = build_rows_v3(record, tokenize)
        self.assertEqual([row["kind"] for row in rows], ["B"])
        self.assertEqual(drops[0]["reason"], "TOO_SHORT")


class RowAuditLoadingTests(unittest.TestCase):
    def validate(self, record, rows, *, stored_audit=None):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            path.write_text(json.dumps({"version": FILE_VERSION, "items": {
                record["problem_id"]: record["audit"] if stored_audit is None else stored_audit}}))
            validate_row_audits(rows, path)

    def test_t1_and_b_links_are_valid_at_load(self):
        record = audited()
        rows, _ = build_rows_v3(record, tokenize)
        self.validate(record, rows)
        self.validate(record, select_rows_v3(rows, "no-note"))

    def test_resolve_links_are_valid_at_load(self):
        record = audited(example(correct=False))
        lesson = "Read each label before comparing the curves."
        record["resolve"] = {**record["t1"], "status": "CORRECT", "prompt": resolve_prompt(record["p0_text"], lesson),
                             "lesson_sha256": text_sha256(lesson)}
        rows, _ = build_rows_v3(record, tokenize)
        self.validate(record, rows)
        rows[0]["lesson_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.validate(record, rows)

    def test_crosslinked_problem_audit_or_source_is_rejected(self):
        record = audited()
        rows, _ = build_rows_v3(record, tokenize)
        for field, value in (("problem_id", "wrong"), ("audit_sha256", "0" * 64), ("source_sha256", "0" * 64)):
            with self.subTest(field=field):
                changed = copy.deepcopy(rows)
                changed[0][field] = value
                with self.assertRaises(ValueError):
                    self.validate(record, changed)

    def test_same_size_t1_and_note_target_or_prompt_tampering_is_rejected_even_after_rehash(self):
        record = audited()
        rows, _ = build_rows_v3(record, tokenize)
        for index in (0, 1):
            for field in ("prompt", "target"):
                with self.subTest(index=index, field=field):
                    changed = copy.deepcopy(rows)
                    changed[index][field] = "X" + changed[index][field][1:]
                    changed[index][field + "_sha256"] = text_sha256(changed[index][field])
                    with self.assertRaises(ValueError):
                        self.validate(record, changed)

    def test_forged_full_pass_without_explicit_review_is_rejected_at_load(self):
        record = example()
        unknown = audit_note(record, gold="B", choices=["red", "blue"], provenance=PROVENANCE)
        reviewed = audited(record)
        rows, _ = build_rows_v3(reviewed, tokenize)
        unknown["content"].update(result="PASS", scope="entire_note")
        unknown["sha256"] = digest({k: v for k, v in unknown.items() if k != "sha256"})
        for row in rows:
            row["audit_sha256"] = unknown["sha256"]
        with self.assertRaises(ValueError):
            self.validate(record, rows, stored_audit=unknown)

    def test_modified_audit_file_item_is_rejected(self):
        record = audited()
        rows, _ = build_rows_v3(record, tokenize)
        changed = copy.deepcopy(record["audit"])
        changed["content"]["result"] = "FAIL"
        with self.assertRaises(ValueError):
            self.validate(record, rows, stored_audit=changed)


if __name__ == "__main__":
    unittest.main()
