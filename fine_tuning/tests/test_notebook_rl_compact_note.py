"""Authored CPU checks for the compact full-history note codec and scorer (no model, no GPU)."""
import copy
import hashlib
import json
import types
import unittest
from dataclasses import replace

import torch

from fine_tuning.notebook_rl import compact_note, updates
from fine_tuning.notebook_rl.engines import RolloutRequest
from fine_tuning.notebook_rl.repair_credit import ComponentCredit, _digest
from fine_tuning.notebook_rl.schema import Group, Sample
from fine_tuning.tests.test_notebook_rl_repair_training import OriginalTokenizer
from fine_tuning.notebook_rl.compact_note import (
    CompactNoteError,
    audit,
    credited_spans,
    parse,
    render,
    verdict_label,
)


def note_json(errors: list | None = None, vision: str = "NO", lesson: str | None = None) -> str:
    data: dict = {"errors": [] if errors is None else errors, "vision_misread": vision}
    if lesson is not None:
        data["lesson"] = lesson
    return json.dumps(data)


def obs(attempt: int, item: str, correct: str) -> dict:
    return {"attempt": attempt, "type": "OBSERVATION", "item": item, "correct": correct}


def check(kind: str, locus: str, status: str) -> dict:
    return {"kind": kind, "locus": locus, "status": status}


def native(*checks: dict) -> dict:
    return {"checks": list(checks)}


def ignore_value(index: int, key: str, value: str) -> str:
    return "COMPLIANT"


def score_one(entry: dict, checks: list | None, *, outcome: str = "CORRECT",
              value_status=ignore_value) -> dict:
    note = parse(note_json([entry]))
    natives = [None if checks is None else native(*checks)]
    result = audit(note, outcomes=[outcome], natives=natives, admission="VISION_OFF",
                   value_status=value_status)
    return result["entries"][0]


def attempt_row(sample_id: str, parent: str | None, text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(sample_id=sample_id, parent_sample_id=parent, text=text)


class CompactNoteTests(unittest.TestCase):
    def assertRejects(self, raw: str, code: str) -> None:
        with self.assertRaises(CompactNoteError) as caught:
            parse(raw)
        self.assertEqual(caught.exception.code, code)

    def test_parse_accepts_bare_and_fenced_json(self) -> None:
        bare = note_json(vision="YES")
        for raw in (bare, f"```json\n{bare}\n```"):
            with self.subTest(raw=raw):
                parsed = parse(raw)
                self.assertEqual(parsed.vision, "YES")
                self.assertEqual(parsed.entries, ())
                self.assertEqual(parsed.lesson, "")
                self.assertIsNone(parsed.lesson_span)

    def test_parse_rejects_fence_with_other_language(self) -> None:
        self.assertRejects(f"```python\n{note_json()}\n```", "FENCE")

    def test_parse_rejects_trailing_prose(self) -> None:
        self.assertRejects(note_json() + " and more", "TRAILING_TEXT")

    def test_parse_rejects_non_json(self) -> None:
        self.assertRejects("not json at all", "INVALID_JSON")

    def test_parse_rejects_bad_top_level_shape(self) -> None:
        self.assertRejects('{"errors": [], "vision_misread": "NO", "extra": 1}', "SCHEMA_FIELDS")
        self.assertRejects('{"errors": []}', "SCHEMA_FIELDS")
        self.assertRejects("[]", "SCHEMA_FIELDS")

    def test_parse_rejects_bad_vision_lesson_and_error_type(self) -> None:
        self.assertRejects(note_json(vision="MAYBE"), "VISION_FIELD")
        self.assertRejects(json.dumps({"errors": [], "vision_misread": "NO", "lesson": 3}),
                           "LESSON_FIELD")
        self.assertRejects(json.dumps({"errors": "x", "vision_misread": "NO"}), "ERROR_LIST")

    def test_entries_past_the_cap_are_kept_but_marked_over_limit(self) -> None:
        errors = [{"attempt": n, "type": "NO_FINAL_ANSWER"} for n in range(1, 6)]
        note = parse(note_json(errors, vision="NO"))
        self.assertEqual(len(note.entries), 5)
        self.assertEqual([entry.shape_error for entry in note.entries[:4]], [None] * 4)
        self.assertEqual(note.entries[4].shape_error, "OVER_LIMIT")
        result = audit(note, outcomes=["INVALID_NO_FINAL"] * 5, natives=[{"checks": []}] * 5,
                       admission="VISION_OFF", value_status=ignore_value)
        self.assertEqual([row["score"] for row in result["entries"]], [1, 1, 1, 1, 0])
        self.assertEqual(result["entries"][4]["reason"], "OVER_LIMIT")
        self.assertEqual(result["reward"], 5 / 5)

    def test_entry_shape_errors_are_named_and_score_zero(self) -> None:
        cases = [
            ("ENTRY_TYPE", {"attempt": 1, "type": "BOGUS"}),
            ("ENTRY_FIELDS", {"attempt": 1, "type": "OBSERVATION", "item": "dir[e4]"}),
            ("ENTRY_ATTEMPT", {"attempt": "1", "type": "NO_FINAL_ANSWER"}),
            ("ENTRY_ATTEMPT", {"attempt": True, "type": "WRONG_STEP"}),
            ("ENTRY_VALUE", obs(1, "   ", "C-H")),
            ("ENTRY_VALUE", obs(1, "= x", "C-H")),
        ]
        for code, entry in cases:
            with self.subTest(code=code, entry=entry):
                parsed = parse(note_json([entry]))
                self.assertEqual(parsed.entries[0].shape_error, code)
                row = score_one(entry, None)
                self.assertEqual((row["score"], row["reason"]), (0, code))

    def test_observation_misread_with_corrected_value_scores_plus_one(self) -> None:
        seen: list[tuple[int, str, str]] = []

        def status(index: int, key: str, value: str) -> str:
            seen.append((index, key, value))
            return "COMPLIANT"

        row = score_one(obs(1, "dir[e4]", "  C-H "), [check("OBSERVATION", "dir[e4]", "VIOLATED")],
                        value_status=status)
        self.assertEqual((row["score"], row["reason"]), (1, "MISREAD_AND_CORRECT_VALUE"))
        self.assertEqual((row["site"], row["proposed_value"]), ("VIOLATED", "COMPLIANT"))
        self.assertEqual(len(seen), 1)
        self.assertEqual((seen[0][0], seen[0][2]), (0, "C-H"))

    def test_observation_misread_with_wrong_or_unknown_value_scores_zero(self) -> None:
        checks = [check("OBSERVATION", "dir[e4]", "VIOLATED")]
        for proposed, reason in (("VIOLATED", "MISREAD_ONLY"),
                                 ("UNVERIFIABLE", "VALUE_UNVERIFIABLE")):
            with self.subTest(proposed=proposed):
                row = score_one(obs(1, "dir[e4]", "C-H"), checks,
                                value_status=lambda i, k, v, s=proposed: s)
                self.assertEqual((row["score"], row["reason"]), (0, reason))

    def test_observation_on_correct_fact_is_penalised(self) -> None:
        row = score_one(obs(1, "dir[e4]", "C-H"), [check("OBSERVATION", "dir[e4]", "COMPLIANT")])
        self.assertEqual((row["score"], row["reason"]), (-1, "FACT_WAS_CORRECT"))

    def test_observation_on_absent_fact_is_unverifiable(self) -> None:
        row = score_one(obs(1, "dir[e4]", "C-H"), [check("OBSERVATION", "dir[e8]", "VIOLATED")])
        self.assertEqual((row["score"], row["reason"]), (0, "FACT_UNVERIFIABLE"))

    def test_item_names_are_normalised_before_matching(self) -> None:
        row = score_one(obs(1, "DIR[ e4 ]", "C-H"), [check("OBSERVATION", "dir[e4]", "VIOLATED")])
        self.assertEqual(row["reason"], "MISREAD_AND_CORRECT_VALUE")

    def test_item_may_repeat_the_attempt_line_and_only_the_name_matches(self) -> None:
        row = score_one(obs(1, "dir[e4] = H->G", "C-H"),
                        [check("OBSERVATION", "dir[e4]", "VIOLATED")])
        self.assertEqual((row["score"], row["reason"]), (1, "MISREAD_AND_CORRECT_VALUE"))
        self.assertEqual(row["site"], "VIOLATED")

    def test_no_final_answer_entries_follow_the_outcome(self) -> None:
        entry = {"attempt": 1, "type": "NO_FINAL_ANSWER"}
        cases = (
            ("INVALID_NO_FINAL", 1, "NO_FINAL_CONFIRMED"),
            ("TRUNCATED", 1, "NO_FINAL_CONFIRMED"),
            ("CORRECT", -1, "ATTEMPT_HAD_FINAL"),
            ("INCORRECT", -1, "ATTEMPT_HAD_FINAL"),
            ("INVALID_RESPONSE_CONTRACT", 0, "OUTCOME_UNAVAILABLE"),
        )
        for outcome, score, reason in cases:
            with self.subTest(outcome=outcome):
                row = score_one(entry, [], outcome=outcome)
                self.assertEqual((row["score"], row["reason"]), (score, reason))

    def test_wrong_step_uses_only_trace_rows(self) -> None:
        entry = {"attempt": 1, "type": "WRONG_STEP"}
        cases = (
            ([check("TRACE", "s1", "VIOLATED"), check("TRACE", "s2", "COMPLIANT")],
             (1, "TRACE_VIOLATION_CONFIRMED")),
            ([check("TRACE", "s1", "COMPLIANT")], (-1, "TRACE_COMPLIANT")),
            ([check("OBSERVATION", "dir[e4]", "VIOLATED")], (0, "TRACE_UNVERIFIABLE")),
            ([], (0, "TRACE_UNVERIFIABLE")),
        )
        for checks, expected in cases:
            with self.subTest(checks=checks):
                row = score_one(entry, checks)
                self.assertEqual((row["score"], row["reason"]), expected)

    def test_attempt_out_of_range_scores_zero(self) -> None:
        for number in (0, 2):
            with self.subTest(number=number):
                row = score_one({"attempt": number, "type": "NO_FINAL_ANSWER"}, [],
                                outcome="INVALID_NO_FINAL")
                self.assertEqual((row["score"], row["reason"]), (0, "ATTEMPT_RANGE"))

    def test_repeated_identity_is_scored_once(self) -> None:
        note = parse(note_json([{"attempt": 1, "type": "NO_FINAL_ANSWER"},
                                {"attempt": 1, "type": "NO_FINAL_ANSWER"}]))
        rows = audit(note, outcomes=["INVALID_NO_FINAL"], natives=[{"checks": []}],
                     admission="VISION_OFF", value_status=ignore_value)["entries"]
        self.assertEqual([(row["score"], row["reason"]) for row in rows],
                         [(1, "NO_FINAL_CONFIRMED"), (0, "DUPLICATE")])

    def test_missing_native_audit_is_unauditable(self) -> None:
        for entry in (obs(1, "dir[e4]", "C-H"), {"attempt": 1, "type": "WRONG_STEP"}):
            with self.subTest(entry=entry):
                row = score_one(entry, None)
                self.assertEqual((row["score"], row["reason"]), (0, "ATTEMPT_UNAUDITABLE"))

    def test_vision_claim_is_scored_against_the_route(self) -> None:
        cases = (
            ("VISION_ON", "YES", 1, "MATCHES_ROUTE"),
            ("VISION_ON", "NO", -1, "CONTRADICTS_ROUTE"),
            ("VISION_OFF", "NO", 1, "MATCHES_ROUTE"),
            ("VISION_OFF", "YES", -1, "CONTRADICTS_ROUTE"),
            ("VISION_OFF", "UNSURE", 0, "UNSURE"),
            ("UNRESOLVED", "YES", 0, "ROUTE_UNRESOLVED"),
            ("UNRESOLVED", "NO", 0, "ROUTE_UNRESOLVED"),
        )
        for admission, claim, score, reason in cases:
            with self.subTest(admission=admission, claim=claim):
                result = audit(parse(note_json(vision=claim)), outcomes=["CORRECT"],
                               natives=[{"checks": []}], admission=admission,
                               value_status=ignore_value)
                self.assertEqual(result["vision"]["score"], score)
                self.assertEqual(result["vision"]["reason"], reason)
                self.assertEqual(result["reward"], score / 5)

    def test_reward_is_exact_sum_over_max_errors_plus_one(self) -> None:
        errors = [
            obs(1, "dir[e4]", "C-H"),
            {"attempt": 2, "type": "NO_FINAL_ANSWER"},
            obs(1, "dir[e8]", "G-H"),
        ]
        note = parse(note_json(errors, vision="YES"))
        natives = [native(check("OBSERVATION", "dir[e4]", "VIOLATED"),
                          check("OBSERVATION", "dir[e8]", "COMPLIANT")), native()]
        result = audit(note, outcomes=["INCORRECT", "TRUNCATED"], natives=natives,
                       admission="VISION_ON", value_status=ignore_value)
        self.assertEqual([row["score"] for row in result["entries"]], [1, 1, -1])
        self.assertEqual(result["vision"]["score"], 1)
        self.assertEqual((result["verified"], result["refuted"]), (2, 1))
        self.assertEqual(result["reward"], 2 / 5)
        self.assertIs(result["lesson_rewarded"], False)

    def test_credited_spans_skip_only_the_lesson_literal(self) -> None:
        raw = note_json([], lesson="Hello there")
        note = parse(raw)
        literal = raw[note.lesson_span.start:note.lesson_span.end]
        self.assertEqual(literal, '"Hello there"')
        spans = credited_spans(raw, note)
        self.assertEqual("".join(raw[s.start:s.end] for s in spans), raw.replace(literal, "", 1))
        for span in spans:
            self.assertFalse(span.start < note.lesson_span.end and note.lesson_span.start < span.end)

    def test_credited_spans_cover_whole_note_without_lesson(self) -> None:
        raw = note_json([])
        self.assertEqual([(s.start, s.end) for s in credited_spans(raw, None)], [(0, len(raw))])
        self.assertEqual([(s.start, s.end) for s in credited_spans(raw, parse(raw))],
                         [(0, len(raw))])

    def test_verdict_labels_are_binary(self) -> None:
        self.assertEqual(verdict_label("CORRECT"), "CORRECT")
        self.assertEqual(verdict_label("INVALID_RESPONSE_CONTRACT"),
                         "NOT GRADED (response format invalid)")
        for outcome in ("INCORRECT", "TRUNCATED", "INVALID_NO_FINAL", None):
            with self.subTest(outcome=outcome):
                self.assertEqual(verdict_label(outcome), "NOT CORRECT")

    def test_render_numbers_attempts_from_one_and_maps_retries(self) -> None:
        attempts = [
            attempt_row("a", None, "first text"),
            attempt_row("b", "a", "second text"),
            attempt_row("c", "b", "third text"),
        ]
        text = render("Which?", [], attempts, ["INCORRECT", "CORRECT", "INVALID_RESPONSE_CONTRACT"],
                      None)
        self.assertIn("=== Attempt 1 (first try) - verdict: NOT CORRECT ===", text)
        self.assertIn("=== Attempt 2 (retry of attempt 1) - verdict: CORRECT ===", text)
        self.assertIn("=== Attempt 3 (retry of attempt 2) - verdict: "
                      "NOT GRADED (response format invalid) ===", text)
        self.assertIn("first text", text)
        self.assertNotIn("Unverified independent re-observation", text)
        self.assertTrue(text.endswith(compact_note.PROMPT))

    def test_render_adds_reread_section_only_when_given(self) -> None:
        attempts = [attempt_row("a", None, "only")]
        without = render("Q", [], attempts, ["CORRECT"], None)
        with_reread = render("Q", [], attempts, ["CORRECT"], "REREAD BODY")
        self.assertNotIn("Unverified independent re-observation", without)
        self.assertIn("=== Unverified independent re-observation (not an attempt, may contain "
                      "errors) ===\nREREAD BODY", with_reread)
        self.assertTrue(with_reread.endswith(compact_note.PROMPT))

    def test_render_rejects_outcome_count_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            render("Q", [], [attempt_row("a", None, "x")], [], None)


class CompactGroupCreditTests(unittest.TestCase):
    """Whole-note reward centering in group_credit, and the compact branch of _repair_vectors."""

    def grade(self, index: int, text: str, *, outcome: str = "CORRECT",
              admission: str = "VISION_ON") -> dict:
        base = {"sample_id": f"s{index}",
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "finish_reason": "stop", "scalar_reward": 0.0,
                "component_credit_applied": False, "eligible_for_rl_loss": False}
        try:
            note = parse(text)
        except CompactNoteError as caught:
            return {**base, "syntax_failure": {"code": caught.code}, "audit": None}
        row = audit(note, outcomes=[outcome], natives=[native()], admission=admission,
                    value_status=ignore_value)
        return {**base, "syntax_failure": None, "audit": json.loads(json.dumps(row))}

    def build(self, texts: list[str], grades: list[dict] | None = None):
        tokenizer = OriginalTokenizer()
        samples = []
        for index, text in enumerate(texts):
            ids = tokenizer.pieces(list(text))
            samples.append(Sample(sample_id=f"s{index}", problem_id="p", group_id="g", kind="DIAG",
                                  turn=1, parent_sample_id=None, prompt_token_ids=[1, 2],
                                  completion_token_ids=ids, sampler_logprobs=[-.1] * len(ids),
                                  finish_reason="stop", text=text, seed=index, policy_version=0))
        group = Group(group_id="g", kind="DIAG", problem_id="p", state_sha256=_digest([1, 2]),
                      samples=samples, usable=True, reason=None)
        if grades is None:
            grades = [self.grade(index, text) for index, text in enumerate(texts)]
        return group, grades, tokenizer

    def mixed(self):
        texts = [note_json([], vision="YES"), note_json([], vision="NO"),
                 note_json([], vision="UNSURE"),
                 note_json([{"attempt": 1, "type": "NO_FINAL_ANSWER"}], vision="NO")]
        group, grades, tokenizer = self.build(texts)
        return group, grades, compact_note.group_credit(group, grades, tokenizer)

    def prepared(self, sample: Sample, *, keep: bool = True):
        old = torch.zeros(1, len(sample.completion_token_ids), dtype=torch.float64)
        mask = torch.ones_like(old, dtype=torch.bool)
        request = RolloutRequest("r", sample.prompt_token_ids, None, [], 1, 8192, [sample.seed])
        action = updates._PreparedAction(sample, request, mask, 0., .5, 4 * 8192, False,
                                         old=old, keep=keep)
        return action, old, mask

    def test_centering_gives_each_note_its_reward_minus_the_group_mean(self) -> None:
        group, grades, credits = self.mixed()
        rewards = [grade["audit"]["reward"] for grade in grades]
        for got, want in zip(rewards, (0.2, -0.2, 0.0, -0.4), strict=True):
            self.assertAlmostEqual(got, want)
        mean = sum(rewards) / len(rewards)
        for sample, credit, reward in zip(group.samples, credits, rewards, strict=True):
            row = credit.unit_rows[0]
            with self.subTest(sample=sample.sample_id):
                self.assertAlmostEqual(row["mean"], mean)
                self.assertAlmostEqual(row["advantage"], reward - mean)
                value = compact_note.coefficient(row["advantage"], len(row["token_indices"]), 8192)
                for index in row["token_indices"]:
                    self.assertAlmostEqual(credit.coefficients[index], value)
                self.assertAlmostEqual(sum(credit.coefficients), value * len(row["token_indices"]))

    def test_identical_notes_have_zero_coefficients_and_no_positive_flag(self) -> None:
        for vision in ("UNSURE", "YES", "NO"):
            group, grades, tokenizer = self.build([note_json([], vision=vision)] * 4)
            for credit in compact_note.group_credit(group, grades, tokenizer):
                with self.subTest(vision=vision, sample=credit.sample_id):
                    self.assertTrue(all(value == 0.0 for value in credit.coefficients))
                    self.assertIs(credit.unit_rows[0]["positive_verified_note"], False)

    def test_note_exactly_at_the_mean_gets_exact_zero_advantage(self) -> None:
        # Score units (-5, 0, 1, 4): summing the float rewards leaves -1.4e-17 for the second note.
        texts = [note_json([], vision="NO")] * 4
        group, grades, tokenizer = self.build(texts)
        for grade, (verified, refuted, vision) in zip(grades, ((0, 4, -1), (0, 0, 0), (0, 0, 1), (3, 0, 1)), strict=True):
            grade["audit"].update(verified=verified, refuted=refuted,
                                  reward=compact_note.note_reward(verified, refuted, vision))
            grade["audit"]["vision"]["score"] = vision
        rows = [credit.unit_rows[0] for credit in compact_note.group_credit(group, grades, tokenizer)]
        self.assertEqual(rows[1]["advantage"], 0.0)
        self.assertEqual(rows[1]["coefficient"], 0.0)
        self.assertEqual([row["positive_verified_note"] for row in rows], [False, False, True, True])

    def test_malformed_note_gets_whole_completion_credit_and_lesson_is_not_credited(self) -> None:
        lesson_text = note_json([], vision="YES", lesson="FREE_LESSON")
        group, grades, tokenizer = self.build(
            [lesson_text, "{", note_json([], vision="NO"), note_json([], vision="NO")])
        credits = compact_note.group_credit(group, grades, tokenizer)
        malformed = credits[1].unit_rows[0]
        self.assertIsNone(grades[1]["audit"])
        self.assertEqual(malformed["token_indices"], list(range(len("{"))))
        self.assertEqual(malformed["reward"], 0.0)
        start = lesson_text.index("FREE_LESSON")
        lesson_row = credits[0].unit_rows[0]
        for index in range(start, start + len("FREE_LESSON")):
            with self.subTest(index=index):
                self.assertNotIn(index, lesson_row["token_indices"])
        self.assertIn(0, lesson_row["token_indices"])

    def test_positive_flag_needs_positive_advantage_and_verified_evidence(self) -> None:
        # Vision-only: 0.25 advantage with a verified vision claim is positive; 0.05 with none is not.
        vision_only = [note_json([], vision="YES"), note_json([], vision="UNSURE"),
                       note_json([], vision="NO"), note_json([], vision="NO")]
        group, grades, tokenizer = self.build(vision_only)
        credits = compact_note.group_credit(group, grades, tokenizer)
        flags = [credit.unit_rows[0]["positive_verified_note"] for credit in credits]
        self.assertEqual(flags, [True, False, False, False])
        # Verified NO_FINAL entry (reward 0.2) against three refuted vision claims: only it is positive.
        verified_text = note_json([{"attempt": 1, "type": "NO_FINAL_ANSWER"}], vision="UNSURE")
        texts = [verified_text] + [note_json([], vision="NO")] * 3
        grades = [self.grade(0, verified_text, outcome="INVALID_NO_FINAL")]
        grades += [self.grade(index, text) for index, text in enumerate(texts[1:], 1)]
        group, grades, tokenizer = self.build(texts, grades)
        credits = compact_note.group_credit(group, grades, tokenizer)
        flags = [credit.unit_rows[0]["positive_verified_note"] for credit in credits]
        self.assertEqual(flags, [True, False, False, False])

    def test_credit_shape_and_bindings(self) -> None:
        texts = [note_json([], vision="YES")] + [note_json([], vision="NO")] * 3
        group, grades, tokenizer = self.build(texts)
        credit = compact_note.group_credit(group, grades, tokenizer)[0]
        row = credit.unit_rows[0]
        self.assertIs(type(credit), ComponentCredit)
        self.assertEqual((credit.sample_id, credit.component), ("s0", "final"))
        self.assertEqual((credit.original_group_size, credit.fixed_units, credit.completion_cap),
                         (4, 1, 8192))
        self.assertEqual(len(credit.unit_rows), 1)
        self.assertEqual(set(row), compact_note.ROW_KEYS)
        self.assertEqual((row["origin_ref"], row["kind"]), ("s0:note", "NOTE"))
        self.assertEqual(credit.source_bindings["version"], compact_note.CREDIT_VERSION)
        self.assertEqual(credit.ownership_sha256, _digest([row]))

    def test_rejects_grades_that_do_not_bind_this_note_or_tampered_arithmetic(self) -> None:
        texts = [note_json([], vision="YES")] + [note_json([], vision="NO")] * 3
        cases = {
            "text digest": lambda grades: grades[0].update(text_sha256="0" * 64),
            "audit reward": lambda grades: grades[0]["audit"].update(
                reward=grades[0]["audit"]["reward"] + 0.2),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                group, grades, tokenizer = self.build(texts)
                mutate(grades)
                with self.assertRaises(ValueError):
                    compact_note.group_credit(group, grades, tokenizer)
        group, grades, tokenizer = self.build(texts)
        with self.assertRaises(ValueError):
            compact_note.group_credit(group, grades, tokenizer, completion_cap=1024)

    def test_repair_vectors_accepts_compact_rows_and_counts_positive_only_when_verified(self) -> None:
        group, _grades, credits = self.mixed()
        for sample, credit in zip(group.samples, credits, strict=True):
            action, old, mask = self.prepared(sample)
            with self.subTest(sample=sample.sample_id):
                _partials, member = updates._repair_vectors(action, credit, old, mask)
                nonzero = sum(1 for value in credit.coefficients if value != 0.0)
                positive = credit.unit_rows[0]["positive_verified_note"]
                self.assertEqual(member["signed_tokens"], nonzero)
                self.assertEqual(member["positive_verified_tokens"], nonzero if positive else 0)

    def test_repair_vectors_rejects_tampered_compact_rows(self) -> None:
        group, _grades, credits = self.mixed()
        action, old, mask = self.prepared(group.samples[0])
        cases = {
            "reward": lambda row: row.update(reward=row["reward"] + 0.2),
            "advantage": lambda row: row.update(advantage=row["advantage"] + 0.1),
            "coefficient": lambda row: row.update(coefficient=row["coefficient"] + 0.5),
            "positive flag": lambda row: row.update(
                positive_verified_note=not row["positive_verified_note"]),
            "kind": lambda row: row.update(kind="TRACE"),
            "extra key": lambda row: row.update(extra=1),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                row = copy.deepcopy(credits[0].unit_rows[0])
                mutate(row)
                tampered = replace(credits[0], unit_rows=(row,), ownership_sha256=_digest([row]))
                with self.assertRaisesRegex(ValueError, "compact note provenance"):
                    updates._repair_vectors(action, tampered, old, mask)


if __name__ == "__main__":
    unittest.main()


class PromptLeakTests(unittest.TestCase):
    def test_prompt_contains_no_concrete_fact_name_or_value(self):
        # A concrete example can coincide with a training item's gold value (found in review).
        import re
        self.assertNotIn("->", compact_note.PROMPT)
        self.assertIsNone(re.search(r"[A-Za-z_]+\[[^\]<>]+\]", compact_note.PROMPT))
