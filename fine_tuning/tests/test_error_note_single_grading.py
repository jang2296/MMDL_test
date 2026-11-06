"""SINGLE v8 grading, note-input and program-filter checks (CPU only), plus a stored-output regression.

The authored verifier below runs the real notebook_rl Verifier.grade logic over authored items/rules, so every
expected status is the actual training comparison. StoredCycleRegression regrades the stored cycle-1/2 outputs
with the v7 and v8 graders when those private directories exist and skips cleanly otherwise.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from fine_tuning.error_note import single
from fine_tuning.error_note.contract import (a2_target_problem, answer_denial, hint_leak_single, note_prompt_single,
                                            parse_note_v2)
from fine_tuning.error_note.single_grading import grade_single, normalize_final
from fine_tuning.notebook_rl.schema import VerifierItem
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests.test_error_note_single_contract import note_record

D61 = ("Then $m\\angle HAC = 61^\\circ$.\n\n### Final Answer:\n\n$$\n\\boxed{61^\\circ}\n$$\n\nSo the answer is "
       "$\\boxed{61^\\circ}$.\n\n---\n\n**Answer:** $\\boxed{61^\\circ}$")
SQRT3 = "Side y = 1 part.\n\n✅ **Answer: y = √3**\n\n(Note: y is opposite 30.)\n\n**Final Answer: \\boxed{\\sqrt{3}}**"
Y7 = "x / 1 = y / √3 = 14 / 2 = 7\n\nTherefore:\n\n> **y = 7√3**\n\n---\n\n✅ Final Answer: **y = 7√3**"


class AuthoredVerifier:
    """Real Verifier.grade over authored items: no vault, no gold outside this object."""

    def __init__(self):
        self.items, self.actors, self.training_rules, self.calls = {}, {}, {}, []

    def add(self, pid, gold, kind, accept=None, *, choices=(), source="authored-test"):
        self.items[pid] = VerifierItem(pid, "mcq" if choices else kind, gold, {}, tuple(choices), {}, {}, (), (), ())
        self.actors[pid] = SimpleNamespace(problem_id=pid, question="Authored question?", choices=tuple(choices),
                                           source=source, mode="plain")
        if not choices:
            self.training_rules[pid] = {"answer_type": kind, "accept": accept or {}}
        return self.actors[pid]

    def grade(self, pid, text, finish_reason, mode, *, answer_contract):
        self.calls.append((answer_contract, text))
        return Verifier.grade(self, pid, text, finish_reason, mode, answer_contract=answer_contract)


TOL = {"rel_tol": "0.05"}
OPEN_CASES = (  # text, gold, kind, accept, source, expected (status, extracted)
    ("Therefore, the difference is 2,399,999 units.", "2399999", "decimal", TOL, None, ("CORRECT", "2399999")),
    ("✅ **Final Answer: $ \\boxed{28.5^\\circ} $**", "28.5", "decimal", {"round_to": 1}, None, ("CORRECT", "28.5")),
    ("Therefore, the smallest percentage value is **2.1 %**.", "2.1", "decimal", TOL, None, ("CORRECT", "2.1")),
    ("Therefore, the total downloads between Sonos and SoundCloud is **31.00 thousand**.", "31", "decimal", TOL, None,
     ("CORRECT", "31.00")),
    ("three", "3", "short_text", None, None, ("CORRECT", "3")),
    ("three", "3", "int", None, None, ("CORRECT", "3")),
    ("Therefore, the answer is 3.84 billion euros.", "3.84", "decimal", TOL, None, ("CORRECT", "3.84")),
    ("✅ **Answer: 3 parts**", "3", "decimal", TOL, None, ("CORRECT", "3")),
    ("The range matches.\n\nFinal Answer: 36.0%-39.7%", "36.0%-39.7%", "short_text", None, None,
     ("CORRECT", "36.0%-39.7%")),
    ("The range matches.\n\nFinal Answer: 36.0%-39.7%", "32.1%-35.7%", "short_text", None, None,
     ("INCORRECT", "36.0%-39.7%")),
    ("metallic", "metal", "short_text", None, "hf-clevr", ("CORRECT", "metal")),
    ("Final Answer: matte", "rubber", "short_text", None, "hf-clevr", ("CORRECT", "rubber")),
    ("metallic", "rubber", "short_text", None, "hf-clevr", ("INCORRECT", "metal")),
    ("metallic", "metal", "short_text", None, "hf-chartqa", ("INCORRECT", "metallic")),  # aliases are per source
    (D61, "32", "decimal", {"abs_tol": "0.000001"}, None, ("INCORRECT", "61")),
    (SQRT3, "1", "decimal", {"abs_tol": "0.000001", "variable": "y"}, None, ("INCORRECT", "1.73205080757")),
    (Y7, "7", "decimal", {"abs_tol": "0.000001", "variable": "y"}, None, ("INCORRECT", "12.124355653")),
    (Y7, "12.12", "decimal", {"abs_tol": "0.01", "variable": "y"}, None, ("CORRECT", "12.124355653")),
    # ambiguous scale/format: never loosened
    ("75", "0.75", "decimal", TOL, None, ("UNVERIFIABLE", "75")),  # x100 scale ambiguity: excluded, not CORRECT
    ("Final Answer: 75%", "0.75", "decimal", TOL, None, ("UNVERIFIABLE", "75%")),
    ("✅ **Answer: Radiography (Chest X-ray)**", "X-Ray", "short_text", None, None,
     ("INCORRECT", "radiography (chest x-ray)")),
    ("Final Answer: X-Ray", "chest X-ray", "short_text", None, None, ("INCORRECT", "x-ray")),
    ("Final Answer: 31 thousand", "31000", "int", None, None, ("INCORRECT", "31")),
    ("Final Answer: 31 thousand", "31 thousand", "short_text", None, None, ("CORRECT", "31 thousand")),
    ("Final Answer: 3.84 million", "3.84 billion", "short_text", None, None, ("INCORRECT", "3.84 million")),
    ("Final Answer: 2 pi", "2", "decimal", TOL, None, ("INCORRECT", "2 pi")),  # undecided: V8's INCORRECT kept
    ("Therefore, the segment is divided into three parts.", "3", "decimal", TOL, None,
     ("INCORRECT", "the segment is divided into three parts")),
    # no single committed answer
    ("Answer: 5\n\nAfter checking again.\n\nFinal Answer: 6", "6", "decimal", TOL, None, ("INVALID_MULTIPLE", None)),
    ("Final Answer: 5 or 6", "6", "decimal", TOL, None, ("INVALID_MULTIPLE", None)),
    ("Final Answer: 21 and 37", "21", "decimal", TOL, None, ("INVALID_MULTIPLE", "21 and 37")),
    ("I could not finish the computation.\nIt depends on more data.", "6", "decimal", TOL, None,
     ("INVALID_NO_FINAL", None)),
)


class NormalizeFinalTests(unittest.TestCase):
    def test_listed_notations(self):
        for value, expected in (
                ("2399999 unit", "2399999"), ("2,399,999 units", "2399999"), ("28.5^\\circ", "28.5"),
                ("$ \\boxed{28.5^{\\circ}} $", "28.5"), ("**61°**.", "61"), ("61 degrees", "61"),
                ("2.1 %", "2.1"), ("2.1 percent", "2.1"), ("31.00 thousand", "31.00"), ("three", "3"),
                ("Three parts", "3"), ("3.84 billion euros", "3.84"), ("74.3 years", "74.3"), ("3 parts", "3"),
                ("\\boxed{\\sqrt{3}}", "√3"), ("y = 7\\sqrt{3}", "7√3"), ("\\text{metal}", "metal"),
                ("36.0%-39.7%", "36.0%-39.7%"), ("63–77", "63-77"), ("$5", "5"),
                ("the total downloads is 31.00 thousand", "31.00"),
                # value-changing or multi-value words are never stripped
                ("5 more than x", "5 more than x"), ("2 pi", "2 pi"), ("3 times", "3 times"), ("21 and 37", "21 and 37"),
                ("three symbols in the legend", "three symbols in the legend"),
                ("the liver is bigger", "the liver is bigger"), ("x^2 + 1", "x^2 + 1")):
            with self.subTest(value=value):
                self.assertEqual(normalize_final(value), expected)
                self.assertEqual(normalize_final(normalize_final(value)), expected)  # idempotent


class GradeSingleTests(unittest.TestCase):
    def test_open_cases_one_committed_answer_one_comparison_chain(self):
        for index, (text, gold, kind, accept, source, expected) in enumerate(OPEN_CASES):
            with self.subTest(text=text, gold=gold):
                verifier = AuthoredVerifier()
                actor = verifier.add(f"p{index}", gold, kind, accept, source=source or "authored-test")
                self.assertEqual(grade_single(verifier, actor, f"p{index}", text, "stop", "plain"), expected)
                self.assertLessEqual(len(verifier.calls), 4)
                calls = verifier.calls
                if calls and calls[-1][0] == "p0-eval-aligned-v1":  # only after an undecided comparison
                    self.assertEqual(calls[-1][1], text)
                    calls = calls[:-1]
                for contract, graded in calls:  # only ever one normalized answer, never the response
                    self.assertEqual(contract, "training-answer-v2")
                    self.assertTrue(graded.startswith("Final answer: ") and "\n" not in graded)

    def test_mcq_is_v8_unchanged_and_length_is_truncated(self):
        verifier = AuthoredVerifier()
        actor = verifier.add("m", "B", "mcq", choices=("red", "blue"))
        self.assertEqual(grade_single(verifier, actor, "m", "Blue is right.\nFinal Answer: B", "stop", "plain"),
                         ("CORRECT", "B"))
        self.assertEqual(verifier.calls, [("p0-eval-aligned-v1", "Blue is right.\nFinal Answer: B")])
        open_actor = verifier.add("o", "3", "decimal", TOL)
        self.assertEqual(grade_single(verifier, open_actor, "o", "Final Answer: 3", "length", "plain"), ("TRUNCATED", None))

    def test_committed_answers_never_get_the_no_committed_penalty(self):
        """\\boxed{61^\\circ}, \\boxed{\\sqrt{3}}, 'y = 7\\sqrt{3}': v7 gave INVALID_NO_FINAL (-1); v8 reads one answer."""
        for text, gold in ((D61, "32"), (SQRT3, "1"), (Y7, "7")):
            verifier = AuthoredVerifier()
            actor = verifier.add("p", gold, "decimal", {"abs_tol": "0.000001", "variable": "y"})
            status, extracted = grade_single(verifier, actor, "p", text, "stop", "plain")
            self.assertEqual(status, "INCORRECT")
            self.assertIsNotNone(extracted)


class NoteAndFilterTests(unittest.TestCase):
    def test_note_prompt_shows_the_grader_read_value_after_the_grading_result(self):
        prompt = note_prompt_single("Question: P0", "... is 2,399,999 units.", "INCORRECT", "2399999", read="2399999 unit")
        self.assertIn("Grading result: WRONG\nThe grader read your final answer as: 2399999 unit\nCorrect answer: 2399999",
                      prompt)
        none = note_prompt_single("Question: P0", "no answer", "INVALID_NO_FINAL", "7", read=None)
        self.assertIn("\nThe grader read your final answer as: no final answer\nCorrect answer: 7", none)

    def test_note_cause_none_and_mismatch_block_only_the_b_row(self):
        audit = {"base_audit": {"format": {"result": "PASS"}, "leak_inputs": {"choices": []},
                                "checks": {"hint_leak": None}}, "single_hint_leak": None, "sha256": "0" * 64}
        for cause, status, reason in (("NONE", "INCORRECT", "NOTE_CAUSE_NONE"), ("NONE", "INVALID_NO_FINAL", "NOTE_CAUSE_NONE"),
                                      ("REASONING", "INVALID_NO_FINAL", "NOTE_CAUSE_MISMATCH"),
                                      ("IMAGE", "INVALID_MULTIPLE", "NOTE_CAUSE_MISMATCH"),
                                      ("CALCULATION", "TRUNCATED", "NOTE_CAUSE_MISMATCH"),
                                      ("NO_FINAL_ANSWER", "TRUNCATED", None), ("REASONING", "INCORRECT", None),
                                      ("NO_FINAL_ANSWER", "INCORRECT", None)):
            with self.subTest(cause=cause, status=status):
                record = note_record(cause, status)
                self.assertEqual(parse_note_v2(record["note"]["text"]).cause, cause)
                decision = single.note_program_decision(audit, record)
                self.assertEqual((decision["accepted"], decision["reason"]), (True, None))  # the re-solve still runs
                self.assertEqual(single.note_cause_problem(record["note"]["text"], status), reason)  # B row gate

    def test_lesson_stem_leak_and_counted_over_rejection(self):
        for lesson, gold, choices, reason in (
                ('Avoid calling a shiny surface "metallic" without checking reflections.', "metal", [], "STEM:meta"),
                ("Name bronchial tree segments by their position.", "Pulmonary bronchus", [], "STEM:bronch"),
                ("An increasing curve has a positive slope.", "B", ["increase", "decrease"], "STEM:increa"),
                ("Count the bars from the left axis.", "metal", [], None),
                ("Check the red channel of the stain.", "E", ["red", "blue"], None)):
            with self.subTest(lesson=lesson):
                self.assertEqual(hint_leak_single(lesson, gold, choices), reason)
        over = ["Consider metabolism of the sample first.", "Use the metadata of the chart.",
                "Read the axis before an increased value is assumed.", "Compare decreasing intervals in order."]
        flagged = [hint_leak_single(lesson, "metal", ["increase", "decrease"]) for lesson in over]
        self.assertEqual(sum(reason is not None and reason.startswith("STEM:") for reason in flagged), 4)  # 4/4 over-rejected

    def test_denial_phrases_and_word_boundary_review_references(self):
        for text, phrase in (("Perhaps the question is flawed, so 3.", "question is flawed"),
                             ("The angle is mislabeled in the figure.", "mislabel"),
                             ("There is a typo in the statement.", "typo in the"),
                             ("The line is not explicitly drawn, so assume 4/4.", "not explicitly drawn"),
                             ("Assuming the question is incomplete, pick 2.", "question is incomplete"),
                             ("This may be a misprint.", "misprint"),
                             ("There is a mistake in the problem.", "mistake in the problem"),
                             ("However, the question is **incomplete** — it does not say which.", "question is incomplete"),
                             ("The bar is not *explicitly* drawn.", "not explicitly drawn"),
                             ("The labels are clear and consistent.", None), ("The unmislabeled bars.", None)):
            with self.subTest(text=text):
                self.assertEqual(answer_denial(text), phrase)
        lesson = "Read the axis labels and the legend carefully before comparing any of the two curves."
        self.assertIsNone(a2_target_problem("This household has personal savings of 75 percent. Final Answer: 0.75", lesson))
        self.assertEqual(a2_target_problem("As per the chart, 0.75. Final Answer: 0.75", lesson), "REVIEW_REFERENCE:as per")
        self.assertEqual(a2_target_problem("Perhaps the question is flawed. Final Answer: 3", lesson),
                         "ANSWER_DENIAL:question is flawed")


CYCLES = {"c1": "pod-v6-cycle1-k2fuyy3cxfnivw", "c2": "pod-v7-cycle2-ame5bsd5hhd0z6"}
ROOT = Path("/home/jang2/mmdl-artifacts/self-note-single-20261105")
PARAMS = Path("/home/jang2/mmdl-artifacts/pod-outbox/jobs-single-v3/single-v5.params.json")
# Listed cycle-2 review cases (claudedocs/analysis_single_v7_cycle2_20261106.md 1-na, 4.2, 5.1): v8 status.
EXPECTED = {
    ("c1", "000547", "t1"): "CORRECT", ("c2", "000547", "t1"): "CORRECT", ("c1", "000547", "resolve"): "CORRECT",
    ("c1", "004301", "t1"): "CORRECT", ("c2", "004301", "t1"): "CORRECT", ("c1", "004301", "resolve"): "CORRECT",
    ("c2", "000555", "t1"): "CORRECT", ("c2", "000555", "resolve"): "CORRECT", ("c1", "000555", "resolve"): "CORRECT",
    ("c1", "000567", "t1"): "CORRECT", ("c2", "000567", "t1"): "CORRECT", ("c1", "000567", "resolve"): "CORRECT",
    ("c1", "000580", "t1"): "CORRECT", ("c2", "000580", "t1"): "CORRECT",
    ("c1", "000592", "t1"): "CORRECT", ("c2", "000592", "t1"): "CORRECT",
    ("c1", "003489", "t1"): "CORRECT", ("c2", "003489", "t1"): "CORRECT", ("c2", "003489", "resolve"): "CORRECT",
    ("c2", "005647", "resolve"): "INCORRECT",  # 'Final Answer: 36.0%-39.7%' is now one committed (wrong) range
    ("c2", "004300", "t1"): "INCORRECT", ("c2", "004302", "t1"): "INCORRECT",  # committed: never NO_COMMITTED
    ("c1", "004304", "t1"): "INCORRECT", ("c2", "004304", "t1"): "INCORRECT",
    ("c1", "000552", "t1"): "UNVERIFIABLE", ("c2", "000552", "t1"): "UNVERIFIABLE",  # '75' vs 0.75: scale-ambiguous
    ("c2", "010573", "t1"): "INCORRECT",  # 'Radiography (Chest X-ray)' vs 'X-Ray' stays
}
# AI-reviewed (human_verified=false) wrong answers of the 64-item review: none may become CORRECT.
REVIEWED_WRONG = {(cycle, pid, "t1") for cycle in CYCLES for pid in (
    "005640", "006549", "004300", "010572", "010633", "010178", "004302", "006551", "003334", "008761", "004303",
    "004304", "011284", "006553", "008068", "000552")} | {
    ("c2", "004299", "t1"), ("c2", "000584", "t1"), ("c2", "010886", "t1"), ("c1", "005640", "resolve"),
    ("c2", "005640", "resolve"), ("c1", "010572", "resolve"), ("c2", "010572", "resolve"), ("c2", "010633", "resolve"),
    ("c2", "004303", "resolve"), ("c1", "006553", "resolve"), ("c2", "006553", "resolve"), ("c1", "004300", "resolve"),
    ("c2", "000584", "resolve"), ("c1", "000552", "resolve"), ("c2", "005647", "resolve")}


@unittest.skipUnless(PARAMS.exists() and all((ROOT / name).is_dir() for name in CYCLES.values()),
                     "stored private cycle outputs are absent")
class StoredCycleRegression(unittest.TestCase):
    def test_v7_to_v8_transitions_on_every_stored_t1_and_resolve(self):
        from fine_tuning.error_note.protocol import SINGLE_ANSWER_UNION
        from fine_tuning.notebook_rl.answers import TURN1_EVAL_CONTRACT
        from fine_tuning.notebook_rl.vault import read_actors

        params = json.loads(PARAMS.read_text())
        root = Path(params["build_root"]) / "train"
        verifier = Verifier(root, training_rules_path=Path(params["training_rules_path"]),
                            training_rules_sha256=params["training_rules_sha256"])
        actors = {actor.problem_id: actor for actor in read_actors(root)}
        moves, seen, stem_only = Counter(), {}, 0
        for cycle, name in CYCLES.items():
            campaign = ROOT / name / "x/single-v5/campaign"
            for phase in sorted(campaign.glob("block-*/phase.json")):
                for pid, record in json.loads(phase.read_text())["records"].items():
                    actor, short = actors[pid], pid.split(":")[3]
                    for stage in ("t1", "resolve"):
                        output = record.get(stage)
                        if output is None:
                            continue
                        args = (pid, output["text"], output["finish_reason"], actor.mode)
                        old = single.grade_answer(verifier, TURN1_EVAL_CONTRACT if actor.choices else SINGLE_ANSWER_UNION,
                                                  *args)[0]
                        new = grade_single(verifier, actor, *args)[0]
                        moves[("mcq" if actor.choices else "open", stage, old, new)] += 1
                        seen[(cycle, short, stage)] = new
                        if actor.choices:
                            self.assertEqual(old, new, (cycle, pid, stage))  # MCQ grading is unchanged
                    parsed = parse_note_v2(record["note"]["text"]) if record.get("note") else None
                    if parsed is not None:
                        reason = hint_leak_single(parsed.lesson, verifier.items[pid].gold, actor.choices)
                        stem_only += reason is not None and reason.startswith("STEM:")
        print("\nv7 -> v8 transitions (kind, stage, v7, v8): count")
        for key, count in sorted(moves.items()):
            print(f"  {key}: {count}{'' if key[2] == key[3] else '  <- changed'}")
        print(f"  stored lessons whose first leak reason is the new STEM rule: {stem_only}")
        for key, status in EXPECTED.items():
            with self.subTest(case=key):
                self.assertEqual(seen.get(key), status)
        for key in sorted(REVIEWED_WRONG & set(seen)):
            with self.subTest(wrong=key):
                self.assertNotEqual(seen[key], "CORRECT")


if __name__ == "__main__":
    unittest.main()
