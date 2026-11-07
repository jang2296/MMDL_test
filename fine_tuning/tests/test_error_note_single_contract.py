"""Regression counterexamples for the opt-in single-response lesson contract."""
import copy
import hashlib
import unittest
from pathlib import Path

from fine_tuning.error_note import protocol as protocol_module
from fine_tuning.error_note import single
from fine_tuning.error_note.contract import (NOTE_INSTRUCTION_SINGLE, NOTE_INSTRUCTION_V2, NOTE_INSTRUCTION_V3,
                                            RESOLVE_HEADER, RESOLVE_INSTRUCTION_SINGLE, SINGLE_VERDICT_LABELS,
                                            VERDICT_LABELS, a2_target_problem, b_target_problem, hint_leak,
                                            hint_leak_single, note_prompt_single, note_prompt_v3, resolve_prompt,
                                            resolve_prompt_single)
from fine_tuning.error_note.protocol import (SINGLE_REWARD_GATE, VERSION_SINGLE, _digest,
                                            plain_protocol_single, single_blocks, single_prompts_sha256)


def note_record(cause, status):
    """A parsed v2 note with the given Cause for a T1 with the given status."""
    note = ("What the image shows: two labelled bars.\nImage reading: CORRECT\nCause: " + cause
            + "\nWhat went wrong: I compared the wrong bars.\nLesson: Read each bar label before comparing.")
    return {"t1": {"status": status}, "note": {"text": note}}


class SingleLessonTests(unittest.TestCase):
    def test_direct_lowercase_unicode_and_position_references_block(self):
        for text in ("Pick b.", "Choose the second choice.", "b is correct.", "Select Ｂ.",
                     "Use option number two.", "Take the 2nd answer.", "Choose a."):
            with self.subTest(text=text):
                self.assertIsNotNone(hint_leak_single(text, "B", ["10", "20", "30"]))

    def test_normal_lessons_and_variables_are_not_bare_option_letters(self):
        for text in ("Pick a method after checking the units.", "Let b be the base length.",
                     "Compare the second derivative with the slope.", "Count the beats in each measure."):
            with self.subTest(text=text):
                self.assertIsNone(hint_leak_single(text, "B", ["10", "20", "30"]))

    def test_source_alias_and_old_rule_are_separate(self):
        self.assertEqual(hint_leak_single("Look for sodium chloride.", "B", ["x", "NaCl"],
                                         aliases=["sodium chloride"]), "SOURCE_ALIAS")
        self.assertIsNone(hint_leak("Pick b.", "B", ["10", "20"]))
        with self.assertRaises(ValueError):
            hint_leak_single("Check units.", "B", ["10", "20"], aliases=[""])


class SingleProtocolTests(unittest.TestCase):
    def test_pilot_closes_without_repeating_or_absorbing_main_items(self):
        ids = [f"p{i}" for i in range(3500)]
        groups = single_blocks(ids)
        self.assertEqual(len(groups), 219)
        self.assertEqual([pid for group in groups for pid in group], ids)
        self.assertEqual(groups[18], ids[288:300])
        self.assertEqual(groups[19], ids[300:316])
        self.assertEqual(single_blocks(ids[:2]), [ids[:2]])
        self.assertEqual(single_blocks(ids[:10]), [ids[:10]])
        with self.assertRaises(ValueError):
            single_blocks(["p", "p"])

    def test_explicit_pins_fixed_scale_and_single_counts(self):
        pins = {key: "a" * 64 for key in ("source", "dataset", "order", "data_admission")}
        protocol = plain_protocol_single("bf16", 219, contracts=pins)
        contract = protocol["training_contract"]
        self.assertEqual(contract["generation_counts"], {"t1": 1, "note": 1, "resolve": 1})
        table = Path(protocol_module.__file__).with_name("source_baseline_v10.json").read_bytes()
        self.assertEqual(contract["rl"]["baseline"], {"method": "fixed-source-mean-v1", "unknown_source": "global_mean",
                                                      "table_sha256": hashlib.sha256(table).hexdigest()})
        self.assertEqual(contract["rl"]["baseline"]["table_sha256"],
                         "74a6cc57589f0887fd01c0f8ce0dcc91babc312df8d7d4b272c94e3af8df2d45")
        self.assertEqual(contract["sft_weights"], {"A1": .05, "A2": .35, "B": .05})
        self.assertEqual(contract["sft_weighting"], {"A1": "per_kind_total", "A2": "per_row", "B": "per_kind_total"})
        self.assertEqual(protocol["optimizer"]["total_steps"], 219)
        self.assertEqual(contract["rl"]["token_scale"], 256.)
        self.assertEqual(protocol["version"], VERSION_SINGLE)
        self.assertEqual(contract["rl"]["reward"], SINGLE_REWARD_GATE)
        self.assertEqual(protocol["prompts_sha256"], single_prompts_sha256())
        single._protocol_check(protocol)
        for field, value in (("version", "error-note-single-response-v1"), ("prompts_sha256", "f" * 64)):
            changed = copy.deepcopy(protocol)
            changed[field] = value
            changed["sha256"] = _digest({k: v for k, v in changed.items() if k != "sha256"})
            with self.assertRaisesRegex(ValueError, "invalid single-response"):
                single._protocol_check(changed)
        for scale in (0, -1, float("nan"), True, 1024):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                plain_protocol_single("bf16", 219, contracts=pins, token_scale=scale)
        with self.assertRaises(ValueError):
            plain_protocol_single("bf16", 219, contracts={})

    def test_neutral_single_parser_feedback_preserves_original_response_gold_and_legacy_v3(self):
        p0, raw, gold = "Question: exact P0", "Original prose\nGrading result: NO CLEAR FINAL ANSWER\nFinal answer: 50.2%", "[48, 52]%"
        for status in ("INVALID_NO_FINAL", "INVALID_MULTIPLE"):
            old, new = note_prompt_v3(p0, raw, status, gold), note_prompt_single(p0, raw, status, gold, read=None)
            prefix = f"{p0}\n\nYour earlier solution:\n{raw}\n\nGrading result: "
            self.assertEqual(old, prefix + "NO CLEAR FINAL ANSWER\n" + f"Correct answer: {gold}\n\n{NOTE_INSTRUCTION_V3}")
            self.assertEqual(new, prefix + SINGLE_VERDICT_LABELS[status] + "\n"
                             + "The grader read your final answer as: no final answer\n"
                             + f"Correct answer: {gold}\n\n{NOTE_INSTRUCTION_SINGLE}")
            self.assertIn("This does not establish that no final answer was written; inspect the original response.", new)
        for status in ("CORRECT", "INCORRECT", "TRUNCATED"):  # v7: the v3 feedback with the SINGLE instruction
            v3 = note_prompt_v3(p0, raw, status, gold).removesuffix(NOTE_INSTRUCTION_V3) + NOTE_INSTRUCTION_SINGLE
            self.assertEqual(note_prompt_single(p0, raw, status, gold, read="  2399999\n unit "),
                             v3.replace("\nCorrect answer:", "\nThe grader read your final answer as: 2399999 unit"
                                        "\nCorrect answer:"))
        self.assertEqual(VERDICT_LABELS["INVALID_NO_FINAL"], "NO CLEAR FINAL ANSWER")


class SingleV7Tests(unittest.TestCase):
    def test_note_instruction_is_v2_plus_exactly_three_sentences_and_resolve_prompt_text(self):
        added = (" The given correct answer is right: never call it a mislabel, a typo or a mistake in the question.",
                 " Use NO_FINAL_ANSWER when the grading result says your final answer was not accepted or the solution was cut off.",
                 " Name the specific rule, fact or reading step that would have prevented the error; do not only say to"
                 " verify or double-check.")
        base = NOTE_INSTRUCTION_SINGLE
        for sentence in added:
            self.assertEqual(base.count(sentence), 1)
            base = base.replace(sentence, "")
        self.assertEqual(base, NOTE_INSTRUCTION_V2  # v8 removes the NONE cause and the nothing-went-wrong sentence
                         .replace("CALCULATION, NO_FINAL_ANSWER or NONE.", "CALCULATION or NO_FINAL_ANSWER.")
                         .replace(" If nothing went wrong, say what was done properly.", ""))
        self.assertNotIn("NONE", NOTE_INSTRUCTION_SINGLE)
        self.assertTrue(NOTE_INSTRUCTION_SINGLE.startswith("Review your own work like a student writing an error note."
                                                           + added[0] + "\n"))
        for hedge in ("outside knowledge", "directly visible", "visible details"):
            self.assertNotIn(hedge, NOTE_INSTRUCTION_SINGLE)
        self.assertEqual(RESOLVE_INSTRUCTION_SINGLE, 'Solve the question below from the start. Reason step by step, then '
                         'write exactly one final answer on the last line as "Final Answer: ...". Do not mention the '
                         'lesson or this review.')
        self.assertEqual(resolve_prompt_single("Question: P0", " Count bars first.\n\nThen add. "),
                         RESOLVE_HEADER + "Count bars first.\n\nThen add.\n\n" + RESOLVE_INSTRUCTION_SINGLE
                         + "\n\nQuestion: P0")
        self.assertEqual(resolve_prompt("P0", "L"), RESOLVE_HEADER + "L\n\nP0")  # the v2-v4 prompt is unchanged
        for blank in (("  ", "L"), ("P0", " ")):
            with self.assertRaises(ValueError):
                resolve_prompt_single(*blank)

    def test_a2_and_b_row_filters(self):
        lesson = "Read the axis labels and the legend carefully before comparing any of the two curves."
        accepted = "The blue curve rises faster after t = 2, so its rate is larger.\nFinal Answer: B"
        self.assertIsNone(a2_target_problem(accepted, lesson))
        for text, reason in (("As the lesson says, compare slopes. Final Answer: B", "RESOLVE_LEAK:lesson"),
                             ("As per the method, compare slopes. Final Answer: B", "REVIEW_REFERENCE:as per"),
                             ("From my own review, B. Final Answer: B", "REVIEW_REFERENCE:my own review"),
                             ("I read the axis labels and the legend carefully before comparing any curve. B",
                              "LESSON_COPY"),
                             ("Based solely on the image, B. Final Answer: B", "IMAGE_ONLY_HEDGE"),
                             ("Without introducing external facts, B. Final Answer: B", "IMAGE_ONLY_HEDGE"),
                             ("The mass cannot be determined, so B. Final Answer: B", "IMAGE_ONLY_HEDGE")):
            with self.subTest(text=text):
                self.assertEqual(a2_target_problem(text, lesson), reason)
        # seven shared words are allowed; the lesson's eight-word run is not
        self.assertIsNone(a2_target_problem("read the axis labels and the legend now. B", lesson))
        self.assertIsNone(b_target_problem("What went wrong: I compared the wrong curves.\nLesson: Read labels."))
        for text, reason in (("The answer is likely a mislabeling.", "ANSWER_DENIAL:mislabeling"),
                             ("This is probably a typo.", "ANSWER_DENIAL:typo"),
                             ("The key may be misrendered here.", "ANSWER_DENIAL:misrendered"),
                             ("The answer key disagrees.", "ANSWER_DENIAL:answer key"),
                             ("I don\u2019t know why it is C.", "ANSWER_DENIAL:i don't know why")):
            with self.subTest(text=text):
                self.assertEqual(b_target_problem(text), reason)
        self.assertIsNone(b_target_problem("Typography of the label was clear."))

    def test_option_text_leak_and_open_item_letters(self):
        choices = ["minor second", "remains a minor second", "healthy", "red", "blue"]  # gold E: wrong options leak too
        self.assertEqual(hint_leak_single("The interval remains a minor second here.", "E", choices), "OPTION_TEXT:A")
        self.assertEqual(hint_leak_single("Tissue that looks healthy can still hide disease.", "E", choices),
                         "OPTION_TEXT:C")
        self.assertIsNone(hint_leak_single("Check the red channel of the stain.", "E", choices))  # 3 chars, one word
        self.assertIsNone(hint_leak("Tissue that looks healthy can still hide disease.", "E", choices))  # old rule kept
        # open item: "answer a ..." and "(a)" are not option letters, the remaining checks still run
        for text in ("Write the answer as a decimal with units.", "Use step (a) before step (b)."):
            with self.subTest(text=text):
                self.assertTrue(hint_leak(text, "42", []).startswith(("OPTION_LETTER", "PAREN_LETTER")))
                self.assertIsNone(hint_leak_single(text, "42", []))
        self.assertEqual(hint_leak_single("Write the answer as a decimal near 42.", "42", []), "GOLD_TEXT")
        self.assertEqual(hint_leak_single("The answer is in the table.", "42", []), "PHRASE:the answer is")
        self.assertEqual(hint_leak_single("Write the answer as sodium chloride.", "NaCl", [], aliases=["sodium chloride"]),
                         "SOURCE_ALIAS")

    def test_program_decision_skips_the_base_letter_check_on_open_items_only(self):
        def decision(base_leak, choices):
            audit = {"base_audit": {"format": {"result": "PASS"}, "leak_inputs": {"choices": choices},
                                    "checks": {"hint_leak": base_leak, "alias_leak": None}},
                     "single_hint_leak": None, "sha256": "0" * 64}
            return single.note_program_decision(audit, note_record("IMAGE", "INCORRECT"))
        lesson, choices = "Remember that x² terms dominate for large inputs.", ["x", "x²", "2x"]
        self.assertIsNone(hint_leak_single(lesson, "B", choices))  # NFKC lesson "x2" vs raw option "x²"
        self.assertEqual(decision(hint_leak(lesson, "B", choices), choices)["reason"], "NOTE_HINT_LEAK")
        self.assertTrue(decision("OPTION_LETTER:A", [])["accepted"])

    def test_protocol_records_scorer_reward_table_and_note_cap(self):
        from fine_tuning.error_note.contract import DENIAL_PHRASES, GRADER_READ_LINE, GRADER_READ_NONE
        from fine_tuning.error_note.single_grading import ALIASES, ALIASES_SHA256, NORMALIZATION_RULES

        pins = {key: "a" * 64 for key in ("source", "dataset", "order", "data_admission")}
        contract = plain_protocol_single("bf16", 2, contracts=pins)["training_contract"]
        self.assertEqual((VERSION_SINGLE, SINGLE_REWARD_GATE),
                         ("error-note-single-response-v10", "single-t1-program-gate-v5"))
        self.assertEqual(contract["rl"]["format_anchor"], {"positions": 4, "beta": .5})
        self.assertEqual(contract["rl"]["beta"], .02)
        self.assertEqual(contract["replay_support"], {
            "method": "stored-generation-support-v1", "block_mean_atol": .05, "row_mean_atol": .1,
            "row_mean_min_tokens": 64, "row_max_atol": 3., "row_failure": "exclude_row_from_rl_kl",
            "max_excluded_rows": 2})
        self.assertEqual(contract["answer_contracts"],
                         {"mcq": "p0-eval-aligned-v1", "open": "v8-extract+normalize+training-compare-v1"})
        scorer = contract["answer_scorer"]
        self.assertEqual((scorer["id"], scorer["rules"], scorer["aliases"], scorer["aliases_sha256"]),
                         ("v8-extract+normalize+training-compare-v1", list(NORMALIZATION_RULES), ALIASES, ALIASES_SHA256))
        self.assertEqual(ALIASES, {"hf-clevr": {"metallic": "metal", "matte": "rubber"}})
        filters = contract["program_filters"]
        self.assertEqual(filters["answer_denial"]["phrases"], list(DENIAL_PHRASES))
        self.assertEqual(filters["note_cause"]["reasons"], ["NOTE_CAUSE_NONE", "NOTE_CAUSE_MISMATCH"])
        self.assertEqual(filters["a2_review_references"]["match"], "word_boundary")
        self.assertIn("first max(4, len-2) letters", filters["lesson_stem"])
        self.assertNotIn("answer_union", contract)
        self.assertEqual(contract["rl"]["reward_values"],
                         {"accepted": 1, "BARE_WRONG": -1, "NO_COMMITTED_ANSWER": -1, "LOOP": -1, "other": 0})
        self.assertEqual(contract["note_max_tokens"], 1024)
        self.assertEqual(contract["max_tokens"], {"t1": 8192, "resolve": 8192})
        self.assertEqual(single_prompts_sha256(), _digest({
            "note": NOTE_INSTRUCTION_SINGLE, "note_feedback": SINGLE_VERDICT_LABELS, "resolve_header": RESOLVE_HEADER,
            "resolve_instruction": RESOLVE_INSTRUCTION_SINGLE,
            "note_grader_read": [GRADER_READ_LINE, GRADER_READ_NONE]}))


if __name__ == "__main__":
    unittest.main()
