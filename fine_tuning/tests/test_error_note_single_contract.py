"""Regression counterexamples for the opt-in single-response lesson contract."""
import copy
import unittest

from fine_tuning.error_note import single
from fine_tuning.error_note.contract import (NOTE_INSTRUCTION_V2, NOTE_INSTRUCTION_V3, SINGLE_VERDICT_LABELS, VERDICT_LABELS,
                                            hint_leak, hint_leak_single, note_prompt_single, note_prompt_v3)
from fine_tuning.error_note.protocol import (SINGLE_REWARD_GATE, VERSION_SINGLE, _digest,
                                            plain_protocol_single, single_blocks, single_prompts_sha256)


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
        protocol = plain_protocol_single("bf16", 219, contracts=pins, token_scale=1024)
        contract = protocol["training_contract"]
        self.assertEqual(contract["generation_counts"], {"t1": 1, "note": 1, "resolve": 1})
        self.assertEqual(contract["rl"]["baseline"], 0)
        self.assertEqual(protocol["optimizer"]["total_steps"], 219)
        self.assertEqual(contract["rl"]["token_scale"], 1024)
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
        for scale in (0, -1, float("nan"), True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                plain_protocol_single("bf16", 219, contracts=pins, token_scale=scale)
        with self.assertRaises(ValueError):
            plain_protocol_single("bf16", 219, contracts={}, token_scale=1024)

    def test_neutral_single_parser_feedback_preserves_original_response_gold_and_legacy_v3(self):
        p0, raw, gold = "Question: exact P0", "Original prose\nGrading result: NO CLEAR FINAL ANSWER\nFinal answer: 50.2%", "[48, 52]%"
        for status in ("INVALID_NO_FINAL", "INVALID_MULTIPLE"):
            old, new = note_prompt_v3(p0, raw, status, gold), note_prompt_single(p0, raw, status, gold)
            prefix = f"{p0}\n\nYour earlier solution:\n{raw}\n\nGrading result: "
            self.assertEqual(old, prefix + "NO CLEAR FINAL ANSWER\n" + f"Correct answer: {gold}\n\n{NOTE_INSTRUCTION_V3}")
            self.assertEqual(new, prefix + SINGLE_VERDICT_LABELS[status] + "\n" + f"Correct answer: {gold}\n\n{NOTE_INSTRUCTION_V2}")
            self.assertIn("This does not establish that no final answer was written; inspect the original response.", new)
        for status in ("CORRECT", "INCORRECT", "TRUNCATED"):  # v5: the v3 feedback with the V2 instruction
            self.assertEqual(note_prompt_single(p0, raw, status, gold),
                             note_prompt_v3(p0, raw, status, gold).removesuffix(NOTE_INSTRUCTION_V3) + NOTE_INSTRUCTION_V2)
        self.assertEqual(VERDICT_LABELS["INVALID_NO_FINAL"], "NO CLEAR FINAL ANSWER")


if __name__ == "__main__":
    unittest.main()
