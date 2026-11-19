import unittest
from copy import deepcopy

from fine_tuning.error_note import v18_rows as v18
from fine_tuning.error_note import v18b_rows as rows
from fine_tuning.tests.test_error_note_v17_rows import TOKENIZER, output, record


QUESTION = ("What shape is the small shiny object that is on the right side of the large rubber object "
            "that is behind the small blue shiny cube?")
DECLARATION = "Therefore, the shape of the small shiny object is a **cylinder**."


def clevr_record(text=DECLARATION, question=QUESTION):
    item = record()
    item.update(source="hf-clevr", source_revision="f" * 40, native_id="train:fixture",
                question=question, p0_text="Question: " + question)
    item["t1"] = output(text, prompt=item["p0_text"], status="INCORRECT", extracted=text)
    item["t1"]["v8_status"] = "INCORRECT"
    return item


class ClevrFormat(unittest.TestCase):
    def test_003539_declaration_is_opt_in_and_parser_grade_stays_incorrect(self):
        item = clevr_record()
        original = deepcopy(item)
        self.assertFalse(v18.format_receipt_v18(item, gold="cylinder")["accepted"])
        form = v18.format_receipt_v18(item, gold="cylinder", _v18b=True)
        self.assertTrue(form["accepted"], form)
        self.assertEqual(form["raw_status"], "INCORRECT")
        self.assertEqual(form["training_grade"], "INCORRECT")
        self.assertEqual(form["semantic_status"], "CORRECT")
        self.assertEqual(form["semantic_certainty"], "SOURCE_FINAL_DECLARATION_ONLY")
        self.assertEqual(form["certification_scope"], "SOURCE_FINAL_SHAPE_ONLY")
        self.assertEqual(item, original)
        other = v18.format_receipt_v18(item, gold="sphere", _v18b=True)
        self.assertEqual(other["extracted_without_gold"], "cylinder")
        self.assertEqual(other["semantic_status"], "INCORRECT")
        self.assertFalse(other["accepted"])

    def test_policy_masks_only_format_and_sft_requires_actual_correction(self):
        item = clevr_record("Read the image carefully.\n" + DECLARATION)
        policy = rows.t1_policy_receipt_v18b(item, tokenizer=TOKENIZER, gold="cylinder")
        selected = bytes(token for token, keep in zip(item["t1"]["generation"]["completion_token_ids"],
                         policy["loss_token_mask"]) if keep).decode()
        self.assertEqual(selected, DECLARATION.split("cylinder")[0])
        self.assertEqual(policy["training_status"], "FORMAT_ERROR")
        self.assertEqual(policy["raw_status"], "INCORRECT")
        self.assertEqual(policy["reason"], "FORMAT_ONLY_CERTIFIED")
        self.assertEqual(policy["reward"], 0)
        candidate = rows.verify_solution_v18b(item, tokenizer=TOKENIZER, gold="cylinder")
        self.assertFalse(candidate["accepted"])
        self.assertIn("FORMAT_CORRECTION_REQUIRED", candidate["reasons"])

    def test_ambiguous_negated_other_referent_and_distractors_are_not_certified(self):
        for text in (
            "The shape of some other object is a cylinder.",
            "Therefore, the shape of the large rubber object is a cylinder.",
            "Therefore, the shape of the small shiny object is not a cylinder.",
            "Therefore, the shape of the small shiny object might be a cylinder.",
            "Therefore, the shape of the small shiny object is a cylinder or sphere.",
            "Therefore, the shape of the small shiny object is a cylinder and not a cube.",
            "Therefore, the shape of the small shiny object is not a cube but a cylinder.",
            '"Therefore, the shape of the small shiny object is a cylinder."',
            "A cylinder is also visible.",
            "Final answer: sphere\n" + DECLARATION,
            "The shape of the small shiny object is a sphere.\n" + DECLARATION,
        ):
            with self.subTest(text=text):
                form = v18.format_receipt_v18(clevr_record(text), gold="cylinder", _v18b=True)
                self.assertFalse(form["accepted"], form)
                self.assertEqual(form["semantic_status"], "UNKNOWN")

    def test_missing_context_wrong_scope_choices_or_truncation_are_not_certified(self):
        for key in ("source_revision", "native_id", "p0_text"):
            with self.subTest(key=key):
                item = clevr_record()
                del item[key]
                self.assertFalse(v18.format_receipt_v18(item, gold="cylinder", _v18b=True)["accepted"])
        for question in ("What material is the small shiny object?",
                         "What shape is the small shiny object or the large object?",
                         "What shape is the small shiny object that is behind the small shiny object?", ""):
            self.assertFalse(v18.format_receipt_v18(clevr_record(question=question),
                                                   gold="cylinder", _v18b=True)["accepted"])
        item = clevr_record()
        item["t1"]["prompt"] = "Question: a different object?"
        self.assertFalse(v18.format_receipt_v18(item, gold="cylinder", _v18b=True)["accepted"])
        item = clevr_record()
        item["t1"]["finish_reason"] = "length"
        self.assertFalse(v18.format_receipt_v18(item, gold="cylinder", _v18b=True)["accepted"])
        self.assertFalse(v18.format_receipt_v18(clevr_record(), gold="cylinder", choices=("cube", "cylinder"),
                                               _v18b=True)["accepted"])


if __name__ == "__main__":
    unittest.main()
