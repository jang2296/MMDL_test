"""Synthetic CPU checks of model-proposed visual rechecks; no fixture is training data."""
from copy import deepcopy
import inspect
import json
import unittest
from unittest.mock import patch

from fine_tuning.error_note import v18_rows as old, v18b_rows as rows
from fine_tuning.error_note.audit import digest, text_sha256
from fine_tuning.tests.test_error_note_v17_rows import TOKENIZER, output, seal, verifier
from fine_tuning.tests.test_error_note_v18_rows import source_record
from fine_tuning.tests.test_error_note_v18b_rows import graph_item


LESSON = "Inspect the requested visual detail before calculating."


def observed(item=None, *, value="9", unit="length", private=None, historical=False):
    item = source_record() if item is None else item
    private = verifier(item) if private is None else private
    item["v17_private_verifier"] = private
    request = old.build_obs_request_v18(item) if historical else rows.build_obs_request_v18b(item)
    item["obs"] = output(json.dumps({"value": value, "unit": unit}), prompt=request["prompt"], status="OBS")
    if historical:
        item = rows.bind_historical_observation_v18b(item)
    return item, private


def check(item, private, *, decision="correct", read_now="7", quote="9", unit="length",
          repairs=None, gold="14", escaped=False):
    ledgers = [rows.audit_t1_v18b(item, verifier=private, tokenizer=TOKENIZER)]
    if item.get("obs"):
        ledgers.append(rows.audit_observation_v18b(item, verifier=private, tokenizer=TOKENIZER)["ledger"])
    request = rows.build_note_request_v18b(item, ledgers=ledgers, gold=gold, tokenizer=TOKENIZER)
    body = {"visual_check": {"read_now": read_now, "quote": quote, "unit": unit, "decision": decision},
            "lesson": LESSON}
    if repairs is not None:
        body["repairs"] = repairs
    text = json.dumps(body)
    if escaped:
        text = text.replace('"7"', r'"\u0037"')
    item["note"] = output(text, prompt=request["prompt"], status="NOTE")
    receipt = rows.verify_note_v18b(item, ledgers=ledgers, verifier=private, choices=[], gold=gold, tokenizer=TOKENIZER)
    return receipt, request, ledgers


def selected(item, receipt):
    row = rows.n_row_v18b(item, tokenizer=TOKENIZER, receipt=receipt)
    return None if row is None else bytes(token for token, keep in zip(
        row["generation"]["completion_token_ids"], row["loss_mask"]) if keep).decode()


class VisualCheck(unittest.TestCase):
    def test_obs_is_only_a_visual_question_with_exact_typed_response_shape(self):
        item = source_record()
        item.update(layer="scene", question="What shape is the cyan object?", source="private-source-name")
        item["evaluation"] = {"gold": "secret-answer"}
        private = verifier(item, [{"kind": "OBJECT_ATTRIBUTE", "entity": [1, "material"], "value": "metal"}],
                           supported_claim_kinds=["OBJECT_ATTRIBUTE"])
        item["v17_private_verifier"] = private
        request = rows.build_obs_request_v18b(item)
        self.assertEqual(request["public_query"], {"target": "material of the first visible object from the left",
                                                 "unit": "", "value_type": "string"})
        for forbidden in (item["question"], item["source"], "secret-answer", '"kind"', '"target"', '"schema"'):
            self.assertNotIn(forbidden, request["prompt"])
        self.assertIn('"unit": ""', request["prompt"])
        graph, _ = graph_item(complete=True)
        prompt = rows.build_obs_request_v18b(graph)["prompt"]
        self.assertIn('"value": <true or false or null>', prompt)
        self.assertIn("bare JSON true or false", prompt)

    def test_note_separates_t1_and_historical_obs_without_private_values(self):
        item, private = observed(historical=True)
        with patch.object(rows, "render_note_target_last", wraps=rows.render_note_target_last) as rendered:
            receipt, request, ledgers = check(item, private)
        self.assertTrue(receipt["n_accepted"], receipt)
        body = rendered.call_args.args[0]
        self.assertEqual(request["public_context_sha256"], digest(body))
        self.assertEqual(request["prompt"], rows.render_note_target_last(body))
        self.assertEqual(body["original_t1"]["question"], item["p0_text"])
        self.assertEqual(body["original_t1"]["answer_grading_result"], "INCORRECT")
        self.assertEqual(body["observation_question"], {"target": "directly printed length label on segment AB", "unit": "length"})
        self.assertEqual(body["observation_prior_value"], "9")
        self.assertEqual(body["observation"], item["obs"]["text"])
        self.assertNotIn("given_answer", body["observation_question"])
        self.assertNotIn("verdict", request["prompt"])
        self.assertNotIn('"repairs": []', request["prompt"])
        if request["note_condition"] == "self_found":
            changed = deepcopy(ledgers)
            changed[-1]["claims"][0]["verdict"] = "VERIFIED"
            changed[-1] = seal({key: value for key, value in changed[-1].items() if key != "sha256"})
            other = rows.build_note_request_v18b(item, ledgers=changed, gold="14", tokenizer=TOKENIZER)
            self.assertEqual(request["prompt"], other["prompt"])

    def test_correct_recheck_trains_only_the_actual_read_now_bytes(self):
        for escaped in (False, True):
            item, private = observed()
            receipt, _, _ = check(item, private, escaped=escaped)
            self.assertTrue(receipt["n_accepted"], receipt)
            self.assertEqual(selected(item, receipt), r"\u0037" if escaped else "7")
            self.assertEqual(receipt["visual_check"], {"model_decision": "correct", "independent_acceptance": True,
                                                     "independent_status": "ACCEPTED", "reason": None})
            self.assertEqual(item["note"]["text"], item["note"]["generation"]["text"])

    def test_bare_boolean_can_share_one_sampled_token_with_json_whitespace(self):
        class MergedBoolean(type(TOKENIZER)):
            def convert_ids_to_tokens(self, values):
                return ["Ġfalse" if value == 501 else super(MergedBoolean, self).convert_ids_to_tokens([value])[0]
                        for value in values]

            def decode(self, values, **kwargs):
                return " false" if values == [501] else ""

        raw = '{"read_now": false}'
        begin = raw.index("false")
        actions = list(raw[:begin - 1].encode()) + [501] + list(raw[begin + 5:].encode()) + [256]
        output = {"text": raw, "generation": {"completion_token_ids": actions}}
        self.assertEqual(rows.boolean_note_token_span(output, MergedBoolean(), [begin, begin + 5]),
                         [begin - 1, begin + 5])

    def test_true_boolean_and_material_revisions_are_independently_verified(self):
        item, private = graph_item(complete=True)
        item["v18b_obs_class"] = False
        item, private = observed(item, value=True, unit="", private=private)
        for reading in (False, "false"):
            receipt, _, _ = check(item, private, quote="true", read_now=reading, unit="", gold="A")
            self.assertTrue(receipt["n_accepted"], receipt)
            self.assertEqual(selected(item, receipt), "false")
            self.assertEqual(receipt["visual_check"]["independent_status"], "ACCEPTED")
        for reading, accepted in (("metal", True), ("rubber", False), ("sphere", False)):
            item = source_record("Final answer: sphere")
            item.update(layer="scene", question="What shape is the cyan object?")
            private = verifier(item, [{"kind": "OBJECT_ATTRIBUTE", "entity": [1, "material"], "value": "metal"}],
                               supported_claim_kinds=["OBJECT_ATTRIBUTE"])
            item, private = observed(item, value="rubber", unit="", private=private)
            receipt, _, _ = check(item, private, quote="rubber", read_now=reading, unit="", gold="cylinder")
            self.assertEqual(receipt["n_accepted"], accepted, receipt)

    def test_duplicate_same_fact_and_non_string_or_malformed_visual_fields_are_rejected(self):
        item, private = observed()
        proposal = {"stage": "T1", "quote": "AB = 9", "corrected_value": "7", "unit": "length"}
        receipt, _, _ = check(item, private, repairs=[proposal])
        self.assertIn("DUPLICATE_CLAIM_REPAIR", receipt["reasons"])
        self.assertIsNone(selected(item, receipt))
        for field, value in (("read_now", True), ("quote", 9), ("decision", "revise"), ("unit", None)):
            with self.subTest(field=field):
                item, private = observed()
                _, _, ledgers = check(item, private)
                obj = json.loads(item["note"]["text"])
                obj["visual_check"][field] = value
                item["note"] = output(json.dumps(obj), prompt=item["note"]["prompt"], status="NOTE")
                receipt = rows.verify_note_v18b(item, ledgers=ledgers, verifier=private,
                                               choices=[], gold="14", tokenizer=TOKENIZER)
                self.assertIn("NOTE_SCHEMA:VISUAL_CHECK", receipt["reasons"])
                self.assertFalse(receipt["resolve_allowed"], receipt)
                self.assertIsNone(selected(item, receipt))

    def test_keep_and_unsure_are_valid_lessons_without_obs_training(self):
        for decision, value in (("keep", "9"), ("unsure", None)):
            item, private = observed()
            receipt, _, _ = check(item, private, decision=decision, read_now=value)
            self.assertFalse(receipt["n_accepted"], receipt)
            self.assertIsNone(selected(item, receipt))
            self.assertTrue(receipt["resolve_allowed"], receipt)
            self.assertEqual(receipt["visual_check"]["model_decision"], decision)
            self.assertEqual(receipt["visual_check"]["independent_status"], "NOT_PROPOSED")

    def test_wrong_value_quote_unit_gold_and_no_value_never_supply_n(self):
        cases = [({"read_now": "8"}, "INCORRECT_SOURCE_REPAIR"),
                 ({"read_now": "9"}, "INCORRECT_SOURCE_REPAIR"),
                 ({"quote": "8"}, "ORIGINAL_QUOTE_NOT_UNIQUE_BOUND_ASSERTION"),
                 ({"unit": "cm"}, "REPAIR_UNIT_MISMATCH"),
                 ({"read_now": "14"}, "FINAL_ANSWER_COPY_NOT_INTERMEDIATE_REPAIR"),
                 ({"read_now": None}, "VISUAL_CHECK_NO_CORRECTED_VALUE")]
        for kwargs, reason in cases:
            with self.subTest(kwargs=kwargs):
                item, private = observed()
                receipt, _, _ = check(item, private, **kwargs)
                self.assertFalse(receipt["n_accepted"], receipt)
                self.assertIn(reason, receipt["reasons"])
                self.assertFalse(receipt["visual_check"]["independent_acceptance"])

    def test_verified_or_unknown_prior_cannot_be_made_into_a_correction(self):
        item, private = observed(value="7")
        receipt, _, _ = check(item, private, quote="7")
        self.assertIn("ORIGINAL_NOT_CONTRADICTED_ASSERTION", receipt["reasons"])
        self.assertIsNone(selected(item, receipt))
        item, private = graph_item(complete=True)
        item["t1"]["status"] = "INCORRECT"
        item, private = observed(item, value=1, unit="", private=private)
        receipt, _, _ = check(item, private, read_now="true", quote="1", unit="", gold="A")
        self.assertIn("ORIGINAL_NOT_CONTRADICTED_ASSERTION", receipt["reasons"])
        self.assertIsNone(selected(item, receipt))

    def test_duplicate_obs_proposals_and_changed_input_are_rejected(self):
        item, private = observed()
        proposal = {"stage": "OBS", "quote": "9", "corrected_value": "7", "unit": "length"}
        receipt, _, _ = check(item, private, repairs=[proposal])
        self.assertIn("DUPLICATE_OBS_REPAIR", receipt["reasons"])
        self.assertIsNone(selected(item, receipt))
        receipt, _, ledgers = check(item, private)
        item["note"]["prompt"] += " altered"
        receipt = rows.verify_note_v18b(item, ledgers=ledgers, verifier=private, choices=[], gold="14", tokenizer=TOKENIZER)
        self.assertIn("NOTE_INPUT_CHANGED", receipt["reasons"])
        self.assertIsNone(selected(item, receipt))

    def test_mixed_t1_and_visual_repairs_remain_independent(self):
        item = source_record("CD = 8\nFinal answer: B")
        private = verifier(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "7"},
                                  {"kind": "SEGMENT_LABEL", "entity": ["C", "D"], "value": "6"}])
        proposal = {"stage": "T1", "quote": "CD = 8", "corrected_value": "6", "unit": "length"}
        for decision, value, expected in (("keep", "9", "6"), ("unsure", None, "6"), ("correct", "7", "76")):
            candidate, private = observed(deepcopy(item), private=private)
            receipt, _, _ = check(candidate, private, decision=decision, read_now=value, repairs=[proposal])
            self.assertTrue(receipt["n_accepted"], receipt)
            self.assertEqual(selected(candidate, receipt), expected)

    def test_note_prompt_fingerprint_covers_both_actual_obs_prompts(self):
        before = rows.prompts_sha256()
        with patch.object(rows, "OBS_INSTRUCTION_V18B", rows.OBS_INSTRUCTION_V18B + " changed"):
            self.assertNotEqual(rows.prompts_sha256(), before)
        def changed_renderer(_body):
            return "changed renderer source"

        with patch.object(rows, "render_note_target_last", changed_renderer):
            self.assertNotEqual(rows.prompts_sha256(), before)

    def test_renderer_is_the_fixed_target_last_candidate_without_placeholder_examples(self):
        self.assertEqual(text_sha256(inspect.getsource(rows.render_note_target_last)),
                         "73c2c1edf7112d86e80bb0afc78a69a150e73029157f7f33404c928709907ef7")
        item, private = observed()
        with patch.object(rows, "render_note_target_last", wraps=rows.render_note_target_last) as rendered:
            _, request, _ = check(item, private)
        body = rendered.call_args.args[0]
        prompt = request["prompt"]
        target = body["observation_question"]["target"]
        self.assertEqual(request["public_context_sha256"], digest(body))
        self.assertTrue(prompt.endswith("Visual question: " + target))
        self.assertLess(prompt.index("Original T1 background"), prompt.index("Your original OBS response:"))
        self.assertLess(prompt.index("Your prior OBS scalar:"), prompt.rindex("Visual question:"))
        for placeholder in ("<old scalar>", "<value or null>", "<abstract lesson>", "keep|correct|unsure"):
            self.assertNotIn(placeholder, prompt)
        retained = {key: value for key, value in body.items() if key not in
                    {"original_t1", "observation", "observation_prior_value", "observation_question"}}
        self.assertIn(json.dumps(retained, sort_keys=True, ensure_ascii=False), prompt)

    def test_target_last_prompt_tamper_cannot_admit_corrected_content(self):
        item, private = observed()
        _, request, ledgers = check(item, private)
        item["note"]["prompt"] = request["prompt"].replace("Visual question:", "Changed visual question:")
        receipt = rows.verify_note_v18b(item, ledgers=ledgers, verifier=private,
                                       choices=[], gold="14", tokenizer=TOKENIZER)
        self.assertIn("NOTE_INPUT_CHANGED", receipt["reasons"])
        self.assertFalse(receipt["resolve_allowed"], receipt)
        self.assertIsNone(selected(item, receipt))

    def test_explicit_corrected_scalar_provenance_excludes_leaked_lesson(self):
        item, private = graph_item(complete=True)
        item["v18b_obs_class"] = False
        item, private = observed(item, value=True, unit="", private=private)
        _, request, ledgers = check(item, private, quote="true", read_now=False, unit="", gold="A")
        text = json.dumps({"visual_check": {"read_now": False, "quote": "true", "unit": "", "decision": "correct"},
                           "lesson": "The corrected value is false for node 0."})
        item["note"] = output(text, prompt=request["prompt"], status="NOTE")
        receipt = rows.verify_note_v18b(item, ledgers=ledgers, verifier=private,
                                       choices=[], gold="A", tokenizer=TOKENIZER)
        self.assertTrue(receipt["n_accepted"], receipt)
        self.assertEqual(receipt["repairs"][0]["proposal_json_path"], "visual_check.read_now")
        self.assertEqual(selected(item, receipt), "false")
        self.assertFalse(receipt["lesson_accepted"], receipt)
        self.assertFalse(receipt["resolve_allowed"], receipt)
        self.assertIn("PROBLEM_NUMBER", receipt["lesson_leak_reasons"])
        self.assertIn("CORRECTED_VALUE", receipt["lesson_leak_reasons"])


if __name__ == "__main__":
    unittest.main()
