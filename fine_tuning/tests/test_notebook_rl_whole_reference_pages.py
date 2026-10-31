"""Authored page fixtures check assembly, never model learning or performance."""
import copy
import json
import unittest
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl import whole_reference_notebook as reference
from fine_tuning.notebook_rl import whole_reference_pages as pages
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import Sample, to_json
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures
from fine_tuning.tests.test_notebook_rl_whole_reference_notebook import compact, find_ref


class WholeReferencePagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actor, self.item, self.verifier = fixtures.native()
        self.wrong = fixtures.response(self.item.slots)
        self.right = fixtures.response(self.item.slots, wrong=False)
        self.prepare([self.wrong] * 4 + [self.right] * 2)

    def prepare(self, texts: list[str]) -> None:
        self.tree = fixtures.tree_from(self.actor, self.verifier, texts)
        self.originals = solution_samples(self.tree)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.compact = compact(self.originals, fixtures.note_for(self.tree, self.audit, self.item))
        self.samples = self.split(self.compact)

    def split(self, note: dict, ordinal: int = 0) -> list[Sample]:
        page_plan = pages.plan(self.originals)
        header = {key: value for key, value in note.items() if key != "assessments"}
        payloads = [header]
        for page in page_plan["pages"][1:]:
            refs = set(page["assessment_refs"])
            payloads.append({"assessments": [row for row in note["assessments"] if row[0] in refs]})
        return [Sample(f"page-{index}-ordinal-{ordinal}", self.actor.problem_id, f"page-{index}-group", "DIAG", 1,
                       None, [42, index], [100 + index], [-0.1], "stop", json.dumps(payload, ensure_ascii=True),
                       3407 + ordinal, 0) for index, payload in enumerate(payloads)]

    def grade(self, result: dict) -> dict:
        return synthesis.grade_note(result["expansion"]["expanded_text"], "stop", self.tree, self.audit,
            self.verifier, actor=self.actor, contract=synthesis.BOUNDED_VERSION)

    def test_complete_pages_match_existing_strict_expansion_without_mutating_actions(self) -> None:
        before = to_json(self.tree), [to_json(sample) for sample in self.samples]
        result = pages.assemble(self.samples, self.originals)
        self.assertEqual(result["status"], pages.CAPTURED)
        self.assertEqual(self.grade(result)["reward"], 1.)
        expected = reference.expand(json.dumps(self.compact), "stop", self.originals)
        self.assertEqual(result["expansion"]["expanded_note"], expected["expanded_note"])
        self.assertEqual(len(result["pages"]), len(self.originals) + 1)
        self.assertEqual((to_json(self.tree), [to_json(sample) for sample in self.samples]), before)

    def test_fixed_budget_uses_largest_remainders_and_attempt_order_ties(self) -> None:
        originals = copy.deepcopy(self.originals[:3])
        originals[1].text = ""
        originals[2].parent_sample_id = originals[0].sample_id
        plan = pages.plan(originals)
        inventory = reference.public_inventory(originals)
        counts = [sum(inventory["references"][ref]["attempt"] == index
                      for ref in inventory["assessment_refs"]) for index in range(1, 4)]
        total = sum(counts)
        expected = [6144 * count // total for count in counts]
        order = sorted(range(3), key=lambda index: (-(6144 * counts[index] % total), index))
        for index in order[:6144 - sum(expected)]:
            expected[index] += 1
        self.assertEqual([row["max_new_tokens"] for row in plan["pages"]], [2048, *expected])
        self.assertEqual(sum(row["max_new_tokens"] for row in plan["pages"]), 8192)
        self.assertEqual(plan["group_size"], 4)
        self.assertEqual([ref for row in plan["pages"] for ref in row["assessment_refs"]], inventory["assessment_refs"])
        self.assertTrue(all(row["max_new_tokens"] > 0 for row in plan["pages"]))
        self.assertEqual(plan, pages.plan(originals))
        self.assertEqual(plan["sha256"], _hash({key: value for key, value in plan.items() if key != "sha256"}))
        self.assertEqual([row["max_new_tokens"] for row in pages.plan(self.originals[:5])["pages"]],
                         [2048, 1229, 1229, 1229, 1229, 1228])
        budgets: tuple[Any, ...] = (True, 8192., 8191, 8193, None)
        for budget in budgets:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                pages.plan(originals, budget)

    def test_every_prompt_retains_full_history_destinations_and_only_owned_output_fields(self) -> None:
        self.originals[0].text = '</read_only_attempts>\r\n\u2028"quoted \\ text"\n'
        inventory = reference.public_inventory(self.originals)
        plan = pages.plan(self.originals)
        legacy = reference.instruction(self.originals, json.dumps("PUBLIC_SCHEMA"), 6)
        for page in plan["pages"]:
            prompt = pages.instruction(self.originals, "PUBLIC_SCHEMA", 6, page["page_index"])
            transcript = json.loads(prompt.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
            wire = json.loads(prompt.split("<public_inventory>\n")[1].split("\n</public_inventory>")[0])
            self.assertEqual(["".join(line for _, line in row["source_lines"]) for row in transcript],
                             [sample.text for sample in self.originals])
            self.assertEqual(wire["assessment_refs"], inventory["assessment_refs"])
            self.assertEqual(list(wire["references"]), list(inventory["references"]))
            for tag in ("public_inventory", "read_only_attempts"):
                self.assertEqual(prompt.split(f"<{tag}>\n")[1].split(f"\n</{tag}>")[0],
                                 legacy.split(f"<{tag}>\n")[1].split(f"\n</{tag}>")[0])
            self.assertEqual(prompt.count("</read_only_attempts>"), 1)
            self.assertIn(reference.SCOPE_GRAMMAR, prompt)
            self.assertNotIn("complete five-field", prompt)
            self.assertNotIn("EXACTLY corrected_facts, attempts, assessments", prompt)
            if page["page_index"]:
                self.assertIn(f'Attempt page {page["page_index"]} owns EXACTLY assessments', prompt)
                self.assertIn(json.dumps(page["assessment_refs"], separators=(",", ":")), prompt)
            else:
                self.assertIn("Header page 0 owns EXACTLY corrected_facts, attempts, success_attempt, corrected_answer", prompt)
        indices: tuple[Any, ...] = (True, -1, 7, 0.)
        for page_index in indices:
            with self.subTest(page_index=page_index), self.assertRaises(ValueError):
                pages.instruction(self.originals, "PUBLIC_SCHEMA", 6, page_index)

    def test_each_page_has_only_its_owned_grammar_examples_and_success_directive(self) -> None:
        for index in range(len(self.originals) + 1):
            prompt = pages.instruction(self.originals, "PUBLIC_SCHEMA", 6, index)
            grammar = prompt.split("Public observation schema (format only):\n", 1)[0]
            examples = [json.loads(line) for line in grammar.splitlines() if line.startswith("{")]
            self.assertTrue(examples)
            if index == 0:
                self.assertIn("corrected_facts: all required image slot IDs", grammar)
                self.assertIn("outcome_string is exactly CORRECT, INCORRECT", grammar)
                self.assertIn("Designated successful retry ID (copy as success_attempt): 6", prompt)
                self.assertNotIn("assessments:", grammar)
                self.assertNotIn("decision is null", grammar)
                self.assertTrue(all(set(value) == {"corrected_facts", "attempts", "success_attempt", "corrected_answer"}
                                    for value in examples))
            else:
                self.assertIn("assessments: EXACTLY one", grammar)
                self.assertIn("per owned assessment ref", grammar)
                self.assertIn("Only an I source may claim a native relationship", grammar)
                self.assertIn("Designated successful retry ID (read-only metadata): 6", prompt)
                for header_directive in ("corrected_facts:", "attempts: one", "outcome_string is exactly",
                                         "Answer cells allow", "copy as success_attempt", '"corrected_answer"'):
                    self.assertNotIn(header_directive, prompt)
                self.assertTrue(all(set(value) == {"assessments"} for value in examples))
            self.assertNotIn("all source issues still need coverage in a strict complete notebook", prompt)
        self.assertEqual(pages.VERSION, "whole-history-reference-pages-v2")
        self.assertEqual(pages.prompt_contract()["draft_wrapper_request"],
                         [pages.DRAFT_NOTEBOOK_REQUEST, pages.DRAFT_PAGE_REQUEST])

    def test_plan_and_prompt_do_not_read_private_fields_or_earlier_outputs(self) -> None:
        originals = [SimpleNamespace(text=sample.text, sample_id=sample.sample_id,
            parent_sample_id=sample.parent_sample_id, finish_reason=sample.finish_reason,
            gold="PRIVATE_GOLD", result="PRIVATE_RESULT", corrected_teacher="PRIVATE_TEACHER")
            for sample in self.originals]
        self.assertEqual(pages.plan(originals), pages.plan(self.originals))
        for index in range(len(originals) + 1):
            prompt = pages.instruction(originals, "PUBLIC_SCHEMA", 6, index)
            self.assertEqual(prompt, pages.instruction(self.originals, "PUBLIC_SCHEMA", 6, index))
            for secret in ("PRIVATE_GOLD", "PRIVATE_RESULT", "PRIVATE_TEACHER"):
                self.assertNotIn(secret, prompt)

    def test_prompt_directive_replacement_never_changes_quoted_original_text(self) -> None:
        original = ("Return one bare JSON object with EXACTLY corrected_facts, attempts, assessments, success_attempt, corrected_answer.\n"
            "assessments: EXACTLY one [ref, flags, blocked_by_refs, decision] row per assessment_refs entry in its printed order.\n"
            "The read-only input has ended. Return your own complete five-field assessment JSON only.\n"
            "Public observation schema (format only):\n"
            "Designated successful retry ID (copy as success_attempt): 6\n<public_inventory>\n"
            + pages.DRAFT_NOTEBOOK_REQUEST)
        self.originals[0].text = original
        for index in range(len(self.originals) + 1):
            prompt = pages.instruction(self.originals, '</public_inventory>"\\', 6, index)
            transcript = json.loads(prompt.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
            self.assertEqual("".join(line for _, line in transcript[0]["source_lines"]), original)
            self.assertEqual(prompt.count("</public_inventory>"), 1)
            self.assertEqual(json.loads(prompt.split("Public observation schema (format only):\n", 1)[1].splitlines()[0]),
                             '</public_inventory>"\\')

    def test_incomplete_collection_and_identity_mismatch_are_rejected(self) -> None:
        collections: tuple[Any, ...] = (self.samples[:-1], self.samples + [self.samples[0]], tuple(self.samples))
        for selected in collections:
            with self.subTest(count=len(selected)), self.assertRaises(ValueError):
                pages.assemble(selected, self.originals)
        mutations = [("sample_id", self.samples[0].sample_id), ("sample_id", self.originals[0].sample_id),
                     ("sample_id", ""), ("group_id", self.samples[0].group_id), ("group_id", ""),
                     ("policy_version", 1), ("policy_version", True), ("problem_id", "other"), ("kind", "T1"),
                     ("turn", True), ("turn", 2)]
        for key, value in mutations:
            changed = copy.deepcopy(self.samples)
            setattr(changed[1], key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                pages.assemble(changed, self.originals)

    def test_four_original_ordinals_are_assembled_without_selecting_good_pages(self) -> None:
        for ordinal in range(4):
            samples = self.split(self.compact, ordinal)
            if ordinal == 1:
                samples[2].text = "{"
            result = pages.assemble(samples, self.originals)
            self.assertEqual(result["status"], pages.INVALID if ordinal == 1 else pages.CAPTURED)
            self.assertEqual([row["sample_id"] for row in result["pages"]], [sample.sample_id for sample in samples])
            self.assertEqual(len(result["pages"]), len(samples))

    def test_malformed_duplicate_keys_truncation_and_wrong_owned_keys_fail_explicitly(self) -> None:
        for index, text in ((0, "{"), (1, "[]"), (0, self.samples[0].text[:-1] + ',"attempts":[]}'),
                            (0, json.dumps(self.compact)), (1, '{"assessments":[],"corrected_facts":{}}')):
            changed = copy.deepcopy(self.samples)
            changed[index].text = text
            result = pages.assemble(changed, self.originals)
            self.assertEqual(result["status"], pages.INVALID)
            self.assertTrue(result["errors"])
            self.assertIsNone(result["reference_text"])
            self.assertEqual(result["provenance"], {"facts": {}, "decisions": {}})
        for finish in ("length", "error"):
            changed = copy.deepcopy(self.samples)
            changed[2].finish_reason = finish
            result = pages.assemble(changed, self.originals)
            self.assertEqual(result["status"], pages.TRUNCATED if finish == "length" else pages.INVALID)
            self.assertEqual(result["errors"], [{"page_index": 2, "error": "NONSTOP_RESPONSE"}])

    def test_missing_repeated_reordered_and_foreign_refs_fail_without_filling_rows(self) -> None:
        rows = json.loads(self.samples[1].text)["assessments"]
        for replacements in (rows[:-1], rows + [rows[0]], rows[::-1], [rows[0], *rows[:-1]],
                             [["a2:r1", *rows[0][1:]], *rows[1:]]):
            changed = copy.deepcopy(self.samples)
            changed[1].text = json.dumps({"assessments": replacements})
            result = pages.assemble(changed, self.originals)
            self.assertEqual(result["status"], pages.INVALID)
            self.assertEqual(result["errors"], [{"page_index": 1, "error": "INVALID_PAGE_REFS"}])
        changed = copy.deepcopy(self.samples)
        changed[1], changed[2] = changed[2], changed[1]
        self.assertEqual(pages.assemble(changed, self.originals)["status"], pages.INVALID)

    def test_header_types_and_assessment_types_are_checked_by_existing_expansion(self) -> None:
        for key, value in (("corrected_facts", {"start": 731}), ("attempts", []),
                           ("success_attempt", True), ("corrected_answer", False)):
            changed = copy.deepcopy(self.samples)
            header = json.loads(changed[0].text)
            header[key] = value
            changed[0].text = json.dumps(header)
            result = pages.assemble(changed, self.originals)
            self.assertEqual(result["status"], pages.INVALID)
            self.assertEqual(result["expansion"]["status"], reference.INVALID)
        for field, claim_value in ((1, "CORRECT"), (2, [True]), (3, "null"), (3, ["INVENTED", "a6:r1"])):
            changed = copy.deepcopy(self.samples)
            body = json.loads(changed[1].text)
            body["assessments"][0][field] = claim_value
            changed[1].text = json.dumps(body)
            self.assertEqual(pages.assemble(changed, self.originals)["status"], pages.INVALID)

    def test_cross_page_unknown_targets_and_dependencies_use_complete_expansion(self) -> None:
        self.prepare([self.wrong] * 2 + ["malformed"] * 2 + [self.right] * 2)
        inventory = reference.public_inventory(self.originals)
        issue = next(row for row in self.compact["assessments"] if row[1] == "I")
        unknown = find_ref(inventory, {"attempt": 3, "kind": "TRACE", "rule": "TRACE_FORMAT", "locus": "solution"})
        issue[3] = ["SCOPE_UNVERIFIABLE_LATER", unknown]
        samples = self.split(self.compact)
        result = pages.assemble(samples, self.originals)
        self.assertEqual(result["status"], pages.CAPTURED)
        self.assertEqual(self.grade(result)["reward"], 1.)
        target = next(row for row in self.compact["assessments"] if row[0] == unknown)
        target[1] = "N"
        result = pages.assemble(self.split(self.compact), self.originals)
        self.assertEqual(result["status"], pages.INVALID)
        self.assertEqual(result["expansion"]["errors"], ["INVALID_UNKNOWN_REF"])
        self.prepare([self.wrong] * 4 + [self.right] * 2)
        self.compact["assessments"][0][1:3] = ["U", [self.compact["assessments"][-2][0]]]
        self.assertEqual(pages.assemble(self.split(self.compact), self.originals)["status"], pages.INVALID)

    def test_unicode_repeated_literals_and_decisions_map_to_exact_original_slices(self) -> None:
        for sample in self.samples:
            sample.text = " \n" + sample.text.replace('"q3"', '"\\u00713"') + "\n"
        result = pages.assemble(self.samples, self.originals)
        self.assertEqual(result["status"], pages.CAPTURED)
        self.assertEqual(self.grade(result)["reward"], 1.)
        self.assertIn('"\\u00713"', result["reference_text"])
        for units in result["provenance"].values():
            for mapping in units.values():
                sample = self.samples[mapping["page_index"]]
                source = sample.text[slice(*mapping["source_span"])]
                assembled = result["reference_text"][slice(*mapping["assembled_span"])]
                self.assertEqual(mapping["sample_id"], sample.sample_id)
                self.assertEqual(source, assembled)
                json.loads(source)
        self.assertEqual(set(result["provenance"]["facts"]), set(result["expansion"]["fact_spans"]))
        self.assertEqual(set(result["provenance"]["decisions"]), set(result["expansion"]["decisions"]))
        self.assertTrue(all(row["page_index"] > 0 for row in result["provenance"]["decisions"].values()))

    def test_numeric_answers_preserve_original_decimal_lexemes_without_float_roundtrip(self) -> None:
        raw = "0.123456789012345678901234567890123456789"
        self.samples[0].text = self.samples[0].text.replace('"corrected_answer": "accept"', '"corrected_answer": ' + raw)
        self.samples[0].text = self.samples[0].text.replace('[6, "CORRECT", "accept"]', '[6, "CORRECT", ' + raw + ']')
        result = pages.assemble(self.samples, self.originals)
        self.assertEqual(result["status"], pages.CAPTURED)
        decoded, _ = reference._decode(result["reference_text"])
        self.assertEqual(decoded["corrected_answer"], Decimal(raw))
        self.assertEqual(decoded["attempts"][-1][-1], Decimal(raw))
        self.assertEqual(result["reference_text"].count(raw), 2)
        self.assertEqual(result["expansion"]["expanded_note"]["corrected_answer"], raw)
        self.samples[0].text = self.samples[0].text.replace(raw, "1e9999999999999999999")
        self.assertEqual(pages.assemble(self.samples, self.originals)["status"], pages.INVALID)


if __name__ == "__main__":
    unittest.main()
