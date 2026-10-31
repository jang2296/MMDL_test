"""Public expansion tests use authored fixtures, never model learning evidence."""
import copy
import json
import unittest
from types import SimpleNamespace

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl import whole_reference_notebook as reference
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures


def find_ref(inventory, endpoint):
    return next(ref for ref, row in inventory["references"].items()
                if all(row[key] == endpoint[key] for key in ("attempt", "kind", "rule", "locus")))


def compact(samples, note):
    inventory = reference.public_inventory(samples)
    rows = {ref: [ref, "N", [], None] for ref in inventory["assessment_refs"]}
    for attempt in note["attempts"]:
        for field, flag in (("issues", "I"), ("unverified", "U")):
            for claim in attempt[field]:
                ref = find_ref(inventory, {"attempt": attempt["attempt"], **claim})
                rows[ref][1] = "IU" if rows[ref][1] not in ("N", flag) else flag
                if flag == "U":
                    rows[ref][2] = [find_ref(inventory, {"attempt": attempt["attempt"], "kind": "TRACE", **dep})
                                    for dep in claim["blocked_by"]]
    for lesson in note["lessons"]:
        ref = find_ref(inventory, lesson["earlier"])
        later = (inventory["absence_refs"][str(lesson["earlier"]["attempt"])]
                 if lesson["relation"] == "NOT_REVISITED" else find_ref(inventory, lesson["later"]))
        rows[ref][3] = [lesson["relation"], later]
    return {"corrected_facts": copy.deepcopy(note["corrected_facts"]),
        "attempts": [[row["attempt"], row["outcome"], row["answer"]] for row in note["attempts"]],
        "assessments": list(rows.values()), "success_attempt": note["success_attempt"],
        "corrected_answer": note["corrected_answer"]}


class WholeReferenceTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        self.wrong = fixtures.response(self.item.slots)
        self.right = fixtures.response(self.item.slots, wrong=False)
        self.prepare([self.wrong] * 4 + [self.right] * 2)

    def prepare(self, texts):
        self.tree = fixtures.tree_from(self.actor, self.verifier, texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.samples = solution_samples(self.tree)
        self.typed = fixtures.note_for(self.tree, self.audit, self.item)
        self.compact = compact(self.samples, self.typed)

    def expand(self, note=None):
        return reference.expand(json.dumps(self.compact if note is None else note), "stop", self.samples)

    def grade(self, expanded):
        return synthesis.grade_note(expanded["expanded_text"], "stop", self.tree, self.audit,
            self.verifier, actor=self.actor, contract=synthesis.BOUNDED_VERSION)

    def test_public_outcomes_match_verifier_response_statuses_and_have_separate_flags(self):
        actor, _, verifier = fixtures.graph_native("shortest")
        cases = [(self.verifier, self.actor, self.right, "stop"),
                 (self.verifier, self.actor, self.wrong, "stop"),
                 (self.verifier, self.actor, "no final answer", "stop"),
                 (self.verifier, self.actor, self.right + "\nFinal answer: reject", "stop"),
                 (self.verifier, self.actor, self.right, "length"),
                 (verifier, actor, "Final answer: not-a-number", "stop")]
        outcomes = [checker.grade(item.problem_id, text, finish, item.mode).status
                    for checker, item, text, finish in cases]
        contract = reference.contract()
        self.assertEqual(contract["version"], "whole-history-reference-assessments-v2")
        self.assertEqual(contract["outcomes"], outcomes)
        self.assertEqual(contract["attempt_columns"], ["attempt", "outcome", "extracted_answer_or_null"])
        self.assertTrue(set(contract["outcomes"]).isdisjoint(contract["flags"]))
        prompt = reference.instruction(self.samples, "PUBLIC_SCHEMA", 6)
        for outcome in outcomes:
            self.assertIn(outcome, prompt)
        self.assertIn("These flags are never attempt outcomes", prompt)

    def test_synthetic_type_examples_are_literal_json_after_prompt_formatting(self):
        prompt = reference.instruction(self.samples, "PUBLIC_SCHEMA", 6)
        examples = {}
        decoder = json.JSONDecoder()
        for line in prompt.splitlines():
            label, _, raw = line.partition(": ")
            if label in {"corrected_facts fragment", "attempt row with answer", "attempt row without extracted answer",
                         "assessment row without decision", "assessment row with relationship"}:
                examples[label] = decoder.raw_decode(raw)[0]
        self.assertEqual(len(examples), 5)
        self.assertEqual(examples["corrected_facts fragment"], {"synthetic_slot": "731"})
        for key in ("attempt row with answer", "attempt row without extracted answer"):
            row = examples[key]
            self.assertEqual(len(row), 3)
            self.assertIs(type(row[0]), int)
            self.assertIn(row[1], reference.contract()["outcomes"])
        self.assertIs(type(examples["attempt row with answer"][2]), str)
        self.assertIsNone(examples["attempt row without extracted answer"][2])
        self.assertEqual(examples["assessment row without decision"], ["<source ref>", "N", [], None])
        decision = examples["assessment row with relationship"][3]
        self.assertEqual(decision, ["CORRECTED", "<later ref>"])
        self.assertIn('Use JSON null, never the string "null"', prompt)
        self.assertIn("unrelated dummy values; do not copy them as item judgments", prompt)

    def test_observed_termination_is_visible_when_identical_text_has_different_outcomes(self):
        from fine_tuning.notebook_rl.answers import extract_final

        row = SimpleNamespace(text="Final answer: 731", sample_id="one", parent_sample_id=None,
                              finish_reason="stop")
        stopped = reference.instruction([row], "PUBLIC_SCHEMA", 1)
        stop_result = extract_final(row.text, row.finish_reason)
        row.finish_reason = "length"
        truncated = reference.instruction([row], "PUBLIC_SCHEMA", 1)
        length_result = extract_final(row.text, row.finish_reason)
        self.assertNotEqual(stopped, truncated)
        self.assertNotEqual(stop_result, length_result)
        for prompt, finish in ((stopped, "stop"), (truncated, "length")):
            transcript = json.loads(prompt.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
            self.assertEqual(transcript[0]["finish_reason"], finish)
            self.assertEqual(transcript[0]["source_lines"], [[1, row.text]])
        self.assertIn('the outcome is TRUNCATED and extracted answer is null', truncated)
        for invalid in (None, True, "unknown"):
            row.finish_reason = invalid
            with self.assertRaises(ValueError):
                reference.instruction([row], "PUBLIC_SCHEMA", 1)

    def test_public_grammar_does_not_read_private_outcomes_answers_or_targets(self):
        private = [SimpleNamespace(text=row.text, sample_id=row.sample_id, parent_sample_id=row.parent_sample_id,
                    finish_reason=row.finish_reason,
                    result=SimpleNamespace(status="PRIVATE_OUTCOME", extracted="PRIVATE_ANSWER", slots="PRIVATE_SLOTS"),
                    gold="PRIVATE_GOLD", expected="PRIVATE_EXPECTED") for row in self.samples]
        prompt = reference.instruction(private, "PUBLIC_SCHEMA", 6)
        self.assertEqual(prompt, reference.instruction(self.samples, "PUBLIC_SCHEMA", 6))
        for hidden in ("PRIVATE_OUTCOME", "PRIVATE_ANSWER", "PRIVATE_SLOTS", "PRIVATE_GOLD", "PRIVATE_EXPECTED"):
            self.assertNotIn(hidden, prompt)
        for sample in self.samples:
            self.assertNotIn(sample.sample_id, reference.prompt_contract())
        for key in ("reward", "expected", "gold", "verified"):
            self.assertNotIn(key, reference.contract())

    def test_attempt_outcome_claims_are_preserved_and_strictly_checked_without_host_repairs(self):
        for flag in reference.contract()["flags"]:
            with self.subTest(flag=flag):
                changed = copy.deepcopy(self.compact)
                changed["attempts"][0][1] = flag
                expanded = self.expand(changed)
                self.assertEqual(expanded["status"], reference.CAPTURED)
                self.assertEqual(expanded["expanded_note"]["attempts"][0]["outcome"], flag)
                self.assertEqual(self.grade(expanded)["reward"], 0.)

    def test_numeric_answer_rendering_preserves_existing_decimal_judgments(self):
        actor, item, verifier = fixtures.graph_native("shortest")
        wrong = {"kind": "shortest", "nodes": ["A", "C", "H"], "edge_ids": None,
                 "weights": [4, 4], "cost": 8}
        right = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None,
                 "weights": [2, 2], "cost": 4}
        for answer, expected in (("8", 1.), ("8.0", 0.)):
            with self.subTest(answer=answer):
                tree = fixtures.tree_from(actor, verifier,
                    [fixtures.graph_response(item.slots, wrong, answer)] * 4
                    + [fixtures.graph_response(item.slots, right, "4")] * 2)
                audit = synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION)
                samples = solution_samples(tree)
                typed = fixtures.note_for(tree, audit, item)
                original, raw = json.dumps(typed), json.dumps(compact(samples, typed))
                if answer == "8":
                    original = original.replace('"corrected_answer": "4"', '"corrected_answer": 4e0')
                    raw = raw.replace('"corrected_answer": "4"', '"corrected_answer": 4e0')
                else:
                    original = original.replace('"answer": "8.0"', '"answer": 8.0000000000000001')
                    raw = raw.replace('"INCORRECT", "8.0"', '"INCORRECT", 8.0000000000000001')
                expanded = reference.expand(raw, "stop", samples)
                direct = synthesis.grade_note(original, "stop", tree, audit, verifier,
                    actor=actor, contract=synthesis.BOUNDED_VERSION)
                rendered = synthesis.grade_note(expanded["expanded_text"], "stop", tree, audit, verifier,
                    actor=actor, contract=synthesis.BOUNDED_VERSION)
                self.assertEqual(direct["reward"], expected)
                self.assertEqual(rendered["reward"], direct["reward"])

    def test_public_inventory_reads_only_public_fields_and_keeps_success_siblings(self):
        public = [SimpleNamespace(text=row.text, sample_id=row.sample_id, parent_sample_id=row.parent_sample_id,
                    finish_reason=row.finish_reason)
                  for row in self.samples]
        inventory = reference.public_inventory(public)
        self.assertEqual(inventory, reference.public_inventory(self.samples))
        self.assertEqual(inventory["retry_pairs"], [[1, 3], [1, 4], [3, 5], [3, 6]])
        self.assertEqual({row["attempt"] for row in inventory["references"].values()}, {None, 1, 2, 3, 4, 5, 6})
        self.assertEqual(reference.validate_audit_coverage(inventory, self.audit)["status"], "AUDIT_COVERAGE_COMPLETE")
        changed = copy.deepcopy(self.audit)
        changed["attempts"][0]["checks"][0]["locus"] = "PRIVATE_UNMAPPABLE"
        with self.assertRaisesRegex(ValueError, "absent from public"):
            reference.validate_audit_coverage(inventory, changed)
        self.assertNotIn("PRIVATE_UNMAPPABLE", json.dumps(inventory))

    def test_all_raw_observation_and_trace_occurrences_survive_malformed_duplicates(self):
        text = ("Observed facts:\r\nDir[ e4 ] = C->H\r\ndir[e4] = ???\r\nno assignment\r\n"
                "Solution:\r\nTrace: {\"kind\":\"dfa\",\"states\":[\"q0\"],\"decision\":\"reject\"}\r\n"
                "Trace: {broken}\r\n")
        public = [SimpleNamespace(text=text, sample_id="one", parent_sample_id=None, finish_reason="stop")]
        inventory = reference.public_inventory(public)
        observed = next(row for row in inventory["references"].values() if row["kind"] == "OBSERVATION")
        self.assertEqual(observed["locus"], "dir[e4]")
        self.assertEqual([row[4] for row in observed["occurrences"]], ["C->H", "???"])
        for occurrence in observed["occurrences"]:
            self.assertEqual(text[occurrence[2]:occurrence[3]], occurrence[5])
        self.assertIn("no assignment\r\n", [row[-1] for row in inventory["coverage"]["raw_observation_lines"]])
        self.assertEqual(len(inventory["trace_lines"]), 2)
        self.assertTrue(any(row["rule"] == "DFA_ACCEPTANCE" for row in inventory["references"].values()))
        for rule in ("TRACE_FORMAT", "TRACE_SCHEMA", "RESPONSE_SCOPE", "NATIVE_CONTRACT"):
            self.assertEqual(sum(row["rule"] == rule for row in inventory["references"].values()), 1)
        unterminated = [SimpleNamespace(text="Observed facts:\nroot = A", sample_id="one", parent_sample_id=None,
                                       finish_reason="stop")]
        self.assertEqual(reference.public_inventory(unterminated)["coverage"]["raw_observation_lines"][0][-1], "root = A")

    def test_compact_input_preserves_every_reference_occurrence_and_full_original(self):
        self.samples[0].text += '\nObserved facts:\ndir[e4] = C->H\ndir[e4] = ???\nraw text\nTrace: {broken}\n'
        inventory = reference.public_inventory(self.samples)
        prompt = reference.instruction(self.samples, "PUBLIC_SCHEMA", 6)
        wire = json.loads(prompt.split("<public_inventory>\n")[1].split("\n</public_inventory>")[0])
        transcript = json.loads(prompt.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        for source, row in zip(self.samples, transcript, strict=True):
            self.assertEqual("".join(text for _, text in row["source_lines"]), source.text)
        self.assertEqual(set(wire["references"]), set(inventory["references"]))
        for ref, row in inventory["references"].items():
            self.assertEqual(wire["references"][ref], [row["attempt"], row["kind"], row["rule"], row["locus"],
                [occurrence[1] for occurrence in row["occurrences"]]])
            for occurrence in row["occurrences"]:
                self.assertEqual(dict(transcript[occurrence[0] - 1]["source_lines"])[occurrence[1]], occurrence[-1])
        for key in ("assessment_refs", "absence_refs", "retry_pairs"):
            self.assertEqual(wire[key], inventory[key])
        self.assertEqual(wire["raw_observation_lines"], [row[:2] for row in inventory["coverage"]["raw_observation_lines"]])
        self.assertEqual(wire["trace_lines"], [row[:2] for row in inventory["trace_lines"]])

    def test_expansion_is_unverified_bookkeeping_and_existing_strict_grader_accepts_fixture(self):
        expanded = self.expand()
        self.assertEqual(expanded["status"], reference.CAPTURED)
        self.assertEqual(set(expanded["expanded_note"]), {"corrected_facts", "attempts", "lessons", "success_attempt", "corrected_answer"})
        self.assertEqual(expanded["expanded_note"]["corrected_facts"], self.typed["corrected_facts"])
        self.assertEqual(len(expanded["expanded_note"]["attempts"]), 6)
        self.assertEqual(self.grade(expanded)["reward"], 1.)
        self.assertNotIn("verified", expanded)
        self.assertNotIn("reward", expanded)
        for row in expanded["expanded_note"]["attempts"]:
            self.assertEqual(row["evidence_ranges"], [[1, len(self.samples[row["attempt"] - 1].text.splitlines())]])
        # A complete public row inventory with no issue claims still earns no strict reward.
        blank = copy.deepcopy(self.compact)
        for row in blank["assessments"]:
            row[1:] = ["N", [], None]
        self.assertEqual(self.expand(blank)["status"], reference.CAPTURED)
        self.assertEqual(self.grade(self.expand(blank))["reward"], 0.)

    def test_correct_successes_still_require_wrong_observation_issue_assessments(self):
        wrong_success = self.right.replace("start = q0", "start = q1")
        self.prepare([self.wrong] * 4 + [wrong_success] * 2)
        expanded = self.expand()
        self.assertEqual(self.audit["attempts"][-1]["outcome"], "CORRECT")
        self.assertEqual(self.grade(expanded)["reward"], 1.)
        wrong_ref = find_ref(expanded["inventory"], {"attempt": 6, "kind": "OBSERVATION", "rule": "OBSERVATION_MATCH", "locus": "start"})
        changed = copy.deepcopy(self.compact)
        next(row for row in changed["assessments"] if row[0] == wrong_ref)[1:] = ["N", [], None]
        self.assertEqual(self.grade(self.expand(changed))["reward"], 0.)

    def test_known_wrong_and_malformed_duplicate_can_expand_both_issue_and_unknown(self):
        mixed = self.wrong.replace("start = q0", "start = q1\nstart = ???")
        self.prepare([mixed] * 4 + [self.right] * 2)
        expanded = self.expand()
        self.assertEqual(expanded["status"], reference.CAPTURED)
        self.assertTrue(any(row[1] == "IU" for row in self.compact["assessments"]))
        self.assertEqual(self.grade(expanded)["reward"], 1.)

    def test_dependencies_and_terminal_absence_are_model_selected_not_invented(self):
        short = fixtures.graph_response(self.item.slots, {"kind": "dfa", "states": ["q0"], "decision": "reject"}, "reject")
        self.prepare([short] * 4 + [self.right] * 2)
        expanded = self.expand()
        unknown = next(row for row in expanded["expanded_note"]["attempts"][0]["unverified"] if row["rule"] == "DFA_ACCEPTANCE")
        self.assertEqual(unknown["blocked_by"], [{"rule": "DFA_COMPLETE_WORD", "locus": "states"}])
        self.assertEqual(self.grade(expanded)["reward"], 1.)
        wrong_success = self.right.replace("start = q0", "start = q1")
        self.prepare([self.wrong] * 4 + [wrong_success] * 2)
        expanded = self.expand()
        absence = next(lesson for lesson in expanded["expanded_note"]["lessons"] if lesson["relation"] == "NOT_REVISITED")
        self.assertEqual(absence["later"], {"attempt": None, "kind": "ABSENCE", "rule": None, "locus": None, "evidence_ranges": []})

    def test_raw_spans_decode_exact_literals_with_unicode_escapes_and_repeated_values(self):
        text = " \n" + json.dumps(self.compact, ensure_ascii=True, indent=2).replace('"q3"', '"\\u00713"') + "\n"
        expanded = reference.expand(text, "stop", self.samples)
        self.assertEqual(expanded["status"], reference.CAPTURED)
        for slot, (start, end) in expanded["fact_spans"].items():
            self.assertEqual(json.loads(text[start:end]), self.compact["corrected_facts"][slot])
        for ref, (start, end) in expanded["decision_spans"].items():
            raw_decision = next(row[3] for row in self.compact["assessments"] if row[0] == ref)
            self.assertEqual(json.loads(text[start:end]), raw_decision)
            self.assertEqual(expanded["decisions"][ref]["relation"], raw_decision[0])
            self.assertEqual(text[start], "[")

    def test_scope_unknown_and_output_relationships_keep_existing_strict_meanings(self):
        self.prepare([self.wrong] * 2 + ["malformed"] * 2 + [self.right] * 2)
        inventory = reference.public_inventory(self.samples)
        issue = next(row for row in self.compact["assessments"] if row[1] == "I")
        later = find_ref(inventory, {"attempt": 3, "kind": "TRACE", "rule": "TRACE_FORMAT", "locus": "solution"})
        issue[3] = ["SCOPE_UNVERIFIABLE_LATER", later]
        expanded = self.expand()
        selected = expanded["decisions"][issue[0]]["later"]
        self.assertEqual(selected["blocked_by"], [])
        self.assertEqual(self.grade(expanded)["reward"], 1.)
        missing = self.wrong.rsplit("\nFinal answer:", 1)[0]
        self.prepare([missing] * 4 + [self.right] * 2)
        inventory = reference.public_inventory(self.samples)
        first = find_ref(inventory, {"attempt": 1, "kind": "OUTPUT", "rule": "FINALIZATION", "locus": "final_answer"})
        last = find_ref(inventory, {"attempt": 5, "kind": "OUTPUT", "rule": "FINALIZATION", "locus": "final_answer"})
        output = next(row for row in self.compact["assessments"] if row[0] == first)
        output[3] = ["FINALIZED", last]
        expanded = self.expand()
        self.assertEqual(output[1], "N")
        self.assertEqual(expanded["decisions"][first]["relation"], "FINALIZED")
        self.assertEqual(self.grade(expanded)["reward"], 1.)

    def test_missing_duplicate_extra_order_bool_and_malformed_rows_fail_closed(self):
        mutations = []
        for rows in (self.compact["assessments"][:-1], self.compact["assessments"] + [self.compact["assessments"][0]],
                     self.compact["assessments"][::-1]):
            mutations.append(self.compact | {"assessments": rows})
        for key, value in (("success_attempt", True), ("success_attempt", 7), ("corrected_answer", False),
                           ("attempts", [[True, "INCORRECT", None]] + self.compact["attempts"][1:]),
                           ("attempts", [[1, "INCORRECT"]] + self.compact["attempts"][1:]),
                           ("attempts", [[1, "INCORRECT", None, "I"]] + self.compact["attempts"][1:]),
                           ("corrected_facts", {"start": True}), ("corrected_facts", {"start": 731}),
                           ("unexpected", [])):
            mutations.append(self.compact | {key: value})
        for field, value in ((0, "not-present"), (1, "CORRECT"), (2, [True]), (3, "null"), (3, "I"),
                             (3, ["INVENTED", "a6:r1"]),
                             (3, ["CORRECTED", "not-present"]), (3, ["CORRECTED", self.compact["assessments"][0][0]])):
            changed = copy.deepcopy(self.compact)
            changed["assessments"][0][field] = value
            mutations.append(changed)
        for note in mutations:
            with self.subTest(note=note):
                expanded = self.expand(note)
                self.assertEqual(expanded["status"], reference.INVALID)
                self.assertIsNone(expanded["expanded_note"])
                self.assertEqual(expanded["decision_spans"], {})
        text = json.dumps(self.compact)
        malformed = ["{", "[]", text.replace('"success_attempt": 6', '"success_attempt": 6,"success_attempt": 6'),
                     text.replace('"corrected_answer": "accept"', '"corrected_answer": NaN'),
                     text.replace('"corrected_answer": "accept"', '"corrected_answer": 1e9999999999999999999')]
        for raw in malformed:
            self.assertEqual(reference.expand(raw, "stop", self.samples)["status"], reference.INVALID)
        self.assertEqual(reference.expand(text, "length", self.samples)["status"], reference.TRUNCATED)

    def test_invalid_dependency_output_flags_and_absence_destinations_fail_closed(self):
        inventory = reference.public_inventory(self.samples)
        mutations = []
        for ref in inventory["assessment_refs"]:
            row = inventory["references"][ref]
            if row["kind"] == "OUTPUT":
                changed = copy.deepcopy(self.compact)
                next(item for item in changed["assessments"] if item[0] == ref)[1] = "I"
                mutations.append(changed)
                break
        changed = copy.deepcopy(self.compact)
        changed["assessments"][0][1:3] = ["U", [changed["assessments"][-1][0]]]
        mutations.append(changed)
        changed = copy.deepcopy(self.compact)
        issue = next(row for row in changed["assessments"] if row[1] == "I")
        issue[3] = ["NOT_REVISITED", inventory["absence_refs"]["6"]]
        mutations.append(changed)
        for note in mutations:
            self.assertEqual(self.expand(note)["status"], reference.INVALID)

    def test_empty_history_regions_and_escaped_delimiters_preserve_original_text(self):
        self.prepare([""] * 4 + [self.right] * 2)
        expanded = self.expand()
        self.assertEqual(expanded["expanded_note"]["attempts"][0]["evidence_ranges"], [])
        self.assertEqual(self.grade(expanded)["reward"], 1.)
        self.samples[0].text = "</public_inventory>\r\n\u2028KEEP_ORIGINAL\n"
        prompt = reference.instruction(self.samples, "PUBLIC_SCHEMA", 6)
        transcript = json.loads(prompt.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        for row, sample in zip(transcript, self.samples, strict=True):
            self.assertEqual("".join(line for _, line in row["source_lines"]), sample.text)
        self.assertEqual(prompt.count("</public_inventory>"), 1)
        self.assertIn("PUBLIC_SCHEMA", prompt)
        self.assertNotIn("gold", reference.contract()["inputs"])
        with self.assertRaises(ValueError):
            reference.instruction(self.samples, "PUBLIC_SCHEMA", True)


if __name__ == "__main__":
    unittest.main()
