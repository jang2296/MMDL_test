"""Public JSON repair diagnostics; authored fixtures never certify actual model improvement."""
import copy
import json
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures


class SynthesisFormatFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        self.texts = [fixtures.response(self.item.slots)] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2
        self.tree = fixtures.tree_from(self.actor, self.verifier, self.texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.note = fixtures.note_for(self.tree, self.audit, self.item)

    def grade(self, note, *, tree=None, audit=None):
        return synthesis.grade_note(json.dumps(note), "stop", tree or self.tree, audit or self.audit,
            self.verifier, actor=self.actor, contract=synthesis.BOUNDED_VERSION)

    @staticmethod
    def payload(report, sample=None):
        return json.loads(synthesis.notebook_feedback(report, sample).split("Verifier feedback:\n", 1)[1])

    def test_string_facts_status_key_and_flat_ranges_are_diagnosed_without_coercion(self):
        bad = copy.deepcopy(self.note)
        bad["corrected_facts"] = "start = q0"
        for attempt in bad["attempts"]:
            attempt["status"] = attempt.pop("outcome")
            attempt["evidence_ranges"] = [1, 23]
        raw = json.dumps(bad)
        before = copy.deepcopy(bad)
        checked = self.grade(bad)
        self.assertEqual((checked["reward"], checked["failure_code"], checked["checks"]), (0., "INVALID_FIELDS", {}))
        errors = checked["format_diagnostics"]["errors"]
        self.assertEqual(len(errors), 1 + 3 * len(bad["attempts"]))
        for error in ({"code": "EXPECTED_OBJECT", "path": "$.corrected_facts"},
                      {"code": "MISSING_KEY", "path": "$.attempts[0].outcome"},
                      {"code": "UNEXPECTED_KEY", "path": "$.attempts[0].status"},
                      {"code": "EXPECTED_RANGE_PAIR", "path": "$.attempts[0].evidence_ranges"}):
            self.assertIn(error, errors)
        self.assertEqual(bad, before)
        self.assertEqual(synthesis.format_diagnostics(raw), checked["format_diagnostics"])
        sample = replace(solution_samples(self.tree)[0], text=raw, reward=0.)
        self.assertEqual(self.payload(checked, sample)["format_diagnostics"], checked["format_diagnostics"])

    def test_nested_required_keys_and_types_have_public_paths_only(self):
        note = copy.deepcopy(self.note)
        note["attempts"][0]["unverified"] = [{"kind": [], "rule": "unresolved", "locus": "submitted",
            "evidence_ranges": [[True, "2"]], "blocked_by": [{"rule": 1, "locus": {}}]}]
        errors = synthesis.format_diagnostics(json.dumps(note))["errors"]
        self.assertIn({"code": "EXPECTED_STRING", "path": "$.attempts[0].unverified[0].kind"}, errors)
        self.assertIn({"code": "EXPECTED_INTEGER", "path": "$.attempts[0].unverified[0].evidence_ranges[0][0]"}, errors)
        self.assertIn({"code": "EXPECTED_INTEGER", "path": "$.attempts[0].unverified[0].evidence_ranges[0][1]"}, errors)
        self.assertIn({"code": "EXPECTED_STRING", "path": "$.attempts[0].unverified[0].blocked_by[0].rule"}, errors)
        self.assertTrue(synthesis._valid_format_diagnostics(synthesis.format_diagnostics(json.dumps(note))))

    def test_missing_containers_and_bad_scalar_types_are_not_silently_defaulted(self):
        for field, value, code in (("attempts", {}, "EXPECTED_LIST"), ("lessons", "none", "EXPECTED_LIST"),
                                   ("success_attempt", True, "EXPECTED_INTEGER"),
                                   ("corrected_answer", {}, "EXPECTED_ANSWER")):
            note = copy.deepcopy(self.note)
            note[field] = value
            self.assertIn({"code": code, "path": "$." + field}, synthesis.format_diagnostics(json.dumps(note))["errors"])
        empty = synthesis.format_diagnostics("{}")
        self.assertEqual(len(empty["errors"]), 5)
        self.assertTrue(all(row["code"] == "MISSING_KEY" for row in empty["errors"]))

    def test_correct_shape_has_no_syntax_errors_but_cannot_pass_semantic_failures(self):
        self.assertEqual(self.grade(self.note)["reward"], 1.)
        for mutation in ("missing_issues", "wrong_fact", "wrong_outcome", "wrong_line_bounds", "attempt_count"):
            note = copy.deepcopy(self.note)
            if mutation == "missing_issues":
                note["attempts"][0]["issues"] = []
                note["lessons"] = []
            elif mutation == "wrong_fact":
                note["corrected_facts"]["start"] = "q3"
            elif mutation == "wrong_outcome":
                note["attempts"][0]["outcome"] = "submitted-untrue-outcome"
            elif mutation == "wrong_line_bounds":
                note["attempts"][0]["evidence_ranges"] = [[-2, 999999]]
            else:
                note["attempts"] = []
            checked = self.grade(note)
            self.assertEqual(checked["reward"], 0.)
            self.assertEqual(checked["format_diagnostics"]["errors"], [])
            sample = replace(solution_samples(self.tree)[0], text=json.dumps(note), reward=0.)
            self.assertEqual(self.payload(checked, sample)["format_diagnostics"]["errors"], [])

    def test_format_feedback_does_not_depend_on_changed_private_expectations(self):
        bad = copy.deepcopy(self.note)
        bad["corrected_facts"] = "a submitted string"
        first = self.grade(bad)
        original = copy.deepcopy(self.audit)
        self.verifier.items[self.actor.problem_id] = replace(self.item, slots=self.item.slots | {"start": "q3"})
        tree = fixtures.tree_from(self.actor, self.verifier, self.texts)
        audit = synthesis.audit_episode(tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.assertNotEqual(audit, original)
        second = self.grade(bad, tree=tree, audit=audit)
        second["private_reference"] = "PRIVATE_REFERENCE_MUST_NOT_APPEAR"
        self.assertEqual(first["format_diagnostics"], second["format_diagnostics"])
        self.assertEqual(self.payload(first), self.payload(second))
        self.assertNotIn("PRIVATE_REFERENCE_MUST_NOT_APPEAR", synthesis.notebook_feedback(second))

    def test_feedback_rejects_forged_diagnostics_or_private_paths(self):
        bad = copy.deepcopy(self.note)
        bad["corrected_facts"] = "submitted string"
        checked = self.grade(bad)
        sample = replace(solution_samples(self.tree)[0], text=json.dumps(bad), reward=0.)
        forged = copy.deepcopy(checked)
        forged["format_diagnostics"]["errors"] = []
        with self.assertRaisesRegex(ValueError, "format diagnostics differ"):
            synthesis.notebook_feedback(forged, sample)
        for field, value in (("code", "PRIVATE_EXPECTED_LABEL"), ("path", "$.private_reference")):
            forged = copy.deepcopy(checked)
            forged["format_diagnostics"]["errors"][0][field] = value
            with self.assertRaises(ValueError):
                synthesis.notebook_feedback(forged)

    def test_empty_attempt_citations_have_public_metadata_before_semantic_checks(self):
        note = copy.deepcopy(self.note)
        for attempt in note["attempts"]:
            attempt["evidence_ranges"] = []
        before = copy.deepcopy(note)
        checked = self.grade(note)
        self.assertEqual((checked["reward"], checked["failure_code"], checked["checks"], checked["attempt_checks"]),
                         (0., "INVALID_FIELDS", {}, []))
        sample = replace(solution_samples(self.tree)[0], text=json.dumps(note), reward=0.)
        payload = self.payload(checked, sample)
        diagnostics = payload["format_diagnostics"]
        self.assertEqual(diagnostics["empty_attempt_evidence_positions"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(diagnostics["submitted_attempt_count"], 6)
        self.assertFalse(diagnostics["empty_attempt_evidence_positions_truncated"])
        self.assertEqual(diagnostics["errors"], [])
        self.assertEqual(payload["failed_attempts"], [])
        self.assertIsNone(payload["checks"]["submitted_lessons_verified"])
        self.assertIn("not claimed\nattempt IDs", synthesis.RETRY_INSTRUCTION)
        self.assertIn("Use [] only for an actually empty original response; never invent a line.", synthesis.RETRY_INSTRUCTION)
        self.assertEqual(note, before)
        with self.assertRaisesRegex(ValueError, "requires its actual sample"):
            self.payload(checked)

    def test_empty_evidence_metadata_uses_json_positions_and_exact_top_level_lists(self):
        note = copy.deepcopy(self.note)
        note["attempts"] = [{"attempt": 600, "evidence_ranges": []},
                            {"attempt": 1, "evidence_ranges": [[]]}, {"evidence_ranges": "[]"},
                            {"evidence_ranges": None}, {"evidence_ranges": {}},
                            {"unverified": [{"evidence_ranges": []}]}, {"attempt": False, "evidence_ranges": []}]
        diagnostics = synthesis.format_diagnostics(json.dumps(note))
        self.assertEqual(diagnostics["empty_attempt_evidence_positions"], [1, 7])
        self.assertEqual(diagnostics["submitted_attempt_count"], 7)
        self.assertTrue(synthesis._valid_format_diagnostics(diagnostics))
        for text, count in (("{}", None), ('{"attempts":{}}', None), ('{"attempts":[]}', 0),
                            ('{"attempts":[],"attempts":[{}]}', None), ("[]", None), ("{", None)):
            with self.subTest(text=text):
                diagnostics = synthesis.format_diagnostics(text)
                self.assertEqual(diagnostics["submitted_attempt_count"], count)
                self.assertEqual(diagnostics["empty_attempt_evidence_positions"], [])
                self.assertFalse(diagnostics["empty_attempt_evidence_positions_truncated"])
                self.assertTrue(synthesis._valid_format_diagnostics(diagnostics))

    def test_empty_evidence_metadata_rejects_tampering_and_requires_the_actual_sample(self):
        note = copy.deepcopy(self.note)
        for index in (0, 2):
            note["attempts"][index]["evidence_ranges"] = []
        checked = self.grade(note)
        sample = replace(solution_samples(self.tree)[0], text=json.dumps(note), reward=0.)
        self.assertTrue(synthesis._valid_format_diagnostics(checked["format_diagnostics"]))
        mutations = [("empty_attempt_evidence_positions", positions) for positions in
                     ([], [2], [1, 1], [3, 1], [0], [7], [True], [1.0], ["1"], [{}], list(range(1, 34)))]
        mutations += [("submitted_attempt_count", count) for count in (None, -1, True, 6.0, 7)]
        mutations += [("empty_attempt_evidence_positions_truncated", flag) for flag in (True, 0, None)]
        for field, value in mutations:
            bad = copy.deepcopy(checked)
            bad["format_diagnostics"][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "format diagnostics differ"):
                self.payload(bad, sample)
        bad = copy.deepcopy(checked)
        bad["format_diagnostics"]["empty_attempt_evidence_positions"] = [2]
        with self.assertRaisesRegex(ValueError, "requires its actual sample"):
            self.payload(bad)
        with self.assertRaisesRegex(ValueError, "differs from the actual note sample"):
            self.payload(checked, replace(sample, text=sample.text + " "))

    def test_empty_evidence_positions_have_their_own_bound_and_truncation(self):
        note = copy.deepcopy(self.note)
        for count in (synthesis.MAX_FORMAT_ERRORS, synthesis.MAX_FORMAT_ERRORS + 10000):
            note["attempts"] = [{"evidence_ranges": []}] * count
            diagnostics = synthesis.format_diagnostics(json.dumps(note))
            with self.subTest(count=count):
                self.assertEqual(diagnostics["empty_attempt_evidence_positions"], list(range(1, 33)))
                self.assertEqual(diagnostics["submitted_attempt_count"], count)
                self.assertEqual(diagnostics["empty_attempt_evidence_positions_truncated"], count > synthesis.MAX_FORMAT_ERRORS)
                self.assertEqual(len(diagnostics["errors"]), synthesis.MAX_FORMAT_ERRORS)
                self.assertTrue(diagnostics["truncated"])
                self.assertTrue(synthesis._valid_format_diagnostics(diagnostics))
                bad = copy.deepcopy(diagnostics)
                bad["empty_attempt_evidence_positions_truncated"] = True
                if count == synthesis.MAX_FORMAT_ERRORS:
                    self.assertFalse(synthesis._valid_format_diagnostics(bad))

    def test_actually_empty_originals_remain_accepted_without_invented_citations(self):
        tree = fixtures.tree_from(self.actor, self.verifier, [""] * 4 + self.texts[4:])
        audit = synthesis.audit_episode(tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        note = fixtures.note_for(tree, audit, self.item)
        checked = self.grade(note, tree=tree, audit=audit)
        self.assertEqual(checked["reward"], 1.)
        self.assertEqual(checked["format_diagnostics"]["empty_attempt_evidence_positions"], [1, 2, 3, 4])
        self.assertEqual(checked["format_diagnostics"]["errors"], [])
        for position in (0, 4):
            bad = copy.deepcopy(note)
            bad["attempts"][position]["evidence_ranges"] = [[1, 1]] if position == 0 else []
            self.assertEqual(self.grade(bad, tree=tree, audit=audit)["reward"], 0.)

    def test_invalid_json_unknown_extras_and_error_limits_are_safe(self):
        for text in ("{", '{"attempts":[],"attempts":[]}', "[" * 2000):
            self.assertEqual(synthesis.format_diagnostics(text)["errors"], [{"code": "INVALID_JSON", "path": "$"}])
        self.assertEqual(synthesis.format_diagnostics("[]")["errors"], [{"code": "EXPECTED_OBJECT", "path": "$"}])
        note = copy.deepcopy(self.note)
        private = "submitted-name-" + "x" * 10000
        note[private] = "submitted value"
        text = json.dumps(synthesis.format_diagnostics(json.dumps(note)))
        self.assertNotIn(private, text)
        note["attempts"] = [0] * (synthesis.MAX_FORMAT_ERRORS + 10)
        bounded = synthesis.format_diagnostics(json.dumps(note))
        self.assertEqual(len(bounded["errors"]), synthesis.MAX_FORMAT_ERRORS)
        self.assertTrue(bounded["truncated"])
        self.assertTrue(synthesis._valid_format_diagnostics(bounded))

    def test_public_shape_preserves_every_original_line_and_supplies_no_facts(self):
        prompt = synthesis.instruction(solution_samples(self.tree), self.audit, "PUBLIC_SLOT_SCHEMA")
        self.assertIn(synthesis.PUBLIC_JSON_SHAPE, prompt)
        self.assertIn('"outcome":"<public outcome>"', prompt)
        self.assertIn('"evidence_ranges":[[<integer>,<integer>]]', prompt)
        rows = json.loads(prompt.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
        for row, sample in zip(rows, solution_samples(self.tree), strict=True):
            self.assertEqual("".join(text for _, text in row["source_lines"]), sample.text)
        self.assertNotIn("q0", synthesis.PUBLIC_JSON_SHAPE)
        self.assertNotIn("states[", synthesis.PUBLIC_JSON_SHAPE)


if __name__ == "__main__":
    unittest.main()
