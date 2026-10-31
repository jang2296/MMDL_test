"""Optional output-presence relationships; authored fixtures are not model learning."""
import copy
import hashlib
import json
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl.notebook_learning import _first_attempt_lines
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures


def output_lesson(samples, earlier=1, later=3):
    def endpoint(attempt):
        text = samples[attempt - 1].text
        return {"attempt": attempt, "kind": "OUTPUT", "rule": "FINALIZATION", "locus": "final_answer",
                "evidence_ranges": [[1, len(text.splitlines())]] if text else []}
    return {"rule": "FINALIZATION", "earlier": endpoint(earlier), "later": endpoint(later),
            "relation": "FINALIZED"}


class SynthesisOutputTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        wrong = fixtures.response(self.item.slots)
        missing = wrong.rsplit("\nFinal answer:", 1)[0]
        self.texts = [missing] * 2 + [wrong] * 2 + [fixtures.response(self.item.slots, wrong=False)] * 2
        self.prepare()

    def prepare(self, texts=None):
        self.tree = fixtures.tree_from(self.actor, self.verifier, self.texts if texts is None else texts)
        self.samples = episode.solution_samples(self.tree)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.note = fixtures.note_for(self.tree, self.audit, self.item)
        self.lesson = output_lesson(self.samples)

    def lesson_check(self, lesson=None, *, audit=None):
        return synthesis._lesson(self.lesson if lesson is None else lesson, self.samples,
                                 self.audit if audit is None else audit, bounded=True)

    def grade(self, note=None, *, audit=None):
        return synthesis.grade_note(json.dumps(self.note if note is None else note), "stop", self.tree,
            self.audit if audit is None else audit, self.verifier, actor=self.actor, contract=synthesis.BOUNDED_VERSION)

    def test_missing_to_wrong_final_is_presence_only_and_correctness_is_separate(self):
        checked = self.lesson_check()
        self.assertTrue(checked["verified"])
        self.assertEqual(checked["credit_type"], "OUTPUT_FINALIZATION")
        self.assertFalse(checked["later_answer_correct"])
        self.assertEqual(checked["covered_issues"], [])
        self.assertEqual(checked["claim"], self.lesson)
        correct = self.lesson_check(output_lesson(self.samples, later=5))
        self.assertTrue(correct["verified"])
        self.assertTrue(correct["later_answer_correct"])
        self.assertEqual(correct["covered_issues"], [])

    def test_output_audit_is_additive_optional_and_does_not_change_native_coverage_or_route(self):
        native = episode.audit_episode(self.tree, self.actor, self.verifier, routing_rule=episode.STRICT_ROUTE_RULE)
        for key in ("route", "admission", "visual_attempts", "all_required_observations_verified",
                    "module_causality_certified", "model_claims_used_for_routing"):
            self.assertEqual(self.audit[key], native[key])
        for attempt in self.audit["attempts"]:
            self.assertEqual(len(attempt["output_checks"]), 1)
            self.assertTrue(all(row["kind"] != "OUTPUT" for key in ("checks", "issues", "unverified")
                                for row in attempt[key]))
            self.assertEqual(attempt["output_checks"][0]["identity"], ["OUTPUT", "FINALIZATION", "same-task"])
        before = self.grade()
        self.assertEqual(before["reward"], 1.)
        self.note["lessons"].append(self.lesson)
        after = self.grade()
        self.assertEqual(after["reward"], before["reward"])
        self.assertEqual(after["attempt_checks"], before["attempt_checks"])
        self.assertEqual(after["checks"], before["checks"])
        self.assertEqual(after["lesson_checks"][-1]["covered_issues"], [])
        self.assertFalse(after["solution_trace_complete"])
        self.assertFalse(after["full_diagnosis"])
        self.assertFalse(after["all_reasoning_verified"])

    def test_empty_marker_length_multiple_retraction_and_trailing_prose_are_excluded(self):
        sample = self.samples[2]
        cases = (("Final answer: ", "stop"), ("Final answer: .", "stop"),
                 (sample.text, "length"), (sample.text + "\nFinal answer: accept", "stop"),
                 (sample.text + "\nActually, retract that answer.", "stop"),
                 (sample.text + "\nAdditional prose.", "stop"), ("missing", "length"))
        for text, finish in cases:
            with self.subTest(text=text[-70:], finish=finish):
                self.assertEqual(synthesis._output_checks(replace(sample, text=text, finish_reason=finish)), [])
        self.assertEqual(synthesis._output_checks(replace(sample, text="Final answer: reject\r\n\t "))[0]["status"],
                         "COMPLIANT")
        self.assertEqual(synthesis._output_checks(replace(sample, text="Observed facts:\nSolution:"))[0]["status"],
                         "VIOLATED")

    def test_sibling_other_relations_bad_fields_and_selective_evidence_are_rejected(self):
        variants = [output_lesson(self.samples, earlier=2), output_lesson(self.samples, earlier=3, later=5)]
        for field, replacement in (("kind", "TRACE"), ("rule", "OBSERVATION_MATCH"), ("locus", "start"),
                                   ("blocked_by", []), ("evidence_ranges", [[1, 1]])):
            bad = copy.deepcopy(self.lesson)
            bad["earlier"][field] = replacement
            variants.append(bad)
        for mutation in ("later_excerpt", "wrong_rule", "wrong_relation", "extra", "bool_index"):
            bad = copy.deepcopy(self.lesson)
            if mutation == "later_excerpt":
                bad["later"]["evidence_ranges"] = [[1, 1]]
            elif mutation == "wrong_rule":
                bad["rule"] = "DFA_TRANSITION"
            elif mutation == "wrong_relation":
                bad["relation"] = "CORRECTED"
            elif mutation == "bool_index":
                bad["earlier"]["attempt"] = True
            else:
                bad["answer_correct"] = True
            variants.append(bad)
        for lesson in variants:
            with self.subTest(lesson=lesson):
                self.assertFalse(self.lesson_check(lesson)["verified"])

    def test_missing_parent_evidence_covers_every_original_line_and_empty_parent_is_bound(self):
        row = self.audit["attempts"][0]["output_checks"][0]
        lines = _first_attempt_lines(self.samples[0].text)
        self.assertEqual(row["refs"], [{"span": [line["start"], line["end"]],
            "sha256": hashlib.sha256(line["text"].encode()).hexdigest()} for line in lines])
        self.prepare([""] + self.texts[1:])
        self.assertEqual(self.audit["attempts"][0]["output_checks"][0]["refs"], [])
        self.assertEqual(self.lesson["earlier"]["evidence_ranges"], [])
        self.assertTrue(self.lesson_check()["verified"])
        self.lesson["earlier"]["evidence_ranges"] = [[1, 1]]
        self.assertFalse(self.lesson_check()["verified"])

    def test_output_cannot_discharge_native_issues_or_unknown_scope_and_is_not_an_attempt_claim(self):
        self.note["lessons"] = [self.lesson]
        report = self.grade()
        self.assertTrue(report["checks"]["lessons_verified"])
        self.assertFalse(report["checks"]["all_issues_synthesized"])
        self.assertEqual(report["reward"], 0.)
        self.note["attempts"][0]["unverified"] = []
        report = self.grade()
        self.assertFalse(report["checks"]["unknown_inventory_verified"])
        self.assertEqual(report["reward"], 0.)
        claim = {key: value for key, value in self.lesson["earlier"].items() if key != "attempt"}
        self.assertFalse(synthesis._claim(claim))
        for location in ("issues", "unverified"):
            note = copy.deepcopy(self.note)
            note["attempts"][0][location].append(claim | ({"blocked_by": []} if location == "unverified" else {}))
            self.assertEqual(self.grade(note)["reward"], 0.)

    def test_raw_finish_parent_and_output_audit_tampering_are_rejected(self):
        bad = copy.deepcopy(self.audit)
        bad["attempts"][0]["output_checks"][0]["refs"][0]["sha256"] = "0" * 64
        self.assertFalse(self.lesson_check(audit=bad)["verified"])
        with self.assertRaises(ValueError):
            self.grade(audit=bad)
        for mutation in ("raw", "finish", "parent", "outcome"):
            self.prepare()
            if mutation == "raw":
                self.samples[0].text += "\n"
            elif mutation == "finish":
                self.samples[0].finish_reason = "length"
            elif mutation == "parent":
                self.samples[2].parent_sample_id = self.samples[1].sample_id
            else:
                self.samples[2].result = replace(self.samples[2].result, status="CORRECT", reward_answer=1.)
                self.assertFalse(self.lesson_check()["verified"])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.grade()

    def test_public_output_shape_has_no_targets_and_legacy_contract_is_unchanged(self):
        self.note["lessons"].append(self.lesson)
        self.assertEqual(synthesis.format_diagnostics(json.dumps(self.note))["errors"], [])
        prompt = synthesis.instruction(self.samples, self.audit, "PUBLIC_SCHEMA")
        self.assertIn("FINALIZED", prompt)
        self.assertIn("presence only", prompt)
        self.assertIn("OUTPUT uses its separate optional FINALIZED checks", prompt)
        rows = json.loads(prompt.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
        self.assertEqual(["".join(text for _, text in row["source_lines"]) for row in rows], self.texts)
        legacy = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.VERSION)
        self.assertTrue(all("output_checks" not in attempt for attempt in legacy["attempts"]))
        self.assertFalse(synthesis._lesson(self.lesson, self.samples, legacy)["verified"])
        self.assertEqual(hashlib.sha256(synthesis.prompt_contract(synthesis.VERSION).encode()).hexdigest(),
                         "e28122119db3e3023bcb9b5fc08eeb940a72d65e71ac30c6303771d67c1b4778")


if __name__ == "__main__":
    unittest.main()
