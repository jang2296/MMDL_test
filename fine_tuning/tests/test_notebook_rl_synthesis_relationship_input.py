"""Public relationship procedure and empty-list feedback; no new semantic targets."""
import copy
import hashlib
import json
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl.episode_notebook import public_transcript, solution_samples
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures


class SynthesisRelationshipInputTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        texts = [fixtures.response(self.item.slots)] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2
        self.tree = fixtures.tree_from(self.actor, self.verifier, texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.note = fixtures.note_for(self.tree, self.audit, self.item)

    def grade(self, note):
        return synthesis.grade_note(json.dumps(note), "stop", self.tree, self.audit, self.verifier,
                                    actor=self.actor, contract=synthesis.BOUNDED_VERSION)

    @staticmethod
    def feedback(report, sample=None):
        return json.loads(synthesis.notebook_feedback(report, sample).split("Verifier feedback:\n", 1)[1])

    def test_count_is_only_the_submitted_list_length_even_when_malformed(self):
        for text, count in (('{"lessons":[]}', 0), ('{"lessons":[{},null,2]}', 3),
                            ('{"lessons":"[]"}', None), ('{}', None), ('[]', None), ('{', None),
                            ('{"lessons":[],"lessons":[1]}', None)):
            with self.subTest(text=text):
                diagnostics = synthesis.format_diagnostics(text)
                self.assertEqual(diagnostics["submitted_lesson_count"], count)
                self.assertTrue(synthesis._valid_format_diagnostics(diagnostics))

    def test_empty_list_is_not_outgoing_verified_lesson_evidence(self):
        note = copy.deepcopy(self.note)
        note["lessons"] = []
        checked = self.grade(note)
        before = copy.deepcopy(checked)
        self.assertTrue(checked["checks"]["lessons_verified"])
        self.assertFalse(checked["checks"]["all_issues_synthesized"])
        payload = self.feedback(checked)
        self.assertNotIn("lessons_verified", payload["checks"])
        self.assertIsNone(payload["checks"]["submitted_lessons_verified"])
        self.assertFalse(payload["checks"]["all_issues_synthesized"])
        self.assertEqual(payload["failed_lessons"], [])
        self.assertEqual(payload["format_diagnostics"]["submitted_lesson_count"], 0)
        self.assertIn("no verified lesson evidence", synthesis.notebook_feedback(checked))
        self.assertEqual(checked, before)

    def test_nonempty_checked_and_unchecked_lists_are_distinguished(self):
        note = copy.deepcopy(self.note)
        note["corrected_facts"]["start"] = "q3"
        sample = replace(solution_samples(self.tree)[0], text=json.dumps(note), reward=0.)
        payload = self.feedback(self.grade(note), sample)
        self.assertTrue(payload["checks"]["submitted_lessons_verified"])
        self.assertEqual(payload["format_diagnostics"]["submitted_lesson_count"], len(note["lessons"]))
        note["lessons"].append("unsupported submitted prose")
        sample = replace(sample, text=json.dumps(note))
        payload = self.feedback(self.grade(note), sample)
        self.assertFalse(payload["checks"]["submitted_lessons_verified"])
        self.assertEqual(payload["failed_lessons"][-1], len(note["lessons"]))
        note["corrected_facts"] = "malformed container"
        payload = self.feedback(self.grade(note))
        self.assertIsNone(payload["checks"]["submitted_lessons_verified"])
        self.assertEqual(payload["format_diagnostics"]["submitted_lesson_count"], len(note["lessons"]))

    def test_forged_or_badly_typed_counts_are_rejected(self):
        note = copy.deepcopy(self.note)
        note["lessons"] = []
        checked = self.grade(note)
        sample = replace(solution_samples(self.tree)[0], text=json.dumps(note), reward=0.)
        for count in (1, -1, True, 0.0):
            bad = copy.deepcopy(checked)
            bad["format_diagnostics"]["submitted_lesson_count"] = count
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "format diagnostics differ"):
                synthesis.notebook_feedback(bad, sample)

    def test_every_public_parent_pair_and_raw_line_is_retained_without_filtering(self):
        samples = solution_samples(self.tree)
        public = public_transcript(samples)
        prompt = synthesis.instruction(samples, self.audit, "PUBLIC_SCHEMA")
        pairs = json.loads(prompt.split("<public_retry_pairs>\n", 1)[1].split("\n</public_retry_pairs>", 1)[0])
        self.assertEqual(pairs, [[row["parent"], row["attempt"]] for row in public if row["parent"] is not None])
        self.assertEqual(pairs, [[1, 3], [1, 4], [3, 5], [3, 6]])
        rows = json.loads(prompt.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
        self.assertEqual(["".join(text for _, text in row["source_lines"]) for row in rows], [sample.text for sample in samples])
        changed = copy.deepcopy(self.audit)
        changed["private_reference"] = "DO_NOT_SUPPLY_TARGETS"
        self.assertEqual(synthesis.instruction(samples, changed, "PUBLIC_SCHEMA"), prompt)
        self.assertIn('"lessons":[', synthesis.PUBLIC_JSON_SHAPE)
        self.assertLess(synthesis.PUBLIC_JSON_SHAPE.index('"lessons"'), synthesis.PUBLIC_JSON_SHAPE.index('"corrected_facts"'))

    def test_parent_without_final_remains_in_the_public_comparisons(self):
        texts = [sample.text for sample in solution_samples(self.tree)]
        texts[0] = texts[0].rsplit("\nFinal answer:", 1)[0].replace("\n", "\r\n")
        tree = fixtures.tree_from(self.actor, self.verifier, texts)
        audit = synthesis.audit_episode(tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        prompt = synthesis.instruction(solution_samples(tree), audit, "PUBLIC_SCHEMA")
        pairs = json.loads(prompt.split("<public_retry_pairs>\n", 1)[1].split("\n</public_retry_pairs>", 1)[0])
        self.assertIn([1, 3], pairs)
        self.assertIn([1, 4], pairs)
        self.assertIn("A parent without a final answer", prompt)
        self.assertIn("abstain honestly", prompt)
        rows = json.loads(prompt.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
        self.assertEqual("".join(text for _, text in rows[0]["source_lines"]), texts[0])

    def test_legacy_prompt_and_feedback_bytes_remain_fixed(self):
        self.assertEqual(hashlib.sha256(synthesis.prompt_contract(synthesis.VERSION).encode()).hexdigest(),
                         "e28122119db3e3023bcb9b5fc08eeb940a72d65e71ac30c6303771d67c1b4778")
        checked = {"reward_version": synthesis.VERSION, "reward": 0., "failure_code": "INVALID_JSON",
                   "checks": {}, "attempt_checks": [], "lesson_checks": []}
        feedback = synthesis.notebook_feedback(checked)
        self.assertEqual(hashlib.sha256(feedback.encode()).hexdigest(),
                         "8dd48a630f02b5247309749478202e59bc1967de19fec634780e3e342f46962f")
        self.assertNotIn("submitted_lesson_count", feedback)
        self.assertNotIn("public_retry_pairs", synthesis.prompt_contract(synthesis.VERSION))


if __name__ == "__main__":
    unittest.main()
