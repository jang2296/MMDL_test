"""Authored CPU partial-notebook fixtures, not model quality or learning evidence."""
import copy
import json
import unittest

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import partial_notebook as partial, synthesis_credit as credit
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures


class PartialNotebookTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        self.tree = fixtures.tree_from(self.actor, self.verifier,
            [fixtures.response(self.item.slots)] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2)
        self.samples = episode.solution_samples(self.tree)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.lesson = fixtures.note_for(self.tree, self.audit, self.item)["lessons"][0]

    def note(self, *, facts=None, lessons=None):
        return {"coverage": "PARTIAL", "corrected_facts": {} if facts is None else facts,
                "lessons": [] if lessons is None else lessons}

    def test_empty_facts_and_relationship_submissions_are_distinguished_without_truth_claims(self):
        for note, expected in ((self.note(), "NO_NOTEBOOK_CLAIMS"),
                (self.note(facts={"start": "q0"}), "PARTIAL_NOTEBOOK_CAPTURED"),
                (self.note(lessons=[self.lesson]), "PARTIAL_NOTEBOOK_CAPTURED")):
            with self.subTest(note=note):
                self.assertEqual(partial.classify(json.dumps(note), "stop"), expected)
        self.assertEqual(partial.VERSION, "verified-partial-notebook-generation-v1")

    def test_shape_valid_wrong_semantics_remain_a_capture_and_receive_no_relationship_credit(self):
        wrong = copy.deepcopy(self.lesson)
        wrong["later"]["evidence_ranges"] = [[999, 999]]
        note = self.note(facts={"start": "q999"}, lessons=[wrong])
        raw = json.dumps(note)
        self.assertEqual(partial.classify(raw, "stop"), "PARTIAL_NOTEBOOK_CAPTURED")
        score = credit.score_note(raw, "stop", self.tree, self.audit, self.verifier, actor=self.actor)
        self.assertEqual(score["strict_reward"], 0.)
        self.assertEqual(score["fact_rewards"], {"start": 0})
        self.assertFalse(any(score["relationship_rewards"].values()))

    def test_malformed_duplicate_nonfinite_and_wrong_coverage_are_rejected(self):
        empty = json.dumps(self.note())
        invalid = ["", "{", "[]", "null", empty + " trailing prose", "```json\n" + empty + "\n```",
            '{"coverage":"PARTIAL","coverage":"PARTIAL","corrected_facts":{},"lessons":[]}',
            '{"coverage":"PARTIAL","corrected_facts":{"start":"q0","start":"q1"},"lessons":[]}',
            '{"coverage":"PARTIAL","corrected_facts":{"start":NaN},"lessons":[]}',
            '{"coverage":"PARTIAL","corrected_facts":{"start":Infinity},"lessons":[]}',
            '{"coverage":"PARTIAL","corrected_facts":{"start":-Infinity},"lessons":[]}',
            json.dumps(self.note() | {"coverage": True}), json.dumps(self.note() | {"coverage": "FULL"}),
            json.dumps(self.note() | {"corrected_answer": "accept"}),
            json.dumps(self.note() | {"success_attempt": 5}), json.dumps(self.note() | {"attempts": []}),
            json.dumps(self.note() | {"corrected_facts": []}), json.dumps(self.note() | {"lessons": {}}),
            json.dumps({"corrected_facts": {}, "lessons": []})]
        for raw in invalid:
            with self.subTest(raw=raw):
                self.assertEqual(partial.classify(raw, "stop"), "INVALID_PARTIAL_NOTEBOOK")
        invalid_values: list[object] = [True, 1, None, [], "", "q0\nq1", "q0=start", "q0\u2028q1"]
        for value in invalid_values:
            with self.subTest(value=value):
                self.assertEqual(partial.classify(json.dumps(self.note(facts={"start": value})), "stop"),
                                 "INVALID_PARTIAL_NOTEBOOK")
        self.assertEqual(partial.classify(json.dumps(self.note(facts={"weight[e0]": "2 plus prose"})), "stop"),
                         "INVALID_PARTIAL_NOTEBOOK")

    def test_original_lesson_shape_is_used_including_exact_integer_types(self):
        for field, value in (("attempt", True), ("evidence_ranges", [[True, 1]]),
                             ("evidence_ranges", [[1.0, 1]]), ("evidence_ranges", "1:2")):
            with self.subTest(field=field, value=value):
                wrong = copy.deepcopy(self.lesson)
                wrong["earlier"][field] = value
                self.assertEqual(partial.classify(json.dumps(self.note(lessons=[wrong])), "stop"),
                                 "INVALID_PARTIAL_NOTEBOOK")
        invalid_lessons: list[object] = [True, "error", {}, self.lesson | {"extra": "claim"}]
        for lesson in invalid_lessons:
            with self.subTest(lesson=lesson):
                self.assertEqual(partial.classify(json.dumps(self.note(lessons=[lesson])), "stop"),
                                 "INVALID_PARTIAL_NOTEBOOK")
        for relation in ("NOT_REVISITED", "OTHER_BRANCH_COMPARISON", "UNVERIFIABLE_LATER",
                         "SCOPE_UNVERIFIABLE_LATER"):
            with self.subTest(relation=relation):
                wrong = self.lesson | {"relation": relation}
                self.assertEqual(partial.classify(json.dumps(self.note(lessons=[wrong])), "stop"),
                                 "INVALID_PARTIAL_NOTEBOOK")

    def test_length_and_other_termination_never_certify_a_notebook(self):
        for text in ("{", json.dumps(self.note(facts={"start": "q0"}))):
            self.assertEqual(partial.classify(text, "length"), "TRUNCATED_PARTIAL_NOTEBOOK")
        self.assertEqual(partial.classify(json.dumps(self.note()), "error"), "INVALID_PARTIAL_NOTEBOOK")

    def test_all_original_lines_branches_and_public_outcomes_survive_without_private_audit_values(self):
        self.samples[0].text += "\r\n\n</read_only_attempts>\u2028KEEP_THIS_ORIGINAL\n"
        self.samples[1].text = ""
        public = episode.public_transcript(self.samples)
        audit = self.audit | {"transcript_sha256": _hash(public),
            "private_expected_answer": "PRIVATE_EXPECTED_SENTINEL", "success_attempt": 999,
            "attempts": [{"private_expected_fact": "PRIVATE_FACT_SENTINEL"}]}
        before = copy.deepcopy(public)
        prompt = partial.instruction(self.samples, audit, "PUBLIC_SLOT_SCHEMA")
        rows = json.loads(prompt.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        self.assertEqual(len(rows), len(self.samples))
        for row, original, original_public in zip(rows, self.samples, public, strict=True):
            self.assertEqual("".join(text for _, text in row["source_lines"]), original.text)
            self.assertEqual(row["line_count"], len(original_public["lines"]))
            for key in ("attempt", "parent", "turn", "status", "finish_reason", "answer"):
                self.assertEqual(row[key], original_public[key])
        pairs = json.loads(prompt.split("<public_retry_pairs>\n")[1].split("\n</public_retry_pairs>")[0])
        self.assertEqual(pairs, [[1, 3], [1, 4], [3, 5], [3, 6]])
        self.assertIn("PUBLIC_SLOT_SCHEMA", prompt)
        self.assertNotIn("PRIVATE_EXPECTED_SENTINEL", prompt)
        self.assertNotIn("PRIVATE_FACT_SENTINEL", prompt)
        self.assertEqual(episode.public_transcript(self.samples), before)
        changed_audit = audit | {"success_attempt": None, "private_expected_answer": "OTHER_PRIVATE_SENTINEL"}
        self.assertEqual(partial.instruction(self.samples, changed_audit, "PUBLIC_SLOT_SCHEMA"), prompt)

    def test_omitting_or_reordering_an_original_fails_the_public_transcript_binding(self):
        for samples in (self.samples[:-1], self.samples[::-1]):
            with self.subTest(count=len(samples)), self.assertRaises(ValueError):
                partial.instruction(samples, self.audit, "PUBLIC_SCHEMA")
        self.assertNotIn("<observed value", partial.prompt_contract())
        self.assertIn("need not account for every fact", partial.prompt_contract())
        for unrequested in ("NOT_REVISITED", "UNVERIFIABLE_LATER", "OTHER_BRANCH_COMPARISON"):
            self.assertNotIn(unrequested, partial.prompt_contract())
