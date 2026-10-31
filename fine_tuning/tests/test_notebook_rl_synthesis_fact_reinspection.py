"""Submitted-fact feedback boundaries; authored checks never establish model improvement."""
import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl.checkpoint import digest
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts, restore_protocol
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures


class SynthesisFactReinspectionTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        self.tree = fixtures.tree_from(self.actor, self.verifier,
            [fixtures.response(self.item.slots)] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.note = fixtures.note_for(self.tree, self.audit, self.item)

    def grade(self, note, finish="stop"):
        text = note if type(note) is str else json.dumps(note)
        checked = synthesis.grade_note(text, finish, self.tree, self.audit, self.verifier,
                                      actor=self.actor, contract=synthesis.BOUNDED_VERSION)
        sample = replace(solution_samples(self.tree)[0], text=text, finish_reason=finish,
                         reward=checked["reward"], result=None)
        return checked, sample

    @staticmethod
    def payload(checked, sample):
        return json.loads(synthesis.notebook_feedback(checked, sample).split("Verifier feedback:\n", 1)[1])

    def test_only_current_submitted_wrong_required_facts_supply_canonical_ids(self):
        for facts, expected in ((self.item.slots, []), ({"accepting": "q0,q1,q2,q3"}, ["accepting"]),
                                ({" START ": "q3"}, ["start"]), ({"start": "q0"}, []),
                                ({"unsupported": "q3"}, []), ({}, [])):
            with self.subTest(facts=facts):
                checked, sample = self.grade({"corrected_facts": facts})
                self.assertEqual(checked["failure_code"], "INVALID_ENVELOPE")
                self.assertEqual(self.payload(checked, sample)["reinspection_slots"], expected)
                receipt = checked["fact_reinspection"]
                self.assertEqual(receipt["episode_audit_sha256"], checked["episode_audit_sha256"])
                if receipt["result"]:
                    self.assertEqual(receipt["result"]["status"], "INVALID_NO_FINAL")
                    self.assertIsNone(receipt["result"]["extracted"])

    def test_invalid_siblings_do_not_prevent_independent_scalar_checks_or_create_values(self):
        note = copy.deepcopy(self.note)
        note["corrected_facts"] = {"trans": {"q0": {"0": "q3"}}, "accepting": "q0,q1,q2,q3"}
        checked, sample = self.grade(note)
        self.assertEqual((checked["failure_code"], checked["reward"], checked["checks"]), ("INVALID_FIELDS", 0., {}))
        self.assertEqual(checked["fact_reinspection"]["facts"], {"accepting": "q0,q1,q2,q3"})
        self.assertEqual(self.payload(checked, sample)["reinspection_slots"], ["accepting"])
        note["corrected_facts"] = {"trans": {"q0": {"0": "q3"}}}
        checked, sample = self.grade(note)
        self.assertEqual(self.payload(checked, sample)["reinspection_slots"], [])

    def test_malformed_unknown_null_non_scalar_and_canonical_collisions_disclose_nothing(self):
        cases = [{"start": None}, {"start": 3}, {"start": True}, {"start": ["q3"]},
                 {"start": "q3\naccepting = q0"}, {"start": "q3\r"}, {"start": "q3\u2028"},
                 {"trans[q0,0]": "q3 because"}, {"start": "?"}, {"start=accepting": "q3"},
                 {"accepting": "q3 because it is circled"}, {"accepting": '["q3"]'},
                 {"accepting": "I do not know"}, {"start": "q 3"}, {"start": "q3,q0"},
                 {"trans[q0,0]": "q3->q2"},
                 {"start": "q3", " START ": "q0"}, {"start": "q3", " START ": {}},
                 "start = q3", None]
        for facts in cases:
            with self.subTest(facts=facts):
                checked, sample = self.grade({"corrected_facts": facts})
                self.assertEqual(self.payload(checked, sample)["reinspection_slots"], [])
        for text in ('{"corrected_facts":{"start":"q3"}',
                     '{"corrected_facts":{"start":"q3","start":"q0"}}', "["):
            checked, sample = self.grade(text)
            self.assertEqual(checked["failure_code"], "INVALID_JSON")
            self.assertEqual(self.payload(checked, sample)["reinspection_slots"], [])
        checked, sample = self.grade({"corrected_facts": {"start": "q3"}}, finish="length")
        self.assertEqual(self.payload(checked, sample)["reinspection_slots"], [])

    def test_proven_absent_optional_relations_and_unparsed_numeric_values_are_not_ids(self):
        actor, item, verifier = fixtures.graph_native("bfs")
        for value in ("1", "1 unit", "not a number", None):
            with self.subTest(value=value):
                note = {"corrected_facts": {"edge[A,H]": value}}
                record = {"note_text_sha256": synthesis._sha(json.dumps(note)), "note_finish_reason": "stop",
                          "episode_audit_sha256": "a" * 64}
                receipt = synthesis._fact_reinspection(note, record, actor.problem_id, verifier)
                checked = record | {"fact_reinspection": receipt}
                self.assertEqual(synthesis._reinspection_feedback(checked, None), [])
                if value == "1":
                    self.assertEqual(receipt["result"]["slots"]["false_relations"], ["edge[a,h]"])
        self.assertNotIn("edge[A,H]", item.slots)

    def test_feedback_never_contains_values_results_expected_inventory_or_teacher_prose(self):
        checked, sample = self.grade({"corrected_facts": {"start": "q3"}})
        checked["private_reference"] = "SECRET_EXPECTED_VALUE_OR_TEACHER_TEXT"
        payload = self.payload(checked, sample)
        self.assertEqual(payload["reinspection_slots"], ["start"])
        rendered = json.dumps(payload)
        for private in ("q0", "q3", "accepting", "trans[", "SECRET_EXPECTED", "wrong_required",
                        "required_correct", "verifier_version", "fact_reinspection", "episode_audit_sha256"):
            self.assertNotIn(private, rendered)
        self.assertIn("not per-attempt causes or expected values", synthesis.RETRY_INSTRUCTION)

    def test_feedback_requires_actual_sample_and_rejects_mismatch_or_receipt_tampering(self):
        checked, sample = self.grade({"corrected_facts": {"start": "q3"}})
        with self.assertRaisesRegex(ValueError, "requires its actual sample"):
            synthesis.notebook_feedback(checked)
        for replacement in ({"text": "{}"}, {"finish_reason": "length"}, {"reward": 1.}, {"problem_id": "other"}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.payload(checked, replace(sample, **replacement))
        for field, value in (("facts", {"start": "q0"}), ("projection_sha256", "f" * 64),
                             ("episode_audit_sha256", "f" * 64), ("result", None)):
            forged = copy.deepcopy(checked)
            forged["fact_reinspection"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.payload(forged, sample)
        forged = copy.deepcopy(checked)
        receipt = forged["fact_reinspection"]
        receipt["facts"] = {"accepting": "q0"}
        receipt["projection_sha256"] = synthesis._sha(synthesis._reinspection_projection(receipt["facts"]))
        receipt["sha256"] = _hash({key: value for key, value in receipt.items() if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "differs from actual sample"):
            self.payload(forged, sample)

    def test_feedback_receipt_roundtrip_preserves_original_scalar_spellings(self):
        checked, sample = self.grade({"corrected_facts": {"trans[q0,0]": "q3", " START ": "\tq3 "}})
        restored = json.loads(json.dumps(checked, sort_keys=True))
        self.assertEqual(self.payload(restored, sample)["reinspection_slots"], ["start", "trans[q0,0]"])
        self.assertEqual(restored["fact_reinspection"]["facts"][" START "], "\tq3 ")

    def test_only_feedback_policy_instruction_and_protocol_hash_change(self):
        current = build_protocol(verified_note_synthesis=True, context_limit=24576)
        current_prompts = local_prompts(current)
        with patch.object(synthesis, "FEEDBACK_POLICY", "whole-history-synthesis-public-output-relationships-and-coverage-v4"), \
                patch.object(synthesis, "RETRY_INSTRUCTION", "old retry instruction"):
            previous = build_protocol(verified_note_synthesis=True, context_limit=24576)
            self.assertEqual(build_protocol()["notebook_feedback_policy"], "submitted-fact-reinspection-slots-v1")
        self.assertEqual({key for key in current if current[key] != previous[key]},
                         {"notebook_feedback_policy", "notebook_retry_instruction_sha256", "sha256"})
        self.assertEqual(current["notebook_retry_instruction_sha256"], digest(synthesis.RETRY_INSTRUCTION))
        self.assertEqual(restore_protocol(current), current)
        self.assertEqual(current_prompts["notebook_feedback_policy"], synthesis.FEEDBACK_POLICY)
        previous["sha256"] = digest({key: value for key, value in previous.items() if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "saved protocol changed"):
            restore_protocol(previous)


if __name__ == "__main__":
    unittest.main()
