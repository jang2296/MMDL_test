"""Submitted-fact locations are bounded extra verifier feedback, not corrected targets."""
import copy
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from fine_tuning.notebook_rl.episode_notebook import (
    FEEDBACK_POLICY, PROMPT, REINSPECTION_INSTRUCTION, STRICT_ROUTE_RULE,
    audit_episode, grade_note, notebook_feedback, solution_hash, validate_update_binding,
)
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import ProblemTree, SlotReport, from_json, to_json
from fine_tuning.notebook_rl.updates import effective_rewards, optimizer_step
from fine_tuning.tests import test_notebook_rl_episode_notebook as fixtures


class FactFeedbackTests(unittest.TestCase):
    setUp = fixtures.EpisodeNotebookTests.setUp
    tree = fixtures.EpisodeNotebookTests.tree
    append = fixtures.EpisodeNotebookTests.append

    def failed_note(self, *, facts=None):
        tree, engine = self.tree()
        audit = audit_episode(tree, self.actor, self.verifier)
        note = fixtures.authored_note(tree, audit)
        note["corrected_facts"] = facts if facts is not None else {"cell[a][b]": "8"}
        return tree, engine, audit, json.dumps(note)

    def bound_failure(self, *, text=None):
        tree, engine, _, bad = self.failed_note()
        original = solution_hash(tree)
        self.prompts["notebook_feedback_policy"] = FEEDBACK_POLICY
        self.prompts["notebook_route_rule"] = STRICT_ROUTE_RULE
        engine.replies.extend([[(bad if text is None else text)] * 4] * 3)
        self.append(tree, engine)
        self.assertEqual(solution_hash(tree), original)
        return tree, engine

    def test_feedback_adds_only_submitted_wrong_ids_and_preserves_binary_reward(self):
        tree, _, audit, text = self.failed_note()
        before = text
        legacy = grade_note(text, "stop", tree, audit, self.verifier)
        current = grade_note(text, "stop", tree, audit, self.verifier, feedback_policy=FEEDBACK_POLICY)
        self.assertEqual(legacy["reward"], 0)
        self.assertEqual(current["reward"], 0)
        self.assertEqual(current["checks"], legacy["checks"])
        self.assertEqual(current["attempt_checks"], legacy["attempt_checks"])
        self.assertEqual(current["submitted_claims"], legacy["submitted_claims"])
        self.assertEqual(current["reinspection_slots"], ["cell[a][b]"])
        self.assertEqual(text, before)
        record, engine = self.bound_failure()
        samples = [s for g in record.groups if g.kind == "DIAG" for s in g.samples]
        self.assertEqual(len(samples), 12)
        self.assertTrue(all(s.text == text and s.reward == 0 and s.result.status == "CORRECT" for s in samples))
        checked = record.notebook["model_notes"][0]
        feedback = notebook_feedback(checked, samples[0])
        self.assertTrue(feedback.startswith(REINSPECTION_INSTRUCTION))
        payload = json.loads(feedback[len(REINSPECTION_INSTRUCTION):])
        self.assertEqual(set(payload), {"failure_code", "checks", "attempt_checks", "remaining_claims_checked",
                                        "feedback_policy", "reinspection_slots"})
        self.assertEqual(payload["reinspection_slots"], ["cell[a][b]"])
        self.assertNotIn("corrected_answer", feedback)
        self.assertNotIn('"8"', feedback)
        self.assertNotIn('"7"', feedback)
        for row in payload["attempt_checks"]:
            self.assertTrue(all(type(value) is bool for key, value in row.items() if key != "attempt"))
        self.assertEqual(record, from_json(ProblemTree, to_json(record)))
        validate_update_binding(record, tokenizer=engine.tokenizer, feedback_policy=FEEDBACK_POLICY)

    def test_unsubmitted_missing_optional_and_unknown_ids_are_not_disclosed(self):
        for facts, expected in (({}, []), ({"cell[unknown][x]": "8"}, []), ({"cell[a][b]": "7"}, [])):
            tree, _, audit, text = self.failed_note(facts=facts)
            checked = grade_note(text, "stop", tree, audit, self.verifier, feedback_policy=FEEDBACK_POLICY)
            self.assertEqual(checked["reinspection_slots"], expected)
        tree, _ = self.bound_failure()
        checked = copy.deepcopy(tree.notebook["model_notes"][0])
        sample = tree.groups[-3].samples[0]
        # A synthetic optional error is not locatable in SlotReport; do not invent its ID.
        optional = replace(sample, result=replace(sample.result, slots=SlotReport(1, 0, 0, (), 0, 1, ())))
        checked["reinspection_slots"] = []
        self.assertEqual(json.loads(notebook_feedback(checked, optional)[len(REINSPECTION_INSTRUCTION):])
                         ["reinspection_slots"], [])
        conflict = replace(sample, result=replace(sample.result,
            slots=SlotReport(1, 1, 0, (), 0, 1, ("cell[a][b]", "private[not_submitted]"))))
        checked["reinspection_slots"] = ["cell[a][b]"]
        self.assertEqual(json.loads(notebook_feedback(checked, conflict)[len(REINSPECTION_INSTRUCTION):])
                         ["reinspection_slots"], ["cell[a][b]"])

    def test_malformed_notes_have_no_fact_receipt_or_hints(self):
        for text in ("{", "{}", "[]", '{"corrected_facts":{"cell[a][b]":"8","cell[a][b]":"7"}}'):
            tree, engine = self.bound_failure(text=text)
            samples = [s for g in tree.groups if g.kind == "DIAG" for s in g.samples]
            self.assertTrue(all(s.result is None and s.reward == 0 for s in samples))
            self.assertTrue(all(row["reinspection_slots"] == [] for row in tree.notebook["model_notes"]))
            validate_update_binding(tree, tokenizer=engine.tokenizer, feedback_policy=FEEDBACK_POLICY)

    def test_rehashed_hint_or_feedback_cannot_change_the_bound_fact_grade(self):
        tree, engine = self.bound_failure()
        for hint in ([], ["private[not_submitted]"], ["cell[a][b]", "cell[a][b]"],
                     ["Cell[a][b]"], [True], "cell[a][b]", {"cell[a][b]": "7"}):
            changed = copy.deepcopy(tree)
            changed.notebook["model_notes"][0]["reinspection_slots"] = hint
            record = changed.notebook["rewarded_notebook"]
            record["rounds"][1]["feedback"] += "forged feedback"
            record["rounds_sha256"] = _hash(record["rounds"])
            with self.subTest(hint=hint), self.assertRaises(ValueError):
                validate_update_binding(changed, tokenizer=engine.tokenizer, feedback_policy=FEEDBACK_POLICY)
        changed = copy.deepcopy(tree)
        changed.groups[-3].samples[0].result = None
        changed.notebook["rewarded_notebook"]["notebook_groups_sha256"] = _hash(
            [json.loads(to_json(g)) for g in changed.groups if g.kind == "DIAG"])
        with self.assertRaisesRegex(ValueError, "bound submitted-fact grade"):
            validate_update_binding(changed, feedback_policy=FEEDBACK_POLICY)
        with self.assertRaisesRegex(ValueError, "actual sample"):
            notebook_feedback(tree.notebook["model_notes"][0])

    def test_legacy_ancestry_stays_valid_but_cannot_be_relabelled_v6(self):
        tree, engine, _, text = self.failed_note()
        engine.replies.extend([[text] * 4] * 3)
        self.append(tree, engine)
        self.assertTrue(all(s.result is None for g in tree.groups if g.kind == "DIAG" for s in g.samples))
        validate_update_binding(tree, tokenizer=engine.tokenizer)
        with self.assertRaisesRegex(ValueError, "feedback policy missing"):
            validate_update_binding(tree, tokenizer=engine.tokenizer, feedback_policy=FEEDBACK_POLICY)
        changed = copy.deepcopy(tree)
        changed.notebook["rewarded_notebook"]["feedback_policy"] = FEEDBACK_POLICY
        with self.assertRaises(ValueError):
            validate_update_binding(changed, feedback_policy=FEEDBACK_POLICY)

    def test_optimizer_rejects_feedback_policy_downgrade_before_model_access(self):
        tree, engine = self.bound_failure()
        protocol = build_protocol()
        protocol["protocol_id"] = "local-nf4-full-history-notebook-v5"
        # This historical fixture targets feedback admission, before reread existed.
        for key in tuple(protocol):
            if key.startswith("notebook_observation_"):
                del protocol[key]
        del protocol["notebook_feedback_policy"]
        optimizer = Mock()
        with self.assertRaisesRegex(ValueError, "feedback policy missing or changed"):
            optimizer_step(SimpleNamespace(inventory={"arm": "MOD-RV"}), optimizer,
                SimpleNamespace(last_epoch=0), [tree], {tree.problem_id: []},
                SimpleNamespace(policy_version=0, tokenizer=engine.tokenizer), protocol,
                "ALG-NB3", step=0, seed=3407)
        optimizer.step.assert_not_called()

    def test_diag_fact_result_does_not_become_an_answer_reward_or_rate(self):
        tree, _ = self.bound_failure()
        legacy = copy.deepcopy(tree)
        for group in legacy.groups:
            if group.kind == "DIAG":
                for sample in group.samples:
                    sample.result = None
        for algorithm in ("ALG-NB3", "CTL-RAND", "CTL-FORMAT"):
            self.assertEqual(effective_rewards(tree.groups, algorithm, 3407, 0),
                             effective_rewards(legacy.groups, algorithm, 3407, 0))
        self.assertEqual(effective_rewards(tree.groups, "ALG-NB3", 3407, 0)[tree.groups[-1].group_id], [0.] * 4)

    def test_protocol_binds_policy_and_generic_precedence_without_private_labels(self):
        protocol = build_protocol()
        self.assertEqual(protocol["protocol_id"], "local-nf4-full-history-notebook-v13")
        self.assertEqual(local_prompts(protocol)["notebook_feedback_policy"], FEEDBACK_POLICY)
        for value in (None, "legacy", {}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                local_prompts(protocol | {"notebook_feedback_policy": value})
        for phrase in ("CORRECT takes priority", "at least half", "VISUAL_RELATION takes priority", "all required"):
            self.assertIn(phrase, PROMPT)


if __name__ == "__main__":
    unittest.main()
