"""Authored CPU histories verify evaluation contracts, never model learning."""
import copy
import json
import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import synthesis_credit, whole_reference_notebook as reference
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import Group, GroupKind, ProblemTree, Sample
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures
from fine_tuning.tests.test_notebook_rl_whole_reference_notebook import compact


class EvaluationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actor, self.item, self.verifier = fixtures.native()
        self.wrong = fixtures.response(self.item.slots)
        self.right = fixtures.response(self.item.slots, wrong=False)
        self.tree = self.evaluation_tree([self.wrong, self.wrong, self.right])
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier,
            contract=synthesis.BOUNDED_VERSION, evaluation_only=True)
        self.note = fixtures.note_for(self.tree, self.audit, self.item)

    def evaluation_tree(self, texts: list[str]) -> ProblemTree:
        groups: list[Group] = []
        kinds: tuple[GroupKind, ...] = ("T1", "T2", "T3")
        for turn, text in enumerate(texts, 1):
            parent = groups[-1].samples[0] if groups else None
            prompt = parent.prompt_token_ids + parent.completion_token_ids + [900 + turn] if parent else [1, 2]
            kind = kinds[turn - 1]
            sample = Sample(f"evaluation-{turn}", self.actor.problem_id, kind, kind, turn,
                parent.sample_id if parent else None, prompt, [100 + turn], [-.1], "stop", text, 3407 + turn, 0,
                self.verifier.grade(self.actor.problem_id, text, "stop", self.actor.mode))
            groups.append(Group(kind, kind, self.actor.problem_id, _hash(prompt), [sample],
                                False, "EVALUATION_SINGLE_SAMPLE"))
        return ProblemTree(self.actor.problem_id, 0, groups, "VISION_OFF", {}, {})

    def grade(self, *, audit: dict | None = None, evaluation_only: bool = True) -> dict:
        return synthesis.grade_note(json.dumps(self.note), "stop", self.tree,
            self.audit if audit is None else audit, self.verifier, actor=self.actor,
            contract=synthesis.BOUNDED_VERSION, evaluation_only=evaluation_only)

    def test_explicit_evaluation_preserves_training_rejection_and_strict_note(self) -> None:
        raw = episode.audit_episode(self.tree, self.actor, self.verifier,
            routing_rule=episode.STRICT_ROUTE_RULE, evaluation_only=True)
        default = episode.audit_episode(self.tree, self.actor, self.verifier,
            routing_rule=episode.STRICT_ROUTE_RULE)
        expected = default | {"training_complete_episode": False, "training_admission": False,
            "complete_episode": True, "success_attempt": 3, "evaluation_only_single_sample_groups": True}
        self.assertEqual(raw, expected)
        for field in ("training_complete_episode", "training_admission", "evaluation_only_single_sample_groups"):
            self.assertEqual(self.audit[field], raw[field])
        self.assertEqual(self.grade()["reward"], 1.)
        self.assertEqual(self.grade()["failure_code"], "NONE")
        self.assertTrue(all(not group.usable for group in self.tree.groups))
        training = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.assertFalse(training["complete_episode"])
        self.assertIsNone(training["success_attempt"])
        for audit in (training, self.audit):
            with self.subTest(audit=audit["complete_episode"]), self.assertRaises(ValueError):
                self.grade(audit=audit, evaluation_only=False)

    def test_reference_grading_requires_explicit_mode_and_keeps_raw_identity(self) -> None:
        text = json.dumps(compact(episode.solution_samples(self.tree), self.note))
        checked = reference.grade_note(text, "stop", self.tree, self.audit, self.verifier,
            actor=self.actor, contract=synthesis.BOUNDED_VERSION, evaluation_only=True)
        self.assertEqual(checked["reward"], 1.)
        self.assertEqual(checked["rendered_strict_report"]["episode_audit_sha256"], _hash(self.audit))
        self.assertEqual(checked["note_text_sha256"], reference._sha(text))
        with self.assertRaises(ValueError):
            reference.grade_note(text, "stop", self.tree, self.audit, self.verifier,
                actor=self.actor, contract=synthesis.BOUNDED_VERSION)

    def test_legacy_grading_reaudits_explicit_mode_and_rejects_stripped_markers(self) -> None:
        audit = episode.audit_episode(self.tree, self.actor, self.verifier,
            routing_rule=episode.STRICT_ROUTE_RULE, evaluation_only=True)
        samples = episode.solution_samples(self.tree)
        final = samples[-1].result
        assert final is not None
        note = {"corrected_facts": self.item.slots, "success_attempt": 3, "corrected_answer": final.extracted,
            "lesson": "Check each actual observation and retry.", "attempts": [
                {"attempt": row["attempt"], "answer": sample.result.extracted if sample.result else None,
                 "line_ranges": [[1, len(sample.text.splitlines())]], "cause": row["cause"],
                 "wrong_slots": row["wrong_slots"]} for row, sample in zip(audit["attempts"], samples)]}
        text = json.dumps(note)
        self.assertEqual(episode.grade_note(text, "stop", self.tree, audit, self.verifier,
            actor=self.actor, evaluation_only=True)["reward"], 1.)
        for supplied in (audit, {key: value for key, value in audit.items()
                                if key not in ("training_complete_episode", "training_admission",
                                               "evaluation_only_single_sample_groups")}):
            with self.assertRaises(ValueError):
                episode.grade_note(text, "stop", self.tree, supplied, self.verifier)
        with self.assertRaises(ValueError):
            episode.grade_note(text, "stop", self.tree, audit, self.verifier, evaluation_only=True)
        forged = audit | {"success_attempt": 2}
        with self.assertRaises(ValueError):
            episode.grade_note(text, "stop", self.tree, forged, self.verifier,
                               actor=self.actor, evaluation_only=True)

    def test_complete_one_two_and_three_attempt_sequences(self) -> None:
        for texts, success in (([self.right], None), ([self.wrong, self.right], 2),
                               ([self.wrong] * 3, None)):
            with self.subTest(count=len(texts), success=success):
                audit = episode.audit_episode(self.evaluation_tree(texts), self.actor, self.verifier,
                    evaluation_only=True)
                self.assertTrue(audit["complete_episode"])
                self.assertIs(audit["training_admission"], False)
                self.assertEqual(audit["success_attempt"], success)
        for count in (1, 2):
            with self.subTest(omitted_retry=count), self.assertRaisesRegex(ValueError, "coverage incomplete"):
                episode.audit_episode(self.evaluation_tree([self.wrong] * count), self.actor, self.verifier,
                                      evaluation_only=True)

    def test_evaluation_group_shape_state_and_coverage_fail_closed(self) -> None:
        changes: dict[str, dict[str, Any]] = {
            "mixed_training_group": {"usable": True, "reason": None,
                "samples": [self.tree.groups[0].samples[0], replace(self.tree.groups[0].samples[0], sample_id="sibling")]},
            "wrong_reason": {"reason": "OTHER"}, "wrong_state": {"state_sha256": "b" * 64},
            "empty_group": {"samples": []}, "wrong_group_kind": {"kind": "DIAG"},
            "wrong_problem": {"problem_id": "other"}, "wrong_sample_identity": {
                "samples": [replace(self.tree.groups[0].samples[0], group_id="other")]},
            "wrong_sample_kind": {"samples": [replace(self.tree.groups[0].samples[0], kind="T2")]},
        }
        for label, fields in changes.items():
            tree = copy.deepcopy(self.tree)
            tree.groups[0] = replace(tree.groups[0], **fields)
            with self.subTest(label=label), self.assertRaises(ValueError):
                episode.audit_episode(tree, self.actor, self.verifier, evaluation_only=True)
        for label in ("skipped", "reordered", "duplicate", "empty", "too_many"):
            tree = copy.deepcopy(self.tree)
            if label == "skipped":
                tree.notebook["skipped_branches"] = [{"kind": "T3", "reason": "CONTEXT_BLOCKED"}]
            elif label == "reordered":
                tree.groups.reverse()
            elif label == "duplicate":
                tree.groups[1] = copy.deepcopy(tree.groups[0])
            elif label == "empty":
                tree.groups = []
            else:
                tree.groups.append(copy.deepcopy(tree.groups[-1]))
            with self.subTest(label=label), self.assertRaises(ValueError):
                episode.audit_episode(tree, self.actor, self.verifier, evaluation_only=True)

    def test_parent_tokens_and_outcomes_still_receive_fresh_validation(self) -> None:
        for label in ("parent", "tokens", "outcome"):
            tree = copy.deepcopy(self.tree)
            sample = tree.groups[1].samples[0]
            assert sample.result is not None
            if label == "parent":
                sample.parent_sample_id = "missing-parent"
            elif label == "tokens":
                sample.prompt_token_ids[0] += 1
                tree.groups[1].state_sha256 = _hash(sample.prompt_token_ids)
            else:
                sample.result = replace(sample.result, extracted="forged")
            with self.subTest(label=label), self.assertRaises(ValueError):
                episode.audit_episode(tree, self.actor, self.verifier, evaluation_only=True)

    def test_blocking_outcomes_remain_blocking(self) -> None:
        tree = self.evaluation_tree([self.wrong])
        sample = tree.groups[0].samples[0]
        assert sample.result is not None
        for status in ("UNVERIFIABLE", "CONTEXT_BLOCKED", "SYSTEM_ERROR"):
            sample.result = replace(sample.result, status=status, extracted=None)
            with self.subTest(status=status), patch.object(self.verifier, "grade", return_value=sample.result), \
                    patch.object(self.verifier, "cause", return_value="UNKNOWN"), \
                    self.assertRaisesRegex(ValueError, "unverifiable evaluation"):
                episode.audit_episode(tree, self.actor, self.verifier, evaluation_only=True)

    def test_forged_evaluation_audit_and_training_credit_are_rejected(self) -> None:
        for forged in (self.audit | {"success_attempt": 2}, self.audit | {"training_admission": True},
                       self.audit | {"complete_episode": False}):
            with self.subTest(forged=forged["success_attempt"]), self.assertRaises(ValueError):
                self.grade(audit=forged)
        with self.assertRaisesRegex(ValueError, "source audit differs"):
            synthesis_credit.score_note(json.dumps(self.note), "stop", self.tree, self.audit,
                                        self.verifier, actor=self.actor)
        with self.assertRaises(ValueError):
            self.grade(audit={key: value for key, value in self.audit.items()
                              if key != "evaluation_only_single_sample_groups"})

    def test_unsupported_native_contract_is_not_enabled_by_evaluation(self) -> None:
        self.verifier.items[self.actor.problem_id] = replace(self.item, accept={})
        audit = synthesis.audit_episode(self.tree, self.actor, self.verifier,
            contract=synthesis.BOUNDED_VERSION, evaluation_only=True)
        self.assertFalse(audit["unknown_regions_accountable"])
        self.assertTrue(any(row["rule"] == "NATIVE_CONTRACT" for attempt in audit["attempts"]
                            for row in attempt["unverified"]))
        note = fixtures.note_for(self.tree, audit, self.item)
        checked = synthesis.grade_note(json.dumps(note), "stop", self.tree, audit, self.verifier,
            actor=self.actor, contract=synthesis.BOUNDED_VERSION, evaluation_only=True)
        self.assertEqual(checked["reward"], 0.)

    def test_default_training_fixture_and_prompts_are_unchanged(self) -> None:
        tree = fixtures.tree_from(self.actor, self.verifier, [self.wrong] * 4 + [self.right] * 2)
        audit = synthesis.audit_episode(tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.assertEqual(audit, synthesis.audit_episode(tree, self.actor, self.verifier,
            contract=synthesis.BOUNDED_VERSION, evaluation_only=False))
        note = fixtures.note_for(tree, audit, self.item)
        checked = synthesis.grade_note(json.dumps(note), "stop", tree, audit, self.verifier,
            actor=self.actor, contract=synthesis.BOUNDED_VERSION)
        expected = {
            "audit": "de52de38f0bbf5fc5ce7f6c8bf2021a640165323c90ece3489d12d8742a18bdd",
            "grade": "91030a594ab4c2c592582a2dedacfbdc03e2bebd8e866a63a76afc6647ccbc5b",
            "synthesis_prompt": "69c1477d6c4dd37e2f164f7b55f91db89deef49960a3ec7e91cb5630d85e3b9e",
            "episode_prompt": "ad8fecc1b2e61b6f4b9ef9ddca71e4277832854661676b90d61b1245134dcdee",
            "reference_prompt": "2d5e14c8611298fd7f912ed691204b05000aeed0ef617b822607ce6760f65aae",
        }
        actual = {"audit": _hash(audit), "grade": _hash(checked),
            "synthesis_prompt": _hash(synthesis.instruction(episode.solution_samples(tree), audit, "PUBLIC_SCHEMA")),
            "episode_prompt": _hash(episode.PROMPT), "reference_prompt": _hash(reference.prompt_contract())}
        self.assertEqual(actual, expected)
        self.assertEqual(checked["reward"], 1.)
        with self.assertRaises(ValueError):
            synthesis.audit_episode(tree, self.actor, self.verifier, evaluation_only=True)


if __name__ == "__main__":
    unittest.main()
