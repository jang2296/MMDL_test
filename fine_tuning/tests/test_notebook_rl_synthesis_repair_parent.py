"""Public repair-parent selection and literal replay, never a semantic success claim."""
import copy
import json
import unittest
from dataclasses import replace
from typing import Any

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl.checkpoint import digest
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts, restore_protocol
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import ProblemTree, VerifierResult, from_json, to_json
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration


class PublicRepairParentTests(unittest.TestCase):
    def setUp(self):
        self.originals = [self.sample("source0", "first\nsecond\n"), self.sample("source1", "last")]
        self.note: dict[str, Any] = {"lessons": [], "corrected_facts": {}, "attempts": [
            {"attempt": index, "outcome": "unverified public claim", "answer": None,
             "evidence_ranges": [[1, 1]], "issues": [], "unverified": []}
            for index in (1, 2)], "success_attempt": 2, "corrected_answer": "unverified"}

    @staticmethod
    def sample(identity, text, finish="stop"):
        from fine_tuning.notebook_rl.schema import Sample

        return Sample(identity, "problem", "group", "DIAG", 1, None, [1], [2], [-.1],
                      finish, text, 1, 0)

    def candidate(self, identity, note=None, finish="stop"):
        return self.sample(identity, json.dumps(self.note if note is None else note), finish)

    def select(self, candidates, originals=None):
        return synthesis.select_repair_parent(candidates, self.originals if originals is None else originals,
                                             policy=synthesis.REPAIR_PARENT_POLICY)

    def test_invalid_first_uses_first_public_shape_valid_sibling_without_reordering(self):
        bad = copy.deepcopy(self.note)
        bad["corrected_facts"] = {"nested": {"unexpected": "object"}}
        candidates = [self.candidate("bad", bad), self.candidate("valid"), self.candidate("later")]
        before = copy.deepcopy(candidates)
        self.assertEqual(self.select(candidates),
                         {"index": 1, "sample_id": "valid", "reason": "FIRST_PUBLIC_SHAPE_VALID"})
        self.assertEqual(candidates, before)
        self.assertEqual(self.select(candidates[1:])["index"], 0)

    def test_all_invalid_falls_back_to_original_first_sample(self):
        candidates = [self.sample("first", "{}"), self.sample("second", "invalid")]
        self.assertEqual(self.select(candidates), {"index": 0, "sample_id": "first", "reason": "FALLBACK_FIRST"})

    def test_length_termination_and_private_rewards_never_choose_a_parent(self):
        candidates = [self.candidate("length", finish="length"), self.candidate("first"), self.candidate("later")]
        expected = self.select(candidates)
        self.assertEqual(expected["index"], 1)
        for rewards in ((1., 0., .5), (0., 0., 0.), (0., 1., 1.)):
            changed = [replace(sample, reward=reward, diag_reward=reward,
                       result=VerifierResult("INCORRECT", "private", 0., None, "private-grade"))
                       for sample, reward in zip(candidates, rewards)]
            self.assertEqual(self.select(changed), expected)

    def test_attempt_count_order_and_every_citation_bound_are_public_requirements(self):
        mutations = []
        for ranges in ([], [[0, 1]], [[1, 3]], [[2, 1]], [[1, 1], [1, 2]], [[True, 1]]):
            note = copy.deepcopy(self.note)
            note["attempts"][0]["evidence_ranges"] = ranges
            mutations.append(note)
        note = copy.deepcopy(self.note)
        note["attempts"].reverse()
        mutations.append(note)
        note = copy.deepcopy(self.note)
        note["attempts"].pop()
        mutations.append(note)
        for key in ("issues", "unverified"):
            note = copy.deepcopy(self.note)
            claim = {"kind": "TRACE", "rule": "public-rule", "locus": "public-locus",
                     "evidence_ranges": [[99, 99]]}
            if key == "unverified":
                claim["blocked_by"] = []
            note["attempts"][0][key] = [claim]
            mutations.append(note)
        for note in mutations:
            with self.subTest(note=note):
                self.assertEqual(self.select([self.candidate("invalid", note), self.candidate("valid")])["index"], 1)

    def test_empty_original_and_terminal_absence_do_not_require_an_invented_line(self):
        originals = [replace(self.originals[0], text=""), self.originals[1]]
        note = copy.deepcopy(self.note)
        note["attempts"][0]["evidence_ranges"] = []
        note["lessons"] = [{"rule": "public-rule", "relation": "NOT_REVISITED",
            "earlier": {"attempt": 1, "kind": "TRACE", "rule": "public-rule", "locus": "public-locus",
                        "evidence_ranges": []},
            "later": {"attempt": None, "kind": "ABSENCE", "rule": None, "locus": None, "evidence_ranges": []}}]
        self.assertEqual(self.select([self.candidate("valid", note)], originals)["reason"], "FIRST_PUBLIC_SHAPE_VALID")
        note["attempts"][0]["evidence_ranges"] = [[1, 1]]
        self.assertEqual(self.select([self.candidate("bad", note)], originals)["reason"], "FALLBACK_FIRST")

    def test_lesson_endpoints_use_the_referenced_original_line_bounds(self):
        note = copy.deepcopy(self.note)
        endpoint = {"attempt": 1, "kind": "TRACE", "rule": "public", "locus": "public", "evidence_ranges": [[2, 2]]}
        note["lessons"] = [{"rule": "public", "relation": "PERSISTED", "earlier": endpoint,
                            "later": endpoint | {"attempt": 2}}]
        self.assertEqual(self.select([self.candidate("bad", note), self.candidate("valid")])["index"], 1)
        note["lessons"][0]["later"]["attempt"] = 3
        self.assertEqual(self.select([self.candidate("bad", note)])["reason"], "FALLBACK_FIRST")

    def test_wrong_or_missing_explicit_policy_is_rejected(self):
        for policy in (None, "", "fixed-first", "first-public-shape-valid-v0"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "repair parent policy"):
                synthesis.select_repair_parent([self.candidate("first")], self.originals, policy=policy)


class RepairParentReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture = integration.SynthesisIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def append(self, *, empty_first=False):
        fixture = self.fixture
        bad = copy.deepcopy(fixture.note)
        if empty_first:
            for attempt in bad["attempts"]:
                attempt["evidence_ranges"] = []
        else:
            bad["corrected_facts"] = {"trans": {"q0": {"0": "q3"}}}
        public = copy.deepcopy(fixture.note)
        public["lessons"] = []
        public["corrected_facts"]["accepting"] = "q0,q1,q2,q3"
        self.round0 = [json.dumps(bad), json.dumps(public), "{}", json.dumps(public)]
        fixture.append([self.round0, [fixture.good] * 4])
        return fixture.tree.notebook["rewarded_notebook"]

    @staticmethod
    def reseal(tree):
        record = tree.notebook["rewarded_notebook"]
        record["notebook_groups_sha256"] = _hash([json.loads(to_json(group)) for group in tree.groups if group.kind == "DIAG"])
        record["rounds_sha256"] = _hash(record["rounds"])

    def test_selected_actual_feedback_and_literal_parent_survive_serialized_fresh_replay(self):
        fixture = self.fixture
        before = copy.deepcopy(fixture.tree.groups)
        record = self.append()
        first, second = fixture.tree.groups[-2:]
        parent = first.samples[1]
        self.assertEqual(fixture.tree.groups[:-2], before)
        self.assertEqual([sample.text for sample in first.samples], self.round0)
        self.assertEqual([sample.reward for sample in first.samples], [0.] * 4)
        self.assertEqual(record["notebook_repair_parent_policy"], synthesis.REPAIR_PARENT_POLICY)
        self.assertEqual(record["rounds"][1]["repair_parent"],
                         {"index": 1, "sample_id": parent.sample_id, "reason": "FIRST_PUBLIC_SHAPE_VALID"})
        checked = fixture.tree.notebook["model_notes"][1]
        feedback = record["rounds"][1]["feedback"]
        self.assertEqual(feedback, episode.notebook_feedback(checked, parent))
        self.assertEqual(checked["failure_code"], "CHECKS_FAILED")
        payload = json.loads(feedback.removeprefix(synthesis.RETRY_INSTRUCTION))
        self.assertEqual(payload["reinspection_slots"], ["accepting"])
        for sample in second.samples:
            self.assertEqual(sample.parent_sample_id, parent.sample_id)
            self.assertEqual(sample.prompt_token_ids,
                             episode.continuation_ids(parent, feedback, fixture.engine.tokenizer))
        restored = from_json(ProblemTree, to_json(fixture.tree))
        receipt = episode.regrade_synthesis(restored, fixture.actor, fixture.verifier, fixture.engine.tokenizer)
        self.assertEqual(len(receipt["samples"]), 8)

    def test_nonempty_original_empty_citations_skip_first_sibling(self):
        record = self.append(empty_first=True)
        self.assertEqual(record["rounds"][1]["repair_parent"]["index"], 1)

    def test_resealed_selection_feedback_group_order_and_sibling_ancestry_tampering_fail(self):
        self.append()
        baseline = self.fixture.tree
        for mutation in ("selection", "boolean_index", "feedback", "order", "parent", "prompt", "reward", "tokens"):
            with self.subTest(mutation=mutation):
                tree = copy.deepcopy(baseline)
                first, second = tree.groups[-2:]
                row = tree.notebook["rewarded_notebook"]["rounds"][1]
                if mutation == "selection":
                    row["repair_parent"]["index"] = 0
                elif mutation == "boolean_index":
                    row["repair_parent"]["index"] = True
                elif mutation == "feedback":
                    row["feedback"] = episode.notebook_feedback(tree.notebook["model_notes"][0], first.samples[0])
                elif mutation == "order":
                    first.samples[0], first.samples[1] = first.samples[1], first.samples[0]
                elif mutation == "parent":
                    second.samples[1].parent_sample_id = first.samples[0].sample_id
                elif mutation == "prompt":
                    second.samples[1].prompt_token_ids = second.samples[1].prompt_token_ids + [1]
                elif mutation == "reward":
                    first.samples[2].reward = first.samples[2].diag_reward = 1.
                    tree.notebook["model_notes"][2]["reward"] = 1.
                else:
                    first.samples[1].completion_token_ids[-1] += 1
                if mutation not in ("parent", "prompt"):
                    self.reseal(tree)
                with self.assertRaises(ValueError):
                    episode.regrade_synthesis(tree, self.fixture.actor, self.fixture.verifier, self.fixture.engine.tokenizer)

    def test_policy_is_required_in_protocol_prompts_and_replayed_record(self):
        fixture = self.fixture
        for kwargs in ({"verified_note_synthesis": True}, {"verified_note_synthesis_credit": True, "context_limit": 24576}):
            protocol = build_protocol(**kwargs)
            self.assertEqual(protocol["notebook_repair_parent_policy"], synthesis.REPAIR_PARENT_POLICY)
            self.assertEqual(local_prompts(protocol)["notebook_repair_parent_policy"], synthesis.REPAIR_PARENT_POLICY)
            for policy in (None, "fixed-first"):
                forged = copy.deepcopy(protocol)
                if policy is None:
                    del forged["notebook_repair_parent_policy"]
                else:
                    forged["notebook_repair_parent_policy"] = policy
                forged["sha256"] = digest({key: value for key, value in forged.items() if key != "sha256"})
                with self.assertRaises(ValueError):
                    restore_protocol(forged)
                with self.assertRaises(ValueError):
                    local_prompts(forged)
        self.assertNotIn("notebook_repair_parent_policy", build_protocol())
        self.assertNotIn("notebook_repair_parent_policy", local_prompts(build_protocol()))
        for policy in (None, "fixed-first"):
            prompts = copy.deepcopy(fixture.prompts)
            if policy is None:
                del prompts["notebook_repair_parent_policy"]
            else:
                prompts["notebook_repair_parent_policy"] = policy
            with self.assertRaisesRegex(ValueError, "repair parent policy"):
                episode.append_episode_notebooks([copy.deepcopy(fixture.tree)], fixture.images, fixture.engine,
                    fixture.verifier, prompts, fixture.cfg, actors={fixture.actor.problem_id: fixture.actor})
        self.assertEqual(fixture.engine.requests, [])
        self.append()
        for policy in (None, "fixed-first"):
            tree = copy.deepcopy(fixture.tree)
            record = tree.notebook["rewarded_notebook"]
            if policy is None:
                del record["notebook_repair_parent_policy"]
            else:
                record["notebook_repair_parent_policy"] = policy
            with self.assertRaisesRegex(ValueError, "repair parent policy"):
                episode.regrade_synthesis(tree, fixture.actor, fixture.verifier, fixture.engine.tokenizer)


if __name__ == "__main__":
    unittest.main()
