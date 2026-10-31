"""Authored CPU fixtures check notebook retries, not learned self-improvement."""

import copy
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from fine_tuning.notebook_rl.episode_notebook import (
    MAX_NOTEBOOK_ROUNDS,
    RETRY_INSTRUCTION,
    audit_episode,
    grade_note,
    notebook_feedback,
    validate_update_binding,
)
from fine_tuning.notebook_rl.losses import group_advantages
from fine_tuning.notebook_rl.rollout import _hash, seeds
from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
from fine_tuning.notebook_rl.updates import action_mask, effective_rewards, has_learning_signal
from fine_tuning.tests import test_notebook_rl_episode_notebook as fixtures


class EpisodeRetryTests(unittest.TestCase):
    setUp = fixtures.EpisodeNotebookTests.setUp
    tree = fixtures.EpisodeNotebookTests.tree
    append = fixtures.EpisodeNotebookTests.append

    def notes(self, tree):
        audit = audit_episode(tree, self.actor, self.verifier)
        valid = fixtures.authored_note(tree, audit)
        invalid = copy.deepcopy(valid)
        invalid["corrected_facts"]["cell[a][b]"] = "8"
        return json.dumps(valid), json.dumps(invalid)

    def three_rounds(self):
        tree, engine = self.tree()
        valid, invalid = self.notes(tree)
        engine.replies.extend([["{", invalid, "{}", "[]"], [invalid] * 4,
                               [valid, invalid, valid, "{}"]])
        report = self.append(tree, engine)
        return tree, engine, report

    def test_format_then_semantic_failure_then_mixed_valid_third_group(self):
        tree, engine, report = self.three_rounds()
        groups = [g for g in tree.groups if g.kind == "DIAG"]
        self.assertEqual(report["generated_groups"], 3)
        self.assertEqual(report["nondegenerate_groups"], 1)
        self.assertEqual([[s.reward for s in g.samples] for g in groups],
                         [[0.0] * 4, [0.0] * 4, [1.0, 0.0, 1.0, 0.0]])
        self.assertEqual(tree.notebook["model_notes"][0]["failure_code"], "INVALID_JSON")
        semantic = tree.notebook["model_notes"][4]
        self.assertEqual(semantic["failure_code"], "CHECKS_FAILED")
        self.assertTrue(semantic["checks"]["all_attempts_verified"])
        self.assertFalse(semantic["checks"]["registered_facts_verified"])
        record = tree.notebook["rewarded_notebook"]
        self.assertEqual(record["stop_reason"], "VALID_NOTE")
        self.assertTrue(record["reward_variance_present"])
        self.assertFalse(record["optimizer_update_verified"])
        self.assertTrue(has_learning_signal(tree, include_notebooks=True))
        self.assertFalse(has_learning_signal(tree, include_notebooks=False))
        self.assertEqual(tree.route, "VISION_ON")
        self.assertEqual(from_json(ProblemTree, to_json(tree)), tree)
        validate_update_binding(tree, tokenizer=engine.tokenizer)

    def test_groups_keep_four_siblings_distinct_seeds_and_independent_advantages(self):
        tree, engine, _ = self.three_rounds()
        groups = [g for g in tree.groups if g.kind == "DIAG"]
        self.assertEqual(len({g.group_id for g in groups}), 3)
        self.assertEqual(len({s.seed for g in groups for s in g.samples}), 12)
        for index, (group, request) in enumerate(zip(groups, engine.requests[-3:], strict=True)):
            with self.subTest(round=index + 1):
                branch = "EPISODE_NOTEBOOK" + (f":REPAIR:{index}" if index else "")
                expected = seeds(tree.problem_id, tree.policy_version, group.samples[0].turn,
                                 branch, 4, master_seed=self.cfg.master_seed)
                self.assertEqual(len(group.samples), 4)
                self.assertEqual([s.seed for s in group.samples], expected)
                self.assertEqual(request.seeds, expected)
                self.assertEqual(len({tuple(s.prompt_token_ids) for s in group.samples}), 1)
                self.assertEqual(len({s.parent_sample_id for s in group.samples}), 1)
                self.assertEqual(group.state_sha256, _hash(group.samples[0].prompt_token_ids))
                advantages = group_advantages(torch.tensor([s.reward for s in group.samples]))
                self.assertEqual(advantages.tolist(), [0.0] * 4 if index < 2
                                 else [0.5, -0.5, 0.5, -0.5])

    def test_all_zero_rewards_stop_at_three_rounds_without_learning_signal(self):
        tree, engine = self.tree()
        engine.replies.extend([["{}"] * 4] * MAX_NOTEBOOK_ROUNDS)
        original = copy.deepcopy(tree.groups)
        report = self.append(tree, engine)
        self.assertEqual(MAX_NOTEBOOK_ROUNDS, 3)
        self.assertEqual(report["generated_groups"], 3)
        self.assertEqual(report["nondegenerate_groups"], 0)
        self.assertEqual(tree.groups[:len(original)], original)
        self.assertEqual(len(engine.requests), len(original) + 3)
        self.assertEqual(tree.notebook["rewarded_notebook"]["stop_reason"], "NOTEBOOK_ROUND_LIMIT")
        self.assertFalse(has_learning_signal(tree, include_notebooks=True))

    def test_all_valid_stops_immediately_but_constant_reward_is_not_learning_signal(self):
        tree, engine = self.tree()
        valid, _ = self.notes(tree)
        engine.replies.append([valid] * 4)
        report = self.append(tree, engine)
        self.assertEqual(report["generated_groups"], 1)
        self.assertEqual([s.reward for s in tree.groups[-1].samples], [1.0] * 4)
        self.assertEqual(tree.notebook["rewarded_notebook"]["stop_reason"], "VALID_NOTE")
        self.assertFalse(tree.notebook["rewarded_notebook"]["reward_variance_present"])
        self.assertFalse(has_learning_signal(tree, include_notebooks=True))

    def test_first_valid_sibling_stops_further_repair_even_when_sample_zero_fails(self):
        tree, engine = self.tree()
        valid, invalid = self.notes(tree)
        engine.replies.extend([["{}"] * 4, [invalid, valid, invalid, invalid]])
        report = self.append(tree, engine)
        self.assertEqual(report["generated_groups"], 2)
        self.assertEqual(report["nondegenerate_groups"], 1)
        self.assertEqual(tree.notebook["rewarded_notebook"]["stop_reason"], "VALID_NOTE")
        self.assertTrue(has_learning_signal(tree, include_notebooks=True))

    def test_retry_preserves_literal_tokens_and_encodes_only_feedback(self):
        tree, engine, _ = self.three_rounds()
        groups = [g for g in tree.groups if g.kind == "DIAG"]
        record = tree.notebook["rewarded_notebook"]
        for index in (1, 2):
            parent, sample = groups[index - 1].samples[0], groups[index].samples[0]
            row = record["rounds"][index]
            feedback = notebook_feedback(tree.notebook["model_notes"][(index - 1) * 4])
            encoded = "\n<|im_start|>user\n" + feedback + "<|im_end|>\n<|im_start|>assistant\n"
            suffix = [900 + ord(c) for c in encoded]
            self.assertEqual(sample.prompt_token_ids,
                             parent.prompt_token_ids + parent.completion_token_ids + suffix)
            self.assertEqual(sample.parent_sample_id, parent.sample_id)
            self.assertEqual(row["feedback"], feedback)
            self.assertEqual(row["continuation_suffix"], suffix)
            self.assertIn(encoded, engine.tokenizer.encoded)
            self.assertIsNone(engine.requests[-3 + index].messages)
        self.assertEqual(json.loads(record["rounds"][1]["feedback"].split("Verifier feedback:\n")[1])
                         ["failure_code"], "INVALID_JSON")
        self.assertFalse(any('"corrected_facts"' in text for text in engine.tokenizer.encoded))

    def test_missing_conversation_boundary_is_added_once_to_retry_prompt(self):
        tree, engine = self.tree()
        valid, _ = self.notes(tree)
        engine.replies.extend([["{}"] * 4, [valid] * 4])
        sample = engine.sample

        def without_boundary(requests):
            outputs = sample(requests)
            if requests[0].request_id.endswith(":EPISODE_NOTEBOOK"):
                outputs[0] = replace(outputs[0], completions=[
                    replace(c, token_ids=c.token_ids[:-1], logprobs=c.logprobs[:-1])
                    for c in outputs[0].completions])
            return outputs

        engine.sample = without_boundary
        self.append(tree, engine)
        parent, retry = tree.groups[-2].samples[0], tree.groups[-1].samples[0]
        end = len(parent.prompt_token_ids) + len(parent.completion_token_ids)
        self.assertEqual(retry.prompt_token_ids[:end], parent.prompt_token_ids + parent.completion_token_ids)
        self.assertEqual(retry.prompt_token_ids[end:end + 2], [151645, 900 + ord("\n")])
        validate_update_binding(tree, tokenizer=engine.tokenizer)

    def test_feedback_contains_only_public_check_results_without_private_sentinels(self):
        tree, _ = self.tree()
        audit = audit_episode(tree, self.actor, self.verifier)
        note = fixtures.authored_note(tree, audit)
        note["corrected_facts"]["cell[a][b]"] = "MODEL_CLAIM_SENTINEL"
        checked = grade_note(json.dumps(note), "stop", tree, audit, self.verifier)
        checked["private_target"] = {"answer": "PRIVATE_ANSWER_SENTINEL", "slots": "PRIVATE_SLOT_SENTINEL"}
        checked["attempt_checks"][0]["expected_cause"] = "PRIVATE_CAUSE_SENTINEL"
        checked["attempt_checks"][0]["resolved_lines"] = ["PRIVATE_EVIDENCE_SENTINEL"]
        feedback = notebook_feedback(checked)
        self.assertTrue(feedback.startswith(RETRY_INSTRUCTION))
        self.assertNotIn("SENTINEL", feedback)
        payload = json.loads(feedback[len(RETRY_INSTRUCTION):])
        self.assertEqual(set(payload), {"failure_code", "checks", "attempt_checks", "remaining_claims_checked"})
        self.assertEqual(payload["failure_code"], "CHECKS_FAILED")
        self.assertEqual(payload["checks"], checked["checks"])
        self.assertEqual(set(payload["attempt_checks"][0]),
                         {"attempt", "history", "cause", "wrong_slots", "conclusion_evidence", "visual_evidence"})
        self.assertNotIn("verified_next_check", payload["attempt_checks"][0])
        self.assertNotIn("next_check_source", payload["attempt_checks"][0])
        self.assertTrue(all(type(value) is bool for row in payload["attempt_checks"]
                            for key, value in row.items() if key != "attempt"))

    def test_malformed_or_successful_feedback_is_rejected(self):
        base = {"failure_code": "INVALID_JSON", "reward": 0.0, "checks": {}, "attempt_checks": []}
        for changed in ({"failure_code": []}, {"failure_code": {}}, {"failure_code": "unknown"},
                        {"failure_code": "NONE"}, {"reward": True}, {"reward": False},
                        {"reward": "0"}, {"reward": 1.0}, {"checks": {"private": False}},
                        {"checks": {"format_valid": 0}}, {"attempt_checks": None},
                        {"attempt_checks": {}}, {"attempt_checks": [None]},
                        {"attempt_checks": [{"attempt": 1}]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                notebook_feedback(base | changed)

    def test_note_action_mask_covers_only_its_generated_completion(self):
        tree, engine, _ = self.three_rounds()
        for group in tree.groups:
            if group.kind != "DIAG":
                continue
            for sample in group.samples:
                for algorithm in ("ALG-T1", "ALG-NB-noteonly"):
                    mask = action_mask(sample, algorithm, engine.tokenizer)
                    self.assertEqual(tuple(mask.shape), (1, len(sample.completion_token_ids)))
                    self.assertTrue(mask.all())
                    self.assertGreater(len(sample.prompt_token_ids), 0)

    def test_answer_only_control_zeroes_every_note_round_without_mutating_records(self):
        tree, _, _ = self.three_rounds()
        before = copy.deepcopy(tree)
        actual = effective_rewards(tree.groups, "ALG-T1", 3407, 0)
        controlled = effective_rewards(tree.groups, "ALG-T1", 3407, 0, notebook_reward_enabled=False)
        for group in tree.groups:
            self.assertEqual(controlled[group.group_id], [0.0] * 4 if group.kind == "DIAG"
                             else actual[group.group_id])
        self.assertEqual(actual[tree.groups[-1].group_id], [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(tree, before)

    def test_parent_feedback_and_round_hash_tampering_are_rejected(self):
        tree, engine, _ = self.three_rounds()
        for case in ("parent", "feedback", "round", "hash"):
            with self.subTest(case=case):
                changed = copy.deepcopy(tree)
                record = changed.notebook["rewarded_notebook"]
                if case == "parent":
                    for sample in changed.groups[-1].samples:
                        sample.parent_sample_id = changed.groups[-2].samples[1].sample_id
                    record["notebook_groups_sha256"] = _hash(
                        [json.loads(to_json(g)) for g in changed.groups if g.kind == "DIAG"])
                elif case == "feedback":
                    record["rounds"][-1]["feedback"] += "Altered feedback."
                elif case == "round":
                    record["rounds"][-1]["round"] = 2
                else:
                    record["rounds_sha256"] = "0" * 64
                if case != "hash":
                    record["rounds_sha256"] = _hash(record["rounds"])
                with self.assertRaises(ValueError):
                    validate_update_binding(changed, tokenizer=engine.tokenizer)

    def test_rehashed_prompt_and_suffix_tampering_still_fails_tokenizer_binding(self):
        tree, engine, _ = self.three_rounds()
        group = tree.groups[-1]
        record = tree.notebook["rewarded_notebook"]
        record["rounds"][-1]["continuation_suffix"][-1] += 1
        for sample in group.samples:
            sample.prompt_token_ids[-1] += 1
        group.state_sha256 = _hash(group.samples[0].prompt_token_ids)
        record["rounds"][-1]["prompt_sha256"] = group.state_sha256
        record["rounds_sha256"] = _hash(record["rounds"])
        record["notebook_groups_sha256"] = _hash(
            [json.loads(to_json(g)) for g in tree.groups if g.kind == "DIAG"])
        with self.assertRaisesRegex(ValueError, "encoded feedback"):
            validate_update_binding(tree, tokenizer=engine.tokenizer)

    def test_budget_or_context_stop_retains_completed_note_group_and_solutions(self):
        for stop in ("BUDGET_STOP", "CONTEXT_BLOCKED"):
            with self.subTest(stop=stop):
                tree, engine = self.tree()
                original = copy.deepcopy(tree.groups)
                engine.replies.append(["{}"] * 4)
                closed = []
                prepare = engine.prepare
                if stop == "CONTEXT_BLOCKED":
                    engine.prepare = lambda request: SimpleNamespace(token_ids=[1] * self.cfg.context_limit) \
                        if ":REPAIR:" in request.request_id else prepare(request)
                report = self.append(tree, engine, on_group=closed.append,
                                     should_stop=lambda: stop == "BUDGET_STOP" and bool(closed))
                self.assertEqual(report["generated_groups"], 1)
                self.assertEqual(tree.groups[:len(original)], original)
                self.assertEqual(closed, tree.groups[len(original):])
                record = tree.notebook["rewarded_notebook"]
                self.assertEqual(record["status"], "GENERATED")
                self.assertEqual(record["stop_reason"], stop)
                self.assertEqual(len(record["rounds"]), 1)
                self.assertFalse(has_learning_signal(tree, include_notebooks=True))
                validate_update_binding(tree, tokenizer=engine.tokenizer)


if __name__ == "__main__":
    unittest.main()
