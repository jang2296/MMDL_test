"""Authored partial-note actions test wiring, not pretrained-model improvement."""
import copy
import json
import unittest

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, partial_notebook, updates
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.notebook_learning import append_rewarded_notebooks
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration
from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleTokenizer


class PartialNotebookIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.f = integration.SynthesisIntegrationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.protocol = build_protocol(verified_note_synthesis_credit=True, partial_notebook=True,
                                       context_limit=24576)
        self.f.protocol = self.protocol
        self.f.engine.tokenizer = ReversibleTokenizer()
        samples = episode.solution_samples(self.f.tree)
        for index, sample in enumerate(samples):
            parent = samples[0] if index in (2, 3) else samples[2] if index in (4, 5) else None
            sample.prompt_token_ids = ([1, 2] if parent is None else
                parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2])
            sample.completion_token_ids = self.f.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)

    @staticmethod
    def note(facts=None, lessons=None):
        return json.dumps({"coverage": "PARTIAL", "corrected_facts": facts or {}, "lessons": lessons or []})

    def append(self, texts):
        observation = "Observed facts:\n" + "\n".join(
            f"{key} = {value}" for key, value in self.f.item.slots.items()) + "\nSolution:"
        self.f.engine.replies = [[observation], texts]
        before = [to_json(sample) for sample in episode.solution_samples(self.f.tree)]
        result = append_rewarded_notebooks([self.f.tree], self.f.images, self.f.engine, self.f.verifier,
            local_prompts(self.protocol), RolloutConfig.from_protocol(self.protocol, "ALG-NB3", master_seed=3407),
            actors={self.f.actor.problem_id: self.f.actor})
        self.assertEqual(before, [to_json(sample) for sample in episode.solution_samples(self.f.tree)])
        self.assertEqual(len(self.f.engine.requests), 2)  # Fresh observation and one n4 note group.
        self.assertFalse(self.f.engine.replies)
        self.record = self.f.tree.notebook["rewarded_notebook"]
        self.group = self.f.tree.groups[-1]
        self.assertEqual(self.record["status"], "GENERATED")
        self.assertEqual(self.record["notebook_generation_policy"], partial_notebook.VERSION)
        self.assertEqual([sample.reward for sample in self.group.samples], [0.] * 4)
        self.assertIsNone(self.record["rounds"][0]["feedback"])
        self.assertIsNone(self.record["rounds"][0]["repair_parent"])
        self.assertEqual(result["generated_groups"], 1)
        episode.regrade_synthesis(self.f.tree, self.f.actor, self.f.verifier, self.f.engine.tokenizer)
        return result

    def test_nonempty_partial_stops_without_claiming_strict_success_and_preserves_all_four(self):
        texts = [self.note({"start": "q0"}), self.note({"start": "q1"}), self.note(), "{broken"]
        self.append(texts)
        self.assertEqual(self.record["stop_reason"], "PARTIAL_NOTEBOOK_CAPTURED")
        self.assertEqual([sample.text for sample in self.group.samples], texts)
        self.assertEqual([row["status"] for row in self.record["partial_artifacts"]],
            ["PARTIAL_NOTEBOOK_CAPTURED", "PARTIAL_NOTEBOOK_CAPTURED", "NO_NOTEBOOK_CLAIMS", "INVALID_PARTIAL_NOTEBOOK"])
        updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier, self.protocol["notebook_objective"])
        scores = updates.notebook_objective_scores(self.f.tree, self.protocol["notebook_objective"])
        self.assertTrue(updates.group_has_signal(self.group, scores))
        self.assertEqual([scores[s.sample_id]["fact_rewards"].get("start", 0) for s in self.group.samples], [1, 0, 0, 0])
        from fine_tuning.notebook_rl.synthesis_credit import group_credit
        credit = group_credit(self.group.samples, scores, self.f.engine.tokenizer)
        self.assertEqual(len(credit), 4)

    def test_empty_and_all_wrong_artifacts_never_gain_truth_from_completion(self):
        for texts, stop in (([self.note()] * 4, "NO_NOTEBOOK_CLAIMS"),
                            ([self.note({"start": "q1"})] * 4, "PARTIAL_NOTEBOOK_CAPTURED"),
                            (["{broken"] * 4, "PARTIAL_NOTEBOOK_CAPTURE_FAILED")):
            with self.subTest(stop=stop):
                self.setUp()
                self.append(texts)
                self.assertEqual(self.record["stop_reason"], stop)
                updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier,
                                                self.protocol["notebook_objective"])
                scores = updates.notebook_objective_scores(self.f.tree, self.protocol["notebook_objective"])
                self.assertFalse(updates.group_has_signal(self.group, scores))
                self.assertTrue(all(score["effective_reward"] == 0 for score in scores.values()))

    def test_policy_stop_artifact_and_extra_group_tampering_are_rejected(self):
        self.append([self.note({"start": "q0"})] * 4)
        original = copy.deepcopy(self.f.tree)
        for field, value in (("notebook_generation_policy", "other"), ("stop_reason", "VALID_NOTE"),
                             ("partial_artifacts", [])):
            tree = copy.deepcopy(original)
            tree.notebook["rewarded_notebook"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                episode.validate_update_binding(tree, tokenizer=self.f.engine.tokenizer)
        with self.assertRaisesRegex(ValueError, "execution protocol"):
            episode.validate_update_binding(original, generation_policy=None)
        tree = copy.deepcopy(original)
        tree.groups.append(copy.deepcopy(tree.groups[-1]))
        record = tree.notebook["rewarded_notebook"]
        record["group_ids"].append(record["group_ids"][-1])
        record["rounds"].append(copy.deepcopy(record["rounds"][-1]))
        record["notebook_groups_sha256"] = _hash([json.loads(to_json(g)) for g in tree.groups if g.kind == "DIAG"])
        record["rounds_sha256"] = _hash(record["rounds"])
        with self.assertRaisesRegex(ValueError, "one original n4"):
            episode.validate_update_binding(tree, tokenizer=self.f.engine.tokenizer)

    def test_optimizer_rejects_cross_protocol_actions_before_mutation(self):
        self.append([self.note({"start": "q0"}), self.note({"start": "q1"})] * 2)
        updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier, self.protocol["notebook_objective"])
        fixture = self.f.optimizer_fixture()
        self.f.protocol = build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        with self.assertRaisesRegex(ValueError, "execution protocol"):
            self.f.step(fixture)

    def test_verified_partial_claim_updates_weights_without_strict_success(self):
        self.append([self.note({"start": "q0"}), self.note({"start": "q1"})] * 2)
        updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier,
                                        self.protocol["notebook_objective"])
        fixture = self.f.optimizer_fixture()
        update = self.f.step(fixture)
        self.assertEqual(update["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual(fixture[2].last_epoch, 1)
        self.assertTrue(torch.count_nonzero(fixture[0].model["L"]))
        self.assertTrue(any(row["changed_groups"] for row in update["notebook_backward"]))
        self.assertEqual([sample.reward for sample in self.group.samples], [0.] * 4)
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))


if __name__ == "__main__":
    unittest.main()
