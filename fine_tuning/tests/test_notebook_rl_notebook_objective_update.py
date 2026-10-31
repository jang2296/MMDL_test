"""Authored CPU reward-integration/rejection fixtures, not learned self-improvement."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fine_tuning.notebook_rl import episode_notebook as notebook, updates
from fine_tuning.notebook_rl.local_profile import build_protocol
from fine_tuning.notebook_rl.local_train import eligible_trees, notebook_learning_gate
from fine_tuning.notebook_rl.notebook_objective import contract
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_notebook_objective as atom_fixtures
from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleEngine


class NotebookObjectiveUpdateTests(unittest.TestCase):
    def setUp(self):
        fixture = atom_fixtures.NotebookObjectiveTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.tree = fixture.fixture, fixture.tree
        self.actor, self.verifier = fixture.actor, fixture.verifier
        self.objective = contract()
        partial = copy.deepcopy(fixture.note)
        partial["corrected_facts"] = {atom_fixtures.A: "7"}
        wrong = copy.deepcopy(fixture.note)
        wrong["corrected_facts"] = {atom_fixtures.A: "999", atom_fixtures.B: "999"}
        self.engine = ReversibleEngine([[json.dumps(partial), json.dumps(wrong)] * 2])
        self.fixture.prompts["notebook_route_rule"] = notebook.STRICT_ROUTE_RULE
        groups = []
        self.fixture.append(self.tree, self.engine, on_group=groups.append,
                            should_stop=lambda: bool(groups))
        self.group = self.tree.groups[-1]
        self.assertEqual(self.group.kind, "DIAG")
        self.assertEqual([s.reward for s in self.group.samples], [0.] * 4)
        # Isolate objective guards here; full v14 reread/profile checks have separate tests.
        self.protocol = build_protocol(verified_note_atoms=True)
        self.protocol.pop("notebook_observation_policy")
        self.protocol.pop("notebook_feedback_policy")

    def bind(self, tree=None):
        return updates.bind_notebook_objective(tree or self.tree, self.actor, self.verifier, self.objective)

    def rehash(self, tree):
        receipt = tree.notebook["learning_objective"]
        receipt["sha256"] = _hash({k: v for k, v in receipt.items() if k != "sha256"})

    def reject_update(self, tree, expected):
        model, optimizer = Mock(name="model_not_constructed"), Mock(name="optimizer_not_stepped")
        bundle = SimpleNamespace(inventory={"arm": "MOD-RV"}, model=model)
        with patch("fine_tuning.notebook_rl.updates.completion_logprobs") as forward:
            with self.assertRaisesRegex(ValueError, expected):
                updates.optimizer_step(bundle, optimizer, SimpleNamespace(last_epoch=0),
                    [tree], {tree.problem_id: [object()]}, self.engine, self.protocol, "ALG-NB3",
                    step=0, seed=3407, observation_actors={tree.problem_id: self.actor},
                    objective_verifier=self.verifier)
            forward.assert_not_called()
        model.assert_not_called()
        optimizer.step.assert_not_called()

    def test_binding_is_explicit_and_preserves_every_sample_and_strict_record(self):
        before = json.loads(to_json(self.tree))
        with self.assertRaisesRegex(ValueError, "stale notebook objective"):
            updates.notebook_objective_scores(self.tree, self.objective)
        receipt = self.bind()
        self.assertIs(receipt, self.tree.notebook["learning_objective"])
        after = json.loads(to_json(self.tree))
        del after["notebook"]["learning_objective"]
        self.assertEqual(after, before)
        scores = updates.notebook_objective_scores(self.tree, self.objective)
        self.assertEqual([scores[s.sample_id]["effective_reward"] for s in self.group.samples], [.125, 0., .125, 0.])
        self.assertTrue(all(row["strict_reward"] == 0 for row in scores.values()))
        with self.assertRaisesRegex(ValueError, "explicitly selected"):
            updates.notebook_objective_scores(self.tree, None)
        with self.assertRaisesRegex(ValueError, "already bound"):
            self.bind()

    def test_unknown_stale_coverage_arithmetic_and_text_bindings_reject(self):
        self.bind()
        with self.assertRaisesRegex(ValueError, "unknown notebook objective"):
            updates.notebook_objective_scores(self.tree, self.objective | {"atomic_weight": .5})
        for version in (None, 15, []):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "unknown notebook objective"):
                updates.notebook_objective_scores(self.tree, self.objective | {"objective_version": version})
        for mutation, expected in (("hash", "stale"), ("policy", "stale"), ("coverage", "coverage"),
                                    ("arithmetic", "arithmetic"), ("text", "arithmetic"),
                                    ("unverified", "arithmetic")):
            with self.subTest(mutation=mutation):
                tree = copy.deepcopy(self.tree)
                receipt = tree.notebook["learning_objective"]
                row = receipt["samples"][self.group.samples[0].sample_id]
                if mutation == "hash":
                    receipt["sha256"] = "a" * 64
                elif mutation == "policy":
                    receipt["policy_version"] += 1
                elif mutation == "coverage":
                    del receipt["samples"][self.group.samples[0].sample_id]
                elif mutation == "arithmetic":
                    row["effective_reward"] += .01
                elif mutation == "text":
                    row["text_sha256"] = "b" * 64
                else:
                    row["objective_independently_verified"] = False
                if mutation != "hash":
                    self.rehash(tree)
                with self.assertRaisesRegex(ValueError, expected):
                    updates.notebook_objective_scores(tree, self.objective)

    def test_partial_note_variance_is_signal_only_for_the_notebook_arm(self):
        self.assertFalse(updates.has_learning_signal(self.tree, include_notebooks=True))
        self.bind()
        self.assertTrue(updates.has_learning_signal(self.tree, include_notebooks=True, notebook_objective=self.objective))
        self.assertFalse(updates.has_learning_signal(self.tree, include_notebooks=False, notebook_objective=self.objective))
        self.assertEqual(eligible_trees([self.tree], "notebook", notebook_objective=self.objective), [self.tree])
        self.assertEqual(eligible_trees([self.tree], "answer-only", notebook_objective=self.objective), [])
        scores = updates.notebook_objective_scores(self.tree, self.objective)
        controlled = updates.effective_rewards(self.tree.groups, "ALG-NB3", 3407, 0,
            notebook_reward_enabled=False, notebook_scores=scores)
        self.assertEqual(controlled[self.group.group_id], [0.] * 4)
        self.assertEqual([s.reward for s in self.group.samples], [0.] * 4)

    def test_fabricated_consistent_arithmetic_requires_real_verifier_regrade_before_update(self):
        self.bind()
        receipt = self.tree.notebook["learning_objective"]
        row = receipt["samples"][self.group.samples[0].sample_id]
        row.update(numerator=0, atom_fraction=0., atomic_reward=0., effective_reward=0.)
        self.rehash(self.tree)
        # Hash and arithmetic alone cannot establish that the original note earned this score.
        self.assertEqual(updates.notebook_objective_scores(self.tree, self.objective)[
            self.group.samples[0].sample_id]["effective_reward"], 0.)
        with patch.object(self.verifier, "grade", wraps=self.verifier.grade) as fresh:
            self.reject_update(self.tree, "freshly regraded notebook objective differs")
            self.assertGreater(fresh.call_count, 0)

    def test_notebook_sample_tokens_must_decode_to_the_rewarded_text_before_update(self):
        sample = self.group.samples[0]
        sample.completion_token_ids[0] += 1
        self.tree.notebook["rewarded_notebook"]["notebook_groups_sha256"] = _hash(
            [json.loads(to_json(group)) for group in self.tree.groups if group.kind == "DIAG"])
        # Rebinding text-only scores cannot authenticate different sampled token actions.
        self.bind()
        updates.notebook_objective_scores(self.tree, self.objective)
        self.reject_update(self.tree, "notebook sampled tokens and rewarded text differ")

    def test_partial_only_notes_cannot_pass_strict_learning_gate_with_claimed_gradients(self):
        self.bind()
        self.reject_partial_expansion()

    def test_components_bind_actual_scores_isolate_control_and_keep_strict_gate(self):
        from fine_tuning.notebook_rl.notebook_objective import COMPONENT_VERSION

        self.objective = contract(COMPONENT_VERSION)
        self.protocol["notebook_objective"] = self.objective
        self.bind()
        scores = updates.notebook_objective_scores(self.tree, self.objective)
        self.assertEqual([scores[s.sample_id]["effective_reward"] for s in self.group.samples],
                         [.21875, .1875, .21875, .1875])
        self.assertEqual([s.reward for s in self.group.samples], [0.] * 4)
        controlled = updates.effective_rewards(self.tree.groups, "ALG-NB3", 3407, 0,
            notebook_reward_enabled=False, notebook_scores=scores)
        self.assertEqual(controlled[self.group.group_id], [0.] * 4)
        self.reject_partial_expansion()
        forged = copy.deepcopy(self.tree)
        row = forged.notebook["learning_objective"]["samples"][self.group.samples[0].sample_id]
        row["component_counts"]["required_facts"] = 0
        self.rehash(forged)
        self.reject_update(forged, "freshly regraded notebook objective differs")

    def test_fact_credit_receipt_tampering_control_and_strict_expansion_gate(self):
        from fine_tuning.notebook_rl.notebook_objective import FACT_CREDIT_VERSION

        self.objective = contract(FACT_CREDIT_VERSION)
        self.protocol["notebook_objective"] = self.objective
        self.bind()
        scores = updates.notebook_objective_scores(self.tree, self.objective)
        self.assertTrue(updates.group_has_signal(self.group, scores))
        controlled = updates.effective_rewards(self.tree.groups, "ALG-NB3", 3407, 0,
            notebook_reward_enabled=False, notebook_scores=scores)
        self.assertEqual(controlled[self.group.group_id], [0.] * 4)
        self.reject_partial_expansion()
        forged = copy.deepcopy(self.tree)
        row = forged.notebook["learning_objective"]["samples"][self.group.samples[0].sample_id]
        row["fact_rewards"][atom_fixtures.A] = 0
        self.rehash(forged)
        self.reject_update(forged, "freshly regraded notebook objective differs")

    def reject_partial_expansion(self):
        scores = updates.notebook_objective_scores(self.tree, self.objective)
        rewards = [scores[s.sample_id]["effective_reward"] for s in self.group.samples]
        mean = sum(rewards) / len(rewards)
        backward = [{"sample_id": s.sample_id, "group_id": self.group.group_id,
            "strict_reward": 0., "effective_reward": scores[s.sample_id]["effective_reward"],
            "atomic_reward": scores[s.sample_id]["atomic_reward"], "advantage": rewards[i] - mean,
            "reward_gradient_isolated": True, "vision_on": True,
            "action_tokens": len(s.completion_token_ids), "changed_groups": ["L"],
            "gradient_before": {g: [None] for g in ("L", "M", "V")},
            "gradient_after": {"L": ["a" * 64], "M": [None], "V": [None]}}
            for i, s in enumerate(self.group.samples)]
        update = {"notebook_reward_enabled": True, "notebook_backward": backward,
            "notebook_objective": self.objective,
            "notebook_objective_receipts": {self.tree.problem_id: self.tree.notebook["learning_objective"]},
            "step_before": 0, "routes": {self.tree.problem_id: True},
            "effective_rewards": {self.group.group_id: rewards}}
        path = self.fixture.root / "raw-0000.json"
        path.write_text(json.dumps({"kind": "TREE", "raw": json.loads(to_json(self.tree))}))
        result = {"notebook_signal_groups": 1, "notebook_gradient_contribution_verified": True}
        self.assertTrue(any(row["advantage"] > 0 for row in backward))
        with self.assertRaisesRegex(ValueError, "positive notebook reward-gradient"):
            notebook_learning_gate(self.fixture.root, result, update, self.protocol)


if __name__ == "__main__":
    unittest.main()
