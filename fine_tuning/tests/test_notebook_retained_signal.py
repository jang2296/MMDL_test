"""Retained policy-coefficient permission on tiny CPU fixtures; no notebook-learning claim."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl import updates
from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.losses import group_loss
from fine_tuning.notebook_rl.notebook_learning import NOTEBOOK_REWARD_VERSION
from fine_tuning.tests import test_notebook_rl_updates as fixtures


class RetainedSignalTests(unittest.TestCase):
    threads: int

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        bundle, protocol = fixtures.bundle(), fixtures.config()
        tree = fixtures.tree(bundle, route="VISION_OFF")
        optimizer, scheduler = make_optimizer(bundle, protocol)
        return bundle, protocol, tree, optimizer, scheduler

    def run_step(self, fixture, **kwargs):
        bundle, protocol, tree, optimizer, scheduler = fixture
        return updates.optimizer_step(bundle, optimizer, scheduler, [tree], {"p": [object()]},
            fixtures.Preparation(), protocol, "ALG-T1", step=0, seed=3407, **kwargs)

    @staticmethod
    def append_group(tree, *, diagnostic=False, rewards=(0., 0.)):
        group = copy.deepcopy(tree.groups[0])
        group.group_id += ":extra"
        if diagnostic:
            group.kind = "DIAG"
        for sample, reward in zip(group.samples, rewards, strict=True):
            sample.group_id = group.group_id
            sample.sample_id += ":extra"
            sample.reward = reward
            if diagnostic:
                sample.kind, sample.result, sample.diag_reward = "DIAG", None, reward
        tree.groups.append(group)
        return group

    @staticmethod
    def reject(samples):
        for sample in samples:
            sample.sampler_logprobs = [value - .1 for value in sample.sampler_logprobs]

    def assert_no_mutation(self, fixture, expected_status, **kwargs):
        bundle, _, _, optimizer, scheduler = fixture
        before = {name: tensor_sha256(parameter) for name, parameter in bundle.model.named_parameters()}
        events: list[dict] = []
        with patch.object(optimizer, "step", wraps=optimizer.step) as step, patch.object(
                scheduler, "step", wraps=scheduler.step) as schedule, patch.object(
                optimizer, "zero_grad", wraps=optimizer.zero_grad) as zero, patch.object(
                updates, "reference_logprobs", side_effect=AssertionError("no reference before permission")):
            result = self.run_step(fixture, require_retained_learning_signal=True, on_progress=events.append, **kwargs)
        self.assertEqual(result["status"], expected_status)
        step.assert_not_called()
        schedule.assert_not_called()
        zero.assert_not_called()
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(optimizer.state, {})
        self.assertEqual(before, {name: tensor_sha256(parameter) for name, parameter in bundle.model.named_parameters()})
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))
        self.assertEqual((result["step_before"], result["step_after"], result["engine_sync_required"]), (0, 0, False))
        self.assertTrue(all(event["stage"] == "old" for event in events))
        return result

    def test_rejected_signal_and_retained_zero_pg_stop_before_reference_or_mutation(self):
        fixture = self.fixture()
        tree = fixture[2]
        self.append_group(tree)
        self.reject(tree.groups[0].samples)
        result = self.assert_no_mutation(fixture, "NO_RETAINED_LEARNING_SIGNAL")
        self.assertEqual(result["retained_learning_signal"], {
            "required": True, "basis": "scaled_policy_coefficients_at_current_equals_old_on_retained_action_tokens",
            "policy_gradient_present": False, "retained_action_count": 2,
            "nonzero_policy_gradient_tokens": 0, "signal_sample_ids": []})
        self.assertEqual(result["n_groups_by_kind"]["T1"], 2)
        self.assertEqual(result["masked_fraction"], .5)

    def test_all_rejected_actions_keep_the_existing_no_matched_status(self):
        fixture = self.fixture()
        self.reject(fixture[2].groups[0].samples)
        result = self.assert_no_mutation(fixture, "NO_MATCHED_ACTIONS")
        self.assertEqual(result["retained_learning_signal"]["retained_action_count"], 0)
        self.assertFalse(result["retained_learning_signal"]["policy_gradient_present"])

    def test_positive_or_negative_retained_answer_coefficient_permits_actual_tiny_update(self):
        for retained_index in (0, 1):
            fixture = self.fixture()
            samples = fixture[2].groups[0].samples
            self.reject([samples[1 - retained_index]])
            with self.subTest(retained_index=retained_index):
                result = self.run_step(fixture, require_retained_learning_signal=True)
                self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
                signal = result["retained_learning_signal"]
                self.assertTrue(signal["policy_gradient_present"])
                self.assertEqual(signal["signal_sample_ids"], [samples[retained_index].sample_id])
                self.assertEqual(signal["nonzero_policy_gradient_tokens"], len(samples[retained_index].completion_token_ids))
                self.assertEqual(fixture[4].last_epoch, 1)

    def test_zero_group_weight_does_not_count_raw_reward_variance(self):
        fixture = self.fixture()
        fixture[1]["loss"]["group_weights"]["T1"] = 0.
        self.append_group(fixture[2], diagnostic=True)
        result = self.assert_no_mutation(fixture, "NO_RETAINED_LEARNING_SIGNAL")
        self.assertEqual(result["retained_learning_signal"]["retained_action_count"], 4)

    def test_answer_only_control_excludes_observed_diag_rewards(self):
        fixture = self.fixture()
        note = self.append_group(fixture[2], diagnostic=True, rewards=(1., 0.))
        fixture[1].update(profile="local-nf4-notebook-rl-v1", notebook_reward=NOTEBOOK_REWARD_VERSION)
        fixture[2].notebook["rewarded_notebook"] = {
            "group_id": note.group_id, "status": "GENERATED", "reward_version": NOTEBOOK_REWARD_VERSION}
        self.reject(fixture[2].groups[0].samples)
        result = self.assert_no_mutation(fixture, "NO_RETAINED_LEARNING_SIGNAL", notebook_reward_enabled=False)
        self.assertEqual(result["effective_rewards"][note.group_id], [0., 0.])
        self.assertEqual([sample.reward for sample in note.samples], [1., 0.])

    def test_default_and_explicit_false_keep_legacy_kl_only_update_and_schema(self):
        source = fixtures.bundle()
        tree = fixtures.tree(source, route="VISION_OFF")
        self.append_group(tree)
        self.reject(tree.groups[0].samples)
        results = []
        for kwargs in ({}, {"require_retained_learning_signal": False}):
            bundle = copy.deepcopy(source)
            protocol = fixtures.config()
            optimizer, scheduler = make_optimizer(bundle, protocol)
            result = self.run_step((bundle, protocol, copy.deepcopy(tree), optimizer, scheduler), **kwargs)
            self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
            self.assertEqual(scheduler.last_epoch, 1)
            self.assertNotIn("retained_learning_signal", result)
            results.append({key: value for key, value in result.items() if key != "seconds"})
        self.assertEqual(*results)

    def test_boolean_and_same_policy_boundaries_reject_before_update(self):
        fixture = self.fixture()
        with patch.object(updates, "completion_logprobs") as forward, patch.object(
                fixture[3], "step", wraps=fixture[3].step) as step:
            for invalid in (None, 0, 1, "true"):
                with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "boolean"):
                    self.run_step(fixture, require_retained_learning_signal=invalid)
            fixture[2].groups[0].samples[0].policy_version = 1
            with self.assertRaises(ValueError):
                self.run_step(fixture, require_retained_learning_signal=True)
        forward.assert_not_called()
        step.assert_not_called()

    @staticmethod
    def action(advantage, partial=None, *, mask=(True, True, True), weight=.5, keep=True):
        return SimpleNamespace(keep=keep, old=torch.full((1, 3), -.1), norm=6, weight=weight,
            advantage=advantage, mask=torch.tensor([mask]), sample=SimpleNamespace(sample_id="coefficient"),
            partial_credit=None if partial is None else {"partial_token_advantages": partial})

    def test_masked_unmapped_canceled_and_zero_scale_coefficients_match_actual_pg_derivative(self):
        cases = [(self.action(.5, mask=(True, False, True)), 2), (self.action(-.5), 3),
                 (self.action(0., [0., .25, 0.], mask=(True, False, True)), 0),
                 (self.action(0., [0., 0., 0.]), 0), (self.action(.5, [-.5] * 3), 0),
                 (self.action(.5, [.25] * 3, weight=0.), 0), (self.action(.5, weight=1e-50), 0),
                 (self.action(0., [.25, 0., -.25]), 2)]
        for action, expected in cases:
            with self.subTest(advantage=action.advantage, partial=action.partial_credit, weight=action.weight):
                signal = updates._retained_learning_signal([action], {"eps_low": .2, "eps_high": .28})
                current = action.old.clone().requires_grad_()
                partial = None if action.partial_credit is None else torch.tensor(
                    [action.partial_credit["partial_token_advantages"]], dtype=current.dtype)
                loss = action.weight * group_loss(current, action.old, action.old, torch.tensor([action.advantage]),
                    action.mask, torch.tensor([True]), eps_low=.2, eps_high=.28, beta=0., norm_const=action.norm,
                    partial_adv=partial)
                derivative = torch.autograd.grad(loss, current)[0]
                self.assertEqual(int(torch.count_nonzero(derivative)), expected)
                self.assertEqual(signal["nonzero_policy_gradient_tokens"], expected)
                self.assertEqual(signal["policy_gradient_present"], bool(expected))

    def test_narrow_clips_use_conservative_independent_terms_for_opposing_advantages(self):
        action = self.action(.5, [-.5] * 3)
        signal = updates._retained_learning_signal([action], {"eps_low": 0., "eps_high": 0.})
        self.assertTrue(signal["policy_gradient_present"])
        self.assertEqual(signal["nonzero_policy_gradient_tokens"], 3)
        self.assertEqual(signal["basis"], "potential_independent_surrogate_policy_coefficients_on_retained_action_tokens")
        current = (action.old + .0005).requires_grad_()
        loss = group_loss(current, action.old, action.old, torch.tensor([action.advantage]), action.mask,
            torch.tensor([True]), eps_low=0., eps_high=0., beta=0., norm_const=action.norm,
            partial_adv=torch.tensor([action.partial_credit["partial_token_advantages"]]))
        self.assertEqual(int(torch.count_nonzero(torch.autograd.grad(loss, current)[0])), 3)


if __name__ == "__main__":
    unittest.main()
