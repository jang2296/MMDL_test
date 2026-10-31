"""Authored native notes and real tiny CPU optimizer/VJP plumbing, not learned improvement."""
import copy
import json
import unittest
from unittest.mock import Mock, patch

import torch

from fine_tuning.notebook_rl import notebook_synthesis as synthesis, synthesis_credit, updates
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.notebook_learning import append_rewarded_notebooks
from fine_tuning.notebook_rl.notebook_objective import SYNTHESIS_CREDIT_VERSION, contract
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests.test_notebook_rl_notebook_synthesis import note_for, response
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration
from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleTokenizer


class SynthesisCreditUpdateTests(unittest.TestCase):
    threads: int

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.fixture = integration.SynthesisIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.tree, self.actor, self.verifier = self.fixture.tree, self.fixture.actor, self.fixture.verifier
        self.engine, self.images = self.fixture.engine, self.fixture.images
        self.protocol = build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        self.fixture.protocol = self.protocol
        self.objective = self.protocol["notebook_objective"]
        self.shifted_samples: set[str] = set()
        self.kl_shift = 0.

    def prepare(self, *, solution_signal=False, equal_relationships=False, mixed_units=False):
        if mixed_units:
            # Short fact literals need whole tokens; 12-character fixture chunks cross them.
            self.engine.tokenizer = ReversibleTokenizer()
            samples = solution_samples(self.tree)
            for index, sample in enumerate(samples):
                parent = samples[0] if index in (2, 3) else samples[2] if index in (4, 5) else None
                sample.prompt_token_ids = ([1, 2] if parent is None else
                    parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2])
                sample.completion_token_ids = self.engine.tokenizer.encode(sample.text) + [2]
                sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
        if solution_signal:
            sample = self.tree.groups[0].samples[1]
            sample.text = response(self.fixture.item.slots, wrong=False)
            sample.completion_token_ids = self.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.result = self.verifier.grade(self.tree.problem_id, sample.text, "stop", "obs")
            sample.reward = 1.
        audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        partial = note_for(self.tree, audit, self.fixture.item)
        partial["corrected_facts"] = {}  # Deliberately keep whole-note strict reward zero.
        missing = copy.deepcopy(partial)
        missing["lessons"] = []
        if mixed_units:
            key = next(iter(self.fixture.item.slots))
            partial["corrected_facts"] = {key: self.fixture.item.slots[key]}
            missing["corrected_facts"] = {key: "q999"}
        good, bad = (json.dumps(note, separators=(",", ":")) for note in (partial, missing))
        notes = [good] * 4 if equal_relationships else [good, bad] * 2
        observation = "Observed facts:\n" + "\n".join(
            f"{key} = {value}" for key, value in self.fixture.item.slots.items()) + "\nSolution:"
        self.engine.replies = [[observation], notes]
        generated = []
        append_rewarded_notebooks([self.tree], self.images, self.engine, self.verifier,
            local_prompts(self.protocol), RolloutConfig.from_protocol(self.protocol, "ALG-NB3", master_seed=3407),
            actors={self.tree.problem_id: self.actor}, on_group=generated.append,
            should_stop=lambda: bool(generated))
        self.group = self.tree.groups[-1]
        updates.bind_notebook_objective(self.tree, self.actor, self.verifier, self.objective)

    def optimizer_fixture(self):
        bundle, optimizer, scheduler, preparation, _, _ = self.fixture.optimizer_fixture()
        # Real adapter tensors have dimensions; preserve optimizer ownership here.
        for key in ("M", "V"):
            bundle.model[key].data = torch.zeros(1, dtype=torch.float64)
        bundle.model["F"] = torch.nn.Parameter(torch.tensor(.25, dtype=torch.float64), requires_grad=False)

        def logprobs(bundle, inputs, completion_len, *, chunk):
            sample = inputs["sample"]
            baseline = torch.tensor([sample.sampler_logprobs], dtype=torch.float64)
            values = bundle.model["L"][torch.tensor(sample.completion_token_ids)].unsqueeze(0)
            values = values + bundle.model["M"]
            if bundle.route.vision_on:
                values = values + bundle.model["V"]
            return baseline + values + (.1 if sample.sample_id in self.shifted_samples else 0.)

        def reference(bundle, inputs, completion_len, *, chunk):
            return torch.tensor([inputs["sample"].sampler_logprobs], dtype=torch.float64) + (
                self.kl_shift if inputs["sample"].kind == "DIAG" else 0.)

        return bundle, optimizer, scheduler, preparation, logprobs, reference

    def step(self, fixture, **kwargs):
        return self.fixture.step(fixture, **kwargs)

    def test_strict_zero_relationship_tokens_have_a_separate_real_vjp_and_off_vision(self):
        self.prepare()
        self.assertEqual(contract(SYNTHESIS_CREDIT_VERSION), synthesis_credit.contract())
        scores = updates.notebook_objective_scores(self.tree, self.objective)
        self.assertEqual([sample.reward for sample in self.group.samples], [0.] * 4)
        self.assertTrue(updates.group_has_signal(self.group, scores))
        self.assertTrue(updates.has_learning_signal(self.tree, include_notebooks=True,
                                                   notebook_objective=self.objective))
        fixture = self.optimizer_fixture()
        bundle, optimizer, scheduler = fixture[:3]
        optimizer.state[bundle.model["V"]] = {"step": torch.tensor(1.),
            "exp_avg": torch.ones_like(bundle.model["V"]), "exp_avg_sq": torch.ones_like(bundle.model["V"])}
        vision_state = copy.deepcopy(optimizer.state[bundle.model["V"]])
        before = to_json(self.tree)
        autograd = torch.autograd.grad

        def measured(*args, **kwargs):
            saved = [None if parameter.grad is None else parameter.grad.clone()
                     for parameter in bundle.model.parameters()]
            result = autograd(*args, **kwargs)
            for parameter, previous in zip(bundle.model.parameters(), saved, strict=True):
                if previous is None:
                    self.assertIsNone(parameter.grad)
                else:
                    torch.testing.assert_close(parameter.grad, previous, atol=0, rtol=0)
            return result

        with patch.object(torch.autograd, "grad", side_effect=measured) as grad, patch.object(
                updates, "group_loss", wraps=updates.group_loss) as loss, patch.object(
                optimizer, "step", wraps=optimizer.step) as step:
            result = self.step(fixture)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual((step.call_count, scheduler.last_epoch, grad.call_count), (1, 1, 1))
        self.assertTrue(grad.call_args.kwargs["retain_graph"])
        self.assertTrue(grad.call_args.kwargs["allow_unused"])
        self.assertNotIn("fact_token_credit", result)
        self.assertEqual(set(result["synthesis_token_credit"]), {sample.sample_id for sample in self.group.samples})
        rows = [row for row in result["notebook_backward"] if row["relationship_gradient"]["measured"]]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        credit = result["synthesis_token_credit"][row["sample_id"]]
        measured_gradient = row["relationship_gradient"]
        self.assertEqual(row["sample_id"], self.group.samples[0].sample_id)
        self.assertEqual(measured_gradient["scope"], "relationship_policy_loss_only")
        self.assertTrue(measured_gradient["finite"])
        self.assertEqual(measured_gradient["nonzero_groups"], ["L", "M"])
        self.assertGreater(measured_gradient["group_norms"]["L"], 0)
        self.assertGreater(measured_gradient["group_norms"]["M"], 0)
        self.assertEqual(measured_gradient["group_norms"]["V"], 0)
        self.assertEqual(measured_gradient["positive_tokens"],
                         sum(value > 0 for value in credit["relationship_token_advantages"]))
        self.assertEqual(measured_gradient["token_count"], row["action_tokens"])
        self.assertEqual(measured_gradient["credit_sha256"], _hash(credit))
        self.assertEqual(row["synthesis_credit_sha256"], _hash(credit))
        isolated = [call for call in loss.call_args_list if call.kwargs["beta"] == 0]
        self.assertEqual(len(isolated), 1)
        call = isolated[0]
        self.assertEqual(call.args[3].tolist(), [0.])
        self.assertEqual(call.kwargs["partial_adv"].tolist(), [credit["relationship_token_advantages"]])
        self.assertEqual(call.kwargs["norm_const"], 4 * self.protocol["rollout"]["diag"]["max_new_tokens"])
        self.assertEqual(float(bundle.model["V"]), 0)
        self.assertEqual(float(bundle.model["F"]), .25)
        for key, value in vision_state.items():
            torch.testing.assert_close(optimizer.state[bundle.model["V"]][key], value, atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))
        self.assertEqual(to_json(self.tree), before)

    def test_measurement_does_not_change_normal_update_or_forward_count(self):
        self.prepare()
        measured, ordinary = self.optimizer_fixture(), self.optimizer_fixture()
        measured_forward, ordinary_forward = Mock(wraps=measured[4]), Mock(wraps=ordinary[4])
        measured = (*measured[:4], measured_forward, measured[5])
        ordinary = (*ordinary[:4], ordinary_forward, ordinary[5])
        with patch.object(updates, "_relationship_gradient", wraps=updates._relationship_gradient):
            first = self.step(measured)
        with patch.object(updates, "_relationship_gradient", return_value={
                "scope": "relationship_policy_loss_only", "measured": False}):
            second = self.step(ordinary)
        for key, parameter in measured[0].model.items():
            torch.testing.assert_close(parameter, ordinary[0].model[key], atol=0, rtol=0)
        self.assertEqual(first["grad_norm"], second["grad_norm"])
        self.assertEqual(first["loss"], second["loss"])
        self.assertEqual(first["step_after"], second["step_after"])
        self.assertEqual(measured_forward.call_count, ordinary_forward.call_count)

    def test_control_disables_both_partial_components_and_keeps_kl(self):
        self.prepare(solution_signal=True, mixed_units=True)
        self.kl_shift = .005
        scores = updates.notebook_objective_scores(self.tree, self.objective)
        credits = synthesis_credit.group_credit(self.group.samples, scores, self.engine.tokenizer)
        self.assertTrue(any(any(value != 0 for value in row["fact_token_advantages"]) for row in credits.values()))
        self.assertTrue(any(any(value != 0 for value in row["relationship_token_advantages"])
                            for row in credits.values()))
        fixture = self.optimizer_fixture()
        before = to_json(self.tree)
        with patch.object(synthesis_credit, "group_credit", wraps=synthesis_credit.group_credit) as credit, patch.object(
                torch.autograd, "grad", wraps=torch.autograd.grad) as grad, patch.object(
                updates, "group_loss", wraps=updates.group_loss) as loss:
            result = self.step(fixture, notebook_reward_enabled=False)
        credit.assert_not_called()
        grad.assert_not_called()
        self.assertNotIn("synthesis_token_credit", result)
        self.assertNotIn("fact_token_credit", result)
        self.assertEqual(result["notebook_backward"], [])
        self.assertEqual(result["effective_rewards"][self.group.group_id], [0.] * 4)
        for call in loss.call_args_list[-4:]:
            self.assertEqual(call.args[3].tolist(), [0.])
            self.assertIsNone(call.kwargs["partial_adv"])
            self.assertEqual(call.kwargs["beta"], self.protocol["loss"]["kl"]["beta"])
        self.assertGreater(result["kl_mean"], 0)
        self.assertEqual(to_json(self.tree), before)

    def test_mis_never_recenters_credit_or_shrinks_full_group_denominator(self):
        self.prepare()
        scores = updates.notebook_objective_scores(self.tree, self.objective)
        expected = synthesis_credit.group_credit(self.group.samples, scores, self.engine.tokenizer)
        self.shifted_samples.add(self.group.samples[0].sample_id)
        with patch.object(updates, "group_loss", wraps=updates.group_loss) as loss:
            result = self.step(self.optimizer_fixture())
        self.assertGreater(result["masked_fraction"], 0)
        self.assertEqual(result["synthesis_token_credit"], expected)
        backward = {row["sample_id"]: row for row in result["notebook_backward"]}
        self.assertNotIn(self.group.samples[0].sample_id, backward)
        measured = [row for row in backward.values() if row["relationship_gradient"]["measured"]]
        self.assertEqual([row["sample_id"] for row in measured], [self.group.samples[2].sample_id])
        for call in loss.call_args_list:
            if call.kwargs["partial_adv"] is not None:
                self.assertEqual(call.kwargs["norm_const"], 4 * self.protocol["rollout"]["diag"]["max_new_tokens"])

    def test_equal_unit_vectors_do_not_fabricate_a_relationship_measurement(self):
        self.prepare(solution_signal=True, equal_relationships=True)
        scores = updates.notebook_objective_scores(self.tree, self.objective)
        self.assertFalse(updates.group_has_signal(self.group, scores))
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            result = self.step(self.optimizer_fixture())
        grad.assert_not_called()
        self.assertTrue(result["notebook_backward"])
        self.assertTrue(all(row["relationship_gradient"] == {
            "scope": "relationship_policy_loss_only", "measured": False} for row in result["notebook_backward"]))

    def test_tampered_scores_fail_before_forward_and_late_receipts_before_step(self):
        self.prepare()
        baseline = copy.deepcopy(self.tree.notebook["learning_objective"])
        first = self.group.samples[0].sample_id
        self.tree.notebook["learning_objective"]["samples"][first]["atomic_reward"] += .1
        receipt = self.tree.notebook["learning_objective"]
        receipt["sha256"] = _hash({key: value for key, value in receipt.items() if key != "sha256"})
        fixture = self.optimizer_fixture()
        with patch.object(updates, "completion_logprobs") as forward, patch.object(
                fixture[1], "step", wraps=fixture[1].step) as step:
            with self.assertRaises(ValueError):
                self.step(fixture)
        forward.assert_not_called()
        step.assert_not_called()
        self.tree.notebook["learning_objective"] = baseline

        def mutate(event):
            if event["stage"] == "optimizer":
                receipt = self.tree.notebook["learning_objective"]
                receipt["samples"][first]["strict_report_sha256"] = "0" * 64
                receipt["sha256"] = _hash({key: value for key, value in receipt.items() if key != "sha256"})

        fixture = self.optimizer_fixture()
        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
            with self.assertRaises(ValueError):
                self.step(fixture, on_progress=mutate)
        step.assert_not_called()
        self.assertEqual(fixture[2].last_epoch, 0)
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))
        self.assertEqual(fixture[1].state, {})

    def test_nonfinite_relationship_vjp_clears_gradients_without_optimizer_commit(self):
        self.prepare()
        fixture = self.optimizer_fixture()
        handle = fixture[0].model["M"].register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        self.addCleanup(handle.remove)
        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
            with self.assertRaisesRegex(RuntimeError, "NONFINITE_RELATIONSHIP_GRADIENT"):
                self.step(fixture)
        step.assert_not_called()
        self.assertEqual(fixture[2].last_epoch, 0)
        self.assertEqual(fixture[1].state, {})
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))

    def test_late_token_credit_mutation_is_recomputed_before_optimizer(self):
        self.prepare()
        captured = []
        original = synthesis_credit.group_credit

        def credit(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result)
            return result

        def mutate(event):
            if event["stage"] == "optimizer":
                row = captured[0][self.group.samples[0].sample_id]
                row["partial_token_advantages"][0] += .01

        fixture = self.optimizer_fixture()
        with patch.object(synthesis_credit, "group_credit", side_effect=credit), patch.object(
                fixture[1], "step", wraps=fixture[1].step) as step:
            with self.assertRaisesRegex(ValueError, "synthesis token credit changed"):
                self.step(fixture, on_progress=mutate)
        self.assertEqual(len(captured), 2)
        step.assert_not_called()
        self.assertEqual(fixture[2].last_epoch, 0)
        self.assertEqual(fixture[1].state, {})
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))


if __name__ == "__main__":
    unittest.main()
