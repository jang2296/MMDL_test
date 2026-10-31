"""Native-only same-forward CPU VJPs; no real-model learning claim."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, paired_notebook, synthesis_credit, updates
from fine_tuning.notebook_rl.engines import RolloutRequest
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import Sample, to_json
from fine_tuning.tests import test_notebook_rl_paired_integration as paired


NATIVE = synthesis_credit.NATIVE_ERROR_RELATIONSHIP
OUTPUT = synthesis_credit.OUTPUT_FINALIZATION
UNMEASURED = {"scope": "native_error_relationship_policy_loss_only", "measured": False}


class NativeRelationshipGradientTests(unittest.TestCase):
    def setUp(self):
        self.params = {key: [torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))]
                       for key in ("L", "M", "V")}
        self.bundle = SimpleNamespace(params=self.params)
        self.current = torch.stack((self.params["L"][0][0], self.params["M"][0][0],
                                    self.params["V"][0][0], self.params["L"][0][1] * 0)).unsqueeze(0)
        self.old = torch.zeros_like(self.current)
        self.mask = torch.ones_like(self.current, dtype=torch.bool)
        self.relationships: list[dict[str, str | float | list[int]]] = [
            {"credit_type": NATIVE, "reward": 1, "advantage": .5, "coefficient": .25, "token_indices": [0]},
            {"credit_type": OUTPUT, "reward": 1, "advantage": .5, "coefficient": .75, "token_indices": [1]}]
        self.credit = {"relationships": self.relationships,
            "relationship_token_advantages": [.25, .75, 0., 0.],
            "partial_token_advantages": [.25, .75, 1., 0.]}
        sample = Sample("sample", "problem", "group", "DIAG", 1, None,
                        [1], [1, 2, 3, 4], [-.1] * 4, "stop", "", 1, 0)
        request = RolloutRequest("request", [1], None, [], 1, 4, [1])
        self.action = updates._PreparedAction(sample, request, self.mask, 3., .5, 4, False, self.credit)
        self.config = {"clip": {"eps_low": .2, "eps_high": .28}}

    def measure(self, current=None, aggregate=None):
        return updates._native_relationship_gradient(self.bundle, self.action,
            self.current if current is None else current, self.old, self.old, self.mask, self.config, aggregate)

    def test_mixed_vector_excludes_output_facts_strict_and_kl_without_touching_grad(self):
        for values in self.params.values():
            values[0].grad = torch.ones_like(values[0])
        saved = {key: values[0].grad.clone() for key, values in self.params.items()}
        aggregate = updates._relationship_gradient(self.bundle, self.action, self.current,
                                                   self.old, self.old, self.mask, self.config)
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            native = self.measure(aggregate=aggregate)
        self.assertEqual(grad.call_count, 1)
        self.assertTrue(grad.call_args.kwargs["retain_graph"])
        self.assertEqual(native["nonzero_groups"], ["L"])
        self.assertEqual(native["group_norms"], {"L": .03125, "M": 0., "V": 0.})
        self.assertGreater(aggregate["group_norms"]["M"], 0)
        self.assertEqual((native["positive_tokens"], native["token_count"]), (1, 1))
        self.assertEqual(native["credit_sha256"], _hash(self.credit))
        self.assertEqual(native["token_advantages_sha256"], _hash([.25, 0., 0., 0.]))
        for key, values in self.params.items():
            torch.testing.assert_close(values[0].grad, saved[key], atol=0, rtol=0)
        self.current.sum().backward()
        self.assertFalse(torch.equal(self.params["L"][0].grad, saved["L"]))

    def test_identical_native_vector_reuses_the_same_forward_measurement(self):
        self.credit["relationships"] = self.relationships[:1]
        self.credit["relationship_token_advantages"] = [.25, 0., 0., 0.]
        aggregate = updates._relationship_gradient(self.bundle, self.action, self.current,
                                                   self.old, self.old, self.mask, self.config)
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            native = self.measure(aggregate=aggregate)
        grad.assert_not_called()
        self.assertEqual(native["basis"], "same_forward_identical_native_only_vector")
        self.assertEqual(native["group_norms"], aggregate["group_norms"])

    def test_masked_or_zero_native_and_output_only_do_not_run_a_vjp(self):
        for mode in ("masked", "zero", "output"):
            with self.subTest(mode=mode):
                if mode == "masked":
                    self.mask[0, 0] = False
                elif mode == "zero":
                    self.mask[0, 0] = True
                    self.relationships[0]["coefficient"] = 0.
                else:
                    self.credit["relationships"] = self.relationships[1:]
                with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
                    self.assertEqual(self.measure(), UNMEASURED)
                grad.assert_not_called()

    def test_positive_native_coefficient_does_not_prove_a_nonzero_derivative(self):
        current = torch.zeros_like(self.current) + self.params["L"][0].sum() * 0
        native = self.measure(current=current)
        self.assertTrue(native["measured"])
        self.assertEqual(native["positive_tokens"], 1)
        self.assertEqual(native["nonzero_groups"], [])
        self.assertEqual(native["group_norms"], dict.fromkeys(("L", "M", "V"), 0.))


class NativeRelationshipOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = paired.PairedNotebookIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f = self.fixture.f
        samples = episode.solution_samples(self.f.tree)
        samples[0].text = samples[0].text.rsplit("\nFinal answer:", 1)[0]
        for index, sample in enumerate(samples):
            parent = samples[0] if index in (2, 3) else samples[2] if index in (4, 5) else None
            sample.prompt_token_ids = ([1, 2] if parent is None else
                parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2])
            sample.completion_token_ids = self.f.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.result = self.f.verifier.grade(self.f.actor.problem_id, sample.text, sample.finish_reason,
                                                 self.f.actor.mode)
            sample.reward = sample.result.reward_answer
        pair_id = next(key for key, pair in paired_notebook.public_pairs(samples).items()
            if pair["earlier"]["attempt"] == 1 and pair["later"]["attempt"] == 3
            and pair["earlier"]["kind"] == "OUTPUT")
        self.output = {"pair_id": pair_id, "relation": "FINALIZED"}
        self.native = self.fixture.compact_lesson()

    def prepare(self, *, mixed=False, output_only=False):
        note = self.fixture.note
        texts = ([note([self.output, self.native]), note(), note(), note()] if mixed else
                 [note([self.output]), note() if output_only else note([self.native]), note(), note()])
        self.fixture.append(texts)
        updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier,
                                       self.fixture.protocol["notebook_objective"])
        return self.f.optimizer_fixture()

    def test_output_first_still_measures_later_native_and_off_route_preserves_vision(self):
        fixture = self.prepare()
        bundle, optimizer, scheduler = fixture[:3]
        before = to_json(self.f.tree)
        vision = bundle.model["V"].detach().clone()
        optimizer.state[bundle.model["V"]] = {"step": torch.tensor(1.),
            "exp_avg": torch.ones_like(vision), "exp_avg_sq": torch.ones_like(vision)}
        vision_state = copy.deepcopy(optimizer.state[bundle.model["V"]])
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            result = self.f.step(fixture)
        self.assertEqual((result["status"], scheduler.last_epoch, grad.call_count),
                         ("UPDATED_UNSYNCHRONIZED", 1, 2))
        rows = result["notebook_backward"]
        self.assertEqual([row["sample_id"] for row in rows if row["relationship_gradient"]["measured"]],
                         [self.fixture.group.samples[0].sample_id])
        measured = [row for row in rows if row["native_relationship_gradient"]["measured"]]
        self.assertEqual([row["sample_id"] for row in measured], [self.fixture.group.samples[1].sample_id])
        native = measured[0]["native_relationship_gradient"]
        self.assertEqual(native["nonzero_groups"], ["L"])
        self.assertGreater(native["positive_tokens"], 0)
        self.assertEqual(native["credit_sha256"], _hash(result["synthesis_token_credit"][measured[0]["sample_id"]]))
        self.assertFalse(measured[0]["vision_on"])
        torch.testing.assert_close(bundle.model["V"], vision, atol=0, rtol=0)
        for key, value in vision_state.items():
            torch.testing.assert_close(optimizer.state[bundle.model["V"]][key], value, atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))
        self.assertEqual(to_json(self.f.tree), before)

    def test_mixed_action_measurement_keeps_normal_update_and_forward_count_identical(self):
        measured = self.prepare(mixed=True)
        ordinary = self.f.optimizer_fixture()
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad, patch.object(
                updates, "completion_logprobs", wraps=measured[4]) as forward:
            result = self.f.step((*measured[:4], forward, measured[5]))
        self.assertEqual(grad.call_count, 2)
        forward_count = forward.call_count
        native = next(row["native_relationship_gradient"] for row in result["notebook_backward"]
                      if row["native_relationship_gradient"]["measured"])
        self.assertEqual(native["basis"], "same_forward_native_only_vector_vjp")
        with patch.object(updates, "_native_relationship_gradient", return_value=UNMEASURED), patch.object(
                updates, "completion_logprobs", wraps=ordinary[4]) as forward:
            baseline = self.f.step((*ordinary[:4], forward, ordinary[5]))
        self.assertEqual(forward_count, forward.call_count)
        for key in ("loss", "grad_norm", "step_after", "effective_rewards", "routes"):
            self.assertEqual(result[key], baseline[key])
        for key, parameter in measured[0].model.items():
            torch.testing.assert_close(parameter, ordinary[0].model[key], atol=0, rtol=0)

    def test_masked_native_tokens_do_not_supply_positive_proof(self):
        fixture = self.prepare(mixed=True)
        scores = updates.notebook_objective_scores(self.f.tree, self.fixture.protocol["notebook_objective"])
        credits = synthesis_credit.group_credit(self.fixture.group.samples, scores, self.f.engine.tokenizer)
        original_mask = updates.action_mask

        def mask_native(sample, algorithm, tokenizer):
            mask = original_mask(sample, algorithm, tokenizer)
            for entry in credits.get(sample.sample_id, {}).get("relationships", []):
                if entry["credit_type"] == NATIVE:
                    mask[0, entry["token_indices"]] = False
            return mask

        with patch.object(updates, "action_mask", side_effect=mask_native), patch.object(
                torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            result = self.f.step(fixture)
        self.assertEqual(grad.call_count, 1)
        self.assertTrue(all(row["native_relationship_gradient"] == UNMEASURED
                            for row in result["notebook_backward"]))

    def test_mis_excluded_native_action_does_not_consume_the_native_measurement(self):
        fixture = self.prepare()
        logprobs = fixture[4]
        excluded = self.fixture.group.samples[1].sample_id

        def shifted(bundle, inputs, completion_len, *, chunk):
            return logprobs(bundle, inputs, completion_len, chunk=chunk) + (
                .1 if inputs["sample"].sample_id == excluded else 0.)

        result = self.f.step((*fixture[:4], shifted, fixture[5]))
        self.assertGreater(result["masked_fraction"], 0)
        self.assertNotIn(excluded, [row["sample_id"] for row in result["notebook_backward"]])
        self.assertTrue(all(row["native_relationship_gradient"] == UNMEASURED
                            for row in result["notebook_backward"]))

    def test_output_only_actions_do_not_add_a_native_vjp(self):
        fixture = self.prepare(output_only=True)
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            result = self.f.step(fixture)
        self.assertEqual(grad.call_count, 1)
        self.assertTrue(all(row["native_relationship_gradient"] == UNMEASURED
                            for row in result["notebook_backward"]))

    def test_nonfinite_native_vjp_clears_gradients_before_commit(self):
        fixture = self.prepare()
        handle = fixture[0].model["L"].register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        self.addCleanup(handle.remove)
        with patch.object(updates, "_relationship_gradient", return_value={
                "scope": "relationship_policy_loss_only", "measured": False}), patch.object(
                fixture[1], "step", wraps=fixture[1].step) as step:
            with self.assertRaisesRegex(RuntimeError, "NONFINITE_RELATIONSHIP_GRADIENT"):
                self.f.step(fixture)
        step.assert_not_called()
        self.assertEqual((fixture[1].state, fixture[2].last_epoch), ({}, 0))
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))


if __name__ == "__main__":
    unittest.main()
