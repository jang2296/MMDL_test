"""Tiny CPU gradients, fixed logical normalization and whole-ordinal MIS checks."""
from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch
from dataclasses import replace

import torch

from fine_tuning.notebook_rl import synthesis_credit as credit, updates, episode_notebook as episode
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as native
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests.test_notebook_rl_pages_credit import page_fixture


class PagesUpdatesTests(unittest.TestCase):
    threads: int

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        fixture = page_fixture(self, partial=True)
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier, fixture.protocol["notebook_objective"])
        return fixture

    def test_one_logical_diag_and_fixed_8192_normalizer_have_real_native_gradient(self):
        fixture = self.fixture()
        optimizer_fixture = fixture.optimizer_fixture()
        bundle, optimizer, scheduler = optimizer_fixture[:3]
        original_loss = updates.group_loss
        weights = []

        def loss(*args, **kwargs):
            if kwargs.get("partial_adv") is not None:
                self.assertEqual(kwargs["norm_const"], 4 * 8192)
            return original_loss(*args, **kwargs)

        original_gradient = updates._relationship_gradient_norms

        def gradient(bundle, action, *args):
            if action.sample.kind == "DIAG":
                weights.append(action.weight)
                self.assertEqual(action.norm, 4 * 8192)
            return original_gradient(bundle, action, *args)

        with patch.object(updates, "group_loss", side_effect=loss), patch.object(
                updates, "_relationship_gradient_norms", side_effect=gradient), patch.object(
                optimizer, "step", wraps=optimizer.step) as step:
            result = fixture.step(optimizer_fixture)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual(step.call_count, 1)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertTrue(torch.count_nonzero(bundle.model["L"]))
        groups = [group for group in fixture.tree.groups if group.usable]
        expected = fixture.protocol["loss"]["group_weights"]["DIAG"] / (
            sum(fixture.protocol["loss"]["group_weights"][group.kind] for group in groups if group.kind != "DIAG")
            + fixture.protocol["loss"]["group_weights"]["DIAG"])
        self.assertTrue(weights)
        self.assertTrue(all(value == expected for value in weights))
        measured = [row["native_relationship_gradient"] for row in result["notebook_backward"]
                    if row["native_relationship_gradient"]["measured"]]
        self.assertTrue(any(row["nonzero_groups"] for row in measured))
        self.assertTrue(result["retained_native_relationship_signal"]["policy_gradient_present"])
        self.assertEqual(updates.effective_mis_deltas(result, fixture.protocol, trees=[fixture.tree]),
                         result["effective_mean_log_ratio_by_sample"])
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))

    def test_bundle_mis_pools_actual_tokens_and_keeps_all_pages_together(self):
        fixture = self.fixture()
        optimizer_fixture = list(fixture.optimizer_fixture())
        baseline = optimizer_fixture[4]
        groups = [group for group in fixture.tree.groups if group.kind == "DIAG"]
        shifted = groups[-1].samples[0].sample_id

        def old(bundle, inputs, completion_len, *, chunk):
            values = baseline(bundle, inputs, completion_len, chunk=chunk)
            return values + (.2 if inputs["sample"].sample_id == shifted else 0.)

        optimizer_fixture[4] = old
        result = fixture.step(optimizer_fixture)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        physical, effective = result["mean_log_ratio_by_sample"], result["effective_mean_log_ratio_by_sample"]
        self.assertGreater(physical[shifted], .05)
        members = [group.samples[0].sample_id for group in groups]
        pooled = float(torch.tensor(sum(result["masked_log_ratio_sum_by_sample"][sid] for sid in members)
            / sum(result["action_token_counts_by_sample"][sid] for sid in members), dtype=torch.float32))
        self.assertLess(pooled, .05)
        self.assertEqual({effective[sid] for sid in members}, {pooled})
        self.assertTrue(result["bundle_mis"]["bundles"][0]["keep"])
        # The physically mismatched page remains in its retained logical ordinal.
        self.assertIn(shifted, {row["sample_id"] for row in result["notebook_backward"]})
        self.assertEqual(updates.validate_page_mis_evidence(result, fixture.protocol), effective)

    def test_bundle_mis_rejects_every_page_and_preserves_original_n4_centering(self):
        fixture = self.fixture()
        optimizer_fixture = list(fixture.optimizer_fixture())
        baseline = optimizer_fixture[4]
        groups = [group for group in fixture.tree.groups if group.kind == "DIAG"]
        rejected = {group.samples[0].sample_id for group in groups}

        def old(bundle, inputs, completion_len, *, chunk):
            values = baseline(bundle, inputs, completion_len, chunk=chunk)
            return values + (.1 if inputs["sample"].sample_id in rejected else 0.)

        optimizer_fixture[4] = old
        expected = {sid: row for group in groups for sid, row in credit.group_credit(group.samples,
            fixture.tree.notebook["learning_objective"]["samples"], fixture.engine.tokenizer).items()}
        result = fixture.step(optimizer_fixture)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertFalse(result["bundle_mis"]["bundles"][0]["keep"])
        self.assertFalse(rejected & {row["sample_id"] for row in result["notebook_backward"]})
        self.assertEqual(result["synthesis_token_credit"], expected)
        self.assertTrue(all(row["relationship_group_means"] == expected[sid]["relationship_group_means"]
                            for sid, row in result["synthesis_token_credit"].items()))

    def test_resealed_membership_counts_effective_delta_and_missing_actions_are_rejected(self):
        fixture = self.fixture()
        result = fixture.step(fixture.optimizer_fixture())
        for mutation in ("count", "delta", "physical", "member", "missing", "binding"):
            changed = copy.deepcopy(result)
            bundle = changed["bundle_mis"]["bundles"][0]
            sid = bundle["sample_ids"][0]
            if mutation == "count":
                changed["action_token_counts_by_sample"][sid] += 1
                memberships = updates._page_memberships(changed["notebook_objective_receipts"])
                effective, evidence = updates._page_mis_evidence(memberships, changed["mean_log_ratio_by_sample"],
                    changed["action_token_counts_by_sample"], fixture.protocol["loss"]["mismatch"]["delta_seq"],
                    changed["masked_log_ratio_sum_by_sample"])
                changed["effective_mean_log_ratio_by_sample"], changed["bundle_mis"] = effective, evidence
            elif mutation == "delta":
                changed["effective_mean_log_ratio_by_sample"][sid] = 9.
            elif mutation == "physical":
                changed["mean_log_ratio_by_sample"][sid] = 9.
            elif mutation == "member":
                bundle["sample_ids"][0], bundle["sample_ids"][1] = bundle["sample_ids"][1], bundle["sample_ids"][0]
                changed["bundle_mis"]["sha256"] = _hash({key: value for key, value in changed["bundle_mis"].items() if key != "sha256"})
            elif mutation == "missing":
                del changed["mean_log_ratio_by_sample"][sid]
            else:
                receipt = changed["notebook_objective_receipts"][fixture.tree.problem_id]
                receipt["samples"][sid]["page_binding"]["raw_pages"][0]["text"] += " "
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                updates.validate_page_mis_evidence(changed, fixture.protocol)

    def test_float32_mis_threshold_has_one_shared_boundary_rule(self):
        memberships = [{"problem_id": "fixture", "bundle_index": 0, "sample_ids": ["left", "right"]}]
        boundary = torch.tensor(.05, dtype=torch.float32)
        toward_zero = float(torch.nextafter(boundary, torch.tensor(0., dtype=torch.float32)))
        outward = float(torch.nextafter(boundary, torch.tensor(math.inf, dtype=torch.float32)))
        for value in (toward_zero, float(boundary), outward):
            for sign in (-1, 1):
                physical = {"left": sign * value, "right": sign * value}
                effective, evidence = updates._page_mis_evidence(memberships, physical,
                    {"left": 1, "right": 3}, .05, {"left": sign * value, "right": sign * value * 3})
                expected = bool(torch.tensor(sign * value, dtype=torch.float32).abs() <= .05)
                self.assertIs(evidence["bundles"][0]["keep"], expected)
                self.assertEqual(set(effective.values()), {sign * value})

    def test_zero_native_vjp_cannot_commit_despite_retained_positive_token_credit(self):
        fixture = self.fixture()
        optimizer_fixture = list(fixture.optimizer_fixture())

        def constant(bundle, inputs, completion_len, *, chunk):
            return torch.tensor([inputs["sample"].sampler_logprobs], dtype=torch.float64) + bundle.model["L"].sum() * 0

        optimizer_fixture[4] = constant
        with patch.object(optimizer_fixture[1], "step", wraps=optimizer_fixture[1].step) as step:
            result = fixture.step(optimizer_fixture)
        self.assertEqual(result["status"], "NO_NATIVE_RELATIONSHIP_GRADIENT")
        self.assertTrue(result["retained_native_relationship_signal"]["policy_gradient_present"])
        step.assert_not_called()
        self.assertEqual(optimizer_fixture[2].last_epoch, 0)
        self.assertTrue(all(parameter.grad is None for parameter in optimizer_fixture[0].model.parameters()))

    def test_answer_signal_tree_without_pages_adds_no_phantom_diag_weight(self):
        fixture = self.fixture()
        actor = replace(fixture.actor, problem_id="fixture-answer-only-no-recovery")
        fixture.verifier.actors[actor.problem_id] = actor
        fixture.verifier.items[actor.problem_id] = replace(fixture.item, problem_id=actor.problem_id)
        wrong, right = native.response(fixture.item.slots), native.response(fixture.item.slots, wrong=False)
        tree = native.tree_from(actor, fixture.verifier, [wrong, right, wrong, wrong, wrong, wrong])
        identities = {sample.sample_id: f"{actor.problem_id}:{sample.sample_id}"
                      for group in tree.groups for sample in group.samples}
        for group in tree.groups:
            group.group_id = f"{actor.problem_id}:{group.group_id}"
            for sample in group.samples:
                sample.sample_id, sample.group_id = identities[sample.sample_id], group.group_id
                if sample.parent_sample_id is not None:
                    sample.parent_sample_id = identities[sample.parent_sample_id]
        samples = episode.solution_samples(tree)
        for index, sample in enumerate(samples):
            sample.reward = float(sample.result is not None and sample.result.status == "CORRECT")
            sample.completion_token_ids = fixture.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            if sample.parent_sample_id is not None:
                parent = next(value for value in samples if value.sample_id == sample.parent_sample_id)
                sample.prompt_token_ids = parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2]
        fixture.images[tree.problem_id] = fixture.images[fixture.tree.problem_id]
        episode.append_episode_notebooks([tree], {tree.problem_id: fixture.images[tree.problem_id]}, fixture.engine,
            fixture.verifier, fixture.prompts, fixture.cfg, actors={tree.problem_id: actor})
        self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "NO_VERIFIED_RECOVERY")
        updates.bind_notebook_objective(tree, actor, fixture.verifier, fixture.protocol["notebook_objective"])
        self.assertEqual(tree.notebook["learning_objective"]["samples"], {})
        self.assertTrue(updates.has_learning_signal(tree, include_notebooks=True,
                                                   notebook_objective=fixture.protocol["notebook_objective"]))
        optimizer_fixture = fixture.optimizer_fixture()
        bundle, optimizer, scheduler, preparation, logprobs, reference = optimizer_fixture
        all_samples = {sample.sample_id: sample for current in (fixture.tree, tree) for group in current.groups for sample in group.samples}
        preparation.model_inputs = lambda request, completion, device: {"sample": all_samples[request.request_id]}
        weights = []
        measure = updates._relationship_gradient_norms

        def gradient(bundle, action, *args):
            weights.append(action.weight)
            return measure(bundle, action, *args)

        with patch.object(updates, "completion_logprobs", side_effect=logprobs), patch.object(
                updates, "reference_logprobs", side_effect=reference), patch.object(
                updates, "_relationship_gradient_norms", side_effect=gradient):
            result = updates.optimizer_step(bundle, optimizer, scheduler, [fixture.tree, tree], fixture.images,
                preparation, fixture.protocol, "ALG-NB3", step=0, seed=3407,
                observation_actors={fixture.actor.problem_id: fixture.actor, actor.problem_id: actor},
                objective_verifier=fixture.verifier)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        lambdas = fixture.protocol["loss"]["group_weights"]
        denominator = sum(lambdas[group.kind] for current in (fixture.tree, tree) for group in current.groups
                          if group.usable and group.kind != "DIAG") + lambdas["DIAG"]
        self.assertTrue(weights)
        self.assertEqual(set(weights), {lambdas["DIAG"] / denominator})
        self.assertEqual(len(result["bundle_mis"]["bundles"]), 4)
        self.assertEqual(updates.effective_mis_deltas(result, fixture.protocol, trees=[tree, fixture.tree]),
                         result["effective_mean_log_ratio_by_sample"])

    def test_missing_page_or_over_cap_is_rejected_before_optimizer(self):
        for mutation in ("missing", "cap"):
            fixture = self.fixture()
            optimizer_fixture = fixture.optimizer_fixture()
            if mutation == "missing":
                fixture.tree.groups.pop()
            else:
                group = fixture.tree.groups[-1]
                cap = fixture.tree.notebook["rewarded_notebook"]["page_plan"]["pages"][-1]["max_new_tokens"]
                group.samples[0].completion_token_ids = [100] * (cap + 1)
                group.samples[0].sampler_logprobs = [-.1] * (cap + 1)
            with patch.object(optimizer_fixture[1], "step") as step, self.assertRaises(ValueError):
                fixture.step(optimizer_fixture)
            step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
