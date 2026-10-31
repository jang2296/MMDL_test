"""Actual tiny CPU assessment gradients; no Qwen or improvement claim."""
import copy
import json
import re
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, local_profile, synthesis_credit as credit, updates
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_native_relationship_gradient as native
from fine_tuning.tests import test_notebook_rl_whole_reference_training_episode as whole
from fine_tuning.tests.test_notebook_rl_synthesis_integration import ChunkTokenizer


UNMEASURED = {"scope": "native_assessment_policy_loss_only", "measured": False}


class AssessmentTokenizer(ChunkTokenizer):
    def encode(self, text, **kwargs):
        encode = super().encode
        return [token for part in re.split(r'("(?:[^"\\]|\\.)*"|\[\])', text)
                for token in encode(part, **kwargs)]


class AssessmentGradientTests(unittest.TestCase):
    def setUp(self):
        self.f = native.NativeRelationshipGradientTests()
        self.f.setUp()
        self.f.credit["assessments"] = [{"ref": "a", "reward": 1, "advantage": .5,
            "coefficient": .25, "token_indices": [0], "expected_flags": "I"}]
        self.f.credit["assessment_token_advantages"] = [.25, 0., 0., 0.]

    def measure(self, current=None):
        f = self.f
        return updates._assessment_gradient(f.bundle, f.action, f.current if current is None else current,
                                            f.old, f.old + .5, f.mask, f.config)

    def test_assessment_only_vjp_excludes_other_channels_and_preserves_accumulated_gradients(self):
        f = self.f
        for values in f.params.values():
            values[0].grad = torch.ones_like(values[0])
        saved = {key: values[0].grad.clone() for key, values in f.params.items()}
        result = self.measure()
        self.assertTrue(updates._assessment_gradient_present(result, f.credit))
        self.assertEqual(result["group_norms"], {"L": .03125, "M": 0., "V": 0.})
        self.assertEqual(result["credit_type"], "NATIVE_ASSESSMENT")
        self.assertEqual(result["token_advantages_sha256"], _hash([.25, 0., 0., 0.]))
        for key, values in f.params.items():
            torch.testing.assert_close(values[0].grad, saved[key], atol=0, rtol=0)

    def test_zero_derivative_does_not_satisfy_assessment_gate(self):
        result = self.measure(self.f.current * 0)
        self.assertTrue(result["measured"])
        self.assertEqual(result["positive_tokens"], 1)
        self.assertFalse(updates._assessment_gradient_present(result, self.f.credit))

    def test_other_channels_mask_mis_and_dtype_cannot_fabricate_assessment_signal(self):
        f = self.f
        f.action.old, f.action.keep = f.old, True
        for mode in ("none", "masked", "mis", "dtype"):
            with self.subTest(mode=mode):
                f.mask[:] = True
                f.action.keep = True
                entry = f.credit["assessments"][0]
                entry["coefficient"] = .25
                entry["reward"] = 1
                if mode == "none":
                    entry["reward"] = 0
                elif mode == "masked":
                    f.mask[0, 0] = False
                elif mode == "mis":
                    f.action.keep = False
                else:
                    entry["coefficient"] = 1e-50
                    f.action.old = f.old.to(torch.float16)
                signal = updates._retained_assessment_signal([f.action])
                self.assertFalse(signal["policy_gradient_present"])
                self.assertEqual(signal["positive_assessment_tokens"], 0)

    def test_retained_signal_classifies_exact_expected_flags(self):
        f = self.f
        f.action.old, f.action.keep = f.old, True
        f.credit["assessments"] = [{"ref": str(index), "reward": 1, "advantage": .5,
            "coefficient": .25, "token_indices": [index], "expected_flags": flags}
            for index, flags in enumerate(("N", "I", "U", "IU"))]
        result = updates._retained_assessment_signal([f.action])
        self.assertEqual(result["positive_assessment_tokens"], 4)
        for value in result["by_expected_flags"].values():
            self.assertEqual(value, {"positive_assessment_tokens": 1, "signal_sample_ids": [f.action.sample.sample_id]})

    def accumulator(self):
        self.f.bundle.model = torch.nn.ParameterDict({key: values[0] for key, values in self.f.params.items()})
        return updates._AssessmentGradientAccumulator(self.f.bundle)

    def test_signed_nonzero_individual_vjps_cancel_exactly(self):
        accumulator = self.accumulator()
        f = self.f
        results = []
        for coefficient in (.25, -.25, .25, -.25):
            f.credit["assessments"][0]["coefficient"] = coefficient
            results.append(updates._assessment_gradient(f.bundle, f.action, f.current, f.old, f.old,
                f.mask, f.config, accumulator))
        self.assertTrue(all(row["group_norms"]["L"] > 0 for row in results))
        self.assertEqual(accumulator.receipt([])["group_norms"], dict.fromkeys(("L", "M", "V"), 0.))

    def test_zero_scalar_assessment_loss_has_nonzero_signed_aggregate(self):
        accumulator = self.accumulator()
        f = self.f
        f.credit["assessments"][0]["coefficient"] = .25
        f.credit["assessments"].append({"ref": "b", "reward": 0, "advantage": -.5,
            "coefficient": -.25, "token_indices": [1], "expected_flags": "N"})
        current = torch.stack((f.params["L"][0][0], f.params["M"][0][0],
                               f.params["V"][0][0], f.params["L"][0][1] * 0)).unsqueeze(0)
        result = updates._assessment_gradient(f.bundle, f.action, current, f.old, f.old,
            f.mask, f.config, accumulator)
        scalar_loss = updates.group_loss(current, f.old, f.old, torch.zeros(1, dtype=current.dtype),
            f.mask, torch.tensor([True]), eps_low=.2, eps_high=.28, beta=0, norm_const=f.action.norm,
            partial_adv=torch.tensor([[.25, -.25, 0., 0.]], dtype=current.dtype))
        self.assertEqual(float(scalar_loss.detach()), 0.)
        self.assertEqual(result["group_norms"], {"L": .03125, "M": .03125, "V": 0.})
        self.assertEqual(accumulator.receipt([])["nonzero_groups"], ["L", "M"])

    def test_negative_only_vjp_and_zero_vector_skip_preserve_gradients(self):
        accumulator = self.accumulator()
        f = self.f
        f.credit["assessments"][0].update(coefficient=-.25, reward=0, advantage=-.5)
        f.params["L"][0].grad = torch.ones_like(f.params["L"][0])
        result = updates._assessment_gradient(f.bundle, f.action, f.current, f.old, f.old,
            f.mask, f.config, accumulator)
        self.assertTrue(result["measured"])
        self.assertEqual(result["positive_tokens"], 0)
        self.assertEqual(accumulator.receipt([])["nonzero_groups"], ["L"])
        torch.testing.assert_close(f.params["L"][0].grad, torch.ones_like(f.params["L"][0]))
        f.credit["assessments"][0]["coefficient"] = 0.
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            self.assertEqual(updates._assessment_gradient(f.bundle, f.action, f.current, f.old, f.old,
                f.mask, f.config, accumulator), UNMEASURED)
        grad.assert_not_called()

    def test_off_on_off_keeps_fixed_parameter_identity(self):
        f = self.f
        accumulator = self.accumulator()
        f.credit["assessments"][0]["token_indices"] = [2]
        for vision_on in (False, True, False):
            f.params["V"][0].requires_grad_(vision_on)
            current = torch.stack((f.params["L"][0][0], f.params["M"][0][0],
                f.params["V"][0][0] if vision_on else f.params["L"][0][1] * 0,
                f.params["L"][0][1] * 0)).unsqueeze(0)
            updates._assessment_gradient(f.bundle, f.action, current, f.old, f.old, f.mask, f.config, accumulator)
        result = accumulator.receipt([])
        self.assertEqual(result["nonzero_groups"], ["V"])
        self.assertEqual([row["group"] for row in result["parameter_inventory"]], ["L", "M", "V"])


class AssessmentOptimizerTests(unittest.TestCase):
    threads: int

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.whole = whole.WholeReferenceTrainingEpisodeTests()
        self.whole.setUp()
        self.addCleanup(self.whole.doCleanups)
        self.f = self.whole.f
        f = self.f
        f.protocol = local_profile.build_protocol(verified_note_synthesis_credit=True,
            verified_note_assessment_credit=True, pair_draft_synthesis=True,
            whole_reference_notebook=True, context_limit=24576)
        f.prompts = local_profile.local_prompts(f.protocol)
        f.cfg = RolloutConfig.from_protocol(f.protocol, "ALG-NB3", master_seed=3407)
        f.engine.tokenizer = AssessmentTokenizer()
        samples = episode.solution_samples(f.tree)
        for index, sample in enumerate(samples):
            parent = samples[0] if index in (2, 3) else samples[2] if index in (4, 5) else None
            sample.prompt_token_ids = ([1, 2] if parent is None else
                parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2])
            sample.completion_token_ids = f.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
        good = json.loads(self.whole.good)
        good["corrected_facts"] = {}
        for row in good["assessments"]:
            row[3] = None
        bad = copy.deepcopy(good)
        target = next(row for row in bad["assessments"] if row[1] == "I")
        self.target_ref = target[0]
        target[1] = "N"
        self.good, self.bad = (json.dumps(note, separators=(",", ":")) for note in (good, bad))

    def prepare(self, notes=None):
        self.whole.append([self.good, self.bad] * 2 if notes is None else notes)
        f = self.f
        updates.bind_notebook_objective(f.tree, f.actor, f.verifier, f.protocol["notebook_objective"])
        return f.optimizer_fixture()

    def test_assessment_alone_drives_real_update_without_native_relationship_proof(self):
        fixture = self.prepare()
        f = self.f
        before = to_json(f.tree)
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            result = f.step(fixture)
        self.assertEqual((result["status"], fixture[2].last_epoch, grad.call_count),
                         ("UPDATED_UNSYNCHRONIZED", 1, 4))
        self.assertTrue(result["assessment_learning_verified"])
        self.assertTrue(result["assessment_commit_required"])
        self.assertNotIn("native_relationship_commit_required", result)
        self.assertNotIn("retained_native_relationship_signal", result)
        self.assertTrue(torch.count_nonzero(fixture[0].model["L"]))
        self.assertEqual([sample.reward for sample in f.tree.groups[-1].samples], [0.] * 4)
        signal = result["retained_assessment_signal"]
        self.assertGreater(signal["by_expected_flags"]["I"]["positive_assessment_tokens"], 0)
        for row in result["notebook_backward"]:
            self.assertFalse(row["native_relationship_gradient"]["measured"])
            token_credit = result["synthesis_token_credit"][row["sample_id"]]
            self.assertFalse(any(token_credit["fact_token_advantages"]))
            self.assertFalse(any(token_credit["relationship_token_advantages"]))
        measured = next(row for row in result["notebook_backward"] if row["assessment_gradient"]["measured"])
        self.assertTrue(updates._assessment_gradient_present(measured["assessment_gradient"],
                         result["synthesis_token_credit"][measured["sample_id"]]))
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))
        self.assertEqual(to_json(f.tree), before)

    def test_mis_excludes_every_positive_assessment_before_reference_or_optimizer(self):
        fixture = self.prepare()
        forward = fixture[4]
        excluded = {sample.sample_id for sample in self.f.tree.groups[-1].samples[::2]}

        def shifted(bundle, inputs, completion_len, *, chunk):
            return forward(bundle, inputs, completion_len, chunk=chunk) + (
                .1 if inputs["sample"].sample_id in excluded else 0.)

        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step, patch.object(
                updates, "reference_logprobs") as reference:
            result = self.f.step((*fixture[:4], shifted, reference))
        self.assertEqual(result["status"], "NO_RETAINED_ASSESSMENT_SIGNAL")
        self.assertFalse(result["assessment_learning_verified"])
        reference.assert_not_called()
        step.assert_not_called()
        self.assertEqual((fixture[1].state, fixture[2].last_epoch), ({}, 0))

    def test_strict_fact_and_relationship_variance_cannot_replace_assessment_variance(self):
        fixture = self.prepare([self.whole.good, self.good] * 2)
        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
            result = self.f.step(fixture)
        self.assertEqual(result["status"], "NO_RETAINED_ASSESSMENT_SIGNAL")
        self.assertFalse(result["assessment_learning_verified"])
        self.assertTrue(any(any(row["fact_token_advantages"]) for row in result["synthesis_token_credit"].values()))
        self.assertTrue(any(any(row["relationship_token_advantages"]) for row in result["synthesis_token_credit"].values()))
        self.assertTrue(any(row["strict_advantage"] > 0 for row in result["synthesis_token_credit"].values()))
        step.assert_not_called()

    def test_zero_assessment_vjp_blocks_commit_even_with_other_real_gradients(self):
        fixture = self.prepare()
        forward = fixture[4]
        scores = updates.notebook_objective_scores(self.f.tree, self.f.protocol["notebook_objective"])
        credits = credit.group_credit(self.f.tree.groups[-1].samples, scores, self.f.engine.tokenizer)

        def detached_assessment(bundle, inputs, completion_len, *, chunk):
            value = forward(bundle, inputs, completion_len, chunk=chunk)
            token_credit = credits.get(inputs["sample"].sample_id)
            if token_credit:
                mask = torch.tensor([token_credit["assessment_token_advantages"]]) != 0
                value = torch.where(mask, value.detach() + value * 0, value)
            return value

        def shifted_reference(bundle, inputs, completion_len, *, chunk):
            return fixture[5](bundle, inputs, completion_len, chunk=chunk) + (
                .005 if inputs["sample"].kind == "DIAG" else 0.)

        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
            result = self.f.step((*fixture[:4], detached_assessment, shifted_reference))
        self.assertEqual(result["status"], "NO_ASSESSMENT_GRADIENT")
        self.assertFalse(result["assessment_learning_verified"])
        self.assertTrue(any(row["changed_groups"] for row in result["notebook_backward"]))
        step.assert_not_called()
        self.assertEqual((fixture[1].state, fixture[2].last_epoch), ({}, 0))
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))

    def test_four_nonzero_signed_vjps_cancel_and_block_optimizer(self):
        fixture = self.prepare()
        scores = updates.notebook_objective_scores(self.f.tree, self.f.protocol["notebook_objective"])
        credits = credit.group_credit(self.f.tree.groups[-1].samples, scores, self.f.engine.tokenizer)

        def same_derivative(bundle, inputs, completion_len, *, chunk):
            sample = inputs["sample"]
            baseline = torch.tensor([sample.sampler_logprobs], dtype=torch.float64)
            token_credit = credits.get(sample.sample_id)
            mask = torch.zeros_like(baseline)
            if token_credit:
                first = next(i for i, value in enumerate(token_credit["assessment_token_advantages"]) if value)
                mask[0, first] = 1.
            return baseline + mask * bundle.model["L"][0]

        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
            result = self.f.step((*fixture[:4], same_derivative, fixture[5]))
        self.assertEqual(result["status"], "NO_ASSESSMENT_GRADIENT")
        self.assertTrue(all(row["assessment_gradient"]["nonzero_groups"] == ["L"] for row in result["notebook_backward"]))
        self.assertEqual(result["assessment_aggregate_gradient"]["nonzero_groups"], [])
        self.assertFalse(result["assessment_learning_verified"])
        step.assert_not_called()
        self.assertEqual((fixture[1].state, fixture[2].last_epoch), ({}, 0))
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))

    def test_assessment_optimizer_rejects_changed_n4_or_cap_before_forward(self):
        fixture = self.prepare()
        for mode in ("n4", "cap"):
            with self.subTest(mode=mode):
                if mode == "n4":
                    sample = self.f.tree.groups[-1].samples.pop()
                else:
                    self.f.protocol["rollout"]["diag"]["max_new_tokens"] = 4096
                with patch.object(updates, "completion_logprobs") as forward:
                    with self.assertRaisesRegex(ValueError, "original n4 and cap8192"):
                        self.f.step(fixture)
                forward.assert_not_called()
                if mode == "n4":
                    self.f.tree.groups[-1].samples.append(sample)

    def test_negative_only_retained_member_and_mis_excluded_member_are_exactly_recorded(self):
        fixture = self.prepare()
        original = fixture[4]
        excluded = self.f.tree.groups[-1].samples[-1].sample_id

        def shifted(bundle, inputs, completion_len, *, chunk):
            return original(bundle, inputs, completion_len, chunk=chunk) + (
                .1 if inputs["sample"].sample_id == excluded else 0.)

        with patch.object(updates, "completion_logprobs", side_effect=shifted) as forward, patch.object(
                updates, "reference_logprobs", side_effect=fixture[5]) as reference:
            result = updates.optimizer_step(fixture[0], fixture[1], fixture[2], [self.f.tree], self.f.images, fixture[3],
                self.f.protocol, "ALG-NB3", step=0, seed=3407,
                observation_actors={self.f.actor.problem_id: self.f.actor}, objective_verifier=self.f.verifier)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        members = result["assessment_aggregate_gradient"]["members"]
        self.assertEqual(len(members), 4)
        self.assertEqual(members[-1]["disposition"], "MIS_EXCLUDED")
        negative = members[1]
        self.assertEqual((negative["disposition"], negative["positive_tokens"]), ("VJP", 0))
        self.assertGreater(negative["token_count"], 0)
        self.assertEqual(forward.call_count, 2 * len(result["mean_log_ratio_by_sample"]) - 1)
        self.assertEqual(reference.call_count, len(result["mean_log_ratio_by_sample"]) - 1)

    def test_retained_zero_vector_is_covered_without_an_extra_vjp(self):
        fixture = self.prepare([self.good, self.bad, "{}", self.bad])
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            result = self.f.step(fixture)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        members = result["assessment_aggregate_gradient"]["members"]
        self.assertEqual((len(members), members[2]["disposition"], members[2]["token_count"]),
                         (4, "ZERO_VECTOR", 0))
        self.assertEqual(grad.call_count, 3)
        self.assertEqual(result["notebook_backward"][2]["assessment_gradient"], UNMEASURED)

    def test_parameter_inventory_rename_at_commit_is_rejected(self):
        fixture = self.prepare()

        def rename(event):
            if event["stage"] == "optimizer":
                fixture[0].model["renamed_L"] = fixture[0].model["L"]
                del fixture[0].model["L"]

        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
            with self.assertRaisesRegex(ValueError, "parameter inventory changed before optimizer"):
                self.f.step(fixture, on_progress=rename)
        step.assert_not_called()
        self.assertEqual((fixture[1].state, fixture[2].last_epoch), ({}, 0))
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))

    def test_uniform_correct_or_wrong_assessments_have_no_relative_update(self):
        for text in (self.good, self.bad):
            with self.subTest(text=text[:40]):
                self.setUp()
                fixture = self.prepare([text] * 4)
                scores = updates.notebook_objective_scores(self.f.tree, self.f.protocol["notebook_objective"])
                self.assertFalse(updates.group_has_signal(self.f.tree.groups[-1], scores))
                with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
                    with self.assertRaisesRegex(ValueError, "exclude signal-free solution trees"):
                        self.f.step(fixture)
                step.assert_not_called()
                self.assertEqual((fixture[1].state, fixture[2].last_epoch), ({}, 0))

    def test_legacy_or_missing_requirement_cannot_select_assessment_credit(self):
        fixture = self.prepare()
        for requirement in (None, updates.NATIVE_RELATIONSHIP_REQUIREMENT, "unknown"):
            with self.subTest(requirement=requirement):
                self.f.protocol["notebook_update_requirement"] = requirement
                with patch.object(updates, "completion_logprobs") as forward:
                    with self.assertRaises(ValueError):
                        self.f.step(fixture)
                forward.assert_not_called()

    def test_late_assessment_credit_mutation_is_recomputed_before_commit(self):
        captured = []
        original = credit.group_credit

        def capture(*args, **kwargs):
            value = original(*args, **kwargs)
            captured.append(value)
            return value

        def mutate(event):
            if event["stage"] == "optimizer":
                row = next(iter(captured[0].values()))
                row["assessment_token_advantages"][0] += .1

        fixture = self.prepare()
        with patch.object(credit, "group_credit", side_effect=capture), patch.object(
                fixture[1], "step", wraps=fixture[1].step) as step:
            with self.assertRaisesRegex(ValueError, "synthesis token credit changed"):
                self.f.step(fixture, on_progress=mutate)
        step.assert_not_called()
        self.assertEqual((fixture[1].state, fixture[2].last_epoch), ({}, 0))
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))

    def test_nonfinite_assessment_vjp_clears_gradients_without_optimizer_state(self):
        fixture = self.prepare()
        handle = fixture[0].model["L"].register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        self.addCleanup(handle.remove)
        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
            with self.assertRaisesRegex(RuntimeError, "NONFINITE_ASSESSMENT_GRADIENT"):
                self.f.step(fixture)
        step.assert_not_called()
        self.assertEqual((fixture[1].state, fixture[2].last_epoch), ({}, 0))
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))


if __name__ == "__main__":
    unittest.main()
