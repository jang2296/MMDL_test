"""Native-only commit permission on authored CPU fixtures, not real-model learning."""
import copy
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import local_profile, synthesis_credit, updates
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests import test_notebook_rl_native_relationship_gradient as fixtures


class NativeCommitGateTests(unittest.TestCase):
    threads: int

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.fixture = fixtures.NativeRelationshipOptimizerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.paired = self.fixture.fixture
        self.f = self.fixture.f
        self.prompts = local_profile.local_prompts(self.paired.protocol)

    def prepare(self, *, required=True, native_count=1):
        note = self.paired.note
        self.paired.append([note([self.fixture.output])]
            + [note([self.fixture.native])] * native_count + [note()] * (3 - native_count))
        updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier,
                                       self.f.protocol["notebook_objective"])
        fixture = self.f.optimizer_fixture()
        if required:
            self.f.protocol["notebook_update_requirement"] = updates.NATIVE_RELATIONSHIP_REQUIREMENT
        return fixture

    def run_step(self, fixture, **kwargs):
        # Protocol/CLI selection is integrated separately; fresh verifier and credit checks run here.
        with patch.object(local_profile, "local_prompts", return_value=self.prompts):
            return self.f.step(fixture, **kwargs)

    def assert_no_step(self, fixture, expected, *, before_reference=False, **kwargs):
        bundle, optimizer, scheduler = fixture[:3]
        before = {key: parameter.detach().clone() for key, parameter in bundle.model.items()}
        state = copy.deepcopy(optimizer.state_dict())
        with patch.object(optimizer, "step", wraps=optimizer.step) as step, patch.object(
                scheduler, "step", wraps=scheduler.step) as schedule, patch.object(
                optimizer, "zero_grad", wraps=optimizer.zero_grad) as zero:
            result = self.run_step(fixture, **kwargs)
        self.assertEqual(result["status"], expected)
        self.assertEqual((result["step_before"], result["step_after"], scheduler.last_epoch), (0, 0, 0))
        self.assertFalse(result["engine_sync_required"])
        step.assert_not_called()
        schedule.assert_not_called()
        self.assertEqual(optimizer.state_dict(), state)
        self.assertEqual(zero.call_count, 0 if before_reference else 1)
        for key, parameter in bundle.model.items():
            torch.testing.assert_close(parameter, before[key], atol=0, rtol=0)
            self.assertIsNone(parameter.grad)
        return result

    def test_output_only_credit_stops_before_reference_and_backward(self):
        fixture = self.prepare(native_count=0)
        events: list[dict] = []
        reference = unittest.mock.Mock(side_effect=AssertionError("native permission precedes reference"))
        result = self.assert_no_step((*fixture[:5], reference), "NO_RETAINED_NATIVE_RELATIONSHIP_SIGNAL",
                                    before_reference=True, on_progress=events.append)
        reference.assert_not_called()
        self.assertFalse(result["retained_native_relationship_signal"]["policy_gradient_present"])
        self.assertTrue(all(event["stage"] == "old" for event in events))

    def test_mis_excluded_native_or_entire_batch_stops_before_reference(self):
        fixture = self.prepare()
        native_id = self.paired.group.samples[1].sample_id
        for all_actions in (False, True):
            def logprobs(bundle, inputs, completion_len, *, chunk):
                value = fixture[4](bundle, inputs, completion_len, chunk=chunk)
                return value + (.1 if all_actions or inputs["sample"].sample_id == native_id else 0.)

            with self.subTest(all_actions=all_actions):
                result = self.assert_no_step((*fixture[:4], logprobs,
                    unittest.mock.Mock(side_effect=AssertionError("no reference"))),
                    "NO_RETAINED_NATIVE_RELATIONSHIP_SIGNAL", before_reference=True)
                self.assertGreater(result["masked_fraction"], 0)

    def test_native_token_mask_excludes_the_only_positive_native_action(self):
        fixture = self.prepare()
        scores = updates.notebook_objective_scores(self.f.tree, self.f.protocol["notebook_objective"])
        credits = synthesis_credit.group_credit(self.paired.group.samples, scores, self.f.engine.tokenizer)
        original = updates.action_mask

        def mask_native(sample, algorithm, tokenizer):
            mask = original(sample, algorithm, tokenizer)
            for entry in credits.get(sample.sample_id, {}).get("relationships", []):
                if entry["credit_type"] == fixtures.NATIVE:
                    mask[0, entry["token_indices"]] = False
            return mask

        with patch.object(updates, "action_mask", side_effect=mask_native):
            self.assert_no_step((*fixture[:5], unittest.mock.Mock(side_effect=AssertionError("no reference"))),
                                "NO_RETAINED_NATIVE_RELATIONSHIP_SIGNAL", before_reference=True)

    def test_dtype_underflow_is_not_positive_retained_native_credit(self):
        fixture = self.prepare()
        original = synthesis_credit.group_credit

        def tiny_native(*args, **kwargs):
            credits = original(*args, **kwargs)
            for credit in credits.values():
                for entry in credit["relationships"]:
                    if entry["credit_type"] == fixtures.NATIVE and entry["coefficient"] > 0:
                        entry["coefficient"] = 1e-50
                        for index in entry["token_indices"]:
                            credit["relationship_token_advantages"][index] = 1e-50
                            credit["partial_token_advantages"][index] = 1e-50
            return credits

        def half_logprobs(*args, **kwargs):
            return fixture[4](*args, **kwargs).to(torch.float16)

        with patch.object(synthesis_credit, "group_credit", side_effect=tiny_native):
            result = self.assert_no_step((*fixture[:4], half_logprobs,
                unittest.mock.Mock(side_effect=AssertionError("no reference"))),
                "NO_RETAINED_NATIVE_RELATIONSHIP_SIGNAL", before_reference=True)
        self.assertTrue(any(entry["coefficient"] == 1e-50
            for credit in result["synthesis_token_credit"].values() for entry in credit["relationships"]))

    def zero_native_logprobs(self, fixture, sample_ids):
        def logprobs(bundle, inputs, completion_len, *, chunk):
            if inputs["sample"].sample_id in sample_ids:
                return torch.tensor([inputs["sample"].sampler_logprobs], dtype=torch.float64) + bundle.model["L"].sum() * 0
            return fixture[4](bundle, inputs, completion_len, chunk=chunk)
        return logprobs

    def test_zero_native_vjp_blocks_output_update_and_clears_existing_gradients(self):
        fixture = self.prepare()
        native_id = self.paired.group.samples[1].sample_id
        gradients_at_commit = []

        def observe(event):
            if event["stage"] == "optimizer":
                gradients_at_commit.append(any(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
                                                for parameter in fixture[0].model.parameters()))

        result = self.assert_no_step((*fixture[:4], self.zero_native_logprobs(fixture, {native_id}), fixture[5]),
                                    "NO_NATIVE_RELATIONSHIP_GRADIENT", on_progress=observe)
        self.assertEqual(gradients_at_commit, [True])
        native = next(row["native_relationship_gradient"] for row in result["notebook_backward"]
                      if row["native_relationship_gradient"]["measured"])
        self.assertGreater(native["positive_tokens"], 0)
        self.assertEqual(native["nonzero_groups"], [])

    def test_required_mode_searches_past_zero_vjp_and_commits_later_native(self):
        fixture = self.prepare(native_count=2)
        first, later = self.paired.group.samples[1:3]
        changed = (*fixture[:4], self.zero_native_logprobs(fixture, {first.sample_id}), fixture[5])
        result = self.run_step(changed)
        self.assertEqual((result["status"], result["step_after"], fixture[2].last_epoch),
                         ("UPDATED_UNSYNCHRONIZED", 1, 1))
        measured = [row for row in result["notebook_backward"] if row["native_relationship_gradient"]["measured"]]
        self.assertEqual([row["sample_id"] for row in measured], [first.sample_id, later.sample_id])
        self.assertFalse(measured[0]["native_relationship_gradient"]["nonzero_groups"])
        self.assertTrue(measured[1]["native_relationship_gradient"]["nonzero_groups"])
        self.assertTrue(fixture[1].state)
        self.assertTrue(torch.count_nonzero(fixture[0].model["L"]))
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))

    def test_required_native_success_preserves_legacy_loss_and_parameters(self):
        fixture = self.prepare()
        legacy_fixture = self.f.optimizer_fixture()
        required = self.run_step(fixture)
        del self.f.protocol["notebook_update_requirement"]
        legacy = self.run_step(legacy_fixture)
        self.assertTrue(required["native_relationship_commit_required"])
        for key in ("status", "loss", "grad_norm", "effective_rewards", "routes"):
            self.assertEqual(required[key], legacy[key])
        self.assertNotIn("native_relationship_commit_required", legacy)
        for key, parameter in fixture[0].model.items():
            torch.testing.assert_close(parameter, legacy_fixture[0].model[key], atol=0, rtol=0)

    def test_legacy_keeps_first_measured_zero_vjp_behavior(self):
        fixture = self.prepare(required=False, native_count=2)
        first = self.paired.group.samples[1]
        result = self.run_step((*fixture[:4], self.zero_native_logprobs(fixture, {first.sample_id}), fixture[5]))
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        measured = [row for row in result["notebook_backward"] if row["native_relationship_gradient"]["measured"]]
        self.assertEqual([row["sample_id"] for row in measured], [first.sample_id])
        self.assertFalse(measured[0]["native_relationship_gradient"]["nonzero_groups"])

    def test_answer_only_control_is_exempt_without_native_success_evidence(self):
        samples = fixtures.episode.solution_samples(self.f.tree)
        sibling = samples[1]
        sibling.text = samples[-1].text
        sibling.completion_token_ids = self.f.engine.tokenizer.encode(sibling.text) + [2]
        sibling.sampler_logprobs = [-.1] * len(sibling.completion_token_ids)
        sibling.result = self.f.verifier.grade(sibling.problem_id, sibling.text, sibling.finish_reason, self.f.actor.mode)
        sibling.reward = sibling.result.reward_answer
        fixture = self.prepare(native_count=0)
        result = self.run_step(fixture, notebook_reward_enabled=False)
        self.assertEqual((result["status"], fixture[2].last_epoch), ("UPDATED_UNSYNCHRONIZED", 1))
        self.assertFalse(result["native_relationship_commit_required"])
        self.assertNotIn("retained_native_relationship_signal", result)
        self.assertNotIn("synthesis_token_credit", result)
        self.assertEqual(result["notebook_backward"], [])

    def test_nonfinite_native_vjp_keeps_health_stop_and_clears_gradients(self):
        fixture = self.prepare()
        handle = fixture[0].model["L"].register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        self.addCleanup(handle.remove)
        with patch.object(updates, "_relationship_gradient", return_value={"measured": False}), patch.object(
                fixture[1], "step", wraps=fixture[1].step) as step, patch.object(
                fixture[2], "step", wraps=fixture[2].step) as schedule:
            with self.assertRaisesRegex(RuntimeError, "NONFINITE_RELATIONSHIP_GRADIENT"):
                self.run_step(fixture)
        step.assert_not_called()
        schedule.assert_not_called()
        self.assertEqual((fixture[1].state, fixture[2].last_epoch), ({}, 0))
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))

    def test_missing_finite_or_matching_credit_proof_blocks_commit(self):
        fixture = self.prepare()
        original = updates._native_relationship_gradient
        for override in ({"finite": False}, {"credit_sha256": _hash("wrong credit")}, {"positive_tokens": 0},
                         {"credit_type": fixtures.OUTPUT}, {"scope": "relationship_policy_loss_only"},
                         {"group_norms": {"L": float("inf"), "M": 0., "V": 0.}}):
            def changed(*args, **kwargs):
                return original(*args, **kwargs) | override

            with self.subTest(override=override), patch.object(updates, "_native_relationship_gradient", side_effect=changed):
                self.assert_no_step(fixture, "NO_NATIVE_RELATIONSHIP_GRADIENT")

    def test_invalid_requirement_and_incompatible_contract_fail_before_old_pass(self):
        fixture = self.prepare()
        original = copy.deepcopy(self.f.protocol)
        invalid_requirements: list[object] = [None, True, 1, {}, "unknown"]
        mutations = [{"notebook_update_requirement": value} for value in invalid_requirements]
        mutations += [{"notebook_objective": None}, {"notebook_route_rule": None}, {"profile": "other"}]
        for mutation in mutations:
            self.f.protocol = original | mutation
            with self.subTest(mutation=mutation), patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
                with self.assertRaisesRegex(ValueError, "requirement|synthesis"):
                    self.run_step((*fixture[:4], unittest.mock.Mock(side_effect=AssertionError("no old")), fixture[5]))
                step.assert_not_called()
                self.assertEqual((fixture[1].state, fixture[2].last_epoch), ({}, 0))


if __name__ == "__main__":
    unittest.main()
