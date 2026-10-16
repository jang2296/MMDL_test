"""Real tiny HF/PEFT G3 comparisons, never pretrained-4B or native vLLM proof."""
import unittest
from unittest.mock import patch

import torch

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl import gate_reference as gates
from fine_tuning.notebook_rl.policy import configure_policy, completion_logprobs
from fine_tuning.tests.test_notebook_rl_policy import config, inputs
from fine_tuning.tests.test_training import tiny_model


def fixtures():
    fresh, base = tiny_model(), tiny_model()
    bundle = configure_policy(base, None, "MOD-RV", config(), enforce_counts=False)
    one = inputs()
    two = {"input_ids": torch.tensor([[4, 3, 3, 3, 3, 5, 4, 3, 3, 3, 3, 5, 9, 10, 11, 2]]),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
        "pixel_values": torch.cat([one["pixel_values"], one["pixel_values"].flip(0)]),
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]])}
    return bundle, fresh, [("single", one, 3), ("multi", two, 3)]


class ReferenceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_initial_full_vocabulary_logits_and_reference_equal_fresh_model(self):
        bundle, fresh, probes = fixtures()
        fresh.eval()
        bundle.route.apply(False)
        before = gates._hashes(bundle.model)
        rng = torch.get_rng_state().clone()
        result = gates._measure_reference(bundle, fresh, probes, initial=True)
        self.assertFalse(result["native_execution"])
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["gpu_gates_certified"])
        self.assertEqual(result["max_abs"], 0)
        self.assertEqual(result["probe_count"], 2)
        for probe in result["probes"]:
            self.assertEqual(probe["policy_logits"]["shape"], [1, 3, 32])
            self.assertTrue(probe["reference_logits"]["bit_equal"])
        self.assertEqual(before, gates._hashes(bundle.model))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(fresh.training)
        self.assertFalse(bundle.route.vision_on)
        self.assertTrue(all(not p.requires_grad for p in bundle.params["V"]))
        self.assertFalse(bundle.model.get_base_model().lm_head._forward_hooks)
        self.assertIsNone(bundle.model.get_base_model().model.rope_deltas)
        self.assertIsNone(fresh.model.rope_deltas)

    def test_reference_remains_equal_after_actual_optimizer_updates_not_initial_policy(self):
        bundle, fresh, probes = fixtures()
        optimizer = torch.optim.AdamW([p for ps in bundle.params.values() for p in ps], lr=.01, weight_decay=0)
        for _ in range(2):
            lp = completion_logprobs(bundle, probes[0][1], 3)
            (lp[0, 0] - lp[0, 1]).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        bundle.route.apply(False)
        before = gates._hashes(bundle.model)
        result = gates._measure_reference(bundle, fresh, probes, initial=False)
        self.assertEqual(result["max_abs"], 0)
        self.assertNotIn("policy_logits", result["probes"][0])
        self.assertNotIn("post_sync_max_abs", result)  # No vLLM synchronization was measured here.
        with self.assertRaisesRegex(ValueError, "initial policy logits differs"):
            gates._measure_reference(bundle, fresh, probes, initial=True)
        self.assertEqual(before, gates._hashes(bundle.model))
        self.assertFalse(bundle.route.vision_on)
        self.assertFalse(bundle.model.get_base_model().lm_head._forward_hooks)

    def test_no_labels_extra_inputs_missing_images_or_changed_reference(self):
        for fault in ("labels", "cache", "missing", "length", "duplicate", "fresh", "base", "same"):
            bundle, fresh, probes = fixtures()
            if fault == "labels":
                probes[0][1]["labels"] = probes[0][1]["input_ids"]
            elif fault == "cache":
                probes[0][1]["past_key_values"] = ()
            elif fault == "missing":
                probes[0][1].pop("pixel_values")
            elif fault == "length":
                probes[0] = (probes[0][0], probes[0][1], 0)
            elif fault == "duplicate":
                probes[1] = probes[0]
            elif fault == "same":
                fresh = bundle.model.get_base_model()
            else:
                model = fresh if fault == "fresh" else bundle.model.get_base_model()
                with torch.no_grad():
                    model.model.visual.patch_embed.proj.weight.add_(1)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                gates._measure_reference(bundle, fresh, probes, initial=True)

    def test_existing_gradients_rejected_not_erased(self):
        bundle, fresh, probes = fixtures()
        p = bundle.params["M"][0]
        p.grad = torch.ones_like(p)
        before = p.grad.clone()
        with self.assertRaisesRegex(ValueError, "clean gradient"):
            gates._measure_reference(bundle, fresh, probes, initial=True)
        self.assertTrue(torch.equal(before, p.grad))
        p.grad = None
        probes[0][1]["pixel_values"].requires_grad_()
        with self.assertRaisesRegex(ValueError, "no input gradients"):
            gates._measure_reference(bundle, fresh, probes, initial=True)

    def test_failed_reference_restores_route_modes_rng_rope_and_removes_hook(self):
        bundle, fresh, probes = fixtures()
        bundle.route.apply(False)
        fresh.eval()
        body = bundle.model.get_base_model().model
        body.rope_deltas = torch.tensor([[7]])
        old_rope = body.rope_deltas
        flags = [p.requires_grad for p in bundle.model.parameters()]
        rng = torch.get_rng_state().clone()
        def bad(*args):
            bundle.route.apply(True)
            torch.rand(5)
            body.rope_deltas = torch.tensor([[99]])
            raise RuntimeError("fixture reference failure")
        with patch.object(gates.policy, "reference_logprobs", side_effect=bad), self.assertRaisesRegex(RuntimeError, "fixture"):
            gates._measure_reference(bundle, fresh, probes, initial=True)
        self.assertEqual(flags, [p.requires_grad for p in bundle.model.parameters()])
        self.assertFalse(bundle.route.vision_on)
        self.assertFalse(fresh.training)
        self.assertIs(body.rope_deltas, old_rope)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(bundle.model.get_base_model().lm_head._forward_hooks)

    def test_wrong_reference_branch_and_silent_route_drift_are_rejected(self):
        for fault in ("logits", "route"):
            bundle, fresh, probes = fixtures()
            bundle.route.apply(False)
            original = gates.policy.reference_logprobs
            def bad(*args):
                result = original(*args)
                if fault == "route":
                    bundle.route.apply(True)
                return result + (1 if fault == "logits" else 0)
            with self.subTest(fault=fault), patch.object(gates.policy, "reference_logprobs", side_effect=bad), self.assertRaises(ValueError):
                gates._measure_reference(bundle, fresh, probes, initial=True)
            self.assertFalse(bundle.route.vision_on)

    def test_inplace_rope_values_restored_after_real_forward_and_failure(self):
        for fail in (False, True):
            bundle, fresh, probes = fixtures()
            bodies = [bundle.model.get_base_model().model, fresh.model]
            originals = [torch.tensor([[7]]), torch.tensor([[11]])]
            snapshots = [value.clone() for value in originals]
            handles = []
            for body, original in zip(bodies, originals):
                body.rope_deltas = original
                def mutate(module, args, value=original):
                    value.add_(1)
                    if fail:
                        raise RuntimeError("fixture in-place RoPE failure")
                handles.append(body.register_forward_pre_hook(mutate))
            try:
                with self.subTest(fail=fail):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "fixture in-place"):
                            gates._measure_reference(bundle, fresh, probes, initial=True)
                    else:
                        result = gates._measure_reference(bundle, fresh, probes, initial=True)
                        self.assertEqual(result["max_abs"], 0)
                    for body, original, snapshot in zip(bodies, originals, snapshots):
                        self.assertIs(body.rope_deltas, original)
                        self.assertTrue(torch.equal(original, snapshot))
            finally:
                for handle in handles:
                    handle.remove()

    def test_forward_gradient_side_effect_rejected_and_cleared_in_both_models(self):
        for target in ("policy", "fresh", "input"):
            for fail in (False, True):
                bundle, fresh, probes = fixtures()
                tensor = (bundle.params["M"][0] if target == "policy" else
                          next(fresh.parameters()) if target == "fresh" else probes[0][1]["pixel_values"])
                def inject(module, args):
                    tensor.grad = torch.ones_like(tensor)
                    if fail:
                        raise RuntimeError("fixture gradient failure")
                handle = fresh.model.register_forward_pre_hook(inject)
                try:
                    error = RuntimeError if fail else ValueError
                    message = "fixture gradient" if fail else "unexpected gradients"
                    with self.subTest(target=target, fail=fail), self.assertRaisesRegex(error, message):
                        gates._measure_reference(bundle, fresh, probes, initial=True)
                    self.assertIsNone(tensor.grad)
                    self.assertTrue(all(p.grad is None for m in (bundle.model, fresh) for p in m.parameters()))
                    self.assertFalse(bundle.model.get_base_model().lm_head._forward_hooks)
                finally:
                    handle.remove()

    def test_public_gate_never_accepts_tiny_cpu_model_and_stage_is_explicit(self):
        bundle, fresh, probes = fixtures()
        for stage in ("INITIAL", "POST_SYNC", "COMPLETE"):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                gates.measure_reference(bundle, fresh, probes, config(), stage=stage)

    def test_compare_is_byte_exact_not_only_equal_numbers(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            gates._compare(torch.tensor([0.]), torch.tensor([-0.]), "signed zero")
        for actual in (torch.tensor([float("nan")]), torch.ones(2), torch.zeros(1).bfloat16()):
            with self.assertRaises(ValueError):
                gates._compare(torch.zeros(1), actual, "bad")
        value = torch.tensor([1., 2.])
        result = gates._compare(value, value.clone(), "good")
        self.assertEqual(result["actual_sha256"], tensor_sha256(value))


if __name__ == "__main__":
    unittest.main()
