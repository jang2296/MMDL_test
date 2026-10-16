"""Real tiny-Qwen gradient fixtures, not pretrained 4B or native CUDA approvals."""
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl import gate_gradients as g
from fine_tuning.notebook_rl import policy
from fine_tuning.tests.test_notebook_rl_policy import bundle, config, inputs


class GradientRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def snapshot(self, b):
        return {"parameters": {n: tensor_sha256(p) for n, p in b.model.named_parameters()},
                "flags": [p.requires_grad for p in b.model.parameters()],
                "modes": [m.training for m in b.model.modules()], "route": b.route.vision_on,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state()}

    def assert_restored(self, b, before):
        after = self.snapshot(b)
        for key in ("parameters", "flags", "modes", "route", "python_rng"):
            self.assertEqual(before[key], after[key])
        self.assertTrue(torch.equal(before["torch_rng"], after["torch_rng"]))
        self.assertEqual(before["numpy_rng"][0], after["numpy_rng"][0])
        self.assertTrue(np.array_equal(before["numpy_rng"][1], after["numpy_rng"][1]))
        self.assertEqual(before["numpy_rng"][2:], after["numpy_rng"][2:])
        self.assertTrue(all(p.grad is None for p in b.model.parameters()))

    def test_actual_off_hard_freeze_on_and_mixed_gradients_restore_all_state(self):
        b = bundle()
        b.model.eval()
        b.model.get_base_model().model.visual.train()
        b.route.apply(False)
        body = b.model.get_base_model().model
        rope = torch.tensor([[7]])
        body.rope_deltas = rope
        before = self.snapshot(b)
        batch = inputs()
        report = g._measure_routes(b, batch, 3, chunk=1)
        for key in ("vision_off_grad_none", "nonvision_grad_finite_nonzero", "hard_freeze_gradient_match",
                    "vision_on_grad_nonzero", "mixed_on_only", "hf_base_unchanged", "installed_parameters_unchanged",
                    "temporary_gradients_cleared", "caller_state_restored"):
            self.assertTrue(report[key], key)
        self.assertEqual(report["vision_applicability"], "MEASURED")
        self.assertEqual(report["hard_freeze_comparison"]["max_abs_error"], 0)
        self.assertEqual(report["mixed_comparison"]["max_abs_error"], 0)
        self.assertEqual(report["device"], "cpu")
        self.assertFalse(report["cuda_policy_execution"])
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["gpu_gates_certified"])
        self.assertFalse(report["optimizer_updated"])
        self.assertIs(body.rope_deltas, rope)
        self.assertTrue(torch.equal(rope, torch.tensor([[7]])))
        self.assertEqual(report["input_fingerprints"], g._fingerprints(batch))
        self.assert_restored(b, before)
        json.dumps(report, allow_nan=False)

    def test_fml_marks_vision_on_and_mixed_not_applicable(self):
        b = bundle("MOD-FML")
        before = self.snapshot(b)
        report = g._measure_routes(b, inputs(), 3)
        self.assertIsNone(report["vision_on_grad_nonzero"])
        self.assertIsNone(report["mixed_on_only"])
        self.assertIsNone(report["on"])
        self.assertEqual(report["vision_applicability"], "NOT_APPLICABLE_MOD_FML")
        self.assertEqual(report["vision_on_status"], "NOT_APPLICABLE_MOD_FML")
        self.assertEqual(report["mixed_status"], "NOT_APPLICABLE_MOD_FML")
        self.assertEqual(report["vision_na_reason"], "NO_INSTALLED_VISION_ADAPTERS")
        self.assertEqual(report["vision_parameter_count"], 0)
        self.assert_restored(b, before)

    def test_preexisting_gradients_are_rejected_without_erasing_caller_state(self):
        b = bundle()
        parameter = b.params["L"][0]
        gradient = torch.zeros_like(parameter)
        parameter.grad = gradient
        before = self.snapshot(b)
        with self.assertRaisesRegex(ValueError, "pre-existing"):
            g._measure_routes(b, inputs(), 3)
        self.assertIs(parameter.grad, gradient)
        self.assertEqual(before["flags"], self.snapshot(b)["flags"])
        self.assertTrue(torch.equal(before["torch_rng"], torch.get_rng_state()))

    def test_gold_fields_input_gradients_and_too_short_actions_rejected(self):
        b = bundle()
        for batch, length in ((inputs() | {"labels": torch.ones(1, 10)}, 3),
                              (inputs() | {"gold": torch.ones(1)}, 3), (inputs(), 1)):
            with self.subTest(length=length), self.assertRaises(ValueError):
                g._measure_routes(b, batch, length)
        batch = inputs()
        batch["pixel_values"].requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "input gradients"):
            g._measure_routes(b, batch, 3)
        batch = inputs()
        gradient = torch.zeros_like(batch["pixel_values"])
        batch["pixel_values"].grad = gradient
        with self.assertRaisesRegex(ValueError, "input gradients"):
            g._measure_routes(b, batch, 3)
        self.assertIs(batch["pixel_values"].grad, gradient)
        self.assertTrue(all(p.grad is None for p in b.model.parameters()))

    def test_exception_restores_rng_modes_route_rope_and_clears_partial_gradients(self):
        b = bundle()
        b.model.eval()
        body = b.model.get_base_model().model
        body.rope_deltas = None
        before = self.snapshot(b)

        def fail(*args, **kwargs):
            random.random()
            np.random.random()
            torch.rand(2)
            body.rope_deltas = torch.ones(1)
            b.params["M"][0].grad = torch.ones_like(b.params["M"][0])
            raise RuntimeError("injected forward failure")

        with patch.object(policy, "completion_logprobs", side_effect=fail), \
                self.assertRaisesRegex(RuntimeError, "injected"):
            g._measure_routes(b, inputs(), 3)
        self.assertIsNone(body.rope_deltas)
        self.assert_restored(b, before)

    def test_broken_route_and_nonfinite_gradients_fail_with_cleanup(self):
        b = bundle()
        before = self.snapshot(b)
        original = b.route.apply
        with patch.object(b.route, "apply", side_effect=lambda enabled: original(True)), \
                self.assertRaisesRegex(ValueError, "route OFF"):
            g._measure_routes(b, inputs(), 3)
        self.assert_restored(b, before)
        handle = b.params["M"][0].register_hook(lambda grad: grad * float("nan"))
        try:
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                g._measure_routes(b, inputs(), 3)
        finally:
            handle.remove()
        self.assert_restored(b, before)

    def test_hard_freeze_comparison_does_not_reuse_the_route_wrapper(self):
        b = bundle()
        before = self.snapshot(b)
        original = policy.completion_logprobs

        def altered(*args, **kwargs):
            result = original(*args, **kwargs)
            parameter = b.params["M"][0]
            return result + (parameter.sum()-parameter.detach().sum())

        with patch.object(policy, "completion_logprobs", side_effect=altered), \
                self.assertRaisesRegex(ValueError, "gradient mismatch"):
            g._measure_routes(b, inputs(), 3)
        self.assert_restored(b, before)

    def test_mixed_accumulation_catches_vision_gradient_leak(self):
        b = bundle()
        before = self.snapshot(b)
        original = policy.completion_logprobs

        def leaking(*args, **kwargs):
            result = original(*args, **kwargs)
            if not b.route.vision_on:
                with torch.no_grad():
                    for parameter in b.params["V"]:
                        if parameter.grad is not None:
                            parameter.grad.add_(1)
            return result

        with patch.object(policy, "completion_logprobs", side_effect=leaking), \
                self.assertRaisesRegex(ValueError, "gradient mismatch"):
            g._measure_routes(b, inputs(), 3)
        self.assert_restored(b, before)

    def test_parameter_and_input_mutation_cannot_return_success(self):
        for mutation in ("parameter", "input"):
            b, batch = bundle(), inputs()
            original = policy.completion_logprobs
            changed = False

            def corrupt(*args, **kwargs):
                nonlocal changed
                if not changed:
                    with torch.no_grad():
                        (b.params["L"][0] if mutation == "parameter" else batch["pixel_values"]).add_(.01)
                    changed = True
                return original(*args, **kwargs)

            with self.subTest(mutation=mutation), patch.object(policy, "completion_logprobs", side_effect=corrupt), \
                    self.assertRaisesRegex(ValueError, "parameter values changed|input tensors changed"):
                g._measure_routes(b, batch, 3)
            self.assertTrue(all(p.grad is None for p in b.model.parameters()))

    def test_public_component_rejects_tiny_cpu_policy(self):
        with self.assertRaisesRegex(ValueError, "actual CUDA"):
            g.measure_gradient_routes(bundle(), inputs(), 3, protocol=config())


if __name__ == "__main__":
    unittest.main()
