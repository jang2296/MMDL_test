"""Real AdamW placement checks; CUDA tests require explicit opt-in, not 4B evidence."""
from __future__ import annotations

import copy
import os
import unittest
import weakref
from unittest.mock import patch

import torch

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl import checkpoint
from fine_tuning.notebook_rl.policy import PolicyBundle, VisionRoute


def fixture(device="cpu"):
    model = torch.nn.Module()
    model.language = torch.nn.ParameterDict({"lora_weight": torch.nn.Parameter(torch.ones(2, 2))})
    model.vision = torch.nn.ParameterDict({"lora_weight": torch.nn.Parameter(torch.ones(2, 2))})
    model.mergers = torch.nn.ModuleList()
    for _ in range(24):
        wrapper = torch.nn.Module()
        wrapper.modules_to_save = torch.nn.ModuleDict({"default": torch.nn.ParameterDict(
            {"weight": torch.nn.Parameter(torch.ones(2, 2))})})
        model.mergers.append(wrapper)
    model.register_parameter("frozen", torch.nn.Parameter(torch.ones(2, 2), requires_grad=False))
    model.to(device)
    params: dict[str, list[torch.nn.Parameter]] = {"L": [], "M": [], "V": []}
    rows = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = "M" if "modules_to_save" in name else "V" if name.startswith("vision.") else "L"
        params[group].append(parameter)
        rows.append({"name": name, "group": group, "shape": list(parameter.shape), "dtype": str(parameter.dtype)})
    bundle = PolicyBundle(model, None, {"trainable": rows}, params, VisionRoute(params["V"]), {})
    protocol = {"optimizer": {"type": "adamw", "schedule": "constant_after_warmup", "warmup_steps": 3,
        "lr": {"L": .001, "M": .002, "V": .003}, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01}}
    optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
    return bundle, optimizer, scheduler, protocol


def advance(bundle, optimizer, scheduler):
    for index, parameter in enumerate(p for ps in bundle.params.values() for p in ps):
        parameter.grad = torch.full_like(parameter, .125 + index * .001)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def signature(value):
    if isinstance(value, torch.Tensor):
        return str(value.device), str(value.dtype), tuple(value.shape), tensor_sha256(value.reshape(-1))
    if isinstance(value, dict):
        return {key: signature(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [signature(item) for item in value]
    return value


class OptimizerParkingTests(unittest.TestCase):
    def test_empty_state_is_noop_and_preserves_objects_scheduler_and_rng(self):
        bundle, optimizer, scheduler, protocol = fixture()
        identities = [id(p) for p in bundle.model.parameters()]
        before = signature((optimizer.state_dict(), scheduler.state_dict(), checkpoint.capture_rng()))
        with patch.object(torch.Tensor, "to", side_effect=AssertionError("CPU state must not move")):
            with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 0):
                self.assertFalse(optimizer.state)
        self.assertEqual(identities, [id(p) for p in bundle.model.parameters()])
        self.assertEqual(before, signature((optimizer.state_dict(), scheduler.state_dict(), checkpoint.capture_rng())))

    def test_existing_cpu_state_and_next_real_adamw_step_are_identical(self):
        bundle, optimizer, scheduler, protocol = fixture()
        baseline, baseline_optimizer, baseline_scheduler, _ = fixture()
        advance(bundle, optimizer, scheduler)
        advance(baseline, baseline_optimizer, baseline_scheduler)
        identities = [id(tensor) for state in optimizer.state.values() for tensor in state.values()]
        before = signature((optimizer.state_dict(), scheduler.state_dict(), checkpoint.capture_rng()))
        with patch.object(torch.Tensor, "to", side_effect=AssertionError("CPU state must not move")):
            with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1):
                self.assertEqual(before, signature((optimizer.state_dict(), scheduler.state_dict(), checkpoint.capture_rng())))
        self.assertEqual(identities, [id(tensor) for state in optimizer.state.values() for tensor in state.values()])
        advance(bundle, optimizer, scheduler)
        advance(baseline, baseline_optimizer, baseline_scheduler)
        self.assertEqual(signature(bundle.model.state_dict()), signature(baseline.model.state_dict()))
        self.assertEqual(signature(optimizer.state_dict()), signature(baseline_optimizer.state_dict()))
        self.assertEqual(scheduler.state_dict(), baseline_scheduler.state_dict())

    def test_body_exception_propagates_with_cpu_state_unchanged(self):
        bundle, optimizer, scheduler, protocol = fixture()
        advance(bundle, optimizer, scheduler)
        before = signature(optimizer.state_dict())
        error = RuntimeError("collection failed")
        with self.assertRaises(RuntimeError) as caught:
            with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1):
                raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(before, signature(optimizer.state_dict()))

    def test_aliases_are_rejected_before_any_transfer_or_body(self):
        for alias in ("same", "view", "other", "parameter", "frozen"):
            with self.subTest(alias=alias):
                bundle, optimizer, scheduler, protocol = fixture()
                advance(bundle, optimizer, scheduler)
                parameter, other = list(optimizer.state)[:2]
                state = optimizer.state[parameter]
                state["exp_avg_sq"] = {"same": state["exp_avg"], "view": state["exp_avg"].view(2, 2),
                    "other": optimizer.state[other]["exp_avg"], "parameter": parameter.detach(),
                    "frozen": bundle.model.frozen.detach()}[alias]
                before = signature(optimizer.state_dict())
                with patch.object(torch.Tensor, "to", side_effect=AssertionError("preflight moved state")):
                    with self.assertRaisesRegex(ValueError, "aliased"):
                        with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1):
                            self.fail("aliased state entered collection")
                self.assertEqual(before, signature(optimizer.state_dict()))

    def test_nested_or_unsupported_state_and_changed_hyperparameters_are_rejected(self):
        for defect in ("nested", "extra", "capturable", "fused", "scheduler", "step_shape"):
            with self.subTest(defect=defect):
                bundle, optimizer, scheduler, protocol = fixture()
                advance(bundle, optimizer, scheduler)
                state = next(iter(optimizer.state.values()))
                if defect == "nested":
                    state["exp_avg"] = {"nested": state["exp_avg"]}
                elif defect == "extra":
                    state["max_exp_avg_sq"] = state["exp_avg_sq"].clone()
                elif defect in ("capturable", "fused"):
                    optimizer.param_groups[0][defect] = True
                elif defect == "scheduler":
                    scheduler.last_epoch += 1
                else:
                    state["step"] = state["step"].reshape(1)
                before = signature(optimizer.state_dict())
                with patch.object(torch.Tensor, "to", side_effect=AssertionError("preflight moved state")):
                    with self.assertRaises(ValueError):
                        with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1):
                            self.fail("unsupported state entered collection")
                self.assertEqual(before, signature(optimizer.state_dict()))

    def test_existing_parameter_alias_is_allowed_and_exit_revalidates(self):
        bundle, optimizer, scheduler, protocol = fixture()
        bundle.model.register_parameter("frozen_tied", torch.nn.Parameter(bundle.model.frozen.detach(), requires_grad=False))
        with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 0):
            pass
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 0):
                optimizer.param_groups[0]["eps"] *= 2

    def test_restore_is_idempotent_and_allows_one_committed_step(self):
        bundle, optimizer, scheduler, protocol = fixture()
        baseline, baseline_optimizer, baseline_scheduler, _ = fixture()
        advance(bundle, optimizer, scheduler)
        advance(baseline, baseline_optimizer, baseline_scheduler)
        with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1) as restore:
            restore()
            restore()
            advance(bundle, optimizer, scheduler)
        advance(baseline, baseline_optimizer, baseline_scheduler)
        self.assertEqual(signature(bundle.model.state_dict()), signature(baseline.model.state_dict()))
        self.assertEqual(signature(optimizer.state_dict()), signature(baseline_optimizer.state_dict()))
        self.assertEqual(scheduler.state_dict(), baseline_scheduler.state_dict())

    def test_step_without_early_restore_is_rejected(self):
        bundle, optimizer, scheduler, protocol = fixture()
        advance(bundle, optimizer, scheduler)
        with self.assertRaisesRegex(ValueError, "scheduler state/step/LR"):
            with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1):
                advance(bundle, optimizer, scheduler)

    def test_restore_with_nothing_parked_moves_nothing(self):
        # CPU-only: no CUDA moment can be parked here, so only the empty-parked path is exercised.
        bundle, optimizer, scheduler, protocol = fixture()
        advance(bundle, optimizer, scheduler)
        before = signature(optimizer.state_dict())
        with patch.object(torch.Tensor, "to") as to:
            with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1) as restore:
                restore()
                restore()
            to.assert_not_called()
        self.assertEqual(before, signature(optimizer.state_dict()))


@unittest.skipUnless(os.environ.get("MMDL_TEST_OPTIMIZER_PARKING_CUDA") == "1",
                     "CUDA parking requires explicit MMDL_TEST_OPTIMIZER_PARKING_CUDA=1")
class CudaOptimizerParkingTests(unittest.TestCase):
    def test_roundtrip_releases_original_moments_preserves_rng_and_next_step(self):
        bundle, optimizer, scheduler, protocol = fixture("cuda")
        advance(bundle, optimizer, scheduler)
        baseline = copy.deepcopy(bundle)
        baseline_optimizer, baseline_scheduler = checkpoint.make_optimizer(baseline, protocol)
        baseline_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        baseline_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
        before = signature((optimizer.state_dict(), scheduler.state_dict(), checkpoint.capture_rng()))
        parked_expected = copy.deepcopy(before[0])
        for state in parked_expected["state"].values():
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = ("cpu", *state[key][1:])
        steps = [state["step"] for state in optimizer.state.values()]
        references = [weakref.ref(state[key]) for state in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq")]
        with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1):
            self.assertTrue(all(state[key].device.type == "cpu" for state in optimizer.state.values() for key in state))
            self.assertTrue(all(reference() is None for reference in references))
            self.assertEqual(signature(optimizer.state_dict()), parked_expected)
        self.assertEqual(before, signature((optimizer.state_dict(), scheduler.state_dict(), checkpoint.capture_rng())))
        self.assertTrue(all(original is state["step"] for original, state in zip(steps, optimizer.state.values(), strict=True)))
        advance(bundle, optimizer, scheduler)
        advance(baseline, baseline_optimizer, baseline_scheduler)
        self.assertEqual(signature(bundle.model.state_dict()), signature(baseline.model.state_dict()))
        self.assertEqual(signature(optimizer.state_dict()), signature(baseline_optimizer.state_dict()))

    def test_partial_transfer_exception_restores_already_parked_moments(self):
        bundle, optimizer, scheduler, protocol = fixture("cuda")
        advance(bundle, optimizer, scheduler)
        before, original_to = signature(optimizer.state_dict()), torch.Tensor.to
        calls = 0
        def transfer(tensor, *args, **kwargs):
            nonlocal calls
            if args and args[0] == "cpu":
                calls += 1
                if calls == 2:
                    raise RuntimeError("transfer failed")
            return original_to(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, "to", transfer), self.assertRaisesRegex(RuntimeError, "transfer failed"):
            with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1):
                self.fail("partial transfer entered collection")
        self.assertEqual(before, signature(optimizer.state_dict()))

    def test_body_exception_restores_cuda_state(self):
        bundle, optimizer, scheduler, protocol = fixture("cuda")
        advance(bundle, optimizer, scheduler)
        before = signature(optimizer.state_dict())
        with self.assertRaisesRegex(RuntimeError, "collection failed"):
            with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1):
                raise RuntimeError("collection failed")
        self.assertEqual(before, signature(optimizer.state_dict()))

    def test_restore_failure_propagates_with_original_body_exception_chained(self):
        bundle, optimizer, scheduler, protocol = fixture("cuda")
        advance(bundle, optimizer, scheduler)
        original_to = torch.Tensor.to
        def transfer(tensor, *args, **kwargs):
            if args and torch.device(args[0]).type == "cuda":
                raise RuntimeError("restore failed")
            return original_to(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, "to", transfer), self.assertRaisesRegex(RuntimeError, "restore failed") as caught:
            with checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, 1):
                raise RuntimeError("collection failed")
        self.assertEqual(str(caught.exception.__context__), "collection failed")


if __name__ == "__main__":
    unittest.main()
