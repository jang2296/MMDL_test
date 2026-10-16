"""Actual tiny-Qwen CPU reward updates, not pretrained 4B/GPU G4 approval."""
import unittest
from unittest.mock import patch

import torch

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.gate_update import _measure_update, measure_update
from fine_tuning.tests.test_notebook_rl_policy import bundle, config
from fine_tuning.tests.test_notebook_rl_updates import Preparation, tree


class MasterUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, arm="MOD-RV"):
        b, protocol, prep = bundle(arm), config(), Preparation()
        opt, scheduler = make_optimizer(b, protocol)
        row = tree(b, route="VISION_OFF")
        return b, opt, scheduler, row, prep, protocol

    def measure(self, fixture, *, public=False):
        b, opt, scheduler, row, prep, protocol = fixture
        return (measure_update if public else _measure_update)(
            b, opt, scheduler, [row], {"p": [object()]}, prep, protocol, "ALG-T1", step=0, seed=3407)

    def test_real_step_complete_master_hashes_and_frozen_probes(self):
        fixture = self.fixture()
        b, opt, scheduler, _, prep, _ = fixture
        with patch("torch.nn.functional.cross_entropy", side_effect=AssertionError("CE forbidden")), \
                patch("fine_tuning.train.AssistantCollator", side_effect=AssertionError("SFT forbidden")):
            report = self.measure(fixture)
        self.assertTrue(report["merger_changed"])
        self.assertTrue(report["frozen_unchanged"])
        self.assertEqual(report["master_tensor_count"], 24)
        self.assertEqual(report["merger_count"], 4)
        self.assertEqual(len(report["master_before"]), 24)
        self.assertEqual(len(report["master_after"]), 24)
        self.assertEqual({v["dtype"] for v in report["master_after"].values()}, {"torch.float32"})
        self.assertEqual(report["base_before"], report["base_after"])
        self.assertTrue(report["frozen_probes"])
        self.assertTrue(all(v["before_sha256"] == v["after_sha256"] for v in report["frozen_probes"].values()))
        named = dict(b.model.named_parameters())
        self.assertTrue(all(v["sha256"] == tensor_sha256(named[n]) for n, v in report["master_after"].items()))
        self.assertTrue(report["changed_probe_tensors"])
        self.assertTrue(all(v == {"before": 0., "after": 1.} for v in report["master_optimizer_steps"].values()))
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertTrue(opt.state)
        self.assertEqual(prep.policy_version, 0)
        self.assertEqual(report["update"]["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertTrue(report["engine_sync_required"])
        self.assertFalse(report["cuda_policy_execution"])
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["gpu_gates_certified"])
        self.assertFalse(report["pinned_architecture_counts"])
        self.assertTrue(all(p.grad is None for p in b.model.parameters()))

    def test_fml_is_measured_not_inapplicable(self):
        report = self.measure(self.fixture("MOD-FML"))
        self.assertTrue(report["merger_changed"])
        self.assertIsNone(report["update"]["grad_norm"]["V"])

    def test_existing_optimizer_state_advances_once_at_declared_cursor(self):
        fixture = self.fixture()
        self.measure(fixture)
        b, opt, scheduler, _, prep, protocol = fixture
        prep.policy_version = 1  # Authored CPU fixture; not sampler sync evidence.
        row = tree(b, version=1, route="VISION_ON")
        report = _measure_update(b, opt, scheduler, [row], {"p": [object()]}, prep, protocol,
                                 "ALG-T1", step=1, seed=3407)
        self.assertTrue(report["merger_changed"])
        self.assertTrue(all(v == {"before": 1., "after": 2.} for v in report["master_optimizer_steps"].values()))
        self.assertEqual(scheduler.last_epoch, 2)

    def test_changed_optimizer_contract_rejected_before_update(self):
        fixture = self.fixture()
        fixture[1].param_groups[0]["lr"] = 0.
        with self.assertRaisesRegex(ValueError, "scheduler state/step/LR"):
            self.measure(fixture)
        self.assertFalse(fixture[1].state)

    def test_public_rejects_cpu_before_update(self):
        fixture = self.fixture()
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.measure(fixture, public=True)
        self.assertEqual(fixture[2].last_epoch, 0)
        self.assertFalse(fixture[1].state)

    def test_rejects_preexisting_gradient_without_erasing_it(self):
        fixture = self.fixture()
        p = fixture[0].params["M"][0]
        p.grad = torch.ones_like(p)
        grad = p.grad
        with self.assertRaisesRegex(ValueError, "pre-existing"):
            self.measure(fixture)
        self.assertIs(p.grad, grad)
        self.assertFalse(fixture[1].state)

    def test_no_matched_actions_cannot_pass(self):
        fixture = self.fixture()
        for sample in fixture[3].groups[0].samples:
            sample.sampler_logprobs = [p - .1 for p in sample.sampler_logprobs]
        with self.assertRaisesRegex(ValueError, "no completed actual"):
            self.measure(fixture)
        self.assertEqual(fixture[2].last_epoch, 0)
        self.assertFalse(fixture[1].state)

    def test_non_fp32_master_and_incomplete_inventory_rejected(self):
        for defect in ("dtype", "inventory"):
            fixture = self.fixture()
            if defect == "dtype":
                fixture[0].params["M"][0].data = fixture[0].params["M"][0].data.bfloat16()
            else:
                fixture[0].inventory["trainable"].pop()
            with self.assertRaises(ValueError):
                self.measure(fixture)
            self.assertFalse(fixture[1].state)

    def test_actual_step_without_master_change_cannot_pass(self):
        fixture = self.fixture()
        masters = fixture[0].params["M"]
        before = [p.detach().clone() for p in masters]
        def restore(*_args):
            with torch.no_grad():
                for p, value in zip(masters, before, strict=True):
                    p.copy_(value)
        handle = fixture[1].register_step_post_hook(restore)
        try:
            with self.assertRaisesRegex(ValueError, "did not change"):
                self.measure(fixture)
        finally:
            handle.remove()
        self.assertEqual(fixture[2].last_epoch, 1)  # No false rollback claim.

    def test_frozen_mutation_during_actual_update_rejected(self):
        fixture = self.fixture()
        p = next(p for n, p in fixture[0].model.named_parameters()
                 if not p.requires_grad and ".lora_" not in n)
        def corrupt(*_args):
            with torch.no_grad():
                p.add_(1)
        handle = fixture[1].register_step_post_hook(corrupt)
        try:
            with self.assertRaisesRegex(ValueError, "frozen base"):
                self.measure(fixture)
        finally:
            handle.remove()
        self.assertEqual(fixture[2].last_epoch, 1)


if __name__ == "__main__":
    unittest.main()
