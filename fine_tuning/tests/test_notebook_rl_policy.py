"""Actual tiny Qwen/PEFT CPU policy gradients, not pretrained-model performance."""
import unittest
from pathlib import Path

import torch

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl.policy import (assert_base_unchanged, completion_logprobs,
    configure_policy, merged_named_tensors, reference_logprobs)
from fine_tuning.notebook_rl.schema import load_protocol
from fine_tuning.tests.test_training import tiny_model


def inputs():
    return {"input_ids": torch.tensor([[4, 3, 3, 3, 3, 5, 9, 10, 11, 2]]),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
            "pixel_values": torch.linspace(-1, 1, 16 * 24).reshape(16, 24),
            "image_grid_thw": torch.tensor([[1, 4, 4]])}


def config():
    return load_protocol(Path(__file__).parents[1] / "notebook_rl/protocol_v1.yaml")


def bundle(arm="MOD-RV"):
    return configure_policy(tiny_model(), None, arm, config(), enforce_counts=False)


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_initial_reference_and_chunked_logprobs_match_real_fresh_qwen(self):
        base, batch = tiny_model().eval(), inputs()
        with torch.no_grad():
            logits = base(**batch).logits[:, -4:-1].float()
            expected = logits.log_softmax(-1).gather(-1, batch["input_ids"][:, -3:].unsqueeze(-1)).squeeze(-1)
        b = configure_policy(base, None, "MOD-RV", config(), enforce_counts=False)
        self.assertTrue(torch.equal(reference_logprobs(b, batch, 3), expected))
        actual = completion_logprobs(b, batch, 3, chunk=1)
        full = completion_logprobs(b, batch, 3, chunk=3)
        self.assertTrue(torch.equal(full, expected))
        # CPU projection kernels at chunk1 can differ by one FP32 ULP from GEMM.
        torch.testing.assert_close(actual, expected, rtol=0, atol=5e-7)
        torch.testing.assert_close(actual, full, rtol=0, atol=5e-7)
        self.assertEqual(actual.shape, (1, 3))

    def test_route_blocks_vision_but_not_merger_or_llm_gradients(self):
        b, batch = bundle(), inputs()
        for vision_on in (False, True):
            b.model.zero_grad(set_to_none=True)
            b.route.apply(vision_on)
            # Signed generated-action objective; no labels or reference answer passed.
            lp = completion_logprobs(b, batch, 3, chunk=1)
            (lp[0, 0] - lp[0, 1]).backward()
            for group in ("L", "M"):
                self.assertTrue(all(p.grad is not None for p in b.params[group]))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in b.params[group]), 0)
            if vision_on:
                self.assertTrue(all(p.grad is not None for p in b.params["V"]))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in b.params["V"]), 0)
            else:
                self.assertTrue(all(p.grad is None for p in b.params["V"]))
        self.assertEqual(len([r for r in b.inventory["trainable"] if r["group"] == "M"]), 24)

    def test_reference_stays_fresh_after_updates_and_route_is_restored(self):
        b, batch = bundle(), inputs()
        initial = reference_logprobs(b, batch, 3)
        opt = torch.optim.AdamW([p for ps in b.params.values() for p in ps], lr=.01, weight_decay=0)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            lp = completion_logprobs(b, batch, 3)
            (lp[0, 0] - lp[0, 1]).backward()
            opt.step()
        b.route.apply(False)
        self.assertTrue(torch.equal(initial, reference_logprobs(b, batch, 3)))
        self.assertFalse(any(p.requires_grad for p in b.params["V"]))
        self.assertFalse(torch.equal(initial, completion_logprobs(b, batch, 3)))
        assert_base_unchanged(b)

    def test_no_momentum_update_on_all_vision_off_step(self):
        b, batch = bundle(), inputs()
        opt = torch.optim.AdamW([p for ps in b.params.values() for p in ps], lr=.01, weight_decay=.1)
        for vision_on in (True, False):
            opt.zero_grad(set_to_none=True)
            b.route.apply(vision_on)
            before = [p.detach().clone() for p in b.params["V"]]
            completion_logprobs(b, batch, 3).sum().backward()
            opt.step()
            if not vision_on:
                self.assertTrue(all(torch.equal(old, p) for old, p in zip(before, b.params["V"], strict=True)))

    def test_out_of_place_export_has_complete_keys_and_preserves_all_parameters(self):
        b = bundle()
        before = {n: tensor_sha256(p) for n, p in b.model.named_parameters()}
        exported = dict(merged_named_tensors(b))
        self.assertEqual(set(exported), set(b.base_sha256))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in exported.values()))
        self.assertEqual(before, {n: tensor_sha256(p) for n, p in b.model.named_parameters()})
        next(iter(exported.values())).zero_()
        self.assertEqual(before, {n: tensor_sha256(p) for n, p in b.model.named_parameters()})
        assert_base_unchanged(b)

    def test_fixed_arm_and_bad_input_guards(self):
        b = bundle("MOD-FML")
        self.assertEqual(b.params["V"], [])
        with self.assertRaises(ValueError):
            b.route.apply(True)
        for bad in (inputs() | {"labels": torch.ones(1, 10)}, inputs() | {"past_key_values": ()},
                    inputs() | {"input_ids": inputs()["input_ids"].float()}):
            with self.assertRaises(ValueError):
                completion_logprobs(b, bad, 3)
        with self.assertRaises(ValueError):
            configure_policy(tiny_model(), None, "MOD-RV", config())
        with torch.no_grad():
            b.model.get_base_model().model.visual.patch_embed.proj.weight.add_(1)
        with self.assertRaises(ValueError):
            assert_base_unchanged(b)


if __name__ == "__main__":
    unittest.main()
