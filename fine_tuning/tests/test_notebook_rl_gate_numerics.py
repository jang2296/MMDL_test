"""Bounded G10 producer checks; these CPU fixtures do not certify a GPU run."""
import json
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import gate_numerics as g


class GateNumericsTests(unittest.TestCase):
    def test_actual_cpu_loss_autograd_evidence_and_source_scope(self):
        before = torch.get_rng_state()
        report = g.measure_numerics("cpu")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(report["numeric_tests_match"])
        self.assertTrue(report["trl_formula_match"])
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["input_dtypes"], ["float32", "bfloat16"])
        self.assertEqual(report["work_dtype"], "float32")
        self.assertEqual(report["trl_source"]["commit"], "aca45154c8b6b1d9b0c06b674c0fa6bcfd9df051")
        self.assertEqual(report["trl_source"]["file_sha256"],
                         "4ac442a6d117838bf26fa3b786feb092c3953b2f9212081646db7fad9b356b41")
        self.assertFalse(report["trl_scope"]["use_bias_correction_kl"])
        self.assertGreater(len(report["limitations"]), 0)
        names = {c["name"] for c in report["cases"]}
        for dtype in ("float32", "bfloat16"):
            for name in ("no_std", "equal_rewards_kl_gradient", "scalar_clipped_gradient",
                         "microbatch_gradient", "group_weights_gradient", "weighted_microbatch_gradient",
                         "ratio_one_gradient",
                         "k3_gradient", "old_ref_adv_detached", "mis_0_1_reject_0_01_keep",
                         "excluded_exact_zero", "all_excluded_gradient", "fp32_work"):
                self.assertIn(f"{dtype}/{name}", names)
            self.assertIn(f"trl/{dtype}/gradient", names)
        self.assertEqual(len(names), len(report["cases"]))
        self.assertTrue(all(c["match"] and c["device"] == "cpu" for c in report["cases"]))
        json.dumps(report, allow_nan=False)

    def test_cuda_unavailable_raises_before_allocation(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch, "empty") as allocate, \
                self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            g.measure_numerics("cuda:0")
        allocate.assert_not_called()

    def test_unsupported_device_does_not_fall_back(self):
        with self.assertRaisesRegex(ValueError, "CPU or CUDA"):
            g.measure_numerics("meta")

    def test_cuda_without_bf16_is_rejected_before_allocation(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "device"), \
                patch.object(torch.cuda, "is_bf16_supported", return_value=False), \
                patch.object(torch, "empty") as allocate, \
                self.assertRaisesRegex(RuntimeError, "lacks BF16"):
            g.measure_numerics("cuda:0")
        allocate.assert_not_called()

    def test_wrong_objective_value_is_not_a_pass(self):
        original = g.losses.group_loss
        with patch.object(g.losses, "group_loss", side_effect=lambda *a, **kw: original(*a, **kw)+.01), \
                self.assertRaisesRegex(RuntimeError, "scalar_clipped_loss.*mismatch"):
            g.measure_numerics("cpu")

    def test_wrong_gradient_with_correct_value_is_not_a_pass(self):
        original = g.losses.group_loss

        def broken(cur, *args, **kwargs):
            result = original(cur, *args, **kwargs)
            return result + (cur-cur.detach()).sum()

        with patch.object(g.losses, "group_loss", side_effect=broken), \
                self.assertRaisesRegex(RuntimeError, "scalar_clipped_gradient.*mismatch"):
            g.measure_numerics("cpu")

    def test_nonfinite_output_is_not_a_pass(self):
        with patch.object(g.losses, "group_loss", return_value=torch.tensor(float("nan"))), \
                self.assertRaisesRegex(RuntimeError, "nonfinite"):
            g.measure_numerics("cpu")

    def test_independent_trl_mismatch_is_not_a_pass(self):
        original = g._trl_dr_grpo
        with patch.object(g, "_trl_dr_grpo", side_effect=lambda *a, **kw: original(*a, **kw)+.01), \
                self.assertRaisesRegex(RuntimeError, "trl/float32/loss.*mismatch"):
            g.measure_numerics("cpu")

    def test_std_normalization_and_missing_mis_are_detected(self):
        original = g.losses.group_advantages
        with patch.object(g.losses, "group_advantages", side_effect=lambda r: original(r)/r.std()), \
                self.assertRaisesRegex(RuntimeError, "no_std.*mismatch"):
            g.measure_numerics("cpu")
        original_mis = g.losses.mis_mask

        def missing_mis(*args, **kwargs):
            keep, delta = original_mis(*args, **kwargs)
            return torch.ones_like(keep), delta

        with patch.object(g.losses, "mis_mask", side_effect=missing_mis), \
                self.assertRaisesRegex(RuntimeError, "mis_0_1_reject_0_01_keep.*mismatch"):
            g.measure_numerics("cpu")


if __name__ == "__main__":
    unittest.main()
