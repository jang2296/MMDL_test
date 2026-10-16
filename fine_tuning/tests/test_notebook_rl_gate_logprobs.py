"""Real tiny HF generation/forward plus synthetic G7 arithmetic, not native proof."""
import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import gate_logprobs as gates
from fine_tuning.notebook_rl.engines import HFRolloutEngine
from fine_tuning.notebook_rl.policy import configure_policy
from fine_tuning.tests.test_notebook_rl_engines import Processor, request
from fine_tuning.tests.test_notebook_rl_policy import config
from fine_tuning.tests.test_training import tiny_model


def fixtures():
    engine = HFRolloutEngine(tiny_model(), Processor(), {"preprocessing": "hf_processor"}, 64)
    bundle = configure_policy(tiny_model(), None, "MOD-RV", config(), enforce_counts=False)
    # Authored tiny seeds avoid generating a second image placeholder. These are
    # not a selected real-data cohort, pretrained outputs or an external gate.
    requests = [replace(request(count, key=f"image-{count}", n=16), max_new_tokens=1) for count in (1, 2)]
    return bundle, engine, requests


class LogprobGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def measure(self, bundle, engine, requests, journal):
        return gates._measure_logprobs(bundle, engine, requests, threshold=.02, delta_seq=.05,
                                      record=journal.append)

    def test_actual_32_generations_and_policy_comparisons_preserve_state(self):
        bundle, engine, requests = fixtures()
        bundle.route.apply(False)
        before = gates._hashes(bundle.model)
        state = torch.get_rng_state().clone()
        body = bundle.model.get_base_model().model
        body.rope_deltas = torch.tensor([[7]])
        rope = body.rope_deltas
        journal = []
        report = self.measure(bundle, engine, requests, journal)
        self.assertEqual(report["trajectories"], 32)
        self.assertEqual(report["tokens"], 32)
        self.assertLess(report["max_abs"], 1e-5)
        self.assertEqual(len(report["rows"]), 32)
        self.assertEqual([row["kind"] for row in journal].count("G7_ENGINE_RETURN"), 2)
        self.assertEqual(len(journal), 34)
        self.assertTrue(report["synthetic_0_1_masked"])
        self.assertFalse(report["native_execution"])
        self.assertFalse(report["thresholds_modified"])
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["gpu_gates_certified"])
        self.assertNotIn("step0_mean_abs", report)  # CPU seam is not native INITIAL.
        self.assertEqual(before, gates._hashes(bundle.model))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertIs(body.rope_deltas, rope)
        self.assertEqual(rope.tolist(), [[7]])
        self.assertFalse(bundle.route.vision_on)
        self.assertTrue(all(p.grad is None for p in bundle.model.parameters()))
        self.assertTrue(bundle.model.training)

    def test_arithmetic_uses_token_weighted_mean_and_signed_sequence_delta(self):
        rows = [{"request_id": str(i), "seed": i, "completion_token_ids": [8, 9],
                 "hf_logprobs": [-2., -2.], "sampler_logprobs": [-2.01, -1.99]} for i in range(32)]
        rows[-1] = {"request_id": "31", "seed": 31, "completion_token_ids": [8],
                    "hf_logprobs": [-2.], "sampler_logprobs": [-2.02]}
        # Binary decimal roundoff: reserve a slightly smaller actual delta.
        rows[-1]["sampler_logprobs"] = [-2.019]
        result = gates._summarize(rows, .02, .005)
        self.assertEqual(result["tokens"], 63)
        self.assertAlmostEqual(result["mean_abs"], (.01 * 62 + .019) / 63)
        self.assertAlmostEqual(result["masked_fraction"], 1 / 32)
        self.assertAlmostEqual(result["per_trajectory"][0]["sequence_sum_delta"], 0)
        self.assertAlmostEqual(result["per_trajectory"][-1]["sequence_sum_delta"], .019)
        self.assertGreater(result["delta_seq_candidate"], 0)
        self.assertLessEqual(result["delta_seq_candidate"], .05)

    def test_summary_rejects_sparse_nonfinite_positive_misaligned_and_excess_delta(self):
        row = {"request_id": "r", "seed": 31, "completion_token_ids": [9],
               "hf_logprobs": [-2.], "sampler_logprobs": [-2.]}
        for defect in ("count", "nan", "positive", "alignment", "mismatch"):
            rows = [copy.deepcopy(row) for _ in range(32)]
            if defect == "count":
                rows.pop()
            elif defect == "nan":
                rows[0]["hf_logprobs"] = [float("nan")]
            elif defect == "positive":
                rows[0]["hf_logprobs"] = [2.]
            elif defect == "alignment":
                rows[0]["sampler_logprobs"] = []
            else:
                rows[0]["sampler_logprobs"] = [-2.1]
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                gates._summarize(rows, .02, .05)

    def test_preconditions_do_not_generate_and_do_not_erase_existing_gradient(self):
        for defect in ("threshold", "count", "coverage", "duplicate", "synchronization", "gradient"):
            bundle, engine, requests = fixtures()
            p = bundle.params["M"][0]
            if defect == "count":
                requests = requests[:1]
            elif defect == "coverage":
                requests[1] = replace(requests[0], request_id="different")
            elif defect == "duplicate":
                requests[1] = replace(requests[1], request_id=requests[0].request_id)
            elif defect == "synchronization":
                engine._synchronized = False
            elif defect == "gradient":
                p.grad = torch.ones_like(p)
            with self.subTest(defect=defect), patch.object(engine, "sample_with_settings") as sample, self.assertRaises(ValueError):
                gates._measure_logprobs(bundle, engine, requests, threshold=.03 if defect == "threshold" else .02,
                                       delta_seq=.05, record=lambda row: None)
            sample.assert_not_called()
            if defect == "gradient":
                self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))

    def test_sampler_prompt_tamper_keeps_raw_return_and_fails(self):
        bundle, engine, requests = fixtures()
        sample = engine.sample_with_settings
        def tamper(*args):
            results = sample(*args)
            results[0].prompt_token_ids[-1] += 1
            return results
        journal = []
        with patch.object(engine, "sample_with_settings", side_effect=tamper), self.assertRaisesRegex(ValueError, "sampler prompt"):
            self.measure(bundle, engine, requests, journal)
        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]["kind"], "G7_ENGINE_RETURN")

    def test_native_boundary_failure_records_error_without_fabricated_payload(self):
        bundle, engine, requests = fixtures()
        journal = []
        with patch.object(engine, "sample_with_settings", side_effect=RuntimeError("invalid backend return")), \
                self.assertRaisesRegex(RuntimeError, "invalid backend"):
            self.measure(bundle, engine, requests, journal)
        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]["kind"], "G7_SAMPLER_ERROR")
        self.assertFalse(journal[0]["backend_invalid_payload_preserved"])

    def test_forward_created_input_gradient_is_rejected_and_cleared(self):
        bundle, engine, requests = fixtures()
        body = bundle.model.get_base_model().model
        observed = []
        def mutate(module, args):
            pixels = engine._pixels["pixel_values"]
            pixels.grad = torch.ones_like(pixels)
            observed.append(pixels)
        hook = body.register_forward_pre_hook(mutate)
        try:
            with self.assertRaisesRegex(ValueError, "input gradients"):
                self.measure(bundle, engine, requests, [])
        finally:
            hook.remove()
        self.assertTrue(observed)
        self.assertTrue(all(pixels.grad is None for pixels in observed))
        self.assertIsNone(engine._pixels["pixel_values"].grad)

    def test_forward_exception_restores_mutated_rope_flags_rng_and_gradients(self):
        bundle, engine, requests = fixtures()
        bundle.route.apply(False)
        body = bundle.model.get_base_model().model
        rope = torch.tensor([[7]])
        body.rope_deltas = rope
        state = torch.get_rng_state().clone()
        def fail(module, args):
            rope.add_(1)
            bundle.route.apply(True)
            bundle.params["M"][0].grad = torch.ones_like(bundle.params["M"][0])
            torch.rand(2)
            raise RuntimeError("fixture failure")
        hook = body.register_forward_pre_hook(fail)
        journal = []
        try:
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                self.measure(bundle, engine, requests, journal)
        finally:
            hook.remove()
        self.assertEqual(journal[0]["kind"], "G7_ENGINE_RETURN")
        self.assertFalse(bundle.route.vision_on)
        self.assertIs(body.rope_deltas, rope)
        self.assertEqual(rope.tolist(), [[7]])
        self.assertTrue(all(p.grad is None for p in bundle.model.parameters()))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_public_rejects_tiny_cpu_model_before_generation(self):
        bundle, engine, requests = fixtures()
        with patch.object(engine, "sample_with_settings") as sample, self.assertRaises(ValueError):
            gates.measure_logprobs(bundle, engine, requests, config(), stage="INITIAL", record=lambda row: None)
        sample.assert_not_called()


if __name__ == "__main__":
    unittest.main()
