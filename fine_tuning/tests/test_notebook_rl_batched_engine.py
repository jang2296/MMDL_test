"""Opt-in row-seeded batched HF sampling vs the serial sampler on the actual tiny CPU Qwen3-VL.

CPU float32 only: GPU/BF16/NF4 drift is not measured here and token equality is not claimed there.
"""
import unittest
from dataclasses import asdict
from unittest.mock import patch

import torch
from PIL import Image
from transformers import LogitNormalization

from fine_tuning.notebook_rl.engines import (ROW_SEEDED_BATCH_SAMPLER, BatchedRolloutResult, HFRolloutEngine,
    RolloutRequest, RolloutResult, SamplingSettings, _RowSeededSampler)
from fine_tuning.notebook_rl.policy import configure_policy
from fine_tuning.notebook_rl.shared_policy_engine import SharedPolicyRolloutEngine
from fine_tuning.tests.test_notebook_rl_policy import config
from fine_tuning.tests import test_notebook_rl_shared_policy_engine as shared
from fine_tuning.tests.test_training import tiny_model

COLORS = ((30, 90, 120), (200, 10, 60))
CAP = 12


def make(images=2, n=4, seeds=None, key="r", colors=COLORS):
    return RolloutRequest(key, None, [{"role": "user", "content": [{"type": "image"} for _ in range(images)]
                                       + [{"type": "text", "text": "question"}]}],
                          [Image.new("RGB", (8, 8), colors[i]) for i in range(images)], n, CAP,
                          list(seeds) if seeds is not None else list(range(31, 31 + n)))


def max_drift(a, b):
    return max((abs(x - y) for p, q in zip(a, b, strict=True)
                for x, y in zip(p.logprobs, q.logprobs, strict=True)), default=0.)


class BatchedEngineTests(unittest.TestCase):
    threads: int

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def engines(self, *batch_rows):
        model, proc = tiny_model(), shared.processor()
        return [HFRolloutEngine(model, proc, {"preprocessing": "hf_processor"}, 64, batch_rows=rows)
                for rows in (None, *batch_rows)]

    def test_rows_draw_distinct_images_in_row_order(self):
        serial, = self.engines()
        swapped = make(colors=COLORS[::-1], n=1)
        self.assertNotEqual(serial.probe_logprobs(make(n=1), [7, 8, 9]), serial.probe_logprobs(swapped, [7, 8, 9]))

    def test_single_row_batched_is_bitwise_serial_with_and_without_eos(self):
        serial, one, wide = self.engines(1, 4)
        for eos in (None, 2, list(range(32))):
            serial.model.generation_config.eos_token_id = eos
            for images in (0, 1, 2):
                for engine, item in ((one, make(images)), (wide, make(images, n=1))):
                    with self.subTest(eos=eos, images=images, batch_rows=engine.batch_rows):
                        expected = serial.sample([item])[0]
                        actual = engine.sample([item])[0]
                        self.assertIs(type(expected), RolloutResult)
                        self.assertEqual(actual.completions, expected.completions)  # exact float equality
                        self.assertEqual(actual.prompt_token_ids, expected.prompt_token_ids)

    def test_n4_batched_tokens_finish_order_rng_and_label(self):
        serial, batched = self.engines(4)
        reasons = set()
        for training in (False, True):
            for eos in (None, 2):
                for images in (0, 1, 2):
                    with self.subTest(training=training, eos=eos, images=images):
                        serial.model.train(training)
                        serial.model.generation_config.eos_token_id = eos
                        item = make(images)
                        expected = serial.sample([item])[0]
                        observed = []
                        state = torch.get_rng_state().clone()
                        actual = batched.sample_with_settings([item], SamplingSettings(),
                            on_completion=lambda *row: observed.append(row))
                        self.assertTrue(torch.equal(state, torch.get_rng_state()))
                        self.assertEqual(batched.model.training, training)
                        result = actual[0]
                        self.assertIs(type(result), BatchedRolloutResult)
                        self.assertEqual((result.batch_rows, result.sampler_version), (4, ROW_SEEDED_BATCH_SAMPLER))
                        self.assertEqual(asdict(result)["sampler_version"], "hf-row-seeded-batch-v1")
                        self.assertNotEqual(result, expected)
                        self.assertEqual(set(asdict(expected)), {"request_id", "prompt_token_ids", "completions"})
                        self.assertEqual(result.prompt_token_ids, expected.prompt_token_ids)
                        self.assertEqual([c.token_ids for c in result.completions],
                                         [c.token_ids for c in expected.completions])
                        self.assertEqual([(c.finish_reason, c.text) for c in result.completions],
                                         [(c.finish_reason, c.text) for c in expected.completions])
                        self.assertLessEqual(max_drift(result.completions, expected.completions), 1e-5)
                        self.assertEqual(observed, [(item.request_id, index, seed, completion) for index, (seed, completion)
                                                    in enumerate(zip(item.seeds, result.completions, strict=True))])
                        self.assertTrue(all(row[3] is not c for row, c in zip(observed, result.completions)))
                        reasons.update(c.finish_reason for c in result.completions)
        self.assertEqual(reasons, {"stop", "length"})

    def test_rows_truncate_independently_at_first_eos_or_cap(self):
        serial, batched = self.engines(4)
        serial.model.generation_config.eos_token_id = None
        free = [c.token_ids for c in serial.sample([make()])[0].completions]
        pad = serial.model.generation_config.pad_token_id
        # A sampled pad id that is not EOS is an ordinary token: rows must not be cut there.
        self.assertTrue(any(pad in row[:-1] for row in free))
        # A non-pad token emitted early by exactly one row becomes EOS: that row stops, the others hit the cap.
        eos = next(t for t in range(32) if t != pad and sum(t in row[:CAP - 2] for row in free) == 1
                   and sum(t in row for row in free) == 1)
        for eos_ids in ([eos], [eos, pad]):  # also with the pad id registered as EOS, as in Qwen.
            serial.model.generation_config.eos_token_id = eos_ids
            expected = serial.sample([make()])[0].completions
            actual = batched.sample([make()])[0].completions
            with self.subTest(eos=eos_ids):
                self.assertEqual([c.token_ids for c in actual], [c.token_ids for c in expected])
                self.assertEqual([c.finish_reason for c in actual], [c.finish_reason for c in expected])
                for row, completion in zip(free, actual, strict=True):
                    stops = [i for i, t in enumerate(row) if t in eos_ids]
                    if stops:
                        self.assertEqual(completion.token_ids, row[:stops[0] + 1])
                        self.assertEqual(completion.finish_reason, "stop")
                    else:
                        self.assertEqual(completion.token_ids, row)
                        self.assertEqual(completion.finish_reason, "length")
                if eos_ids == [eos]:
                    self.assertEqual(sorted(len(c.token_ids) < CAP for c in actual), [False, False, False, True])

    def test_seeds_stay_per_row_when_swapped(self):
        _, batched = self.engines(4)
        batched.model.generation_config.eos_token_id = 2
        first = batched.sample([make(seeds=[31, 32, 33, 34])])[0].completions
        second = batched.sample([make(seeds=[32, 31, 33, 34])])[0].completions
        self.assertNotEqual(first[0].token_ids, first[1].token_ids)
        for a, b in ((first[0], second[1]), (first[1], second[0]), (first[2], second[2]), (first[3], second[3])):
            self.assertEqual((a.token_ids, a.finish_reason), (b.token_ids, b.finish_reason))
            torch.testing.assert_close(torch.tensor(a.logprobs), torch.tensor(b.logprobs), atol=1e-6, rtol=0)

    def test_chunks_are_greedy_over_forced_draws_and_persist_before_the_next_chunk(self):
        serial, batched = self.engines(3)
        model = batched.model
        model.generation_config.temperature, model.generation_config.top_k = .7, 20  # model defaults must not leak
        generate, chain, configs = model.generate, model._get_logits_processor, model._prepare_generation_config
        rows, processors, effective, observed = [], [], [], []

        def chunk(*args, **kwargs):
            rows.append((kwargs["input_ids"].shape[0], len(observed)))
            return generate(*args, **kwargs)

        def chained(*args, **kwargs):
            result = chain(*args, **kwargs)
            processors.append(tuple(type(p) for p in result))
            return result

        def prepared(*args, **kwargs):
            cfg, model_kwargs = configs(*args, **kwargs)
            effective.append((cfg.do_sample, cfg.repetition_penalty, cfg.renormalize_logits))
            return cfg, model_kwargs

        item = make()
        with patch.object(model, "generate", side_effect=chunk), patch.object(
                model, "_get_logits_processor", side_effect=chained), patch.object(
                model, "_prepare_generation_config", side_effect=prepared):
            result = batched.sample([item], on_completion=lambda *row: observed.append(row))[0]
        self.assertEqual(rows, [(3, 0), (1, 3)])
        self.assertEqual(processors, [(LogitNormalization, _RowSeededSampler)] * 2)
        self.assertEqual(effective, [(False, 1.0, False)] * 2)
        self.assertEqual([row[1] for row in observed], [0, 1, 2, 3])
        self.assertEqual(result.batch_rows, 3)
        expected = serial.sample([item])[0].completions
        self.assertEqual([c.token_ids for c in result.completions], [c.token_ids for c in expected])
        self.assertLessEqual(max_drift(result.completions, expected), 1e-5)

    def test_batch_rows_validation_and_shared_engine_passthrough(self):
        for bad in (0, -1, True, 1.5, "2"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "batch_rows"):
                HFRolloutEngine(tiny_model(), shared.processor(), {"preprocessing": "hf_processor"}, 64, batch_rows=bad)
        bundle = configure_policy(tiny_model(), shared.processor(), "MOD-RV", config(), enforce_counts=False)
        with self.assertRaisesRegex(ValueError, "batch_rows"):
            SharedPolicyRolloutEngine(bundle, {"preprocessing": "hf_processor"}, 64, logprob_chunk=1, batch_rows=0)
        fixture = shared.SharedPolicyEngineTests()
        outputs = []
        for rows in (None, 2):
            engine = SharedPolicyRolloutEngine(bundle, {"preprocessing": "hf_processor"}, 64,
                                               logprob_chunk=1, batch_rows=rows)
            fixture.certify(bundle, engine)
            bundle.route.apply(False)
            outputs.append(engine.sample([make(n=3)])[0])
            self.assertFalse(bundle.route.vision_on)
        serial, batched = outputs
        self.assertIs(type(serial), RolloutResult)
        self.assertEqual((type(batched), batched.batch_rows), (BatchedRolloutResult, 2))
        self.assertEqual([c.token_ids for c in batched.completions], [c.token_ids for c in serial.completions])
        self.assertLessEqual(max_drift(batched.completions, serial.completions), 1e-5)


if __name__ == "__main__":
    unittest.main()
