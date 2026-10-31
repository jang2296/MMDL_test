"""Bounded score retention and exact legacy parity on actual tiny CPU Qwen3-VL."""
import unittest
import weakref
from unittest.mock import patch

import torch
from transformers import GenerationConfig, LogitNormalization
from transformers.generation.utils import GenerateDecoderOnlyOutput

from fine_tuning.notebook_rl.engines import Completion, HFRolloutEngine, _SelectedTokenLogprobs
from fine_tuning.tests.test_notebook_rl_engines import Processor, request
from fine_tuning.tests.test_training import tiny_model


class StreamingScoresTests(unittest.TestCase):
    threads: int

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def engine(self):
        return HFRolloutEngine(tiny_model(), Processor(), {"preprocessing": "hf_processor"}, 64)

    def legacy(self, engine, item):
        prepared = engine.prepare(item)
        batch = engine.model_inputs(item, [], torch.device("cpu"))
        model, completions = engine.model, []
        eos = model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            repetition_penalty=1.0, max_new_tokens=item.max_new_tokens,
            eos_token_id=eos, pad_token_id=model.generation_config.pad_token_id,
            return_dict_in_generate=True, output_scores=True, use_cache=True,
            renormalize_logits=True)
        training = model.training
        model.eval()
        try:
            for seed in item.seeds:
                with torch.random.fork_rng(), torch.no_grad():
                    torch.manual_seed(seed)
                    output = model.generate(**batch, generation_config=cfg, use_model_defaults=False)
                ids = output.sequences[0, len(prepared.token_ids):].tolist()
                lps = [float(score[0].float().log_softmax(-1)[token])
                       for token, score in zip(ids, output.scores, strict=True)]
                output["past_key_values"] = None
                del output
                completions.append(Completion(ids, lps, "stop" if ids[-1] in eos_ids else "length",
                    engine.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)))
        finally:
            model.train(training)
        return completions

    def test_actual_single_multi_image_eos_length_exact_legacy_and_rng_parity(self):
        engine = self.engine()
        original = engine.model._get_logits_processor
        observed = []

        def processors(*args, **kwargs):
            result = original(*args, **kwargs)
            observed.append(tuple(type(processor) for processor in result))
            return result

        for training in (False, True):
            for count in (0, 1, 2):
                for stop in (False, True):
                    with self.subTest(training=training, images=count, stop=stop):
                        engine.model.train(training)
                        # Every sampled first token is EOS in the short-stop case.
                        engine.model.generation_config.eos_token_id = list(range(32)) if stop else None
                        item = request(count)
                        state = torch.get_rng_state().clone()
                        expected = self.legacy(engine, item)
                        self.assertTrue(torch.equal(state, torch.get_rng_state()))
                        with patch.object(engine.model, "_get_logits_processor", side_effect=processors):
                            actual = engine.sample([item])[0]
                        self.assertEqual(actual.completions, expected)
                        self.assertEqual([len(row.token_ids) for row in actual.completions],
                                         [1 if stop else item.max_new_tokens] * item.n)
                        self.assertEqual(engine.model.training, training)
                        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(observed, [(LogitNormalization, _SelectedTokenLogprobs)] * len(observed))
        self.assertTrue(observed)

    def test_collector_retains_one_vector_and_owning_scalars_without_cpu_transfers(self):
        collector = _SelectedTokenLogprobs(2)
        ids, references, expected = torch.tensor([[9, 10]]), [], []
        for step in range(128):
            scores = torch.randn(1, 257).log_softmax(-1)
            token = step % 257
            expected.append(float(scores[0].float().log_softmax(-1)[token]))
            references.append(weakref.ref(scores))
            state = torch.get_rng_state().clone()
            with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("per-token CPU transfer")), patch.object(
                    torch.Tensor, "tolist", side_effect=AssertionError("per-token CPU transfer")), patch.object(
                    torch.Tensor, "item", side_effect=AssertionError("per-token scalar transfer")):
                self.assertIs(collector(ids, scores), scores)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            del scores
            self.assertTrue(all(reference() is None for reference in references[:-1]))
            self.assertIsNotNone(references[-1]())
            self.assertTrue(all(value.numel() == 1 and value.untyped_storage().nbytes() == value.element_size()
                                for value in collector.logprobs))
            ids = torch.cat((ids, torch.tensor([[token]])), dim=-1)
        self.assertEqual(collector.finish(ids), expected)
        self.assertTrue(all(reference() is None for reference in references))
        collector.clear()
        self.assertEqual(collector.logprobs, [])

    def test_generation_failures_clear_collector_cache_restore_training_and_rng(self):
        engine = self.engine()
        for training in (False, True):
            for failure in ("interrupted", "empty", "missing", "prefix", "count", "nonfinite", "duplicate"):
                with self.subTest(training=training, failure=failure):
                    collectors, outputs = [], []
                    engine.model.train(training)

                    def generate(**kwargs):
                        torch.rand(1)
                        ids = kwargs["input_ids"]
                        collector = kwargs["logits_processor"][-1]
                        collectors.append(collector)
                        if failure != "missing":
                            scores = torch.full((1, 32), float("nan") if failure == "nonfinite" else 0.)
                            kwargs["logits_processor"](ids, scores)
                        if failure == "interrupted":
                            raise RuntimeError("interrupted")
                        if failure == "duplicate":
                            kwargs["logits_processor"](ids, scores)
                        sequences = torch.cat((ids, torch.tensor([[11]])), dim=-1)
                        if failure == "empty":
                            sequences = ids
                        elif failure == "count":
                            sequences = torch.cat((sequences, torch.tensor([[12]])), dim=-1)
                        elif failure == "prefix":
                            sequences[0, 0] += 1
                        output = GenerateDecoderOnlyOutput(sequences=sequences, past_key_values=torch.ones(1))
                        outputs.append(output)
                        return output

                    state = torch.get_rng_state().clone()
                    with patch.object(engine.model, "generate", side_effect=generate), self.assertRaises(RuntimeError):
                        engine.sample([request(n=1)])
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))
                    self.assertEqual(engine.model.training, training)
                    for collector in collectors:
                        self.assertIsNone(collector.scores)
                        self.assertIsNone(collector.input_ids)
                        self.assertEqual(collector.logprobs, [])
                    self.assertTrue(all(output.past_key_values is None for output in outputs))

    def test_empty_and_mismatched_collector_calls_fail_closed(self):
        collector = _SelectedTokenLogprobs(2)
        for sequences in (torch.tensor([[9, 10]]), torch.tensor([[9, 10, 11]])):
            with self.assertRaisesRegex(RuntimeError, "streamed generation scores"):
                collector.finish(sequences)
        for ids, scores in ((torch.tensor([[9]]), torch.zeros(1, 32)),
                            (torch.tensor([[9, 10]]), torch.zeros(2, 32)),
                            (torch.tensor([9, 10]), torch.zeros(1, 32))):
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                collector(ids, scores)


if __name__ == "__main__":
    unittest.main()
