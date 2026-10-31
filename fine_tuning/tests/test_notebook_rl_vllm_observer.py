"""CPU checks with mocked vLLM: opt-in prefix-cache flag and per-completion observer. No GPU."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl.engines import (Completion, InputPreparation, QWEN_OFFICIAL,
    SamplingSettings, VLLMRolloutEngine, vllm_kwargs)
from fine_tuning.tests.test_notebook_rl_engines import Processor, Tokenizer, request
from fine_tuning.tests.test_training import tiny_model

IMAGE_CFG = {"preprocessing": "qwen_vl_utils_0_0_14", "min_pixels": 1003520, "max_pixels": 4014080}
BUDGETS = (10737418240, 16, 16384)
# Request "a" has seeds 31-32, request "b" has seeds 31-33 (request(n) uses range(31, 31+n)).
ORDER = [("a", 0, 31), ("a", 1, 32), ("b", 0, 31), ("b", 1, 32), ("b", 2, 33)]


def fake_sampling_params(**kwargs):
    return SimpleNamespace(**kwargs)


class VLLMObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def run_observed(self, method, *args):
        engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
        InputPreparation.__init__(engine, Processor(), tiny_model().config,
                                  {"preprocessing": "hf_processor"}, 64)
        engine._call, engine._synchronized = 0, True
        observed = []

        def generate(prompts, params, **kwargs):
            # Observers must not run while the single vLLM call is still generating.
            self.assertEqual(observed, [])
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[SimpleNamespace(
                token_ids=[12 + j], logprobs=[{12 + j: SimpleNamespace(logprob=-1.)}],
                finish_reason="stop")]) for j, p in enumerate(prompts)]

        engine.llm = SimpleNamespace(generate=generate)
        requests = [request(key="a", n=2), request(key="b", n=3)]
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=fake_sampling_params)}):
            results = getattr(engine, method)(requests, *args,
                                              on_completion=lambda *row: observed.append(row))
        return results, observed

    def test_on_completion_runs_after_generate_in_request_then_index_order(self):
        for method, args in (("sample", ()), ("sample_with_settings", (QWEN_OFFICIAL,))):
            with self.subTest(method=method):
                results, observed = self.run_observed(method, *args)
                by_id = {r.request_id: r for r in results}
                self.assertEqual([c.token_ids for r in results for c in r.completions],
                                 [[12], [13], [14], [15], [16]])
                self.assertEqual(observed, [(rid, i, seed, by_id[rid].completions[i])
                                            for rid, i, seed in ORDER])
                for (rid, i, _), row in zip(ORDER, observed, strict=True):
                    self.assertIs(type(row[3]), Completion)
                    self.assertIsNot(row[3], by_id[rid].completions[i])

    def test_none_observer_keeps_sample_dispatch_shape(self):
        engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
        r = request()
        with patch.object(engine, "sample_with_settings", return_value=[]) as sample:
            self.assertEqual(engine.sample([r]), [])
            sample.assert_called_with([r], SamplingSettings())
            self.assertEqual(engine.sample([r], on_completion=None), [])
            sample.assert_called_with([r], SamplingSettings())
            self.assertEqual(sample.call_count, 2)

    def test_prefix_caching_flag_default_false_true_passthrough_and_bool_only(self):
        default = vllm_kwargs(Path("model"), IMAGE_CFG, *BUDGETS)
        enabled = vllm_kwargs(Path("model"), IMAGE_CFG, *BUDGETS, enable_prefix_caching=True)
        self.assertIs(default["enable_prefix_caching"], False)
        self.assertIs(enabled["enable_prefix_caching"], True)
        self.assertEqual({k: v for k, v in enabled.items() if k != "enable_prefix_caching"},
                         {k: v for k, v in default.items() if k != "enable_prefix_caching"})
        for value in (1, None, "yes"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                vllm_kwargs(Path("model"), IMAGE_CFG, *BUDGETS, enable_prefix_caching=value)

        captured = []

        def fake_llm(**kwargs):
            captured.append(kwargs)
            return SimpleNamespace()

        processor = SimpleNamespace(tokenizer=Tokenizer(), image_processor=SimpleNamespace(size=None))
        with (patch("fine_tuning.notebook_rl.engines.version", return_value="0.11.0"),
              patch.dict(os.environ, {}),
              patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
              patch("transformers.AutoConfig.from_pretrained", return_value=tiny_model().config),
              patch("fine_tuning.notebook_rl.engines._weight_inventory", return_value={}),
              patch.dict(sys.modules, {"vllm": SimpleNamespace(LLM=fake_llm)})):
            VLLMRolloutEngine(Path("model"), IMAGE_CFG, *BUDGETS)
            VLLMRolloutEngine(Path("model"), IMAGE_CFG, *BUDGETS, enable_prefix_caching=True)
        self.assertIs(captured[0]["enable_prefix_caching"], False)
        self.assertIs(captured[1]["enable_prefix_caching"], True)
