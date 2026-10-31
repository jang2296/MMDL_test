"""Tiny actual HF inference and mocked vLLM boundary checks, not GPU certification."""
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.notebook_rl.engines import (Completion, HFRolloutEngine, RolloutRequest,
    VLLMRolloutEngine, InputPreparation, Prepared, SamplingSettings, QWEN_OFFICIAL,
    _SelectedTokenLogprobs, _weight_inventory, vllm_kwargs)
from fine_tuning.tests.test_training import tiny_model


class Tokenizer:
    def encode(self, text, **kwargs):
        return [9, 10]

    def decode(self, ids, **kwargs):
        return ":".join(map(str, ids))


class Processor:
    tokenizer = Tokenizer()
    image_processor = SimpleNamespace(merge_size=2)

    def apply_chat_template(self, messages, **kwargs):
        return "fixed authored CPU fixture"

    def __call__(self, text, images, **kwargs):
        ids = [token for _ in images for token in [4, 3, 3, 3, 3, 5]] + [9, 10]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "pixel_values": torch.linspace(-1, 1, 16*24*len(images)).reshape(-1, 24),
                "image_grid_thw": torch.tensor([[1, 4, 4] for _ in images])}


def request(images=0, ids=None, key="r", n=2):
    return RolloutRequest(key, ids, None if ids is not None else [{"role": "user", "content": "fixture"}],
                          [Image.new("RGB", (4, 4)) for _ in range(images)], n, 3, list(range(31, 31+n)))


class EncoderManager:
    """vLLM0.11 EncoderCacheManager initial-state/idle reference contract."""
    def __init__(self, cache_size):
        self.cache_size = self.num_free_slots = self.num_freeable_slots = cache_size
        self.cached, self.freeable, self.freed = {}, {}, []


class InprocClient:
    def __init__(self, scheduler):
        self.engine_core = SimpleNamespace(scheduler=scheduler)


VLLM_MODULES = {"vllm.v1.engine.core_client": SimpleNamespace(InprocClient=InprocClient)}


def pushed_engine(*, prefix_ack=True, encoder_ack=True, busy=False, loaded=True):
    engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
    engine.policy_version, engine._synchronized, engine._push_complete = 0, True, True
    engine._initialization_open = True
    engine._last_loaded_version = None
    engine._weight_shapes = {"w": (1,), "merger.bias": (2,)}
    manager = EncoderManager(64)
    manager.cached["old-v0"] = set()
    manager.freeable["old-v0"] = 4
    manager.num_free_slots = 60
    scheduler = SimpleNamespace(encoder_cache_manager=manager,
        has_unfinished_requests=lambda: busy, reset_prefix_cache=lambda: prefix_ack)
    worker = SimpleNamespace(model_runner=SimpleNamespace(encoder_cache={"old-v0": torch.ones(4)}))
    parameters = {n: torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
                  for n, shape in engine._weight_shapes.items()}
    def load_weights(pairs):
        names = []
        for name, value in pairs:
            parameters[name].data.copy_(value)
            names.append(name)
        return names if loaded else names[:-1]
    engine.llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=InprocClient(scheduler)),
        apply_model=lambda f: [f(SimpleNamespace(load_weights=load_weights, named_parameters=lambda: parameters.items()))],
        collective_rpc=lambda f: [f(worker)] if encoder_ack else [False],
        # V1 public wrapper returns None: implementation must not rely on it.
        reset_prefix_cache=lambda: None)
    return engine, scheduler, worker, parameters


def weights():
    return [("w", torch.ones(1, dtype=torch.bfloat16)), ("merger.bias", torch.ones(2, dtype=torch.bfloat16))]


def expand_token_prompt(ids, lengths):
    """Exact 0.11 _apply_matches rule: replace the first unused original token per item.

    Source: multimodal/processing.py:668-791 and models/qwen3_vl.py:929-937,972-976.
    The native source functions were independently exercised without importing CUDA.
    """
    result, offset = [], 0
    for length in lengths:
        index = ids.index(3, offset)
        result.extend(ids[offset:index] + [3]*length)
        offset = index + 1
    return result + ids[offset:]


class EnginesTests(unittest.TestCase):
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

    def test_actual_tiny_transformers_seed_order_logprobs_and_rng_restore(self):
        engine = self.engine()
        state = torch.get_rng_state().clone()
        first = engine.sample([request()])[0]
        second = engine.sample([request()])[0]
        self.assertEqual(first, second)
        self.assertEqual(len(first.completions), 2)
        self.assertEqual(first.prompt_token_ids, [9, 10])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(engine.model.training)
        for c in first.completions:
            self.assertEqual(len(c.token_ids), len(c.logprobs))
            self.assertIn(c.finish_reason, ("stop", "length"))
            expected = engine.probe_logprobs(request(), c.token_ids)
            torch.testing.assert_close(torch.tensor(c.logprobs), torch.tensor(expected), atol=1e-5, rtol=1e-5)

    def test_actual_cache_released_before_scores_without_changing_completions(self):
        from transformers import GenerationConfig

        engine = self.engine()
        generate = engine.model.generate
        finish = _SelectedTokenLogprobs.finish
        for training in (False, True):
            for count in (0, 1, 2):
                item = request(count)
                expected, extracted, finish_checks = [], [], []
                engine.model.train(training)

                def observed_generate(*args, **kwargs):
                    legacy_cfg = GenerationConfig.from_dict(kwargs["generation_config"].to_dict())
                    legacy_cfg.output_scores, legacy_cfg.renormalize_logits = True, True
                    with torch.random.fork_rng():
                        legacy = generate(*args, **{**kwargs, "generation_config": legacy_cfg,
                                                   "logits_processor": None})
                    ids = legacy.sequences[0, kwargs["input_ids"].shape[1]:].tolist()
                    lps = [float(score[0].float().log_softmax(-1)[token])
                           for token, score in zip(ids, legacy.scores, strict=True)]
                    legacy["past_key_values"] = None
                    del legacy
                    output = generate(*args, **kwargs)
                    self.assertIsNone(output.scores)
                    self.assertIs(output["past_key_values"], output.past_key_values)
                    refs = [weakref.ref(output.past_key_values)] + [
                        weakref.ref(tensor) for layer in output.past_key_values.layers
                        for tensor in (layer.keys, layer.values)]
                    self.assertTrue(all(reference() is not None for reference in refs))
                    self.assertEqual(output.sequences[0, kwargs["input_ids"].shape[1]:].tolist(), ids)
                    eos = engine.model.generation_config.eos_token_id
                    eos_ids = eos if isinstance(eos, list) else [eos]
                    expected.append(Completion(ids, lps, "stop" if ids[-1] in eos_ids else "length",
                        engine.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)))
                    finish_checks.append((weakref.ref(output), refs))
                    return output

                def streamed_finish(collector, sequences):
                    output_ref, refs = finish_checks.pop(0)
                    live_output = output_ref()
                    assert live_output is not None
                    self.assertIsNone(live_output.past_key_values)
                    self.assertIsNone(live_output["past_key_values"])
                    self.assertTrue(all(reference() is None for reference in refs))
                    extracted.append(True)
                    return finish(collector, sequences)

                state = torch.get_rng_state().clone()
                with self.subTest(training=training, images=count), patch.object(
                        engine.model, "generate", side_effect=observed_generate), patch.object(
                        _SelectedTokenLogprobs, "finish", streamed_finish):
                    result = engine.sample([item])[0]
                    self.assertEqual(result.completions, expected)
                    self.assertEqual(len(extracted), item.n)
                    self.assertEqual(engine.model.training, training)
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_model_defaults_cannot_override_explicit_rollout_recipe(self):
        engine = self.engine()
        model_defaults = engine.model.generation_config
        model_defaults.temperature, model_defaults.top_p = .7, .8
        model_defaults.bos_token_id, model_defaults.transformers_version = 1, "4.57.1"
        original = engine.model._prepare_generation_config
        observed = []
        def prepare(*args, **kwargs):
            effective, model_kwargs = original(*args, **kwargs)
            observed.append((effective.do_sample, effective.temperature, effective.top_p,
                             effective.top_k, effective.repetition_penalty, effective.bos_token_id))
            return effective, model_kwargs
        with patch.object(engine.model, "_prepare_generation_config", side_effect=prepare):
            result = engine.sample([request()])[0]
        self.assertEqual(observed, [(True, 1.0, 1.0, 0, 1.0, 1)]*2)
        self.assertEqual((model_defaults.temperature, model_defaults.top_p), (.7, .8))
        for completion in result.completions:
            expected = engine.probe_logprobs(request(), completion.token_ids)
            torch.testing.assert_close(torch.tensor(completion.logprobs), torch.tensor(expected),
                                       atol=1e-5, rtol=1e-5)

    def test_explicit_sampling_pins_and_hf_official_refusal(self):
        self.assertEqual(SamplingSettings().name, "temperature_1")
        self.assertEqual(QWEN_OFFICIAL.name, "qwen_official")
        for settings in (SamplingSettings(float("nan")), SamplingSettings(top_k=True),
                         SamplingSettings(.7, .8, 20, 0., 1.)):
            with self.assertRaisesRegex(ValueError, "pinned"):
                settings.validate()
        with self.assertRaisesRegex(NotImplementedError, "presence penalty"):
            self.engine().sample_with_settings([request()], QWEN_OFFICIAL)

    def test_vllm_native_official_sampling_parameters_are_not_ignored(self):
        engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
        InputPreparation.__init__(engine, Processor(), tiny_model().config, {"preprocessing": "hf_processor"}, 64)
        engine._call, engine._synchronized = 0, True
        captured = []
        def generate(prompts, params, **kwargs):
            captured.extend(params)
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[SimpleNamespace(
                token_ids=[12], logprobs=[{12: SimpleNamespace(logprob=-1.)}], finish_reason="stop")]) for p in prompts]
        engine.llm = SimpleNamespace(generate=generate)
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw))}):
            engine.sample_with_settings([request()], QWEN_OFFICIAL)
        self.assertEqual([(p.temperature, p.top_p, p.top_k, p.presence_penalty, p.repetition_penalty)
                          for p in captured], [(.7, .8, 20, 1.5, 1.)]*2)

    def test_actual_image_inputs_retry_ids_and_immutable_pixels(self):
        engine = self.engine()
        for n in (1, 2):
            original = request(n)
            prepared = engine.prepare(original)
            retry = request(n, ids=prepared.token_ids+[11, 2, 9], key="retry")
            actual = engine.prepare(retry)
            self.assertEqual(actual.token_ids, retry.prompt_token_ids)
            self.assertTrue(torch.equal(prepared.pixels["pixel_values"], actual.pixels["pixel_values"]))
            probabilities = engine.probe_logprobs(retry, [12, 13])
            self.assertEqual(len(probabilities), 2)
            self.assertTrue(all(p < 0 for p in probabilities))

    def test_request_context_and_image_mismatch_fail_closed(self):
        engine = self.engine()
        for bad in (request(ids=[9]*63), request(images=1, ids=[9, 10]),
                    RolloutRequest("x", [9], [], [], 1, 2, [2]),
                    RolloutRequest("x", [9], None, [], 2, 2, [2, 2])):
            with self.assertRaises(ValueError):
                engine.prepare(bad)
        with self.assertRaises(ValueError):
            engine.sample([request(), request()])

    def test_transformers_complete_weight_push_and_version(self):
        engine = self.engine()
        weights = [(n, p.detach().clone()+.01) for n, p in engine.model.named_parameters()]
        with self.assertRaises(ValueError):
            engine.push_weights(weights[:-1], 1)
        engine.push_weights(weights, 1)
        for n, p in engine.model.named_parameters():
            self.assertTrue(torch.equal(p, dict(weights)[n]))
        with self.assertRaises(ValueError):
            engine.push_weights(weights, 1)

    def test_transformers_initialize_arbitrary_resume_version_requires_two_actual_image_probes(self):
        engine, reference = self.engine(), self.engine()
        named = [(n, p.detach().clone()+.01) for n, p in reference.model.named_parameters()]
        reference.push_weights(named, 1)
        probes = [(request(count, key=f"probe-{count}"), [12, 13],
                   reference.probe_logprobs(request(count), [12, 13])) for count in (1, 2)]
        engine.prepare(request(1))  # CPU preparation alone does not consume the fresh window.
        engine.initialize_weights(iter(named), policy_version=19)
        self.assertEqual(engine.policy_version, 19)
        for n, p in engine.model.named_parameters():
            torch.testing.assert_close(p, dict(named)[n], atol=0, rtol=0)
        for invalid in ([], probes[:1], [probes[0], probes[0]]):
            with self.assertRaises(ValueError):
                engine.validate_sync(invalid, .02)
            with self.assertRaisesRegex(RuntimeError, "probes"):
                engine.sample([request()])
        with self.assertRaises(ValueError):
            engine.push_weights(named, 20)
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            engine.validate_sync([(r, ids, [p-1 for p in ref]) for r, ids, ref in probes], .02)
        reports = engine.validate_sync(probes, .02)
        self.assertEqual([r["policy_version"] for r in reports], [19, 19])
        self.assertEqual(len(engine.sample([request()])[0].completions), 2)
        engine.push_weights(named, 20)
        with self.assertRaises(ValueError):
            engine.push_weights(named, 22)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            engine.initialize_weights(named, 20)

    def test_initialize_rejects_invalid_versions_and_used_engines(self):
        for make in (self.engine, lambda: pushed_engine()[0]):
            engine = make()
            for bad in (-1, True, 1.5):
                with self.subTest(backend=type(engine).__name__, bad=bad), self.assertRaises(ValueError):
                    engine.initialize_weights([], bad)
            self.assertTrue(engine._initialization_open)
        for operation in (lambda e: e.sample([request()]),
                          lambda e: e.probe_logprobs(request(), [12]),
                          lambda e: e.validate_sync([], .02)):
            engine = self.engine()
            try:
                operation(engine)
            except ValueError:
                pass
            with self.assertRaisesRegex(RuntimeError, "fresh engine"):
                engine.initialize_weights([], 4)
        engine = self.engine()
        engine.push_weights([(n, p.detach()) for n, p in engine.model.named_parameters()], 1)
        with self.assertRaisesRegex(RuntimeError, "fresh engine"):
            engine.initialize_weights([], 0)

    def test_failed_transformers_initialize_is_one_shot_and_never_certifiable(self):
        for corrupt in ("missing", "duplicate", "dtype", "identity"):
            engine = self.engine()
            named = [(n, p.detach().clone()) for n, p in engine.model.named_parameters()]
            if corrupt == "missing":
                named.pop()
            elif corrupt == "duplicate":
                named.append(named[0])
            elif corrupt == "dtype":
                named[0] = (named[0][0], named[0][1].double())
            else:
                engine.model.extra_parameter = torch.nn.Parameter(torch.zeros(1))
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                engine.initialize_weights(named, 8)
            self.assertEqual(engine.policy_version, 0)
            with self.assertRaisesRegex(RuntimeError, "probes"):
                engine.sample([])
            with self.assertRaisesRegex(RuntimeError, "partial"):
                engine.validate_sync([], .02)
            with self.assertRaisesRegex(RuntimeError, "only once"):
                engine.initialize_weights(named, 8)

    def test_initialize_accepts_zero_version_but_still_requires_probes(self):
        engine = self.engine()
        named = [(n, p.detach()) for n, p in engine.model.named_parameters()]
        engine.initialize_weights(named, 0)
        self.assertEqual(engine.policy_version, 0)
        with self.assertRaisesRegex(RuntimeError, "probes"):
            engine.sample([])
        engine, *_ = pushed_engine()
        self.assertIsNone(engine._last_loaded_version)
        with patch.dict(sys.modules, VLLM_MODULES):
            engine.initialize_weights(weights(), 0)
        self.assertEqual(engine.policy_version, 0)
        self.assertEqual(engine._last_loaded_version, 0)
        self.assertFalse(engine._synchronized)

    def test_vllm_initialize_uses_acknowledged_cache_clear_and_actual_probe_path(self):
        engine, scheduler, worker, parameters = pushed_engine()
        InputPreparation.__init__(engine, Processor(), tiny_model().config, {"preprocessing": "hf_processor"}, 64)
        engine._call = 0
        calls = []
        def generate(prompts, params, **kwargs):
            calls.extend(prompts)
            outputs = []
            for prompt in prompts:
                ids = expand_token_prompt(prompt["prompt_token_ids"], [4]*len(prompt["multi_modal_data"]["image"]))
                outputs.append(SimpleNamespace(prompt_token_ids=ids,
                    outputs=[SimpleNamespace(token_ids=[12], logprobs=[{12: SimpleNamespace(logprob=-1.)}],
                                             finish_reason="stop")],
                    prompt_logprobs=[None]+[{t: SimpleNamespace(logprob=-1.)} for t in ids[1:]]))
            return outputs
        engine.llm.generate = generate
        modules = {**VLLM_MODULES, "vllm": SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw))}
        with patch.dict(sys.modules, modules):
            engine.initialize_weights(iter(weights()), policy_version=37)
            self.assertEqual(engine.policy_version, 37)
            self.assertEqual(engine._last_loaded_version, 37)
            self.assertTrue(all(bool((p == 1).all()) for p in parameters.values()))
            self.assertEqual(worker.model_runner.encoder_cache, {})
            self.assertEqual(scheduler.encoder_cache_manager.num_free_slots, 64)
            with self.assertRaisesRegex(RuntimeError, "probes"):
                engine.sample([request(1)])
            self.assertEqual(calls, [])
            probes = [(request(n, key=f"probe-{n}"), [12, 13], [-1., -1.]) for n in (1, 2)]
            self.assertEqual(len(engine.validate_sync(probes, .02)), 2)
            self.assertEqual(len(calls), 2)
            self.assertEqual(engine.sample([request(1)])[0].completions[0].token_ids, [12])
            with self.assertRaisesRegex(RuntimeError, "only once"):
                engine.initialize_weights(weights(), 37)
            engine.push_weights(weights(), 38)
            self.assertFalse(engine._synchronized)
            self.assertEqual(engine._last_loaded_version, 38)

    def test_failed_vllm_initialize_is_one_shot_and_uncommitted(self):
        for kwargs, pairs in (({}, weights()[:1]), ({"prefix_ack": False}, weights()),
                              ({"encoder_ack": False}, weights()), ({"loaded": False}, weights())):
            engine, *_ = pushed_engine(**kwargs)
            with self.subTest(kwargs=kwargs), patch.dict(sys.modules, VLLM_MODULES):
                with self.assertRaises((ValueError, RuntimeError)):
                    engine.initialize_weights(pairs, 6)
            self.assertEqual(engine.policy_version, 0)
            self.assertFalse(engine._synchronized)
            self.assertIsNone(engine._last_loaded_version)
            with self.assertRaisesRegex(RuntimeError, "partial"):
                engine.validate_sync([], .02)
            with self.assertRaisesRegex(RuntimeError, "only once"):
                engine.initialize_weights(weights(), 6)

    def test_vllm_exact_arguments_do_not_silently_shrink(self):
        args = vllm_kwargs(Path("model"), {"preprocessing": "qwen_vl_utils_0_0_14",
            "min_pixels": 1003520, "max_pixels": 4014080}, 10737418240, 16, 16384)
        self.assertEqual(args["kv_cache_memory_bytes"], 10737418240)
        self.assertEqual(args["gpu_memory_utilization"], 0.7)
        self.assertEqual(args["max_num_seqs"], 16)
        self.assertEqual(args["max_model_len"], 16384)
        self.assertEqual(args["dtype"], "bfloat16")
        self.assertFalse(args["enforce_eager"])
        self.assertFalse(args["mm_processor_kwargs"]["do_resize"])
        self.assertEqual(args["mm_processor_cache_gb"], 0)
        self.assertIsNone(args["quantization"])

    def test_vllm_postpush_fail_closed_until_both_probes_pass(self):
        engine, scheduler, worker, _ = pushed_engine()
        with patch.dict(sys.modules, VLLM_MODULES):
            engine.push_weights(iter(weights()), 1)
        self.assertEqual(worker.model_runner.encoder_cache, {})
        self.assertEqual(scheduler.encoder_cache_manager.cached, {})
        self.assertEqual(scheduler.encoder_cache_manager.num_free_slots, 64)
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=None)}):
            with self.assertRaisesRegex(RuntimeError, "probes"):
                engine.sample([])
        engine.probe_logprobs = lambda r, c: [-1.0] * len(c)
        probes = [(request(1), [12], [-1.0]), (request(2), [12], [-1.0])]
        self.assertEqual(len(engine.validate_sync(probes, .02)), 2)
        self.assertTrue(engine._synchronized)
        with patch.dict(sys.modules, VLLM_MODULES):
            engine.push_weights(iter(weights()), 2)
        with self.assertRaises(RuntimeError):
            engine.validate_sync([(r, ids, [-2.0]) for r, ids, _ in probes], .02)
        self.assertFalse(engine._synchronized)

    def test_vllm_source_rule_transport_expands_once_and_preserves_literal_history(self):
        engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
        InputPreparation.__init__(engine, Processor(), tiny_model().config, {"preprocessing": "hf_processor"}, 64)
        engine._call = 0
        for count in (1, 2):
            prepared = engine.prepare(request(count))
            prior = [11, 151645, 8, 7, 9]
            ids = prepared.token_ids + prior
            transported = engine._prompt(prepared, ids)
            self.assertEqual(transported["prompt_token_ids"].count(3), count)
            self.assertEqual(expand_token_prompt(transported["prompt_token_ids"], [4]*count), ids)
            self.assertEqual(transported["prompt_token_ids"][-len(prior):], prior)
            self.assertEqual(prepared.token_ids.count(3), 4*count)
            self.assertNotEqual(expand_token_prompt(ids, [4]*count), ids)
        bad = Prepared([4, 3, 5], {}, [Image.new("RGB", (4, 4))], "key")
        with self.assertRaisesRegex(ValueError, "registered"):
            engine._prompt(bad, bad.token_ids)

    def test_vllm_sample_and_probe_transport_compare_original_expanded_ids(self):
        engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
        InputPreparation.__init__(engine, Processor(), tiny_model().config, {"preprocessing": "hf_processor"}, 64)
        engine._call, engine._synchronized = 0, True
        calls = []
        def generate(prompts, params, **kwargs):
            calls.extend(prompts)
            outputs = []
            for prompt in prompts:
                ids = expand_token_prompt(prompt["prompt_token_ids"], [4]*len(prompt["multi_modal_data"]["image"]))
                outputs.append(SimpleNamespace(prompt_token_ids=ids,
                    outputs=[SimpleNamespace(token_ids=[12], logprobs=[{12: SimpleNamespace(logprob=-1.)}],
                                             finish_reason="stop")],
                    prompt_logprobs=[None] + [{token: SimpleNamespace(logprob=-1.)} for token in ids[1:]]))
            return outputs
        engine.llm = SimpleNamespace(generate=generate)
        module = SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw))
        with patch.dict(sys.modules, {"vllm": module}):
            for count in (1, 2):
                r = request(count)
                expected = engine.prepare(r).token_ids
                result = engine.sample([r])[0]
                self.assertEqual(result.prompt_token_ids, expected)
                with self.assertRaisesRegex(RuntimeError, "fresh engine"):
                    engine.initialize_weights([], 9)
                self.assertEqual(engine.probe_logprobs(r, [12, 13]), [-1., -1.])
                self.assertEqual(calls[-1]["prompt_token_ids"][-2:], [12, 13])
            def altered(*args, **kwargs):
                outputs = generate(*args, **kwargs)
                outputs[0].prompt_token_ids.append(99)
                return outputs
            engine.llm.generate = altered
            with self.assertRaisesRegex(RuntimeError, "actual vLLM"):
                engine.sample([request(1)])
            with self.assertRaisesRegex(RuntimeError, "probe input"):
                engine.probe_logprobs(request(1), [12])
        self.assertEqual(len({u for call in calls for u in call["multi_modal_uuids"]["image"]}),
                         sum(len(call["multi_modal_uuids"]["image"]) for call in calls))

    def test_vllm_partial_duplicate_bad_dtype_shape_nonfinite_and_native_missing_refused(self):
        invalid = [weights()[:1], weights()+weights()[:1], weights()+[("unknown", torch.ones(1))],
                   [("w", torch.ones(1)), weights()[1]],
                   [("w", torch.ones(2, dtype=torch.bfloat16)), weights()[1]],
                   [("w", torch.tensor([float("nan")], dtype=torch.bfloat16)), weights()[1]]]
        for bad in invalid:
            engine, *_ = pushed_engine()
            with self.subTest(bad=bad[0][0]), patch.dict(sys.modules, VLLM_MODULES), self.assertRaises(ValueError):
                engine.push_weights(iter(bad), 1)
            self.assertEqual(engine.policy_version, 0)
            self.assertFalse(engine._synchronized)
            with self.assertRaisesRegex(RuntimeError, "partial"):
                engine.validate_sync([], .02)
        engine, *_ = pushed_engine(loaded=False)
        with patch.dict(sys.modules, VLLM_MODULES), self.assertRaisesRegex(RuntimeError, "native"):
            engine.push_weights(iter(weights()), 1)

    def test_vllm_cache_idle_topology_and_ack_failures_leave_version_uncommitted(self):
        for kwargs in ({"prefix_ack": False}, {"prefix_ack": None}, {"encoder_ack": False}, {"busy": True}):
            engine, *_ = pushed_engine(**kwargs)
            with self.subTest(kwargs=kwargs), patch.dict(sys.modules, VLLM_MODULES), self.assertRaises(RuntimeError):
                engine.push_weights(iter(weights()), 1)
            self.assertEqual(engine.policy_version, 0)
            self.assertFalse(engine._push_complete)
        engine, *_ = pushed_engine()
        engine.llm.llm_engine.engine_core = SimpleNamespace()
        with patch.dict(sys.modules, VLLM_MODULES), self.assertRaisesRegex(RuntimeError, "in-process"):
            engine.push_weights(iter(weights()), 1)
        engine, *_ = pushed_engine()
        def failed_rpc(callback):
            raise RuntimeError("worker cache clear failed")
        engine.llm.collective_rpc = failed_rpc
        with patch.dict(sys.modules, VLLM_MODULES), self.assertRaisesRegex(RuntimeError, "worker cache"):
            engine.push_weights(iter(weights()), 1)
        self.assertEqual(engine.policy_version, 0)
        self.assertFalse(engine._push_complete)

    def test_meta_weight_inventory_matches_actual_tiny_tied_architecture(self):
        model = tiny_model()
        for tied in (False, True):
            model.config.tie_word_embeddings = model.config.text_config.tie_word_embeddings = tied
            expected = tiny_model()
            expected.config.tie_word_embeddings = expected.config.text_config.tie_word_embeddings = tied
            expected.tie_weights()
            self.assertEqual(_weight_inventory(model.config), {n: tuple(p.shape) for n, p in expected.named_parameters()})


if __name__ == "__main__":
    unittest.main()
