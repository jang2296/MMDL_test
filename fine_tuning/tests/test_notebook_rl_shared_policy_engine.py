"""Actual tiny Qwen/PEFT and processor checks; no pretrained or CUDA claims."""
from dataclasses import replace
import unittest
import weakref
from unittest.mock import patch

import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor,
                          Qwen3VLVideoProcessor)

from fine_tuning.notebook_rl.engines import RolloutRequest, _SelectedTokenLogprobs
from fine_tuning.notebook_rl.policy import (assert_base_unchanged, completion_logprobs,
                                           configure_policy, reference_logprobs)
from fine_tuning.notebook_rl.shared_policy_engine import SharedPolicyRolloutEngine
from fine_tuning.tests.test_notebook_rl_policy import config
from fine_tuning.tests.test_training import tiny_model


def processor():
    words = ["<pad>", "<unk>", "<eos>", "<|image_pad|>", "<|vision_start|>",
             "<|vision_end|>", "<|video_pad|>", "a", "b", "question", "assistant"]
    vocabulary = dict(zip(words, range(len(words)), strict=True))
    raw = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=raw, unk_token="<unk>",
        eos_token="<eos>", pad_token="<pad>", additional_special_tokens=words[3:7])
    template = ("{% for message in messages %}{% for block in message['content'] %}"
                "{% if block['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
                "{% else %} {{ block['text'] }} {% endif %}{% endfor %}{% endfor %} assistant")
    return Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(patch_size=2,
        temporal_patch_size=2, merge_size=2, min_pixels=64, max_pixels=64),
        tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(), chat_template=template)


def request(count=1, *, key=None, n=1):
    return RolloutRequest(key or f"images-{count}", None,
        [{"role": "user", "content": [{"type": "image"} for _ in range(count)]
            + [{"type": "text", "text": "question"}]}],
        [Image.new("RGB", (8, 8), (30 + i, 90, 120)) for i in range(count)],
        n, 3, list(range(31, 31+n)))


class SharedPolicyEngineTests(unittest.TestCase):
    threads: int

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def engine(self):
        bundle = configure_policy(tiny_model(), processor(), "MOD-RV", config(), enforce_counts=False)
        engine = SharedPolicyRolloutEngine(bundle, {"preprocessing": "hf_processor"}, 64, logprob_chunk=1)
        return bundle, engine

    def certify(self, bundle, engine):
        probes = []
        for count in (1, 2):
            item, ids = request(count), [12, 13]
            with torch.no_grad():
                expected = completion_logprobs(bundle,
                    engine.model_inputs(item, ids, torch.device("cpu")), len(ids), chunk=1)[0].tolist()
            probes.append((item, ids, expected))
        reports = engine.validate_sync(probes, .02)
        self.assertEqual([row["policy_version"] for row in reports], [engine.policy_version]*2)
        self.assertTrue(all(row["max_abs_delta"] == 0 for row in reports))

    def test_same_model_actual_processor_generation_logprobs_and_score_release(self):
        bundle = configure_policy(tiny_model(), processor(), "MOD-RV", config(), enforce_counts=False)
        with patch("copy.deepcopy", side_effect=AssertionError("no model copy")), patch(
                "fine_tuning.notebook_rl.policy.merged_named_tensors", side_effect=AssertionError("no merge")), patch(
                "fine_tuning.notebook_rl.engines._weight_inventory", side_effect=AssertionError("no second model")):
            engine = SharedPolicyRolloutEngine(bundle, {"preprocessing": "hf_processor"}, 64, logprob_chunk=1)
        self.assertIs(engine.model, bundle.model)
        self.assertIs(engine.processor, bundle.processor)
        with self.assertRaisesRegex(RuntimeError, "probes"):
            engine.sample([request()])
        self.certify(bundle, engine)
        bundle.model.generation_config.temperature = .7
        bundle.model.generation_config.top_p = .8
        bundle.model.generation_config.transformers_version = "4.57.1"
        score_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        cache_refs: list[weakref.ReferenceType] = []
        original = bundle.model.generate
        observe = _SelectedTokenLogprobs.__call__
        finish = _SelectedTokenLogprobs.finish

        def streamed_scores(collector, ids, scores):
            self.assertTrue(all(reference() is None for reference in score_refs[:-1]))
            score_refs.append(weakref.ref(scores))
            return observe(collector, ids, scores)

        def generate(*args, **kwargs):
            self.assertTrue(all(reference() is None for reference in score_refs))
            output = original(*args, **kwargs)
            self.assertIsNone(output.scores)
            cache_refs[:] = [weakref.ref(output.past_key_values)] + [
                weakref.ref(tensor) for layer in output.past_key_values.layers
                for tensor in (layer.keys, layer.values)]

            return output

        def streamed_finish(collector, sequences):
            self.assertTrue(all(reference() is None for reference in cache_refs))
            return finish(collector, sequences)

        state = torch.get_rng_state().clone()
        with patch.object(bundle.model, "generate", side_effect=generate), patch.object(
                _SelectedTokenLogprobs, "__call__", streamed_scores), patch.object(
                _SelectedTokenLogprobs, "finish", streamed_finish):
            result = engine.sample([request(n=2)])[0]
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(len(result.completions), 2)
        self.assertTrue(all(reference() is None for reference in score_refs))
        for completion in result.completions:
            actual = engine.probe_logprobs(request(), completion.token_ids)
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(completion.logprobs), rtol=0, atol=5e-7)
        assert_base_unchanged(bundle)

    def test_actual_update_requires_new_version_and_two_fresh_probes(self):
        bundle, engine = self.engine()
        self.certify(bundle, engine)
        batch = engine.model_inputs(request(), [12, 13], torch.device("cpu"))
        bundle.route.apply(False)
        reference = reference_logprobs(bundle, batch, 2).clone()
        before = engine.probe_logprobs(request(), [12, 13])
        optimizer = torch.optim.AdamW([p for params in bundle.params.values() for p in params], lr=.01)
        optimizer.zero_grad(set_to_none=True)
        completion_logprobs(bundle, batch, 2, chunk=1).sum().backward()
        optimizer.step()
        self.assertNotEqual(before, engine.probe_logprobs(request(), [12, 13]))
        self.assertTrue(torch.equal(reference, reference_logprobs(bundle, batch, 2)))
        engine.mark_updated(bundle, 1)
        with self.assertRaisesRegex(RuntimeError, "probes"):
            engine.sample([request()])
        with self.assertRaises(ValueError):
            engine.validate_sync([(request(), [12], [-1.])], .02)
        self.certify(bundle, engine)
        self.assertEqual(len(engine.sample([request()])[0].completions), 1)
        self.assertFalse(bundle.route.vision_on)
        self.assertFalse(any(p.requires_grad for p in bundle.params["V"]))
        assert_base_unchanged(bundle)

    def test_identity_partitions_versions_and_raw_weight_loading_fail_closed(self):
        bundle, engine = self.engine()
        self.certify(bundle, engine)
        with self.assertRaisesRegex(ValueError, "identity"):
            engine.mark_updated(replace(bundle), 1)
        for version in (0, 2, True, 1.5):
            with self.assertRaises(ValueError):
                engine.mark_updated(bundle, version)
        for operation in (engine.initialize_weights, engine.push_weights, engine._load_weights):
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                operation(iter(()), 1)
        original = bundle.params["L"][0]
        bundle.params["L"][0] = torch.nn.Parameter(original.detach().clone())
        with self.assertRaisesRegex(ValueError, "ownership"):
            engine.mark_updated(bundle, 1)
        with self.assertRaisesRegex(ValueError, "ownership"):
            engine.sample([request()])

    def test_trainer_synchronize_reuses_live_weights_without_merged_export(self):
        from fine_tuning.notebook_rl.trainer import SyncProbe, synchronize
        bundle, engine = self.engine()
        probes = [SyncProbe(request(n), [12, 13]) for n in (1, 2)]
        with patch("fine_tuning.notebook_rl.trainer.merged_named_tensors", side_effect=AssertionError("no weight export")):
            synchronize(bundle, engine, engine, probes, config(), 0, push=False)
            batch = engine.model_inputs(request(), [12, 13], torch.device("cpu"))
            completion_logprobs(bundle, batch, 2, chunk=1).sum().backward()
            optimizer = torch.optim.SGD([p for ps in bundle.params.values() for p in ps], lr=.01)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            synchronize(bundle, engine, engine, probes, config(), 1, push=True)
        self.assertEqual(engine.policy_version, 1)
        self.assertTrue(engine._synchronized)
        self.assertIs(engine.model, bundle.model)

    def test_training_mode_vision_route_and_bounded_projection_restored_even_on_error(self):
        bundle, engine = self.engine()
        self.certify(bundle, engine)
        for training in (False, True):
            for vision_on in (False, True):
                bundle.model.train(training)
                bundle.route.apply(vision_on)
                with patch.object(bundle.model, "forward", side_effect=AssertionError("no full vocabulary forward")), patch.object(
                        bundle.model.get_base_model().lm_head, "forward", wraps=bundle.model.get_base_model().lm_head.forward) as projection:
                    engine.probe_logprobs(request(), [12, 13])
                self.assertTrue(all(call.args[0].shape[1] <= 1 for call in projection.call_args_list))
                engine.sample([request()])
                with patch.object(bundle.model, "generate", side_effect=RuntimeError("authored failure")):
                    with self.assertRaisesRegex(RuntimeError, "authored failure"):
                        engine.sample([request()])
                with patch("fine_tuning.notebook_rl.shared_policy_engine.completion_logprobs", side_effect=RuntimeError("probe failure")):
                    with self.assertRaisesRegex(RuntimeError, "probe failure"):
                        engine.probe_logprobs(request(), [12])
                self.assertEqual(bundle.model.training, training)
                self.assertEqual(bundle.route.vision_on, vision_on)
                self.assertTrue(all(p.requires_grad == vision_on for p in bundle.params["V"]))


if __name__ == "__main__":
    unittest.main()
