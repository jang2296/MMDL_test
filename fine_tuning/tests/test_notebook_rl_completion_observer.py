"""Tiny actual CPU generation observers; no Qwen checkpoint or learning claims."""
import tempfile
import unittest
import weakref
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl.engines import HFRolloutEngine, SamplingSettings, _SelectedTokenLogprobs
from fine_tuning.notebook_rl.local_eval import read, save
from fine_tuning.tests import test_notebook_rl_engines as hf
from fine_tuning.tests import test_notebook_rl_shared_policy_engine as shared


class CompletionObserverTests(unittest.TestCase):
    threads: int

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def engine(self, kind):
        if kind == "hf":
            return None, hf.EnginesTests().engine(), hf.request(images=1, n=4)
        fixture = shared.SharedPolicyEngineTests()
        bundle, engine = fixture.engine()
        fixture.certify(bundle, engine)
        bundle.route.apply(False)
        return bundle, engine, shared.request(count=1, n=4)

    def test_actual_n4_observation_preserves_outputs_rng_and_model_mode(self):
        for kind in ("hf", "shared"):
            bundle, engine, request = self.engine(kind)
            self.addCleanup(request.images[0].close)
            for training in (False, True):
                with self.subTest(kind=kind, training=training):
                    engine.model.train(training)
                    rng = torch.get_rng_state().clone()
                    expected = engine.sample([request])
                    observed = []
                    actual = engine.sample_with_settings([request], SamplingSettings(),
                        on_completion=lambda *row: observed.append(row))
                    self.assertEqual(actual, expected)
                    self.assertEqual(observed, [(request.request_id, index, seed, completion)
                        for index, (seed, completion) in enumerate(zip(
                            request.seeds, expected[0].completions, strict=True))])
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertEqual(engine.model.training, training)
                    self.assertEqual(request.n, 4)
                    if bundle is not None:
                        self.assertFalse(bundle.route.vision_on)

    def test_none_observer_preserves_existing_dispatch_call_shapes(self):
        _, engine, request = self.engine("hf")
        self.addCleanup(request.images[0].close)
        with patch.object(engine, "sample_with_settings", return_value=[]) as sample:
            self.assertEqual(engine.sample([request], on_completion=None), [])
            sample.assert_called_once_with([request], SamplingSettings())
        _, engine, request = self.engine("shared")
        self.addCleanup(request.images[0].close)
        with patch.object(HFRolloutEngine, "sample_with_settings", return_value=[]) as sample:
            self.assertEqual(engine.sample_with_settings([request], SamplingSettings(), on_completion=None), [])
            sample.assert_called_once_with([request], SamplingSettings())

    def test_first_completion_saved_after_cleanup_before_later_generate_failure(self):
        for kind in ("hf", "shared"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                bundle, engine, request = self.engine(kind)
                self.addCleanup(request.images[0].close)
                engine.model.train(True)
                path = Path(directory) / "completion-0.json"
                generate = engine.model.generate
                clear = _SelectedTokenLogprobs.clear
                collectors = []
                cache_refs: list[weakref.ReferenceType] = []
                calls = 0

                def actual_generate(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        self.assertTrue(path.is_file())
                        raise RuntimeError("deliberate later generation failure")
                    output = generate(*args, **kwargs)
                    cache_refs.append(weakref.ref(output.past_key_values))
                    return output

                def cleared(collector):
                    clear(collector)
                    collectors.append(collector)

                def persist(request_id, index, seed, completion):
                    self.assertTrue(collectors)
                    self.assertTrue(all(collector.scores is None and collector.input_ids is None
                                        and not collector.logprobs for collector in collectors))
                    self.assertTrue(all(reference() is None for reference in cache_refs))
                    save(path, {"request_id": request_id, "index": index, "seed": seed,
                                "completion": asdict(completion)})

                with patch.object(engine.model, "generate", side_effect=actual_generate), patch.object(
                        _SelectedTokenLogprobs, "clear", cleared), self.assertRaisesRegex(
                            RuntimeError, "later generation failure"):
                    engine.sample([request], on_completion=persist)
                saved = read(path)
                self.assertEqual((saved["request_id"], saved["index"], saved["seed"]),
                                 (request.request_id, 0, request.seeds[0]))
                self.assertEqual(len(saved["completion"]["token_ids"]), len(saved["completion"]["logprobs"]))
                self.assertTrue(saved["completion"]["token_ids"])
                self.assertEqual(calls, 2)
                self.assertTrue(engine.model.training)
                if bundle is not None:
                    self.assertFalse(bundle.route.vision_on)

    def test_observer_failure_stops_generation_and_restores_training_and_route(self):
        for kind in ("hf", "shared"):
            with self.subTest(kind=kind):
                bundle, engine, request = self.engine(kind)
                self.addCleanup(request.images[0].close)
                engine.model.train(True)
                rng = torch.get_rng_state().clone()

                def fail(*_):
                    if bundle is not None:
                        bundle.route.apply(True)
                    raise OSError("deliberate observer write failure")

                with patch.object(engine.model, "generate", wraps=engine.model.generate) as generated:
                    with self.assertRaisesRegex(OSError, "observer write failure"):
                        engine.sample([request], on_completion=fail)
                    self.assertEqual(generated.call_count, 1)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(engine.model.training)
                if bundle is not None:
                    self.assertFalse(bundle.route.vision_on)
                    self.assertFalse(any(parameter.requires_grad for parameter in bundle.params["V"]))

    def test_invalid_actual_completion_is_not_observed(self):
        for kind in ("hf", "shared"):
            with self.subTest(kind=kind):
                _, engine, request = self.engine(kind)
                self.addCleanup(request.images[0].close)
                engine.model.train(True)
                observed = []

                def invalid_logprobs(collector, sequences):
                    return [.5] * (sequences.shape[1] - collector.prompt_length)

                with patch.object(_SelectedTokenLogprobs, "finish", invalid_logprobs), patch.object(
                        engine.model, "generate", wraps=engine.model.generate) as generated:
                    with self.assertRaisesRegex(RuntimeError, "completion/log probability contract"):
                        engine.sample([request], on_completion=lambda *row: observed.append(row))
                    self.assertEqual(generated.call_count, 1)
                self.assertEqual(observed, [])
                self.assertTrue(engine.model.training)

    def test_observer_lists_do_not_alias_returned_completions(self):
        for kind in ("hf", "shared"):
            with self.subTest(kind=kind):
                _, engine, request = self.engine(kind)
                self.addCleanup(request.images[0].close)
                expected = engine.sample([request])
                seen = []

                def mutate(request_id, index, seed, completion):
                    seen.append((request_id, index, seed))
                    completion.token_ids[:] = [-999]
                    completion.logprobs.clear()

                self.assertEqual(engine.sample([request], on_completion=mutate), expected)
                self.assertEqual(seen, [(request.request_id, index, seed)
                                      for index, seed in enumerate(request.seeds)])


if __name__ == "__main__":
    unittest.main()
