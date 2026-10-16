"""Authored CPU rollouts plus actual tiny RL updates; never native 4B/GPU evidence."""
import hashlib
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.engines import Completion, Prepared, RolloutRequest, RolloutResult
from fine_tuning.notebook_rl.gate_throughput import _Memory, _measure_step, _native_budgets, measure_step
from fine_tuning.notebook_rl.policy import completion_logprobs
from fine_tuning.notebook_rl.prompts import load_prompts
from fine_tuning.notebook_rl.schema import VerifierResult
from fine_tuning.notebook_rl.trainer import SyncProbe
from fine_tuning.notebook_rl.updates import _stage_time
from fine_tuning.tests.test_notebook_rl_policy import bundle, config
from fine_tuning.tests.test_notebook_rl_schema import records
from fine_tuning.tests.test_notebook_rl_trainer import SyncFixture


class Tokens:
    unk_token_id = -1

    def encode(self, text, **kwargs):
        return [8, 9]  # Authored tiny vocabulary, not production chat template evidence.

    def convert_tokens_to_ids(self, token):
        return 2


class TinySampler(SyncFixture):
    """Scripted final answers; actual per-action tiny-model log probabilities."""
    def __init__(self, model, *, all_correct=False, fail_push=False, mismatch=False):
        super().__init__(fail_push=fail_push)
        self.model, self.tokenizer = model, Tokens()
        self.all_correct, self.mismatch = all_correct, mismatch
        self.calls = []

    def prepare(self, request):
        count = len(request.images)
        ids = request.prompt_token_ids or [v for _ in request.images for v in [4, 3, 3, 3, 3, 5]]+[9]
        return Prepared(list(ids), {
            "pixel_values": torch.linspace(-1, 1, count*16*24).reshape(-1, 24),
            "image_grid_thw": torch.tensor([[1, 4, 4]]*count)}, request.images, "cpu-fixture")

    def model_inputs(self, request, completion, device):
        value = self.prepare(request)
        ids = torch.tensor([value.token_ids+completion], device=device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                **{k: v.to(device) for k, v in value.pixels.items()}}

    def sample(self, requests):
        self.calls.extend(requests)
        result = []
        for request in requests:
            samples = []
            for i in range(request.n):
                completion = [10+i, 11, 2]
                with torch.no_grad():
                    lp = completion_logprobs(self.model,
                        self.model_inputs(request, completion, torch.device("cpu")), 3)[0].tolist()
                if self.mismatch and ":T3:" in request.request_id:
                    lp = [x-.2 for x in lp]
                correct = self.all_correct or i == 0
                samples.append(Completion(completion, lp, "stop", "Final answer: " + ("7" if correct else "8")))
            result.append(RolloutResult(request.request_id, self.prepare(request).token_ids, samples))
        return result


class TinyVerifier:
    def grade(self, pid, text, finish, mode):
        correct = text == "Final answer: 7"
        return VerifierResult("CORRECT" if correct else "INCORRECT", "7" if correct else "8",
                              float(correct), None, "authored-cpu-fixture")

    def cause(self, pid, result):
        return None

    def route(self, pid, results, rule):
        return "VISION_OFF", {"fixture_only": True}


class ThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, *, images=2, **sampler_options):
        b, protocol = bundle("MOD-FML"), config()
        protocol["batch"] = {"problems_per_step": 1, "max_attempted_problems_per_step": 3}
        protocol["rollout"]["n"] = {1: 2, 2: 2, 3: 2}
        protocol["rollout"]["branch"] = {1: 1, 2: 1}
        protocol["rollout"]["max_new_tokens"] = {1: 3, 2: 3, 3: 3}
        actor = replace(records()[0], problem_id="stress", mode="plain",
                        image_paths=tuple(f"images/{i}.png" for i in range(images)),
                        image_sha256=tuple(hashlib.sha256(str(i).encode()).hexdigest() for i in range(images)))
        opt, sched = make_optimizer(b, protocol)
        engine = TinySampler(b, **sampler_options)
        created = []
        def loader(actor):
            loaded = [Image.new("RGB", (4, 4), (i, 20, 30)) for i in range(len(actor.image_paths))]
            created.extend(loaded)
            return loaded
        probes = [SyncProbe(RolloutRequest(str(n), None, [{}], loader(replace(actor,
            image_paths=tuple(f"images/{i}.png" for i in range(n)))), 1, 3, [3407]), [10, 11, 2]) for n in (1, 2)]
        self.addCleanup(lambda: [im.close() for im in created])
        journal = []
        kwargs = dict(bundle=b, optimizer=opt, scheduler=sched, engine=engine, preparation=engine,
            actors={"stress": actor}, structures={"stress": "b"*64}, loader=loader,
            verifier=TinyVerifier(), prompts=load_prompts(Path(__file__).parents[1]/"notebook_rl/prompts_v1.yaml"),
            protocol=protocol, probes=probes, algorithm="ALG-NB3", step=0, seed=3407,
            should_stop=lambda: False, record=journal.append)
        return kwargs, journal, created

    def test_actual_tiny_update_all_stages_but_no_native_memory_claim(self):
        kwargs, journal, images = self.fixture()
        with patch("torch.nn.functional.cross_entropy", side_effect=AssertionError("CE forbidden")):
            result = _measure_step(**kwargs)
        self.assertEqual(result["status"], "STEP_MEASURED_NOT_GPU_APPROVAL")
        self.assertTrue(result["longest_sample_step_complete"])
        self.assertEqual(result["witness"]["turn"], 3)
        self.assertEqual(result["rollout_output_tokens"], 18)
        self.assertEqual(result["rollout_calls"], 3)
        self.assertEqual(kwargs["scheduler"].last_epoch, 1)
        self.assertEqual(kwargs["engine"].pushes, [1])
        self.assertTrue(result["G4"]["merger_changed"])
        self.assertTrue(all(result["seconds"][key] > 0 for key in ("old", "ref", "train", "push", "rollout")))
        self.assertGreater(result["step_seconds"], sum(result["seconds"].values()))
        self.assertIsNone(result["peak_memory_bytes"])
        self.assertFalse(result["native_execution"])
        self.assertFalse(result["memory_observed"])
        self.assertFalse(result["gpu_gates_certified"])
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["engine_sync_required"])
        self.assertEqual(journal[-1]["kind"], "G11_RESULT")
        self.assertEqual(len([r for r in journal if r["kind"] == "G11_SAMPLER_RETURN"]), 3)
        # Probe images belong to caller; cohort and collected images are closed.
        for image in images[3:]:
            with self.assertRaises(ValueError):
                image.tobytes()

    def test_single_image_cannot_pass_stress_or_start_rollouts(self):
        kwargs, journal, _ = self.fixture(images=1)
        result = _measure_step(**kwargs)
        self.assertEqual(result["incomplete_reason"], "MAXIMUM_IMAGE_COUNT_NOT_IN_COHORT")
        self.assertFalse(result["longest_sample_step_complete"])
        self.assertFalse(kwargs["engine"].calls)
        self.assertFalse(kwargs["optimizer"].state)
        self.assertEqual(journal[-1]["status"], "INCOMPLETE")

    def test_no_retry_no_learning_signal_is_incomplete_not_an_invented_retry(self):
        kwargs, journal, _ = self.fixture(all_correct=True)
        result = _measure_step(**kwargs)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["collection_status"], "EXHAUSTED")
        self.assertEqual(len(kwargs["engine"].calls), 1)
        self.assertFalse(kwargs["optimizer"].state)
        self.assertTrue(any(r["kind"] == "G11_TREE" for r in journal))

    def test_missing_third_attempt_fails_before_update(self):
        kwargs, _, _ = self.fixture()
        original = kwargs["engine"].sample
        def early_success(requests):
            kwargs["engine"].all_correct = ":T2:" in requests[0].request_id
            return original(requests)
        kwargs["engine"].sample = early_success
        result = _measure_step(**kwargs)
        self.assertEqual(result["incomplete_reason"], "LONGEST_INPUT_ACTUAL_MAXIMUM_RETRY_NOT_OBSERVED")
        self.assertFalse(kwargs["optimizer"].state)

    def test_mismatch_masked_longest_action_is_not_memory_pass(self):
        kwargs, _, _ = self.fixture(mismatch=True)
        result = _measure_step(**kwargs)
        self.assertEqual(result["incomplete_reason"], "LONGEST_ACTION_MISMATCH_MASKED_NOT_TRAINED")
        self.assertFalse(result["longest_sample_step_complete"])
        self.assertTrue(result["weights_updated"])
        self.assertFalse(result["engine_sync_required"])
        self.assertEqual(kwargs["engine"].pushes, [1])

    def test_failed_sync_retains_actual_update_and_unsynchronized_state(self):
        kwargs, journal, _ = self.fixture(fail_push=True)
        with self.assertRaisesRegex(RuntimeError, "fixture push failed"):
            _measure_step(**kwargs)
        result = journal[-1]
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(result["weights_updated"])
        self.assertTrue(result["engine_sync_required"])
        self.assertFalse(result["longest_sample_step_complete"])
        self.assertEqual(kwargs["scheduler"].last_epoch, 1)

    def test_input_changes_after_ranking_fail_before_update(self):
        kwargs, journal, _ = self.fixture()
        loader, count = kwargs["loader"], [0]
        def changed(actor):
            count[0] += 1
            images = loader(actor)
            if count[0] == 2:
                images[0].putpixel((0, 0), (200, 0, 0))
            return images
        kwargs["loader"] = changed
        with self.assertRaisesRegex(ValueError, "input changed"):
            _measure_step(**kwargs)
        self.assertFalse(kwargs["optimizer"].state)
        self.assertEqual(journal[-1]["status"], "FAILED")

    def test_budget_stop_before_preparation_makes_no_calls(self):
        kwargs, _, _ = self.fixture()
        kwargs["should_stop"] = lambda: True
        result = _measure_step(**kwargs)
        self.assertIn("BUDGET_STOP", result["incomplete_reason"])
        self.assertFalse(kwargs["engine"].calls)

    def test_grading_image_mutation_first_or_last_turn_prevents_update(self):
        for turn in (1, 3):
            with self.subTest(turn=turn):
                kwargs, journal, _ = self.fixture()
                verifier, engine = kwargs["verifier"], kwargs["engine"]
                original = verifier.grade
                def grade(*args):
                    if f":T{turn}:" in engine.calls[-1].request_id:
                        engine.calls[-1].images[0].putpixel((0, 0), (255, 0, 0))
                    return original(*args)
                verifier.grade = grade
                with self.assertRaisesRegex(ValueError, "input changed"):
                    _measure_step(**kwargs)
                self.assertFalse(kwargs["optimizer"].state)
                self.assertEqual(journal[-1]["status"], "FAILED")

    def test_processor_tensor_drift_without_image_mutation_prevents_update(self):
        for stage in ("retry", "trainer"):
            with self.subTest(stage=stage):
                kwargs, journal, _ = self.fixture()
                engine = kwargs["engine"]
                target = engine if stage == "retry" else TinySampler(kwargs["bundle"])
                if stage == "trainer":
                    kwargs["preparation"] = target
                original = target.prepare
                def prepare(request):
                    value = original(request)
                    if request.prompt_token_ids is not None:
                        value.pixels["pixel_values"][0, 0] += 1
                    return value
                target.prepare = prepare
                with self.assertRaisesRegex(ValueError, "input changed"):
                    _measure_step(**kwargs)
                self.assertFalse(kwargs["optimizer"].state)
                self.assertEqual(journal[-1]["status"], "FAILED")

    def test_native_budget_projection_rejects_silent_budget_and_cache_changes(self):
        # Configuration boundary test, not an allocated native vLLM engine.
        protocol = config()
        fixed, image = protocol["engine"]["vllm"], protocol["image"]
        mm = SimpleNamespace(mm_processor_cache_gb=0, limit_per_prompt={"image": 2, "video": 0},
            mm_processor_kwargs={"size": {"shortest_edge": image["min_pixels"],
                "longest_edge": image["max_pixels"]}, "do_resize": False})
        model = SimpleNamespace(multimodal_config=mm, max_model_len=protocol["context_limit"],
            enforce_eager=False, generation_config="vllm", override_generation_config={})
        cache = SimpleNamespace(kv_cache_memory_bytes=fixed["kv_cache_memory_bytes"], enable_prefix_caching=False)
        scheduler = SimpleNamespace(max_num_seqs=fixed["max_num_seqs"], max_model_len=protocol["context_limit"])
        cfg = SimpleNamespace(model_config=model, cache_config=cache, scheduler_config=scheduler)
        engine = SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=cfg)))
        self.assertEqual(_native_budgets(engine, protocol)["kv_cache_memory_bytes"], 10737418240)
        for owner, key, value in ((cache, "kv_cache_memory_bytes", 1024), (scheduler, "max_num_seqs", 1),
                (scheduler, "max_model_len", 8192), (model, "enforce_eager", True),
                (cache, "enable_prefix_caching", True), (mm, "mm_processor_cache_gb", 4),
                (mm, "limit_per_prompt", {"image": 1}), (model, "generation_config", "auto"),
                (model, "override_generation_config", {"temperature": .5})):
            with self.subTest(key=key), patch.object(owner, key, value), \
                    self.assertRaisesRegex(ValueError, "native cache/scheduler/input"):
                _native_budgets(engine, protocol)

    def test_public_rejects_cpu_before_any_update(self):
        kwargs, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "CUDA"):
            measure_step(**kwargs)
        self.assertFalse(kwargs["optimizer"].state)

    def test_stage_clock_synchronizes_only_explicit_cuda_device(self):
        with patch("torch.cuda.synchronize") as sync:
            self.assertGreater(_stage_time(torch.device("cpu")), 0)
            sync.assert_not_called()
            self.assertGreater(_stage_time(torch.device("cuda:2")), 0)
            sync.assert_called_once_with(torch.device("cuda:2"))

    def test_device_samples_and_allocator_peaks_are_not_added_or_called_native(self):
        # API-boundary regression only, no GPU allocation or physical measurement.
        memory = _Memory(torch.device("cuda:2"))
        current = [(60, 150)]
        with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                patch("torch.cuda.mem_get_info", side_effect=lambda device: current[0]), \
                patch("torch.cuda.max_memory_allocated", return_value=50), \
                patch("torch.cuda.max_memory_reserved", return_value=70):
            memory.start()
            current[0] = (80, 150)
            result = memory.finish()
        self.assertGreaterEqual(result["device_samples"], 2)
        self.assertEqual(result["device_sampled_peak_bytes"], 90)
        self.assertEqual(result["torch_peak_allocated_bytes"], 50)
        self.assertEqual(result["torch_peak_reserved_bytes"], 70)
        self.assertEqual(result["device_total_bytes"], 150)
        self.assertFalse(memory.thread.is_alive())

    def test_memory_observer_error_cannot_be_reported_as_success(self):
        memory = _Memory(torch.device("cuda:0"))
        memory.error = "RuntimeError"
        with patch("torch.cuda.mem_get_info", return_value=(10, 100)), \
                self.assertRaisesRegex(ValueError, "observer failed"):
            memory.finish()
        self.assertTrue(memory.stop.is_set())


if __name__ == "__main__":
    unittest.main()
