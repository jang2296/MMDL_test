"""CPU checks of the vLLM SINGLE stream with an authored engine; no GPU, model weights or real vLLM engine."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from vllm.logprobs import Logprob

from fine_tuning.error_note import llm_lora, rl
from fine_tuning.error_note.collect import WAIT, StreamStats
from fine_tuning.error_note.collect_v4 import validate_single_output
from fine_tuning.error_note.single_vllm import SingleVllmSampler, support_from_logprobs

PROMPT = [7, 8, 9]
EOS = 2


def row(values: dict[int, float]) -> dict:
    return {token: Logprob(value) for token, value in values.items()}


class FakeEngine:
    """add_request/step like vLLM V1: every active request finishes on the next step with an authored completion."""

    def __init__(self, completions):
        self.completions, self.active, self.params = completions, [], {}

    def add_request(self, request_id, prompt, params):
        self.active.append(request_id)
        self.params[request_id] = params

    def step(self):
        done, self.active = self.active, []
        return [SimpleNamespace(request_id=key, finished=True, prompt_token_ids=PROMPT,
                                outputs=[self.completions[key]]) for key in done]

    def abort_request(self, ids):
        self.active = [key for key in self.active if key not in ids]


def sampler(engine):
    owner = SimpleNamespace(prepare=lambda request: SimpleNamespace(token_ids=PROMPT),
                            _prompt=lambda prepared, ids: {"prompt_token_ids": list(ids)},
                            processor=None, model_config=None,
                            tokenizer=SimpleNamespace(decode=lambda ids, **_: " ".join(map(str, ids))),
                            llm=SimpleNamespace(llm_engine=engine))
    value = SingleVllmSampler.__new__(SingleVllmSampler)
    value.engine, value.max_num_seqs, value.sleep, value.eos_token_ids = owner, 16, False, [EOS]
    value.loaded_policy, value.asleep = "c" * 64, False
    value.image = {"min_pixels": 1003520, "max_pixels": 4014080}  # __init__ takes it from the plan's protocol
    return value


def request(pid):
    return {"request_id": f"{pid}:t1", "problem_id": pid, "block_id": 0, "stage": "t1", "seed": 5, "max_tokens": 8,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}],
            "images": [Image.new("RGB", (4, 4))]}


class SupportTests(unittest.TestCase):
    def test_finite_processed_logprobs_are_the_support_and_include_the_action(self):
        value, support = support_from_logprobs(5, row({5: -.2, 3: -1.7, 9: -math.inf, 1: -math.inf}))
        self.assertEqual((value, support), (-.2, [3, 5]))
        with self.assertRaisesRegex(RuntimeError, "outside"):
            support_from_logprobs(9, row({5: -.2, 9: -math.inf}))
        with self.assertRaisesRegex(RuntimeError, "outside"):
            support_from_logprobs(4, row({t: -3. for t in range(21)} | {4: -1.}))


class StreamTests(unittest.TestCase):
    def build(self, completions):
        engine = FakeEngine(completions)
        value = sampler(engine)
        args = SimpleNamespace(policy_sha256="c" * 64, protocol_sha256="a" * 64)
        item = {"prompt_token_ids": PROMPT, "input_tokens": 3, "input_sha256": "b" * 64}
        with patch("fine_tuning.error_note.single_vllm.SingleVllmSampler.sync", return_value="c" * 64), \
                patch("fine_tuning.notebook_rl.input_preparation.prepare_protocol_inputs",
                      return_value=(item, [])) as prepare:
            stream = value.stream_builder(None, args, StreamStats())
            self.assertTrue(stream.continuous)
            results = list(stream([request("p0"), WAIT, request("p1")], StreamStats(), None))
        # every request is prepared with the sampler's own image config, never the module constant
        self.assertTrue(prepare.call_args_list and all(call.args[-1] is value.image for call in prepare.call_args_list))
        return results, engine

    def test_completion_records_actions_support_and_exact_prompt_and_passes_ledger_validation(self):
        good = SimpleNamespace(token_ids=[4, EOS], finish_reason="stop",
                               logprobs=[row({4: -.1, 6: -2.5, 8: -math.inf}), row({EOS: -.05, 3: -3.})])
        outputs, engine = self.build({"p0:t1": good, "p1:t1": good})
        self.assertEqual([r["request_id"] for r, _ in outputs], ["p0:t1", "p1:t1"])
        params = engine.params["p0:t1"]
        self.assertEqual((params.temperature, params.top_p, params.top_k, params.presence_penalty, params.logprobs),
                         (.7, .8, 20, 1.5, 20))
        generation = outputs[0][1]["generation"]
        self.assertEqual(generation["support_token_ids"], [[4, 6], [EOS, 3]])
        self.assertEqual(generation["support_sha256"], [rl.digest([4, 6]), rl.digest([EOS, 3])])
        self.assertEqual(generation["sampling_logprobs"], [-.1, -.05])
        self.assertEqual(generation["completion_token_ids"], [4, EOS])
        validate_single_output(request("p0"), outputs[0][1], policy_sha256="c" * 64, protocol_sha256="a" * 64)

    def test_stop_without_eos_or_missing_logprobs_fails_closed(self):
        bad = {"p0:t1": SimpleNamespace(token_ids=[4], finish_reason="stop", logprobs=[row({4: -.1})]),
               "p1:t1": SimpleNamespace(token_ids=[4], finish_reason="length", logprobs=None)}
        with self.assertRaisesRegex(RuntimeError, "EOS|log-probability"):
            self.build(bad)


class SleepTests(unittest.TestCase):
    def test_release_frees_engine_and_sync_wakes_and_pushes_complete_weights(self):
        calls = []
        llm = SimpleNamespace(sleep=lambda level: calls.append(("sleep", level)), wake_up=lambda: calls.append(("wake",)))
        engine = SimpleNamespace(llm=llm, policy_version=3,
                                 _load_weights=lambda named, version: calls.append(("push", version, len(list(named)))))
        value = SingleVllmSampler.__new__(SingleVllmSampler)
        value.engine, value.sleep, value.asleep, value.loaded_policy = engine, True, False, "x" * 64
        with patch.object(torch.cuda, "empty_cache"):
            value.release()
            self.assertTrue(value.asleep)
            self.assertIsNone(value.loaded_policy)
            with patch("fine_tuning.notebook_rl.policy.merged_named_tensors", return_value=iter([("w", torch.ones(1))])), \
                    patch("fine_tuning.error_note.rl.policy_sha256", return_value="y" * 64):
                self.assertEqual(value.sync(object()), "y" * 64)
                value.sync(object())  # unchanged policy: no second push
        self.assertEqual(calls, [("sleep", 2), ("wake",), ("push", 4, 1)])


def tiny_bundle():
    """Real tiny Qwen3-VL under PEFT with non-zero LoRA B and perturbed Merger masters."""
    from fine_tuning.notebook_rl.policy import configure_policy
    from fine_tuning.notebook_rl.schema import load_protocol
    from fine_tuning.tests.test_training import tiny_model

    protocol = load_protocol(Path(__file__).parents[1] / "notebook_rl/protocol_v1.yaml")
    bundle = configure_policy(tiny_model(), None, "MOD-RV", protocol, enforce_counts=False)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for name, parameter in bundle.model.named_parameters():
            if ".lora_B." in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return bundle


def perturb(bundle):
    with torch.no_grad():
        next(p for n, p in bundle.model.named_parameters() if ".lora_B." in n).add_(1.0)


class LoraFileTests(unittest.TestCase):
    def test_llm_adapter_loads_in_vllm_exactly(self):
        from vllm.config import LoRAConfig
        from vllm.lora.models import LoRAModel
        from vllm.lora.peft_helper import PEFTHelper
        from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration as Q

        bundle = tiny_bundle()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter"
            llm_lora.write_llm_adapter(bundle, path)
            helper = PEFTHelper.from_local_dir(str(path), None)
            helper.validate_legal(LoRAConfig(max_lora_rank=16, max_loras=1))
            model = LoRAModel.from_local_checkpoint(str(path), list(llm_lora.LLM_LEAVES), helper, lora_model_id=1,
                                                    device="cpu", dtype=torch.bfloat16, embedding_modules={},
                                                    embedding_padding_modules=[], weights_mapper=Q.hf_to_vllm_mapper)
        modules = [f"self_attn.{x}_proj" for x in "qkvo"] + [f"mlp.{x}_proj" for x in ("gate", "up", "down")]
        expected = {f"language_model.model.layers.{i}.{m}" for i in range(3) for m in modules}
        self.assertEqual(set(model.loras), expected)
        self.assertFalse(any("visual" in name for name in model.loras))
        base = bundle.model.get_base_model()
        for name, weights in model.loras.items():
            layer = base.get_submodule("model.language_model." + name.removeprefix("language_model.model."))
            a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
            self.assertTrue(torch.equal(weights.lora_a, a.bfloat16()), name)
            self.assertTrue(torch.equal(weights.lora_b, b.bfloat16() * 2), name)
            self.assertTrue(bool(b.abs().sum() > 0))
        # Vision LoRA stays out of the file and goes to the hooks with the same exact BF16 scaling.
        vision = llm_lora.vision_lora(bundle)
        self.assertEqual(sorted(vision), sorted(f"visual.blocks.{i}.attn.{x}" for i in range(3) for x in ("qkv", "proj")))
        layer = base.get_submodule("model.visual.blocks.0.attn.qkv")
        self.assertTrue(torch.equal(vision["visual.blocks.0.attn.qkv"][1], layer.lora_B["default"].weight.bfloat16() * 2))

    def test_mm_scope_patch_filters_vision_and_mergers(self):
        from vllm.lora.models import LoRAModelManager
        from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration as Q

        llm_lora.scope_qwen3vl_lora()
        patched = Q.get_mm_mapping
        llm_lora.scope_qwen3vl_lora()  # idempotent
        self.assertIs(Q.get_mm_mapping, patched)
        manager = SimpleNamespace(supports_mm=True, model=Q.__new__(Q))
        filtered = LoRAModelManager._filter_unsupported_mm_module
        for name in ("visual.blocks.0.attn.proj", "visual.blocks.0.mlp.linear_fc1", "visual.merger.linear_fc2",
                     "visual.deepstack_merger_list.2.linear_fc1"):
            self.assertTrue(filtered(manager, name), name)
        self.assertFalse(filtered(manager, "language_model.model.layers.0.self_attn.qkv_proj"))
        with patch.object(Q, "get_mm_mapping", patched.mmdl_original):  # upstream 0.11.0 prefixes never match
            self.assertFalse(filtered(manager, "visual.blocks.0.attn.proj"))


class FakeLoraCore:
    def __init__(self, calls):
        self.calls, self.loaded = calls, set()

    def remove_lora(self, lora_id):
        self.calls.append(("remove", lora_id))
        self.loaded.discard(lora_id)
        return True

    def add_lora(self, request):
        self.calls.append(("add", request.lora_int_id))
        self.loaded.add(request.lora_int_id)
        return True

    def list_loras(self):
        return set(self.loaded)


class LoraSyncTests(unittest.TestCase):
    def test_lora_sync_order_and_fresh_ids(self):
        calls, bundle = [], tiny_bundle()
        core = FakeLoraCore(calls)
        llm = SimpleNamespace(sleep=lambda level: calls.append(("sleep", level)), wake_up=lambda: calls.append(("wake",)),
                              llm_engine=core)
        hooks = SimpleNamespace(load=lambda weights: calls.append(("hooks", len(weights))))

        def push(engine, named, version):
            names = [name for name, _ in named]  # the full canonical inventory is consumed
            self.assertEqual(set(names), set(bundle.base_sha256))
            calls.append(("push", version))
            engine.policy_version = version

        value = SingleVllmSampler.__new__(SingleVllmSampler)
        value.engine = SimpleNamespace(llm=llm, policy_version=0)
        value.lora, value.sleep, value.asleep, value.loaded_policy = True, True, False, None
        value.vision_hooks, value._lora_id, value._preflight_done = hooks, 0, False
        verified = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.cuda, "empty_cache"), \
                patch.object(llm_lora, "push_mergers", push), \
                patch.object(SingleVllmSampler, "verify_lora", lambda self, b: verified.append(self.lora_request)):
            value.lora_dir = Path(tmp)
            value.release()
            first = value.sync(bundle)
            self.assertEqual(value.sync(bundle), first)  # unchanged policy: no-op
            perturb(bundle)
            value.sync(bundle)
            value.release()
            value.sync(bundle)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["3"])  # replaced adapters are deleted
        self.assertEqual(calls, [("sleep", 1), ("wake",), ("hooks", 6), ("push", 1), ("add", 1),
                                 ("hooks", 6), ("push", 2), ("remove", 1), ("add", 2),
                                 ("sleep", 1), ("wake",), ("hooks", 6), ("push", 3), ("remove", 2), ("add", 3)])
        self.assertEqual([r.lora_int_id for r in verified], [1])  # preflight once, before any generation
        self.assertEqual((value.lora_request.lora_int_id, core.list_loras()), (3, {3}))


class LoraFakeEngine(FakeEngine):
    def __init__(self, completions, recorded=None):
        super().__init__(completions)
        self.recorded, self.requests = recorded, {}
        self.engine_core = SimpleNamespace(engine_core=SimpleNamespace(scheduler=SimpleNamespace(requests=self.requests)))

    def add_request(self, request_id, prompt, params, lora_request=None):
        super().add_request(request_id, prompt, params)
        self.requests[request_id] = SimpleNamespace(lora_request=self.recorded or lora_request)


class LoraStreamTests(unittest.TestCase):
    def build(self, engine):
        from vllm.lora.request import LoRARequest

        value = sampler(engine)
        value.lora, value.lora_request = True, LoRARequest("policy-c", 7, "/unused")
        args = SimpleNamespace(policy_sha256="c" * 64, protocol_sha256="a" * 64)
        item = {"prompt_token_ids": PROMPT, "input_tokens": 3, "input_sha256": "b" * 64}
        with patch("fine_tuning.error_note.single_vllm.SingleVllmSampler.sync", return_value="c" * 64), \
                patch("fine_tuning.notebook_rl.input_preparation.prepare_protocol_inputs", return_value=(item, [])):
            stats = StreamStats()
            stream = value.stream_builder(None, args, stats)
            return list(stream([request("p0"), request("p1")], StreamStats(), None)), stats

    def test_lora_stream_attaches_request(self):
        from vllm.lora.request import LoRARequest

        good = SimpleNamespace(token_ids=[4, EOS], finish_reason="stop", logprobs=[row({4: -.1}), row({EOS: -.05})])
        engine = LoraFakeEngine({"p0:t1": good, "p1:t1": good})
        outputs, stats = self.build(engine)
        self.assertEqual({r.lora_request.lora_int_id for r in engine.requests.values()}, {7})
        self.assertEqual({o["generation"]["engine"] for _, o in outputs}, {llm_lora.ENGINE_LORA})
        self.assertEqual(stats.receipt_extra["single_sampler"], llm_lora.ENGINE_LORA)
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            self.build(LoraFakeEngine({"p0:t1": good}, recorded=LoRARequest("other", 6, "/x")))


class MergerManager:
    def __init__(self, cache_size=1):
        self.cache_size, self.cached, self.freeable, self.freed = cache_size, {}, {}, []


class PushMergerTests(unittest.TestCase):
    SHAPES = {"model.visual.merger.linear_fc1.weight": (2, 2), "model.visual.deepstack_merger_list.0.linear_fc1.weight": (2, 2),
              "model.language_model.layers.0.self_attn.q_proj.weight": (2, 2)}

    def engine(self, vllm_names, loaded):
        from vllm.v1.engine.core_client import InprocClient

        scheduler = SimpleNamespace(has_unfinished_requests=lambda: False, encoder_cache_manager=MergerManager(),
                                    reset_prefix_cache=lambda: True)
        client = InprocClient.__new__(InprocClient)
        client.engine_core = SimpleNamespace(scheduler=scheduler)

        def load_weights(weights):
            loaded.extend(name for name, _ in weights)
            return {name.removeprefix("model.") for name in loaded}

        model = SimpleNamespace(named_parameters=lambda: [(n, None) for n in vllm_names], load_weights=load_weights)
        llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=client), apply_model=lambda fn: [fn(model)],
                              collective_rpc=lambda fn: [True])
        return SimpleNamespace(llm=llm, _weight_shapes=self.SHAPES, policy_version=0)

    def named(self, **changes):
        values = {name: torch.zeros(shape, dtype=torch.bfloat16) for name, shape in self.SHAPES.items()} | changes
        return [(name, value) for name, value in values.items() if value is not None]

    def test_push_mergers_rejects_incomplete_or_foreign(self):
        vllm_names = ["visual.merger.linear_fc1.weight", "visual.deepstack_merger_list.0.linear_fc1.weight",
                      "language_model.model.layers.0.self_attn.qkv_proj.base_layer.weight"]
        loaded: list[str] = []
        engine = self.engine(vllm_names, loaded)
        llm_lora.push_mergers(engine, self.named(), 1)
        self.assertEqual(sorted(loaded), sorted(n for n in self.SHAPES if "visual" in n))  # Mergers only
        self.assertEqual((engine.policy_version, engine._last_loaded_version, engine._push_complete), (1, 1, True))
        merger = "model.visual.merger.linear_fc1.weight"
        bad_inputs = [self.named(**{merger: None}), self.named(**{merger: torch.zeros(2, 2)}),
                      self.named() + [("model.visual.blocks.0.attn.qkv.lora_A", torch.zeros(1))]]
        for named in bad_inputs:  # missing Merger, FP32 Merger, foreign name
            with self.assertRaises(ValueError):
                llm_lora.push_mergers(self.engine(vllm_names, []), named, 1)
        with self.assertRaisesRegex(RuntimeError, "Merger loading"):  # vLLM holds a Merger tensor that was not loaded
            llm_lora.push_mergers(self.engine(vllm_names + ["visual.merger.norm.weight"], []), self.named(), 1)


class SlopeTests(unittest.TestCase):
    def test_slope_passes_exact_application_and_rejects_scale_or_missing_vision(self):
        g = torch.Generator().manual_seed(3)
        h_off = (-5 + torch.randn(64, generator=g)).tolist()
        d_llm, d_vis = torch.randn(64, generator=g) * .3, torch.randn(64, generator=g) * .3
        h_llm = (torch.tensor(h_off) + d_llm).tolist()
        h_on = (torch.tensor(h_llm) + d_vis).tolist()
        noise = (torch.randn(64, generator=g) * .005).tolist()
        shift = lambda x: [a + b for a, b in zip(x, noise)]  # noqa: E731
        good = llm_lora.slope_check(shift(h_on), shift(h_llm), shift(h_off), h_on, h_llm, h_off, cap=.04)
        self.assertTrue(good["pass"])
        self.assertTrue(all(good[k]["pass"] for k in ("total", "llm", "vision")))
        half = (torch.tensor(h_off) + d_llm / 2).tolist()  # alpha written as 16
        bad = llm_lora.slope_check(shift((torch.tensor(half) + d_vis).tolist()), shift(half), shift(h_off),
                                   h_on, h_llm, h_off, cap=.04)
        self.assertFalse(bad["pass"])
        self.assertFalse(bad["llm"]["pass"])
        no_hooks = llm_lora.slope_check(shift(h_llm), shift(h_llm), shift(h_off), h_on, h_llm, h_off, cap=.04)
        self.assertFalse(no_hooks["vision"]["pass"])
        flat = llm_lora.slope_check(h_off, h_off, h_off, h_off, h_off, h_off, cap=.04)
        self.assertTrue(flat["pass"] and flat["llm"]["pass"] is None)  # no effect: only the cap applies


if __name__ == "__main__":
    unittest.main()
