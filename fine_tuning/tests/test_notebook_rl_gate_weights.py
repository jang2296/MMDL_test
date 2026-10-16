"""Authored tiny packed-worker CPU checks, never pretrained/native vLLM evidence."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from fine_tuning.notebook_rl import gate_weights as g
from fine_tuning.notebook_rl import policy
from fine_tuning.notebook_rl.engines import VLLMRolloutEngine


def register(model, name, value):
    parts = name.split(".")
    for part in parts[:-1]:
        if not hasattr(model, part):
            model.add_module(part, nn.Module())
        model = getattr(model, part)
    model.register_parameter(parts[-1], value if isinstance(value, nn.Parameter) else nn.Parameter(value))


def fixture():
    source = {}

    def add(name, shape):
        value = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
        source[name] = (value + len(source)).bfloat16()

    add("model.language_model.embed_tokens.weight", (8, 2))
    for name, rows in (("q", 4), ("k", 2), ("v", 2)):
        add(f"model.language_model.layers.0.self_attn.{name}_proj.weight", (rows, 2))
    for name in ("gate", "up"):
        add(f"model.language_model.layers.0.mlp.{name}_proj.weight", (3, 2))
    add("model.language_model.layers.0.self_attn.o_proj.weight", (2, 4))
    add("model.language_model.layers.0.mlp.down_proj.weight", (2, 3))
    add("model.language_model.norm.weight", (2,))
    add("model.visual.blocks.0.attn.qkv.weight", (6, 2))
    add("model.visual.blocks.0.attn.qkv.bias", (6,))
    for root in ("model.visual.merger", *[f"model.visual.deepstack_merger_list.{i}" for i in range(3)]):
        for suffix in ("norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias",
                       "linear_fc2.weight", "linear_fc2.bias"):
            add(root + "." + suffix, (2, 2) if suffix.startswith("linear_") and suffix.endswith("weight") else (2,))
    worker = nn.Module()
    # Deliberately authored independently from the production mapping helper.
    for name, tensor in source.items():
        if any(f"self_attn.{p}_proj" in name for p in "qkv") or any(f"mlp.{p}_proj" in name for p in ("gate", "up")):
            continue
        target = name.replace("model.language_model.", "language_model.model.").removeprefix("model.")
        register(worker, target, tensor.clone())
    register(worker, "language_model.model.layers.0.self_attn.qkv_proj.weight",
             torch.cat([source[f"model.language_model.layers.0.self_attn.{p}_proj.weight"] for p in "qkv"]))
    register(worker, "language_model.model.layers.0.mlp.gate_up_proj.weight",
             torch.cat([source[f"model.language_model.layers.0.mlp.{p}_proj.weight"] for p in ("gate", "up")]))
    register(worker, "language_model.lm_head.weight", worker.get_parameter("language_model.model.embed_tokens.weight"))
    return source, worker


def measure(source, worker):
    return g._measure_parity(iter(source.items()), {n: tuple(t.shape) for n, t in source.items()},
                             lambda callback: [callback(worker)])


class FixtureInproc:
    def __init__(self, scheduler):
        self.engine_core = SimpleNamespace(scheduler=scheduler, model_executor=FixtureExecutor())


class FixtureExecutor:
    def __init__(self):
        self.driver_worker = SimpleNamespace(device=torch.device("cuda:0"))


def runtime():
    engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
    engine.policy_version, engine._last_loaded_version = 3, 3
    engine._initialization_open, engine._push_complete = False, True
    scheduler = SimpleNamespace(has_unfinished_requests=lambda: False,
                                encoder_cache_manager=SimpleNamespace(cached={}))
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
        data_parallel_size=1, world_size=1), quant_config=None, lora_config=None,
        model_config=SimpleNamespace(quantization=None, dtype=torch.bfloat16))
    engine.llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=FixtureInproc(scheduler), vllm_config=cfg))
    return engine, scheduler, cfg


class WeightParityTests(unittest.TestCase):
    def test_complete_packed_slice_ledger_is_read_only_and_not_native(self):
        source, worker = fixture()
        before = {n: p.detach().clone() for n, p in worker.named_parameters()}
        report = measure(source, worker)
        self.assertTrue(report["all_tensors_match"])
        self.assertTrue(report["all_four_mergers"])
        self.assertEqual(report["merger_tensor_count"], 24)
        self.assertEqual(report["canonical_tensor_count"], len(source))
        self.assertEqual(report["worker_tensor_count"], len(source)-3)
        self.assertEqual(report["device"], "cpu")
        self.assertFalse(report["native_execution"])
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["gpu_gates_certified"])
        ledger = {r["canonical_name"]: r for r in report["ledger"]}
        for name, start, stop in (("q", 0, 4), ("k", 4, 6), ("v", 6, 8)):
            row = ledger[f"model.language_model.layers.0.self_attn.{name}_proj.weight"]
            self.assertEqual((row["start"], row["stop"]), (start, stop))
        self.assertEqual(ledger["model.language_model.layers.0.mlp.up_proj.weight"]["start"], 3)
        self.assertTrue(all(torch.equal(before[n], p) for n, p in worker.named_parameters()))
        json.dumps(report, allow_nan=False)

    def test_corrupted_packed_and_direct_tensors_rejected(self):
        for name in ("language_model.model.layers.0.self_attn.qkv_proj.weight",
                     "language_model.model.layers.0.mlp.gate_up_proj.weight", "visual.deepstack_merger_list.2.norm.weight"):
            with self.subTest(name=name):
                source, worker = fixture()
                with torch.no_grad():
                    worker.get_parameter(name).flatten()[-1].add_(1)
                with self.assertRaisesRegex(ValueError, "readback mismatch"):
                    measure(source, worker)

    def test_missing_extra_shape_dtype_nonfinite_and_untied_head_rejected(self):
        for failure in ("missing", "extra", "shape", "dtype", "nonfinite", "untied", "alias"):
            with self.subTest(failure=failure):
                source, worker = fixture()
                name = "visual.merger.norm.weight"
                parent = worker.get_submodule("visual.merger.norm")
                if failure == "missing":
                    del parent.weight
                elif failure == "extra":
                    register(worker, "unexpected.weight", torch.ones(2, dtype=torch.bfloat16))
                elif failure == "shape":
                    parent.weight = nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
                elif failure == "dtype":
                    parent.float()
                elif failure == "nonfinite":
                    with torch.no_grad():
                        worker.get_parameter(name)[0] = float("nan")
                elif failure == "untied":
                    worker.language_model.lm_head.weight = nn.Parameter(worker.language_model.lm_head.weight.clone())
                else:
                    register(worker, "unexpected_alias", worker.get_parameter(name))
                with self.assertRaises(ValueError):
                    measure(source, worker)

    def test_missing_duplicate_wrong_dtype_and_missing_merger_source_rejected(self):
        source, worker = fixture()
        shapes = {n: tuple(t.shape) for n, t in source.items()}
        pairs = list(source.items())
        for named in (pairs[:-1], pairs+pairs[:1], [(n, t.float()) for n, t in pairs]):
            with self.assertRaises(ValueError):
                g._measure_parity(iter(named), shapes, lambda callback: [callback(worker)])
        for missing in ("model.language_model.layers.0.self_attn.k_proj.weight", "model.visual.merger.norm.bias"):
            partial = {n: t for n, t in source.items() if n != missing}
            with self.assertRaises(ValueError):
                measure(partial, worker)

    def test_wrong_packing_order_and_multiple_workers_rejected(self):
        source, worker = fixture()
        target = worker.get_parameter("language_model.model.layers.0.self_attn.qkv_proj.weight")
        with torch.no_grad():
            target.copy_(torch.cat([source[f"model.language_model.layers.0.self_attn.{p}_proj.weight"] for p in "vkq"]))
        with self.assertRaisesRegex(ValueError, "readback mismatch"):
            measure(source, worker)
        source, worker = fixture()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            g._measure_parity(iter(source.items()), {n: tuple(t.shape) for n, t in source.items()},
                              lambda callback: [callback(worker), callback(worker)])

    def test_distinct_parameters_with_shared_storage_are_rejected(self):
        source, worker = fixture()
        name = "model.visual.deepstack_merger_list.0.norm.weight"
        other = "model.visual.deepstack_merger_list.1.norm.weight"
        source[other] = source[name].clone()
        left = worker.get_submodule("visual.deepstack_merger_list.0.norm")
        right = worker.get_submodule("visual.deepstack_merger_list.1.norm")
        right.weight = nn.Parameter(left.weight.detach())
        with self.assertRaisesRegex(ValueError, "shared worker tensor storage"):
            measure(source, worker)

    def test_runtime_guards_and_explicit_successful_zero_version_push(self):
        modules = SimpleNamespace(InprocClient=FixtureInproc, UniProcExecutor=FixtureExecutor)
        with patch.object(g, "version", return_value="0.11.0"), patch.object(g, "import_module", return_value=modules):
            engine, scheduler, cfg = runtime()
            self.assertEqual(g.require_idle_engine(engine)["policy_version"], 3)
            self.assertEqual(g.require_idle_engine(engine)["device"], "cuda:0")
            engine.policy_version = engine._last_loaded_version = 0
            self.assertEqual(g.require_idle_engine(engine)["last_loaded_version"], 0)
            engine._last_loaded_version = None
            with self.assertRaisesRegex(ValueError, "explicit completed"):
                g.require_idle_engine(engine)
            self.assertIsNone(g.require_idle_engine(engine, require_push=False)["last_loaded_version"])
            for failure in ("busy", "encoder", "partial", "marker", "tp", "pp", "quant", "dtype"):
                with self.subTest(failure=failure):
                    engine, scheduler, cfg = runtime()
                    if failure == "busy":
                        scheduler.has_unfinished_requests = lambda: True
                    elif failure == "encoder":
                        scheduler.encoder_cache_manager.cached = {"pending": {"request"}}
                    elif failure == "partial":
                        engine._push_complete = False
                    elif failure == "marker":
                        engine._last_loaded_version -= 1
                    elif failure in ("tp", "pp"):
                        setattr(cfg.parallel_config, "tensor_parallel_size" if failure == "tp" else "pipeline_parallel_size", 2)
                    elif failure == "quant":
                        cfg.quant_config = "forbidden"
                    else:
                        cfg.model_config.dtype = torch.float32
                    with self.assertRaises(ValueError):
                        g.require_idle_engine(engine)

    def test_wrong_engine_version_and_client_rejected_without_native_load(self):
        with self.assertRaises(ValueError):
            g.require_idle_engine(object())
        engine, _, _ = runtime()
        with patch.object(g, "version", return_value="0.12.0"), self.assertRaises(ValueError):
            g.require_idle_engine(engine)
        with patch.object(g, "version", return_value="0.11.0"), \
                patch.object(g, "import_module", return_value=SimpleNamespace(InprocClient=dict)), \
                self.assertRaisesRegex(ValueError, "in-process"):
            g.require_idle_engine(engine)

    def test_public_native_measurement_rejects_fixture_engine(self):
        with self.assertRaises(ValueError):
            g.measure_weight_parity(SimpleNamespace(), SimpleNamespace())

    def test_public_boundary_rechecks_base_and_runtime_even_after_failed_readback(self):
        # Control-flow doubles only; they do not exercise a real native model.
        for fault in (None, "readback", "device", "version"):
            shapes = {str(i): (1,) for i in range(713)}
            bundle = SimpleNamespace(inventory={"base_id": g.BASE_ID, "base_revision": g.BASE_REVISION},
                model=SimpleNamespace(config=object()), base_sha256={n: "authored" for n in shapes})
            engine = SimpleNamespace(model_config=object(), _weight_shapes=shapes,
                                     llm=SimpleNamespace(apply_model=object()))
            runtime = {"device": "cuda:0", "policy_version": 3}
            report = {"worker_tensor_count": 605, "device": "cuda:1" if fault == "device" else "cuda:0"}
            with self.subTest(fault=fault), \
                    patch.object(g, "require_idle_engine", side_effect=[runtime, runtime | {"policy_version": 4}
                        if fault == "version" else runtime]), \
                    patch.object(g, "_pinned_config"), \
                    patch.object(g, "_measure_parity", return_value=report,
                        side_effect=ValueError("readback") if fault == "readback" else None), \
                    patch.object(policy, "merged_named_tensors", return_value=iter(())), \
                    patch.object(policy, "assert_base_unchanged") as base_check:
                if fault:
                    with self.assertRaises(ValueError):
                        g.measure_weight_parity(bundle, engine)
                else:
                    result = g.measure_weight_parity(bundle, engine)
                    self.assertTrue(result["hf_base_unchanged"])
                    self.assertEqual(result["policy_version"], 3)
                self.assertEqual(base_check.call_count, 2)


if __name__ == "__main__":
    unittest.main()
