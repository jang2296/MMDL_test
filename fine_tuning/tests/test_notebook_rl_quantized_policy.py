"""CPU validation fixtures only: no actual bitsandbytes kernels or pretrained 4B run."""
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from fine_tuning.modules import BASE_REVISION, tensor_sha256
from fine_tuning.notebook_rl import quantized_policy as qp
from fine_tuning.notebook_rl.policy import PolicyBundle, VisionRoute, configure_policy
from fine_tuning.tests.test_notebook_rl_policy import config


class QuantStateFixture:
    def __init__(self):
        self.quant_type, self.dtype, self.blocksize = "nf4", torch.bfloat16, 64
        self.nested, self.shape = True, (4, 4)
        self.absmax = torch.ones(1, dtype=torch.uint8)
        self.code = torch.linspace(-1, 1, 16)
        self.nested_absmax, self.nested_code = torch.ones(1), torch.linspace(-1, 1, 256)

    def as_dict(self, packed=False):
        assert packed
        wire = json.dumps({"shape": self.shape, "dtype": str(self.dtype), "blocksize": self.blocksize})
        return {"absmax": self.absmax, "quant_map": self.code,
                "nested_absmax": self.nested_absmax, "nested_quant_map": self.nested_code,
                "quant_state.bitsandbytes__nf4": torch.tensor(list(wire.encode()), dtype=torch.uint8)}


class Params4bitFixture(nn.Parameter):
    def __new__(cls):
        result = super().__new__(cls, torch.zeros(8, 1, dtype=torch.uint8), requires_grad=False)
        result.quant_state = QuantStateFixture()
        result.quant_type, result.compress_statistics = "nf4", True
        result.quant_storage, result.bnb_quantized = torch.uint8, True
        return result


class Linear4bitFixture(nn.Linear):
    def __init__(self):
        super().__init__(4, 4, bias=False)
        self.weight = Params4bitFixture()
        self.quant_state, self.compute_dtype = self.weight.quant_state, torch.bfloat16

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for name, value in self.quant_state.as_dict(packed=True).items():
            destination[prefix + "weight." + name] = value


class BaseFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ConfigFixture(model_type="qwen3_vl")
        self.generation_config = ConfigFixture()
        self.is_loaded_in_4bit = True
        self.model = nn.Module()
        self.model.language = nn.ModuleList([Linear4bitFixture() for _ in range(252)])
        self.model.visual = nn.Module()
        self.model.visual.layers = nn.ModuleList([Linear4bitFixture() for _ in range(48)])
        self.model.visual.merger = nn.Sequential(nn.Linear(4, 4, dtype=torch.bfloat16))
        self.model.visual.deepstack_merger_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4, dtype=torch.bfloat16)) for _ in range(3)])
        self.lm_head = nn.Linear(4, 4, dtype=torch.bfloat16)

    def get_base_model(self):
        return self

    def save_pretrained(self, directory, *, state_dict, **kwargs):
        from safetensors.torch import save_file

        save_file({name: value.contiguous() for name, value in state_dict.items()},
                  str(Path(directory) / "model.safetensors"))
        (Path(directory) / "config.json").write_text('{"model_type":"cpu_fixture"}\n')


class ProcessorFixture:
    def save_pretrained(self, directory):
        (Path(directory) / "preprocessor_config.json").write_text('{"cpu_fixture":true}\n')


class ConfigFixture(SimpleNamespace):
    def save_pretrained(self, directory):
        (Path(directory) / "config.json").write_text('{"model_type":"cpu_fixture"}\n')


def inventory_fixture():
    return {"targets": {"L": [f"model.language.{i}" for i in range(252)],
                        "V": [f"model.visual.layers.{i}" for i in range(48)]},
            "mergers": list(qp.MERGERS)}


class QuantizedPolicyTests(unittest.TestCase):
    def setUp(self):
        bnb, bnb_nn = types.ModuleType("bitsandbytes"), types.ModuleType("bitsandbytes.nn")
        bnb_nn.Linear4bit, bnb_nn.Params4bit = Linear4bitFixture, Params4bitFixture
        bnb.nn = bnb_nn
        self.bnb = patch.dict(sys.modules, {"bitsandbytes": bnb, "bitsandbytes.nn": bnb_nn})
        self.bnb.start()
        self.addCleanup(self.bnb.stop)

    def test_contract_is_explicit_and_returns_fresh_collections(self):
        first = qp.quantization_contract()
        first["excluded_modules"].clear()
        self.assertEqual(qp.quantization_contract()["excluded_modules"], [*qp.MERGERS, "lm_head"])
        protocol = config()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / BASE_REVISION
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "separate NF4"):
                qp._validate_protocol(source, "MOD-RV", protocol, 0)
            protocol["quantization"] = qp.quantization_contract()
            qp._validate_protocol(source, "MOD-RV", protocol, 0)
            for arm, device in (("MOD-FML", 0), ("MOD-RV", True), ("MOD-RV", -1)):
                with self.subTest(arm=arm, device=device), self.assertRaises(ValueError):
                    qp._validate_protocol(source, arm, protocol, device)
            for field, value in (("quantization", {"quant_type": "fp4"}),
                                 ("forbidden", {}), ("initial_checkpoint", {"model_id": "other"})):
                changed = copy.deepcopy(protocol)
                changed[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    qp._validate_protocol(source, "MOD-RV", changed, 0)
            protocol["model"]["modules_to_save"].pop()
            with self.assertRaises(ValueError):
                qp._validate_protocol(source, "MOD-RV", protocol, 0)

    def test_loader_refuses_cpu_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / BASE_REVISION
            source.mkdir()
            protocol = config() | {"quantization": qp.quantization_contract()}
            with (patch.object(qp, "_versions", return_value=qp.PINNED_VERSIONS),
                  patch.object(torch.cuda, "is_available", return_value=False)):
                with self.assertRaisesRegex(ValueError, "no CPU/offload"):
                    qp.build_quantized_policy(source, "MOD-RV", protocol)

    def test_inspection_requires_actual_quantization_and_all_exclusions(self):
        base = BaseFixture()
        with patch.object(qp, "inspect_modules", return_value=inventory_fixture()):
            names = qp._inspect_quantized_base(base, torch.device("cpu"))
            self.assertEqual(len(names), 300)
            for path, replacement in (("model.language.0", nn.Linear(4, 4)),
                                      ("model.visual.layers.0", nn.Linear(4, 4)),
                                      ("model.visual.merger.0", Linear4bitFixture()),
                                      ("lm_head", Linear4bitFixture())):
                old = base.get_submodule(path)
                base.set_submodule(path, replacement)
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "NF4"):
                    qp._inspect_quantized_base(base, torch.device("cpu"))
                base.set_submodule(path, old)
            base.model.visual.layers[0].compute_dtype = torch.float16
            with self.assertRaisesRegex(ValueError, "actual NF4"):
                qp._inspect_quantized_base(base, torch.device("cpu"))

    def test_scale_code_metadata_and_nested_state_changes_are_detected(self):
        base = nn.Sequential(Linear4bitFixture())
        original = qp.quantization_state_sha256(base)
        state = base[0].quant_state
        for value in (state.absmax, state.code, state.nested_absmax, state.nested_code):
            before = value.clone()
            value.add_(1)
            self.assertNotEqual(qp.quantization_state_sha256(base), original)
            value.copy_(before)
        state.shape = (2, 8)
        self.assertNotEqual(qp.quantization_state_sha256(base), original)
        state.quant_type = "fp4"
        with self.assertRaisesRegex(ValueError, "double-quantization state"):
            qp.quantization_state_sha256(base)
        state.quant_type, state.shape = "nf4", (4, 4)
        state.nested_absmax.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            qp.quantization_state_sha256(base)

    def test_peft_base_layer_prefix_is_canonicalized(self):
        original = nn.Sequential(Linear4bitFixture())
        before = qp.quantization_state_sha256(original)
        layer = original[0]
        original[0] = nn.Module()
        original[0].base_layer = layer
        self.assertEqual(qp.quantization_state_sha256(original), before)
        with self.assertRaisesRegex(ValueError, "no actual"):
            qp.quantization_state_sha256(nn.Linear(4, 4))

    def test_extra_frozen_check_catches_state_mutation_beyond_parameter_bytes(self):
        base = BaseFixture()
        qp._register_bf16_inputs(base)
        bundle = PolicyBundle(base, None, {}, {}, VisionRoute([]),
            {name: tensor_sha256(p) for name, p in base.named_parameters()})
        versions = qp.PINNED_VERSIONS
        bundle.inventory["quantization"] = {"contract": qp.quantization_contract(),
            "versions": versions, "state_sha256": qp.quantization_state_sha256(base)}
        with patch.object(qp, "_versions", return_value=versions):
            qp.assert_quantized_base_unchanged(bundle)
            base.model.visual.layers[0].quant_state.code.add_(1)
            with self.assertRaisesRegex(ValueError, "quantization state"):
                qp.assert_quantized_base_unchanged(bundle)

    def test_loader_dispatches_exact_nf4_and_existing_policy_without_base_upcasting(self):
        base, expected = object(), object()
        model = BaseFixture()
        # Loader dispatch is mocked; this fixture does not execute a CUDA or NF4 kernel.
        bundle = PolicyBundle(model, expected, {}, {"L": [nn.Parameter(torch.ones(1))],
            "M": [nn.Parameter(torch.ones(1))], "V": [nn.Parameter(torch.ones(1))]}, VisionRoute([]), {})
        protocol = config() | {"quantization": qp.quantization_contract()}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / BASE_REVISION
            source.mkdir()
            with (patch.object(qp, "_cuda_device", return_value=torch.device("cpu")),
                  patch("transformers.BitsAndBytesConfig") as bnb_config,
                  patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained", return_value=base) as loader,
                  patch("transformers.AutoProcessor.from_pretrained", return_value=expected) as processor,
                  patch.object(qp, "_inspect_quantized_base", return_value=["actual.module"]),
                  patch.object(qp, "_register_bf16_inputs") as hooks,
                  patch.object(qp, "quantization_state_sha256", return_value={"actual.module": {}}),
                  patch.object(qp, "configure_policy", return_value=bundle) as configure,
                  patch.object(qp, "assert_quantized_base_unchanged") as frozen,
                  patch.object(qp, "_versions", return_value=qp.PINNED_VERSIONS)):
                self.assertIs(qp.build_quantized_policy(source, "MOD-RV", protocol), bundle)
            bnb_config.assert_called_once_with(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.uint8, llm_int8_skip_modules=[*qp.MERGERS, "lm_head"],
                llm_int8_enable_fp32_cpu_offload=False)
            kwargs = loader.call_args.kwargs
            self.assertEqual(kwargs["device_map"], {"": 0})
            self.assertTrue(kwargs["local_files_only"])
            self.assertFalse(kwargs["trust_remote_code"])
            self.assertIs(kwargs["dtype"], torch.bfloat16)
            self.assertEqual(processor.call_args.kwargs["max_pixels"], protocol["image"]["max_pixels"])
            configure.assert_called_once_with(base, expected, "MOD-RV", protocol)
            hooks.assert_called_once_with(base)
            frozen.assert_called_once_with(bundle)
            self.assertEqual(bundle.inventory["quantization"]["reference"], "adapter_disabled_nf4_f0")

    def test_runtime_pin_rejects_different_library_version(self):
        with patch.object(qp, "version", side_effect=lambda name: qp.PINNED_VERSIONS[name]):
            self.assertEqual(qp._versions(), qp.PINNED_VERSIONS)
        with patch.object(qp, "version", return_value="different"):
            with self.assertRaisesRegex(ValueError, "exact separately pinned"):
                qp._versions()

    def test_frozen_export_keeps_original_mergers_and_every_quantization_tensor(self):
        base = BaseFixture()
        reference = {name: tensor_sha256(p) for name, p in base.named_parameters()}
        for root in qp.MERGERS:
            original = base.get_submodule(root)
            wrapper = nn.Module()
            wrapper.original_module = original
            wrapper.modules_to_save = nn.ModuleDict({"default": copy.deepcopy(original).float()})
            with torch.no_grad():
                wrapper.modules_to_save["default"][0].weight.add_(10)
            base.set_submodule(root, wrapper)
        layer = base.model.language[0]
        base.model.language[0] = nn.Module()
        base.model.language[0].base_layer = layer
        base.model.language[0].lora_A = nn.ModuleDict({"default": nn.Linear(4, 4)})
        bundle = PolicyBundle(base, ProcessorFixture(), {}, {}, VisionRoute([]), reference)
        qp._register_bf16_inputs(base)
        bundle.inventory["quantization"] = {"contract": qp.quantization_contract(),
            "versions": qp.PINNED_VERSIONS, "state_sha256": qp.quantization_state_sha256(base)}
        state = qp._frozen_state_dict(bundle)
        for name, digest in reference.items():
            self.assertEqual(tensor_sha256(state[name]), digest)
        self.assertFalse(any("lora_" in name or "modules_to_save" in name for name in state))
        self.assertEqual(len([name for name in state if ".weight.quant_state." in name]), 300)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nf4"
            with patch.object(qp, "_versions", return_value=qp.PINNED_VERSIONS):
                metadata = qp.save_quantized_base(bundle, destination)
                self.assertEqual(qp.read_quantized_base_metadata(destination), metadata)
                with self.assertRaises(FileExistsError):
                    qp.save_quantized_base(bundle, destination)
                with patch.object(qp, "_build_quantized_policy", return_value=bundle):
                    protocol = config() | {"quantization": qp.quantization_contract()}
                    self.assertIs(qp.load_quantized_policy(destination, "MOD-RV", protocol), bundle)
                (destination / "model.safetensors").write_bytes(b"damaged")
                with self.assertRaisesRegex(ValueError, "payload checksum"):
                    qp.read_quantized_base_metadata(destination)

    def test_cache_rejects_extra_files_symlinks_and_incomplete_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                qp.read_quantized_base_metadata(path)
            (path / "file").write_text("payload")
            (path / "symlink").symlink_to(path / "file")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                qp.read_quantized_base_metadata(path)


class CanonicalExportTests(unittest.TestCase):
    def test_explicit_bf16_input_cast_preserves_autograd_and_frozen_parameters(self):
        layer = nn.Linear(4, 4, dtype=torch.bfloat16).requires_grad_(False)
        before = {name: tensor_sha256(value) for name, value in layer.named_parameters()}
        layer.register_forward_pre_hook(qp._bf16_linear_input)
        value = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
        output = layer(value)
        self.assertIs(output.dtype, torch.bfloat16)
        output.float().square().sum().backward()
        self.assertIsNotNone(value.grad)
        self.assertTrue(bool(torch.isfinite(value.grad).all()))
        self.assertGreater(float(value.grad.abs().sum()), 0)
        self.assertEqual({name: tensor_sha256(p) for name, p in layer.named_parameters()}, before)
        ready = (value.to(torch.bfloat16),)
        self.assertIs(qp._bf16_linear_input(layer, ready), ready)
        with self.assertRaisesRegex(ValueError, "floating"):
            qp._bf16_linear_input(layer, (torch.ones(1, dtype=torch.long),))

    def test_reload_audit_blocks_original_model_files_without_global_test_hook(self):
        from fine_tuning.notebook_rl.quantized_cache import block_original_access

        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            original.mkdir()
            with patch.object(sys, "addaudithook") as install:
                attempts = block_original_access(original)
            hook = install.call_args.args[0]
            self.assertEqual(attempts["denied_original_access_attempts"], 0)
            with self.assertRaisesRegex(PermissionError, "original BF16"):
                hook("open", (str(original / "weights"), "r", 0))
            hook("open", (str(Path(directory) / "outside"), "r", 0))
            hook("open", (1, "r", 0))
            self.assertEqual(attempts["denied_original_access_attempts"], 1)

    def test_readiness_comparison_requires_identical_tokens_and_strict_finite_logprobs(self):
        from fine_tuning.notebook_rl.quantized_cache import _compare

        before = {"images": {"synthetic.png": "hash"}, "generations": [
            {"request_id": "one", "prompt_token_ids": [1, 2], "completions": [
                {"token_ids": [3, 4], "text": "fixture", "finish_reason": "stop", "logprobs": [-.2, -.3]}]}],
            "completion_logprobs": {"one": [-.2, -.3]}}
        self.assertEqual(_compare(before, copy.deepcopy(before))["max_abs_logprob_delta"], 0)
        for value in (float("nan"), -.4):
            changed = copy.deepcopy(before)
            changed["completion_logprobs"]["one"][0] = value
            with self.assertRaisesRegex(ValueError, "probability difference"):
                _compare(before, changed)
        changed = copy.deepcopy(before)
        changed["generations"][0]["completions"][0]["token_ids"][0] = 5
        with self.assertRaisesRegex(ValueError, "generation differs"):
            _compare(before, changed)

    def test_real_transformers_serialization_retains_tying_and_original_mergers(self):
        from safetensors.torch import load_file

        from fine_tuning.tests.test_training import tiny_model

        base = tiny_model().to(dtype=torch.bfloat16)
        base.config.tie_word_embeddings = base.config.text_config.tie_word_embeddings = True
        base.tie_weights()
        self.assertIs(base.get_input_embeddings().weight, base.lm_head.weight)
        bundle = configure_policy(base, ProcessorFixture(), "MOD-RV", config(), enforce_counts=False)
        bundle.inventory["quantization"] = {"state_sha256": {}}
        for root in qp.MERGERS:
            master = base.get_submodule(root).modules_to_save["default"]
            with torch.no_grad():
                next(master.parameters()).add_(10)
        # A quantizer makes HF 4.57.1 discard state_dict. The cache writer must
        # preserve canonical tensors independently of that HF code path.
        base.hf_quantizer = SimpleNamespace(get_state_dict_and_metadata=lambda *args: (None, {}))
        with tempfile.TemporaryDirectory() as directory:
            # This is a real HF serialization check with tiny BF16 CPU weights,
            # not an actual NF4/pretrained model load, forward or reload.
            destination = Path(directory) / "cache"
            with patch.object(qp, "assert_quantized_base_unchanged"):
                qp.save_quantized_base(bundle, destination)
            saved = load_file(str(destination / "model.safetensors"))
            saved_config = json.loads((destination / "config.json").read_text())
        self.assertTrue(saved_config["tie_word_embeddings"])
        self.assertNotIn("lm_head.weight", saved)
        self.assertIn("model.language_model.embed_tokens.weight", saved)
        self.assertFalse(any("lora_" in name or "modules_to_save" in name for name in saved))
        for name, digest in bundle.base_sha256.items():
            self.assertEqual(tensor_sha256(saved[name]), digest)


if __name__ == "__main__":
    unittest.main()
