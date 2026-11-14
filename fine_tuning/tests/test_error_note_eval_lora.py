"""v15 MMMU evaluation with unmerged adapters (prereg v10): export, model reference, preregistration, engine wiring, pod plan.

CPU tests reuse the authored tiny single-campaign fixture of test_error_note_single_eval on a v15 plan, with nonzero LoRA B
and Merger masters. The opt-in GPU test (MMDL_TEST_VLLM_LORA_CUDA=1) builds the real B2 evaluation backend for kind
unmerged-lora on the tiny random Qwen3-VL of test_error_note_single_vllm_lora_gpu, runs the preflight checks and generates
two synthetic answers; run it as `python -m fine_tuning.tests.test_error_note_eval_lora` for a truthful exit status.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import yaml

from fine_tuning.error_note import llm_lora, lora_preflight
from fine_tuning.error_note.protocol import VERSION_SINGLE_V15
from fine_tuning.notebook_rl import checkpoint as ckpt, export, proof_eval as pe
from fine_tuning.tests import test_error_note_single_eval as single_eval
from fine_tuning.tests.test_notebook_rl_checkpoint import fresh_tiny
from fine_tuning.tests.test_notebook_rl_proof_eval import A40, B2, IDS, FakeLLM, backend_mocks, variant_cfg, write_subset
from mmdl.runtime.artifacts import sha256_file

GPU = os.environ.get("MMDL_TEST_VLLM_LORA_CUDA") == "1"


def perturbed_tiny(model_path, arm, protocol):
    """fresh_tiny with nonzero LoRA B and moved Merger masters, so merged and unmerged exports differ."""
    bundle = fresh_tiny(model_path, arm, protocol)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name, parameter in bundle.model.named_parameters():
            if ".lora_B." in name or ".modules_to_save.default." in name:
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return bundle


def halves_and_b2(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    b2 = root / "b2.jsonl"
    b2.write_text("".join(json.dumps({"id": sid, "correct": i >= 342}) + "\n" for i, sid in enumerate(IDS)))
    for kind in pe.HALVES:
        pe.make_subset(kind, root / f"{kind}.json", b2)
    return (b2, write_subset(root / "wrong.json", b2), write_subset(root / "rest.json", b2, kind="rest"),
            root / "half-a.json", root / "half-b.json")


class V15UnmergedExportTests(unittest.TestCase):
    STAGES = {19: "pilot", 20: "stopped", 21: "final"}
    SINGLE_VERSION = VERSION_SINGLE_V15
    EXECUTION_MAX_BLOCKS = None

    @classmethod
    def setUpClass(cls) -> None:
        with mock.patch.object(single_eval, "fresh_tiny", perturbed_tiny):
            single_eval.SingleCampaignEvaluationV12.setUpClass.__func__(cls)
        cls.ref_path = cls.exports[21] / "model_ref.json"
        cls.model = cls.exports[21] / "model"

    def bundle(self):
        return ckpt.load_eval(self.base, self.out / "checkpoint-21", binding=single_eval.campaign._binding(self.plan),
                              _test_policy_factory=fresh_tiny)

    def test_v15_export_keeps_the_base_bits_and_writes_both_adapters_unmerged(self) -> None:
        from fine_tuning.notebook_rl.policy import merged_named_tensors
        from safetensors.torch import load_file

        ref = json.loads(self.ref_path.read_text())
        self.assertEqual((ref["kind"], ref["llm_lora"], ref["llm_lora"]["scaling"]), (pe.LORA_KIND, pe.lora_section(), 2.0))
        self.assertEqual(ref["export"]["training_protocol"]["rollout_adapter"], "vllm-lora-unmerged+vision-hooks")
        rows = {row["path"]: row for row in ref["files"]}
        self.assertEqual([row["path"] for row in ref["files"]], sorted(rows))
        for name in [*export.LORA_FILES, "model.safetensors", "export_manifest.json"]:
            self.assertEqual((rows[name]["sha256"], rows[name]["bytes"]),
                             (sha256_file(self.model / name), (self.model / name).stat().st_size), name)
        manifest = export.validate_export_reference(self.model, ref)
        self.assertEqual((manifest["format"], manifest["precision_policy"], ref["export"]["precision_policy"]),
                         (export.FORMAT_LORA, export.PRECISION_LORA, export.PRECISION_LORA))
        self.assertEqual(manifest["lora"]["scaling"], 2.0)
        bundle = self.bundle()
        files = lora_preflight.check_files(self.model, self.base, bundle)
        self.assertTrue(files["pass"], files)
        self.assertGreater(files["merger_tensors"], 0)
        self.assertEqual(files["vision_modules"], len(bundle.inventory["targets"]["V"]))
        self.assertEqual(files["llm_adapter_tensors"], 2 * len(bundle.inventory["targets"]["L"]))
        # the merged (v1) export of the same checkpoint would have changed the adapted weights; this one did not
        shards, base = load_file(str(self.model / "model.safetensors")), load_file(str(self.base / "model.safetensors"))
        merged = dict(merged_named_tensors(bundle))
        target = bundle.inventory["targets"]["L"][0] + ".weight"
        self.assertTrue(torch.equal(shards[target], base[target]))
        self.assertFalse(torch.equal(merged[target], base[target]))
        mergers = [n for n in shards if n.startswith(llm_lora.HF_MERGERS)]
        self.assertTrue(all(torch.equal(shards[n], merged[n]) for n in mergers))
        self.assertTrue(any(not torch.equal(shards[n], base[n]) for n in mergers))

    def test_pf5_amplified_adapter_takes_fresh_ids_and_the_export_request_comes_back(self) -> None:
        from safetensors.torch import load_file
        from vllm.lora.request import LoRARequest

        loaded, calls = {pe.LORA_INT_ID}, []

        def remove(lora_id):
            calls.append(("remove", lora_id))
            return lora_id in loaded and loaded.discard(lora_id) is None

        def add(request):
            calls.append(("add", request.lora_int_id))
            return loaded.add(request.lora_int_id) is None

        engine = SimpleNamespace(list_loras=lambda: set(loaded), remove_lora=remove, add_lora=add)
        real = LoRARequest(pe.LORA_NAME, pe.LORA_INT_ID, str(self.model / export.LLM_LORA_DIR))
        exported = load_file(str(self.model / export.LLM_LORA_DIR / "adapter_model.safetensors"))
        with tempfile.TemporaryDirectory(prefix="v15-amplified-") as tmp:
            swap = lora_preflight.adapter_swap(engine, self.bundle(), real, Path(tmp))
            first = swap(llm_lora.AMPLIFY)
            amplified = load_file(str(Path(first.lora_path) / "adapter_model.safetensors"))
            self.assertIs(swap(1), real)
            second = swap(llm_lora.AMPLIFY)
        self.assertEqual(len({pe.LORA_INT_ID, first.lora_int_id, second.lora_int_id}), 3)
        self.assertEqual(calls, [("remove", 1), ("add", first.lora_int_id), ("remove", first.lora_int_id), ("add", 1),
                                 ("remove", 1), ("add", second.lora_int_id)])
        self.assertEqual(set(amplified), set(exported))
        for key, value in exported.items():  # A as exported, B x AMPLIFY exactly (FP32 master)
            factor = llm_lora.AMPLIFY if ".lora_B." in key else 1
            self.assertTrue(torch.equal(amplified[key], value * factor), key)

    def test_v15_export_reference_refuses_tampered_missing_or_extra_adapter_files(self) -> None:
        ref = json.loads(self.ref_path.read_text())
        with tempfile.TemporaryDirectory(prefix="v15-tamper-") as tmp:
            for case in ("tampered", "missing", "extra"):
                model = Path(shutil.copytree(self.model, Path(tmp) / case))
                if case == "tampered":
                    path = model / export.VISION_LORA_FILE
                    path.write_bytes(path.read_bytes()[:-1] + b"\0")
                elif case == "missing":
                    (model / export.LORA_FILES[0]).unlink()
                else:
                    (model / "other").mkdir()
                with self.assertRaises((ValueError, FileNotFoundError), msg=case):
                    export.validate_export_reference(model, ref)

    def test_v9_refuses_and_v10_declares_the_unmerged_export_with_the_v9_rules(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v15-v10-") as tmp:
            root = Path(tmp)
            b2, wrong, rest, half_a, half_b = halves_and_b2(root)
            for fmt in ("v7", "v9"):
                with self.assertRaisesRegex(ValueError, "preregistration v10 only"):
                    pe.preregister(root / f"{fmt}.json", wrong, rest, b2, fmt, ft_single_model_ref=self.ref_path,
                                   **({"half_a_subset": half_a, "half_b_subset": half_b} if fmt == "v9" else {}))
            prereg = root / "v10.json"
            pe.preregister(prereg, wrong, rest, b2, "v10", ft_single_model_ref=self.ref_path, half_a_subset=half_a,
                           half_b_subset=half_b)
            loaded, _ = pe.load_preregistration(prereg)
            v9 = json.loads(json.dumps(pe.build_preregistration_v9(wrong, rest, half_a, half_b, b2, self.ref_path,
                                                                   model_ref=pe._lora_model_ref)))
            extra = {"format", "arms", "model_application", "limitations"}
            self.assertEqual({k: v for k, v in loaded.items() if k not in extra}, {k: v for k, v in v9.items() if k not in extra})
            self.assertEqual(loaded["format"], pe.FORMAT_V10)
            self.assertEqual(loaded["limitations"], v9["limitations"] + [pe.CONFOUND_EN])
            self.assertEqual(loaded["arms"][pe.ARM_SINGLE]["definition"],
                             v9["arms"][pe.ARM_SINGLE]["definition"] + ". " + pe.APPLICATION_EN)
            self.assertEqual(loaded["arms"]["F0"], v9["arms"]["F0"])
            self.assertEqual(loaded["model_application"]["application"], pe.APPLICATION)
            gate = loaded["screen_gate"]
            self.assertEqual((gate["arm"], gate["variant"], gate["subset"], gate["items"], gate["minimum_correct"]),
                             (pe.ARM_SINGLE, pe.FINAL_VARIANT_V8, "half-a", 450, gate["reference_correct"] + 6))
            self.assertEqual(loaded["runs"], [{"arm": arm, "variant": B2, "subset": half, "role": "statistical"}
                                              for arm in ("F0", pe.ARM_SINGLE) for half in pe.HALVES])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pe.main(["verify-single-export", "--model-ref", str(self.ref_path), "--checkpoint",
                                          str(self.out / "checkpoint-21"), "--binding", str(self.plan_file),
                                          "--prereg", str(prereg)]), 0)
            with self.assertRaisesRegex(ValueError, "artifact root of the arm's half-a run"):
                pe.ledger_start(prereg, pe.ARM_SINGLE, B2, "half-b-run", root / "ledger.jsonl", "half-b")
            self.assertEqual(pe._final_model_ref(self.ref_path, pe.FORMAT_V10)[1],
                             {k: v for k, v in loaded["arms"][pe.ARM_SINGLE].items() if k != "definition"}
                             | {"model_ref": str(self.ref_path.resolve())})

    def test_plain_verify_model_refuses_the_lora_kind_and_the_registered_one_checks_every_file(self) -> None:
        from mmdl.evaluation import engine

        ref, original = json.loads(self.ref_path.read_text()), engine.verify_model
        with self.assertRaisesRegex(ValueError, "Unknown model manifest kind"):
            engine.verify_model(self.model, ref)  # plain scripts/eval.sh cannot evaluate a v15 export
        with pe.registered_variants():
            engine.verify_model(self.model, ref)
            with self.assertRaisesRegex(ValueError, "llm_lora section"):
                engine.verify_model(self.model, {k: v for k, v in ref.items() if k != "llm_lora"})
            tampered = copy.deepcopy(ref)
            next(r for r in tampered["files"] if r["path"] == export.VISION_LORA_FILE)["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "integrity"):
                engine.verify_model(self.model, tampered)
        self.assertIs(engine.verify_model, original)

    def test_v10_plan_runs_the_preflight_before_the_first_ledger_start_on_half_a_only(self) -> None:
        from fine_tuning.notebook_rl.pod import chain, eval_stage

        with tempfile.TemporaryDirectory(prefix="v15-plan-") as tmp:
            root = Path(tmp)
            b2, wrong, rest, half_a, half_b = halves_and_b2(root)
            prereg = root / "v10.json"
            pe.preregister(prereg, wrong, rest, b2, "v10", ft_single_model_ref=self.ref_path, half_a_subset=half_a,
                           half_b_subset=half_b)
            volume = root / "volume"
            proof = volume / "eval-data/proof"
            proof.mkdir(parents=True)
            pinned = Path(shutil.copyfile(prereg, proof / "prereg.json"))
            params = {"version": eval_stage.PARAMS_VERSION, "commit": "a" * 40, "repo_root": str(pe.ROOT),
                      "python": f"{volume}/venv-0123456789abcdef/bin/python", "volume_root": str(volume),
                      "artifact_root": str(proof / "artifacts"), "prereg": str(pinned), "prereg_sha256": sha256_file(pinned),
                      "ledger": str(proof / "ledger.jsonl"), "run_prefix": "v15", "arm": pe.ARM_SINGLE, "seeds": [3407],
                      "vllm_inprocess": True, "eval_slots": "auto", "checkpoint": str(self.out / "checkpoint-21"),
                      "checkpoint_metadata_sha256": "2" * 64, "binding": str(self.plan_file), "binding_sha256": "3" * 64,
                      "training_commit": "f" * 40, "ft_export": str(self.exports[21]), "reuse_export": True,
                      "ft_model_ref_sha256": "4" * 64}
            specs = {}
            for half, source in zip(pe.HALVES, (half_a, half_b)):
                subset = Path(shutil.copyfile(source, proof / f"{half}.json"))
                specs[half] = eval_stage.plan("eval-ft", params | {"subset": str(subset), "subset_tag": half,
                                                                   "subset_sha256": sha256_file(subset)})
                self.assertIs(chain.check_spec(specs[half]), specs[half])
            names = [step["name"] for step in specs["half-a"]["steps"]]
            first_start = next(i for i, name in enumerate(names) if name.startswith("ledger-"))
            index = names.index("lora-preflight")
            self.assertLess(names.index("verify-single-export"), index)
            self.assertLess(index, first_start)
            step = specs["half-a"]["steps"][index]
            report = f"{self.exports[21]}/lora_preflight/preflight.json"
            self.assertEqual(step["argv"][:5], ["/usr/bin/env", f"{pe.VLLM_INPROCESS_ENV}=0", f"{pe.SLOTS_ENV}=auto",
                                                params["python"], "-m"])
            self.assertEqual(step["argv"][5:7], [eval_stage.PREFLIGHT, "check"])
            self.assertEqual((step["argv"][step["argv"].index("--out") + 1], step["expect_files"]), (report, [report]))
            self.assertEqual(step["argv"][step["argv"].index("--protocol") + 1],
                             f"{pe.ROOT}/{pe._rel(pe.VARIANTS[pe.FINAL_VARIANT_V8][0])}")
            self.assertNotIn("lora-preflight", [step["name"] for step in specs["half-b"]["steps"]])


class LoraEngineWiringTests(unittest.TestCase):
    """CPU stand-ins for CUDA and vLLM (test_notebook_rl_proof_eval.backend_mocks); the real VLLMBackend.__init__ runs."""

    def backends(self, *kinds, env: str | None = "0"):
        from mmdl.evaluation.backends.vllm_backend import VLLMBackend

        hw, b2 = yaml.safe_load(A40.read_text()), variant_cfg(B2)
        attached: list = []
        FakeLLM.calls.clear()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ))
            os.environ.pop(pe.VLLM_INPROCESS_ENV, None)
            if env is not None:
                os.environ[pe.VLLM_INPROCESS_ENV] = env
            stack.enter_context(backend_mocks())
            stack.enter_context(mock.patch.object(pe, "attach_lora", lambda backend, path: attached.append((backend, path))))
            scope = stack.enter_context(mock.patch.object(llm_lora, "scope_qwen3vl_lora"))
            stack.enter_context(pe.registered_variants(slots=16))
            built = [VLLMBackend("model", b2, hw, kind, "base") for kind in kinds]
        return built, list(FakeLLM.calls), attached, scope.call_count

    def test_lora_kind_adds_only_the_engine_args_and_attaches_the_adapter(self) -> None:
        (merged, lora), calls, attached, scoped = self.backends("merged", pe.LORA_KIND)
        plain, served = calls
        self.assertEqual(served, plain | llm_lora.ENGINE_ARGS)
        self.assertEqual(attached, [(lora, Path("model"))])
        self.assertEqual(scoped, 1)
        self.assertEqual((lora.batch_size, lora.cfg["execution"]["batch_size"]), (16, 32))

    def test_no_adapter_kinds_get_exactly_the_v14_llm_kwargs(self) -> None:
        backends, calls, attached, scoped = self.backends("merged", "full", "base")
        v14 = {"model", "tokenizer", "dtype", "tensor_parallel_size", "max_num_seqs", "max_model_len",
               "gpu_memory_utilization", "quantization", "generation_config", "mm_processor_kwargs",
               "enable_prefix_caching", "async_scheduling", "enforce_eager", "limit_mm_per_prompt", "trust_remote_code"}
        self.assertEqual([set(call) for call in calls], [v14] * 3)
        self.assertEqual([(c["max_num_seqs"], c["max_model_len"], c["gpu_memory_utilization"]) for c in calls],
                         [(16, 40960, 0.90)] * 3)
        self.assertEqual((attached, scoped), ([], 0))
        self.assertTrue(all(not hasattr(b, "lora_request") for b in backends))

    def test_lora_kind_needs_the_in_process_engine_and_a_registered_variant(self) -> None:
        from mmdl.evaluation.backends.vllm_backend import VLLMBackend

        for env in (None, "1"):
            with self.assertRaisesRegex(RuntimeError, "in-process continuous engine"):
                self.backends(pe.LORA_KIND, env=env)
        hw = yaml.safe_load(A40.read_text())
        with backend_mocks(), pe.registered_variants():
            with self.assertRaisesRegex(ValueError, "registered variant"):
                VLLMBackend("model", variant_cfg(pe.COURSE), hw, pe.LORA_KIND, "base")

    def test_registration_names_the_lora_hooks_and_restores_every_patch(self) -> None:
        from mmdl.evaluation import engine
        from mmdl.evaluation.backends.vllm_backend import VLLMBackend
        from mmdl.runtime import contracts

        with mock.patch.object(pe, "gpu_info", return_value=("NVIDIA A40", 46068)):
            patched = pe.variant_registration(B2)["patched"]
        self.assertIn(pe.LORA_KIND, patched["mmdl.evaluation.engine.verify_model"])
        self.assertIn("vision-LoRA forward hooks", patched["unmerged-lora"])
        functions = (contracts.validate_configs, VLLMBackend.__init__, engine.verify_model)
        with pe.registered_variants():
            self.assertNotEqual((contracts.validate_configs, VLLMBackend.__init__, engine.verify_model), functions)
        self.assertEqual((contracts.validate_configs, VLLMBackend.__init__, engine.verify_model), functions)


class ThroughputTests(unittest.TestCase):
    """lora_preflight.throughput on a fake backend: the dict, the explicit lora_request=None, the token-count check."""

    SLOTS = 4

    def run_throughput(self, per_request: int = lora_preflight.THROUGHPUT_TOKENS):
        calls, lora = [], object()

        def generate(prompts, params, **kwargs):
            calls.append((prompts, params, kwargs))
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[0] * per_request)]) for _ in prompts]

        params_made: list = []
        vllm = SimpleNamespace(SamplingParams=lambda **kw: params_made.append(kw) or kw)
        prep = SimpleNamespace(
            processor=SimpleNamespace(apply_chat_template=lambda messages, **kw: "chat:" + messages[0]["content"][0]["text"]),
            tokenizer=SimpleNamespace(encode=lambda text, **kw: [len(text), 7]))
        backend = SimpleNamespace(llm=SimpleNamespace(generate=generate), lora_request=lora)
        clock = iter([0.0, 2.0, 10.0, 11.0])  # adapter run 2 s, base run 1 s
        with mock.patch.dict(sys.modules, {"vllm": vllm}), \
                mock.patch.object(lora_preflight, "time", SimpleNamespace(perf_counter=lambda: next(clock))):
            return lora_preflight.throughput(backend, prep, self.SLOTS), calls, lora, params_made

    def test_records_adapter_and_base_speed_with_an_explicit_none_for_the_base_run(self) -> None:
        report, calls, lora, params_made = self.run_throughput()
        total = self.SLOTS * 256
        self.assertEqual(report, {"slots": self.SLOTS, "tokens_per_request": 256, "lora_tokens_per_second": total / 2,
                                  "base_tokens_per_second": total / 1, "ratio": 2.0, "limit": 1.5})
        self.assertEqual(params_made, [{"n": 1, "temperature": 0.0, "max_tokens": 256, "ignore_eos": True,
                                        "detokenize": False}])
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][2]["lora_request"], lora)
        self.assertIn("lora_request", calls[1][2])
        self.assertIsNone(calls[1][2]["lora_request"])
        for prompts, _, kwargs in calls:
            self.assertEqual(len(prompts), self.SLOTS)
            self.assertEqual(prompts[0], {"prompt_token_ids": [len("chat:" + lora_preflight.CONTINUATION), 7]})
            self.assertIs(kwargs["use_tqdm"], False)

    def test_a_short_generation_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected 1024"):
            self.run_throughput(per_request=255)


def write_export_like(bundle, base: Path, out: Path) -> Path:
    """The v15 export layout from an in-memory policy: base shard with BF16 Merger masters, both adapter files."""
    from safetensors.torch import load_file, save_file

    from fine_tuning.notebook_rl.policy import merged_named_tensors

    shutil.copytree(base, out)
    tensors = load_file(str(out / "model.safetensors"))
    mergers = {n: v.cpu().contiguous() for n, v in merged_named_tensors(bundle) if n.startswith(llm_lora.HF_MERGERS)}
    if not mergers or not set(mergers) <= set(tensors):
        raise AssertionError("tiny export: Merger names differ from the base shard")
    save_file(tensors | mergers, str(out / "model.safetensors"), metadata={"format": "pt"})
    llm_lora.write_llm_adapter(bundle, out / export.LLM_LORA_DIR)
    (out / export.VISION_LORA_DIR).mkdir()
    save_file({f"{n}.{part}": t.cpu().contiguous() for n, (a, b) in llm_lora.vision_lora(bundle).items()
               for part, t in (("lora_A", a), ("lora_B", b))}, str(out / export.VISION_LORA_FILE))
    return out


@unittest.skipUnless(GPU, "vLLM LoRA GPU check requires explicit MMDL_TEST_VLLM_LORA_CUDA=1")
class TinyQwenEvalLoraGpuTests(unittest.TestCase):
    def test_eval_engine_preflight_and_two_answers_with_the_adapter_on(self) -> None:
        from mmdl.evaluation.backends import vllm_backend
        from mmdl.runtime import contracts

        from fine_tuning.tests import test_error_note_single_vllm_lora_gpu as tiny

        tmp = tempfile.TemporaryDirectory(prefix="tiny-qwen3vl-eval-")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        base = tiny.build_tiny(root / "base")
        bundle = tiny.policy(base)  # BF16 on CUDA, perturbed L/V LoRA B and Merger masters
        model = write_export_like(bundle, base, root / "model")
        self.assertTrue(lora_preflight.check_files(model, base, bundle)["pass"])
        protocol = pe.VARIANTS[pe.FINAL_VARIANT_V8][0]
        cfg = yaml.safe_load(protocol.read_text())
        prep = lora_preflight.probe_preparation(base, model, cfg["image"], cfg["execution"]["max_model_len"])
        pairs = lora_preflight.probe_pairs(prep, cfg["image"])
        reference = llm_lora.hf_columns(bundle, prep, pairs)
        real = vllm_backend._vllm

        def low_memory():  # the backend's fixed 0.90 does not fit beside the HF policy on an 8 GB laptop GPU
            llm_cls, sampling, version = real()
            return (lambda **kwargs: llm_cls(**kwargs | {"gpu_memory_utilization": 0.45})), sampling, version

        with mock.patch.dict(os.environ, {pe.VLLM_INPROCESS_ENV: "0"}), \
                mock.patch.object(vllm_backend, "_vllm", low_memory), pe.registered_variants(slots=2):
            run_cfg, hw = contracts.load_configs(protocol, A40)
            backend = vllm_backend.VLLMBackend(str(model), run_cfg, hw, pe.LORA_KIND, str(base))
        report = lora_preflight.engine_checks(backend, bundle, prep, pairs, reference)
        print("EVAL_LORA_PREFLIGHT " + json.dumps(report, sort_keys=True, default=str), file=sys.stderr, flush=True)
        self.assertTrue(report["pass"], report)
        amplified = report["pf5"]["amplified"]
        self.assertTrue(amplified["llm"]["pass"] and amplified["vision"]["pass"])  # both amplified slopes tested
        self.assertTrue(report["pf2_restored"]["pass"])
        self.assertEqual(backend.llm.llm_engine.list_loras(), {pe.LORA_INT_ID})  # the export's adapter is back
        backend.vision_hooks.on = False  # control: the vision adapter not applied
        control = lora_preflight.engine_checks(backend, bundle, prep, pairs, reference)
        backend.vision_hooks.on = True
        self.assertFalse(control["pass"] or control["pf2"]["pass"] or control["pf5"]["amplified"]["vision"]["pass"])
        speed = lora_preflight.throughput(backend, prep, 2)  # the real engine: the LoRA slot, then lora_request=None
        print("EVAL_LORA_THROUGHPUT " + json.dumps(speed, sort_keys=True), file=sys.stderr, flush=True)
        self.assertTrue(speed["lora_tokens_per_second"] > 0 and speed["base_tokens_per_second"] > 0, speed)
        backend.cfg = copy.deepcopy(backend.cfg)
        backend.cfg["generation"]["max_new_tokens"] = 16  # synthetic answers only; the protocol run keeps 32768
        requests = [{"sample_id": f"lora-{i}", "messages": request.messages, "images": request.images, "seed": 7 + i}
                    for i, (request, _) in enumerate(pairs)]
        rows = list(backend.generate_stream(requests))  # add_request asserts each live request carries the LoRA
        self.assertEqual(sorted(row["sample_id"] for row in rows), ["lora-0", "lora-1"])
        print("EVAL_LORA_ANSWERS " + json.dumps([{k: row.get(k) for k in ("sample_id", "finish_reason", "output_tokens")}
                                                 for row in rows], default=str), file=sys.stderr, flush=True)
        self.assertEqual(backend.describe()["llm_lora"]["lora_request"], [pe.LORA_NAME, pe.LORA_INT_ID, "llm_lora"])
        with self.assertRaisesRegex(RuntimeError, "batch path"):
            backend.generate_batch(requests[:1])


if __name__ == "__main__":
    outcome = unittest.main(exit=False).result.wasSuccessful()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if outcome else 1)  # skip the final GC of an in-process vLLM engine (see the GPU test module docstring)
