"""Opt-in local GPU check of the v15 unmerged-LoRA rollout path on a tiny random Qwen3-VL in vLLM 0.11.0.

MMDL_TEST_VLLM_LORA_CUDA=1 enables it (MMDL_TEST_VLLM_LORA_SLEEP=0 runs without sleep mode). Tiny random weights with
the real tokenizer/processor: this proves the wiring and the preflight's power, not 4B numerics, CUDA-graph sizes,
throughput or pod memory.

Run it as `python -m fine_tuning.tests.test_error_note_single_vllm_lora_gpu` for a truthful exit status: on WSL an
in-process vLLM engine with sleep mode (CuMem) segfaults in the interpreter's final garbage collection after the
tests have finished, on the merged v14 path as well, so `python -m unittest` reports OK and then exits 139.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SNAPSHOT = (Path.home() / "mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots"
            / "ebb281ec70b05090aa6165b016eac8ec08e71b17")
ENABLED = os.environ.get("MMDL_TEST_VLLM_LORA_CUDA") == "1"
SLEEP = os.environ.get("MMDL_TEST_VLLM_LORA_SLEEP", "1") == "1"
EOS = [151643, 151645]
SCALE = {"L": 0.25, "V": 1.0, "M": 0.02}  # LoRA B / Merger noise large enough for PF5 to test both slopes


def build_tiny(dest: Path) -> Path:
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = json.loads((SNAPSHOT / "config.json").read_text())
    config["text_config"].update(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                                 head_dim=64, num_key_value_heads=2)
    config["text_config"]["rope_scaling"]["mrope_section"] = [8, 12, 12]
    config["vision_config"].update(depth=3, hidden_size=128, intermediate_size=256, num_heads=2, out_hidden_size=256,
                                   deepstack_visual_indexes=[0, 1, 2])
    torch.manual_seed(3407)
    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**config)).to(torch.bfloat16)
    model.save_pretrained(dest, safe_serialization=True)
    for name in ("chat_template.json", "merges.txt", "preprocessor_config.json", "tokenizer.json",
                 "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"):
        shutil.copyfile(SNAPSHOT / name, dest / name)
    return dest


def sampler(path: Path):
    from fine_tuning.error_note.single_vllm import SingleVllmSampler

    return SingleVllmSampler(path, kv_cache_gib=.25, max_num_seqs=2, gpu_memory_utilization=.3, sleep=SLEEP,
                             eos_token_ids=EOS, lora=True)


def policy(path: Path):
    """HF policy on the same BF16 snapshot with perturbed L/V LoRA B and Merger masters."""
    import torch
    from transformers import Qwen3VLForConditionalGeneration

    from fine_tuning.notebook_rl.policy import configure_policy
    from fine_tuning.notebook_rl.schema import load_protocol

    base = Qwen3VLForConditionalGeneration.from_pretrained(str(path), dtype=torch.bfloat16).to("cuda")
    protocol = load_protocol(Path(__file__).parents[1] / "notebook_rl/protocol_v1.yaml")
    bundle = configure_policy(base, None, "MOD-RV", protocol, enforce_counts=False)
    generator = torch.Generator(device="cuda").manual_seed(5)
    with torch.no_grad():
        for name, parameter in bundle.model.named_parameters():
            group = "M" if ".modules_to_save.default." in name else "V" if ".visual." in name else "L"
            if ".lora_B." in name or group == "M":
                noise = torch.randn(parameter.shape, generator=generator, device="cuda") * SCALE[group]
                parameter.add_(noise.to(parameter.dtype))
    return bundle


def unscoped_main(path: str) -> None:
    """Negative control in its own process: the engine built without the scope patch must fail PF1."""
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration as Q

    from fine_tuning.error_note import llm_lora

    upstream = getattr(Q.get_mm_mapping, "mmdl_original", Q.get_mm_mapping)
    with patch.object(llm_lora, "scope_qwen3vl_lora", lambda: None), patch.object(Q, "get_mm_mapping", upstream):
        sampler(Path(path))


def report_line(text: str) -> dict:
    lines = [line for line in text.splitlines() if line.startswith("SINGLE_LORA_PREFLIGHT ")]
    if not lines:
        raise AssertionError("no SINGLE_LORA_PREFLIGHT line")
    return json.loads(lines[-1].removeprefix("SINGLE_LORA_PREFLIGHT "))


@unittest.skipUnless(ENABLED, "vLLM LoRA GPU check requires explicit MMDL_TEST_VLLM_LORA_CUDA=1")
class TinyQwenLoraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="tiny-qwen3vl-")
        cls.path = build_tiny(Path(cls.tmp.name) / "model")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_engine_without_scope_patch_fails_pf1(self):
        code = "import sys; from fine_tuning.tests.test_error_note_single_vllm_lora_gpu import unscoped_main; " \
               "unscoped_main(sys.argv[1])"
        run = subprocess.run([sys.executable, "-c", code, str(self.path)], cwd=Path(__file__).parents[2],
                             capture_output=True, text=True, timeout=900)
        sys.stderr.write(run.stderr[-6000:])
        self.assertNotEqual(run.returncode, 0)
        if "SINGLE_LORA_PREFLIGHT " in run.stderr:  # engine started: PF1 must see the vision wrappers
            report = report_line(run.stderr)
            self.assertFalse(report["pf1"]["pass"])
            self.assertTrue(report["pf1"]["foreign_modules"] or report["pf1"]["renamed_vision_weights"])
        else:  # vLLM 0.11.0 fails at start-up: a vision proj LoRA wrapper runs punica with the decoder token map
            self.assertIn("self.proj(context_layer)", run.stderr)
            self.assertIn("assert token_lora_mapping.size(0) == M", run.stderr)

    def test_b_preflight_passes_and_controls_fail(self):
        from fine_tuning.error_note import llm_lora

        value, bundle = sampler(self.path), policy(self.path)
        self.addCleanup(shutil.rmtree, value.lora_dir, True)
        value.sync(bundle)  # first LoRA sync runs PF1-PF5 and raises on failure
        report = value.preflight_report
        self.assertTrue(report["pass"])
        self.assertTrue(report["pf5"]["llm"]["pass"] and report["pf5"]["vision"]["pass"])  # both slopes tested
        self.assertIs(report["pf4"]["pass"], True if SLEEP else None)

        value.vision_hooks.on = False  # control 1: vision hooks off
        with self.assertRaises(RuntimeError):
            value.verify_lora(bundle)
        self.assertFalse(value.preflight_report["pf2"]["pass"])
        self.assertFalse(value.preflight_report["pf5"]["vision"]["pass"])
        value.vision_hooks.on = True

        original = llm_lora.adapter_config  # control 2: alpha written as 16 (vLLM scaling 1 instead of 2)
        with patch.object(llm_lora, "adapter_config", lambda b: original(b) | {"lora_alpha": 16}):
            value._swap_adapter(bundle, value.loaded_policy)
        with self.assertRaises(RuntimeError):
            value.verify_lora(bundle)
        self.assertFalse(value.preflight_report["pf2"]["pass"])
        self.assertFalse(value.preflight_report["pf5"]["llm"]["pass"])

        value._swap_adapter(bundle, value.loaded_policy)  # restored adapter, fresh id
        value.release()  # real level-1 sleep in LoRA mode, then wake + Merger push + new adapter
        value.sync(bundle)
        self.assertEqual(value.lora_request.lora_int_id, 4 if SLEEP else 3)
        self.assertTrue(value.verify_lora(bundle)["pass"])
        self.check_stream(value, bundle)

    def check_stream(self, value, bundle):
        """The campaign's own stream (continuous add_request/step through _LoraEngine) samples the unmerged-LoRA
        policy: HF replays the sampled actions' log q within the training replay bound with the adapter on, not off."""
        import torch

        from fine_tuning.error_note import llm_lora, rl
        from fine_tuning.error_note.collect import SINGLE_SAMPLING, StreamStats
        from fine_tuning.error_note.collect_v4 import validate_single_output
        from fine_tuning.error_note.single_vllm import probe_requests
        from fine_tuning.notebook_rl.engines import RolloutRequest

        args = SimpleNamespace(policy_sha256=rl.policy_sha256(bundle), protocol_sha256="a" * 64)
        requests = [{"request_id": f"p{i}:t1", "problem_id": f"p{i}", "block_id": 0, "stage": "t1", "seed": 5 + i,
                     "max_tokens": 32, "messages": probe.messages, "images": probe.images}
                    for i, probe in enumerate(probe_requests(value.image))]
        stats = StreamStats()
        outputs = list(value.stream_builder(bundle, args, stats)(requests, StreamStats(), None))
        self.assertEqual(sorted(request["request_id"] for request, _ in outputs), ["p0:t1", "p1:t1"])
        self.assertEqual(stats.receipt_extra["single_sampler"], llm_lora.ENGINE_LORA)
        base, device = bundle.model.get_base_model(), next(bundle.model.parameters()).device

        def replay(request, generation):
            ids = generation["completion_token_ids"]
            probe = RolloutRequest(request["request_id"], None, request["messages"], request["images"], 1, len(ids),
                                   [request["seed"]])
            inputs = value.engine.model_inputs(probe, ids, device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = base.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
                scores = base.lm_head(hidden[0, -len(ids) - 1:-1]).float()
            done = torch.tensor([ids], device=device)
            return [float(rl.stored_support_logprob(scores[t:t + 1], done[:, :t], ids[t], support,
                                                     temperature=SINGLE_SAMPLING["temperature"],
                                                     presence_penalty=SINGLE_SAMPLING["presence_penalty"]))
                    for t, support in enumerate(generation["support_token_ids"])]

        on = off = count = 0.0
        for request, output in outputs:
            validate_single_output(request, output, policy_sha256=args.policy_sha256, protocol_sha256="a" * 64)
            generation = output["generation"]
            self.assertEqual(generation["engine"], llm_lora.ENGINE_LORA)
            sampled = generation["sampling_logprobs"]
            on += sum(abs(a - b) for a, b in zip(replay(request, generation), sampled, strict=True))
            with llm_lora.adapters_off(bundle, "L", "V"):
                off += sum(abs(a - b) for a, b in zip(replay(request, generation), sampled, strict=True))
            count += len(sampled)
        print(f"SINGLE_LORA_STREAM tokens {int(count)} replay_on {on / count:.5f} replay_off {off / count:.5f}",
              file=sys.stderr, flush=True)
        self.assertLessEqual(on / count, 0.05)  # protocol replay block bound
        self.assertGreater(off / count, 5 * max(on / count, 0.01))


if __name__ == "__main__":
    outcome = unittest.main(exit=False).result.wasSuccessful()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if outcome else 1)  # skip the final GC that segfaults under WSL CuMem (see module docstring)
