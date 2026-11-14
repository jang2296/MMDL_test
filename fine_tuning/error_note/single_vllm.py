"""SINGLE v3 generation on an in-process vLLM 0.11 engine: continuous batching, policy pushed after every update.

The HF/PEFT bundle stays the training authority. Before a block generates, its LoRA-merged BF16 weights (Merger
masters rounded to BF16, as in the eval export) are pushed with the existing complete-inventory loader of
notebook_rl.engines.VLLMRolloutEngine. vLLM's processed log-probabilities (presence penalty -> temperature ->
top-k/top-p, the order rl.processed_scores uses) give each sampled action's log q and the support it was drawn
from; every block's training replay checks them against the HF policy (protocol replay bounds). Sleep mode
level 2 frees vLLM weights and KV cache while the HF policy trains; the next push restores complete weights.

v15 LoRA mode (`lora=True`, llm_lora): the original BF16 snapshot stays in vLLM; the LLM adapter is a fresh vLLM LoRA
per policy, the vision adapter runs through exact hooks, only the Mergers are pushed, and sleep is level 1 (the pool
also holds the LoRA slots). The first LoRA sync runs the PF1-PF5 preflight before any generation; PF5 probes an
amplified copy of the adapter, after which the real adapter is restored under a fresh id and re-checked bit-exactly.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Iterator

from . import llm_lora
from .collect import SINGLE_SAMPLING, Stream, StreamStats, continuous_stream, single_request_binding, single_request_sha256
from .contract import CONTEXT_LIMIT, IMAGE

ENGINE = "vllm-0.11.0-processed-logprobs"
TOP_LOGPROBS = 20  # == SINGLE_SAMPLING top_k: every token the sampler could draw is returned


@contextmanager
def _nondeterministic():
    """vLLM sampling is batch-dependent anyway; HF replay alone needs deterministic kernels."""
    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def support_from_logprobs(token: int, row: dict) -> tuple[float, list[int]]:
    """Finite processed log-probs are exactly the top-k/top-p support; the sampled token must be inside it."""
    support = sorted(t for t, value in row.items() if math.isfinite(value.logprob))
    if token not in support or len(support) > SINGLE_SAMPLING["top_k"]:
        raise RuntimeError("SINGLE vLLM: sampled action outside its returned processed support")
    return float(row[token].logprob), support


class _LoraEngine:
    """continuous_stream calls add_request(rid, prompt, params); attach the policy's LoRA and confirm it on the live
    scheduler request (vLLM 0.11 V1 RequestOutput never carries lora_request, output_processor.py:251)."""

    def __init__(self, inner: Any, lora: Any):
        self.inner, self.lora = inner, lora

    def add_request(self, request_id: str, prompt: Any, params: Any) -> None:
        self.inner.add_request(request_id, prompt, params, lora_request=self.lora)
        live = self.inner.engine_core.engine_core.scheduler.requests[request_id].lora_request
        if live is None or live.lora_int_id != self.lora.lora_int_id:
            raise RuntimeError("SINGLE vLLM: request is not bound to the current policy LoRA")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


PROBE_TEXT = "Describe every visible detail of the image, then explain what it could represent."
PROBE_TOKENS = 64


def probe_requests(image: dict) -> list:
    """PF5 inputs: fixed text with 1 and 2 deterministic noise images of at least min_pixels (no task data)."""
    import numpy as np
    from PIL import Image

    from fine_tuning.notebook_rl.engines import RolloutRequest

    side = -(-(math.isqrt(image["min_pixels"] - 1) + 1) // 32) * 32  # ceil(sqrt(min_pixels)) up to a multiple of 32
    if not image["min_pixels"] <= side * side <= image["max_pixels"]:
        raise ValueError("SINGLE vLLM: probe image size outside the image budget")
    rng = np.random.RandomState(3407)
    pictures = [Image.fromarray(rng.randint(0, 256, (side, side, 3), dtype=np.uint8)) for _ in range(3)]
    out = []
    for index, images in enumerate((pictures[:1], pictures[1:])):
        messages = [{"role": "user", "content": [*({"type": "image"} for _ in images),
                                                 {"type": "text", "text": PROBE_TEXT}]}]
        out.append(RolloutRequest(f"lora-preflight-{index}", None, messages, images, 1, PROBE_TOKENS, [3407]))
    return out


class SingleVllmSampler:
    """One engine for the whole campaign; `stream_builder` matches collect.hf_single_stream_builder's signature. The image config comes from the plan's protocol."""

    lora = False  # class defaults keep __new__-built samplers on the merged-push path
    lora_request: Any = None
    vision_hooks: Any = None

    def __init__(self, model_path: Path, *, kv_cache_gib: float, max_num_seqs: int, gpu_memory_utilization: float,
                 sleep: bool, eos_token_ids: Iterable[int], image: dict | None = None, lora: bool = False):
        self.image = dict(IMAGE if image is None else image)
        low, high = self.image.get("min_pixels"), self.image.get("max_pixels")
        if not (isinstance(low, int) and isinstance(high, int) and 0 < low <= high):
            raise ValueError("SINGLE vLLM: image config needs positive integer min_pixels <= max_pixels")
        from importlib import import_module
        from importlib.metadata import version

        from fine_tuning.notebook_rl.engines import InputPreparation, VLLMRolloutEngine, _weight_inventory, vllm_kwargs

        if version("vllm") != "0.11.0":
            raise RuntimeError("SINGLE vLLM: vllm==0.11.0 required")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # in-process engine: weight push and add_request/step
        from transformers import AutoConfig, AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
                                                  min_pixels=self.image["min_pixels"], max_pixels=self.image["max_pixels"])
        processor.image_processor.size = {"shortest_edge": self.image["min_pixels"], "longest_edge": self.image["max_pixels"]}
        engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)  # reuse prepare/_prompt/_load_weights, own LLM kwargs
        InputPreparation.__init__(engine, processor, AutoConfig.from_pretrained(str(model_path), local_files_only=True),
                                  self.image, CONTEXT_LIMIT)
        engine._weight_shapes = _weight_inventory(engine.model_config)
        engine.model_dir = Path(model_path).resolve()
        kwargs = vllm_kwargs(Path(model_path), self.image, int(kv_cache_gib * 2**30), max_num_seqs, CONTEXT_LIMIT,
                             gpu_memory_utilization=float(gpu_memory_utilization))
        extra: dict = {}
        if lora:
            self.pf0 = llm_lora.host_memory(bool(sleep))
            if not self.pf0["pass"]:
                self._report({"pf0": self.pf0, "pass": False})
                raise RuntimeError("SINGLE vLLM: PF0 host memory below the level-1 sleep requirement")
            llm_lora.scope_qwen3vl_lora()
            extra = dict(llm_lora.ENGINE_ARGS)
            if extra.keys() & kwargs.keys():
                raise ValueError("SINGLE vLLM: LoRA engine arguments overlap the pinned vLLM arguments")
        with _nondeterministic():
            engine.llm = import_module("vllm").LLM(**kwargs, logprobs_mode="processed_logprobs",
                                                   max_logprobs=TOP_LOGPROBS, enable_sleep_mode=bool(sleep), **extra)
        engine._call, engine._synchronized, engine._push_complete, engine._last_loaded_version = 0, True, True, None
        self.engine, self.max_num_seqs, self.sleep = engine, max_num_seqs, bool(sleep)
        self.eos_token_ids = sorted(set(eos_token_ids))
        self.loaded_policy: str | None = None  # the disk snapshot is F0, but a push always precedes generation
        self.asleep = False
        if lora:
            self.lora = True
            self.frozen_fp = llm_lora.base_fingerprint(engine.llm)  # before any push
            self.vision_hooks = engine.llm.apply_model(llm_lora.install_vision_hooks)[0]
            pf1 = self._scope()
            if not pf1["pass"]:
                self._report({"pf0": self.pf0, "pf1": pf1, "pass": False})
                raise RuntimeError("SINGLE vLLM: PF1 LoRA scope check failed")
            self.lora_dir = Path(tempfile.mkdtemp(prefix="single-vllm-lora-"))
            self._lora_id, self._preflight_done = 0, False
            self.preflight_report: dict | None = None

    def release(self) -> None:
        """Free vLLM weights and KV cache before an HF update (sleep level 2: weights are re-pushed anyway; level 1 in
        LoRA mode: the weight pool also holds the LoRA slots and is restored bit for bit)."""
        if self.sleep and not self.asleep:
            import torch

            self.engine.llm.sleep(level=1 if self.lora else 2)
            self.asleep, self.loaded_policy = True, None
            torch.cuda.empty_cache()

    def sync(self, bundle: Any) -> str:
        from fine_tuning.notebook_rl.policy import merged_named_tensors

        from .rl import policy_sha256

        policy = policy_sha256(bundle)
        if policy != self.loaded_policy:
            if self.asleep:
                import torch

                torch.cuda.empty_cache()
                self.engine.llm.wake_up()
                self.asleep = False
            if self.lora:
                with _nondeterministic():
                    self.vision_hooks.load(llm_lora.vision_lora(bundle))
                    llm_lora.push_mergers(self.engine, merged_named_tensors(bundle), self.engine.policy_version + 1)
                    self._swap_adapter(bundle, policy)
                if not self._preflight_done:
                    self.verify_lora(bundle)
                    self._preflight_done = True
            else:
                with _nondeterministic():
                    self.engine._load_weights(merged_named_tensors(bundle), self.engine.policy_version + 1)
            self.loaded_policy = policy
        return policy

    def _swap_adapter(self, bundle: Any, policy: str, factor: float = 1) -> Any:
        """A fresh, never reused id per policy: vLLM's LRU manager only touches an id it already knows. `factor` != 1
        loads the PF5 amplified probe adapter (B x factor) under its own fresh id."""
        from vllm.lora.request import LoRARequest

        new_id = self._lora_id + 1
        self._lora_id = new_id
        path = self.lora_dir / str(new_id)
        llm_lora.write_llm_adapter(bundle, path, factor)
        core, old = self.engine.llm.llm_engine, self.lora_request
        self.lora_request = None
        if old is not None:
            if core.remove_lora(old.lora_int_id) is not True:
                raise RuntimeError("SINGLE vLLM: previous LoRA not removed")
            shutil.rmtree(old.lora_path)
        name = f"policy-{policy[:16]}" if factor == 1 else f"probe-x{factor:g}-{policy[:16]}"
        request = LoRARequest(name, new_id, str(path))
        if core.add_lora(request) is not True or core.list_loras() != {new_id}:
            raise RuntimeError("SINGLE vLLM: policy LoRA not loaded as the only adapter")
        self.lora_request = request
        return request

    def _report(self, report: dict) -> None:
        self.preflight_report = report
        print("SINGLE_LORA_PREFLIGHT " + json.dumps(report, sort_keys=True, default=str), file=sys.stderr, flush=True)

    def _scope(self) -> dict:
        config = self.engine.model_config
        return llm_lora.check_scope(self.engine.llm, self.vision_hooks, config.text_config.num_hidden_layers,
                                    config.vision_config.depth)

    def _probe(self, request: Any, completion: list[int], lora: Any) -> list[float]:
        """Teacher-forced raw log-probs (engines.probe_logprobs recipe) with or without the policy LoRA."""
        from vllm import SamplingParams

        prepared = self.engine.prepare(request)
        ids = prepared.token_ids + completion
        params = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=0, presence_penalty=0.0, repetition_penalty=1.0,
                                max_tokens=1, seed=request.seeds[0], prompt_logprobs=0, detokenize=False)
        output = self.engine.llm.generate([self.engine._prompt(prepared, ids)], params, use_tqdm=False,
                                          lora_request=lora)[0]
        if list(output.prompt_token_ids) != ids or output.prompt_logprobs is None:
            raise RuntimeError("SINGLE vLLM: probe input mismatch or missing teacher-forced log-probs")
        rows = output.prompt_logprobs[len(prepared.token_ids):]
        return [float(row[t].logprob) for row, t in zip(rows, completion, strict=True)]

    def _pf5(self, bundle: Any) -> tuple[dict, Any]:
        """llm_lora.slope_check on greedy probe completions of the engine as it generates (real and amplified adapter,
        llm_lora.vllm_columns); returns the report and the request of the real adapter restored under a fresh id."""
        from vllm import SamplingParams

        from fine_tuning.notebook_rl.engines import VLLMRolloutEngine

        from .rl import policy_sha256

        pairs = []
        for request in probe_requests(self.image):
            prepared = self.engine.prepare(request)
            greedy = SamplingParams(n=1, temperature=0.0, max_tokens=PROBE_TOKENS, ignore_eos=True, detokenize=False)
            with _nondeterministic():
                output = self.engine.llm.generate([self.engine._prompt(prepared, prepared.token_ids)], greedy,
                                                  use_tqdm=False, lora_request=self.lora_request)[0]
            completion = list(output.outputs[0].token_ids)
            if len(completion) != PROBE_TOKENS:
                raise RuntimeError("SINGLE vLLM: PF5 greedy probe length differs")
            pairs.append((request, completion))
        policy = policy_sha256(bundle)
        with _nondeterministic():
            columns, restored = llm_lora.vllm_columns(bundle, self.vision_hooks, self._probe, pairs, self.lora_request,
                                                      lambda factor: self._swap_adapter(bundle, policy, factor))
        reference = llm_lora.hf_columns(bundle, self.engine, pairs)
        return llm_lora.slope_check(**columns, **reference, cap=VLLMRolloutEngine.sync_threshold_cap), restored

    def verify_lora(self, bundle: Any) -> dict:
        """PF1-PF5 (PF0 ran at start-up); prints SINGLE_LORA_PREFLIGHT {json} and raises on any failure. PF5 tests the
        application mechanism on an amplified adapter (llm_lora.slope_check; the real adapter's slope is a diagnostic),
        then the real adapter returns under a fresh id and PF2 is repeated bit-exactly (pf2_restored)."""
        import torch

        llm = self.engine.llm
        report: dict = {"pf0": self.pf0, "pf1": self._scope(), "lora_int_id": self.lora_request.lora_int_id}
        if report["pf1"]["pass"]:
            lora_id = self.lora_request.lora_int_id
            report["pf2"] = llm_lora.check_identity(llm, bundle, lora_id, self.vision_hooks)
            report["pf3"] = llm_lora.check_base(llm, bundle, self.frozen_fp)
            if self.sleep:
                with _nondeterministic():
                    llm.sleep(level=1)
                    torch.cuda.empty_cache()
                    llm.wake_up()
                identity = llm_lora.check_identity(llm, bundle, lora_id, self.vision_hooks)
                base = llm_lora.check_base(llm, bundle, self.frozen_fp)
                report["pf4"] = {"identity": identity["pass"], "base": base["pass"],
                                 "pass": identity["pass"] and base["pass"]}
            else:
                report["pf4"] = {"pass": None, "reason": "sleep mode off"}
            report["pf5"], restored = self._pf5(bundle)
            report["restored_lora_int_id"] = restored.lora_int_id
            report["pf2_restored"] = llm_lora.check_identity(llm, bundle, restored.lora_int_id, self.vision_hooks)
        report["pass"] = bool(report["pf1"]["pass"]) and all(
            report[k]["pass"] is True for k in ("pf2", "pf3", "pf5", "pf2_restored")) and report["pf4"]["pass"] is not False
        self._report(report)
        if not report["pass"]:
            raise RuntimeError("SINGLE vLLM: LoRA preflight failed")
        return report

    def stream_builder(self, bundle: Any, args: argparse.Namespace, stats: StreamStats) -> Stream:
        from vllm import SamplingParams

        from fine_tuning.notebook_rl.engines import RolloutRequest
        from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs

        from .rl import digest, sampling_sha256

        policy, protocol = args.policy_sha256, args.protocol_sha256
        if self.sync(bundle) != policy:
            raise ValueError("SINGLE vLLM: stale live policy")
        engine, sampling_hash = self.engine, sampling_sha256(SINGLE_SAMPLING)
        label = llm_lora.ENGINE_LORA if self.lora else ENGINE
        stats.receipt_extra.update(single_sampler=label, single_batch_rows=self.max_num_seqs)
        eos = set(self.eos_token_ids)

        def prepare(request: dict) -> tuple[dict, Any, dict]:
            binding = single_request_binding(request)
            item, _ = prepare_protocol_inputs(engine.processor, engine.model_config, request["messages"],
                                              request["images"], self.image)
            if item["input_tokens"] + request["max_tokens"] > CONTEXT_LIMIT:
                raise ValueError("SINGLE vLLM: prompt plus completion exceeds the context limit")
            prepared = engine.prepare(RolloutRequest(binding["request_id"], None, request["messages"], request["images"],
                                                     1, request["max_tokens"], [request["seed"]]))
            if prepared.token_ids != list(item["prompt_token_ids"]):
                raise RuntimeError("SINGLE vLLM: engine prompt IDs differ from the HF training input")
            params = SamplingParams(n=1, temperature=SINGLE_SAMPLING["temperature"], top_p=SINGLE_SAMPLING["top_p"],
                                    top_k=SINGLE_SAMPLING["top_k"], presence_penalty=SINGLE_SAMPLING["presence_penalty"],
                                    repetition_penalty=1.0, max_tokens=request["max_tokens"], seed=request["seed"],
                                    stop_token_ids=self.eos_token_ids, logprobs=TOP_LOGPROBS,
                                    detokenize=False)  # text is decoded from the action IDs in finish()
            return engine._prompt(prepared, prepared.token_ids), params, item

        def finish(output: Any, request: dict, item: dict) -> dict:
            if len(output.outputs) != 1 or list(output.prompt_token_ids) != list(item["prompt_token_ids"]):
                raise RuntimeError("SINGLE vLLM: one completion on the exact HF prompt IDs required")
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            if completion.logprobs is None or len(completion.logprobs) != len(ids):
                raise RuntimeError("SINGLE vLLM: one processed log-probability row per action required")
            reason = completion.finish_reason
            if reason not in ("stop", "length") or (reason == "stop" and (not ids or ids[-1] not in eos)):
                raise RuntimeError(f"SINGLE vLLM: unsupported finish {reason!r} or stop without an EOS action")
            logprobs, supports = [], []
            for token, row in zip(ids, completion.logprobs, strict=True):
                value, support = support_from_logprobs(token, row)
                logprobs.append(value)
                supports.append(support)
            text = engine.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generation = single_request_binding(request) | {
                "prompt_token_ids": list(item["prompt_token_ids"]), "completion_token_ids": ids,
                "sampling_logprobs": logprobs, "support_sha256": [digest(s) for s in supports],
                "support_token_ids": supports, "sampling": dict(SINGLE_SAMPLING), "policy_sha256": policy,
                "protocol_sha256": protocol, "sampling_sha256": sampling_hash, "input_sha256": item["input_sha256"],
                "finish_reason": reason, "request_sha256": single_request_sha256(request), "text": text,
                "eos_token_ids": self.eos_token_ids, "completion_count": 1, "engine": label}
            return {"text": text, "finish_reason": reason, "prompt_tokens": len(item["prompt_token_ids"]),
                    "completion_tokens": len(ids), "completion_count": 1, "generation": generation}

        def stream(requests: Iterable[dict], stats_: StreamStats, stop_after: float | None) -> Iterator[tuple[dict, dict]]:
            if self.sync(bundle) != policy:
                raise ValueError("SINGLE vLLM: policy changed before generation")
            core = engine.llm.llm_engine
            if self.lora:
                if self.lora_request is None:
                    raise RuntimeError("SINGLE vLLM: LoRA mode without a loaded policy adapter")
                core = _LoraEngine(core, self.lora_request)
            with _nondeterministic():
                yield from continuous_stream(core, requests, self.max_num_seqs, prepare, finish,
                                             stats_, stop_after=stop_after)

        setattr(stream, "continuous", True)
        return stream
