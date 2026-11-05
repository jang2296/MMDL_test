"""SINGLE v3 generation on an in-process vLLM 0.11 engine: continuous batching, policy pushed after every update.

The HF/PEFT bundle stays the training authority. Before a block generates, its LoRA-merged BF16 weights (Merger
masters rounded to BF16, as in the eval export) are pushed with the existing complete-inventory loader of
notebook_rl.engines.VLLMRolloutEngine. vLLM's processed log-probabilities (presence penalty -> temperature ->
top-k/top-p, the order rl.processed_scores uses) give each sampled action's log q and the support it was drawn
from; every block's training replay checks them against the HF policy (protocol replay bounds). Sleep mode
level 2 frees vLLM weights and KV cache while the HF policy trains; the next push restores complete weights.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

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


class SingleVllmSampler:
    """One engine for the whole campaign; `stream_builder` matches collect.hf_single_stream_builder's signature."""

    def __init__(self, model_path: Path, *, kv_cache_gib: float, max_num_seqs: int, gpu_memory_utilization: float,
                 sleep: bool, eos_token_ids: Iterable[int]):
        from importlib import import_module
        from importlib.metadata import version

        from fine_tuning.notebook_rl.engines import InputPreparation, VLLMRolloutEngine, _weight_inventory, vllm_kwargs

        if version("vllm") != "0.11.0":
            raise RuntimeError("SINGLE vLLM: vllm==0.11.0 required")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # in-process engine: weight push and add_request/step
        from transformers import AutoConfig, AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
                                                  min_pixels=IMAGE["min_pixels"], max_pixels=IMAGE["max_pixels"])
        processor.image_processor.size = {"shortest_edge": IMAGE["min_pixels"], "longest_edge": IMAGE["max_pixels"]}
        engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)  # reuse prepare/_prompt/_load_weights, own LLM kwargs
        InputPreparation.__init__(engine, processor, AutoConfig.from_pretrained(str(model_path), local_files_only=True),
                                  IMAGE, CONTEXT_LIMIT)
        engine._weight_shapes = _weight_inventory(engine.model_config)
        engine.model_dir = Path(model_path).resolve()
        kwargs = vllm_kwargs(Path(model_path), IMAGE, int(kv_cache_gib * 2**30), max_num_seqs, CONTEXT_LIMIT,
                             gpu_memory_utilization=float(gpu_memory_utilization))
        with _nondeterministic():
            engine.llm = import_module("vllm").LLM(**kwargs, logprobs_mode="processed_logprobs",
                                                   max_logprobs=TOP_LOGPROBS, enable_sleep_mode=bool(sleep))
        engine._call, engine._synchronized, engine._push_complete, engine._last_loaded_version = 0, True, True, None
        self.engine, self.max_num_seqs, self.sleep = engine, max_num_seqs, bool(sleep)
        self.eos_token_ids = sorted(set(eos_token_ids))
        self.loaded_policy: str | None = None  # the disk snapshot is F0, but a push always precedes generation
        self.asleep = False

    def release(self) -> None:
        """Free vLLM weights and KV cache before an HF update (sleep level 2: weights are re-pushed anyway)."""
        if self.sleep and not self.asleep:
            import torch

            self.engine.llm.sleep(level=2)
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
            with _nondeterministic():
                self.engine._load_weights(merged_named_tensors(bundle), self.engine.policy_version + 1)
            self.loaded_policy = policy
        return policy

    def stream_builder(self, bundle: Any, args: argparse.Namespace, stats: StreamStats) -> Stream:
        from vllm import SamplingParams

        from fine_tuning.notebook_rl.engines import RolloutRequest
        from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs

        from .rl import digest, sampling_sha256

        policy, protocol = args.policy_sha256, args.protocol_sha256
        if self.sync(bundle) != policy:
            raise ValueError("SINGLE vLLM: stale live policy")
        engine, sampling_hash = self.engine, sampling_sha256(SINGLE_SAMPLING)
        stats.receipt_extra.update(single_sampler=ENGINE, single_batch_rows=self.max_num_seqs)
        eos = set(self.eos_token_ids)

        def prepare(request: dict) -> tuple[dict, Any, dict]:
            binding = single_request_binding(request)
            item, _ = prepare_protocol_inputs(engine.processor, engine.model_config, request["messages"],
                                              request["images"], IMAGE)
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
                "eos_token_ids": self.eos_token_ids, "completion_count": 1, "engine": ENGINE}
            return {"text": text, "finish_reason": reason, "prompt_tokens": len(item["prompt_token_ids"]),
                    "completion_tokens": len(ids), "completion_count": 1, "generation": generation}

        def stream(requests: Iterable[dict], stats_: StreamStats, stop_after: float | None) -> Iterator[tuple[dict, dict]]:
            if self.sync(bundle) != policy:
                raise ValueError("SINGLE vLLM: policy changed before generation")
            with _nondeterministic():
                yield from continuous_stream(engine.llm.llm_engine, requests, self.max_num_seqs, prepare, finish,
                                             stats_, stop_after=stop_after)

        setattr(stream, "continuous", True)
        return stream
