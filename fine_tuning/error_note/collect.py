"""Collect stage of the error-note path: one gold-blind solve per item, one note per item with the gold visible, rows.

Plan: claudedocs/research_self_note_finetuning_plan_20261101.md section 3.
CLI: python -m fine_tuning.error_note.collect {solve,note,sweep,rows} ... (see _parser). Heavy imports stay inside functions.
Generation is a continuous add_request/step loop with the evaluation backend's prompt shape and sampling arguments;
every completion is appended to its JSONL immediately, so a killed run resumes by skipping the ids already written.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from fine_tuning.notebook_rl.prompts import p0_eval_text, user_message

from .contract import (CONTEXT_LIMIT, IMAGE, MAX_NEW_TOKENS, TURN1_ANSWER_CONTRACT, build_rows, note_prompt, parse_note,
                       sample_seed)
from .order import read_order

PREFETCH_WORKERS = 8
THROUGHPUT_EVERY = 60.0
SKIPPED_T1_STATUSES = ("UNVERIFIABLE", "CONTEXT_BLOCKED")
MIB = 1024 * 1024
# Auto sizing (--kv-cache-gib auto, --seqs auto) from measured Qwen3-VL-4B BF16 vLLM runs; keep these numbers in sync with them.
WEIGHTS_GIB = 8.68            # model weights
OTHER_VLLM_GIB = 5.0          # other vLLM memory at 32-48 concurrency (A40 peak 41.66 GiB with KV 28 GiB)
KV_TOKENS_PER_GIB = 7280      # 8 GiB of KV cache gave 58,240 tokens
MEAN_CONTEXT_TOKENS = 4800    # time-weighted mean context of a running sequence
AUTO_RESERVE_GIB = 16         # kept outside the KV cache: WEIGHTS_GIB + OTHER_VLLM_GIB plus about 2.3 GiB of margin
MIN_AUTO_KV_GIB = 8
SEQS_STEP = 8
SEQS_FACTORS = (0.75, 1.0, 1.25)


class InputTooLong(ValueError):
    """The prepared prompt plus the output cap exceeds the context limit; the item is recorded, never truncated."""


@dataclass
class StreamStats:
    steps: int = 0
    step_seconds: float = 0.0
    active_sum: int = 0
    startup_seconds: float = 0.0
    stopped_early: bool = False
    metrics: Callable[[], dict | None] | None = None  # set by the engine builder: {"generation_tokens", "preemptions"}
    on_step: Callable[[], None] | None = None
    receipt_extra: dict = field(default_factory=dict)  # merged into the final receipt (resolved KV size and its source)


Stream = Callable[[Iterable[dict], StreamStats, float | None], Iterator[tuple[dict, dict]]]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ files and small utilities

def append_jsonl(path: Path, row: dict) -> None:
    data = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    with open(path, "ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl_done(path: Path, *, repair: bool = True) -> dict[str, dict]:
    """{problem_id: row}. A final line cut by a kill is dropped with a warning (and cut from the file when repair)."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = path.read_bytes()
    *lines, tail = raw.split(b"\n")  # tail is b"" when the file ends with a newline
    rows = [json.loads(line) for line in lines if line.strip()]
    if tail.strip():
        try:
            rows.append(json.loads(tail))
            if repair:
                with open(path, "ab") as stream:
                    stream.write(b"\n")
        except ValueError:
            warnings.warn(f"{path}: dropped a truncated final line of {len(tail)} bytes", RuntimeWarning)
            if repair:
                with open(path, "r+b") as stream:
                    stream.truncate(len(raw) - len(tail))
    done: dict[str, dict] = {}
    for row in rows:
        if row["problem_id"] in done:
            raise ValueError(f"{path}: duplicate problem_id {row['problem_id']}")
        done[row["problem_id"]] = row
    return done


def write_atomic(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    write_atomic(path, json.dumps(value, indent=1, ensure_ascii=False) + "\n")


def bounded_map(fn: Callable, iterable: Iterable, *, workers: int, buffer: int) -> Iterator:
    """fn over iterable on `workers` threads, results in input order, at most `buffer` futures outstanding."""
    if workers < 1 or buffer < 1:
        raise ValueError("workers and buffer must be at least 1")
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        pending: collections.deque = collections.deque()
        try:
            for item in iterable:
                pending.append(pool.submit(fn, item))
                if len(pending) >= buffer:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


def _gpu_sample() -> tuple[int | None, int | None, int | None]:
    """(utilization percent, memory used MiB, memory total MiB) of the first GPU; nulls when nvidia-smi is unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5)
        util, used, total = (int(v) for v in result.stdout.splitlines()[0].split(","))
        return util, used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None, None


def _cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat") as stream:
            values = [int(v) for v in stream.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    return sum(values), values[3] + (values[4] if len(values) > 4 else 0)  # total, idle + iowait


def _mem_available_kib() -> int | None:
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


class ResourceLog:
    """Background thread: one JSON line per interval (GPU, CPU, load, free RAM), plus one at start and one at exit."""

    def __init__(self, path: Path, every: float = 60.0):
        self.path, self.every, self._prev = Path(path), every, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        cpu = _cpu_times()
        percent = None
        if cpu and self._prev and cpu[0] > self._prev[0]:
            percent = round(100 * (1 - (cpu[1] - self._prev[1]) / (cpu[0] - self._prev[0])), 1)
        self._prev = cpu
        util, used, total = _gpu_sample()
        try:
            load = list(os.getloadavg())
        except OSError:
            load = None
        append_jsonl(self.path, {"utc": _utc(), "gpu_util_percent": util, "gpu_mem_used_mib": used,
                                 "gpu_mem_total_mib": total, "cpu_percent": percent, "loadavg": load,
                                 "mem_available_kib": _mem_available_kib()})

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.every):
            self._sample()

    def __enter__(self) -> "ResourceLog":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


# ------------------------------------------------------------------ guards (build-root mode of passrate._install_guards)

def install_training_guards(build_root: Path, *, vllm_schema: dict | None = None) -> list[Path]:
    """Lock checks, optional vLLM schema step, HF imports, then the file guards. Returns the forbidden benchmark roots.

    vllm_schema: a dict that receives "schema_dependency_guard" (pass {} when vLLM will be imported); None for no vLLM.
    """
    from importlib import import_module

    from fine_tuning.notebook_rl.dataset import SPLITS
    from fine_tuning.notebook_rl.guards import install_guards, prepare_vllm_schema_dependency
    from fine_tuning.notebook_rl.quantized_policy import _versions
    from fine_tuning.notebook_rl.shared_smoke import _prepare_hf_dependencies

    _versions()
    if vllm_schema is not None:
        vllm_schema["schema_dependency_guard"] = prepare_vllm_schema_dependency()
        # Imported before the file guards: a first import after them dies on any hard-linked stdlib or package file.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        for module in ("vllm", "vllm.sampling_params", "logging.config"):
            import_module(module)
    _prepare_hf_dependencies()
    import_module("bitsandbytes")
    hubs = {Path.home() / "mmdl-cache/huggingface/hub",
            Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")}
    forbidden = [Path.home() / "mmdl-data/evaluation",
                 *(hub / name for hub in sorted(hubs) for name in ("datasets--MMMU--MMMU", "datasets--MMMU--MMMU_Pro"))]
    vaults = [Path(build_root) / split / name for split in SPLITS for name in ("verifier_private.jsonl", "lineage.jsonl")]
    install_guards(forbidden_roots=forbidden, vault_files=vaults, allow_vault=False, allow_benchmark=False)
    return forbidden


# ------------------------------------------------------------------ continuous generation

def continuous_stream(engine: Any, requests: Iterable[dict], limit: int, prepare: Callable, finish: Callable,
                      stats: StreamStats, *, stop_after: float | None = None,
                      clock: Callable[[], float] = time.monotonic) -> Iterator[tuple[dict, dict]]:
    """Keep `limit` requests active on `engine` (add_request/abort_request/step), yield (request, output dict) per finish.

    prepare(request) -> (prompt, params, info), raising InputTooLong to record the item without generating.
    finish(engine_output, request, info) -> output dict. Request dicts need a unique "request_id".
    After `stop_after` seconds the active requests are aborted and the stream ends with stats.stopped_early.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    stop_at = None if stop_after is None else clock() + stop_after
    iterator, active, blocked, seen = iter(requests), {}, collections.deque(), set()
    exhausted = False

    def fill() -> None:
        nonlocal exhausted
        while not exhausted and len(active) < limit:
            request = next(iterator, None)
            if request is None:
                exhausted = True
                return
            request_id = request["request_id"]
            if request_id in seen:
                raise ValueError(f"duplicate request_id {request_id}")
            seen.add(request_id)
            try:
                prompt, params, info = prepare(request)
            except InputTooLong as exc:
                blocked.append((request, {"blocked": True, "error": str(exc)}))
                continue
            engine.add_request(request_id, prompt, params)
            active[request_id] = {"request": request, "info": info, "seconds": 0.0}

    try:
        fill()
        while True:
            while blocked:
                yield blocked.popleft()
            if not active:
                break
            if stop_at is not None and clock() >= stop_at:
                stats.stopped_early = True
                break
            count, began = len(active), clock()
            outputs = engine.step()
            elapsed = clock() - began
            stats.steps += 1
            stats.step_seconds += elapsed
            stats.active_sum += count
            for state in active.values():
                state["seconds"] += elapsed / count
            for output in outputs:
                state = active.get(output.request_id)
                if state is None:
                    raise RuntimeError(f"engine returned an unknown request id: {output.request_id}")
                if output.finished:
                    del active[output.request_id]
                    row = finish(output, state["request"], state["info"])
                    row["seconds"] = state["seconds"]
                    yield state["request"], row
            if stats.on_step is not None:
                stats.on_step()
            fill()
    finally:
        if active:
            handling_error = sys.exc_info()[0] is not None
            try:
                engine.abort_request(list(active))
            except Exception as exc:
                if not handling_error:
                    raise
                warnings.warn(f"Failed to abort outstanding requests: {exc}", RuntimeWarning)


def run_generation(stage: str, out: Path, requests: Iterable[dict], stream: Stream, stats: StreamStats,
                   handle: Callable[[dict, dict], dict], *, total: int, S: int, B: int, engine: str,
                   stop_after: float | None = None, every: float = THROUGHPUT_EVERY,
                   gpu_probe: Callable | None = None) -> dict:
    """Append handle(request, output) rows to <stage>.jsonl as they finish; throughput.jsonl every `every` seconds;
    receipt-<stage>.json at exit (also after an error or a stop deadline)."""
    out = Path(out)
    counts = {"done": 0, "tokens": 0, "length": 0}
    peak: list[int | None] = [None]
    last = [time.monotonic()]

    def snapshot() -> dict:
        probe = gpu_probe or _gpu_sample
        used = probe()[1]
        if used is not None:
            peak[0] = used if peak[0] is None else max(peak[0], used)
        try:
            metrics = stats.metrics() if stats.metrics else None
        except Exception:
            metrics = None
        generated = metrics["generation_tokens"] if metrics else stats.active_sum
        seconds = stats.step_seconds
        return {"utc": _utc(), "stage": stage, "engine": engine, "S": S, "B": B, "items_total": total,
                "items_done": counts["done"], "finished_tokens": counts["tokens"], "generation_seconds": seconds,
                "startup_seconds": stats.startup_seconds,
                "tokens_per_second": generated / seconds if seconds > 0 else None,
                "finished_tokens_per_second": counts["tokens"] / seconds if seconds > 0 else None,
                "steps": stats.steps, "mean_step_ms": 1000 * seconds / stats.steps if stats.steps else None,
                "mean_active": stats.active_sum / stats.steps if stats.steps else None,
                "preemptions": metrics["preemptions"] if metrics else None,
                "generation_tokens": metrics["generation_tokens"] if metrics else None,
                "token_source": "vllm_metrics" if metrics else "active_times_steps",
                "peak_gpu_mem_mib": peak[0], "length_finishes": counts["length"], "stopped_early": stats.stopped_early}

    def tick(force: bool = False) -> None:
        now = time.monotonic()
        if force or now - last[0] >= every:
            last[0] = now
            append_jsonl(out / "throughput.jsonl", snapshot())

    stats.on_step = tick
    error = None
    try:
        iterator = iter(requests)
        first = next(iterator, None)
        if first is not None:  # nothing to generate means no engine is ever built
            with contextlib.closing(stream(itertools.chain([first], iterator), stats, stop_after)) as outputs:
                for request, output in outputs:
                    row = handle(request, output)
                    append_jsonl(out / f"{stage}.jsonl", row)
                    counts["done"] += 1
                    if not output.get("blocked"):
                        counts["tokens"] += output["completion_tokens"]
                        counts["length"] += output["finish_reason"] == "length"
                    tick()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        tick(force=True)
        receipt = snapshot() | stats.receipt_extra | {"completed": error is None and not stats.stopped_early, "error": error}
        write_json(out / f"receipt-{stage}.json", receipt)
    return receipt


# ------------------------------------------------------------------ solve and note orchestration

def _blocked_fields(output: dict) -> dict:
    return {"status": "CONTEXT_BLOCKED", "error": output["error"]}


def run_solve(items: list[dict], actors: dict, load: Callable, stream: Stream, stats: StreamStats, grade: Callable,
              out: Path, *, S: int, B: int, engine: str, stop_after: float | None = None,
              every: float = THROUGHPUT_EVERY, gpu_probe: Callable | None = None) -> dict:
    """Gold-blind turn 1. grade(problem_id, text, finish_reason, mode) -> (status, extracted); there is no gold here."""
    out = Path(out)
    done = read_jsonl_done(out / "t1.jsonl")
    todo = [item["problem_id"] for item in items if item["problem_id"] not in done]

    def requests() -> Iterator[dict]:
        for actor, images in bounded_map(lambda pid: (actors[pid], load(actors[pid])), todo,
                                         workers=PREFETCH_WORKERS, buffer=2 * S):
            yield {"request_id": actor.problem_id, "messages": [user_message(p0_eval_text(actor.question, actor.choices), len(images))],
                   "images": images, "seed": sample_seed(actor.problem_id, "t1"), "max_tokens": MAX_NEW_TOKENS,
                   "mode": actor.mode}

    def handle(request: dict, output: dict) -> dict:
        row = {"problem_id": request["request_id"], "stage": "t1", "seed": request["seed"]}
        if output.get("blocked"):
            return row | _blocked_fields(output) | {"prompt_tokens": None, "completion_tokens": 0, "extracted": None,
                                                    "seconds": 0.0, "engine": engine}
        status, extracted = grade(request["request_id"], output["text"], output["finish_reason"], request["mode"])
        return row | {"prompt_tokens": output["prompt_tokens"], "completion_tokens": output["completion_tokens"],
                      "finish_reason": output["finish_reason"], "text": output["text"], "status": status,
                      "extracted": extracted, "seconds": output["seconds"], "engine": engine}

    return run_generation("t1", out, requests(), stream, stats, handle, total=len(todo), S=S, B=B, engine=engine,
                          stop_after=stop_after, every=every, gpu_probe=gpu_probe)


def run_note(items: list[dict], actors: dict, t1: dict[str, dict], gold: dict[str, str], load: Callable, stream: Stream,
             stats: StreamStats, grade: Callable, out: Path, *, S: int, B: int, engine: str,
             stop_after: float | None = None, every: float = THROUGHPUT_EVERY, gpu_probe: Callable | None = None) -> dict:
    """Turn 2: the note prompt shows the gold. Items without a usable T1 get a SKIPPED row and no generation."""
    out = Path(out)
    ids = [item["problem_id"] for item in items]
    missing = [pid for pid in ids if pid not in t1]
    if missing:
        raise ValueError(f"no T1 row for {len(missing)} items, first {missing[:3]}")
    done = read_jsonl_done(out / "note.jsonl")
    for pid in ids:
        if pid not in done and t1[pid]["status"] in SKIPPED_T1_STATUSES:
            done[pid] = {"problem_id": pid, "stage": "note", "status": "SKIPPED_" + t1[pid]["status"]}
            append_jsonl(out / "note.jsonl", done[pid])
    todo = [pid for pid in ids if pid not in done]

    def requests() -> Iterator[dict]:
        for actor, images in bounded_map(lambda pid: (actors[pid], load(actors[pid])), todo,
                                         workers=PREFETCH_WORKERS, buffer=2 * S):
            pid = actor.problem_id
            prompt = note_prompt(p0_eval_text(actor.question, actor.choices), t1[pid]["text"], t1[pid]["status"], gold[pid])
            yield {"request_id": pid, "messages": [user_message(prompt, len(images))], "images": images,
                   "seed": sample_seed(pid, "note"), "max_tokens": MAX_NEW_TOKENS, "prompt": prompt, "mode": actor.mode}

    def handle(request: dict, output: dict) -> dict:
        pid = request["request_id"]
        row = {"problem_id": pid, "stage": "note", "seed": request["seed"]}
        if output.get("blocked"):
            return row | _blocked_fields(output) | {"prompt": request["prompt"], "engine": engine}
        parsed = parse_note(output["text"])
        corrected = grade(pid, parsed.corrected_text, output["finish_reason"], request["mode"]) if parsed else (None, None)
        return row | {"prompt": request["prompt"], "prompt_tokens": output["prompt_tokens"],
                      "completion_tokens": output["completion_tokens"], "finish_reason": output["finish_reason"],
                      "text": output["text"], "image_reading": parsed.image_reading if parsed else None,
                      "cause": parsed.cause if parsed else None, "corrected_status": corrected[0],
                      "corrected_extracted": corrected[1], "seconds": output["seconds"], "engine": engine}

    return run_generation("note", out, requests(), stream, stats, handle, total=len(todo), S=S, B=B, engine=engine,
                          stop_after=stop_after, every=every, gpu_probe=gpu_probe)


# ------------------------------------------------------------------ auto sizing from the GPU's memory

def gpu_total_gib() -> float:
    """Total memory of GPU 0 in GiB from nvidia-smi, so the calling process holds no CUDA context."""
    total_mib = _gpu_sample()[2]
    if total_mib is None:
        raise RuntimeError("--kv-cache-gib auto: nvidia-smi did not report the GPU memory")
    return total_mib / 1024


def auto_kv_gib(total_gib: float) -> int:
    kv = math.floor(total_gib - AUTO_RESERVE_GIB)
    if kv < MIN_AUTO_KV_GIB:
        raise ValueError(f"--kv-cache-gib auto: a {total_gib:.1f} GiB GPU leaves {kv} GiB for the KV cache, "
                         f"below the minimum of {MIN_AUTO_KV_GIB}")
    return kv


def auto_seqs(kv_gib: int) -> list[int]:
    """Sweep candidates around the concurrency the KV cache holds at the mean context: multiples of 8, sorted, distinct."""
    estimate = kv_gib * KV_TOKENS_PER_GIB / MEAN_CONTEXT_TOKENS
    return sorted({max(SEQS_STEP, round(factor * estimate) // SEQS_STEP * SEQS_STEP) for factor in SEQS_FACTORS})


def resolve_kv(kv_arg: float | str, total_gib: Callable[[], float] = gpu_total_gib) -> dict:
    """The GPU is read only for "auto"."""
    if kv_arg != "auto":
        return {"gpu_total_gib": None, "kv_cache_gib_resolved": kv_arg, "kv_source": "fixed"}
    total = total_gib()
    return {"gpu_total_gib": total, "kv_cache_gib_resolved": auto_kv_gib(total), "kv_source": "auto"}


# ------------------------------------------------------------------ engines (GPU; not exercised by the unit tests)

def vllm_metrics(llm: Any) -> dict | None:
    try:
        values = {m.name: m.value for m in llm.get_metrics() if hasattr(m, "value")}
        return {"generation_tokens": int(values["vllm:generation_tokens"]),
                "preemptions": int(values.get("vllm:num_preemptions", 0))}
    except Exception:
        return None


def _vllm_stream_builder(args: argparse.Namespace, stats: StreamStats) -> Stream:
    def stream(requests: Iterable[dict], stats_: StreamStats, stop_after: float | None) -> Iterator[tuple[dict, dict]]:
        kv = resolve_kv(args.kv_cache_gib)  # before the heavy imports: a GPU too small for "auto" fails fast
        stats_.receipt_extra |= kv
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from importlib.metadata import version

        from transformers import AutoConfig, AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind

        from fine_tuning.notebook_rl.engines import QWEN_OFFICIAL, vllm_kwargs
        from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs

        if version("vllm") != "0.11.0":
            raise RuntimeError("collect: vllm==0.11.0 required")
        began = time.monotonic()
        model_path = str(args.model_path)
        kwargs = vllm_kwargs(Path(model_path), IMAGE, int(kv["kv_cache_gib_resolved"] * 2**30), args.max_num_seqs, CONTEXT_LIMIT,
                             gpu_memory_utilization=float(args.gpu_memory_utilization))
        kwargs |= {"max_num_batched_tokens": args.max_num_batched_tokens, "disable_log_stats": False}
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, min_pixels=IMAGE["min_pixels"],
                                                  max_pixels=IMAGE["max_pixels"])
        processor.image_processor.size = {"shortest_edge": IMAGE["min_pixels"], "longest_edge": IMAGE["max_pixels"]}
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        llm = LLM(**kwargs)
        stats.startup_seconds = time.monotonic() - began
        stats.metrics = lambda: vllm_metrics(llm)

        def prepare(request: dict) -> tuple[dict, Any, int]:
            item, images = prepare_protocol_inputs(processor, config, request["messages"], request["images"], IMAGE)
            if item["input_tokens"] + request["max_tokens"] > CONTEXT_LIMIT:
                raise InputTooLong(f"{item['input_tokens']} input + {request['max_tokens']} output tokens exceed {CONTEXT_LIMIT}")
            # The same arguments as the evaluation backend's _sampling_params; only the seed and the cap differ per call.
            params = SamplingParams(n=1, temperature=QWEN_OFFICIAL.temperature, top_p=QWEN_OFFICIAL.top_p,
                                    top_k=QWEN_OFFICIAL.top_k, repetition_penalty=QWEN_OFFICIAL.repetition_penalty,
                                    presence_penalty=QWEN_OFFICIAL.presence_penalty, max_tokens=request["max_tokens"],
                                    seed=request["seed"], output_kind=RequestOutputKind.FINAL_ONLY)
            return {"prompt": item["chat_prompt"], "multi_modal_data": {"image": images}}, params, item["input_tokens"]

        def finish(output: Any, request: dict, info: int) -> dict:
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM did not return exactly one sampled completion")
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            return {"text": processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    "finish_reason": completion.finish_reason or "unknown",
                    "prompt_tokens": len(output.prompt_token_ids), "completion_tokens": len(ids)}

        yield from continuous_stream(llm.llm_engine, requests, args.max_num_seqs, prepare, finish, stats_,
                                     stop_after=stop_after)

    return stream


class GeneratedOnlyPresencePenalty:
    """src/mmdl/runtime/generation.py GeneratedOnlyPresencePenalty (copied: fine_tuning must work alone)."""

    def __init__(self, prompt_length: int, penalty: float):
        if prompt_length < 0 or penalty < 0:
            raise ValueError("Invalid prompt length or penalty")
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        generated = input_ids[:, self.prompt_length:]
        if generated.numel() == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return scores - seen.to(scores.dtype) * self.penalty


def _hf_stream_builder(args: argparse.Namespace, stats: StreamStats) -> Stream:
    """Local NF4 functional runs: one sequence at a time, like pod/gpu_smoke.py."""
    def stream(requests: Iterable[dict], stats_: StreamStats, stop_after: float | None) -> Iterator[tuple[dict, dict]]:
        import torch
        from transformers import GenerationConfig, LogitsProcessorList

        from fine_tuning.error_note.protocol import ARM, plain_protocol
        from fine_tuning.notebook_rl.engines import QWEN_OFFICIAL as SAMPLING
        from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs
        from fine_tuning.notebook_rl.quantized_policy import build_quantized_policy

        began = time.monotonic()
        bundle = build_quantized_policy(Path(args.model_path), ARM, plain_protocol("nf4"))
        model, processor = bundle.model.eval(), bundle.processor
        stats_.startup_seconds = time.monotonic() - began
        base = model.generation_config
        eos_ids = set(base.eos_token_id) if isinstance(base.eos_token_id, list) else {base.eos_token_id}
        stop_at = None if stop_after is None else time.monotonic() + stop_after
        for request in requests:
            if stop_at is not None and time.monotonic() >= stop_at:
                stats_.stopped_early = True
                return
            item, _ = prepare_protocol_inputs(processor, model.config, request["messages"], request["images"], IMAGE)
            length = item["input_tokens"]
            if length + request["max_tokens"] > CONTEXT_LIMIT:
                yield request, {"blocked": True, "error": f"{length} input + {request['max_tokens']} output tokens exceed {CONTEXT_LIMIT}"}
                continue
            config = GenerationConfig(
                do_sample=True, num_beams=1, temperature=SAMPLING.temperature, top_p=SAMPLING.top_p, top_k=SAMPLING.top_k,
                repetition_penalty=SAMPLING.repetition_penalty, max_new_tokens=request["max_tokens"],
                eos_token_id=base.eos_token_id, pad_token_id=base.pad_token_id, use_cache=True)
            torch.manual_seed(request["seed"])
            began = time.monotonic()
            # The Merger masters are FP32 while the frozen base computes in BF16 (policy._logprobs uses the same autocast).
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    **item["inputs"].to("cuda:0"), generation_config=config, use_model_defaults=False,
                    logits_processor=LogitsProcessorList([GeneratedOnlyPresencePenalty(length, SAMPLING.presence_penalty)]))
            torch.cuda.synchronize()
            seconds = time.monotonic() - began
            ids = getattr(generated, "sequences", generated)[0, length:].tolist()
            finish_reason = "stop" if ids and ids[-1] in eos_ids else "length"
            if finish_reason == "stop":
                ids = ids[:-1]
            stats_.steps += len(ids)
            stats_.step_seconds += seconds
            stats_.active_sum += len(ids)
            if stats_.on_step is not None:
                stats_.on_step()
            yield request, {"text": processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                            "finish_reason": finish_reason, "prompt_tokens": length, "completion_tokens": len(ids),
                            "seconds": seconds}

    return stream


# ------------------------------------------------------------------ sweep

def _child_command(args: argparse.Namespace, seqs: int, stop_after: float | None) -> list[str]:
    command = [sys.executable, "-m", "fine_tuning.error_note.collect", "solve",
               "--build-root", str(args.build_root), "--order", str(args.order), "--limit", str(args.limit),
               "--model-path", str(args.model_path), "--output", str(args.output), "--engine", args.engine,
               "--kv-cache-gib", str(args.kv_cache_gib), "--max-num-batched-tokens", str(args.max_num_batched_tokens),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--max-num-seqs", str(seqs)]
    return command + ([] if stop_after is None else ["--stop-after-seconds", str(stop_after)])


def _run_child(command: list[str]) -> int:
    return subprocess.run(command).returncode


def sweep(args: argparse.Namespace, run_child: Callable[[list[str]], int] = _run_child,
          total_gib: Callable[[], float] = gpu_total_gib) -> int:
    """One short solve child per --seqs value (or "auto": candidates from the KV size), then a final child with the
    fastest S that runs the rest to completion. The children get --kv-cache-gib as given and resolve it themselves."""
    out, items = Path(args.output), _prefix(args)
    kv = resolve_kv(args.kv_cache_gib, total_gib)
    seqs_values = (auto_seqs(int(kv["kv_cache_gib_resolved"])) if args.seqs == "auto"
                   else [int(v) for v in args.seqs.split(",")])
    out.mkdir(parents=True, exist_ok=True)
    total, ids = len(items), {item["problem_id"] for item in items}

    def finished() -> int:
        return len(ids & set(read_jsonl_done(out / "t1.jsonl", repair=False)))

    runs: list[dict] = []
    for seqs in seqs_values:
        if finished() >= total:
            break  # everything finished early; the remaining values are skipped
        (out / "receipt-t1.json").unlink(missing_ok=True)
        code = run_child(_child_command(args, seqs, args.seconds))
        if code:
            return code
        receipt = json.loads((out / "receipt-t1.json").read_text())
        write_json(out / f"receipt-t1-s{seqs}.json", receipt)
        runs.append(receipt)
    measured = [run for run in runs if run.get("tokens_per_second") is not None]
    best = max(measured, key=lambda run: (run["tokens_per_second"], -run["S"]))["S"] if measured else None
    result = {"runs": runs, "best_seqs": best} | kv | {"seqs_resolved": seqs_values}
    write_json(out / "sweep.json", result)
    if best is not None and finished() < total:
        code = run_child(_child_command(args, best, None))
        if code:
            return code
        result["final"] = json.loads((out / "receipt-t1.json").read_text())
        write_json(out / "sweep.json", result)
    return 0


# ------------------------------------------------------------------ rows

def make_records(items: list[dict], actors: dict, t1: dict[str, dict], note: dict[str, dict]) -> list[dict]:
    """The inputs of contract.build_rows. A note row with a status (SKIPPED / CONTEXT_BLOCKED) counts as no note."""
    records = []
    for item in items:
        pid = item["problem_id"]
        if pid not in t1:
            raise ValueError(f"no T1 row for {pid}")
        actor, first, second = actors[pid], t1[pid], note.get(pid)
        record = {"problem_id": pid, "subject": item["subject"], "source": item["source"],
                  "p0_text": p0_eval_text(actor.question, actor.choices), "image_paths": list(actor.image_paths),
                  "t1": {"text": first.get("text", ""), "finish_reason": first.get("finish_reason") or "",
                         "status": first["status"]}}
        if second is not None and "status" not in second:
            record["note"] = {key: second[key] for key in ("prompt", "text", "finish_reason", "corrected_status")}
        records.append(record)
    return records


def summarize(records: list[dict], rows: list[dict], drops: list[dict]) -> dict:
    """Counts only; no text."""
    parsed = [parse_note(r["note"]["text"]) if "note" in r else "NO_NOTE" for r in records]
    by_kind: dict[str, list[int]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row["target_tokens"])
    return {
        "items": len(records), "rows": len(rows), "rows_by_kind": dict(Counter(r["kind"] for r in rows)),
        "rows_by_target_source": dict(Counter(r["target_source"] for r in rows)),
        "rows_by_route": dict(Counter(r["route"] for r in rows)),
        "rows_by_subject": dict(Counter(r["subject"] for r in rows)),
        "drops": len(drops), "drops_by_reason": dict(Counter(d["reason"] for d in drops)),
        "image_reading": dict(Counter(p if p == "NO_NOTE" else "UNPARSED" if p is None else p.image_reading for p in parsed)),
        "cause": dict(Counter(p if p == "NO_NOTE" else "UNPARSED" if p is None else p.cause for p in parsed)),
        "t1_status": dict(Counter(r["t1"]["status"] for r in records)),
        "mean_target_tokens": {kind: round(sum(v) / len(v), 1) for kind, v in by_kind.items()},
    }


def collect_rows(records: list[dict], tokenize: Callable[[str], list[int]]) -> tuple[list[dict], list[dict], dict]:
    rows: list[dict] = []
    drops: list[dict] = []
    for record in records:
        built, dropped = build_rows(record, tokenize)
        rows += built
        drops += dropped
    return rows, drops, summarize(records, rows, drops)


# ------------------------------------------------------------------ CLI

def _prefix(args: argparse.Namespace) -> list[dict]:
    items = read_order(Path(args.order))["items"]
    if not 1 <= args.limit <= len(items):
        raise ValueError(f"--limit must be 1..{len(items)}")
    return items[:args.limit]


def _grader(service: Any) -> Callable:
    def grade(problem_id: str, text: str, finish_reason: str, mode: str) -> tuple[str, str | None]:
        result = service.grade(problem_id, text, finish_reason, mode, answer_contract=TURN1_ANSWER_CONTRACT)
        return result.status, result.extracted
    return grade


def _generate(args: argparse.Namespace, stage: str) -> int:
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.vault import read_actors, read_verifier
    from fine_tuning.notebook_rl.verifier_service import VerifierService

    build_root, out, vllm = Path(args.build_root), Path(args.output), args.engine == "vllm"
    out.mkdir(parents=True, exist_ok=True)
    items = _prefix(args)
    seqs = args.max_num_seqs
    gold = None
    if stage == "note":
        if args.max_num_seqs_from:
            seqs = json.loads(Path(args.max_num_seqs_from).read_text())["best_seqs"]
            if type(seqs) is not int:
                raise ValueError("--max-num-seqs-from: the sweep has no best_seqs")
        args.max_num_seqs = seqs
        verifier = read_verifier(build_root / "train")  # the gold is read before any guard, in this process only
        gold = {item["problem_id"]: verifier[item["problem_id"]].gold for item in items}
        del verifier
    forbidden = install_training_guards(build_root, vllm_schema={} if vllm else None)
    actors = {actor.problem_id: actor for actor in read_actors(build_root / "train")}
    unknown = [item["problem_id"] for item in items if item["problem_id"] not in actors]
    if unknown:
        raise ValueError(f"{len(unknown)} ordered items are not in the train split, first {unknown[:3]}")
    stats = StreamStats()
    stream = (_vllm_stream_builder if vllm else _hf_stream_builder)(args, stats)
    options = dict(S=seqs, B=args.max_num_batched_tokens, engine=args.engine, stop_after=args.stop_after_seconds)
    load = lambda actor: load_images(actor, build_root / "images")  # noqa: E731
    with VerifierService(build_root / "train", forbidden_roots=forbidden) as service, ResourceLog(out / "resource.jsonl"):
        if stage == "solve":
            receipt = run_solve(items, actors, load, stream, stats, _grader(service), out, **options)
        else:
            t1 = read_jsonl_done(out / "t1.jsonl", repair=False)
            receipt = run_note(items, actors, t1, gold, load, stream, stats, _grader(service), out, **options)
    print(json.dumps(receipt))
    return 0


def _rows(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from fine_tuning.notebook_rl.vault import read_actors

    out = Path(args.output)
    destination = Path(args.rows_output) if args.rows_output else out
    destination.mkdir(parents=True, exist_ok=True)
    items = _prefix(args)
    actors = {actor.problem_id: actor for actor in read_actors(Path(args.build_root) / "train")}
    records = make_records(items, actors, read_jsonl_done(out / "t1.jsonl", repair=False),
                           read_jsonl_done(out / "note.jsonl", repair=False))
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True)
    rows, drops, summary = collect_rows(records, lambda text: tokenizer.encode(text, add_special_tokens=False))
    write_atomic(destination / "rows.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_atomic(destination / "drops.jsonl", "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in drops))
    write_json(destination / "rows_summary.json", summary)
    print(json.dumps(summary))
    return 0


def _kv_arg(text: str) -> float | str:
    if text == "auto":
        return text
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive number or 'auto'")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--build-root", type=Path, required=True)
        p.add_argument("--order", type=Path, required=True)
        p.add_argument("--limit", type=int, required=True)
        p.add_argument("--output", type=Path, required=True)

    def engine(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model-path", type=Path, required=True)
        p.add_argument("--engine", choices=("vllm", "hf"), required=True)
        p.add_argument("--kv-cache-gib", type=_kv_arg, default=28, help="GiB of KV cache, or auto (GPU memory - 16)")
        p.add_argument("--max-num-batched-tokens", type=int, default=16384)
        p.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    def run_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-num-seqs", type=int, default=32)
        p.add_argument("--stop-after-seconds", type=float, default=None)

    for name in ("solve", "note", "sweep"):
        p = sub.add_parser(name)
        common(p)
        engine(p)
        if name != "sweep":
            run_options(p)
    sub.choices["note"].add_argument("--max-num-seqs-from", type=Path, default=None)
    sub.choices["sweep"].add_argument("--seqs", required=True, help="comma list of --max-num-seqs values, or auto")
    sub.choices["sweep"].add_argument("--seconds", type=float, default=300.0)
    rows = sub.add_parser("rows")
    common(rows)
    rows.add_argument("--tokenizer-path", type=Path, required=True)
    rows.add_argument("--rows-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "sweep":
        return sweep(args)
    if args.command == "rows":
        return _rows(args)
    return _generate(args, args.command)


if __name__ == "__main__":
    raise SystemExit(main())
