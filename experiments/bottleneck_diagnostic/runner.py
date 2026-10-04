"""Append-only internal diagnostics with pinned inference and cumulative budget gates."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .input_builder import MODES, build_request, canonical_hash, validate_sample
from .oracle_validation import validate_oracle


class BudgetStop(RuntimeError):
    """The active request was aborted before a partial output could be called complete."""


def remaining_window(protocol: dict[str, Any]) -> float:
    """Absolute UTC operational window, independent of process restart/run budgets."""
    value = protocol.get("overall_deadline_utc")
    if value is None:
        return math.inf
    if not isinstance(value, str):
        raise ValueError("overall_deadline_utc must be an aware UTC ISO timestamp")
    deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    offset = deadline.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("overall_deadline_utc must be an aware UTC ISO timestamp")
    return max(0.0, deadline.timestamp() - time.time())


def _json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid append-only JSONL {path}:{number}; refusing repair") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{number}")
            rows.append(row)
    return rows


def _append(path: Path, row: dict[str, Any]) -> None:
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["sample_id"], row["mode"], row["repetition"]


def _oracles(values: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if isinstance(values, Mapping):
        entries: list[Any] = []
        for value in values.values():
            entries.extend(value.values() if isinstance(value, dict) and "level" not in value else [value])
    else:
        entries = list(values or [])
    result = {}
    for oracle in entries:
        if not isinstance(oracle, dict) or "sample_id" not in oracle or "level" not in oracle:
            raise ValueError("Oracle manifest must contain records with sample_id and level")
        key = oracle["sample_id"], oracle["level"]
        if key in result:
            raise ValueError(f"Duplicate oracle version for {key}; freeze one approved version")
        result[key] = oracle
    return result


def _verify_images(data_dir: Path, samples: list[dict[str, Any]]) -> None:
    from PIL import Image

    verified: dict[Path, Any] = {}
    for sample in samples:
        for item in sample["images"]:
            path = (data_dir / item["path"]).resolve(strict=True)
            if not path.is_relative_to(data_dir):
                raise ValueError("Image path escapes the diagnostic data directory")
            expected = item["sha256"], item["original_size"]
            if path in verified:
                if verified[path] != expected:
                    raise ValueError("Shared image has conflicting hash/size metadata")
                continue
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != item["sha256"]:
                raise ValueError(f"Original image hash mismatch: {item['path']}")
            with Image.open(path) as image:
                if list(image.size) != item["original_size"]:
                    raise ValueError(f"Original image size mismatch: {item['path']}")
                image.verify()
            verified[path] = expected


def _loaded_request(request: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    from PIL import Image

    loaded = copy.deepcopy(request)
    images = []
    for metadata in request["images"]:
        path = (data_dir / metadata["path"]).resolve(strict=True)
        if not path.is_relative_to(data_dir):
            raise ValueError("Image path escaped data root")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != metadata["sha256"]:
                raise ValueError("Image changed after preflight")
        with Image.open(path) as original:
            if list(original.size) != metadata["original_size"]:
                raise ValueError("Image dimensions changed after preflight")
            images.append(original.convert("RGB"))
    blocks = [block for block in loaded["messages"][0]["content"] if block["type"] == "image"]
    for block, image in zip(blocks, images, strict=True):
        block["image"] = image
    loaded["images"] = images
    return loaded


def run_diagnostics(data_dir: str | Path, output_dir: str | Path, protocol: dict[str, Any],
                    samples: Iterable[dict[str, Any]], oracles: Any,
                    backend: Callable[[dict[str, Any]], dict[str, Any]],
                    budget_seconds: float = 14400,
                    repetitions: Iterable[int] = (3407,)) -> list[dict[str, Any]]:
    """Execute valid cells only. Injected backends produce TEST_ONLY records by default.

    Resume uses the same protocol/data/oracle/code identities, preserves completed cells,
    and reuses the original P text. Budgets include previous sessions and aborted requests.
    """
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("budget_seconds must be finite and positive")
    remaining_window(protocol)  # Reject malformed operational bounds before touching outputs.
    root = Path(data_dir).resolve(strict=True)
    output = Path(output_dir).resolve()
    values = list(samples)
    repeats = list(repetitions)
    if not values or not repeats or len(set(repeats)) != len(repeats):
        raise ValueError("Non-empty samples and unique repetitions are required")
    modes = list(protocol.get("modes", MODES[:-1]))
    policy = protocol.get("oracle_review_policy", "HUMAN_REVIEW_REQUIRED")
    if policy not in {"HUMAN_REVIEW_REQUIRED", "ALLOW_AI_REVIEWED_DIAGNOSTIC"}:
        raise ValueError("Unknown oracle_review_policy")
    if not modes or len(set(modes)) != len(modes) or set(modes) - set(MODES):
        raise ValueError("Protocol requires unique supported modes")
    if any(mode in modes for mode in ("M", "MI")) and "P" not in modes:
        raise ValueError("M/MI protocol must include P")
    # P must precede its dependants; other modes retain the frozen protocol order.
    modes = [mode for mode in modes if mode not in {"M", "MI"}] + [
        mode for mode in modes if mode in {"M", "MI"}
    ]
    for sample in values:
        validate_sample(sample)
        build_request(sample, "I", repeats[0])
    if len({sample["sample_id"] for sample in values}) != len(values):
        raise ValueError("Sample IDs must be unique")
    for rep in repeats:
        build_request(values[0], "I", rep)
    oracle_map = _oracles(oracles)
    _verify_images(root, values)
    token_budget = protocol.get("total_output_token_budget", 250000)
    if not isinstance(token_budget, int) or isinstance(token_budget, bool) or token_budget < 1:
        raise ValueError("total_output_token_budget must be a positive integer")
    sources = [Path(__file__), Path(__file__).with_name("input_builder.py"),
               Path(__file__).with_name("oracle_validation.py"), Path(__file__).with_name("scoring.py"),
               Path(__file__).with_name("local_backend.py"), Path(__file__).with_name("ingest.py"),
               Path(__file__).with_name("cli.py")]
    repository = Path(__file__).resolve().parents[2]
    sources.extend(repository / "src/mmdl/evaluation" / name for name in (
        "prompt.py", "parsers.py", "final_answer_parser.py", "final_answer_parser_v8.py",
        "backends/vllm_backend.py", "backends/input_preparation.py", "backends/transformers_backend.py",
    ))
    sources.extend(repository / "src/mmdl/runtime" / name for name in ("placement.py", "generation.py"))
    sources.append(repository / "third_party/mmmu/eval_utils.py")
    code_hash = canonical_hash({str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sources})
    identity = {
        "protocol": protocol, "protocol_hash": canonical_hash(protocol),
        "sample_manifest_hash": canonical_hash(values),
        "oracle_manifest_hash": canonical_hash(sorted(oracle_map.values(), key=lambda item: (item["sample_id"], item["level"]))),
        "repetitions": repeats, "budget_seconds": budget_seconds,
        "total_output_token_budget": token_budget, "code_hash": code_hash,
        "execution_kind": "ACTUAL_INFERENCE" if getattr(backend, "actual_inference", False) is True else "TEST_ONLY",
        "backend_evaluation_config": getattr(backend, "cfg", None),
        "backend_hardware_profile": getattr(backend, "hardware_profile", None),
        "model_files_manifest_hash": getattr(backend, "model_files_manifest_hash", None),
        "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    # One owner per output directory prevents duplicate completion records on concurrent resume.
    lock = (output / ".runner.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("Another diagnostic runner owns this output directory") from exc
    try:
        frozen_path = output / "protocol.json"
        if frozen_path.exists():
            if json.loads(frozen_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("Resume refused: protocol/data/oracle/code/budget identity changed")
        else:
            if any(output.glob("*.jsonl")):
                raise ValueError("Cannot initialize over existing unfrozen diagnostic records")
            _atomic_json(frozen_path, identity)
        return _run_locked(root, output, identity, values, oracle_map, backend, modes, repeats)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _run_locked(root: Path, output: Path, identity: dict[str, Any],
                samples: list[dict[str, Any]], oracle_map: dict[tuple[str, str], dict[str, Any]],
                backend: Callable[[dict[str, Any]], dict[str, Any]], modes: list[str],
                repetitions: list[int]) -> list[dict[str, Any]]:
    response_path = output / "responses.jsonl"
    history = _json_rows(response_path)
    latest = {}
    completed = set()
    consumed_tokens = 0
    for row in history:
        stored_hash = row.get("record_hash")
        if stored_hash != canonical_hash({key: value for key, value in row.items() if key != "record_hash"}):
            raise ValueError("Response ledger record hash mismatch")
        key = _key(row)
        if row.get("protocol_hash") != identity["protocol_hash"] or row.get("run_kind") != identity["execution_kind"]:
            raise ValueError("Response ledger protocol/execution identity differs from frozen run")
        if key in completed:
            raise ValueError("Completed diagnostic response was overwritten/duplicated")
        if row["runtime_status"] == "COMPLETED":
            if row["response_hash"] != canonical_hash(row["raw_response"]):
                raise ValueError("Completed raw response hash mismatch")
            completed.add(key)
        consumed_tokens += row.get("charged_output_tokens", row.get("output_tokens", 0))
        latest[key] = row
    attempts = {row["attempt_id"] for row in history if "attempt_id" in row}
    audits = _json_rows(output / "input_audit.jsonl")
    # A crash after enqueue can spend tokens without a completion: charge its full reserved allowance.
    consumed_tokens += sum(row["max_new_tokens"] for row in audits if row["attempt_id"] not in attempts)
    budgets = _json_rows(output / "budget_events.jsonl")
    consumed_seconds = sum(row.get("elapsed_seconds", 0) for row in budgets if row["event"] == "SESSION_END")
    starts = {row["session_id"]: row for row in budgets if row["event"] == "SESSION_START"}
    ends = {row["session_id"] for row in budgets if row["event"] == "SESSION_END"}
    # An unclosed session is conservatively charged through recovery, including downtime.
    for session_id in starts.keys() - ends:
        elapsed = max(0.0, time.time() - starts[session_id]["started_epoch"]) + starts[session_id].get("model_load_seconds", 0)
        consumed_seconds += elapsed
        _append(output / "budget_events.jsonl", {
            "event": "SESSION_END", "session_id": session_id, "elapsed_seconds": elapsed,
            "recovered_unclosed_session": True,
        })
    start = time.monotonic()
    load_seconds = getattr(backend, "load_seconds", 0)
    if not isinstance(load_seconds, (int, float)) or not math.isfinite(load_seconds) or load_seconds < 0:
        raise ValueError("Backend load_seconds must be finite and nonnegative")
    consumed_seconds += load_seconds
    session_id = f"{time.time_ns()}-{os.getpid()}"
    _append(output / "budget_events.jsonl", {
        "event": "SESSION_START", "session_id": session_id, "started_epoch": time.time(),
        "previous_elapsed_seconds": consumed_seconds, "previous_output_tokens": consumed_tokens,
        "model_load_seconds": load_seconds,
    })
    deadline = start + min(max(0, identity["budget_seconds"] - consumed_seconds),
                           remaining_window(identity["protocol"]))
    run_kind = identity["execution_kind"]
    stop = False
    policy = identity["protocol"].get("oracle_review_policy", "HUMAN_REVIEW_REQUIRED")
    legacy_roles = {row["sample_id"]: row for row in _json_rows(root / "input_audit.jsonl")} if "LEGACY" in modes else {}

    def save(row: dict[str, Any]) -> None:
        row["record_hash"] = canonical_hash(row)
        _append(response_path, row)
        latest[_key(row)] = row

    try:
        for sample in samples:
            for rep in repetitions:
                for mode in modes:
                    key = sample["sample_id"], mode, rep
                    previous = latest.get(key)
                    if previous and previous["runtime_status"] in {"COMPLETED", "BLOCKED", "EXCLUDED"}:
                        if previous["runtime_status"] == "COMPLETED":
                            p = latest.get((sample["sample_id"], "P", rep)) if mode in {"M", "MI"} else None
                            oracle = oracle_map.get((sample["sample_id"], "O1" if mode == "O1" else "O0")) if mode in {"O0", "O1", "IO0"} else None
                            request = build_request(sample, mode, rep, oracle, p, policy)
                            if request["cache_key"] != previous["request_hash"]:
                                raise ValueError("Resume request or P response differs from completed cell")
                        continue
                    base = {
                        "run_id": identity["protocol"].get("run_id", output.name),
                        "sample_id": sample["sample_id"], "subject": sample["subject"],
                        "family_id": sample["family_id"], "skill_tags": sample["skill_tags"],
                        "mode": mode, "repetition": rep, "protocol_hash": identity["protocol_hash"],
                        "run_kind": run_kind, "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
                    }
                    oracle = None
                    if mode == "LEGACY" and (sample["sample_id"] not in legacy_roles
                            or legacy_roles[sample["sample_id"]]["excluded_images"]):
                        save(base | {"runtime_status": "BLOCKED",
                                     "reason": "LEGACY_ORIGINAL_IMAGE_ROLE_EQUIVALENCE_NOT_VERIFIED"})
                        continue
                    if mode in {"O0", "O1", "IO0"}:
                        level = "O1" if mode == "O1" else "O0"
                        oracle = oracle_map.get((sample["sample_id"], level))
                        verdict = validate_oracle(oracle, sample, level, policy)
                        if verdict["status"] != "VALID":
                            save(base | {"runtime_status": verdict["status"], "reason": verdict["reason"]})
                            continue
                    p = latest.get((sample["sample_id"], "P", rep)) if mode in {"M", "MI"} else None
                    if mode in {"M", "MI"} and (p is None or p["runtime_status"] != "COMPLETED"):
                        save(base | {"runtime_status": "BUDGET_STOP" if stop else "BLOCKED", "reason": "P_NOT_COMPLETED"})
                        continue
                    if mode == "P" and not sample["images"]:
                        save(base | {"runtime_status": "EXCLUDED", "reason": "NO_IMAGE_STRUCTURE_TASK"})
                        continue
                    request = build_request(sample, mode, rep, oracle, p, policy)
                    audit = base | {
                        "stage_seed": request["seed"], "prompt_hash": request["prompt_hash"],
                        "request_hash": request["cache_key"],
                        "cache_key": canonical_hash([identity["protocol_hash"], request["cache_key"]]),
                        "oracle_hash": request["oracle_hash"],
                        "perception_response_hash": request["perception_response_hash"],
                        "verification_kind": request["verification_kind"], "oracle_review_policy": policy,
                        "image_hashes": [image["sha256"] for image in request["images"]],
                        "max_new_tokens": request["max_new_tokens"],
                    }
                    if (stop or time.monotonic() >= deadline or
                            consumed_tokens + request["max_new_tokens"] > identity["total_output_token_budget"]):
                        stop = True
                        if not previous or previous["runtime_status"] != "BUDGET_STOP":
                            save(audit | {"runtime_status": "BUDGET_STOP", "reason": "TIME_OR_OUTPUT_TOKEN_BUDGET"})
                        continue
                    audit["attempt_id"] = canonical_hash([session_id, key, time.time_ns()])
                    _append(output / "input_audit.jsonl", audit | {"request": request})
                    execution_request = request | {"deadline_monotonic": deadline}
                    before = time.monotonic()
                    try:
                        result = backend(_loaded_request(execution_request, root))
                        if not isinstance(result, dict) or not isinstance(result.get("raw_response"), str):
                            raise ValueError("Backend must return raw_response text and measured token counts")
                        tokens = result.get("generated_tokens", result.get("output_tokens"))
                        if not isinstance(tokens, int) or isinstance(tokens, bool) or not 0 <= tokens <= request["max_new_tokens"]:
                            raise ValueError("Backend output token count violates the frozen generation limit")
                        if result.get("finish_reason") not in {"stop", "eos", "length"}:
                            raise ValueError("Backend returned a missing/unsupported finish reason")
                        counts = result.get("input_tokens")
                        if not isinstance(counts, int) or isinstance(counts, bool) or counts < 1:
                            raise ValueError("Backend must record actual positive input_tokens")
                        save(audit | {
                            "runtime_status": "COMPLETED", "raw_response": result["raw_response"],
                            "response_hash": canonical_hash(result["raw_response"]),
                            "finish_reason": result["finish_reason"], "input_tokens": counts,
                            "output_tokens": tokens, "elapsed_seconds": time.monotonic() - before,
                            "image_grid_thw": result.get("image_grid_thw", []),
                            "model_revision": result.get("model_revision"),
                            "backend": result.get("backend", "injected"),
                            "backend_version": result.get("backend_version"),
                            "peak_memory": result.get("peak_memory"), "backend_audit": result,
                        })
                        consumed_tokens += tokens
                    except BudgetStop as exc:
                        stop = True
                        consumed_tokens += request["max_new_tokens"]
                        save(audit | {"runtime_status": "BUDGET_STOP", "reason": str(exc),
                                      "charged_output_tokens": request["max_new_tokens"],
                                      "elapsed_seconds": time.monotonic() - before})
                    except Exception as exc:
                        consumed_tokens += request["max_new_tokens"]
                        save(audit | {"runtime_status": "SYSTEM_ERROR", "reason": f"{type(exc).__name__}: {exc}",
                                      "charged_output_tokens": request["max_new_tokens"],
                                      "elapsed_seconds": time.monotonic() - before})
    finally:
        _append(output / "budget_events.jsonl", {
            "event": "SESSION_END", "session_id": session_id,
            "elapsed_seconds": time.monotonic() - start + load_seconds,
            "total_output_tokens": consumed_tokens,
        })
    return list(latest.values())


def make_vllm_backend(model_path: str, cfg: dict[str, Any], hw: dict[str, Any]) -> Any:
    """Lazy GPU import keeps manifest/scoring/CPU checks independent of CUDA packages."""
    from mmdl.evaluation.backends.vllm_backend import VLLMBackend
    from mmdl.evaluation.backends.input_preparation import prepare_protocol_inputs, tensor_identity
    from mmdl.runtime.artifacts import digest
    from mmdl.runtime.contracts import MODEL_REVISION

    class DiagnosticVLLMBackend(VLLMBackend):
        actual_inference = True

        def _prepare_request(self, request: dict[str, Any], *, final_only: bool) -> tuple[Any, Any, Any]:
            if request["images"]:
                item, images = prepare_protocol_inputs(
                    self.processor, self.model_config, request["messages"], request["images"], self.cfg["image"],
                )
                prompt = {"prompt": item["chat_prompt"], "multi_modal_data": {"image": images}}
            else:
                text = self.processor.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
                inputs = self.processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                ids = inputs["input_ids"][0].tolist()
                if self.model_config.image_token_id in ids:
                    raise ValueError("Text-only request unexpectedly contains an image token")
                tensors = {name: tensor_identity(value) for name, value in inputs.items()}
                item = {"chat_prompt": text, "input_tokens": len(ids), "prompt_token_ids": ids,
                        "image_grid_thw": [], "input_tensors": tensors, "input_sha256": digest(tensors)}
                prompt = {"prompt": text}
            maximum = request["max_new_tokens"]
            if item["input_tokens"] + maximum > self.max_model_len:
                raise ValueError("Input plus output allowance exceeds context; refusing truncation")
            generation = self.cfg["generation"]
            params = self._sampling_params_cls(
                n=1, temperature=generation["temperature"], top_p=generation["top_p"],
                top_k=generation["top_k"], repetition_penalty=generation["repetition_penalty"],
                presence_penalty=generation["presence_penalty"], max_tokens=maximum,
                seed=request["seed"], output_kind=self._request_output_kind,
            )
            return item, prompt, params

        def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
            # ponytail: serial calls preserve P dependencies; batch independent cells if throughput demands it.
            request = request | {"sample_id": canonical_hash([
                request["sample_id"], request["mode"], request["repetition"], request["cache_key"],
            ])}
            engine = self.llm.llm_engine
            item, prompt, params = self._prepare_request(request, final_only=True)
            identifier = request["sample_id"]
            if time.monotonic() >= request["deadline_monotonic"]:
                raise BudgetStop("TIME_BUDGET_BEFORE_ENQUEUE")
            engine.add_request(identifier, prompt, params)
            start = time.monotonic()
            active = True
            try:
                while active:
                    if time.monotonic() >= request["deadline_monotonic"]:
                        raise BudgetStop("TIME_BUDGET_ACTIVE_REQUEST_ABORTED")
                    for result in engine.step():
                        if result.request_id != identifier:
                            raise RuntimeError("Unexpected diagnostic engine request ID")
                        if result.finished:
                            active = False
                            row = self._row(request, item, params, result,
                                            generation_seconds=time.monotonic() - start,
                                            timing_policy="serial_request_wall")
                            image_tokens = sum(
                                token == self.model_config.image_token_id for token in item["prompt_token_ids"]
                            )
                            return row | {"model_revision": MODEL_REVISION, "backend": "vllm",
                                          "backend_version": self.vllm_version,
                                          "image_tokens": image_tokens,
                                          "input_text_tokens": item["input_tokens"] - image_tokens,
                                          "engine_token_agreement_verified": True,
                                          "peak_memory": None}
                raise RuntimeError("Engine completed without a response")
            finally:
                if active:
                    engine.abort_request([identifier])

    frozen = {
        "model": {"id": "Qwen/Qwen3-VL-4B-Instruct", "revision": MODEL_REVISION, "dtype": "bfloat16"},
        "image": {"min_pixels": 1003520, "max_pixels": 4014080,
                  "preprocessing": "qwen_vl_utils_0_0_14"},
        "generation": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                       "repetition_penalty": 1.0, "presence_penalty": 1.5},
        "execution": {"backend": "vllm", "batch_size": 2, "max_model_len": 40960},
    }
    for section, requirements in frozen.items():
        for field, value in requirements.items():
            if cfg.get(section, {}).get(field) != value:
                raise ValueError(f"Diagnostic backend refuses changed frozen {section}.{field}")
    backend = DiagnosticVLLMBackend(model_path, cfg, hw)
    backend.hardware_profile = copy.deepcopy(hw)
    return backend
