"""Gold-isolated external evaluation; CPU adapters test contracts, not GPU accuracy.

The injected engine must expose model_dir, policy_version, supported_sampling,
prepare, tokenizer, and sample_with_settings(requests, SamplingSettings). Sampling
settings are explicit: a T1-only engine cannot silently stand in for official eval.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from .dataset import load_images
from .engines import Completion, QWEN_OFFICIAL, RolloutRequest, SamplingSettings, _completion
from .prompts import load_prompts, retry_text, turn1_text, user_message
from .rollout import continuation_ids
from .schema import Sample, VerifierResult, to_json
from .vault import read_actors
from .verifier_service import VerifierService

MODES = frozenset(("first", "pass_at_k", "forced_second", "feedback_second", "blind"))
SCORED = frozenset(("CORRECT", "INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"))
RETRYABLE = SCORED - {"CORRECT"}
FORCED_REVIEW = "Review your solution and give a final answer"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(problem_id: str, run_seed: int, sampler: str, index: int, turn: int = 1) -> int:
    value = f"external-v1|{run_seed}|{problem_id}|{sampler}|{index}|{turn}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % 2**31


def pass_at_k(n: int, correct: int, k: int) -> float:
    if (any(type(v) is not int for v in (n, correct, k)) or not 0 <= correct <= n or not 1 <= k <= n):
        raise ValueError("external: pass@k needs 0<=correct<=n and 1<=k<=n")
    return 1. - (math.comb(n-correct, k)/math.comb(n, k) if n-correct >= k else 0.)


def _item_mean(rows: list[dict], value) -> tuple[float | None, int]:
    items: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = value(row)
        if score is not None:
            items[row["problem_id"]].append(float(score))
    return (statistics.mean(statistics.mean(v) for v in items.values()), len(items)) if items else (None, 0)


def _cell(rows: list[dict], k_list: tuple[int, ...]) -> dict:
    applicable = [r for r in rows if r["applicable"] and r["planned"]]
    trials: dict[tuple, list[dict]] = defaultdict(list)
    for row in applicable:
        trials[(row["problem_id"], row["run_seed"], row["parent_record_id"])].append(row)
    expected = sum(values[0]["expected_n"] for values in trials.values())
    scored = [r for r in applicable if r["status"] in SCORED and type(r["correct"]) is bool]
    complete = len(scored) == expected and all(r["planned"] for r in rows)
    observed, n_observed = _item_mean(scored, lambda r: r["correct"])
    generated = [r for r in rows if r["n_tokens"] is not None]
    slot_value, n_slot_items = _item_mean(scored, lambda r: (
        r["slots"]["required_correct"]/r["slots"]["required"]
        if r["slots"] is not None and r["slots"]["required"] else None))
    counts = {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}
    result: dict[str, Any] = {"status": "COMPLETE" if complete else "INCOMPLETE", "n_items": len({r["problem_id"] for r in rows}),
        "n_expected": expected, "n_scored": len(scored), "n_inapplicable": sum(not r["applicable"] for r in rows),
        "accuracy": observed if complete else None, "observed_accuracy": observed,
        "n_observed_items": n_observed, "status_counts": counts,
        "required_slot_accuracy": {"value": slot_value, "n_items": n_slot_items,
            "n_observations": sum(r["slots"] is not None and r["slots"]["required"] > 0 for r in scored)},
        "n_generated": len(generated), "length_median": statistics.median(r["n_tokens"] for r in generated) if generated else None,
        "truncation_rate": sum(r["finish_reason"] == "length" for r in generated)/len(generated) if generated else None,
        "short_fraction_le32": sum(r["n_tokens"] <= 32 for r in generated)/len(generated) if generated else None,
        "one_line_rate": sum(len(r["raw_text"].strip().splitlines()) == 1 for r in generated)/len(generated) if generated else None,
        "no_final_rate": sum(r["status"] == "INVALID_NO_FINAL" for r in scored)/len(scored) if scored else None}
    if rows[0]["mode"] == "pass_at_k":
        pass_trials: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for row in rows:
            pass_trials[row["problem_id"], row["run_seed"]].append(row)
        estimates = []
        for (pid, _), values in pass_trials.items():
            if len(values) == 8 and all(r["status"] in SCORED for r in values):
                c = sum(r["correct"] for r in values)
                estimates.append({"problem_id": pid, **{str(k): pass_at_k(8, c, k) for k in k_list}})
        result["pass_at_k"] = {str(k): {"value": _item_mean(estimates, lambda r: r[str(k)])[0] if complete else None,
            "observed_value": _item_mean(estimates, lambda r: r[str(k)])[0],
            "n_items": _item_mean(estimates, lambda r: r[str(k)])[1],
            "n_complete_trials": len(estimates), "n_expected_trials": len(pass_trials)} for k in k_list}
    if rows[0]["mode"] in ("forced_second", "feedback_second"):
        pairs = [r for r in scored if type(r["parent_correct"]) is bool]
        paired_complete = len(pairs) == expected and complete
        result["paired"] = {"n_expected": expected, "n_complete": len(pairs)}
        for name, was_correct in (("i_to_c", False), ("c_to_i", True)):
            selected = [r for r in pairs if r["parent_correct"] is was_correct]
            metric, n_items = _item_mean(selected, lambda r: r["correct"] is not was_correct)
            result["paired"][name] = {"value": metric if paired_complete else None, "observed_value": metric,
                "n_items": n_items, "denominator": len(selected),
                "numerator": sum(r["correct"] is not was_correct for r in selected)}
        repeat, n_repeat = _item_mean(pairs, lambda r: r["extracted"] == r["parent_extracted"]
            if r["extracted"] is not None and r["parent_extracted"] is not None else None)
        result["paired"]["exact_extracted_repeat"] = {"value": repeat, "n_items": n_repeat,
            "denominator": sum(r["extracted"] is not None and r["parent_extracted"] is not None for r in pairs)}
        baseline, n_baseline = _item_mean(applicable, lambda r: r["elimination_baseline"])
        result["paired"]["mcq_elimination_baseline"] = {"value": baseline, "n_items": n_baseline,
            "denominator": sum(r["elimination_baseline"] is not None for r in applicable)}
    return result


def summarize(rows: list[dict], k_list: tuple[int, ...] = (1, 4, 8)) -> dict:
    if (not rows or not k_list or len(set(k_list)) != len(k_list)
            or any(type(k) is not int or not 1 <= k <= 8 for k in k_list)):
        raise ValueError("external: nonempty records and distinct pass@k values in 1..8 required")
    ids = [r["record_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("external: duplicate evaluation record identity")
    cells: dict[tuple, list] = defaultdict(list)
    for row in rows:
        for family in {row["family"], "*"}:
            for prompt_mode in {row["prompt_mode"], "*"}:
                for answer_kind in {row["answer_kind"], "*"}:
                    cells[(row["model"], row["split"], family, row["mode"], row["sampler"], prompt_mode, answer_kind)].append(row)
    output = []
    for key, values in sorted(cells.items()):
        names = ("model", "split", "family", "mode", "sampler", "prompt_mode", "answer_kind")
        output.append(dict(zip(names, key, strict=True)) | _cell(values, k_list))
    primary = [r for r in output if r["family"] == r["prompt_mode"] == r["answer_kind"] == "*"
               and r["mode"] == "first" and r["sampler"] == "qwen_official"]
    return {"status": "COMPLETE" if all(r["status"] == "COMPLETE" for r in output) else "INCOMPLETE",
        "n_records": len(rows), "n_items": len({r["problem_id"] for r in rows}), "cells": output,
        "primary_first_attempt": [{"split": r["split"], "model": r["model"], "accuracy": r["accuracy"],
            "n_items": r["n_items"], "n_expected": r["n_expected"], "n_scored": r["n_scored"]} for r in primary],
        "denominator_policy": "equal item weight after averaging requested repetitions; failures are null, never ordinary incorrect",
        "gpu_execution_certified": False}


def evaluate(model_dir: Path, split_root: Path, output: Path, modes, k_list=(1, 4, 8), seeds=(3407,), *,
             engine: Any, protocol: dict, image_root: Path | None = None, split: str | None = None,
             forbidden_roots=(), _test_verifier: Any = None) -> dict:
    """Write every requested/failed observation; summary is the final completion marker.

    A private verifier service is the production default. `_test_verifier` is only
    for authored CPU fixtures; it is never passed to the model/engine adapter.
    """
    model_dir, split_root, output = Path(model_dir), Path(split_root), Path(output)
    modes, k_list, seeds = tuple(modes), tuple(k_list), tuple(seeds)
    split = split or split_root.name
    if (not modes or len(set(modes)) != len(modes) or not set(modes) <= MODES
            or split not in ("dev", "sealed", "style_sealed") or not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds)
            or not k_list or len(set(k_list)) != len(k_list)
            or any(type(k) is not int or not 1 <= k <= 8 for k in k_list)):
        raise ValueError("external: invalid modes, held-out split, seeds, or pass@k")
    if (protocol["evaluation"]["external_first_attempt"]["n"] != 4
            or protocol["evaluation"]["pass_at_k"]["n"] != 8):
        raise ValueError("external: pinned n=4 first attempts/n=8 pass@k required")
    prompts = load_prompts(Path(__file__).with_name("prompts_v1.yaml"))
    if protocol["prompts_sha256"] != prompts["sha256"]:
        raise ValueError("external: actual prompts differ from protocol")
    need_official = bool(set(modes)-{"pass_at_k"})
    required = {"temperature_1"} if "first" in modes or "pass_at_k" in modes else set()
    if need_official:
        required.add("qwen_official")
    adapter_model = getattr(engine, "model_dir", None)
    if (not callable(getattr(engine, "sample_with_settings", None))
            or not required <= getattr(engine, "supported_sampling", set())
            or not isinstance(adapter_model, (str, Path)) or Path(adapter_model).resolve() != model_dir.resolve()):
        raise ValueError("external: matching model and explicit sampling-capable adapter required")
    version = engine.policy_version
    if type(version) is not int or version < 0:
        raise ValueError("external: explicit nonnegative fixed policy version required")
    actors = sorted(read_actors(split_root), key=lambda a: a.problem_id)
    if not actors:
        raise ValueError("external: empty evaluation split")
    image_root = Path(image_root) if image_root is not None else split_root.parent / "images"
    protected = Path(__file__).resolve().parents[2] / "MMDL"
    if any(output.resolve().is_relative_to(root.resolve()) for root in (model_dir, split_root, image_root, protected)):
        raise ValueError("external: output must be outside model/data/protected sources")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "notebook-rl-external-v1", "model": str(model_dir), "split": split,
        "modes": modes, "k_list": k_list, "seeds": seeds, "policy_version": version,
        "protocol_sha256": protocol["sha256"], "prompts_sha256": prompts["sha256"],
        "actor_inputs_sha256": _sha(split_root / "actor_inputs.jsonl"),
        "vault_manifest_sha256": _sha(split_root / "manifest.yaml"), "n_items": len(actors),
        "sampling": {s.name: {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k,
            "presence_penalty": s.presence_penalty, "repetition_penalty": s.repetition_penalty}
            for s in (QWEN_OFFICIAL, SamplingSettings())},
        "code_sha256": {name: _sha(Path(__file__).with_name(name)) for name in
            ("evaluate_external.py", "engines.py", "dataset.py", "rollout.py", "prompts.py", "schema.py")},
        "engine_class": type(engine).__name__, "gpu_execution_certified": False}
    with (output / "run_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    rows: list[dict] = []
    costs = {"sampler_calls_attempted": 0, "sampler_response_calls": 0,
             "returned_completions": 0, "returned_tokens": 0}
    failed_state = False
    context = nullcontext(_test_verifier) if _test_verifier is not None else VerifierService(
        split_root, forbidden_roots=list(forbidden_roots))
    with context as verifier, (output / "results.jsonl").open("x", encoding="utf-8") as stream, (
            output / "sampler_calls.jsonl").open("x", encoding="utf-8") as raw_stream:
        verifier_metadata = {"test_only": True} if _test_verifier is not None else verifier.metadata
        with (output / "verifier_manifest.json").open("x", encoding="utf-8") as verifier_stream:
            json.dump(verifier_metadata, verifier_stream, ensure_ascii=False, allow_nan=False, indent=2)
            verifier_stream.write("\n")
        def record(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(row)

        for actor in actors:
            try:
                images = load_images(actor, image_root)
                image_error = False
            except (OSError, ValueError):
                images, image_error = [], True
            try:
                for run_seed in seeds:
                    def request_for(mode, sampling, count, parent=None, index=0, blind=False):
                        condition = "root" if parent is None else mode
                        rid = f"{actor.problem_id}:external:{run_seed}:{sampling.name}:{mode}:{condition}:{index}"
                        if parent is None:
                            ids = None
                            messages = [user_message(turn1_text(actor, prompts), len(images))]
                        else:
                            prior = Sample(parent["record_id"], actor.problem_id, "external", "T1", 1, None,
                                parent["prompt_token_ids"], parent["completion_token_ids"], parent["logprobs"],
                                parent["finish_reason"], parent["raw_text"], parent["seed"], version)
                            feedback = FORCED_REVIEW if mode == "forced_second" else retry_text(
                                parent["status"], parent["extracted"], actor.mode, actor.family, prompts)
                            ids, messages = continuation_ids(prior, feedback, engine.tokenizer), None
                        actual_images = [Image.new("RGB", im.size, (128, 128, 128)) for im in images] if blind else images
                        turn = 1 if parent is None else 2
                        sample_seeds = [_seed(actor.problem_id, run_seed, sampling.name, i if parent is None else index, turn)
                                        for i in range(count)]
                        return RolloutRequest(rid, ids, messages, actual_images, count,
                            protocol["rollout"]["max_new_tokens"][turn], sample_seeds)

                    def query(mode, sampling, count, *, parent=None, index=0, blind=False):
                        nonlocal failed_state
                        request = request_for(mode, sampling, count, parent, index, blind)
                        request_wire = {"request_id": request.request_id, "messages": request.messages,
                            "prompt_token_ids": request.prompt_token_ids, "n": request.n,
                            "max_new_tokens": request.max_new_tokens, "seeds": request.seeds,
                            "images": [{"size": list(im.size), "rgb_sha256": hashlib.sha256(im.tobytes()).hexdigest()}
                                       for im in request.images]}
                        outputs, prompt_ids, actual_prompt, error = [], request.prompt_token_ids or [], None, None
                        if image_error:
                            error = "IMAGE_INPUT_ERROR"
                        elif failed_state:
                            error = "POLICY_VERSION_CHANGED"
                        else:
                            try:
                                if engine.policy_version != version:
                                    failed_state = True
                                    raise RuntimeError("POLICY_VERSION_CHANGED")
                                prepared = engine.prepare(request)
                                prompt_ids = list(prepared.token_ids)
                                costs["sampler_calls_attempted"] += 1
                                results = engine.sample_with_settings([request], sampling)
                                costs["sampler_response_calls"] += 1
                                costs["returned_completions"] += sum(len(r.completions) for r in results)
                                costs["returned_tokens"] += sum(len(c.token_ids) for r in results for c in r.completions)
                                # Preserve every actual response even if count/identity validation fails.
                                raw_stream.write(json.dumps({"mode": mode, "sampler": sampling.name,
                                    "request": request_wire, "results": [asdict(r) for r in results]},
                                    ensure_ascii=False, allow_nan=False)+"\n")
                                raw_stream.flush()
                                os.fsync(raw_stream.fileno())
                                if results:
                                    outputs = results[0].completions
                                    actual_prompt = list(results[0].prompt_token_ids)
                                if engine.policy_version != version:
                                    failed_state = True
                                    raise RuntimeError("POLICY_VERSION_CHANGED")
                                if len(results) != 1 or results[0].request_id != request.request_id:
                                    raise RuntimeError("RESULT_IDENTITY")
                                if list(results[0].prompt_token_ids) != prompt_ids or (
                                        request.prompt_token_ids is not None and prompt_ids != request.prompt_token_ids):
                                    error = "PROMPT_IDENTITY"
                                if len(outputs) > count:
                                    error = "UNEXPECTED_COMPLETION_COUNT"
                            except Exception as exc:
                                error = "CONTEXT_BLOCKED" if str(exc).startswith("CONTEXT_BLOCKED:") else type(exc).__name__
                        result_rows = []
                        for position in range(max(count, len(outputs))):
                            completion = outputs[position] if position < len(outputs) else None
                            row = {"model": model_dir.name, "problem_id": actor.problem_id, "family": actor.family,
                                "split": split, "mode": mode, "sampler": sampling.name, "prompt_mode": actor.mode,
                                "answer_kind": "mcq" if actor.answer_format == "mcq" else "open", "k": index if parent else position,
                                "run_seed": run_seed, "seed": request.seeds[position] if position < count else None,
                                "record_id": f"{request.request_id}:{mode}:{position}", "applicable": True, "planned": position < count,
                                "expected_n": count,
                                "status": "SYSTEM_ERROR" if error else "MISSING_COMPLETION", "correct": None,
                                "extracted": None, "slots": None, "finish_reason": None, "n_tokens": None,
                                "raw_text": None, "prompt_token_ids": prompt_ids, "completion_token_ids": [], "logprobs": [],
                                "actual_prompt_token_ids": actual_prompt,
                                "parent_record_id": parent["record_id"] if parent else None,
                                "parent_correct": parent["correct"] if parent else None,
                                "parent_extracted": parent["extracted"] if parent else None,
                                "elimination_baseline": (1/(len(actor.choices)-int(parent["status"] == "INCORRECT")))
                                    if parent and mode == "feedback_second" and actor.choices else None,
                                "error_code": error,
                                "request": request_wire}
                            if error == "CONTEXT_BLOCKED":
                                row["status"] = "CONTEXT_BLOCKED"
                            if completion is not None:
                                row.update(raw_text=completion.text, completion_token_ids=list(completion.token_ids),
                                    logprobs=list(completion.logprobs), finish_reason=completion.finish_reason,
                                    n_tokens=len(completion.token_ids))
                                if not error:
                                    try:
                                        if type(completion) is not Completion:
                                            raise ValueError("unsupported completion")
                                        _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
                                        if len(completion.token_ids) > request.max_new_tokens:
                                            raise ValueError("over-cap completion")
                                        grade: VerifierResult = verifier.grade(actor.problem_id, completion.text,
                                            completion.finish_reason, actor.mode)
                                        to_json(grade)
                                        row.update(status=grade.status, extracted=grade.extracted,
                                            correct=grade.status == "CORRECT" if grade.status in SCORED else None,
                                            slots=json.loads(to_json(grade.slots)) if grade.slots else None)
                                    except Exception as exc:
                                        row.update(status="SYSTEM_ERROR", error_code=type(exc).__name__)
                            record(row)
                            result_rows.append(row)
                        if blind:
                            for image in request.images:
                                image.close()
                        return result_rows

                    first = []
                    if set(modes) & {"first", "forced_second", "feedback_second"}:
                        first = query("first", QWEN_OFFICIAL, 4)
                    if "pass_at_k" in modes or "first" in modes:
                        temperature_rows = query("pass_at_k" if "pass_at_k" in modes else "first",
                                                 SamplingSettings(), 8 if "pass_at_k" in modes else 4)
                        if "first" in modes and "pass_at_k" in modes:
                            for row in temperature_rows[:4]:
                                record(row | {"mode": "first", "record_id": row["record_id"]+":first-reuse",
                                              "expected_n": 4,
                                              "reused_from": row["record_id"]})
                    for mode in ("forced_second", "feedback_second"):
                        if mode not in modes:
                            continue
                        for index, parent in enumerate(first[:4]):
                            can_retry = (parent["status"] in SCORED or mode == "forced_second" and
                                         parent["status"] == "UNVERIFIABLE" and parent["error_code"] is None)
                            if not can_retry or mode == "feedback_second" and parent["status"] == "CORRECT":
                                record(parent | {"record_id": parent["record_id"]+":"+mode+":not-generated",
                                    "mode": mode, "parent_record_id": parent["record_id"],
                                    "parent_correct": parent["correct"], "parent_extracted": parent["extracted"],
                                    "applicable": parent["status"] != "CORRECT", "correct": None, "extracted": None,
                                    "expected_n": 1,
                                    "status": "NOT_RETRIED_CORRECT" if parent["status"] == "CORRECT" else "DEPENDENCY_ERROR",
                                    "raw_text": None, "completion_token_ids": [], "logprobs": [], "n_tokens": None,
                                    "finish_reason": None, "slots": None, "elimination_baseline": None, "request": None})
                            else:
                                query(mode, QWEN_OFFICIAL, 1, parent=parent, index=index)
                    if "blind" in modes:
                        query("blind", QWEN_OFFICIAL, 4, blind=True)
            finally:
                for image in images:
                    image.close()
    summary = summarize(rows, k_list)
    summary["sampling_cost_counts"] = costs
    summary.update(results_sha256=_sha(output / "results.jsonl"), run_manifest_sha256=_sha(output / "run_manifest.json"),
                   sampler_calls_sha256=_sha(output / "sampler_calls.jsonl"),
                   verifier_manifest_sha256=_sha(output / "verifier_manifest.json"))
    if (_sha(split_root / "actor_inputs.jsonl") != manifest["actor_inputs_sha256"]
            or _sha(split_root / "manifest.yaml") != manifest["vault_manifest_sha256"]):
        raise ValueError("external: evaluation source changed; partial raw preserved")
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return summary
