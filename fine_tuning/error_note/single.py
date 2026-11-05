"""Same-policy, single-response RL + self-note SFT with program-only row admission (SINGLE v3).

The CLI collects a preregistered local prefix and proves backward without an optimizer.
Rows are admitted by program checks only (graded correct, normal stop, no loop, no lesson leak);
note/solution content is unverified. Paid execution is absent.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from .audit import audit_single_note, digest, single_note_review_bindings, text_sha256, validate_single_note_audit
from .collect import ResourceLog, StreamStats, hf_single_stream_builder, install_training_guards, write_json
from .collect_v3 import file_sha, generation_identity
from .collect_v4 import run_single_requests
from .contract import (CONTEXT_LIMIT, MASTER_SEED, MAX_NEW_TOKENS, ROUTE_FULL, ROUTE_LM,
                       note_prompt_single, parse_note_v2, resolve_prompt, sample_seed)
from .protocol import ARM, SINGLE_REWARD_GATE, VERSION_SINGLE, plain_protocol_single, single_blocks, single_prompts_sha256
from .v4 import target_problem

VERSION = "single-response-runtime-v3"
COMPONENTS = ("PG", "KL", "A1", "A2", "B")
GRADED = frozenset(("CORRECT", "INCORRECT", "TRUNCATED", "INVALID_NO_FINAL", "INVALID_MULTIPLE"))
# The verifier could not compare (e.g. an unbound unit): reward 0, no note (no truthful wrong/right feedback exists).
UNVERIFIABLE = "UNVERIFIABLE"


@contextmanager
def _runtime_lock(directory: Path, name: str):
    """An overlapping writer must stop before generation, phase mutation, or backward starts."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("single runtime: HOLD concurrent writer already owns this run") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("single runtime: expected a JSON object")
    return value


def _sealed(path: Path, value: dict) -> None:
    """Append a new immutable artifact, or verify its exact existing value."""
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"single runtime: immutable binding changed: {path.name}")
    else:
        write_json(path, value)


def _state(path: Path, value: dict | None = None) -> dict:
    if value is None:
        data = _read(path)
        if data.get("sha256") != digest({k: v for k, v in data.items() if k != "sha256"}):
            raise ValueError("single runtime: corrupt phase state")
        return data
    body = {k: v for k, v in value.items() if k != "sha256"}
    result = body | {"sha256": digest(body)}
    write_json(path, result)
    return result


def source_manifest() -> dict:
    """Describe actual source bytes, including an uncommitted working tree."""
    root = Path(__file__).resolve().parents[1]
    files = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob("*.py"))
             if not {"tests", "runs", "generated", "__pycache__"}.intersection(p.relative_to(root).parts)}
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return {"base_commit": base, "actual_files": files,
            "identity": "actual source manifest; base_commit does not assert a clean committed tree"}


def _protocol_check(protocol: dict) -> None:
    from .protocol import _digest

    if (protocol.get("version") != VERSION_SINGLE
            or protocol.get("sha256") != _digest({k: v for k, v in protocol.items() if k != "sha256"})
            or protocol.get("prompts_sha256") != single_prompts_sha256()
            or protocol["training_contract"].get("version") != VERSION_SINGLE
            or protocol["training_contract"]["rl"].get("reward") != SINGLE_REWARD_GATE
            or protocol["training_contract"]["generation_counts"] != {"t1": 1, "note": 1, "resolve": 1}):
        raise ValueError("single runtime: invalid single-response protocol")


def _candidate(record: dict, kind: str) -> dict | None:
    stage = {"RL": "t1", "A1": "t1", "A2": "resolve", "B": "note"}[kind]
    output = record.get(stage)
    if output is None:
        return None
    actions = output["generation"]["completion_token_ids"]
    if kind != "RL" and (not output["text"].strip()
                         or target_problem(output["text"], actions, output["finish_reason"], "CORRECT", min_tokens=1)):
        return None
    if kind in ("A1", "A2") and output["status"] != "CORRECT":
        return None
    parsed = parse_note_v2(record["note"]["text"]) if record.get("note") else None
    route = ROUTE_FULL if kind in ("A2", "B") and parsed and parsed.image_reading == "MISREAD" else ROUTE_LM
    return {"row_id": record["problem_id"] + ":" + kind, "kind": kind, "route": route,
            **{k: record[k] for k in ("problem_id", "image_paths", "image_sha256", "subject", "source")},
            "prompt": output["prompt"] if kind == "B" else record["p0_text"], "target": output["text"],
            "target_source": {"RL": "T1", "A1": "T1", "A2": "RESOLVE", "B": "NOTE"}[kind],
            "generation": output["generation"], "complete": True,
            "status": output.get("status", "NOTE"), "reward": 0 if kind == "RL" else int(record["t1"]["status"] == "CORRECT")}


def _single_reward_decision(row: dict, evidence: dict) -> dict:
    """Rebuild the program gate from the original T1: R=1 iff graded CORRECT and an admissible A1 target."""
    try:
        if (row["kind"] != "RL" or row["target_source"] != "T1" or row["route"] != ROUTE_LM
                or row["status"] not in GRADED | {"LOOP_STOPPED", UNVERIFIABLE}
                or not isinstance(evidence, dict) or set(evidence) != {"version", "t1_metadata", "provenance", "decision"}
                or evidence["version"] != SINGLE_REWARD_GATE or not isinstance(evidence["t1_metadata"], dict)
                or {"text", "generation", "prompt", "status"} & set(evidence["t1_metadata"])
                or row["target"] != row["generation"]["text"]):
            raise ValueError("single reward: malformed or unsupported original-T1 evidence")
        t1 = evidence["t1_metadata"] | {"text": row["target"], "generation": row["generation"],
                                        "prompt": row["prompt"], "status": row["status"]}
        record = {key: row[key] for key in ("problem_id", "image_paths", "image_sha256", "subject", "source")}
        record.update(p0_text=row["prompt"], t1=t1)
        bindings = single_note_review_bindings(record, evidence["provenance"])
        if row["status"] != "CORRECT":
            reason = "T1_UNVERIFIABLE" if row["status"] == UNVERIFIABLE else "T1_NOT_CORRECT"
        else:
            problem = target_problem(row["target"], row["generation"]["completion_token_ids"], t1["finish_reason"],
                                     "CORRECT", min_tokens=1)
            reason = None if _candidate(record, "A1") is not None else "T1_INELIGIBLE_" + (problem or "EMPTY_TARGET")
        return {"accepted": reason is None, "reason": reason, "bindings": bindings, "method": "program"}
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("single reward: malformed original-T1 evidence") from error


def bind_single_reward(row: dict, t1: dict, provenance: dict) -> dict:
    """Attach evidence without duplicating response text or sampled actions in the target."""
    if any(t1[key] != row[field] for key, field in (("text", "target"), ("prompt", "prompt"),
                                                  ("generation", "generation"), ("status", "status"))):
        raise ValueError("single reward: original T1 differs from the RL row")
    evidence: dict = {"version": SINGLE_REWARD_GATE, "t1_metadata": {k: v for k, v in t1.items()
                if k not in {"text", "generation", "prompt", "status"}}, "provenance": provenance, "decision": None}
    evidence["decision"] = _single_reward_decision(row, evidence)
    row.update(reward=int(evidence["decision"]["accepted"]), reward_evidence=evidence)
    return row


def validate_single_reward(row: dict) -> dict:
    evidence = row.get("reward_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("single reward: original-T1 program-gate evidence required")
    decision = _single_reward_decision(row, evidence)
    if (evidence["decision"] != decision or type(row.get("reward")) is not int
            or row["reward"] != int(decision["accepted"])):
        raise ValueError("single reward: reward/decision differs from the program gate on the original solution")
    return decision


def note_program_decision(audit: dict) -> dict:
    """Program-only note admission: parsed note that stopped normally and passed every bounded leak check."""
    base = audit["base_audit"]
    if base["format"]["result"] != "PASS":
        reason = "NOTE_FORMAT_INVALID"
    elif (failed := next((key for key, value in base["checks"].items() if value is not None), None)) is not None:
        reason = "NOTE_" + failed.upper()
    elif audit["single_hint_leak"] is not None:
        reason = "LESSON_LEAK_" + audit["single_hint_leak"]
    else:
        reason = None
    return {"accepted": reason is None, "reason": reason, "method": "program", "audit_sha256": audit["sha256"]}


def run_single_block(bundle: Any, **kwargs) -> dict:
    """Serialize the complete generation/review/row transaction for one output block."""
    with _runtime_lock(Path(kwargs["output"]), ".runtime.lock"):
        return _run_single_block(bundle, **kwargs)


def _run_single_block(bundle: Any, *, actors: dict, items: list[dict], provenance: dict, gold: Callable[[str], str],
                     grade: Callable, load_images: Callable, protocol: dict, output: Path, block_id: int = 0,
                     limit: int | None = None, stream: Callable | None = None,
                     stream_args: argparse.Namespace | None = None, stop_after: float | None = None) -> dict:
    """Collect at most 16 distinct items; resume completed actions; admit rows by program checks only.

    ``items`` is the immutable whole block, even when limit first requests its prefix of two.
    Gold is requested only while constructing the note/audit stage. It never reaches T1/retry requests.
    """
    from fine_tuning.notebook_rl.prompts import p0_eval_text, user_message
    from .rl import policy_sha256

    _protocol_check(protocol)
    ids = [item["problem_id"] for item in items]
    if not 1 <= len(ids) <= 16 or len(set(ids)) != len(ids) or any(pid not in actors or pid not in provenance for pid in ids):
        raise ValueError("single runtime: unique, source-bound block of 1..16 actors required")
    limit = len(ids) if limit is None else limit
    if type(limit) is not int or not 1 <= limit <= len(ids):
        raise ValueError("single runtime: requested prefix outside the fixed block")
    policy = policy_sha256(bundle)
    binding = {"version": VERSION, "block_id": block_id, "items": items, "protocol_sha256": protocol["sha256"],
               "policy_sha256": policy, "actors_sha256": digest({pid: vars(actors[pid]) for pid in ids}),
               "provenance_sha256": digest({pid: provenance[pid] for pid in ids})}
    output.mkdir(parents=True, exist_ok=True)
    _sealed(output / "binding.json", binding)
    state_path = output / "phase.json"
    state = _state(state_path) if state_path.exists() else {"version": VERSION, "binding_sha256": digest(binding),
                                                          "records": {}, "limit": 0}
    if state["binding_sha256"] != digest(binding) or limit < state["limit"]:
        raise ValueError("single runtime: stale policy/binding or prefix moved backwards")
    state["limit"] = limit
    active = ids[:limit]
    stats = StreamStats()
    if stream is None:
        args = argparse.Namespace(**vars(stream_args)) if stream_args is not None else argparse.Namespace()
        args.policy_sha256, args.protocol_sha256 = policy, protocol["sha256"]
        stream = (getattr(args, "stream_builder", None) or hf_single_stream_builder)(bundle, args, stats)

    def checkpoint(status: str, **extra) -> dict:
        if policy_sha256(bundle) != policy:
            raise ValueError("single runtime: policy changed during collection")
        state.update(status=status, **extra)
        return _state(state_path, state)

    def lesson(pid: str) -> str:
        parsed = parse_note_v2(state["records"][pid]["note"]["text"])
        if parsed is None:
            raise ValueError("single runtime: accepted note is no longer parseable")
        return parsed.lesson

    def request(pid: str, stage: str, prompt: str) -> dict:
        return {"request_id": f"{pid}:{stage}", "problem_id": pid, "block_id": block_id, "stage": stage,
                "seed": sample_seed(pid, "t1" if stage == "t1" else stage), "max_tokens": MAX_NEW_TOKENS,
                "messages": [user_message(prompt, len(actors[pid].image_paths))], "images": load_images(actors[pid]),
                "prompt": prompt}

    def accept(stage: str, pid: str, output_: dict, prompt: str) -> None:
        result = dict(output_) | {"prompt": prompt}
        if stage != "note":
            status, extracted = grade(pid, result["text"], result["finish_reason"], actors[pid].mode)
            if status not in GRADED | {UNVERIFIABLE}:
                raise ValueError(f"single runtime: ungraded/system outcome blocks the block: {status}")
            result.update(status=status, extracted=extracted)
        if stage == "resolve":
            result["lesson_sha256"] = text_sha256(lesson(pid))
        previous = state["records"][pid].get(stage)
        if previous is not None and previous != result:
            raise ValueError("single runtime: completed generation/grade changed")
        state["records"][pid][stage] = result
        checkpoint("COLLECTING_" + stage.upper())

    def note_prompt(pid: str) -> str:
        t1 = state["records"][pid]["t1"]
        return note_prompt_single(state["records"][pid]["p0_text"], t1["text"], t1["status"], gold(pid))

    def note_decision(pid: str) -> tuple[dict, dict]:
        record, source = state["records"][pid], provenance[pid]
        audit = audit_single_note(record, gold=gold(pid), choices=actors[pid].choices, provenance=source,
                                  aliases=source.get("leak_aliases", ()),
                                  acronym_candidates=source.get("acronym_candidates", ()))
        validate_single_note_audit(audit, record, source)
        return audit, note_program_decision(audit)

    def is_wrong(pid: str) -> bool:
        return state["records"][pid]["t1"]["status"] not in ("CORRECT", UNVERIFIABLE)

    def generate(stage: str, prompts: dict[str, str]) -> bool:
        requests = [request(pid, stage, prompt) for pid, prompt in prompts.items()]
        if not requests:
            return True
        results = run_single_requests([{k: v for k, v in r.items() if k != "prompt"} for r in requests], stream, stats,
                                      output, policy_sha256=policy, protocol_sha256=protocol["sha256"], stop_after=stop_after)
        for item in requests:
            if item["request_id"] in results:
                accept(stage, item["problem_id"], results[item["request_id"]], item["prompt"])
        return all(stage in state["records"][pid] for pid in prompts)

    def pipeline() -> None:
        """Continuous engines: a question's note/re-solve starts as soon as its own previous stage ends."""
        prompts: dict[str, str] = {}

        def submit(item: dict) -> dict:
            prompts[item["request_id"]] = item.pop("prompt")
            return item

        def follow(item: dict, result: dict) -> list[dict]:
            pid, stage = item["problem_id"], item["stage"]
            accept(stage, pid, result, prompts[item["request_id"]])
            if stage == "t1" and is_wrong(pid):
                return [submit(request(pid, "note", note_prompt(pid)))]
            if stage == "note" and note_decision(pid)[1]["accepted"]:
                return [submit(request(pid, "resolve", resolve_prompt(state["records"][pid]["p0_text"], lesson(pid))))]
            return []

        run_single_requests([submit(request(pid, "t1", state["records"][pid]["p0_text"])) for pid in active], stream,
                            stats, output, policy_sha256=policy, protocol_sha256=protocol["sha256"],
                            stop_after=stop_after, follow_up=follow)

    for item in items[:limit]:
        pid, actor = item["problem_id"], actors[item["problem_id"]]
        metadata = {"problem_id": pid, "p0_text": p0_eval_text(actor.question, actor.choices),
                    "image_paths": list(actor.image_paths), "image_sha256": list(actor.image_sha256),
                    "subject": item["subject"], "source": actor.source}
        if pid not in state["records"]:
            state["records"][pid] = metadata
        elif any(state["records"][pid][k] != value for k, value in metadata.items()):
            raise ValueError("single runtime: actor metadata changed")
    if getattr(stream, "continuous", False):
        pipeline()  # The staged pass below then reuses every completion from the ledger without generating.
    if not generate("t1", {pid: state["records"][pid]["p0_text"] for pid in active}):
        return checkpoint("INCOMPLETE_T1")
    wrong = [pid for pid in active if is_wrong(pid)]
    if not generate("note", {pid: note_prompt(pid) for pid in wrong}):
        return checkpoint("INCOMPLETE_NOTE")
    audits, note_decisions = {}, {}
    for pid in wrong:
        audits[pid], note_decisions[pid] = note_decision(pid)
    approved = [pid for pid in wrong if note_decisions[pid]["accepted"]]
    if not generate("resolve", {pid: resolve_prompt(state["records"][pid]["p0_text"], lesson(pid))
                                for pid in approved}):
        return checkpoint("INCOMPLETE_RESOLVE", note_audits=audits, note_decisions=note_decisions)
    rows, decisions = [], {}
    for pid in active:
        record = state["records"][pid]
        rl_row = _candidate(record, "RL")
        if rl_row is None:
            raise ValueError("single runtime: original T1 is missing")
        rows.append(bind_single_reward(rl_row, record["t1"], provenance[pid]))
        kind = "A1" if record["t1"]["status"] == "CORRECT" else "A2"
        if kind == "A2" and pid not in approved:
            continue
        row = _candidate(record, kind)
        decisions[f"{pid}:{kind}"] = {"accepted": row is not None, "method": "program",
                                       "reason": None if row is not None else f"{kind}_NOT_ADMISSIBLE"}
        if row is None:
            continue
        rows.append(row)
        if kind == "A2" and (note_row := _candidate(record, "B")) is not None:
            rows.append(note_row)
    contract = protocol["training_contract"] | {"expected_problem_ids": active, "policy_sha256": policy,
        "reward_provenance_sha256": {pid: digest(provenance[pid]) for pid in active},
        "reward_evidence_sha256": {row["problem_id"]: digest(row["reward_evidence"]) for row in rows if row["kind"] == "RL"}}
    result = {"version": VERSION, "status": "ROWS_READY", "binding_sha256": digest(binding), "limit": limit,
              "rows": rows, "rows_sha256": digest(rows), "contract": contract, "note_audits": audits,
              "note_decisions": note_decisions, "row_decisions": decisions}
    write_json(output / f"rows-{limit}.json", result)
    return checkpoint("ROWS_READY", note_audits=audits, note_decisions=note_decisions, row_decisions=decisions,
                      rows_path=f"rows-{limit}.json")


def prove_single_backward(bundle: Any, encoded: list[dict], contract: dict, *, precision: str,
                          token_budget: int = 0, capacity: Any = None) -> dict:
    """Independent component gradients; no optimizer, scheduler, or model checkpoint exists here."""
    from .train import _all_parameter_hashes, _gradient_norms
    from .train_v4 import backward_single_block
    from fine_tuning.notebook_rl.policy import assert_base_unchanged
    from fine_tuning.notebook_rl.quantized_policy import assert_quantized_base_unchanged

    check_base = assert_quantized_base_unchanged if precision == "nf4" else assert_base_unchanged
    before = _all_parameter_hashes(bundle)
    report: dict[str, Any] = {"status": "HOLD", "optimizer_steps": 0, "scheduler_steps": 0, "weights_saved": False,
              "rows_by_kind": dict(Counter(row["kind"] for row in encoded)), "components": {}, "missing": []}
    try:
        check_base(bundle)
        for name in COMPONENTS:
            started = time.monotonic()
            bundle.model.zero_grad(set_to_none=True)
            stats = backward_single_block(bundle, encoded, contract, token_budget, components=(name,), capacity=capacity)
            norms = _gradient_norms(bundle)
            rows = [r for r in encoded if r["kind"] == ("RL" if name in ("PG", "KL") else name)]
            signal = bool(stats["has_gradient"])
            full = name in ("A2", "B") and any(row["route"] == ROUTE_FULL for row in rows)
            expected = {"L": signal, "M": signal, "V": signal and full}
            positive = {group: value is not None and value > 0 for group, value in norms.items()}
            valid = all(positive.get(group, False) == need for group, need in expected.items())
            report["components"][name] = {"stats": stats, "gradient_norms": norms, "route_valid": valid,
                                           "zero_F0_KL_allowed": name == "KL" and not signal,
                                           "seconds": time.monotonic() - started}
            if name != "KL" and (not rows or not signal):
                report["missing"].append(name)
            if not valid:
                report["missing"].append(name + ":GRADIENT_ROUTE")
        a2 = {r["problem_id"] for r in encoded if r["kind"] == "A2"}
        notes = {r["problem_id"] for r in encoded if r["kind"] == "B"}
        incorrect = {r["problem_id"] for r in encoded if r["kind"] == "RL" and r["status"] == "INCORRECT"}
        report["paired_incorrect_a2_b"] = sorted(a2 & notes & incorrect)
        if not report["paired_incorrect_a2_b"]:
            report["missing"].append("SAME_PROBLEM_INCORRECT_A2_B")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["missing"].append("BACKWARD_FAILED")
    finally:
        bundle.model.zero_grad(set_to_none=True)
        check_base(bundle)
        report["all_parameters_unchanged"] = before == _all_parameter_hashes(bundle)
        report["base_unchanged"] = True
        report["gradients_cleared"] = all(p.grad is None for p in bundle.model.parameters())
    report["status"] = ("FAIL" if report.get("error") or not report["all_parameters_unchanged"] else
                        "HOLD" if report["missing"] else "PASS")
    return report


def run_joint_single_update(bundle: Any, optimizer: Any, scheduler: Any, encoded: list[dict], contract: dict, *,
                            cursor: dict, block_ids: list[str], allow_update: bool = False, token_budget: int = 0,
                            capacity: Any = None) -> dict:
    """Explicit caller opt-in, one joint step; retain the caller's optimizer across the pilot boundary."""
    import torch
    from .rl import policy_sha256
    from .train_v4 import backward_single_block

    if allow_update is not True:
        raise ValueError("single runtime: parameter updates require explicit caller opt-in")
    slot = cursor["slot"]
    if (type(slot) is not int or not 0 <= slot < contract["total_slots"]
            or cursor["policy_sha256"] != contract["policy_sha256"]
            or cursor["policy_sha256"] != policy_sha256(bundle)
            or contract["expected_problem_ids"] != block_ids or cursor["consumed_problem_ids"][-len(block_ids):] == block_ids
            or set(block_ids) & set(cursor["consumed_problem_ids"])):
        raise ValueError("single runtime: duplicate, stale or invalid continuation cursor")
    optimizer.zero_grad(set_to_none=True)
    try:
        if capacity is not None:
            capacity.optimizer = optimizer
        stats = backward_single_block(bundle, encoded, contract, token_budget, capacity=capacity)
        if stats["has_gradient"]:
            for parameters in bundle.params.values():
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                if not bool(torch.isfinite(norm)):
                    raise ValueError("single runtime: nonfinite gradient")
            optimizer.step()
        scheduler.step()
    finally:
        optimizer.zero_grad(set_to_none=True)
    cursor.update(slot=slot + 1, updates=cursor["updates"] + int(stats["has_gradient"]),
                  consumed_problem_ids=cursor["consumed_problem_ids"] + block_ids,
                  policy_sha256=policy_sha256(bundle))
    return {"stats": stats, "cursor": cursor, "optimizer_steps": int(stats["has_gradient"]), "scheduled_slots": 1}


def save_single_continuation(bundle: Any, optimizer: Any, scheduler: Any, *, cursor: dict,
                             path: Path, protocol: dict, binding: dict, allow_save: bool = False) -> None:
    """Use the existing atomic checkpoint format; the caller must separately authorize real weight saving."""
    import os
    from fine_tuning.notebook_rl import checkpoint

    _protocol_check(protocol)
    if allow_save is not True or path.exists() or path.with_name(path.name + ".tmp").exists():
        raise ValueError("single runtime: explicit save opt-in and unused checkpoint paths required")
    bundle.route.apply(True)
    temporary = path.with_name(path.name + ".tmp")
    checkpoint.save(bundle, optimizer, scheduler, cursor["slot"],
                    {"epoch": 0, "position": cursor["slot"], "discarded_problem_ids": [], "single": cursor},
                    checkpoint.capture_rng(), temporary, protocol=protocol, binding=binding)
    os.replace(temporary, path)


def load_single_continuation(model_path: Path, path: Path, protocol: dict, *, binding: dict,
                             device: str = "cpu", policy_factory: Callable | None = None) -> tuple:
    """Restore the existing BF16/CPU checkpoint format, including moments, LR, RNG and single-use cursor."""
    from fine_tuning.notebook_rl import checkpoint
    from .rl import policy_sha256

    _protocol_check(protocol)
    if protocol["precision"] != "bf16":
        raise ValueError("single runtime: NF4 update/reload is not validated; use no-update local proof")
    bundle, optimizer, scheduler, slot, saved = checkpoint.load_resume(
        model_path, path, protocol, binding=binding, device=device, policy_factory=policy_factory)
    cursor = saved.get("single", {})
    if (cursor.get("slot") != slot or saved["position"] != slot
            or not 0 <= slot <= protocol["training_contract"]["total_slots"]
            or cursor.get("policy_sha256") != policy_sha256(bundle)):
        raise ValueError("single runtime: continuation policy/slot mismatch")
    return bundle, optimizer, scheduler, cursor


def _validate_data(args: argparse.Namespace) -> tuple[dict, dict]:
    from fine_tuning.notebook_rl.local_run import frozen_approval_output
    from .frozen_data import verify_frozen_data
    from .order import read_order

    approval = _read(args.frozen_approval)
    order = read_order(args.order, expected_sha256=args.order_sha256)
    if args.mode == "frozen3500":
        checked = verify_frozen_data(args.build_root, approval, args.order, order_sha256=args.order_sha256)
    else:
        selected, _, checked = frozen_approval_output(args.build_root, approval)
        if len(order["items"]) < 10 or any(i["problem_id"] not in selected["selected_ids"] for i in order["items"]):
            raise ValueError("single runtime: local smoke order is not an approved training selection")
        checked = checked | {"scope": "old approved pool local fixed10 only; not 3500-ready"}
    return order, checked


def prepare(args: argparse.Namespace) -> dict:
    """Serialize CPU preparation, including immutable input/source/capacity binding."""
    with _runtime_lock(args.output, ".prepare.lock"):
        return _prepare(args)


def _prepare(args: argparse.Namespace) -> dict:
    """CPU-only freeze. Revalidate source, data and model-file identity on every invocation."""
    import math

    measured = getattr(args, "capacity_profile", None)
    measured_sha = getattr(args, "capacity_profile_sha256", None)
    capture_path = getattr(args, "capture_capacity", None)
    reserve = getattr(args, "capacity_reserve_bytes", None)
    if (measured is None) != (measured_sha is None):
        raise ValueError("single runtime: capacity profile and externally pinned SHA256 are required together")
    if (capture_path is None) != (reserve is None) or capture_path is not None and (
            measured is not None or type(reserve) is not int or reserve <= 0 or capture_path.exists()):
        raise ValueError("single runtime: capture requires a fresh output, explicit positive reserve, and no admission profile")
    for name, explicit_default, measured_default in (("max_num_seqs", 1, 16),
            ("max_num_batched_tokens", CONTEXT_LIMIT, 16 * CONTEXT_LIMIT), ("token_budget", 0, 16 * CONTEXT_LIMIT)):
        if getattr(args, name, None) is None:
            setattr(args, name, measured_default if measured is not None else explicit_default)
    precision = getattr(args, "precision", "nf4")
    if precision not in ("nf4", "bf16"):
        raise ValueError("single runtime: explicit nf4 or bf16 precision required")

    if (type(args.max_num_seqs) is not int or not 1 <= args.max_num_seqs <= 16
            or type(args.max_num_batched_tokens) is not int or args.max_num_batched_tokens < 1
            or type(args.token_budget) is not int or args.token_budget < 0
            or args.stop_after_seconds is not None and (not math.isfinite(args.stop_after_seconds) or args.stop_after_seconds <= 0)):
        raise ValueError("single runtime: explicit positive generation capacity/deadline and nonnegative training budget required")
    order, admission = _validate_data(args)
    from fine_tuning.notebook_rl.schema import to_json
    from fine_tuning.notebook_rl.vault import read_actors
    from .answer_contract import VERSION as RULES_VERSION
    from .source_evidence import VERSION as SOURCE_VERSION

    document = _read(args.source_provenance)
    sources = document["items"]
    actors = {actor.problem_id: actor for actor in read_actors(args.build_root / "train")}
    train_sha = file_sha(args.build_root / "train" / "manifest.yaml")
    rules = _read(args.training_rules)
    if (document.get("version") != SOURCE_VERSION or rules.get("version") != RULES_VERSION
            or rules.get("vault_manifest_sha256") != train_sha):
        raise ValueError("single runtime: source evidence/rules are not bound to the approved training vault")
    for item in order["items"][:10]:
        pid = item["problem_id"]
        actor, evidence = actors.get(pid), sources.get(pid, {})
        if (actor is None or item["source"] != actor.source or evidence.get("source") != actor.source
                or evidence.get("actor_sha256") != text_sha256(to_json(actor))
                or evidence.get("image_sha256") != list(actor.image_sha256)
                or evidence.get("vault_manifest_sha256") != train_sha or evidence.get("split") != "train"):
            raise ValueError("single runtime: fixed10 source/actor/original image binding mismatch")
    source = source_manifest()
    ids = [i["problem_id"] for i in order["items"]]
    blocks = single_blocks(ids)
    model_config = _read(args.model_path / "config.json")
    if (args.model_path / "nf4_base.json").exists() or model_config.get("quantization_config") is not None:
        raise ValueError("single runtime: --model-path requires the raw BF16 snapshot; NF4 cache needs its separate verified loader")
    generation = generation_identity(argparse.Namespace(engine="hf", model_path=args.model_path, code_commit=source["base_commit"])) | {"precision": precision}
    contracts = {"source": digest(source), "dataset": file_sha(args.build_root / "manifest.yaml"),
                 "order": args.order_sha256, "data_admission": digest(admission)}
    protocol = plain_protocol_single(precision, len(blocks), contracts=contracts, token_scale=args.token_scale)
    body = {"version": VERSION, "mode": args.mode, "source": source, "generation": generation,
            "protocol": protocol, "data_admission": admission, "fixed10": order["items"][:10],
            "blocks": blocks, "source_provenance_sha256": digest(sources), "rules_sha256": file_sha(args.training_rules),
            "capacity": {"max_num_seqs": args.max_num_seqs, "padded_generation_token_budget": args.max_num_batched_tokens,
                         "training_token_budget": args.token_budget, "context_limit": CONTEXT_LIMIT,
                         "stop_after_seconds": args.stop_after_seconds},
            "runtime_implemented": True, "actual_model_execution": False, "cloud_ready": False}
    if measured is not None:
        from .capacity import load_profile

        if not isinstance(measured_sha, str):
            raise ValueError("single runtime: expected capacity profile SHA256 required")
        profile = load_profile(measured, measured_sha)
        expected = {"source_sha256": contracts["source"], "model_sha256": generation["model_files_sha256"], "precision": precision}
        if any(profile["binding"].get(key) != value for key, value in expected.items()):
            raise ValueError("single runtime: measured capacity source/model/precision differs from this plan")
        body["capacity"]["measured_profile"] = {"path": str(Path(measured).absolute()), "file_sha256": measured_sha}
    if capture_path is not None:
        # The fresh partial filename may change across review/resume invocations; capture behavior stays sealed.
        body["capacity"]["capture"] = {"enabled": True, "reserve_bytes": reserve}
    _sealed(args.output / "plan.json", body)
    return body


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "dry-run"))
    for name in ("build-root", "order", "frozen-approval", "source-provenance", "training-rules", "model-path", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--order-sha256", required=True)
    parser.add_argument("--mode", choices=("local-smoke", "frozen3500"), default="local-smoke")
    parser.add_argument("--token-scale", type=float, default=1024.)
    parser.add_argument("--limit", type=int, choices=(2, 10), default=2)
    parser.add_argument("--precision", choices=("nf4", "bf16"), default="nf4")
    parser.add_argument("--capacity-profile", type=Path)
    parser.add_argument("--capacity-profile-sha256")
    parser.add_argument("--capture-capacity", type=Path, help="fresh actual-measurement partial; never generates extra answers")
    parser.add_argument("--capacity-reserve-bytes", type=int, help="explicit safety reserve, paired with --capture-capacity")
    parser.add_argument("--max-num-seqs", type=int, help="default 1; measured-profile mode ceiling 16")
    parser.add_argument("--max-num-batched-tokens", type=int, help="default one context; profile mode ceiling 16 contexts")
    parser.add_argument("--token-budget", type=int, help="default singleton rows; profile mode ceiling 16 contexts")
    parser.add_argument("--stop-after-seconds", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with _runtime_lock(args.output, ".cli.lock"):
        return _main(args)


def _main(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    plan = prepare(args)
    if args.command == "prepare":
        print(json.dumps({"status": "PREPARED", "model_loaded": False, "cloud_ready": False}))
        return 0
    import random
    import numpy as np
    import torch
    from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.policy import enable_deterministic_execution
    from fine_tuning.notebook_rl.policy import build_policy
    from fine_tuning.notebook_rl.quantized_policy import build_quantized_policy
    from fine_tuning.notebook_rl.vault import read_actors, read_verifier
    from fine_tuning.notebook_rl.verifier_service import VerifierService
    from .train_v4 import encode_single_row

    enable_deterministic_execution()
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)
    actors = {a.problem_id: a for a in read_actors(args.build_root / "train")}
    sources = _read(args.source_provenance)["items"]
    # The trusted controller keeps only fixed10 gold strings; model requests never receive this mapping.
    private = read_verifier(args.build_root / "train")
    gold_values = {item["problem_id"]: private[item["problem_id"]].gold for item in plan["fixed10"]}
    del private
    private_files = [args.training_rules, args.source_provenance]
    forbidden = install_training_guards(args.build_root, private_files=private_files)
    profile = None
    if "measured_profile" in plan["capacity"]:
        from .capacity import load_profile

        descriptor = plan["capacity"]["measured_profile"]
        profile = load_profile(Path(descriptor["path"]), descriptor["file_sha256"])
    precision = getattr(args, "precision", "nf4")
    bundle = (build_quantized_policy if precision == "nf4" else build_policy)(args.model_path, ARM, plan["protocol"])
    if precision == "bf16":
        bundle.model.to("cuda:0")
    capacity, capture = None, None
    identity = {"source_sha256": digest(plan["source"]), "model_sha256": plan["generation"]["model_files_sha256"], "precision": precision}
    if profile is not None:
        from .capacity import Capacity

        capacity = Capacity(bundle, profile, identity)
        args.capacity_session = capacity
    if getattr(args, "capture_capacity", None) is not None:
        from .capacity import Capture

        capture = Capture(bundle, identity, args.capacity_reserve_bytes)
        args.capacity_capture = capture

    def gold(pid):
        return gold_values[pid]

    try:
        with VerifierService(args.build_root / "train", forbidden_roots=forbidden, training_rules_path=args.training_rules,
                             training_rules_sha256=plan["rules_sha256"]) as verifier, ResourceLog(args.output / "resources.jsonl"):
            def grade(pid, text, finish_reason, mode):
                result = verifier.grade(pid, text, finish_reason, mode, answer_contract=TRAINING_ANSWER_CONTRACT)
                return result.status, result.extracted

            state = run_single_block(bundle, actors=actors, items=plan["fixed10"], provenance=sources, gold=gold, grade=grade,
                                     load_images=lambda actor: load_images(actor, args.build_root / "images"),
                                     protocol=plan["protocol"], output=args.output / "block-0", limit=args.limit,
                                     stream_args=args, stop_after=args.stop_after_seconds)
        if args.command == "dry-run" and state["status"] == "ROWS_READY":
            closed = _read(args.output / "block-0" / state["rows_path"])
            encoded = [encode_single_row(bundle.processor, bundle.model.config, row,
                                         load_images(actors[row["problem_id"]], args.build_root / "images")) for row in closed["rows"]]
            with ResourceLog(args.output / "backward-resources.jsonl"):
                report = prove_single_backward(bundle, encoded, closed["contract"], precision=precision,
                                               token_budget=args.token_budget, capacity=capacity)
            _sealed(args.output / f"dry-run-{args.limit}.json", report)
            if capture is not None:
                if report.get("error") or report["status"] == "FAIL" or report.get("all_parameters_unchanged") is False:
                    capture.failed = True
                else:
                    from .capacity import capture_backward

                    try:
                        capture_backward(capture, encoded, closed["contract"], component="RL", token_budget=args.token_budget)
                        for kind in ("A1", "A2", "B"):
                            if any(row["kind"] == kind for row in encoded):
                                capture_backward(capture, encoded, closed["contract"], component=kind, token_budget=args.token_budget)
                    except BaseException:
                        capture.failed = True
                        raise
            print(json.dumps(report))
            return 0 if report["status"] == "PASS" else 2
        print(json.dumps({"status": state["status"], "waiting": state.get("waiting", []), "weights_saved": False}))
        return 0 if state["status"] == "ROWS_READY" else 2
    finally:
        if capture is not None:
            pending = sys.exception()
            try:
                partial = capture.write(args.capture_capacity)
                print(json.dumps({"capacity_capture": str(args.capture_capacity), "status": partial["status"],
                                  "file_sha256": file_sha(args.capture_capacity), "entries": len(partial["entries"])}))
            except Exception as error:
                if pending is None:
                    raise
                pending.add_note(f"capacity partial publication failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
