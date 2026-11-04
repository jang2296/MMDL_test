"""The plain training protocol dict for fine_tuning.notebook_rl.policy and checkpoint.

Learning rates are the recipe G6 actually ran; only the training target changed (plan section 3.4).
Version 2 (claudedocs/plan_self_note_v2_20261102.md section 4): the same model, LoRA, learning rates and clip;
16 items per update and a linear decay to 0 after the 10-step warmup.
"""
from __future__ import annotations

import hashlib
import json

from fine_tuning.modules import BASE_ID, BASE_REVISION
from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT

from .contract import (CONTEXT_LIMIT, IMAGE, LOOP_RATIO, LOOP_WINDOW, MASTER_SEED, MAX_NEW_TOKENS, MAX_TARGET_TOKENS,
                       MIN_TARGET_TOKENS, NOTE_INSTRUCTION, NOTE_INSTRUCTION_V2, NOTE_INSTRUCTION_V3, RESOLVE_HEADER)

ARM = "MOD-RV"  # installs Vision, LLM and Merger trainables; the per-item route decides who gets gradients
MERGERS = ["model.visual.merger", "model.visual.deepstack_merger_list.0",
           "model.visual.deepstack_merger_list.1", "model.visual.deepstack_merger_list.2"]
FORBIDDEN = {"additional_sft": False, "teacher_or_upper_model_labels": True, "benchmark_training": True}
GRAD_CLIP = 1.0
VERSION_V2 = "error-note-protocol-v2"
PER_UPDATE_ITEMS_V2 = 16
VERSION_V3 = "error-note-protocol-v3"
ADMISSION_VERSION_V3 = "error-note-admission-v4"
V3_CONTRACT_HASHES = frozenset(("rows", "order", "audit", "sidecar", "solution_audit", "admission"))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def plain_protocol(precision: str) -> dict:
    """precision: "bf16" on the rented GPU, "nf4" for local functional checks only."""
    if precision not in ("bf16", "nf4"):
        raise ValueError("protocol: precision must be bf16 or nf4")
    body: dict = {
        "version": "error-note-protocol-v1", "precision": precision, "forbidden": FORBIDDEN,
        "initial_checkpoint": {"model_id": BASE_ID, "revision": BASE_REVISION},
        "image": dict(IMAGE), "max_new_tokens": MAX_NEW_TOKENS, "context_limit": CONTEXT_LIMIT,
        "model": {"lora": {"r": 16, "alpha": 32, "dropout": 0.0, "bias": "none"}, "modules_to_save": MERGERS},
        "expected_trainable": {"L": 33030144, "M": 109105152, "V": 2359296},
        "optimizer": {"type": "adamw", "schedule": "constant_after_warmup", "warmup_steps": 10,
                      "lr": {"L": 5e-5, "M": 1e-5, "V": 1e-5}, "betas": [0.9, 0.95], "eps": 1e-8,
                      "weight_decay": 0.0, "grad_clip_per_group": GRAD_CLIP},
        "prompts_sha256": hashlib.sha256(NOTE_INSTRUCTION.encode()).hexdigest(),
    }
    if precision == "nf4":
        from fine_tuning.notebook_rl.quantized_policy import quantization_contract

        body["quantization"] = quantization_contract()
    return body | {"sha256": _digest(body)}


def plain_protocol_v2(precision: str, total_steps: int) -> dict:
    """total_steps: the optimizer steps of the run (its non-empty update groups); the multiplier reaches 0 there."""
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("protocol: total_steps must be a positive integer")
    body = {key: value for key, value in plain_protocol(precision).items() if key != "sha256"}
    body |= {"version": VERSION_V2, "per_update_items": PER_UPDATE_ITEMS_V2,
             "optimizer": body["optimizer"] | {"schedule": "linear_decay_after_warmup", "total_steps": total_steps},
             "prompts_sha256": _digest({"note": NOTE_INSTRUCTION_V2, "resolve_header": RESOLVE_HEADER})}
    return body | {"sha256": _digest(body)}


def plain_protocol_v3(precision: str, total_steps: int, *, arm: str, contracts: dict[str, str]) -> dict:
    """The common admitted update schedule, with arm-specific loss and immutable input receipts."""
    from .audit import SOLUTION_REVIEW_VERSION

    if arm not in ("note", "no-note"):
        raise ValueError("protocol v3: arm must be note or no-note")
    if set(contracts) != V3_CONTRACT_HASHES or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in contracts.values()):
        raise ValueError("protocol v3: rows/order/audit/sidecar/solution-audit/admission SHA256 bindings required")
    body = {key: value for key, value in plain_protocol_v2(precision, total_steps).items() if key != "sha256"}
    body |= {"version": VERSION_V3,
             "prompts_sha256": _digest({"note": NOTE_INSTRUCTION_V3, "resolve_header": RESOLVE_HEADER}),
             "training_contract": {"arm": arm, "hashes": dict(contracts), "master_seed": MASTER_SEED,
                                   "answer_contract": TRAINING_ANSWER_CONTRACT, "audit_contract": "note-audit-v1",
                                   "solution_content_contract": SOLUTION_REVIEW_VERSION,
                                   "admission": ADMISSION_VERSION_V3, "active_group": "at_least_one_encoded_A",
                                   "route": "preserve_same_A_in_both_arms", "unexpected_drop": "STOP",
                                   "length_policy": {"min_target_tokens": MIN_TARGET_TOKENS,
                                                     "max_target_tokens": MAX_TARGET_TOKENS,
                                                     "loop_window": LOOP_WINDOW, "loop_ratio": LOOP_RATIO},
                                   "loss": {"row": "mean_target_and_eos_nll", "A": 0.9,
                                            "B": 0.1 if arm == "note" else 0.0, "renormalize_missing_kind": False}}}
    return body | {"sha256": _digest(body)}
