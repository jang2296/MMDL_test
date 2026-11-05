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
                       MIN_TARGET_TOKENS, NOTE_INSTRUCTION, NOTE_INSTRUCTION_V2, NOTE_INSTRUCTION_V3, RESOLVE_HEADER,
                       SINGLE_VERDICT_LABELS)

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


VERSION_V4 = "error-note-protocol-v4"
V4_CONTRACT_HASHES = frozenset(("rows", "order", "init"))


def plain_protocol_v4(precision: str, total_steps: int, *, arm: str, round_index: int, start: int, limit: int,
                      contracts: dict[str, str | None]) -> dict:
    """RL + note SFT (claudedocs/plan_rl_sft_hybrid_v4_20261104.md section 3.5). contracts: rows/order SHA256 and the
    init checkpoint metadata SHA256 (None for a round that starts from F0). Every loss value is fixed here."""
    from .v4 import ARMS, LR_V4, RL_SETTINGS, ROLLOUT_N, SFT_WEIGHTS, VERSION as COLLECTION_VERSION

    if arm not in ARMS:
        raise ValueError(f"protocol v4: arm must be one of {ARMS}")
    if any(type(v) is not int for v in (round_index, start, limit)) or round_index < 1 or start < 0 or limit < 1:
        raise ValueError("protocol v4: round >= 1, start >= 0 and limit >= 1 required")
    if set(contracts) != V4_CONTRACT_HASHES or any(
            value is not None and (not isinstance(value, str) or len(value) != 64
                                   or any(c not in "0123456789abcdef" for c in value))
            for value in contracts.values()) or contracts["rows"] is None or contracts["order"] is None:
        raise ValueError("protocol v4: rows/order SHA256 and an init SHA256 or null required")
    if (round_index == 1) != (contracts["init"] is None):
        raise ValueError("protocol v4: round 1 starts from F0 and later rounds from an init checkpoint")
    body = {key: value for key, value in plain_protocol_v2(precision, total_steps).items() if key != "sha256"}
    weights = dict(SFT_WEIGHTS) if arm == "hybrid" else {**SFT_WEIGHTS, "A2": 0.0, "B": 0.0}
    body |= {"version": VERSION_V4, "optimizer": body["optimizer"] | {"lr": dict(LR_V4)},
             "prompts_sha256": _digest({"note": NOTE_INSTRUCTION_V3, "resolve_header": RESOLVE_HEADER,
                                        "collection": COLLECTION_VERSION}),
             "training_contract": {"arm": arm, "round": round_index, "start": start, "limit": limit,
                                   "hashes": dict(contracts), "master_seed": MASTER_SEED, "rollout_n": ROLLOUT_N,
                                   "rl": dict(RL_SETTINGS), "sft_weights": weights,
                                   "sft_row": "mean_target_and_eos_nll", "renormalize_missing_kind": False,
                                   "length_policy": {"min_target_tokens": MIN_TARGET_TOKENS,
                                                     "max_target_tokens": MAX_TARGET_TOKENS,
                                                     "loop_window": LOOP_WINDOW, "loop_ratio": LOOP_RATIO}}}
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


# v3 (user 2026-11-05): program-only row admission; RL replay over the stored generation-time support.
# v4: MCQ graded by the V8-aligned contract, >=32-token solutions for reward/A1/A2 (pilot v3 collapsed to bare
# answers), replay drift bounded per block and on long rows (one-token rows measure engine noise, not binding).
VERSION_SINGLE = "error-note-single-response-v4"
SINGLE_REWARD_GATE = "single-t1-program-gate-v2"
SINGLE_REPLAY = "stored-generation-support-v1"
SINGLE_REPLAY_BLOCK_MEAN_ATOL = 0.05
SINGLE_REPLAY_ROW_MEAN_ATOL = 0.1
SINGLE_REPLAY_ROW_MEAN_MIN_TOKENS = 64
SINGLE_REPLAY_MAX_ATOL = 3.0
SINGLE_ANSWER_CONTRACTS = {"mcq": "p0-eval-aligned-v1", "open": "training-answer-v2"}
SINGLE_CONTRACT_HASHES = frozenset(("source", "dataset", "order", "data_admission"))
SINGLE_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


def single_prompts_sha256() -> str:
    return _digest({"note": NOTE_INSTRUCTION_V3, "note_feedback": SINGLE_VERDICT_LABELS, "resolve_header": RESOLVE_HEADER})


def single_blocks(problem_ids: list[str], *, pilot_count: int = 300) -> list[list[str]]:
    """Pilot and continuation close separate 16-item blocks; every original ID occurs once."""
    if (not problem_ids or any(not isinstance(pid, str) or not pid for pid in problem_ids)
            or len(set(problem_ids)) != len(problem_ids) or type(pilot_count) is not int or pilot_count < 1):
        raise ValueError("single protocol: unique nonempty problem IDs and positive pilot count required")
    boundary = min(pilot_count, len(problem_ids))
    return [part[start:start + PER_UPDATE_ITEMS_V2]
            for part in (problem_ids[:boundary], problem_ids[boundary:])
            for start in range(0, len(part), PER_UPDATE_ITEMS_V2)]


def plain_protocol_single(precision: str, total_slots: int, *, contracts: dict[str, str],
                          token_scale: float, beta: float = 0.02,
                          sft_weights: dict[str, float] | None = None) -> dict:
    """Explicit local-data/source pins and an on-policy, one-response objective; no execution permission."""
    import math

    weights = {"A1": 0.05, "A2": 0.5, "B": 0.05} if sft_weights is None else dict(sft_weights)
    if (set(contracts) != SINGLE_CONTRACT_HASHES or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in contracts.values())):
        raise ValueError("single protocol: explicit source, dataset, order and data-admission SHA256 required")
    if (isinstance(token_scale, bool) or not isinstance(token_scale, (int, float))
            or not math.isfinite(token_scale) or token_scale <= 0
            or isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0
            or set(weights) != {"A1", "A2", "B"} or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 for value in weights.values())):
        raise ValueError("single protocol: finite fixed normalization and nonnegative weights required")
    body = {key: value for key, value in plain_protocol_v2(precision, total_slots).items() if key != "sha256"}
    body |= {"version": VERSION_SINGLE, "sampling": dict(SINGLE_SAMPLING),
             "prompts_sha256": single_prompts_sha256(),
             "training_contract": {
                 "version": VERSION_SINGLE, "hashes": dict(contracts), "master_seed": MASTER_SEED, "per_update_items": PER_UPDATE_ITEMS_V2,
                 "generation_counts": {"t1": 1, "note": 1, "resolve": 1}, "sampling": dict(SINGLE_SAMPLING),
                 "rl": {"method": "reinforce", "baseline": 0.0, "token_scale": float(token_scale),
                        "beta": float(beta), "reward": SINGLE_REWARD_GATE,
                        "kl": "raw_F0_to_current_on_all_completed_T1_prefixes"},
                 "sft_weights": weights, "sft_row": "mean_original_target_token_nll",
                 "length_policy": {"min_target_tokens": MIN_TARGET_TOKENS, "note_min_target_tokens": 1,
                                   "max_target_tokens": MAX_TARGET_TOKENS,
                                   "content_acceptance": "program_checks_only_content_unverified"},
                 "answer_contracts": dict(SINGLE_ANSWER_CONTRACTS),
                 "renormalize_missing_kind": False, "replay": "original_actions_same_live_policy_once",
                 "replay_support": {"method": SINGLE_REPLAY, "block_mean_atol": SINGLE_REPLAY_BLOCK_MEAN_ATOL,
                                    "row_mean_atol": SINGLE_REPLAY_ROW_MEAN_ATOL,
                                    "row_mean_min_tokens": SINGLE_REPLAY_ROW_MEAN_MIN_TOKENS,
                                    "row_max_atol": SINGLE_REPLAY_MAX_ATOL},
                 "zero_signal": "consume_slot_without_optimizer_or_decay", "total_slots": total_slots,
                 "pilot_boundary": 300, "pilot_main_optimizer": "continuous",
                 "routes": {"RL": "LM", "A1": "LM", "A2_B": "accepted_note_self_report"}}}
    return body | {"sha256": _digest(body)}
