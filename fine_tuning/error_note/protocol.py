"""The plain training protocol dict for fine_tuning.notebook_rl.policy and checkpoint.

Learning rates are the recipe G6 actually ran; only the training target changed (plan section 3.4).
Version 2 (claudedocs/plan_self_note_v2_20261102.md section 4): the same model, LoRA, learning rates and clip;
16 items per update and a linear decay to 0 after the 10-step warmup.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from fine_tuning.modules import BASE_ID, BASE_REVISION
from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT, TURN1_EVAL_CONTRACT

from .contract import (CONTEXT_LIMIT, IMAGE, LOOP_RATIO, LOOP_WINDOW, MASTER_SEED, MAX_NEW_TOKENS, MAX_TARGET_TOKENS,
                       MIN_TARGET_TOKENS, NOTE_INSTRUCTION, NOTE_INSTRUCTION_SINGLE, NOTE_INSTRUCTION_V2, NOTE_INSTRUCTION_V3,
                       REFERENCE_HEADER, RESOLVE_HEADER, RESOLVE_INSTRUCTION_SINGLE, SINGLE_VERDICT_LABELS, DENIAL_PHRASES,
                       GRADER_READ_LINE, GRADER_READ_NONE, LESSON_STEM_RULE, _REVIEW_REFERENCES)
from .single_grading import ALIASES, ALIASES_SHA256, NORMALIZATION_RULES, NUMBER_WORDS, SCORER, SUFFIX_STOP

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
# v5 (2026-11-06, main v4 run stopped at slot 75): V2 note instruction (V3 taught image-only hedging) and reward -1
# for a normally finished T1 under 32 tokens (bare answers rose 28% -> 44% with no penalty), see
# claudedocs/analysis_single_rlsft_failure_20261106.md.
# v6 (2026-11-06): open answers also use the V8-aligned contract MMMU scores with. training-answer-v2 rejected
# common final-answer notation (emoji/bold/heading, a repeated boxed value): on the stored v4 T1, 262 of 514 open
# answers were INVALID, so their notes reviewed parser output; V8 grading gives +33 correct open answers
# (-24 bare range lines) and A2-eligible re-solves 50 -> 76.
# v7 (2026-11-06, cycle-1 64-item review): open answers V8 first, then training-answer-v2 only to turn a non-CORRECT
# V8 result CORRECT (unit/word false negatives); reward -1 only for a wrong bare answer, a stopped answer the parser
# could not commit to, or a looping wrong answer (correct bare answers 0); notes capped at 1024 tokens; note and
# re-solve instructions; program filters on A2 (lesson echo, image-only hedging) and B (answer denial).
# v8 (2026-11-06, cycle-2 64-item review, claudedocs/analysis_single_v7_cycle2_20261106.md section 5): open answers
# graded by single_grading (V8 extraction, recorded notation normalization incl. a per-source alias table, the
# training comparison); a committed answer never gets NO_COMMITTED_ANSWER; the note input shows the grader-read
# value; notes with Cause NONE, or a non-NO_FINAL_ANSWER cause for an unaccepted/cut-off answer, are rejected;
# lesson stem leak check; answer-denial phrases block T1 reward/A1 and A2; review phrases on word boundaries.
# v9 (2026-11-07, v8 main run stopped at block 27 of 219 on one row's vLLM-vs-HF replay mismatch, and the T1
# answer-first share rose within sources as in v4): a block may drop up to two replay-failing RL rows from PG, KL and
# the anchor (block mean still covers every row; fixed denominators), and the first T1 positions carry an
# always-on raw F0 KL anchor so the reason-first/answer-first decision stays near the original model.
# v10 (2026-11-07, claudedocs/analysis_why_rlsft_not_improving_20261107.md: over v9's 55 updates the weighted loss was
# A2 9.12, anchor 1.58, B 0.62, A1 0.57, PG 0.013, KL 0.004 and 51% of rewards were 0): T1 PG uses a fixed per-source
# baseline (mean v9 T1 reward of blocks 0-54, shipped as source_baseline_v10.json and pinned by SHA256), so every
# non-excluded RL row carries A = reward - b(source); token scale 256 instead of 1024; A2 SFT weight 0.35 per row
# instead of 0.5 shared by the block's A2 rows. Rewards, gates, prompts, anchor, KL and replay are v9's.
# v11 (2026-11-08, claudedocs/plan_v11_20261108.md): no baseline; reward gate v6 adds REASONED_WRONG and
# TRUNCATED_NO_LOOP penalties and a fixed advantage table (+1 / -1 / -0.5 / 0); negative rows use the raw
# full-vocabulary log p (a one-token stored support gives log q = 0 and no gradient); per-row cap min(1, 2048/n);
# token scale 1024; the note prompt may show a dataset reference after the correct answer. LOOP rows are exempt from
# the row cap (weight 1.0, as in v9) so a long repetition loop keeps its full penalty mass.
# v12 (2026-11-11, claudedocs/plan_v12_20261111.md items 1-4, 6): reward gate v7 (REASONED_WRONG 0, new CONTAINED_OPEN
# 0 for an open answer that contains the gold's words), lesson leak rule v2 (wrong-option names allowed unless they
# eliminate), B rows from every accepted note whatever the re-solve gave, and H rows: a dataset's human-written
# solution (ScienceQA lecture + solution + one final-answer line) as LM-only SFT under P0, weight 0.02 per kind (0.10 gave an H gradient 7x the PG gradient in the local NF4 block, 2026-11-11).
# Everything else is v11b. plain_protocol_single(version=VERSION_SINGLE) still builds the byte-identical v11 protocol.
# v13 (2026-11-12, claudedocs/research_v12_drop_and_67_path_20261112.md section 7.3): the answers MMMU loses most are
# self-doubt spirals ("Wait ..." after a finished first derivation until the token cap), so v13 trains finishing only.
# Reward gate v8 = gate v7 plus two program fields: a LOOP / TRUNCATED_NO_LOOP row carries the spiral window (from the
# third marker token, or token 1024 when there are fewer than three, for at most 2048 tokens) and its PG term reads the
# raw log p of that window alone at advantage -1 with no row cap, so the first derivation is never pushed down; an
# accepted row is praised (+1, and an A1 row) only when it has at least 256 tokens, shorter correct answers carry 0.
# No H rows. Lesson leak rule v2, B rows, KL, anchor, replay, SFT weights and everything else are v12's.
VERSION_SINGLE_V8 = "error-note-single-response-v8"
VERSION_SINGLE_V9 = "error-note-single-response-v9"
VERSION_SINGLE_V10 = "error-note-single-response-v10"
VERSION_SINGLE = "error-note-single-response-v11"
VERSION_SINGLE_V12 = "error-note-single-response-v12"
VERSION_SINGLE_V13 = "error-note-single-response-v13"
SINGLE_TOKEN_SCALE_V10 = 256.0
SINGLE_TOKEN_SCALE_V11 = 1024.0
SINGLE_ROW_TOKEN_CAP_V11 = 2048
SINGLE_ROW_TOKEN_CAP_EXEMPT_V11 = ("LOOP",)
SINGLE_NEGATIVE_LOGPROB_V11 = "raw_full_vocab_log_softmax_T1_no_presence"
SINGLE_ADVANTAGE_VERSION_V11 = "penalty-table-v1"
SINGLE_ADVANTAGE_V11 = {"accepted": 1.0, "BARE_WRONG": -1.0, "NO_COMMITTED_ANSWER": -1.0, "LOOP": -1.0,
                        "REASONED_WRONG": -0.5, "TRUNCATED_NO_LOOP": -0.5, "other": 0.0}
SINGLE_PLAN_EXCEPTIONS_V11 = [
    "2026-11-05 plan: PG used only the stored sampling distribution q; v11 negative-advantage rows use the raw "
    "full-vocabulary log p (T=1, no presence penalty) so one-token supports still carry gradient",
    "2026-11-05 plan: the PG denominator was length-independent and fixed; v11 scales each row by min(1, 2048/n) "
    "so long rows do not dominate the block"]
SINGLE_SFT_WEIGHTS_V10 = {"A1": 0.05, "A2": 0.35, "B": 0.05}
SINGLE_SFT_WEIGHTING_V10 = {"A1": "per_kind_total", "A2": "per_row", "B": "per_kind_total"}
SINGLE_ADVANTAGE_VERSION_V12 = "penalty-table-v2"
SINGLE_ADVANTAGE_V12 = {"accepted": 1.0, "BARE_WRONG": -1.0, "NO_COMMITTED_ANSWER": -1.0, "LOOP": -1.0,
                        "REASONED_WRONG": 0.0, "TRUNCATED_NO_LOOP": -0.5, "CONTAINED_OPEN": 0.0, "other": 0.0}
SINGLE_CONTAINED_OPEN_RULE = ("open item (no options), T1 INCORRECT and stopped with a committed (extracted) answer whose "
                              "tokens contain the gold's answer tokens contiguously while the sequences differ; gate v7 "
                              "order: accepted, LOOP, TRUNCATED_NO_LOOP, NO_COMMITTED_ANSWER, CONTAINED_OPEN, BARE_WRONG, "
                              "REASONED_WRONG; gold = the source-bound provenance label")
SINGLE_SFT_WEIGHTS_V12 = SINGLE_SFT_WEIGHTS_V10 | {"H": 0.02}
SINGLE_SFT_WEIGHTING_V12 = SINGLE_SFT_WEIGHTING_V10 | {"H": "per_kind_total"}
SINGLE_ADVANTAGE_VERSION_V13 = "penalty-table-v3"
SINGLE_ADVANTAGE_V13 = {"accepted_praised": 1.0, "accepted": 0.0, "BARE_WRONG": -1.0, "NO_COMMITTED_ANSWER": -1.0,
                        "LOOP": -1.0, "TRUNCATED_NO_LOOP": -1.0, "REASONED_WRONG": 0.0, "CONTAINED_OPEN": 0.0, "other": 0.0}
# The pinned Qwen3-VL tokenizer's two single tokens that decode to "Wait" ("Wait", " Wait"); prepare re-checks them.
SINGLE_SPIRAL_V13 = {"version": "spiral-window-v1", "marker": "Wait", "marker_token_ids": [13824, 14190],
                     "marker_count": 3, "fallback_start": 1024, "window_tokens": 2048,
                     "penalties": ["LOOP", "TRUNCATED_NO_LOOP"], "logprob": "raw", "row_weight": 1.0}
SINGLE_PRAISE_V13 = {"version": "praise-v1", "min_tokens": 256, "applies_to": ["PG", "A1"]}


def spiral_window(token_ids: list[int]) -> list[int]:
    """v13: [start, end) of a cut-off or looping T1's penalized tokens. start = the index of the third marker token,
    or min(1024, n // 2) with fewer than three markers; end = min(n, start + 2048)."""
    if not isinstance(token_ids, list) or not token_ids or any(type(token) is not int for token in token_ids):
        raise ValueError("single protocol: spiral window needs the sampled token IDs")
    rule, count = SINGLE_SPIRAL_V13, len(token_ids)
    marks = [index for index, token in enumerate(token_ids) if token in rule["marker_token_ids"]]
    start = marks[rule["marker_count"] - 1] if len(marks) >= rule["marker_count"] else min(rule["fallback_start"], count // 2)
    return [start, min(count, start + rule["window_tokens"])]
SINGLE_BASELINE_METHOD = "fixed-source-mean-v1"
SINGLE_BASELINE_TABLE = Path(__file__).with_name("source_baseline_v10.json")
SINGLE_REWARD_GATE_V5 = "single-t1-program-gate-v5"
SINGLE_REWARD_GATE = "single-t1-program-gate-v6"
SINGLE_REWARD_GATE_V7 = "single-t1-program-gate-v7"
SINGLE_REWARD_GATE_V8 = "single-t1-program-gate-v8"
SINGLE_NOTE_MAX_TOKENS = 1024
SINGLE_ANSWER_UNION = f"{TURN1_EVAL_CONTRACT}|{TRAINING_ANSWER_CONTRACT}"  # v7 open contract, kept for old callers
SINGLE_OPEN_SCORER = SCORER
SINGLE_REWARD_VALUES = {"accepted": 1, "BARE_WRONG": -1, "NO_COMMITTED_ANSWER": -1, "LOOP": -1,
                        "REASONED_WRONG": -1, "TRUNCATED_NO_LOOP": -1, "other": 0}
SINGLE_REPLAY = "stored-generation-support-v1"
SINGLE_REPLAY_BLOCK_MEAN_ATOL = 0.05
SINGLE_REPLAY_ROW_MEAN_ATOL = 0.1
SINGLE_REPLAY_ROW_MEAN_MIN_TOKENS = 64
SINGLE_REPLAY_MAX_ATOL = 3.0
SINGLE_REPLAY_ROW_FAILURE = "exclude_row_from_rl_kl"
SINGLE_REPLAY_MAX_EXCLUDED_ROWS = 2
SINGLE_FORMAT_ANCHOR_POSITIONS = 4
SINGLE_FORMAT_ANCHOR_BETA = 0.5
SINGLE_ANSWER_CONTRACTS = {"mcq": TURN1_EVAL_CONTRACT, "open": SINGLE_OPEN_SCORER}
SINGLE_NOTE_CAUSE_RULE = {
    "reject_cause": ["NONE"], "unaccepted_statuses": ["INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"],
    "unaccepted_cause": "NO_FINAL_ANSWER", "reasons": ["NOTE_CAUSE_NONE", "NOTE_CAUSE_MISMATCH"],
    "applies_to": "B rows only; the note's lesson still drives its one re-solve"}
SINGLE_CONTRACT_HASHES = frozenset(("source", "dataset", "order", "data_admission"))
SINGLE_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


def single_prompts_sha256() -> str:
    return _digest({"note": NOTE_INSTRUCTION_SINGLE, "note_feedback": SINGLE_VERDICT_LABELS, "resolve_header": RESOLVE_HEADER,
                    "resolve_instruction": RESOLVE_INSTRUCTION_SINGLE,
                    "note_grader_read": [GRADER_READ_LINE, GRADER_READ_NONE], "note_reference": REFERENCE_HEADER})


def single_baseline_contract() -> dict:
    """The v10 RL baseline pin: method plus the SHA256 of the shipped per-source table bytes."""
    return {"method": SINGLE_BASELINE_METHOD, "table_sha256": hashlib.sha256(SINGLE_BASELINE_TABLE.read_bytes()).hexdigest(),
            "unknown_source": "global_mean"}


def single_source_baseline(problem_id: str, pin: dict) -> dict:
    """b(source) for one RL row from the shipped table; the table bytes must still match the sealed pin."""
    raw = SINGLE_BASELINE_TABLE.read_bytes()
    if pin != {"method": SINGLE_BASELINE_METHOD, "table_sha256": hashlib.sha256(raw).hexdigest(),
               "unknown_source": "global_mean"}:
        raise ValueError("single baseline: shipped source table differs from the sealed baseline pin")
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("single baseline: problem ID required")
    table, source = json.loads(raw), problem_id.split(":")[0]
    value = table["sources"][source]["mean"] if source in table["sources"] else table["global_mean"]
    return {"method": SINGLE_BASELINE_METHOD, "table_sha256": pin["table_sha256"], "source": source, "value": float(value)}


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
                          token_scale: float = SINGLE_TOKEN_SCALE_V11, beta: float = 0.02,
                          sft_weights: dict[str, float] | None = None, note_reference: dict | None = None,
                          version: str = VERSION_SINGLE) -> dict:
    """Explicit local-data/source pins and an on-policy, one-response objective; no execution permission (v11, or
    v12 / v13 with ``version=VERSION_SINGLE_V12`` / ``VERSION_SINGLE_V13``)."""
    import math

    if version not in (VERSION_SINGLE, VERSION_SINGLE_V12, VERSION_SINGLE_V13):
        raise ValueError("single protocol: version must be v11, v12 or v13")
    v13 = version == VERSION_SINGLE_V13
    v12 = version == VERSION_SINGLE_V12  # H rows are v12 only; v13 shares v12's note filters and B rule below
    weights = dict(SINGLE_SFT_WEIGHTS_V12 if v12 else SINGLE_SFT_WEIGHTS_V10) if sft_weights is None else dict(sft_weights)
    reference = {"version": "none"} if note_reference is None else note_reference
    if token_scale != SINGLE_TOKEN_SCALE_V11:
        raise ValueError(f"single protocol: v11 fixes the PG token scale at {SINGLE_TOKEN_SCALE_V11}")
    if not isinstance(reference, dict) or not isinstance(reference.get("version"), str):
        raise ValueError("single protocol: note_reference must be a dict with a string version")
    if (set(contracts) != SINGLE_CONTRACT_HASHES or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in contracts.values())):
        raise ValueError("single protocol: explicit source, dataset, order and data-admission SHA256 required")
    if (isinstance(token_scale, bool) or not isinstance(token_scale, (int, float))
            or not math.isfinite(token_scale) or token_scale <= 0
            or isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0
            or set(weights) != set(SINGLE_SFT_WEIGHTS_V12 if v12 else SINGLE_SFT_WEIGHTS_V10) or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 for value in weights.values())):
        raise ValueError("single protocol: finite fixed normalization and nonnegative weights required")
    body = {key: value for key, value in plain_protocol_v2(precision, total_slots).items() if key != "sha256"}
    body |= {"version": VERSION_SINGLE, "sampling": dict(SINGLE_SAMPLING),
             "prompts_sha256": single_prompts_sha256(),
             "training_contract": {
                 "version": VERSION_SINGLE, "hashes": dict(contracts), "master_seed": MASTER_SEED, "per_update_items": PER_UPDATE_ITEMS_V2,
                 "generation_counts": {"t1": 1, "note": 1, "resolve": 1}, "sampling": dict(SINGLE_SAMPLING),
                 "max_tokens": {"t1": MAX_NEW_TOKENS, "resolve": MAX_NEW_TOKENS}, "note_max_tokens": SINGLE_NOTE_MAX_TOKENS,
                 "rl": {"method": "reinforce", "baseline": "none",
                        "advantage": {"version": SINGLE_ADVANTAGE_VERSION_V11, "values": dict(SINGLE_ADVANTAGE_V11)},
                        "token_scale": float(token_scale), "row_token_cap": SINGLE_ROW_TOKEN_CAP_V11,
                        "row_token_cap_exempt": list(SINGLE_ROW_TOKEN_CAP_EXEMPT_V11),
                        "negative_logprob": SINGLE_NEGATIVE_LOGPROB_V11, "beta": float(beta), "reward": SINGLE_REWARD_GATE,
                        "reward_values": dict(SINGLE_REWARD_VALUES),
                        "kl": "raw_F0_to_current_on_all_completed_T1_prefixes",
                        "format_anchor": {"positions": SINGLE_FORMAT_ANCHOR_POSITIONS, "beta": SINGLE_FORMAT_ANCHOR_BETA},
                        "plan_exceptions": list(SINGLE_PLAN_EXCEPTIONS_V11)},
                 "note_reference": dict(reference),
                 "sft_weights": weights, "sft_weighting": dict(SINGLE_SFT_WEIGHTING_V10),
                 "sft_row": "mean_original_target_token_nll",
                 "length_policy": {"min_target_tokens": MIN_TARGET_TOKENS, "note_min_target_tokens": 1,
                                   "max_target_tokens": MAX_TARGET_TOKENS,
                                   "content_acceptance": "program_checks_only_content_unverified"},
                 "answer_contracts": dict(SINGLE_ANSWER_CONTRACTS),
                 "answer_scorer": {"id": SINGLE_OPEN_SCORER, "rules": list(NORMALIZATION_RULES),
                                   "suffix_stop": sorted(SUFFIX_STOP), "number_words": list(NUMBER_WORDS),
                                   "aliases": ALIASES, "aliases_sha256": ALIASES_SHA256,
                                   "committed_answer_never_no_committed_penalty": True},
                 "program_filters": {"note_cause": SINGLE_NOTE_CAUSE_RULE, "lesson_stem": LESSON_STEM_RULE,
                                     "answer_denial": {"phrases": list(DENIAL_PHRASES), "match": "word_start_after_markdown_emphasis_removed",
                                                       "blocks": ["T1_reward", "A1", "A2"]},
                                     "a2_review_references": {"phrases": list(_REVIEW_REFERENCES),
                                                              "match": "word_boundary"}},
                 "renormalize_missing_kind": False, "replay": "original_actions_same_live_policy_once",
                 "replay_support": {"method": SINGLE_REPLAY, "block_mean_atol": SINGLE_REPLAY_BLOCK_MEAN_ATOL,
                                    "row_mean_atol": SINGLE_REPLAY_ROW_MEAN_ATOL,
                                    "row_mean_min_tokens": SINGLE_REPLAY_ROW_MEAN_MIN_TOKENS,
                                    "row_max_atol": SINGLE_REPLAY_MAX_ATOL,
                                    "row_failure": SINGLE_REPLAY_ROW_FAILURE,
                                    "max_excluded_rows": SINGLE_REPLAY_MAX_EXCLUDED_ROWS},
                 "zero_signal": "consume_slot_without_optimizer_or_decay", "total_slots": total_slots,
                 "pilot_boundary": 300, "pilot_main_optimizer": "continuous",
                 "routes": {"RL": "LM", "A1": "LM", "A2_B": "accepted_note_self_report"}}}
    if v12 or v13:
        from .contract import (_ANSWER_DENIAL_V12, HUMAN_FINAL_LINE, LESSON_LEAK_RULE_V2, LESSON_LEAK_VERSION_V2,
                               ANSWER_TOKEN_RULE)
        from .references import HUMAN_SOLUTION_FIELDS, HUMAN_SOLUTION_VERSION, human_solution_sha256

        contract = body["training_contract"]
        body["version"] = contract["version"] = version
        contract["rl"] |= {"reward": SINGLE_REWARD_GATE_V7, "reward_values": dict(SINGLE_REWARD_VALUES, CONTAINED_OPEN=0),
                           "advantage": {"version": SINGLE_ADVANTAGE_VERSION_V12, "values": dict(SINGLE_ADVANTAGE_V12)},
                           "contained_open": {"rule": SINGLE_CONTAINED_OPEN_RULE, "tokens": ANSWER_TOKEN_RULE}}
        contract["program_filters"] |= {
            "lesson_stem": LESSON_LEAK_RULE_V2, "lesson_leak": LESSON_LEAK_VERSION_V2,
            "b_answer_denial": {"pattern": _ANSWER_DENIAL_V12.pattern, "match": "lowercase_whitespace_collapsed"}}
        contract["b_rows"] = "every_program_accepted_note_whatever_the_resolve_result"
    if v13:
        contract["rl"] |= {"reward": SINGLE_REWARD_GATE_V8,
                           "advantage": {"version": SINGLE_ADVANTAGE_VERSION_V13, "values": dict(SINGLE_ADVANTAGE_V13)},
                           "spiral_window": copy.deepcopy(SINGLE_SPIRAL_V13), "praise": copy.deepcopy(SINGLE_PRAISE_V13)}
    if v12:
        contract["sft_weighting"] = dict(SINGLE_SFT_WEIGHTING_V12)
        contract["human_solution"] = {
            "version": HUMAN_SOLUTION_VERSION, "fields": {k: list(v) for k, v in HUMAN_SOLUTION_FIELDS.items()},
            "fields_sha256": human_solution_sha256(), "target": "lecture?\\n\\nsolution\\n\\n" + HUMAN_FINAL_LINE,
            "mcq_answer": "{gold letter}. {gold option text}", "min_target_tokens": MIN_TARGET_TOKENS,
            "end_token": "appended_like_sampled_rows", "items": "every_block_item_of_a_mapped_source",
            "excluded_from": ["PG", "KL", "format_anchor", "replay"]}
        contract["routes"] = contract["routes"] | {"H": "LM"}
    return body | {"sha256": _digest(body)}
