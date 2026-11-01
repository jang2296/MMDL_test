"""The plain training protocol dict for fine_tuning.notebook_rl.policy and checkpoint.

Learning rates are the recipe G6 actually ran; only the training target changed (plan section 3.4).
"""
from __future__ import annotations

import hashlib
import json

from fine_tuning.modules import BASE_ID, BASE_REVISION

from .contract import CONTEXT_LIMIT, IMAGE, MAX_NEW_TOKENS, NOTE_INSTRUCTION

ARM = "MOD-RV"  # installs Vision, LLM and Merger trainables; the per-item route decides who gets gradients
MERGERS = ["model.visual.merger", "model.visual.deepstack_merger_list.0",
           "model.visual.deepstack_merger_list.1", "model.visual.deepstack_merger_list.2"]
FORBIDDEN = {"additional_sft": False, "teacher_or_upper_model_labels": True, "benchmark_training": True}
GRAD_CLIP = 1.0


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
