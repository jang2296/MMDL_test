"""Gold-blind, stateless diagnostic messages; all payloads remain benchmark-derived."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from mmdl.evaluation.prompt import build_prompt

MODES = ("I", "T0", "O0", "O1", "IO0", "P", "M", "MI", "LEGACY")
FINAL_INSTRUCTION = (
    "Answer the question using the provided information. You may give a brief solution.\n"
    "End with exactly one line:\nFINAL_ANSWER: <one option label or one short answer>\n"
    "Do not provide multiple alternative final answers."
)
PERCEPTION_INSTRUCTION = (
    "Describe the visible symbols, values, positions and connections.\n"
    "Do not solve the question. Do not select an answer option.\n"
    "Use the supplied JSON schema. Use null for facts you cannot determine.\n"
    "Return one JSON object only. Describe every supplied image in its supplied order."
    " The marker field is an image ID, not its content/type: copy the supplied ID exactly,"
    " including angle brackets (for example <image 1>)."
)
_SAMPLE_FIELDS = {
    "sample_id", "subject", "question_type", "question", "options", "hint", "images",
    "family_id", "skill_tags", "provenance", "training_allowed", "source_dataset",
    "source_revision", "source_split",
}
_IMAGE_FIELDS = {"field", "marker", "role", "order", "path", "sha256", "original_size"}
_MARKER = re.compile(r"<image\s+\d+>")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def stage_seed(base_seed: int, sample_id: str, mode: str) -> int:
    if base_seed not in (3407, 3408, 3409):
        raise ValueError("Diagnostic repetitions must use base seeds 3407, 3408, 3409")
    text = f"{base_seed}:{sample_id}" + (":P" if mode == "P" else "")
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2**31)


def validate_sample(sample: dict[str, Any]) -> None:
    unknown = set(sample) - _SAMPLE_FIELDS
    if unknown:
        raise ValueError(f"Inference sample contains forbidden/unknown fields: {sorted(unknown)}")
    required = _SAMPLE_FIELDS - {"hint", "source_dataset", "source_revision", "source_split"}
    if required - set(sample):
        raise ValueError(f"Missing sample fields: {sorted(required - set(sample))}")
    if sample["provenance"] != "BENCHMARK_DERIVED" or sample["training_allowed"] is not False:
        raise ValueError("Diagnostic inputs must be BENCHMARK_DERIVED, training_allowed=false")
    for name in ("sample_id", "subject", "question_type", "question", "family_id"):
        if not isinstance(sample[name], str) or not sample[name].strip():
            raise ValueError(f"Sample {name} must be non-empty text")
    if not isinstance(sample["options"], list) or not all(
        isinstance(value, str) for value in sample["options"]
    ):
        raise ValueError("options must be an ordered list of text")
    if not isinstance(sample["skill_tags"], list) or not all(
        isinstance(value, str) for value in sample["skill_tags"]
    ):
        raise ValueError("skill_tags must be a list of text")
    if sample.get("hint") is not None and not isinstance(sample["hint"], str):
        raise ValueError("hint must be text or null")
    images = sample["images"]
    if not isinstance(images, list):
        raise ValueError("images must be a list")
    seen: set[str] = set()
    previous = -1
    texts = [sample["question"], sample.get("hint") or "", *sample["options"]]
    referenced = set(_MARKER.findall("\n".join(texts)))
    for image in images:
        if (not isinstance(image, dict) or _IMAGE_FIELDS - set(image)
                or set(image) - (_IMAGE_FIELDS | {"source_bytes_sha256"})):
            raise ValueError("Image metadata must contain only the declared image fields")
        marker = image["marker"]
        if not isinstance(marker, str) or not _MARKER.fullmatch(marker) or marker in seen:
            raise ValueError("Image markers must be valid and unique")
        if image["role"] not in {"question", "option", "question_and_option"}:
            raise ValueError("Explanation-only images are forbidden")
        q = marker in sample["question"] or marker in (sample.get("hint") or "")
        o = any(marker in value for value in sample["options"])
        expected = "question_and_option" if q and o else "question" if q else "option" if o else None
        if expected != image["role"]:
            raise ValueError("Image role does not match original question/option references")
        if (not isinstance(image["order"], int) or isinstance(image["order"], bool)
                or image["order"] <= previous):
            raise ValueError("Image input order must be strictly increasing")
        if not re.fullmatch(r"[0-9a-f]{64}", str(image["sha256"])):
            raise ValueError("Image SHA256 is invalid")
        path = image["path"]
        if not isinstance(path, str) or not path or "\\" in path or path.startswith("/"):
            raise ValueError("Image path must be relative POSIX text")
        if any(part in ("", ".", "..") for part in path.split("/")):
            raise ValueError("Image path contains unsafe components")
        size = image["original_size"]
        if not isinstance(size, list) or len(size) != 2 or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in size
        ):
            raise ValueError("original_size must contain two positive integers")
        previous = image["order"]
        seen.add(marker)
    if seen != referenced:
        raise ValueError("Image coverage differs from original question/options")


def perception_schema(sample: dict[str, Any]) -> dict[str, Any]:
    """Predeclared broad schemas, selected without question/gold-dependent fields."""
    domains = {
        "Music": ["clefs", "staff_positions", "accidentals", "durations", "bar_order"],
        "Computer_Science": ["nodes", "edges", "directions", "weights", "initial_state"],
        "Electronics": ["components", "values", "nodes", "connections", "polarities"],
        "Energy_and_Power": ["components", "values", "nodes", "connections", "polarities"],
        "Accounting": ["row_labels", "column_labels", "cells", "units", "time_basis"],
        "Finance": ["row_labels", "column_labels", "cells", "units", "time_basis"],
    }
    fields = domains.get(sample["subject"], ["visible_text", "symbols", "values", "positions", "connections"])
    properties: dict[str, Any] = {"marker": {"type": "string", "enum": [im["marker"] for im in sample["images"]]}}
    properties.update({field: {"type": ["array", "null"]} for field in fields})
    properties["uncertainties"] = {"type": ["array", "null"]}
    return {
        "type": "object", "additionalProperties": False, "required": ["images"],
        "properties": {"images": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties,
        }}},
    }


def build_request(sample: dict[str, Any], mode: str, rep: int,
                  oracle: dict[str, Any] | None = None,
                  perception: dict[str, Any] | str | None = None,
                  oracle_review_policy: str = "HUMAN_REVIEW_REQUIRED") -> dict[str, Any]:
    """Construct one independent user message, with no previous conversation state."""
    from .oracle_validation import validate_oracle

    validate_sample(sample)
    if mode not in MODES:
        raise ValueError(f"Unknown diagnostic mode: {mode}")
    image_mode = mode in {"I", "IO0", "P", "MI", "LEGACY"}
    images = sample["images"] if image_mode else []
    if mode == "P" and not images:
        raise ValueError("P requires actual images")
    oracle_id = None
    perception_id = None
    raw: Any = ""
    if mode in {"O0", "O1", "IO0"}:
        level = "O1" if mode == "O1" else "O0"
        verdict = validate_oracle(oracle, sample, level, oracle_review_policy)
        if verdict["status"] != "VALID":
            raise ValueError(f"Oracle condition {verdict['status']}: {verdict['reason']}")
        assert oracle is not None
        oracle_id = oracle["oracle_hash"]
    elif oracle is not None:
        raise ValueError("Oracle supplied to a non-oracle condition")
    if mode in {"M", "MI"}:
        if isinstance(perception, dict):
            if (perception.get("sample_id") != sample["sample_id"] or perception.get("mode") != "P"
                    or perception.get("repetition") != rep or perception.get("runtime_status") != "COMPLETED"):
                raise ValueError("M/MI require the completed P response for this sample/repetition")
            raw = perception.get("raw_response")
        else:
            raw = perception
        if not isinstance(raw, str):
            raise ValueError("M/MI require the actual unedited P raw text")
        perception_id = canonical_hash(raw)
    elif perception is not None:
        raise ValueError("P output supplied to an unrelated condition")
    if mode == "P":
        schema = perception_schema(sample)
        text = PERCEPTION_INSTRUCTION + "\nSchema:\n" + json.dumps(schema, sort_keys=True)
        text += "\nImage identities in supplied order:\n" + json.dumps(
            [image["marker"] for image in images], ensure_ascii=False,
        )
    else:
        text = build_prompt(sample, Path(__file__).resolve().parents[2])
        if mode in {"O0", "O1", "IO0"}:
            assert oracle is not None
            facts = oracle["facts"] if isinstance(oracle["facts"], str) else json.dumps(
                oracle["facts"], ensure_ascii=False, sort_keys=True,
            )
            text = f"Provided visual information ({oracle['level']}):\n{facts}\n\n{text}"
        elif mode in {"M", "MI"}:
            text = "Unedited model structure extraction:\n" + raw + "\n\n" + text
        if mode != "LEGACY":
            text += "\n\n" + FINAL_INSTRUCTION
    messages = [{"role": "user", "content": [
        *[{"type": "image", "image": image["path"]} for image in images],
        {"type": "text", "text": text},
    ]}]
    result = {
        "sample_id": sample["sample_id"], "mode": mode, "repetition": rep,
        "seed": stage_seed(rep, sample["sample_id"], mode), "messages": messages,
        "images": [dict(image) for image in images],
        "max_new_tokens": 1536 if mode == "P" else 32768 if mode == "LEGACY" else 4096,
        "oracle_hash": oracle_id, "perception_response_hash": perception_id,
        "oracle_review_policy": oracle_review_policy,
        "verification_kind": verdict.get("verification_kind") if oracle_id else None,
        "prompt_hash": canonical_hash(messages),
        "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
    }
    result["cache_key"] = canonical_hash(result)
    return result
