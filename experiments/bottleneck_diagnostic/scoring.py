"""Post-inference scoring: gold-blind extraction, immutable V8 comparison, structure audit."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from mmdl.evaluation.final_answer_parser_v8 import (
    _open_equal,
    extract_open,
    score_response,
)
from mmdl.evaluation.parsers import _question_type

from .input_builder import canonical_hash, perception_schema
from .oracle_validation import validate_oracle

SCORING_PROTOCOL = "diagnostic-final-interface-v2-exact-option"


def extract_diagnostic(raw: Any, question_type: str, options: list[str],
                       question: str = "") -> dict[str, Any]:
    """The only answer candidate is one terminal FINAL_ANSWER line, never a gold search."""
    if not isinstance(raw, str) or not raw.strip():
        return {"answer": None, "status": "EMPTY", "rule": "empty", "audit_flags": []}
    markers = re.findall(r"(?im)^\s*FINAL_ANSWER\s*:", raw)
    terminal = re.fullmatch(r"FINAL_ANSWER: ([^\r\n]+)", raw.rstrip().splitlines()[-1])
    if len(markers) != 1 or terminal is None:
        return {"answer": None, "status": "NO_PARSE", "rule": "missing_or_multiple_terminal_final",
                "audit_flags": ["FINAL_INTERFACE_INVALID"]}
    candidate = terminal.group(1).strip()
    if _question_type(question_type) == "mcq":
        matches = [chr(65 + i) for i, option in enumerate(options) if candidate == option.strip()]
        if re.fullmatch(r"[A-Z]", candidate) and ord(candidate) - 65 < len(options):
            if any(label != candidate for label in matches):
                return {"answer": None, "status": "NO_PARSE", "rule": "label_content_collision",
                        "audit_flags": ["FINAL_INTERFACE_AMBIGUOUS"]}
            return {"answer": candidate, "status": "PARSED", "rule": "diagnostic_single_label", "audit_flags": []}
        labeled = re.fullmatch(r"([A-Z])\.\s+(.+)", candidate)
        if labeled:
            label, content = labeled.groups()
            index = ord(label) - 65
            if index < len(options) and content == options[index].strip():
                return {"answer": label, "status": "PARSED", "rule": "diagnostic_label_option_exact", "audit_flags": []}
            return {"answer": None, "status": "NO_PARSE", "rule": "label_option_mismatch",
                    "audit_flags": ["FINAL_INTERFACE_AMBIGUOUS"]}
        if len(matches) == 1:
            return {"answer": matches[0], "status": "PARSED", "rule": "diagnostic_option_text_exact", "audit_flags": []}
        return {"answer": None, "status": "NO_PARSE", "rule": "invalid_single_label",
                "audit_flags": ["FINAL_INTERFACE_INVALID"]}
    extracted = extract_open(candidate, "stop", question=question)
    return {"answer": extracted["answer"], "status": extracted["status"],
            "rule": extracted["rule"], "audit_flags": [] if extracted["status"] == "PARSED" else ["OPEN_FINAL_AMBIGUOUS"]}


def _compare(extracted: dict[str, Any], sample: dict[str, Any], gold: Any) -> bool:
    if extracted["status"] != "PARSED":
        return False
    aliases = gold if isinstance(gold, (list, tuple, set)) else [gold]
    if _question_type(sample["question_type"]) == "mcq":
        return any(str(answer).strip().upper() == extracted["answer"] for answer in aliases)
    return any(_open_equal(extracted["answer"], answer, sample["question"], extracted["rule"]) for answer in aliases)


def _schema_error(value: Any, schema: dict[str, Any]) -> bool:
    types = schema.get("type")
    types = types if isinstance(types, list) else [types]
    actual = "null" if value is None else "object" if isinstance(value, dict) else "array" if isinstance(value, list) else "string" if isinstance(value, str) else "other"
    if actual not in types:
        return True
    if actual == "object":
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) - set(value):
            return True
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return True
        return any(_schema_error(item, properties[key]) for key, item in value.items() if key in properties)
    if actual == "array" and "items" in schema:
        return any(_schema_error(item, schema["items"]) for item in value)
    return False


def score_perception(record: dict[str, Any], sample: dict[str, Any],
                     oracle: dict[str, Any] | None) -> dict[str, Any]:
    """Exact JSON structure checks; textual/semantic oracle comparison awaits human review."""
    result: dict[str, Any] = {
        "structure_status": "PENDING_REVIEW", "structure_exact": None,
        "v8_score": None, "diagnostic_score": None, "audit_flags": [],
    }
    if record.get("runtime_status") != "COMPLETED":
        return result | {"structure_status": "NOT_EXECUTED"}
    text = record["raw_response"].strip()
    match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1)
    try:
        extracted = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return result | {"schema_parse_status": "INVALID", "audit_flags": ["P_SCHEMA_PARSE_FAILURE"]}
    schema = perception_schema(sample)
    if _schema_error(extracted, schema):
        return result | {"schema_parse_status": "INVALID", "audit_flags": ["P_SCHEMA_INVALID"]}
    markers = [item["marker"] for item in extracted["images"]]
    if markers != [image["marker"] for image in sample["images"]]:
        return result | {"schema_parse_status": "INVALID", "audit_flags": ["P_IMAGE_COVERAGE_OR_ORDER_MISMATCH"]}
    result["schema_parse_status"] = "VALID"
    result["structure_response_hash"] = canonical_hash(extracted)
    verdict = validate_oracle(oracle, sample, "O0", record.get("oracle_review_policy", "HUMAN_REVIEW_REQUIRED"))
    if verdict["status"] != "VALID":
        return result | {"oracle_review_reason": verdict["reason"]}
    assert oracle is not None
    expected = oracle["facts"]
    if not isinstance(expected, dict) or _schema_error(expected, schema):
        return result | {"oracle_review_reason": "APPROVED_O0_NOT_IN_PREDECLARED_JSON_SCHEMA"}
    if [item["marker"] for item in expected["images"]] != markers:
        return result | {"oracle_review_reason": "APPROVED_O0_IMAGE_ALIGNMENT_REQUIRES_REVIEW"}
    correct = missing = extra = total = 0
    relation_expected: set[str] = set()
    relation_actual: set[str] = set()
    fields = []
    for reference, actual in zip(expected["images"], extracted["images"], strict=True):
        for field in reference:
            if field in {"marker", "uncertainties"}:
                continue
            total += 1
            exact = actual[field] == reference[field]
            correct += exact
            missing += actual[field] is None and reference[field] is not None
            if isinstance(reference[field], list) and isinstance(actual[field], list):
                reference_set = {canonical_hash(value) for value in reference[field]}
                actual_set = {canonical_hash(value) for value in actual[field]}
                extra += len(actual_set - reference_set)
                if field in {"edges", "connections", "relations"}:
                    relation_expected.update(f"{reference['marker']}:{field}:{value}" for value in reference_set)
                    relation_actual.update(f"{reference['marker']}:{field}:{value}" for value in actual_set)
            fields.append({"marker": reference["marker"], "field": field, "exact": exact,
                           "missing": actual[field] is None and reference[field] is not None})
    overlap = len(relation_actual & relation_expected)
    return result | {
        "structure_status": "SCORED_EXACT_JSON", "comparison_scope": "exact_json_not_semantic_adjudication",
        "structure_exact": all(field["exact"] for field in fields), "fields": fields,
        "field_accuracy": correct / total if total else None,
        "missing_fields": missing, "total_fields": total,
        "unmatched_list_items": extra,
        "relation_precision": overlap / len(relation_actual) if relation_actual else None,
        "relation_recall": overlap / len(relation_expected) if relation_expected else None,
        "oracle_hash": oracle["oracle_hash"],
    }


def score_record(record: dict[str, Any], sample: dict[str, Any], gold: Any,
                 oracle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Keep runtime, V8, diagnostic and audit judgments separate."""
    if record["sample_id"] != sample["sample_id"]:
        raise ValueError("Score sample ID mismatch")
    base = {key: record[key] for key in ("sample_id", "mode", "repetition", "runtime_status")}
    base.update({"provenance": "BENCHMARK_DERIVED", "training_allowed": False,
                 "diagnostic_scoring_protocol": SCORING_PROTOCOL,
                 "v8_score_scope": "Unchanged V8 applied to NEW diagnostic raw format, not the frozen 558/900 baseline",
                 "audit_status": "NOT_REVIEWED", "audit_flags": []})
    if record["mode"] == "P":
        return base | score_perception(record, sample, oracle)
    if record["runtime_status"] != "COMPLETED":
        return base | {"v8_parsed_answer": None, "v8_score": None,
                       "diagnostic_parsed_answer": None, "diagnostic_score": None,
                       "scoring_status": "NOT_EXECUTED"}
    extracted = extract_diagnostic(record["raw_response"], sample["question_type"], sample["options"], sample["question"])
    v8 = score_response(record["raw_response"], sample["question_type"], sample["options"], gold,
                        record.get("finish_reason"), question=sample["question"])
    diagnostic = _compare(extracted, sample, gold)
    flags = list(extracted["audit_flags"])
    if record.get("finish_reason") == "length":
        flags.append("OUTPUT_LIMIT_REACHED")
    if v8["correct"] != diagnostic:
        flags.append("PARSER_JUDGMENT_DIFFERENCE")
    if record["mode"] == "LEGACY":
        flags.append("LEGACY_HAS_NO_DIAGNOSTIC_FINAL_CONTRACT")
    return base | {
        "scoring_status": "SCORED", "response_hash": canonical_hash(record["raw_response"]),
        "v8_parsed_answer": v8["parsed_answer"], "v8_score": v8["correct"],
        "v8_parse_status": v8["parse_status"], "v8_extraction_rule": v8["extraction_rule"],
        "diagnostic_parsed_answer": extracted["answer"], "diagnostic_score": diagnostic,
        "diagnostic_parse_status": extracted["status"], "diagnostic_extraction_rule": extracted["rule"],
        "audit_flags": flags,
    }


def score_responses(records: Iterable[dict[str, Any]], samples: Iterable[dict[str, Any]],
                    gold_by_id: Mapping[str, Any],
                    oracles: Mapping[tuple[str, str], dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Explicit post-run gold boundary; the runtime never imports or calls this function."""
    sample_map = {sample["sample_id"]: sample for sample in samples}
    scores = []
    for record in records:
        sample_id = record["sample_id"]
        if sample_id not in sample_map or sample_id not in gold_by_id:
            raise ValueError(f"Missing sample/gold for score record {sample_id}")
        gold = gold_by_id[sample_id]
        if isinstance(gold, dict):
            gold = gold["answer"]
        scores.append(score_record(record, sample_map[sample_id], gold,
                                   (oracles or {}).get((sample_id, "O0"))))
    return scores
