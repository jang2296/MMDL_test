"""Conservative admissibility gates; validation never promotes a candidate oracle."""

from __future__ import annotations

import re
from typing import Any

from .input_builder import canonical_hash


def oracle_hash(oracle: dict[str, Any]) -> str:
    return canonical_hash({key: value for key, value in oracle.items() if key != "oracle_hash"})


def validate_oracle(oracle: dict[str, Any] | None, sample: dict[str, Any],
                    level: str | None = None,
                    oracle_review_policy: str = "HUMAN_REVIEW_REQUIRED") -> dict[str, str]:
    """VALID/BLOCKED/EXCLUDED metadata gate, not an automated semantic certification."""
    def verdict(status: str, reason: str) -> dict[str, str]:
        return {"status": status, "reason": reason}

    if oracle_review_policy not in {"HUMAN_REVIEW_REQUIRED", "ALLOW_AI_REVIEWED_DIAGNOSTIC"}:
        raise ValueError("Unknown oracle_review_policy")
    if oracle is None:
        return verdict("BLOCKED", "MISSING_ORACLE")
    if not isinstance(oracle, dict):
        return verdict("BLOCKED", "MALFORMED_ORACLE")
    leakage = oracle.get("target_leakage_status")
    if leakage in {"TARGET_REVEALED", "NOT_APPLICABLE", "N/A"}:
        return verdict("EXCLUDED", str(leakage))
    required = {
        "sample_id", "level", "annotation_version", "facts", "source_image_hashes",
        "evidence_locations", "completeness_status", "target_leakage_status",
        "annotator_exposure_status", "reviewer_status", "reviewer_id", "annotator_id",
        "approved", "oracle_hash", "provenance", "training_allowed",
    }
    if required - set(oracle):
        return verdict("BLOCKED", "MISSING_REVIEW_FIELDS")
    if oracle["sample_id"] != sample["sample_id"] or oracle["level"] not in {"O0", "O1"}:
        return verdict("BLOCKED", "ORACLE_ID_OR_LEVEL_MISMATCH")
    if level is not None and oracle["level"] != level:
        return verdict("BLOCKED", "ORACLE_LEVEL_MISMATCH")
    if oracle["approved"] is not True:
        return verdict("BLOCKED", "UNAPPROVED_ORACLE")
    if leakage != "VALID" or oracle["completeness_status"] != "COMPLETE":
        return verdict("BLOCKED", "INCOMPLETE_OR_UNVALIDATED_ORACLE")
    review = oracle["reviewer_status"]
    if review == "AI_INDEPENDENT_REVIEWED" and oracle_review_policy != "ALLOW_AI_REVIEWED_DIAGNOSTIC":
        return verdict("BLOCKED", "AI_REVIEW_REQUIRES_EXPLICIT_DIAGNOSTIC_POLICY")
    if review not in {"INDEPENDENT_REVIEWED", "AI_INDEPENDENT_REVIEWED"}:
        return verdict("BLOCKED", "INDEPENDENT_REVIEW_REQUIRED")
    for name in ("annotation_version", "annotator_exposure_status", "reviewer_id", "annotator_id"):
        if not isinstance(oracle[name], str) or not oracle[name].strip():
            return verdict("BLOCKED", "MISSING_ANNOTATOR_OR_REVIEW_IDENTITY")
    if oracle["reviewer_id"] == oracle["annotator_id"]:
        return verdict("BLOCKED", "REVIEWER_NOT_INDEPENDENT")
    if oracle["provenance"] != "BENCHMARK_DERIVED" or oracle["training_allowed"] is not False:
        return verdict("BLOCKED", "ORACLE_PROVENANCE_INVALID")
    expected = [image["sha256"] for image in sample["images"]]
    if oracle["source_image_hashes"] != expected:
        return verdict("BLOCKED", "SOURCE_IMAGE_COVERAGE_OR_ORDER_MISMATCH")
    if not isinstance(oracle["evidence_locations"], (dict, list)) or not oracle["evidence_locations"]:
        return verdict("BLOCKED", "MISSING_IMAGE_EVIDENCE")
    facts = oracle["facts"]
    if not isinstance(facts, (dict, str)) or not facts:
        return verdict("BLOCKED", "MISSING_FACTS")
    # This detects explicit answer-interface leaks; semantic leakage still requires the reviewer.
    def target_field(value: Any) -> bool:
        if isinstance(value, dict):
            return any(str(key).lower() in {"answer", "gold", "correct_answer", "final_answer", "solution"}
                       or target_field(item) for key, item in value.items())
        return isinstance(value, list) and any(target_field(item) for item in value)

    if target_field(facts):
        return verdict("EXCLUDED", "EXPLICIT_TARGET_FIELD")
    if re.search(r"(?:FINAL_ANSWER\s*:|(?:correct|final)\s+answer\s*(?:is|:))", str(facts), re.I):
        return verdict("EXCLUDED", "EXPLICIT_TARGET_DECLARATION")
    if oracle["oracle_hash"] != oracle_hash(oracle):
        return verdict("BLOCKED", "ORACLE_HASH_MISMATCH")
    return verdict("VALID", "APPROVED_INDEPENDENT_REVIEW") | {
        "verification_kind": "AI_REVIEWED_NOT_HUMAN_VERIFIED" if review == "AI_INDEPENDENT_REVIEWED"
        else "INDEPENDENT_REVIEW_METADATA_VALIDATED",
    }
