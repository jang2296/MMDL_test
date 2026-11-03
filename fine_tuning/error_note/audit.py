"""Private, source-bound note review receipts; no model, network or universal prose verifier.

Only an explicit full source review can admit a whole note or clear a lesson. Program/format checks
remain bounded evidence. Review evidence is metadata, never a replacement training target.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contract import _has_words, hint_leak, parse_note_v2

VERSION = "note-audit-v1"
FILE_VERSION = "note-audit-file-v1"
REVIEW_VERSION = "note-review-v1"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def review_bindings(record: dict, provenance: dict) -> dict:
    """Hash exact generated text plus all training-input metadata, not merely file sizes."""
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("audit: nonempty source provenance required")
    note = record.get("note")
    text = note["text"] if note else ""
    parsed = parse_note_v2(text)
    return {"note_sha256": text_sha256(text), "note_record_sha256": digest(note),
            "t1_sha256": digest(record["t1"]), "source_sha256": digest(provenance),
            "input_sha256": digest({key: record[key] for key in ("p0_text", "image_paths", "subject", "source")}),
            "p0_sha256": text_sha256(record["p0_text"]), "t1_text_sha256": text_sha256(record["t1"]["text"]),
            "note_target_sha256": text_sha256(parsed.note_text) if parsed else None,
            "note_prompt_sha256": text_sha256(note["prompt"]) if note else None,
            "lesson_sha256": text_sha256(parsed.lesson) if parsed else None}


def _review(review: dict | None, record: dict, bindings: dict) -> None:
    if review is None:
        return
    if (not isinstance(review, dict) or review.get("version") != REVIEW_VERSION or
            review.get("problem_id") != record["problem_id"] or review.get("bindings") != bindings):
        raise ValueError("audit: review identity/bindings mismatch")
    kind = review.get("reviewer_kind")
    if (kind not in ("human", "ai_review", "program") or type(review.get("human_verified")) is not bool or
            (review["human_verified"] and kind != "human")):
        raise ValueError("audit: reviewer kind/human_verified mismatch")
    if review.get("method") not in ("source_review", "bounded_check", "format_only", "no_contradiction", "regex"):
        raise ValueError("audit: unsupported review method")
    for key, results, scopes in (("content", ("PASS", "FAIL", "UNKNOWN"), ("entire_note", "partial")),
                                 ("leak", ("CLEAR", "LEAK", "UNCERTAIN"), ("entire_lesson", "partial"))):
        check = review.get(key)
        if not isinstance(check, dict) or check.get("result") not in results or check.get("scope") not in scopes:
            raise ValueError(f"audit: invalid {key} review result/scope")
        evidence = check.get("evidence")
        if (not isinstance(evidence, list) or any(not isinstance(ref, str) or not ref.strip() for ref in evidence) or
                (check["result"] in ("PASS", "FAIL", "CLEAR", "LEAK") and not evidence)):
            raise ValueError(f"audit: {key} review requires explicit evidence references")


def _strings(values: tuple[str, ...] | list[str]) -> list[str]:
    if not isinstance(values, (tuple, list)) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("audit: aliases/acronym candidates must be nonempty strings")
    return list(values)


def audit_note(record: dict, *, gold: str, choices: tuple[str, ...] | list[str], provenance: dict,
               review: dict | None = None, aliases: tuple[str, ...] | list[str] = (),
               acronym_candidates: tuple[str, ...] | list[str] = ()) -> dict:
    """Return a deterministic private audit. Missing/partial reviews never certify whole free prose.

    aliases are source-confirmed answer/option synonyms. acronym_candidates are unconfirmed potential
    synonyms: a matching candidate blocks CLEAR, including when a reviewer overlooked it.
    """
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError("audit: nonblank gold required for private leak checks")
    choices, aliases, acronym_candidates = _strings(choices), _strings(aliases), _strings(acronym_candidates)
    bindings = review_bindings(record, provenance)
    _review(review, record, bindings)
    note = record.get("note")
    parsed = parse_note_v2(note["text"]) if note else None
    format_ok = note is not None and parsed is not None and note["finish_reason"] == "stop"
    content = {"result": "UNKNOWN", "scope": "partial", "reason": "MISSING_REVIEW"}
    leak = {"result": "UNCERTAIN", "scope": "partial", "reason": "MISSING_REVIEW"}
    if review is not None:
        full_review = review["method"] == "source_review" and review["reviewer_kind"] in ("human", "ai_review")
        for key, decision, positive in (("content", content, "PASS"), ("leak", leak, "CLEAR")):
            check = review[key]
            required_scope = "entire_note" if key == "content" else "entire_lesson"
            if check["result"] == positive and (not full_review or check["scope"] != required_scope):
                decision.update(scope=check["scope"], reason="BOUNDED_REVIEW_ONLY")
            else:
                decision.update(result=check["result"], scope=check["scope"], reason="EXPLICIT_SOURCE_REVIEW")
    checks: dict[str, str | None] = {
        "hint_leak": None, "alias_leak": None, "acronym_candidate": None, "explicit_contradiction": None}
    if parsed is not None:
        checks["hint_leak"] = hint_leak(parsed.lesson, gold, choices)
        checks["alias_leak"] = next((v for v in aliases if _has_words(parsed.lesson, v)), None)
        checks["acronym_candidate"] = next((v for v in acronym_candidates if _has_words(parsed.lesson, v)), None)
        if parsed.image_reading == "MISREAD" and parsed.cause == "NONE":
            checks["explicit_contradiction"] = "MISREAD_WITH_CAUSE_NONE"
            content.update(result="FAIL", scope="partial", reason="MISREAD_WITH_CAUSE_NONE")
        if checks["hint_leak"] or checks["alias_leak"]:
            leak.update(result="LEAK", scope="partial", reason=checks["hint_leak"] or "SOURCE_ALIAS")
        elif checks["acronym_candidate"]:
            leak.update(result="UNCERTAIN", scope="partial", reason="ACRONYM_CANDIDATE")
    if not format_ok:
        content.update(result="UNKNOWN", scope="partial", reason="FORMAT_INVALID")
        leak.update(result="UNCERTAIN", scope="partial", reason="FORMAT_INVALID")
    body = {"version": VERSION, "problem_id": record["problem_id"], "bindings": bindings,
            "format": {"result": "PASS" if format_ok else "FAIL"}, "content": content, "leak": leak,
            "checks": checks, "review": review, "review_sha256": digest(review) if review is not None else None,
            "leak_inputs": {"gold": gold, "choices": choices, "aliases": aliases,
                            "acronym_candidates": acronym_candidates}}
    return body | {"sha256": digest(body)}


def validate_audit(audit: dict, record: dict, provenance: dict) -> dict:
    """Recompute decisions and all bindings; reject stale text, source metadata or self-digest."""
    if not isinstance(audit, dict) or audit.get("version") != VERSION:
        raise ValueError("audit: unsupported audit version")
    if audit.get("sha256") != digest({k: v for k, v in audit.items() if k != "sha256"}):
        raise ValueError("audit: sha256 mismatch")
    try:
        expected = audit_note(record, provenance=provenance, review=audit["review"], **audit["leak_inputs"])
    except (KeyError, TypeError) as error:
        raise ValueError("audit: malformed audit") from error
    if audit != expected:
        raise ValueError("audit: record/source/decision binding mismatch")
    return audit


def _read_items(path: Path, expected_sha256: str | None = None) -> dict[str, dict]:
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("audit: file sha256 mismatch")
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"version", "items"} or data["version"] != FILE_VERSION:
        raise ValueError("audit: invalid file wrapper")
    if not isinstance(data["items"], dict):
        raise ValueError("audit: file items must be an object")
    for pid, audit in data["items"].items():
        if (not isinstance(audit, dict) or audit.get("version") != VERSION or audit.get("problem_id") != pid or
                audit.get("sha256") != digest({k: v for k, v in audit.items() if k != "sha256"})):
            raise ValueError("audit: file item identity/sha256 mismatch")
        if audit.get("review_sha256") != (digest(audit["review"]) if audit.get("review") is not None else None):
            raise ValueError("audit: review sha256 mismatch")
        bindings = audit.get("bindings")
        if not isinstance(bindings, dict):
            raise ValueError("audit: file item bindings must be an object")
        _review(audit.get("review"), {"problem_id": pid}, bindings)
    return data["items"]


def read_audits(path: Path, records: list[dict] | dict[str, dict], provenances: dict[str, dict], *,
                expected_sha256: str | None = None) -> dict[str, dict]:
    """Read a private note-audit-file-v1 wrapper and validate every item against current inputs."""
    items = _read_items(path, expected_sha256)
    if isinstance(records, list):
        by_id = {r["problem_id"]: r for r in records}
        if len(by_id) != len(records):
            raise ValueError("audit: duplicate record problem_id")
    else:
        by_id = records
    if set(items) != set(by_id) or not set(items) <= set(provenances):
        raise ValueError("audit: file/record/source ids mismatch")
    for pid, audit in items.items():
        if not isinstance(audit, dict) or audit.get("problem_id") != pid or by_id[pid]["problem_id"] != pid:
            raise ValueError("audit: file item problem_id mismatch")
        validate_audit(audit, by_id[pid], provenances[pid])
    return items


def validate_row_audits(rows: list[dict], audit_path: Path) -> None:
    """Verify immutable row links before admission/loading; the caller also pins the whole audit file.

    NOTE/T1 targets are matched to pre-generation audit bindings. RESOLVE's output is collected later:
    its self-hash and audited lesson are checked here; correctness remains the collector's grading evidence.
    """
    audits = _read_items(audit_path)
    for row in rows:
        try:
            audit = audits[row["problem_id"]]
            bindings = audit["bindings"]
            if (row["audit_sha256"] != audit["sha256"] or row["source_sha256"] != bindings["source_sha256"] or
                    row["prompt_sha256"] != text_sha256(row["prompt"]) or
                    row["target_sha256"] != text_sha256(row["target"])):
                raise ValueError("audit: row hash/crosslink mismatch")
            review = audit["review"]
            content_ok = (audit["format"]["result"] == "PASS" and audit["content"]["result"] == "PASS" and
                          audit["content"]["scope"] == "entire_note" and review is not None and
                          review["method"] == "source_review" and review["reviewer_kind"] in ("human", "ai_review") and
                          review["content"]["result"] == "PASS" and review["content"]["scope"] == "entire_note" and
                          audit["checks"]["explicit_contradiction"] is None)
            if row["kind"] == "B" and row["target_source"] == "NOTE":
                if (not content_ok or row["target_sha256"] != bindings["note_target_sha256"] or
                        row["prompt_sha256"] != bindings["note_prompt_sha256"]):
                    raise ValueError("audit: note row target/prompt/content mismatch")
            elif row["kind"] == "A" and row["target_source"] == "T1":
                if (row["target_sha256"] != bindings["t1_text_sha256"] or
                        row["prompt_sha256"] != bindings["p0_sha256"]):
                    raise ValueError("audit: T1 row target/prompt mismatch")
            elif row["kind"] == "A" and row["target_source"] == "RESOLVE":
                clear = (content_ok and audit["leak"]["result"] == "CLEAR" and
                         audit["leak"]["scope"] == "entire_lesson" and review["leak"]["result"] == "CLEAR" and
                         review["leak"]["scope"] == "entire_lesson" and
                         all(audit["checks"][key] is None for key in ("hint_leak", "alias_leak", "acronym_candidate")))
                if (not clear or row["lesson_sha256"] != bindings["lesson_sha256"] or
                        row["prompt_sha256"] != bindings["p0_sha256"]):
                    raise ValueError("audit: resolve row lesson/content/prompt mismatch")
            else:
                raise ValueError("audit: invalid row kind/target_source")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("audit: missing or malformed row/audit linkage") from error
