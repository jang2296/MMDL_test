"""Gold-blind, private P/O0 review units; no automatic semantic matching.

Queue: --data PREPARED --run pilot=RUN --run coverage=RUN --output FRESH_DIR
Review: --queue QUEUE --reviews EDITED_TEMPLATE --output FRESH_DIR
One top-level list item is one unit, including an entire object or tuple. Review
each candidate and reference once. A supported grouped candidate may explicitly
cover several references; every matched reference has exactly one owner. The
reference facts were frozen before inference; this atomization is post-run and
does not establish a preregistered list of relations needed to solve the question.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .cli import selected, validated_run
from .ingest import read_rows, sha, write_json, write_rows
from .input_builder import canonical_hash, perception_schema
from .oracle_validation import validate_oracle
from .scoring import _schema_error

Row = dict[str, Any]
PROVENANCE = {"provenance": "BENCHMARK_DERIVED", "training_allowed": False}
PROTOCOL = "p-top-level-item-explicit-coverage-v1"
CANDIDATE = {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", "NOT_APPLICABLE", "UNREVIEWED"}
REFERENCE = {"MATCHED", "OMITTED", "UNVERIFIABLE", "NOT_APPLICABLE", "UNREVIEWED"}
REVIEW_KINDS = {"AI_REVIEWED_NOT_HUMAN_VERIFIED", "HUMAN_REVIEW_DECLARED_NOT_CERTIFIED"}
RELATION_FIELDS = {"edges", "directions", "connections", "polarities", "positions",
                   "staff_positions", "bar_order", "cells"}


def units(facts: Row, side: str, source_hash: str) -> tuple[list[Row], list[Row]]:
    """Keep tuple/object contents together; null/empty fields do not invent facts."""
    atoms, fields = [], []
    for image in facts["images"]:
        for field, values in image.items():
            if field in {"marker", "uncertainties"}:
                continue
            fields.append({"marker": image["marker"], "field": field,
                           "state": "NULL" if values is None else "EMPTY" if not values else "ITEMS",
                           "items": len(values) if values is not None else 0})
            for index, value in enumerate(values or []):
                atom = {"side": side, "source_hash": source_hash, "marker": image["marker"],
                        "field": field, "index": index, "value": value}
                atoms.append(atom | {"unit_id": canonical_hash(atom)})
    return atoms, fields


def packet(record: Row, sample: Row, oracle: Row | None) -> Row:
    """No question, answer, option, score, or previous review enters the packet."""
    schema = perception_schema(sample)
    markers = [image["marker"] for image in sample["images"]]
    result: Row = {"protocol": PROTOCOL, "sample_id": record["sample_id"],
        "subject": sample["subject"], "repetition": record["repetition"],
        "runtime_status": record["runtime_status"], "schema": schema,
        "images": [{k: image[k] for k in ("marker", "path", "sha256", "original_size")}
                   for image in sample["images"]],
        "p_record_hash": record.get("record_hash"), "p_response_hash": record.get("response_hash"),
        "oracle_hash": oracle.get("oracle_hash") if oracle else None,
        "candidate_units": [], "reference_units": [], "candidate_fields": [],
        "reference_fields": [], "uncertainties": {}, **PROVENANCE}
    verdict = validate_oracle(oracle, sample, "O0", record.get(
        "oracle_review_policy", "HUMAN_REVIEW_REQUIRED"))
    result["oracle_gate"] = verdict
    result["schema_parse_status"] = "NOT_EXECUTED"
    if record["runtime_status"] == "COMPLETED":
        text = record["raw_response"].strip()
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        try:
            facts = json.loads(match.group(1) if match else text)
            valid = not _schema_error(facts, schema) and [
                image["marker"] for image in facts["images"]] == markers
        except (json.JSONDecodeError, TypeError, KeyError):
            valid = False
        result["schema_parse_status"] = "VALID" if valid else "INVALID"
        if valid:
            result["candidate_units"], result["candidate_fields"] = units(
                facts, "candidate", record["response_hash"])
            result["uncertainties"]["candidate"] = [
                {"marker": im["marker"], "value": im["uncertainties"]} for im in facts["images"]]
    if verdict["status"] == "VALID":
        assert oracle is not None
        reference = oracle["facts"]
        if (_schema_error(reference, schema) or [im["marker"] for im in reference["images"]]
                != markers):
            result["oracle_gate"] = {"status": "BLOCKED", "reason": "REFERENCE_SCHEMA_INVALID"}
        else:
            result["reference_units"], result["reference_fields"] = units(
                reference, "reference", oracle["oracle_hash"])
            result["uncertainties"]["reference"] = [
                {"marker": im["marker"], "value": im["uncertainties"]}
                for im in reference["images"]]
    result["comparison_status"] = ("ELIGIBLE" if result["schema_parse_status"] == "VALID"
        and result["oracle_gate"]["status"] == "VALID" else "NOT_COMPARABLE")
    result["packet_hash"] = canonical_hash(result)
    return result


def template(case: Row) -> Row:
    return {"packet_hash": case["packet_hash"], "reviewer_id": None, "verification_kind": None,
        "coverage_status": "UNREVIEWED", "coverage_evidence": None,
        "candidate_decisions": [{"unit_id": u["unit_id"], "status": "UNREVIEWED",
                                 "reference_ids": [], "evidence": None}
                                for u in case["candidate_units"]],
        "reference_decisions": [{"unit_id": u["unit_id"], "status": "UNREVIEWED",
                                 "candidate_id": None, "evidence": None}
                                for u in case["reference_units"]], **PROVENANCE}


def validated_decisions(case: Row, review: Row) -> tuple[list[Row], list[Row]]:
    if (case["packet_hash"] != canonical_hash({k: v for k, v in case.items() if k != "packet_hash"})
            or review.get("packet_hash") != case["packet_hash"]):
        raise ValueError("Review packet hash mismatch")
    if review.get("provenance") != "BENCHMARK_DERIVED" or review.get("training_allowed") is not False:
        raise ValueError("Review provenance mismatch")
    if case.get("provenance", "BENCHMARK_DERIVED") != "BENCHMARK_DERIVED" or case.get("training_allowed", False) is not False:
        raise ValueError("Packet provenance mismatch")
    indexed = []
    for side, statuses in (("candidate", CANDIDATE), ("reference", REFERENCE)):
        atoms = case[side + "_units"]
        for atom in atoms:
            if atom["unit_id"] != canonical_hash({k: v for k, v in atom.items() if k != "unit_id"}):
                raise ValueError("Unit hash mismatch")
        decisions = review[side + "_decisions"]
        mapping = {d["unit_id"]: d for d in decisions}
        if len(mapping) != len(decisions) or set(mapping) != {u["unit_id"] for u in atoms}:
            raise ValueError("Every unit requires exactly one explicit decision")
        for decision in decisions:
            if decision["status"] not in statuses:
                raise ValueError("Unknown review status")
            if decision["status"] != "UNREVIEWED":
                if (case["comparison_status"] != "ELIGIBLE"
                        or review.get("verification_kind") not in REVIEW_KINDS
                        or not isinstance(review.get("reviewer_id"), str)
                        or not review["reviewer_id"].strip()
                        or not isinstance(decision.get("evidence"), str)
                        or not decision["evidence"].strip()):
                    raise ValueError("Reviewed units need eligible sources, reviewer and evidence")
        indexed.append(mapping)
    candidates, references = indexed
    coverage = review.get("coverage_status")
    if coverage not in {"UNREVIEWED", "PARTIAL", "COMPLETE"}:
        raise ValueError("Unknown packet coverage review status")
    reviewed = any(d["status"] != "UNREVIEWED" for d in [*candidates.values(), *references.values()])
    if coverage == "UNREVIEWED" and reviewed:
        raise ValueError("Reviewed decisions require explicit packet coverage review")
    if coverage != "UNREVIEWED":
        if (case["comparison_status"] != "ELIGIBLE"
                or review.get("verification_kind") not in REVIEW_KINDS
                or not isinstance(review.get("reviewer_id"), str)
                or not review["reviewer_id"].strip()
                or not isinstance(review.get("coverage_evidence"), str)
                or not review["coverage_evidence"].strip()):
            raise ValueError("Packet coverage needs eligible sources, reviewer and evidence")
    if coverage == "COMPLETE" and any(d["status"] == "UNREVIEWED"
                                      for d in [*candidates.values(), *references.values()]):
        raise ValueError("Complete packet coverage cannot contain unreviewed units")
    owners: dict[str, str] = {}
    for cid, decision in candidates.items():
        refs = decision["reference_ids"]
        if (not isinstance(refs, list) or not all(isinstance(rid, str) for rid in refs)
                or len(set(refs)) != len(refs) or set(refs) - references.keys()):
            raise ValueError("Unknown or duplicate reference mapping")
        if refs and decision["status"] != "SUPPORTED":
            raise ValueError("Only supported candidates can cover references")
        for rid in refs:
            if rid in owners:
                raise ValueError("Reference coverage must have one candidate owner")
            owners[rid] = cid
    for rid, decision in references.items():
        cid = decision["candidate_id"]
        if decision["status"] == "MATCHED":
            if cid is None or owners.get(rid) != cid:
                raise ValueError("Matched reference requires reciprocal supported coverage")
        elif cid is not None or rid in owners:
            raise ValueError("Only matched references may have candidate coverage")
    return list(candidates.values()), list(references.values())


def measures(candidates: list[Row], references: list[Row], eligible: bool) -> Row:
    c, r = Counter(d["status"] for d in candidates), Counter(d["status"] for d in references)
    ready = eligible and not c["UNREVIEWED"] and not r["UNREVIEWED"]
    def ratio(n: int, d: int) -> float | None:
        return n / d if ready and d else None
    c_all = c["SUPPORTED"] + c["CONTRADICTED"] + c["UNVERIFIABLE"]
    r_all = r["MATCHED"] + r["OMITTED"] + r["UNVERIFIABLE"]
    return {"status": "REVIEWED_REFERENCE_CONDITIONAL" if ready else "NOT_MEASURED",
        "candidate_counts": {s: c[s] for s in sorted(CANDIDATE)},
        "reference_counts": {s: r[s] for s in sorted(REFERENCE)},
        "semantic_precision_adjudicable": ratio(c["SUPPORTED"], c["SUPPORTED"] + c["CONTRADICTED"]),
        "supported_fraction_all_applicable": ratio(c["SUPPORTED"], c_all),
        "contradicted_fraction_all_applicable": ratio(c["CONTRADICTED"], c_all),
        "unverifiable_fraction_all_applicable": ratio(c["UNVERIFIABLE"], c_all),
        "matched_reference_recall_adjudicable": ratio(r["MATCHED"], r["MATCHED"] + r["OMITTED"]),
        "matched_reference_fraction_all_applicable": ratio(r["MATCHED"], r_all),
        "omission_fraction_adjudicable": ratio(r["OMITTED"], r["MATCHED"] + r["OMITTED"]),
        "reference_unverifiable_fraction_all_applicable": ratio(r["UNVERIFIABLE"], r_all),
        "precision_denominator_adjudicable": c["SUPPORTED"] + c["CONTRADICTED"],
        "recall_denominator_adjudicable": r["MATCHED"] + r["OMITTED"],
        "applicable_candidate_units": c_all, "applicable_reference_units": r_all,
        "candidate_units_total": sum(c.values()), "reference_units_total": sum(r.values()),
        "candidate_denominator_status": "UNREVIEWED" if c["UNREVIEWED"] else
            "NOT_APPLICABLE" if not c_all else "REVIEWED",
        "reference_denominator_status": "UNREVIEWED" if r["UNREVIEWED"] else
            "NOT_APPLICABLE" if not r_all else "REVIEWED",
        "empty_candidate_denominator": "NOT_APPLICABLE" if not c_all and not c["UNREVIEWED"] else None,
        "empty_reference_denominator": "NOT_APPLICABLE" if not r_all and not r["UNREVIEWED"] else None}


def summarize(cases: list[Row], reviews: list[Row]) -> Row:
    by_hash = {r["packet_hash"]: r for r in reviews}
    if len(by_hash) != len(reviews) or set(by_hash) != {c["packet_hash"] for c in cases}:
        raise ValueError("Every packet requires exactly one review entry")
    candidates, references, results = [], [], []
    fields: dict[str, tuple[list[Row], list[Row]]] = {}
    eligible = [c for c in cases if c["comparison_status"] == "ELIGIBLE"]
    complete = bool(eligible) and all(by_hash[c["packet_hash"]]["coverage_status"] == "COMPLETE"
                                      for c in eligible)
    for case in cases:
        c, r = validated_decisions(case, by_hash[case["packet_hash"]])
        results.append({"packet_hash": case["packet_hash"], "sample_id": case["sample_id"],
            **measures(c, r, case["comparison_status"] == "ELIGIBLE"
                       and by_hash[case["packet_hash"]]["coverage_status"] == "COMPLETE")})
        if case["comparison_status"] != "ELIGIBLE":
            continue
        candidates.extend(c)
        references.extend(r)
        for side, decisions in (("candidate", c), ("reference", r)):
            unit_fields = {u["unit_id"]: u["field"] for u in case[side + "_units"]}
            for d in decisions:
                groups = fields.setdefault(unit_fields[d["unit_id"]], ([], []))
                groups[0 if side == "candidate" else 1].append(d)
    return {"protocol": PROTOCOL, "scope": "Conditional on archived approved O0 and explicit reviews; "
        "no image truth, human verification, module causality or benchmark score certification",
        "unit_policy": "Whole top-level list item; no recursive leaves; null/empty fields create no units",
        "reference_policy": "Archived pre-inference O0 facts; atomization defined post-run",
        "mapping_policy": "Explicit disjoint coverage; grouped candidate may cover several references",
        "cases": len(cases), "eligible_cases": len(eligible),
        "not_comparable_cases": len(cases) - len(eligible),
        "schema_statuses": dict(Counter(c["schema_parse_status"] for c in cases)),
        "schema_failures": sum(c["schema_parse_status"] == "INVALID" for c in cases),
        "schema_completed_denominator": sum(c["runtime_status"] == "COMPLETED" for c in cases),
        "schema_parse_failure_fraction": (sum(c["schema_parse_status"] == "INVALID" for c in cases)
            / sum(c["runtime_status"] == "COMPLETED" for c in cases))
            if any(c["runtime_status"] == "COMPLETED" for c in cases) else None,
        "oracle_gate_statuses": dict(Counter(c["oracle_gate"]["status"] for c in cases)),
        "reviewed_cases": sum(r["status"] == "REVIEWED_REFERENCE_CONDITIONAL" for r in results),
        "overall": measures(candidates, references, complete),
        "per_field": {name: measures(c, r, complete) for name, (c, r) in sorted(fields.items())},
        "relations": measures([d for c in cases if c["comparison_status"] == "ELIGIBLE"
            for d in by_hash[c["packet_hash"]]["candidate_decisions"]
            if d["unit_id"] in {u["unit_id"] for u in c["candidate_units"]
                                 if u["field"] in RELATION_FIELDS}],
            [d for c in cases if c["comparison_status"] == "ELIGIBLE"
             for d in by_hash[c["packet_hash"]]["reference_decisions"]
             if d["unit_id"] in {u["unit_id"] for u in c["reference_units"]
                                  if u["field"] in RELATION_FIELDS}], complete),
        "relation_scope": "Schema relation-bearing fields selected post-run; not critical solving relations",
        "critical_full_structure_exact": {"status": "NOT_MEASURED", "value": None,
            "reason": "No demonstrably preregistered critical solving relation list"},
        "invented_nonvisible_content_rate": {"status": "NOT_MEASURED", "value": None,
            "reason": "Contradicted or unmatched reference content does not certify invisible invention"},
        "per_case": results, **PROVENANCE}


def export(data: Path | None, runs: list[tuple[str, Path]], output: Path,
           queue: Path | None = None, reviews: Path | None = None) -> Row:
    output = output.resolve()
    private = Path(__file__).resolve().parents[2] / "analysis_exports/bottleneck_20261003"
    if output.parent != private or not output.name.startswith("p_structure_measurement_") or output.exists():
        raise ValueError("Output must be a fresh private p_structure_measurement_* directory")
    sources: dict[str, str] = {}
    def freeze(path: Path) -> Path:
        path = path.resolve(strict=True)
        sources[str(path)] = sha(path)
        return path
    cases = []
    if queue is not None:
        if data is not None or runs or reviews is None:
            raise ValueError("Review requires queue and reviews only")
        frozen_queue = freeze(queue)
        receipt = json.loads(freeze(queue.parent / "receipt.json").read_text())
        if receipt["outputs"].get(queue.name) != sha(frozen_queue):
            raise ValueError("Queue differs from its source receipt")
        for source, digest in receipt["sources"].items():
            if sha(freeze(Path(source))) != digest:
                raise ValueError("Archived queue source differs from receipt")
        cases = read_rows(frozen_queue)
        decisions = read_rows(freeze(reviews))
    else:
        if data is None or not runs or reviews is not None:
            raise ValueError("Queue requires data and runs only")
        for name in ("samples.jsonl", "sample_selection.json"):
            freeze(data / name)
        seen = set()
        for suite, run in runs:
            samples = selected(data, suite)
            for name in ("protocol.json", "responses.jsonl"):
                freeze(run / name)
            identity, _, latest, _ = validated_run(run, samples)
            oracle_path = freeze(run.parent / "input/oracles_used.jsonl")
            oracles = read_rows(oracle_path)
            if canonical_hash(sorted(oracles, key=lambda o: (o["sample_id"], o["level"]))) != identity["oracle_manifest_hash"]:
                raise ValueError("Archived pre-inference oracle manifest mismatch")
            oracle_map = {(o["sample_id"], o["level"]): o for o in oracles}
            if len(oracle_map) != len(oracles):
                raise ValueError("Duplicate oracle IDs")
            sample_map = {s["sample_id"]: s for s in samples}
            for row in latest:
                if row["mode"] != "P":
                    continue
                key = row["sample_id"], row["repetition"]
                if key in seen:
                    raise ValueError("Overlapping P cells")
                seen.add(key)
                cases.append(packet(row, sample_map[row["sample_id"]],
                                    oracle_map.get((row["sample_id"], "O0"))))
        decisions = [template(c) for c in cases]
    if not cases:
        raise ValueError("No P cases")
    summary = summarize(cases, decisions)
    for path, digest in sources.items():
        if sha(path) != digest:
            raise ValueError("Input changed during read-only measurement")
    output.mkdir(exist_ok=False)
    write_rows(output / "queue.jsonl", cases)
    write_rows(output / "review_template.jsonl" if queue is None else output / "validated_reviews.jsonl", decisions)
    write_json(output / "summary.json", summary)
    write_json(output / "receipt.json", {"sources": sources, "source_unmodified": True,
        "utility_sha256": sha(Path(__file__)), "outputs": {p.name: sha(p) for p in output.iterdir()},
        "gold_accessed": False, "automatic_semantic_matches": 0, **PROVENANCE})
    return {"output": str(output), "cases": summary["cases"],
            "eligible_cases": summary["eligible_cases"], "reviewed_cases": summary["reviewed_cases"],
            "status": summary["overall"]["status"], **PROVENANCE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run", action="append", default=[], metavar="SUITE=DIR")
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        runs = []
        for value in args.run:
            suite, separator, path = value.partition("=")
            if not separator or suite not in {"pilot", "coverage"} or not path:
                raise ValueError("--run must be pilot=DIR or coverage=DIR")
            runs.append((suite, Path(path)))
        result = export(args.data, runs, args.output, args.queue, args.reviews)
    except (OSError, KeyError, ValueError, TypeError) as error:
        parser.exit(2, f"Private structure review: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
