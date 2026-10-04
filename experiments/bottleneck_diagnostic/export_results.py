"""Private, read-only post-run scoring/export; never creates an inference run."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mmdl.evaluation.parsers import scoring_source_files

from .cli import selected, validated_run
from .ingest import read_rows, sha, write_json, write_rows
from .input_builder import canonical_hash
from .metrics import PAIRS, cluster, summarize
from .scoring import SCORING_PROTOCOL, score_responses

Row = dict[str, Any]
PROVENANCE = {"provenance": "BENCHMARK_DERIVED", "training_allowed": False}


def compatible_union(runs: list[Row]) -> tuple[list[Row], list[Row], list[Row]]:
    """Combine independently validated, disjoint suites, never average summaries."""
    if not runs:
        raise ValueError("At least one run is required")
    if len(runs) > 1 and any(r["suite"] in {"stress", "local"} for r in runs):
        raise ValueError("Stress/local exploratory runs must be exported standalone")
    first = runs[0]["identity"]
    keys = ("protocol_hash", "code_hash", "execution_kind", "repetitions",
            "backend_evaluation_config", "backend_hardware_profile", "model_files_manifest_hash")
    seen: set[str] = set()
    samples, scores, attempts = [], [], []
    for run in runs:
        if any(run["identity"].get(k) != first.get(k) for k in keys):
            raise ValueError("Incompatible protocol/inference-code/repetition/backend identities")
        ids = {s["sample_id"] for s in run["samples"]}
        if ids & seen or len(ids) != len(run["samples"]):
            raise ValueError("Overlapping/duplicate sample IDs; no silent deduplication")
        seen.update(ids)
        samples.extend(run["samples"])
        scores.extend(run["scores"])
        attempts.extend(run["attempts"])
    return samples, scores, attempts


def cost_rows(attempts: list[Row]) -> Row:
    by_mode: dict[str, list[Row]] = defaultdict(list)
    for row in attempts:
        by_mode[row["mode"]].append(row)

    def total(rows: list[Row]) -> Row:
        image = sum(r.get("backend_audit", {}).get("image_tokens", 0) or 0 for r in rows)
        inputs = sum(r.get("input_tokens", 0) or 0 for r in rows)
        if image > inputs:
            raise ValueError("Image token count exceeds total input tokens")
        memory = [r.get("backend_audit", {}).get("observed_device_memory_used_bytes")
                  for r in rows]
        snapshots = [v for v in memory if type(v) is int and v >= 0]
        actual = [r for r in rows if r.get("run_kind") == "ACTUAL_INFERENCE"]
        completed = sum(r.get("runtime_status") == "COMPLETED" for r in actual)
        execution_attempts = [r for r in actual if r.get("attempt_id")]
        unsuccessful = sum(r.get("runtime_status") != "COMPLETED" for r in execution_attempts)
        uncertain_calls = unsuccessful or any(r.get("runtime_status") == "SYSTEM_ERROR"
                                              for r in actual)
        return {"ledger_attempt_records": len(rows),
                "completed_calls": completed,
                "recorded_execution_path_attempts": len(execution_attempts),
                "unresolved_unsuccessful_execution_attempts": unsuccessful,
                "actual_call_count": None if uncertain_calls else completed,
                "actual_call_count_status": "UNCERTAIN" if uncertain_calls else "PROVEN_COMPLETED",
                "input_tokens": inputs, "input_text_tokens": inputs - image,
                "image_tokens": image,
                "output_tokens": sum(r.get("output_tokens", 0) or 0 for r in rows),
                "budget_reserved_output_tokens": sum(r.get("charged_output_tokens", 0) or 0
                                                       for r in rows),
                "request_seconds": sum(r.get("elapsed_seconds", 0) or 0 for r in rows),
                "max_observed_device_memory_bytes": max(snapshots, default=None),
                "memory_measurement": "snapshot_not_true_peak",
                "true_peak_gpu_bytes": None, "usd": None}

    latest = {(r["sample_id"], r["mode"], r["repetition"]): r for r in attempts}
    logical = {}
    for mode in ("M", "MI"):
        pairs = [(p, r) for (sid, m, rep), r in latest.items() if m == mode
                 and r["runtime_status"] == "COMPLETED"
                 and (p := latest.get((sid, "P", rep))) is not None
                 and p["runtime_status"] == "COMPLETED"]
        matched = {(r["sample_id"], r["repetition"]) for _, r in pairs}
        logical["P+" + mode] = total([r for r in attempts if r["mode"] in {"P", mode}
            and (r["sample_id"], r["repetition"]) in matched]) | {
            "matched_cells": len(pairs),
            "accounting": "Logical pipeline cost; shared P is counted once in actual_total, "
                          "not charged twice by summing these alternative pipeline views."}
    return {"per_mode": {m: total(rows) for m, rows in sorted(by_mode.items())},
            "actual_total": total(attempts), "logical_matched_pipelines": logical,
            "request_sum_is_wall_time": False, "cost_matched_compute_effect": None,
            "attempt_scope": "attempt_id records entry before image loading, not backend invocation; "
                             "only completed actual calls are proven. Unsuccessful execution paths "
                             "leave actual_call_count uncertain/null.",
            "token_scope": "output_tokens are recorded generated tokens; budget_reserved_output_tokens "
                           "are charged_output_tokens reservations, NOT observed generation.",
            "request_time_scope": "Sum of recorded elapsed_seconds across all ledger attempts, "
                                  "including unsuccessful attempts; not wall time."}


def structure_summary(scores: list[Row]) -> Row:
    rows = [r for r in scores if r["mode"] == "P"]
    scored = [r for r in rows if r.get("structure_status") == "SCORED_EXACT_JSON"]
    fields = [f for r in scored for f in r.get("fields", [])]
    field_groups: dict[str, list[Row]] = defaultdict(list)
    for f in fields:
        field_groups[f["field"]].append(f)
    return {"scope": "Exact JSON representation comparison, NOT semantic certification",
            "requested_cells": len(rows), "schema_statuses": dict(Counter(
                r.get("schema_parse_status", "NOT_AVAILABLE") for r in rows)),
            "structure_statuses": dict(Counter(r.get("structure_status") for r in rows)),
            "oracle_comparable_cells": len(scored),
            "structure_exact_cells": sum(r["structure_exact"] is True for r in scored),
            "field_exact_count": sum(f["exact"] for f in fields), "field_count": len(fields),
            "field_exact_rate": sum(f["exact"] for f in fields) / len(fields) if fields else None,
            "missing_fields": sum(r.get("missing_fields", 0) for r in scored),
            "unmatched_list_items": sum(r.get("unmatched_list_items", 0) for r in scored),
            "per_field": {name: {"count": len(fs), "exact": sum(f["exact"] for f in fs),
                                    "missing": sum(f["missing"] for f in fs)}
                          for name, fs in sorted(field_groups.items())},
            "relations": {name: {"nonnull_cells": len(vs),
                                  "macro_mean": sum(vs) / len(vs) if vs else None}
                          for name in ("relation_precision", "relation_recall")
                          for vs in [[r[name] for r in scored if r.get(name) is not None]]},
            "semantic_field_accuracy": None, "semantic_relation_precision": None,
            "semantic_relation_recall": None, "semantic_structure_exact_rate": None,
            "caveat": "Text synonyms/order/schema granularity can mismatch; unmatched JSON "
                      "items are not certified hallucinations. Null relation metrics lack denominators."}


def decision_envelope(entries: list[Row], denominator: int) -> Row:
    """Conditional decision envelope, not bounds on true semantic accuracy."""
    if len(entries) > denominator:
        raise ValueError("Decision rows exceed fixed denominator")
    true = sum(r["status"] == "RESOLVED" and r.get("candidate_correct") is True
               for r in entries)
    false = sum(r["status"] == "RESOLVED" and r.get("candidate_correct") is False
                for r in entries)
    resolved = sum(r["status"] == "RESOLVED" for r in entries)
    if resolved != true + false:
        raise ValueError("RESOLVED requires a boolean candidate decision")
    unresolved = denominator - resolved
    return {"denominator": denominator, "resolved_true": true, "resolved_false": false,
            "unresolved": unresolved, "missing": denominator - len(entries),
            "statuses": dict(Counter(r["status"] for r in entries)),
            "conditional_lower": true / denominator,
            "conditional_upper": (true + unresolved) / denominator,
            "scope": "Conditional on accepted RESOLVED decisions; NOT true-semantic accuracy bounds"}


def gemma_export(directory: Path, source: Path) -> tuple[Row, list[Row]]:
    from experiments.gemma_verifier import runtime as g
    from .finish_runpod import audit_result

    frozen = {p: sha(p) for p in [source, *(directory / n for n in
              ("run_manifest.json", "ledger.jsonl", "summaries.jsonl"))] if p.exists()}
    source_sha256 = frozen[source]
    checked = audit_result(directory, source)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    extraction_mode = manifest.get("extraction_mode", "literal")
    sources = g.load_rows(source)
    source_map = {r["id"]: r for r in sources}
    counts = Counter(r["subject"] for r in sources)
    if len(counts) != 30 or set(counts.values()) != {30}:
        raise ValueError("Gemma requires original900,30 subjects x30")
    if sum(r["correct"] for r in sources) != 558:
        raise ValueError("Gemma source does not match frozen558/900 V8 baseline")
    entries = read_rows(directory / "ledger.jsonl") if (directory / "ledger.jsonl").exists() else []
    compared = 0
    for entry in entries:
        if entry["status"] not in {"RESOLVED", "REVIEW", "SYSTEM_ERROR", "PENDING", "BUDGET_STOP"}:
            raise ValueError("Unknown Gemma terminal/partial status")
        row = source_map[entry["id"]]
        if entry["subject"] != row["subject"] or entry["raw_response_sha256"] != hashlib.sha256(
                row["raw_response"].encode()).hexdigest():
            raise ValueError("Gemma source/raw identity mismatch")
        if entry["input_sha256"] != g.digest(g.extraction_input(row, extraction_mode)):
            raise ValueError("Gemma input hash mismatch")
        if entry["status"] not in {"RESOLVED", "REVIEW"}:
            continue
        if entry["requests"]["extraction"] != g.request_payload(row, manifest["model"], extraction_mode):
            raise ValueError("Gemma extraction request differs from frozen source")
        judge = g._checked_response(entry["http_responses"]["extraction"], manifest["model"])
        if judge != entry["judge_raw"]:
            raise ValueError("Gemma actual response differs from judge JSON")
        mapping = None
        if entry.get("mapping_raw") is not None:
            mapping = g._checked_response(entry["http_responses"]["mapping"], manifest["model"])
            if mapping != entry["mapping_raw"]:
                raise ValueError("Gemma mapping response differs")
            answer = entry["frozen_submitted_answer"]
            if entry["requests"]["mapping"] != g.mapping_request(row, answer, manifest["model"]):
                raise ValueError("Gemma mapping request differs")
        equiv = None
        if entry.get("equivalence_raw") is not None:
            equiv = g._checked_response(entry["http_responses"]["equivalence"], manifest["model"])
            if equiv != entry["equivalence_raw"]:
                raise ValueError("Gemma equivalence response differs")
            answer = entry["validated_extraction"]["json"]["extracted_answer"]
            if entry["requests"]["equivalence"] != g.equivalence_request(row, answer, manifest["model"]):
                raise ValueError("Gemma equivalence request differs")
        replay = g.audit_rows([row], {row["id"]: judge},
                              {row["id"]: equiv} if equiv is not None else None,
                              mapping_outputs={row["id"]: mapping} if mapping is not None else None,
                              extraction_mode=extraction_mode)[0]
        if any(entry.get(k) != value for k, value in replay.items()):
            raise ValueError("Gemma pure audit replay differs from saved decisions")
        compared += 1
    if (directory / "summaries.jsonl").exists() and any(
            checked.get(key) != value for key, value in g.summarize(entries, extraction_mode).items()):
        raise ValueError("Gemma summary replay differs from saved decisions")
    queue = [{"id": r["id"], "subject": r["subject"], "status": r["status"],
              "review_reasons": r.get("review_reasons", []),
              "source_path": str(source), "source_sha256": source_sha256,
              "raw_response_sha256": r["raw_response_sha256"],
              "input_sha256": r["input_sha256"], "judge_raw": r.get("judge_raw"),
              "validated_extraction": r.get("validated_extraction"), **PROVENANCE}
             for r in entries if r["status"] != "RESOLVED"]
    present = {r["id"] for r in entries}
    queue.extend({"id": r["id"], "subject": r["subject"], "status": "MISSING",
                  "source_path": str(source), "source_sha256": source_sha256,
                  "input_sha256": g.digest(g.extraction_input(r, extraction_mode)),
                  "raw_response_sha256": hashlib.sha256(r["raw_response"].encode()).hexdigest(),
                  **PROVENANCE}
                 for r in sources if r["id"] not in present)
    per_subject = {subject: decision_envelope([r for r in entries if r["subject"] == subject], 30)
                   for subject in sorted({r["subject"] for r in sources})}
    for subject, envelope in per_subject.items():
        group = [r for r in entries if r["subject"] == subject]
        envelope.update(stored_v8_correct=sum(r["correct"] for r in sources
                        if r["subject"] == subject),
                        candidate_correct_not_certified=sum(r.get("candidate_correct") is True
                                                            for r in group),
                        candidate_labels=dict(Counter(r["candidate"] for r in group
                                                      if r.get("candidate"))))
    by_stage = {}
    for stage in ("extraction", "mapping", "equivalence"):
        responses = [r["http_responses"][stage] for r in entries
                     if stage in r.get("http_responses", {})]
        by_stage[stage] = {"response_calls": len(responses),
            "recorded_request_attempts": sum(stage in r.get("requests", {}) for r in entries),
            "input_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in responses),
            "output_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in responses),
            "usd": None, "true_peak_gpu_bytes": None}
    if sha(source) != manifest["source_sha256"]:
        raise ValueError("Gemma source changed during replay")
    if any(sha(p) != digest for p, digest in frozen.items()):
        raise ValueError("Gemma ledger/manifest changed during replay; use a closed stage")
    return {"validation": checked, "replayed_terminal_rows": compared,
            "stored_v8_correct": sum(r["correct"] for r in sources),
            "candidate_correct_not_certified": sum(r.get("candidate_correct") is True for r in entries),
            "candidate_labels": dict(Counter(r["candidate"] for r in entries if r.get("candidate"))),
            "decision_envelope": decision_envelope(entries, 900), "per_subject": per_subject,
            "source_sha256": source_sha256, "stage_costs": by_stage, "replaces_v8": False,
            "independent_human_verified": False, **PROVENANCE}, queue


def _csv(path: Path, rows: list[Row]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                         else v for k, v in row.items()} for row in rows)


def export(data: Path, oracle_path: Path, specs: list[tuple[str, Path]], output: Path,
           gemma_run: Path | None = None, gemma_source: Path | None = None) -> Row:
    data, oracle_path, output = data.resolve(strict=True), oracle_path.resolve(strict=True), output.resolve()
    if output.exists() or "MMDL" in output.parts or "analysis_exports" not in output.parts:
        raise ValueError("Output must be a fresh private analysis_exports directory")
    if bool(gemma_run) != bool(gemma_source):
        raise ValueError("Gemma run and source must be supplied together")
    sources = {str(p): sha(p) for p in [data / "receipt.json", data / "gold.jsonl",
               data / "samples.jsonl", data / "sample_selection.json", oracle_path]}
    receipt = json.loads((data / "receipt.json").read_text())
    if sha(data / "gold.jsonl") != receipt["files"]["gold.jsonl"]:
        raise ValueError("Frozen gold ledger hash mismatch")
    gold = {r["sample_id"]: ast.literal_eval(r["answer"]) if isinstance(r["answer"], str)
            and r["answer"].startswith("[") else r["answer"] for r in read_rows(data / "gold.jsonl")}
    oracles = read_rows(oracle_path)
    oracle_map = {(r["sample_id"], r["level"]): r for r in oracles}
    if len(oracle_map) != len(oracles):
        raise ValueError("Duplicate oracle IDs/levels")
    runs = []
    for suite, directory in specs:
        directory = directory.resolve(strict=True)
        samples = selected(data, suite)
        if output.is_relative_to(directory) or output.is_relative_to(data):
            raise ValueError("Output must not be nested inside an input run or prepared data")
        for name in ("protocol.json", "responses.jsonl", "scores.jsonl", "summary.json"):
            path = directory / name
            if path.exists():
                sources[str(path)] = sha(path)
        identity, attempts, latest, missing = validated_run(directory, samples)
        if any(r["runtime_status"] not in {"COMPLETED", "BLOCKED", "EXCLUDED", "PENDING",
                                           "SYSTEM_ERROR", "BUDGET_STOP"} for r in latest):
            raise ValueError("Unknown diagnostic runtime status")
        ids = {s["sample_id"] for s in samples}
        subset = [o for o in oracles if o["sample_id"] in ids]
        possible = [canonical_hash(sorted(os, key=lambda o: (o["sample_id"], o["level"])))
                    for os in (oracles, subset)]
        if identity["oracle_manifest_hash"] not in possible:
            raise ValueError("Frozen inference oracle manifest differs from supplied oracle inputs")
        indexed = {(r["sample_id"], r["mode"], r["repetition"]): r for r in latest}
        for row in latest:
            if row["mode"] in {"M", "MI"} and row["runtime_status"] == "COMPLETED":
                p = indexed.get((row["sample_id"], "P", row["repetition"]))
                if not p or p.get("response_hash") != row.get("perception_response_hash"):
                    raise ValueError("M/MI does not reuse the exact P response hash")
        runs.append({"suite": suite, "path": str(directory), "identity": identity,
                     "samples": samples, "attempts": attempts, "latest": latest,
                     "missing": missing, "scores": score_responses(latest, samples, gold, oracle_map)})
    samples, scores, attempts = compatible_union(runs)
    latest = [r for run in runs for r in run["latest"]]
    summary = summarize(scores, samples, latest)
    for group_name, group in summary["groups"].items():
        ids = {s["sample_id"] for s in samples if group_name == "overall_selected"
               or group_name == "subject/" + s["subject"]
               or group_name == "cluster/" + cluster(s["subject"])
               or group_name.removeprefix("image_type/") in s["skill_tags"]}
        for mode, values in group["modes"].items():
            comparable = [r for r in scores if r["sample_id"] in ids and r["mode"] == mode
                          and type(r.get("v8_score")) is bool]
            values.update(v8_applicable_completed=len(comparable),
                v8_correct=sum(r["v8_score"] for r in comparable),
                v8_accuracy=sum(r["v8_score"] for r in comparable) / len(comparable)
                if comparable else None)
    summary["cost"] = cost_rows(attempts)
    summary.update(status="INCOMPLETE" if any(r["runtime_status"] in {
        "PENDING", "SYSTEM_ERROR", "BUDGET_STOP"} for r in latest) else "ALL_DECLARED_CELLS_TERMINAL",
        actual_run_completion_not_inferred=True, scoring_protocol=SCORING_PROTOCOL,
        P_structure=structure_summary(scores), source_runs=[{"suite": r["suite"], "path": r["path"],
            "identity": r["identity"], "missing_cells": r["missing"]} for r in runs],
        frozen_baseline={"correct": 558, "denominator": 900, "replaced": False},
        v8_comparison_scope="Unchanged V8 applied to NEW diagnostic raw; not historical32K comparison",
        exploratory_conditions=[m for m in {r["mode"] for r in scores} if m == "LEGACY"],
        exploratory_suite=any(r["suite"] in {"stress", "local"} for r in runs),
        environment="Use source-run doctor/freeze artifacts; configs are not actual environment certification")
    gemma_queue: list[Row] = []
    if gemma_run is not None and gemma_source is not None:
        summary["gemma"], gemma_queue = gemma_export(gemma_run, gemma_source)
        for path in [gemma_source, *(gemma_run / n for n in
                     ("run_manifest.json", "ledger.jsonl", "summaries.jsonl"))]:
            if path.exists():
                sources[str(path.resolve())] = sha(path)
    for source_name, digest in sources.items():
        if sha(source_name) != digest:
            raise ValueError("Input changed during read-only export")
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "scores_v2.jsonl", scores)
    transitions: list[Row] = []
    subjects: list[Row] = []
    skills: list[Row] = []
    for name, group in summary["groups"].items():
        transitions.extend({"group": name, **pair} for pair in group["paired"])
        target = subjects if name.startswith("subject/") else skills if name.startswith("image_type/") else None
        if target is not None:
            target.extend({"group": name, "questions": group["questions"], "mode": mode, **values}
                          for mode, values in group["modes"].items())
    indexed_scores = {(r["sample_id"], r["mode"], r["repetition"]): r for r in scores}
    sample_map = {s["sample_id"]: s for s in samples}
    response_map = {(r["sample_id"], r["mode"], r["repetition"]): r for r in latest}
    run_map = {s["sample_id"]: r["path"] for r in runs for s in r["samples"]}
    pairs = []
    for (sid, mode, rep), row in indexed_scores.items():
        for before, after in PAIRS:
            other = indexed_scores.get((sid, after, rep))
            if mode == before and other and type(row.get("diagnostic_score")) is bool and type(other.get("diagnostic_score")) is bool:
                pairs.append({"sample_id": sid, "family_id": sample_map[sid]["family_id"],
                    "subject": sample_map[sid]["subject"], "repetition": rep,
                    "before": before, "after": after, "before_correct": row["diagnostic_score"],
                    "after_correct": other["diagnostic_score"],
                    "source_run": run_map[sid],
                    "before_response_hash": response_map[sid, before, rep].get("response_hash"),
                    "after_response_hash": response_map[sid, after, rep].get("response_hash"),
                    "delta": int(other["diagnostic_score"]) - int(row["diagnostic_score"])})
    coverage = [{"mode": mode, "runtime_status": status, "cells": count}
                for mode, statuses in summary["oracle_coverage"].items() for status, count in statuses.items()]
    for name, rows in (("transitions", transitions), ("per_subject", subjects), ("per_skill", skills),
                       ("oracle_coverage", coverage), ("paired_cases", pairs)):
        _csv(output / (name + ".csv"), rows)
    if gemma_run is not None:
        write_rows(output / "gemma_review_queue.jsonl", gemma_queue)
    write_json(output / "summary.json", summary)
    write_json(output / "source_hashes.json", {"sources": sources,
        "exporter_sha256": sha(Path(__file__)), "source_unmodified": True,
        "post_run_scoring_sources": {str(p): sha(p) for p in [*[Path(__file__).with_name(n)
            for n in ("scoring.py", "metrics.py", "cli.py", "input_builder.py", "oracle_validation.py")],
            *scoring_source_files("team-final-answer-v8")]},
        "outputs": {p.name: sha(p) for p in output.iterdir()}, **PROVENANCE})
    return {"output": str(output), "status": summary["status"], "questions": len(samples),
            "score_rows": len(scores), "summary_sha256": sha(output / "summary.json"), **PROVENANCE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="SUITE=DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gemma-run", type=Path)
    parser.add_argument("--gemma-source", type=Path)
    args = parser.parse_args()
    try:
        specs = []
        for item in args.run:
            suite, sep, path = item.partition("=")
            if not sep or suite not in {"stress", "local", "pilot", "coverage", "expansion", "expansion-remainder"} or not path:
                raise ValueError("--run requires a supported SUITE=DIR")
            specs.append((suite, Path(path)))
        result = export(args.data, args.oracles, specs, args.output, args.gemma_run, args.gemma_source)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Private export: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
