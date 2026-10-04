"""Rescore three recovered runs and audit scoring-only changes against preserved V6.

This is CPU-only development regression, not new inference or model-error diagnosis.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
from pathlib import Path
import subprocess
import sys

from mmdl.evaluation.parsers import score_with_parser, scoring_source_files
from mmdl.runtime.artifacts import atomic_text, digest, read_json, sha256_file, write_json
from scripts.validate_rescore import validate


RUNS = {"baseline_2048": "baseline_review.json", "continuous_32k": "continuous_review.json",
        "official_b2": "official_b2_review.json"}
PARSER = "team-final-answer-v8"


def run(v6_root: Path, audit_root: Path, output: Path, public: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    initial_hashes = [sha256_file(path) for path in scoring_source_files(PARSER)]
    scoring_hash = digest([PARSER, *initial_hashes])
    summary = {"parser": PARSER, "scoring_sha256": scoring_hash,
               "new_inference": False, "semantic_error_free": False,
               "validation_scope": "CPU replay, source integrity and development regression; no independent holdout",
               "runs": {}}
    table = io.StringIO(newline="")
    csv_writer = csv.writer(table, lineterminator="\n")
    csv_writer.writerow(["run", "subject", "correct", "total", "accuracy", "generation_seconds", "sample_seconds"])
    for name, ledger_name in RUNS.items():
        reference = v6_root / name
        source_record = read_json(reference / "source.json")
        source = Path(source_record["source_run"])
        if sha256_file(source / "predictions.jsonl") != source_record["source_predictions_sha256"]:
            raise ValueError(f"Original source changed: {name}")
        destination = output / name
        subprocess.run([sys.executable, "-c", "from mmdl.evaluation.cli import rescore_main; rescore_main()",
                        "--artifact-root", str(source.parent.parent), "--run-id", source.name,
                        "--parser", PARSER, "--output-dir", str(destination)], check=True)
        verification = validate(source, destination)
        write_json(destination / "verification.json", verification)
        old = {r["id"]: r for p in sorted((reference / "samples").glob("*.json")) if (r := read_json(p))}
        rows = [read_json(p) for p in sorted((destination / "samples").glob("*.json"))]
        current = {r["id"]: r for r in rows}
        if set(old) != set(current):
            raise ValueError(f"V6/V8 ID sets differ: {name}")
        transitions: Counter[str] = Counter()
        changes = []
        for row in rows:
            before = old[row["id"]]
            transitions[f"{int(before['correct'])}->{int(row['correct'])}"] += 1
            if any(before.get(key) != row[key] for key in ("correct", "parsed_answer", "parse_status")):
                changes.append({"id": row["id"], "v6_answer": before["parsed_answer"],
                                "v8_answer": row["parsed_answer"], "v6_correct": before["correct"],
                                "v8_correct": row["correct"], "finish_reason": row["finish_reason"],
                                "rule": row["extraction_rule"], "evidence": row["extraction_evidence"]})
            alternative = score_with_parser(row["raw_response"], row["question_type"], row["options"],
                                            "__gold_permutation_probe__", PARSER, row.get("finish_reason"),
                                            question=row["question"])
            if any(alternative[k] != row[k] for k in alternative if k != "correct"):
                raise ValueError(f"Gold-dependent extraction: {row['id']}")
        ledger = read_json(audit_root / ledger_name)
        cases = ledger if isinstance(ledger, list) else ledger.get("records", ledger.get("cases"))
        regression = [{"id": c["id"], "category": c["category"],
                       "v8_correct": current[c["id"]]["correct"],
                       "v8_rule": current[c["id"]]["extraction_rule"]} for c in cases]
        write_json(destination / "v6_comparison.json", {"transitions": dict(transitions), "changes": changes})
        write_json(destination / "historical_regression.json", regression)
        scores = read_json(destination / "summary.json")
        counts = Counter(c["category"] for c in regression)
        recovered = Counter(c["category"] for c in regression if c["v8_correct"])
        summary["runs"][name] = {
            "correct": scores["correct"], "denominator": 900, "accuracy": scores["accuracy"],
            "no_parse": sum(r["parse_status"] == "NO_PARSE" for r in rows),
            "empty": sum(r["parse_status"] == "EMPTY" for r in rows),
            "length_total": sum(r["finish_reason"] == "length" for r in rows),
            "length_correct": sum(r["finish_reason"] == "length" and r["correct"] for r in rows),
            "v6_transitions": dict(transitions), "regression_counts": dict(counts),
            "regression_correct": dict(recovered), "gold_independence_checked": 900,
            "record_hashes_and_replay_verified": 900, "all_non_scoring_fields_unchanged": True,
            "source_predictions_sha256": sha256_file(source / "predictions.jsonl"),
            "v8_predictions_sha256": sha256_file(destination / "predictions.jsonl"),
        }
        for subject in scores["subject_scores"]:
            csv_writer.writerow([name, subject["subject"], subject["correct"], 30, subject["accuracy"],
                                 subject["generation_seconds"], subject["sample_seconds"]])
        print(f"{name}: {scores['correct']}/900; historical regression {dict(recovered)} / {dict(counts)}", flush=True)
    if initial_hashes != [sha256_file(path) for path in scoring_source_files(PARSER)]:
        raise ValueError("Scorer changed during suite; discard this candidate, do not publish")
    summary["status"] = "CPU_REPLAY_VERIFIED"
    write_json(output / "summary.json", summary)
    atomic_text(output / "subject_scores.csv", table.getvalue())
    if public is not None:
        public.mkdir(parents=True, exist_ok=False)
        write_json(public / "summary.json", summary)
        atomic_text(public / "subject_scores.csv", table.getvalue())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--public", type=Path, help="optional new directory for text-free summary/subject CSV")
    args = parser.parse_args()
    run(args.v6_root, args.audit_root, args.output, args.public)
