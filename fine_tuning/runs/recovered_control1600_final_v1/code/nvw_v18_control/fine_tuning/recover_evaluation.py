"""Recover a terminated F0 summary-save failure, never repair or regenerate its raw rows."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

from .audit import dump, read_rows, sha
from .evaluate import BASE_ID, BASE_REVISION, sample_seed, score_with_parser, verify_cohort_images


def process_gone(launch):
    pid = int(launch["pid"])
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return True
    # A recycled PID is not the original owned process.
    return stat.read_text().rsplit(")", 1)[1].split()[19] != str(launch["pid_birth"])


def recover(parent, cohort, protocol_path, launch_path, model_receipt_path, log_path, output, owned_pod_id):
    parent, cohort, output = Path(parent), Path(cohort), Path(output)
    launch = json.loads(Path(launch_path).read_text())
    model = json.loads(Path(model_receipt_path).read_text())
    protocol = json.loads(Path(protocol_path).read_text())
    manifest = json.loads((cohort/"manifest.json").read_text())
    command = launch["command"]
    def argument(flag):
        return command[command.index(flag)+1]
    if (os.environ.get("RUNPOD_POD_ID") != owned_pod_id or launch["owned_pod_id"] != owned_pod_id
            or not process_gone(launch) or "--checkpoint" in command
            or command[command.index("fine_tuning.evaluate")+1] != "run"
            or Path(argument("--output")) != parent or Path(argument("--cohort")) != cohort
            or Path(argument("--protocol")) != Path(protocol_path)
            or model.get("status") != "MODEL_HASH_VERIFIED"
            or model["manifest"]["repo_id"] != BASE_ID or model["manifest"]["revision"] != BASE_REVISION
            or model["model_path"] != argument("--model-path")
            or launch["cohort_sha256"] != sha(cohort/"manifest.json")
            or launch["sources"]["protocol.json"] != sha(protocol_path)):
        raise ValueError("Recovery identity/hash/process-closure evidence mismatch")
    if (sha(cohort/"inputs.jsonl") != manifest["inputs_sha256"]
            or sha(cohort/"gold.jsonl") != manifest["gold_sha256"]):
        raise ValueError("Frozen cohort changed")
    verify_cohort_images(cohort, manifest)
    broken = (parent/"summary.json").read_text()
    log = Path(log_path).read_text()
    try:
        json.loads(broken)
    except json.JSONDecodeError:
        pass
    else:
        raise ValueError("A valid closed summary must not be reconstructed")
    if ('TypeError: \'<\' not supported between instances of \'NoneType\' and \'str\'' not in log
            or '"finish_reasons":' not in broken):
        raise ValueError("Not the evidenced terminal summary serialization failure")
    prefix = json.loads(broken.split('"finish_reasons":', 1)[0].rstrip().rstrip(",") + "\n}")
    rows = read_rows(parent/"predictions.jsonl")
    inputs = read_rows(cohort/"inputs.jsonl")
    gold = {r["id"]: r for r in read_rows(cohort/"gold.jsonl")}
    if (len(rows) != manifest["count"] or len({r["id"] for r in inputs}) != len(inputs)
            or [r["id"] for r in rows] != [r["id"] for r in inputs]):
        raise ValueError("Recovery ordered cohort coverage mismatch")
    done, stopped = [], False
    required = {"raw_response", "output_token_ids", "finish_reason", "seed", "seconds", "input_tokens",
                "output_tokens", "input_sha256", "prompt_token_ids", "image_grid_thw", "processed_image_sizes",
                "correct", "subject"}
    for row, record in zip(rows, inputs, strict=True):
        if row["status"] == "BUDGET_STOP" and row.get("correct") is None:
            stopped = True
            continue
        if row["status"] != "COMPLETED" or stopped or required-set(row):
            raise ValueError("Recovery needs a complete observed prefix then budget-stop tail")
        if (any(row[k] is None for k in required) or row["seed"] != sample_seed(row["id"], protocol["master_seed"])
                or row["output_tokens"] != len(row["output_token_ids"])
                or row["input_tokens"] != len(row["prompt_token_ids"]) or row["subject"] != record["subject"]
                or row["finish_reason"] not in {"stop", "length", "other"}
                or not math.isfinite(row["seconds"]) or row["seconds"] < 0):
            raise ValueError("Recovered observed input/token/seed mismatch")
        score = score_with_parser(row["raw_response"], record["question_type"], record["options"],
                                  gold[row["id"]]["answer"], "team-final-answer-v8", row["finish_reason"],
                                  question=record["question"])
        if score["correct"] is not row["correct"]:
            raise ValueError("Recovered stored V8 score mismatch")
        done.append(row)
    if (not stopped or prefix["checkpoint"] != "F0-frozen" or prefix["checkpoint_metadata_sha256"] is not None
            or prefix["cohort_sha256"] != sha(cohort/"manifest.json") or prefix["benchmark_derived"] is not True
            or prefix["completed"] != len(done) or prefix["denominator"] != len(rows)
            or prefix["correct"] != sum(r["correct"] is True for r in done) or prefix["accuracy"] is not None):
        raise ValueError("Damaged summary prefix and preserved ledger disagree")
    start = dt.datetime.fromisoformat(launch["started_utc"])
    elapsed = (parent/"summary.json").stat().st_mtime-start.timestamp()
    if elapsed <= 0:
        raise ValueError("Invalid observed terminal timing")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parent/"predictions.jsonl", output/"predictions.jsonl")
    summary = dict(prefix, status="BUDGET_STOP", model_id=BASE_ID, model_revision=BASE_REVISION,
        protocol_sha256=sha(protocol_path), predictions_sha256=sha(output/"predictions.jsonl"),
        evaluator_source_sha256=launch["sources"]["evaluate.py"],
        finish_reasons=dict(Counter(r["finish_reason"] for r in done)), budget_stopped=len(rows)-len(done),
        seconds=elapsed, seconds_kind="PROCESS_LAUNCH_TO_FAILED_SUMMARY_MTIME_OBSERVATION",
        generation_seconds=sum(r["seconds"] for r in done), peak_allocated_bytes=None, peak_reserved_bytes=None,
        parser="team-final-answer-v8", historical_62_percent_comparable=False,
        training_allowed=False, public_upload_allowed=False)
    dump(output/"recovery_receipt.json", {"status":"CLOSED_F0_BUDGET_STOP_SUMMARY_RECOVERED_NOT_NEW_INFERENCE",
        "parent":str(parent), "parent_predictions_sha256":sha(parent/"predictions.jsonl"),
        "damaged_summary_sha256":sha(parent/"summary.json"), "launch_sha256":sha(launch_path),
        "model_receipt_sha256":sha(model_receipt_path), "terminal_log_sha256":sha(log_path),
        "actual_launch_evaluator_sha256":launch["sources"]["evaluate.py"],
        "finish_time_file_evaluator_sha256_not_execution_proof":prefix["evaluator_source_sha256"],
        "process_gone":True, "parent_unchanged":True, "unknown_peak_memory_not_fabricated":True})
    dump(output/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "cohort", "protocol", "launch-receipt", "model-receipt", "log", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--owned-pod-id", required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.parent, args.cohort, args.protocol, args.launch_receipt,
                             args.model_receipt, args.log, args.output, args.owned_pod_id)))


if __name__ == "__main__":
    main()
