"""Private K1/K2 capability probes using the existing, frozen Qwen diagnostic runner."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

from experiments.bottleneck_diagnostic.cli import validated_run
from experiments.bottleneck_diagnostic.ingest import (
    sha, verify_model_snapshot, write_json, write_rows,
)
from experiments.bottleneck_diagnostic.input_builder import build_request, canonical_hash
from experiments.bottleneck_diagnostic.scoring import score_responses


def samples_from_bank(path: Path) -> tuple[dict, list[dict]]:
    """Reject answer-side fields rather than trusting a scrubbed model payload."""
    bank = json.loads(path.read_text())
    rows = bank["items"]
    allowed = {"id", "subject", "stage", "pair_id", "skill", "question", "options"}
    samples = []
    for row in rows:
        if set(row) != allowed or row["stage"] not in {"K1", "K2"}:
            raise ValueError("Knowledge input has unknown/gold fields or invalid stage")
        if (len(row["options"]) != 4 or len(set(row["options"])) != 4
                or any(not isinstance(v, str) or not v.strip() for v in row.values()
                       if not isinstance(v, list))):
            raise ValueError("Each probe needs text fields and four distinct choices")
        sample = {"sample_id": row["id"], "subject": row["subject"],
                  "question_type": "multiple-choice", "question": row["question"],
                  "options": row["options"], "hint": None, "images": [],
                  "family_id": row["pair_id"], "skill_tags": ["knowledge_probe", row["stage"], row["skill"]],
                  "provenance": "BENCHMARK_DERIVED", "training_allowed": False}
        build_request(sample, "T0", 3407)  # Shared gold-blind contract and final format.
        samples.append(sample)
    counts = Counter((r["subject"], r["stage"]) for r in rows)
    pairs = Counter((r["pair_id"], r["stage"]) for r in rows)
    pair_ids = {r["pair_id"] for r in rows}
    if (len(samples) != 60 or len({s["sample_id"] for s in samples}) != 60
            or len(counts) != 30 or set(counts.values()) != {2}
            or len(pairs) != 60 or set(pairs.values()) != {1} or len(pair_ids) != 30
            or any(len({(r["subject"], r["skill"]) for r in rows if r["pair_id"] == pair}) != 1
                   for pair in pair_ids)):
        raise ValueError("Freeze 15 subjects x two K1/K2 pairs before observing outputs")
    if any(len({r["skill"] for r in rows if r["subject"] == subject}) != 2
           for subject, _ in counts):
        raise ValueError("Each subject needs two distinct preregistered skills")
    return bank, samples


def protocol(path: Path) -> dict:
    return {"protocol_id": "qwen-knowledge-rule-probe-v1", "modes": ["T0"],
            "stage_labels": ["K1", "K2"], "source_sha256": sha(path),
            "wrapper_sha256": sha(Path(__file__)), "final_output_limit": 4096,
            "first_run_seed": 3407, "repeat_seeds": [3407],
            "total_output_token_budget": 245760,
            "overall_deadline_utc": "2026-10-03T03:52:38Z",
            "scope": "Authored knowledge/rule probes, NOT matched MMMU causal interventions or population accuracy",
            "provenance": "BENCHMARK_DERIVED", "training_allowed": False}


def dry_run(cases: Path, output: Path) -> dict:
    bank, samples = samples_from_bank(cases)
    output.mkdir(parents=True, exist_ok=False)
    requests = [build_request(s, "T0", 3407) for s in samples]
    write_rows(output / "requests.jsonl", requests)
    result = {"status": "CPU_DRY_RUN_NOT_INFERENCE", "count": len(samples),
              "subjects": dict(Counter(s["subject"] for s in samples)),
              "requests_hash": canonical_hash(requests), "protocol": protocol(cases),
              "actual_inference_calls": 0, "images": 0, "bank_metadata": {k: v for k, v in bank.items() if k != "items"}}
    write_json(output / "receipt.json", result)
    return result


def score(cases: Path, answers: Path, output: Path, destination: Path) -> dict:
    _, samples = samples_from_bank(cases)
    identity, history, latest, missing = validated_run(output, samples)
    frozen = identity["protocol"]
    expected = protocol(cases)
    if (identity["repetitions"] != [3407]
            or any(frozen.get(k) != v for k, v in expected.items())):
        raise ValueError("Probe bank differs from actual frozen run")
    key = json.loads(answers.read_text())
    gold = key["answers"]
    if set(gold) != {s["sample_id"] for s in samples}:
        raise ValueError("Answer key must cover exactly the predeclared probes")
    labels = {sid: row["answer"] for sid, row in gold.items()}
    if set(labels.values()) - {"A", "B", "C", "D"}:
        raise ValueError("Answer key contains an invalid choice")
    scored = score_responses(latest, samples, labels)
    stages = {s["sample_id"]: s["skill_tags"][1] for s in samples}
    subjects = {s["sample_id"]: s["subject"] for s in samples}
    groups: dict[str, dict[str, Any]] = {}
    for subject in ["ALL", *sorted(set(subjects.values()))]:
        groups[subject] = {}
        for stage in ("K1", "K2"):
            rows = [r for r in scored if stages[r["sample_id"]] == stage
                    and (subject == "ALL" or subjects[r["sample_id"]] == subject)]
            completed = [r for r in rows if r["runtime_status"] == "COMPLETED"]
            groups[subject][stage] = {"declared": len(rows), "completed": len(completed),
                "correct": sum(r["diagnostic_score"] is True for r in completed),
                "accuracy": sum(r["diagnostic_score"] is True for r in completed) / len(rows) if len(completed) == len(rows) else None,
                "no_parse": sum(r.get("diagnostic_parse_status") != "PARSED" for r in completed),
                "limit_terminated": sum("OUTPUT_LIMIT_REACHED" in r.get("audit_flags", []) for r in completed)}
    actual = [r for r in history if r["runtime_status"] == "COMPLETED"]
    complete = len(actual) == 60 and not missing
    result = {"status": "TEST_ONLY_NOT_ACTUAL" if identity["execution_kind"] != "ACTUAL_INFERENCE"
              else "COMPLETED" if complete else "INCOMPLETE",
        "execution_kind": identity["execution_kind"], "groups": groups, "missing": missing,
        "protocol_hash": identity["protocol_hash"], "answer_key_sha256": sha(answers),
        "response_sha256": sha(output / "responses.jsonl"),
        "cost": {"completed_calls": len(actual), "input_tokens": sum(r["input_tokens"] for r in actual),
                 "output_tokens": sum(r["output_tokens"] for r in actual),
                 "request_seconds": sum(r["elapsed_seconds"] for r in actual)},
        "warning": "Two questions per stage/subject; simpler authored probes, not matched K3, not MMMU accuracy or module attribution",
        "provenance": "BENCHMARK_DERIVED", "training_allowed": False}
    destination.mkdir(parents=True, exist_ok=False)
    write_rows(destination / "scores.jsonl", scored)
    write_json(destination / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["dry-run", "run", "managed-run", "score"])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--model-manifest", type=Path, default=Path("manifests/models/baseline.json"))
    parser.add_argument("--budget-seconds", type=float, default=3600)
    parser.add_argument("--controller", type=Path)
    args = parser.parse_args()
    if args.command == "dry-run":
        result = dry_run(args.cases, args.output)
    elif args.command == "managed-run":
        if (args.controller is None or args.model_path is None
                or sha(args.controller) != "fc2b0bb73867f9ed32bf8c71fcf3b308b1086c4c39500aaaca2a2d7e4b22c86f"
                or os.environ.get("RUNPOD_POD_ID") != "h447a4u6c0vq3k"):
            parser.error("Exact owned Pod and verified existing cleanup controller required")
        spec = importlib.util.spec_from_file_location("knowledge_owned_controller", args.controller)
        if spec is None or spec.loader is None:
            raise ValueError("Existing controller is unavailable")
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        controller.gpu_idle()
        args.output.mkdir(parents=True, exist_ok=False)
        dry_run(args.cases, args.output / "dry-run")
        child = [sys.executable, str(Path(__file__)), "run", "--cases", str(args.cases),
                 "--output", str(args.output / "run"), "--model-path", args.model_path,
                 "--model-manifest", str(args.model_manifest), "--budget-seconds", str(args.budget_seconds)]
        end = min(controller.cutoff() - 1800, time.time() + args.budget_seconds + 120)
        result = {"status": "STARTED", "cases_sha256": sha(args.cases),
                  "wrapper_sha256": sha(Path(__file__)), "controller_sha256": sha(args.controller)}
        def interrupted(signum, _frame):
            raise TimeoutError(f"Managed probe interrupted by signal {signum}")
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in handlers:
            signal.signal(sig, interrupted)
        try:
            result["returncode"] = controller.run(child, args.output, "knowledge", dict(os.environ), end)
            result["status"] = "PROCESS_EXITED_NOT_SCORE_CERTIFICATION"
        except Exception as exc:
            result.update(status="BLOCKED_OR_FAILED_PRESERVED", error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                controller.gpu_idle()
            except Exception as exc:
                result.update(status="CLEANUP_NOT_VERIFIED", cleanup_error=f"{type(exc).__name__}: {exc}")
            write_json(args.output / "controller_summary.json", result)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    elif args.command == "score":
        if args.answers is None or args.score_output is None:
            parser.error("score requires separate --answers and fresh --score-output")
        result = score(args.cases, args.answers, args.output, args.score_output)
    else:
        import yaml
        from experiments.bottleneck_diagnostic.runner import make_vllm_backend, remaining_window, run_diagnostics
        if (args.model_path is None or remaining_window(protocol(args.cases)) <= 0
                or not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 3600):
            parser.error("Verified model path and unexpired execution window required")
        _, samples = samples_from_bank(args.cases)
        frozen = protocol(args.cases)
        cfg = yaml.safe_load(Path("configs/eval/mmmu_val_v8.yaml").read_text())
        hw = yaml.safe_load(Path("configs/hardware/rtx3090_24gb.yaml").read_text())
        verified = verify_model_snapshot(args.model_path, args.model_manifest)
        if remaining_window(frozen) <= 0:
            raise TimeoutError("Execution window expired after actual model-byte verification")
        frozen["evaluation_config"] = cfg
        backend = make_vllm_backend(args.model_path, cfg, hw)
        backend.model_files_manifest_hash = verified
        rows = run_diagnostics(args.cases.parent, args.output, frozen, samples, [], backend, args.budget_seconds)
        result = {"statuses": dict(Counter(r["runtime_status"] for r in rows)), "count": len(rows)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
