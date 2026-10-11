"""Clean-clone 8K inference and separate CPU scoring of immutable completions."""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from experiments.bottleneck_diagnostic.metrics import CLUSTERS, cluster, transition
from experiments.bottleneck_diagnostic.runner import _loaded_request, _verify_images
from fine_tuning.runpod_g6 import checkout
from mmdl.evaluation.parsers import score_with_parser, scoring_source_files
from mmdl.runtime.artifacts import atomic_text, digest, read_json, sha256_file, write_json
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION

from .inputs import build_request

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.yaml")
RULES = Path(__file__).with_name("rules.yaml")
ARMS = ("F0", "FL", "FM", "FV", "FML")
MODES = ("I", "T", "IT", "IR", "TR")
PAIRS = (("I", "T"), ("I", "IT"), ("T", "IT"), ("I", "IR"), ("T", "TR"))


def rows(path):
    values = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("JSONL object rows required")
    return values


def unique(values, key="sample_id"):
    indexed = {value[key]: value for value in values}
    if len(indexed) != len(values):
        raise ValueError("Duplicate input/response identity")
    return indexed


def validate_completion(row, expected):
    if (row.get("record_hash") != digest({key: value for key, value in row.items() if key != "record_hash"})
            or row.get("runtime_status") != "COMPLETED" or row.get("request") != expected
            or row.get("request_hash") != expected["request_hash"]
            or row.get("sample_id") != expected["sample_id"]
            or row.get("original_sample_id") != expected["original_sample_id"]
            or row.get("mode") != expected["mode"] or row.get("repetition") != 3407):
        raise ValueError("Invalid completed or altered inference artifact")


def model_identity(model_path, arm):
    if arm == "F0":
        manifest = read_json(ROOT / "manifests/models/baseline.json")
        if manifest["repo_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
            raise ValueError("Pinned model identity mismatch")
        files = {row["path"]: row["sha256"] for row in manifest["files"]}
    else:
        manifest = read_json(model_path / "export_manifest.json")
        if (manifest.get("status") != "EXPORTED" or manifest.get("arm") != arm
                or manifest.get("base_id") != MODEL_ID
                or manifest.get("base_revision") != MODEL_REVISION
                or manifest.get("format") != "derived-hf-bf16-v1"):
            raise ValueError("A complete hash-bound derived checkpoint is required")
        files = manifest["files"]
    for name, expected in files.items():
        path = (model_path / name).resolve(strict=True)
        if (not path.is_relative_to(model_path.resolve()) and arm != "F0"):
            raise ValueError("Derived model file escapes export directory")
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256_file(path) != expected:
            raise ValueError("Model file/hash mismatch")
    return {"arm": arm, "manifest_sha256": digest(manifest), "files": files,
            "derived_precision_policy": manifest.get("precision_policy")}


def make_plan(data_root, arm, phase):
    if arm not in ARMS or phase not in ("smoke", "full"):
        raise ValueError("Unknown arm or phase")
    samples = unique(rows(data_root / "samples.jsonl"))
    oracles = unique(rows(data_root / "oracles.jsonl"))
    selection = read_json(data_root / "selection.json")
    if digest({key: value for key, value in selection.items() if key != "selection_hash"}) != selection["selection_hash"]:
        raise ValueError("Selection manifest hash mismatch")
    if (len(selection["selected_ids"]) != 48 or set(selection["selected_ids"]) != set(samples)
            or not set(selection["subset_ids"]) <= set(samples)
            or len(set(selection["subset_ids"])) != 12
            or set(oracles) != set(samples)):
        raise ValueError("Sample/oracle/selection coverage mismatch")
    for sid, sample in samples.items():
        if (digest(sample) != selection["sample_hashes"][sid]
                or oracles[sid]["oracle_hash"] != selection["oracle_hashes"][sid]):
            raise ValueError("Prepared sample/oracle differs from frozen selection")
    strata = unique(selection["strata"])
    if set(strata) != set(samples):
        raise ValueError("Selection strata coverage mismatch")
    for group in CLUSTERS:
        members = [row for sid, row in strata.items() if cluster(samples[sid]["subject"]) == group]
        subjects = Counter(samples[row["sample_id"]]["subject"] for row in members)
        if (len(members) != 12 or Counter(row["historical_correct"] for row in members) != {True: 6, False: 6}
                or len(subjects) < 3 or max(subjects.values()) > 4):
            raise ValueError("Diagnostic strata changed from frozen 12/cluster, 6/6 and subject caps")
    if len(samples) != 48 or len(selection["subset_ids"]) != 12:
        raise ValueError("Frozen 48-question/12-checkpoint-subset selection required")
    ids = selection["subset_ids"] if arm != "F0" or phase == "smoke" else selection["selected_ids"]
    modes = MODES if arm == "F0" else ("I", "T")
    rules = yaml.safe_load(RULES.read_text())
    requests = []
    for sample_id in ids:
        sample = samples[sample_id]
        for mode in modes:
            request = build_request(sample, mode,
                                    oracles[sample_id] if mode in ("T", "IT", "TR") else None,
                                    rules[sample["subject"]] if mode in ("IR", "TR") else None)
            request["original_sample_id"] = sample_id
            request["sample_id"] = f"{arm}__{sample_id}__{mode}"
            request["request_hash"] = digest(request)
            requests.append(request)
    return requests, samples


def run_inference(data_root, model_path, base_path, output, arm, phase, slots, deadline_utc,
                  commit, backend_factory=None):
    if type(slots) is not int or slots not in (2, 4, 8):
        raise ValueError("Explicit diagnostic concurrency must be 2, 4 or 8")
    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    if deadline.utcoffset() is None or datetime.now(timezone.utc) >= deadline:
        raise ValueError("Unexpired timezone-aware deadline required")
    checkout(ROOT, commit)
    protected = ROOT / "MMDL"
    for path in (output, data_root):
        if path.resolve().is_relative_to(protected):
            raise ValueError("Professor submission checkout is protected")
    requests, samples = make_plan(data_root, arm, phase)
    _verify_images(data_root.resolve(), list(samples.values()))
    config = yaml.safe_load(PROTOCOL.read_text())
    config["execution"]["batch_size"] = slots
    identity = {"protocol": config, "model": model_identity(model_path, arm),
                "processor_base": model_identity(base_path, "F0"),
                "code_commit": commit, "deadline_utc": deadline_utc,
                "input_files": {name: sha256_file(data_root / name)
                                for name in ("samples.jsonl", "oracles.jsonl", "selection.json")},
                "rules_sha256": sha256_file(RULES),
                "run_kind": "ACTUAL_INFERENCE" if backend_factory is None else "TEST_ONLY",
                "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
                "public_upload_allowed": False}
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != identity:
                raise ValueError("Resume changed model/protocol/inputs/rules/seed/deadline")
        else:
            if any(path.name != ".lock" for path in output.iterdir()):
                raise ValueError("Nonempty unbound output directory")
            write_json(manifest_path, identity)
        response_root = output / "responses"
        response_root.mkdir(exist_ok=True)
        done = {}
        for path in response_root.glob("*.json"):
            row = read_json(path)
            if (path.stem != row["request_hash"] or row["runtime_status"] != "COMPLETED"
                    or row.get("record_hash") != digest({key: value for key, value in row.items()
                                                        if key != "record_hash"})):
                raise ValueError("Invalid completed response artifact")
            if row["sample_id"] in done:
                raise ValueError("Duplicate durable completion")
            done[row["sample_id"]] = row
        requested = unique(requests)
        full, _ = make_plan(data_root, arm, "full")
        full_map = unique(full)
        if set(done) - set(requested):
            # A smoke is a strict prefix/subset of its full continuation, never the reverse.
            if set(done) - set(full_map):
                raise ValueError("Output contains unknown request identities")
        for sid, row in done.items():
            expected = full_map.get(sid)
            if not expected or row["request_hash"] != expected["request_hash"]:
                raise ValueError("Completed request changed under resume")
            validate_completion(row, expected)
        pending = [request for request in requests if request["sample_id"] not in done]
        start = time.monotonic()
        def status(state):
            complete = sum(request["sample_id"] in done for request in requests)
            write_json(output / "status.json", {"status": state, "arm": arm, "phase": phase,
                       "expected": len(requests), "completed": complete,
                       "missing": len(requests) - complete, "max_active_requests": slots,
                       "wall_seconds_this_process": time.monotonic() - start,
                       "gpu_deadline_utc": deadline_utc})
        status("RUNNING")
        if not pending:
            status("COMPLETED")
            return
        if backend_factory is None:
            from .backend import create_backend
            backend_factory = create_backend
        backend = backend_factory(model_path, base_path, arm, config, deadline_utc)
        write_json(output / f"backend_{phase}.json", backend.describe())
        def loaded():
            for request in pending:
                yield _loaded_request(request, data_root.resolve())
        stream = backend.generate_stream(loaded())
        try:
            for raw in stream:
                sid = raw["sample_id"]
                request = requested[sid]
                if sid in done or not isinstance(raw.get("raw_response"), str):
                    raise ValueError("Duplicate or invalid backend completion")
                row = raw | {"original_sample_id": request["original_sample_id"],
                             "arm": arm, "mode": request["mode"], "repetition": 3407,
                             "request_hash": request["request_hash"],
                             "request": request, "runtime_status": "COMPLETED",
                             "run_kind": identity["run_kind"], "public_upload_allowed": False,
                             "provenance": "BENCHMARK_DERIVED", "training_allowed": False}
                row["record_hash"] = digest(row)
                write_json(response_root / (request["request_hash"] + ".json"), row)
                done[sid] = row
                status("RUNNING")
                print(json.dumps({"arm": arm, "completed": len(done), "request": sid,
                                  "tokens": raw.get("generated_tokens"),
                                  "finish_reason": raw.get("finish_reason")}), flush=True)
        except BaseException:
            status("INCOMPLETE")
            raise
        finally:
            stream.close()
        status("COMPLETED" if all(request["sample_id"] in done for request in requests)
               else "INCOMPLETE")


def score(data_root, run_roots, output):
    if output.resolve().is_relative_to(ROOT / "MMDL"):
        raise ValueError("Professor submission checkout is protected")
    if output.exists():
        raise FileExistsError("Scoring export must be new; inference remains immutable")
    samples = unique(rows(data_root / "samples.jsonl"))
    gold = unique(rows(data_root / "gold.jsonl"))
    selection = read_json(data_root / "selection.json")
    if (set(gold) != set(samples) or selection.get("gold_hashes") != {
            sid: digest(value) for sid, value in gold.items()}):
        raise ValueError("Selected gold differs from its immutable preparation binding")
    scored, sources = [], {}
    planned = [request for arm in ARMS for request in make_plan(data_root, arm, "full")[0]]
    seen_arms = set()
    for run_root in run_roots:
        manifest = read_json(run_root / "manifest.json")
        arm = manifest["model"]["arm"]
        if arm in seen_arms:
            raise ValueError("Duplicate arm run roots")
        seen_arms.add(arm)
        config = yaml.safe_load(PROTOCOL.read_text())
        config["execution"]["batch_size"] = manifest["protocol"]["execution"]["batch_size"]
        if (manifest["protocol"] != config
                or config["execution"]["batch_size"] not in (2, 4, 8)):
            raise ValueError("Scoring requires the frozen 8K diagnostic protocol")
        requests, _ = make_plan(data_root, arm, "full")
        plan = unique(requests)
        if (manifest["input_files"] != {name: sha256_file(data_root / name)
                                       for name in ("samples.jsonl", "oracles.jsonl", "selection.json")}
                or manifest["rules_sha256"] != sha256_file(RULES)):
            raise ValueError("Scoring inputs/rules differ from inference manifest")
        sources[str(run_root)] = sha256_file(run_root / "manifest.json")
        for path in sorted((run_root / "responses").glob("*.json")):
            raw = read_json(path)
            if (raw.get("record_hash") != digest({key: value for key, value in raw.items()
                                                  if key != "record_hash"})
                    or raw["sample_id"] not in plan
                    or raw["request_hash"] != plan[raw["sample_id"]]["request_hash"]
                    or path.stem != raw["request_hash"] or raw["arm"] != arm
                    or raw["runtime_status"] != "COMPLETED"):
                raise ValueError("Unbound or altered inference completion")
            expected = plan[raw["sample_id"]]
            validate_completion(raw, expected)
            sid = expected["original_sample_id"]
            sample = samples[sid]
            verdict = score_with_parser(raw["raw_response"], sample["question_type"],
                       sample["options"], gold[sid]["answer"], "team-final-answer-v8",
                       raw["finish_reason"], question=sample["question"])
            scored.append({"sample_id": sid, "arm": raw["arm"], "mode": raw["mode"],
                           "repetition": 3407, "v8_score": verdict["correct"],
                           "diagnostic_score": verdict["correct"], **verdict,
                           "runtime_status": "COMPLETED", "finish_reason": raw["finish_reason"],
                           "run_kind": raw["run_kind"], "public_upload_allowed": False,
                           "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
                           "generated_tokens": raw["generated_tokens"],
                           "input_tokens": raw["input_tokens"],
                           "generation_seconds": raw.get("generation_seconds"),
                           "request_latency_seconds": raw.get("request_latency_seconds"),
                           "image_grid_thw": raw.get("image_grid_thw"),
                           "audit_status": "AI_REVIEWED_NOT_HUMAN_VERIFIED" if raw["mode"] != "I"
                                           else "NOT_SEMANTICALLY_AUDITED",
                           "response_path": str(path), "response_sha256": sha256_file(path)})
    keys = [(row["sample_id"], row["arm"], row["mode"], row["repetition"]) for row in scored]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate scored cells; do not combine smoke and full exports")
    unique(planned)
    completed_ids = {f"{row['arm']}__{row['sample_id']}__{row['mode']}" for row in scored}
    missing = [request["sample_id"] for request in planned if request["sample_id"] not in completed_ids]
    groups = {}
    for arm in ARMS:
        selected = [row for row in scored if row["arm"] == arm]
        for group in ("all_selected", *sorted({cluster(s["subject"]) for s in samples.values()})):
            subset = [row for row in selected if group == "all_selected"
                      or cluster(samples[row["sample_id"]]["subject"]) == group]
            expected_ids = [sid for sid in (read_json(data_root / "selection.json")["selected_ids"]
                            if arm == "F0" else read_json(data_root / "selection.json")["subset_ids"])
                            if group == "all_selected" or cluster(samples[sid]["subject"]) == group]
            groups[f"{arm}/{group}"] = {"conditions": {mode: {
                "status": "NOT_APPLICABLE" if arm != "F0" and mode not in ("I", "T")
                          else "COMPLETED" if len([row for row in subset if row["mode"] == mode]) == len(expected_ids)
                          else "INCOMPLETE",
                "planned_applicable": len(expected_ids) if arm == "F0" or mode in ("I", "T") else 0,
                "completed": len(cells := [row for row in subset if row["mode"] == mode]),
                "correct": sum(row["correct"] for row in cells),
                "accuracy": sum(row["correct"] for row in cells) / len(expected_ids)
                            if expected_ids and len(cells) == len(expected_ids) else None,
                "partial_completed_accuracy": sum(row["correct"] for row in cells) / len(cells) if cells else None,
                "finish_reasons": dict(Counter(row["finish_reason"] for row in cells)),
                "parse_statuses": dict(Counter(row["parse_status"] for row in cells))}
                for mode in MODES}, "transitions": [transition(subset, list(samples.values()), a, b)
                                                        for a, b in PAIRS]}
    output.mkdir(parents=True, exist_ok=False)
    atomic_text(output / "scores.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored))
    cross_arm = [row | {"mode": row["arm"] + "/" + row["mode"]} for row in scored]
    run_kinds = {row["run_kind"] for row in scored}
    write_json(output / "summary.json", {"scope": "48 stratified diagnostic items, not MMMU population accuracy",
               "status": "COMPLETED" if len(scored) == 336 and not missing else "INCOMPLETE",
               "run_kind": next(iter(run_kinds)) if len(run_kinds) == 1 else "MIXED_OR_EMPTY",
               "missing_request_ids": missing,
               "completed_cells": len(scored), "maximum_planned_cells": 336, "groups": groups,
               "checkpoint_transitions": [transition(cross_arm, list(samples.values()),
                                                       "F0/" + mode, arm + "/" + mode)
                                            for arm in ARMS[1:] for mode in ("I", "T")],
               "cost": {"input_tokens": sum(row["input_tokens"] for row in scored),
                        "output_tokens": sum(row["generated_tokens"] for row in scored),
                        "apportioned_generation_seconds": sum(row.get("generation_seconds") or 0 for row in scored)},
               "source_manifests": sources, "gold_sha256": sha256_file(data_root / "gold.jsonl"),
               "parser_sources": {str(path.relative_to(ROOT)): sha256_file(path)
                                  for path in scoring_source_files("team-final-answer-v8")},
               "module_attribution": "Information recovery does not uniquely identify an internal module",
               "actual_inference_cells": sum(row["run_kind"] == "ACTUAL_INFERENCE" for row in scored),
               "public_upload_allowed": False,
               "provenance": "BENCHMARK_DERIVED", "training_allowed": False})
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    infer = sub.add_parser("infer")
    for name in ("data-root", "model-path", "base-path", "output"):
        infer.add_argument("--" + name, type=Path, required=True)
    infer.add_argument("--arm", choices=ARMS, required=True)
    infer.add_argument("--phase", choices=("smoke", "full"), required=True)
    infer.add_argument("--slots", type=int, choices=(2, 4, 8), required=True)
    infer.add_argument("--deadline-utc", required=True)
    infer.add_argument("--commit", required=True)
    scorer = sub.add_parser("score")
    scorer.add_argument("--data-root", type=Path, required=True)
    scorer.add_argument("--runs", nargs="+", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "infer":
        if not re.fullmatch("[0-9a-f]{40}", args.commit):
            parser.error("Full commit SHA required")
        run_inference(args.data_root, args.model_path, args.base_path, args.output,
                      args.arm, args.phase, args.slots, args.deadline_utc, args.commit)
    else:
        score(args.data_root, args.runs, args.output)


if __name__ == "__main__":
    main()
