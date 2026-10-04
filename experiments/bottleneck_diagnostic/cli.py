"""Private diagnosis CLI; never synchronizes or publishes the MMDL checkout."""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

from .ingest import ingest, read_rows, sha, verify_frozen, verify_model_snapshot, write_json, write_rows
from .input_builder import build_request, canonical_hash, validate_sample
from .metrics import all_subject_first_pass, summarize
from .oracle_validation import validate_oracle


def selected(data, suite):
    selection = json.loads((data / "sample_selection.json").read_text())
    if suite == "local":
        ids = ["validation_Public_Health_14", "validation_Music_3"]
    elif suite == "expansion-remainder":
        completed = set(selection["pilot_ids"]) | set(all_subject_first_pass(selection))
        ids = [sid for sid in selection["expansion_ids"] if sid not in completed]
    else:
        ids = all_subject_first_pass(selection) if suite == "coverage" else selection[suite + "_ids"]
    samples = {s["sample_id"]: s for s in read_rows(data / "samples.jsonl")}
    return [samples[sid] for sid in ids]


def dry_run(data, suite, output, oracles=None):
    samples = selected(data, suite)
    oracle_map = {(o["sample_id"], o["level"]): o for o in read_rows(oracles)} if oracles else {}
    requests = []
    roles = {r["sample_id"]: r for r in read_rows(data / "input_audit.jsonl")}
    for sample in samples:
        validate_sample(sample)
        for mode in ("I", "T0", "P", "LEGACY"):
            if mode == "LEGACY" and roles[sample["sample_id"]]["excluded_images"]:
                requests.append({"sample_id": sample["sample_id"], "mode": mode,
                                 "status": "BLOCKED", "reason": "LEGACY_IMAGE_ROLES_DIFFER_FROM_ORIGINAL"})
                continue
            requests.append(build_request(sample, mode, 3407))
        for mode in ("O0", "O1", "IO0"):
            level = "O1" if mode == "O1" else "O0"
            oracle = oracle_map.get((sample["sample_id"], level))
            verdict = validate_oracle(oracle, sample, level, "ALLOW_AI_REVIEWED_DIAGNOSTIC")
            if verdict["status"] == "VALID":
                requests.append(build_request(sample, mode, 3407, oracle=oracle,
                                              oracle_review_policy="ALLOW_AI_REVIEWED_DIAGNOSTIC"))
            else:
                requests.append({"sample_id": sample["sample_id"], "mode": mode, **verdict})
        for mode in ("M", "MI"):
            requests.append({"sample_id": sample["sample_id"], "mode": mode,
                             "status": "BLOCKED", "reason": "DRY_RUN_HAS_NO_ACTUAL_P_RESPONSE"})
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "requests.jsonl", requests)
    receipt = {"status": "CPU_DRY_RUN_COMPLETED_NOT_GPU_SMOKE", "suite": suite,
               "questions": len(samples), "subjects": dict(Counter(s["subject"] for s in samples)),
               "request_hash": canonical_hash(requests), "actual_inference_calls": 0,
               "provenance": "BENCHMARK_DERIVED", "training_allowed": False}
    write_json(output / "receipt.json", receipt)
    return receipt


def validated_run(output, samples):
    """Read-only ledger validation plus explicit absent cells; never invent completions."""
    identity = json.loads((output / "protocol.json").read_text())
    if canonical_hash(identity["protocol"]) != identity["protocol_hash"]:
        raise ValueError("Frozen inference protocol hash differs")
    if canonical_hash(samples) != identity["sample_manifest_hash"]:
        raise ValueError("Scoring sample manifest differs from frozen inference")
    responses = read_rows(output / "responses.jsonl") if (output / "responses.jsonl").exists() else []
    expected = {(s["sample_id"], m, rep) for s in samples
                for m in identity["protocol"]["modes"] for rep in identity["repetitions"]}
    latest, completed = {}, set()
    for row in responses:
        key = row["sample_id"], row["mode"], row["repetition"]
        if (key not in expected or key in completed
                or row.get("record_hash") != canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
                or row["protocol_hash"] != identity["protocol_hash"]
                or row["run_kind"] != identity["execution_kind"]):
            raise ValueError("Scoring ledger key/record/protocol/execution integrity failed")
        if row["runtime_status"] == "COMPLETED":
            if row["response_hash"] != canonical_hash(row["raw_response"]):
                raise ValueError("Scoring raw response hash differs")
            completed.add(key)
        latest[key] = row
    missing = sorted(expected - latest.keys())
    for sid, mode, rep in missing:
        latest[sid, mode, rep] = {"sample_id": sid, "mode": mode, "repetition": rep,
                                 "runtime_status": "PENDING", "run_kind": "NOT_EXECUTED"}
    return identity, responses, list(latest.values()), missing


def audit_processor(data, model_path, suite, output):
    import yaml
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor
    from mmdl.evaluation.backends.input_preparation import prepare_protocol_inputs
    from .ingest import MODEL_REVISION

    cfg = yaml.safe_load(Path("configs/eval/mmmu_val_v8.yaml").read_text())
    processor = AutoProcessor.from_pretrained(model_path, revision=MODEL_REVISION, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": cfg["image"]["min_pixels"], "longest_edge": cfg["image"]["max_pixels"]}
    model = AutoConfig.from_pretrained(model_path, revision=MODEL_REVISION, local_files_only=True)
    rows = []
    for sample in selected(data, suite):
        for mode in ("I", "T0", "P"):
            request = build_request(sample, mode, 3407)
            images = []
            for metadata in request["images"]:
                with Image.open(data / metadata["path"]) as original:
                    images.append(original.convert("RGB"))
            messages = request["messages"]
            blocks = [b for b in messages[0]["content"] if b["type"] == "image"]
            for block, image in zip(blocks, images, strict=True):
                block["image"] = image
            if images:
                item, _ = prepare_protocol_inputs(processor, model, messages, images, cfg["image"])
            else:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
                item = {"prompt_token_ids": ids, "input_tokens": len(ids), "image_grid_thw": [],
                        "processed_image_sizes": [], "input_sha256": canonical_hash(ids)}
            image_tokens = sum(token == model.image_token_id for token in item["prompt_token_ids"])
            if item["input_tokens"] + request["max_new_tokens"] > cfg["execution"]["max_model_len"]:
                raise ValueError("CPU audit found context overflow; no automatic truncation")
            rows.append({"sample_id": sample["sample_id"], "mode": mode,
                         "original_image_sizes": [m["original_size"] for m in request["images"]],
                         "processed_image_sizes": item.get("processed_image_sizes"),
                         "image_grid_thw": item["image_grid_thw"], "image_tokens": image_tokens,
                         "total_input_tokens": item["input_tokens"], "input_sha256": item["input_sha256"],
                         "model_revision": MODEL_REVISION, "scope": "Actual pinned CPU processor, NOT vLLM internal pixel tensor certification",
                         "provenance": "BENCHMARK_DERIVED", "training_allowed": False})
    write_rows(output, rows)
    return {"status": "ACTUAL_CPU_PROCESSOR_AUDITED", "rows": len(rows), "image_tokens": sum(r["image_tokens"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "dry-run", "processor-audit", "inventory", "run", "local-smoke", "score", "verify"])
    parser.add_argument("--data", type=Path, default=Path("analysis_exports/bottleneck_20261003/prepared"))
    parser.add_argument("--source", type=Path, default=Path("analysis_exports/mmmu_20260924/team_final_v8/official_b2"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--model-manifest", type=Path, default=Path("manifests/models/baseline.json"))
    parser.add_argument("--suite", choices=["local", "stress", "pilot", "coverage", "expansion", "expansion-remainder"], default="stress")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score-output", type=Path, help="New score artifact directory when a run was previously scored")
    parser.add_argument("--oracles", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--hardware", type=Path, default=Path("configs/hardware/rtx3090_24gb.yaml"))
    parser.add_argument("--budget-seconds", type=float, default=14400)
    args = parser.parse_args()
    if args.command == "prepare":
        result = ingest(args.source, args.data_root, args.data, args.model_manifest, "MMDL")
    elif args.command == "verify":
        result = verify_frozen(args.data)
    else:
        if args.output is None:
            parser.error("--output required")
        if args.command == "dry-run":
            result = dry_run(args.data, args.suite, args.output, args.oracles)
        elif args.command == "processor-audit":
            result = audit_processor(args.data, args.model_path, args.suite, args.output)
        elif args.command == "inventory":
            from .inventory import module_inventory
            result = module_inventory(args.model_path, args.model_manifest, args.output)
        elif args.command in {"run", "local-smoke"}:
            import yaml
            from .runner import BudgetStop, make_vllm_backend, remaining_window, run_diagnostics
            if args.protocol is None or args.model_path is None:
                parser.error("run requires --protocol and --model-path")
            protocol = json.loads(args.protocol.read_text())
            if remaining_window(protocol) <= 0:
                raise BudgetStop("OVERALL_OPERATIONAL_WINDOW_EXPIRED_BEFORE_MODEL_LOAD")
            cfg = yaml.safe_load(Path("configs/eval/mmmu_val_v8.yaml").read_text())
            hw = yaml.safe_load(args.hardware.read_text())
            protocol["evaluation_config"] = cfg
            verified_model = verify_model_snapshot(args.model_path, args.model_manifest)
            if remaining_window(protocol) <= 0:
                raise BudgetStop("OVERALL_OPERATIONAL_WINDOW_EXPIRED_AFTER_MODEL_BYTE_VERIFICATION")
            if args.command == "local-smoke":
                from .local_backend import make_local_backend
                backend = make_local_backend(args.model_path, cfg, hw)
            else:
                backend = make_vllm_backend(args.model_path, cfg, hw)
            backend.model_files_manifest_hash = verified_model
            records = run_diagnostics(args.data, args.output, protocol, selected(args.data, args.suite),
                                      read_rows(args.oracles) if args.oracles else [], backend, args.budget_seconds)
            result = {"status_counts": dict(Counter(r["runtime_status"] for r in records)), "requests": len(records)}
        else:
            from .scoring import SCORING_PROTOCOL, score_responses
            from mmdl.evaluation.parsers import scoring_source_files
            samples = selected(args.data, args.suite)
            identity, responses, latest, missing = validated_run(args.output, samples)
            prepared_receipt = json.loads((args.data / "receipt.json").read_text())
            if sha(args.data / "gold.jsonl") != prepared_receipt["files"]["gold.jsonl"]:
                raise ValueError("Gold ledger changed after pinned data audit")
            gold = {r["sample_id"]: ast.literal_eval(r["answer"]) if isinstance(r["answer"], str) and r["answer"].startswith("[") else r["answer"] for r in read_rows(args.data / "gold.jsonl")}
            oracles = {(r["sample_id"], r["level"]): r for r in read_rows(args.oracles)} if args.oracles else {}
            scores = score_responses(latest, samples, gold, oracles)
            score_output = args.score_output or args.output
            score_output.mkdir(parents=True, exist_ok=True)
            write_rows(score_output / "scores.jsonl", scores)
            result = summarize(scores, samples, latest)
            result.update(status="INCOMPLETE" if missing or any(r["runtime_status"] in {"SYSTEM_ERROR", "BUDGET_STOP"} for r in latest) else "SCORED_ALL_DECLARED_CELLS",
                          missing_cells=missing, expected_cells=len(latest),
                          scoring_identity={"inference_protocol_hash": identity["protocol_hash"],
                                            "diagnostic_scoring_protocol": SCORING_PROTOCOL,
                                            "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[2])): sha(path)
                                                              for path in [Path(__file__), Path(__file__).with_name("scoring.py"),
                                                                           Path(__file__).with_name("metrics.py"), *scoring_source_files("team-final-answer-v8")]},
                                            "gold_sha256": sha(args.data / "gold.jsonl"),
                                            "post_run_oracles_sha256": sha(args.oracles) if args.oracles else None})
            # Charge every observed attempt, not only its latest completion.
            result["cost"] = summarize(scores, samples, responses)["cost"]
            write_json(score_output / "summary.json", result)
            result = {"actual_inference_cells": result["actual_inference_cells"],
                      "runtime_statuses": result["runtime_statuses"], "groups": len(result["groups"]),
                      "summary_path": str(score_output / "summary.json")}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
