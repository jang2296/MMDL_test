"""Bounded fresh-F0 TabMWP A/B SFT and frozen external evaluation; no automatic expansion."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from experiments.input_recovery.runner import ROOT, model_identity
from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json

from .budget import effective_deadline, utc
from .run_experiment import Controller, exposure_matches
from .runpod_g6 import checkout
from .tabmwp_trial import contract, pair_check


def input_gate(args):
    """Require actual closed diagnosis, reviewed CPU data and the exact new execution authority."""
    from .tabmwp_evaluate import load_cohort
    from .train import approved_records

    setup = read_json(args.runtime_root / "setup/model.json")
    if (setup.get("status") != "MODEL_HASH_VERIFIED" or setup.get("code_commit") != args.commit
            or setup.get("code_repository") != "jang2296/MMDL_test"):
        raise ValueError("Same exact-commit model-ready setup required")
    model = Path(setup["model_path"]).resolve(strict=True)
    if not model.is_relative_to(args.runtime_root.resolve()):
        raise ValueError("Pinned base must belong to the verified runtime")
    model_identity(model, "F0")
    diagnosis = read_json(args.diagnostic_status)
    if (diagnosis.get("status") != "DIAGNOSTIC_INFERENCE_COMPLETED"
            or diagnosis.get("actual_cells") != 336 or diagnosis.get("training_executed") is not False
            or diagnosis.get("owned_pod_id") != args.owned_pod_id
            or re.fullmatch(r"[0-9a-f]{40}", str(diagnosis.get("code_commit", ""))) is None):
        raise ValueError("The actual fixed 336-cell diagnosis must close first")
    diagnostic_outputs = Path(diagnosis.get("outputs", "")).resolve(strict=True)
    if (not diagnostic_outputs.is_dir() or diagnostic_outputs.is_relative_to(ROOT)
            or not Path(diagnosis["outputs"]).is_absolute()):
        raise ValueError("Actual diagnostic outputs must exist outside the Git checkout")
    diagnostic_execution = {key: diagnosis[key] for key in ("code_commit", "owned_pod_id", "outputs")}
    trial = read_json(args.trial_root / "trial_manifest.json")
    receipt = read_json(args.approval_root / "cpu_approval_receipt.json")
    trial_sha = sha256_file(args.trial_root / "trial_manifest.json")
    receipt_sha = sha256_file(args.approval_root / "cpu_approval_receipt.json")
    if (trial.get("trial_hash") != digest({k: v for k, v in trial.items() if k != "trial_hash"})
            or trial.get("training_contract") != contract()
            or receipt.get("status") != "DATA_APPROVED" or receipt.get("training_allowed") is not True
            or receipt.get("training_execution_authorized") is not False
            or receipt.get("trial_manifest_sha256") != trial_sha):
        raise ValueError("Complete reviewed A/B CPU approval required")
    if any(sha256_file(Path(__file__).with_name(name)) != expected
           for name, expected in trial.get("source_sha256", {}).items()) or set(trial.get("source_sha256", {})) != {"train.py", "modules.py", "tabmwp_trial.py"}:
        raise ValueError("Trial producer and training sources differ")
    cohort, _samples = load_cohort(args.cohort_root)
    if any(args.cohort_root.rglob("gold.jsonl")):
        raise ValueError("External evaluation gold must remain off the inference Pod")
    manifests, variants = {}, {}
    for candidate in ("A", "B"):
        path = args.approval_root / candidate / "training_manifest.json"
        rows, manifest = approved_records(
            args.approval_root / candidate / "approved/records.jsonl", path, args.trial_root)
        if (len(rows) != 2000 or manifest.get("tabmwp_condition") != candidate
                or manifest.get("tabmwp_trial_hash") != trial["trial_hash"]
                or manifest.get("tabmwp_trial_manifest_sha256") != trial_sha
                or manifest.get("tabmwp_trial_receipt_sha256") != receipt_sha
                or manifest.get("tabmwp_preparation_hash") != cohort["tabmwp_preparation_hash"]
                or manifest.get("training_contract") != contract()
                or manifest.get("record_order_sha256") != trial["record_order_sha256"]
                or manifest.get("consumption_order_sha256") != trial["consumption_order_sha256"]):
            raise ValueError("Wrong condition, population or reviewed training contract")
        variants[candidate], manifests[candidate] = rows, manifest
    training_binding = cohort.get("approved_training_records", {})
    if training_binding.get("record_count") != 2000 or training_binding.get("sha256") != manifests["A"]["records_sha256"]:
        raise ValueError("External exclusion cohort must bind this exact approved A training population")
    comparable = {v: [dict(row, processor_receipt_sha256="VARIANT_SPECIFIC") for row in rows]
                  for v, rows in variants.items()}
    pair_check(comparable["A"], comparable["B"], trial["retention_ids"])
    bound = {"trial_manifest_sha256": trial_sha, "cpu_approval_receipt_sha256": receipt_sha,
             "diagnostic_status_sha256": sha256_file(args.diagnostic_status),
             "diagnostic_execution": diagnostic_execution,
             "external_manifest_sha256": sha256_file(args.cohort_root / "manifest.json"),
             "training_manifests_sha256": {v: sha256_file(args.approval_root / v / "training_manifest.json")
                                            for v in ("A", "B")}}
    authority = read_json(args.authorization)
    required = {"status": "APPROVED_FOR_BOUNDED_TABMWP_AB", "training_execution_authorized": True,
                "code_commit": args.commit, "owned_pod_id": args.owned_pod_id,
                "training_contract": contract(), "input_binding": bound,
                "teacher_generated_supervision_allowed": False, "automatic_expansion_allowed": False}
    if any(authority.get(key) != value for key, value in required.items()):
        raise ValueError("Exact new A/B execution authority missing; CPU approval is not training")
    return model, manifests, bound


def run(args, controller):
    model, manifests, bound = input_gate(args)
    write_json(controller.output / "input_binding.json", bound)
    files = [args.trial_root / "trial_manifest.json", args.approval_root / "cpu_approval_receipt.json",
             args.cohort_root / "manifest.json", args.cohort_root / "samples.jsonl",
             args.diagnostic_status, args.authorization]
    for candidate in ("A", "B"):
        files += [args.approval_root / candidate / "training_manifest.json",
                  args.approval_root / candidate / "approved/records.jsonl"]
    frozen_files = {path: sha256_file(path) for path in files}

    def stage(name, command, maximum_seconds=None):
        if any(sha256_file(path) != expected for path, expected in frozen_files.items()):
            raise ValueError("Frozen training/evaluation authority or inputs changed")
        controller.stage(name, command, maximum_seconds)

    def evaluate(candidate, candidate_model, checkpoint=None):
        command = [sys.executable, "-m", "fine_tuning.tabmwp_evaluate", "infer",
                   "--data-root", str(args.cohort_root), "--model-path", str(candidate_model),
                   "--base-path", str(model), "--output", str(controller.output / (candidate + "_external")),
                   "--candidate", candidate, "--slots", "4", "--deadline-utc", controller.deadline.isoformat(),
                   "--commit", args.commit]
        if checkpoint is not None:
            command += ["--checkpoint-metadata", str(checkpoint / "metadata.json"),
                        "--training-manifest", str(args.approval_root / candidate / "training_manifest.json"),
                        "--training-manifest-sha256", bound["training_manifests_sha256"][candidate]]
        stage(candidate + "_external", command)
        status = read_json(controller.output / (candidate + "_external/status.json"))
        if status.get("status") != "COMPLETED" or status.get("completed") != 132:
            raise ValueError("Actual complete external responses required")

    evaluate("F0", model)
    metadata_by_variant = {}
    pixels = contract()["input_contract"]
    for candidate in ("A", "B"):
        # Both invocations receive the pinned original base, never the other variant's checkpoint.
        output = controller.output / candidate
        command = [sys.executable, "-m", "fine_tuning.train", "--model-path", str(model),
                   "--records", str(args.approval_root / candidate / "approved/records.jsonl"),
                   "--manifest", str(args.approval_root / candidate / "training_manifest.json"),
                   "--image-root", str(args.trial_root), "--output", str(output), "--arm", "FM",
                   "--seed", "3407", "--accumulation", "16", "--sequence-budget", "4096",
                   "--min-pixels", str(pixels["min_pixels"]), "--max-pixels", str(pixels["max_pixels"]),
                   "--episode-pixels", str(pixels["episode_pixels"]),
                   "--budget-seconds", str(max(1, controller.remaining() - 120))]
        stage(candidate + "_train", command)
        checkpoint = output / "checkpoint"
        metadata = read_json(checkpoint / "metadata.json")
        if (not exposure_matches(metadata) or metadata.get("termination") != "EPOCH_COMPLETE"
                or metadata.get("planned_records") != 2000 or metadata.get("exposure", {}).get("updates") != 125
                or metadata.get("arm") != "FM" or metadata.get("seed") != 3407
                or metadata.get("epochs") != 1 or metadata.get("budgets", {}).get("accumulation") != 16
                or metadata.get("base_precision") != "BF16" or metadata.get("trainable_master_precision") != "FP32"
                or metadata.get("consumed_order_sha256") != manifests[candidate]["consumption_order_sha256"]
                or metadata.get("manifest_sha256") != bound["training_manifests_sha256"][candidate]
                or metadata.get("records_sha256") != manifests[candidate]["records_sha256"]):
            raise TimeoutError("A/B exposure incomplete; preserve checkpoint without scoring it as a finished trial")
        metadata_by_variant[candidate] = metadata
        stage(candidate + "_reload", [sys.executable, "-m", "fine_tuning.train",
                         "--model-path", str(model), "--verify-checkpoint", str(checkpoint)], 600)
        exported = controller.output / (candidate + "_derived")
        stage(candidate + "_export", [sys.executable, "-m", "experiments.input_recovery.checkpoints",
                         "--model-path", str(model), "--checkpoint", str(checkpoint), "--output", str(exported)])
        evaluate(candidate, exported, checkpoint)
    if metadata_by_variant["A"]["consumed_order_sha256"] != metadata_by_variant["B"]["consumed_order_sha256"]:
        raise ValueError("Actual A/B record consumption differs")
    return {"status": "AB_TRAINING_AND_EXTERNAL_INFERENCE_COMPLETED", "training_executed": True,
            "rows_per_variant": 2000, "updates_per_variant": 125, "external_actual_cells": 396,
            "scoring_status": "PENDING_LOCAL_GOLD_COMPARISON", "automatic_expansion_allowed": False,
            "target_token_exposure": {v: m["exposure"]["target_tokens"] for v, m in metadata_by_variant.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-root", "trial-root", "approval-root", "cohort-root", "diagnostic-status",
                 "authorization", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("commit", "owned-pod-id", "gpu-cutoff-utc", "billing-start-utc"):
        parser.add_argument("--" + name, required=True)
    for name in ("balance-usd", "reserve-usd", "hourly-usd"):
        parser.add_argument("--" + name, type=float, required=True)
    args = parser.parse_args()
    checkout(ROOT, args.commit)
    if os.environ.get("RUNPOD_POD_ID") != args.owned_pod_id:
        parser.error("Exact owned Pod identity required")
    for path in (args.runtime_root, args.trial_root, args.approval_root, args.cohort_root,
                 args.diagnostic_status, args.authorization, args.output):
        canonical = path.resolve()
        if not canonical.is_relative_to(Path("/workspace")) or canonical == Path("/workspace") or canonical.is_relative_to(ROOT):
            parser.error("Private artifacts require explicit persistent /workspace children outside Git")
    deadline = effective_deadline(utc(args.gpu_cutoff_utc), utc(args.billing_start_utc),
                                  args.balance_usd, args.reserve_usd, args.hourly_usd)
    controller = Controller(args.output, deadline, Path(__file__).parent)
    result = {"status": "INCOMPLETE"}
    try:
        result = run(args, controller)
    except TimeoutError as exc:
        result = {"status": "BUDGET_STOP", "reason": str(exc)}
    except BaseException as exc:
        result = {"status": "INCOMPLETE", "reason": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        result.update(code_commit=args.commit, owned_pod_id=args.owned_pod_id,
                      gpu_deadline_utc=deadline.isoformat(), stage_seconds=controller.stage_seconds)
        write_json(args.output / "status.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
