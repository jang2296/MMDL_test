"""Additional FL-only NVW 0/10 mixture pilot; never the original 10K or G7."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

from . import audit, prepare_controls, prepare_training, run_experiment as runner
from .modules import BASE_ID, BASE_REVISION
from .report import compare_inputs, external_groups, load_evaluation, paired_counts
from .train import approved_records, order_hash


def control_inputs(parent: Path, reference: Path, control: Path, smoke_root: Path, smoke_image_root: Path):
    """Reconstruct the approved substitution; copied receipts alone are insufficient."""
    rows, parent_manifest, _completed, parent_files, _ = prepare_training.closed_pilot(parent)
    reference_rows, reference_manifest = approved_records(
        reference/"approved/records.jsonl", reference/"common_manifest.json", reference)
    control_rows, manifest = approved_records(
        control/"approved/records.jsonl", control/"common_manifest.json", control)
    smoke_rows, smoke_manifest = approved_records(smoke_root/"approved/records.jsonl",
                                                  smoke_root/"smoke_manifest.json", smoke_image_root)
    receipt_path = control/"control_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    subset = json.loads((reference/"subset_receipt.json").read_text())
    reference_files = {name: audit.sha(audit.image_path(reference, name)) for name in
                       ("approved/records.jsonl", "common_manifest.json", "subset_receipt.json", "smoke_exclusion/records.jsonl")}
    if (len(control_rows) != 1600 or len(reference_rows) != 1600
            or len(smoke_rows) != 360 or len({r["root_id"] for r in smoke_rows}) != 120
            or receipt.get("status") != "DATA_PREPARATION_COMPLETED"
            or receipt.get("kind") != "TRACE_NVW_0_VS_10_PERCENT_CONTROL"
            or receipt.get("selection_seed") != 3407
            or receipt.get("record_count_per_condition") != 1600
            or receipt.get("common_unchanged_TRACE_records") != 1440
            or receipt.get("replacement_TRACE_records") != 160
            or receipt.get("control_source_counts") != {"trace": 1600}
            or receipt.get("reference_source_counts") != {"trace": 1440, "nvw-independent-procedural-v1": 160}
            or receipt.get("parent_files_sha256") != parent_files
            or receipt.get("reference_files_sha256") != reference_files
            or Path(receipt.get("reference", "")).resolve() != reference.resolve()
            or subset.get("status") != "READY" or subset.get("parent_files_sha256") != parent_files
            or subset.get("records_sha256") != audit.sha(reference/"approved/records.jsonl")
            or reference_manifest.get("preparation_receipt_sha256") != audit.sha(reference/"subset_receipt.json")
            or reference_manifest.get("pilot_scale") != subset.get("pilot_scale")
            or manifest.get("preparation_receipt_sha256") != audit.sha(receipt_path)
            or receipt.get("records_sha256") != audit.sha(control/"approved/records.jsonl")
            or receipt.get("source_sha256") != audit.sha(Path(prepare_controls.__file__))):
        raise ValueError("Control/reference immutable preparation lineage mismatch")
    eligible, exclusion, _, _ = prepare_training.smoke_exclusion(rows, smoke_root/"approved/records.jsonl")
    if (subset.get("smoke_exclusion") != exclusion
            or receipt.get("reference_smoke_exclusion") != exclusion
            or reference_manifest.get("pilot_scale", {}).get("smoke_exclusion") != exclusion
            or audit.sha(reference/exclusion["evidence_path"]) != exclusion["records_sha256"]
            or prepare_controls.select_trace_control(eligible, reference_rows) != control_rows):
        raise ValueError("Control rows or smoke exclusion differ from approved substitution")
    for key in ("sources", "known_parent_ids", "source_sha256", "contamination_audit"):
        if any(m.get(key) != parent_manifest.get(key) for m in (manifest, reference_manifest)):
            raise ValueError(f"Control/reference inherited approval differs: {key}")
    policy = {"kind": receipt["kind"], "nvw_fraction": 0, "record_count": 1600, "seed": 3407}
    input_audit = manifest.get("model_input_audit", {})
    if (manifest.get("mixture_policy") != policy
            or json.loads((control/"mixture_policy.json").read_text()) != policy
            or manifest.get("mixture_policy_sha256") != audit.sha(control/"mixture_policy.json")
            or input_audit.get("status") != "PASS" or input_audit.get("record_count") != 1600
            or input_audit.get("kind") != "INHERITED_APPROVED_TRACE_CONTROL"
            or input_audit.get("processor_reexecuted") is not False
            or input_audit.get("parent_record_count") != len(rows)
            or input_audit.get("parent_manifest_sha256") != parent_files["common_manifest.json"]
            or any(input_audit.get(key) != parent_manifest["model_input_audit"].get(key)
                   for key in ("processor_revision", "sequence_budget", "episode_pixels"))):
        raise ValueError("Control policy/processor approval mismatch")
    frozen = {}
    for root, names in ((parent, parent_files), (reference, receipt.get("reference_files_sha256", {}))):
        for name, digest in names.items():
            source = audit.image_path(root, name)
            copied = audit.image_path(control/("parent_evidence" if root == parent else "reference_evidence"), name)
            if audit.sha(source) != digest or audit.sha(copied) != digest:
                raise ValueError("Copied approval evidence differs from its original")
            frozen[str(source.resolve())] = digest
            frozen[str(copied.resolve())] = digest
    for root, selected, selected_manifest in ((control, control_rows, manifest),
                                              (reference, reference_rows, reference_manifest)):
        for row in selected:
            audit.check_lineage(row, parent_manifest["sources"])
        expected_images = {name: parent_manifest["files"].get(name)
                           for row in selected for name in row["image_paths"]}
        if selected_manifest["files"] != expected_images:
            raise ValueError("Control/reference image approval differs from parent")
        for name, digest in expected_images.items():
            frozen[str(audit.image_path(root, name).resolve())] = digest
    for path in (receipt_path, control/"mixture_policy.json", control/"common_manifest.json",
                 control/"approved/records.jsonl", reference/"common_manifest.json",
                 reference/"approved/records.jsonl", reference/"subset_receipt.json",
                 smoke_root/"approved/records.jsonl", smoke_root/"smoke_manifest.json"):
        frozen[str(path.resolve())] = audit.sha(path)
    for name in {name for row in smoke_rows for name in row["image_paths"]}:
        frozen[str(audit.image_path(smoke_image_root, name).resolve())] = smoke_manifest["files"][name]
    return control_rows, reference_rows, smoke_rows, frozen


def verify_training(output: Path, rows: list, manifest: Path, cfg: dict, reference: dict):
    """Verify the fresh FL journals and checkpoint without inventing a closed four-arm run."""
    checkpoint = output/"checkpoint"
    metadata = json.loads((checkpoint/"metadata.json").read_text())
    indices = list(range(len(rows)))
    random.Random(3407).shuffle(indices)
    ids = [rows[i]["id"] for i in indices]
    consumed, updates = audit.read_rows(output/"consumed_ids.jsonl"), audit.read_rows(output/"updates.jsonl")
    budgets = {key: cfg[key] for key in ("min_pixels", "max_pixels", "episode_pixels")}
    budgets.update(sequence_budget=cfg["max_sequence_tokens"], accumulation=cfg["gradient_accumulation"], max_steps=0)
    if (not runner.exposure_matches(metadata) or metadata.get("termination") != "EPOCH_COMPLETE"
            or metadata.get("arm") != "FL" or metadata.get("seed") != 3407
            or metadata.get("planned_records") != 1600 or metadata["exposure"].get("records") != 1600
            or metadata.get("base_id") != BASE_ID or metadata.get("base_revision") != BASE_REVISION
            or metadata.get("epochs") != 1 or metadata.get("base_precision") != "BF16"
            or not math.isfinite(metadata.get("seconds", 0)) or metadata.get("seconds", 0) <= 0
            or type(metadata["exposure"].get("visual_tokens")) is not int or metadata["exposure"]["visual_tokens"] <= 0
            or metadata.get("trainable_master_precision") != "FP32"
            or metadata.get("manifest_sha256") != audit.sha(manifest)
            or metadata.get("records_sha256") != audit.sha(manifest.parent/"approved/records.jsonl")
            or metadata.get("source_sha256") != {n: audit.sha(Path(__file__).with_name(n)) for n in ("train.py", "modules.py")}
            or metadata.get("consumed_ids_sha256") != audit.sha(output/"consumed_ids.jsonl")
            or metadata.get("consumed_order_sha256") != order_hash(ids)
            or any(metadata.get("budgets", {}).get(k) != v for k, v in budgets.items())
            or [r.get("id") for r in consumed] != ids
            or [r.get("position") for r in consumed] != list(range(1600))
            or len(updates) != 100 or metadata["exposure"].get("updates") != 100
            or metadata["exposure"].get("images") != sum(len(r["image_paths"]) for r in rows)
            or metadata.get("inventory") != reference.get("inventory")):
        raise ValueError("Fresh control FL identity/exposure/module proof mismatch")
    if (not metadata["inventory"].get("trainable")
            or metadata["inventory"].get("arm") != "FL"
            or {r.get("group") for r in metadata["inventory"]["trainable"]} != {"L"}):
        raise ValueError("Fresh control requires a nonempty language-only FL inventory")
    for key in ("input_tokens", "target_tokens"):
        if (any(type(r.get(key)) is not int or r[key] <= 0 for r in consumed)
                or sum(r[key] for r in consumed) != metadata["exposure"].get(key)):
            raise ValueError("Fresh control token ledger mismatch")
    for step, update in enumerate(updates, 1):
        if (update.get("step") != step or update.get("consumed_records") != step*16
                or update.get("consumed_order_sha256") != order_hash(ids[:step*16])
                or update.get("loss_tokens") != sum(r["target_tokens"] for r in consumed[(step-1)*16:step*16])
                or any(set(update.get(key, {})) != {"L"}
                       or any(not math.isfinite(v) or v <= 0 for v in update[key].values())
                       for key in ("gradient_l2", "sampled_update_l1"))
                or update.get("frozen_sampled_checks", 0) <= 0
                or not 0 < update.get("peak_allocated_bytes", 0) <= update.get("peak_reserved_bytes", 0)):
            raise ValueError("Fresh control gradient/update/frozen/memory proof mismatch")
    if (not metadata.get("files") or not metadata.get("state_sha256")
            or set(metadata["state_sha256"]) != {r["name"] for r in metadata["inventory"]["trainable"]}
            or not {"adapter/adapter_config.json", "adapter/adapter_model.safetensors"} <= set(metadata["files"])):
        raise ValueError("Fresh control checkpoint payload incomplete")
    for name, digest in metadata["files"].items():
        if audit.sha(audit.image_path(checkpoint, name)) != digest:
            raise ValueError("Fresh control checkpoint payload changed")
    return metadata


def require_idle_gpu():
    probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
    if probe.returncode or probe.stdout.strip():
        raise ValueError("GPU idle proof failed or another compute process is active; no process is killed")


def run(args, controller):
    _previous_status, _previous_journal, previous_files = runner.closed_controller(args.previous_controller)
    protocol = json.loads(args.protocol.read_text())
    if protocol != json.loads(Path(__file__).with_name("protocol.json").read_text()) or protocol["master_seed"] != 3407:
        raise ValueError("Control requires the unchanged frozen protocol")
    cfg = protocol["training"]
    rows, reference_rows, smoke_rows, frozen = control_inputs(
        args.parent, args.reference, args.control, args.smoke_root, args.smoke_image_root)
    runner.pilot_scale_evidence(reference_rows, json.loads((args.reference/"common_manifest.json").read_text()),
                                cfg["pilot_records"], 1600, 360, 3407)
    _smokes, smoke_files, smoke_receipt = runner.reuse_smoke(args.smoke_controller, smoke_rows,
        args.smoke_root/"approved/records.jsonl", args.smoke_root/"smoke_manifest.json", cfg, 3407)
    pilots, training_files, training_receipt = runner.reuse_pilot(args.reference_training_controller,
        reference_rows, args.reference/"approved/records.jsonl", args.reference/"common_manifest.json",
        cfg, 3407, args.protocol, args.reference)
    if "FL" not in pilots:
        raise ValueError("Closed 10% FL checkpoint unavailable")
    checkpoint = Path(training_receipt["arms"]["FL"]["checkpoint"])
    cohorts = {"mmmu": args.mmmu_cohort, "external": args.external_cohort}
    baselines = {"mmmu": args.baseline_mmmu, "external": args.baseline_external}
    for name, cohort in cohorts.items():
        runner.cohort_gate(cohort, 60)
        runner.evaluation_gate(baselines[name], cohort, args.protocol, baseline=True)
    results, eval_files, eval_receipt = runner.reuse_pilot_evaluations(args.reference_evaluation_controller,
        {"FL": checkpoint}, baselines, cohorts, args.protocol)
    if set(results) != {("FL", "mmmu"), ("FL", "external")}:
        raise ValueError("Both closed 10% FL paired cohorts are required")
    frozen.update(smoke_files | training_files | eval_files | previous_files)
    for path in (args.protocol, args.model_path/"config.json"):
        frozen[str(path.resolve())] = audit.sha(path)
    if (controller.output/"launch.json").exists():
        frozen[str((controller.output/"launch.json").resolve())] = audit.sha(controller.output/"launch.json")
    for name, cohort in cohorts.items():
        for filename in ("manifest.json", "inputs.jsonl", "gold.jsonl"):
            frozen[str((cohort/filename).resolve())] = audit.sha(cohort/filename)
        for filename in ("summary.json", "predictions.jsonl"):
            frozen[str((baselines[name]/filename).resolve())] = audit.sha(baselines[name]/filename)
    estimate = 2*pilots["FL"]["seconds"] + 150 + sum(
        json.loads((path/"summary.json").read_text())["seconds"] for path in results.values())
    declaration = {"arm": "FL", "selection": "resource_based_closed_reference_available_before_control_results",
        "smoke_records_root": str(args.smoke_root.resolve()), "smoke_image_root": str(args.smoke_image_root.resolve()),
        "both_conditions_preregistered": False, "records_per_condition": 1600, "updates_per_condition": 100,
        "training_tokens_expected_equal": False, "original_10K_execution_completed": False,
        "original_G7_completed": False, "A0_A3_completed": False,
        "required_seconds_estimate": estimate, "estimate_policy": "2x closed FL training +150s reload +closed paired evaluations",
        "files": frozen, "reference_training": training_receipt, "reference_evaluation": eval_receipt,
        "smoke": smoke_receipt, "public_upload_allowed": False}
    audit.dump(controller.output/"input_freeze.json", declaration)
    frozen[str((controller.output/"input_freeze.json").resolve())] = audit.sha(controller.output/"input_freeze.json")
    controller.event(stage="control_budget_gate", required_seconds=estimate, remaining_seconds=controller.remaining())
    if estimate >= controller.remaining()-120:
        return {"status": "BUDGET_STOP", "GPU_training_started": False, "required_seconds_estimate": estimate}

    def stage(name, command, maximum=None):
        if any(audit.sha(Path(path)) != digest for path, digest in frozen.items()):
            raise ValueError("Frozen control/reference proof changed")
        require_idle_gpu()
        controller.stage(name, command, maximum)

    output = controller.output/"pilot_FL"
    command = [sys.executable, "-m", "fine_tuning.train", "--model-path", str(args.model_path),
        "--records", str(args.control/"approved/records.jsonl"), "--manifest", str(args.control/"common_manifest.json"),
        "--image-root", str(args.control), "--output", str(output), "--arm", "FL", "--seed", "3407",
        "--budget-seconds", str(max(1, controller.remaining()-120))]
    for flag, value in (("min-pixels", cfg["min_pixels"]), ("max-pixels", cfg["max_pixels"]),
                        ("episode-pixels", cfg["episode_pixels"]), ("sequence-budget", cfg["max_sequence_tokens"]),
                        ("accumulation", cfg["gradient_accumulation"])):
        command.extend(["--"+flag, str(value)])
    stage("pilot_FL_train", command)
    metadata = json.loads((output/"checkpoint/metadata.json").read_text())
    if not runner.exposure_matches(metadata):
        raise TimeoutError("Control full exposure incomplete; checkpoint retained")
    metadata = verify_training(output, rows, args.control/"common_manifest.json", cfg, pilots["FL"])
    for path in (output/"checkpoint/metadata.json", output/"consumed_ids.jsonl", output/"updates.jsonl"):
        frozen[str(path.resolve())] = audit.sha(path)
    for name, digest in metadata["files"].items():
        frozen[str(audit.image_path(output/"checkpoint", name).resolve())] = digest
    stage("pilot_FL_reload", [sys.executable, "-m", "fine_tuning.train", "--model-path", str(args.model_path),
                              "--verify-checkpoint", str(output/"checkpoint")], 600)
    comparisons = {}
    for name, cohort in cohorts.items():
        evaluated = controller.output/f"pilot_FL_{name}"
        stage(f"pilot_FL_{name}", [sys.executable, "-m", "fine_tuning.evaluate",
            "run" if name == "mmmu" else "run-external", "--model-path", str(args.model_path),
            "--cohort", str(cohort), "--output", str(evaluated), "--protocol", str(args.protocol),
            "--checkpoint", str(output/"checkpoint"), "--budget-seconds", str(max(1, int(controller.remaining()-120)))])
        summary = json.loads((evaluated/"summary.json").read_text())
        if summary.get("status") == "BUDGET_STOP":
            raise TimeoutError(f"Control {name} evaluation incomplete; results retained")
        if (summary.get("checkpoint_metadata_sha256") != audit.sha(output/"checkpoint/metadata.json")
                or Path(summary.get("checkpoint", "")).resolve() != (output/"checkpoint").resolve()
                or summary.get("evaluator_source_sha256") != audit.sha(Path(__file__).with_name("evaluate.py"))
                or summary.get("benchmark_derived") is not (name == "mmmu")):
            raise ValueError("Control evaluation checkpoint/source/cohort identity mismatch")
        paired = runner.paired_input_gate(evaluated, baselines[name], cohort, args.protocol)
        audit.dump(evaluated/"input_equivalence.json", paired)
        _, reference_predictions = load_evaluation(results["FL", name])
        _, predictions = load_evaluation(evaluated)
        compare_inputs(reference_predictions, predictions)
        comparisons[name] = {"direction": "NVW0_minus_NVW10",
            "helper_condition_labels": {"f0": "trained_NVW10_FL_reference", "ft": "trained_NVW0_FL_control"},
            "paired": paired_counts(reference_predictions, predictions),
            "control_summary_sha256": audit.sha(evaluated/"summary.json"),
            "reference_summary_sha256": audit.sha(results["FL", name]/"summary.json")}
        if name == "external":
            comparisons[name].update(groups=external_groups(reference_predictions, predictions), mixed_metrics_not_MMMU=True)
        for filename in ("summary.json", "predictions.jsonl", "input_equivalence.json"):
            frozen[str((evaluated/filename).resolve())] = audit.sha(evaluated/filename)
    if any(audit.sha(Path(path)) != digest for path, digest in frozen.items()):
        raise ValueError("Frozen proof changed before report")
    report = dict(declaration, status="ADDITIONAL_FL_NVW0_VS10_PILOT_COMPLETED",
                  control_exposure=metadata["exposure"], reference_exposure=pilots["FL"]["exposure"], comparisons=comparisons)
    audit.dump(controller.output/"report.json", report)
    return {"status": report["status"], "GPU_training_started": True, "report_sha256": audit.sha(controller.output/"report.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "reference", "control", "smoke-root", "smoke-image-root", "smoke-controller",
                 "reference-training-controller", "reference-evaluation-controller", "previous-controller",
                 "baseline-mmmu", "baseline-external", "mmmu-cohort", "external-cohort", "protocol", "model-path", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("owned-pod-id", "gpu-cutoff-utc", "billing-start-utc"):
        parser.add_argument("--"+name, required=True)
    for name in ("balance-usd", "reserve-usd", "hourly-usd"):
        parser.add_argument("--"+name, type=float, required=True)
    args = parser.parse_args()
    if (not args.owned_pod_id or os.environ.get("RUNPOD_POD_ID") != args.owned_pod_id
            or not args.model_path.is_dir() or args.model_path.name != BASE_REVISION):
        parser.error("Exact owned Pod and pinned local F0 snapshot required")
    if args.reserve_usd <= 0:
        parser.error("A positive recovery reserve is required")
    deadline = runner.effective_deadline(runner.utc(args.gpu_cutoff_utc), runner.utc(args.billing_start_utc),
                                        args.balance_usd, args.reserve_usd, args.hourly_usd)
    controller = runner.Controller(args.output, deadline, Path(__file__).parent)
    audit.dump(args.output/"launch.json", {key: str(value) if isinstance(value, Path) else value
                                         for key, value in vars(args).items()})
    try:
        result = run(args, controller)
    except TimeoutError as error:
        result = {"status": "BUDGET_STOP", "reason": str(error)}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result = {"status": "BLOCKED", "reason": f"{type(error).__name__}: {error}"}
    result.update(deadline_utc=deadline.isoformat(), stages=controller.stage_seconds,
                  original_10K_execution_completed=False, original_G7_completed=False, A0_A3_completed=False,
                  arm="FL", public_upload_allowed=False)
    audit.dump(args.output/"status.json", result)
    controller.event(stage="controller", **result)
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
