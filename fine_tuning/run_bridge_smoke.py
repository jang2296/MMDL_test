"""Bounded fresh-F0 FL GPU input/update/reload check, not fullG7 or a score experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from . import audit, circuit_bridge, run_experiment as runner
from .graph_bridge import SOURCE_ID, VERSION
from .modules import BASE_REVISION
from .recover_evaluation import process_gone
from .run_control import require_idle_gpu, verify_fl_training
from .train import approved_records, validate_training_input_contract


def inputs(approval: Path, image_root: Path, child: Path, condition: str = "A3"):
    if condition not in {"A0", "A1", "A2", "A3"}:
        raise ValueError("Unknown graph bridge condition")
    records, manifest_path = approval/f"{condition}_approved/records.jsonl", approval/f"{condition}_approved/manifest.json"
    rows, manifest = approved_records(records, manifest_path, image_root)
    contract = audit.training_input_contract(child)
    validate_training_input_contract(manifest, contract["min_pixels"], contract["max_pixels"],
                                    contract["episode_pixels"], contract["sequence_budget"], child)
    original = image_root/f"{condition}.jsonl"
    pool = json.loads((image_root/"manifest.json").read_text())
    circuit = pool.get("version") == circuit_bridge.VERSION
    if circuit:
        circuit_bridge.verify(image_root)
    elif pool.get("version") != VERSION:
        raise ValueError("Unknown bridge source pool")
    candidates = {r["id"]: r for r in audit.read_rows(original)}
    train_candidates = {sid: r for sid, r in candidates.items() if r["split"] == "train"}
    root_count = len({r["root_id"] for r in train_candidates.values()}) if circuit else 28
    record_count = 3*root_count if circuit else 84
    source_id = circuit_bridge.SOURCE_ID if circuit else SOURCE_ID
    processor_path = approval/f"{condition}_processor.json"
    processor = json.loads(processor_path.read_text())
    if (root_count <= 0 or len(rows) != record_count or len({r["root_id"] for r in rows}) != root_count
            or {r["id"] for r in rows} != set(train_candidates)
            or Counter(r["view"] for r in rows) != {"qa": root_count, "structure": root_count, "grounded": root_count}
            or any(r.get("condition") != condition or r.get("source_id") != source_id for r in rows)
            or processor.get("producer_sha256") != audit.sha(Path(audit.__file__))
            or processor.get("records_sha256") != audit.sha(original)
            or processor.get("status") != "PASS" or processor.get("kind") != "ACTUAL_PROCESSOR_PREFLIGHT"
            or processor.get("processor_class") != "Qwen3VLProcessor" or processor.get("truncation") is not False
            or processor.get("training_input_contract") != contract
            or manifest["model_input_audit"].get("receipt_sha256") != audit.sha(processor_path)):
        raise ValueError(f"Fresh actual {condition} approval/source/processor identity mismatch")
    for row in rows:
        expected = dict(audit.model_input_record(candidates[row["id"]]), training_allowed=True,
                        processor_receipt_sha256=audit.sha(processor_path))
        proof = processor["records"].get(row["id"], {})
        if (row != expected or proof.get("status") != "PASS"
                or proof.get("input_sha256") != audit.input_fingerprint(row, image_root)
                or not 0 < proof.get("sequence_tokens", 0) <= contract["sequence_budget"]
                or proof.get("assistant_tokens", 0) <= 0
                or len(proof.get("image_grid_thw", [])) != len(row["image_paths"])
                or set(proof.get("tensor_sha256", {})) != {
                    "input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"}):
            raise ValueError(f"{condition} row differs from fresh approved actual input")
    frozen = {str(p.resolve()): audit.sha(p) for p in (records, manifest_path, processor_path, original, child,
              image_root/"manifest.json", image_root/"pairing.jsonl")}
    frozen.update({str(audit.image_path(image_root, name).resolve()): digest
                   for name, digest in manifest["files"].items()})
    return rows, manifest_path, records, contract, frozen


def verify_processor_exposure(output: Path, approval: Path, condition: str, metadata: dict, merge_size: int):
    proofs = json.loads((approval/f"{condition}_processor.json").read_text())["records"]
    consumed = audit.read_rows(output/"consumed_ids.jsonl")
    if (any(row["input_tokens"] != proofs[row["id"]]["sequence_tokens"]
            or row["target_tokens"] != proofs[row["id"]]["assistant_tokens"] for row in consumed)
            or metadata["exposure"]["visual_tokens"] != sum(
                sum(math.prod(grid) for grid in proofs[row["id"]]["image_grid_thw"])//merge_size**2 for row in consumed)):
        raise ValueError("GPU consumed token exposure differs from the approved actual processor proof")


def run(args, controller):
    if not process_gone({"pid": args.previous_pid, "pid_birth": args.previous_birth}):
        raise ValueError("The exact previous controller is still live; no overlapping GPU work")
    _, _, previous_files = runner.closed_controller(args.previous_controller)
    rows, manifest, records, contract, frozen = inputs(args.approval, args.image_root, args.input_protocol)
    if rows[0].get("source_id") == circuit_bridge.SOURCE_ID:
        raise ValueError("The historical Graph-only smoke is not a Circuit GPU protocol; use the four-condition controller")
    reference_path = args.reference_checkpoint/"metadata.json"
    reference = json.loads(reference_path.read_text())
    if (reference.get("arm") != "FL" or reference.get("base_revision") != BASE_REVISION
            or not runner.exposure_matches(reference)):
        raise ValueError("A closed fresh-F0 FL inventory reference is required")
    frozen.update(previous_files)
    frozen[str(reference_path.resolve())] = audit.sha(reference_path)
    frozen[str((args.model_path/"config.json").resolve())] = audit.sha(args.model_path/"config.json")
    merge_size = json.loads((args.model_path/"config.json").read_text())["vision_config"]["spatial_merge_size"]
    if type(merge_size) is not int or merge_size <= 0:
        raise ValueError("Invalid pinned model vision merge size")
    declaration = {"kind": "BOUNDED_GRAPH_A3_ALL_INPUT_GPU_TRAINING_SMOKE", "condition": "A3", "arm": "FL",
        "each_arm_starts_from": "F0", "checkpoint_initialization": "F0-not-reference-checkpoint",
        "module_selection": "closed FL update/reload inventory available; not selected by MMMU accuracy",
        "generator_version": VERSION, "training_input_contract": contract, "records": 84, "roots": 28,
        "seed": 3407, "planned_updates": 6, "spatial_merge_size": merge_size,
        "training_stage_limit_seconds": 600, "reload_stage_limit_seconds": 300,
        "budget_forecast_measured_for_bridge": False, "memory_fit_before_execution": "UNKNOWN",
        "full_G7_completed": False, "A0_A3_performance_comparison_completed": False,
        "professional_non_synthetic_transfer_completed": False, "public_upload_allowed": False, "files": frozen}
    audit.dump(controller.output/"input_freeze.json", declaration)
    if controller.remaining() < 1050:
        return {"status": "BUDGET_STOP", "GPU_training_started": False,
                "reason": "900sec bounded train/reload plus30sec termination and120sec recovery margin do not fit"}

    def stage(name, command, limit):
        if any(audit.sha(Path(p)) != digest for p, digest in frozen.items()):
            raise ValueError("Frozen input/source evidence changed")
        require_idle_gpu()
        if controller.remaining() < limit+150:
            raise TimeoutError("Stage cap plus30sec process termination and120sec recovery margin no longer fit")
        controller.stage(name, command, maximum_seconds=limit)

    output = controller.output/"A3_FL"
    stage("A3_FL_train", [sys.executable, "-m", "fine_tuning.train", "--model-path", str(args.model_path),
        "--records", str(records), "--manifest", str(manifest), "--image-root", str(args.image_root),
        "--output", str(output), "--arm", "FL", "--seed", "3407", "--accumulation", "16", "--max-steps", "0",
        "--input-protocol", str(args.input_protocol), "--min-pixels", str(contract["min_pixels"]),
        "--max-pixels", str(contract["max_pixels"]), "--episode-pixels", str(contract["episode_pixels"]),
        "--sequence-budget", str(contract["sequence_budget"]), "--budget-seconds", "540"], 600)
    saved = json.loads((output/"checkpoint/metadata.json").read_text())
    if saved.get("termination") == "BUDGET_STOP":
        raise TimeoutError("Bounded bridge training stopped before the full84 epoch; partial checkpoint preserved")
    cfg = {**contract, "max_sequence_tokens": contract["sequence_budget"], "gradient_accumulation": 16}
    metadata = verify_fl_training(output, rows, records, manifest, cfg, reference, contract)
    verify_processor_exposure(output, args.approval, "A3", metadata, merge_size)
    for name in ("checkpoint/metadata.json", "consumed_ids.jsonl", "updates.jsonl"):
        frozen[str((output/name).resolve())] = audit.sha(output/name)
    frozen.update({str(audit.image_path(output/"checkpoint", name).resolve()): digest
                   for name, digest in metadata["files"].items()})
    stage("A3_FL_reload", [sys.executable, "-m", "fine_tuning.train", "--model-path", str(args.model_path),
        "--verify-checkpoint", str(output/"checkpoint")], 300)
    log = (controller.output/"A3_FL_reload.log").read_text()
    reload_proof = [json.loads(line) for line in log.splitlines() if line.startswith('{"status": "RELOAD_VERIFIED"')]
    if len(reload_proof) != 1 or Path(reload_proof[0]["checkpoint"]).resolve() != (output/"checkpoint").resolve():
        raise ValueError("Missing fresh-process reload verification output")
    return {"status": "BOUNDED_GRAPH_A3_GPU_SMOKE_COMPLETED", "GPU_training_started": True,
            "all84_inputs_forward_backward_updated": True, "fresh_process_reload_verified": True,
            "exposure": metadata["exposure"], "checkpoint_metadata_sha256": audit.sha(output/"checkpoint/metadata.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("approval", "image-root", "input-protocol", "previous-controller", "reference-checkpoint", "model-path", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--previous-pid", type=int, required=True)
    for name in ("previous-birth", "owned-pod-id", "gpu-cutoff-utc", "billing-start-utc"):
        parser.add_argument("--"+name, required=True)
    for name in ("balance-usd", "reserve-usd", "hourly-usd"):
        parser.add_argument("--"+name, type=float, required=True)
    args = parser.parse_args()
    if (os.environ.get("RUNPOD_POD_ID") != args.owned_pod_id or not args.owned_pod_id
            or args.previous_pid <= 0 or not args.previous_birth.isdecimal() or args.reserve_usd <= 0
            or not args.model_path.is_dir() or args.model_path.name != BASE_REVISION):
        parser.error("Exact owned Pod, prior process identity, recovery reserve and pinned local F0 snapshot required")
    deadline = runner.effective_deadline(runner.utc(args.gpu_cutoff_utc), runner.utc(args.billing_start_utc),
                                        args.balance_usd, args.reserve_usd, args.hourly_usd)
    controller = runner.Controller(args.output, deadline, Path(__file__).parent)
    audit.dump(args.output/"launch.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        result = run(args, controller)
    except TimeoutError as error:
        result = {"status": "BUDGET_STOP", "reason": str(error)}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result = {"status": "BLOCKED", "reason": f"{type(error).__name__}: {error}"}
    result.update(deadline_utc=deadline.isoformat(), stages=controller.stage_seconds,
                  full_G7_completed=False, A0_A3_performance_comparison_completed=False,
                  professional_non_synthetic_transfer_completed=False, public_upload_allowed=False)
    audit.dump(args.output/"status.json", result)
    controller.event(stage="controller", **result)
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
