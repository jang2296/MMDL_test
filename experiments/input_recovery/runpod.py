"""Execute 336 independent diagnostic cells from the approved exact GitHub clone."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fine_tuning.budget import effective_deadline, utc
from fine_tuning.run_experiment import Controller
from fine_tuning.runpod_g6 import checkout
from mmdl.runtime.artifacts import read_json, write_json

from .runner import ROOT, model_identity


def run(args, controller):
    runtime = args.runtime_root.resolve()
    setup = read_json(runtime / "setup/model.json")
    if (setup.get("status") != "MODEL_HASH_VERIFIED" or setup.get("code_commit") != args.commit
            or setup.get("code_repository") != "jang2296/MMDL_test"):
        raise ValueError("Verified setup from the same Git commit required")
    model = Path(setup["model_path"]).resolve(strict=True)
    if not model.is_relative_to(runtime):
        raise ValueError("Pinned base must belong to the verified runtime")
    model_identity(model, "F0")
    if (args.data_root / "gold.jsonl").exists():
        raise ValueError("Do not upload the diagnostic gold file to the inference Pod")
    outputs = runtime / "runs/input_recovery"
    outputs.mkdir(parents=True, exist_ok=True)
    def stage(name, module, *values):
        controller.stage(name, [sys.executable, "-m", "experiments.input_recovery." + module,
                               *map(str, values)])
    def infer(arm, candidate, phase):
        output = outputs / arm
        stage(arm + "_" + phase, "runner", "infer", "--data-root", args.data_root,
              "--model-path", candidate, "--base-path", model, "--output", output,
              "--arm", arm, "--phase", phase, "--slots", args.slots,
              "--deadline-utc", controller.deadline.isoformat(), "--commit", args.commit)
        status = read_json(output / "status.json")
        expected = 60 if arm == "F0" and phase == "smoke" else 240 if arm == "F0" else 24
        if status.get("status") != "COMPLETED" or status.get("completed") != expected:
            raise ValueError("Actual expected diagnostic completions missing")
    infer("F0", model, "smoke")
    infer("F0", model, "full")
    for arm in ("FL", "FM", "FV", "FML"):
        exported = runtime / "derived_models" / arm
        stage(arm + "_export", "checkpoints", "--model-path", model,
              "--checkpoint", args.checkpoint_root / arm, "--output", exported)
        model_identity(exported, arm)
        infer(arm, exported, "full")
    return {"status": "DIAGNOSTIC_INFERENCE_COMPLETED", "actual_cells": 336,
            "scoring_status": "PENDING_LOCAL_GOLD_COMPARISON", "outputs": str(outputs),
            "training_executed": False, "provenance": "BENCHMARK_DERIVED",
            "training_allowed": False, "public_upload_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-root", "data-root", "checkpoint-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("commit", "owned-pod-id", "gpu-cutoff-utc", "billing-start-utc"):
        parser.add_argument("--" + name, required=True)
    for name in ("balance-usd", "reserve-usd", "hourly-usd"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--slots", type=int, choices=(2, 4, 8), default=4)
    args = parser.parse_args()
    checkout(ROOT, args.commit)
    if not args.owned_pod_id or os.environ.get("RUNPOD_POD_ID") != args.owned_pod_id:
        parser.error("Exact owned Pod identity required")
    for path in (args.runtime_root, args.data_root, args.checkpoint_root, args.output):
        canonical = path.resolve()
        if not canonical.is_relative_to(Path("/workspace")) or canonical == Path("/workspace"):
            parser.error("Explicit child paths on persistent /workspace required")
        if canonical.is_relative_to(ROOT):
            parser.error("Private artifacts must be outside the Git checkout")
    deadline = effective_deadline(utc(args.gpu_cutoff_utc), utc(args.billing_start_utc),
                                  args.balance_usd, args.reserve_usd, args.hourly_usd)
    controller = Controller(args.output, deadline, Path(__file__).parent)
    result = {"status": "INCOMPLETE"}
    try:
        result = run(args, controller)
    except TimeoutError as exc:
        result = {"status": "BUDGET_STOP", "reason": str(exc), "training_executed": False}
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
