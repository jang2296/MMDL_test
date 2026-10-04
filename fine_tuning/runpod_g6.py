"""Fresh full-10K G6 from a pinned MMDL_test clone, with bounded preparation too."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .audit import dump, sha
from .run_experiment import Controller, effective_deadline, utc


def checkout(repo, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Exact 40-character commit required")
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain"):
        raise ValueError("Pinned clean checkout required")
    if git("remote", "get-url", "origin") not in {
            "https://github.com/jang2296/MMDL_test.git", "git@github.com:jang2296/MMDL_test.git"}:
        raise ValueError("MMDL_test origin required")


def expected_shortfall(raw):
    receipt = json.loads((raw / "preparation_receipt.json").read_text())
    if (receipt.get("status") != "INCOMPLETE_QUOTA" or receipt.get("stage") != "pilot"
            or receipt.get("expected_records") != 10000
            or type(receipt.get("quarantined_records")) is not int
            or not 0 < receipt["quarantined_records"] < 9000
            or (raw / "common_manifest.json").exists()):
        raise ValueError("Only a recorded contamination-quarantine shortfall can be completed")
    return receipt


def run(args, controller, repo):
    root = args.runtime_root
    setup = json.loads((root / "setup/model.json").read_text())
    if (setup.get("status") != "MODEL_HASH_VERIFIED" or setup.get("code_commit") != args.commit
            or setup.get("code_repository") != "jang2296/MMDL_test"):
        raise ValueError("Current exact-commit setup receipt required")
    expected = json.loads((repo / "manifests/models/baseline.json").read_text())
    if setup.get("manifest", {}).get("files") != expected["files"]:
        raise ValueError("Pinned model manifest mismatch")
    model = Path(setup["model_path"])
    if not model.resolve().is_relative_to(root):
        raise ValueError("Model must belong to this runtime")
    for entry in expected["files"]:
        if sha(model / entry["path"]) != entry["sha256"]:
            raise ValueError("Pinned model bytes changed after setup")

    def stage(name, module, *values):
        cleanup = {"termination_grace_seconds": 45} if module == "run_experiment" else {}
        controller.stage(name, [sys.executable, "-m", "fine_tuning." + module,
                                *map(str, values)], **cleanup)

    generated = root / "generated"
    for name, count, smoke, seed in (("prototype", 100, 120, 3407),
                                     ("pilot_pool", 600, 0, 3407),
                                     ("replacement_pool", 300, 0, 3410)):
        stage("generate_" + name, "nvw", "--output", generated / name,
              "--per-family", count, "--smoke-roots", smoke, "--seed", seed)
        stage("verify_" + name, "nvw", "--output", generated / name, "--verify")
    stage("benchmarks", "prepare_sources", "--task-root", root, "--stage", "benchmarks")
    index = root / "benchmark-exclusion/fingerprint_index.json"
    stage("index", "prepare_training", "index", "--coverage",
          root / "benchmark-exclusion/coverage.json", "--output", index)
    stage("trace", "prepare_sources", "--task-root", root, "--stage", "trace",
          "--inventory", repo / "fine_tuning/source_inventory_20261004.json", "--trace-shards", 4)
    approval = ["--source-registry", repo / "fine_tuning/source_registry_v1.json",
                "--benchmark-index", index, "--model-path", model]
    trace = ["--trace-downloads", root / "setup/trace-downloads.json",
             "--trace-cache-root", root / "hf-cache/hub"]
    smoke = root / "training/smoke"
    raw = root / "training/pilot_raw"
    complete = root / "training/pilot_complete"
    common = root / "training/common10k"
    stage("approve_smoke", "prepare_training", "smoke", "--nvw-root", generated / "prototype",
          *approval, "--output", smoke)
    try:
        stage("approve_pilot", "prepare_training", "pilot", "--nvw-root", generated / "pilot_pool",
              *trace, *approval, "--output", raw)
    except RuntimeError:
        receipt = expected_shortfall(raw)
        controller.event(stage="approve_pilot", status="RECORDED_QUARANTINE_SHORTFALL",
                         quarantined=receipt["quarantined_records"])
    stage("complete_pilot", "prepare_training", "complete-pilot", "--parent", raw,
          "--parent-source-root", repo / "fine_tuning", *trace, *approval, "--output", complete)
    stage("disjoint_pilot", "prepare_training", "disjoint-full-pilot", "--parent", complete,
          "--nvw-root", generated / "replacement_pool", "--smoke-records", smoke / "approved/records.jsonl",
          "--smoke-manifest", smoke / "smoke_manifest.json", *approval, "--output", common)
    mm = root / "evaluation/mmmu60"
    external = root / "evaluation/external60"
    stage("mmmu_cohort", "evaluate", "prepare-mmmu", "--data-root", root / "data/evaluation/mmmu",
          "--per-subject", 2, "--output", mm)
    stage("external_cohort", "evaluate", "prepare-external", "--nvw-root", generated / "prototype",
          "--trace-root", raw, "--roots-per-family", 4, "--trace-count", 24, "--output", external)
    experiment = root / "runs/g6_full10k"
    # Let the inner controller close its separately grouped GPU child before the outer watchdog.
    inner_cutoff = (controller.deadline - dt.timedelta(seconds=120)).isoformat()
    stage("full10k", "run_experiment", "--model-path", model,
          "--smoke-records", smoke / "approved/records.jsonl", "--smoke-manifest", smoke / "smoke_manifest.json",
          "--pilot-records", common / "approved/records.jsonl", "--pilot-manifest", common / "common_manifest.json",
          "--image-root", generated / "prototype", "--pilot-image-root", common,
          "--mmmu-cohort", mm, "--external-cohort", external,
          "--protocol", repo / "fine_tuning/protocol.json", "--output", experiment,
          "--owned-pod-id", args.owned_pod_id, "--gpu-cutoff-utc", inner_cutoff,
          "--billing-start-utc", args.billing_start_utc, "--balance-usd", args.balance_usd,
          "--reserve-usd", args.reserve_usd, "--hourly-usd", args.hourly_usd)
    result = json.loads((experiment / "status.json").read_text())
    if result.get("status") != "FOUR_ARM_PILOT_EVALUATED":
        return result
    stage("report", "report", "--controller", experiment,
          "--baseline-mmmu", experiment / "baseline_F0_mmmu",
          "--baseline-external", experiment / "baseline_F0_external", "--output", root / "reports/g6")
    return {"status": "FOUR_ARM_PILOT_EVALUATED", "report": str(root / "reports/g6")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    for name in ("commit", "owned-pod-id", "gpu-cutoff-utc", "billing-start-utc"):
        parser.add_argument("--" + name, required=True)
    for name in ("balance-usd", "reserve-usd", "hourly-usd"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    checkout(repo, args.commit)
    root = args.runtime_root
    if not root.is_absolute() or root != root.resolve() or not root.is_relative_to("/workspace") or root == Path("/workspace"):
        parser.error("Canonical child of /workspace required")
    deadline = effective_deadline(utc(args.gpu_cutoff_utc), utc(args.billing_start_utc),
                                  args.balance_usd, args.reserve_usd, args.hourly_usd)
    if not args.execute:
        print(json.dumps({"status": "DRY_RUN", "commit": args.commit, "deadline_utc": deadline.isoformat(),
                          "pilot_records": 10000, "training_started": False,
                          "steps": ["own_generation", "exclusion", "normal_approval", "smoke", "four_arm_full10k", "report"]}))
        return
    if os.environ.get("RUNPOD_POD_ID") != args.owned_pod_id:
        parser.error("Exact owned RunPod identity required")
    subprocess.run(["mountpoint", "-q", "/workspace"], check=True, timeout=10)
    os.environ.update(HF_HOME=str(root / "hf-cache"), HF_HUB_ENABLE_HF_TRANSFER="0",
                      PYTHONUNBUFFERED="1", MMDL_STORAGE_RESERVE_GIB="25")
    controller = Controller(root / "runs/g6_launch", deadline, repo / "fine_tuning")
    try:
        result = run(args, controller, repo)
    except TimeoutError as exc:
        result = {"status": "BUDGET_STOP", "reason": str(exc)}
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        result = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
    result.update(commit=args.commit, deadline_utc=deadline.isoformat(), stages=controller.stage_seconds)
    dump(controller.output / "status.json", result)
    controller.event(stage="controller", **result)
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
