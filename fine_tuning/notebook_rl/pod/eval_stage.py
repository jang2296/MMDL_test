"""Plan the proof's MMMU-val evaluation stages as pod/chain.py specs (`pod-chain-spec-v1`).

CLI: python -m fine_tuning.notebook_rl.pod.eval_stage plan --stage {eval-f0,eval-ft,eval-course}
         --params PARAMS.json --output STEPS.json
Writes STEPS.json once (an existing file is refused) and prints its sha256, the value the approved job's
main argv passes as `chain run --spec STEPS.json --spec-sha256 HEX`. Same params -> same bytes. Executes
nothing; analysis (`proof_eval analyze`) stays local after collection.

Arms, variants, seeds and config files come from proof_eval.VARIANTS, the table its preregistration is
built from: statistical runs F0 and FT x c16 seeds 3407/3408/3409 (`proof_eval run-eval`), one course run
FT x mmmu-val-v8 (c2, seed 3407; plain scripts/eval.sh). Every stage pins its inputs first (`chain pin`),
then appends all of its ledger STARTs, then runs; the ledger START of a run always precedes the run, and
an omitted or replaced seed stays visible to `analyze --ledger`. Run ids: <run_prefix>-<arm>-<tag>-evaluation.

params (`eval-stage-params-v1`), absolute canonical paths:
  all stages   commit (40-hex clone commit, --require-commit), repo_root (the clone), python
               (<volume_root>/venv-<16 hex>/bin/python), volume_root, artifact_root, prereg, prereg_sha256,
               ledger, run_prefix; artifact_root, ledger and prereg under <volume_root>/eval-data/
  eval-ft      + checkpoint, checkpoint_metadata_sha256 (sha256 of <checkpoint>/metadata.json, which pins
               every checkpoint file), binding, binding_sha256, training_commit, ft_export (new dir)
  eval-course  + ft_export, ft_model_ref_sha256 (sha256 of <ft_export>/model_ref.json written by eval-ft)
"""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from pathlib import Path
from typing import Any

from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION

from .. import proof_eval
from .common import HEX40, HEX64, sha256_bytes
from .job import EVAL_STAGES
from .verify import EVAL_CACHE, EVAL_DATA_ROOT, EVAL_DIR

PARAMS_VERSION = "eval-stage-params-v1"
SPEC_VERSION = "pod-chain-spec-v1"
CHAIN = "fine_tuning.notebook_rl.pod.chain"
PROOF = "fine_tuning.notebook_rl.proof_eval"
HARDWARE = "fine_tuning/notebook_rl/eval_configs/hardware/a40_48gb.yaml"
BASE_REF = "manifests/models/baseline.json"
COMMON = {"version", "commit", "repo_root", "python", "volume_root", "artifact_root", "prereg",
          "prereg_sha256", "ledger", "run_prefix"}
EXTRA = {"eval-f0": set(),
         "eval-ft": {"checkpoint", "checkpoint_metadata_sha256", "binding", "binding_sha256",
                     "training_commit", "ft_export"},
         "eval-course": {"ft_export", "ft_model_ref_sha256"}}
PREFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,40}\Z")
VENV_PYTHON = re.compile(r"/venv-[0-9a-f]{16}/bin/python\Z")
# Raw responses (with MMMU questions and gold), the ledger and the prereg live under <volume_root>/eval-data,
# the root the training preflight proves every training-side guard forbids.
UNDER_EVAL_ROOT = ("artifact_root", "ledger", "prereg")


def _check(stage: str, params: Any) -> dict[str, Any]:
    if stage not in EVAL_STAGES:
        raise ValueError(f"stage must be one of {EVAL_STAGES}")
    keys = COMMON | EXTRA[stage]
    if not isinstance(params, dict) or set(params) != keys:
        raise ValueError(f"{stage} params must have exactly the keys {sorted(keys)}")
    if params["version"] != PARAMS_VERSION:
        raise ValueError(f"params version must be {PARAMS_VERSION}")
    for key in keys - {"version", "run_prefix"}:
        value = params[key]
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if key.endswith("commit"):
            ok = bool(HEX40.match(value))
        elif key.endswith("sha256"):
            ok = bool(HEX64.match(value))
        else:
            ok = value.startswith("/") and value != "/" and posixpath.normpath(value) == value
        if not ok:
            raise ValueError(f"{key} is malformed: {value!r}")
    if not isinstance(params["run_prefix"], str) or not PREFIX.match(params["run_prefix"]):
        raise ValueError("run_prefix must match [A-Za-z0-9][A-Za-z0-9_.-]{0,40}")
    if (not params["python"].startswith(params["volume_root"] + "/")
            or not VENV_PYTHON.match(params["python"][len(params["volume_root"]):])):
        raise ValueError("python must be <volume_root>/venv-<lock sha256[:16]>/bin/python")
    volume = params["volume_root"]
    reserved = (f"{volume}/{EVAL_DATA_ROOT}", f"{volume}/{EVAL_CACHE}")
    for key in UNDER_EVAL_ROOT:
        value = params[key]
        if (not value.startswith(f"{volume}/{EVAL_DIR}/")
                or any(value == r or value.startswith(r + "/") for r in reserved)):
            raise ValueError(f"{key} must be under {volume}/{EVAL_DIR}/ (outside mmmu/ and hf-cache/) "
                             "so that the training guards forbid it")
    return params


def _runs(stage: str) -> list[tuple[str, str]]:
    """(arm, variant) per stage, in the order of proof_eval.VARIANTS (= the preregistered runs)."""
    statistical = [v for v, (_, _, _, role) in proof_eval.VARIANTS.items() if role == "statistical"]
    return {"eval-f0": [("F0", v) for v in statistical], "eval-ft": [("FT", v) for v in statistical],
            "eval-course": [("FT", proof_eval.COURSE)]}[stage]


def run_id(prefix: str, arm: str, variant: str) -> str:
    seed = proof_eval.VARIANTS[variant][1]
    tag = "course" if variant == proof_eval.COURSE else "c16"
    return f"{prefix}-{arm.lower()}-{tag}-s{seed}-evaluation"


def _step(name: str, argv: list[str], expect: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "argv": argv, "expect_files": expect or []}


def plan(stage: str, params: dict[str, Any]) -> dict[str, Any]:
    p = _check(stage, params)
    py, repo, artifacts = p["python"], p["repo_root"], p["artifact_root"]
    snapshot = (f"{p['volume_root']}/hf-cache/hub/models--{MODEL_ID.replace('/', '--')}"
                f"/snapshots/{MODEL_REVISION}")
    pins = [("prereg", p["prereg"], p["prereg_sha256"])]
    if stage == "eval-ft":
        pins += [("checkpoint", f"{p['checkpoint']}/metadata.json", p["checkpoint_metadata_sha256"]),
                 ("binding", p["binding"], p["binding_sha256"])]
    if stage == "eval-course":
        pins += [("ft-model-ref", f"{p['ft_export']}/model_ref.json", p["ft_model_ref_sha256"])]
    steps = [_step(f"pin-{name}", [py, "-m", CHAIN, "pin", "--file", path, "--sha256", sha])
             for name, path, sha in pins]
    if stage == "eval-f0":
        model_ref, model_path = f"{repo}/{BASE_REF}", snapshot
    else:
        model_ref, model_path = f"{p['ft_export']}/model_ref.json", f"{p['ft_export']}/model"
    if stage == "eval-ft":
        steps.append(_step("export-ft", [
            py, "-m", PROOF, "export", "--checkpoint", p["checkpoint"], "--binding", p["binding"],
            "--base", snapshot, "--output", p["ft_export"], "--code-commit", p["training_commit"]],
            [model_ref, f"{model_path}/export_manifest.json"]))
    runs = [(arm, variant, run_id(p["run_prefix"], arm, variant)) for arm, variant in _runs(stage)]
    for arm, variant, rid in runs:
        steps.append(_step(f"ledger-{rid}", [
            py, "-m", PROOF, "ledger-start", "--prereg", p["prereg"], "--arm", arm, "--variant", variant,
            "--run-id", rid, "--ledger", p["ledger"]], [p["ledger"]]))
    for arm, variant, rid in runs:
        protocol = f"{repo}/{proof_eval._rel(proof_eval.VARIANTS[variant][0])}"
        flags = ["--protocol", protocol, "--hardware", f"{repo}/{HARDWARE}", "--model-ref", model_ref,
                 "--model-path", model_path, "--base-path", snapshot,
                 "--data-root", f"{p['volume_root']}/{EVAL_DATA_ROOT}", "--artifact-root", artifacts,
                 "--public-root", f"{artifacts}/public", "--require-commit", p["commit"],
                 "--job-id", rid.removesuffix("-evaluation"), "--run-role", "evaluation", "--run-id", rid,
                 "--mode", "full", "--no-download"]
        run_dir = f"{artifacts}/runs/{rid}"
        expect = [f"{run_dir}/summary.json", f"{run_dir}/predictions.jsonl"]
        if variant == proof_eval.COURSE:  # the course run is the plain entry point, never run-eval
            argv = ["/usr/bin/env", f"MMDL_PYTHON={py}", "bash", f"{repo}/scripts/eval.sh", *flags]
        else:
            argv = [py, "-m", PROOF, "run-eval", *flags]
            expect.append(f"{run_dir}/{proof_eval.REGISTRATION_FILE}")
        steps.append(_step(f"eval-{rid}", argv, expect))
    canonical = json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    # job_stage: chain.run refuses this spec under a job of another stage (a training-stage job, whose preflight
    # skips the eval_data check and is bound to the training-data pins, must never carry an MMMU evaluation).
    return {"version": SPEC_VERSION, "meta": {"stage": stage, "job_stage": stage,
                                              "params_sha256": sha256_bytes(canonical.encode()),
                                              "generator": "fine_tuning.notebook_rl.pod.eval_stage plan"},
            "steps": steps}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="write the chain spec of one evaluation stage")
    plan_parser.add_argument("--stage", choices=EVAL_STAGES, required=True)
    plan_parser.add_argument("--params", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        text = json.dumps(plan(args.stage, json.loads(args.params.read_text(encoding="utf-8"))),
                          indent=2, sort_keys=True) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except (OSError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(sha256_bytes(text.encode()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
