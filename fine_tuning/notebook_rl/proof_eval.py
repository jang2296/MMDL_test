"""Pre-registered F0 / FT-NOTE / FT-SOLUTION MMMU-val proof: merged export, 8k run wrapper, preregistration, analysis, gate.

Nothing here trains or selects. ``run-eval`` only registers the three 8k variant IDs inside its own process
and then calls the unchanged scripts/eval.sh entry point; src/ is never edited.
The statistical (paired F0/FT) variants differ from mmmu-val-v8 only in seed, 32 slots (the configs' batch_size; a
run executes MMDL_EVAL_SLOTS slots, see ``resolve_slots``), an 8192-token output cap, the image budget 262144..1310720 pixels (an evaluation budget, not the RL training budget) and a 24576
context. The c16 variant files stay on disk but are not registered.
Scope: every preregistered run is a ``--mode partial`` run on one of two subsets of the 900 MMMU-val items, both
derived from the frozen B2 + team-final-answer-v8 baseline and bound by sha256 in the preregistration
(evaluation only, never training): "wrong" = the 342 items B2 got wrong (proof-eval-subset-v1) and "rest" = the
558 items B2 got right (proof-eval-subset-rest-v1). ``make-subset --kind {wrong,rest}`` writes either file.
The course run (mmmu-val-v8, 900 items, plain scripts/eval.sh) keeps its code path but is not preregistered.
Arms: F0 (untrained base), FT-NOTE (primary, fixed in advance) and FT-SOLUTION (comparison); each has exactly one
checkpoint. ``analyze`` re-scores preserved raw responses with the frozen V8 scorer (the only scorer here) and
reduces several seeds only as means (CLAUDE.md §2.8): no reducer argument, no repeated (arm, variant, subset) run,
no per-item seed choice, no best seed. ``gate`` applies the preregistered 300-item decision rule to the "wrong"
subset; it is a threshold for spending more, not proof of improvement.
Error-note version 2 (claudedocs/plan_self_note_v2_20261102.md section 1): ``preregister --format v4`` writes a
separate preregistration (format notebook-rl-proof-prereg-v4) with arms F0 and FT-V2 on the existing
mmmu-val-v8-8k-seed3407-v2 variant, wrong + rest subsets (900 items). ``gate-62`` passes iff the FT-V2 900-item V8
correct count of that one seed exceeds B2's 558; one run, so not proof of improvement.
Error-note version 3 uses ``preregister --format v5 --ft-v3-model-ref REF``: one declared FT-NOTE-V3 checkpoint,
seed 3407 and the same 8k variant, wrong + rest exactly once. ``gate-62`` checks >=559; ``gate-67.4`` checks >=607.
Error-note version 4 (RL + note SFT) uses ``preregister --format v6 --ft-v4-model-ref REF`` the same way with the one
final FT-HYB-V4 export of its last round (``verify-v4-export`` before ledger START).
The single-response RL+SFT campaign (error_note.single_campaign) uses ``preregister --format v7 --ft-single-model-ref REF``:
one final FT-SINGLE export, same variant/seed/subsets, PASS iff > 558 (``gate-62``); ``export`` and
``verify-single-export`` take the campaign's sealed campaign-plan.json as ``--binding``.
Only F0 runs with the same evaluation code, registration and hardware are paired; historical counts are display only.

Run ledger (append-only JSONL, hash-chained, fsynced): ``ledger-start --prereg P --arm A --variant V --subset S
--run-id R --ledger L`` must precede every evaluation run (``run-eval`` or the plain course ``scripts/eval.sh``; R is
the run directory name artifact_root/runs/R). It refuses a second start of the same (arm, variant, subset) unless
the first is recorded with ``ledger-fail --ledger L --run-id R --reason ...`` (refused for a finished run).
``analyze``/``gate`` take every run directory of the ledger: they refuse a run that was never started, started under
another (arm, variant, subset) or recorded FAILED, and any live started run that is not passed (an omitted or
replaced run). A passed run that is not finished is listed as unfinished and left out of the numbers.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mmdl.evaluation.parsers import score_with_parser, scoring_source_files
from mmdl.runtime.artifacts import digest, sha256_file
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION, SUBJECTS, sample_seed

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FORMAT = "notebook-rl-proof-prereg-v3"
SUBSET_VERSION = "proof-eval-subset-v1"
REST_SUBSET_VERSION = "proof-eval-subset-rest-v1"
ARMS = ("F0", "FT-NOTE", "FT-SOLUTION", "FT-NOTE-3417")
FORMAT_V4 = "notebook-rl-proof-prereg-v4"  # error-note version 2: arms F0 and FT-V2
ARM_V2 = "FT-V2"
ARMS_V4 = ("F0", ARM_V2)
FORMAT_V5 = "notebook-rl-proof-prereg-v5"
ARM_V3 = "FT-NOTE-V3"
ARMS_V5 = ("F0", ARM_V3)
FORMAT_V6 = "notebook-rl-proof-prereg-v6"  # error-note version 4 (RL + note SFT): arms F0 and FT-HYB-V4
ARM_V4 = "FT-HYB-V4"
ARMS_V6 = ("F0", ARM_V4)
FORMAT_V7 = "notebook-rl-proof-prereg-v7"  # single-response RL + SFT campaign: arms F0 and FT-SINGLE
FORMAT_V8 = "notebook-rl-proof-prereg-v8"  # v7 arms; one 900-item run per arm (subset "all")
ARM_SINGLE = "FT-SINGLE"
ALL_ARMS = ARMS + (ARM_V2, ARM_V3, ARM_V4, ARM_SINGLE)
FINAL_ARMS = {FORMAT_V5: ARM_V3, FORMAT_V6: ARM_V4, FORMAT_V7: ARM_SINGLE, FORMAT_V8: ARM_SINGLE}  # one checkpoint declared before a single 900-item pass
# final arm -> (preregistration tag, export tag, checkpoint kind, gate format, gate result key)
FINAL_TAGS = {ARM_V3: ("v5", "v3", "NOTE", "error-note-final-gate-v3", "ft_note_v3"),
              ARM_V4: ("v6", "v4", "hybrid", "error-note-final-gate-v4", "ft_hyb_v4"),
              ARM_SINGLE: ("v7", "single", "single-campaign", "error-note-final-gate-single-v1", "ft_single")}
SINGLE_PILOT = (19, 300)  # single_campaign: the pilot ends after slot 19 = the first 300 items
B2_CORRECT = 558
VLLM_INPROCESS_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"
# subset tag -> (file version, B2 key it must equal)
SUBSETS = {"wrong": (SUBSET_VERSION, "wrong_ids"), "rest": (REST_SUBSET_VERSION, "right_ids")}
# v8 (user 2026-11-06): one 900-item run per arm over the sorted original IDs instead of wrong first, then rest.
# SUBSETS stays the v3-v7 shard pair; SUBSET_FILES adds the one-run subset file.
ALL_SUBSET_VERSION = "proof-eval-subset-all-v1"
SUBSET_FILES = SUBSETS | {"all": (ALL_SUBSET_VERSION, "all_ids")}
SEEDS = (3407, 3408, 3409)
# 300-item gate (claudedocs/research_self_note_finetuning_plan_20261101.md 6.3), fixed before any run
GATE_PASS_DIFF = 6.5
GATE_COLLAPSE_TOKEN_RATIO = 0.5
GATE_COLLAPSE_COUNT_RATIO = 1.10
# gate-62 (claudedocs/plan_self_note_v2_20261102.md section 1): one seed, wrong + rest = 900 items, vs B2's 558
GATE_62_VARIANT = "mmmu-val-v8-8k-seed3407-v2"
GATE_62_NOTE = ("one run per arm: a 900-item accuracy near 62% has a run-to-run standard error of about 1.6 points "
                "(sqrt(0.62 * 0.38 / 900)), so exceeding 558 by 1-2 points (9-18 items) is not distinguishable from "
                "chance; PASS is not proof of improvement. The run also differs from B2 in the 8192-token cap, the "
                "262144..1310720 image budget, the slots and the GPU")
# The evaluator's summary status of a run that finished every planned item: full runs end COMPLETE, partial
# (--sample-id) runs end PARTIAL; anything else (INCOMPLETE, RUNNING, absent) is not finished.
FINISHED = {"course": "COMPLETE", "statistical": "PARTIAL"}
PARSER = "team-final-answer-v8"
COURSE = "mmmu-val-v8"
# variant id -> (config file, seed, max concurrent requests, role)
VARIANTS = {COURSE: (ROOT / "configs/eval/mmmu_val_v8.yaml", 3407, 2, "course")} | {
    f"mmmu-val-v8-8k-seed{s}-v2": (HERE / f"eval_configs/mmmu_val_v8_8k_seed{s}.yaml", s, 32, "statistical")
    for s in SEEDS}
B2_PREDICTIONS = ROOT / "analysis_exports/mmmu_20260924/team_final_v8/official_b2/predictions.jsonl"
BASE_MODEL_REF = ROOT / "manifests/models/baseline.json"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 3407
DISCIPLINES = {
    "Art & Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health & Medicine": ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine",
                          "Pharmacy", "Public_Health"],
    "Humanities & Social Science": ["History", "Literature", "Psychology", "Sociology"],
    "Tech & Engineering": ["Agriculture", "Architecture_and_Engineering", "Computer_Science", "Electronics",
                           "Energy_and_Power", "Materials", "Mechanical_Engineering"],
}
SINGLE_ANSWER_RULE = (
    "Each item is scored from the one answer generated once per (arm, protocol variant, subset). No best-of-N, "
    "majority vote, reranking, verifier/feedback retry, ensemble, or per-item seed choice. Seeds are "
    "combined only as the mean of per-seed correctness. Each arm has exactly one evaluated checkpoint. "
    "Repeated runs of the same (arm, variant, subset) are refused, never chosen between. An incomplete run is "
    "resumed in place (same run_id) or reported as failed; it is never replaced because of its score.")


def _subject(sample_id: str) -> str:
    for subject in sorted(SUBJECTS, key=len, reverse=True):
        if sample_id.startswith(f"validation_{subject}_"):
            return subject
    raise ValueError(f"proof_eval: unknown MMMU subject for {sample_id}")


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _write_once(path: Path, text: str) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"proof_eval: refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _scorer_files() -> dict[str, str]:
    return {_rel(p): sha256_file(p) for p in scoring_source_files(PARSER)}


# ---------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------

def export_for_eval(checkpoint: Path, binding_file: Path, base: Path, output: Path, code_commit: str, *,
                    _test_policy_factory=None, _test_base_manifest: Path | None = None) -> dict:
    """Merge one checkpoint to BF16 HF weights plus a ``kind: merged`` eval model-ref manifest."""
    from fine_tuning.notebook_rl.checkpoint import read_metadata
    from fine_tuning.notebook_rl.export import export_merged, validate_export_identity

    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("proof_eval: --code-commit must be the full 40-hex training commit")
    binding, plan = _binding_file(binding_file)
    checkpoint, output = Path(checkpoint), Path(output)
    metadata = read_metadata(checkpoint, binding=binding)
    version = metadata["protocol"].get("version")
    v3_protocol = metadata["protocol"] if version in ("error-note-protocol-v3", "error-note-protocol-v4") else None
    if v3_protocol is not None:
        expected = _expected_training_protocol(v3_protocol)
        if v3_protocol != expected or metadata["step"] != expected["optimizer"]["total_steps"]:
            tag = "v3" if version == "error-note-protocol-v3" else "v4"
            raise ValueError(f"proof_eval: {tag} export requires the final BF16 checkpoint of its fixed protocol")
    campaign = _single_campaign_export(metadata, plan, checkpoint) if plan is not None or version == _single_version() else None
    if campaign is not None:
        v3_protocol = metadata["protocol"]
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "model"
    export = export_merged(checkpoint, Path(base), model_dir, metadata["arm"], binding=binding,
                           _test_policy_factory=_test_policy_factory, _test_base_manifest=_test_base_manifest)
    identity = validate_export_identity(model_dir, metadata["arm"])
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(model_dir.iterdir())]
    model_ref = {
        "kind": "merged", "base": {"id": MODEL_ID, "revision": MODEL_REVISION}, "dtype": "bfloat16",
        "processor_revision": MODEL_REVISION, "usage": "evaluation_only",
        "train_config_sha256": metadata["resolved_protocol_sha256"],
        "data_manifest_sha256": binding["data_sha256"], "code_commit": code_commit,
        "artifact_revision": sha256_file(model_dir / "export_manifest.json"),
        "files": files,
        "export": {
            "arm": metadata["arm"], "step": metadata["step"], "vision_on": metadata["vision_on"],
            "binding": binding, "export_manifest_sha256": identity["manifest_sha256"],
            "checkpoint_files": export["checkpoint_files"],
            "checkpoint_state_sha256": digest(export["checkpoint_state_sha256"]),
            "mergers": export["mergers"], "base_manifest_sha256": export["base_manifest_sha256"],
            "precision_policy": export["precision_policy"],
            "unmerged_inference_bit_equivalent": export["precision_policy"]["unmerged_inference_bit_equivalent"],
            **({"training_protocol": v3_protocol} if v3_protocol is not None else {}),
            **({"campaign": campaign} if campaign is not None else {}),
        },
    }
    _write_once(output / "model_ref.json", json.dumps(model_ref, indent=2, sort_keys=True) + "\n")
    return model_ref


def _single_version() -> str:
    from fine_tuning.error_note.protocol import VERSION_SINGLE

    return VERSION_SINGLE


def _binding_file(path: Path) -> tuple[dict, dict | None]:
    """(binding, plan): a binding JSON, a launch.json holding 'binding', or a single campaign's sealed
    campaign-plan.json, whose checkpoint binding is single_campaign._binding(plan) (plan returned, else None)."""
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(loaded, dict) and str(loaded.get("version")).startswith("single-campaign-"):
        from fine_tuning.error_note import single_campaign

        single_campaign._plan_check(loaded)
        return single_campaign._binding(loaded), loaded
    return (loaded["binding"] if isinstance(loaded, dict) and "binding" in loaded else loaded), None


def _single_stage(slot: int, consumed: int, last: int) -> str:
    """pilot (slot 19, 300 items), final (the last slot) or stopped: a main checkpoint after the pilot of a
    campaign the user ended early (2026-11-06, length collapse), evaluated once like a final one."""
    if slot == last:
        return "final"
    if (slot, consumed) == SINGLE_PILOT:
        return "pilot"
    if SINGLE_PILOT[0] < slot < last:
        return "stopped"
    raise ValueError("proof_eval: single-response export requires the pilot (slot 19, 300 items) or the final "
                     "campaign checkpoint, or the newest checkpoint of a stopped main run")


def _single_campaign_export(metadata: dict, plan: dict | None, checkpoint: Path) -> dict:
    """The pilot, final or stopped checkpoint of the one campaign whose sealed plan binds it. A stopped one must
    be the campaign's newest committed checkpoint (campaign-state.json beside it), so no earlier one is picked."""
    from fine_tuning.error_note import single_campaign
    from fine_tuning.error_note.order import _ids_sha256

    if plan is None or metadata["protocol"] != plan["protocol"]:
        raise ValueError("proof_eval: a single-response export requires --binding <its campaign-plan.json>")
    cursor = metadata["cursor"].get("single", {})
    single_campaign._cursor(plan, cursor, cursor.get("policy_sha256"))
    slot, consumed, last = cursor["slot"], cursor["consumed_problem_ids"], len(plan["blocks"])
    stage = _single_stage(slot, len(consumed), last)
    if metadata["step"] != slot:
        raise ValueError("proof_eval: single-response checkpoint step differs from its campaign slot")
    if stage == "stopped":
        state = json.loads((Path(checkpoint).parent / "campaign-state.json").read_text(encoding="utf-8"))
        if (state.get("plan_sha256") != plan["sha256"] or state["cursor"]["slot"] != slot or state.get("pending")
                or state["checkpoint"]["file_sha256"] != sha256_file(Path(checkpoint) / "metadata.json")):
            raise ValueError("proof_eval: a stopped single-response export must be the campaign's newest checkpoint")
    return {"version": plan["version"], "plan_sha256": plan["sha256"], "stage": stage,
            "slot": slot, "updates": cursor["updates"], "consumed_ids_sha256": _ids_sha256(consumed),
            "policy_sha256": cursor["policy_sha256"]}


def _expected_training_protocol(protocol: dict) -> dict:
    """The error-note v3/v4 protocol recomputed from its recorded parameters (precision bf16)."""
    from fine_tuning.error_note.protocol import plain_protocol_v3, plain_protocol_v4

    contract = protocol.get("training_contract", {})
    steps = protocol.get("optimizer", {}).get("total_steps")
    if protocol.get("version") == "error-note-protocol-v4":
        return plain_protocol_v4("bf16", steps, arm=contract.get("arm"), round_index=contract.get("round"),
                                 start=contract.get("start"), limit=contract.get("limit"),
                                 contracts=contract.get("hashes", {}))
    return plain_protocol_v3("bf16", steps, arm=contract.get("arm"), contracts=contract.get("hashes", {}))


def _v4_model_ref(path: Path) -> tuple[dict, dict]:
    """The declared final hybrid export of error-note v4 (its last round), validated like the v3 NOTE export."""
    from fine_tuning.error_note.protocol import ARM

    ref = json.loads(Path(path).read_text(encoding="utf-8"))
    export = ref.get("export", {})
    protocol = export.get("training_protocol", {})
    contract = protocol.get("training_contract", {})
    if protocol.get("version") != "error-note-protocol-v4":
        raise ValueError("proof_eval: v6 evaluation requires an error-note-protocol-v4 export")
    expected = _expected_training_protocol(protocol)
    binding = export.get("binding", {})
    if (ref.get("kind") != "merged" or ref.get("dtype") != "bfloat16" or not ref.get("files")
            or ref.get("base") != {"id": MODEL_ID, "revision": MODEL_REVISION}
            or ref.get("processor_revision") != MODEL_REVISION or export.get("arm") != ARM
            or contract.get("arm") != "hybrid" or protocol != expected
            or ref.get("train_config_sha256") != digest(protocol)
            or binding.get("protocol_sha256") != protocol["sha256"]
            or binding.get("prompts_sha256") != protocol["prompts_sha256"]
            or binding.get("data_sha256") != contract["hashes"]["rows"]
            or ref.get("data_manifest_sha256") != binding["data_sha256"]
            or export.get("step") != protocol["optimizer"]["total_steps"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(export.get("checkpoint_files", {}).get("metadata.json", "")))):
        raise ValueError("proof_eval: v6 evaluation requires its one final BF16 hybrid export and matching protocol binding")
    declared = {"model_ref": str(Path(path).resolve()), "model_ref_sha256": sha256_file(path),
                "model_sha256": digest(ref["files"]), "checkpoint_metadata_sha256": export["checkpoint_files"]["metadata.json"],
                "train_config_sha256": ref["train_config_sha256"], "round": contract["round"]}
    return ref, declared


def _final_model_ref(path: Path, fmt: str) -> tuple[dict, dict]:
    return {FORMAT_V6: _v4_model_ref, FORMAT_V7: _single_model_ref, FORMAT_V8: _single_model_ref}.get(fmt, _v3_model_ref)(path)


def _single_model_ref(path: Path) -> tuple[dict, dict]:
    """The declared final export of one single-response campaign (its last slot), validated like the v4 export."""
    from fine_tuning.error_note import single
    from fine_tuning.error_note.protocol import ARM

    ref = json.loads(Path(path).read_text(encoding="utf-8"))
    export = ref.get("export", {})
    protocol, campaign, binding = (export.get(key, {}) for key in ("training_protocol", "campaign", "binding"))
    if protocol.get("version") != _single_version():
        raise ValueError("proof_eval: v7 evaluation requires an error-note-single-response export")
    single._protocol_check(protocol)
    if (ref.get("kind") != "merged" or ref.get("dtype") != "bfloat16" or not ref.get("files")
            or ref.get("base") != {"id": MODEL_ID, "revision": MODEL_REVISION}
            or ref.get("processor_revision") != MODEL_REVISION or export.get("arm") != ARM
            or export.get("step") != campaign.get("slot") or campaign.get("stage") not in ("final", "stopped")
            or type(campaign.get("slot")) is not int
            or campaign["stage"] != _single_stage(campaign["slot"], 0, protocol["training_contract"]["total_slots"])
            or ref.get("train_config_sha256") != digest(protocol)
            or binding.get("protocol_sha256") != protocol["sha256"]
            or binding.get("prompts_sha256") != protocol["prompts_sha256"]
            or binding.get("data_sha256") != campaign.get("plan_sha256")
            or ref.get("data_manifest_sha256") != binding["data_sha256"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(export.get("checkpoint_files", {}).get("metadata.json", "")))):
        raise ValueError("proof_eval: v7 evaluation requires its one final BF16 single-campaign export and matching "
                         "protocol binding")
    declared = {"model_ref": str(Path(path).resolve()), "model_ref_sha256": sha256_file(path),
                "model_sha256": digest(ref["files"]), "checkpoint_metadata_sha256": export["checkpoint_files"]["metadata.json"],
                "train_config_sha256": ref["train_config_sha256"], "campaign_plan_sha256": campaign["plan_sha256"]}
    return ref, declared


def _v3_model_ref(path: Path) -> tuple[dict, dict]:
    """Validate the declared final NOTE export's metadata; evaluator verify_model checks the actual model files."""
    from fine_tuning.error_note.protocol import ARM, plain_protocol_v3

    ref = json.loads(Path(path).read_text(encoding="utf-8"))
    export = ref.get("export", {})
    protocol = export.get("training_protocol", {})
    contract = protocol.get("training_contract", {})
    expected = plain_protocol_v3("bf16", protocol.get("optimizer", {}).get("total_steps"),
                                 arm=contract.get("arm"), contracts=contract.get("hashes", {}))
    binding = export.get("binding", {})
    if (ref.get("kind") != "merged" or ref.get("dtype") != "bfloat16" or not ref.get("files")
            or ref.get("base") != {"id": MODEL_ID, "revision": MODEL_REVISION}
            or ref.get("processor_revision") != MODEL_REVISION or export.get("arm") != ARM
            or contract.get("arm") != "note" or protocol != expected
            or ref.get("train_config_sha256") != digest(protocol)
            or binding.get("protocol_sha256") != protocol["sha256"]
            or binding.get("prompts_sha256") != protocol["prompts_sha256"]
            or binding.get("data_sha256") != contract["hashes"]["rows"]
            or ref.get("data_manifest_sha256") != binding["data_sha256"]
            or export.get("step") != protocol["optimizer"]["total_steps"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(export.get("checkpoint_files", {}).get("metadata.json", "")))):
        raise ValueError("proof_eval: v3 evaluation requires its one final BF16 NOTE export and matching protocol binding")
    declared = {"model_ref": str(Path(path).resolve()), "model_ref_sha256": sha256_file(path),
                "model_sha256": digest(ref["files"]), "checkpoint_metadata_sha256": export["checkpoint_files"]["metadata.json"],
                "train_config_sha256": ref["train_config_sha256"]}
    return ref, declared


def verify_v3_export(model_ref: Path, checkpoint: Path, binding_file: Path, prereg_path: Path) -> None:
    """Before ledger START, connect the declared model reference to the pinned source checkpoint (v5 or v6)."""
    from fine_tuning.notebook_rl.checkpoint import read_metadata
    from fine_tuning.notebook_rl.export import validate_export_reference

    prereg, _ = load_preregistration(prereg_path)
    if prereg["format"] not in FINAL_ARMS:
        raise ValueError("proof_eval: export verification needs a v5, v6 or v7 preregistration")
    arm = FINAL_ARMS[prereg["format"]]
    tag = FINAL_TAGS[arm][1]
    ref, declared = _final_model_ref(model_ref, prereg["format"])
    if any(prereg["arms"][arm].get(k) != v for k, v in declared.items() if k != "model_ref"):
        raise ValueError(f"proof_eval: {tag} export differs from the declared final checkpoint")
    binding, plan = _binding_file(binding_file)
    metadata = read_metadata(checkpoint, binding=binding)
    manifest = validate_export_reference(Path(model_ref).parent / "model", ref)
    if (sha256_file(Path(checkpoint) / "metadata.json") != declared["checkpoint_metadata_sha256"]
            or metadata["protocol"] != ref["export"]["training_protocol"] or metadata["binding"] != ref["export"]["binding"]
            or metadata["resolved_protocol_sha256"] != declared["train_config_sha256"]
            or metadata["arm"] != ref["export"]["arm"] or metadata["step"] != ref["export"]["step"]
            or manifest["checkpoint_files"] != {p.name: sha256_file(p) for p in Path(checkpoint).iterdir()}
            or manifest["checkpoint_state_sha256"] != metadata["state_sha256"]
            or arm == ARM_SINGLE and _single_campaign_export(metadata, plan, checkpoint) != ref["export"]["campaign"]):
        raise ValueError(f"proof_eval: {tag} export checkpoint metadata or binding differs")


# ---------------------------------------------------------------------------------------------
# 8k run wrapper: in-process registration of exactly three IDs; src/ files stay untouched
# ---------------------------------------------------------------------------------------------

V8 = "mmmu-val-v8"
# The only differences of a registered variant from its mmmu-val-v8 twin besides protocol_id and seed.
# Slots = concurrent requests = vLLM max_num_seqs. 32 is the batch_size of the preregistered configs (their
# config_digest) and the default; a run may execute more via MMDL_EVAL_SLOTS (resolve_slots), recorded in its
# registration. analyze/gate refuse to compare runs of different slots or GPUs.
FAST_SLOTS = 32
SLOTS_ENV = "MMDL_EVAL_SLOTS"
# auto_slots inputs (the backend fixes gpu_memory_utilization at 0.90, vllm_backend.py)
GPU_MEMORY_UTILIZATION = 0.90
WEIGHTS_GIB = 8.68  # model weights
OTHER_VLLM_GIB = 5.0  # other vLLM memory, about
KV_TOKENS_PER_GIB = 7280  # KV-cache tokens per GiB
MEAN_CONTEXT_TOKENS = 4800  # time-weighted mean context of a running request
SLOT_STEP = 8
FAST_MAX_NEW_TOKENS = 8192
FAST_PIXELS = {"min_pixels": 262144, "max_pixels": 1310720}  # evaluation budget; the RL path (local_profile) keeps its own, smaller max_pixels
FAST_MAX_MODEL_LEN = 24576
REGISTERED = {vid: seed for vid, (_, seed, _, role) in VARIANTS.items() if role == "statistical"}
PATCHED_SRC = ("src/mmdl/runtime/contracts.py", "src/mmdl/evaluation/backends/vllm_backend.py",
               "src/mmdl/evaluation/cli.py", "src/mmdl/evaluation/engine.py")
REGISTRATION_FILE = "variant_registration.json"


def auto_slots(total_gib: float) -> int:
    """Slots a GPU of ``total_gib`` GiB can feed: KV memory left after weights and other vLLM memory, as running
    requests of the mean context, rounded down to a multiple of 8; never below the 32 of the preregistered configs."""
    kv_gib = GPU_MEMORY_UTILIZATION * total_gib - WEIGHTS_GIB - OTHER_VLLM_GIB
    return max(FAST_SLOTS, math.floor(kv_gib * KV_TOKENS_PER_GIB / MEAN_CONTEXT_TOKENS / SLOT_STEP) * SLOT_STEP)


def gpu_info() -> tuple[str | None, int | None]:
    """(name, total MiB) of the first GPU from nvidia-smi; (None, None) when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        name, total = out.splitlines()[0].rsplit(",", 1)
        return name.strip(), int(total)
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None, None


def resolve_slots() -> dict:
    """Slots of this run-eval process, resolved once: MMDL_EVAL_SLOTS = positive int | "auto" | unset (= 32)."""
    raw = os.environ.get(SLOTS_ENV)
    name, total_mib = gpu_info()
    if raw is None:
        slots, source = FAST_SLOTS, "default"
    elif raw == "auto":
        if total_mib is None:
            raise ValueError(f"proof_eval: {SLOTS_ENV}=auto needs nvidia-smi to report the GPU memory")
        slots, source = auto_slots(total_mib / 1024), "auto"
    elif re.fullmatch(r"[1-9][0-9]*", raw):
        slots, source = int(raw), "env"
    else:
        raise ValueError(f"proof_eval: {SLOTS_ENV} must be a positive integer or 'auto', not {raw!r}")
    return {"slots": slots, "slots_source": source, "gpu_total_mib": total_mib, "gpu_name": name}


def variant_registration(protocol_id: str, resolved: dict | None = None) -> dict:
    """What run-eval registers and the exact src/ + proof_eval bytes it patched around.

    ``resolved`` = resolve_slots() (default: resolved now); its slots and GPU facts are part of the registration."""
    from mmdl.runtime import contracts

    resolved = resolve_slots() if resolved is None else resolved
    inprocess = os.environ.get(VLLM_INPROCESS_ENV)
    return {
        **({"vllm_env": {VLLM_INPROCESS_ENV: inprocess}} if inprocess is not None else {}),
        "format": "proof-eval-variant-registration-v1", "protocol_id": protocol_id,
        "protocol_file_sha256": sha256_file(VARIANTS[protocol_id][0]),
        **resolved,
        "registered": {pid: {"seed": seed, "parser": PARSER, "max_new_tokens": FAST_MAX_NEW_TOKENS,
                             "control_images": _fast_images(contracts),
                             "execution_profile": _fast_execution(contracts, resolved["slots"])}
                       for pid, seed in REGISTERED.items()},
        "patched": {
            "mmdl.runtime.contracts.validate_configs": "registered IDs: exact seed, config batch_size 32, 8192 output "
                                                       "tokens, 262144..1310720 pixels, 24576 context, V8 parser; all "
                                                       "else validated as the mmmu-val-v8 twin; other IDs unchanged",
            "mmdl.evaluation.backends.vllm_backend.VLLMBackend.__init__": "registered IDs: slot check sees 2, vLLM "
                                                                          "max_num_seqs=batch_size=slots; other IDs unchanged"},
        "src_files": {p: sha256_file(ROOT / p) for p in PATCHED_SRC},
        "proof_eval_sha256": sha256_file(Path(__file__)),
        "entry_point": "python -m fine_tuning.notebook_rl.proof_eval run-eval <scripts/eval.sh arguments>; "
                       "command.json reproduce_command (plain eval.sh) does not reproduce this run",
    }


def _record_registration(run_dir: Path, registration: dict) -> None:
    if not run_dir.is_dir():
        raise RuntimeError("proof_eval: run directory absent at backend start; evaluator structure changed")
    path = run_dir / REGISTRATION_FILE
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != registration:
            raise ValueError("proof_eval: resume with a different variant registration, src or proof_eval code")
        return
    _write_once(path, json.dumps(registration, indent=2, sort_keys=True) + "\n")


def _fast_images(contracts) -> dict:
    return dict(contracts.CONTROL_IMAGES[V8], **FAST_PIXELS)


def _fast_execution(contracts, slots: int = FAST_SLOTS) -> dict:
    return dict(contracts.EXECUTION_PROFILES[V8], batch_size=slots, max_model_len=FAST_MAX_MODEL_LEN)


@contextmanager
def registered_variants(on_backend: Callable[[], None] | None = None, slots: int = FAST_SLOTS) -> Iterator[None]:
    """Register exactly the 8k IDs in this process; tables and functions are restored on exit.

    ``slots`` is what the backend executes. Registered configs keep batch_size 32 (their preregistered
    config_digest, so the resolved config of every run stays the preregistered one): the validator accepts
    exactly that and the backend is widened to ``slots``."""
    from mmdl.evaluation.backends import vllm_backend
    from mmdl.runtime import contracts

    if any(pid in contracts.CONTROL_IMAGES or pid in contracts.EXECUTION_PROFILES for pid in REGISTERED):
        raise RuntimeError("proof_eval: 8k IDs already registered (nested use or src changed)")
    original_validate = contracts.validate_configs
    original_init = vllm_backend.VLLMBackend.__init__

    def validate_configs(cfg, hw):
        pid = cfg.get("protocol_id")
        if pid not in REGISTERED:
            return original_validate(cfg, hw)
        seed, tokens = cfg["generation"].get("seed"), cfg["generation"].get("max_new_tokens")
        if (type(seed) is not int or seed != REGISTERED[pid] or cfg.get("parser") != PARSER
                or type(tokens) is not int or tokens != FAST_MAX_NEW_TOKENS
                or cfg.get("image") != contracts.CONTROL_IMAGES[pid]
                or cfg["execution"] != contracts.EXECUTION_PROFILES[pid]):
            raise ValueError(f"proof_eval: {pid} requires seed {REGISTERED[pid]}, config batch_size {FAST_SLOTS} "
                             f"(this run executes {slots} slots), 8192 output tokens, 262144..1310720 pixels, "
                             "24576 context and the V8 parser")
        twin = copy.deepcopy(cfg)
        twin["protocol_id"], twin["generation"]["seed"], twin["generation"]["max_new_tokens"] = V8, 3407, 32768
        twin["image"] = dict(contracts.CONTROL_IMAGES[V8])
        twin["execution"] = dict(contracts.EXECUTION_PROFILES[V8])
        return original_validate(twin, hw)

    def init(self, model_path, cfg, hw, kind="base", base_path=None):
        if cfg.get("protocol_id") not in REGISTERED:
            return original_init(self, model_path, cfg, hw, kind, base_path)
        validate_configs(cfg, hw)
        if on_backend is not None:
            on_backend()
        view = copy.deepcopy(cfg)
        view["execution"]["batch_size"] = 2
        real = vllm_backend._vllm

        def widened():
            llm_cls, sampling_cls, version = real()

            def llm(**kwargs):
                if kwargs.get("max_num_seqs") != 2:
                    raise RuntimeError("proof_eval: unexpected VLLMBackend structure; refusing to widen slots")
                return llm_cls(**kwargs | {"max_num_seqs": slots})
            return llm, sampling_cls, version

        vllm_backend._vllm = widened
        try:
            original_init(self, model_path, view, hw, kind, base_path)
        finally:
            vllm_backend._vllm = real
        if self.batch_size != 2 or self.cfg is not view:
            raise RuntimeError("proof_eval: unexpected VLLMBackend structure after load")
        self.cfg, self.batch_size = cfg, slots

    try:
        for pid in REGISTERED:
            contracts.CONTROL_IMAGES[pid] = _fast_images(contracts)
            contracts.EXECUTION_PROFILES[pid] = _fast_execution(contracts)
        setattr(contracts, "validate_configs", validate_configs)
        setattr(vllm_backend.VLLMBackend, "__init__", init)
        yield
    finally:
        setattr(vllm_backend.VLLMBackend, "__init__", original_init)
        setattr(contracts, "validate_configs", original_validate)
        for pid in REGISTERED:
            contracts.CONTROL_IMAGES.pop(pid, None)
            contracts.EXECUTION_PROFILES.pop(pid, None)


def run_eval(argv: list[str]) -> int:
    """Same argv as scripts/eval.sh, for one of the three 8k variant files only."""
    from mmdl.runtime.artifacts import default_path

    pre = argparse.ArgumentParser(prog="proof_eval run-eval", add_help=False)
    pre.add_argument("--protocol", type=Path, required=True)
    pre.add_argument("--run-id", required=True)
    pre.add_argument("--artifact-root", type=Path, default=default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts"))
    known, _ = pre.parse_known_args(argv)
    protocol = known.protocol if known.protocol.is_absolute() else ROOT / known.protocol
    pid = yaml.safe_load(protocol.read_text(encoding="utf-8")).get("protocol_id")
    if pid not in REGISTERED or sha256_file(protocol) != sha256_file(VARIANTS[pid][0]):
        raise ValueError("proof_eval: run-eval runs only the three preregistered 8k variant files; "
                         "the c2 course run (mmmu-val-v8) uses plain scripts/eval.sh")
    artifact_root = known.artifact_root if known.artifact_root.is_absolute() else ROOT / known.artifact_root
    run_dir = artifact_root / "runs" / known.run_id
    resolved = resolve_slots()
    registration = variant_registration(pid, resolved)
    print(f"proof_eval: {resolved['slots']} slots ({resolved['slots_source']}) on {resolved['gpu_name']} "
          f"{resolved['gpu_total_mib']} MiB", flush=True)
    from mmdl.evaluation import cli

    cwd, saved_argv = Path.cwd(), sys.argv
    os.chdir(ROOT)  # scripts/eval.sh runs the CLI from the repository root
    sys.argv = ["mmdl.evaluation.cli", *argv]
    try:
        with registered_variants(on_backend=lambda: _record_registration(run_dir, registration),
                                 slots=resolved["slots"]):
            cli.main()
    finally:
        sys.argv = saved_argv
        os.chdir(cwd)
        if run_dir.is_dir() and not (run_dir / REGISTRATION_FILE).exists():
            _write_once(run_dir / REGISTRATION_FILE, json.dumps(registration, indent=2, sort_keys=True) + "\n")
    return 0


# ---------------------------------------------------------------------------------------------
# preregistration
# ---------------------------------------------------------------------------------------------

def _b2_split(path: Path) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    subjects: dict[str, int] = defaultdict(int)
    for sid in ids:
        subjects[_subject(sid)] += 1
    wrong = sorted(row["id"] for row in rows if row["correct"] is False)
    right = sorted(row["id"] for row in rows if row["correct"] is True)
    if (len(ids) != 900 or len(set(ids)) != 900 or set(subjects) != set(SUBJECTS)
            or any(v != 30 for v in subjects.values()) or (len(wrong), len(right)) != (342, 558)):
        raise ValueError("proof_eval: frozen B2 V8 reference must be 900 items, 558 right / 342 wrong")
    return {"source": _rel(Path(path)), "sha256": sha256_file(path), "scoring": PARSER,
            "wrong_ids": wrong, "right_ids": right, "all_ids": sorted(ids)}


def _subset(path: Path, b2: dict, tag: str = "wrong") -> dict:
    """An evaluation subset file: exactly the sorted frozen B2 V8 wrong ("wrong"), right ("rest") or all ("all") IDs
    of the same B2 file."""
    version, key = SUBSET_FILES[tag]
    path = Path(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    ids = loaded.get("sample_ids") if isinstance(loaded, dict) else None
    if (loaded.get("version") != version or not isinstance(ids, list) or ids != b2[key]
            or loaded.get("count") != len(ids) or loaded.get("source_sha256") != b2["sha256"]):
        raise ValueError(f"proof_eval: the {tag} subset must be the sorted frozen B2 V8 {key.removesuffix('_ids')} IDs "
                         "of the same B2 file")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "version": version, "count": len(ids),
            "source": loaded.get("source"), "source_sha256": loaded["source_sha256"], "ids": ids,
            "rule": f"B2 + team-final-answer-v8 {key.removesuffix('_ids')} items; evaluation only, never training "
                    "(CLAUDE.md §2.2)"}


def make_subset(kind: str, output: Path, b2_predictions: Path = B2_PREDICTIONS) -> dict:
    """Write the "wrong" (342), "rest" (558) or "all" (900) subset file from the frozen B2 file; write-once."""
    version, key = SUBSET_FILES[kind]
    b2 = _b2_split(b2_predictions)
    value = {"version": version, "purpose": f"MMMU-val evaluation subset ({kind}); evaluation only, never training",
             "source": b2["source"], "source_sha256": b2["sha256"], "count": len(b2[key]), "sample_ids": b2[key]}
    _write_once(output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def _variant_entry(vid: str) -> dict:
    path, seed, concurrency, role = VARIANTS[vid]
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (cfg["protocol_id"] != vid or cfg["generation"]["seed"] != seed or cfg["parser"] != PARSER
            or cfg["execution"]["batch_size"] != concurrency or cfg["status"] != "FROZEN"):
        raise ValueError(f"proof_eval: variant file disagrees with its declaration: {vid}")
    return {"path": _rel(path), "sha256": sha256_file(path), "config_digest": digest(cfg),
            "seed": seed, "concurrency": concurrency, "role": role}


def _paired_run_rules(n: int, m: int) -> list[str]:
    return [f"each (arm, variant, subset) run at most once with --mode partial and one --sample-id "
            f"per subset ID; status {FINISHED['statistical']}; run_manifest expected_ids, "
            f"identity.ids_sha256 and predictions are exactly the {n} (wrong) or {m} (rest) subset "
            "IDs; a stage runs only the requested seeds, the rest of the listed runs may never exist",
            "all arms of one variant share identity.protocol_sha256",
            "all statistical runs share identity.code_sha256",
            "every record seed equals sample_seed(variant seed, id)",
            "stored V8 correctness/parse_status must equal the re-score",
            "each arm evaluates one checkpoint; the arms evaluate different checkpoints"]


def _scorer() -> dict:
    return {"parser": PARSER, "files": _scorer_files(),
            "only_scorer": "strict team-final-answer-v8 re-score; proof_eval has no second (Qwen-style) "
                           "score: Qwen's published MMMU scoring uses an LLM-assisted judge, which is not "
                           "used without approval (CLAUDE.md §7.3)"}


def build_preregistration(subset: Path, rest_subset: Path, b2_predictions: Path = B2_PREDICTIONS) -> dict:
    variants = {vid: _variant_entry(vid) for vid, row in VARIANTS.items()
                if row[3] == "statistical"}  # the course run is not in the proof plan
    statistical = list(variants)
    runs = [{"arm": arm, "variant": v, "subset": tag, "role": "statistical"}
            for arm in ARMS for v in statistical for tag in SUBSETS]
    base_ref = json.loads(BASE_MODEL_REF.read_text(encoding="utf-8"))
    b2 = _b2_split(b2_predictions)
    items = {"wrong": _subset(subset, b2, "wrong"), "rest": _subset(rest_subset, b2, "rest")}
    n, m = items["wrong"]["count"], items["rest"]["count"]
    return {
        "format": FORMAT, "status": "FROZEN",
        "arms": {
            "F0": {"definition": f"untrained BF16 base {MODEL_ID}@{MODEL_REVISION}, re-run on the subsets "
                                 "(the stored B2 answers are not F0 and not the comparator)",
                   "model_ref": _rel(BASE_MODEL_REF), "model_sha256": digest(base_ref["files"])},
            "FT-NOTE": {"definition": "merged BF16 export of the single checkpoint trained with the note "
                                      "(proof_eval export); the primary arm, fixed in advance",
                        "model_ref": "bound at analysis by --ft-note-model-ref; all FT-NOTE runs share its model_sha256"},
            "FT-SOLUTION": {"definition": "merged BF16 export of the single checkpoint trained without the note "
                                          "(proof_eval export); a comparison arm that never replaces FT-NOTE",
                            "model_ref": "bound at analysis by --ft-solution-model-ref; all FT-SOLUTION runs share "
                                         "its model_sha256"},
            "FT-NOTE-3417": {"definition": "merged BF16 export of the single checkpoint trained with the note on all "
                                           "3,417 ordered items; run only after the 300-item gate passes",
                             "model_ref": "bound at analysis by --ft-note-3417-model-ref; all FT-NOTE-3417 runs share "
                                          "its model_sha256"},
        },
        "evaluation_subsets": items,
        "protocol_variants": variants,
        "runs": runs,
        "paired_run_rules": _paired_run_rules(n, m),
        "scorer": _scorer(),
        "primary_endpoint": {
            "statistic": f"wrong subset ({n} items): mean over the seeds both finished of (FT-NOTE strict V8 correct "
                         "count - F0 correct count); per-seed counts, gains, losses and exact McNemar are reported "
                         "for every finished (arm, seed)",
            "arm": "FT-NOTE", "comparator": "F0", "subset": "wrong", "variants": statistical, "items": n,
            "ci": {"method": "paired item bootstrap over the subset items, percentile",
                   "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "level": 0.95},
        },
        "gate_300": {
            "subset": "wrong", "arm": "FT-NOTE", "comparator": "F0", "seeds": "the seeds both arms finished",
            "mean_diff": "mean FT-NOTE correct - mean F0 correct over those seeds",
            "PASS": f"mean_diff >= {GATE_PASS_DIFF}, or 3 seeds finished and FT-NOTE > F0 in each (with fewer than 3 "
                    "seeds only the mean_diff branch can pass)",
            "AMBIGUOUS": "mean_diff > 0 otherwise", "STOP": "otherwise",
            "collapse_alert": f"FT-NOTE median generated tokens < {GATE_COLLAPSE_TOKEN_RATIO} x F0, or its no-parse "
                              f"or length-finish count > {GATE_COLLAPSE_COUNT_RATIO} x F0 (seed means)",
            "ft_solution": "reported beside FT-NOTE; never changes the status and never replaces FT-NOTE",
            "not_proof_of_improvement": True,
        },
        "sensitivity": {"mcnemar_exact_two_sided": {"pairs": "per-item correctness F0 vs arm on one subset, per seed "
                                                             "(every finished seed), seeds never pooled"}},
        "secondary_endpoints": {
            "generation_counts": "per (subset, arm, seed): no-parse count, length-finish count, median and mean "
                                 "generated tokens",
            "full_900": "correct count and accuracy of an (arm, seed) whose wrong and rest runs both finished",
            "per_discipline": "seed-mean accuracy per arm",
            "per_subject": "seed-mean accuracy per arm (table, no CI)",
        },
        "disciplines": DISCIPLINES,
        "single_answer_rule": SINGLE_ANSWER_RULE,
        "mmmu_val_budget": {"checkpoints_per_arm": 1, "runs": len(runs), "items_per_run": {"wrong": n, "rest": m},
                            "rule": "no other MMMU-val runs, seeds, items or checkpoints; MMMU-val never selects a "
                                    "checkpoint"},
        "course_facing_run": {"status": "DROPPED", "variant": COURSE,
                              "rule": "user decision 2026-10-31: the proof evaluates only the subsets; the course "
                                      "protocol code path is kept but no course run is preregistered"},
        "limitations": [
            f"the wrong subset ({n} items) was chosen because B2 answered them wrongly once; a re-run of the "
            "untrained base answers some of them correctly by sampling alone (regression to the mean), so the "
            "comparator is the paired F0 re-run on the same IDs and seeds, never B2's 0 correct",
            f"accuracy on one subset is not an MMMU-val accuracy and is not comparable to the 62.00% B2 baseline or "
            f"the official 67.4; a {n + m}-item accuracy exists only for an (arm, seed) whose wrong and rest runs "
            "both finished",
            "the 8k variants (8192 output tokens, 262144..1310720 pixels, 32 concurrent) differ from the course "
            "protocol (32768 tokens, official pixels, 2 concurrent)",
            "the 300-item gate is a threshold for spending more, not evidence of improvement",
        ],
        "notes": [
            "user decision 2026-10-31 (speed and cost of the RunPod proof): 8K output, 262144..1310720 pixels and 32 "
            "concurrent requests for all arms; the course-protocol run is dropped from the proof plan",
            "supersedes proof-prereg-v2 (format notebook-rl-proof-prereg-v2: arms F0/FT, 342 items, earlier 8k variants)",
        ],
        "b2_reference": b2,
    }


def build_preregistration_v4(subset: Path, rest_subset: Path, b2_predictions: Path = B2_PREDICTIONS) -> dict:
    """Error-note version 2: arms F0 and FT-V2, the existing 8k seed-3407 variant, wrong + rest (900 items)."""
    variants = {GATE_62_VARIANT: _variant_entry(GATE_62_VARIANT)}
    runs = [{"arm": arm, "variant": GATE_62_VARIANT, "subset": tag, "role": "statistical"}
            for arm in ARMS_V4 for tag in SUBSETS]
    base_ref = json.loads(BASE_MODEL_REF.read_text(encoding="utf-8"))
    b2 = _b2_split(b2_predictions)
    items = {"wrong": _subset(subset, b2, "wrong"), "rest": _subset(rest_subset, b2, "rest")}
    n, m = items["wrong"]["count"], items["rest"]["count"]
    return {
        "format": FORMAT_V4, "status": "FROZEN",
        "arms": {
            "F0": {"definition": f"untrained BF16 base {MODEL_ID}@{MODEL_REVISION}, re-run on the subsets "
                                 "(optional in this ledger; the stored B2 answers are not F0)",
                   "model_ref": _rel(BASE_MODEL_REF), "model_sha256": digest(base_ref["files"])},
            ARM_V2: {"definition": "merged BF16 export of the single checkpoint of error-note version 2 (v2 note "
                                   "without a corrected solution, one lesson-hinted gold-blind re-solve, protocol "
                                   "error-note-protocol-v2, item_order_v2.json, arm note); fixed in advance",
                     "model_ref": "bound at analysis by --ft-v2-model-ref; all FT-V2 runs share its model_sha256"},
        },
        "evaluation_subsets": items,
        "protocol_variants": variants,
        "runs": runs,
        "paired_run_rules": _paired_run_rules(n, m),
        "scorer": _scorer(),
        "gate_62": {
            "arm": ARM_V2, "variant": GATE_62_VARIANT, "items": n + m,
            "statistic": f"FT-V2 strict V8 correct count over wrong ({n}) + rest ({m}) runs of seed 3407",
            "PASS": f"correct > {B2_CORRECT} (the B2 + team-final-answer-v8 baseline, 62.00%)", "FAIL": "otherwise",
            "f0": "reported beside FT-V2 when this ledger has both F0 runs; never changes the status",
            "not_proof_of_improvement": True, "note": GATE_62_NOTE,
        },
        "disciplines": DISCIPLINES,
        "single_answer_rule": SINGLE_ANSWER_RULE,
        "mmmu_val_budget": {"checkpoints_per_arm": 1, "runs": len(runs), "items_per_run": {"wrong": n, "rest": m},
                            "rule": "no other MMMU-val runs, seeds, items or checkpoints; MMMU-val never selects a "
                                    "checkpoint"},
        "course_facing_run": {"status": "DROPPED", "variant": COURSE,
                              "rule": "user decision 2026-11-02: one 8k run of 900 items (wrong + rest), compared "
                                      "with B2's 558"},
        "limitations": [
            GATE_62_NOTE,
            "B2's 558 is a stored run of another protocol (32768 tokens, official pixels, 2 concurrent, RTX 3090); "
            "an F0 re-run under this variant is the paired comparator when it exists",
        ],
        "notes": ["user decision 2026-11-02: one run of 900 items, PASS iff above 62.00% (558); the 8k variant and "
                  "the 262144..1310720 image budget of the training are kept (no high-resolution variant)",
                  f"separate from {FORMAT} (its own ledger); arms of one ledger are never mixed with another's"],
        "b2_reference": b2,
    }


def build_preregistration_v5(subset: Path, rest_subset: Path, b2_predictions: Path, ft_model_ref: Path) -> dict:
    """Final v3 NOTE checkpoint, declared before its single 900-item seed-3407 evaluation."""
    _, declared = _v3_model_ref(ft_model_ref)
    value = build_preregistration_v5_shape(subset, rest_subset, b2_predictions, ARM_V3, declared,
                                           "single final BF16 NOTE checkpoint of error-note-protocol-v3, fixed before MMMU evaluation")
    value["format"] = FORMAT_V5
    value["course_facing_run"]["rule"] = "one final FT-NOTE-V3 8k pass over the 900 original IDs (wrong + rest)"
    value["notes"][0] = "one final NOTE checkpoint, one seed (3407), no additional MMMU seeds or checkpoint selection"
    return value


def build_preregistration_v6(subset: Path, rest_subset: Path, b2_predictions: Path, ft_model_ref: Path) -> dict:
    """Final v4 hybrid checkpoint (its last round), declared before its single 900-item seed-3407 evaluation."""
    _, declared = _v4_model_ref(ft_model_ref)
    value = build_preregistration_v5_shape(subset, rest_subset, b2_predictions, ARM_V4, declared,
                                           "single final BF16 hybrid checkpoint of error-note-protocol-v4 "
                                           "(RL + note SFT, last round), fixed before MMMU evaluation")
    value["format"] = FORMAT_V6
    value["course_facing_run"]["rule"] = "one final FT-HYB-V4 8k pass over the 900 original IDs (wrong + rest)"
    value["notes"][0] = "one final hybrid checkpoint, one seed (3407), no additional MMMU seeds or checkpoint selection"
    return value


def build_preregistration_v7(subset: Path, rest_subset: Path, b2_predictions: Path, ft_model_ref: Path) -> dict:
    """Final single-response campaign checkpoint, declared before its single 900-item seed-3407 evaluation."""
    _, declared = _single_model_ref(ft_model_ref)
    value = build_preregistration_v5_shape(subset, rest_subset, b2_predictions, ARM_SINGLE, declared,
                                           "single final BF16 checkpoint of the single-response RL + note SFT campaign "
                                           "(error-note-single-response-v3, all 3,500 items), fixed before MMMU evaluation")
    value["format"] = FORMAT_V7
    value["course_facing_run"]["rule"] = "one final FT-SINGLE 8k pass over the 900 original IDs (wrong + rest)"
    value["notes"][0] = "one final campaign checkpoint, one seed (3407), no additional MMMU seeds or checkpoint selection"
    value["notes"].append("historical 8k F0 = 541/900 (prereg-v4 ledger, 2026-11-02) is a display-only reference "
                          "(gate-62 --f0-reference 541), never paired or part of the rule")
    return value


def build_preregistration_v8(subset: Path, rest_subset: Path, all_subset: Path, b2_predictions: Path,
                             ft_model_ref: Path) -> dict:
    """v7 with one 900-item run per arm over the sorted original IDs (the wrong/rest files only bind the B2 split)."""
    value = build_preregistration_v7(subset, rest_subset, b2_predictions, ft_model_ref)
    items = {"all": _subset(all_subset, _b2_split(b2_predictions), "all")}
    value["format"] = FORMAT_V8
    value["evaluation_subsets"] = items
    value["runs"] = [{"arm": name, "variant": GATE_62_VARIANT, "subset": "all", "role": "statistical"}
                     for name in ("F0", ARM_SINGLE)]
    value["paired_run_rules"] = [rule.replace("exactly the 342 (wrong) or 558 (rest) subset IDs",
                                              "exactly the 900 sorted original IDs") for rule in value["paired_run_rules"]]
    value["mmmu_val_budget"]["items_per_run"] = {"all": 900}
    value["mmmu_val_budget"]["runs"] = len(value["runs"])
    value["course_facing_run"]["rule"] = "one final FT-SINGLE 8k pass over the 900 sorted original IDs in one run"
    value["notes"][1] = "one 900-item run per arm (user 2026-11-06: not the wrong subset first, then the rest)"
    return value


def build_preregistration_v5_shape(subset: Path, rest_subset: Path, b2_predictions: Path, arm: str, declared: dict,
                                   definition: str) -> dict:
    value = build_preregistration_v4(subset, rest_subset, b2_predictions)
    value["arms"] = {"F0": value["arms"]["F0"], arm: {"definition": definition, **declared}}
    value["runs"] = [{"arm": name, "variant": GATE_62_VARIANT, "subset": tag, "role": "statistical"}
                     for name in ("F0", arm) for tag in SUBSETS]
    value["gate_62"] = {"arm": arm, "variant": GATE_62_VARIANT, "items": 900,
                        "minimum_correct": 559, "not_proof_of_improvement": True}
    value["gate_67.4"] = {"arm": arm, "variant": GATE_62_VARIANT, "items": 900,
                          "minimum_correct": 607, "not_proof_of_improvement": True}
    value["notes"] = ["one final checkpoint, one seed (3407), no additional MMMU seeds or checkpoint selection",
                      "wrong and rest are disjoint storage shards of one 900-item evaluation, not a 342-item pilot gate",
                      "optional current F0 has the same protocol, code, registration and hardware; historical F0 is display only"]
    return value


def preregister(output: Path, subset: Path, rest_subset: Path, b2_predictions: Path = B2_PREDICTIONS,
                fmt: str = "v3", ft_v3_model_ref: Path | None = None, ft_v4_model_ref: Path | None = None,
                ft_single_model_ref: Path | None = None, all_subset: Path | None = None) -> str:
    """Write-once preregistration plus ``<output>.sha256``; v5/v6/v7 pin a previously exported final model reference."""
    output = Path(output)
    if (fmt == "v5") != (ft_v3_model_ref is not None):
        raise ValueError("proof_eval: --format v5 requires --ft-v3-model-ref; the flag is v5-only")
    if (fmt == "v6") != (ft_v4_model_ref is not None):
        raise ValueError("proof_eval: --format v6 requires --ft-v4-model-ref; the flag is v6-only")
    if (fmt in ("v7", "v8")) != (ft_single_model_ref is not None):
        raise ValueError("proof_eval: --format v7/v8 requires --ft-single-model-ref; the flag is v7/v8-only")
    if (fmt == "v8") != (all_subset is not None):
        raise ValueError("proof_eval: --format v8 requires --all-subset; the flag is v8-only")
    if all_subset is not None:
        value = build_preregistration_v8(subset, rest_subset, all_subset, b2_predictions, ft_single_model_ref)
    elif ft_single_model_ref is not None:
        value = build_preregistration_v7(subset, rest_subset, b2_predictions, ft_single_model_ref)
    elif ft_v4_model_ref is not None:
        value = build_preregistration_v6(subset, rest_subset, b2_predictions, ft_v4_model_ref)
    elif ft_v3_model_ref is not None:
        value = build_preregistration_v5(subset, rest_subset, b2_predictions, ft_v3_model_ref)
    else:
        value = {"v3": build_preregistration, "v4": build_preregistration_v4}[fmt](subset, rest_subset, b2_predictions)
    _write_once(output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    value = sha256_file(output)
    _write_once(output.with_name(output.name + ".sha256"), value + "\n")
    return value


def load_preregistration(path: Path) -> tuple[dict, str]:
    path = Path(path)
    value = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="ascii").strip() != value:
        raise ValueError("proof_eval: preregistration hash sidecar missing or mismatched")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    if prereg.get("format") not in (FORMAT, FORMAT_V4, *FINAL_ARMS) or prereg.get("status") != "FROZEN":
        raise ValueError("proof_eval: not a frozen proof preregistration")
    if prereg["scorer"]["files"] != _scorer_files():
        raise ValueError("proof_eval: V8 scorer source changed since preregistration")
    if prereg["format"] in FINAL_ARMS:
        final = FINAL_ARMS[prereg["format"]]
        tags = ("all",) if prereg["format"] == FORMAT_V8 else tuple(SUBSETS)
        expected_runs = [{"arm": arm, "variant": GATE_62_VARIANT, "subset": tag, "role": "statistical"}
                         for arm in ("F0", final) for tag in tags]
        subsets = prereg["evaluation_subsets"]
        b2_ids = sorted(prereg["b2_reference"]["wrong_ids"] + prereg["b2_reference"]["right_ids"])
        expected_ids = ({"all": (b2_ids, 900)} if prereg["format"] == FORMAT_V8 else
                        {"wrong": (prereg["b2_reference"]["wrong_ids"], 342), "rest": (prereg["b2_reference"]["right_ids"], 558)})
        if (set(prereg["arms"]) != {"F0", final} or prereg["runs"] != expected_runs
                or prereg["protocol_variants"] != {GATE_62_VARIANT: _variant_entry(GATE_62_VARIANT)}
                or set(subsets) != set(tags) or len(set(b2_ids)) != 900
                or any(subsets[tag]["ids"] != ids or subsets[tag]["count"] != n for tag, (ids, n) in expected_ids.items())
                or set().union(*(subsets[tag]["ids"] for tag in tags)) != set(b2_ids)):
            tag, _, kind, *_ = FINAL_TAGS[final]
            raise ValueError(f"proof_eval: {tag} requires one declared {kind} checkpoint, "
                             "seed 3407 and all 900 disjoint original IDs")
        for key in ("model_ref_sha256", "model_sha256", "checkpoint_metadata_sha256", "train_config_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(prereg["arms"][final].get(key, ""))):
                tag, _, kind, *_ = FINAL_TAGS[final]
                raise ValueError(f"proof_eval: {tag} final {kind} model reference is not bound")
    return prereg, value


# ---------------------------------------------------------------------------------------------
# run ledger: one START line per evaluation run, so analyze can see an omitted repeat
# ---------------------------------------------------------------------------------------------

LEDGER_FORMAT = "proof-eval-ledger-v2"
GENESIS = "0" * 64
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_EVENT_KEYS = {"START": {"arm", "variant", "subset", "run_id", "preregistration_sha256"}, "FAIL": {"run_id", "reason"}}
_COMMON_KEYS = {"format", "event", "prev", "time"}


def _check_event(events: list[dict], event: dict) -> None:
    """The ledger rules, applied to every line on read and to the new line before it is appended."""
    starts = {e["run_id"]: e for e in events if e["event"] == "START"}
    failed = {e["run_id"] for e in events if e["event"] == "FAIL"}
    if event["event"] == "FAIL":
        reason = event["reason"]
        if event["run_id"] not in starts or event["run_id"] in failed or type(reason) is not str or not reason.strip():
            raise ValueError(f"proof_eval: ledger FAIL needs a started, not yet failed run and a reason: {event['run_id']}")
        return
    if not _RUN_ID.fullmatch(str(event["run_id"])) or event["run_id"] in starts:
        raise ValueError(f"proof_eval: ledger run_id invalid or already started: {event['run_id']}")
    if event["arm"] not in ALL_ARMS or event["subset"] not in SUBSET_FILES:
        raise ValueError(f"proof_eval: ledger arm {event['arm']!r} or subset {event['subset']!r} is not known")
    if any(e["preregistration_sha256"] != event["preregistration_sha256"] for e in starts.values()):
        raise ValueError("proof_eval: one ledger serves one preregistration")
    key = (event["arm"], event["variant"], event["subset"])
    live = [r for r, e in starts.items() if (e["arm"], e["variant"], e["subset"]) == key and r not in failed]
    if live:
        raise ValueError(f"proof_eval: {'/'.join(key)} already started as {live[0]}; "
                         "a second start needs ledger-fail on the first (never replace a run by its score)")


def read_ledger(path: Path) -> tuple[list[dict], str]:
    """Validated events and the head hash; a broken chain, malformed line or rule violation raises."""
    return _parse(Path(path).read_bytes())


def _parse(raw: bytes) -> tuple[list[dict], str]:
    if raw and not raw.endswith(b"\n"):
        raise ValueError("proof_eval: ledger ends with a partial line")
    events: list[dict] = []
    head = GENESIS
    for line in raw.splitlines(keepends=True):
        event = json.loads(line)
        kind = event.get("event") if type(event) is dict else None
        if (kind not in _EVENT_KEYS or set(event) != _COMMON_KEYS | _EVENT_KEYS[kind]
                or event["format"] != LEDGER_FORMAT or event["prev"] != head):
            raise ValueError("proof_eval: ledger line malformed or hash chain broken")
        _check_event(events, event)
        events.append(event)
        head = hashlib.sha256(line).hexdigest()
    return events, head


def _append(path: Path, kind: str, *, single_attempt: bool = False, **fields: str) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600), "r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        events, head = _parse(stream.read())
        if single_attempt and any(e["event"] == "START" and
                                  all(e[key] == fields[key] for key in ("arm", "variant", "subset"))
                                  for e in events):
            raise ValueError("proof_eval: v5 forbids replacing a started run, including a failed run")
        event = {"format": LEDGER_FORMAT, "event": kind, "prev": head, "time": datetime.now(timezone.utc).isoformat(),
                 **fields}
        _check_event(events, event)
        stream.write(json.dumps(event, sort_keys=True).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event


def ledger_start(prereg_path: Path, arm: str, variant: str, run_id: str, ledger: Path, subset: str) -> dict:
    prereg, prereg_sha = load_preregistration(prereg_path)
    if (arm, variant, subset) not in {(r["arm"], r["variant"], r["subset"]) for r in prereg["runs"]}:
        raise ValueError(f"proof_eval: {(arm, variant, subset)} is not a preregistered run")
    return _append(ledger, "START", single_attempt=prereg["format"] in FINAL_ARMS,
                   arm=arm, variant=variant, subset=subset, run_id=run_id,
                   preregistration_sha256=prereg_sha)


def ledger_fail(ledger: Path, run_id: str, reason: str, artifact_root: Path) -> dict:
    """Record a started run as FAILED (crashed, abandoned); a finished run directory cannot be declared failed."""
    summary = Path(artifact_root) / "runs" / run_id / "summary.json"
    status = json.loads(summary.read_text(encoding="utf-8")).get("status") if summary.is_file() else None
    if status in FINISHED.values():
        raise ValueError(f"proof_eval: {run_id} is {status}; a finished run cannot be replaced by marking it failed")
    return _append(ledger, "FAIL", run_id=run_id, reason=reason)


# ---------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------

def load_run(path: Path, prereg: dict) -> dict:
    """Validate one eval run directory (or its predictions.jsonl) against the preregistration.

    A statistical run must be the partial run on exactly one preregistered subset (status PARTIAL); a course
    run, if a preregistration ever lists one, the full 900-item run (status COMPLETE)."""
    path = Path(path)
    run_dir = path.parent if path.is_file() else path
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    identity = manifest.get("identity", {})
    cfg = yaml.safe_load((run_dir / "resolved_eval_config.yaml").read_text(encoding="utf-8"))
    matches = [v for v, row in prereg["protocol_variants"].items() if row["config_digest"] == digest(cfg)]
    if len(matches) != 1:
        raise ValueError(f"proof_eval: {run_dir} protocol config is not preregistered")
    variant = prereg["protocol_variants"][matches[0]]
    if not {"protocol_sha256", "model_sha256", "code_sha256"} <= identity.keys():
        raise ValueError(f"proof_eval: {run_dir} is not an original evaluation run")
    if variant["role"] == "course":
        mode, subset = "full", None
        expected = sorted(set(prereg["b2_reference"]["wrong_ids"]) | set(prereg["b2_reference"]["right_ids"]))
    else:
        mode = "partial"
        subset = next((tag for tag, row in prereg["evaluation_subsets"].items()
                       if manifest.get("expected_ids") == sorted(row["ids"])), None)
        expected = sorted(prereg["evaluation_subsets"][subset or "wrong"]["ids"])
    if identity.get("mode") != mode:
        raise ValueError(f"proof_eval: {run_dir} is not an original {mode} evaluation run")
    if manifest.get("expected_ids") != expected or identity.get("ids_sha256") != digest(expected):
        raise ValueError(f"proof_eval: {run_dir} sample set differs from the preregistered subset IDs "
                         f"({len(expected)} expected)")
    status = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get("status")
    if status != FINISHED[variant["role"]]:
        raise ValueError(f"proof_eval: {run_dir} is {status}, not {FINISHED[variant['role']]}")
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError(f"proof_eval: {run_dir} must hold the {len(expected)} preregistered items exactly once")
    items = {}
    for row in rows:
        if row.get("_record_sha256") != digest({k: v for k, v in row.items() if k != "_record_sha256"}):
            raise ValueError(f"proof_eval: record integrity failure {row['id']}")
        if row["seed"] != sample_seed(variant["seed"], row["id"]):
            raise ValueError(f"proof_eval: record seed is not the preregistered seed {row['id']}")
        score = score_with_parser(row["raw_response"], row["question_type"], row["options"], row["answer"],
                                  PARSER, row.get("finish_reason"), question=row.get("question", ""))
        if score["correct"] != row["correct"] or score["parse_status"] != row["parse_status"]:
            raise ValueError(f"proof_eval: stored score differs from V8 re-score {row['id']}")
        items[row["id"]] = {"correct": score["correct"], "length": row.get("finish_reason") == "length",
                            "missing_final": score["parse_status"] != "PARSED", "tokens": row["generated_tokens"]}
    registration = run_dir / REGISTRATION_FILE
    return {"variant": matches[0], "seed": variant["seed"], "subset": subset, "run_dir": str(run_dir.resolve()),
            "identity": {k: identity.get(k) for k in ("protocol_sha256", "model_sha256", "code_sha256",
                                                     "git_commit", "job_id", "run_role", "mode", "ids_sha256")},
            "registration": json.loads(registration.read_text(encoding="utf-8")) if registration.exists() else None,
            "items": items}


def seed_mean(per_seed: list[dict[str, bool]]) -> dict[str, float]:
    """The only multi-seed reducer: per-item mean of 0/1 values over every preregistered seed."""
    if not per_seed or any(set(s) != set(per_seed[0]) for s in per_seed):
        raise ValueError("proof_eval: seeds must cover identical items")
    return {i: sum(bool(s[i]) for s in per_seed) / len(per_seed) for i in per_seed[0]}


def paired_bootstrap(diff: list[float], resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> list[float]:
    values = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.concatenate([values[rng.integers(0, len(values), (min(1000, resamples - k), len(values)))].mean(1)
                            for k in range(0, resamples, 1000)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def mcnemar_exact(f0: dict[str, bool], ft: dict[str, bool]) -> dict:
    b = sum(f0[i] and not ft[i] for i in f0)
    c = sum(ft[i] and not f0[i] for i in f0)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"f0_right_ft_wrong": b, "f0_wrong_ft_right": c, "p_value": p}


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _cell(run: dict, ids: list[str]) -> dict:
    """Counts and token statistics of one (arm, seed) run on one subset."""
    items = run["items"]
    tokens = [items[i]["tokens"] for i in ids]
    return {"correct": sum(items[i]["correct"] for i in ids), "no_parse": sum(items[i]["missing_final"] for i in ids),
            "length_finish": sum(items[i]["length"] for i in ids), "tokens_median": statistics.median(tokens),
            "tokens_mean": statistics.fmean(tokens), "correct_ids": [i for i in ids if items[i]["correct"]]}


def _grouped(mean_by_id: dict[str, float], ids: list[str], groups: dict[str, list[str]]) -> dict[str, float]:
    return {g: _mean(mean_by_id[i] for i in members) for g, subjects in groups.items()
            if (members := [i for i in ids if _subject(i) in subjects])}


def _subset_report(ids: list[str], by_arm: dict[str, dict[str, dict]]) -> dict:
    """by_arm: arm -> seed (str) -> run, all on this subset. Seeds are combined only as means."""
    n = len(ids)
    arms: dict[str, Any] = {}
    item_mean: dict[str, dict[str, float]] = {}
    for arm, seeds in by_arm.items():
        cells = {s: _cell(run, ids) for s, run in sorted(seeds.items())}
        item_mean[arm] = seed_mean([{i: run["items"][i]["correct"] for i in ids} for run in seeds.values()])
        mean_correct = _mean(c["correct"] for c in cells.values())
        if not math.isclose(mean_correct / n, _mean(item_mean[arm].values()), abs_tol=1e-12):
            raise AssertionError("proof_eval: item mean and seed mean disagree")
        arms[arm] = {"seeds": cells, "mean_correct": mean_correct, "accuracy": mean_correct / n}
    f0 = by_arm.get("F0", {})
    for arm in arms:
        common = sorted(set(f0) & set(by_arm[arm])) if arm != "F0" else []
        if not common:
            continue
        paired = {}
        for s in common:
            m = mcnemar_exact({i: f0[s]["items"][i]["correct"] for i in ids},
                              {i: by_arm[arm][s]["items"][i]["correct"] for i in ids})
            gains, losses = m["f0_wrong_ft_right"], m["f0_right_ft_wrong"]
            paired[s] = {"gains": gains, "losses": losses, "net": gains - losses, "mcnemar": m}
        base = seed_mean([{i: f0[s]["items"][i]["correct"] for i in ids} for s in common])
        mine = seed_mean([{i: by_arm[arm][s]["items"][i]["correct"] for i in ids} for s in common])
        diff = [mine[i] - base[i] for i in ids]
        arms[arm].update(
            seeds_compared=common, paired_vs_F0=paired,
            mean_diff_vs_F0=_mean(arms[arm]["seeds"][s]["correct"] - arms["F0"]["seeds"][s]["correct"] for s in common),
            mean_diff_ci95=[x * n for x in paired_bootstrap(diff)])
    return {"n": n, "ids": ids, "arms": arms,
            "per_subject": {a: _grouped(m, ids, {s: [s] for s in SUBJECTS}) for a, m in item_mean.items()},
            "per_discipline": {a: _grouped(m, ids, DISCIPLINES) for a, m in item_mean.items()}}


def _require_same_slots_and_gpu(runs: list[dict]) -> None:
    """Concurrency and hardware shift sampling numerics slightly: runs compared or pooled must share both.

    A registration without the keys (written before they existed) counts as 32 slots on an unknown GPU."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for run in runs:
        reg = run["registration"] or {}
        groups[(reg.get("slots", FAST_SLOTS), reg.get("gpu_name"))].append(Path(run["run_dir"]).name)
    if len(groups) > 1:
        raise ValueError("proof_eval: runs of different slots or GPU cannot be compared or pooled: " + "; ".join(
            f"slots={slots} gpu={gpu}: {', '.join(sorted(ids))}" for (slots, gpu), ids in sorted(groups.items(), key=str)))


def analyze(prereg_path: Path, run_paths: list[Path], ft_model_refs: dict[str, Path] | None, ledger: Path) -> dict:
    """Every run directory of the ledger: finished runs are scored, unfinished ones listed; seeds only as means."""
    prereg, prereg_sha = load_preregistration(prereg_path)
    events, head = read_ledger(ledger)
    starts = {e["run_id"]: e for e in events if e["event"] == "START"}
    failed = {e["run_id"] for e in events if e["event"] == "FAIL"}
    keys = [(e["arm"], e["variant"], e["subset"]) for e in starts.values()]
    final = FINAL_ARMS.get(prereg["format"])
    if final is not None and len(keys) != len(set(keys)):
        raise ValueError(f"proof_eval: {FINAL_TAGS[final][0]} ledger contains repeated attempts, including failed runs")
    if any(e["preregistration_sha256"] != prereg_sha for e in starts.values()):
        raise ValueError("proof_eval: the ledger belongs to another preregistration")
    allowed = {(r["arm"], r["variant"], r["subset"]) for r in prereg["runs"]}
    arms = [arm for arm in ALL_ARMS if arm in prereg["arms"]]
    base_sha = prereg["arms"]["F0"]["model_sha256"]
    runs: dict[tuple[str, str, str], dict] = {}
    seen: set[str] = set()
    unfinished: list[str] = []
    for path in run_paths:
        run_dir = Path(path).parent if Path(path).is_file() else Path(path)
        run_id = run_dir.name
        if run_id in seen:
            raise ValueError(f"proof_eval: duplicate run directory {run_id}; choosing among repeated runs is prohibited")
        seen.add(run_id)
        if run_id not in starts:
            raise ValueError(f"proof_eval: run {run_id} was never started in the ledger")
        if run_id in failed:
            raise ValueError(f"proof_eval: run {run_id} is recorded in the ledger as FAILED")
        summary = run_dir / "summary.json"
        status = json.loads(summary.read_text(encoding="utf-8")).get("status") if summary.is_file() else None
        if status not in FINISHED.values():  # still running, interrupted or never written; a COMPLETE one fails load_run
            unfinished.append(run_id)
            continue
        run, start = load_run(run_dir, prereg), starts[run_id]
        key = (start["arm"], run["variant"], run["subset"])
        if key != (start["arm"], start["variant"], start["subset"]) or key not in allowed:
            raise ValueError(f"proof_eval: run {run_id} is recorded in the ledger as another (arm, variant, subset)")
        if (run["identity"]["model_sha256"] == base_sha) != (start["arm"] == "F0"):
            raise ValueError(f"proof_eval: run {run_id} does not evaluate the model of its arm {start['arm']}")
        runs[key] = run
    omitted = sorted(r for r in starts if r not in failed and r not in seen)
    if omitted:
        raise ValueError(f"proof_eval: ledger run {omitted[0]} is not among the analyzed runs (omitted or replaced run)")
    if not runs:
        raise ValueError("proof_eval: no finished run to analyze")
    shas = {arm: {r["identity"]["model_sha256"] for key, r in runs.items() if key[0] == arm} for arm in arms}
    if any(len(v) > 1 for v in shas.values()):
        raise ValueError("proof_eval: every run of an arm must evaluate its one bound export (one checkpoint per arm)")
    bound = [next(iter(v)) for v in shas.values() if v]
    if len(set(bound)) != len(bound):
        raise ValueError("proof_eval: the arms must evaluate different checkpoints (one checkpoint per arm)")
    if final is not None and shas[final] and shas[final] != {prereg["arms"][final]["model_sha256"]}:
        raise ValueError(f"proof_eval: {final} run differs from the checkpoint declared before evaluation")
    refs = {}
    for arm, ref_path in (ft_model_refs or {}).items():
        if arm not in arms or arm == "F0":
            raise ValueError(f"proof_eval: {arm} is not a fine-tuned arm")
        ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
        if arm == final:
            ref, declared = _final_model_ref(ref_path, prereg["format"])
            if any(prereg["arms"][arm].get(k) != v for k, v in declared.items() if k != "model_ref"):
                raise ValueError(f"proof_eval: {arm} model reference differs from its preregistration")
        if shas[arm] and shas[arm] != {digest(ref["files"])}:
            raise ValueError(f"proof_eval: every {arm} run must evaluate the one bound export (one checkpoint per arm)")
        refs[arm] = {"sha256": sha256_file(Path(ref_path)), "model_sha256": digest(ref["files"]),
                     "export": ref.get("export")}
    if final is not None and final not in refs:
        refs[final] = dict(prereg["arms"][final])
    if len({r["identity"]["code_sha256"] for r in runs.values()}) != 1:
        raise ValueError("proof_eval: statistical runs must share one evaluation code hash")
    for variant in {r["variant"] for r in runs.values()}:
        if len({r["identity"]["protocol_sha256"] for r in runs.values() if r["variant"] == variant}) != 1:
            raise ValueError(f"proof_eval: protocol hash differs between arms for {variant}")
    for key, run in runs.items():
        reg = run["registration"]
        if (reg is None or reg.get("protocol_id") != key[1]
                or reg.get("protocol_file_sha256") != prereg["protocol_variants"][key[1]]["sha256"]):
            raise ValueError(f"proof_eval: statistical run {key} lacks its matching {REGISTRATION_FILE}")
    _require_same_slots_and_gpu(list(runs.values()))
    if len({digest({k: x for k, x in r["registration"].items() if k not in ("protocol_id", "protocol_file_sha256")})
            for r in runs.values()}) != 1:
        raise ValueError("proof_eval: statistical runs must share one registration (src and proof_eval bytes)")

    subsets = {tag: _subset_report(sorted(row["ids"]), {
        arm: {str(r["seed"]): r for key, r in runs.items() if key[0] == arm and key[2] == tag} for arm in arms
        if any(key[0] == arm and key[2] == tag for key in runs)}) for tag, row in prereg["evaluation_subsets"].items()}
    full_900: dict[str, Any] = {}
    for arm in arms if "all" in subsets else ():  # v8: the one 900-item run is the full count
        cells = {s: {"correct": c["correct"], "accuracy": c["correct"] / subsets["all"]["n"]}
                 for s, c in subsets["all"]["arms"].get(arm, {"seeds": {}})["seeds"].items()}
        if cells:
            full_900[arm] = {"items": subsets["all"]["n"], "seeds": cells,
                             "mean_correct": _mean(c["correct"] for c in cells.values())}
    for arm in arms if "all" not in subsets else ():
        total = subsets["wrong"]["n"] + subsets["rest"]["n"]
        wrong = subsets["wrong"]["arms"].get(arm, {"seeds": {}})["seeds"]
        rest = subsets["rest"]["arms"].get(arm, {"seeds": {}})["seeds"]
        cells = {s: {"correct": wrong[s]["correct"] + rest[s]["correct"], "accuracy": (wrong[s]["correct"] + rest[s]["correct"]) / total,
                     "wrong_subset_correct": wrong[s]["correct"], "rest_subset_correct": rest[s]["correct"]}
                 for s in sorted(set(wrong) & set(rest))}
        if cells:
            full_900[arm] = {"items": total, "seeds": cells, "mean_correct": _mean(c["correct"] for c in cells.values())}
    return {
        "format": "notebook-rl-proof-report-v2", "preregistration_sha256": prereg_sha,
        **({"preregistration_format": prereg["format"]} if final is not None else {}),
        "single_answer_rule": prereg["single_answer_rule"],
        "ledger": {"file_sha256": sha256_file(ledger), "head": head, "starts": len(starts),
                   "failed_run_ids": sorted(failed), "unfinished_run_ids": sorted(unfinished)},
        "runs": [{"arm": key[0], "variant": key[1], "subset": key[2], "seed": r["seed"], "run_dir": r["run_dir"],
                  "identity": r["identity"], "registration": r["registration"]} for key, r in sorted(runs.items())],
        "ft_model_refs": refs, "scorer": PARSER, "subsets": subsets, "full_900": full_900,
        "course_facing": dict(prereg["course_facing_run"]), "limitations": prereg.get("limitations", []),
    }


def gate_300(results: dict) -> dict:
    """The preregistered 300-item decision rule on the "wrong" subset of an analyze report (see prereg gate_300).

    Seeds are the ones both F0 and FT-NOTE finished and are combined only as means. Token statistics and the
    no-parse / length-finish counts of the collapse alert are the seed means of the per-seed values. A threshold for
    spending more, never proof of improvement."""
    if results.get("preregistration_format") in FINAL_ARMS:
        tag = FINAL_TAGS[FINAL_ARMS[results["preregistration_format"]]][1]
        raise ValueError(f"proof_eval: {tag} final evaluation has no 342-item pilot gate")
    _require_same_slots_and_gpu(results.get("runs", []))
    wrong = results["subsets"]["wrong"]
    f0, note = wrong["arms"].get("F0"), wrong["arms"].get("FT-NOTE")
    seeds = sorted(set(f0["seeds"]) & set(note["seeds"])) if f0 and note else []
    if not seeds:
        raise ValueError("proof_eval: the gate needs a seed finished by both F0 and FT-NOTE on the wrong subset")
    f0_counts, note_counts = ([arm["seeds"][s]["correct"] for s in seeds] for arm in (f0, note))
    mean_diff = _mean(note_counts) - _mean(f0_counts)
    by_mean = mean_diff >= GATE_PASS_DIFF
    by_seeds = len(seeds) == len(SEEDS) and all(a > b for a, b in zip(note_counts, f0_counts))
    status = "PASS" if by_mean or by_seeds else "AMBIGUOUS" if mean_diff > 0 else "STOP"
    f0_wrong = set.intersection(*(set(wrong["ids"]) - set(f0["seeds"][s]["correct_ids"]) for s in seeds))
    note_right = set().union(*(set(note["seeds"][s]["correct_ids"]) for s in seeds))
    seed_mean_of = lambda arm, key: _mean(arm["seeds"][s][key] for s in seeds)  # noqa: E731
    collapse: dict[str, Any] = {"tokens_median": {"F0": seed_mean_of(f0, "tokens_median"),
                                                  "FT-NOTE": seed_mean_of(note, "tokens_median")}}
    collapse["tokens_median"]["alert"] = (collapse["tokens_median"]["FT-NOTE"]
                                          < GATE_COLLAPSE_TOKEN_RATIO * collapse["tokens_median"]["F0"])
    for key in ("no_parse", "length_finish"):
        collapse[key] = {"F0": seed_mean_of(f0, key), "FT-NOTE": seed_mean_of(note, key)}
        collapse[key]["alert"] = collapse[key]["FT-NOTE"] > GATE_COLLAPSE_COUNT_RATIO * collapse[key]["F0"]
    solution = wrong["arms"].get("FT-SOLUTION")
    shared = sorted(set(f0["seeds"]) & set(solution["seeds"])) if solution else []
    return {
        "format": "notebook-rl-gate-300-v1", "not_proof_of_improvement": True,
        "preregistration_sha256": results.get("preregistration_sha256"), "subset": "wrong", "items": wrong["n"],
        "status": status, "pass_via": [name for name, hit in (("mean_diff", by_mean), ("all_seeds", by_seeds)) if hit],
        "seeds_used": [int(s) for s in seeds], "fewer_than_3_seeds": len(seeds) < len(SEEDS),
        "note": ("fewer than 3 seeds finished: PASS is possible only through the mean_diff branch"
                 if len(seeds) < len(SEEDS) else "3 seeds finished: both PASS branches applied"),
        "per_seed": {s: {"F0": a, "FT-NOTE": b} for s, a, b in zip(seeds, f0_counts, note_counts)},
        "mean_f0": _mean(f0_counts), "mean_ft_note": _mean(note_counts), "mean_diff": mean_diff,
        "recovered_ids": len(f0_wrong & note_right), "collapse_alert": any(c["alert"] for c in collapse.values()),
        "collapse": collapse,
        "ft_solution": ({"seeds_used": [int(s) for s in shared],
                         "mean_correct": _mean(solution["seeds"][s]["correct"] for s in shared),
                         "mean_f0_correct": _mean(f0["seeds"][s]["correct"] for s in shared),
                         "mean_diff": _mean(solution["seeds"][s]["correct"] - f0["seeds"][s]["correct"] for s in shared),
                         "note": "reported beside FT-NOTE; does not change the status"} if shared else None),
    }


def gate_62(results: dict, f0_reference: int | None = None) -> dict:
    """The preregistered version-2 rule on an analyze report of a v4 preregistration: PASS iff the FT-V2 900-item V8
    correct count (wrong + rest runs of the one seed) > 558. F0 of the same ledger and an F0 reference number from
    another ledger are shown beside it and never change the status."""
    if results.get("preregistration_format") in FINAL_ARMS:
        return gate_v3(results, 62, f0_reference)
    _require_same_slots_and_gpu(results.get("runs", []))
    seed = str(VARIANTS[GATE_62_VARIANT][1])
    ft = results["full_900"].get(ARM_V2, {"seeds": {}})["seeds"].get(seed)
    if ft is None:
        raise ValueError(f"proof_eval: gate-62 needs finished FT-V2 wrong and rest runs of seed {seed}")
    f0 = results["full_900"].get("F0", {"seeds": {}})["seeds"].get(seed)
    total = results["full_900"][ARM_V2]["items"]
    if f0_reference is not None and (type(f0_reference) is not int or not 0 <= f0_reference <= total):
        raise ValueError(f"proof_eval: --f0-reference must be a correct count between 0 and {total}")
    return {
        "format": "error-note-gate-62-v1", "not_proof_of_improvement": True,
        "preregistration_sha256": results.get("preregistration_sha256"), "variant": GATE_62_VARIANT, "seed": int(seed),
        "status": "PASS" if ft["correct"] > B2_CORRECT else "FAIL",
        "rule": f"PASS iff the FT-V2 900-item V8 correct count > {B2_CORRECT} (B2, 62.00%)",
        "ft_v2": ft, "b2_correct": B2_CORRECT, "margin_vs_b2": ft["correct"] - B2_CORRECT,
        "f0": (dict(f0, ft_v2_minus_f0=ft["correct"] - f0["correct"]) if f0 is not None
               else "not run in this ledger"),
        "f0_reference": (None if f0_reference is None else {
            "correct": f0_reference, "accuracy": f0_reference / total,
            "source": f"F0 x {GATE_62_VARIANT} wrong + rest of the v2 ledger ({FORMAT}); display only, another "
                      "ledger and another registration, never part of the rule"}),
        "note": GATE_62_NOTE,
    }


def gate_v3(results: dict, target_percent: float, f0_reference: int | None = None) -> dict:
    """One final v3 (v5) or v4 (v6) pass: numeric targets, current paired F0 and explicitly unpaired references."""
    if results.get("preregistration_format") not in FINAL_ARMS or target_percent not in (62, 67.4):
        raise ValueError("proof_eval: v3 final gates require a v5 preregistration and target 62 or 67.4 "
                         "(v4: a v6 preregistration; single campaign: v7)")
    _require_same_slots_and_gpu(results.get("runs", []))
    arm = FINAL_ARMS[results["preregistration_format"]]
    _, tag, _, gate_format, result_key = FINAL_TAGS[arm]
    seed = "3407"
    block = results["full_900"].get(arm, {})
    if block.get("items") != 900 or set(block.get("seeds", {})) != {seed}:
        raise ValueError(f"proof_eval: {tag} final gate requires both {arm} subsets, "
                         "exactly 900 items and seed 3407 only")
    ft = block["seeds"][seed]
    minimum = 559 if target_percent == 62 else 607
    if f0_reference is not None and (type(f0_reference) is not int or not 0 <= f0_reference <= 900):
        raise ValueError("proof_eval: --f0-reference must be a correct count between 0 and 900")
    f0 = results["full_900"].get("F0", {}).get("seeds", {}).get(seed)
    paired = None
    if f0 is not None:
        truth: dict[str, dict[str, bool]] = {"F0": {}, arm: {}}
        for sub in results["subsets"].values():
            for arm in truth:
                correct = set(sub["arms"][arm]["seeds"][seed]["correct_ids"])
                truth[arm].update({sid: sid in correct for sid in sub["ids"]})
        paired = {"correct": f0["correct"], "accuracy": f0["accuracy"], "ft_minus_f0": ft["correct"] - f0["correct"],
                  "mcnemar": mcnemar_exact(truth["F0"], truth[arm]), "conditions": "same current evaluation registration"}
    return {"format": gate_format, "not_proof_of_improvement": True,
            "preregistration_sha256": results.get("preregistration_sha256"), "variant": GATE_62_VARIANT, "seed": 3407,
            "arm": arm, "items": 900, "target_percent": target_percent, "minimum_correct": minimum,
            "status": "PASS" if ft["correct"] >= minimum else "FAIL", result_key: ft,
            "rule": f"PASS iff the one final {arm} 900-item V8 correct count >= {minimum} (> {target_percent}%)",
            "paired_f0": paired,
            "f0_reference": None if f0_reference is None else {
                "correct": f0_reference, "accuracy": f0_reference / 900,
                "source": "historical F0 under different conditions/registration; display only, never paired or part of the rule"},
            "b2_correct": B2_CORRECT, "margin_vs_b2": ft["correct"] - B2_CORRECT,
            "note": GATE_62_NOTE}


def render_markdown(report: dict) -> str:
    pct = lambda x: f"{100 * x:.2f}"  # noqa: E731
    lines = ["# Notebook-RL MMMU-val proof report", "", f"Preregistration sha256 `{report['preregistration_sha256']}`",
             f"Scorer: strict {report['scorer']} only; seeds combined only as means.", ""]
    if report["ledger"]["unfinished_run_ids"]:
        lines += [f"Unfinished runs (not scored): {', '.join(report['ledger']['unfinished_run_ids'])}", ""]
    for tag, sub in report["subsets"].items():
        if not sub["arms"]:
            lines += [f"## Subset {tag} ({sub['n']} items): no finished run", ""]
            continue
        lines += [f"## Subset {tag} ({sub['n']} items)", "",
                  "| arm | seed | correct | no-parse | length | median tokens | mean tokens | gains | losses | net | McNemar p |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm, block in sub["arms"].items():
            for seed, c in block["seeds"].items():
                p = block.get("paired_vs_F0", {}).get(seed)
                pair = [p["gains"], p["losses"], p["net"], f"{p['mcnemar']['p_value']:.4g}"] if p else ["-"] * 4
                lines.append(f"| {arm} | {seed} | {c['correct']} | {c['no_parse']} | {c['length_finish']} | "
                             f"{c['tokens_median']:g} | {c['tokens_mean']:.1f} | " + " | ".join(map(str, pair)) + " |")
        lines += ["", "Mean correct per arm: " + ", ".join(
            f"{arm} {block['mean_correct']:.2f} ({pct(block['accuracy'])}%)"
            + (f", {block['mean_diff_vs_F0']:+.2f} vs F0 [95% {block['mean_diff_ci95'][0]:+.2f}, "
               f"{block['mean_diff_ci95'][1]:+.2f}]" if "mean_diff_vs_F0" in block else "")
            for arm, block in sub["arms"].items()), ""]
        lines += ["| subject | " + " | ".join(sub["per_subject"]) + " |", "|---|" + "---|" * len(sub["per_subject"])]
        subjects = sorted({s for row in sub["per_subject"].values() for s in row})
        lines += [f"| {s} | " + " | ".join(pct(row[s]) if s in row else "-" for row in sub["per_subject"].values()) + " |"
                  for s in subjects]
        lines.append("")
    if report["full_900"]:
        lines += ["## Full 900 (wrong + rest, or the one v8 run; same arm and seed)", "", "| arm | seed | correct | accuracy |", "|---|---|---|---|"]
        lines += [f"| {arm} | {seed} | {c['correct']} | {pct(c['accuracy'])}% |"
                  for arm, block in report["full_900"].items() for seed, c in block["seeds"].items()]
        lines.append("")
    lines += [f"Course-facing run: {report['course_facing']['status']}", ""]
    lines += [f"- Limitation: {text}" for text in report["limitations"]]
    lines += ["", report["single_answer_rule"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run-eval"]:
        return run_eval(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run-eval", help="scripts/eval.sh arguments for one 8k variant file (passed through verbatim)")
    exp = sub.add_parser("export", help="merge one local_train checkpoint into a BF16 eval model + model_ref.json")
    exp.add_argument("--checkpoint", type=Path, required=True)
    exp.add_argument("--binding", type=Path, required=True, help="binding JSON or launch.json holding 'binding'")
    exp.add_argument("--base", type=Path, required=True, help="pinned BF16 snapshot directory")
    exp.add_argument("--output", type=Path, required=True, help="new directory: model/ + model_ref.json")
    exp.add_argument("--code-commit", required=True, help="40-hex Git commit of the training code")
    for name in ("verify-v3-export", "verify-v4-export", "verify-single-export"):
        check_export = sub.add_parser(name, help="connect the declared final (v5: v3 NOTE, v6: v4 hybrid, v7: single "
                                                 "campaign, --binding its campaign-plan.json) export to its source checkpoint")
        for flag in ("model-ref", "checkpoint", "binding", "prereg"):
            check_export.add_argument(f"--{flag}", type=Path, required=True)
    mk = sub.add_parser("make-subset", help="write the wrong (342) or rest (558) evaluation subset file from the B2 file")
    mk.add_argument("--kind", choices=tuple(SUBSET_FILES), required=True)
    mk.add_argument("--output", type=Path, required=True)
    mk.add_argument("--b2-predictions", type=Path, default=B2_PREDICTIONS)
    pre = sub.add_parser("preregister", help="write the frozen preregistration and its .sha256")
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--subset", type=Path, required=True, help="proof-eval-subset-v1 JSON: the B2 V8 wrong IDs")
    pre.add_argument("--rest-subset", type=Path, required=True, help="proof-eval-subset-rest-v1 JSON: the B2 V8 right IDs")
    pre.add_argument("--b2-predictions", type=Path, default=B2_PREDICTIONS)
    pre.add_argument("--all-subset", type=Path, help="v8 only: proof-eval-subset-all-v1 JSON, the 900 sorted original IDs")
    pre.add_argument("--format", choices=("v3", "v4", "v5", "v6", "v7", "v8"), default="v3",
                     help=f"v3 = {FORMAT}, v4 = {FORMAT_V4}, v5 = {FORMAT_V5} (final FT-NOTE-V3, declared checkpoint), "
                          f"v6 = {FORMAT_V6} (final FT-HYB-V4), v7 = {FORMAT_V7} (final FT-SINGLE), "
                          f"v8 = {FORMAT_V8} (final FT-SINGLE, one 900-item run)")
    pre.add_argument("--ft-v3-model-ref", type=Path, help="required for v5; final NOTE export fixed before evaluation")
    pre.add_argument("--ft-v4-model-ref", type=Path, help="required for v6; final hybrid export fixed before evaluation")
    pre.add_argument("--ft-single-model-ref", type=Path, help="required for v7; final campaign export fixed before evaluation")
    start = sub.add_parser("ledger-start", help="append the START of one evaluation run (before run-eval/eval.sh); "
                           "refused when that (arm, variant, subset) has a started run not recorded as FAILED")
    start.add_argument("--prereg", type=Path, required=True)
    start.add_argument("--arm", choices=ALL_ARMS, required=True)
    start.add_argument("--variant", required=True, help="a preregistered protocol variant id")
    start.add_argument("--subset", choices=tuple(SUBSET_FILES), required=True)
    start.add_argument("--run-id", required=True, help="the run directory name artifact_root/runs/<run-id>")
    start.add_argument("--ledger", type=Path, required=True)
    fail = sub.add_parser("ledger-fail", help="record a started run as FAILED (refused for a finished COMPLETE/PARTIAL run)")
    fail.add_argument("--ledger", type=Path, required=True)
    fail.add_argument("--run-id", required=True)
    fail.add_argument("--reason", required=True)
    fail.add_argument("--artifact-root", type=Path, default=None, help="default: MMDL_ARTIFACT_ROOT, as run-eval")
    ana = sub.add_parser("analyze", help="score the finished runs of the ledger; writes <output>.json and <output>.md")
    gate = sub.add_parser("gate", help="apply the preregistered 300-item gate (wrong subset, FT-NOTE vs F0); writes --output")
    gate62 = sub.add_parser("gate-62", help="v4 preregistration: PASS iff the FT-V2 900-item V8 correct count > 558; "
                                            "v5: FT-NOTE-V3 >=559; writes --output")
    gate674 = sub.add_parser("gate-67.4", help="v5/v6 only: final FT-NOTE-V3 / FT-HYB-V4 900-item V8 correct count >=607")
    for command in (ana, gate, gate62, gate674):
        command.add_argument("--prereg", type=Path, required=True)
        command.add_argument("--ledger", type=Path, required=True, help="the run ledger: every run dir of it is passed "
                             "(a live started run left out is refused)")
        command.add_argument("--runs", type=Path, nargs="+", required=True,
                             help="eval run dirs or their predictions.jsonl (dir name = ledger run id)")
    ana.add_argument("--ft-note-model-ref", type=Path, default=None)
    ana.add_argument("--ft-solution-model-ref", type=Path, default=None)
    ana.add_argument("--ft-note-3417-model-ref", type=Path, default=None)
    ana.add_argument("--ft-v2-model-ref", type=Path, default=None)
    ana.add_argument("--ft-v3-model-ref", type=Path, default=None)
    ana.add_argument("--ft-v4-model-ref", type=Path, default=None)
    ana.add_argument("--ft-single-model-ref", type=Path, default=None)
    ana.add_argument("--output", type=Path, required=True, help="report path; .json and .md are written")
    gate.add_argument("--output", type=Path, required=True, help="gate result JSON (write-once)")
    gate62.add_argument("--output", type=Path, required=True, help="gate result JSON (write-once)")
    gate62.add_argument("--f0-reference", type=int, default=None,
                        help="F0 900-item correct count from the v2 ledger, shown for display only")
    gate674.add_argument("--output", type=Path, required=True, help="gate result JSON (write-once)")
    gate674.add_argument("--f0-reference", type=int, default=None, help="historical F0 correct count, display only")
    args = parser.parse_args(argv)
    if args.command == "export":
        ref = export_for_eval(args.checkpoint, args.binding, args.base, args.output, args.code_commit)
        print(f"EXPORTED {args.output / 'model_ref.json'} unmerged_inference_bit_equivalent="
              f"{ref['export']['unmerged_inference_bit_equivalent']}")
    elif args.command in ("verify-v3-export", "verify-v4-export", "verify-single-export"):
        verify_v3_export(args.model_ref, args.checkpoint, args.binding, args.prereg)
        print(f"{args.command.split('-')[1].upper()}_EXPORT_VERIFIED {args.model_ref}")
    elif args.command == "make-subset":
        value = make_subset(args.kind, args.output, args.b2_predictions)
        print(f"SUBSET {args.kind} {args.output} count={value['count']} sha256={sha256_file(args.output)}")
    elif args.command == "preregister":
        print(f"PREREGISTERED {args.output} sha256="
              f"{preregister(args.output, args.subset, args.rest_subset, args.b2_predictions, args.format, args.ft_v3_model_ref, args.ft_v4_model_ref, args.ft_single_model_ref, all_subset=args.all_subset)}")
    elif args.command == "ledger-start":
        event = ledger_start(args.prereg, args.arm, args.variant, args.run_id, args.ledger, args.subset)
        print(f"LEDGER_STARTED {event['arm']} {event['variant']} {event['subset']} {event['run_id']}")
    elif args.command == "ledger-fail":
        from mmdl.runtime.artifacts import default_path

        root = args.artifact_root or default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts")
        print(f"LEDGER_FAILED {ledger_fail(args.ledger, args.run_id, args.reason, root)['run_id']}")
    elif args.command == "gate":
        result = gate_300(analyze(args.prereg, args.runs, None, args.ledger))
        _write_once(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"GATE {result['status']} mean_diff={result['mean_diff']:.3f} collapse_alert={result['collapse_alert']}")
    elif args.command == "gate-62":
        result = gate_62(analyze(args.prereg, args.runs, None, args.ledger), args.f0_reference)
        _write_once(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
        arm_key = next(key for key in ("ft_note_v3", "ft_hyb_v4", "ft_single", "ft_v2") if key in result)
        print(f"GATE62 {result['status']} {arm_key}={result[arm_key]['correct']} b2={B2_CORRECT}")
    elif args.command == "gate-67.4":
        result = gate_v3(analyze(args.prereg, args.runs, None, args.ledger), 67.4, args.f0_reference)
        _write_once(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
        arm_key = next(key for key in ("ft_note_v3", "ft_hyb_v4", "ft_single") if key in result)
        print(f"GATE67.4 {result['status']} {arm_key}={result[arm_key]['correct']} minimum=607")
    else:
        refs = {arm: path for arm, path in (("FT-NOTE", args.ft_note_model_ref),
                                            ("FT-SOLUTION", args.ft_solution_model_ref),
                                            ("FT-NOTE-3417", args.ft_note_3417_model_ref),
                                            (ARM_V2, args.ft_v2_model_ref), (ARM_V3, args.ft_v3_model_ref),
                                            (ARM_V4, args.ft_v4_model_ref), (ARM_SINGLE, args.ft_single_model_ref)) if path}
        report = analyze(args.prereg, args.runs, refs, args.ledger)
        _write_once(args.output.with_suffix(".json"), json.dumps(report, indent=2, sort_keys=True) + "\n")
        _write_once(args.output.with_suffix(".md"), render_markdown(report))
        print(f"ANALYZED {args.output.with_suffix('.json')} runs={len(report['runs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
