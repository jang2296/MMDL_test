"""Pre-registered F0-vs-FT MMMU-val proof: merged export, fast8k run wrapper, preregistration, analysis.

Nothing here trains or selects. ``run-eval`` only registers the three fast8k variant IDs inside its
own process and then calls the unchanged scripts/eval.sh entry point; src/ is never edited.
The statistical (paired F0/FT) variants differ from mmmu-val-v8 only in seed, 32 slots, an 8192-token
output cap, the training image budget 262144..1003520 pixels and a 24576 context (user decision
2026-10-31: speed/cost of the RunPod proof). The c16 variant files stay on disk but are not registered.
Scope (user decision 2026-10-31): every preregistered run is a ``--mode partial`` run on exactly the
342 MMMU-val items the frozen B2 + team-final-answer-v8 baseline got wrong (the subset file, bound by
sha256 in the preregistration; evaluation only, never training). The course run (mmmu-val-v8, 900
items, plain scripts/eval.sh) keeps its code path but is not preregistered.
``analyze`` re-scores preserved raw responses with the frozen V8 scorer (the only scorer here) and
reduces several seeds only by the per-item mean (CLAUDE.md §2.8): there is no reducer argument, no
repeated (arm, protocol) run, and no second FT checkpoint.

Run ledger (append-only JSONL, hash-chained, fsynced): ``ledger-start --prereg P --arm A --variant V --run-id R
--ledger L`` must precede every evaluation run (``run-eval`` or the plain course ``scripts/eval.sh``; R is the run
directory name artifact_root/runs/R). It refuses a second start of the same (arm, variant) unless the first is
recorded with ``ledger-fail --ledger L --run-id R --reason ...`` (refused for a finished run). ``analyze --ledger L``
is required: it refuses a run that was never started, started under another (arm, variant) or recorded FAILED, and any
live started run of a registered (arm, variant) that is not the one analyzed (an omitted or replaced repeat).
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
FORMAT = "notebook-rl-proof-prereg-v2"
SUBSET_VERSION = "proof-eval-subset-v1"
# The evaluator's summary status of a run that finished every planned item: full runs end COMPLETE, partial
# (--sample-id) runs end PARTIAL; anything else (INCOMPLETE, RUNNING, absent) is not finished.
FINISHED = {"course": "COMPLETE", "statistical": "PARTIAL"}
PARSER = "team-final-answer-v8"
COURSE = "mmmu-val-v8"
SENSITIVITY = "mmmu-val-v8-fast8k-seed3407-v1"
# variant id -> (config file, seed, max concurrent requests, role)
VARIANTS = {COURSE: (ROOT / "configs/eval/mmmu_val_v8.yaml", 3407, 2, "course")} | {
    f"mmmu-val-v8-fast8k-seed{s}-v1": (HERE / f"eval_configs/mmmu_val_v8_fast8k_seed{s}.yaml", s, 32, "statistical")
    for s in (3407, 3408, 3409)}
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
    "Each item is scored from the one answer generated once per (arm, protocol variant). No best-of-N, "
    "majority vote, reranking, verifier/feedback retry, ensemble, or per-item seed choice. Seeds are "
    "combined only as the mean of per-seed correctness. Each arm has exactly one evaluated checkpoint. "
    "Repeated runs of the same (arm, variant) are refused, never chosen between. An incomplete run is "
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
    loaded = json.loads(Path(binding_file).read_text(encoding="utf-8"))
    binding = loaded["binding"] if isinstance(loaded, dict) and "binding" in loaded else loaded
    checkpoint, output = Path(checkpoint), Path(output)
    metadata = read_metadata(checkpoint, binding=binding)
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
        },
    }
    _write_once(output / "model_ref.json", json.dumps(model_ref, indent=2, sort_keys=True) + "\n")
    return model_ref


# ---------------------------------------------------------------------------------------------
# fast8k run wrapper: in-process registration of exactly three IDs; src/ files stay untouched
# ---------------------------------------------------------------------------------------------

V8 = "mmmu-val-v8"
# The only differences of a registered variant from its mmmu-val-v8 twin besides protocol_id and seed.
FAST_SLOTS = 32
FAST_MAX_NEW_TOKENS = 8192
FAST_PIXELS = {"min_pixels": 262144, "max_pixels": 1003520}  # = the training image budget (local_profile)
FAST_MAX_MODEL_LEN = 24576
REGISTERED = {vid: seed for vid, (_, seed, _, role) in VARIANTS.items() if role == "statistical"}
PATCHED_SRC = ("src/mmdl/runtime/contracts.py", "src/mmdl/evaluation/backends/vllm_backend.py",
               "src/mmdl/evaluation/cli.py", "src/mmdl/evaluation/engine.py")
REGISTRATION_FILE = "variant_registration.json"


def variant_registration(protocol_id: str) -> dict:
    """What run-eval registers and the exact src/ + proof_eval bytes it patched around."""
    from mmdl.runtime import contracts

    return {
        "format": "proof-eval-variant-registration-v1", "protocol_id": protocol_id,
        "protocol_file_sha256": sha256_file(VARIANTS[protocol_id][0]),
        "registered": {pid: {"seed": seed, "parser": PARSER, "max_new_tokens": FAST_MAX_NEW_TOKENS,
                             "control_images": _fast_images(contracts), "execution_profile": _fast_execution(contracts)}
                       for pid, seed in REGISTERED.items()},
        "patched": {
            "mmdl.runtime.contracts.validate_configs": "registered IDs: exact seed, 32 slots, 8192 output tokens, "
                                                       "262144..1003520 pixels, 24576 context, V8 parser; all else "
                                                       "validated as the mmmu-val-v8 twin; other IDs unchanged",
            "mmdl.evaluation.backends.vllm_backend.VLLMBackend.__init__": "registered IDs: slot check sees 2, vLLM "
                                                                          "max_num_seqs=32, batch_size=32; other IDs unchanged"},
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


def _fast_execution(contracts) -> dict:
    return dict(contracts.EXECUTION_PROFILES[V8], batch_size=FAST_SLOTS, max_model_len=FAST_MAX_MODEL_LEN)


@contextmanager
def registered_variants(on_backend: Callable[[], None] | None = None) -> Iterator[None]:
    """Register exactly the fast8k IDs in this process; tables and functions are restored on exit."""
    from mmdl.evaluation.backends import vllm_backend
    from mmdl.runtime import contracts

    if any(pid in contracts.CONTROL_IMAGES or pid in contracts.EXECUTION_PROFILES for pid in REGISTERED):
        raise RuntimeError("proof_eval: fast8k IDs already registered (nested use or src changed)")
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
            raise ValueError(f"proof_eval: {pid} requires seed {REGISTERED[pid]}, 32 slots, 8192 output tokens, "
                             "262144..1003520 pixels, 24576 context and the V8 parser")
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
                return llm_cls(**kwargs | {"max_num_seqs": FAST_SLOTS})
            return llm, sampling_cls, version

        vllm_backend._vllm = widened
        try:
            original_init(self, model_path, view, hw, kind, base_path)
        finally:
            vllm_backend._vllm = real
        if self.batch_size != 2 or self.cfg is not view:
            raise RuntimeError("proof_eval: unexpected VLLMBackend structure after load")
        self.cfg, self.batch_size = cfg, FAST_SLOTS

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
    """Same argv as scripts/eval.sh, for one of the three fast8k variant files only."""
    from mmdl.runtime.artifacts import default_path

    pre = argparse.ArgumentParser(prog="proof_eval run-eval", add_help=False)
    pre.add_argument("--protocol", type=Path, required=True)
    pre.add_argument("--run-id", required=True)
    pre.add_argument("--artifact-root", type=Path, default=default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts"))
    known, _ = pre.parse_known_args(argv)
    protocol = known.protocol if known.protocol.is_absolute() else ROOT / known.protocol
    pid = yaml.safe_load(protocol.read_text(encoding="utf-8")).get("protocol_id")
    if pid not in REGISTERED or sha256_file(protocol) != sha256_file(VARIANTS[pid][0]):
        raise ValueError("proof_eval: run-eval runs only the three preregistered fast8k variant files; "
                         "the c2 course run (mmmu-val-v8) uses plain scripts/eval.sh")
    artifact_root = known.artifact_root if known.artifact_root.is_absolute() else ROOT / known.artifact_root
    run_dir = artifact_root / "runs" / known.run_id
    registration = variant_registration(pid)
    from mmdl.evaluation import cli

    cwd, saved_argv = Path.cwd(), sys.argv
    os.chdir(ROOT)  # scripts/eval.sh runs the CLI from the repository root
    sys.argv = ["mmdl.evaluation.cli", *argv]
    try:
        with registered_variants(on_backend=lambda: _record_registration(run_dir, registration)):
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
            "wrong_ids": wrong, "right_ids": right}


def _subset(path: Path, b2: dict) -> dict:
    """The evaluation subset file: exactly the frozen B2 V8 wrong IDs, derived from the same B2 file."""
    path = Path(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    ids = loaded.get("sample_ids") if isinstance(loaded, dict) else None
    if (loaded.get("version") != SUBSET_VERSION or not isinstance(ids, list) or ids != b2["wrong_ids"]
            or loaded.get("count") != len(ids) or loaded.get("source_sha256") != b2["sha256"]):
        raise ValueError("proof_eval: the subset must be the sorted frozen B2 V8 wrong IDs of the same B2 file")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "version": SUBSET_VERSION, "count": len(ids),
            "source": loaded.get("source"), "source_sha256": loaded["source_sha256"], "ids": ids,
            "rule": "B2 + team-final-answer-v8 wrong items; evaluation only, never training (CLAUDE.md §2.2)"}


def build_preregistration(subset: Path, b2_predictions: Path = B2_PREDICTIONS) -> dict:
    variants = {}
    for vid, (path, seed, concurrency, role) in VARIANTS.items():
        if role != "statistical":  # the course run is not in the 342-item proof plan
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (cfg["protocol_id"] != vid or cfg["generation"]["seed"] != seed or cfg["parser"] != PARSER
                or cfg["execution"]["batch_size"] != concurrency or cfg["status"] != "FROZEN"):
            raise ValueError(f"proof_eval: variant file disagrees with its declaration: {vid}")
        variants[vid] = {"path": _rel(path), "sha256": sha256_file(path), "config_digest": digest(cfg),
                         "seed": seed, "concurrency": concurrency, "role": role}
    statistical = list(variants)
    runs = [{"arm": arm, "variant": v, "role": "statistical"} for arm in ("F0", "FT") for v in statistical]
    base_ref = json.loads(BASE_MODEL_REF.read_text(encoding="utf-8"))
    b2 = _b2_split(b2_predictions)
    items = _subset(subset, b2)
    n = items["count"]
    return {
        "format": FORMAT, "status": "FROZEN",
        "arms": {
            "F0": {"definition": f"untrained BF16 base {MODEL_ID}@{MODEL_REVISION}, re-run on the subset "
                                 "(the stored B2 answers are not F0 and not the comparator)",
                   "model_ref": _rel(BASE_MODEL_REF), "model_sha256": digest(base_ref["files"])},
            "FT": {"definition": "merged BF16 export of the single notebook-pipeline checkpoint (proof_eval export)",
                   "model_ref": "bound at analysis by --ft-model-ref; all FT runs must share its model_sha256"},
        },
        "evaluation_subset": items,
        "protocol_variants": variants,
        "runs": runs,
        "paired_run_rules": [f"each (arm, variant) run exactly once with --mode partial and one --sample-id per "
                             f"subset ID; status {FINISHED['statistical']}; run_manifest expected_ids, "
                             f"identity.ids_sha256 and predictions are exactly the {n} subset IDs",
                             "F0 and FT of one variant share identity.protocol_sha256",
                             "all statistical runs share identity.code_sha256",
                             "every record seed equals sample_seed(variant seed, id)",
                             "stored V8 correctness/parse_status must equal the re-score"],
        "scorer": {"parser": PARSER, "files": _scorer_files(),
                   "only_scorer": "strict team-final-answer-v8 re-score; proof_eval has no second (Qwen-style) "
                                  "score: Qwen's published MMMU scoring uses an LLM-assisted judge, which is not "
                                  "used without approval (CLAUDE.md §7.3)"},
        "primary_endpoint": {
            "statistic": f"mean over the {n} subset items of (FT seed-mean correctness - F0 seed-mean correctness) "
                         f"= FT - F0 of the seed mean of per-seed strict V8 accuracy on the subset; each per-seed "
                         f"accuracy = items answered correctly / {n} (= recovered count / {n})",
            "variants": statistical, "items": n,
            "ci": {"method": "paired item bootstrap over the subset items, percentile",
                   "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "level": 0.95},
        },
        "sensitivity": {"mcnemar_exact_two_sided": {"variant": SENSITIVITY,
                                                    "pairs": "per-item correctness F0 vs FT on the subset, seed 3407 only"}},
        "secondary_endpoints": {
            "length_cutoff_rate": "seed-mean share with finish_reason == length, F0 and FT",
            "missing_final_answer_rate": "seed-mean share with V8 parse_status != PARSED, F0 and FT",
            "per_discipline": "seed-mean accuracy per arm and FT-F0 with the same paired bootstrap",
            "per_subject": "seed-mean accuracy per arm and FT-F0 (table, no CI)",
        },
        "disciplines": DISCIPLINES,
        "single_answer_rule": SINGLE_ANSWER_RULE,
        "mmmu_val_budget": {"checkpoints_per_arm": 1, "runs": len(runs), "items_per_run": n,
                            "rule": "no other MMMU-val runs, seeds, items or checkpoints; MMMU-val never selects a "
                                    "checkpoint"},
        "course_facing_run": {"status": "DROPPED", "variant": COURSE,
                              "rule": "user decision 2026-10-31: the proof evaluates only the subset; the course "
                                      "protocol code path is kept but no course run is preregistered"},
        "limitations": [
            f"{n}-only: regressions on the {len(b2['right_ids'])} items B2 answered correctly cannot be detected "
            "(user decision 2026-10-31, budget)",
            "the subset was chosen because B2 answered these items wrongly once; a re-run of the untrained base "
            "answers some of them correctly by sampling alone (regression to the mean), so the comparator is the "
            "paired F0 re-run on the same IDs and seeds, never B2's 0 correct",
            f"accuracy on the {n} subset is not an MMMU-val accuracy and is not comparable to the 62.00% B2 "
            "baseline or the official 67.4",
            "the fast8k variants (8192 output tokens, 262144..1003520 pixels, 32 concurrent) differ from the "
            "course protocol (32768 tokens, official pixels, 2 concurrent)",
        ],
        "notes": [
            "user decision 2026-10-31 (speed and cost of the RunPod proof): 8K output, the training image budget "
            "and 32 concurrent requests for the paired arms (fast8k replaces c16); the proof runs only on the "
            f"{n} B2 V8 wrong items; the course-protocol run is dropped from the proof plan",
            "supersedes proof-prereg/prereg.json (format notebook-rl-proof-prereg-v1, 900 items, c16, course run)",
        ],
        "b2_reference": b2,
    }


def preregister(output: Path, subset: Path, b2_predictions: Path = B2_PREDICTIONS) -> str:
    """Write-once preregistration plus ``<output>.sha256``; returns the file hash."""
    output = Path(output)
    _write_once(output, json.dumps(build_preregistration(subset, b2_predictions), indent=2, sort_keys=True) + "\n")
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
    if prereg.get("format") != FORMAT or prereg.get("status") != "FROZEN":
        raise ValueError("proof_eval: not a frozen proof preregistration")
    if prereg["scorer"]["files"] != _scorer_files():
        raise ValueError("proof_eval: V8 scorer source changed since preregistration")
    return prereg, value


# ---------------------------------------------------------------------------------------------
# run ledger: one START line per evaluation run, so analyze can see an omitted repeat
# ---------------------------------------------------------------------------------------------

LEDGER_FORMAT = "proof-eval-ledger-v1"
GENESIS = "0" * 64
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_EVENT_KEYS = {"START": {"arm", "variant", "run_id", "preregistration_sha256"}, "FAIL": {"run_id", "reason"}}
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
    if any(e["preregistration_sha256"] != event["preregistration_sha256"] for e in starts.values()):
        raise ValueError("proof_eval: one ledger serves one preregistration")
    live = [r for r, e in starts.items() if (e["arm"], e["variant"]) == (event["arm"], event["variant"])
            and r not in failed]
    if live:
        raise ValueError(f"proof_eval: {event['arm']}/{event['variant']} already started as {live[0]}; "
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


def _append(path: Path, kind: str, **fields: str) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600), "r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        events, head = _parse(stream.read())
        event = {"format": LEDGER_FORMAT, "event": kind, "prev": head, "time": datetime.now(timezone.utc).isoformat(),
                 **fields}
        _check_event(events, event)
        stream.write(json.dumps(event, sort_keys=True).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event


def ledger_start(prereg_path: Path, arm: str, variant: str, run_id: str, ledger: Path) -> dict:
    prereg, prereg_sha = load_preregistration(prereg_path)
    if (arm, variant) not in {(r["arm"], r["variant"]) for r in prereg["runs"]}:
        raise ValueError(f"proof_eval: {(arm, variant)} is not a preregistered run")
    return _append(ledger, "START", arm=arm, variant=variant, run_id=run_id, preregistration_sha256=prereg_sha)


def ledger_fail(ledger: Path, run_id: str, reason: str, artifact_root: Path) -> dict:
    """Record a started run as FAILED (crashed, abandoned); a finished run directory cannot be declared failed."""
    summary = Path(artifact_root) / "runs" / run_id / "summary.json"
    status = json.loads(summary.read_text(encoding="utf-8")).get("status") if summary.is_file() else None
    if status in FINISHED.values():
        raise ValueError(f"proof_eval: {run_id} is {status}; a finished run cannot be replaced by marking it failed")
    return _append(ledger, "FAIL", run_id=run_id, reason=reason)


def check_ledger(ledger: Path, prereg_sha: str, runs: dict[tuple[str, str], dict]) -> dict:
    """Every analyzed run was started under its (arm, variant); no live started run is left out of the analysis."""
    events, head = read_ledger(ledger)
    starts = {e["run_id"]: e for e in events if e["event"] == "START"}
    failed = {e["run_id"] for e in events if e["event"] == "FAIL"}
    if any(e["preregistration_sha256"] != prereg_sha for e in starts.values()):
        raise ValueError("proof_eval: the ledger belongs to another preregistration")
    analyzed = {key: Path(run["run_dir"]).name for key, run in runs.items()}
    for key, run_id in analyzed.items():
        if run_id not in starts:
            raise ValueError(f"proof_eval: run {run_id} was never started in the ledger")
        if (starts[run_id]["arm"], starts[run_id]["variant"]) != key or run_id in failed:
            raise ValueError(f"proof_eval: run {run_id} is recorded in the ledger as another (arm, variant) or FAILED")
    for run_id, event in starts.items():
        key = (event["arm"], event["variant"])
        if run_id not in failed and analyzed.get(key) != run_id:
            raise ValueError(f"proof_eval: ledger run {run_id} {key} is not the one analyzed (omitted or replaced run)")
    return {"file_sha256": sha256_file(ledger), "head": head, "starts": len(starts), "failed_run_ids": sorted(failed)}


# ---------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------

def load_run(path: Path, prereg: dict) -> dict:
    """Validate one eval run directory (or its predictions.jsonl) against the preregistration.

    A statistical run must be the partial run on exactly the preregistered subset (status PARTIAL); a course
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
    if variant["role"] == "course":
        mode, expected = "full", sorted(set(prereg["b2_reference"]["wrong_ids"]) | set(prereg["b2_reference"]["right_ids"]))
    else:
        mode, expected = "partial", sorted(prereg["evaluation_subset"]["ids"])
    if not {"protocol_sha256", "model_sha256", "code_sha256"} <= identity.keys() or identity.get("mode") != mode:
        raise ValueError(f"proof_eval: {run_dir} is not an original {mode} evaluation run")
    if manifest.get("expected_ids") != expected or identity.get("ids_sha256") != digest(expected):
        raise ValueError(f"proof_eval: {run_dir} sample set differs from the preregistered {len(expected)} IDs")
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
                            "missing_final": score["parse_status"] != "PARSED"}
    arm = "F0" if identity["model_sha256"] == prereg["arms"]["F0"]["model_sha256"] else "FT"
    registration = run_dir / REGISTRATION_FILE
    return {"arm": arm, "variant": matches[0], "seed": variant["seed"], "run_dir": str(run_dir.resolve()),
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


def analyze(prereg_path: Path, run_paths: list[Path], ft_model_ref: Path, ledger: Path) -> dict:
    prereg, prereg_sha = load_preregistration(prereg_path)
    allowed = {(r["arm"], r["variant"]): r["role"] for r in prereg["runs"]}
    runs: dict[tuple[str, str], dict] = {}
    for path in run_paths:
        run = load_run(path, prereg)
        key = (run["arm"], run["variant"])
        if key not in allowed:
            raise ValueError(f"proof_eval: run {key} is not preregistered")
        if key in runs:
            raise ValueError(f"proof_eval: duplicate run for {key}; choosing among repeated runs is prohibited")
        runs[key] = run
    statistical = prereg["primary_endpoint"]["variants"]
    missing = [k for k, role in allowed.items() if role == "statistical" and k not in runs]
    if missing:
        raise ValueError(f"proof_eval: preregistered statistical runs missing: {missing}")
    ref = json.loads(Path(ft_model_ref).read_text(encoding="utf-8"))
    ft_sha = digest(ref["files"])
    if {r["identity"]["model_sha256"] for (arm, _), r in runs.items() if arm == "FT"} != {ft_sha}:
        raise ValueError("proof_eval: every FT run must evaluate the one bound FT export (one checkpoint per arm)")
    stat_runs = [runs[(arm, v)] for arm in ("F0", "FT") for v in statistical]
    if len({r["identity"]["code_sha256"] for r in stat_runs}) != 1:
        raise ValueError("proof_eval: statistical runs must share one evaluation code hash")
    for v in statistical:
        if runs[("F0", v)]["identity"]["protocol_sha256"] != runs[("FT", v)]["identity"]["protocol_sha256"]:
            raise ValueError(f"proof_eval: F0/FT protocol hash differs for {v}")
    for (arm, v), run in runs.items():
        reg = run["registration"]
        if allowed[(arm, v)] == "course":
            if reg is not None:
                raise ValueError("proof_eval: the course run must come from plain scripts/eval.sh")
        elif (reg is None or reg.get("protocol_id") != v
              or reg.get("protocol_file_sha256") != prereg["protocol_variants"][v]["sha256"]):
            raise ValueError(f"proof_eval: statistical run {(arm, v)} lacks its matching {REGISTRATION_FILE}")
    if len({digest({k: x for k, x in r["registration"].items() if k not in ("protocol_id", "protocol_file_sha256")})
            for r in stat_runs}) != 1:
        raise ValueError("proof_eval: statistical runs must share one registration (src and proof_eval bytes)")
    ledger_summary = check_ledger(ledger, prereg_sha, runs)

    ids = sorted(runs[("F0", statistical[0])]["items"])  # = the preregistered subset (load_run)
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("F0", "FT"):
        seeds = [runs[(arm, v)] for v in statistical]
        correct = seed_mean([{i: r["items"][i]["correct"] for i in ids} for r in seeds])
        per_seed = {str(r["seed"]): _mean(r["items"][i]["correct"] for i in ids) for r in seeds}
        count_per_seed = {str(r["seed"]): sum(r["items"][i]["correct"] for i in ids) for r in seeds}
        accuracy = _mean(correct.values())
        if not math.isclose(accuracy, _mean(per_seed.values()), abs_tol=1e-12):
            raise AssertionError("proof_eval: item mean and seed mean disagree")
        arms[arm] = {
            "item_mean": correct, "accuracy": accuracy, "accuracy_per_seed": per_seed,
            "correct_per_seed": count_per_seed,
            "length_cutoff_rate": _mean(seed_mean([{i: r["items"][i]["length"] for i in ids} for r in seeds]).values()),
            "missing_final_answer_rate": _mean(
                seed_mean([{i: r["items"][i]["missing_final"] for i in ids} for r in seeds]).values()),
        }
    diff = {i: arms["FT"]["item_mean"][i] - arms["F0"]["item_mean"][i] for i in ids}
    sensitivity = prereg["sensitivity"]["mcnemar_exact_two_sided"]["variant"]
    sens = runs[("F0", sensitivity)]["items"], runs[("FT", sensitivity)]["items"]
    report: dict[str, Any] = {
        "format": "notebook-rl-proof-report-v1", "preregistration_sha256": prereg_sha,
        "single_answer_rule": prereg["single_answer_rule"], "ledger": ledger_summary,
        "runs": [{k: r[k] for k in ("arm", "variant", "seed", "run_dir", "identity", "registration")}
                 for r in runs.values()],
        "ft_model_ref": {"sha256": sha256_file(Path(ft_model_ref)), "model_sha256": ft_sha,
                         "export": ref.get("export")},
        "primary": {"items": len(ids), "scorer": PARSER,
                    "FT_minus_F0": _mean(diff.values()), "ci95": paired_bootstrap(list(diff.values())),
                    "F0_accuracy": arms["F0"]["accuracy"], "FT_accuracy": arms["FT"]["accuracy"],
                    "accuracy_per_seed": {a: arms[a]["accuracy_per_seed"] for a in arms},
                    "correct_per_seed": {a: arms[a]["correct_per_seed"] for a in arms}},
        "sensitivity_mcnemar_seed3407": mcnemar_exact({i: sens[0][i]["correct"] for i in ids},
                                                      {i: sens[1][i]["correct"] for i in ids}),
        "secondary": {a: {k: arms[a][k] for k in ("length_cutoff_rate", "missing_final_answer_rate")}
                      for a in arms},
        "limitations": prereg.get("limitations", []),
        "per_discipline": {}, "per_subject": {},
    }
    for name, subjects in DISCIPLINES.items():
        members = [i for i in ids if _subject(i) in subjects]
        if not members:
            continue
        report["per_discipline"][name] = {
            "n": len(members), **{a: _mean(arms[a]["item_mean"][i] for i in members) for a in arms},
            "FT_minus_F0": _mean(diff[i] for i in members), "ci95": paired_bootstrap([diff[i] for i in members])}
    for subject in SUBJECTS:
        members = [i for i in ids if _subject(i) == subject]
        if not members:
            continue
        report["per_subject"][subject] = {"n": len(members), **{a: _mean(arms[a]["item_mean"][i] for i in members)
                                                                for a in arms},
                                          "FT_minus_F0": _mean(diff[i] for i in members)}
    course = runs.get(("FT", COURSE))
    if ("FT", COURSE) not in allowed:
        report["course_facing"] = dict(prereg["course_facing_run"])
    elif course is None:
        report["course_facing"] = {"status": "NOT_RUN"}
    else:
        wrong, right = prereg["b2_reference"]["wrong_ids"], prereg["b2_reference"]["right_ids"]
        c = {i: row["correct"] for i, row in course["items"].items()}
        report["course_facing"] = {
            "status": "COMPLETE", "variant": COURSE, "seed": 3407, "correct": sum(c.values()),
            "accuracy": _mean(c.values()), "b2_wrong_recovery": _mean(c[i] for i in wrong),
            "b2_right_retention": _mean(c[i] for i in right),
            "note": "single preregistered course-protocol FT run; not pooled with the statistical runs"}
    return report


def render_markdown(report: dict) -> str:
    p = report["primary"]
    pct = lambda x: f"{100 * x:.2f}"  # noqa: E731
    lines = ["# Notebook-RL MMMU-val proof report", "",
             f"Preregistration sha256 `{report['preregistration_sha256']}`", "",
             f"Primary ({p['items']} preregistered items, statistical variants, seeds "
             f"{'/'.join(p['accuracy_per_seed']['F0'])}, seed-mean strict V8 only): "
             f"F0 {pct(p['F0_accuracy'])}%, "
             f"FT {pct(p['FT_accuracy'])}%, FT-F0 {pct(p['FT_minus_F0'])} pp "
             f"[95% paired bootstrap {pct(p['ci95'][0])}, {pct(p['ci95'][1])}]", "",
             "| arm | " + " | ".join(f"seed {s} correct/{p['items']}" for s in p["accuracy_per_seed"]["F0"]) + " |",
             "|---|" + "---|" * len(p["accuracy_per_seed"]["F0"])]
    for arm, seeds in p["accuracy_per_seed"].items():
        lines.append(f"| {arm} | " + " | ".join(f"{p['correct_per_seed'][arm][s]} ({pct(v)}%)"
                                                for s, v in seeds.items()) + " |")
    m = report["sensitivity_mcnemar_seed3407"]
    lines += ["", f"McNemar exact (seed 3407): F0-only right {m['f0_right_ft_wrong']}, "
              f"FT-only right {m['f0_wrong_ft_right']}, p = {m['p_value']:.4g}", "",
              "| secondary | F0 | FT |", "|---|---|---|"]
    for key in report["secondary"]["F0"]:
        lines.append(f"| {key} | {pct(report['secondary']['F0'][key])} | {pct(report['secondary']['FT'][key])} |")
    lines += ["", "| discipline | n | F0 | FT | FT-F0 | 95% CI |", "|---|---|---|---|---|---|"]
    for name, row in report["per_discipline"].items():
        lines.append(f"| {name} | {row['n']} | {pct(row['F0'])} | {pct(row['FT'])} | {pct(row['FT_minus_F0'])} | "
                     f"[{pct(row['ci95'][0])}, {pct(row['ci95'][1])}] |")
    lines += ["", "| subject | n | F0 | FT | FT-F0 |", "|---|---|---|---|---|"]
    for name, row in report["per_subject"].items():
        lines.append(f"| {name} | {row['n']} | {pct(row['F0'])} | {pct(row['FT'])} | {pct(row['FT_minus_F0'])} |")
    course = report["course_facing"]
    lines += ["", "Course-facing run (FT, mmmu-val-v8, c2, seed 3407): " + (
        f"{course['correct']}/900 = {pct(course['accuracy'])}% (not pooled)" if course["status"] == "COMPLETE"
        else course["status"]), ""]
    lines += [f"- Limitation: {text}" for text in report["limitations"]]
    lines += ["", report["single_answer_rule"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run-eval"]:
        return run_eval(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run-eval", help="scripts/eval.sh arguments for one fast8k variant file (passed through verbatim)")
    exp = sub.add_parser("export", help="merge one local_train checkpoint into a BF16 eval model + model_ref.json")
    exp.add_argument("--checkpoint", type=Path, required=True)
    exp.add_argument("--binding", type=Path, required=True, help="binding JSON or launch.json holding 'binding'")
    exp.add_argument("--base", type=Path, required=True, help="pinned BF16 snapshot directory")
    exp.add_argument("--output", type=Path, required=True, help="new directory: model/ + model_ref.json")
    exp.add_argument("--code-commit", required=True, help="40-hex Git commit of the training code")
    pre = sub.add_parser("preregister", help="write the frozen preregistration and its .sha256")
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--subset", type=Path, required=True, help="proof-eval-subset-v1 JSON: the B2 V8 wrong IDs")
    pre.add_argument("--b2-predictions", type=Path, default=B2_PREDICTIONS)
    start = sub.add_parser("ledger-start", help="append the START of one evaluation run (before run-eval/eval.sh); "
                           "refused when that (arm, variant) has a started run not recorded as FAILED")
    start.add_argument("--prereg", type=Path, required=True)
    start.add_argument("--arm", choices=("F0", "FT"), required=True)
    start.add_argument("--variant", required=True, help="a preregistered protocol variant id")
    start.add_argument("--run-id", required=True, help="the run directory name artifact_root/runs/<run-id>")
    start.add_argument("--ledger", type=Path, required=True)
    fail = sub.add_parser("ledger-fail", help="record a started run as FAILED (refused for a finished COMPLETE/PARTIAL run)")
    fail.add_argument("--ledger", type=Path, required=True)
    fail.add_argument("--run-id", required=True)
    fail.add_argument("--reason", required=True)
    fail.add_argument("--artifact-root", type=Path, default=None, help="default: MMDL_ARTIFACT_ROOT, as run-eval")
    ana = sub.add_parser("analyze", help="score preregistered runs; writes <output>.json and <output>.md")
    ana.add_argument("--prereg", type=Path, required=True)
    ana.add_argument("--ledger", type=Path, required=True, help="the run ledger: every analyzed run was started in it "
                     "and no live started run is left out")
    ana.add_argument("--runs", type=Path, nargs="+", required=True, help="eval run dirs or their predictions.jsonl")
    ana.add_argument("--ft-model-ref", type=Path, required=True)
    ana.add_argument("--output", type=Path, required=True, help="report path; .json and .md are written")
    args = parser.parse_args(argv)
    if args.command == "export":
        ref = export_for_eval(args.checkpoint, args.binding, args.base, args.output, args.code_commit)
        print(f"EXPORTED {args.output / 'model_ref.json'} unmerged_inference_bit_equivalent="
              f"{ref['export']['unmerged_inference_bit_equivalent']}")
    elif args.command == "preregister":
        print(f"PREREGISTERED {args.output} sha256={preregister(args.output, args.subset, args.b2_predictions)}")
    elif args.command == "ledger-start":
        event = ledger_start(args.prereg, args.arm, args.variant, args.run_id, args.ledger)
        print(f"LEDGER_STARTED {event['arm']} {event['variant']} {event['run_id']}")
    elif args.command == "ledger-fail":
        from mmdl.runtime.artifacts import default_path

        root = args.artifact_root or default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts")
        print(f"LEDGER_FAILED {ledger_fail(args.ledger, args.run_id, args.reason, root)['run_id']}")
    else:
        report = analyze(args.prereg, args.runs, args.ft_model_ref, args.ledger)
        _write_once(args.output.with_suffix(".json"), json.dumps(report, indent=2, sort_keys=True) + "\n")
        _write_once(args.output.with_suffix(".md"), render_markdown(report))
        print(f"ANALYZED {args.output.with_suffix('.json')} FT-F0={report['primary']['FT_minus_F0']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
