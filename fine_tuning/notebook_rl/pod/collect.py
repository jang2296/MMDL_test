"""Recovery and cleanup bookkeeping. Never calls RunPod or any API; the user performs and
confirms every remote action, this module only records evidence.

pack      (on the Pod)  tar of CLOSED run dirs only + sha256 manifest; --unclosed-dead-owner also
                        takes a hard-killed run (dead owner, no process env names the run dir),
                        labelled UNCLOSED in the manifest
pack-eval (on the Pod)  one MMMU evaluation stage (eval-f0|eval-ft; eval-course is not planned): its run dir
                        plus exactly what `proof_eval analyze` reads (the stage's planned run dirs under
                        artifact_root, the ledger, the prereg and, for FT, model_ref.json; never model
                        weights), same manifest/verify/ledger flow, manifest "kind": "eval"
verify    (local)       re-hash the downloaded tar, validate members, safe extraction, receipt; an eval
                        pack and its extraction directory must lie under the private artifact root
                        ($MMDL_ARTIFACT_ROOT, default ~/mmdl-artifacts), never in the repo
state                   append-only, hash-chained ledger with the fixed cleanup order

CLI: collect pack --out-dir DIR --name NAME [--unclosed-dead-owner] RUN_DIR [RUN_DIR ...]
     collect pack-eval --params EVAL_STAGE_PARAMS.json --stage STAGE --run-dir RUN_DIR --out-dir DIR [--name NAME]
     collect verify --archive TAR --manifest JSON --extract-to NEW_DIR --receipt OUT.json
     collect state --ledger FILE show
     collect state --ledger FILE append STATE [--evidence key=value ...] [--receipt-file FILE]
                   [--run-ledger FILE ...]
OWNED_STORAGE_DELETED needs --run-ledger for every run that used the storage (this ledger included);
each must already hold REMOTE_VERIFIED and LOCAL_VERIFIED (one cleanup ledger per pack, training or eval).
LOCAL_VERIFIED needs the typed pod_id of REMOTE_VERIFIED to equal the pod_id the pack recorded on the Pod
(RUNPOD_POD_ID), via the receipt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .common import (HEX64, HashingReader, parse_utc, process_startticks, read_json, sha256_bytes,
                     sha256_file, utc_now, write_json)
from .runner import TERMINAL_PREFIXES

PACK_VERSION = "pod-pack-v1"
EVAL_KIND = "eval"
# Exactly what proof_eval.load_run reads from a run directory (no samples/, prompts/, logs, source.patch).
EVAL_RUN_FILES = ("run_manifest.json", "summary.json", "predictions.jsonl", "resolved_eval_config.yaml",
                  "variant_registration.json")
EVAL_REQUIRED = EVAL_RUN_FILES[:4]  # variant_registration.json: required for the fast8k (statistical) runs
LEDGER_VERSION = "pod-cleanup-ledger-v1"
ORDER = ("REMOTE_VERIFIED", "LOCAL_VERIFIED", "POD_TERMINATED", "OWNED_STORAGE_DELETED",
         "CLEANUP_VERIFIED")
RESULT_FILES = ("result.json", "receipt.json", "throughput_receipt.json")
SKEW = dt.timedelta(seconds=60)
PROC = Path("/proc")


class CollectError(ValueError):
    pass


# ---------------------------------------------------------------- pack

def owner_alive(run_dir: Path) -> bool:
    """True when the recorded owner PID is alive with the recorded start time.

    Fails closed: a PID that exists but whose /proc start time cannot be read counts as alive.
    """
    try:
        owner = read_json(run_dir / "owner.json")
        pid, ticks = int(owner["pid"]), int(owner["startticks"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CollectError(f"{run_dir.name}: owner.json missing or unreadable ({exc})") from exc
    current = process_startticks(pid)
    if current is not None:
        return current == ticks
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_dir_holders(run_dir: Path) -> list[str]:
    """PIDs whose environment still carries MMDL_RUN_DIR=<run_dir> (start.sh exports it).

    A SIGKILLed start.sh leaves its session-leader children running; they keep the variable.
    Fails closed: an environ that cannot be read is reported as a holder, and so is a missing
    /proc. Only a process that vanished mid-scan is ignored.
    """
    if not PROC.is_dir():
        raise CollectError(f"{PROC} unavailable: cannot show that nothing still uses the run dir")
    wanted = {os.fsencode(f"MMDL_RUN_DIR={path}") for path in (os.path.abspath(run_dir),
                                                              run_dir.resolve())}
    holders = []
    for entry in PROC.glob("[0-9]*"):
        try:
            environment = (entry / "environ").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            holders.append(f"{entry.name} (environ unreadable)")
            continue
        if wanted & set(environment.split(b"\0")):
            holders.append(entry.name)
    return holders


def closed_run(run_dir: Path, allow_unclosed: bool = False,
               results: tuple[str, ...] = RESULT_FILES) -> str:
    """Raise unless the run is closed: owner gone, terminal state, and a result when it succeeded.

    A failed run (PREFLIGHT_FAILED, SMOKE_FAILED, INTERRUPTED, MAIN_EXIT != 0) is packable
    without a result file: its run_state.json and logs are the evidence that must be recovered.
    With allow_unclosed a non-terminal state (hard kill: no INTERRUPTED was ever written) is
    also accepted, but only when no process env still names the run dir; returns "UNCLOSED"
    for those and "CLOSED" otherwise. A MAIN_EXIT 0 run must hold one of `results` (an evaluation
    stage keeps its results outside its run dir and passes chain.jsonl).
    """
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise CollectError(f"{run_dir}: not a real directory")
    if owner_alive(run_dir):
        raise CollectError(f"{run_dir.name}: owner process is still alive")
    try:
        state = str(read_json(run_dir / "run_state.json")["state"])
    except (OSError, ValueError, KeyError) as exc:
        raise CollectError(f"{run_dir.name}: run_state.json missing or unreadable") from exc
    if not state.startswith(TERMINAL_PREFIXES):
        if not allow_unclosed:
            raise CollectError(f"{run_dir.name}: run is not closed (state {state})")
        holders = run_dir_holders(run_dir)
        if holders:
            raise CollectError(f"{run_dir.name}: processes still name this run dir: {holders}")
        return "UNCLOSED"
    if state == "MAIN_EXIT 0" and not any((run_dir / name).is_file() for name in results):
        raise CollectError(f"{run_dir.name}: successful run lacks result/receipt {results}")
    return "CLOSED"


def _check_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise CollectError("archive name must match [A-Za-z0-9._-]+")


def pack(run_dirs: list[Path], out_dir: Path, name: str,
         unclosed_dead_owner: bool = False) -> dict[str, Any]:
    _check_name(name)
    names = [path.name for path in run_dirs]
    if not run_dirs or len(set(names)) != len(names):
        raise CollectError("one or more run dirs with distinct names required")
    unclosed = [path.name for path in run_dirs
                if closed_run(path, unclosed_dead_owner) == "UNCLOSED"]

    def fill(tar: tarfile.TarFile, records: list[dict[str, Any]]) -> None:
        for run_dir in run_dirs:
            _tar_run_dir(tar, run_dir, records)

    return _seal(out_dir, name, fill, runs=names, closure="UNCLOSED" if unclosed else "CLOSED",
                 unclosed_runs=unclosed)


def _seal(out_dir: Path, name: str, fill: Callable[[tarfile.TarFile, list[dict[str, Any]]], None],
          **fields: Any) -> dict[str, Any]:
    """Write <name>.tar (atomic rename, never overwrites) and <name>.manifest.json; fill adds members."""
    out_dir.mkdir(parents=True, exist_ok=True)
    archive, manifest_path = out_dir / f"{name}.tar", out_dir / f"{name}.manifest.json"
    if archive.exists() or manifest_path.exists():
        raise CollectError("archive or manifest already exists; refusing to overwrite")
    records: list[dict[str, Any]] = []
    partial = out_dir / f".{name}.tar.partial"
    try:
        with tarfile.open(partial, "x") as tar:
            fill(tar, records)
        os.rename(partial, archive)
    finally:
        partial.unlink(missing_ok=True)  # only the temp file this call created
    manifest = {"version": PACK_VERSION, "archive": archive.name,
                "archive_bytes": archive.stat().st_size, "archive_sha256": sha256_file(archive),
                **fields, "files": records, "created_utc": utc_now(),
                "pod_id": os.environ.get("RUNPOD_POD_ID")}
    write_json(manifest_path, manifest)
    return manifest


def _add_member(tar: tarfile.TarFile, path: Path, arcname: str, records: list[dict[str, Any]]) -> None:
    if path.is_symlink() or not (path.is_dir() or path.is_file()):
        raise CollectError(f"{path}: only regular files and directories are packed")
    info = tar.gettarinfo(str(path), arcname=arcname)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    if path.is_dir():
        tar.addfile(info)
        return
    with open(path, "rb") as stream:
        reader = HashingReader(stream)
        tar.addfile(info, reader)
    if reader.count != info.size:
        raise CollectError(f"{path}: changed while packing")
    records.append({"path": info.name, "bytes": info.size, "sha256": reader.digest.hexdigest(),
                    "source": os.path.abspath(path)})


def _tar_run_dir(tar: tarfile.TarFile, run_dir: Path, records: list[dict[str, Any]]) -> None:
    for path in sorted([run_dir, *run_dir.rglob("*")]):
        _add_member(tar, path, str(Path(run_dir.name) / path.relative_to(run_dir)), records)


# ---------------------------------------------------------------- pack-eval

def _plain_file(path: Path, root: Path) -> Path:
    """A regular file whose real path is exactly `path` below the real `root`: no symlink anywhere."""
    try:
        wanted = root.resolve() / path.relative_to(root)
    except ValueError as exc:
        raise CollectError(f"{path}: outside {root}") from exc
    if path.is_symlink() or not path.is_file() or path.resolve() != wanted:
        raise CollectError(f"{path}: not a regular file inside {root} (symlinks are refused)")
    return path


def pack_eval(params: Any, stage: str, run_dir: Path, out_dir: Path,
              name: str | None = None) -> dict[str, Any]:
    """Pack one evaluation stage: its start.sh run dir and the inputs of `proof_eval analyze`.

    Packed (archive paths): <run dir name>/** (logs, chain.jsonl), eval/runs/<run id>/<EVAL_RUN_FILES>
    for each run the stage plans, eval/<ledger>, eval/<prereg> + .sha256 and, for eval-ft/eval-course,
    eval/model_ref.json. The merged model weights are never packed. Every planned run needs its ledger
    START and must be finished (summary PARTIAL for the subset runs, COMPLETE for a full run), or be recorded
    FAILED in the ledger (packed as it is, marked FAILED).
    """
    from .. import proof_eval  # lazy: numpy/yaml are not needed by pack, verify or state
    from . import chain, eval_stage

    p = eval_stage._check(stage, params)
    name = name or run_dir.name
    _check_name(name)
    if run_dir.name == "eval":
        raise CollectError("the start.sh run dir may not be named 'eval' (archive prefix)")
    if not os.environ.get("RUNPOD_POD_ID"):
        raise CollectError("RUNPOD_POD_ID is not set: the pack records the Pod it ran on")
    # The run dir must have run exactly the spec these params plan (the bytes `eval_stage plan` writes).
    spec_sha = sha256_bytes((json.dumps(eval_stage.plan(stage, p), indent=2, sort_keys=True) + "\n").encode())
    closed_run(run_dir, results=(chain.LOG_NAME,))
    if not (run_dir / chain.LOG_NAME).is_file():
        raise CollectError(f"{run_dir.name}: no {chain.LOG_NAME}, the chain never started; "
                           "recover its logs with `collect pack`")
    try:
        log = chain.verify_log(run_dir / chain.LOG_NAME, spec_sha)
    except (OSError, ValueError) as exc:
        raise CollectError(f"{run_dir.name}: {chain.LOG_NAME} is not the log of the spec these params plan "
                           f"({exc})") from exc
    if read_json(run_dir / "run_state.json")["state"] == "MAIN_EXIT 0" and (
            not log or (log[-1]["event"], log[-1].get("status")) != ("CHAIN_END", "COMPLETE")):
        raise CollectError(f"{run_dir.name}: MAIN_EXIT 0 without a COMPLETE chain end")

    volume_eval, artifacts = Path(p["volume_root"]) / eval_stage.EVAL_DIR, Path(p["artifact_root"])
    ledger = _plain_file(Path(p["ledger"]), volume_eval)
    events, _ = proof_eval.read_ledger(ledger)  # hash chain and ledger rules
    starts = {e["run_id"]: e for e in events if e["event"] == "START"}
    failed = {e["run_id"] for e in events if e["event"] == "FAIL"}
    sources: list[tuple[Path, str]] = []
    eval_runs: list[dict[str, Any]] = []
    for arm, variant in eval_stage._runs(stage):
        rid = eval_stage.run_id(p["run_prefix"], arm, variant)
        start = starts.get(rid)
        if (start is None or (start["arm"], start["variant"]) != (arm, variant)
                or start["preregistration_sha256"] != p["prereg_sha256"]):
            raise CollectError(f"{rid}: no ledger START for {arm}/{variant} under this preregistration")
        base = artifacts / "runs" / rid
        present = [f for f in EVAL_RUN_FILES if (base / f).exists() or (base / f).is_symlink()]
        for file in present:
            _plain_file(base / file, artifacts)
        summary = read_json(base / "summary.json") if "summary.json" in present else None
        role = proof_eval.VARIANTS[variant][3]
        finished = proof_eval.FINISHED[role]  # COMPLETE (full course run) or PARTIAL (the subset runs)
        done = isinstance(summary, dict) and summary.get("status") == finished
        if rid in failed:
            if done:
                raise CollectError(f"{rid}: the ledger records FAILED but summary.json is {finished}")
            status = "FAILED"
        elif done:
            missing = [f for f in (*EVAL_REQUIRED, proof_eval.REGISTRATION_FILE if role == "statistical" else None)
                       if f and f not in present]
            if missing:
                raise CollectError(f"{rid}: {finished} run lacks {missing}")
            status = finished
        else:
            raise CollectError(f"{rid}: summary is not {finished} and the ledger has no FAIL for it "
                               "(record it with proof_eval ledger-fail first)")
        sources += [(base / f, f"eval/runs/{rid}/{f}") for f in present]
        eval_runs.append({"run_id": rid, "arm": arm, "variant": variant, "status": status,
                          "dir": f"eval/runs/{rid}", "files": present})
    prereg = _plain_file(Path(p["prereg"]), volume_eval)
    sidecar = _plain_file(Path(p["prereg"] + ".sha256"), volume_eval)
    if sha256_file(prereg) != p["prereg_sha256"] or sidecar.read_text(encoding="ascii").strip() != p["prereg_sha256"]:
        raise CollectError("prereg or its .sha256 differs from params.prereg_sha256")
    sources += [(ledger, f"eval/{ledger.name}"), (prereg, f"eval/{prereg.name}"),
                (sidecar, f"eval/{sidecar.name}")]
    model_ref_sha = None
    if stage != "eval-f0":
        export = Path(p["ft_export"])
        model_ref = _plain_file(export / "model_ref.json", export)  # the manifest only, never export/model
        model_ref_sha = sha256_file(model_ref)
        if stage == "eval-course" and model_ref_sha != p["ft_model_ref_sha256"]:
            raise CollectError("model_ref.json differs from params.ft_model_ref_sha256")
        sources.append((model_ref, "eval/model_ref.json"))
    if len({arc for _, arc in sources}) != len(sources):
        raise CollectError("two eval inputs would share one archive path")

    def fill(tar: tarfile.TarFile, records: list[dict[str, Any]]) -> None:
        _tar_run_dir(tar, run_dir, records)
        for path, arcname in sources:
            _add_member(tar, path, arcname, records)

    return _seal(out_dir, name, fill, kind=EVAL_KIND, stage=stage, runs=[run_dir.name], closure="CLOSED",
                 unclosed_runs=[], eval_runs=eval_runs, eval_run_ids=[r["run_id"] for r in eval_runs],
                 spec_sha256=spec_sha, ledger_sha256=sha256_file(ledger), prereg_sha256=p["prereg_sha256"],
                 model_ref_sha256=model_ref_sha)


# ---------------------------------------------------------------- verify

def check_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = tar.getmembers()
    for member in members:
        parts = Path(member.name).parts
        if (not member.name or Path(member.name).is_absolute() or ".." in parts
                or member.name.startswith(("/", "\\")) or "\\" in member.name):
            raise CollectError(f"unsafe member path: {member.name!r}")
        if not (member.isreg() or member.isdir()):
            raise CollectError(f"member is not a regular file or directory "
                               f"(symlink/hardlink/device refused): {member.name!r}")
    return members


def private_root() -> Path:
    """Where eval packs (raw MMMU responses, never published) may live locally."""
    return Path(os.environ.get("MMDL_ARTIFACT_ROOT") or Path.home() / "mmdl-artifacts").resolve()


def _require_private(path: Path, label: str) -> None:
    if not path.resolve().is_relative_to(private_root()):
        raise CollectError(f"{label} {path} must lie under {private_root()} (private, never the repo)")


def verify_archive(archive: Path, manifest_path: Path, extract_to: Path,
                   receipt: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("version") != PACK_VERSION:
        raise CollectError("not a pod-pack-v1 manifest")
    is_eval = manifest.get("kind") == EVAL_KIND
    if is_eval:
        _require_private(archive, "eval archive")
        _require_private(extract_to, "eval extraction directory")
    if sha256_file(archive) != manifest["archive_sha256"]:
        raise CollectError("archive sha256 differs from the pack manifest")
    if extract_to.exists() or receipt.exists():
        raise CollectError("extraction directory or receipt already exists")
    with tarfile.open(archive, "r") as tar:
        members = check_members(tar)
        listed = {m.name for m in members if m.isreg()}
        if listed != {f["path"] for f in manifest["files"]}:
            raise CollectError("archive file set differs from the pack manifest")
        extract_to.mkdir(parents=True, mode=0o700 if is_eval else 0o777)
        tar.extractall(extract_to, members=members, filter="data")
    root = extract_to.resolve()
    for record in manifest["files"]:
        path = (extract_to / record["path"]).resolve()
        if root not in path.parents or path.is_symlink():
            raise CollectError(f"extracted path escapes the target: {record['path']}")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise CollectError(f"extracted file differs from manifest: {record['path']}")
    result = {"version": "pod-local-receipt-v1", "status": "LOCAL_VERIFIED",
              "archive_sha256": manifest["archive_sha256"],
              "manifest_sha256": sha256_file(manifest_path),
              "closure": manifest.get("closure", "CLOSED"), "pod_id": manifest.get("pod_id"),
              "files": len(manifest["files"]),
              "bytes": sum(f["bytes"] for f in manifest["files"]), "extracted_to": str(root),
              "verified_utc": utc_now()}
    if is_eval:
        result |= {"kind": EVAL_KIND, "stage": manifest["stage"],
                   "eval_runs": [{k: r[k] for k in ("run_id", "status")} for r in manifest["eval_runs"]]}
    write_json(receipt, result)
    return result


# ---------------------------------------------------------------- ledger

def _confirmed_user(evidence: dict[str, str]) -> None:
    if evidence.get("confirmed_by") != "user":
        raise CollectError("evidence confirmed_by=user required (the user performs and confirms "
                           "remote actions; nothing here calls an API)")


def _require(evidence: dict[str, str], *keys: str) -> None:
    missing = [key for key in keys if not evidence.get(key)]
    if missing:
        raise CollectError(f"evidence missing: {missing}")


def _when(evidence: dict[str, str], key: str) -> dt.datetime:
    try:
        return parse_utc(evidence[key])
    except ValueError as exc:
        raise CollectError(f"{key} must be a timezone-aware ISO time ({exc})") from exc


def check_evidence(state: str, evidence: dict[str, str], records: list[dict[str, Any]],
                   now: dt.datetime, receipt: dict[str, Any] | None = None) -> None:
    first = records[0]["evidence"] if records else {}
    if state == "REMOTE_VERIFIED":
        _require(evidence, "archive_sha256", "manifest_sha256", "pod_id")
        for key in ("archive_sha256", "manifest_sha256"):
            if not HEX64.match(evidence[key]):
                raise CollectError(f"{key} must be a lowercase sha256")
    elif state == "LOCAL_VERIFIED":
        _require(evidence, "archive_sha256", "local_receipt_sha256")
        if not HEX64.match(evidence["local_receipt_sha256"]):
            raise CollectError("local_receipt_sha256 must be a lowercase sha256")
        if evidence["archive_sha256"] != first["archive_sha256"]:
            raise CollectError("archive_sha256 differs from REMOTE_VERIFIED")
        receipt = receipt or {}
        if (receipt.get("version") != "pod-local-receipt-v1"
                or receipt.get("status") != "LOCAL_VERIFIED"
                or receipt.get("archive_sha256") != first["archive_sha256"]
                or receipt.get("manifest_sha256") != first["manifest_sha256"]):
            raise CollectError("LOCAL_VERIFIED needs the receipt written by `collect verify` for "
                               "the REMOTE_VERIFIED archive and manifest (--receipt-file)")
        if receipt.get("pod_id") != first["pod_id"]:
            raise CollectError(f"pod_id typed at REMOTE_VERIFIED ({first['pod_id']}) differs from "
                               f"the pod_id the pack recorded ({receipt.get('pod_id')})")
    elif state == "POD_TERMINATED":
        _require(evidence, "pod_id", "terminated_utc")
        _confirmed_user(evidence)
        if evidence["pod_id"] != first["pod_id"]:
            raise CollectError("pod_id differs from REMOTE_VERIFIED")
        stamp = _when(evidence, "terminated_utc")
        if stamp > now + SKEW or stamp < parse_utc(records[-1]["utc"]) - SKEW:
            raise CollectError("terminated_utc must lie between LOCAL_VERIFIED and now")
    elif state == "OWNED_STORAGE_DELETED":
        _require(evidence, "deleted_resources", "deleted_utc", "run_ledgers")
        _confirmed_user(evidence)
        stamp = _when(evidence, "deleted_utc")
        if stamp > now + SKEW or stamp < parse_utc(records[-1]["evidence"]["terminated_utc"]) - SKEW:
            raise CollectError("deleted_utc must lie between terminated_utc and now")
    elif state == "CLEANUP_VERIFIED":
        _require(evidence, "residual_query_utc", "residual_count")
        _confirmed_user(evidence)
        if evidence["residual_count"] != "0":
            raise CollectError("residual_count must be 0 to verify cleanup")
        stamp = _when(evidence, "residual_query_utc")
        if stamp > now + SKEW or stamp < parse_utc(records[-1]["evidence"]["deleted_utc"]) - SKEW:
            raise CollectError("residual_query_utc must lie between deleted_utc and now")


def _record_hash(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records, previous = [], None
    for number, line in enumerate(path.read_text().splitlines()):
        record = json.loads(line)
        expected = ORDER[number] if number < len(ORDER) else None
        if (record.get("version") != LEDGER_VERSION or record.get("seq") != number
                or record.get("state") != expected or record.get("prev_sha256") != previous
                or record.get("record_sha256") != _record_hash(record)):
            raise CollectError(f"ledger corrupt or out of order at line {number + 1}")
        records.append(record)
        previous = record["record_sha256"]
    return records


def run_ledgers_evidence(current: Path, ledgers: list[Path]) -> str:
    """Every run that used the storage has REMOTE_VERIFIED and LOCAL_VERIFIED; JSON summary."""
    resolved = [Path(ledger).resolve() for ledger in ledgers]
    if not resolved or len(set(resolved)) != len(resolved) or current.resolve() not in resolved:
        raise CollectError("OWNED_STORAGE_DELETED needs --run-ledger for every run that used the "
                           "storage, this ledger included, each once")
    summary = []
    for ledger in resolved:
        records = read_ledger(ledger)
        if len(records) < 2:
            raise CollectError(f"{ledger}: run lacks REMOTE_VERIFIED + LOCAL_VERIFIED")
        summary.append({"ledger": str(ledger), "pod_id": records[0]["evidence"]["pod_id"],
                        "archive_sha256": records[0]["evidence"]["archive_sha256"],
                        "last_record_sha256": records[-1]["record_sha256"]})
    return json.dumps(summary, sort_keys=True)


def append_state(path: Path, state: str, evidence: dict[str, str],
                 now: dt.datetime | None = None, receipt_file: Path | None = None,
                 run_ledgers: list[Path] | None = None) -> dict[str, Any]:
    if state not in ORDER:
        raise CollectError(f"unknown state {state!r}; order is {' -> '.join(ORDER)}")
    receipt = None
    if receipt_file is not None:
        receipt = read_json(receipt_file)
        if not isinstance(receipt, dict):
            raise CollectError("receipt file is not a JSON object")
        evidence = {**evidence, "local_receipt_sha256": sha256_file(receipt_file)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        records = read_ledger(path)
        expected = ORDER[len(records)] if len(records) < len(ORDER) else None
        if state != expected:
            raise CollectError(f"out-of-order transition: next must be {expected}, got {state}")
        if state == "OWNED_STORAGE_DELETED":
            evidence = {**evidence, "run_ledgers": run_ledgers_evidence(path, run_ledgers or [])}
        check_evidence(state, evidence, records, now or dt.datetime.now(dt.timezone.utc), receipt)
        record: dict[str, Any] = {"version": LEDGER_VERSION, "seq": len(records), "state": state,
                                  "utc": utc_now(), "evidence": evidence,
                                  "prev_sha256": records[-1]["record_sha256"] if records else None}
        record["record_sha256"] = _record_hash(record)
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--unclosed-dead-owner", action="store_true",
                   help="also pack a hard-killed run whose owner is dead and that no process "
                        "still names; the manifest is labelled UNCLOSED")
    p.add_argument("run_dirs", nargs="+")
    e = sub.add_parser("pack-eval")
    e.add_argument("--params", required=True, help="the eval-stage-params-v1 file `eval_stage plan` read")
    e.add_argument("--stage", required=True, choices=("eval-f0", "eval-ft", "eval-course"))
    e.add_argument("--run-dir", required=True, help="the stage's start.sh run dir")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--name", help="archive name, default: the run dir name")
    v = sub.add_parser("verify")
    v.add_argument("--archive", required=True)
    v.add_argument("--manifest", required=True)
    v.add_argument("--extract-to", required=True)
    v.add_argument("--receipt", required=True)
    s = sub.add_parser("state")
    s.add_argument("--ledger", required=True)
    action = s.add_subparsers(dest="action", required=True)
    action.add_parser("show")
    a = action.add_parser("append")
    a.add_argument("state")
    a.add_argument("--evidence", action="append", default=[], metavar="KEY=VALUE")
    a.add_argument("--receipt-file", help="sets local_receipt_sha256 from this file's bytes")
    a.add_argument("--run-ledger", action="append", default=[], metavar="LEDGER",
                   help="OWNED_STORAGE_DELETED: ledger of every run that used the storage")
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            result = pack([Path(d) for d in args.run_dirs], Path(args.out_dir), args.name,
                          args.unclosed_dead_owner)
            print(json.dumps({k: result[k] for k in ("archive", "archive_sha256", "runs",
                                                     "closure")}))
        elif args.command == "pack-eval":
            result = pack_eval(read_json(args.params), args.stage, Path(args.run_dir),
                               Path(args.out_dir), args.name)
            print(json.dumps({k: result[k] for k in ("archive", "archive_sha256", "runs", "closure",
                                                     "stage", "eval_runs")}))
        elif args.command == "verify":
            result = verify_archive(Path(args.archive), Path(args.manifest),
                                    Path(args.extract_to), Path(args.receipt))
            print(json.dumps(result, indent=2))
        elif args.action == "show":
            print(json.dumps(read_ledger(Path(args.ledger)), indent=2))
        else:
            evidence = dict(item.partition("=")[::2] for item in args.evidence)
            receipt_file = Path(args.receipt_file) if args.receipt_file else None
            print(json.dumps(append_state(Path(args.ledger), args.state, evidence,
                                          receipt_file=receipt_file,
                                          run_ledgers=[Path(p) for p in args.run_ledger]),
                             indent=2))
        return 0
    except (CollectError, OSError, ValueError, KeyError, tarfile.TarError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
