"""Private, dry-run-first continuation; never restarts the pilot or manages cloud resources."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import time
from typing import Any

DEADLINE = "2026-10-03T04:22:38Z"
POD = "h447a4u6c0vq3k"
BUNDLE = Path("/workspace/private_bundle")
EVIDENCE = BUNDLE / "analysis_exports/bottleneck_20261003"
DATA = EVIDENCE / "prepared"
ORACLES = EVIDENCE / "oracles/all_reviewed.jsonl"
QWEN = Path("/workspace/bottleneck-runtime")
GEMMA = Path("/workspace/gemma-verifier")
GEMMA_CODE = Path("/workspace/private_finish_v8")
PREFLIGHT = GEMMA_CODE / "experiments/gemma_verifier/preflight.py"
CPU = Path("/workspace/artifacts/gemma-cpu900-v8")
OUTPUT = Path("/workspace/artifacts/private-finish-v8")
SAMPLER_SOURCES = {
    "vllm/v1/sample/ops/topk_topp_sampler.py": "35b9a1ffeb1e50b73b0ddd1ffc0f968910ee21cf9cc5455dd879fde547b6ddc1",
    "vllm/envs.py": "1e76ec0ef393d38cd98310e8f6bcf2665741c6c48aa9ec1848c2e08427fc593c",
}
INVENTORY_SHA = "4aa9d623ae470ea97f156c9deb26710fb25bbb5cd9283124ba462970e1775e57"
PREFLIGHT_SHA = "b8d7f0b8b855598c985904b31e587f2878b3019bf339912398e6ba5325d50d81"
SETUP_EXCEPTION = "experiments/bottleneck_diagnostic/runpod_setup.sh"
SETUP_EXCEPTION_SHA = "6b82d22788ac2f334d10f16ccc73862486ed35b612cce03beadad6ab346ce29a"
MODES = ["I", "T0", "O0", "O1", "IO0", "P", "M", "MI"]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def cutoff() -> float:
    return datetime.fromisoformat(DEADLINE.replace("Z", "+00:00")).timestamp()


def pilot_gate(identity: dict[str, Any], history: list[dict[str, Any]],
               latest: list[dict[str, Any]], missing: list[Any],
               protocol: dict[str, Any], expected: int = 480) -> dict[str, Any]:
    """Inputs must come from the original CLI's validated_run, not saved score summaries."""
    statuses = Counter(row["runtime_status"] for row in latest)
    completed = [row for row in history if row["runtime_status"] == "COMPLETED"]
    seconds = [row["elapsed_seconds"] for row in completed]
    if any(type(n) not in (int, float) or not math.isfinite(n) or n <= 0 for n in seconds):
        raise ValueError("Pilot request times must be actual, positive and finite")
    complete = (not missing and len(latest) == expected and identity["protocol"] == protocol
                and identity["execution_kind"] == "ACTUAL_INFERENCE"
                and identity["repetitions"] == [3407]
                and set(statuses) <= {"COMPLETED", "BLOCKED", "EXCLUDED"}
                and not any(row["runtime_status"] in {"SYSTEM_ERROR", "BUDGET_STOP"}
                            for row in history))
    return {"status": "COMPLETE" if complete else "PARTIAL_OR_INVALID", "expected_cells": expected,
            "observed_cells": len(latest), "missing_cells": len(missing), "statuses": dict(statuses),
            "completed_calls": len(completed), "completed_request_seconds": sum(seconds),
            "mean_request_seconds": sum(seconds) / len(seconds) if seconds else None,
            "input_tokens": sum(row.get("input_tokens", 0) for row in completed),
            "output_tokens": sum(row.get("output_tokens", 0) for row in completed),
            "protocol_hash": identity["protocol_hash"]}


def coverage_gate(pilot: dict[str, Any], remaining: float, requested: bool,
                  *, gemma_finished: bool = False) -> dict[str, Any]:
    estimate = (1.5 * pilot["mean_request_seconds"] * 336 + 180
                if pilot.get("mean_request_seconds") is not None else None)
    reason = ("NOT_REQUESTED" if not requested else "PILOT_NOT_COMPLETE" if pilot["status"] != "COMPLETE"
              else "NO_COMPLETED_REQUEST_TIMING" if not pilot["completed_calls"]
              else "INSUFFICIENT_WINDOW_AFTER_RESERVES" if estimate is None or estimate > remaining - (0 if gemma_finished else 10800) - 1800
              else "MEASURED_TIME_GATE_PASS")
    return {"allowed": reason == "MEASURED_TIME_GATE_PASS", "reason": reason,
            "declared_cells": 336, "selected_questions": 42, "estimate_seconds": estimate,
            "remaining_seconds": remaining, "gemma_reserve_seconds": 0 if gemma_finished else 10800,
            "recovery_reserve_seconds": 1800, "startup_allowance_seconds": 180,
            "estimate_is_not_completion_or_speed_guarantee": True}


def after_gemma_gate(result: dict[str, Any], remaining: float, requested: bool) -> dict[str, Any] | None:
    coverage = result.get("coverage", {})
    audit = result.get("gemma_audit", {})
    validation = audit.get("controller_validation", {})
    if (audit.get("status") in {"RESOLVED", "REVIEW"}
            and result.get("gemma_audit_returncode") == 0
            and validation.get("all900_actual_terminal") is True
            and validation.get("ledger_rows") == validation.get("unique_ids") == 900
            and validation.get("missing_ids") == 0
            and not result.get("cleanup_error") and requested
            and coverage.get("reason") == "INSUFFICIENT_WINDOW_AFTER_RESERVES"
            and not coverage.get("attempted")):
        return coverage_gate(result["pilot"], remaining, True, gemma_finished=True)
    return None


def process_identity(pid: int, proc: Path = Path("/proc")) -> dict[str, Any] | None:
    try:
        text = (proc / str(pid) / "stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        command = (proc / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    return {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "session": int(fields[3]),
            "start_ticks": int(fields[19]), "command_sha256": hashlib.sha256(command).hexdigest()}


def verify_parent(current: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if current is None:
        return False
    if any(current[name] != expected[name] for name in ("pid", "start_ticks")):
        raise ValueError("Parent PID was reused or its command/start identity changed; never signal it")
    if current["state"] in {"Z", "X"}:
        return False  # A zombie's cmdline is empty; its birth still identifies the terminal parent.
    if current["command_sha256"] != expected["command_sha256"]:
        raise ValueError("Parent command identity changed; never signal it")
    return True


def verify_bundle(bundle: Path = BUNDLE) -> dict[str, str]:
    inventory = bundle / "TRANSFER_FILES.json"
    if sha(inventory) != INVENTORY_SHA:
        raise ValueError("Immutable v1 bundle inventory identity mismatch")
    hashes = {}
    for item in read(inventory):
        relative = Path(item["path"])
        path = bundle / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
            raise ValueError("Unsafe bundle inventory path")
        if not path.resolve(strict=True).is_relative_to(bundle.resolve(strict=True)):
            raise ValueError("Bundle path escaped through a symlink")
        observed = sha(path)
        authorized_setup = str(relative) == SETUP_EXCEPTION and observed == SETUP_EXCEPTION_SHA
        if not authorized_setup and (path.stat().st_size != item["bytes"] or observed != item["sha256"]):
            raise ValueError(f"Immutable bundle file changed: {relative}")
        if str(relative) in hashes:
            raise ValueError("Duplicate inventory path")
        hashes[str(relative)] = observed
    return hashes


def verify_cpu_identity(bundle: Path = BUNDLE, preflight: Path = PREFLIGHT,
                        cpu: Path = CPU, gemma_code: Path | None = None) -> dict[str, Any]:
    from experiments.gemma_verifier import runtime as g

    manifest, summary = read(cpu / "manifest.json"), read(cpu / "summary.json")
    extraction_mode = manifest.get("extraction_mode", "literal")
    protocol = g.protocol_id(extraction_mode)
    directory = (bundle if gemma_code is None else gemma_code) / "experiments/gemma_verifier"
    components = {p.name: sha(p) for p in [preflight, directory / "runtime.py",
                                          directory / "deployment.json", directory / "start_server.sh"]}
    if "extraction_mode" in manifest:
        components["source_segments.py"] = sha(directory / "source_segments.py")
    spec = read(directory / "deployment.json")
    source = bundle / "analysis_exports/bottleneck_20261003/gemma_source_v8.jsonl"
    if (manifest["code_sha256"] != digest(components)
            or summary.get("extraction_mode", "literal") != extraction_mode
            or manifest.get("protocol", protocol) != protocol
            or summary.get("protocol", protocol) != protocol
            or (extraction_mode == "segments" and (
                manifest.get("protocol") != protocol or summary.get("protocol") != protocol))
            or manifest["source_sha256"] != sha(source)
            or manifest["model"] != spec["model_id"] or manifest["revision"] != spec["model_revision"]
            or manifest["kind"] != "ACTUAL_CPU_TOKENIZER" or manifest["source_count"] != 900
            or manifest["max_model_len"] != 131072 or manifest["output_allowance"] != 1024
            or manifest["input_truncation"] is not False
            or summary["source_sha256"] != manifest["source_sha256"]
            or summary["manifest_sha256"] != sha(cpu / "manifest.json")
            or summary["token_budget_sha256"] != sha(cpu / "token_budget.jsonl")
            or summary["status"] != "CPU_TOKENIZER_PASS" or summary["counted"] != 900
            or summary["max_input_plus_output"] > 131072
            or summary["context_overflows"] != 0 or summary["truncation"] is not False
            or summary["full_audit_allowed"] is not False or summary["judge_calls"] != 0
            or manifest["tokenizer_files_sha256"] != {
                name: spec["expected_model_files"][name]["sha256"]
                for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
            or manifest["runtime_chat_template_sha256"] != spec["expected_model_files"]["chat_template.jinja"]["sha256"]):
        raise ValueError("Immutable Gemma CPU/source/model/template/code identity mismatch")
    return {"manifest_sha256": sha(cpu / "manifest.json"), "summary_sha256": sha(cpu / "summary.json"),
            "code_sha256": manifest["code_sha256"], "source_sha256": manifest["source_sha256"]}


def audit_result(output: Path, source: Path) -> dict[str, Any]:
    """Read append-only actual audit artifacts; REVIEW is not a final assisted score."""
    from experiments.gemma_verifier import runtime as g

    manifest = read(output / "run_manifest.json")
    extraction_mode = manifest.get("extraction_mode", "literal")
    protocol = g.protocol_id(extraction_mode)
    if (manifest["source_sha256"] != sha(source) or manifest["source_count"] != 900
            or manifest.get("protocol", protocol) != protocol
            or (extraction_mode == "segments" and manifest.get("protocol") != protocol)
            or manifest["execute"] is not True or manifest["semantic_open"] is not True
            or manifest["model"] != "google/gemma-4-E4B-it"
            or manifest["revision"] != "ee0ef6023621cff504d758262d4e04895a5af4a2"
            or manifest["endpoint"] != "http://127.0.0.1:8000/v1"
            or manifest["model_manifest_sha256"] != digest(manifest["model_manifest"])):
        raise ValueError("Actual audit source/model/execute manifest identity failed")
    path = output / "summaries.jsonl"
    summaries = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    summary = summaries[-1] if summaries else {"status": "INCOMPLETE_NO_SUMMARY", "assisted_accuracy": None}
    ledger = output / "ledger.jsonl"
    entries = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    source_rows = [json.loads(line) for line in source.read_text().splitlines()]
    expected = {row.get("id", row.get("sample_id")) for row in source_rows}
    if len(source_rows) != 900 or len(expected) != 900:
        raise ValueError("Gemma source is not frozen900")
    seen = set()
    for entry in entries:
        if (entry["id"] not in expected or entry["id"] in seen
                or entry["entry_sha256"] != digest({k: v for k, v in entry.items() if k != "entry_sha256"})):
            raise ValueError("Actual audit record hash/ID integrity failed")
        seen.add(entry["id"])
        if entry["status"] in {"RESOLVED", "REVIEW"} and (
                not entry.get("requests", {}).get("extraction")
                or not entry.get("http_responses", {}).get("extraction")):
            raise ValueError("Terminal audit record lacks actual extraction request/response")
        if entry["status"] in {"RESOLVED", "REVIEW"}:
            if any(entry[name] != manifest[name] for name in
                   ("model", "revision", "model_manifest_sha256", "protocol_sha256")):
                raise ValueError("Terminal audit model/protocol identity differs")
    statuses = Counter(entry["status"] for entry in entries)
    subjects = Counter(entry["subject"] for entry in entries)
    resolved = (seen == expected and statuses["RESOLVED"] == 900
                and len(subjects) == 30 and set(subjects.values()) == {30})
    terminal = seen == expected and set(statuses) <= {"RESOLVED", "REVIEW"}
    if summaries:
        if (summary["statuses"] != dict(statuses) or summary["source_count"] != len(entries)
                or summary.get("extraction_mode", extraction_mode) != extraction_mode
                or summary.get("protocol", protocol) != protocol
                or (extraction_mode == "segments" and summary.get("protocol") != protocol)
                or summary["denominator"] != 900 or summary["execute"] is not True
                or summary["dry_run"] is not False or summary["replaces_v8"] is not False
                or (summary["status"] == "RESOLVED") != resolved
                or (not resolved and summary["assisted_accuracy"] is not None)):
            raise ValueError("Actual audit summary disagrees with ledger or exposes an unresolved score")
        if resolved and summary["assisted_accuracy"] != sum(e["candidate_correct"] is True for e in entries) / 900:
            raise ValueError("Resolved assisted score arithmetic differs")
    seconds = sum(entry.get("elapsed_seconds", 0) for entry in entries)
    usage = [response.get("usage", {}) for entry in entries
             for stage, response in entry.get("http_responses", {}).items()
             if stage in {"extraction", "mapping", "equivalence"}]
    return {**summary, "controller_validation": {"ledger_rows": len(entries), "unique_ids": len(seen),
            "missing_ids": len(expected - seen), "statuses": dict(statuses), "all900_actual_terminal": terminal,
            "all900_resolved": resolved, "final_score_allowed": resolved and bool(summaries),
            "elapsed_request_seconds": seconds, "actual_completion_responses": len(usage),
            "input_tokens": sum(row.get("prompt_tokens", 0) for row in usage),
            "output_tokens": sum(row.get("completion_tokens", 0) for row in usage),
            "ledger_sha256": sha(ledger) if ledger.exists() else None,
            "run_manifest_sha256": sha(output / "run_manifest.json"),
            "summaries_sha256": sha(path) if path.exists() else None}}


class OwnedProcess:
    """Track exact process births; cleanup never signals an unrelated PID or broad process name."""

    def __init__(self, command: list[str], log: Path, env: dict[str, str], end: float,
                 cwd: Path = BUNDLE):
        require_window(end, "process construction")
        if not callable(getattr(os, "pidfd_open", None)) or not callable(getattr(signal, "pidfd_send_signal", None)):
            raise OSError("pidfd cleanup is required before launching any child")
        probe = os.pidfd_open(os.getpid(), 0)  # Check kernel support without sending any signal.
        os.close(probe)
        self.log = log.open("xb")
        try:
            require_window(end, "Popen")
            self.child = subprocess.Popen(command, cwd=cwd, env=env, stdout=self.log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.log.close()
            raise
        self.known: dict[int, int] = {}
        self.track()

    def track(self) -> None:
        observed = {}
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                try:
                    item = process_identity(int(path.name))
                except (OSError, ValueError):
                    continue
                if item is not None:
                    observed[item["pid"]] = item
        anchors = {pid for pid, birth in self.known.items()
                   if pid in observed and observed[pid]["start_ticks"] == birth}
        if not self.known and self.child.pid in observed:
            anchors.add(self.child.pid)
        while True:
            descendants = {pid for pid, item in observed.items() if item["ppid"] in anchors}
            group = {pid for pid, item in observed.items()
                     if anchors and item["session"] == self.child.pid}
            expanded = anchors | descendants | group
            if expanded == anchors:
                break
            anchors = expanded
        self.known.update({pid: observed[pid]["start_ticks"] for pid in anchors})

    def signal_owned(self, sig: int) -> None:
        if not callable(getattr(os, "pidfd_open", None)) or not callable(getattr(signal, "pidfd_send_signal", None)):
            raise OSError("pidfd cleanup is required; refusing unsafe numeric-PID signals")
        for pid, birth in self.known.items():
            try:
                handle = os.pidfd_open(pid, 0)
            except ProcessLookupError:
                continue
            try:
                # The handle is bound first: later PID reuse cannot redirect this signal.
                current = process_identity(pid)
                if current is not None and current["start_ticks"] == birth and current["state"] not in {"Z", "X"}:
                    try:
                        signal.pidfd_send_signal(handle, sig, None, 0)
                    except ProcessLookupError:
                        pass
            finally:
                os.close(handle)

    def close(self) -> None:
        try:
            self.close_owned()
        finally:
            self.log.close()

    def close_owned(self) -> None:
        self.track()
        self.signal_owned(signal.SIGTERM)
        end = time.monotonic() + 10
        while time.monotonic() < end:
            self.track()
            alive = [process_identity(pid) for pid in self.known]
            if not any(p and p["start_ticks"] == self.known[p["pid"]] and p["state"] not in {"Z", "X"}
                       for p in alive):
                break
            time.sleep(.2)
        self.signal_owned(signal.SIGKILL)
        self.child.wait(timeout=10)
        end = time.monotonic() + 5
        while True:
            alive = [process_identity(pid) for pid in self.known]
            if not any(p and p["start_ticks"] == self.known[p["pid"]] and p["state"] not in {"Z", "X"}
                       for p in alive):
                break
            if time.monotonic() >= end:
                raise TimeoutError("Owned descendant remained alive after exact-PID cleanup")
            time.sleep(.2)


def gpu_idle() -> None:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=10)
    if result.stdout.strip():
        raise ValueError("GPU compute processes remain; no concurrent model load and no automatic kill")


def gemma_environment(env: dict[str, str], root: Path = GEMMA) -> dict[str, str]:
    """Explicit native sampler, not an automatic precision/context/backend fallback."""
    for relative, expected in SAMPLER_SOURCES.items():
        if sha(root / "venv/lib/python3.12/site-packages" / relative) != expected:
            raise ValueError("Observed vLLM sampler/environment sources changed")
    return dict(env, PATH=f"{root}/venv/bin:{os.environ.get('PATH', '')}",
                HF_HOME=str(root / "hf-cache"), VLLM_USE_FLASHINFER_SAMPLER="0",
                PYTHONPATH=f"{GEMMA_CODE}:{env.get('PYTHONPATH', '')}")


def event(output: Path, phase: str, **values: Any) -> None:
    row = {"utc": datetime.now(timezone.utc).isoformat(), "phase": phase, **values}
    with (output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_window(end: float, phase: str) -> None:
    if not math.isfinite(end) or time.time() >= end:
        raise TimeoutError(phase + " absolute time window expired before execution")


def run(command: list[str], output: Path, name: str, env: dict[str, str], end: float,
        cwd: Path = BUNDLE) -> int:
    require_window(end, name)
    event(output, name, status="STARTED", command=command, command_sha256=digest(command))
    require_window(end, name)
    owned = OwnedProcess(command, output / (name + ".log"), env, end, cwd)
    started = time.monotonic()
    try:
        while True:
            require_window(end, name)
            rc = owned.child.poll()
            if rc is not None:
                require_window(end, name)
                break
            owned.track()
            time.sleep(max(0, min(1, end - time.time())))
        event(output, name, status="EXITED", returncode=rc, elapsed_seconds=time.monotonic() - started)
        return rc
    except (OSError, ValueError, TimeoutError) as error:
        event(output, name, status="FAILED_OR_TIMED_OUT", reason=str(error),
              elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        owned.close()
        event(output, name, status="OWNED_PROCESSES_CLEANED", returncode=owned.child.returncode,
              tracked_births=owned.known, elapsed_seconds=time.monotonic() - started)


PILOT_INSPECT = """
import json,sys,yaml
from pathlib import Path
from experiments.bottleneck_diagnostic.cli import selected,validated_run
data,out,protocol=map(Path,sys.argv[1:4])
suite=sys.argv[4]
samples=selected(data,suite)
if len(samples)!={'pilot':60,'coverage':42}[suite]: raise ValueError('Suite selection count differs')
identity,history,latest,missing=validated_run(out,samples)
expected=json.loads(protocol.read_text())
expected['evaluation_config']=yaml.safe_load(Path('configs/eval/mmmu_val_v8.yaml').read_text())
print(json.dumps({'identity':identity,'history':[{k:r[k] for k in ('runtime_status','elapsed_seconds','input_tokens','output_tokens') if k in r} for r in history], 'latest':[{'runtime_status':r['runtime_status']} for r in latest], 'missing':missing,'protocol':expected}))
"""


def run_coverage(gate: dict[str, Any], env: dict[str, str]) -> None:
    """One fresh attempt only; failed or budget-stopped coverage remains preserved."""
    gate["attempted"] = True
    try:
        gpu_idle()
        verify_bundle()
        model = read(QWEN / "setup/model.json")
        baseline = read(BUNDLE / "manifests/models/baseline.json")
        if (model["pod_id"] != POD or model["lock_sha256"] != sha(BUNDLE / "env/requirements-vllm.lock")
                or any(model[name] != baseline[name] for name in ("repo_id", "revision", "files"))):
            raise ValueError("Qwen setup/model/lock identity changed")
        model_path = Path(model["model_path"]).resolve(strict=True)
        if not model_path.is_relative_to((QWEN / "hf-cache").resolve(strict=True)):
            raise ValueError("Qwen model escaped isolated snapshot cache")
        python = str(QWEN / "venv/bin/python")
        selection = [python, "-c", "from experiments.bottleneck_diagnostic.cli import selected; from pathlib import Path; import sys,importlib.metadata as m; assert len(selected(Path(sys.argv[1]),'coverage'))==42; assert all(m.version(k)==v for k,v in {'torch':'2.8.0+cu128','transformers':'4.57.1','vllm':'0.11.0'}.items())", str(DATA)]
        if run(selection, OUTPUT, "coverage_selection", env, min(time.time() + 60, cutoff() - 1800)) != 0:
            raise ValueError("Fixed coverage selection/environment failed")
        budget = math.ceil(gate["estimate_seconds"])
        target = OUTPUT / "coverage"
        if target.exists():
            raise ValueError("Existing coverage artifacts refused; no implicit resume/repeat")
        command = [python, "-m", "experiments.bottleneck_diagnostic.cli", "run", "--data", str(DATA),
                   "--suite", "coverage", "--output", str(target), "--protocol",
                   str(BUNDLE / "experiments/bottleneck_diagnostic/protocol.json"),
                   "--model-path", str(model_path), "--oracles", str(ORACLES), "--budget-seconds", str(budget)]
        gate["returncode"] = run(command, OUTPUT, "coverage_run", env,
                                  min(time.time() + budget, cutoff() - gate["gemma_reserve_seconds"] - 1800))
        gate["status"] = "RUN_EXITED_NOT_INDEPENDENTLY_VERIFIED"
        if (target / "responses.jsonl").exists():
            gate["responses_sha256"] = sha(target / "responses.jsonl")
            validate = [python, "-c", PILOT_INSPECT, str(DATA), str(target),
                        str(BUNDLE / "experiments/bottleneck_diagnostic/protocol.json"), "coverage"]
            if run(validate, OUTPUT, "coverage_validate", env,
                   cutoff() - gate["gemma_reserve_seconds"] - 1800) == 0:
                gate["validated_cells"] = pilot_gate(**read(OUTPUT / "coverage_validate.log"), expected=336)
            score = [python, "-m", "experiments.bottleneck_diagnostic.cli", "score", "--data", str(DATA),
                     "--suite", "coverage", "--output", str(target), "--oracles", str(ORACLES)]
            gate["score_returncode"] = run(score, OUTPUT, "coverage_score", env,
                                           cutoff() - gate["gemma_reserve_seconds"] - 1800)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.SubprocessError) as error:
        gate["status"] = "BLOCKED_OR_FAILED_PRESERVED"
        gate["error"] = str(error)
        event(OUTPUT, "coverage_failure", reason=str(error))


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.pod_id != POD or os.environ.get("RUNPOD_POD_ID") != POD:
        raise ValueError("Explicit authorized Pod ID must match RUNPOD_POD_ID")
    if not os.path.ismount("/workspace"):
        raise ValueError("Persistent /workspace mount is required")
    if args.parent_pid != 888 or not args.parent_start_ticks or not args.parent_command_sha256:
        raise ValueError("Execution requires trusted PID888 start ticks and raw cmdline SHA256")
    if time.time() >= cutoff() - 1800:
        raise TimeoutError("Recovery reserve has started; no new model load")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {"status": "RUNNING", "pod_id": POD, "deadline_utc": DEADLINE,
                             "full30_tested": False, "provenance": "BENCHMARK_DERIVED",
                             "training_allowed": False, "public_upload_allowed": False}
    env = dict(os.environ, PYTHONPATH=f"{BUNDLE}/src:{BUNDLE}", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    server = None
    try:
        hashes = verify_bundle()
        if sha(PREFLIGHT) != PREFLIGHT_SHA:
            raise ValueError("Gemma preflight is not the immutable deployed source")
        result["source_hashes"] = hashes
        result["authorized_setup_exception"] = {"path": SETUP_EXCEPTION, "sha256": SETUP_EXCEPTION_SHA,
                                               "transfer_archive_sha256": "e6cfa4bc93735145f08b6f5eb10c1a8cbc53ccc39264c23945ac7e5544fe887d"}
        result["controller_sha256"] = sha(Path(__file__))
        result["cpu_identity"] = verify_cpu_identity(gemma_code=GEMMA_CODE)
        protocol = BUNDLE / "experiments/bottleneck_diagnostic/protocol.json"
        frozen = read(protocol)
        if frozen["overall_deadline_utc"] != DEADLINE or frozen["modes"] != MODES:
            raise ValueError("Frozen diagnostic protocol changed")
        expected = {"pid": 888, "start_ticks": args.parent_start_ticks,
                    "command_sha256": args.parent_command_sha256}
        event(OUTPUT, "pilot_wait", status="WAITING", parent_identity=expected)
        while verify_parent(process_identity(888), expected):
            if time.time() >= cutoff() - 1800:
                raise TimeoutError("Pilot remains active at recovery cutoff; never killed or restarted")
            time.sleep(min(args.poll_seconds, max(.1, cutoff() - 1800 - time.time())))
        event(OUTPUT, "pilot_wait", status="TERMINAL_OR_ABSENT", parent_identity=expected)
        python = str(QWEN / "venv/bin/python")
        pilot = QWEN / "results/pilot"
        inspect = [python, "-c", PILOT_INSPECT, str(DATA), str(pilot), str(protocol), "pilot"]
        if run(inspect, OUTPUT, "pilot_validate", env, min(cutoff() - 1800, time.time() + 180)) == 0:
            values = read(OUTPUT / "pilot_validate.log")
            try:
                result["pilot"] = pilot_gate(**values)
            except (ValueError, KeyError, TypeError) as error:
                result["pilot"] = {"status": "PARTIAL_OR_INVALID", "completed_calls": 0,
                                   "mean_request_seconds": None, "reason": str(error)}
            result["pilot"]["responses_sha256"] = sha(pilot / "responses.jsonl")
            result["pilot"]["identity_sha256"] = sha(pilot / "protocol.json")
        else:
            result["pilot"] = {"status": "PARTIAL_OR_INVALID", "completed_calls": 0,
                               "mean_request_seconds": None, "reason": "VALIDATED_RUN_FAILED"}
        if (pilot / "responses.jsonl").exists():
            score = [python, "-m", "experiments.bottleneck_diagnostic.cli", "score", "--data", str(DATA),
                     "--suite", "pilot", "--output", str(pilot), "--score-output", str(OUTPUT / "pilot-score"),
                     "--oracles", str(ORACLES)]
            result["pilot_score_returncode"] = run(score, OUTPUT, "pilot_score", env,
                                                   min(cutoff() - 1800, time.time() + 180))
        coverage = coverage_gate(result["pilot"], cutoff() - time.time(), args.coverage)
        result["coverage"] = coverage
        event(OUTPUT, "coverage_gate", **coverage)
        if coverage["allowed"]:
            run_coverage(coverage, env)
        # Pilot partiality does not block independent Gemma; an occupied GPU or changed identity does.
        gpu_idle()
        verify_bundle()
        verify_cpu_identity(gemma_code=GEMMA_CODE)
        end = cutoff() - 1800
        if time.time() >= end:
            raise TimeoutError("No Gemma window before fixed recovery reserve")
        gemma_env = gemma_environment(env)
        result["server_runtime_policy"] = {
            "VLLM_USE_FLASHINFER_SAMPLER": "0", "sampler": "vllm_pytorch_native",
            "installed_source_sha256": SAMPLER_SOURCES,
            "reason": "FlashInfer sampler warmup required absent nvcc; explicit supported opt-out",
            "model_precision_context_and_sampling_changed": False,
            "judge_prompt_protocol": "gemma-frozen-final-audit-v6-exhaustive-choice-assessment",
            "exhaustive_choice_assessment": True,
            "submission_only_extraction": True,
            "native_cross_field_schema_branches": True,
            "chat_template_content_format": "string",
            "content_format_reason": "CPU HF string and vLLM auto OpenAI-list differed by one system-boundary space; explicitly match string",
            "failed_original_output_preserved": "/workspace/artifacts/private-finish-v1",
            "failed_smoke_output_preserved": "/workspace/artifacts/private-finish-v7",
        }
        state = OUTPUT / "gemma-server"
        timeout_binary = shutil.which("timeout")
        if timeout_binary is None or end - time.time() <= 10:
            raise ValueError("Native hard-timeout guard or Gemma termination margin is missing")
        command = [timeout_binary, "--signal=TERM", "--kill-after=10s",
                   f"{end - time.time() - 10:.3f}s", "bash", str(GEMMA_CODE / "experiments/gemma_verifier/start_server.sh"),
                   "--model-path", str(GEMMA / "model"), "--output", str(state), "--execute"]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 8000))  # Refuse another listener; never adopt/kill an existing server.
        event(OUTPUT, "gemma_server", status="STARTED", command=command, command_sha256=digest(command),
              absolute_cutoff_utc="2026-10-03T03:52:38Z")
        server = OwnedProcess(command, OUTPUT / "gemma-server.log", gemma_env, end)
        ready = False
        readiness_end = min(end, time.time() + 600)
        while time.time() < readiness_end and server.child.poll() is None:
            server.track()
            try:
                connection = http.client.HTTPConnection("127.0.0.1", 8000,
                                                        timeout=max(.1, min(5, readiness_end-time.time())))
                try:
                    connection.request("GET", "/v1/models")
                    response = connection.getresponse()
                    models = json.loads(response.read(1024 * 1024)) if response.status == 200 else {}
                    ready = any(item["id"] == "google/gemma-4-E4B-it" for item in models.get("data", []))
                finally:
                    connection.close()
            except (OSError, ValueError, KeyError):
                pass
            if ready:
                break
            time.sleep(1)
        if not ready:
            raise TimeoutError("Gemma loopback readiness failed; preserve logs, no fallback")
        if "FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0." not in (OUTPUT / "gemma-server.log").read_text():
            raise ValueError("Server did not attest the explicit native sampler selection")
        identity = read(state / "model_identity.json")
        identity["server_runtime_policy"] = result["server_runtime_policy"]
        identity["controller_sha256"] = sha(Path(__file__))
        identity_path = OUTPUT / "gemma-server-policy/model_identity.json"
        identity_path.parent.mkdir(parents=True, exist_ok=False)
        # Preserve launcher identity; bind the observed engine policy in a separate immutable artifact.
        with identity_path.open("x") as stream:
            json.dump(identity, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        event(OUTPUT, "gemma_server", status="READY_POLICY_BOUND",
              model_identity_sha256=sha(identity_path), **result["server_runtime_policy"])
        source = EVIDENCE / "gemma_source_v8.jsonl"
        common = [str(GEMMA / "venv/bin/python"), str(PREFLIGHT), "smoke", "--source", str(source),
                  "--output", str(OUTPUT / "gemma-smoke"), "--cpu-preflight", str(CPU),
                  "--model-manifest", str(identity_path), "--timeout", "600",
                  "--deadline-utc", "2026-10-03T03:52:38Z", "--execute"]
        rc = run(common, OUTPUT, "gemma_smoke", gemma_env, end)
        smoke = read(OUTPUT / "gemma-smoke/summary.json") if (OUTPUT / "gemma-smoke/summary.json").exists() else {}
        result["gemma_smoke"] = smoke
        if rc != 0 or smoke.get("status") != "GPU_SMOKE_PASS" or smoke.get("full_audit_allowed") is not True:
            raise ValueError("Actual Gemma smoke gate failed; full900 prohibited")
        ledger = (OUTPUT / "gemma-smoke/smoke.jsonl").read_text().splitlines()
        if len(ledger) != 29 or smoke.get("cases") != 29:
            raise ValueError("Actual smoke did not preserve all29 declared cases")
        audit = list(common)
        audit[2] = "audit"
        audit[audit.index("--output") + 1] = str(OUTPUT / "gemma-audit")
        audit.extend(["--smoke-gate", str(OUTPUT / "gemma-smoke"), "--semantic-open"])
        result["gemma_audit_returncode"] = run(audit, OUTPUT, "gemma_audit", gemma_env, end)
        result["gemma_audit"] = audit_result(OUTPUT / "gemma-audit", source)
        result["status"] = "CONTINUATION_EXITED_LOCAL_RECOVERY_PENDING"
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.SubprocessError) as error:
        result["status"] = "BLOCKED_OR_FAILED_PRESERVED"
        result["error"] = str(error)
        event(OUTPUT, "failure", status=result["status"], reason=str(error))
    finally:
        if server is not None:
            try:
                server.close()
                event(OUTPUT, "gemma_server", status="OWNED_DESCENDANTS_CLEANED", tracked_births=server.known)
            except (OSError, TimeoutError, subprocess.SubprocessError) as error:
                result["status"] = "BLOCKED_OR_FAILED_PRESERVED"
                result["cleanup_error"] = str(error)
        if (OUTPUT / "gemma-audit").exists() and "gemma_audit" not in result:
            try:
                result["gemma_audit"] = audit_result(OUTPUT / "gemma-audit", EVIDENCE / "gemma_source_v8.jsonl")
            except (OSError, ValueError, KeyError, TypeError) as error:
                result["gemma_audit"] = {"status": "INCOMPLETE_VALIDATION_FAILED", "error": str(error),
                                         "final_score_allowed": False}
        fallback = after_gemma_gate(result, cutoff() - time.time(), args.coverage)
        if fallback is not None:
            result["coverage_after_gemma"] = fallback
            event(OUTPUT, "coverage_after_gemma_gate", **fallback)
            if fallback["allowed"]:
                run_coverage(fallback, env)
        with (OUTPUT / "summary.json").open("x") as stream:
            json.dump(result, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--parent-pid", type=int, default=888)
    parser.add_argument("--parent-start-ticks", type=int)
    parser.add_argument("--parent-command-sha256")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.pod_id != POD or not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 60:
        parser.error("Authorized Pod ID and polling interval in(0,60] required")
    if not args.execute:
        print(json.dumps({"status": "DRY_PLAN_NO_GPU_OR_PROCESS_OPERATIONS", "pod_id": POD,
                          "parent_pid": args.parent_pid, "deadline_utc": DEADLINE, "coverage_requested": args.coverage,
                          "output": str(OUTPUT), "VLLM_USE_FLASHINFER_SAMPLER": "0",
                          "plan": ["validate frozen inference/new preflight+CPU900", "wait same PID888 identity without kill/restart",
                                   "pilot480 gate", "optional measured coverage336 gate", "GPU empty gate",
                                   "Gemma pinned BF16 server", "actual smoke29 gate", "gated audit900",
                                   "owned descendants cleanup", "root local recovery pending"]}, sort_keys=True))
        return 0
    if (args.parent_start_ticks is None or args.parent_start_ticks < 1
            or args.parent_command_sha256 is None or len(args.parent_command_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.parent_command_sha256)):
        parser.error("Trusted positive parent start ticks and raw cmdline SHA256 required")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Controller interrupted by signal{signum}; cleanup only owned processes")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        result = execute(args)
    except (OSError, ValueError, TimeoutError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "CONTINUATION_EXITED_LOCAL_RECOVERY_PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
