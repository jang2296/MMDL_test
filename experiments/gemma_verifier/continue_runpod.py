"""Same-Pod, immutable-gate continuation; dry by default, no cloud lifecycle operations."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any

from experiments.bottleneck_diagnostic import finish_runpod as f
from experiments.bottleneck_diagnostic.export_results import gemma_export
from . import preflight as p, runtime as r

POD = "re5qljwwyq7yui"
DEADLINE = "2026-10-03T06:07:38Z"
PARENT = Path("/workspace/artifacts/gemma-segments-v1")
ROOT = Path("/workspace/gemma-verifier")
SOURCE = Path("/workspace/private_bundle/analysis_exports/bottleneck_20261003/gemma_source_v8.jsonl")
SOURCE_SHA = "c1f552ee245c66b627baa7f948122f40b063ef1beaef08a458c6088c620fa917"
ATTESTATION_FIELDS = {"controller_sha256", "server_runtime_policy"}
EXPORT_SOURCE = Path(gemma_export.__code__.co_filename)


def records(path: Path) -> list[dict[str, Any]]:
    items = [r.json_object(line) for line in path.read_text().splitlines()]
    if any(item.get("entry_sha256") != r.digest({k: v for k, v in item.items() if k != "entry_sha256"})
           for item in items) or len({item["id"] for item in items}) != len(items):
        raise ValueError("Record hashes or unique IDs differ")
    return items


def closed(parent: Path, pod: str) -> dict[str, Any]:
    summary = f.read(parent / "summary.json")
    if (summary.get("pod_id") != pod or summary.get("extraction_mode") != "segments"
            or summary.get("protocol") != r.protocol_id("segments")
            or summary.get("status") not in {"EXITED_LOCAL_RECOVERY_PENDING", "BLOCKED_OR_FAILED_PRESERVED"}
            or "owned_server_births_cleaned" not in summary or "cleanup_error" in summary):
        raise ValueError("Parent is not closed with successful owned cleanup")
    return summary


def static_identity(prior: dict[str, Any], fresh: dict[str, Any]) -> None:
    expected = {k: v for k, v in prior.items() if k not in ATTESTATION_FIELDS}
    if fresh != expected:
        raise ValueError("Fresh server static model/environment identity differs")


def inheritance(parent: Path, generated: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old, new = f.read(parent / "audit/run_manifest.json"), f.read(generated / "run_manifest.json")
    if old.get("execute") is not True or any(old.get(k) != v for k, v in new.items()
                                            if k not in {"deadline_unix", "execute"}):
        raise ValueError("Source/model/protocol/cache identity mismatch")
    expected = {item["id"]: item for item in records(generated / "prepared.jsonl")}
    prepared, entries = records(parent / "audit/prepared.jsonl"), records(parent / "audit/ledger.jsonl")
    for entry in prepared:
        if entry != expected.get(entry["id"]):
            raise ValueError("Prepared source payload/input/key mismatch")
    for entry in entries:
        request = expected.get(entry["id"])
        if request is None or entry.get("resume_key") != request["resume_key"]:
            raise ValueError("Resume input/protocol/model key mismatch")
        if entry["status"] not in {"RESOLVED", "REVIEW", "SYSTEM_ERROR", "BUDGET_STOP", "PENDING"}:
            raise ValueError("Unknown parent status")
        if entry["status"] in {"RESOLVED", "REVIEW", "SYSTEM_ERROR"} and "extraction" in entry.get("http_responses", {}):
            if (entry.get("attempt", 0) < 1 or entry.get("requests", {}).get("extraction") != request["payload"]
                    or any(entry.get(k) != old[k] for k in
                           ("model", "revision", "model_manifest_sha256", "protocol_sha256"))):
                raise ValueError("Inherited record lacks actual attempt or matching identity")
    return prepared, [e for e in entries if e["status"] in {"RESOLVED", "REVIEW", "SYSTEM_ERROR"}
                      and "extraction" in e.get("http_responses", {})]


def stage_cost(entries: list[dict[str, Any]], inherited: set[str]) -> dict[str, Any]:
    fresh = [e for e in entries if e["id"] not in inherited]
    return {stage: {"recorded_requests": sum(stage in e.get("requests", {}) for e in fresh),
                   "response_calls": len(responses),
                   "input_tokens": sum(x.get("prompt_tokens", 0) for x in usages),
                   "output_tokens": sum(x.get("completion_tokens", 0) for x in usages)}
            for stage in ("extraction", "mapping", "equivalence")
            for responses in [[e["http_responses"][stage] for e in fresh if stage in e.get("http_responses", {})]]
            for usages in [[x["usage"] for x in responses if isinstance(x, dict) and isinstance(x.get("usage"), dict)]]}


def execute(pod: str, output: Path, *, run: bool = False) -> dict[str, Any]:
    end, code = p._deadline(DEADLINE), Path(p.__file__).resolve().parents[2]
    f.require_window(end, "continuation preparation")
    output = output.resolve()
    if (pod != POD or not output.is_relative_to(Path("/workspace/artifacts"))
            or output == Path("/workspace/artifacts") or output.exists()
            or output.is_relative_to(PARENT) or PARENT.is_relative_to(output)):
        raise ValueError("Exact owned Pod and fresh independent private output required")
    closed(PARENT, pod)
    if f.sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Frozen source changed")
    identity_path = PARENT / "model_identity.json"
    prior = f.read(identity_path)
    gemma_export(PARENT / "audit", SOURCE)
    paths = [SOURCE, identity_path, PARENT / "summary.json", *(PARENT / "audit" / name for name in
             ("run_manifest.json", "ledger.jsonl", "prepared.jsonl"))]
    paths += [PARENT / stage / name for stage, names in
              (("cpu", ("manifest.json", "summary.json", "token_budget.jsonl")),
               ("smoke", ("manifest.json", "summary.json", "smoke.jsonl"))) for name in names]
    frozen = {str(path): f.sha(path) for path in paths}
    with tempfile.TemporaryDirectory() as temporary:
        prepared_dir = Path(temporary) / "prepared"
        p.gated_audit(SOURCE, PARENT / "cpu", PARENT / "smoke", prepared_dir, identity_path,
                      prior["endpoint"], semantic_open=True, deadline_utc=DEADLINE, extraction_mode="segments")
        prepared, reused = inheritance(PARENT, prepared_dir)
    inherited = {item["id"] for item in reused}
    result: dict[str, Any] = {"status": "DRY_PLAN_NO_GPU", "pod_id": pod, "deadline_utc": DEADLINE,
        "extraction_mode": "segments", "protocol": r.protocol_id("segments"), "source_hashes": frozen,
        "controller_sha256": f.sha(Path(__file__)), "code_sha256": p._code_hash(),
        "operation_source_hashes": {str(path): f.sha(path) for path in
            (Path(__file__), Path(f.__file__), EXPORT_SOURCE)},
        "identity_provenance": "PRIOR_ATTESTATION_REUSED; old controller did not launch this server",
        "failed_attempt_provenance": "No-response failures remain in the unchanged parent ledger, not the reusable output cache",
        "accounting_scope": "New records only; recorded requests may fail before HTTP completion; no dollar cost inferred",
        "inherited_ids": sorted(inherited), "inherited_count": len(inherited), "remaining": 900 - len(inherited),
        "replaces_v8": False, "training_allowed": False, "public_upload_allowed": False}
    if any(f.sha(Path(path)) != sha for path, sha in frozen.items()):
        raise ValueError("Parent/source changed during validation")
    if not run:
        return result
    if os.environ.get("RUNPOD_POD_ID") != pod or not os.path.ismount("/workspace"):
        raise ValueError("Actual native Pod identity and persistent mount must match")
    output.mkdir(parents=True, exist_ok=False)
    server = None
    try:
        if len(inherited) == 900:
            result["status"] = "NOTHING_REMAINING_NO_GPU"
            return result
        f.gpu_idle()
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", 8000))
        env = f.gemma_environment(dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                                      TRANSFORMERS_OFFLINE="1"), ROOT)
        env["PYTHONPATH"] = f"{code / 'src'}:{code}"
        timeout = shutil.which("timeout")
        if not timeout or end - time.time() <= 10:
            raise ValueError("Hard timeout/cleanup margin missing")
        command = [timeout, "--signal=TERM", "--kill-after=10s", f"{end-time.time()-10:.3f}s", "bash",
                   str(code / "experiments/gemma_verifier/start_server.sh"), "--model-path", str(ROOT / "model"),
                   "--output", str(output / "server"), "--execute"]
        f.event(output, "server", command=command, command_sha256=r.digest(command))
        server = f.OwnedProcess(command, output / "server.log", env, end, code)
        ready_end = min(end - 10, time.time() + 600)
        while True:
            f.require_window(ready_end, "server readiness")
            server.track()
            if server.child.poll() is not None:
                raise ValueError("Owned server exited before readiness")
            try:
                version = r.http_json(prior["endpoint"], "/version", timeout=min(3, ready_end-time.time()))
                models = r.http_json(prior["endpoint"], "/v1/models", timeout=min(3, ready_end-time.time()))
                if version.get("version") != prior["vllm_version"] or not any(
                        item.get("id") == prior["model_id"] for item in models.get("data", [])):
                    raise ValueError("Fresh endpoint model/version differs")
                break
            except OSError:
                time.sleep(min(1, max(0, ready_end-time.time())))
        static_identity(prior, f.read(output / "server/model_identity.json"))
        if "FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0." not in (output / "server.log").read_text():
            raise ValueError("Fresh native sampler startup not observed")
        p._write(output / "fresh_server_observation.json", {"version": version, "models": models,
                 "controller_sha256": result["controller_sha256"], "prior_identity_sha256": frozen[str(identity_path)]})
        if (p._code_hash() != result["code_sha256"]
                or any(f.sha(Path(path)) != sha for path, sha in frozen.items())):
            raise ValueError("Source/gates/parent/code changed before inheritance")
        shutil.copyfile(identity_path, output / "model_identity.json")
        audit = output / "audit"
        audit.mkdir()
        for name, items in (("prepared.jsonl", prepared), ("ledger.jsonl", reused)):
            for item in items:
                r.append_json(audit / name, item)
        result["audit"] = p.gated_audit(SOURCE, PARENT / "cpu", PARENT / "smoke", audit,
            output / "model_identity.json", prior["endpoint"], run=True, semantic_open=True,
            deadline_utc=DEADLINE, timeout=600, extraction_mode="segments")
        result["audit_validation"] = f.audit_result(audit, SOURCE)
        result["status"] = ("EXITED_LOCAL_RECOVERY_PENDING" if result["audit_validation"][
            "controller_validation"]["all900_actual_terminal"] else "INCOMPLETE_LOCAL_RECOVERY_PENDING")
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.SubprocessError) as error:
        result.update(status="BLOCKED_OR_FAILED_PRESERVED", error=str(error))
    finally:
        if server is not None:
            try:
                server.close()
                f.gpu_idle()
                result["owned_server_births_cleaned"] = server.known
            except (OSError, TimeoutError, ValueError, subprocess.SubprocessError) as error:
                result["cleanup_error"] = str(error)
        try:
            if (output / "audit/ledger.jsonl").exists():
                result["new_stage_cost"] = stage_cost(records(output / "audit/ledger.jsonl"), inherited)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update(status="BLOCKED_OR_FAILED_PRESERVED", accounting_error=str(error))
        result["closed"] = True
        p._write(output / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Interrupted by signal{signum}; close owned processes only")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = execute(args.pod_id, args.output, run=args.execute)
    print(r.canonical(result))
    return int(result["status"] not in {"DRY_PLAN_NO_GPU", "NOTHING_REMAINING_NO_GPU",
                                       "EXITED_LOCAL_RECOVERY_PENDING"} or "cleanup_error" in result)


if __name__ == "__main__":
    raise SystemExit(main())
