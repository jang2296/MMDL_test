"""One private, deadline-bound Gemma-only run; cloud lifecycle remains with the caller."""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
from typing import Any

from experiments.bottleneck_diagnostic import finish_runpod as f
from . import preflight, runtime as r


def execute(pod: str, source: Path, root: Path, output: Path, deadline: str,
            *, run: bool = False) -> dict[str, Any]:
    end = preflight._deadline(deadline)
    f.require_window(end, "Gemma-only preparation")
    code = Path(__file__).resolve().parents[2]
    source, root, output = source.resolve(), root.resolve(), output.resolve()
    if (not pod or not root.is_relative_to(Path("/workspace")) or root == Path("/workspace")
            or not output.is_relative_to(Path("/workspace")) or output == Path("/workspace")
            or output.exists() or source.is_relative_to(output) or root.is_relative_to(output)):
        raise ValueError("Fresh private /workspace outputs and an explicit Pod ID are required")
    rows, source_sha = preflight._full_rows(source)
    result: dict[str, Any] = {"status": "DRY_PLAN_NO_GPU", "pod_id": pod,
        "deadline_utc": deadline, "source_sha256": source_sha, "source_count": len(rows),
        "extraction_mode": "segments", "protocol": r.protocol_id("segments"),
        "training_allowed": False, "public_upload_allowed": False, "replaces_v8": False}
    if not run:
        return result
    if os.environ.get("RUNPOD_POD_ID") != pod or not os.path.ismount("/workspace"):
        raise ValueError("Authorized actual Pod and persistent workspace must match")
    output.mkdir(parents=True, exist_ok=False)
    server = None
    python = str(root / "venv/bin/python")
    common = [python, "-m", "experiments.gemma_verifier.preflight", "cpu", "--source", str(source),
              "--output", str(output / "cpu"), "--deadline-utc", deadline,
              "--extraction-mode", "segments"]
    try:
        env = f.gemma_environment(dict(os.environ, PYTHONPATH=f"{code / 'src'}:{code}",
            PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"), root)
        env["PYTHONPATH"] = f"{code / 'src'}:{code}"
        if f.run(common + ["--snapshot", str(root / "model")], output, "cpu", env, end, code):
            raise ValueError("CPU900 full-input gate failed")
        result["cpu"] = f.read(output / "cpu/summary.json")
        f.gpu_idle()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 8000))
        timeout = shutil.which("timeout")
        if not timeout or end - time.time() <= 10:
            raise ValueError("Hard timeout or termination margin missing")
        command = [timeout, "--signal=TERM", "--kill-after=10s", f"{end-time.time()-10:.3f}s",
            "bash", str(code / "experiments/gemma_verifier/start_server.sh"),
            "--model-path", str(root / "model"), "--output", str(output / "server"), "--execute"]
        server = f.OwnedProcess(command, output / "server.log", env, end, code)
        ready_end = min(end, time.time() + 600)
        ready = False
        while time.time() < ready_end and server.child.poll() is None:
            server.track()
            connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3)
            try:
                connection.request("GET", "/v1/models")
                response = connection.getresponse()
                models = json.loads(response.read(1024 * 1024)) if response.status == 200 else {}
                ready = any(item["id"] == "google/gemma-4-E4B-it" for item in models.get("data", []))
            except (OSError, ValueError, KeyError):
                pass
            finally:
                connection.close()
            if ready:
                break
            time.sleep(min(1, max(0, ready_end-time.time())))
        if not ready:
            raise TimeoutError("Loopback Gemma readiness failed; no fallback")
        if "FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0." not in (output / "server.log").read_text():
            raise ValueError("Native sampler policy was not observed")
        identity = f.read(output / "server/model_identity.json")
        identity["server_runtime_policy"] = {"VLLM_USE_FLASHINFER_SAMPLER": "0",
            "chat_template_content_format": "string", "submission_only_extraction": True,
            "source_segment_citations": True, "native_cross_field_schema_branches": True}
        identity["controller_sha256"] = f.sha(Path(__file__))
        with (output / "model_identity.json").open("x") as stream:
            json.dump(identity, stream, sort_keys=True)
            stream.write("\n")
        common[3] = "smoke"
        common[common.index("--output") + 1] = str(output / "smoke")
        common += ["--cpu-preflight", str(output / "cpu"), "--model-manifest",
                   str(output / "model_identity.json"), "--timeout", "600", "--execute"]
        if f.run(common, output, "smoke", env, end, code):
            raise ValueError("Actual smoke29 failed; full900 prohibited")
        common[3] = "audit"
        common[common.index("--output") + 1] = str(output / "audit")
        common += ["--smoke-gate", str(output / "smoke"), "--semantic-open"]
        result["audit_returncode"] = f.run(common, output, "audit", env, end, code)
        result["audit"] = f.audit_result(output / "audit", source)
        result["status"] = "EXITED_LOCAL_RECOVERY_PENDING"
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.SubprocessError) as error:
        result["status"], result["error"] = "BLOCKED_OR_FAILED_PRESERVED", str(error)
    finally:
        if server is not None:
            try:
                server.close()
                result["owned_server_births_cleaned"] = server.known
            except (OSError, TimeoutError, subprocess.SubprocessError) as error:
                result["cleanup_error"] = str(error)
        with (output / "summary.json").open("x") as stream:
            json.dump(result, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Interrupted by signal{signum}; close only owned processes")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = execute(args.pod_id, args.source, args.root, args.output, args.deadline_utc, run=args.execute)
    print(json.dumps({k: v for k, v in result.items() if k not in {"cpu", "audit"}}, sort_keys=True))
    return int(result["status"] == "BLOCKED_OR_FAILED_PRESERVED"
               or bool(result.get("audit_returncode")) or "cleanup_error" in result)


if __name__ == "__main__":
    raise SystemExit(main())
