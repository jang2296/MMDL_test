"""CPU-only Gemma controller boundaries; every process/HTTP call is mocked."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any
import unittest
from unittest import mock

from experiments.gemma_verifier import rerun


class RerunTests(unittest.TestCase):
    deadline = "2099-01-01T00:00:00Z"
    pod = "synthetic-owned-pod"

    def test_dry_plan_has_no_side_effects_or_process_calls(self) -> None:
        output = Path("/workspace/synthetic-rerun-never-created/output")
        with mock.patch.object(rerun.preflight, "_full_rows", return_value=([{}] * 900, "a" * 64)), \
                mock.patch.object(rerun.f, "run") as run, \
                mock.patch.object(rerun.f, "OwnedProcess") as process, \
                mock.patch.object(rerun.f, "gpu_idle") as gpu, \
                mock.patch.object(rerun.http.client, "HTTPConnection") as http, \
                mock.patch.object(Path, "mkdir") as mkdir:
            result = rerun.execute(self.pod, Path("/workspace/source.jsonl"),
                                   Path("/workspace/synthetic-model"), output, self.deadline)
        self.assertEqual(result["status"], "DRY_PLAN_NO_GPU")
        self.assertEqual(result["source_count"], 900)
        self.assertEqual(result["extraction_mode"], "segments")
        self.assertEqual(result["protocol"], rerun.r.protocol_id("segments"))
        for call in (run, process, gpu, http, mkdir):
            call.assert_not_called()

    def test_invalid_output_is_rejected_before_reading_source(self) -> None:
        for output in (Path("/workspace"), Path("/tmp/synthetic-rerun")):
            with self.subTest(output=output), \
                    mock.patch.object(rerun.preflight, "_full_rows") as read, \
                    self.assertRaisesRegex(ValueError, "Fresh private"):
                rerun.execute(self.pod, Path("/workspace/source.jsonl"),
                              Path("/workspace/synthetic-model"), output, self.deadline)
            read.assert_not_called()

    def test_expired_time_precedes_source_or_process_work(self) -> None:
        with mock.patch.object(rerun.preflight, "_full_rows") as read, \
                mock.patch.object(rerun.f, "run") as run, \
                self.assertRaises(TimeoutError):
            rerun.execute(self.pod, Path("/workspace/source.jsonl"),
                          Path("/workspace/synthetic-model"),
                          Path("/workspace/synthetic-output"), "2000-01-01T00:00:00Z", run=True)
        read.assert_not_called()
        run.assert_not_called()

    def execution_mocks(self, stack: ExitStack, *, cpu_returncode: int = 0) -> tuple[Path, Any, Any, Any]:
        temporary = stack.enter_context(tempfile.TemporaryDirectory())
        output = Path(temporary) / "output"
        # Only the /workspace containment predicate is simulated. Writes stay in
        # this disposable local directory; no real Pod/mount/environment exists.
        stack.enter_context(mock.patch.object(Path, "is_relative_to", autospec=True,
                                              side_effect=lambda _path, other: other == Path("/workspace")))
        stack.enter_context(mock.patch.dict(rerun.os.environ, {"RUNPOD_POD_ID": self.pod}))
        stack.enter_context(mock.patch.object(rerun.os.path, "ismount", return_value=True))
        stack.enter_context(mock.patch.object(rerun.preflight, "_full_rows",
                                              return_value=([{}] * 900, "a" * 64)))
        stack.enter_context(mock.patch.object(rerun.f, "gemma_environment",
                                              side_effect=lambda env, _root: env))
        run = stack.enter_context(mock.patch.object(rerun.f, "run", return_value=cpu_returncode))
        gpu = stack.enter_context(mock.patch.object(rerun.f, "gpu_idle"))
        process = stack.enter_context(mock.patch.object(rerun.f, "OwnedProcess"))
        return output, run, gpu, process

    def call_execution(self, output: Path) -> dict[str, Any]:
        return rerun.execute(self.pod, Path("/workspace/synthetic-source.jsonl"),
                             Path("/workspace/synthetic-model"), output, self.deadline, run=True)

    def test_first_cpu_gate_failure_never_loads_server_and_preserves_receipt(self) -> None:
        with ExitStack() as stack:
            output, run, gpu, process = self.execution_mocks(stack, cpu_returncode=1)
            http = stack.enter_context(mock.patch.object(rerun.http.client, "HTTPConnection"))
            result = self.call_execution(output)
            self.assertEqual(result["status"], "BLOCKED_OR_FAILED_PRESERVED")
            self.assertIn("CPU900", result["error"])
            self.assertEqual(json.loads((output / "summary.json").read_text()), result)
            self.assertEqual(run.call_args.args[0][3], "cpu")
            self.assertIn("segments", run.call_args.args[0])
            gpu.assert_not_called()
            process.assert_not_called()
            http.assert_not_called()

    def test_subprocess_gpu_gate_failure_is_preserved_as_failure(self) -> None:
        with ExitStack() as stack:
            output, _run, gpu, process = self.execution_mocks(stack)
            stack.enter_context(mock.patch.object(rerun.f, "read", return_value={}))
            gpu.side_effect = subprocess.CalledProcessError(1, ["synthetic-nvidia-smi"])
            result = self.call_execution(output)
            self.assertEqual(result["status"], "BLOCKED_OR_FAILED_PRESERVED")
            self.assertEqual(json.loads((output / "summary.json").read_text()), result)
            process.assert_not_called()

    def test_smoke_failure_closes_owned_server_and_never_runs_audit(self) -> None:
        with ExitStack() as stack:
            output, run, _gpu, process = self.execution_mocks(stack)
            commands: list[list[str]] = []

            def stage(command: list[str], *_args: Any) -> int:
                commands.append(list(command))
                if command[3] == "cpu":
                    (output / "server.log").write_text(
                        "FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.")
                    return 0
                return 1

            run.side_effect = stage
            stack.enter_context(mock.patch.object(rerun.f, "read", return_value={}))
            stack.enter_context(mock.patch.object(rerun.shutil, "which", return_value="/usr/bin/timeout"))
            stack.enter_context(mock.patch.object(rerun.socket, "socket"))
            connection = stack.enter_context(mock.patch.object(rerun.http.client, "HTTPConnection"))
            response = connection.return_value.getresponse.return_value
            response.status = 200
            response.read.return_value = b'{"data":[{"id":"google/gemma-4-E4B-it"}]}'
            process.return_value.child.poll.return_value = None
            process.return_value.known = {123: 456}
            result = self.call_execution(output)
            self.assertEqual(result["status"], "BLOCKED_OR_FAILED_PRESERVED")
            self.assertIn("smoke29", result["error"])
            self.assertEqual([command[3] for command in commands], ["cpu", "smoke"])
            self.assertEqual(commands[1][commands[1].index("--output") + 1], str(output / "smoke"))
            self.assertIn("--cpu-preflight", commands[1])
            process.return_value.close.assert_called_once_with()
            self.assertEqual(result["owned_server_births_cleaned"], {123: 456})

    def test_main_reports_audit_or_cleanup_failure_as_nonzero(self) -> None:
        args = ["synthetic-rerun", "--pod-id", self.pod, "--source", "/workspace/source.jsonl",
                "--root", "/workspace/synthetic-model", "--output", "/workspace/synthetic-output",
                "--deadline-utc", self.deadline]
        for failure in ({"audit_returncode": 1}, {"cleanup_error": "synthetic cleanup failure"}):
            with self.subTest(failure=failure), \
                    mock.patch("sys.argv", args), \
                    mock.patch.object(rerun.signal, "signal"), \
                    mock.patch.object(rerun, "execute", return_value={
                        "status": "EXITED_LOCAL_RECOVERY_PENDING", **failure}), \
                    mock.patch("builtins.print"):
                self.assertEqual(rerun.main(), 1)


if __name__ == "__main__":
    unittest.main()
