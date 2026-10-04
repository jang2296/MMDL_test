"""Explicit synthetic CPU fixtures; no actual Pod/GPU/server operation."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

from experiments.bottleneck_diagnostic import finish_runpod as f


class FinishTests(unittest.TestCase):
    def test_explicit_sampler_preserves_identity_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes = {}
            for relative in f.SAMPLER_SOURCES:
                path = root / "venv/lib/python3.12/site-packages" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)
                hashes[relative] = f.sha(path)
            with mock.patch.object(f, "SAMPLER_SOURCES", hashes):
                env = f.gemma_environment({"PRESERVE": "yes", "VLLM_USE_FLASHINFER_SAMPLER": "1"}, root)
                self.assertEqual(env["VLLM_USE_FLASHINFER_SAMPLER"], "0")
                self.assertEqual(env["PRESERVE"], "yes")
                self.assertTrue(env["PYTHONPATH"].startswith(str(f.GEMMA_CODE) + ":"))
                path.write_text("changed")
                with self.assertRaisesRegex(ValueError, "sources changed"):
                    f.gemma_environment({}, root)

    def pilot(self):
        protocol = {"modes": f.MODES, "overall_deadline_utc": f.DEADLINE}
        identity = {"protocol": protocol, "protocol_hash": f.digest(protocol),
                    "execution_kind": "ACTUAL_INFERENCE", "repetitions": [3407]}
        history = [{"runtime_status": "COMPLETED", "elapsed_seconds": 10,
                    "input_tokens": 20, "output_tokens": 30} for _ in range(478)]
        latest = history + [{"runtime_status": "BLOCKED"}, {"runtime_status": "EXCLUDED"}]
        return identity, history, latest, [], protocol

    def test_complete_and_partial_pilot_gates_and_real_cost(self):
        values = self.pilot()
        result = f.pilot_gate(*values)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["completed_calls"], 478)
        self.assertEqual(result["completed_request_seconds"], 4780)
        self.assertEqual(result["output_tokens"], 14340)
        for status in ("BUDGET_STOP", "SYSTEM_ERROR", "PENDING"):
            changed = copy.deepcopy(values)
            changed[2][0]["runtime_status"] = status
            self.assertEqual(f.pilot_gate(*changed)["status"], "PARTIAL_OR_INVALID")
        changed = copy.deepcopy(values)
        changed[3].append(["synthetic_missing", "I", 3407])
        self.assertEqual(f.pilot_gate(*changed)["status"], "PARTIAL_OR_INVALID")
        changed = copy.deepcopy(values)
        changed[0]["execution_kind"] = "TEST_ONLY"
        self.assertEqual(f.pilot_gate(*changed)["status"], "PARTIAL_OR_INVALID")
        changed = copy.deepcopy(values)
        changed[1][0]["elapsed_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            f.pilot_gate(*changed)

    def test_coverage_measured_reserves_and_no_repeat_fallback(self):
        pilot = f.pilot_gate(*self.pilot())
        self.assertEqual(f.coverage_gate(pilot, 17820, True)["estimate_seconds"], 5220)
        self.assertTrue(f.coverage_gate(pilot, 17820, True)["allowed"])
        skipped = f.coverage_gate(pilot, 17819, True)
        self.assertFalse(skipped["allowed"])
        self.assertFalse(f.coverage_gate(pilot | {"status": "PARTIAL_OR_INVALID"}, 99999, True)["allowed"])
        self.assertFalse(f.coverage_gate(pilot, 99999, False)["allowed"])
        complete = {"status": "REVIEW", "controller_validation": {"all900_actual_terminal": True,
                    "ledger_rows": 900, "unique_ids": 900, "missing_ids": 0}}
        result = {"pilot": pilot, "coverage": skipped, "gemma_audit": complete, "gemma_audit_returncode": 0}
        self.assertTrue(f.after_gemma_gate(result, 7020, True)["allowed"])
        self.assertFalse(f.after_gemma_gate(result, 7019, True)["allowed"])
        self.assertIsNone(f.after_gemma_gate(result | {"coverage": skipped | {"attempted": True}}, 99999, True))
        self.assertIsNone(f.after_gemma_gate(result | {"cleanup_error": "uncertain"}, 99999, True))
        self.assertIsNone(f.after_gemma_gate(result | {"gemma_audit": {}}, 99999, True))
        for failure in ({"status": "INCOMPLETE_VALIDATION_FAILED"}, {"status": "INCOMPLETE"},
                        complete | {"controller_validation": complete["controller_validation"] | {"all900_actual_terminal": False}},
                        complete | {"controller_validation": complete["controller_validation"] | {"ledger_rows": 899, "unique_ids": 899, "missing_ids": 1}}):
            self.assertIsNone(f.after_gemma_gate(result | {"gemma_audit": failure}, 99999, True))
        self.assertIsNone(f.after_gemma_gate(result | {"gemma_audit_returncode": 1}, 99999, True))

    def test_parent_birth_command_and_terminal_zombie(self):
        expected = {"pid": 888, "start_ticks": 123, "command_sha256": "a" * 64}
        current = expected | {"state": "S"}
        self.assertTrue(f.verify_parent(current, expected))
        self.assertFalse(f.verify_parent(None, expected))
        self.assertFalse(f.verify_parent(current | {"state": "Z", "command_sha256": "empty"}, expected))
        for field in ("start_ticks", "command_sha256"):
            with self.assertRaises(ValueError):
                f.verify_parent(current | {field: "different"}, expected)

    def test_exact_birth_cleanup_does_not_signal_reused_or_other_pid(self):
        owned = object.__new__(f.OwnedProcess)
        owned.known = {100: 20, 101: 21, 102: 22}
        states = {100: {"start_ticks": 20, "state": "S"},
                  101: {"start_ticks": 999, "state": "S"},
                  102: {"start_ticks": 22, "state": "Z"}}
        sequence = []
        def opened(pid, flags):
            sequence.append(("open", pid))
            return pid + 1000
        def checked(pid):
            sequence.append(("validate", pid))
            return states[pid]
        with mock.patch.object(f, "process_identity", side_effect=checked), \
                mock.patch.object(f.os, "pidfd_open", side_effect=opened, create=True), \
                mock.patch.object(f.signal, "pidfd_send_signal", create=True) as send, \
                mock.patch.object(f.os, "close") as close, mock.patch.object(f.os, "kill") as kill:
            owned.signal_owned(signal.SIGTERM)
        send.assert_called_once_with(1100, signal.SIGTERM, None, 0)
        self.assertEqual(sequence, [(action, pid) for pid in (100, 101, 102) for action in ("open", "validate")])
        self.assertEqual(close.call_args_list, [mock.call(1100), mock.call(1101), mock.call(1102)])
        kill.assert_not_called()

    def test_pidfd_unavailable_open_error_and_exited_target_never_fall_back(self):
        owned = object.__new__(f.OwnedProcess)
        owned.known = {100: 20}
        with mock.patch.object(f.os, "pidfd_open", None, create=True), \
                mock.patch.object(f.signal, "pidfd_send_signal", create=True) as send, mock.patch.object(f.os, "kill") as kill:
            with self.assertRaisesRegex(OSError, "pidfd cleanup is required"):
                owned.signal_owned(signal.SIGTERM)
            send.assert_not_called()
            kill.assert_not_called()
        for error, raises in [(ProcessLookupError(), False), (OSError("kernel refusal"), True)]:
            with mock.patch.object(f.os, "pidfd_open", side_effect=error, create=True), \
                    mock.patch.object(f.signal, "pidfd_send_signal", create=True) as send, mock.patch.object(f.os, "kill") as kill:
                if raises:
                    with self.assertRaises(OSError):
                        owned.signal_owned(signal.SIGTERM)
                else:
                    owned.signal_owned(signal.SIGTERM)
                send.assert_not_called()
                kill.assert_not_called()
        with mock.patch.object(f.os, "pidfd_open", return_value=77, create=True), \
                mock.patch.object(f, "process_identity", return_value={"start_ticks": 20, "state": "S"}), \
                mock.patch.object(f.signal, "pidfd_send_signal", side_effect=OSError("signal refusal"), create=True), \
                mock.patch.object(f.os, "close") as close, mock.patch.object(f.os, "kill") as kill:
            with self.assertRaisesRegex(OSError, "signal refusal"):
                owned.signal_owned(signal.SIGTERM)
            close.assert_called_once_with(77)
            kill.assert_not_called()

    def test_expired_run_never_constructs_even_if_child_would_exit_immediately(self):
        for times in ([11], [9, 11]):
            with mock.patch.object(f.time, "time", side_effect=times), mock.patch.object(f, "event"), \
                    mock.patch.object(f, "OwnedProcess") as constructor:
                constructor.return_value.child.poll.return_value = 0
                with self.assertRaises(TimeoutError):
                    f.run(["synthetic-only"], Path("/synthetic"), "expired", {}, 10)
                constructor.assert_not_called()
        owned = mock.Mock()
        owned.child.poll.return_value = 0
        with mock.patch.object(f.time, "time", side_effect=[9, 9, 11]), mock.patch.object(f, "event"), \
                mock.patch.object(f, "OwnedProcess", return_value=owned):
            with self.assertRaises(TimeoutError):
                f.run(["synthetic-only"], Path("/synthetic"), "fast-exit", {}, 10)
            owned.child.poll.assert_not_called()
            owned.close.assert_called_once()

    def test_constructor_rechecks_deadline_immediately_before_popen(self):
        log = mock.Mock()
        with mock.patch.object(f.time, "time", side_effect=[9, 11]), \
                mock.patch.object(f.os, "pidfd_open", return_value=99, create=True), \
                mock.patch.object(f.signal, "pidfd_send_signal", create=True), mock.patch.object(f.os, "close"), \
                mock.patch.object(Path, "open", return_value=log), mock.patch.object(f.subprocess, "Popen") as popen:
            with self.assertRaises(TimeoutError):
                f.OwnedProcess(["synthetic-only"], Path("/synthetic"), {}, 10)
            popen.assert_not_called()
            log.close.assert_called_once()

    def test_cpu_identity_matches_deployed_not_later_local_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "experiments/gemma_verifier"
            directory.mkdir(parents=True)
            evidence = root / "analysis_exports/bottleneck_20261003"
            evidence.mkdir(parents=True)
            (evidence / "gemma_source_v8.jsonl").write_text("explicit synthetic source\n")
            preflight = root / "deployed/preflight.py"
            preflight.parent.mkdir()
            preflight.write_text("explicit synthetic deployed v5\n")
            for name in ("runtime.py", "start_server.sh"):
                (directory / name).write_text("explicit synthetic " + name)
            names = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
            spec = {"model_id": "synthetic-model", "model_revision": "synthetic-revision",
                    "expected_model_files": {name: {"sha256": "a" * 64} for name in names}}
            (directory / "deployment.json").write_text(json.dumps(spec))
            cpu = root / "cpu"
            cpu.mkdir()
            (cpu / "token_budget.jsonl").write_text("explicit synthetic ledger\n")
            components = {p.name: f.sha(p) for p in [preflight, directory / "runtime.py",
                         directory / "deployment.json", directory / "start_server.sh"]}
            manifest = {"code_sha256": f.digest(components), "source_sha256": f.sha(evidence / "gemma_source_v8.jsonl"),
                        "model": spec["model_id"], "revision": spec["model_revision"], "kind": "ACTUAL_CPU_TOKENIZER",
                        "source_count": 900, "input_truncation": False,
                        "max_model_len": 131072, "output_allowance": 1024,
                        "tokenizer_files_sha256": {name: "a" * 64 for name in names},
                        "runtime_chat_template_sha256": "a" * 64}
            (cpu / "manifest.json").write_text(json.dumps(manifest))
            summary = {"source_sha256": manifest["source_sha256"], "manifest_sha256": f.sha(cpu / "manifest.json"),
                       "token_budget_sha256": f.sha(cpu / "token_budget.jsonl"), "status": "CPU_TOKENIZER_PASS",
                       "counted": 900, "context_overflows": 0, "truncation": False,
                       "max_input_plus_output": 51003,
                       "full_audit_allowed": False, "judge_calls": 0}
            (cpu / "summary.json").write_text(json.dumps(summary))
            self.assertEqual(f.verify_cpu_identity(root, preflight, cpu)["code_sha256"], manifest["code_sha256"])
            preflight.write_text("different later local annotation\n")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                f.verify_cpu_identity(root, preflight, cpu)

    def test_only_attested_setup_exception_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / f.SETUP_EXCEPTION
            path.parent.mkdir(parents=True)
            path.write_text("explicit synthetic authorized setup replacement")
            original = {"path": f.SETUP_EXCEPTION, "bytes": 1, "sha256": "b" * 64}
            (root / "TRANSFER_FILES.json").write_text(json.dumps([original]))
            with mock.patch.object(f, "INVENTORY_SHA", f.sha(root / "TRANSFER_FILES.json")), mock.patch.object(f, "SETUP_EXCEPTION_SHA", f.sha(path)):
                self.assertEqual(f.verify_bundle(root)[f.SETUP_EXCEPTION], f.sha(path))
                path.write_text("unapproved mutation")
                with self.assertRaisesRegex(ValueError, "file changed"):
                    f.verify_bundle(root)

    def test_append_only_audit_summary_review_never_final_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            entries = [{"id": f"synthetic_{i}", "subject": f"synthetic_subject_{i // 30}",
                        "status": "REVIEW", "candidate_correct": False,
                        "requests": {"extraction": {"explicit": "synthetic"}},
                        "http_responses": {"extraction": {"usage": {"prompt_tokens": 2, "completion_tokens": 3}},
                                           "mapping": {"usage": {"prompt_tokens": 4, "completion_tokens": 5}}},
                        "elapsed_seconds": .1} for i in range(900)]
            source.write_text("".join(json.dumps({"id": row["id"]}) + "\n" for row in entries))
            manifest = {"source_sha256": f.sha(source), "source_count": 900, "execute": True,
                        "semantic_open": True, "model": "google/gemma-4-E4B-it",
                        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
                        "endpoint": "http://127.0.0.1:8000/v1", "model_manifest": {"explicit": "synthetic"},
                        "model_manifest_sha256": f.digest({"explicit": "synthetic"}), "protocol_sha256": "a" * 64}
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            for entry in entries:
                entry.update({name: manifest[name] for name in
                             ("model", "revision", "model_manifest_sha256", "protocol_sha256")})
            def save():
                (root / "ledger.jsonl").write_text("".join(json.dumps(row | {"entry_sha256": f.digest(row)}) + "\n" for row in entries))
            save()
            summary = {"status": "REVIEW", "assisted_accuracy": None, "statuses": {"REVIEW": 900},
                       "source_count": 900, "denominator": 900, "execute": True, "dry_run": False,
                       "replaces_v8": False}
            (root / "summaries.jsonl").write_text(json.dumps({"old": "ignored"}) + "\n" + json.dumps(summary) + "\n")
            result = f.audit_result(root, source)
            self.assertTrue(result["controller_validation"]["all900_actual_terminal"])
            self.assertFalse(result["controller_validation"]["final_score_allowed"])
            self.assertEqual(result["controller_validation"]["output_tokens"], 7200)
            self.assertEqual(result["controller_validation"]["input_tokens"], 5400)
            self.assertEqual(result["controller_validation"]["actual_completion_responses"], 1800)
            (root / "summaries.jsonl").write_text(json.dumps(summary | {"assisted_accuracy": .5}) + "\n")
            with self.assertRaisesRegex(ValueError, "unresolved score"):
                f.audit_result(root, source)
            (root / "summaries.jsonl").unlink()
            entries.pop()
            save()
            result = f.audit_result(root, source)
            self.assertEqual(result["status"], "INCOMPLETE_NO_SUMMARY")
            self.assertFalse(result["controller_validation"]["all900_actual_terminal"])

    def test_default_dry_plan_does_not_inspect_processes_or_gpu(self):
        stream = io.StringIO()
        with mock.patch("sys.argv", ["finish_runpod.py", "--pod-id", f.POD, "--coverage"]), \
                mock.patch.object(f, "execute") as execute, \
                mock.patch.object(f, "process_identity") as process, \
                mock.patch.object(f, "gpu_idle") as gpu, redirect_stdout(stream):
            self.assertEqual(f.main(), 0)
        execute.assert_not_called()
        process.assert_not_called()
        gpu.assert_not_called()
        self.assertEqual(json.loads(stream.getvalue())["status"], "DRY_PLAN_NO_GPU_OR_PROCESS_OPERATIONS")


if __name__ == "__main__":
    unittest.main()
