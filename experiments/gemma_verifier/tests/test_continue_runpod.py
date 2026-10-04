"""Synthetic CPU-only continuation contracts; never a Pod, signal, or live inference."""
from __future__ import annotations

import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest import mock

from experiments.gemma_verifier import continue_runpod as c


class ContinuationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent, self.prepared = self.root / "parent", self.root / "validation"
        (self.parent / "audit").mkdir(parents=True)
        self.prepared.mkdir()
        self.manifest = {"execute": True, "deadline_unix": 1, "source_sha256": "source",
                         "model": "synthetic", "revision": "frozen", "protocol_sha256": "protocol",
                         "model_manifest_sha256": "identity", "extraction_mode": "segments"}
        c.p._write(self.parent / "audit/run_manifest.json", self.manifest)
        c.p._write(self.prepared / "run_manifest.json", self.manifest | {"execute": False, "deadline_unix": 2})
        self.requests = [{"id": f"case{i}", "resume_key": f"key{i}", "input_sha256": f"input{i}",
                          "payload": {"messages": [f"raw{i}"]}} for i in range(3)]
        for request in self.requests:
            self.append(self.prepared / "prepared.jsonl", request)
            self.append(self.parent / "audit/prepared.jsonl", request)
        self.entries = [self.entry(i, status) for i, status in enumerate(("REVIEW", "SYSTEM_ERROR", "BUDGET_STOP"))]
        for entry in self.entries:
            self.append(self.parent / "audit/ledger.jsonl", entry)

    @staticmethod
    def append(path: Path, entry: dict[str, Any]) -> None:
        c.r.append_json(path, entry | {"entry_sha256": c.r.digest(entry)})

    def entry(self, index: int, status: str) -> dict[str, Any]:
        return {k: self.manifest[k] for k in ("model", "revision", "model_manifest_sha256", "protocol_sha256")} | {
            "id": f"case{index}", "resume_key": f"key{index}", "status": status,
            "attempt": 0 if status == "BUDGET_STOP" else 1,
            "http_responses": {} if status == "BUDGET_STOP" else {"extraction": {
                "id": f"synthetic-completion-{index}", "choices": [], "usage": {
                    "prompt_tokens": 10, "completion_tokens": 1}}},
            "requests": {} if status == "BUDGET_STOP" else {"extraction": self.requests[index]["payload"]}}

    def replace_manifest(self, changes: dict[str, Any]) -> None:
        path = self.parent / "audit/run_manifest.json"
        path.write_text(c.r.canonical(self.manifest | changes))

    def replace_entries(self, entries: list[dict[str, Any]]) -> None:
        path = self.parent / "audit/ledger.jsonl"
        path.write_text("".join(c.r.canonical(e | {"entry_sha256": c.r.digest(e)}) + "\n" for e in entries))

    def test_actual_errors_reused_budget_excluded_and_parent_bytes_unchanged(self) -> None:
        before = {p: p.read_bytes() for p in (self.parent / "audit").iterdir()}
        prepared, inherited = c.inheritance(self.parent, self.prepared)
        self.assertEqual(len(prepared), 3)
        self.assertEqual([e["id"] for e in inherited], ["case0", "case1"])
        self.assertEqual(before, {p: p.read_bytes() for p in (self.parent / "audit").iterdir()})

    def test_source_model_protocol_mode_changes_rejected(self) -> None:
        for field in ("source_sha256", "model", "revision", "protocol_sha256", "model_manifest_sha256", "extraction_mode"):
            with self.subTest(field=field):
                self.replace_manifest({field: "changed"})
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    c.inheritance(self.parent, self.prepared)
        self.replace_manifest({"execute": False})
        with self.assertRaises(ValueError):
            c.inheritance(self.parent, self.prepared)

    def test_no_response_system_error_excluded_but_returned_malformed_output_preserved(self) -> None:
        no_response = self.entries[1] | {"http_responses": {"extraction_tokenization": {"count": 10}}}
        self.replace_entries([self.entries[0], no_response, self.entries[2]])
        before = (self.parent / "audit/ledger.jsonl").read_bytes()
        _prepared, inherited = c.inheritance(self.parent, self.prepared)
        self.assertEqual([e["id"] for e in inherited], ["case0"])
        self.assertEqual(before, (self.parent / "audit/ledger.jsonl").read_bytes())
        cost = c.stage_cost([no_response], set())
        self.assertEqual(cost["extraction"]["response_calls"], 0)
        self.assertEqual(cost["extraction"]["input_tokens"], 0)
        self.assertEqual(cost["extraction"]["output_tokens"], 0)
        malformed = self.entries[1] | {"http_responses": {"extraction": {"invalid": "actual returned JSON"}}}
        self.replace_entries([malformed])
        _prepared, inherited = c.inheritance(self.parent, self.prepared)
        self.assertEqual(inherited[0]["http_responses"]["extraction"], malformed["http_responses"]["extraction"])
        for actual_returned_json in (None, ["invalid native shape"], {"usage": None}):
            malformed = self.entries[1] | {"http_responses": {"extraction": actual_returned_json}}
            self.replace_entries([malformed])
            _prepared, inherited = c.inheritance(self.parent, self.prepared)
            self.assertEqual(len(inherited), 1)
            cost = c.stage_cost(inherited, set())
            self.assertEqual(cost["extraction"]["response_calls"], 1)
            self.assertEqual(cost["extraction"]["input_tokens"], 0)

    def test_partial_parent_json_is_rejected_not_trimmed(self) -> None:
        path = self.parent / "audit/ledger.jsonl"
        original = path.read_bytes() + b'{"id":"partial"'
        path.write_bytes(original)
        with self.assertRaises(ValueError):
            c.inheritance(self.parent, self.prepared)
        self.assertEqual(path.read_bytes(), original)

    def test_resume_key_payload_and_missing_actual_attempt_rejected(self) -> None:
        for mutation in ({"resume_key": "bad"}, {"requests": {"extraction": {"messages": ["changed"]}}},
                         {"attempt": 0}, {"model_manifest_sha256": "changed"}):
            with self.subTest(mutation=mutation):
                self.replace_entries([self.entries[0] | mutation])
                with self.assertRaises(ValueError):
                    c.inheritance(self.parent, self.prepared)
        path = self.parent / "audit/prepared.jsonl"
        path.write_text(c.r.canonical(self.requests[0] | {"entry_sha256": "bad"}) + "\n")
        with self.assertRaisesRegex(ValueError, "hashes"):
            c.inheritance(self.parent, self.prepared)

    def test_parent_not_closed_wrong_pod_or_cleanup_failure_rejected(self) -> None:
        good = {"pod_id": c.POD, "protocol": c.r.protocol_id("segments"), "extraction_mode": "segments",
                "status": "BLOCKED_OR_FAILED_PRESERVED", "owned_server_births_cleaned": {"123": 456}}
        with mock.patch.object(c.f, "read", return_value=good):
            self.assertEqual(c.closed(self.parent, c.POD), good)
        for changes in ({"status": "RUNNING"}, {"cleanup_error": "still alive"}, {"pod_id": "other"},
                        {"extraction_mode": "literal"}, {"protocol": "old"}):
            with mock.patch.object(c.f, "read", return_value=good | changes), self.assertRaises(ValueError):
                c.closed(self.parent, c.POD)
        without_cleanup = {k: v for k, v in good.items() if k != "owned_server_births_cleaned"}
        with mock.patch.object(c.f, "read", return_value=without_cleanup), self.assertRaises(ValueError):
            c.closed(self.parent, c.POD)

    def test_fresh_static_identity_exact_except_explicit_prior_attestation(self) -> None:
        static = {"model_id": "gemma", "revision": "frozen", "dtype": "bfloat16", "server_started": False,
                  "environment_lock_sha256": "env", "weights_manifest_sha256": "weights"}
        prior = static | {"controller_sha256": "OLD", "server_runtime_policy": {"sampler": "native"}}
        c.static_identity(prior, static)
        self.assertEqual(prior["controller_sha256"], "OLD")
        for mutation in ({"dtype": "float16"}, {"revision": "other"}, {"environment_lock_sha256": "new"},
                         {"server_started": True}, {"extra": "unattested"}, {"controller_sha256": "NEW"}):
            with self.assertRaises(ValueError):
                c.static_identity(prior, static | mutation)

    def test_cost_excludes_all_inherited_stage_calls(self) -> None:
        inherited = copy.deepcopy(self.entries[0]) | {"http_responses": {
            "extraction": {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}}}
        fresh = self.entry(2, "REVIEW") | {"requests": {"extraction": {}, "mapping": {}}, "http_responses": {
            "extraction": {"usage": {"prompt_tokens": 10, "completion_tokens": 4}},
            "mapping": {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}}}
        cost = c.stage_cost([inherited, fresh], {"case0"})
        self.assertEqual(cost["extraction"], {"recorded_requests": 1, "response_calls": 1,
                                              "input_tokens": 10, "output_tokens": 4})
        self.assertEqual(cost["mapping"]["input_tokens"], 7)
        self.assertEqual(cost["equivalence"]["response_calls"], 0)

    def test_expired_or_other_pod_rejected_before_any_gpu_work(self) -> None:
        with mock.patch.object(c.p, "_deadline", return_value=0), mock.patch.object(c.f, "OwnedProcess") as process, \
                self.assertRaises(TimeoutError):
            c.execute(c.POD, Path("/workspace/artifacts/synthetic-continuation"), run=True)
        process.assert_not_called()
        with mock.patch.object(c.f, "require_window"), mock.patch.object(c, "closed") as closed, \
                mock.patch.object(c.f, "OwnedProcess") as process, self.assertRaises(ValueError):
            c.execute("other", Path("/workspace/artifacts/synthetic-continuation"), run=True)
        closed.assert_not_called()
        process.assert_not_called()

    def execute_mocks(self, stack: ExitStack) -> tuple[Path, Any, Any, Any]:
        output = self.root / "output"
        identity = {"model_id": "synthetic", "vllm_version": "synthetic", "endpoint": "http://127.0.0.1:8000/v1"}
        stack.enter_context(mock.patch.object(Path, "is_relative_to", autospec=True,
            side_effect=lambda _path, other: other == Path("/workspace/artifacts")))
        stack.enter_context(mock.patch.dict(c.os.environ, {"RUNPOD_POD_ID": c.POD}))
        stack.enter_context(mock.patch.object(c.os.path, "ismount", return_value=True))
        stack.enter_context(mock.patch.object(c, "closed"))
        stack.enter_context(mock.patch.object(c, "gemma_export"))
        stack.enter_context(mock.patch.object(c.f, "sha", return_value=c.SOURCE_SHA))
        stack.enter_context(mock.patch.object(c.f, "read", return_value=identity))
        stack.enter_context(mock.patch.object(c.p, "_code_hash", return_value="synthetic-code"))
        audit = stack.enter_context(mock.patch.object(c.p, "gated_audit", return_value={"status": "INCOMPLETE"}))
        stack.enter_context(mock.patch.object(c, "inheritance", return_value=([], [])))
        stack.enter_context(mock.patch.object(c.f, "gemma_environment", side_effect=lambda env, _root: env))
        stack.enter_context(mock.patch.object(c.shutil, "which", return_value="/usr/bin/timeout"))
        stack.enter_context(mock.patch.object(c.shutil, "copyfile"))
        stack.enter_context(mock.patch.object(c.socket, "socket"))
        gpu = stack.enter_context(mock.patch.object(c.f, "gpu_idle"))
        process = stack.enter_context(mock.patch.object(c.f, "OwnedProcess"))
        process.return_value.child.poll.return_value = None
        process.return_value.known = {123: 456}
        def started(*_args: Any) -> Any:
            (output / "server.log").write_text(
                "FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.")
            return process.return_value
        process.side_effect = started
        def http(_endpoint: str, path: str, **_kwargs: Any) -> dict[str, Any]:
            return {"version": "synthetic"} if path == "/version" else {"data": [{"id": "synthetic"}]}
        stack.enter_context(mock.patch.object(c.r, "http_json", side_effect=http))
        stack.enter_context(mock.patch.object(c.f, "audit_result", return_value={
            "controller_validation": {"all900_actual_terminal": False}}))
        return output, audit, gpu, process

    def test_dry_plan_never_launches_gpu_or_creates_actual_output(self) -> None:
        with ExitStack() as stack:
            output, audit, gpu, process = self.execute_mocks(stack)
            result = c.execute(c.POD, output)
            self.assertEqual(result["status"], "DRY_PLAN_NO_GPU")
            self.assertFalse(output.exists())
            self.assertFalse(audit.call_args.kwargs.get("run", False))
            gpu.assert_not_called()
            process.assert_not_called()

    def test_gpu_occupied_refuses_server_and_writes_closed_failure(self) -> None:
        with ExitStack() as stack:
            output, audit, gpu, process = self.execute_mocks(stack)
            gpu.side_effect = ValueError("GPU compute processes remain")
            result = c.execute(c.POD, output, run=True)
            self.assertEqual(result["status"], "BLOCKED_OR_FAILED_PRESERVED")
            self.assertTrue(result["closed"])
            self.assertTrue((output / "summary.json").exists())
            process.assert_not_called()
            self.assertEqual(audit.call_count, 1)

    def test_fresh_server_mismatch_blocks_audit_and_cleans_only_owned(self) -> None:
        with ExitStack() as stack:
            output, audit, gpu, process = self.execute_mocks(stack)
            stack.enter_context(mock.patch.object(c, "static_identity", side_effect=ValueError("identity differs")))
            result = c.execute(c.POD, output, run=True)
            self.assertEqual(result["status"], "BLOCKED_OR_FAILED_PRESERVED")
            self.assertEqual(audit.call_count, 1)
            process.return_value.close.assert_called_once_with()
            self.assertEqual(gpu.call_count, 2)

    def test_partial_audit_is_not_complete_and_uses_fresh_true_execute_dir(self) -> None:
        with ExitStack() as stack:
            output, audit, _gpu, process = self.execute_mocks(stack)
            result = c.execute(c.POD, output, run=True)
            self.assertEqual(result["status"], "INCOMPLETE_LOCAL_RECOVERY_PENDING")
            first, second = audit.call_args_list
            self.assertNotEqual(first.args[3], second.args[3])
            self.assertFalse(first.kwargs.get("run", False))
            self.assertIs(second.kwargs["run"], True)
            self.assertEqual(second.args[3], output / "audit")
            self.assertEqual(second.kwargs["deadline_utc"], c.DEADLINE)
            self.assertEqual(second.kwargs["extraction_mode"], "segments")
            process.return_value.close.assert_called_once_with()
            self.assertTrue(result["closed"])

    def test_changed_frozen_source_refuses_prepare_and_gpu(self) -> None:
        with ExitStack() as stack:
            output, audit, gpu, process = self.execute_mocks(stack)
            stack.enter_context(mock.patch.object(c.f, "sha", return_value="changed-source"))
            with self.assertRaisesRegex(ValueError, "Frozen source"):
                c.execute(c.POD, output, run=True)
            audit.assert_not_called()
            gpu.assert_not_called()
            process.assert_not_called()

    def test_actual_endpoint_mismatch_prevents_reusing_old_gate(self) -> None:
        with ExitStack() as stack:
            output, audit, _gpu, process = self.execute_mocks(stack)
            stack.enter_context(mock.patch.object(c.r, "http_json", return_value={"version": "different"}))
            result = c.execute(c.POD, output, run=True)
            self.assertEqual(result["status"], "BLOCKED_OR_FAILED_PRESERVED")
            self.assertIn("endpoint model/version", result["error"])
            self.assertEqual(audit.call_count, 1)
            process.return_value.close.assert_called_once_with()

    def test_port_probe_uses_reuseaddr_not_reuseport_and_listener_conflict_blocks(self) -> None:
        with ExitStack() as stack:
            output, audit, _gpu, process = self.execute_mocks(stack)
            sock = stack.enter_context(mock.patch.object(c.socket, "socket"))
            probe = sock.return_value.__enter__.return_value
            probe.bind.side_effect = OSError("live listener already bound")
            result = c.execute(c.POD, output, run=True)
            probe.setsockopt.assert_called_once_with(c.socket.SOL_SOCKET, c.socket.SO_REUSEADDR, 1)
            probe.bind.assert_called_once_with(("127.0.0.1", 8000))
            self.assertEqual(result["status"], "BLOCKED_OR_FAILED_PRESERVED")
            self.assertEqual(audit.call_count, 1)
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
