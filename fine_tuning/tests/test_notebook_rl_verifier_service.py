"""Authored vaults and fresh interpreters; no benchmark/model processes."""
import hashlib
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl.schema import DiagTarget, VerifierResult
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier_service import (
    MAX_FRAME_BYTES, VerifierService, VerifierServiceError,
)
from fine_tuning.tests.test_notebook_rl_schema import records


class VerifierServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "vault"
        actor, verifier, lineage, *_ = records()
        self.actor, self.item = actor, verifier
        self.secret = "PRIVATE-ORACLE-DO-NOT-RETURN-38491"
        verifier = replace(verifier, oracle_fact_lines=(self.secret,), accept={"private_annotation": self.secret})
        write_split(self.root, [actor], [verifier], [lineage])

    def service(self, **kwargs):
        service = VerifierService(self.root, forbidden_roots=[self.base / "benchmark"], **kwargs)
        self.addCleanup(service.close)
        return service

    def test_fresh_process_all_five_methods_and_bounded_metadata(self):
        service = self.service()
        self.assertNotEqual(service.metadata["pid"], os.getpid())
        self.assertTrue(service.metadata["python_guards_installed"])
        self.assertFalse(service.metadata["os_sandbox"])
        self.assertEqual(service.metadata["item_count"], 1)
        text = "Observed facts:\ncell[a][b] = 8\nSolution:\nFinal answer: 8"
        result = service.grade(self.actor.problem_id, text, "stop", "obs")
        self.assertEqual(result.status, "INCORRECT")
        self.assertEqual(service.cause(self.actor.problem_id, result), "VISUAL_MISREAD")
        target = service.diagnosis_target(self.actor.problem_id, result)
        self.assertEqual(target, DiagTarget("VISUAL", ("cell[a][b]",)))
        route, evidence = service.route(self.actor.problem_id, [result], "verified-any-visual-v1")
        self.assertEqual(route, "VISION_ON")
        self.assertFalse(evidence["module_causality_certified"])
        note = "Cause: VISUAL_MISREAD\nWrong item: cell[a][b]\nLesson: Read the cell."
        self.assertEqual(service.grade_diagnosis(self.actor.problem_id, note, target), 1.25)
        correct = service.grade(self.actor.problem_id, "Final answer: 7", "stop", "plain")
        self.assertEqual(correct.reward_answer, 1)
        self.assertIsNone(service.cause(self.actor.problem_id, correct))
        self.assertIsNone(service.diagnosis_target(self.actor.problem_id, correct))
        public = json.dumps([service.metadata, result.__dict__, evidence, target.__dict__], default=str)
        self.assertNotIn(self.secret, public)
        for field in ("oracle_fact_lines", "printed_text", "gold", "accept", "source_revision"):
            self.assertNotIn(f'"{field}"', public)
        service.close()
        self.assertIsNotNone(service._process.poll())
        with self.assertRaisesRegex(VerifierServiceError, "CLOSED"):
            service.grade(self.actor.problem_id, "Final answer: 7", "stop", "plain")

    def test_actor_never_opens_private_vault_even_with_installed_guards(self):
        # The parent harness may own private fixtures; an independently guarded actor does not.
        script = """
import json, pathlib, sys
from fine_tuning.notebook_rl.guards import install_guards
from fine_tuning.notebook_rl.verifier_service import VerifierService
root=pathlib.Path(sys.argv[1])
private=[root/'verifier_private.jsonl',root/'lineage.jsonl']
install_guards(forbidden_roots=[root.parent/'benchmark'],vault_files=private,
               allow_vault=False,allow_benchmark=False)
opened=[]
def audit(event,args):
    if event=='open' and isinstance(args[0],str): opened.append(args[0])
sys.addaudithook(audit)
with VerifierService(root,forbidden_roots=[root.parent/'benchmark']) as service:
    assert service.grade('fake:family:train:1','Final answer: 8','stop','plain').status=='INCORRECT'
    assert not any(str(p) in opened for p in private)
    for path in private:
        try: path.read_bytes()
        except PermissionError: pass
        else: raise AssertionError('private open allowed')
    print(json.dumps({'worker':service.metadata['pid'],'actor_guarded':True}))
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.root)], capture_output=True,
                                text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["actor_guarded"])
        self.assertNotIn(self.secret, result.stdout + result.stderr)

    def test_wrong_problem_methods_types_and_unknown_keys_are_public_errors(self):
        service = self.service()
        with self.assertRaisesRegex(VerifierServiceError, "UNKNOWN_PROBLEM"):
            service.grade("not-registered", "Final answer: 8", "stop", "plain")
        for method, pid, params in (("items", self.actor.problem_id, {}),
                                    ("grade", self.actor.problem_id, {"text": 9, "finish_reason": "stop", "mode": "plain"}),
                                    ("grade", self.actor.problem_id, {"text": "8", "finish_reason": "stop", "mode": "plain", "gold": self.secret}),
                                    ("grade", True, {"text": "8", "finish_reason": "stop", "mode": "plain"}),
                                    ("grade", self.actor.problem_id, {"text": "8", "finish_reason": "eos", "mode": "plain"})):
            with self.subTest(method=method, params=params), self.assertRaisesRegex(VerifierServiceError, "INVALID_REQUEST") as caught:
                service._request(method, pid, params)
            self.assertNotIn(self.secret, str(caught.exception))
        self.assertEqual(service.grade(self.actor.problem_id, "Final answer: 7", "stop", "plain").status, "CORRECT")

    def test_fabricated_and_cross_problem_results_or_targets_rejected(self):
        service = self.service()
        fake = VerifierResult("INCORRECT", "8", 0.0, None, "strict-final-v1")
        with self.assertRaisesRegex(VerifierServiceError, "UNISSUED_RESULT"):
            service.cause(self.actor.problem_id, fake)
        with self.assertRaisesRegex(VerifierServiceError, "UNISSUED_TARGET"):
            service.grade_diagnosis(self.actor.problem_id, "anything", DiagTarget("FINALIZATION", ()))
        actual = service.grade(self.actor.problem_id, "No answer", "stop", "plain")
        with self.assertRaisesRegex(VerifierServiceError, "UNISSUED_RESULT"):
            service.route(self.actor.problem_id, [replace(actual, verifier_version="forged")], "verified-any-visual-v1")
        target = service.diagnosis_target(self.actor.problem_id, actual)
        self.assertEqual(target, DiagTarget("FINALIZATION", ()))
        self.assertEqual(service.grade_diagnosis(self.actor.problem_id,
            "Cause: FINALIZATION\nWrong item: none\nLesson: Finish the answer.", target), 1.0)

    def test_request_bounds_do_not_write_oversize_frame_or_desynchronize(self):
        service = self.service(max_request_bytes=256)
        with self.assertRaisesRegex(ValueError, "REQUEST_LIMIT"):
            service.grade(self.actor.problem_id, "x" * 1000, "stop", "plain")
        self.assertEqual(service._sequence, 0)
        self.assertEqual(service.grade(self.actor.problem_id, "Final answer: 8", "stop", "plain").status, "INCORRECT")
        # A first-turn CONTEXT_BLOCKED rollout legitimately has no graded results.
        route, evidence = service.route(self.actor.problem_id, [], "verified-any-visual-v1")
        self.assertEqual((route, evidence["observed_evidence"]), ("VISION_OFF", "UNKNOWN"))
        with self.assertRaisesRegex(ValueError, "RESULT_LIMIT"):
            service.route(self.actor.problem_id, [None] * 257, "verified-any-visual-v1")

    def test_timeout_kills_owned_process_and_disallows_fallback(self):
        service = self.service(timeout=5)
        service._timeout = 0.25
        os.kill(service._process.pid, signal.SIGSTOP)
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "TIMEOUT"):
            service.grade(self.actor.problem_id, "Final answer: 8", "stop", "plain")
        self.assertLess(time.monotonic() - started, 4)
        self.assertIsNotNone(service._process.poll())
        with self.assertRaisesRegex(VerifierServiceError, "CLOSED"):
            service.grade(self.actor.problem_id, "Final answer: 7", "stop", "plain")

    def test_missing_tampered_and_benchmark_vault_fail_startup_without_private_errors(self):
        alias = self.base / "vault-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        for root, forbidden in ((self.base / "missing", []), (self.root, [self.root]), (alias, [])):
            with self.subTest(root=root), self.assertRaisesRegex(VerifierServiceError, "VAULT_UNAVAILABLE") as caught:
                VerifierService(root, forbidden_roots=forbidden, timeout=5)
            self.assertNotIn(self.secret, str(caught.exception))
            self.assertNotIn(str(root), str(caught.exception))
        private = self.root / "verifier_private.jsonl"
        private.write_bytes(private.read_bytes() + b" ")
        with self.assertRaisesRegex(VerifierServiceError, "VAULT_UNAVAILABLE"):
            self.service()

    def raw_worker(self):
        process = subprocess.Popen([sys.executable, "-m", "fine_tuning.notebook_rl.verifier_service", "--worker",
                                    "--vault-root", str(self.root)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def cleanup():
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
        self.addCleanup(cleanup)
        def receive():
            length = struct.unpack("!I", process.stdout.read(4))[0]
            self.assertLessEqual(length, MAX_FRAME_BYTES)
            return json.loads(process.stdout.read(length))
        self.assertEqual(receive()["id"], 0)
        return process, receive

    def test_raw_protocol_duplicate_or_boolean_request_id_fails_closed(self):
        for request_id in (True, 0, 2):
            process, receive = self.raw_worker()
            raw = json.dumps({"id": request_id, "method": "grade", "problem_id": self.actor.problem_id,
                              "params": {"text": self.secret, "finish_reason": "stop", "mode": "plain"}}).encode()
            process.stdin.write(struct.pack("!I", len(raw)) + raw)
            process.stdin.flush()
            self.assertEqual(receive(), {"id": 0, "ok": False, "error": "INVALID_REQUEST"})
            process.wait(timeout=5)
            self.assertEqual(process.returncode, 3)
        process, receive = self.raw_worker()
        raw = b'{"id":1,"id":1}'
        process.stdin.write(struct.pack("!I", len(raw)) + raw)
        process.stdin.flush()
        process.wait(timeout=5)
        self.assertEqual(process.stdout.read(), b"")
        self.assertEqual(process.returncode, 3)

    def test_raw_protocol_oversize_frame_and_non_json_never_unpickled(self):
        for payload in (struct.pack("!I", MAX_FRAME_BYTES + 1), struct.pack("!I", 4) + b"\x80\x04N."):
            process, _ = self.raw_worker()
            process.stdin.write(payload)
            process.stdin.flush()
            process.wait(timeout=5)
            self.assertEqual(process.returncode, 3)
            self.assertEqual(process.stdout.read(), b"")
            self.assertNotIn(self.secret.encode(), process.stderr.read())

    def test_mismatched_response_id_closes_worker(self):
        service = self.service()
        original = service._response
        def mismatch(request_id, deadline):
            return original(request_id + 1, deadline)
        with patch.object(service, "_response", side_effect=mismatch):
            with self.assertRaisesRegex(VerifierServiceError, "INVALID_RESPONSE"):
                service.grade(self.actor.problem_id, "Final answer: 8", "stop", "plain")
        self.assertIsNotNone(service._process.poll())

    def test_no_pickle_no_global_start_method_change_and_metadata_copy(self):
        source = Path(__file__).parents[1] / "notebook_rl" / "verifier_service.py"
        code = source.read_text()
        self.assertNotIn("import pickle", code)
        self.assertNotIn("multiprocessing", code)
        service = self.service()
        metadata = service.metadata
        metadata["code_sha256"].clear()
        self.assertTrue(service.metadata["code_sha256"])
        self.assertEqual(service.metadata["code_sha256"][source.name], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertFalse(hasattr(service, "items"))

    def test_invalid_constructor_bounds_do_not_spawn(self):
        for kwargs in ({"timeout": True}, {"timeout": 0}, {"timeout": float("nan")},
                       {"max_request_bytes": MAX_FRAME_BYTES + 1}, {"max_request_bytes": True}):
            with self.subTest(kwargs=kwargs), patch("subprocess.Popen") as spawn, self.assertRaises(ValueError):
                VerifierService(self.root, forbidden_roots=[], **kwargs)
            spawn.assert_not_called()

    def test_fresh_interpreter_does_not_inherit_parent_module_state(self):
        # The worker's assert_no_teacher_imports would reject inherited sys.modules.
        with patch.dict(sys.modules, {"openai": object()}):
            service = self.service()
        self.assertEqual(service.grade(self.actor.problem_id, "Final answer: 8", "stop", "plain").status, "INCORRECT")
        self.assertEqual(service.metadata["vault_manifest_sha256"],
                         hashlib.sha256((self.root / "manifest.yaml").read_bytes()).hexdigest())

    def test_issued_results_are_bound_to_problem_not_only_content(self):
        actor, verifier, lineage, *_ = records()
        other_id = actor.problem_id + ":other"
        other_actor, other_verifier, other_lineage = (replace(actor, problem_id=other_id),
            replace(verifier, problem_id=other_id), replace(lineage, problem_id=other_id))
        self.root = self.base / "two-items"
        write_split(self.root, [actor, other_actor], [verifier, other_verifier], [lineage, other_lineage])
        service = self.service()
        actual = service.grade(actor.problem_id, "No final answer", "stop", "plain")
        target = service.diagnosis_target(actor.problem_id, actual)
        with self.assertRaisesRegex(VerifierServiceError, "UNISSUED_RESULT"):
            service.cause(other_id, actual)
        with self.assertRaisesRegex(VerifierServiceError, "UNISSUED_TARGET"):
            service.grade_diagnosis(other_id, "anything", target)

    def test_private_extra_field_or_extracted_not_in_actor_text_closes_client(self):
        for kind in ("extra", "extracted"):
            service = self.service()
            response = {"status": "CORRECT", "extracted": self.secret, "reward_answer": 1.0,
                        "slots": None, "verifier_version": "strict-final-v1"}
            if kind == "extra":
                response["gold"] = self.secret
            with patch.object(service, "_response", return_value=response), self.assertRaisesRegex(
                    VerifierServiceError, "INVALID_RESPONSE") as caught:
                service.grade(self.actor.problem_id, "Final answer: 8", "stop", "plain")
            self.assertNotIn(self.secret, str(caught.exception))
            self.assertIsNotNone(service._process.poll())


if __name__ == "__main__":
    unittest.main()
