"""Program-authored vault fixtures and real CPU verifier children; no model/GPU."""
import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl.generators import graph, graph_pair
from fine_tuning.notebook_rl.reasoning_trace import validate_trace_report
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.notebook_rl.verifier_service import (
    VERSION, VerifierService, VerifierServiceError, _dispatch, _Issued,
)
from fine_tuning.tests.test_notebook_rl_schema import records


class TraceServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = [graph.generate(index, 3407, "train", "obs") for index in range(4)]
        cls.problems.append(graph_pair.generate(0, 3407, "train", "obs"))

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "vault"
        actor, item, lineage, *_ = records()
        self.secret = "PRIVATE-ANNOTATION-DO-NOT-RETURN-69270"
        item = replace(item, oracle_fact_lines=(self.secret,), accept={"annotation": self.secret})
        self.unsupported = actor
        write_split(self.root, [p.actor for p in self.problems] + [actor],
                    [p.verifier for p in self.problems] + [item],
                    [p.lineage for p in self.problems] + [lineage])
        self.verifier = Verifier(self.root)

    def service(self):
        service = VerifierService(self.root, forbidden_roots=[self.base / "benchmark"])
        self.addCleanup(service.close)
        return service

    def bfs_text(self):
        problem = self.problems[1]
        world = problem.verifier.accept["graph_world"]
        pending, seen, steps = ["A"], {"A"}, []
        while pending:
            node = pending.pop(0)
            new = sorted(b for a, b, _ in world["edges"] if a == node and b not in seen)
            steps.append({"node": node, "new": new})
            pending.extend(new)
            seen.update(new)
        trace = {"kind": "bfs", "steps": steps}
        return ("Observed facts:\r\n" + "\r\n".join(problem.verifier.oracle_fact_lines)
                + "\r\nSolution:\r\nTrace: " + json.dumps(trace)
                + "\r\nFinal answer: " + problem.verifier.gold + "\r\n")

    def failed_bfs_text(self):
        text = self.bfs_text()
        line = next(line for line in text.splitlines() if line.startswith("Trace: "))
        trace = json.loads(line[len("Trace: "):])
        trace["steps"][0]["node"] = "H"
        return text.replace(line, "Trace: " + json.dumps(trace), 1), len(trace["steps"])

    def dfa_text(self, states):
        trace = {"kind": "dfa", "states": states, "decision": "reject"}
        return "Solution: \t\r\n\tTrace:\t" + json.dumps(trace) + " \t\r\nFinal answer: reject"

    def test_isolated_native_trace_matches_direct_and_pins_sources(self):
        service = self.service()
        self.assertNotEqual(service.metadata["pid"], os.getpid())
        self.assertEqual(service.metadata["version"], VERSION)
        here = Path(__file__).resolve().parents[1] / "notebook_rl"
        for source in ("reasoning_trace.py", "generators/graph.py", "generators/graph_pair.py"):
            self.assertEqual(service.metadata["code_sha256"][source],
                             hashlib.sha256((here / source).read_bytes()).hexdigest())
        text = self.bfs_text()
        pid = self.problems[1].actor.problem_id
        expected = self.verifier.trace(pid, text, "stop")
        self.assertEqual(expected["status"], "COMPLIANT")
        actual = service.trace(pid, text, "stop")
        self.assertEqual(actual, expected)
        validate_trace_report(actual, text, "stop")
        self.assertTrue(any(row["observation_refs"] for row in actual["checks"]))
        self.assertEqual(actual["text_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(actual["finish_reason"], "stop")
        for problem in self.problems:
            with self.subTest(source=problem.actor.source, question=problem.actor.question):
                raw = "Solution:\nNo structured trace supplied.\nFinal answer: 999"
                self.assertEqual(service.trace(problem.actor.problem_id, raw, "stop"),
                                 self.verifier.trace(problem.actor.problem_id, raw, "stop"))

    def test_positive_trace_violation_and_length_binding_survive_rpc(self):
        service = self.service()
        pid = self.problems[0].actor.problem_id
        text = ('Solution:\nTrace: {"kind":"shortest","nodes":["A","H"],'
                '"edge_ids":null,"weights":[1],"cost":1}\nFinal answer: 1')
        for finish in ("stop", "length"):
            expected = self.verifier.trace(pid, text, finish)
            actual = service.trace(pid, text, finish)
            self.assertEqual(actual, expected)
            self.assertEqual(actual["status"], "VIOLATED")
            self.assertEqual(actual["finish_reason"], finish)

    def test_failed_bfs_suffix_inventory_and_blockers_survive_rpc(self):
        service = self.service()
        text, count = self.failed_bfs_text()
        pid = self.problems[1].actor.problem_id
        report = service.trace(pid, text, "stop")
        self.assertEqual(report, self.verifier.trace(pid, text, "stop"))
        self.assertEqual(report["version"], "native-reasoning-trace-v4")
        self.assertEqual(report["status"], "VIOLATED")
        self.assertFalse(report["coverage"]["complete"])
        rows = [row for row in report["checks"] if row["rule"].startswith("BFS_")]
        self.assertEqual({(row["rule"], row["locus"]) for row in rows},
                         {("BFS_COMPLETENESS", "steps")} | {
                             (rule, f"steps[{index}].{field}") for index in range(count)
                             for rule, field in (("BFS_FIFO", "node"), ("BFS_NEW_NEIGHBORS", "new"))})
        blocker = {"rule": "BFS_FIFO", "locus": "steps[0].node"}
        self.assertEqual(rows[0]["status"], "VIOLATED")
        self.assertEqual(rows[0]["blocked_by"], [])
        for row in rows[1:]:
            self.assertEqual(row["status"], "UNVERIFIABLE")
            self.assertEqual(row["blocked_by"], [blocker])

    def test_dfa_acceptance_dependencies_survive_rpc_without_private_gold_or_replacement(self):
        service = self.service()
        problem = self.problems[2]
        pid = problem.actor.problem_id
        for states in (["q0"] * 4, ["q0"] * 4 + ["q9"]):
            text = self.dfa_text(states)
            expected = self.verifier.trace(pid, text, "stop")
            report = service.trace(pid, text, "stop")
            self.assertEqual(report, expected)
            with patch.dict(self.verifier.items, {pid: replace(problem.verifier, gold=self.secret)}):
                self.assertEqual(report, self.verifier.trace(pid, text, "stop"))
            validate_trace_report(report, text, "stop")
            cause = ({"rule": "DFA_COMPLETE_WORD", "locus": "states"} if len(states) != 5
                     else {"rule": "DFA_TRANSITION", "locus": "states[3:5]"})
            self.assertEqual((report["checks"][-1]["status"], report["checks"][-1]["blocked_by"]),
                             ("UNVERIFIABLE", [cause]))
            self.assertEqual(report["status"], "VIOLATED")
            self.assertFalse(report["coverage"]["complete"])
            start, end = report["trace"]["span"]
            self.assertEqual(report["trace"]["line"], text[start:end])
            self.assertTrue(report["trace"]["line"].startswith("\tTrace:\t"))
            self.assertTrue(report["trace"]["line"].endswith(" \t"))
            for key in ("gold", "expected", "reference_trace", "replacement_trace", "graph_world"):
                self.assertNotIn('"' + key + '"', json.dumps(report))
            self.assertNotIn(self.secret, json.dumps(report))

    def test_forged_dfa_acceptance_dependencies_close_client(self):
        pid = self.problems[2].actor.problem_id
        for states in (["q9"] * 4, ["q9", "q9", "q0", "q0", "q9"]):
            text = self.dfa_text(states)
            report = self.verifier.trace(pid, text, "stop")
            for mutation in ("empty", "unrelated", "unproven"):
                service = self.service()
                forged = copy.deepcopy(report)
                acceptance = forged["checks"][-1]
                if mutation == "empty":
                    acceptance["blocked_by"] = []
                elif mutation == "unrelated":
                    acceptance["blocked_by"] = [{"rule": "DFA_START", "locus": "states[0]"}]
                else:
                    cause = acceptance["blocked_by"][0]
                    next(row for row in forged["checks"] if row["rule"] == cause["rule"]
                         and row["locus"] == cause["locus"])["status"] = "COMPLIANT"
                    forged["issues"] = [row for row in forged["checks"] if row["status"] != "COMPLIANT"]
                with self.subTest(states=states, mutation=mutation), \
                        patch.object(service, "_response", return_value=forged), \
                        self.assertRaisesRegex(VerifierServiceError, "INVALID_RESPONSE"):
                    service.trace(pid, text, "stop")
                self.assertIsNotNone(service._process.poll())

    def test_forged_blocked_references_close_client(self):
        text, _ = self.failed_bfs_text()
        pid = self.problems[1].actor.problem_id
        report = self.verifier.trace(pid, text, "stop")
        for kind in ("missing", "empty", "unknown", "forward", "compliant", "self", "duplicate", "private",
                     "drop_suffix", "repeat_suffix"):
            service = self.service()
            forged = copy.deepcopy(report)
            row = next(row for row in forged["checks"] if row["blocked_by"])
            if kind == "missing":
                del row["blocked_by"]
            elif kind == "empty":
                row["blocked_by"] = []
            elif kind == "unknown":
                row["blocked_by"] = [{"rule": "BFS_FIFO", "locus": "steps[99].node"}]
            elif kind == "forward":
                row["blocked_by"] = [{"rule": "BFS_COMPLETENESS", "locus": "steps"}]
            elif kind == "compliant":
                row["blocked_by"] = [{"rule": "RESPONSE_SCOPE", "locus": "response"}]
            elif kind == "self":
                row["blocked_by"] = [{"rule": row["rule"], "locus": row["locus"]}]
            elif kind == "duplicate":
                row["blocked_by"] *= 2
            elif kind == "private":
                row["blocked_by"][0]["expected"] = self.secret
            else:
                index = forged["checks"].index(row)
                if kind == "drop_suffix":
                    forged["checks"].pop(index)
                else:
                    forged["checks"].insert(index, copy.deepcopy(row))
                forged["issues"] = [entry for entry in forged["checks"] if entry["status"] != "COMPLIANT"]
            with self.subTest(kind=kind), patch.object(service, "_response", return_value=forged), \
                    self.assertRaisesRegex(VerifierServiceError, "INVALID_RESPONSE") as caught:
                service.trace(pid, text, "stop")
            self.assertNotIn(self.secret, str(caught.exception))
            self.assertIsNotNone(service._process.poll())

    def test_response_scope_cannot_be_laundered_into_full_coverage(self):
        pid = self.problems[1].actor.problem_id
        for kind in ("prefix", "suffix", "boolean"):
            service = self.service()
            text = self.bfs_text()
            if kind == "prefix":
                text = "Unverified claim before the response.\r\n" + text
            elif kind == "suffix":
                text += "Unverified claim after the final answer.\r\n"
            report = self.verifier.trace(pid, text, "stop")
            forged = copy.deepcopy(report)
            if kind == "boolean":
                forged["checks"][0]["status"] = True
            else:
                row = next(row for row in forged["checks"] if row["rule"] == "RESPONSE_SCOPE")
                self.assertEqual(row["status"], "UNVERIFIABLE")
                row["status"] = "COMPLIANT"
                forged["issues"] = [row for row in forged["checks"] if row["status"] != "COMPLIANT"]
                forged["status"] = "COMPLIANT"
                forged["coverage"].update(
                    checked=sorted({row["rule"] for row in forged["checks"]}), unverified=[], complete=True)
            with self.subTest(kind=kind), patch.object(service, "_response", return_value=forged), \
                    self.assertRaisesRegex(VerifierServiceError, "INVALID_RESPONSE"):
                service.trace(pid, text, "stop")
            self.assertIsNotNone(service._process.poll())

    def test_constructor_uses_trusted_same_vault_actor_and_legacy_grade_needs_no_actor(self):
        problem = self.problems[1]
        self.assertEqual(self.verifier.actors[problem.actor.problem_id], problem.actor)
        with patch("fine_tuning.notebook_rl.reasoning_trace.check_trace", return_value={}) as check:
            self.verifier.trace(problem.actor.problem_id, "raw", "length")
        check.assert_called_once_with(actor=problem.actor, item=problem.verifier,
                                      text="raw", finish_reason="length")
        legacy = Verifier.__new__(Verifier)
        legacy.items = self.verifier.items
        result = legacy.grade(self.unsupported.problem_id, "Final answer: 7", "stop", "plain")
        self.assertEqual(result.status, "CORRECT")

    def test_trace_rejects_forged_question_source_traversal_and_extra_parameters(self):
        service = self.service()
        pid, text = self.problems[1].actor.problem_id, self.bfs_text()
        for key in ("actor", "item", "question", "source", "traversal", "expected", "gold"):
            with self.subTest(key=key), self.assertRaisesRegex(VerifierServiceError, "INVALID_REQUEST"):
                service._request("trace", pid, {"text": text, "finish_reason": "stop", key: self.secret})
        for args in ({"text": text}, {"text": 9, "finish_reason": "stop"},
                     {"text": text, "finish_reason": "eos"}, {"text": text, "finish_reason": True}):
            with self.subTest(args=args), self.assertRaisesRegex(VerifierServiceError, "INVALID_REQUEST"):
                service._request("trace", pid, args)
        with self.assertRaisesRegex(VerifierServiceError, "UNKNOWN_PROBLEM"):
            service.trace("not-registered", text, "stop")
        self.assertEqual(service.trace(pid, text, "stop")["status"], "COMPLIANT")

    def test_public_report_excludes_private_annotations_and_expected_values(self):
        service = self.service()
        report = service.trace(self.unsupported.problem_id, "Solution:\nFinal answer: 8", "stop")
        self.assertEqual(report["status"], "UNVERIFIABLE")
        public = json.dumps([service.metadata, report], allow_nan=False)
        self.assertNotIn(self.secret, public)
        for key in ("gold", "expected", "graph_world", "oracle_fact_lines", "printed_text",
                    "accept", "reference_trace", "teacher_trace"):
            self.assertNotIn('"' + key + '"', public)
        self.assertEqual(service.grade(self.unsupported.problem_id, "Final answer: 7", "stop", "plain").status,
                         "CORRECT")

    def test_malformed_or_wrong_raw_bound_response_closes_client(self):
        text, pid = self.bfs_text(), self.problems[1].actor.problem_id
        report = self.verifier.trace(pid, text, "stop")
        for kind in ("private", "locus", "missing_trace", "hash", "finish", "observation", "nonfinite", "type"):
            service = self.service()
            forged = copy.deepcopy(report)
            if kind == "private":
                forged["gold"] = self.secret
            elif kind == "locus":
                forged["checks"][0]["locus"] = self.secret
            elif kind == "missing_trace":
                forged["trace"] = None
            elif kind == "hash":
                forged["text_sha256"] = "0" * 64
            elif kind == "finish":
                forged["finish_reason"] = "length"
            elif kind == "observation":
                ref = next(row["observation_refs"][0] for row in forged["checks"] if row["observation_refs"])
                ref["span"][0] += 1
            elif kind == "nonfinite":
                forged["coverage"]["kind"] = float("nan")
            else:
                forged = [forged]
            with self.subTest(kind=kind), patch.object(service, "_response", return_value=forged), \
                    self.assertRaisesRegex(VerifierServiceError, "INVALID_RESPONSE") as caught:
                service.trace(pid, text, "stop")
            self.assertNotIn(self.secret, str(caught.exception))
            self.assertIsNotNone(service._process.poll())
            with self.assertRaisesRegex(VerifierServiceError, "CLOSED"):
                service.trace(pid, text, "stop")

    def test_worker_dispatch_validates_report_before_serializing(self):
        text, pid = self.bfs_text(), self.problems[1].actor.problem_id
        forged = self.verifier.trace(pid, text, "stop") | {"expected": self.secret}
        request = {"id": 1, "method": "trace", "problem_id": pid,
                   "params": {"text": text, "finish_reason": "stop"}}
        with patch.object(self.verifier, "trace", return_value=forged), self.assertRaises(ValueError) as caught:
            _dispatch(self.verifier, _Issued(), request)
        self.assertNotIn(self.secret, str(caught.exception))

    def test_trace_rpc_preserves_actor_vault_benchmark_and_teacher_guards(self):
        script = """
import importlib, json, pathlib, sys
from fine_tuning.notebook_rl.guards import install_guards
from fine_tuning.notebook_rl.verifier_service import VerifierService
root = pathlib.Path(sys.argv[1])
private = [root/'verifier_private.jsonl', root/'lineage.jsonl']
benchmark = root.parent/'benchmark'
install_guards(forbidden_roots=[benchmark], vault_files=private,
               allow_vault=False, allow_benchmark=False)
with VerifierService(root, forbidden_roots=[benchmark]) as service:
    report = service.trace(sys.argv[2], 'Solution:\\nFinal answer: 8', 'stop')
    assert report['status'] == 'UNVERIFIABLE'
    for path in [*private, benchmark/'forbidden.txt']:
        try: path.read_bytes()
        except PermissionError: pass
        else: raise AssertionError('private or benchmark read allowed')
    for module in ('openai', 'anthropic'):
        try: importlib.import_module(module)
        except PermissionError: pass
        else: raise AssertionError('teacher import allowed')
    print(json.dumps({'guarded': True}))
"""
        benchmark = self.base / "benchmark"
        benchmark.mkdir()
        (benchmark / "forbidden.txt").write_text(self.secret)
        result = subprocess.run([sys.executable, "-B", "-c", script, str(self.root), self.unsupported.problem_id],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"guarded": True})
        self.assertNotIn(self.secret, result.stdout + result.stderr)

    def test_worker_trusted_imports_precede_guards_without_weakening_them(self):
        script = """
import importlib, pathlib, sys
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.notebook_rl.verifier_service import _worker
root = pathlib.Path(sys.argv[1])
benchmark = root.parent/'benchmark'
original = Verifier.trace
def trace(self, *args):
    try: importlib.import_module('openai')
    except PermissionError: pass
    else: raise AssertionError('teacher import allowed')
    try: (benchmark/'forbidden.txt').read_bytes()
    except PermissionError: pass
    else: raise AssertionError('benchmark read allowed')
    return original(self, *args)
Verifier.trace = trace
raise SystemExit(_worker(root, [benchmark]))
"""
        benchmark = self.base / "benchmark"
        benchmark.mkdir()
        (benchmark / "forbidden.txt").write_text(self.secret)
        pid, text = self.problems[1].actor.problem_id, self.bfs_text()
        request = json.dumps({"id": 1, "method": "trace", "problem_id": pid,
                              "params": {"text": text, "finish_reason": "stop"}}).encode()
        result = subprocess.run([sys.executable, "-B", "-c", script, str(self.root)],
                                input=struct.pack("!I", len(request)) + request,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        frames, raw = [], result.stdout
        while raw:
            length = struct.unpack("!I", raw[:4])[0]
            frames.append(json.loads(raw[4:4 + length]))
            raw = raw[4 + length:]
        self.assertEqual([frame["id"] for frame in frames], [0, 1])
        self.assertEqual(frames[1], {"id": 1, "ok": True,
                                   "result": self.verifier.trace(pid, text, "stop")})
        self.assertNotIn(self.secret.encode(), result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
