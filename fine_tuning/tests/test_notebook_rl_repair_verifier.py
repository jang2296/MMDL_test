"""Authored repair responses exercise CPU verifier isolation, not model learning."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl.generators import circuit, graph
from fine_tuning.notebook_rl.generators.common import GeneratedProblem
from fine_tuning.notebook_rl.reasoning_trace import check_trace, validate_trace_report
from fine_tuning.notebook_rl.repair_contract import MAX_RAW_CHARS, VERSION, RepairContractError, parse_retry_sections
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import VERSION as GRADE_VERSION, Verifier
from fine_tuning.notebook_rl.verifier_service import VerifierService, VerifierServiceError
from fine_tuning.tests.test_notebook_rl_circuit_trace import fixture_trace as circuit_fixture, response as circuit_response
from fine_tuning.tests.test_notebook_rl_reasoning_trace import fixture_trace, response


def joint(body: str, *, newline: str = "\n") -> str:
    check = {"version": VERSION, "claims": [{"origin_ref": "parent/관측/🙂", "intent": "RECHECK_OBSERVATION",
              "rule": "DIRECTED_EDGE", "proposed_value":
              '과거 인용 🙂 Observed facts: edge[A,H] = 999; Solution: Trace: {}; Final answer: 777777'}],
             "uncertain_refs": []}
    return ("ErrorCheck: " + json.dumps(check, ensure_ascii=False)
            + newline + "RevisedSolution:" + newline + body)


class RepairVerifierTests(unittest.TestCase):
    graphs: list[GeneratedProblem]
    circuit: GeneratedProblem

    @classmethod
    def setUpClass(cls) -> None:
        cls.graphs = [graph.generate(index, 3407, "train", "obs") for index in range(4)]
        cls.circuit = circuit.generate(2, 20261021, "sealed", "obs")

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "vault"
        self.secret = "PRIVATE-ANNOTATION-DO-NOT-RETURN-54821"
        problems = [*self.graphs, self.circuit]
        write_split(self.root, [p.actor for p in problems],
                    [replace(p.verifier, printed_text=(self.secret,)) for p in problems],
                    [p.lineage for p in problems])
        self.verifier = Verifier(self.root)

    def service(self) -> VerifierService:
        service = VerifierService(self.root, forbidden_roots=[self.root.parent / "benchmark"])
        self.addCleanup(service.close)
        return service

    def test_fresh_service_projects_graph_tree_dfa_and_circuit_with_raw_offsets(self) -> None:
        service = self.service()
        self.assertNotEqual(service.metadata["pid"], os.getpid())
        codec = Path(__file__).parents[1] / "notebook_rl" / "repair_contract.py"
        self.assertEqual(service.metadata["code_sha256"]["repair_contract.py"],
                         hashlib.sha256(codec.read_bytes()).hexdigest())
        for problem in [*self.graphs, self.circuit]:
            is_circuit = problem is self.circuit
            body = (circuit_response(problem, circuit_fixture(problem)) if is_circuit
                    else response(problem, fixture_trace(problem))).replace("\n", "\r\n")
            raw, pid = joint(body, newline="\r\n"), problem.actor.problem_id
            sections = parse_retry_sections(raw, contract_version=VERSION)
            legacy = self.verifier.trace(pid, body, "stop")
            with self.subTest(source=problem.actor.source, family=problem.actor.family):
                result = service.grade(pid, raw, "stop", "obs", response_contract=VERSION)
                expected = replace(self.verifier.grade(pid, body, "stop", "obs"),
                                   verifier_version=GRADE_VERSION + "+" + VERSION)
                self.assertEqual(result, expected)
                self.assertEqual(result.status, "CORRECT")
                report = service.trace(pid, raw, "stop", response_contract=VERSION)
                self.assertEqual(report, self.verifier.trace(pid, raw, "stop", response_contract=VERSION))
                validate_trace_report(report, raw, "stop", response_contract=VERSION)
                self.assertEqual(report["status"], "COMPLIANT")
                self.assertEqual(report["version"], legacy["version"] + "+" + VERSION)
                self.assertEqual(report["response_contract"], VERSION)
                self.assertEqual(report["text_sha256"], hashlib.sha256(raw.encode()).hexdigest())
                self.assertEqual(report["solution_projection"], {
                    "span": [sections.solution_span.start, len(raw)],
                    "sha256": hashlib.sha256(body.encode()).hexdigest()})
                self.assertEqual(report["trace"]["line"], legacy["trace"]["line"])
                self.assertEqual(report["trace"]["span"],
                                 [v + sections.solution_span.start for v in legacy["trace"]["span"]])
                start, end = report["trace"]["span"]
                self.assertEqual(raw[start:end], report["trace"]["line"])
                for row in report["checks"]:
                    for ref in row["observation_refs"]:
                        start, end = ref["span"]
                        self.assertGreaterEqual(start, sections.solution_span.start)
                        self.assertEqual(ref["sha256"], hashlib.sha256(raw[start:end].encode()).hexdigest())
                self.assertNotIn(self.secret, json.dumps(report))

    def test_violated_checks_and_aliased_issues_shift_once_without_changing_input(self) -> None:
        problem = self.graphs[0]
        trace = fixture_trace(problem)
        trace["weights"][0] += 1
        body = response(problem, trace)
        raw, pid = joint(body), problem.actor.problem_id
        legacy = self.verifier.trace(pid, body, "stop")
        self.assertTrue(any(row["observation_refs"] for row in legacy["issues"]))
        original = copy.deepcopy(legacy)
        with patch("fine_tuning.notebook_rl.reasoning_trace.check_trace", return_value=legacy):
            report = check_trace(actor=problem.actor, item=problem.verifier, text=raw,
                                 finish_reason="stop", response_contract=VERSION)
        self.assertEqual(self.service().trace(pid, raw, "stop", response_contract=VERSION), report)
        self.assertEqual(report["status"], "VIOLATED")
        self.assertEqual(legacy, original)
        offset = len(raw) - len(body)
        for field in ("checks", "issues"):
            for actual, expected in zip(report[field], legacy[field], strict=True):
                self.assertEqual(actual["status"], expected["status"])
                self.assertEqual(actual["blocked_by"], expected["blocked_by"])
                for ref, old in zip(actual["observation_refs"], expected["observation_refs"], strict=True):
                    self.assertEqual(ref["span"], [v + offset for v in old["span"]])
        before = copy.deepcopy(report)
        validate_trace_report(report, raw, "stop", response_contract=VERSION)
        self.assertEqual(report, before)
        malformed_values: tuple[object, ...] = ({}, "", ())
        for field in ("checks", "issues"):
            for malformed in malformed_values:
                forged = copy.deepcopy(report)
                forged[field][0]["observation_refs"] = malformed
                with self.subTest(field=field, refs=type(malformed)), self.assertRaises(ValueError):
                    validate_trace_report(forged, raw, "stop", response_contract=VERSION)

    def test_quoted_old_answer_facts_and_trace_never_supply_missing_revised_content(self) -> None:
        service, problem = self.service(), self.graphs[0]
        pid = problem.actor.problem_id
        for body in ("Solution:\nNo structured trace supplied.", "Solution:\nFinal answer: 999"):
            with self.subTest(body=body):
                raw = joint(body)
                result = service.grade(pid, raw, "stop", "obs", response_contract=VERSION)
                self.assertEqual(result.slots.parsed if result.slots else None, 0)
                self.assertEqual(result.status, "INVALID_NO_FINAL" if "999" not in body else "INCORRECT")
                self.assertNotEqual(result.extracted, "777777")
                report = service.trace(pid, raw, "stop", response_contract=VERSION)
                self.assertIsNone(report["trace"])
                self.assertEqual(report["status"], "UNVERIFIABLE")

    def test_length_finish_keeps_native_violation_but_never_rewards_final_answer(self) -> None:
        problem = self.graphs[0]
        trace = fixture_trace(problem)
        trace["weights"][0] += 1
        raw, pid = joint(response(problem, trace)), problem.actor.problem_id
        service = self.service()
        result = service.grade(pid, raw, "length", "obs", response_contract=VERSION)
        self.assertEqual((result.status, result.reward_answer, result.extracted), ("TRUNCATED", 0, None))
        report = service.trace(pid, raw, "length", response_contract=VERSION)
        self.assertEqual(report["status"], "VIOLATED")
        self.assertEqual(report["finish_reason"], "length")
        with self.assertRaises(ValueError):
            validate_trace_report(report, raw, "stop", response_contract=VERSION)

    def test_malformed_and_wrong_contract_requests_are_bounded_and_worker_survives(self) -> None:
        service, problem = self.service(), self.graphs[0]
        body, pid = response(problem, fixture_trace(problem)), problem.actor.problem_id
        raw = joint(body)
        invalid = [body, "prefix\n" + raw, raw[:raw.index("RevisedSolution:")],
                   'ErrorCheck: {"version":', raw.replace("RevisedSolution:\n", "RevisedSolution:", 1),
                   "x" * (MAX_RAW_CHARS + 1), "ErrorCheck: " + self.secret]
        for text in invalid:
            for method in ("grade", "trace"):
                params = {"text": text, "finish_reason": "stop", "response_contract": VERSION}
                if method == "grade":
                    params["mode"] = "obs"
                with self.subTest(method=method, length=len(text)), self.assertRaisesRegex(
                        VerifierServiceError, "INVALID_RESPONSE_CONTRACT") as caught:
                    service._request(method, pid, params)
                self.assertNotIn(self.secret, str(caught.exception))
        contracts: tuple[object, ...] = ("repair-check-v0", "", True, {})
        for contract in contracts:
            for method in ("grade", "trace"):
                params = {"text": raw, "finish_reason": "stop", "response_contract": contract}
                if method == "grade":
                    params["mode"] = "obs"
                with self.subTest(method=method, contract=contract), self.assertRaisesRegex(
                        VerifierServiceError, "INVALID_RESPONSE_CONTRACT"):
                    service._request(method, pid, params)
        with self.assertRaises(RepairContractError):
            self.verifier.grade(pid, body, "stop", "obs", response_contract=VERSION)
        self.assertIsNone(service._process.poll())
        self.assertEqual(service.grade(pid, raw, "stop", "obs", response_contract=VERSION).status, "CORRECT")

    def test_explicit_opt_in_preserves_legacy_wire_and_rejects_cross_contract_reports(self) -> None:
        service, problem = self.service(), self.graphs[0]
        pid, body = problem.actor.problem_id, response(problem, fixture_trace(problem))
        expected_grade = self.verifier.grade(pid, body, "stop", "obs")
        expected_trace = self.verifier.trace(pid, body, "stop")
        with patch("fine_tuning.notebook_rl.verifier.parse_retry_sections", side_effect=AssertionError), \
                patch("fine_tuning.notebook_rl.reasoning_trace.parse_retry_sections", side_effect=AssertionError):
            self.assertEqual(to_json(self.verifier.grade(pid, body, "stop", "obs")), to_json(expected_grade))
            self.assertEqual(self.verifier.trace(pid, body, "stop"), expected_trace)
        self.assertEqual(service.grade(pid, body, "stop", "obs"), expected_grade)
        self.assertEqual(service.trace(pid, body, "stop"), expected_trace)
        raw = joint(body)
        legacy = service.trace(pid, raw, "stop")
        self.assertEqual(legacy["status"], "UNVERIFIABLE")
        self.assertNotIn("response_contract", legacy)
        report = service.trace(pid, raw, "stop", response_contract=VERSION)
        for contract in (None, "repair-check-v0"):
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                validate_trace_report(report, raw, "stop", response_contract=contract)
        with self.assertRaises(ValueError):
            validate_trace_report(legacy, raw, "stop", response_contract=VERSION)

    def test_projection_report_rejects_hash_offset_bool_contract_and_private_forgery(self) -> None:
        for problem in (self.graphs[0], self.circuit):
            body = (circuit_response(problem, circuit_fixture(problem)) if problem is self.circuit
                    else response(problem, fixture_trace(problem)))
            raw, pid = joint(body), problem.actor.problem_id
            report = self.verifier.trace(pid, raw, "stop", response_contract=VERSION)
            for field, value in (("text_sha256", hashlib.sha256(body.encode()).hexdigest()),
                                 ("response_contract", None), ("version", report["version"].split("+")[0]),
                                 ("gold", self.secret)):
                forged = copy.deepcopy(report)
                forged[field] = value
                with self.subTest(family=problem.actor.family, field=field), self.assertRaises(ValueError):
                    validate_trace_report(forged, raw, "stop", response_contract=VERSION)
            for mutation in ("shift", "bool", "hash", "projection_shift", "projection_bool", "projection_hash",
                             "projection_private", "line", "raw_drift", "oversize"):
                forged = copy.deepcopy(report)
                text = raw
                if mutation == "shift":
                    forged["trace"]["span"][0] += 1
                elif mutation == "bool":
                    forged["trace"]["span"][0] = True
                elif mutation == "hash":
                    forged["trace"]["sha256"] = "0" * 64
                elif mutation == "projection_shift":
                    forged["solution_projection"]["span"][0] += 1
                elif mutation == "projection_bool":
                    forged["solution_projection"]["span"][0] = True
                elif mutation == "projection_hash":
                    forged["solution_projection"]["sha256"] = report["text_sha256"]
                elif mutation == "projection_private":
                    forged["solution_projection"]["expected"] = self.secret
                elif mutation == "line":
                    forged["trace"]["line"] = forged["trace"]["span"]
                elif mutation == "raw_drift":
                    text = raw.replace("과거", "현재", 1)
                else:
                    forged["semantic_scope"] = "x" * 65536
                with self.subTest(family=problem.actor.family, mutation=mutation), self.assertRaises(ValueError):
                    validate_trace_report(forged, text, "stop", response_contract=VERSION)
            for mutation in ("shift", "bool", "private"):
                if problem is self.circuit:
                    continue
                forged = copy.deepcopy(report)
                ref = next(row["observation_refs"][0] for row in forged["checks"] if row["observation_refs"])
                if mutation == "shift":
                    ref["span"][0] -= 1
                elif mutation == "bool":
                    ref["span"][0] = False
                else:
                    ref["expected"] = self.secret
                with self.subTest(ref=mutation), self.assertRaises(ValueError):
                    validate_trace_report(forged, raw, "stop", response_contract=VERSION)

    def test_client_closes_on_suffix_external_answer_old_version_or_private_report(self) -> None:
        problem = self.graphs[0]
        raw, pid = joint(response(problem, fixture_trace(problem))), problem.actor.problem_id
        for mutation in ("answer", "grade_version", "private_trace", "trace_offset"):
            service = self.service()
            if mutation in ("answer", "grade_version"):
                result = self.verifier.grade(pid, raw, "stop", "obs", response_contract=VERSION)
                forged = json.loads(to_json(result))
                forged["extracted" if mutation == "answer" else "verifier_version"] = (
                    "777777" if mutation == "answer" else GRADE_VERSION)
                with self.subTest(mutation=mutation), patch.object(service, "_response", return_value=forged), \
                        self.assertRaisesRegex(VerifierServiceError, "INVALID_RESPONSE"):
                    service.grade(pid, raw, "stop", "obs", response_contract=VERSION)
            else:
                forged = self.verifier.trace(pid, raw, "stop", response_contract=VERSION)
                if mutation == "private_trace":
                    forged["expected"] = self.secret
                else:
                    forged["trace"]["span"][0] += 1
                with self.subTest(mutation=mutation), patch.object(service, "_response", return_value=forged), \
                        self.assertRaisesRegex(VerifierServiceError, "INVALID_RESPONSE"):
                    service.trace(pid, raw, "stop", response_contract=VERSION)
            self.assertIsNotNone(service._process.poll())


if __name__ == "__main__":
    unittest.main()
