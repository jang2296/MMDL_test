"""Authored native samples verify repair audit plumbing, never model learning."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl.reasoning_trace import _finish
from fine_tuning.notebook_rl.generators import circuit, graph
from fine_tuning.notebook_rl.generators.common import GeneratedProblem
from fine_tuning.notebook_rl.repair_contract import MAX_RAW_CHARS, VERSION, RepairContractError, parse_retry_sections
from fine_tuning.notebook_rl.schema import Sample, VerifierResult, to_json
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.notebook_rl.verifier_service import VerifierService
from fine_tuning.tests.test_notebook_rl_circuit_trace import fixture_trace as circuit_fixture, response as circuit_response
from fine_tuning.tests.test_notebook_rl_reasoning_trace import fixture_trace, response
from fine_tuning.tests.test_notebook_rl_repair_verifier import joint
from fine_tuning.tests.test_notebook_rl_schema import records


class RepairNativeTests(unittest.TestCase):
    problem: GeneratedProblem
    circuit: GeneratedProblem

    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = graph.generate(0, 3407, "train", "obs")
        cls.circuit = circuit.generate(2, 20261021, "sealed", "obs")

    def setUp(self) -> None:
        self.verifier = Verifier.__new__(Verifier)
        self.verifier.actors = {p.actor.problem_id: p.actor for p in (self.problem, self.circuit)}
        self.verifier.items = {p.actor.problem_id: p.verifier for p in (self.problem, self.circuit)}
        self.body = response(self.problem, fixture_trace(self.problem))

    def sample(self, text: str, *, problem: GeneratedProblem | None = None, finish: str = "stop") -> Sample:
        problem = self.problem if problem is None else problem
        return Sample("actual-child", problem.actor.problem_id, "T2", "T2", 2, "actual-parent",
                      [1, 2], [3], [-.1], finish, text, 3407, 0)

    def test_default_helpers_and_audit_keep_legacy_outputs_without_trusting_cached_result(self) -> None:
        sample = self.sample(self.body)
        sample.result = VerifierResult("INCORRECT", "777777", 0., None, "untrusted-cache")
        before = to_json(sample)
        report = self.verifier.trace(sample.problem_id, sample.text, sample.finish_reason)
        checks = synthesis._merge_checks(synthesis._observation_checks(sample, self.verifier)
                                         + synthesis._trace_checks(sample, report))
        expected_result = self.verifier.grade(sample.problem_id, sample.text, "stop", self.problem.actor.mode)
        with patch("fine_tuning.notebook_rl.repair_contract.parse_retry_sections", side_effect=AssertionError):
            audit = synthesis.audit_native_sample(sample, self.problem.actor, self.verifier)
            self.assertEqual(synthesis._trace_checks(sample, report, response_contract=None),
                             synthesis._trace_checks(sample, report))
            self.assertEqual(synthesis._observation_checks(sample, self.verifier, response_contract=None),
                             synthesis._observation_checks(sample, self.verifier))
            self.assertEqual(synthesis._output_checks(sample, response_contract=None), synthesis._output_checks(sample))
        self.assertEqual(set(audit), {"version", "sample_id", "problem_id", "response_contract", "raw_sha256",
                         "finish_reason", "solution_projection", "result", "trace_report", "native_trace",
                         "checks", "issues", "unverified", "output_checks"})
        self.assertEqual(audit["version"], synthesis.NATIVE_SAMPLE_AUDIT_VERSION)
        self.assertEqual(audit["result"], json.loads(to_json(expected_result)))
        self.assertEqual(audit["checks"], checks)
        self.assertEqual(audit["issues"], [row for row in checks if row["status"] == "VIOLATED"])
        self.assertEqual(audit["unverified"], [row for row in checks if row["status"] == "UNVERIFIABLE"])
        self.assertEqual(audit["output_checks"], synthesis._output_checks(sample))
        self.assertEqual(audit["trace_report"], report)
        self.assertEqual(audit["native_trace"], fixture_trace(self.problem))
        self.assertEqual(audit["solution_projection"], {"span": [0, len(self.body)],
                         "sha256": hashlib.sha256(self.body.encode()).hexdigest()})
        self.assertEqual(to_json(sample), before)

    def test_typed_unicode_crlf_audit_lifts_all_refs_without_prefix_claims(self) -> None:
        for problem in (self.problem, self.circuit):
            body = (circuit_response(problem, circuit_fixture(problem)) if problem is self.circuit
                    else response(problem, fixture_trace(problem))).replace("\n", "\r\n")
            raw = joint(body, newline="\r\n")
            sections = parse_retry_sections(raw, contract_version=VERSION)
            legacy = synthesis.audit_native_sample(self.sample(body, problem=problem), problem.actor, self.verifier)
            sample = self.sample(raw, problem=problem)
            before = to_json(sample)
            audit = synthesis.audit_native_sample(sample, problem.actor, self.verifier, response_contract=VERSION)
            with self.subTest(family=problem.actor.family):
                self.assertEqual(audit["native_trace"], legacy["native_trace"])
                self.assertEqual(audit["result"]["status"], "CORRECT")
                self.assertEqual(audit["raw_sha256"], hashlib.sha256(raw.encode()).hexdigest())
                self.assertEqual(audit["solution_projection"], {"span": [sections.solution_span.start, len(raw)],
                                 "sha256": hashlib.sha256(body.encode()).hexdigest()})
                for field in ("checks", "issues", "unverified", "output_checks"):
                    for actual, old in zip(audit[field], legacy[field], strict=True):
                        self.assertEqual({key: value for key, value in actual.items() if key != "refs"},
                                         {key: value for key, value in old.items() if key != "refs"})
                        for ref, old_ref in zip(actual["refs"], old["refs"], strict=True):
                            self.assertEqual(ref["span"], [v + sections.solution_span.start for v in old_ref["span"]])
                            start, end = ref["span"]
                            self.assertEqual(raw[start:end], body[old_ref["span"][0]:old_ref["span"][1]])
                            self.assertEqual(ref["sha256"], hashlib.sha256(raw[start:end].encode()).hexdigest())
                self.assertEqual(to_json(sample), before)

    def test_invalid_cached_result_and_rewards_are_ignored_without_mutation(self) -> None:
        sample = self.sample(joint(self.body))
        expected = synthesis.audit_native_sample(sample, self.problem.actor, self.verifier, response_contract=VERSION)
        cached_result, cached_diag = object(), object()
        poisoned = replace(sample, result=cached_result, diag_reward=cached_diag, reward=float("nan"))
        self.assertEqual(synthesis.audit_native_sample(poisoned, self.problem.actor, self.verifier,
                                                      response_contract=VERSION), expected)
        self.assertIs(poisoned.result, cached_result)
        self.assertIs(poisoned.diag_reward, cached_diag)
        self.assertNotEqual(poisoned.reward, poisoned.reward)

    def test_grading_callback_cannot_mix_mutated_caller_text_with_snapshot_evidence(self) -> None:
        raw = joint(self.body)
        cached_result, cached_diag = object(), object()
        sample = replace(self.sample(raw), result=cached_result, diag_reward=cached_diag, reward=float("nan"))
        expected = synthesis.audit_native_sample(sample, self.problem.actor, self.verifier, response_contract=VERSION)
        changed_trace = fixture_trace(self.problem)
        changed_trace["weights"][0] += 1
        changed_raw = joint(response(self.problem, changed_trace))
        grade = self.verifier.grade

        def mutate(problem_id: str, text: str, finish: str, mode: str, **options):
            result = grade(problem_id, text, finish, mode, **options)
            sample.text = changed_raw
            return result

        with patch.object(self.verifier, "grade", side_effect=mutate):
            actual = synthesis.audit_native_sample(sample, self.problem.actor, self.verifier,
                                                   response_contract=VERSION)
        self.assertEqual(actual, expected)
        self.assertEqual(sample.text, changed_raw)
        self.assertIs(sample.result, cached_result)
        self.assertIs(sample.diag_reward, cached_diag)
        self.assertEqual(actual["raw_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        start, end = actual["solution_projection"]["span"]
        self.assertEqual(actual["solution_projection"]["sha256"], hashlib.sha256(raw[start:end].encode()).hexdigest())

    def test_cleaned_final_answer_matches_the_real_fresh_extraction(self) -> None:
        body = self.body.rsplit("Final answer:", 1)[0] + "Final answer: **$" + self.problem.verifier.gold + "$**."
        for text, contract in ((body, None), (joint(body), VERSION)):
            audit = synthesis.audit_native_sample(self.sample(text), self.problem.actor, self.verifier,
                                                  response_contract=contract)
            self.assertEqual(audit["result"]["status"], "CORRECT")
            self.assertEqual(audit["result"]["extracted"], self.problem.verifier.gold)

    def test_error_check_cannot_supply_observations_trace_or_final_output(self) -> None:
        for body in ("Solution:\nNo authored native trace.", "Solution:\nFinal answer: 999"):
            raw = joint(body)
            audit = synthesis.audit_native_sample(self.sample(raw), self.problem.actor, self.verifier,
                                                  response_contract=VERSION)
            with self.subTest(body=body):
                self.assertIsNone(audit["native_trace"])
                self.assertFalse(any(row["kind"] == "OBSERVATION" for row in audit["checks"]))
                self.assertTrue(any(row["rule"] == "TRACE_FORMAT" for row in audit["unverified"]))
                self.assertEqual(audit["result"]["status"],
                                 "INVALID_NO_FINAL" if "999" not in body else "INCORRECT")
                self.assertEqual(audit["output_checks"][0]["status"],
                                 "VIOLATED" if "999" not in body else "COMPLIANT")
                for row in audit["output_checks"]:
                    for ref in row["refs"]:
                        self.assertGreaterEqual(ref["span"][0], len(raw) - len(body))

    def test_native_repair_can_be_supported_while_both_answers_are_wrong(self) -> None:
        trace = fixture_trace(self.problem)
        wrong_trace = copy.deepcopy(trace)
        wrong_trace["weights"][0] += 1
        bodies = [response(self.problem, value).rsplit("Final answer:", 1)[0] + "Final answer: 999"
                  for value in (wrong_trace, trace)]
        audits = [synthesis.audit_native_sample(self.sample(joint(body)), self.problem.actor, self.verifier,
                                               response_contract=VERSION) for body in bodies]
        self.assertEqual([audit["result"]["status"] for audit in audits], ["INCORRECT", "INCORRECT"])
        before, after = [next(row for row in audit["checks"] if row["rule"] == "EDGE_WEIGHT") for audit in audits]
        self.assertEqual((before["status"], after["status"]), ("VIOLATED", "COMPLIANT"))
        self.assertEqual(before["identity"], after["identity"])
        self.assertFalse(any(key in audits[1] for key in ("reward", "route", "success_attempt", "complete_episode")))

    def test_correct_final_answer_keeps_native_violation_and_length_keeps_evidence(self) -> None:
        trace = fixture_trace(self.problem)
        trace["weights"][0] += 1
        raw = joint(response(self.problem, trace))
        for finish in ("stop", "length"):
            audit = synthesis.audit_native_sample(self.sample(raw, finish=finish), self.problem.actor,
                                                  self.verifier, response_contract=VERSION)
            with self.subTest(finish=finish):
                self.assertEqual(audit["result"]["status"], "CORRECT" if finish == "stop" else "TRUNCATED")
                self.assertTrue(any(row["rule"] == "EDGE_WEIGHT" for row in audit["issues"]))
                self.assertEqual(audit["finish_reason"], finish)
                if finish == "length":
                    self.assertEqual(audit["output_checks"], [])

    def test_plain_mode_and_absent_observations_do_not_certify_vision(self) -> None:
        body = self.body[self.body.index("Solution:"):]
        for mode in ("obs", "plain"):
            actor = replace(self.problem.actor, mode=mode)
            audit = synthesis.audit_native_sample(self.sample(body), actor, self.verifier)
            with self.subTest(mode=mode):
                self.assertEqual(audit["trace_report"]["status"], "COMPLIANT")
                self.assertFalse(any(row["kind"] == "OBSERVATION" for row in audit["checks"]))
                if mode == "plain":
                    self.assertIsNone(audit["result"]["slots"])
                else:
                    self.assertEqual(audit["result"]["slots"]["parsed"], 0)
                self.assertNotIn("all_required_observations_verified", audit)
                self.assertNotIn("route", audit)

    def test_bad_identity_mode_contract_and_typed_grammar_fail_before_verifier_calls(self) -> None:
        sample = self.sample(self.body)
        cases = [(replace(sample, problem_id="other"), self.problem.actor, None),
                 (sample, replace(self.problem.actor, problem_id="other"), None),
                 (sample, replace(self.problem.actor, mode="unsupported"), None),
                 (replace(sample, kind="DIAG"), self.problem.actor, None),
                 (replace(sample, turn=1), self.problem.actor, None),
                 (replace(sample, text="x" * (MAX_RAW_CHARS + 1)), self.problem.actor, None),
                 (sample, self.problem.actor, "repair-check-v0")]
        for actual, actor, contract in cases:
            with self.subTest(contract=contract, actor=actor.problem_id), \
                    patch.object(self.verifier, "grade") as grade, self.assertRaises(ValueError):
                synthesis.audit_native_sample(actual, actor, self.verifier, response_contract=contract)
            grade.assert_not_called()
        raw = joint(self.body)
        for text in (self.body, "prefix\n" + raw, raw[:raw.index("RevisedSolution:")],
                     raw.replace("RevisedSolution:\n", "RevisedSolution:", 1)):
            with self.subTest(length=len(text)), patch.object(self.verifier, "grade") as grade, \
                    self.assertRaises(RepairContractError):
                synthesis.audit_native_sample(self.sample(text), self.problem.actor, self.verifier,
                                              response_contract=VERSION)
            grade.assert_not_called()

    def test_poisoned_fresh_reports_and_answers_are_rejected(self) -> None:
        raw, secret = joint(self.body), "PRIVATE-DO-NOT-RETURN-64290"
        sample = self.sample(raw)
        report = self.verifier.trace(sample.problem_id, raw, "stop", response_contract=VERSION)
        for mutation in ("private", "hash", "offset", "contract", "refs_type", "report_type", "false_format_root"):
            forged = copy.deepcopy(report)
            if mutation == "private":
                forged["gold"] = secret
            elif mutation == "hash":
                forged["text_sha256"] = "0" * 64
            elif mutation == "offset":
                forged["trace"]["span"][0] += 1
            elif mutation == "contract":
                forged["response_contract"] = None
            elif mutation == "refs_type":
                forged["checks"][0]["observation_refs"] = {}
            elif mutation == "false_format_root":
                forged["trace"] = None
                forged["checks"] = [{"rule": "TRACE_FORMAT", "locus": "solution", "status": "UNVERIFIABLE",
                                     "observation_refs": [], "blocked_by": []}]
                forged = _finish(forged)
            returned_report = [forged] if mutation == "report_type" else forged
            with self.subTest(mutation=mutation), patch.object(self.verifier, "trace", return_value=returned_report), \
                    self.assertRaises(ValueError) as caught:
                synthesis.audit_native_sample(sample, self.problem.actor, self.verifier, response_contract=VERSION)
            self.assertNotIn(secret, str(caught.exception))
        for poison in (VerifierResult("CORRECT", secret, 1., None, "poison"),
                       VerifierResult("INCORRECT", "A", 0., None, "poison"),
                       VerifierResult("INVALID_NO_FINAL", None, 0., None, "poison")):
            with self.subTest(extracted=poison.extracted), patch.object(self.verifier, "grade", return_value=poison), \
                    self.assertRaises(ValueError) as caught:
                synthesis.audit_native_sample(sample, self.problem.actor, self.verifier, response_contract=VERSION)
            self.assertNotIn(secret, str(caught.exception))

    def test_legitimate_unsupported_and_format_reports_remain_auditable(self) -> None:
        actor, item, *_ = records()
        self.verifier.actors[actor.problem_id], self.verifier.items[actor.problem_id] = actor, item
        unsupported = replace(self.sample('Solution:\nTrace: {"kind":"unregistered"}\nFinal answer: 7'),
                              problem_id=actor.problem_id)
        audit = synthesis.audit_native_sample(unsupported, actor, self.verifier)
        self.assertIsNone(audit["native_trace"])
        self.assertEqual([row["rule"] for row in audit["unverified"]], ["NATIVE_CONTRACT"])
        body = "Solution:\nNo authored structured trace."
        audit = synthesis.audit_native_sample(self.sample(joint(body)), self.problem.actor,
                                              self.verifier, response_contract=VERSION)
        self.assertIsNone(audit["native_trace"])
        self.assertTrue(any(row["rule"] == "TRACE_FORMAT" for row in audit["unverified"]))
        key, value = next(iter(self.problem.verifier.slots.items()))
        duplicate = self.body.replace(f"{key} = {value}", f"{key} = {value}\n{key} = {value}", 1)
        audit = synthesis.audit_native_sample(self.sample(joint(duplicate)), self.problem.actor,
                                              self.verifier, response_contract=VERSION)
        self.assertIsNotNone(audit["native_trace"])
        self.assertTrue(any(row["rule"] == "TRACE_FORMAT" for row in audit["unverified"]))

    def test_fresh_service_roundtrip_keeps_private_annotation_out_of_audit(self) -> None:
        secret = "PRIVATE-ANNOTATION-DO-NOT-RETURN-83901"
        problems = (self.problem, self.circuit)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "vault"
            write_split(root, [p.actor for p in problems],
                        [replace(p.verifier, printed_text=(secret,)) for p in problems], [p.lineage for p in problems])
            with VerifierService(root, forbidden_roots=[root.parent / "benchmark"]) as service:
                for problem in problems:
                    body = (circuit_response(problem, circuit_fixture(problem)) if problem is self.circuit
                            else response(problem, fixture_trace(problem)))
                    sample = self.sample(joint(body), problem=problem)
                    expected = synthesis.audit_native_sample(sample, problem.actor, self.verifier,
                                                             response_contract=VERSION)
                    actual = synthesis.audit_native_sample(sample, problem.actor, service, response_contract=VERSION)
                    self.assertEqual(actual, expected)
                    self.assertNotIn(secret, json.dumps(actual))


if __name__ == "__main__":
    unittest.main()
