"""Opt-in retry_out_of_choice incorrect-v1 rule; authored CPU fixtures, not Qwen behaviour."""
from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

from fine_tuning.notebook_rl import answers, local_profile, local_run, notebook_synthesis, verifier_service
from fine_tuning.notebook_rl import prompts as builders
from fine_tuning.notebook_rl.repair_contract import VERSION as RETRY_VERSION
from fine_tuning.notebook_rl.schema import Sample, VerifierItem, to_json
from fine_tuning.notebook_rl.verifier import RETRY_OUT_OF_CHOICE_TAG, VERSION as VERIFIER_VERSION
from fine_tuning.notebook_rl.verifier_service import VerifierService, VerifierServiceError
from fine_tuning.tests import test_notebook_rl_eval_aligned as eval_aligned
from fine_tuning.tests.test_notebook_rl_eval_aligned import EVAL, INT, MCQ, VERIFIER, eval_flagged, flagged, grade
from fine_tuning.tests.test_notebook_rl_route_finalization import (ACTOR, ITEM, RIGHT, WRONG, Pipeline, note, retry,
                                                                   flagged as route_flagged)

RULE = answers.RETRY_OUT_OF_CHOICE
OUTSIDE = retry("Solution:\nNo listed count fits.\nFinal answer: None of the options are correct.")
# Hashes before the option existed; an absent option must keep them byte-identical.
FLAGGED_SHA256 = "32db41b8e6665618d239316fc0fe95290a5cc4577a3899dafea6e4b0546d3f90"
EVAL_FLAGGED_SHA256 = "d72af95301fadd9d2de5b3687c0c2bd32ccb87447d7c7991aa017eafc928d982"


def retry_grade(actor, answer: str, finish: str = "stop", **options):
    return grade(actor, retry(f"Solution:\nx\n{answer}"), finish, response_contract=RETRY_VERSION, **options)


class GradeTests(unittest.TestCase):
    def test_mcq_retry_answer_naming_no_option_is_incorrect_only_under_the_rule(self) -> None:
        # MCQ choices "2","3","4","5", gold B.
        cases = (("Final answer: None of the options are correct.", "INCORRECT", "UNVERIFIABLE"),
                 ("Final answer: 7", "INCORRECT", "UNVERIFIABLE"),
                 ("Final answer: E", "INCORRECT", "UNVERIFIABLE"),
                 ("Final answer: B", "CORRECT", "CORRECT"),
                 ("Final answer: C", "INCORRECT", "INCORRECT"),
                 # May name an option (label in any case, option text): never re-labelled.
                 ("Final answer: b", "UNVERIFIABLE", "UNVERIFIABLE"),
                 ("Final answer: (B) 3", "UNVERIFIABLE", "UNVERIFIABLE"),
                 ("Final answer: 3", "UNVERIFIABLE", "UNVERIFIABLE"),
                 ("Final answer: None is a correct option", "UNVERIFIABLE", "UNVERIFIABLE"),
                 # Missing, multiple and truncated finals keep their own statuses.
                 ("No final line here.", "INVALID_NO_FINAL", "INVALID_NO_FINAL"),
                 ("Final answer: B\nFinal answer: C", "INVALID_MULTIPLE", "INVALID_MULTIPLE"),
                 ("Final answer: 7 or 9", "INVALID_MULTIPLE", "INVALID_MULTIPLE"))
        for answer, ruled, plain in cases:
            with self.subTest(answer=answer):
                bound, unbound = retry_grade(MCQ, answer, retry_out_of_choice=RULE), retry_grade(MCQ, answer)
                self.assertEqual((bound.status, unbound.status), (ruled, plain))
                self.assertEqual((bound.extracted, bound.reward_answer), (unbound.extracted, unbound.reward_answer))
                self.assertEqual(bound.verifier_version, VERIFIER_VERSION + "+" + RETRY_VERSION + RETRY_OUT_OF_CHOICE_TAG)
                self.assertEqual(unbound.verifier_version, VERIFIER_VERSION + "+" + RETRY_VERSION)
        self.assertEqual(retry_grade(MCQ, "Final answer: 7", "length", retry_out_of_choice=RULE).status, "TRUNCATED")

    def test_non_mcq_and_uncomparable_gold_are_unchanged(self) -> None:
        for answer, status in (("Final answer: twelve", "UNVERIFIABLE"), ("Final answer: 12", "CORRECT"),
                               ("Final answer: 13", "INCORRECT")):
            with self.subTest(answer=answer):
                self.assertEqual(retry_grade(INT, answer, retry_out_of_choice=RULE).status, status)
                self.assertEqual(retry_grade(INT, answer).status, status)
        bad = replace(MCQ, problem_id=MCQ.problem_id + ":bad-gold")
        verifier = replace_items(bad, VerifierItem(bad.problem_id, "mcq", "Z", {}, bad.choices, {}, {}, (), (), ()))
        text = retry("Solution:\nx\nFinal answer: None of the options are correct.")
        result = verifier.grade(bad.problem_id, text, "stop", "obs", response_contract=RETRY_VERSION,
                                retry_out_of_choice=RULE)
        self.assertEqual(result.status, "UNVERIFIABLE")

    def test_misuse_fails_closed(self) -> None:
        for options in ({"retry_out_of_choice": RULE}, {"retry_out_of_choice": "junk", "response_contract": RETRY_VERSION},
                        {"retry_out_of_choice": RULE, "answer_contract": EVAL}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                grade(MCQ, "Final answer: 7", **options)


def replace_items(actor, item):
    verifier = type(VERIFIER).__new__(type(VERIFIER))
    verifier.actors, verifier.items = {actor.problem_id: actor}, {actor.problem_id: item}
    return verifier


class AuditAndServiceTests(unittest.TestCase):
    @staticmethod
    def sample(turn: int) -> Sample:
        return Sample(f"p:T{turn}:0", MCQ.problem_id, f"p:T{turn}", f"T{turn}", turn, None if turn == 1 else "p:T1:0",
                      [1], [2], [-.1], "stop", OUTSIDE if turn > 1 else "Final answer: 7", 0, 0)

    def test_native_audit_binds_the_rule_both_ways(self) -> None:
        options = {"response_contract": RETRY_VERSION}
        audit = notebook_synthesis.audit_native_sample(self.sample(2), MCQ, VERIFIER, retry_out_of_choice=RULE, **options)
        self.assertEqual(audit["result"]["status"], "INCORRECT")
        self.assertEqual(notebook_synthesis.audit_native_sample(self.sample(2), MCQ, VERIFIER, **options)["result"]["status"],
                         "UNVERIFIABLE")
        self.assertEqual(notebook_synthesis.grade_options(1, EVAL, RULE), {"answer_contract": EVAL})
        self.assertEqual(notebook_synthesis.grade_options(2, EVAL, RULE), {"retry_out_of_choice": RULE})
        self.assertEqual(notebook_synthesis.grade_options(3, EVAL, None), {})
        dropped, added = MagicMock(wraps=VERIFIER), MagicMock(wraps=VERIFIER)
        dropped.grade.side_effect = lambda *args, **kwargs: VERIFIER.grade(*args, response_contract=RETRY_VERSION)
        added.grade.side_effect = lambda *args, **kwargs: VERIFIER.grade(*args, response_contract=RETRY_VERSION,
                                                                         retry_out_of_choice=RULE)
        for verifier, rule in ((dropped, RULE), (added, None)):
            with self.subTest(rule=rule), self.assertRaisesRegex(ValueError, "not bound"):
                notebook_synthesis.audit_native_sample(self.sample(2), MCQ, verifier, retry_out_of_choice=rule, **options)
        for sample, extra in ((self.sample(1), {}), (self.sample(2), {"response_contract": None})):
            with self.subTest(turn=sample.turn), self.assertRaises(ValueError):
                notebook_synthesis.audit_native_sample(sample, MCQ, VERIFIER, **(options | extra), retry_out_of_choice=RULE)

    def test_worker_and_client_apply_and_check_the_rule(self) -> None:
        def dispatch(verifier, params: dict) -> dict:
            return verifier_service._dispatch(verifier, verifier_service._Issued(), {
                "id": 1, "method": "grade", "problem_id": MCQ.problem_id,
                "params": {"text": OUTSIDE, "finish_reason": "stop", "mode": "obs"} | params})

        ruled = {"response_contract": RETRY_VERSION, "retry_out_of_choice": RULE}
        self.assertEqual(dispatch(VERIFIER, ruled)["status"], "INCORRECT")
        self.assertEqual(dispatch(VERIFIER, {"response_contract": RETRY_VERSION})["status"], "UNVERIFIABLE")
        stripped = MagicMock(wraps=VERIFIER)
        stripped.items, stripped.actors = VERIFIER.items, VERIFIER.actors
        stripped.grade.side_effect = lambda *args, **kwargs: VERIFIER.grade(*args, response_contract=RETRY_VERSION)
        for verifier, params, code in ((stripped, ruled, "VERIFIER_ERROR"),
                                       (VERIFIER, {"retry_out_of_choice": RULE}, "INVALID_REQUEST"),
                                       (VERIFIER, ruled | {"answer_contract": EVAL}, "INVALID_REQUEST")):
            with self.subTest(code=code), self.assertRaisesRegex(verifier_service._Rejected, code):
                dispatch(verifier, params)

        def client(request) -> VerifierService:
            service = VerifierService.__new__(VerifierService)
            service._request, service.close = request, lambda: None
            return service

        honest = client(lambda method, pid, params: verifier_service._dispatch(
            VERIFIER, verifier_service._Issued(), {"id": 1, "method": method, "problem_id": pid, "params": params}))
        untagged = client(lambda method, pid, params: json.loads(to_json(VERIFIER.grade(
            pid, params["text"], params["finish_reason"], params["mode"], response_contract=RETRY_VERSION))))
        call = (MCQ.problem_id, OUTSIDE, "stop", "obs")
        result = honest.grade(*call, response_contract=RETRY_VERSION, retry_out_of_choice=RULE)
        self.assertEqual((result.status, result.verifier_version[-len(RETRY_OUT_OF_CHOICE_TAG):]),
                         ("INCORRECT", RETRY_OUT_OF_CHOICE_TAG))
        self.assertEqual(honest.grade(*call, response_contract=RETRY_VERSION).status, "UNVERIFIABLE")
        with self.assertRaises(VerifierServiceError):
            untagged.grade(*call, response_contract=RETRY_VERSION, retry_out_of_choice=RULE)


class PipelineTests(unittest.TestCase):
    """local-test-v2 minimal (2026-10-31) shape: T1 P0 graded by V8, T2 = [wrong letter, outside the options]."""

    def collect(self, protocol: dict, replies: list[list[str]]):
        p = Pipeline(self, protocol, slotless=True, actor=ACTOR, item=ITEM)
        tree, engine = p.f.collect(replies)
        return p, tree, engine

    def test_bound_rule_grades_incorrect_keeps_the_group_usable_and_admits_the_tree(self) -> None:
        protocol = route_flagged(turn1_contract=EVAL, retry_out_of_choice=RULE)
        p, tree, engine = self.collect(protocol, [["So the answer is (C)"] * 4, [retry(WRONG), OUTSIDE],
                                                  [OUTSIDE, retry(RIGHT)]])
        p.f.attach(tree, engine, [note([{"attempt": 1, "type": "WRONG_STEP"}])] * 4)
        self.assertEqual([group.kind for group in tree.groups], ["T1", "T2", "T3", "DIAG"])
        t1, t2, t3 = tree.groups[:3]
        self.assertEqual([row.result.status for row in t1.samples], ["INCORRECT"] * 4)
        self.assertEqual((t2.usable, t2.reason, [row.result.status for row in t2.samples]),
                         (True, None, ["INCORRECT", "INCORRECT"]))
        self.assertEqual([row.result.status for row in t3.samples], ["INCORRECT", "CORRECT"])
        self.assertTrue(all(row.result.verifier_version.endswith(RETRY_OUT_OF_CHOICE_TAG)
                            for group in (t2, t3) for row in group.samples))
        self.assertFalse(any(RETRY_OUT_OF_CHOICE_TAG in row.result.verifier_version for row in t1.samples))
        record = tree.notebook["rewarded_notebook"]
        self.assertEqual((record["status"], record["source_audit"]["retry_out_of_choice"]), ("GENERATED", RULE))
        receipt = p.f.prepare(tree, engine)
        self.assertTrue(receipt["admitted"])
        self.assertEqual(receipt["admission_reasons"], [])

    def test_unbound_rule_keeps_the_old_unverifiable_non_admitted_tree(self) -> None:
        p, tree, engine = self.collect(route_flagged(turn1_contract=EVAL),
                                       [["So the answer is (C)"] * 4, [retry(WRONG), OUTSIDE]])
        p.f.attach(tree, engine)
        t2 = tree.groups[1]
        self.assertEqual((t2.usable, [row.result.status for row in t2.samples]), (False, ["INCORRECT", "UNVERIFIABLE"]))
        self.assertNotIn("retry_out_of_choice", tree.notebook["rewarded_notebook"]["source_audit"])
        receipt = p.f.prepare(tree, engine)
        self.assertFalse(receipt["admitted"])
        self.assertIn("UNUSABLE_SOLUTION_GROUP", receipt["admission_reasons"])


class ProtocolTests(unittest.TestCase):
    def test_absent_option_keeps_hashes_and_bound_option_round_trips(self) -> None:
        self.assertEqual((flagged()["sha256"], eval_flagged()["sha256"]), (FLAGGED_SHA256, EVAL_FLAGGED_SHA256))
        self.assertNotIn("repair_retry_out_of_choice", eval_flagged())
        self.assertNotIn("repair_retry_out_of_choice", local_profile.local_prompts(eval_flagged()))
        bound = eval_flagged(retry_out_of_choice=RULE)
        self.assertEqual(bound["repair_retry_out_of_choice"], RULE)
        self.assertNotEqual(bound["sha256"], EVAL_FLAGGED_SHA256)
        self.assertEqual({key: value for key, value in bound.items() if key not in ("repair_retry_out_of_choice", "sha256")},
                         {key: value for key, value in eval_flagged().items() if key != "sha256"})
        self.assertEqual(local_profile.restore_protocol(json.loads(json.dumps(bound))), bound)
        prompts = local_profile.local_prompts(bound)
        self.assertEqual(builders.for_actor(MCQ, prompts)["repair_retry_out_of_choice"], RULE)
        for mutate in (lambda protocol: protocol.update(repair_retry_out_of_choice="junk"),
                       lambda protocol: protocol.pop("repair_retry_out_of_choice"),
                       lambda protocol: protocol.pop("repair_turn1_contract")):
            tampered = json.loads(json.dumps(bound))
            mutate(tampered)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                local_profile.restore_protocol(tampered)
        for options in ({"retry_out_of_choice": RULE}, {"turn1_contract": EVAL, "retry_out_of_choice": "junk"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                flagged(**options)


class CliTests(unittest.TestCase):
    parse = staticmethod(eval_aligned.CliAndReadinessTests.parse)
    setUp = eval_aligned.CliAndReadinessTests.setUp
    cache_check = eval_aligned.CliAndReadinessTests.cache_check
    FLAGS = [*eval_aligned.CliAndReadinessTests.NO_CAP, "--retry-out-of-choice", RULE]

    def test_flag_binds_through_the_shared_helper_and_cache_readiness(self) -> None:
        args = self.parse(self.FLAGS)
        self.assertIsNone(local_profile.repair_cli_error(args, repair_notebook=True))
        keywords = local_profile.repair_cli_keywords(args)
        self.assertEqual(keywords["retry_out_of_choice"], RULE)
        expected = local_profile.build_protocol(**eval_aligned.CliAndReadinessTests.FLAGGED_OPTIONS, turn1_contract=EVAL,
                                                retry_out_of_choice=RULE)
        self.assertEqual(local_profile.build_protocol(repair_notebook=True, context_limit=24576, **keywords), expected)
        self.assertNotIn("retry_out_of_choice", local_profile.repair_cli_keywords(self.parse(eval_aligned.CliAndReadinessTests.NO_CAP)))
        message = local_profile.repair_cli_error(
            self.parse([*eval_aligned.CliAndReadinessTests.NO_CONTRACT, "--retry-out-of-choice", RULE]), repair_notebook=True)
        self.assertIn("--retry-out-of-choice requires --turn1-contract", message)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            self.parse([*eval_aligned.CliAndReadinessTests.NO_CAP, "--retry-out-of-choice", "junk"])
        check = self.cache_check("retry", self.FLAGS)
        self.assertEqual(json.loads((check / "protocol.json").read_text())["sha256"], expected["sha256"])
        self.assertEqual(local_run.cache_readiness(check, self.root / "model", expected)["status"],
                         "NF4_CACHE_FRESH_PROCESS_RELOAD_GENERATION_VERIFIED")
        with self.assertRaises(ValueError):
            local_run.cache_readiness(check, self.root / "model",
                                      local_profile.build_protocol(**eval_aligned.CliAndReadinessTests.FLAGGED_OPTIONS, turn1_contract=EVAL))


if __name__ == "__main__":
    unittest.main()
