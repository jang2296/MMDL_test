"""Scripted n1 CPU evidence, never Qwen training or measured model performance."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import local_ladder_generation as generation
from fine_tuning.notebook_rl import local_ladder_recovery as recovery
from fine_tuning.notebook_rl import repair_contract, repair_notebook
from fine_tuning.notebook_rl import prompts
from fine_tuning.notebook_rl.engines import Completion, RolloutResult
from fine_tuning.notebook_rl.local_ladder_gate import _validate_protocol
from fine_tuning.notebook_rl.generators.common import GeneratedProblem
from fine_tuning.notebook_rl.schema import ProblemTree, from_json
from fine_tuning.tests import test_notebook_rl_circuit_trace as trace_fixture
from fine_tuning.tests import test_notebook_rl_reference_ladder_evaluation as reference_fixture
from fine_tuning.tests.test_notebook_rl_repair_origin_wire import canonical_rows


class RepairEngine(reference_fixture.ReferenceEngine):
    def sample(self, requests):
        request = requests[0]
        if request.request_id.endswith(":FRESH_OBSERVATION"):
            tokens = self.tokenizer.encode("authored unverified reread")
            return [RolloutResult(request.request_id, self.prepare(request).token_ids,
                [Completion(tokens, [-.5] * len(tokens), "stop", "authored unverified reread")])]
        self.calls.append(request)
        pid, stage = request.request_id.rsplit(":", 1)
        problem = self.rows[pid]
        outcome = self.script.get((pid, stage), "correct")
        if stage == "NOTE":
            body = request.messages[0]["content"][-1]["text"]
            payload = json.loads(body.split("<read_only_history>\n")[1].split("\n</read_only_history>")[0])
            attempts = [{"sample_id": row["sample_id"], "outcome": row["status"], "answer": row["answer"]}
                        for row in payload["attempts"]]
            claims = []
            if outcome in ("claim", "false_claim", "persisted"):
                trace = trace_fixture.fixture_trace(problem)
                selected = [row for row in canonical_rows(payload["public_origins"])
                    if row[2:5] == ["TRACE", "CIRCUIT_NODE_VOLTAGE", f"nodes[{len(trace['nodes'])-1}]"]]
                claims = [{"origin_ref": selected[0][0], "later_ref": selected[1][0],
                    "relation": "PERSISTED" if outcome == "persisted" else "CORRECTED", "rule": selected[0][3],
                    "repaired_value": None if outcome == "persisted" else {"field": "nodes", "value": trace["nodes"][-1]}}]
            text = json.dumps({"version": repair_contract.FINAL_VERSION, "scope": "BOUNDED_CLAIMS_FULL_HISTORY",
                "attempts": attempts, "success_sample_id": payload["success_sample_id"],
                "claims": claims, "uncertain_refs": [], "lesson_text": "untrusted"})
            if outcome == "malformed":
                text = "{broken"
        elif outcome == "malformed":
            text = 'ErrorCheck: {broken}\nRevisedSolution:\nFinal answer: ' + problem.verifier.gold
        else:
            trace = trace_fixture.fixture_trace(problem)
            if outcome == "native_issue":
                trace["nodes"][-1][1] = "-1"
            text = trace_fixture.response(problem, trace)
            if outcome == "wrong":
                text = text.rsplit("Final answer:", 1)[0] + "Final answer: 999999 A"
            if outcome == "nofinal":
                text = text.rsplit("Final answer:", 1)[0]
            if stage != "T1":
                feedback = self.tokenizer.decode([token for token in request.prompt_token_ids
                    if token in self.tokenizer.pieces]).split("<public_origins>\n")[-1]
                origins = canonical_rows(json.loads(feedback.split("\n</public_origins>")[0]))
                origin = next(row for row in origins if row[2:5] == ["TRACE", "CIRCUIT_NODE_VOLTAGE", f"nodes[{len(trace['nodes'])-1}]"])
                check = {"version": repair_contract.VERSION, "claims": [{"origin_ref": origin[0],
                    "intent": "RECHECK_OPERATION", "rule": origin[3],
                    "proposed_value": {"field": "nodes", "value": trace["nodes"][-1]}}], "uncertain_refs": []}
                if outcome == "bad_version":
                    check["version"] = "unsupported-model-version"
                text = "ErrorCheck: " + json.dumps(check) + "\nRevisedSolution:\n" + text
        tokens = self.tokenizer.encode(text)
        return [RolloutResult(request.request_id, self.prepare(request).token_ids,
            [Completion(tokens, [-.5] * len(tokens), "length" if outcome == "truncated" else "stop", text)])]


class RepairLadderTests(unittest.TestCase):
    rows: list[GeneratedProblem]
    cohort: Path
    root: Path
    sha: str

    @classmethod
    def setUpClass(cls):
        reference_fixture.ReferenceLadderEvaluationTests.setUpClass.__func__(cls)

    def setUp(self):
        reference_fixture.ReferenceLadderEvaluationTests.setUp(self)
        self.templates = generation.circuit_evaluation_prompts(generation.CIRCUIT_REPAIR_PROMPT_POLICY)

    def run_repair(self, script=None, name="repair", version=0):
        engine = RepairEngine(self.rows, script=script, version=version)
        result = recovery.diagnose(self.cohort, self.root / name, engine, self.templates,
            expected_preparation_sha256=self.sha, local_protocol_sha256="a" * 64,
            deadline=time.monotonic() + 180, context_limit=24576)
        self.assertEqual(result["status"], "COMPLETE_NATIVE_LADDER_RECOVERY", result.get("reason"))
        return result, engine

    def test_native_correct_answer_repair_and_fresh_replay(self):
        pid = self.rows[0].actor.problem_id
        from fine_tuning.notebook_rl import rollout, updates
        with patch.object(rollout, "build_tree", side_effect=AssertionError("training collector called")), \
                patch.object(repair_notebook, "append", side_effect=AssertionError("training final collector called")):
            result, engine = self.run_repair({(pid, "T1"): "native_issue", (pid, "NOTE"): "claim"})
        self.assertEqual((result["first_attempt_accuracy"], result["final_accuracy"]), (1., 1.))
        self.assertEqual(result["repair_metrics"]["verified_native_corrected_count"], 1)
        self.assertEqual(result["repair_metrics"]["final_note_count"], 1)
        self.assertEqual(result["repair_metrics"]["new_error_scope"], "identity-bound-native-child-issues-absent-from-parent")
        self.assertIsNone(result["note_verified_count"])
        self.assertEqual(result["repair_metrics"]["final_claims"]["submitted_claim_precision"], 1.)
        self.assertEqual(result["repair_metrics"]["final_claims"]["verified_claim_count"], 1)
        _validate_protocol(result["protocol"])
        replayed = recovery.replay(self.cohort, result, expected_preparation_sha256=self.sha)
        self.assertEqual(replayed["repair_metrics"], result["repair_metrics"])
        self.assertNotEqual(replayed["verifier_process"]["pid"], result["verifier_process"]["pid"])
        self.assertEqual([row.request_id.rsplit(":", 1)[1] for row in engine.calls[10:]], ["T2", "NOTE"])
        call = result["calls"][0]
        first = result["generation"]["rows"][0]
        prefix = first["input"]["prompt_token_ids"] + first["completion_token_ids"]
        self.assertEqual(call["input"]["prompt_token_ids"][:len(prefix)], prefix)
        self.assertEqual(call["seed"], recovery._seed(pid, "T2"))
        self.assertTrue(all(not group["usable"] and group["reason"] == "EVALUATION_SINGLE_SAMPLE"
            for entry in result["trees"] for group in entry["tree"]["groups"]))
        tree = from_json(ProblemTree, json.dumps(result["trees"][0]["tree"]))
        self.assertFalse(updates.has_learning_signal(tree))
        for mutation in ("raw", "feedback", "image", "metric", "group", "contract"):
            changed = copy.deepcopy(result)
            row = changed["calls"][0]
            if mutation == "raw":
                row["text"] += "changed"
            elif mutation == "feedback":
                row["feedback"] += "changed"
            elif mutation == "image":
                row["input"]["pixel_tensors"] = {}
            elif mutation == "metric":
                changed["repair_metrics"]["verified_native_corrected_count"] += 1
            elif mutation == "group":
                changed["trees"][0]["tree"]["groups"][0]["usable"] = True
            else:
                row["response_contract"] = None
            row["input_sha256"] = generation._digest(row["input"])
            self.rehash(row)
            self.rehash(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)

    @staticmethod
    def rehash(row):
        row["record_sha256"] = generation._digest({key: value for key, value in row.items() if key != "record_sha256"})

    def test_reread_after_t1_reused_by_t3_and_preserved_on_context_or_budget_skip(self):
        pid = self.rows[0].actor.problem_id
        original_prepare, original_sample = RepairEngine.prepare, RepairEngine.sample
        for mode in ("context", "budget"):
            now = [100.0]
            fresh_calls = []
            engine = RepairEngine(self.rows, script={(pid, "T1"): "wrong"}, version=0)
            def prepare(instance, request):
                if mode == "context" and request.request_id == pid + ":T2":
                    raise ValueError("CONTEXT_BLOCKED: authored retry reservation")
                return original_prepare(instance, request)
            def sample(instance, requests):
                result = original_sample(instance, requests)
                if requests[0].request_id.endswith(":FRESH_OBSERVATION"):
                    fresh_calls.append(requests[0])
                    if mode == "budget":
                        now[0] = 1000.0
                return result
            with patch.object(RepairEngine, "prepare", prepare), patch.object(RepairEngine, "sample", sample), \
                    patch.object(recovery.time, "monotonic", side_effect=lambda: now[0]):
                result = recovery.diagnose(self.cohort, self.root / ("reread-skip-" + mode), engine,
                    self.templates, expected_preparation_sha256=self.sha,
                    local_protocol_sha256="a" * 64, deadline=500.0, context_limit=24576)
            self.assertEqual(result["status"], "INCOMPLETE")
            tree = result["partial_trees"][0]
            self.assertEqual(len(fresh_calls), 1)
            self.assertEqual([row["event"]["phase"] for row in result["fresh_observation_events"]],
                             ["STARTED", "COMPLETED"])
            self.assertIn("fresh_observation", tree["notebook"])
            self.assertEqual(tree["notebook"]["skipped_branches"], [{"kind": "T2",
                "parent_sample_id": pid + ":T1:0", "status": "CONTEXT_BLOCKED" if mode == "context" else "BUDGET_STOP"}])
            self.assertEqual(result["calls"], [])

    def test_completed_persisted_retry_deadline_failure_is_never_a_skip(self):
        pid = self.rows[0].actor.problem_id
        original_sample = RepairEngine.sample
        now = [100.0]
        engine = RepairEngine(self.rows, script={(pid, "T1"): "wrong"}, version=0)
        def sample(instance, requests):
            result = original_sample(instance, requests)
            if requests[0].request_id == pid + ":T2":
                now[0] = 1000.0
            return result
        destination = self.root / "completed-retry-deadline"
        with patch.object(RepairEngine, "sample", sample), \
                patch.object(recovery.time, "monotonic", side_effect=lambda: now[0]):
            result = recovery.diagnose(self.cohort, destination, engine, self.templates,
                expected_preparation_sha256=self.sha, local_protocol_sha256="a" * 64,
                deadline=500.0, context_limit=24576)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(len(result["calls"]), 1)
        self.assertEqual(result["calls"][0]["stage"], "T2")
        self.assertEqual(result["calls"][0]["finish_reason"], "stop")
        self.assertTrue((destination / "call-0000.json").is_file())
        self.assertEqual(result["partial_trees"][0]["notebook"]["skipped_branches"], [])
        self.assertIn("fresh_observation", result["partial_trees"][0]["notebook"])

    def test_fresh_event_integer_types_are_exact_even_after_rehash(self):
        pid = self.rows[0].actor.problem_id
        result, _ = self.run_repair({(pid, "T1"): "wrong"}, name="fresh-integer-types")
        for key in ("before_call_index", "policy_version"):
            changed = copy.deepcopy(result)
            event = changed["fresh_observation_events"][0]
            if key == "policy_version":
                event["event"][key] = False
            else:
                event[key] = False
            self.rehash(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery._validate_complete(changed)

    def test_latest_correct_source_survives_later_wrong_retry_and_malformed_is_not_rescued(self):
        a, b = [row.actor.problem_id for row in self.rows[:2]]
        result, _ = self.run_repair({(a, "T1"): "wrong", (a, "T2"): "native_issue", (a, "T3"): "wrong", (a, "NOTE"): "false_claim",
            (b, "T1"): "wrong", (b, "T2"): "malformed", (b, "T3"): "malformed"})
        self.assertEqual(result["final_statuses"][a], "INCORRECT")
        self.assertEqual(result["trees"][0]["episode_audit"]["success_sample_id"], a + ":T2:0")
        self.assertEqual(result["final_statuses"][b], "INVALID_RESPONSE_CONTRACT")
        self.assertIsNone(result["trees"][1]["tree"]["groups"][-1]["samples"][0]["result"])
        self.assertIsNone(result["trees"][1]["note"])
        self.assertEqual(result["trees"][1]["episode_audit"]["routing"]["admission"], "UNRESOLVED")
        self.assertEqual(result["repair_metrics"]["malformed_response_count"], 2)
        self.assertGreaterEqual(result["repair_metrics"]["false_accusation_count"], 1)
        recovery.replay(self.cohort, result, expected_preparation_sha256=self.sha)

    def test_malformed_retry_keeps_prior_correct_source_and_full_final_history(self):
        pid = self.rows[0].actor.problem_id
        result, engine = self.run_repair({(pid, "T1"): "native_issue", (pid, "T2"): "malformed", (pid, "T3"): "malformed"})
        entry = result["trees"][0]
        self.assertEqual(entry["episode_audit"]["success_sample_id"], pid + ":T1:0")
        self.assertEqual(result["final_statuses"][pid], "INVALID_RESPONSE_CONTRACT")
        self.assertEqual([row.request_id.rsplit(":", 1)[1] for row in engine.calls[10:]], ["T2", "T3", "NOTE"])
        self.assertEqual(result["repair_metrics"]["final_note_count"], 1)
        self.assertEqual(result["repair_metrics"]["final_claims"]["eligible_history_count"], 1)
        self.assertEqual(result["repair_metrics"]["final_claims"]["known_issue_count"], 3)
        self.assertEqual(result["repair_metrics"]["final_claims"]["known_issue_coverage"], 0.)
        note_row = result["calls"][-1]
        body = note_row["input"]["messages"][0]["content"][-1]["text"]
        history = json.loads(body.split("<read_only_history>\n")[1].split("\n</read_only_history>")[0])
        self.assertEqual([row["status"] for row in history["attempts"]],
                         ["CORRECT", "INVALID_RESPONSE_CONTRACT", "INVALID_RESPONSE_CONTRACT"])
        self.assertEqual(history["attempts"][1]["raw"], result["calls"][0]["text"])
        self.assertIsNone(history["attempts"][1]["answer"])
        self.assertFalse(entry["episode_audit"]["routing"]["all_required_observations_verified"])
        parsed = repair_contract.parse_final_note(note_row["text"], contract_version=repair_contract.FINAL_VERSION)
        self.assertEqual([row["outcome"] for row in parsed.attempts],
                         ["CORRECT", "INVALID_RESPONSE_CONTRACT", "INVALID_RESPONSE_CONTRACT"])
        self.assertEqual(parsed.attempts[1]["answer"], None)
        checked = entry["note"]["checked"]["audit"]
        self.assertTrue(checked["metadata_valid"])
        self.assertEqual(checked["metadata_errors"], [])
        missing_note = copy.deepcopy(entry)
        missing_note["note"] = None
        missing_metrics = recovery._repair_metrics([missing_note], [])["repair_metrics"]
        self.assertEqual(missing_metrics["final_note_count"], 0)
        self.assertEqual(missing_metrics["final_claims"]["known_issue_count"], 3)
        self.assertEqual(missing_metrics["final_claims"]["known_issue_coverage"], 0.)
        serialized = json.loads(json.dumps(result, sort_keys=True))
        recovery.replay(self.cohort, serialized, expected_preparation_sha256=self.sha)
        for mutation in ("grade", "syntax", "metadata", "outcome"):
            changed = copy.deepcopy(serialized)
            if mutation == "grade":
                changed["trees"][0]["tree"]["groups"][1]["samples"][0]["result"] = entry["tree"]["groups"][0]["samples"][0]["result"]
            elif mutation == "syntax":
                changed["trees"][0]["episode_audit"]["native_audits"][pid + ":T2:0"]["syntax_failure"]["code"] = "INVALID_FIELD"
            elif mutation == "metadata":
                changed["calls"][-1]["input"]["messages"][0]["content"][-1]["text"] = body.replace(
                    '"status":"INVALID_RESPONSE_CONTRACT"', '"status":"CORRECT"')
                changed["calls"][-1]["input_sha256"] = generation._digest(changed["calls"][-1]["input"])
                self.rehash(changed["calls"][-1])
            else:
                changed["final_statuses"][pid] = "UNAVAILABLE"
            self.rehash(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)

    def test_bad_model_version_is_invalid_response_and_wrong_caller_contract_is_fatal(self):
        pid = self.rows[0].actor.problem_id
        result, _ = self.run_repair({(pid, "T1"): "wrong", (pid, "T2"): "bad_version", (pid, "T3"): "bad_version"})
        self.assertEqual(result["final_statuses"][pid], "INVALID_RESPONSE_CONTRACT")
        self.assertIsNone(result["trees"][0]["note"])
        audited = result["trees"][0]["episode_audit"]["native_audits"][pid + ":T2:0"]
        self.assertEqual(audited["syntax_failure"]["code"], "INVALID_FIELD")
        self.assertEqual(result["repair_metrics"]["invalid_response_contract_count"], 2)
        recovery.replay(self.cohort, result, expected_preparation_sha256=self.sha)
        with self.assertRaisesRegex(repair_contract.RepairContractError, "CONTRACT_VERSION"):
            repair_contract.public_attempt_outcome(result["calls"][0]["text"],
                response_contract="unsupported-caller-contract", result=None)
        changed = copy.deepcopy(result)
        changed["calls"][0]["response_contract"] = "unsupported-caller-contract"
        self.rehash(changed["calls"][0])
        self.rehash(changed)
        with self.assertRaises(ValueError):
            recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)

    def test_unavailable_valid_grade_still_blocks_success_and_metrics_separate_failure_kinds(self):
        a, b, c = [row.actor.problem_id for row in self.rows[:3]]
        result, _ = self.run_repair({(a, "T1"): "wrong", (a, "T2"): "malformed",
            (a, "T3"): "malformed",
            (b, "T1"): "nofinal", (c, "T1"): "truncated"})
        metrics = result["repair_metrics"]
        self.assertEqual((metrics["invalid_response_contract_count"], metrics["no_final_count"], metrics["output_limit_count"]), (2, 1, 1))
        self.assertEqual(result["first_statuses"][b], "INVALID_NO_FINAL")
        self.assertEqual(result["first_statuses"][c], "TRUNCATED")
        self.assertEqual(result["final_statuses"][a], "INVALID_RESPONSE_CONTRACT")
        self.assertEqual(result["initially_incorrect"], 1)
        self.assertEqual(result["initially_failed"], 3)
        self.assertEqual(result["first_attempt_accuracy"], .7)
        entry = result["trees"][1]
        tree = from_json(ProblemTree, json.dumps(entry["tree"]))
        tree.groups[0].samples[0].result = None
        with self.assertRaisesRegex(ValueError, "source outcome unavailable"):
            recovery._repair_episode(tree, entry["episode_audit"]["native_audits"])
        entry["tree"] = recovery._wire(tree)
        with self.assertRaisesRegex(ValueError, "source outcome unavailable"):
            recovery._repair_metrics([entry], [])

    def test_new_policy_requires_exact_context_and_circuit_domain(self):
        with self.assertRaises(ValueError):
            recovery.protocol(self.templates, "a" * 64)
        first = generation.protocol(self.templates, "a" * 64, context_limit=24576)
        with self.assertRaises(ValueError):
            _validate_protocol(first)
        for key in ("evaluator_version", "retry_prompt_sha256"):
            contract = recovery.protocol(self.templates, "a" * 64, context_limit=24576)
            contract[key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validate_protocol(contract)
        with self.assertRaises(ValueError):
            prompts.turn1_text(replace(self.rows[0].actor, source="gen-graph-v1"), self.templates)

    def test_malformed_t2_recovers_grammar_in_t3_without_native_correction_credit(self):
        pid = self.rows[0].actor.problem_id
        result, engine = self.run_repair({(pid, "T1"): "wrong", (pid, "T2"): "malformed"})
        entry = result["trees"][0]
        self.assertEqual([row.request_id.rsplit(":", 1)[1] for row in engine.calls[10:]], ["T2", "T3", "NOTE"])
        self.assertEqual(result["final_statuses"][pid], "CORRECT")
        self.assertEqual(result["repair_metrics"]["verified_native_corrected_count"], 0)
        self.assertEqual(len(entry["online_audits"]), 1)
        self.assertEqual(entry["online_audits"][0]["child_id"], pid + ":T3:0")
        parent = result["calls"][0]
        child = result["calls"][1]
        prefix = parent["input"]["prompt_token_ids"] + parent["completion_token_ids"]
        self.assertEqual(child["input"]["prompt_token_ids"][:len(prefix)], prefix)
        self.assertIn("grammar: INVALID_JSON", child["feedback"])
        self.assertIn("Answer correctness: NOT JUDGED", child["feedback"])
        self.assertIsNone(entry["tree"]["groups"][1]["samples"][0]["result"])
        self.assertEqual(entry["tree"]["groups"][1]["samples"][0]["reward"], 0)
        replayed = recovery.replay(self.cohort, result, expected_preparation_sha256=self.sha)
        self.assertEqual(replayed["repair_metrics"], result["repair_metrics"])

    def test_absent_retry_has_no_final_and_zero_denominators_are_null(self):
        result, _ = self.run_repair()
        self.assertEqual(result["calls"], [])
        self.assertEqual(result["repair_metrics"]["final_note_count"], 0)
        self.assertIsNone(result["repair_metrics"]["final_claims"]["submitted_claim_precision"])
        self.assertIsNone(result["repair_metrics"]["final_claims"]["known_issue_coverage"])
        self.assertIsNone(result["repair_metrics"]["final_claims"]["native_correction_coverage"])

    def test_malformed_final_preserves_independent_coverage_denominators(self):
        pid = self.rows[0].actor.problem_id
        result, _ = self.run_repair({(pid, "T1"): "native_issue", (pid, "NOTE"): "malformed"})
        metrics = result["repair_metrics"]
        self.assertEqual(metrics["final_syntax_failure_count"], 1)
        self.assertEqual(metrics["final_claims"]["known_issue_count"], 3)
        self.assertEqual(metrics["final_claims"]["correction_opportunity_count"], 3)
        self.assertEqual(metrics["final_claims"]["known_issue_coverage"], 0.)
        self.assertEqual(metrics["final_claims"]["native_correction_coverage"], 0.)
        self.assertIsNone(metrics["final_claims"]["submitted_claim_precision"])
        recovery.replay(self.cohort, result, expected_preparation_sha256=self.sha)

    def test_persisted_claim_reviews_both_error_endpoints(self):
        pid = self.rows[0].actor.problem_id
        result, _ = self.run_repair({(pid, "T1"): "native_issue", (pid, "T2"): "native_issue", (pid, "NOTE"): "persisted"})
        metrics = result["repair_metrics"]
        self.assertEqual(metrics["known_native_issue_count"], 6)
        self.assertEqual(metrics["unreviewed_count"], 4)
        self.assertEqual(metrics["final_claims"]["known_issue_count"], 6)
        self.assertEqual(metrics["final_claims"]["covered_known_issue_count"], 2)
        self.assertEqual(metrics["final_claims"]["known_issue_coverage"], 1 / 3)
        recovery.replay(self.cohort, result, expected_preparation_sha256=self.sha)

    def test_paired_repair_metrics_preserve_the_exact_evaluator(self):
        before, _ = self.run_repair(name="before")
        after, _ = self.run_repair(name="after", version=1)
        compared = recovery.validate_paired_records(before, after)
        self.assertEqual(compared["repair_metrics"], {"left": before["repair_metrics"], "right": after["repair_metrics"]})
        after["protocol"]["evaluator_version"] = recovery.VERSION
        after["protocol_sha256"] = generation._digest(after["protocol"])
        self.rehash(after)
        with self.assertRaises(ValueError):
            recovery.validate_paired_records(before, after)

    def test_public_helper_preserves_training_request_bytes(self):
        from fine_tuning.tests.test_notebook_rl_repair_notebook import RepairNotebookTests
        from fine_tuning.tests.test_notebook_rl_repair_verifier import joint
        fixture = RepairNotebookTests()
        fixture.setUp()
        tree, _ = fixture.collect([[fixture.wrong] * 4, [joint(fixture.right)] * 2])
        audit = repair_notebook.audit_tree(tree, fixture.actor, fixture.verifier, prompts=fixture.prompts,
            master_seed=fixture.cfg.master_seed)
        after = repair_notebook._request(tree, fixture.actor, fixture.prompts, audit, fixture.images[fixture.actor.problem_id])
        feedback = [{key: call[key] for key in ("group_id", "turn", "parent_sample_id", "feedback", "feedback_sha256")}
                    for call in tree.notebook["repair_collector"]["calls"]]
        self.assertEqual(after.messages, recovery._repair_note_messages(fixture.actor, fixture.prompts, tree,
            audit, feedback))
        self.assertEqual(recovery._hash(after.messages), "bcb4ba9efd0066e59596b5692330b6c50072c41c0d1f6aa4155bad0662ff698c")
        self.assertEqual(after.seeds, [884010108, 1681053265, 1841857784, 974984049])
        self.assertEqual((after.n, after.max_new_tokens), (4, 8192))

    def test_public_helper_keeps_malformed_sibling_before_valid_correct_source(self):
        from fine_tuning.tests.test_notebook_rl_repair_notebook import RepairNotebookTests
        from fine_tuning.tests.test_notebook_rl_repair_verifier import joint
        fixture = RepairNotebookTests()
        fixture.setUp()
        malformed = "ErrorCheck: {broken}\nRevisedSolution:\n" + fixture.right
        tree, _ = fixture.collect([[fixture.wrong] * 4, [malformed, joint(fixture.right)], [joint(fixture.right)] * 2])
        audit = repair_notebook.audit_tree(tree, fixture.actor, fixture.verifier, prompts=fixture.prompts,
            master_seed=fixture.cfg.master_seed)
        request = repair_notebook._request(tree, fixture.actor, fixture.prompts, audit, fixture.images[fixture.actor.problem_id])
        feedback = [{key: call[key] for key in ("group_id", "turn", "parent_sample_id", "feedback", "feedback_sha256")}
                    for call in tree.notebook["repair_collector"]["calls"]]
        self.assertEqual(request.messages, recovery._repair_note_messages(fixture.actor, fixture.prompts,
            tree, audit, feedback))
        payload = json.loads(request.messages[0]["content"][-1]["text"].split("<read_only_history>\n")[1].split(
            "\n</read_only_history>")[0])
        self.assertEqual(len(payload["attempts"]), 8)
        self.assertEqual([row["status"] for row in payload["attempts"][4:6]], ["INVALID_RESPONSE_CONTRACT", "CORRECT"])
        self.assertIsNone(payload["attempts"][4]["answer"])
        self.assertEqual(payload["attempts"][4]["raw"], malformed)
        self.assertEqual(payload["success_sample_id"], payload["attempts"][6]["sample_id"])
        self.assertEqual(audit["unavailable_result_ids"], [])
        self.assertEqual(audit["missing_result_ids"], [payload["attempts"][4]["sample_id"]])

