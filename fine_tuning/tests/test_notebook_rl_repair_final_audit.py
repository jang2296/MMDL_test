"""CPU fixtures for bounded supplied-history final semantics, not model learning."""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl.generators import circuit
from fine_tuning.notebook_rl.repair_contract import FINAL_VERSION, ORIGIN_VERSION, VERSION, RepairContractError, public_origin_inventory
from fine_tuning.notebook_rl.repair_credit import audit_final_note
from fine_tuning.notebook_rl.schema import Sample
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests.test_notebook_rl_circuit_trace import fixture_trace as circuit_trace, response as circuit_response
from fine_tuning.tests.test_notebook_rl_notebook_synthesis import graph_native, graph_response
from fine_tuning.tests.test_notebook_rl_repair_credit import joint, sample


class FinalRepairAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actor, self.item, self.verifier = graph_native("shortest")
        self.trace = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None,
                      "weights": [2, 2], "cost": 4}
        parent = sample(self.actor, "p", graph_response(self.item.slots, {**self.trace, "cost": 3}, "4"))
        child = self.retry("c", self.trace, parent)
        self.prepare([parent, child])

    def retry(self, name: str, trace: dict, parent: Sample, *, facts=None, answer="4", finish="stop") -> Sample:
        body = graph_response(self.item.slots if facts is None else facts, trace, answer)
        return sample(self.actor, name, joint(body, parent.sample_id + ":r1", intent="UNKNOWN"), parent, finish)

    def prepare(self, history: list[Sample]) -> None:
        self.history = history
        self.contracts = {row.sample_id: None if row.turn == 1 else VERSION for row in history}
        self.inventory = public_origin_inventory(history, contract_version=ORIGIN_VERSION,
                                                response_contracts=self.contracts)

    def ref(self, sample_id: str, rule="COST_SUM", locus: str | None = None) -> str:
        return next(row.ref_id for row in self.inventory.origins if row.sample_id == sample_id and row.rule == rule
                    and (locus is None or row.locus == locus))

    def claim(self, relation="CORRECTED", *, origin="p", later="c", rule="COST_SUM", locus=None,
              later_locus=None, repaired=None, blockers: list[str] | None = None) -> dict:
        result = {"origin_ref": self.ref(origin, rule, locus), "later_ref": self.ref(later, rule, later_locus or locus),
                  "relation": relation, "rule": rule,
                  "repaired_value": {"field": "cost", "value": 4} if relation == "CORRECTED" and repaired is None else repaired}
        if relation == "UNKNOWN":
            result.update(unknown_reason="DEPENDENCY_BLOCKED", blocked_by=blockers or [])
        return result

    def note(self, claims: list[dict] | None = None, *, attempts=None, success=None,
             uncertain: list[str] | None = None, lesson="human prose", finish="stop") -> Sample:
        if attempts is None:
            attempts = []
            correct = []
            for row in self.history:
                result = self.verifier.grade(self.actor.problem_id, row.text, row.finish_reason, "obs",
                                             response_contract=self.contracts[row.sample_id])
                attempts.append({"sample_id": row.sample_id, "outcome": result.status, "answer": result.extracted})
                if result.status == "CORRECT":
                    correct.append(row)
            if success is None and correct:
                turn = max(row.turn for row in correct)
                success = next(row.sample_id for row in correct if row.turn == turn)
        data = {"version": FINAL_VERSION, "scope": "BOUNDED_CLAIMS_FULL_HISTORY", "attempts": attempts,
                "success_sample_id": success or self.history[0].sample_id, "claims": claims or [],
                "uncertain_refs": uncertain or [], "lesson_text": lesson}
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return Sample("note", self.actor.problem_id, "diag-group", "DIAG", 1, None,
                      [77], [88], [-.1], finish, text, 3407, 0)

    def audit(self, note: Sample, *, inventory=None, contracts=None):
        return audit_final_note(note, self.history, self.actor, self.verifier,
            inventory=self.inventory if inventory is None else inventory,
            response_contracts=self.contracts if contracts is None else contracts)

    def test_corrected_claim_has_exact_raw_ownership_and_bounded_scope(self):
        note = self.note([self.claim()])
        audit = self.audit(note)
        self.assertTrue(audit.metadata_valid)
        self.assertTrue(audit.accurate_submitted_note)
        self.assertEqual(audit.witnesses[0].claim_verdict, "VERIFIED")
        self.assertEqual(sum(row.value for row in audit.unit_scores), 1)
        self.assertEqual(len(audit.unit_scores), self.inventory.fixed_units)
        owned = [span for row in audit.unit_scores for span in row.raw_spans]
        self.assertEqual(len(owned), 1)
        self.assertEqual(json.loads(note.text[owned[0].start:owned[0].end]), self.claim())
        self.assertNotIn("human prose", note.text[owned[0].start:owned[0].end])
        self.assertFalse(audit.source_bindings["full_tree_attachment_verified"])
        self.assertFalse(audit.source_bindings["token_text_correspondence_verified"])
        self.assertFalse(audit.source_bindings["lesson_semantics_verified"])
        self.assertEqual(audit.source_bindings["actual_attempts"][0]["outcome"], "CORRECT")
        self.assertEqual(audit.metrics["native_correction_count"], 1)
        self.assertEqual(audit.metrics["native_correction_coverage"], 1.0)

    def test_normal_source_false_accusation_is_refuted_not_dropped_from_precision(self):
        claim = self.claim(rule="PATH_ENDPOINTS", locus="nodes", repaired={"field": "nodes", "value": ["A", "B", "H"]})
        audit = self.audit(self.note([claim]))
        self.assertEqual(audit.witnesses[0].claim_verdict, "REFUTED")
        self.assertEqual(sum(row.value for row in audit.unit_scores), -1)
        self.assertFalse(audit.accurate_submitted_note)
        self.assertEqual(audit.metrics["submitted_claim_precision"], 0.0)
        self.assertEqual(audit.metrics["false_claim_count"], 1)

    def test_true_false_true_retains_signed_scores_and_distinct_coverage(self):
        parent = sample(self.actor, "p", graph_response({**self.item.slots, "edge[A,B]": "9"},
                                                       {**self.trace, "cost": 3}, "4"))
        self.prepare([parent, self.retry("c", self.trace, parent)])
        claims = [self.claim(), self.claim(rule="PATH_ENDPOINTS", locus="nodes",
                                           repaired={"field": "nodes", "value": ["A", "B", "H"]}),
                  self.claim(rule="OBSERVATION_MATCH", locus="edge[a,b]", repaired="2")]
        audit = self.audit(self.note(claims))
        self.assertEqual([row.claim_verdict for row in audit.witnesses], ["VERIFIED", "REFUTED", "VERIFIED"])
        self.assertEqual(sorted(row.value for row in audit.unit_scores if row.value), [-1, 1, 1])
        self.assertEqual(audit.metrics["submitted_claim_precision"], 2 / 3)
        self.assertTrue(audit.metrics["complete_known_issue_coverage"])
        self.assertEqual(audit.metrics["native_correction_count"], 2)
        self.assertFalse(audit.accurate_submitted_note)

    def test_persisted_positive_and_definitive_wrong_relation_negative(self):
        parent = self.history[0]
        self.prepare([parent, self.retry("c", {**self.trace, "cost": 3}, parent)])
        audit = self.audit(self.note([self.claim("PERSISTED")]))
        self.assertEqual(audit.witnesses[0].claim_verdict, "VERIFIED")
        self.assertEqual(sum(row.value for row in audit.unit_scores), 1)
        self.assertEqual(audit.metrics["covered_known_issue_count"], 2)
        self.assertEqual(audit.metrics["native_correction_count"], 0)
        self.assertEqual(self.audit(self.note([self.claim()])).witnesses[0].claim_verdict, "REFUTED")
        self.prepare([parent, self.retry("c", self.trace, parent)])
        self.assertEqual(self.audit(self.note([self.claim("PERSISTED")])).witnesses[0].claim_verdict, "REFUTED")

    def test_branch_comparison_requires_shared_recorded_ancestor_and_compliant_target(self):
        root = sample(self.actor, "root", graph_response(self.item.slots, self.trace, "4"))
        first = self.retry("p", {**self.trace, "cost": 3}, root)
        second = self.retry("c", self.trace, root)
        self.prepare([root, first, second])
        branch = self.claim("OTHER_BRANCH_COMPARISON")
        audit = self.audit(self.note([branch]))
        self.assertEqual(audit.witnesses[0].claim_verdict, "VERIFIED")
        self.assertEqual(audit.metrics["native_correction_count"], 0)
        self.assertEqual(audit.source_bindings["actual_success_sample_id"], "p")
        self.assertEqual(self.audit(self.note([self.claim()])).witnesses[0].claim_verdict, "STRUCTURAL_INVALID")
        self.prepare([root, first, self.retry("c", {**self.trace, "cost": 3}, root)])
        self.assertEqual(self.audit(self.note([self.claim("OTHER_BRANCH_COMPARISON")])).witnesses[0].claim_verdict, "REFUTED")
        self.prepare([sample(self.actor, "p", graph_response(self.item.slots, {**self.trace, "cost": 3}, "4")),
                      sample(self.actor, "c", graph_response(self.item.slots, self.trace, "4"))])
        self.assertEqual(self.audit(self.note([self.claim("OTHER_BRANCH_COMPARISON")])).witnesses[0].claim_verdict,
                         "STRUCTURAL_INVALID")

    def unknown_history(self) -> list[str]:
        parent = sample(self.actor, "p", graph_response(self.item.slots,
            {**self.trace, "weights": [3, 2], "cost": 5}, "4"))
        child = self.retry("c", {**self.trace, "nodes": ["B", "A", "H"], "weights": [3, 2], "cost": 5}, parent)
        self.prepare([parent, child])
        return [self.ref("c", "DIRECTED_EDGE", "nodes[0:2]")]

    def test_unknown_requires_exact_positive_violations_dependencies_and_raw_binding(self):
        blockers = self.unknown_history()
        claim = self.claim("UNKNOWN", rule="EDGE_WEIGHT", locus="weights[0]", blockers=blockers)
        audit = self.audit(self.note([claim]))
        self.assertEqual(audit.witnesses[0].claim_verdict, "VERIFIED")
        self.assertEqual(sum(row.value for row in audit.unit_scores), 1)
        self.assertGreater(audit.metrics["fresh_unknown_count"], 0)
        self.assertEqual(audit.metrics["covered_known_issue_count"], 1)
        self.assertGreater(audit.metrics["uncovered_known_issue_count"], 0)
        self.assertEqual(audit.metrics["native_correction_count"], 0)
        earlier = self.ref("c", "PATH_ENDPOINTS", "nodes")
        for refs in ([self.ref("p", "DIRECTED_EDGE", "nodes[0:2]")], [earlier, *blockers],
                     [*blockers, earlier], ["foreign:r1"]):
            bad = self.claim("UNKNOWN", rule="EDGE_WEIGHT", locus="weights[0]", blockers=refs)
            audited = self.audit(self.note([bad]))
            self.assertEqual(audited.witnesses[0].claim_verdict, "STRUCTURAL_INVALID")
            self.assertEqual(sum(row.value for row in audited.unit_scores), 0)
        with self.assertRaises(RepairContractError):
            self.audit(self.note([self.claim("UNKNOWN", rule="EDGE_WEIGHT", locus="weights[0]", blockers=[])]))

    def test_supported_unknown_definitive_contradiction_is_refuted(self):
        blockers = [self.ref("c", "DIRECTED_EDGE", "nodes[0:2]")]
        claim = self.claim("UNKNOWN", blockers=blockers)
        audit = self.audit(self.note([claim]))
        self.assertEqual(audit.witnesses[0].claim_verdict, "REFUTED")
        self.assertEqual(audit.witnesses[0].reason, "DESTINATION_CONTRADICTS_RELATION")
        self.assertEqual(sum(row.value for row in audit.unit_scores), -1)
        parent = sample(self.actor, "p", graph_response(self.item.slots, self.trace, "4"))
        self.prepare([parent, self.retry("c", {**self.trace, "cost": 3}, parent)])
        claim = self.claim("UNKNOWN", blockers=[self.ref("c", "DIRECTED_EDGE", "nodes[0:2]")])
        audit = self.audit(self.note([claim]))
        self.assertEqual(audit.witnesses[0].reason, "SOURCE_PROVEN_COMPLIANT")
        self.assertEqual(sum(row.value for row in audit.unit_scores), -1)

    def test_generic_observation_unknown_is_zero_not_a_false_accusation_penalty(self):
        parent = sample(self.actor, "p", graph_response({**self.item.slots, "edge[B,H]": "9"}, self.trace, "4"))
        child = self.retry("c", self.trace, parent, facts={**self.item.slots, "edge[B,H]": "???"})
        self.prepare([parent, child])
        claim = self.claim("UNKNOWN", rule="OBSERVATION_MATCH", locus="edge[b,h]",
                           blockers=[self.ref("c", "OBSERVATION_MATCH", "edge[a,b]")])
        audit = self.audit(self.note([claim]))
        self.assertEqual(audit.witnesses[0].claim_verdict, "UNVERIFIABLE")
        self.assertEqual(sum(row.value for row in audit.unit_scores), 0)
        body = graph_response(self.item.slots, self.trace, "4")
        body = body[:body.index("Trace:")] + "Final answer: 4"
        parent = sample(self.actor, "p", body)
        child = sample(self.actor, "c", joint(body, "p:r1", intent="UNKNOWN"), parent)
        self.prepare([parent, child])
        generic = self.claim("UNKNOWN", rule="TRACE_FORMAT", locus="solution",
                             blockers=[self.ref("c", "OBSERVATION_MATCH", "edge[a,b]")])
        audit = self.audit(self.note([generic]))
        self.assertEqual(audit.witnesses[0].claim_verdict, "UNVERIFIABLE")
        self.assertEqual(sum(row.value for row in audit.unit_scores), 0)

    def test_invalid_final_reference_kind_rule_and_order_never_expand_frozen_units(self):
        original = self.claim()
        for change in ({"origin_ref": "foreign:r1"}, {"later_ref": "note:r1"},
                       {"later_ref": self.ref("c", "OBSERVATION_MATCH", "edge[a,b]")},
                       {"rule": "DIRECTED_EDGE"},
                       {"origin_ref": original["later_ref"], "later_ref": original["origin_ref"]}):
            audit = self.audit(self.note([{**original, **change}]))
            self.assertEqual(audit.witnesses[0].claim_verdict, "STRUCTURAL_INVALID")
            self.assertEqual(len(audit.unit_scores), self.inventory.fixed_units)
            self.assertEqual(sum(row.value for row in audit.unit_scores), 0)
            self.assertEqual(audit.metrics["submitted_claim_count"], 1)
            self.assertEqual(audit.metrics["submitted_claim_precision"], 0.0)

    def test_operand_identity_mismatch_and_unsupported_positive_proof_do_not_become_refuted(self):
        parent = sample(self.actor, "p", graph_response(self.item.slots,
            {**self.trace, "weights": [3, 2], "cost": 5}, "4"))
        child = self.retry("c", {**self.trace, "nodes": ["A", "C", "H"], "weights": [4, 4], "cost": 8}, parent)
        self.prepare([parent, child])
        claim = self.claim(rule="EDGE_WEIGHT", locus="weights[0]", repaired={"field": "weights", "value": 4})
        self.assertEqual(self.audit(self.note([claim])).witnesses[0].claim_verdict, "STRUCTURAL_INVALID")
        self.setUp()
        with patch("fine_tuning.notebook_rl.repair_credit.operation_patch", return_value=None):
            self.assertEqual(self.audit(self.note([self.claim()])).witnesses[0].claim_verdict, "UNVERIFIABLE")
        with patch.object(synthesis, "_lesson", return_value={"verified": False}):
            self.assertEqual(self.audit(self.note([self.claim()])).witnesses[0].claim_verdict, "UNVERIFIABLE")
        mismatch = self.claim(repaired={"field": "cost", "value": 4.0})
        self.assertEqual(self.audit(self.note([mismatch])).witnesses[0].claim_verdict, "REFUTED")

    def test_finalized_uses_projected_stopped_output_with_correctness_irrelevant_and_reward_zero(self):
        parent = self.history[0]
        missing = parent.text.rsplit("\nFinal answer:", 1)[0]
        parent = replace(parent, text=missing)
        child = self.retry("c", self.trace, parent)
        child.text = joint(child.text.split("\nRevisedSolution:\n", 1)[1], "p:r1", intent="UNKNOWN")
        correct = sample(self.actor, "root-success", graph_response(self.item.slots, self.trace, "4"))
        child.text = child.text.replace("Final answer: 4", "Final answer: 999")
        self.prepare([parent, correct, child])
        claim = self.claim("FINALIZED", rule="FINALIZATION", locus="final_answer")
        with patch.object(synthesis, "_output_endpoint", side_effect=AssertionError("unprojected output helper used")):
            audit = self.audit(self.note([claim]))
        self.assertTrue(audit.accurate_submitted_note)
        self.assertEqual(audit.metrics["verified_output_count"], 1)
        self.assertEqual(audit.metrics["native_claim_count"], 0)
        self.assertEqual(audit.metrics["native_correction_count"], 0)
        self.assertEqual(sum(row.value for row in audit.unit_scores), 0)
        self.prepare([parent, correct, replace(child, finish_reason="length")])
        self.assertEqual(self.audit(self.note([self.claim("FINALIZED", rule="FINALIZATION", locus="final_answer")])).witnesses[0].claim_verdict,
                         "UNVERIFIABLE")

    def test_incorrect_metadata_zeroes_applied_scores_but_retains_underlying_signed_verdicts(self):
        note = self.note([self.claim()])
        original = json.loads(note.text)
        variants = []
        for change in ("answer", "outcome", "order", "missing", "success", "uncertain"):
            value = copy.deepcopy(original)
            if change == "answer":
                value["attempts"][0]["answer"] = "999"
            elif change == "outcome":
                value["attempts"][0]["outcome"] = "INCORRECT"
            elif change == "order":
                value["attempts"].reverse()
            elif change == "missing":
                value["attempts"] = value["attempts"][1:]
            elif change == "success":
                value["success_sample_id"] = "p"
            else:
                value["uncertain_refs"] = ["foreign:r1"]
            variants.append(replace(note, text=json.dumps(value)))
        for invalid in variants:
            audit = self.audit(invalid)
            self.assertFalse(audit.metadata_valid)
            self.assertFalse(audit.accurate_submitted_note)
            self.assertEqual(audit.witnesses[0].claim_verdict, "VERIFIED")
            self.assertEqual(sum(row.value for row in audit.proposed_unit_scores), 1)
            self.assertEqual(sum(row.value for row in audit.unit_scores), 0)
        false_claim = self.claim(rule="PATH_ENDPOINTS", locus="nodes", repaired={"field": "nodes", "value": ["A", "B", "H"]})
        value = json.loads(self.note([false_claim]).text)
        value["attempts"][0]["outcome"] = "INCORRECT"
        audit = self.audit(replace(note, text=json.dumps(value)))
        self.assertEqual(audit.witnesses[0].claim_verdict, "REFUTED")
        self.assertEqual(sum(row.value for row in audit.proposed_unit_scores), -1)
        self.assertEqual(sum(row.value for row in audit.unit_scores), 0)
        uncertain = [row.ref_id for row in self.inventory.origins if row.rule != "COST_SUM"][:2]
        self.assertFalse(self.audit(self.note([self.claim()], uncertain=list(reversed(uncertain)))).metadata_valid)

    def test_empty_claims_zero_denominators_t1_success_and_latest_first_ordinal_selection(self):
        audit = self.audit(self.note([]))
        self.assertIsNone(audit.metrics["submitted_claim_precision"])
        self.assertFalse(audit.accurate_submitted_note)
        self.assertEqual(audit.metrics["submitted_claim_count"], 0)
        uncertain = self.audit(self.note([], uncertain=[self.ref("p")]))
        self.assertEqual(uncertain.metrics["declared_uncertain_count"], 1)
        self.assertEqual(uncertain.metrics["covered_known_issue_count"], 0)
        root = sample(self.actor, "root", graph_response(self.item.slots, self.trace, "4"))
        self.prepare([root])
        audit = self.audit(self.note([]))
        self.assertTrue(audit.metadata_valid)
        self.assertIsNone(audit.metrics["known_issue_coverage"])
        self.assertIsNone(audit.metrics["complete_known_issue_coverage"])
        self.assertEqual(audit.source_bindings["actual_success_sample_id"], "root")
        self.prepare([root, self.retry("a", self.trace, root), self.retry("b", self.trace, root)])
        self.assertEqual(self.audit(self.note([])).source_bindings["actual_success_sample_id"], "a")
        self.assertFalse(self.audit(self.note([], success="b")).metadata_valid)
        self.assertFalse(self.audit(self.note([], success="root")).metadata_valid)

    def test_length_local_signed_claims_and_both_existing_diag_parent_forms(self):
        note = self.note([self.claim()], finish="length")
        audit = self.audit(note)
        self.assertTrue(audit.metadata_valid)
        self.assertEqual(sum(row.value for row in audit.unit_scores), 1)
        self.assertTrue(audit.source_bindings["whole_response_truncated"])
        parented = replace(note, parent_sample_id="c", turn=2)
        self.assertTrue(self.audit(parented).metadata_valid)
        for invalid in (replace(note, kind="T1"), replace(note, group_id=self.history[0].group_id),
                        replace(note, sample_id="p"), replace(note, policy_version=1),
                        replace(note, parent_sample_id="c", turn=1), replace(note, turn=2),
                        replace(note, parent_sample_id="foreign")):
            with self.assertRaises(ValueError):
                self.audit(invalid)

    def test_mutation_callbacks_cannot_change_snapshots_and_cached_verdicts_are_ignored(self):
        note = self.note([self.claim()])
        baseline = self.audit(note)
        for row in [*self.history, note]:
            row.result = object()  # type: ignore[assignment]
            row.reward, row.diag_reward = float("nan"), float("inf")
        self.assertEqual(self.audit(note), baseline)
        native = synthesis.audit_native_sample

        def mutate(*args, **kwargs):
            result = native(*args, **kwargs)
            self.history[0].text = "mutated caller raw"
            self.history[1].prompt_token_ids.append(999)
            self.contracts["p"] = VERSION
            note.text = "mutated note"
            return result

        with patch.object(synthesis, "audit_native_sample", side_effect=mutate):
            self.assertEqual(self.audit(note), baseline)

    def test_inventory_contract_raw_and_no_success_boundaries_are_integrity_failures(self):
        note = self.note([self.claim()])
        for inventory in (replace(self.inventory, sha256="0" * 64),
                          replace(self.inventory, parent_sample_id="p")):
            with self.assertRaises(ValueError):
                self.audit(note, inventory=inventory)
        with self.assertRaises(ValueError):
            self.audit(note, contracts={**self.contracts, "note": VERSION})
        with self.assertRaises(RepairContractError):
            self.audit(replace(note, text=note.text[:-1]))
        self.history = [replace(row, text=row.text.replace("Final answer: 4", "Final answer: 999")) for row in self.history]
        self.inventory = public_origin_inventory(self.history, contract_version=ORIGIN_VERSION, response_contracts=self.contracts)
        with self.assertRaisesRegex(ValueError, "no independently correct"):
            self.audit(note)

    def test_malformed_prior_has_only_fresh_syntax_outcome_and_no_native_proof(self):
        root = self.history[0]
        malformed = self.retry("bad", self.trace, root)
        malformed.text = "ErrorCheck: {broken}\n" + graph_response(self.item.slots, self.trace, "4")
        self.prepare([root, malformed])
        attempts: list[dict] = [{"sample_id": "p", "outcome": "CORRECT", "answer": "4"},
                    {"sample_id": "bad", "outcome": "INVALID_RESPONSE_CONTRACT", "answer": None}]
        note = self.note([], attempts=attempts, success="p")
        audit = self.audit(note)
        self.assertTrue(audit.metadata_valid)
        self.assertFalse(audit.accurate_submitted_note)
        self.assertIsNone(audit.source_bindings["native_audits"]["bad"])
        actual = audit.source_bindings["actual_attempts"][1]
        self.assertEqual((actual["outcome_available"], actual["outcome"], actual["answer"]),
                         (True, "INVALID_RESPONSE_CONTRACT", None))
        self.assertEqual(sum(row.value for row in audit.unit_scores), 0)
        self.assertEqual(audit.metrics["native_correction_count"], 0)
        for forged in (attempts[:1], [attempts[0], attempts[1] | {"outcome": "UNVERIFIABLE"}],
                       [attempts[0], attempts[1] | {"answer": "4"}],
                       [attempts[0] | {"outcome": "INVALID_RESPONSE_CONTRACT", "answer": None}, attempts[1]]):
            rejected = self.audit(self.note([], attempts=forged, success="p"))
            self.assertFalse(rejected.metadata_valid)
            self.assertIn("ATTEMPT_METADATA_DIFFERS", rejected.metadata_errors)
            self.assertFalse(any(row.value for row in rejected.unit_scores))

    def test_forged_projection_failure_for_valid_raw_is_an_integrity_error(self):
        note = self.note([])
        with patch.object(synthesis, "audit_native_sample", side_effect=RepairContractError("INVALID_JSON")):
            with self.assertRaisesRegex(ValueError, "projection failure differs"):
                self.audit(note)

    def test_intrinsic_circuit_null_dependency_does_not_gain_unknown_reward(self):
        problem = circuit.generate(2, 20261021, "sealed", "obs")
        self.actor = problem.actor
        self.verifier = Verifier.__new__(Verifier)
        self.verifier.actors, self.verifier.items = {self.actor.problem_id: self.actor}, {self.actor.problem_id: problem.verifier}
        original = circuit_trace(problem)
        wrong = copy.deepcopy(original)
        wrong["input"][1] = str(Fraction(wrong["input"][1]) + 1)
        parent = sample(self.actor, "p", circuit_response(problem, wrong))
        target = copy.deepcopy(original)
        target["ohm"][0][5] = None
        child = sample(self.actor, "c", joint(circuit_response(problem, target), "p:r1", intent="UNKNOWN"), parent)
        self.prepare([parent, child])
        claim = self.claim("UNKNOWN", rule="CIRCUIT_INPUT_CURRENT_LINK", locus="input[1]",
                           blockers=[self.ref("c", "CIRCUIT_OHM", "ohm[0]")])
        audit = self.audit(self.note([claim]))
        self.assertEqual(audit.witnesses[0].claim_verdict, "UNVERIFIABLE")
        self.assertEqual(audit.witnesses[0].reason, "BLOCKER_NOT_PROVEN_VIOLATED")
        self.assertEqual(sum(row.value for row in audit.unit_scores), 0)


if __name__ == "__main__":
    unittest.main()
