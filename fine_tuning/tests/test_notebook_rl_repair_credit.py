"""Authored CPU verifier evidence for local repair; no training or model claims."""
from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import replace
from fractions import Fraction
from typing import Any
from unittest.mock import patch

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl.generators import circuit
from fine_tuning.notebook_rl.repair_contract import (
    ORIGIN_VERSION, VERSION, RawSpan, RepairContractError, parse_retry_sections, public_origin_inventory,
)
from fine_tuning.notebook_rl.repair_credit import AUDIT_VERSION, audit_online_repair, operation_patch
from fine_tuning.notebook_rl.schema import Sample, to_json
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests.test_notebook_rl_circuit_trace import fixture_trace as circuit_trace, response as circuit_response
from fine_tuning.tests.test_notebook_rl_notebook_synthesis import graph_native, graph_response


def sample(actor: Any, name: str, text: str, parent: Sample | None = None, finish: str = "stop") -> Sample:
    turn = 1 if parent is None else parent.turn + 1
    prompt = [1, 2] if parent is None else parent.prompt_token_ids + parent.completion_token_ids + [900 + turn]
    return Sample(name, actor.problem_id, f"g{turn}", f"T{turn}", turn,
                  parent.sample_id if parent else None, prompt, [100 + turn], [-.1], finish, text, 3407, 0)


def joint(body: str, ref: str, *, rule: str = "COST_SUM", proposed: Any = None,
          intent: str = "RECHECK_OPERATION", uncertain: list[str] | None = None) -> str:
    check = {"version": VERSION, "claims": [{"origin_ref": ref, "intent": intent,
              "rule": rule, "proposed_value": proposed}], "uncertain_refs": uncertain or []}
    return "ErrorCheck: " + json.dumps(check, ensure_ascii=False, separators=(",", ":")) + "\nRevisedSolution:\n" + body


class OnlineRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actor, self.item, self.verifier = graph_native("shortest")
        self.trace = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None,
                      "weights": [2, 2], "cost": 4}
        wrong = {**self.trace, "cost": 3}
        self.parent = sample(self.actor, "parent", graph_response(self.item.slots, wrong, "4"))

    def inventory(self, history: list[Sample] | None = None, contracts: dict[str, str | None] | None = None,
                  parent: str | None = None):
        history = [self.parent] if history is None else history
        contracts = {row.sample_id: None for row in history} if contracts is None else contracts
        return public_origin_inventory(history, contract_version=ORIGIN_VERSION,
            response_contracts=contracts, parent_sample_id=parent or history[-1].sample_id)

    def origin(self, rule: str = "COST_SUM", *, inventory=None):
        inventory = self.inventory() if inventory is None else inventory
        return next(row for row in inventory.origins if row.rule == rule)

    def child(self, *, body: str | None = None, ref: str | None = None, proposed: Any = None,
              rule: str = "COST_SUM", intent: str = "RECHECK_OPERATION", finish: str = "stop",
              uncertain: list[str] | None = None) -> Sample:
        body = graph_response(self.item.slots, self.trace, "4") if body is None else body
        ref = self.origin().ref_id if ref is None else ref
        proposed = {"field": "cost", "value": 4} if proposed is None and intent != "UNKNOWN" else proposed
        return sample(self.actor, "child", joint(body, ref, rule=rule, proposed=proposed,
                                                intent=intent, uncertain=uncertain), self.parent, finish)

    def audit(self, child: Sample, *, history: list[Sample] | None = None, inventory=None,
              contracts: dict[str, str | None] | None = None, sections=None):
        history = [self.parent] if history is None else history
        contracts = ({**{row.sample_id: None for row in history}, child.sample_id: VERSION}
                     if contracts is None else contracts)
        inventory = self.inventory(history, {key: contracts[key] for key in contracts if key != child.sample_id},
                                   child.parent_sample_id) if inventory is None else inventory
        sections = parse_retry_sections(child.text, contract_version=VERSION) if sections is None else sections
        return audit_online_repair(history, child, self.actor, self.verifier,
                                   inventory=inventory, response_contracts=contracts, sections=sections)

    def test_correct_final_parent_with_native_error_can_repair_and_preserves_fixed_units(self):
        child = self.child()
        self.assertEqual(self.verifier.grade(self.actor.problem_id, self.parent.text, "stop", "obs").status, "CORRECT")
        audit = self.audit(child)
        self.assertEqual(audit.witnesses[0].claim_verdict, "VERIFIED")
        self.assertEqual(audit.witnesses[0].relation, "CORRECTED")
        self.assertEqual(sum(row.value for row in audit.unit_scores), 1)
        self.assertEqual(len(audit.unit_scores), self.inventory().fixed_units)
        unit = next(row for row in audit.unit_scores if row.value)
        self.assertEqual(unit.verifier_version, AUDIT_VERSION)
        self.assertEqual(child.text[unit.raw_spans[0].start:unit.raw_spans[0].end],
                         json.dumps(json.loads(child.text[len("ErrorCheck: "):].split("\n", 1)[0])["claims"][0],
                                    ensure_ascii=False, separators=(",", ":")))
        self.assertTrue(all(row.raw_spans == () for row in audit.unit_scores if row.reason == "NOT_SUBMITTED"))
        self.assertEqual(audit.source_bindings["actor_sha256"], hashlib.sha256(to_json(self.actor).encode()).hexdigest())
        for evidence in audit.source_bindings["native_audits"].values():
            self.assertTrue(evidence["result_verifier_version"])
            self.assertTrue(evidence["trace_verifier_version"])
        self.assertFalse(audit.source_bindings["token_text_correspondence_verified"])

    def test_wrong_final_and_completed_length_child_still_allow_verified_local_unit(self):
        body = graph_response(self.item.slots, self.trace, "999")
        for finish in ("stop", "length"):
            child = self.child(body=body, finish=finish)
            audit = self.audit(child)
            self.assertEqual(audit.witnesses[0].claim_verdict, "VERIFIED")
            self.assertEqual(audit.source_bindings["whole_response_truncated"], finish == "length")
            self.assertEqual(audit.source_bindings["semantic_scope"], "native_rule_local_repair")
            self.assertFalse(audit.source_bindings["all_reasoning_verified"])
            self.assertFalse(audit.source_bindings["module_causality_certified"])
            expected = "TRUNCATED" if finish == "length" else "INCORRECT"
            self.assertEqual(self.verifier.grade(self.actor.problem_id, child.text, finish, "obs",
                                                response_contract=VERSION).status, expected)

    def test_observation_patch_requires_exact_single_emitted_string_and_compliance(self):
        wrong_facts = {**self.item.slots, "edge[A,B]": "9"}
        self.parent = sample(self.actor, "parent", graph_response(wrong_facts, self.trace, "4"))
        ref = next(row.ref_id for row in self.inventory().origins if row.kind == "OBSERVATION" and row.locus == "edge[a,b]")
        for proposed, value in (("2", 1), (" 2\t", 1), ("2 ohm", 0), ("02", 0)):
            child = self.child(ref=ref, rule="OBSERVATION_MATCH", intent="RECHECK_OBSERVATION", proposed=proposed)
            self.assertEqual(sum(row.value for row in self.audit(child).unit_scores), value)
        body = graph_response(self.item.slots, self.trace, "4").replace("\nSolution:", "\nedge[A,B] = ???\nSolution:")
        child = self.child(body=body, ref=ref, rule="OBSERVATION_MATCH", intent="RECHECK_OBSERVATION", proposed="2")
        self.assertEqual(self.audit(child).witnesses[0].reason, "CHILD_CHECK_MISSING_OR_AMBIGUOUS")

    def test_unknown_intent_source_normality_and_intent_only_never_earn(self):
        child = self.child(intent="UNKNOWN")
        self.assertEqual(self.audit(child).witnesses[0].reason, "UNKNOWN_INTENT")
        child = self.child(body="Solution:\nFinal answer: 4")
        self.assertEqual(sum(row.value for row in self.audit(child).unit_scores), 0)
        self.parent = sample(self.actor, "parent", graph_response(self.item.slots, self.trace, "4"))
        audit = self.audit(self.child())
        self.assertEqual(audit.witnesses[0].reason, "SOURCE_NOT_VERIFIED_VIOLATED")
        self.assertEqual(sum(row.value for row in audit.unit_scores), 0)

    def test_patch_is_tied_to_rule_locus_and_json_type_without_numeric_conversion(self):
        for proposed in ({"field": "weights", "value": 4}, {"field": "cost", "value": 4.0},
                         {"field": "cost", "value": "4"}, {"field": "cost", "value": 3}):
            audit = self.audit(self.child(proposed=proposed))
            self.assertEqual(audit.witnesses[0].reason, "PROPOSED_PATCH_DIFFERS_FROM_EMITTED")
            self.assertEqual(sum(row.value for row in audit.unit_scores), 0)

    def test_foreign_sibling_future_and_wrong_rule_kind_are_structural_zero(self):
        sibling = replace(self.parent, sample_id="sibling")
        sibling_inventory = public_origin_inventory([self.parent, sibling], contract_version=ORIGIN_VERSION,
                                                    response_contracts={"parent": None, "sibling": None})
        sibling_ref = next(row.ref_id for row in sibling_inventory.origins if row.sample_id == "sibling" and row.rule == "COST_SUM")
        for ref in ("foreign:r1", sibling_ref, "child:r1"):
            child = self.child(ref=ref)
            audit = self.audit(child, history=[self.parent, sibling])
            self.assertEqual(audit.witnesses[0].claim_verdict, "STRUCTURAL_INVALID")
            self.assertEqual(len(audit.unit_scores), self.inventory().fixed_units)
            self.assertNotIn(ref, [row.origin_ref for row in audit.unit_scores])
            self.assertEqual(sum(row.value for row in audit.unit_scores), 0)
        for child in (self.child(rule="DIRECTED_EDGE"), self.child(intent="RECHECK_OBSERVATION", proposed="4")):
            self.assertEqual(self.audit(child).witnesses[0].reason, "ORIGIN_RULE_OR_KIND_MISMATCH")

    def test_uncertain_refs_require_inventory_membership_and_order_and_receive_no_credit(self):
        inventory = self.inventory()
        eligible = [row.ref_id for row in inventory.origins if row.rule != "COST_SUM"]
        valid = self.audit(self.child(uncertain=eligible[:2]))
        self.assertTrue(valid.source_bindings["uncertain_refs_valid"])
        self.assertEqual(sum(row.value for row in valid.unit_scores), 1)
        for refs in (["foreign:r1"], list(reversed(eligible[:2]))):
            audit = self.audit(self.child(uncertain=refs))
            self.assertFalse(audit.source_bindings["uncertain_refs_valid"])
            self.assertEqual(audit.witnesses[0].reason, "INVALID_UNCERTAIN_REFS")
            self.assertEqual(sum(row.value for row in audit.unit_scores), 0)

    def test_same_array_index_with_different_operands_and_multiple_revisits_fail_closed(self):
        parent_trace = {**self.trace, "weights": [3, 2], "cost": 5}
        self.parent = sample(self.actor, "parent", graph_response(self.item.slots, parent_trace, "4"))
        ref = self.origin("EDGE_WEIGHT").ref_id
        changed = {**self.trace, "nodes": ["A", "C", "H"], "weights": [4, 4], "cost": 8}
        child = self.child(body=graph_response(self.item.slots, changed, "8"), ref=ref,
                           rule="EDGE_WEIGHT", proposed={"field": "weights", "value": 4})
        self.assertEqual(self.audit(child).witnesses[0].reason, "CHILD_CHECK_MISSING_OR_AMBIGUOUS")
        repeated = {**self.trace, "nodes": ["A", "B", "A", "B", "H"], "weights": [2, 2, 2, 2], "cost": 8}
        child = self.child(body=graph_response(self.item.slots, repeated, "8"), ref=ref,
                           rule="EDGE_WEIGHT", proposed={"field": "weights", "value": 2})
        self.assertEqual(self.audit(child).witnesses[0].reason, "CHILD_CHECK_MISSING_OR_AMBIGUOUS")

    def test_malformed_prior_source_is_unknown_without_legacy_fallback(self):
        root = self.parent
        malformed = sample(self.actor, "parent", "ErrorCheck: {broken}\n" + graph_response(self.item.slots, self.trace, "4"), root)
        root = replace(root, sample_id="root")
        malformed.parent_sample_id = root.sample_id
        contracts = {"root": None, "parent": VERSION}
        inventory = self.inventory([root, malformed], contracts, "parent")
        origin = next(row for row in inventory.origins if row.sample_id == "parent" and row.rule == "TRACE_FORMAT")
        child = sample(self.actor, "child", joint(graph_response(self.item.slots, self.trace, "4"), origin.ref_id,
                       rule="TRACE_FORMAT", proposed={"field": "cost", "value": 4}), malformed)
        audit = self.audit(child, history=[root, malformed], inventory=inventory,
                           contracts={**contracts, "child": VERSION})
        self.assertTrue(audit.witnesses[0].reason.startswith("MALFORMED_SOURCE_"))
        self.assertNotIn("parent", audit.source_bindings["native_audits"])
        self.assertEqual(sum(row.value for row in audit.unit_scores), 0)

    def test_incomplete_current_json_and_trace_do_not_become_positive_units(self):
        child = self.child()
        sections = parse_retry_sections(child.text, contract_version=VERSION)
        incomplete = replace(child, text='ErrorCheck: {"version":', finish_reason="length")
        with self.assertRaises(RepairContractError):
            self.audit(incomplete, sections=sections)
        child = self.child(body='Solution:\nTrace: {"kind":"shortest",', finish="length")
        self.assertEqual(sum(row.value for row in self.audit(child).unit_scores), 0)

    def test_stored_result_reward_and_diag_are_ignored_but_wire_inputs_remain_strict(self):
        child = self.child()
        baseline = self.audit(child)
        self.parent.result = child.result = object()  # type: ignore[assignment]
        self.parent.reward = child.reward = float("nan")
        self.parent.diag_reward = child.diag_reward = float("inf")
        self.assertEqual(baseline, self.audit(child))
        with self.assertRaises(ValueError):
            self.audit(replace(child, sampler_logprobs=[float("nan")]))

    def test_inventory_raw_sections_contract_and_ancestry_mutations_raise_before_private_checks(self):
        contracts: Any
        child, inventory = self.child(), self.inventory()
        forged = [replace(inventory, sha256="0" * 64),
            replace(inventory, origins=(replace(inventory.origins[0], raw_sha="0" * 64), *inventory.origins[1:])),
            replace(inventory, sources=(replace(inventory.sources[0], solution_sha="0" * 64),)),
            replace(inventory, origins=(replace(inventory.origins[0], occurrence_spans=(RawSpan(0, 1),)), *inventory.origins[1:]))]
        for value in forged:
            with patch.object(synthesis, "audit_native_sample") as native:
                with self.assertRaises(ValueError):
                    self.audit(child, inventory=value)
                native.assert_not_called()
        sections = parse_retry_sections(child.text, contract_version=VERSION)
        with self.assertRaises(RepairContractError):
            self.audit(child, sections=replace(sections, raw_sha="0" * 64))
        for contracts in ({"parent": None}, {"parent": VERSION, "child": VERSION},
                          {"parent": None, "child": None}, {"parent": None, "child": VERSION, "future": VERSION}):
            with self.assertRaises(ValueError):
                self.audit(child, inventory=inventory, contracts=contracts)
        for changed in (replace(child, prompt_token_ids=[999]), replace(child, policy_version=1),
                        replace(child, parent_sample_id="future"), replace(child, problem_id="foreign"),
                        replace(child, kind="T3"), replace(child, turn=3), replace(child, sample_id="parent")):
            with self.assertRaises(ValueError):
                self.audit(changed, inventory=inventory, sections=sections)
        changed_parent = replace(self.parent, text=self.parent.text.replace('"cost":3', '"cost":5'))
        with self.assertRaises(ValueError):
            self.audit(child, history=[changed_parent], inventory=inventory)

    def test_every_retry_prefix_is_checked_and_mutable_caller_state_is_detached(self):
        child, inventory = self.child(), self.inventory()
        contracts = {"parent": None, "child": VERSION}
        baseline = self.audit(child, inventory=inventory, contracts=contracts)
        native = synthesis.audit_native_sample

        def mutate(*args, **kwargs):
            report = native(*args, **kwargs)
            self.parent.completion_token_ids.append(999)
            child.text = "mutated caller raw"
            contracts["parent"] = VERSION
            return report

        with patch.object(synthesis, "audit_native_sample", side_effect=mutate):
            audited = self.audit(child, inventory=inventory, contracts=contracts)
        self.assertEqual(audited, baseline)
        self.assertEqual(audited.source_bindings["response_contracts"]["parent"], None)
        root = sample(self.actor, "root", graph_response(self.item.slots, self.trace, "4"))
        prior = sample(self.actor, "prior", joint(graph_response(self.item.slots, self.trace, "4"), "root:r1",
                                                 proposed=None, intent="UNKNOWN"), root)
        prior.prompt_token_ids[0] = 999
        next_child = sample(self.actor, "next", joint(graph_response(self.item.slots, self.trace, "4"), "root:r1",
                                                      proposed=None, intent="UNKNOWN"), prior)
        valid_inventory = public_origin_inventory([root, prior], contract_version=ORIGIN_VERSION,
            response_contracts={"root": None, "prior": VERSION}, parent_sample_id="prior")
        with self.assertRaisesRegex(ValueError, "prefix"):
            self.audit(next_child, history=[root, prior], inventory=valid_inventory,
                       contracts={"root": None, "prior": VERSION, "next": VERSION})

    def test_circuit_local_ohm_repair_preserves_rationals_without_certifying_physics(self):
        problem = circuit.generate(2, 20261021, "sealed", "obs")
        self.actor = problem.actor
        self.verifier = Verifier.__new__(Verifier)
        self.verifier.actors, self.verifier.items = {problem.actor.problem_id: problem.actor}, {problem.actor.problem_id: problem.verifier}
        corrected = circuit_trace(problem)
        corrected["nodes"][-1][1] = "-1"
        wrong = copy.deepcopy(corrected)
        wrong["ohm"][0][5] = str(Fraction(wrong["ohm"][0][5]) + 1)
        self.parent = sample(self.actor, "parent", circuit_response(problem, wrong))
        ref = next(row.ref_id for row in self.inventory().origins if row.rule == "CIRCUIT_OHM" and row.locus == "ohm[0]")
        proposal = operation_patch(corrected, "CIRCUIT_OHM", "ohm[0]")
        assert proposal is not None
        child = self.child(body=circuit_response(problem, corrected), ref=ref, rule="CIRCUIT_OHM", proposed=proposal)
        audit = self.audit(child)
        self.assertEqual(sum(row.value for row in audit.unit_scores), 1)
        self.assertFalse(audit.source_bindings["all_reasoning_verified"])
        report = self.verifier.trace(self.actor.problem_id, child.text, "stop", response_contract=VERSION)
        self.assertTrue(any(row["rule"] == "CIRCUIT_NODE_VOLTAGE" for row in report["issues"]))
        changed = copy.deepcopy(proposal)
        rational = Fraction(changed["value"][5])
        changed["value"][5] = f"{rational.numerator * 2}/{rational.denominator * 2}"
        child = self.child(body=circuit_response(problem, corrected), ref=ref, rule="CIRCUIT_OHM", proposed=changed)
        self.assertEqual(self.audit(child).witnesses[0].reason, "PROPOSED_PATCH_DIFFERS_FROM_EMITTED")


class OperationPatchTests(unittest.TestCase):
    def test_every_supported_family_maps_the_exact_authored_field_piece(self):
        traces: list[tuple[dict, list[tuple[str, str, str, Any]]]] = [
            ({"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 4},
             [("PATH_ENDPOINTS", "nodes", "nodes", ["A", "B", "H"]),
              ("MINIMALITY", "nodes", "nodes", ["A", "B", "H"]),
              ("DIRECTED_EDGE", "nodes[0:2]", "nodes", ["A", "B"]),
              ("EDGE_WEIGHT", "weights[1]", "weights", 2), ("COST_SUM", "cost", "cost", 4)]),
            ({"kind": "bfs", "steps": [{"node": "A", "new": ["B"]}]},
             [("BFS_FIFO", "steps[0].node", "steps", {"node": "A", "new": ["B"]}),
              ("BFS_NEW_NEIGHBORS", "steps[0].new", "steps", {"node": "A", "new": ["B"]}),
              ("BFS_COMPLETENESS", "steps", "steps", [{"node": "A", "new": ["B"]}])]),
            ({"kind": "tree", "traversal": "preorder", "events": [{"op": "enter", "node": "A"},
                                                                     {"op": "child", "parent": "A", "side": "L", "node": "B"}]},
             [("TREE_START", "events[0]", "events", {"op": "enter", "node": "A"}),
              ("TREE_CHILD_EDGE", "events[1]", "events", {"op": "child", "parent": "A", "side": "L", "node": "B"}),
              ("TREE_EVENT_ORDER", "events[0]", "events", {"op": "enter", "node": "A"}),
              ("TREE_COMPLETENESS", "events", "events", None)]),
            ({"kind": "dfa", "states": ["q0", "q1"], "decision": "reject"},
             [("DFA_START", "states[0]", "states", "q0"),
              ("DFA_TRANSITION", "states[0:2]", "states", ["q0", "q1"]),
              ("DFA_COMPLETE_WORD", "states", "states", ["q0", "q1"]),
              ("DFA_ACCEPTANCE", "decision", "decision", "reject")])]
        for trace, cases in traces:
            for rule, locus, field, value in cases:
                expected = trace["events"] if rule == "TREE_COMPLETENESS" else value
                self.assertEqual(operation_patch(trace, rule, locus), {"field": field, "value": expected})
            self.assertIsNone(operation_patch(trace, "NOT_A_RULE", "nodes"))
            self.assertIsNone(operation_patch(trace, "RESPONSE_SCOPE", "response"))
        self.assertIsNone(operation_patch({"kind": "shortest", "nodes": [], "weights": [], "cost": True}, "COST_SUM", "cost"))

    def test_circuit_row_input_query_and_coverage_values_are_lossless_and_detached(self):
        problem = circuit.generate(2, 20261021, "sealed", "obs")
        trace = circuit_trace(problem)
        from fine_tuning.notebook_rl.circuit_trace import inventory
        for rule, locus in inventory(trace):
            if rule == "RESPONSE_SCOPE":
                self.assertIsNone(operation_patch(trace, rule, locus))
                continue
            field = locus.split("[", 1)[0]
            expected = trace[field]
            if "[" in locus and field not in ("input", "query"):
                expected = expected[int(locus.split("[", 1)[1][:-1])]
            self.assertEqual(operation_patch(trace, rule, locus), {"field": field, "value": expected})
        patch_value = operation_patch(trace, "CIRCUIT_OHM", "ohm[0]")
        assert patch_value is not None
        patch_value["value"][5] = "mutated"
        self.assertNotEqual(trace["ohm"][0][5], "mutated")


if __name__ == "__main__":
    unittest.main()
