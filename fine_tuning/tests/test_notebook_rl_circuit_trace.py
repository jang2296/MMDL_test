"""Exact CPU fixtures certify trace checks, never model quality or learning."""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

from fine_tuning.notebook_rl import circuit_trace, reasoning_trace
from fine_tuning.notebook_rl.generators import circuit
from fine_tuning.notebook_rl.generators.common import GeneratedProblem


def fixture_trace(problem) -> dict:
    world = problem.verifier.accept["circuit_world"]
    solution, second = circuit._solutions(world)
    assert solution == second
    nodes = solution["node_voltages_V"]
    components = [comp for comp in circuit._components(world) if comp["kind"] == "RESISTOR"]
    currents = solution["resistor_currents_A"]
    quantity, target = problem.verifier.accept["circuit_query"]
    value = (currents[target] if quantity == "current" else nodes[target] if quantity == "voltage"
             else solution["equivalent_R_ohm"])
    return {"kind": "circuit_ladder", "nodes": [[node, str(value)] for node, value in nodes.items()],
            "ohm": [[comp["name"], comp["a"], comp["b"], comp["value"],
                     str(nodes[comp["a"]] - nodes[comp["b"]]), str(currents[comp["name"]])]
                    for comp in components],
            "kcl": [[node, [[comp["name"], 1 if comp["a"] == node else -1]
                             for comp in components if node in (comp["a"], comp["b"])], "0"]
                    for node in list(nodes)[len(world["sources"]) + 1:]],
            "input": [str(sum(world["sources"])), str(solution["source_current_A"]),
                      str(solution["equivalent_R_ohm"])],
            "query": [quantity, target, str(value), {"current": "A", "voltage": "V", "resistance": "ohm"}[quantity]]}


def response(problem, trace: dict) -> str:
    facts = "\n".join(f"{key} = {value}" for key, value in problem.verifier.slots.items())
    return (f"Observed facts:\n{facts}\nSolution:\nTrace: "
            + json.dumps(trace, separators=(",", ":")) + f"\nFinal answer: {problem.verifier.gold}")


class CircuitTraceTests(unittest.TestCase):
    problems: list[GeneratedProblem]
    one_source_two_stage: GeneratedProblem
    problem: GeneratedProblem

    @classmethod
    def setUpClass(cls):
        # The same ten sealed topology/index contracts used by local_ladder.
        cls.problems = [circuit.generate(index, 20261021, "sealed", "obs")
                        for index in (2, 5, 8, 10, 11, 13, 14, 16, 17, 20)]
        cls.one_source_two_stage = circuit.generate(1, 20261021, "sealed", "obs")
        cls.problem = cls.problems[0]

    def audit(self, problem=None, trace=None, *, text=None, item=None):
        problem = self.problem if problem is None else problem
        trace = fixture_trace(problem) if trace is None else trace
        text = response(problem, trace) if text is None else text
        report = reasoning_trace.check_trace(actor=problem.actor, item=problem.verifier if item is None else item,
                                            text=text, finish_reason="stop")
        reasoning_trace.validate_trace_report(report, text, "stop")
        return report

    @staticmethod
    def row(report, rule, locus):
        return next(row for row in report["checks"] if (row["rule"], row["locus"]) == (rule, locus))

    def test_sealed_ten_exact_dual_solver_fixtures_and_line_bounds(self):
        topologies = set()
        for problem in [*self.problems, self.one_source_two_stage]:
            trace = fixture_trace(problem)
            world = problem.verifier.accept["circuit_world"]
            topologies.add((len(world["sources"]), len(world["series"])))
            with self.subTest(problem=problem.actor.problem_id):
                report = self.audit(problem, trace)
                self.assertEqual(report["status"], "COMPLIANT")
                self.assertEqual(report["version"], circuit_trace.VERSION)
                self.assertEqual(report["semantic_scope"], circuit_trace.SCOPE)
                self.assertEqual(len(report["checks"]), len(circuit_trace.inventory(trace)))
                self.assertLessEqual(len(report["trace"]["line"]), 4096)
                self.assertEqual(reasoning_trace._inventory(trace), circuit_trace.inventory(trace))
        self.assertEqual(topologies, {(1, 2), (1, 3), (2, 2), (2, 3)})

    def test_gold_is_never_read_and_true_final_does_not_erase_wrong_node(self):
        trace = fixture_trace(self.problem)
        trace["nodes"][-1][1] = "-1"
        original = self.audit(trace=trace)
        changed = self.audit(trace=trace, item=replace(self.problem.verifier, gold="deliberately unrelated"))
        self.assertEqual(original, changed)
        self.assertEqual(original["status"], "VIOLATED")
        self.assertEqual(self.row(original, "CIRCUIT_NODE_VOLTAGE", f"nodes[{len(trace['nodes']) - 1}]")["status"],
                         "VIOLATED")

    def test_source_polarity_and_resistor_endpoints_are_native_checks(self):
        problem = next(p for p in self.problems if len(p.verifier.accept["circuit_world"]["sources"]) == 2)
        trace = fixture_trace(problem)
        trace["nodes"][1][1] = str(-Fraction(trace["nodes"][1][1]))
        report = self.audit(problem, trace)
        self.assertEqual(self.row(report, "CIRCUIT_NODE_VOLTAGE", "nodes[1]")["status"], "VIOLATED")
        trace = fixture_trace(problem)
        trace["ohm"][0][1], trace["ohm"][0][2] = trace["ohm"][0][2], trace["ohm"][0][1]
        report = self.audit(problem, trace)
        self.assertEqual(self.row(report, "CIRCUIT_INCIDENCE", "ohm[0]")["status"], "VIOLATED")

    def test_local_arithmetic_is_independent_of_physical_operand_accuracy(self):
        trace = fixture_trace(self.problem)
        trace["nodes"][-1][1] = "-1"
        nodes = dict(trace["nodes"])
        for row in trace["ohm"]:
            row[4] = str(Fraction(nodes[row[1]]) - Fraction(nodes[row[2]]))
            row[5] = str(Fraction(row[4]) / Fraction(row[3]))
        report = self.audit(trace=trace)
        self.assertEqual(report["status"], "VIOLATED")
        for row in report["checks"]:
            if row["rule"] in ("CIRCUIT_DROP", "CIRCUIT_OHM"):
                self.assertEqual(row["status"], "COMPLIANT")
                self.assertEqual(row["blocked_by"], [])

    def test_resistance_wrong_but_ohm_locally_correct(self):
        trace = fixture_trace(self.problem)
        trace["ohm"][0][3] = "49"
        trace["ohm"][0][5] = str(Fraction(trace["ohm"][0][4]) / 49)
        report = self.audit(trace=trace)
        self.assertEqual(self.row(report, "CIRCUIT_RESISTANCE", "ohm[0]")["status"], "VIOLATED")
        self.assertEqual(self.row(report, "CIRCUIT_OHM", "ohm[0]")["status"], "COMPLIANT")

    def test_kcl_requires_exact_signed_complete_incidence_and_authored_sum(self):
        for mutation in ("sign", "missing", "sum", "order"):
            trace = fixture_trace(self.problem)
            if mutation == "sign":
                trace["kcl"][0][1][0][1] *= -1
            elif mutation == "missing":
                trace["kcl"][0][1].pop()
            elif mutation == "sum":
                trace["kcl"][0][2] = "1"
            else:
                trace["kcl"][0][1].reverse()
            report = self.audit(trace=trace)
            with self.subTest(mutation=mutation):
                self.assertEqual(report["status"], "VIOLATED")
                rule = "CIRCUIT_KCL_SUM" if mutation == "sum" else "CIRCUIT_KCL_INCIDENCE"
                self.assertEqual(self.row(report, rule, "kcl[0]")["status"], "VIOLATED")

    def test_null_propagation_exact_dependencies_and_no_false_compliance(self):
        trace = fixture_trace(self.problem)
        trace["nodes"][1][1] = None
        trace["ohm"][0][3:] = [None, None, None]
        trace["input"] = [None, None, None]
        trace["query"][2] = None
        report = self.audit(trace=trace)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertFalse(report["coverage"]["complete"])
        self.assertEqual(self.row(report, "CIRCUIT_NODE_VOLTAGE", "nodes[1]")["blocked_by"], [])
        self.assertEqual(self.row(report, "CIRCUIT_DROP", "ohm[0]")["blocked_by"],
                         [{"rule": "CIRCUIT_NODE_VOLTAGE", "locus": "nodes[1]"}])
        self.assertEqual(self.row(report, "CIRCUIT_OHM", "ohm[0]")["blocked_by"],
                         [{"rule": "CIRCUIT_RESISTANCE", "locus": "ohm[0]"},
                          {"rule": "CIRCUIT_DROP", "locus": "ohm[0]"}])
        for row in report["issues"]:
            self.assertTrue(circuit_trace.supported_unknown(row, report["checks"]))

    def test_missing_and_reordered_operation_coverage(self):
        for field, rule in (("nodes", "CIRCUIT_NODE_COVERAGE"), ("ohm", "CIRCUIT_RESISTOR_COVERAGE"),
                            ("kcl", "CIRCUIT_KCL_COVERAGE")):
            trace = fixture_trace(self.problem)
            trace[field].reverse()
            with self.subTest(field=field):
                self.assertEqual(self.row(self.audit(trace=trace), rule, field)["status"], "VIOLATED")
        trace = fixture_trace(self.problem)
        trace["kcl"].pop()
        self.assertEqual(self.row(self.audit(trace=trace), "CIRCUIT_KCL_COVERAGE", "kcl")["status"], "VIOLATED")
        trace = fixture_trace(self.problem)
        trace["ohm"].pop()
        for row in trace["kcl"]:
            row[1] = [term for term in row[1] if term[0] != "R6"]
        self.assertEqual(self.row(self.audit(trace=trace), "CIRCUIT_RESISTOR_COVERAGE", "ohm")["status"], "VIOLATED")

    def test_query_and_input_values_units_and_links(self):
        for key, index, value in (("input", 0, "-1"), ("input", 1, "0"), ("input", 2, "0"),
                                  ("query", 2, "0"), ("query", 3, "V")):
            trace = fixture_trace(self.problem)
            trace[key][index] = value
            with self.subTest(field=key, index=index):
                self.assertEqual(self.audit(trace=trace)["status"], "VIOLATED")

    def test_semantic_identity_uses_actual_node_resistor_orientation(self):
        trace = fixture_trace(self.problem)
        reordered = copy.deepcopy(trace)
        reordered["ohm"].reverse()
        self.assertEqual(circuit_trace.identity(trace, "CIRCUIT_OHM", "ohm[0]"),
                         circuit_trace.identity(reordered, "CIRCUIT_OHM", "ohm[5]"))
        self.assertNotEqual(circuit_trace.identity(trace, "CIRCUIT_OHM", "ohm[0]"),
                            circuit_trace.identity(reordered, "CIRCUIT_OHM", "ohm[0]"))
        reordered = copy.deepcopy(trace)
        reordered["ohm"][0][1:3] = reversed(reordered["ohm"][0][1:3])
        self.assertNotEqual(circuit_trace.identity(trace, "CIRCUIT_OHM", "ohm[0]"),
                            circuit_trace.identity(reordered, "CIRCUIT_OHM", "ohm[0]"))

    def test_numeric_bounds_duplicate_rows_extra_fields_and_line_limit(self):
        for value in (True, 1, 1.0, "1.0", "1e1", "01", "+1", "-0", "1/0", "2/4", "1/1", "1/-2", "0/2",
                      " 1", "1 ", "1" * 33, "1/" + "1" * 33, "NaN", "Infinity"):
            trace = fixture_trace(self.problem)
            trace["nodes"][0][1] = value
            with self.subTest(value=value):
                report = self.audit(trace=trace)
                self.assertEqual(report["checks"][0]["rule"], "TRACE_SCHEMA")
        for field in ("nodes", "ohm", "kcl"):
            trace = fixture_trace(self.problem)
            trace[field][0] = trace[field][1]
            self.assertEqual(self.audit(trace=trace)["checks"][0]["rule"], "TRACE_SCHEMA")
        trace = fixture_trace(self.problem)
        trace["extra"] = []
        self.assertEqual(self.audit(trace=trace)["checks"][0]["rule"], "TRACE_SCHEMA")
        text = response(self.problem, fixture_trace(self.problem))
        self.assertEqual(self.audit(text=text.replace("Trace:", "Trace:" + " " * 4096))["checks"][0]["rule"],
                         "TRACE_FORMAT")

    def test_wire_rejects_forged_null_blockers_arithmetic_inventory_and_evidence(self):
        trace = fixture_trace(self.problem)
        trace["nodes"][1][1] = None
        text = response(self.problem, trace)
        report = self.audit(trace=trace)
        for mutation in ("omit_blocker", "forge_blocker", "native_uncertainty", "local_status", "drop", "order", "evidence"):
            forged = copy.deepcopy(report)
            if mutation == "omit_blocker":
                self.row(forged, "CIRCUIT_DROP", "ohm[0]")["blocked_by"] = []
            elif mutation == "forge_blocker":
                self.row(forged, "CIRCUIT_DROP", "ohm[0]")["blocked_by"] = [
                    {"rule": "CIRCUIT_NODE_VOLTAGE", "locus": "nodes[0]"}]
            elif mutation == "native_uncertainty":
                self.row(forged, "CIRCUIT_NODE_VOLTAGE", "nodes[0]")["status"] = "UNVERIFIABLE"
            elif mutation == "local_status":
                self.row(forged, "CIRCUIT_DROP", "ohm[0]")["status"] = "COMPLIANT"
                self.row(forged, "CIRCUIT_DROP", "ohm[0]")["blocked_by"] = []
            elif mutation == "drop":
                forged["checks"].pop()
            elif mutation == "order":
                forged["checks"][4:6] = reversed(forged["checks"][4:6])
            else:
                forged["trace"]["span"][0] += 1
            reasoning_trace._finish(forged)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reasoning_trace.validate_trace_report(forged, text, "stop")

    def test_native_contract_rejects_bridge_unknown_question_mutation_and_solver_disagreement(self):
        bridge = circuit.generate(0, 20261021, "sealed", "obs")
        self.assertEqual(self.audit(bridge, trace={})["checks"][0]["rule"], "NATIVE_CONTRACT")
        actor = replace(self.problem.actor, question=self.problem.actor.question + " changed")
        report = reasoning_trace.check_trace(actor=actor, item=self.problem.verifier,
                                            text=response(self.problem, fixture_trace(self.problem)), finish_reason="stop")
        self.assertEqual(report["checks"][0]["rule"], "NATIVE_CONTRACT")
        reasoning_trace.validate_trace_report(report, response(self.problem, fixture_trace(self.problem)), "stop")
        trace = fixture_trace(self.problem)
        with patch.object(circuit, "_solutions", return_value=({"a": 1}, {"a": 2})):
            self.assertEqual(self.audit(trace=trace)["checks"][0]["rule"], "NATIVE_CONTRACT")

    def test_instruction_contains_only_public_schema_and_scope_is_bound(self):
        for problem in self.problems:
            instruction = reasoning_trace.instruction_for_question(problem.actor.question, problem.actor.source)
            self.assertIn(circuit_trace.VERSION, instruction)
            self.assertIn("canonical reduced rational", instruction)
            self.assertNotIn("node_voltages_V", instruction)
            self.assertNotIn("{", instruction)
        text = response(self.problem, fixture_trace(self.problem))
        report = self.audit()
        report["semantic_scope"] = reasoning_trace._SCOPE
        with self.assertRaises(ValueError):
            reasoning_trace.validate_trace_report(report, text, "stop")


if __name__ == "__main__":
    unittest.main()
