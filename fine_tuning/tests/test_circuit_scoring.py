"""Hand-authored CPU contracts; no model, solver, benchmark or external input."""
import copy
import json
import unittest

from fine_tuning.circuit_scoring import CONTRACT_VERSION, extract_response, score_circuit


STRUCTURE = {
    "nodes": ["N0", "N1", "N2", "N3"], "reference_node": "N0",
    "components": [
        {"id": "C0", "type": "RESISTOR", "pins": ["N1", "N2"], "value": 2, "unit": "ohm"},
        {"id": "C1", "type": "RESISTOR", "pins": ["N2", "N0"], "value": 6, "unit": "ohm"},
        {"id": "C2", "type": "RESISTOR", "pins": ["N3", "N0"], "value": 3, "unit": "ohm"},
        {"id": "C3", "type": "DC_SOURCE", "pins": ["N1", "N0"], "positive_pin": "N1",
         "value": 12, "unit": "V"},
        {"id": "C4", "type": "SWITCH", "pins": ["N2", "N3"], "state": "CLOSED"},
    ],
}
CLOSED = {"source_current_A": "3", "node_voltages_V": {"N0": "0", "N1": "12", "N2": "6", "N3": "6"},
          "resistor_currents_A": {"C0": "3", "C1": "1", "C2": "2"}, "option": "B"}
OPEN = {"source_current_A": "3/2", "node_voltages_V": {"N0": "0", "N1": "12", "N2": "9", "N3": "0"},
        "resistor_currents_A": {"C0": "3/2", "C1": "3/2", "C2": "0"}, "option": "C"}


class CircuitScoringTests(unittest.TestCase):
    def score(self, prediction, target, view):
        return score_circuit(json.dumps(prediction), json.dumps(target), view)

    def test_single_letter_only_and_no_candidate_rescue(self):
        self.assertTrue(score_circuit(" B \n", "B", "qa")["correct"])
        self.assertFalse(score_circuit("A", "B", "qa")["correct"])
        for raw in ("not B", "B or C", "A then B", "Final answer: B", '"B"', "B.", "b", "E", ""):
            with self.subTest(raw=raw):
                result = score_circuit(raw, "B", "qa")
                self.assertFalse(result["correct"])
                self.assertEqual("NO_PARSE", result["parse_status"])

    def test_structure_order_is_irrelevant_but_pin_direction_is_exact(self):
        predicted = copy.deepcopy(STRUCTURE)
        predicted["nodes"].reverse()
        predicted["components"].reverse()
        self.assertTrue(self.score(predicted, STRUCTURE, "structure")["correct"])
        for component in range(5):
            predicted = copy.deepcopy(STRUCTURE)
            predicted["components"][component]["pins"].reverse()
            result = self.score(predicted, STRUCTURE, "structure")
            self.assertFalse(result["correct"])
            self.assertEqual("PARSED", result["parse_status"])

    def test_valid_wrong_polarity_and_component_assignment_are_parsed_before_comparison(self):
        predicted = copy.deepcopy(STRUCTURE)
        predicted["components"][3]["positive_pin"] = "N0"
        result = self.score(predicted, STRUCTURE, "structure")
        self.assertEqual("PARSED", result["parse_status"])
        self.assertFalse(result["circuit_structure_exact"])
        predicted = copy.deepcopy(STRUCTURE)
        predicted["components"][0]["id"], predicted["components"][3]["id"] = "C3", "C0"
        result = self.score(predicted, STRUCTURE, "structure")
        self.assertEqual("PARSED", result["parse_status"])
        self.assertFalse(result["correct"])

    def test_wrong_connectivity_value_polarity_state_reference_or_unit_fails(self):
        for index, key, value in ((0, "pins", ["N1", "N0"]), (0, "value", 4),
                                  (3, "positive_pin", "N0"), (4, "state", "OPEN"),
                                  (0, "unit", "kohm"), (3, "unit", "mV"), (0, "type", "CAPACITOR")):
            predicted = copy.deepcopy(STRUCTURE)
            predicted["components"][index][key] = value
            self.assertFalse(self.score(predicted, STRUCTURE, "structure")["correct"])
        predicted = dict(STRUCTURE, reference_node="N1")
        self.assertFalse(self.score(predicted, STRUCTURE, "structure")["correct"])

    def test_unknown_duplicate_missing_components_nodes_or_pins_fail(self):
        variants = []
        for key, value in (("id", "C5"), ("id", "C1"), ("pins", ["N1", "N9"]),
                           ("pins", ["N1", "N1"]), ("value", True), ("value", 2.0)):
            item = copy.deepcopy(STRUCTURE)
            item["components"][0][key] = value
            variants.append(item)
        variants += [dict(STRUCTURE, nodes=["N0", "N1", "N2", "N2"]),
                     dict(STRUCTURE, components=STRUCTURE["components"][:-1]),
                     dict(STRUCTURE, components=STRUCTURE["components"]+[STRUCTURE["components"][0]])]
        for item in variants:
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    extract_response(json.dumps(item), "structure")

    def test_exact_rationals_accept_equivalent_decimals_and_integers(self):
        predicted = copy.deepcopy(OPEN)
        predicted["source_current_A"] = "6/4"
        predicted["resistor_currents_A"].update(C0=1.5, C1="1.50", C2=0)
        predicted["node_voltages_V"].update(N0="0/3", N1="1.2e1", N2=9)
        self.assertTrue(self.score(predicted, OPEN, "grounded")["correct"])
        predicted["source_current_A"] = "1.50000000000000000001"
        self.assertFalse(self.score(predicted, OPEN, "grounded")["correct"])

    def test_every_grounded_field_required_and_correct_letter_cannot_rescue_current(self):
        for key in CLOSED:
            predicted = copy.deepcopy(CLOSED)
            del predicted[key]
            self.assertFalse(self.score(predicted, CLOSED, "grounded")["correct"])
        predicted = copy.deepcopy(CLOSED)
        predicted["resistor_currents_A"]["C1"] = "2"
        result = self.score(predicted, CLOSED, "grounded")
        self.assertFalse(result["correct"])
        self.assertTrue(result["option_exact"])
        self.assertFalse(result["currents_exact"])
        predicted = dict(CLOSED, option="C")
        self.assertFalse(self.score(predicted, CLOSED, "grounded")["correct"])

    def test_no_units_conversion_booleans_nonfinite_or_arithmetic_expressions(self):
        for value in (True, None, [], {}, "3 A", "3000 mA", "6/2 A", "1+2", "NaN",
                      "Infinity", "-Infinity", "1/0", "1e9999", "9"*129, float("nan"), float("inf")):
            with self.subTest(value=value):
                predicted = dict(CLOSED, source_current_A=value)
                with self.assertRaises(ValueError):
                    extract_response(json.dumps(predicted), "grounded")

    def test_exact_decimal_is_not_binary_float_approximation(self):
        reference = dict(CLOSED, source_current_A="1/10")
        self.assertTrue(self.score(dict(CLOSED, source_current_A=0.1), reference, "grounded")["correct"])
        reference = dict(CLOSED, source_current_A="1/3")
        self.assertFalse(self.score(dict(CLOSED, source_current_A=1/3), reference, "grounded")["correct"])

    def test_only_one_bare_json_object_and_duplicate_keys_fail(self):
        raw = json.dumps(CLOSED)
        for malformed in (raw+raw, "reason: "+raw, raw+" B", "```json\n"+raw+"\n```", "["+raw+"]",
                          raw.replace('"option": "B"', '"option": "C", "option": "B"'),
                          raw.replace('"C1": "1"', '"C1": "1", "C1": "1"')):
            self.assertFalse(score_circuit(malformed, raw, "grounded")["correct"])

    def test_extra_fields_unknown_maps_and_answer_injection_fail(self):
        for view, target in (("structure", STRUCTURE), ("grounded", CLOSED)):
            for extra in ("answer", "reasoning", "world", "gold", "question"):
                self.assertFalse(self.score(dict(target, **{extra: "B"}), target, view)["correct"])
        predicted = copy.deepcopy(STRUCTURE)
        predicted["components"][0]["answer"] = "B"
        self.assertFalse(self.score(predicted, STRUCTURE, "structure")["correct"])
        predicted = copy.deepcopy(CLOSED)
        predicted["node_voltages_V"]["N4"] = "0"
        self.assertFalse(self.score(predicted, CLOSED, "grounded")["correct"])

    def test_extraction_is_independent_of_gold_and_does_not_solve(self):
        raw = json.dumps(CLOSED)
        observed = extract_response(raw, "grounded")
        wrong = dict(CLOSED, option="D", source_current_A="4")
        self.assertFalse(score_circuit(raw, json.dumps(wrong), "grounded")["correct"])
        self.assertEqual(observed, extract_response(raw, "grounded"))
        # A schema-valid but physically false target is compared as supplied.
        self.assertTrue(self.score(wrong, wrong, "grounded")["correct"])

    def test_unknown_contracts_views_and_invalid_gold_raise(self):
        for version in ("independent-dc-circuit-bridge-v1", "future", None):
            with self.assertRaises(ValueError):
                extract_response("B", "qa", contract_version=version)
        with self.assertRaises(ValueError):
            score_circuit("B", "B", "unknown")
        with self.assertRaises(ValueError):
            score_circuit("B", "not B", "qa")
        self.assertEqual("independent-dc-circuit-bridge-v2", CONTRACT_VERSION)

    def test_incomplete_unknown_finish_and_oversized_response_fail_closed(self):
        for reason in ("length", "error", None):
            self.assertFalse(score_circuit("B", "B", "qa", reason)["correct"])
        with self.assertRaises(ValueError):
            extract_response(" "*16385+"B", "qa")
        self.assertTrue(score_circuit("B", "B", "qa", "eos")["correct"])


if __name__ == "__main__":
    unittest.main()
