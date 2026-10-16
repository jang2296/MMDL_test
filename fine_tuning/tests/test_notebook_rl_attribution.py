"""Functional cause evidence is not proof of internal module responsibility."""
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl.attribution import attempt_cause, diagnosis_target, problem_route, random_route
from fine_tuning.notebook_rl.observations import compare_slots
from fine_tuning.notebook_rl.schema import DiagTarget, VerifierResult
from fine_tuning.tests.test_notebook_rl_schema import records


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.item = records()[1]

    def result(self, observed=None, status="INCORRECT", item=None):
        item = item or self.item
        slots = None if observed is None else compare_slots(observed, item)
        return VerifierResult(status, "7" if status == "CORRECT" else "8", float(status == "CORRECT"), slots, "fixture")

    def test_missing_plain_empty_required_and_system_are_unknown(self):
        self.assertEqual(attempt_cause(self.result(), self.item), "UNKNOWN")
        self.assertEqual(attempt_cause(self.result({}), self.item), "UNKNOWN")
        empty = replace(self.item, required_slots=())
        self.assertEqual(attempt_cause(self.result({"cell[a][b]": ["7"]}, item=empty), empty), "UNKNOWN")
        self.assertEqual(attempt_cause(self.result(status="UNVERIFIABLE"), self.item), "UNKNOWN")
        self.assertIsNone(diagnosis_target(self.result(), self.item))

    def test_finalization_is_not_perception_failure(self):
        for status in ("INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"):
            result = self.result(status=status)
            self.assertEqual(attempt_cause(result, self.item), "FINALIZATION")
            self.assertEqual(diagnosis_target(result, self.item), DiagTarget("FINALIZATION", ()))

    def test_visual_nonvisual_and_correct_answers(self):
        bad = self.result({"cell[a][b]": ["8"]})
        self.assertEqual(attempt_cause(bad, self.item), "VISUAL_MISREAD")
        self.assertEqual(diagnosis_target(bad, self.item), DiagTarget("VISUAL", ("cell[a][b]",)))
        good = self.result({"cell[a][b]": ["7"]})
        self.assertEqual(attempt_cause(good, self.item), "NON_VISUAL")
        self.assertIsNone(attempt_cause(replace(bad, status="CORRECT", reward_answer=1.0), self.item))
        edge = replace(self.item, slots={"wire[r1]": "a-b"}, slot_types={"wire[r1]": "relation"}, required_slots=("wire[r1]",))
        self.assertEqual(attempt_cause(self.result({"wire[r1]": ["a-c"]}, item=edge), edge), "VISUAL_RELATION")

    def test_half_parsed_boundary_and_partial_correct_unknown(self):
        keys = ("cell[a]", "cell[b]", "cell[c]", "cell[d]")
        item = replace(self.item, slots=dict.fromkeys(keys, "7"), slot_types=dict.fromkeys(keys, "number"), required_slots=keys)
        self.assertEqual(attempt_cause(self.result({keys[0]: ["8"]}, item=item), item), "UNKNOWN")
        for status in ("INCORRECT", "CORRECT", "SYSTEM_ERROR", "UNVERIFIABLE", "TRUNCATED"):
            partial = self.result({keys[0]: ["8"]}, status=status, item=item)
            self.assertEqual(problem_route([partial], item, "verified-any-visual-v1")[0], "VISION_OFF")
        self.assertEqual(attempt_cause(self.result({keys[0]: ["8"], keys[1]: ["7"]}, item=item), item), "VISUAL_MISREAD")
        self.assertEqual(attempt_cause(self.result({keys[0]: ["7"], keys[1]: ["7"]}, item=item), item), "UNKNOWN")

    def test_any_wrong_required_including_correct_answer_routes_vision(self):
        right = self.result({"cell[a][b]": ["7"]})
        wrong = self.result({"cell[a][b]": ["8"]}, status="CORRECT")
        route, evidence = problem_route([right, wrong], self.item, "verified-any-visual-v1")
        self.assertEqual(route, "VISION_ON")
        self.assertEqual(evidence["wrong_required"], [{"result_index": 1, "wrong_slots": ["cell[a][b]"]}])
        self.assertNotIn("expected", str(evidence))
        self.assertFalse(evidence["module_causality_certified"])
        self.assertEqual(problem_route([right], self.item, "verified-any-visual-v1")[0], "VISION_OFF")
        self.assertEqual(problem_route([self.result()], self.item, "verified-any-visual-v1")[1]["observed_evidence"], "UNKNOWN")

    def test_probe_is_explicit_separate_and_conflict_not_overclaimed(self):
        probe = {"oracle": {"n": 4, "correct": 3, "baseline_n": 8, "baseline_correct": 0}}
        with self.assertRaises(ValueError):
            problem_route([self.result()], self.item, "verified-any-visual-v1", probe)
        self.assertEqual(problem_route([self.result()], self.item, "verified-any-visual-v1+probe", probe)[0], "VISION_ON")
        conflict = probe | {"self_perception": {"n": 4, "correct": 3}}
        route, evidence = problem_route([self.result()], self.item, "verified-any-visual-v1+probe", conflict)
        self.assertEqual((route, evidence["probe_evidence"]), ("VISION_OFF", "UNKNOWN"))
        for bad in ({"gold": "7"}, {"oracle": {"n": 4, "correct": 5, "baseline_n": 8, "baseline_correct": 0}},
                    {"self_perception": {"n": 1, "correct": 1}}):
            with self.assertRaises(ValueError):
                problem_route([], self.item, "verified-any-visual-v1+probe", bad)

    def test_invalid_slot_report_rejected(self):
        result = self.result({"cell[a][b]": ["8"]})
        bad = replace(result, slots=replace(result.slots, wrong_required=("unknown",)))
        with self.assertRaises(ValueError):
            problem_route([bad], self.item, "verified-any-visual-v1")

    def test_deterministic_random_route_rate(self):
        routes = [random_route(.3, f"problem:{i}") for i in range(10_000)]
        self.assertLess(abs(routes.count("VISION_ON") / 10_000 - .3), .02)
        self.assertEqual(routes[:20], [random_route(.3, f"problem:{i}") for i in range(20)])
        self.assertEqual(random_route(0, "key"), "VISION_OFF")
        self.assertEqual(random_route(1, "key"), "VISION_ON")
        with self.assertRaises(ValueError):
            random_route(float("nan"), "key")


if __name__ == "__main__":
    unittest.main()
