"""Authored CPU fixtures check the circuit notebook contract, never learned improvement."""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl import circuit_trace, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import whole_reference_notebook as reference
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.notebook_rl.generators import circuit
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import Group, ProblemTree, Sample, to_json
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests.test_notebook_rl_circuit_trace import fixture_trace, response
from fine_tuning.tests.test_notebook_rl_notebook_synthesis import note_for
from fine_tuning.tests.test_notebook_rl_whole_reference_notebook import compact, find_ref


class CircuitNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = circuit.generate(2, 20261021, "sealed", "obs")
        self.actor, self.item = self.problem.actor, self.problem.verifier
        self.verifier = Verifier.__new__(Verifier)
        self.verifier.actors = {self.actor.problem_id: self.actor}
        self.verifier.items = {self.actor.problem_id: self.item}
        self.right_trace = fixture_trace(self.problem)
        self.right = response(self.problem, self.right_trace)
        wrong = copy.deepcopy(self.right_trace)
        wrong["nodes"][-1][1] = "-1"
        self.wrong = response(self.problem, wrong).rsplit("Final answer:", 1)[0] + "Final answer: -999 A"
        self.prepare([self.wrong, self.right])

    def prepare(self, texts: list[str]) -> None:
        groups: list[Group] = []
        for turn, text in enumerate(texts, 1):
            parent = groups[-1].samples[0] if groups else None
            prompt = parent.prompt_token_ids + parent.completion_token_ids + [900 + turn] if parent else [1, 2]
            kind = f"T{turn}"
            sample = Sample(f"circuit-{turn}", self.actor.problem_id, kind, kind, turn,
                parent.sample_id if parent else None, prompt, [100 + turn], [-.1], "stop", text,
                3407 + turn, 0, self.verifier.grade(self.actor.problem_id, text, "stop", self.actor.mode))
            groups.append(Group(kind, kind, self.actor.problem_id, _hash(prompt), [sample],
                                False, "EVALUATION_SINGLE_SAMPLE"))
        self.tree = ProblemTree(self.actor.problem_id, 0, groups, "VISION_OFF", {}, {})
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier,
            contract=synthesis.BOUNDED_VERSION, evaluation_only=True)
        self.samples = solution_samples(self.tree)
        self.note = compact(self.samples, note_for(self.tree, self.audit, self.item))

    def grade(self, note: dict | None = None) -> dict:
        return reference.grade_note(json.dumps(self.note if note is None else note), "stop",
            self.tree, self.audit, self.verifier, actor=self.actor,
            contract=synthesis.BOUNDED_VERSION, evaluation_only=True)

    def test_exact_native_circuit_recovery_and_full_notebook(self) -> None:
        before = to_json(self.tree), copy.deepcopy(self.audit)
        checked = self.grade()
        self.assertEqual(checked["reward"], 1.)
        self.assertEqual(checked["expansion_status"], reference.CAPTURED)
        self.assertTrue(synthesis.successful(checked["rendered_strict_report"], synthesis.BOUNDED_VERSION))
        self.assertTrue(any(row[3] and row[3][0] == "CORRECTED" for row in self.note["assessments"]))
        self.assertEqual((to_json(self.tree), self.audit), before)
        self.assertIs(self.audit["training_admission"], False)

    def test_null_dependencies_remain_explicit_and_accountable(self) -> None:
        trace = copy.deepcopy(self.right_trace)
        trace["nodes"][1][1] = None
        trace["ohm"][0][3:] = [None, None, None]
        text = response(self.problem, trace).rsplit("Final answer:", 1)[0] + "Final answer: -999 A"
        self.prepare([text, self.right])
        self.assertTrue(self.audit["unknown_regions_accountable"])
        self.assertTrue(any(row[1] == "U" and row[2] for row in self.note["assessments"]))
        checked = self.grade()
        self.assertEqual(checked["expansion_status"], reference.CAPTURED)
        self.assertEqual(checked["reward"], 1.)
        forged = copy.deepcopy(self.note)
        next(row for row in forged["assessments"] if row[2])[2] = []
        self.assertEqual(self.grade(forged)["reward"], 0.)

    def test_wrong_nonnull_prerequisite_does_not_hide_checkable_arithmetic(self) -> None:
        inventory = reference.public_inventory(self.samples)
        node_ref = find_ref(inventory, {"attempt": 1, "kind": "TRACE", "rule": "CIRCUIT_NODE_VOLTAGE",
            "locus": f"nodes[{len(self.right_trace['nodes']) - 1}]"})
        dependent_ref = next(ref for ref, row in inventory["references"].items()
                             if row["attempt"] == 1 and row["rule"] == "CIRCUIT_DROP")
        forged = copy.deepcopy(self.note)
        row = next(row for row in forged["assessments"] if row[0] == dependent_ref)
        row[1:3] = ["U", [node_ref]]
        checked = self.grade(forged)
        self.assertEqual(checked["reward"], 0.)
        self.assertEqual(checked["expansion_status"], reference.INVALID)

    def test_complete_circuit_facts_and_polarity_are_rechecked(self) -> None:
        checked = self.grade()
        self.assertEqual(len(checked["fact_reinspection"]["facts"]), len(self.item.slots))
        for key in ("comp[R1]", "wire[R1]", "dir[E1]"):
            with self.subTest(key=key):
                forged = copy.deepcopy(self.note)
                actual = next(name for name in forged["corrected_facts"] if name.lower() == key.lower())
                forged["corrected_facts"][actual] = {
                    "comp[R1]": "1 V", "wire[R1]": "N0-N5", "dir[E1]": "N0->N1"
                }[key]
                self.assertEqual(self.grade(forged)["reward"], 0.)

    def test_same_numeric_wrong_units_and_prose_cannot_receive_fact_credit(self) -> None:
        for component, suffixes in (("comp[R1]", ("V", "A", "nonsense", "ohm therefore correct")),
                                    ("comp[E1]", ("ohm", "A", "nonsense"))):
            key = next(key for key in self.item.slots if key.lower() == component.lower())
            number = self.item.slots[key].split()[0]
            for suffix in suffixes:
                with self.subTest(component=component, suffix=suffix):
                    forged = copy.deepcopy(self.note)
                    forged["corrected_facts"][key] = number + " " + suffix
                    checked = self.grade(forged)
                    self.assertEqual(checked["reward"], 0.)
                    self.assertEqual(checked["expansion_status"], reference.INVALID)
                    altered = self.wrong.replace(f"{key} = {self.item.slots[key]}", f"{key} = {number} {suffix}")
                    self.prepare([altered, self.right])
                    rows = [row for row in self.audit["attempts"][0]["checks"]
                            if row["kind"] == "OBSERVATION" and row["locus"] == key.lower()]
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["status"], "VIOLATED" if suffix in ("V", "A", "ohm")
                                     else "UNVERIFIABLE")
                    self.assertEqual(self.audit["routing_rule"], synthesis.CIRCUIT_EVALUATION_ROUTING)
                    self.assertEqual(self.audit["admission"], "VISION_ON" if suffix in ("V", "A", "ohm")
                                     else "UNRESOLVED")
                    self.assertEqual(self.audit["complete_observation_attempts"], [2])
                    self.assertFalse(self.audit["all_required_observations_verified"])
                    self.assertFalse(self.audit["training_admission"])
                    self.assertEqual(self.grade()["reward"], 1.)

    def test_prompt_is_public_complete_and_gold_independent(self) -> None:
        text = reference.evaluation_generation_text(self.actor, self.samples, "PUBLIC_SCHEMA", 2)
        inventory = reference.public_inventory(self.samples)
        self.verifier.items[self.actor.problem_id] = replace(self.item, gold="unrelated-hidden-gold")
        self.assertEqual(text, reference.evaluation_generation_text(self.actor, self.samples, "PUBLIC_SCHEMA", 2))
        self.assertEqual(inventory, reference.public_inventory(self.samples))
        self.assertNotIn("unrelated-hidden-gold", text)
        self.assertIn("CIRCUIT_*", text)
        self.assertIn("then V for comp[E1]/comp[E2] or ohm for comp[R1]..comp[R6]", text)
        self.assertEqual(reference.evaluation_generation_contract()["trace_contract"], circuit_trace.VERSION)
        self.assertIn(json.dumps(self.right, ensure_ascii=False)[1:-1].split("\\n")[0], text)
        with self.assertRaises(ValueError):
            reference.evaluation_generation_text(replace(self.actor, mode="plain"), self.samples, "PUBLIC_SCHEMA", 2)

    def test_evaluation_cannot_be_used_as_training_note(self) -> None:
        with self.assertRaises(ValueError):
            reference.grade_note(json.dumps(self.note), "stop", self.tree, self.audit, self.verifier,
                actor=self.actor, contract=synthesis.BOUNDED_VERSION)


if __name__ == "__main__":
    unittest.main()
