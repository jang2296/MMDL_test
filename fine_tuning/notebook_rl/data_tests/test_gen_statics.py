"""Authored beam statics checks: exact algebra, rendered records, actual consumers."""
import hashlib
import io
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

from fine_tuning.notebook_rl.answers import compare
from fine_tuning.notebook_rl.generators import statics
from fine_tuning.notebook_rl.generators.common import matches_split, png_chunks
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.schema import to_json


class StaticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = []
        for index in range(240):
            split = ("train", "dev", "sealed", "style_sealed")[index % 4]
            cls.problems.append(statics.generate(index, 3407, split, "obs",
                                                 "shift" if split == "style_sealed" else "default"))

    def test_hand_calculated_point_and_uniform_load_cases(self):
        # Simple support, 10kN at4m and2kN/m from2m to8m: RA12, RB10.
        base = {"kind": "simple", "length": 10, "loads": [[4, 10]], "uniform": [2, 8, 2]}
        expected = {"reaction": Fraction(12), "reaction_b": Fraction(10),
                    "max_moment": Fraction(44), "max_location": Fraction(4)}
        self.assertEqual(statics._equilibrium(base), expected)
        self.assertEqual(statics._sympy(base), expected)
        cantilever = base | {"kind": "cantilever"}
        expected = {"reaction": Fraction(22), "reaction_b": Fraction(0),
                    "max_moment": Fraction(100), "max_location": Fraction(0)}
        self.assertEqual(statics._equilibrium(cantilever), expected)
        self.assertEqual(statics._sympy(cantilever), expected)
        # Uniform loading gives an interior stationary point, not just a load node.
        uniform = {"kind": "simple", "length": 8, "loads": [[2, 4]], "uniform": [0, 8, 2]}
        expected = {"reaction": Fraction(11), "reaction_b": Fraction(9),
                    "max_moment": Fraction(81, 4), "max_location": Fraction(7, 2)}
        self.assertEqual(statics._equilibrium(uniform), expected)
        self.assertEqual(statics._sympy(uniform), expected)

    def test_signed_forces_plateau_tie_and_cantilever_interior_maximum(self):
        plateau = {"kind": "simple", "length": 10, "loads": [[2, 10], [8, 10]], "uniform": None}
        for solver in (statics._equilibrium, statics._sympy):
            self.assertEqual(solver(plateau)["max_location"], 2)
            self.assertEqual(solver(plateau)["max_moment"], 20)
        # Upward tip-side load balances the root couple; the peak is in the span.
        interior = {"kind": "cantilever", "length": 10,
                    "loads": [[2, 20], [8, -5]], "uniform": None}
        expected = {"reaction": Fraction(15), "reaction_b": Fraction(0),
                    "max_moment": Fraction(30), "max_location": Fraction(2)}
        self.assertEqual(statics._equilibrium(interior), expected)
        self.assertEqual(statics._sympy(interior), expected)
        reversed_loads = interior | {"loads": [[2, -20], [8, 5]]}
        self.assertEqual(statics._equilibrium(reversed_loads),
                         expected | {"reaction": Fraction(-15)})
        self.assertEqual(statics._sympy(reversed_loads), statics._equilibrium(reversed_loads))
        self.assertIn("smallest x", statics._question("max_location")[0])

    def test_500_exact_dual_solver_structures(self):
        counts, signs, kinds, uniform_counts = set(), set(), set(), set()
        for index in range(500):
            structure = statics._select(index, 3408, "train", "default")
            counts.add(len(structure["loads"]))
            signs.update(1 if p > 0 else -1 for _, p in structure["loads"])
            kinds.add(structure["kind"])
            uniform_counts.add(structure["uniform"] is not None)
            equilibrium = statics._equilibrium(structure)
            self.assertEqual(equilibrium, statics._sympy(structure))
            if structure["kind"] == "simple":
                self.assertGreaterEqual(equilibrium["reaction"], 0)
                self.assertGreaterEqual(equilibrium["reaction_b"], 0)
        self.assertEqual(counts, {1, 2, 3})
        self.assertEqual(signs, {-1, 1})
        self.assertEqual(kinds, {"simple", "cantilever"})
        self.assertEqual(uniform_counts, {True, False})

    def test_80_self_checks_determinism_split_and_actual_consumer_contracts(self):
        by_split = {s: set() for s in ("train", "dev", "sealed", "style_sealed")}
        for p in self.problems[:80]:
            statics.self_check(p)
            by_split[p.lineage.split].add(p.lineage.structure_hash)
            self.assertTrue(matches_split(p.lineage.structure_hash, p.lineage.split,
                                          "shift" if p.lineage.split == "style_sealed" else "default"))
            self.assertIs(compare(p.verifier.gold, p.verifier), True)
            self.assertIs(compare(p.verifier.gold.rsplit(" ", 1)[0], p.verifier), False)
            text = "Observed facts:\n" + "\n".join(p.verifier.oracle_fact_lines) + "\nSolution:\n"
            report = compare_slots(parse_observations(text), p.verifier)
            self.assertEqual(report.all_correct, len(p.verifier.slots))
            self.assertEqual(report.required_correct, len(p.verifier.required_slots))
            self.assertEqual(report.wrong_required, ())
            self.assertEqual(report.conflicting, ())
            self.assertEqual(p.actor.answer_format, "unit_number")
            self.assertEqual(p.actor.choices, ())
            self.assertFalse(p.lineage.benchmark_derived or p.lineage.model_touched)
            for forbidden in ('"gold":', '"slots":', '"accept":'):
                self.assertNotIn(forbidden, to_json(p.actor))
        for a in by_split:
            for b in by_split:
                if a != b:
                    self.assertTrue(by_split[a].isdisjoint(by_split[b]))

    def test_100_same_question_image_dependent_pairs(self):
        buckets, pairs, templates = {}, 0, set()
        for p in self.problems:
            prior = buckets.setdefault(p.actor.question, [])
            other = next((q for q in prior if q.verifier.gold != p.verifier.gold), None)
            if other is not None:
                self.assertNotEqual(other.image_bytes, p.image_bytes)
                self.assertNotEqual(other.lineage.structure_hash, p.lineage.structure_hash)
                templates.add(p.lineage.template_id)
                pairs += 1
                if pairs == 100:
                    break
            prior.append(p)
        self.assertEqual(pairs, 100)
        self.assertEqual(templates, set(statics._TASKS))

    def test_png_slots_style_prompt_and_no_global_rng_mutation(self):
        import random
        prompt = yaml.safe_load((Path(__file__).parents[1] / "prompts_v1.yaml").read_text())
        self.assertEqual(statics.OBSERVATION_INSTRUCTION, prompt["observation_instruction"][statics.FAMILY])
        state = random.getstate()
        p = statics.generate(0, 3407, "train", "plain")
        self.assertEqual(random.getstate(), state)
        self.assertEqual(p.image_bytes, self.problems[0].image_bytes)
        self.assertEqual(p.verifier, self.problems[0].verifier)
        for p in self.problems[:12]:
            self.assertEqual(hashlib.sha256(p.image_bytes[0]).hexdigest(), p.actor.image_sha256[0])
            self.assertTrue(set(png_chunks(p.image_bytes[0])) <= {b"IHDR", b"IDAT", b"IEND"})
            with Image.open(io.BytesIO(p.image_bytes[0])) as image:
                self.assertGreaterEqual(image.width, 1000)
                self.assertGreaterEqual(image.height, 500)
                self.assertEqual(image.info, {})
            self.assertEqual(set(p.verifier.slots), set(p.verifier.required_slots))
            self.assertEqual(p.verifier.slot_types["span"], "number")
        structure = statics._select(0, 3407, "train", "default")
        raw, printed = statics._render(structure, "default")
        shifted, shifted_printed = statics._render(structure, "shift")
        self.assertNotEqual(raw, shifted)
        self.assertEqual(printed, shifted_printed)

    def test_unit_and_exact_rounding_policy(self):
        for x, expected in ((Fraction(25, 100), "0.3"), (Fraction(-25, 100), "-0.3"),
                            (Fraction(-1, 100), "0.0")):
            self.assertEqual(statics._rounded(x), expected)
        p = next(p for p in self.problems if p.lineage.template_id == "max_moment")
        self.assertIs(compare(p.verifier.gold.replace("kN*m", "kNm"), p.verifier), True)
        self.assertIs(compare(p.verifier.gold.replace("kN*m", "Nm"), p.verifier), False)

    def test_invalid_tampered_inputs_and_solver_disagreement_fail_closed(self):
        p = self.problems[0]
        for item in (replace(p, verifier=replace(p.verifier, gold="0.0 kN")),
                     replace(p, image_bytes=(p.image_bytes[0][:-1],)),
                     replace(p, actor=replace(p.actor, family="table")),
                     replace(p, verifier=replace(p.verifier, slots={})),
                     replace(p, lineage=replace(p.lineage, generator_version="other"))):
            with self.assertRaisesRegex(ValueError, "statics:"):
                statics.self_check(item)
        for delta in ({"index": True}, {"seed": -1}, {"index": -1}, {"mode": "other"},
                      {"split": "other"}, {"style": "shift"}, {"split": "style_sealed"}):
            with self.assertRaises(ValueError):
                statics.generate(**({"index": 0, "seed": 3407, "split": "train", "mode": "obs"}
                                    | delta))
        with patch.object(statics, "_sympy", return_value={}):
            with self.assertRaisesRegex(ValueError, "solvers disagree"):
                statics.generate(0, 3407, "train", "obs")
        with patch.object(statics, "matches_split", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "1000 candidates"):
                statics.generate(0, 3407, "train", "obs")


if __name__ == "__main__":
    unittest.main()
