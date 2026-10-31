"""Authored thermodynamics/energy checks: exact solvers, rendered records, actual consumers."""
import hashlib
import io
import random
import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl.answers import compare
from fine_tuning.notebook_rl.dataset import generate_pool
from fine_tuning.notebook_rl.generators import get_generator, thermo
from fine_tuning.notebook_rl.generators.common import matches_split, png_chunks
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.schema import to_json

SPLITS = ("train", "dev", "sealed", "style_sealed")


def _style(split):
    return "shift" if split == "style_sealed" else "default"


def _cycle(*states):
    return {"task": "cycle_work", "states": [[label, v, p] for label, (v, p) in zip("ABCD", states)]}


class ThermoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = [thermo.generate(index, 3407, SPLITS[index % 4], "obs", _style(SPLITS[index % 4]))
                        for index in range(48)]

    def test_registry_and_module_contract(self):
        self.assertIs(get_generator("thermo"), thermo)
        self.assertEqual((thermo.GENERATOR_VERSION, thermo.FAMILY), ("gen-thermo-v1", "thermo"))
        for key in ("state[", "flow[", "power["):
            self.assertIn(key, thermo.OBSERVATION_INSTRUCTION)
        self.assertTrue(thermo.OBSERVATION_INSTRUCTION.endswith("\n"))

    def test_hand_checked_cycle_work(self):
        # Right on the bottom leg, up, left on the top leg: counter-clockwise, 0.4 m^3 x 200 kPa.
        ccw = _cycle((2, 100), (6, 100), (6, 300), (2, 300))
        # Up the left side, down-right to the bottom, back left: clockwise, 0.5 x 0.4 x 200.
        cw = _cycle((2, 100), (2, 300), (6, 100))
        for solver in (thermo._work_shoelace, thermo._work_legs):
            self.assertEqual(solver(ccw), Fraction(-80))
            self.assertEqual(solver(cw), Fraction(40))
        self.assertEqual(thermo._answer(ccw), ("-80.0", "decimal", {"round_to": 1}))
        self.assertEqual(thermo._answer(cw), ("40.0", "decimal", {"round_to": 1}))
        self.assertEqual(thermo._slots(cw), {"state[A]": "0.2,100", "state[B]": "0.2,300",
                                             "state[C]": "0.6,100"})

    def test_hand_checked_efficiency_and_energy(self):
        sankey = {"task": "sankey_efficiency", "input": 800, "useful": 300,
                  "losses": [["Friction", 200], ["Exhaust gas", 300]]}
        self.assertEqual(thermo._efficiency(sankey), Fraction(75, 2))
        self.assertEqual(thermo._efficiency_from_losses(sankey), Fraction(75, 2))
        self.assertEqual(thermo._answer(sankey), ("37.5", "decimal", {"round_to": 1}))
        self.assertEqual(thermo._slots(sankey), {"flow[Input]": "800 kW", "flow[Useful output]": "300 kW",
                                                 "flow[Friction]": "200 kW", "flow[Exhaust gas]": "300 kW"})
        load = {"task": "load_energy", "values": [10, 20, 30, 5, 80, 5]}
        self.assertEqual(thermo._answer(load), ("150", "int", {}))
        self.assertEqual(thermo._slots(load)["power[4-5]"], "80 kW")
        self.assertEqual(thermo._rounded(Fraction(2917, 100)), "29.2")
        self.assertEqual(thermo._rounded(Fraction(-25, 100)), "-0.3")
        self.assertTrue(thermo._tie(Fraction(1225, 100)))
        self.assertFalse(thermo._tie(Fraction(1250, 100)))
        degenerate = dict(sankey, useful=0, losses=[["Friction", 800]])
        with self.assertRaisesRegex(ValueError, "thermo:"):
            thermo._answer(degenerate)

    def test_degenerate_polygons_rejected(self):
        self.assertTrue(thermo._simple_polygon([(2, 100), (6, 100), (6, 300), (2, 300)]))
        self.assertTrue(thermo._simple_polygon([(2, 100), (2, 300), (6, 100)]))
        self.assertFalse(thermo._simple_polygon([(2, 100), (6, 300), (6, 100), (2, 300)]))  # bow tie
        self.assertFalse(thermo._simple_polygon([(2, 100), (4, 200), (6, 300)]))  # collinear: zero area
        self.assertFalse(thermo._simple_polygon([(2, 100), (4, 100), (4, 100), (2, 300)]))  # duplicate
        self.assertFalse(thermo._simple_polygon([(2, 100), (6, 100), (4, 100), (4, 300)]))  # 3 collinear
        self.assertTrue(thermo._segments_meet((2, 100), (6, 300), (6, 100), (2, 300)))
        self.assertTrue(thermo._segments_meet((2, 100), (6, 100), (4, 100), (8, 100)))  # overlap
        self.assertTrue(thermo._segments_meet((2, 100), (6, 100), (6, 100), (8, 300)))  # touch
        self.assertFalse(thermo._segments_meet((2, 100), (6, 100), (2, 300), (6, 300)))

    def test_sampled_structures_cover_every_variant_and_stay_exact(self):
        cycle_sizes, orientation, losses = set(), set(), set()
        lengths = set()
        for index in range(600):
            for task, offset in (("cycle_work", 0), ("sankey_efficiency", 1), ("load_energy", 2)):
                structure = thermo._select(index * 3 + offset, 3408, "train", "default")
                self.assertEqual(structure["task"], task)
                gold, answer_type, _ = thermo._answer(structure)
                if task == "cycle_work":
                    states = structure["states"]
                    cycle_sizes.add(len(states))
                    orientation.add(thermo._work_shoelace(structure) > 0)
                    for _, v, p in states:
                        self.assertTrue(1 <= v <= 10 and 100 <= p <= 900)
                    self.assertTrue(thermo._simple_polygon([(v, p) for _, v, p in states]))
                    self.assertEqual(thermo._work_shoelace(structure), thermo._work_legs(structure))
                elif task == "sankey_efficiency":
                    parts = [structure["useful"], *(kw for _, kw in structure["losses"])]
                    self.assertEqual(sum(parts), structure["input"])
                    self.assertTrue(100 <= structure["input"] <= 2000)
                    self.assertEqual(len({name for name, _ in structure["losses"]}), len(parts) - 1)
                    losses.add(len(structure["losses"]))
                    self.assertFalse(thermo._tie(thermo._efficiency(structure)))
                else:
                    values = structure["values"]
                    lengths.add(len(values))
                    self.assertTrue(all(5 <= kw <= 80 for kw in values) and len(set(values)) > 1)
                    self.assertEqual(gold, str(sum(values)))
                    self.assertEqual(answer_type, "int")
        self.assertEqual(cycle_sizes, {3, 4})
        self.assertEqual(orientation, {True, False})
        self.assertEqual(losses, {1, 2, 3})
        self.assertEqual(lengths, set(range(6, 13)))

    def test_determinism_split_banding_and_tasks_by_index(self):
        state = random.getstate()
        by_split = {split: set() for split in SPLITS}
        for p in self.problems:
            split = p.lineage.split
            again = thermo.generate(int(p.actor.problem_id.split(":")[-2]), 3407, split, "obs", _style(split))
            self.assertEqual(p, again)
            self.assertTrue(matches_split(p.lineage.structure_hash, split, _style(split)))
            by_split[split].add(p.lineage.structure_hash)
        self.assertEqual(random.getstate(), state)
        for a in SPLITS:
            for b in SPLITS:
                if a != b:
                    self.assertTrue(by_split[a].isdisjoint(by_split[b]))
        for index, p in enumerate(self.problems):
            self.assertEqual(p.lineage.template_id, thermo._TASKS[index % 3])
            self.assertEqual(p.actor.source, thermo.GENERATOR_VERSION)
        plain = thermo.generate(0, 3407, "train", "plain")
        self.assertEqual(plain.image_bytes, self.problems[0].image_bytes)
        self.assertEqual(plain.verifier, self.problems[0].verifier)

    def test_self_checks_and_actual_consumer_contracts(self):
        questions = {}
        for p in self.problems:
            thermo.self_check(p)
            self.assertIs(compare(p.verifier.gold, p.verifier), True)
            task = p.lineage.template_id
            if task == "load_energy":
                self.assertEqual(p.actor.answer_format, "int")
                self.assertIs(compare(str(int(p.verifier.gold) + 1), p.verifier), False)
            else:
                self.assertEqual(p.actor.answer_format, "decimal")
                self.assertIs(compare(str(float(p.verifier.gold) + 1), p.verifier), False)
            text = "Observed facts:\n" + "\n".join(p.verifier.oracle_fact_lines) + "\nSolution:\n"
            report = compare_slots(parse_observations(text), p.verifier)
            self.assertEqual(report.all_correct, len(p.verifier.slots))
            self.assertEqual(report.required_correct, len(p.verifier.required_slots))
            self.assertEqual(report.wrong_required, ())
            self.assertEqual(report.conflicting, ())
            self.assertEqual(set(p.verifier.slots), set(p.verifier.required_slots))
            self.assertEqual(p.actor.choices, ())
            self.assertFalse(p.lineage.benchmark_derived or p.lineage.model_touched)
            for forbidden in ('"gold":', '"slots":', '"accept":'):
                self.assertNotIn(forbidden, to_json(p.actor))
            questions[task] = p.actor.question
            # Every slot value is drawn in the picture, so the answer depends only on readable values.
            rows = [line.split() for line in p.verifier.printed_text]
            for key, value in p.verifier.slots.items():
                if task == "cycle_work":
                    self.assertIn([key[6], *value.split(",")], rows)
                elif task == "sankey_efficiency":
                    self.assertIn(f"{key[5:-1]}: {value}", p.verifier.printed_text)
                else:
                    self.assertIn(value, p.verifier.printed_text)
        self.assertTrue(questions["cycle_work"].startswith(
            "The diagram shows a cycle of an ideal gas. What is the net work done BY the gas during "
            "one cycle, in kJ? (Positive if the gas does net work.)"))
        self.assertTrue(questions["sankey_efficiency"].startswith(
            "What is the efficiency of the device, in percent?"))
        self.assertEqual(questions["load_energy"],
                         "How much electrical energy is consumed over the whole period shown, in kWh?")

    def test_png_and_style_shift(self):
        for p in self.problems[:12]:
            self.assertEqual(len(p.image_bytes), 1)
            self.assertEqual(hashlib.sha256(p.image_bytes[0]).hexdigest(), p.actor.image_sha256[0])
            self.assertTrue(set(png_chunks(p.image_bytes[0])) <= {b"IHDR", b"IDAT", b"IEND"})
            with Image.open(io.BytesIO(p.image_bytes[0])) as image:
                self.assertGreaterEqual(image.width, 1000)
                self.assertGreaterEqual(image.height, 500)
                self.assertEqual(image.info, {})
        for index in range(3):
            structure = thermo._select(index, 3407, "train", "default")
            render = thermo._RENDER[structure["task"]]
            raw, printed = render(structure, "default")
            shifted, shifted_printed = render(structure, "shift")
            self.assertNotEqual(raw, shifted)
            self.assertEqual(printed, shifted_printed)

    def test_generate_pool_is_deterministic_and_unique(self):
        for split in ("train", "sealed"):
            first = generate_pool("thermo", 6, 3407, split)
            second = generate_pool("thermo", 6, 3407, split)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 6)
            self.assertEqual(len({p.lineage.structure_hash for p in first}), 6)
            for p in first:
                self.assertEqual(p.lineage.split, split)
                self.assertEqual(p.lineage.generator_version, thermo.GENERATOR_VERSION)

    def test_invalid_tampered_inputs_and_solver_disagreement_fail_closed(self):
        p = self.problems[0]
        for item in (replace(p, verifier=replace(p.verifier, gold="0.0")),
                     replace(p, image_bytes=(p.image_bytes[0][:-1],)),
                     replace(p, actor=replace(p.actor, family="table")),
                     replace(p, verifier=replace(p.verifier, slots={})),
                     replace(p, lineage=replace(p.lineage, generator_version="other"))):
            with self.assertRaisesRegex(ValueError, "thermo:"):
                thermo.self_check(item)
        for delta in ({"index": True}, {"seed": -1}, {"index": -1}, {"mode": "other"},
                      {"split": "other"}, {"style": "shift"}, {"split": "style_sealed"}):
            with self.assertRaises(ValueError):
                thermo.generate(**({"index": 0, "seed": 3407, "split": "train", "mode": "obs"} | delta))
        with patch.object(thermo, "_work_legs", return_value=Fraction(0)):
            with self.assertRaisesRegex(ValueError, "solvers disagree"):
                thermo.generate(0, 3407, "train", "obs")
        with patch.object(thermo, "_efficiency_from_losses", return_value=Fraction(0)):
            with self.assertRaisesRegex(ValueError, "solvers disagree"):
                thermo.generate(1, 3407, "train", "obs")
        with patch.object(thermo, "matches_split", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "1000 candidates"):
                thermo.generate(0, 3407, "train", "obs")


if __name__ == "__main__":
    unittest.main()
