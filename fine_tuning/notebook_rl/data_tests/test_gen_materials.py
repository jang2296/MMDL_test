"""Authored stress-strain checks: exact solvers, degenerate rejection, records, consumers."""
import hashlib
import io
import random
import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl.answers import compare, number
from fine_tuning.notebook_rl.dataset import generate_pool
from fine_tuning.notebook_rl.generators import materials
from fine_tuning.notebook_rl.generators.common import matches_split, png_chunks
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.schema import to_json

SPLITS = ("train", "dev", "sealed", "style_sealed")


def _style(split):
    return "shift" if split == "style_sealed" else "default"


class MaterialsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = [materials.generate(i, 3407, SPLITS[i % 4], "obs", _style(SPLITS[i % 4]))
                        for i in range(120)]

    def test_hand_checked_solver_values_for_each_task(self):
        # P at 0.2 % and 400 MPa: 400 MPa / 0.002 = 200 GPa; peak 700 MPa; fracture at 14.0 %.
        first = {"points": [["P", 2, 400], ["H1", 30, 520], ["H2", 60, 610],
                            ["U", 100, 700], ["F", 140, 520]]}
        expected = {"youngs_modulus": Fraction(200), "ultimate_strength": Fraction(700),
                    "fracture_strain": Fraction(14)}
        # P at 0.3 % and 300 MPa: 300 / 0.003 = 100 GPa; peak 690 MPa; fracture at 15.0 %.
        second = {"points": [["P", 3, 300], ["H1", 20, 420], ["H2", 50, 560], ["H3", 90, 640],
                             ["U", 120, 690], ["F", 150, 500]]}
        again = {"youngs_modulus": Fraction(100), "ultimate_strength": Fraction(690),
                 "fracture_strain": Fraction(15)}
        # P at 0.4 % and 960 MPa is 240 GPa; a fracture strain of 7.5 % is not a whole percent.
        third = {"points": [["P", 4, 960], ["H1", 15, 1010], ["H2", 40, 1100],
                            ["U", 60, 1180], ["F", 75, 900]]}
        last = {"youngs_modulus": Fraction(240), "ultimate_strength": Fraction(1180),
                "fracture_strain": Fraction(15, 2)}
        for structure, answers in ((first, expected), (second, again), (third, last)):
            materials._validate(structure)
            self.assertEqual(materials._solve(structure), answers)
            self.assertEqual(materials._fields(structure), answers)
        self.assertEqual(materials._slots(first)["point[P]"], "0.2,400")
        self.assertEqual(materials._slots(third)["point[F]"], "7.5,900")
        for value, text in ((Fraction(14), "14.0"), (Fraction(15, 2), "7.5"),
                            (Fraction(1, 2), "0.5"), (Fraction(0), "0.0")):
            self.assertEqual(materials._decimal(value), text)
        for bad in (Fraction(1, 3), Fraction(-1, 2), Fraction(1, 20)):
            with self.assertRaisesRegex(ValueError, "materials:"):
                materials._decimal(bad)

    def test_degenerate_curves_are_rejected(self):
        good = [["P", 2, 400], ["H1", 30, 520], ["H2", 60, 610], ["U", 100, 700], ["F", 140, 520]]
        materials._validate({"points": good})

        def changed(edits):
            points = [list(p) for p in good]
            for (row, column), value in edits.items():
                points[row][column] = value
            return {"points": points}

        cases = {
            "strain equal": changed({(2, 1): 30}),
            "strain decreasing": changed({(2, 1): 20}),
            "fracture strain before U": changed({(4, 1): 90}),
            "hardening stress falling": changed({(2, 2): 500}),
            "U not maximum": changed({(2, 2): 800}),
            "F equals U": changed({(4, 2): 700}),
            "F above U": changed({(4, 2): 750}),
            "F not positive": changed({(4, 2): 0}),
            "modulus not integer": changed({(0, 2): 401}),
            "modulus below 50": changed({(0, 2): 98}),
            "modulus above 250": changed({(0, 2): 502}),
            "non integer value": changed({(1, 2): 520.5}),
        }
        for name, structure in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "materials:"):
                materials._validate(structure)
        for points in (good[:2] + good[3:], good[::-1], good[:1] + good[2:],
                       [["P", 2, 400], ["H1", 30, 520], ["H2", 60, 610], ["H3", 70, 650],
                        ["H4", 80, 660], ["U", 100, 700], ["F", 140, 520]]):
            with self.assertRaisesRegex(ValueError, "materials:"):
                materials._validate({"points": points})

    def test_500_selected_structures_are_valid_and_diverse(self):
        counts, tasks, moduli = set(), set(), set()
        for index in range(500):
            structure = materials._select(index, 3408, "train", "default")
            materials._validate(structure)
            answers = materials._solve(structure)
            self.assertEqual(answers, materials._fields(structure))
            self.assertEqual(answers["youngs_modulus"].denominator, 1)
            self.assertTrue(50 <= answers["youngs_modulus"] <= 250)
            counts.add(len(structure["points"]))
            tasks.add(materials._TASKS[index % 3])
            moduli.add(int(answers["youngs_modulus"]))
        self.assertEqual(counts, {5, 6})
        self.assertEqual(tasks, set(materials._TASKS))
        self.assertGreater(len(moduli), 100)

    def test_determinism_and_split_banding_with_self_checks(self):
        by_split = {s: set() for s in SPLITS}
        for p in self.problems[:60]:
            materials.self_check(p)
            split = p.lineage.split
            by_split[split].add(p.lineage.structure_hash)
            self.assertTrue(matches_split(p.lineage.structure_hash, split, _style(split)))
            self.assertEqual(p.lineage.template_id, materials._TASKS[
                int(p.actor.problem_id.split(":")[-2]) % 3])
        for a in by_split:
            for b in by_split:
                if a != b:
                    self.assertTrue(by_split[a].isdisjoint(by_split[b]))
        state = random.getstate()
        for index in (0, 1, 2, 3, 57, 118):
            p = self.problems[index]
            split = p.lineage.split
            self.assertEqual(p, materials.generate(index, 3407, split, "obs", _style(split)))
        self.assertEqual(random.getstate(), state)
        plain = materials.generate(0, 3407, "train", "plain")
        self.assertEqual(plain.image_bytes, self.problems[0].image_bytes)
        self.assertEqual(plain.verifier, self.problems[0].verifier)
        self.assertEqual(plain.actor.mode, "plain")

    def test_answers_questions_and_actual_consumer_contracts(self):
        for p in self.problems[:60]:
            gold, task = p.verifier.gold, p.lineage.template_id
            self.assertIs(compare(gold, p.verifier), True)
            step = Fraction(1) if task != "fracture_strain" else Fraction(1, 10)
            wrong = number(gold) + step
            self.assertIs(compare(str(wrong) if step == 1 else materials._decimal(wrong),
                                  p.verifier), False)
            if task == "fracture_strain":
                self.assertEqual(p.actor.answer_format, "decimal")
                self.assertEqual(p.verifier.accept, {"round_to": 1})
                self.assertRegex(gold, r"^\d+\.\d$")
                self.assertIs(compare(str(number(gold) + Fraction(1, 100)), p.verifier), True)
            else:
                self.assertEqual(p.actor.answer_format, "int")
                self.assertRegex(gold, r"^\d+$")
            text = "Observed facts:\n" + "\n".join(p.verifier.oracle_fact_lines) + "\nSolution:\n"
            report = compare_slots(parse_observations(text), p.verifier)
            self.assertEqual(report.all_correct, len(p.verifier.slots))
            self.assertEqual(report.required_correct, len(p.verifier.required_slots))
            self.assertEqual(report.wrong_required, ())
            self.assertEqual(report.conflicting, ())
            self.assertEqual(set(p.verifier.slots), set(p.verifier.required_slots))
            self.assertTrue(set(p.verifier.slot_types.values()) == {"text"})
            self.assertEqual(p.actor.choices, ())
            self.assertFalse(p.lineage.benchmark_derived or p.lineage.model_touched)
            for forbidden in ('"gold":', '"slots":', '"accept":'):
                self.assertNotIn(forbidden, to_json(p.actor))
        questions = {t: materials._question(t)[0] for t in materials._TASKS}
        self.assertIn("Young's modulus", questions["youngs_modulus"])
        self.assertIn("in GPa", questions["youngs_modulus"])
        self.assertIn("ultimate tensile strength", questions["ultimate_strength"])
        self.assertIn("in MPa", questions["ultimate_strength"])
        self.assertIn("strain at fracture, in percent", questions["fracture_strain"])
        self.assertIn("point[<label>]", materials.OBSERVATION_INSTRUCTION)
        self.assertTrue(materials.OBSERVATION_INSTRUCTION.endswith("\n"))

    def test_80_same_question_image_dependent_pairs(self):
        buckets, pairs, templates = {}, 0, set()
        for p in self.problems:
            prior = buckets.setdefault(p.actor.question, [])
            other = next((q for q in prior if q.verifier.gold != p.verifier.gold), None)
            if other is not None:
                self.assertNotEqual(other.image_bytes, p.image_bytes)
                self.assertNotEqual(other.lineage.structure_hash, p.lineage.structure_hash)
                templates.add(p.lineage.template_id)
                pairs += 1
                if pairs == 80:
                    break
            prior.append(p)
        self.assertEqual(pairs, 80)
        self.assertEqual(templates, set(materials._TASKS))

    def test_png_printed_text_and_style(self):
        for p in self.problems[:12]:
            self.assertEqual(hashlib.sha256(p.image_bytes[0]).hexdigest(), p.actor.image_sha256[0])
            self.assertTrue(set(png_chunks(p.image_bytes[0])) <= {b"IHDR", b"IDAT", b"IEND"})
            with Image.open(io.BytesIO(p.image_bytes[0])) as image:
                self.assertGreaterEqual(image.width, 1000)
                self.assertGreaterEqual(image.height, 500)
                self.assertEqual(image.info, {})
            for label, value in p.verifier.slots.items():
                strain, stress = value.split(",")
                self.assertIn(f"{label[6:-1]} ({strain}, {stress})", p.verifier.printed_text)
            for text in ("Strain (%)", "Stress (MPa)"):
                self.assertIn(text, p.verifier.printed_text)
        structure = materials._select(0, 3407, "train", "default")
        raw, printed = materials._render(structure, "default")
        shifted, shifted_printed = materials._render(structure, "shift")
        self.assertNotEqual(raw, shifted)
        self.assertEqual(printed, shifted_printed)

    def _assert_collision_free(self, structure, style):
        captured = {}

        def capture(figure, *, dpi):
            captured["figure"], captured["dpi"] = figure, dpi
            return b""

        with patch.object(materials, "png_bytes", capture):
            materials._render(structure, style)
        figure = captured["figure"]
        self.assertEqual(figure.dpi, captured["dpi"])
        figure.draw_without_rendering()
        renderer = figure.canvas.get_renderer()
        ax = figure.axes[0]
        axes_box = ax.get_window_extent(renderer)
        ticks = [t.get_window_extent(renderer)
                 for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text()]
        curve_path = ax.lines[0].get_transform().transform_path(ax.lines[0].get_path())
        boxes = [t.get_window_extent(renderer) for t in ax.texts]
        self.assertEqual(len(boxes), len(structure["points"]))
        for i, box in enumerate(boxes):
            label = ax.texts[i].get_text()
            self.assertTrue(axes_box.x0 <= box.x0 and axes_box.y0 <= box.y0
                            and box.x1 <= axes_box.x1 and box.y1 <= axes_box.y1,
                            f"{label} leaves the axes")
            self.assertFalse(any(box.overlaps(t) for t in ticks), f"{label} hits a tick label")
            self.assertFalse(any(box.overlaps(o) for o in boxes[:i]), f"{label} hits a label")
            self.assertFalse(curve_path.intersects_bbox(box, filled=False),
                             f"{label} crosses the curve")

    def test_point_labels_are_collision_free(self):
        for i in range(15):
            for split, style in (("train", "default"), ("style_sealed", "shift")):
                with self.subTest(index=i, style=style):
                    self._assert_collision_free(materials._select(i, 3407, split, style), style)
        # The first train candidate here (P at 0.1 %, 52 MPa under a 700 MPa axis) has no clear
        # P label; selection must skip it instead of failing the whole pool.
        self._assert_collision_free(materials._select(387, 9001, "train", "default"), "default")
        adversarial = (
            {"points": [["P", 1, 250], ["H1", 11, 290], ["H2", 21, 330], ["U", 31, 370],
                        ["F", 91, 200]]},
            {"points": [["P", 4, 200], ["H1", 14, 240], ["H2", 24, 280], ["H3", 34, 320],
                        ["U", 44, 360], ["F", 104, 190]]},
            {"points": [["P", 2, 500], ["H1", 12, 540], ["H2", 22, 580], ["H3", 62, 1100],
                        ["U", 162, 1250], ["F", 222, 630]]},
            {"points": [["P", 4, 200], ["H1", 14, 350], ["H2", 24, 500], ["U", 34, 650],
                        ["F", 94, 330]]},
        )
        for n, structure in enumerate(adversarial):
            materials._validate(structure)
            for style in ("default", "shift"):
                with self.subTest(adversarial=n, style=style):
                    self._assert_collision_free(structure, style)

    def test_label_placement_fails_closed(self):
        structure = materials._select(0, 3407, "train", "default")
        with patch.object(materials, "_candidates",
                          return_value=(((-3000, 0), "right", "center"),)):
            with self.assertRaisesRegex(ValueError,
                                        "materials: no collision-free label placement"):
                materials._render(structure, "default")

    def test_invalid_tampered_inputs_and_solver_disagreement_fail_closed(self):
        p = self.problems[0]
        for item in (replace(p, verifier=replace(p.verifier, gold="1")),
                     replace(p, image_bytes=(p.image_bytes[0][:-1],)),
                     replace(p, actor=replace(p.actor, family="table")),
                     replace(p, verifier=replace(p.verifier, slots={}, slot_types={},
                                                 required_slots=())),
                     replace(p, lineage=replace(p.lineage, generator_version="other"))):
            with self.assertRaisesRegex(ValueError, "materials:"):
                materials.self_check(item)
        with self.assertRaisesRegex(ValueError, "materials:"):
            materials.self_check("not a problem")
        for delta in ({"index": True}, {"seed": -1}, {"index": -1}, {"mode": "other"},
                      {"split": "other"}, {"style": "shift"}, {"split": "style_sealed"}):
            with self.assertRaises(ValueError):
                materials.generate(**({"index": 0, "seed": 3407, "split": "train", "mode": "obs"}
                                      | delta))
        with patch.object(materials, "_fields", return_value={}):
            with self.assertRaisesRegex(ValueError, "solvers disagree"):
                materials.generate(0, 3407, "train", "obs")
        with patch.object(materials, "matches_split", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "1000 candidates"):
                materials.generate(0, 3407, "train", "obs")

    def test_generate_pool_train_and_sealed(self):
        for split in ("train", "sealed"):
            pool = generate_pool("materials", 6, 3407, split)
            self.assertEqual(len(pool), 6)
            self.assertEqual(len({p.lineage.structure_hash for p in pool}), 6)
            for p in pool:
                self.assertEqual(p.lineage.split, split)
                self.assertEqual(p.actor.family, "materials")
                self.assertIs(compare(p.verifier.gold, p.verifier), True)


if __name__ == "__main__":
    unittest.main()
