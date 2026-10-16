"""CPU authored-chart contract checks; no benchmark, model, or approval claims."""
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
from fine_tuning.notebook_rl.generators import chart
from fine_tuning.notebook_rl.generators.common import matches_split, png_chunks, structure_hash
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.schema import to_json


class ChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = []
        for index in range(1000):
            split = ("train", "dev", "sealed", "style_sealed")[index % 4]
            cls.problems.append(chart.generate(index, 3407, split, "obs",
                                               "shift" if split == "style_sealed" else "default"))

    def test_determinism_and_global_rng_independence(self):
        import random
        state = random.getstate()
        for index in (0, 1, 2, 3, 101, 107, 302, 399):
            p = self.problems[index]
            again = chart.generate(index, 3407, p.lineage.split, "obs",
                                   "shift" if p.lineage.split == "style_sealed" else "default")
            self.assertEqual(p, again)
        self.assertEqual(state, random.getstate())

    def test_500_self_checks_and_structure_split_disjointness(self):
        by_split = {key: set() for key in ("train", "dev", "sealed", "style_sealed")}
        for p in self.problems[:500]:
            chart.self_check(p)
            digest = p.lineage.structure_hash
            by_split[p.lineage.split].add(digest)
            eligible = [s for s in by_split if matches_split(
                digest, s, "shift" if s == "style_sealed" else "default")]
            self.assertEqual(eligible, [p.lineage.split])
        for left in by_split:
            for right in by_split:
                if left != right:
                    self.assertTrue(by_split[left].isdisjoint(by_split[right]))

    def test_100_same_question_different_images_change_answer_pairs(self):
        pairs = 0
        buckets = {}
        for p in self.problems:
            if p.actor.answer_format == "mcq":
                continue
            key = (p.actor.question, p.actor.family)
            bucket = buckets.setdefault(key, [])
            other = next((q for q in bucket if q.verifier.gold != p.verifier.gold), None)
            if other is not None:
                self.assertNotEqual(other.lineage.structure_hash, p.lineage.structure_hash)
                self.assertNotEqual(other.image_bytes, p.image_bytes)
                pairs += 1
                if pairs == 100:
                    break
            bucket.append(p)
        self.assertEqual(pairs, 100)

    def test_1000_answer_formats_mcq_distractors_and_template_coverage(self):
        open_count, combinations, positions = 0, set(), set()
        series_counts, item_counts, scales = set(), set(), set()
        for index, p in enumerate(self.problems):
            template = p.lineage.template_id
            combinations.add((p.actor.family, template))
            style = "shift" if p.lineage.split == "style_sealed" else "default"
            structure = chart._select(index, 3407, p.lineage.split, style)
            series_counts.add(len(structure["series"]))
            item_counts.add(len(structure["items"]))
            scales.add(structure["scale"])
            direct = chart._array_answer(structure, template)
            self.assertEqual(direct, chart._pandas_answer(structure, template))
            gold_value = direct if isinstance(direct, str) else chart._decimal(direct)
            if p.actor.answer_format == "mcq":
                self.assertEqual(len(p.actor.choices), 10)
                self.assertEqual(len(set(p.actor.choices)), 10)
                position = ord(p.verifier.gold) - 65
                positions.add(position)
                self.assertEqual(p.actor.choices[position], gold_value)
                self.assertEqual(p.actor.choices.count(gold_value), 1)
                self.assertEqual(p.actor.choices, p.verifier.choices)
                for value in p.actor.choices:
                    Fraction(value)
            else:
                open_count += 1
                self.assertEqual(p.verifier.gold, gold_value)
            self.assertFalse(p.lineage.benchmark_derived or p.lineage.model_touched)
            self.assertEqual(p.lineage.label_provenance, "PROGRAM")
            self.assertEqual(p.lineage.structure_hash, structure_hash(structure))
        self.assertGreaterEqual(open_count, 600)
        self.assertEqual(combinations, {(family, task) for family in chart._KINDS
                                        for task in chart._TASKS})
        self.assertEqual(series_counts, {2, 3, 4})
        self.assertEqual(item_counts, {4, 5, 6, 7, 8})
        self.assertEqual(scales, {1, 10})
        self.assertEqual(positions, set(range(10)))

    def test_family_slot_observation_and_required_fact_contracts(self):
        prompts = yaml.safe_load((Path(__file__).parents[1] / "prompts_v1.yaml").read_text())
        self.assertEqual(chart.OBSERVATION_INSTRUCTION,
                         prompts["observation_instruction"][chart.FAMILY])
        for p in self.problems[:60]:
            family = p.actor.family
            pattern = {"chart_bar": r"bar\[[^\]]+\]\[[^\]]+\]",
                       "chart_line": r"point\[[^\]]+\]\[[^\]]+\]",
                       "table": r"cell\[[^\]]+\]\[[^\]]+\]"}[family]
            for key, value in p.verifier.slots.items():
                self.assertRegex(key, "^" + pattern + "$")
                self.assertIn(value, p.verifier.printed_text)
                self.assertEqual(p.verifier.slot_types[key], "number")
            self.assertTrue(set(p.verifier.required_slots) <= p.verifier.slots.keys())
            self.assertEqual(p.verifier.oracle_fact_lines,
                             tuple(f"{k} = {v}" for k, v in p.verifier.slots.items()))
            if p.lineage.template_id == "growth":
                self.assertEqual(len(p.verifier.required_slots), 2)
            elif p.lineage.template_id == "margin":
                self.assertEqual(len(p.verifier.required_slots), 2)
            actor_json = to_json(p.actor)
            for forbidden in ('"gold":', '"slots":', '"required_slots":', '"accept":'):
                self.assertNotIn(forbidden, actor_json)

    def test_exact_independent_solvers_rounding_and_first_tie(self):
        structure = {"series": ["Revenue", "Cost"], "items": ["P1", "P2", "P3", "P4"],
                     "values": [[100, 120, 200, 300], [50, 40, 120, 230]], "scale": 1}
        for task, expected in (("growth", Fraction(20)), ("margin", Fraction(50)),
                               ("largest_difference", Fraction(80)), ("largest_item", "P2")):
            self.assertEqual(chart._array_answer(structure, task), expected)
            self.assertEqual(chart._pandas_answer(structure, task), expected)
        self.assertIn("If tied", chart._question("largest_item"))
        for number, expected in ((Fraction(25, 100), "0.3"), (Fraction(-25, 100), "-0.3"),
                                 (Fraction(1, 100), "0.0"), (Fraction(-1, 100), "0.0"),
                                 (Fraction(99995, 100), "1000.0")):
            self.assertEqual(chart._decimal(number), expected)
        with patch.object(chart, "_pandas_answer", return_value="broken"):
            with self.assertRaisesRegex(ValueError, "solvers disagree"):
                chart.generate(0, 3407, "train", "plain")

    def test_answer_and_observation_consumer_compatibility(self):
        for p in self.problems[:60]:
            with self.subTest(family=p.actor.family, task=p.lineage.template_id):
                self.assertIs(compare(p.verifier.gold, p.verifier), True)
                text = "Observed facts:\n" + "\n".join(p.verifier.oracle_fact_lines) + "\nSolution:\n"
                report = compare_slots(parse_observations(text), p.verifier)
                self.assertEqual(report.all_correct, len(p.verifier.slots))
                self.assertEqual(report.required_correct, len(p.verifier.required_slots))
                self.assertEqual(report.wrong_required, ())
                self.assertEqual(report.conflicting, ())

    def test_png_metadata_hash_decode_and_style_render_change(self):
        for p in self.problems[:16]:
            raw = p.image_bytes[0]
            self.assertEqual(hashlib.sha256(raw).hexdigest(), p.actor.image_sha256[0])
            self.assertTrue(set(png_chunks(raw)) <= {b"IHDR", b"IDAT", b"IEND"})
            with Image.open(io.BytesIO(raw)) as image:
                self.assertIn(image.mode, ("RGB", "RGBA"))
                self.assertGreaterEqual(image.width, 1000)
                self.assertGreaterEqual(image.height, 600)
                self.assertEqual(image.info, {})
        structure = chart._select(0, 3407, "train", "default")
        for family in chart._KINDS:
            default, printed = chart._render(structure, family, "default")
            shifted, printed_shift = chart._render(structure, family, "shift")
            self.assertNotEqual(default, shifted)
            self.assertEqual(printed, printed_shift)
        plain = chart.generate(0, 3407, "train", "plain")
        self.assertEqual(plain.image_bytes, self.problems[0].image_bytes)
        self.assertEqual(plain.verifier, self.problems[0].verifier)

    def test_self_check_rejects_tampered_records_and_inputs(self):
        p = self.problems[0]
        for malformed in (replace(p, verifier=replace(p.verifier, gold="wrong")),
                          replace(p, actor=replace(p.actor, question="wrong")),
                          replace(p, verifier=replace(p.verifier, slots={"wrong": "1"})),
                          replace(p, image_bytes=(p.image_bytes[0][:-1],)),
                          replace(p, lineage=replace(p.lineage, structure_hash="0" * 16)),
                          replace(p, lineage=replace(p.lineage, generator_version="other"))):
            with self.assertRaisesRegex(ValueError, "chart:"):
                chart.self_check(malformed)
        for changes in ({"index": -1}, {"index": True}, {"seed": -1}, {"seed": True},
                        {"split": "other"}, {"mode": "other"}, {"style": "other"},
                        {"style": "shift"}, {"split": "style_sealed"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                chart.generate(**({"index": 0, "seed": 3407, "split": "train", "mode": "obs"}
                                  | changes))
        with patch.object(chart, "matches_split", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "1000 candidates"):
                chart.generate(0, 3407, "train", "obs")


if __name__ == "__main__":
    unittest.main()
