"""Authored pixel/structure fixtures, not benchmark or training records."""
import io
import unittest
from dataclasses import asdict

from PIL import Image

from fine_tuning.notebook_rl.generators.common import (
    build, matches_split, png_bytes, png_chunks, split_of, structure_hash,
)
from fine_tuning.notebook_rl.schema import validate_actor


class CommonTests(unittest.TestCase):
    def test_canonical_structure_hash(self):
        self.assertEqual(structure_hash({"a": 1, "b": [2]}), structure_hash({"b": [2], "a": 1}))
        for obj in ({1: 2}, {"a": float("nan")}, {"a": {1: 2}}, object()):
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                structure_hash(obj)

    def test_png_determinism_pixels_and_metadata(self):
        with Image.new("RGB", (12, 9), (21, 67, 230)) as image:
            a = png_bytes(image, dpi=100)
            image.info.update({"Software": "irrelevant", "date": "not serialized", "comment": "gold"})
            b = png_bytes(image, dpi=200)
            self.assertEqual(a, b)
            self.assertEqual(png_chunks(a), (b"IHDR", b"IDAT", b"IEND"))
            with Image.open(io.BytesIO(a)) as loaded:
                self.assertEqual(loaded.tobytes(), image.tobytes())
            self.assertEqual(image.info["comment"], "gold")
        with Image.new("RGBA", (2, 2), (255, 0, 0, 40)) as image:
            raw = png_bytes(image, dpi=72)
            with Image.open(io.BytesIO(raw)) as loaded:
                self.assertEqual(image.tobytes(), loaded.tobytes())
        for invalid in (b"not png", a[:-1], a + b"junk", a[:40] + b"bad" + a[43:]):
            with self.assertRaises(ValueError):
                png_chunks(invalid)

    def test_figure_png(self):
        from matplotlib.figure import Figure
        figure = Figure(figsize=(2, 1))
        figure.subplots().plot([0, 1], [1, 0])
        self.assertEqual(png_bytes(figure, dpi=72), png_bytes(figure, dpi=72))
        for dpi in (0, -1, True, "100"):
            with self.assertRaises(ValueError):
                png_bytes(figure, dpi=dpi)

    def test_split_distribution_and_boundaries(self):
        counts = dict.fromkeys(("train", "dev", "sealed"), 0)
        for i in range(10000):
            counts[split_of(structure_hash({"index": i}))] += 1
        for split, fraction in zip(counts, (0.8, 0.1, 0.1), strict=True):
            self.assertLess(abs(counts[split] / 10000 - fraction), 0.02)
        self.assertEqual(split_of("0000000000000000"), "train")
        self.assertEqual(split_of("ffffffffffffffff"), "sealed")
        for ratios in ((0.8, 0.1), (0.8, 0.1, 0.2), (1, 0, 0), (True, 0.1, 0.1)):
            with self.assertRaises(ValueError):
                split_of("0000000000000000", ratios)

    def test_shifted_structures_are_disjoint(self):
        for i in range(1000):
            digest = structure_hash({"i": i})
            eligible = [s for s in ("train", "dev", "sealed", "style_sealed")
                        if matches_split(digest, s, "shift" if s == "style_sealed" else "default")]
            self.assertEqual(len(eligible), 1)
        with self.assertRaisesRegex(ValueError, "exclusive"):
            matches_split("ffffffffffffffff", "train", "shift")

    def test_build_validated_separate_records(self):
        structure = {"fixture": "no benchmark", "cells": [3, 4]}
        digest = structure_hash(structure)
        split = split_of(digest)
        if split == "sealed" and not matches_split(digest, split):
            split = "style_sealed"
        with Image.new("RGB", (4, 4), "white") as image:
            raw = png_bytes(image, dpi=72)
        kwargs = dict(source="fixture", family="table", index=0, split=split, mode="obs",
                      question="What is the sum of the two displayed cells?", choices=(),
                      answer_type="int", gold="7", accept={}, slots={"a": "3", "b": "4"},
                      slot_types={"a": "number", "b": "number"}, required_slots=("a", "b"),
                      oracle_fact_lines=("a=3", "b=4"), printed_text=("3", "4"),
                      structure=structure, template_id="sum", image_bytes=(raw,),
                      generator_version="fixture-v1", seed=3407,
                      style="shift" if split == "style_sealed" else "default")
        p = build(**kwargs)
        validate_actor(p.actor)
        self.assertEqual(build(**kwargs), p)
        self.assertNotIn("gold", asdict(p.actor))
        self.assertEqual(p.verifier.gold, "7")
        self.assertEqual(p.actor.problem_id, p.lineage.problem_id)
        self.assertFalse(p.lineage.benchmark_derived)
        self.assertFalse(p.lineage.model_touched)
        for update in ({"source": "../escape"}, {"index": True}, {"image_bytes": ()},
                       {"gold": ""}, {"required_slots": ("missing",)}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                build(**(kwargs | update))


if __name__ == "__main__":
    unittest.main()
