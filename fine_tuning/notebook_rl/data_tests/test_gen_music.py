"""Authored notation checks; these do not certify human render review."""
import hashlib
import io
import random
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml
from music21 import converter
from PIL import Image

from fine_tuning.notebook_rl.answers import compare, extract_final
from fine_tuning.notebook_rl.generators import music
from fine_tuning.notebook_rl.generators.common import matches_split, png_chunks
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.schema import from_json, to_json


class MusicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = []
        for i in range(500):
            split = ("train", "dev", "sealed", "style_sealed")[i % 4]
            cls.problems.append(music.generate(i, 3407, split, "obs",
                                               "shift" if split == "style_sealed" else "default"))

    def test_500_dual_solver_render_selfchecks_and_disjointness(self):
        for p in self.problems:
            music.self_check(p)
            eligible = [s for s in ("train", "dev", "sealed", "style_sealed")
                        if matches_split(p.lineage.structure_hash, s,
                                         "shift" if s == "style_sealed" else "default")]
            self.assertEqual(eligible, [p.lineage.split])
        self.assertEqual({p.lineage.template_id for p in self.problems}, set(music._TASKS))

    def test_same_question_image_dependency_100_pairs(self):
        buckets, pairs = {}, 0
        for p in self.problems:
            old = buckets.setdefault(p.actor.question, [])
            other = next((q for q in old if q.verifier.gold != p.verifier.gold), None)
            if other is not None:
                self.assertNotEqual(other.image_bytes, p.image_bytes)
                pairs += 1
                if pairs == 100:
                    break
            old.append(p)
        self.assertEqual(pairs, 100)

    def test_determinism_mode_and_global_rng(self):
        state = random.getstate()
        for i in (0, 1, 7, 19):
            p = self.problems[i]
            style = "shift" if p.lineage.split == "style_sealed" else "default"
            self.assertEqual(p, music.generate(i, 3407, p.lineage.split, "obs", style))
            plain = music.generate(i, 3407, p.lineage.split, "plain", style)
            self.assertEqual(plain.image_bytes, p.image_bytes)
            self.assertEqual(plain.verifier, p.verifier)
        self.assertEqual(random.getstate(), state)

    def test_exact_answer_slot_and_json_consumers(self):
        config = yaml.safe_load((Path(__file__).parents[1] / "prompts_v1.yaml").read_text())
        self.assertEqual(config["observation_instruction"]["music"], music.OBSERVATION_INSTRUCTION)
        for p in self.problems[:60]:
            final = extract_final("Solution: checked.\nFinal answer: " + p.verifier.gold, "stop")
            self.assertEqual(final.status, "OK")
            self.assertIs(compare(final.answer, p.verifier), True)
            text = "Observed facts:\n" + "\n".join(p.verifier.oracle_fact_lines) + "\nSolution:\n"
            report = compare_slots(parse_observations(text), p.verifier)
            self.assertEqual(report.all_correct, len(p.verifier.slots))
            self.assertEqual(report.required_correct, len(p.verifier.required_slots))
            self.assertEqual(report.wrong_required, ())
            for record in (p.actor, p.verifier, p.lineage):
                self.assertEqual(from_json(type(record), to_json(record)), record)
            self.assertEqual(p.actor.answer_format, "short_text")

    def test_musicxml_roundtrip_preserves_written_pitches_durations(self):
        for i, p in enumerate(self.problems[:40]):
            style = "shift" if p.lineage.split == "style_sealed" else "default"
            world = music._select(i, 3407, p.lineage.split, style)
            score = converter.parseData(music._xml(world), format="musicxml")
            notes = list(score.recurse().notes)
            self.assertEqual([n.nameWithOctave.replace("-", "b") for n in notes],
                             [music._pitch(*v) for v in world["pitches"]])
            self.assertEqual([n.lyric for n in notes],
                             [f"{j // world['beats'] + 1}.{j % world['beats'] + 1}"
                              for j in range(len(notes))])
            duration = .5 if world["beat_type"] == 8 else 1
            self.assertTrue(all(n.quarterLength == duration for n in notes))

    def test_explicit_accidental_cancellation_within_bar(self):
        world = {"clef": "treble", "sharps": 0, "beats": 4, "beat_type": 4,
                 "measures": 1, "pitches": [[28, 1], [28, 0], [28, 0], [28, -1]]}
        tree = ET.fromstring(music._xml(world))
        self.assertEqual([n.findtext("accidental") for n in tree.findall(".//note")],
                         ["sharp", "natural", None, "flat"])

    def test_hand_calculated_theory(self):
        p = music.generate(0, 3407, "train", "obs")
        self.assertEqual(p.verifier.gold, "m3")
        self.assertFalse(compare("M3", p.verifier))
        for first, second, expected in (([28, 0], [30, 0], "M3"), ([28, 0], [30, -1], "m3"),
                                         ([28, 0], [32, 0], "P5"), ([35, 0], [28, 0], "P8"),
                                         ([28, 0], [37, 0], "M10")):
            self.assertEqual(music._manual_interval(first, second), expected)
        self.assertEqual(music._signature(-3), {"B": -1, "E": -1, "A": -1})
        self.assertEqual(music._pitch(28, -1), "Cb4")
        with patch.object(music, "_manual_interval", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "solvers disagree"):
                music.generate(0, 3407, "train", "obs")

    def test_png_nonblank_hash_metadata_and_style(self):
        for p in self.problems[:16]:
            raw = p.image_bytes[0]
            self.assertEqual(hashlib.sha256(raw).hexdigest(), p.actor.image_sha256[0])
            self.assertTrue(set(png_chunks(raw)) <= {b"IHDR", b"IDAT", b"IEND"})
            with Image.open(io.BytesIO(raw)) as picture:
                self.assertGreaterEqual(picture.width, 400)
                self.assertGreaterEqual(picture.height, 80)
                self.assertLess(picture.convert("L").getextrema()[0], 50)
                self.assertEqual(picture.getpixel((0, 0))[:3], (255, 255, 255))
        world = music._select(0, 3407, "train", "default")
        self.assertNotEqual(music._render(world, "default"), music._render(world, "shift"))

    def test_tampering_and_invalid_inputs(self):
        p = self.problems[0]
        for bad in (replace(p, verifier=replace(p.verifier, gold="wrong")),
                    replace(p, verifier=replace(p.verifier, accept={"aliases": ["anything"]})),
                    replace(p, actor=replace(p.actor, question="wrong")),
                    replace(p, image_bytes=(p.image_bytes[0][:-1],))):
            with self.assertRaises(ValueError):
                music.self_check(bad)
        for override in ({"index": -1}, {"index": True}, {"seed": True},
                         {"mode": "other"}, {"style": "shift"}):
            with self.assertRaises(ValueError):
                music.generate(**({"index": 0, "seed": 3407, "split": "train", "mode": "obs"} | override))


if __name__ == "__main__":
    unittest.main()
