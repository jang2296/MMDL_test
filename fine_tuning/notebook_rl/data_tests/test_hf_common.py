"""Authored fixtures only; no real dataset rows, upstream images, or benchmark data."""
import hashlib
import io
import unittest

from PIL import Image, PngImagePlugin

from fine_tuning.notebook_rl.adapters import hf_common as hf
from fine_tuning.notebook_rl.generators.common import png_chunks
from fine_tuning.notebook_rl.schema import to_json


def encode(image, fmt="PNG", **kwargs):
    out = io.BytesIO()
    image.save(out, format=fmt, **kwargs)
    return out.getvalue()


def decode(raw):
    return Image.open(io.BytesIO(raw))


SPEC = hf.SourceSpec("hf-test", "org/dataset", "a" * 40, "cc-by-4.0", "HUMAN", "authored annotators", "test_family")


def candidate(**overrides):
    fields = {"source": "hf-test", "native_id": "row-7", "subject": "Biology", "question": "Which is green?",
              "choices": ("Leaf", "Sky", "Sea", "Sand"), "answer_type": "mcq", "gold": "A",
              "images": (encode(Image.new("RGB", (16, 12), (10, 200, 30))),)}
    fields.update(overrides)
    return hf.Candidate(**fields)


class NormalizeImageTests(unittest.TestCase):
    def test_rgba_is_composited_on_white(self):
        rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
        rgba.putpixel((0, 0), (0, 0, 0, 128))
        rgba.putpixel((1, 0), (0, 0, 255, 255))
        with decode(hf.normalize_image(encode(rgba))) as out:
            self.assertEqual(out.mode, "RGB")
            self.assertEqual(out.getpixel((3, 3)), (255, 255, 255))
            self.assertEqual(out.getpixel((1, 0)), (0, 0, 255))
            self.assertTrue(all(120 <= v <= 135 for v in out.getpixel((0, 0))))

    def test_oversize_downscaled_keeping_aspect(self):
        raw = encode(Image.new("RGB", (1800, 1200), (5, 6, 7)))
        with decode(hf.normalize_image(raw)) as out:
            width, height = out.size
        self.assertLessEqual(width * height, hf.MAX_STORED_PIXELS)
        self.assertGreater(width * height, 0.98 * hf.MAX_STORED_PIXELS)
        self.assertAlmostEqual(width / height, 1.5, delta=0.01)

    def test_small_image_keeps_size_and_pixels(self):
        src = Image.new("RGB", (9, 7), (1, 2, 3))
        src.putpixel((2, 2), (200, 100, 50))
        with decode(hf.normalize_image(encode(src))) as out:
            self.assertEqual(out.size, (9, 7))
            self.assertEqual(out.getpixel((2, 2)), (200, 100, 50))

    def test_animated_gif_rejected(self):
        frames = [Image.new("RGB", (6, 6), (255, 0, 0)), Image.new("RGB", (6, 6), (0, 0, 255))]
        raw = encode(frames[0], "GIF", save_all=True, append_images=frames[1:])
        self.assertGreater(decode(raw).n_frames, 1)
        with self.assertRaises(ValueError):
            hf.normalize_image(raw)

    def test_output_is_clean_png(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("comment", "metadata that must not survive")
        raw = encode(Image.new("RGB", (8, 8), (9, 9, 9)), pnginfo=info)
        self.assertIn(b"tEXt", raw)
        out = hf.normalize_image(raw)
        self.assertNotIn(b"tEXt", out)
        self.assertEqual(png_chunks(out)[0], b"IHDR")

    def test_non_image_bytes_rejected(self):
        with self.assertRaises(Exception):
            hf.normalize_image(b"not an image")


class PixelHashTests(unittest.TestCase):
    def test_stable_across_reencoding_and_sensitive_to_pixels(self):
        image = Image.new("RGB", (20, 10), (3, 4, 5))
        image.putpixel((4, 4), (250, 0, 0))
        a = hf.pixel_sha256(encode(image, compress_level=0))
        b = hf.pixel_sha256(encode(image, compress_level=9, optimize=True))
        self.assertEqual(a, b)
        self.assertRegex(a, r"^[0-9a-f]{64}$")
        image.putpixel((4, 4), (251, 0, 0))
        self.assertNotEqual(a, hf.pixel_sha256(encode(image)))

    def test_size_is_part_of_the_hash(self):
        tall = hf.pixel_sha256(encode(Image.new("RGB", (4, 8), (1, 1, 1))))
        wide = hf.pixel_sha256(encode(Image.new("RGB", (8, 4), (1, 1, 1))))
        self.assertNotEqual(tall, wide)


class SplitTests(unittest.TestCase):
    def test_family_digest_uses_family_key_or_native_id(self):
        self.assertEqual(hf.family_digest(candidate(family_key="fam")), hf.family_digest(candidate(native_id="other", family_key="fam")))
        self.assertNotEqual(hf.family_digest(candidate(family_key="fam")), hf.family_digest(candidate(family_key="fam2")))
        self.assertEqual(hf.family_digest(candidate()), hf.family_digest(candidate(family_key="row-7")))

    def test_exactly_one_stable_split(self):
        seen = set()
        for i in range(600):
            c = candidate(native_id=f"n{i}")
            split = hf.split_for(c)
            self.assertEqual(split, hf.split_for(c))
            self.assertIn(split, ("train", "dev", "sealed", "style_sealed"))
            digest = hf.family_digest(c)
            hits = [s for s in ("train", "dev", "sealed", "style_sealed")
                    if hf.matches_split(digest, s, "shift" if s == "style_sealed" else "default")]
            self.assertEqual(hits, [split])
            seen.add(split)
        self.assertIn("train", seen)
        self.assertIn("dev", seen)


class McqTests(unittest.TestCase):
    def test_mcq_from_answer_ok_and_stripped(self):
        self.assertEqual(hf.mcq_from_answer(" Q? ", [" a ", "b", "c"], 2), ("Q?", ("a", "b", "c"), "C"))

    def test_mcq_from_answer_rejects_bad_input(self):
        bad = [
            ([" a", "a "], 0),                    # duplicate after strip
            (["a", "  "], 0),                     # empty after strip
            (["a"], 0),                           # too few
            (list("abcdefghijk"), 0),             # too many
            (["a", "b"], 2),                      # out of range
            (["a", "b"], -1),
        ]
        for options, index in bad:
            with self.subTest(options=options, index=index), self.assertRaises(ValueError):
                hf.mcq_from_answer("Q?", options, index)

    def test_program_mcq_deterministic_and_gold_once(self):
        pool = ["cat", "dog", "bird", "fish", "frog", "ant"]
        first = hf.program_mcq("cat", pool, key="k1")
        self.assertEqual(first, hf.program_mcq("cat", pool, key="k1"))
        positions = set()
        for i in range(40):
            choices, letter = hf.program_mcq("cat", pool, key=f"k{i}")
            self.assertEqual(len(choices), 4)
            self.assertEqual(choices.count("cat"), 1)
            self.assertEqual(len(set(choices)), 4)
            self.assertEqual(choices["ABCD".index(letter)], "cat")
            self.assertTrue(set(choices) <= set(pool))
            positions.add(letter)
        self.assertGreater(len(positions), 1)

    def test_program_mcq_small_pool_and_empty_pool(self):
        choices, letter = hf.program_mcq("cat", ["cat", "dog"], key="k")
        self.assertEqual(sorted(choices), ["cat", "dog"])
        self.assertEqual(choices["AB".index(letter)], "cat")
        for pool in ([], ["cat"]):
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                hf.program_mcq("cat", pool, key="k")


class TextTests(unittest.TestCase):
    def test_strip_instruction_suffix(self):
        for text, expected in (
            ("What is 2+2?\nShort answer required.", "What is 2+2?"),
            ("Pick one.\nAnswer with the letter.", "Pick one."),
            ("Name it.\nBe succinct.", "Name it."),
            ("Name it.\nBe succinct.  \n", "Name it."),
            ("How much?\nAnswer scale should be: percent.", "How much?\nAnswer scale should be: percent."),
            ("No instruction here", "No instruction here"),
        ):
            with self.subTest(text=text):
                self.assertEqual(hf.strip_instruction_suffix(text), expected)

    def test_strip_removes_only_one_line(self):
        self.assertEqual(hf.strip_instruction_suffix("Q?\nBe brief.\nShort answer required."), "Q?\nBe brief.")

    def test_parse_letter_choices_ai2d(self):
        user = ("Question: What do respiration and combustion give out\nChoices:\nA. Oxygen\nB. Carbon dioxide\n"
                "C. Nitrogen\nD. Heat\nAnswer with the letter.")
        self.assertEqual(hf.parse_letter_choices(user),
                         ("What do respiration and combustion give out", ("Oxygen", "Carbon dioxide", "Nitrogen", "Heat")))

    def test_parse_letter_choices_non_matching(self):
        for user in (
            "What is the capital?\nShort answer required.",
            "Question: q\nA. x\nB. y\nAnswer with the letter.",                       # no Choices header
            "Question: q\nChoices:\nA. x\nC. y\nAnswer with the letter.",             # letters not consecutive
            "Question: q\nChoices:\nA. x\nAnswer with the letter.",                   # one choice
            "Question: q\nChoices:\nA. x\nB. y",                                      # no instruction line
            "Question: \nChoices:\nA. x\nB. y\nAnswer with the letter.",              # empty question
        ):
            with self.subTest(user=user):
                self.assertIsNone(hf.parse_letter_choices(user))


class ToProblemTests(unittest.TestCase):
    def test_records_validate_and_carry_human_lineage(self):
        c = candidate(accept={"alias": ["leaf"]}, images=(encode(Image.new("RGBA", (16, 12), (0, 0, 0, 0))),
                                                          encode(Image.new("RGB", (8, 8), (1, 2, 3)))))
        problem = hf.to_problem(c, SPEC, index=7)
        actor, verifier, lineage = problem.actor, problem.verifier, problem.lineage
        for record in (actor, verifier, lineage):
            to_json(record)
        digest = hf.family_digest(c)
        split = hf.split_for(c)
        pid = f"hf-test:test_family:{split}:000007:{digest}"
        self.assertEqual((actor.problem_id, verifier.problem_id, lineage.problem_id), (pid, pid, pid))
        self.assertRegex(pid, r"^hf-test:test_family:(train|dev|sealed|style_sealed):000007:[0-9a-f]{16}$")
        self.assertEqual(actor.image_paths, tuple(f"hf-test/{split}/000007-{digest}-{i}.png" for i in range(2)))
        self.assertEqual(actor.image_sha256, tuple(hashlib.sha256(x).hexdigest() for x in problem.image_bytes))
        for raw in problem.image_bytes:
            png_chunks(raw)
        self.assertEqual((actor.source, actor.family, actor.mode, actor.answer_format), ("hf-test", "test_family", "plain", "mcq"))
        self.assertEqual((actor.question, actor.choices), (c.question, c.choices))
        self.assertEqual((verifier.gold, verifier.accept, verifier.choices, verifier.answer_type),
                         ("A", {"alias": ["leaf"]}, c.choices, "mcq"))
        self.assertEqual((verifier.slots, verifier.slot_types, verifier.required_slots), ({}, {}, ()))
        self.assertEqual((lineage.source, lineage.source_revision, lineage.license, lineage.label_provenance),
                         ("hf-test", "a" * 40, "cc-by-4.0", "HUMAN"))
        self.assertEqual(lineage.image_provenance, f"hf:org/dataset@{'a' * 40}")
        self.assertEqual((lineage.upstream_origin, lineage.generator_version, lineage.generator_seed),
                         ("authored annotators", None, None))
        self.assertEqual((lineage.template_id, lineage.structure_hash, lineage.split, lineage.parents),
                         ("hf-test-row-v1", digest, split, ()))
        self.assertFalse(lineage.benchmark_derived or lineage.model_touched)

    def test_non_mcq_candidate(self):
        c = candidate(answer_type="short_text", choices=(), gold="leaf", native_id="row-9")
        problem = hf.to_problem(c, SPEC, index=0, mode="obs")
        self.assertEqual((problem.actor.mode, problem.actor.choices, problem.verifier.gold), ("obs", (), "leaf"))

    def test_rejections(self):
        image = encode(Image.new("RGB", (4, 4), (1, 1, 1)))
        for name, c in (
            ("source", candidate(source="other")),
            ("subject", candidate(subject="Underwater_Basketry")),
            ("gold out of range", candidate(gold="E")),
            ("gold not a letter", candidate(gold="AB")),
            ("gold empty", candidate(gold="")),
            ("too few choices", candidate(choices=("Leaf",), gold="A")),
            ("too many choices", candidate(choices=tuple("abcdefghijk"), gold="A")),
            ("no images", candidate(images=())),
            ("three images", candidate(images=(image, image, image))),
            ("choices on non-mcq", candidate(answer_type="short_text", gold="leaf")),
        ):
            with self.subTest(name), self.assertRaises(ValueError):
                hf.to_problem(c, SPEC, index=1)

    def test_candidate_order_is_label_blind_and_deterministic(self):
        a, b = candidate(gold="A"), candidate(gold="B")
        self.assertEqual(hf.candidate_order(a, 3), hf.candidate_order(b, 3))
        self.assertNotEqual(hf.candidate_order(a, 3), hf.candidate_order(a, 4))


if __name__ == "__main__":
    unittest.main()
