"""Authored parquet fixtures only; no real dataset rows, upstream images, or benchmark data."""
import hashlib
import io
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fine_tuning.notebook_rl.adapters import hf_common as hf
from fine_tuning.notebook_rl.adapters import hf_science as m

IMAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
SCIENCEQA_SCHEMA = pa.schema([
    ("image", IMAGE), ("question", pa.string()), ("choices", pa.list_(pa.string())),
    ("answer", pa.int8()), ("hint", pa.string()), ("subject", pa.string()), ("topic", pa.string()),
    ("grade", pa.string()), ("category", pa.string()), ("lecture", pa.string()), ("solution", pa.string()),
])
TQA_SCHEMA = pa.schema([
    ("image", IMAGE), ("question", pa.string()), ("options", pa.list_(pa.string())), ("answer", pa.int64()),
])
GEOMETRY_SCHEMA = pa.schema([("images", pa.list_(IMAGE)), ("problem", pa.string()), ("answer", pa.string())])


def png(color):
    out = io.BytesIO()
    Image.new("RGB", (6, 5), color).save(out, format="PNG")
    return out.getvalue()


RED, GREEN, BLUE = png((255, 0, 0)), png((0, 255, 0)), png((0, 0, 255))


def image(raw):
    return {"bytes": raw, "path": None}


def write(path, schema, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def sqa_row(**fields):
    row = {"image": image(RED), "question": "Which is a mammal?", "choices": ["Fern", "Dog"], "answer": 1,
           "hint": "", "subject": "natural science", "topic": "biology", "grade": "grade3",
           "category": "science", "lecture": "LECTURE_SECRET", "solution": "SOLUTION_SECRET"}
    row.update(fields)
    return row


def tqa_row(**fields):
    row = {"image": image(RED), "question": "Which organ pumps blood?", "options": ["Lung", "Heart"], "answer": 1}
    row.update(fields)
    return row


def geo_row(**fields):
    row = {"images": [image(RED)], "problem": "<image>Find x.", "answer": "3"}
    row.update(fields)
    return row


class RepoTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


class ScienceQATests(RepoTests):
    def sqa(self, rows):
        repo = self.root / "derek-thomas__ScienceQA"
        write(repo / "data" / "train-00000-of-00001-1028f23e353fbe3e.parquet", SCIENCEQA_SCHEMA, rows)
        return list(m.iter_scienceqa(repo))

    def test_valid_row_uses_hint_and_hides_upstream_text(self):
        (c,) = self.sqa([sqa_row(hint="Look at the picture.", question="Which is a mammal?")])
        self.assertEqual(c.question, "Context: Look at the picture.\nWhich is a mammal?")
        self.assertEqual((c.choices, c.answer_type, c.gold, c.subject, c.native_id),
                         (("Fern", "Dog"), "mcq", "B", "Biology", "train:0"))
        self.assertEqual((c.accept, c.images, c.family_key), ({}, (RED,), hashlib.sha256(RED).hexdigest()))
        self.assertNotIn("SECRET", repr(c))

    def test_invalid_rows_skipped_and_blank_hint_dropped(self):
        got = self.sqa([
            sqa_row(image=None),                                     # no image
            sqa_row(topic="units-and-measurement"),                  # unmapped topic
            sqa_row(answer=5),                                       # answer index out of range
            sqa_row(image=image(BLUE), topic="civics", hint=None, question="Who leads?",
                    choices=["Mayor", "Judge"], answer=0),           # kept: blank hint, Sociology
            sqa_row(choices=None),                                   # no choices
            sqa_row(image=image(None)),                              # image struct without bytes
            sqa_row(image=image(GREEN), question=None),              # no question
        ])
        self.assertEqual([(c.native_id, c.subject, c.question) for c in got],
                         [("train:3", "Sociology", "Who leads?")])

    def test_topic_to_subject_mapping(self):
        expected = {
            "biology": "Biology", "chemistry": "Chemistry", "physics": "Physics",
            "geography": "Geography", "earth-science": "Geography", "us-history": "History",
            "world-history": "History", "economics": "Economics", "civics": "Sociology",
            "global-studies": "Sociology", "writing-strategies": "Literature",
            "reading-comprehension": "Literature", "literacy-in-science": "Literature",
            "vocabulary": "Literature", "word-study": "Literature",
        }
        rows = [sqa_row(image=image(png((i, 0, 0))), topic=topic, question=f"Q {topic}")
                for i, topic in enumerate(expected)]
        rows.append(sqa_row(image=image(GREEN), topic="units-and-measurement", question="Q unmapped"))
        got = {c.question: c.subject for c in self.sqa(rows)}
        self.assertEqual(got, {f"Q {topic}": subject for topic, subject in expected.items()})


SQA_RAW = Path("/home/jang2/mmdl-data/training/raw/hf-pool-v1/derek-thomas__ScienceQA")


class ScienceQAFilterTests(RepoTests):
    def run_filter(self, rows):
        repo = self.root / "derek-thomas__ScienceQA"
        write(repo / "data" / "train-00000-of-00001-1028f23e353fbe3e.parquet", SCIENCEQA_SCHEMA, rows)
        dropped = Counter()
        return [c.native_id for c in m.iter_scienceqa(repo, dropped)], dropped

    def test_filter_version_is_pinned(self):
        self.assertEqual(m.FILTER_VERSION, "scienceqa-filter-v1")

    def test_gold_text_as_whole_phrase_in_hint_is_dropped(self):
        rows = [
            sqa_row(hint="A DOG barks!"),                                              # 0: whole word, any case/punct
            sqa_row(hint="A dogfish swims."),                                          # 1: substring only: kept
            sqa_row(hint="Pick the ordinary one.", choices=["Fern", "Golden  Retriever"], answer=1),  # 2: kept
            sqa_row(hint="It is a golden retriever, a friendly breed.",
                    choices=["Fern", "Golden  Retriever"], answer=1),                  # 3: multi-word phrase
            sqa_row(hint="The fern is green."),                                        # 4: other choice only: kept
            sqa_row(hint=None),                                                        # 5: no hint: kept
        ]
        kept, dropped = self.run_filter(rows)
        self.assertEqual(kept, ["train:1", "train:2", "train:4", "train:5"])
        self.assertEqual(dict(dropped), {"valid": 6, "gold_in_hint": 2, "any": 2})

    def test_scientific_name_categories_are_dropped(self):
        rows = [sqa_row(category=c) for c in
                ("Scientific names", "Classification and scientific names", "Classification", "scientific names", None)]
        kept, dropped = self.run_filter(rows)
        self.assertEqual(kept, ["train:2", "train:3", "train:4"])   # exact, case-sensitive category values
        self.assertEqual(dict(dropped), {"valid": 5, "scientific_names": 2, "any": 2})

    def test_hint_longer_than_200_chars_is_dropped(self):
        kept, dropped = self.run_filter([sqa_row(hint="x" * 200), sqa_row(hint="x" * 201)])
        self.assertEqual(kept, ["train:0"])
        self.assertEqual(dict(dropped), {"valid": 2, "long_hint": 1, "any": 1})

    def test_rules_overlap_and_count_once_in_any(self):
        kept, dropped = self.run_filter([sqa_row(hint="dog " * 60, category="Scientific names")])
        self.assertEqual(kept, [])
        self.assertEqual(dict(dropped), {"valid": 1, "gold_in_hint": 1, "scientific_names": 1,
                                         "long_hint": 1, "any": 1})

    def test_invalid_rows_are_not_counted_as_valid(self):
        _, dropped = self.run_filter([sqa_row(image=None), sqa_row(answer=5), sqa_row(topic="units-and-measurement")])
        self.assertEqual(dict(dropped), {})

    @unittest.skipUnless(SQA_RAW.exists(), "local ScienceQA parquet not present")
    def test_real_train_parquet_counts(self):
        dropped = Counter()
        kept = sum(1 for _ in m.iter_scienceqa(SQA_RAW, dropped))
        print(f"\nScienceQA {m.FILTER_VERSION}: {dict(dropped)} kept={kept}")
        self.assertEqual(dict(dropped), {"valid": 5825, "gold_in_hint": 424, "scientific_names": 293,
                                         "long_hint": 1039, "any": 1430})
        self.assertEqual(kept, 5825 - 1430)


class TQATests(RepoTests):
    def test_reads_train_files_in_order_and_maps_subjects(self):
        repo = self.root / "yyyyifan__TQA" / "data"
        write(repo / "train-00000-of-00002.parquet", TQA_SCHEMA, [
            tqa_row(question="What is the phase of the Moon?", options=["New", "Full"], answer=1),
            tqa_row(image=image(None)),                                                  # no image: skip
            tqa_row(),                                                                   # Biology
            tqa_row(question="The sun gives off energy.", options=["Heat", "Sound"], answer=0),
            tqa_row(options=["Heat", " Heat"], answer=0),                                # duplicate: skip
        ])
        write(repo / "train-00001-of-00002.parquet", TQA_SCHEMA, [
            tqa_row(image=image(GREEN), question="What does a lever change?", options=["Force", "Mass"], answer=0),
            tqa_row(image=image(GREEN), question="What is a molecule made of?", options=["Atoms", "Cells"], answer=0),
            tqa_row(question="Which formation is shown?", options=["Ridge", "Cliff"], answer=0),  # no 'ion' hit
        ])
        write(repo / "validation-00000-of-00001.parquet", TQA_SCHEMA,
              [tqa_row(question="What is the Moon?", options=["A", "B"], answer=0)])
        got = list(m.iter_tqa(self.root / "yyyyifan__TQA"))
        self.assertEqual([(c.native_id, c.subject, c.gold) for c in got], [
            ("train-00000-of-00002:0", "Geography", "B"),
            ("train-00000-of-00002:2", "Biology", "B"),
            ("train-00000-of-00002:3", "Geography", "A"),
            ("train-00001-of-00002:0", "Physics", "A"),
            ("train-00001-of-00002:1", "Chemistry", "A"),
            ("train-00001-of-00002:2", "Biology", "A"),
        ])
        self.assertEqual((got[0].choices, got[0].answer_type, got[0].images), (("New", "Full"), "mcq", (RED,)))

    def test_geography_keywords_match_at_word_start(self):
        words = ("glacier", "waterfall", "river", "ocean", "weather", "cloud", "wind", "mountain",
                 "delta", "tsunami", "earthquake", "climate", "desert")
        repo = self.root / "yyyyifan__TQA" / "data"
        write(repo / "train-00000-of-00001.parquet", TQA_SCHEMA,
              [tqa_row(question=f"What is a {word} shaped like?", options=["Flat", "Round"], answer=0)
               for word in words])
        got = list(m.iter_tqa(self.root / "yyyyifan__TQA"))
        self.assertEqual([c.subject for c in got], ["Geography"] * len(words))


class GeometryTests(RepoTests):
    def test_numeric_answers_only_with_round_to(self):
        repo = self.root / "hiyouga__geometry3k" / "data"
        write(repo / "train-00000-of-00001.parquet", GEOMETRY_SCHEMA, [
            geo_row(problem="<image>Find the length of AB.", answer="3"),
            geo_row(problem="<image>Find x.", answer="11.7"),
            geo_row(answer="2 \\sqrt { 221 }"),                  # LaTeX: skip
            geo_row(images=[image(RED), image(BLUE)]),           # two images: skip
            geo_row(images=[]),                                  # no image: skip
            geo_row(images=[image(None)]),                       # null bytes: skip
            geo_row(problem="<image>Find the area.", answer="-4.25"),
            geo_row(problem="  Area of the shaded region?  ", answer=" 120 "),
        ])
        got = list(m.iter_geometry3k(self.root / "hiyouga__geometry3k"))
        self.assertEqual([(c.native_id, c.question, c.gold, c.accept, c.answer_type, c.subject, c.choices)
                          for c in got], [
            ("train:0", "Find the length of AB.", "3", {"abs_tol": "0.000001"}, "decimal", "Math", ()),
            ("train:1", "Find x.", "11.7", {"round_to": 1}, "decimal", "Math", ()),
            ("train:6", "Find the area.", "-4.25", {"round_to": 2}, "decimal", "Math", ()),
            ("train:7", "Area of the shaded region?", "120", {"abs_tol": "0.000001"}, "decimal", "Math", ()),
        ])

    def test_missing_train_files_raise(self):
        with self.assertRaises(FileNotFoundError):
            list(m.iter_geometry3k(self.root / "hiyouga__geometry3k"))


class SpecAndProblemTests(RepoTests):
    def test_pinned_specs_and_iterator_keys(self):
        self.assertEqual(set(m.SPECS), set(m.ITERATORS))
        self.assertEqual(m.SPECS["hf-scienceqa"], hf.SourceSpec(
            "hf-scienceqa", "derek-thomas/ScienceQA", "f18b0a70359ebfb41f658fd564208d0355b013f4",
            "CC-BY-NC-SA-4.0", "HUMAN", "IXL curriculum items (ScienceQA)", "science_diagram"))
        self.assertEqual(m.SPECS["hf-tqa"], hf.SourceSpec(
            "hf-tqa", "yyyyifan/TQA", "1e9a4d2a5900cb3c97d1e8d4d78ce927584895bb",
            "CC-BY-NC-3.0", "HUMAN", "CK-12 textbook diagram questions (TQA)", "science_diagram"))
        self.assertEqual(m.SPECS["hf-geometry3k"], hf.SourceSpec(
            "hf-geometry3k", "hiyouga/geometry3k", "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7",
            "MIT", "HUMAN", "Geometry3K annotators via InterGPS", "geometry"))

    def test_candidates_build_verifiable_problems(self):
        sqa_dir = self.root / "derek-thomas__ScienceQA"
        write(sqa_dir / "data" / "train-00000-of-00001-1028f23e353fbe3e.parquet", SCIENCEQA_SCHEMA,
              [sqa_row(topic="physics", question="What is a magnet?")])
        geo_dir = self.root / "hiyouga__geometry3k"
        write(geo_dir / "data" / "train-00000-of-00001.parquet", GEOMETRY_SCHEMA, [geo_row(answer="11.7")])
        (science,) = m.iter_scienceqa(sqa_dir)
        (geometry,) = m.iter_geometry3k(geo_dir)
        for candidate, source in ((science, "hf-scienceqa"), (geometry, "hf-geometry3k")):
            with self.subTest(source=source):
                problem = hf.to_problem(candidate, m.SPECS[source], index=0)
                self.assertEqual((problem.verifier.gold, problem.verifier.accept, problem.verifier.answer_type),
                                 (candidate.gold, candidate.accept, candidate.answer_type))


if __name__ == "__main__":
    unittest.main()
