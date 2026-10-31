"""Authored fixtures only: tiny synthetic parquet shards in a temp directory. No network, no GPU."""
import io
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fine_tuning.notebook_rl.adapters import hf_art as art
from fine_tuning.notebook_rl.adapters import hf_common as hf

IMAGE_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def png(color):
    out = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(out, format="PNG")
    return out.getvalue()


def gold_text(c):
    return c.choices["ABCDEFGHIJ".index(c.gold)]


def assert_converts(test, candidates):
    for c in candidates:
        with test.subTest(native_id=c.native_id):
            problem = hf.to_problem(c, art.SPECS[c.source], index=0)
            test.assertEqual((problem.verifier.gold, problem.lineage.label_provenance), (c.gold, "HUMAN"))


# The Met: (Object Name, Department, jpg bytes or None). Object ID is the 1-based row number.
def met_rows():
    rows = [("Vase", "Greek and Roman Art", png((i, 0, 0))) for i in range(14)]
    rows.append((" Vase ", "Greek and Roman Art", png((0, 1, 0))))   # stripped: 15 rows in total
    rows += [("Bowl", "Asian Art", png((0, 0, i))) for i in range(15)]
    rows += [("Dress", "Costume Institute", png((9, i, 0))) for i in range(15)]
    rows += [("Hat", "Costume Institute", png((5, 5, i))) for i in range(14)]   # below the 15 cut
    rows += [("Blank", "Asian Art", None) for _ in range(15)]                    # no image bytes
    return rows


def write_met(root):
    directory = root / "default" / "train"
    directory.mkdir(parents=True)
    rows = met_rows()
    # Two shards: the 'Bowl' name count must span both files.
    for name, start, stop in (("00000.parquet", 0, 20), ("00001.parquet", 20, len(rows))):
        part = rows[start:stop]
        table = pa.table({
            "Object ID": pa.array(list(range(start + 1, stop + 1)), type=pa.int64()),
            "Department": pa.array([r[1] for r in part], type=pa.string()),
            "Object Name": pa.array([r[0] for r in part], type=pa.string()),
            "jpg": pa.array([r[2] for r in part], type=pa.binary()),
        })
        pq.write_table(table, directory / name)


class MetTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_met(self.root)
        self.rows = met_rows()
        self.candidates = list(art.iter_met(self.root))
        self.by_id = {c.native_id: c for c in self.candidates}

    def test_pool_spans_shards_and_drops_small_and_imageless_names(self):
        self.assertEqual(len(self.candidates), 45)  # Vase 15 + Bowl 15 + Dress 15; Hat and Blank excluded
        for c in self.candidates:
            self.assertEqual(set(c.choices), {"Vase", "Bowl", "Dress"})

    def test_subject_and_question_follow_department(self):
        self.assertEqual(self.by_id["met:1"].subject, "Art")
        self.assertEqual(self.by_id["met:31"].subject, "Design")  # Costume Institute
        self.assertEqual(
            self.by_id["met:1"].question,
            "This object is from The Metropolitan Museum of Art (Greek and Roman Art). What kind of object is it?",
        )

    def test_gold_is_stripped_object_name_and_family_is_object_id(self):
        stripped = self.by_id["met:15"]
        self.assertEqual(gold_text(stripped), "Vase")
        self.assertEqual(stripped.family_key, "met:15")
        self.assertEqual(stripped.images, (self.rows[14][2],))

    def test_converts_to_verified_problems(self):
        assert_converts(self, self.candidates)


COOPER_ROWS = (
    # (record_id, medium, has image bytes)
    ("chndm_1926-43-5-553", "Etching on laid paper", True),
    ("chndm_1930-1-2-a", "Graphite on paper", True),
    ("chndm_1930-1-2-b", "Graphite on paper", True),
    ("chndm_1930-1-3-B", "Tin-glazed earthenware, underglaze", True),
    ("chndm_1950-2-1", "Watercolor over graphite", True),
    ("chndm_1896-2-32-a_f", "Etching and engraving", True),
    ("chndm_1940-9-1", "Unknown medium text", True),        # no technique match: skipped
    ("chndm_1940-9-2", "Machine-printed paper", False),     # no image bytes: skipped
)


def write_cooper(root):
    directory = root / "data"
    directory.mkdir(parents=True)
    table = pa.table({
        "image": pa.array(
            [{"bytes": png((i * 30, i, i)), "path": None} if has else None
             for i, (_, _, has) in enumerate(COOPER_ROWS)],
            type=IMAGE_STRUCT,
        ),
        "medium": pa.array([r[1] for r in COOPER_ROWS], type=pa.string()),
        "record_id": pa.array([r[0] for r in COOPER_ROWS], type=pa.string()),
    })
    pq.write_table(table, directory / "train-00000-of-00001.parquet")


class CooperHewittTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_cooper(self.root)
        self.candidates = list(art.iter_cooperhewitt(self.root))
        self.by_id = {c.native_id: c for c in self.candidates}

    def test_rows_without_technique_or_image_are_skipped(self):
        self.assertEqual(
            [c.native_id for c in self.candidates],
            ["chndm_1926-43-5-553", "chndm_1930-1-2-a", "chndm_1930-1-2-b", "chndm_1930-1-3-B", "chndm_1950-2-1",
             "chndm_1896-2-32-a_f"],
        )

    def test_gold_is_first_priority_technique(self):
        self.assertEqual(gold_text(self.by_id["chndm_1926-43-5-553"]), "etching")
        self.assertEqual(gold_text(self.by_id["chndm_1930-1-3-B"]), "tin-glazed earthenware")
        self.assertEqual(gold_text(self.by_id["chndm_1950-2-1"]), "watercolor")  # watercolor rule precedes graphite

    def test_subject_from_technique(self):
        self.assertEqual(self.by_id["chndm_1926-43-5-553"].subject, "Art")
        self.assertEqual(self.by_id["chndm_1930-1-2-a"].subject, "Art")
        self.assertEqual(self.by_id["chndm_1930-1-3-B"].subject, "Design")

    def test_family_key_drops_trailing_part_letter(self):
        self.assertEqual(self.by_id["chndm_1930-1-2-a"].family_key, "chndm_1930-1-2")
        self.assertEqual(self.by_id["chndm_1930-1-2-b"].family_key, "chndm_1930-1-2")
        self.assertEqual(self.by_id["chndm_1930-1-3-B"].family_key, "chndm_1930-1-3")
        self.assertEqual(self.by_id["chndm_1926-43-5-553"].family_key, "chndm_1926-43-5-553")
        self.assertEqual(self.by_id["chndm_1896-2-32-a_f"].family_key, "chndm_1896-2-32")  # part range
        self.assertEqual(art._cooper_family_key("chndm_1938-88-7-a-2"), "chndm_1938-88-7")  # sub-part

    def test_co_named_technique_is_never_a_distractor(self):
        for key in ("chndm_1950-2-1", "chndm_1896-2-32-a_f"):
            c = self.by_id[key]
            named = art._techniques(dict((r[0], r[1]) for r in COOPER_ROWS)[key])
            self.assertEqual(gold_text(c), named[0])
            self.assertFalse(set(named[1:]) & set(c.choices))

    def test_pool_is_all_distinct_technique_names(self):
        pool = {name for _, name in art.TECHNIQUE_RULES}
        for c in self.candidates:
            self.assertEqual(len(c.choices), 4)
            self.assertTrue(set(c.choices) <= pool)

    def test_converts_to_verified_problems(self):
        assert_converts(self, self.candidates)


CRELLO_ROWS = (
    # (id, category class id, has preview bytes)
    ("c-1", 0, True),     # holidaysCelebration
    ("c-2", 22, True),    # 'all' catch-all: skipped
    ("c-3", 23, True),    # class id outside the README list: skipped
    ("c-4", 11, False),   # no preview bytes: skipped
    ("c-5", 19, True),    # industry
    ("c-6", 19, True),    # same title as c-5 (another format of one design)
)
CRELLO_TITLES = {"c-5": "Factory Day", "c-6": " factory day "}


def write_crello(root):
    directory = root / "data"
    directory.mkdir(parents=True)
    table = pa.table({
        "id": pa.array([r[0] for r in CRELLO_ROWS], type=pa.string()),
        "category": pa.array([r[1] for r in CRELLO_ROWS], type=pa.int64()),
        "title": pa.array([CRELLO_TITLES.get(r[0], "None") for r in CRELLO_ROWS], type=pa.string()),
        "preview": pa.array(
            [{"bytes": png((i * 40, 0, 0)), "path": None} if has else None
             for i, (_, _, has) in enumerate(CRELLO_ROWS)],
            type=IMAGE_STRUCT,
        ),
    })
    pq.write_table(table, directory / "train-00000-of-00001.parquet")


class CrelloTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_crello(self.root)
        self.candidates = list(art.iter_crello(self.root))
        self.by_id = {c.native_id: c for c in self.candidates}

    def test_invalid_rows_are_skipped(self):
        self.assertEqual([c.native_id for c in self.candidates], ["c-1", "c-5", "c-6"])

    def test_gold_is_camel_case_category_text(self):
        self.assertEqual(gold_text(self.by_id["c-1"]), "holidays celebration")
        self.assertEqual(art._crello_text("socialActivityCharity"), "social activity charity")

    def test_question_subject_family(self):
        c = self.by_id["c-5"]
        self.assertEqual(c.question, "This is a graphic design template. What is it designed for?")
        self.assertEqual((c.subject, c.family_key), ("Design", "crello:title:factory day"))
        self.assertEqual(self.by_id["c-6"].family_key, c.family_key)
        self.assertEqual(self.by_id["c-1"].family_key, "crello:c-1")  # 'None' title: per-id family

    def test_pool_is_all_category_names(self):
        pool = {art._crello_text(n) for n in art.CRELLO_CATEGORY_NAMES} - {"all"}
        self.assertEqual(len(pool), 22)
        for c in self.candidates:
            self.assertEqual(len(c.choices), 4)
            self.assertTrue(set(c.choices) <= pool)

    def test_converts_to_verified_problems(self):
        assert_converts(self, self.candidates)


class ShardAndRegistryTests(unittest.TestCase):
    def test_missing_shards_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                list(art.iter_cooperhewitt(Path(tmp)))

    def test_specs_pin_full_revisions_and_iterators_match(self):
        self.assertEqual(set(art.SPECS), {"hf-met", "hf-cooperhewitt", "hf-crello"})
        self.assertEqual(set(art.ITERATORS), set(art.SPECS))
        for spec in art.SPECS.values():
            self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
            self.assertRegex(spec.family, r"^[a-z0-9_]+$")
            self.assertEqual(spec.label_provenance, "HUMAN")


if __name__ == "__main__":
    unittest.main()
