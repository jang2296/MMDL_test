"""Authored fixtures only: tiny synthetic zip, split list, leaf map and parquet in a temp directory."""
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fine_tuning.notebook_rl.adapters import hf_agri as agri
from fine_tuning.notebook_rl.adapters import hf_common as hf

PV_MEMBERS = (
    "raw/color/Apple___Apple_scab/s1___Mary_HL 1.JPG",
    "raw/color/Apple___healthy/s2___Mary_HL 2.JPG",
    "raw/color/Apple/malformed.JPG",                        # class dir without '___': skipped
    "raw/color/Corn_(maize)___Common_rust_/s3___Rutg._HL 3.JPG",
    "raw/color/Tomato___healthy/s4___Tom 4.JPG",            # crop with only healthy: pool < 2, skipped
    "raw/color/Apple___Black_rot/s5___Mary_HL 5.JPG",
    "raw/color/Apple___Apple_scab/s6___Unknown 6.JPG",      # tag not in leaf map: class dir + tag is the family
    "raw/color/Apple___Black_rot/s7___Gone 7.JPG",          # listed but absent from zip: skipped
)
PV_MISSING_FROM_ZIP = {"raw/color/Apple___Black_rot/s7___Gone 7.JPG"}
PV_LEAF_MAP = {
    "mary_hl 1": ["Apple___scab:::1.0"],
    "mary_hl 2": ["Soybean___healthy:::9.0", "Apple___healthy:::2.0"],   # cross-class tag collision
    "mary_hl 5": ["Apple___Black_rot:::5.0", "Apple___Black_rot:::6.0"],  # unresolvable: sorted join
    "rutg._hl 3": ["Corn___rust:::3.0"],
}
DW_ROWS = (
    ((10, 0, 0), "Lantana"),
    ((0, 10, 0), "Negative"),
    (None, "Parkinsonia"),                                  # no image bytes: skipped, species still in pool
    ((0, 0, 10), "Lantana"),
    ((10, 10, 0), "Siam weed"),
)


def png(color):
    out = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(out, format="PNG")
    return out.getvalue()


def write_plantvillage(root):
    (root / "splits").mkdir()
    (root / "leaf_grouping").mkdir()
    (root / "splits" / "color_train.txt").write_text("\n".join(PV_MEMBERS) + "\n", encoding="utf-8")
    (root / "leaf_grouping" / "leaf-map.json").write_text(json.dumps(PV_LEAF_MAP), encoding="utf-8")
    with zipfile.ZipFile(root / "data.zip", "w") as archive:
        for i, member in enumerate(PV_MEMBERS):
            if member not in PV_MISSING_FROM_ZIP:
                archive.writestr(member, png((i * 20, 20, 20)))


def write_deepweeds(root):
    (root / "data").mkdir()
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    images = [{"bytes": png(color), "path": None} if color is not None else None for color, _ in DW_ROWS]
    table = pa.table({
        "image": pa.array(images, type=image_type),
        "species": pa.array([s for _, s in DW_ROWS], type=pa.string()),
        "label": pa.array(list(range(len(DW_ROWS))), type=pa.int64()),
    })
    pq.write_table(table, root / "data" / "train-00000-of-00001.parquet")


class PlantVillageTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_plantvillage(self.root)
        self.candidates = list(agri.iter_plantvillage(self.root))
        self.by_member = {c.native_id: c for c in self.candidates}

    def test_train_members_in_order_minus_skipped(self):
        self.assertEqual([c.native_id for c in self.candidates], [PV_MEMBERS[i] for i in (0, 1, 3, 5, 6)])

    def test_gold_is_the_class_condition(self):
        expected = {
            PV_MEMBERS[0]: "Apple scab", PV_MEMBERS[1]: "healthy (no disease)", PV_MEMBERS[3]: "Common rust",
            PV_MEMBERS[5]: "Black rot", PV_MEMBERS[6]: "Apple scab",
        }
        for member, text in expected.items():
            with self.subTest(member=member):
                c = self.by_member[member]
                self.assertEqual(c.choices["ABCDEFGHIJ".index(c.gold)], text)
                self.assertEqual((c.source, c.subject, c.answer_type), ("hf-plantvillage", "Agriculture", "mcq"))

    def test_pool_is_same_crop_plus_healthy(self):
        self.assertEqual(set(self.by_member[PV_MEMBERS[0]].choices), {"Apple scab", "Black rot", "healthy (no disease)"})
        self.assertEqual(set(self.by_member[PV_MEMBERS[3]].choices), {"Common rust", "healthy (no disease)"})

    def test_question_and_image_bytes(self):
        c = self.by_member[PV_MEMBERS[3]]
        self.assertEqual(c.question, "This photo shows a Corn (maize) leaf. Which condition does the leaf show?")
        self.assertEqual(self.by_member[PV_MEMBERS[0]].question,
                         "This photo shows an Apple leaf. Which condition does the leaf show?")
        with zipfile.ZipFile(self.root / "data.zip") as archive:
            self.assertEqual(c.images, (archive.read(PV_MEMBERS[3]),))

    def test_family_key_from_leaf_map_or_class_tag_fallback(self):
        self.assertEqual(self.by_member[PV_MEMBERS[0]].family_key, "Apple___scab:::1.0")
        self.assertEqual(self.by_member[PV_MEMBERS[1]].family_key, "Apple___healthy:::2.0")
        self.assertEqual(self.by_member[PV_MEMBERS[5]].family_key, "Apple___Black_rot:::5.0|Apple___Black_rot:::6.0")
        self.assertEqual(self.by_member[PV_MEMBERS[3]].family_key, "Corn___rust:::3.0")
        self.assertEqual(self.by_member[PV_MEMBERS[6]].family_key, "Apple___Apple_scab/unknown 6")

    def test_file_name_copies_share_the_fallback_family(self):
        original = agri._plant_family_key("raw/color/Apple___Apple_scab/s6___Unknown 6.JPG", {})
        self.assertEqual(original, "Apple___Apple_scab/unknown 6")
        for name in ("Unknown 6 copy.JPG", "Unknown 6 Copy 2.JPG", "Unknown 6 copy 13.jpg"):
            with self.subTest(name=name):
                self.assertEqual(agri._plant_family_key(f"raw/color/Apple___Apple_scab/s8___{name}", {}), original)
        self.assertEqual(
            agri._plant_family_key("raw/color/Apple___Apple_scab/s9___Copy machine 1.JPG", {}),
            "Apple___Apple_scab/copy machine 1",
        )

    def test_candidates_convert_to_verified_problems(self):
        for c in self.candidates:
            with self.subTest(native_id=c.native_id):
                problem = hf.to_problem(c, agri.SPECS[c.source], index=0)
                self.assertEqual((problem.lineage.label_provenance, problem.verifier.gold), ("HUMAN", c.gold))


class DeepWeedsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_deepweeds(self.root)
        self.candidates = list(agri.iter_deepweeds(self.root))
        self.by_id = {c.native_id: c for c in self.candidates}

    def test_valid_rows_only_with_row_index_ids(self):
        self.assertEqual([c.native_id for c in self.candidates], ["train:0", "train:1", "train:3", "train:4"])

    def test_gold_text_and_pool(self):
        expected = {"train:0": "Lantana", "train:1": "none of the listed weed species",
                    "train:3": "Lantana", "train:4": "Siam weed"}
        pool = ["Lantana", "Parkinsonia", "Siam weed", "none of the listed weed species"]
        for native_id, text in expected.items():
            with self.subTest(native_id=native_id):
                c = self.by_id[native_id]
                self.assertEqual(c.choices["ABCD".index(c.gold)], text)
                self.assertEqual(sorted(c.choices), pool)
                self.assertEqual((c.question, c.subject, c.answer_type), (agri.DEEPWEEDS_QUESTION, "Agriculture", "mcq"))

    def test_family_key_is_image_sha256(self):
        c = self.by_id["train:0"]
        image = png((10, 0, 0))
        self.assertEqual(c.images, (image,))
        self.assertEqual(c.family_key, hashlib.sha256(image).hexdigest())

    def test_iteration_is_deterministic(self):
        self.assertEqual(self.candidates, list(agri.iter_deepweeds(self.root)))

    def test_candidates_convert_to_verified_problems(self):
        for c in self.candidates:
            with self.subTest(native_id=c.native_id):
                problem = hf.to_problem(c, agri.SPECS[c.source], index=0)
                self.assertEqual((problem.actor.choices, problem.lineage.source), (c.choices, "hf-deepweeds"))


class CropArticleTests(unittest.TestCase):
    def test_vowel_initial_takes_an(self):
        for crop, article in (("Orange", "an"), ("Apple", "an"), ("apple", "an"), ("Tomato", "a"),
                              ("Corn (maize)", "a"), ("", "a")):
            with self.subTest(crop=crop):
                self.assertEqual(agri._crop_article(crop), article)


class SpecTests(unittest.TestCase):
    def test_pinned_sources_and_iterators(self):
        self.assertEqual(set(agri.SPECS), {"hf-plantvillage", "hf-deepweeds"})
        self.assertEqual(set(agri.ITERATORS), set(agri.SPECS))
        for source, spec in agri.SPECS.items():
            with self.subTest(source=source):
                self.assertEqual(spec.source, source)
                self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
                self.assertEqual((spec.label_provenance, spec.family), ("HUMAN", "plant_photo"))
        self.assertEqual(agri.SPECS["hf-plantvillage"].repo, "mohanty/PlantVillage")
        self.assertEqual(agri.SPECS["hf-deepweeds"].license, "CC-BY-4.0")


if __name__ == "__main__":
    unittest.main()
