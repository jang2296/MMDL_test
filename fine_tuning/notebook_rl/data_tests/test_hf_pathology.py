"""Authored fixtures only: tiny synthetic parquet shards and a synthetic 12 x 5000 sine ECG in a temp directory."""
import hashlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fine_tuning.notebook_rl.adapters import hf_common as hf
from fine_tuning.notebook_rl.adapters import hf_pathology as path_mod

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NCT_SHARD_0 = "NCT_CRC_HE_100K-00000-of-00002-aaaa"
NCT_SHARD_1 = "NCT_CRC_HE_100K-00001-of-00002-bbbb"
BM_SHARD_0 = "train-00000-of-00002"
BM_SHARD_1 = "train-00001-of-00002"


def png(color):
    out = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(out, format="PNG")
    return out.getvalue()


def write_labeled(root, shards):
    (root / "data").mkdir(exist_ok=True)
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    for stem, rows in shards.items():
        images = [{"bytes": png(color), "path": None} if color is not None else None for color, _ in rows]
        table = pa.table({
            "image": pa.array(images, type=image_type),
            "label": pa.array([label for _, label in rows], type=pa.int64()),
        })
        pq.write_table(table, root / "data" / f"{stem}.parquet")


def sine_signal(amplitude=1.0, samples=5000, rows=12):
    return [[amplitude * math.sin(2 * math.pi * (r + 1) * t / samples * 10) for t in range(samples)]
            for r in range(rows)]


class NctTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_labeled(self.root, {
            NCT_SHARD_1: [((10, 10, 0), 6)],
            NCT_SHARD_0: [((10, 0, 0), 0), ((0, 10, 0), 8), (None, 3), ((0, 0, 10), 99), ((20, 0, 0), 4)],
            "NCT_CRC_HE_100K_NONORM-00000-of-00001-cccc": [((30, 0, 0), 0)],   # different dataset file: ignored
            "CRC_VAL_HE_7K-00000-of-00001-dddd": [((40, 0, 0), 0)],            # validation file: ignored
        })
        self.candidates = list(path_mod.iter_nctcrc(self.root))
        self.by_id = {c.native_id: c for c in self.candidates}

    def test_sorted_shards_valid_rows_and_row_index_ids(self):
        self.assertEqual([c.native_id for c in self.candidates],
                         [f"{NCT_SHARD_0}:0", f"{NCT_SHARD_0}:1", f"{NCT_SHARD_0}:4", f"{NCT_SHARD_1}:0"])

    def test_gold_name_and_pool(self):
        expected = {f"{NCT_SHARD_0}:0": "adipose tissue", f"{NCT_SHARD_0}:1": "colorectal adenocarcinoma epithelium",
                    f"{NCT_SHARD_0}:4": "mucus", f"{NCT_SHARD_1}:0": "normal colon mucosa"}
        for native_id, text in expected.items():
            with self.subTest(native_id=native_id):
                c = self.by_id[native_id]
                self.assertEqual(c.choices["ABCD".index(c.gold)], text)
                self.assertEqual(len(set(c.choices)), 4)
                self.assertLessEqual(set(c.choices), set(path_mod.NCT_NAMES.values()))
                self.assertEqual((c.source, c.subject, c.answer_type), ("hf-nctcrc", "Basic_Medical_Science", "mcq"))
                self.assertEqual(c.question, "This is an H&E-stained human tissue patch. "
                                             "Which tissue class does it show?")

    def test_names_are_the_nine_readable_classes(self):
        self.assertEqual(list(path_mod.NCT_NAMES), list(path_mod.NCT_CODES))
        self.assertEqual(path_mod.NCT_NAMES["BACK"], "background (no tissue)")
        self.assertEqual(path_mod.NCT_NAMES["STR"], "cancer-associated stroma")

    def test_family_key_is_image_sha256_and_deterministic(self):
        c = self.by_id[f"{NCT_SHARD_0}:0"]
        self.assertEqual(c.images, (png((10, 0, 0)),))
        self.assertEqual(c.family_key, hashlib.sha256(png((10, 0, 0))).hexdigest())
        self.assertEqual(self.candidates, list(path_mod.iter_nctcrc(self.root)))

    def test_missing_shards_raise(self):
        with tempfile.TemporaryDirectory() as empty:
            (Path(empty) / "data").mkdir()
            with self.assertRaises(FileNotFoundError):
                list(path_mod.iter_nctcrc(Path(empty)))

    def test_candidates_convert_to_verified_problems(self):
        for c in self.candidates:
            with self.subTest(native_id=c.native_id):
                problem = hf.to_problem(c, path_mod.SPECS[c.source], index=0)
                self.assertEqual((problem.lineage.label_provenance, problem.verifier.gold), ("HUMAN", c.gold))


class BoneMarrowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_labeled(self.root, {
            BM_SHARD_1: [((10, 10, 0), 20)],
            BM_SHARD_0: [((10, 0, 0), 2), ((0, 10, 0), 1), ((0, 0, 10), 16), ((5, 5, 5), 17),
                         ((20, 0, 0), 3), ((0, 20, 0), 21), ((0, 0, 20), 0)],
        })
        self.candidates = list(path_mod.iter_bonemarrow(self.root))
        self.by_id = {c.native_id: c for c in self.candidates}

    def test_artefact_other_and_out_of_range_labels_are_skipped(self):
        self.assertEqual(set(path_mod.BM_CODES) - set(path_mod.BM_NAMES), {"ART", "NIF", "OTH"})
        self.assertEqual([c.native_id for c in self.candidates],
                         [f"{BM_SHARD_0}:{i}" for i in (0, 4, 6)] + [f"{BM_SHARD_1}:0"])

    def test_gold_name_pool_and_question(self):
        expected = {f"{BM_SHARD_0}:0": "basophil", f"{BM_SHARD_0}:4": "blast",
                    f"{BM_SHARD_0}:6": "abnormal eosinophil", f"{BM_SHARD_1}:0": "promyelocyte"}
        self.assertEqual(len(path_mod.BM_NAMES), 18)
        for native_id, text in expected.items():
            with self.subTest(native_id=native_id):
                c = self.by_id[native_id]
                self.assertEqual(c.choices["ABCD".index(c.gold)], text)
                self.assertEqual(len(set(c.choices)), 4)
                self.assertLessEqual(set(c.choices), set(path_mod.BM_NAMES.values()))
                self.assertEqual((c.source, c.subject), ("hf-bonemarrow", "Diagnostics_and_Laboratory_Medicine"))
                self.assertEqual(c.question, "This is a single cell from a May-Grünwald-Giemsa stained bone "
                                             "marrow smear. Which cell type is it?")

    def test_family_key_is_image_sha256(self):
        c = self.by_id[f"{BM_SHARD_0}:0"]
        self.assertEqual(c.family_key, hashlib.sha256(png((10, 0, 0))).hexdigest())

    def test_candidates_convert_to_verified_problems(self):
        for c in self.candidates:
            with self.subTest(native_id=c.native_id):
                problem = hf.to_problem(c, path_mod.SPECS[c.source], index=0)
                self.assertEqual(problem.lineage.source, "hf-bonemarrow")


# (ecg_id, patient_id, strat_fold, validated, superclasses, leads, signal kind); kinds: ok, short, nan, few_rows
PTBXL_ROWS = (
    (1, 10, 1, True, ["NORM"], LEADS, "ok"),
    (2, 10, 8, True, ["MI"], LEADS, "ok"),
    (3, 11, 9, True, ["CD"], LEADS, "ok"),          # fold 9: skipped
    (4, 12, 2, False, ["STTC"], LEADS, "ok"),       # not human validated: skipped
    (5, 13, 3, True, ["NORM", "MI"], LEADS, "ok"),  # two superclasses: skipped
    (6, 14, 4, True, [], LEADS, "ok"),              # no superclass: skipped
    (7, 15, 5, True, ["STTC"], LEADS[:11], "few_rows"),  # 11 leads: skipped
    (8, 16, 6, True, ["CD"], LEADS, "short"),       # 100 samples instead of fs*10: skipped
    (9, 17, 7, True, ["HYP"], LEADS, "ok"),
    (10, 18, 7, True, ["XYZ"], LEADS, "ok"),        # unknown superclass: skipped
    (11, 19, 1, True, ["NORM"], LEADS, "nan"),      # non-finite signal: skipped
)


def write_ptbxl(root):
    (root / "data").mkdir()
    signals = []
    for ecg_id, *_, kind in PTBXL_ROWS:
        signal = sine_signal(samples=100) if kind == "short" else sine_signal(rows=11) if kind == "few_rows" \
            else sine_signal(amplitude=0.1 * ecg_id)
        if kind == "nan":
            signal[3][17] = float("nan")
        signals.append(signal)
    n = len(PTBXL_ROWS)
    table = pa.table({
        "ecg_id": pa.array([r[0] for r in PTBXL_ROWS], type=pa.int32()),
        "patient_id": pa.array([r[1] for r in PTBXL_ROWS], type=pa.int32()),
        "strat_fold": pa.array([r[2] for r in PTBXL_ROWS], type=pa.int8()),
        "validated_by_human": pa.array([r[3] for r in PTBXL_ROWS], type=pa.bool_()),
        "report": pa.array(["SECRET-GERMAN-REPORT"] * n, type=pa.string()),
        "superclasses": pa.array([r[4] for r in PTBXL_ROWS], type=pa.list_(pa.string())),
        "fs": pa.array([500] * n, type=pa.int16()),
        "leads": pa.array([r[5] for r in PTBXL_ROWS], type=pa.list_(pa.string())),
        "signal": pa.array(signals, type=pa.list_(pa.list_(pa.float32()))),
    })
    pq.write_table(table, root / "data" / "train-00000-of-00001.parquet")


class PtbxlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        write_ptbxl(cls.root)
        cls.candidates = list(path_mod.iter_ptbxl(cls.root))
        cls.by_id = {c.native_id: c for c in cls.candidates}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_only_validated_single_label_complete_signals(self):
        self.assertEqual([c.native_id for c in self.candidates], ["ecg:1", "ecg:2", "ecg:9"])

    def test_gold_name_all_five_options_and_question(self):
        expected = {"ecg:1": "normal ECG", "ecg:2": "myocardial infarction", "ecg:9": "hypertrophy"}
        for native_id, text in expected.items():
            with self.subTest(native_id=native_id):
                c = self.by_id[native_id]
                self.assertEqual(c.choices["ABCDE".index(c.gold)], text)
                self.assertEqual(set(c.choices), set(path_mod.PTBXL_NAMES.values()))
                self.assertEqual(len(c.choices), 5)
                self.assertEqual((c.source, c.subject, c.answer_type), ("hf-ptbxl", "Diagnostics_and_Laboratory_Medicine", "mcq"))
                self.assertEqual(c.question, "This is a 12-lead resting ECG (10 s, 25 mm/s, 10 mm/mV). "
                                             "Which diagnostic class best describes it?")

    def test_family_key_is_the_patient(self):
        self.assertEqual(self.by_id["ecg:1"].family_key, "patient:10")
        self.assertEqual(self.by_id["ecg:2"].family_key, "patient:10")
        self.assertEqual(self.by_id["ecg:9"].family_key, "patient:17")
        self.assertEqual(hf.split_for(self.by_id["ecg:1"]), hf.split_for(self.by_id["ecg:2"]))

    def test_image_is_the_rendered_ecg(self):
        with Image.open(io.BytesIO(self.by_id["ecg:1"].images[0])) as image:
            self.assertEqual((image.format, image.size), ("PNG", (1600, 1900)))
        self.assertTrue(self.by_id["ecg:1"].images != self.by_id["ecg:9"].images)

    def test_report_column_is_never_read(self):
        seen = []
        original = pq.ParquetFile.iter_batches

        def spy(self, *args, **kwargs):
            seen.append(kwargs.get("columns"))
            return original(self, *args, **kwargs)

        with mock.patch.object(pq.ParquetFile, "iter_batches", spy):
            list(path_mod.iter_ptbxl(self.root))
        self.assertTrue(seen)
        self.assertTrue(all(columns is not None and "report" not in columns for columns in seen))

    def test_candidates_convert_to_verified_problems(self):
        problem = hf.to_problem(self.by_id["ecg:2"], path_mod.SPECS["hf-ptbxl"], index=0)
        self.assertEqual((problem.lineage.label_provenance, problem.verifier.gold), ("HUMAN", self.by_id["ecg:2"].gold))


class RenderEcgTests(unittest.TestCase):
    def test_canvas_grid_and_determinism(self):
        data = path_mod.render_ecg(sine_signal(), LEADS, 500)
        self.assertEqual(data, path_mod.render_ecg(sine_signal(), LEADS, 500))
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual((image.format, image.size), ("PNG", (1600, 1900)))
            rgb = image.convert("RGB")
        self.assertEqual(rgb.getpixel((5, 5)), (255, 255, 255))                      # margin stays white
        self.assertEqual(rgb.getpixel((path_mod.ECG_LEFT, 60)), path_mod.ECG_GRID)   # vertical grid line
        self.assertEqual(rgb.getpixel((60, round(path_mod.ECG_TOP))), path_mod.ECG_GRID)  # horizontal grid line

    def test_amplitude_keeps_10_mm_per_mv_and_only_artefacts_are_cut(self):
        # a 2 mV lead 1 is drawn at its true height (overlapping strip 0, as on paper);
        # a 50 mV artefact on lead 6 stops at ECG_MAX_SWING above its baseline
        signal = [[0.0] * 5000 for _ in range(12)]
        signal[1] = [2.0] * 5000
        signal[6] = [50.0] * 5000
        with Image.open(io.BytesIO(path_mod.render_ecg(signal, LEADS, 500))) as image:
            rgb = image.convert("RGB")
        top, strip = path_mod.ECG_TOP, path_mod.ECG_STRIP_H
        px_per_mv = path_mod.ECG_CELL / 0.5
        x = path_mod.ECG_LEFT + 700
        dark = [y for y in range(1900) if rgb.getpixel((x, y)) == (0, 0, 0)]
        lead1 = top + 1.5 * strip - 2.0 * px_per_mv
        self.assertTrue(any(abs(y - lead1) <= 2 for y in dark))
        self.assertFalse(any(abs(y - (top + 1.5 * strip)) <= 2 for y in dark))      # not clamped to its baseline strip
        lead6 = top + 6.5 * strip - path_mod.ECG_MAX_SWING
        self.assertTrue(any(abs(y - lead6) <= 2 for y in dark))
        self.assertFalse(any(top + 4.5 * strip + 3 < y < lead6 - 3 for y in dark))  # nothing above the cut

    def test_signal_and_lead_names_change_the_image(self):
        flat = [[0.0] * 5000 for _ in range(12)]
        base = path_mod.render_ecg(flat, LEADS, 500)
        self.assertTrue(base != path_mod.render_ecg(sine_signal(), LEADS, 500))
        renamed = path_mod.render_ecg(flat, ["X"] + LEADS[1:], 500)
        with Image.open(io.BytesIO(base)) as a, Image.open(io.BytesIO(renamed)) as b:
            box = (path_mod.ECG_LEFT + 2, round(path_mod.ECG_TOP), path_mod.ECG_LEFT + 80, round(path_mod.ECG_TOP) + 30)
            self.assertNotEqual(a.crop(box).tobytes(), b.crop(box).tobytes())
            same = (0, 400, 1600, 1900)       # below the first strip nothing differs
            self.assertEqual(a.crop(same).tobytes(), b.crop(same).tobytes())

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            path_mod.render_ecg(sine_signal(rows=11), LEADS[:11], 500)
        with self.assertRaises(ValueError):
            path_mod.render_ecg(sine_signal(samples=100), LEADS, 500)
        nan = sine_signal()
        nan[0][0] = float("nan")
        with self.assertRaises(ValueError):
            path_mod.render_ecg(nan, LEADS, 500)


class SpecTests(unittest.TestCase):
    def test_pinned_sources_and_iterators(self):
        self.assertEqual(set(path_mod.SPECS), {"hf-nctcrc", "hf-bonemarrow", "hf-ptbxl"})
        self.assertEqual(set(path_mod.ITERATORS), set(path_mod.SPECS))
        expected = {
            "hf-nctcrc": ("1aurent/NCT-CRC-HE", "97765ecb14bf11ce7bb8779a1097d187844a67cb", "histology"),
            "hf-bonemarrow": ("ekim15/bone_marrow_cell_dataset", "a9c0cf3bf5ea1f0b502f50de40fece30a82f3aed", "cytology"),
            "hf-ptbxl": ("tirandazdylan/ptb-xl-ecg", "9e69b42bdb58a3beb39354efa096676b8fb928fd", "ecg"),
        }
        for source, spec in path_mod.SPECS.items():
            with self.subTest(source=source):
                self.assertEqual(spec.source, source)
                self.assertEqual((spec.repo, spec.revision, spec.family), expected[source])
                self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
                self.assertEqual((spec.license, spec.label_provenance), ("CC-BY-4.0", "HUMAN"))


if __name__ == "__main__":
    unittest.main()
