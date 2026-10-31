"""Synthetic-fixture tests for the hf-vqarad and hf-slake converters (no network, no GPU)."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fine_tuning.notebook_rl.adapters import hf_medical as m
from fine_tuning.notebook_rl.adapters.hf_common import to_problem

_IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
_VQARAD_SCHEMA = pa.schema([("image", _IMAGE_TYPE), ("question", pa.string()), ("answer", pa.string())])


def _image(raw: bytes | None) -> dict | None:
    return None if raw is None else {"bytes": raw, "path": None}


def _png(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _write_vqarad(repo: Path, rows: list[dict]) -> Path:
    (repo / "data").mkdir(parents=True)
    table = pa.Table.from_pylist(rows, schema=_VQARAD_SCHEMA)
    pq.write_table(table, repo / "data" / "train-00000-of-00001-test.parquet")
    return repo


class AnswerRuleTests(unittest.TestCase):
    def test_yes_no_become_two_option_mcq(self):
        self.assertEqual(m._qa_answer("is it?", " YES "), ("Is it?", ("Yes", "No"), "mcq", "A"))
        self.assertEqual(m._qa_answer("is it?", "no"), ("Is it?", ("Yes", "No"), "mcq", "B"))

    def test_short_answer_bounds(self):
        self.assertEqual(m._qa_answer("q", "a b c"), ("Q", (), "short_text", "a b c"))
        self.assertIsNone(m._qa_answer("q", "a b c d"))
        self.assertIsNone(m._qa_answer("q", "a,b"))
        self.assertEqual(m._qa_answer("q", "x" * 40), ("Q", (), "short_text", "x" * 40))
        self.assertIsNone(m._qa_answer("q", "x" * 41))

    def test_question_keeps_rest_and_capitalizes_first_letter(self):
        self.assertEqual(m._qa_answer("  what ORGAN? ", "ct")[0], "What ORGAN?")

    def test_empty_or_non_string_fields_are_skipped(self):
        self.assertIsNone(m._qa_answer("  ", "ct"))
        self.assertIsNone(m._qa_answer("q", "  "))
        self.assertIsNone(m._qa_answer("q", None))


class VqaradTests(unittest.TestCase):
    def test_rows_map_to_candidates_and_skip_invalid(self):
        rows = [
            {"image": _image(b"img-a"), "question": "is there a fracture", "answer": "Yes"},
            {"image": _image(b"img-b"), "question": " what organ? ", "answer": " ct "},
            {"image": _image(b"img-c"), "question": "describe it", "answer": "long, multi word"},
            {"image": _image(None), "question": "where", "answer": "left"},
            {"image": _image(b"img-d"), "question": "is it normal", "answer": "NO"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            got = list(m.iter_vqarad(_write_vqarad(Path(tmp), rows)))
        self.assertEqual(
            [(c.native_id, c.subject, c.question, c.choices, c.answer_type, c.gold) for c in got],
            [("train:0", "Clinical_Medicine", "Is there a fracture", ("Yes", "No"), "mcq", "A"),
             ("train:1", "Clinical_Medicine", "What organ?", (), "short_text", "ct"),
             ("train:4", "Clinical_Medicine", "Is it normal", ("Yes", "No"), "mcq", "B")],
        )
        self.assertEqual(got[1].images, (b"img-b",))
        self.assertEqual(got[1].family_key, hashlib.sha256(b"img-b").hexdigest())

    def test_converted_rows_build_problems(self):
        rows = [
            {"image": _image(_png((255, 0, 0))), "question": "is there a fracture", "answer": "no"},
            {"image": _image(_png((0, 255, 0))), "question": "what organ", "answer": "ct"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidates = list(m.iter_vqarad(_write_vqarad(Path(tmp), rows)))
        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertIsNotNone(to_problem(candidate, m.SPECS["hf-vqarad"], index=0))

    def test_missing_train_shards_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                list(m.iter_vqarad(Path(tmp)))


class SlakeTests(unittest.TestCase):
    def test_english_rows_with_zip_images_map_to_candidates(self):
        rows = [
            {"qid": 1, "img_name": "xmlab1/source.jpg", "q_lang": "en", "question": "where is the lesion?",
             "answer": "Liver", "answer_type": "OPEN", "content_type": "Organ"},
            {"qid": 2, "img_name": "xmlab1/source.jpg", "q_lang": "en", "question": "is there a mass",
             "answer": "yes", "answer_type": "CLOSED", "content_type": "Abnormality"},
            {"qid": 3, "img_name": "xmlab1/source.jpg", "q_lang": "zh", "question": "where",
             "answer": "liver", "answer_type": "OPEN", "content_type": "Organ"},
            {"qid": 4, "img_name": "xmlab9/source.jpg", "q_lang": "en", "question": "where",
             "answer": "Lung", "answer_type": "OPEN", "content_type": "Organ"},
            {"qid": 5, "img_name": "xmlab2/source.jpg", "q_lang": "en", "question": "how many",
             "answer": "two, maybe", "answer_type": "OPEN", "content_type": "Quantity"},
            {"qid": 6, "img_name": "xmlab2/source.jpg", "q_lang": "en", "question": "what plane",
             "answer": "axial", "answer_type": "OPEN", "content_type": "Plane"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "train.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            with zipfile.ZipFile(repo / "imgs.zip", "w") as archive:
                archive.writestr("imgs/xmlab1/source.jpg", b"slake-1")
                archive.writestr("imgs/xmlab2/source.jpg", b"slake-2")
            got = list(m.iter_slake(repo))
        self.assertEqual(
            [(c.native_id, c.subject, c.question, c.answer_type, c.gold, c.family_key) for c in got],
            [("train:1", "Basic_Medical_Science", "Where is the lesion?", "short_text", "Liver", "xmlab1"),
             ("train:2", "Clinical_Medicine", "Is there a mass", "mcq", "A", "xmlab1"),
             ("train:6", "Basic_Medical_Science", "What plane", "short_text", "axial", "xmlab2")],
        )
        self.assertEqual(got[0].images, (b"slake-1",))
        self.assertEqual(got[2].images, (b"slake-2",))


SLAKE_RAW = Path("/home/jang2/mmdl-data/training/raw/hf-pool-v1/BoKelvin__SLAKE")


def _slake_row(qid: int, answer, img: str = "xmlab1/source.jpg", **fields) -> dict:
    row = {"qid": qid, "img_name": img, "q_lang": "en", "question": "what is this", "answer": answer,
           "answer_type": "OPEN", "content_type": "Organ"}
    row.update(fields)
    return row


class SlakeFilterTests(unittest.TestCase):
    def test_filter_version_is_pinned(self):
        self.assertEqual(m.FILTER_VERSION, "slake-filter-v1")

    def test_prefix_answers(self):
        answers = ["Cardiomegal", "Cardiomega", "Cardiomegaly", "Both Lung", "Both Lungs", "Lung", "Lungs",
                   "Liver", "Liver cancer", "Heart", "Heartbeat", " CT ", "CT scan", "", None, "Brain"]
        self.assertEqual(m.slake_prefix_answers(answers), {
            "both lung": ("both lungs",),                              # case-insensitive, whitespace-stripped
            "cardiomega": ("cardiomegal", "cardiomegaly"),
            "cardiomegal": ("cardiomegaly",),
            "heart": ("heartbeat",),                                   # exactly 5 chars: caught
        })                                                             # Lung (4 chars), Liver + space, CT: kept

    def test_prefix_needs_a_strict_longer_answer_with_alphabetic_next_char(self):
        self.assertEqual(m.slake_prefix_answers(["Lung", "Lungs"]), {})                     # < 5 chars
        self.assertEqual(m.slake_prefix_answers(["Liver", "Liver-cancer", "Liver cancer"]), {})
        self.assertEqual(m.slake_prefix_answers(["Liver", "Liver"]), {})                    # equal is not strict
        self.assertEqual(m.slake_prefix_answers(["Liver", "Livers"]), {"liver": ("livers",)})

    def test_iter_slake_drops_empty_and_truncated_answers(self):
        rows = [
            _slake_row(1, "Cardiomegaly"), _slake_row(2, "Cardiomegal"), _slake_row(3, "Cardiomega"),
            _slake_row(4, "Both Lungs"), _slake_row(5, "Both Lung"), _slake_row(6, "Lung"),
            _slake_row(7, ""), _slake_row(8, "   "), _slake_row(9, None),
            _slake_row(10, "Liver"), _slake_row(11, "Liver cancer"),
            _slake_row(12, "Cardiomegal", q_lang="zh"),            # other language: neither kept nor counted
        ]
        dropped = Counter()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "train.json").write_text(json.dumps(rows), encoding="utf-8")
            with zipfile.ZipFile(repo / "imgs.zip", "w") as archive:
                archive.writestr("imgs/xmlab1/source.jpg", b"slake-1")
            got = list(m.iter_slake(repo, dropped))
        self.assertEqual([c.native_id for c in got], ["train:1", "train:4", "train:6", "train:10", "train:11"])
        self.assertEqual(dict(dropped), {"empty_answer": 3, "prefix_answer": 3})

    @unittest.skipUnless(SLAKE_RAW.exists(), "local SLAKE train.json not present")
    def test_real_train_json_prefix_pairs(self):
        rows = json.loads((SLAKE_RAW / "train.json").read_text(encoding="utf-8"))
        prefixes = m.slake_prefix_answers(r["answer"] for r in rows if r["q_lang"] == "en")
        print(f"\nSLAKE {m.FILTER_VERSION}: {len(prefixes)} truncated answers (prefix -> longer answers)")
        rows_per_prefix = Counter(r["answer"].strip().casefold() for r in rows
                                  if r["q_lang"] == "en" and r["answer"].strip().casefold() in prefixes)
        for prefix, longer in prefixes.items():
            print(f"  {prefix!r} ({rows_per_prefix[prefix]} rows) -> {list(longer)}")
        self.assertIn("cardiomegal", prefixes)
        self.assertIn("cardiomega", prefixes)
        self.assertIn("both lung", prefixes)
        dropped = Counter()
        zipped = SLAKE_RAW / "imgs.zip"
        self.assertTrue(zipped.exists())
        kept = sum(1 for _ in m.iter_slake(SLAKE_RAW, dropped))
        print(f"SLAKE {m.FILTER_VERSION}: kept={kept} dropped={dict(dropped)}")
        self.assertGreater(dropped["prefix_answer"], 0)


class SpecTests(unittest.TestCase):
    def test_specs_are_pinned_human_radiology(self):
        self.assertEqual(set(m.SPECS), {"hf-vqarad", "hf-slake"})
        for spec in m.SPECS.values():
            self.assertEqual(len(spec.revision), 40)
            self.assertEqual(spec.label_provenance, "HUMAN")
            self.assertEqual(spec.family, "radiology")
        self.assertEqual(set(m.ITERATORS), set(m.SPECS))


if __name__ == "__main__":
    unittest.main()
