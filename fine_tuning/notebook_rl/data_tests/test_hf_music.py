"""Synthetic-fixture tests for the SSMR-Bench converter. No network, no real dataset."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fine_tuning.notebook_rl.adapters import hf_music

_SCHEMA = pa.schema([
    ("question", pa.string()),
    ("abc_context", pa.string()),
    ("sheet_music_img", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("correct_answer", pa.string()),
    ("incorrect_answer1", pa.string()),
    ("incorrect_answer2", pa.string()),
    ("incorrect_answer3", pa.string()),
])


def _row(**overrides: object) -> dict:
    row: dict = {
        "question": "What is the key signature?", "abc_context": None,
        "sheet_music_img": {"bytes": b"sheet-0", "path": None},
        "correct_answer": "G major", "incorrect_answer1": "C major",
        "incorrect_answer2": "D minor", "incorrect_answer3": "A minor",
    }
    row.update(overrides)
    return row


ROWS = [
    _row(),                                                            # 0 valid
    _row(abc_context="X:1\nK:C\nCDEF|"),                               # 1 ABC text -> skipped
    _row(sheet_music_img=None),                                        # 2 no image -> skipped
    _row(sheet_music_img={"bytes": b"", "path": None}),                # 3 empty image -> skipped
    _row(incorrect_answer2=None),                                      # 4 missing answer -> skipped
    _row(incorrect_answer1="G major"),                                 # 5 duplicate answer -> skipped
    _row(question="  How many beats?  ", abc_context="",               # 6 empty ABC kept, text stripped
         sheet_music_img={"bytes": b"sheet-6", "path": None},
         correct_answer=" 4 ", incorrect_answer1="3", incorrect_answer2="2", incorrect_answer3="1"),
    _row(question="   "),                                              # 7 blank question -> skipped
]


def _candidates(rows: list[dict]) -> list:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "Sylence__SSMR-Bench"
        (repo / "data").mkdir(parents=True)
        table = pa.Table.from_pylist(rows, schema=_SCHEMA)
        pq.write_table(table, repo / "data" / "Train_visual-00000-of-00001.parquet")
        return list(hf_music.ITERATORS["hf-ssmr"](repo))


class SsmrConverterTests(unittest.TestCase):
    def test_only_clean_rows_are_kept_with_original_row_index(self) -> None:
        self.assertEqual([c.native_id for c in _candidates(ROWS)], ["Train_visual:0", "Train_visual:6"])

    def test_valid_row_gold_points_at_correct_answer(self) -> None:
        candidate = _candidates(ROWS)[0]
        self.assertEqual(candidate.source, "hf-ssmr")
        self.assertEqual(candidate.subject, "Music")
        self.assertEqual(candidate.answer_type, "mcq")
        self.assertEqual(candidate.question, "What is the key signature?")
        self.assertEqual(sorted(candidate.choices), sorted(["G major", "C major", "D minor", "A minor"]))
        self.assertEqual(candidate.choices["ABCDEFGHIJ".index(candidate.gold)], "G major")
        self.assertEqual(candidate.accept, {})
        self.assertEqual(candidate.images, (b"sheet-0",))
        self.assertEqual(candidate.family_key, hashlib.sha256(b"sheet-0").hexdigest())

    def test_padded_text_is_stripped_and_gold_follows_answer(self) -> None:
        candidate = _candidates(ROWS)[1]
        self.assertEqual(candidate.question, "How many beats?")
        self.assertEqual(sorted(candidate.choices), ["1", "2", "3", "4"])
        self.assertEqual(candidate.choices["ABCDEFGHIJ".index(candidate.gold)], "4")

    def test_abc_text_never_enters_a_candidate(self) -> None:
        self.assertNotIn("K:C", repr(_candidates(ROWS)))

    def test_program_labels_are_deterministic(self) -> None:
        self.assertEqual(_candidates(ROWS), _candidates(ROWS))

    def test_spec_pins_source_license_and_family(self) -> None:
        spec = hf_music.SPECS["hf-ssmr"]
        self.assertEqual(spec.repo, "Sylence/SSMR-Bench")
        self.assertEqual(len(spec.revision), 40)
        self.assertTrue(set(spec.revision) <= set("0123456789abcdef"))
        self.assertEqual((spec.license, spec.label_provenance, spec.family), ("MIT", "PROGRAM", "music"))

    def test_missing_train_shard_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                list(hf_music.ITERATORS["hf-ssmr"](Path(tmp)))


if __name__ == "__main__":
    unittest.main()
