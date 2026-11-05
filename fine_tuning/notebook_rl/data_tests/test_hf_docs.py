"""Unit tests for hf_docs (ChartQA and TAT-DQA). Synthetic fixtures in temporary directories only."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fine_tuning.notebook_rl.adapters import hf_docs

CHART_IMAGE = b"\x89PNG-chart"
PAGE_PNG = b"\x89PNG-page-one"
CHART_SCHEMA = pa.schema([
    ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("query", pa.string()),
    ("label", pa.list_(pa.string())),
    ("human_or_machine", pa.int64()),
])


def _chart_row(query: str, label: list[str], human: int = 0, image: bytes | None = CHART_IMAGE) -> dict:
    return {"image": None if image is None else {"bytes": image, "path": None},
            "query": query, "label": label, "human_or_machine": human}


def _write_chartqa(root: Path, rows: list[dict]) -> Path:
    (root / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=CHART_SCHEMA),
                   root / "data" / "train-00000-of-00001.parquet")
    return root


def _tat_q(uid: str, question: str, answer, answer_type: str = "arithmetic", scale: str = "") -> dict:
    return {"uid": uid, "question": question, "answer": answer, "answer_type": answer_type, "scale": scale,
            "derivation": "SECRET-DERIVATION", "facts": ["x"], "block_mapping": []}


def _tat_entry(doc_uid: str, page: int, questions: list[dict]) -> dict:
    return {"doc": {"uid": doc_uid, "page": page, "source": f"{doc_uid}.pdf"}, "questions": questions}


def _write_tatdqa(root: Path, entries: list[dict], members: dict[str, bytes]) -> Path:
    (root / "tatdqa_dataset_train.json").write_text(json.dumps(entries), encoding="utf-8")
    with zipfile.ZipFile(root / "tatdqa_docs_train.zip", "w") as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    return root


class ChartqaTest(unittest.TestCase):
    def test_keeps_only_human_single_label_rows_with_image(self):
        rows = [
            _chart_row("How many bars?", ["4"]),                # row 0 kept
            _chart_row("How many bars?", ["4"], human=1),       # machine-generated
            _chart_row("How many bars?", ["4", "5"]),           # two labels
            _chart_row("How many bars?", []),                   # no label
            _chart_row("How many bars?", ["4"], image=None),    # missing image
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidates = list(hf_docs.iter_chartqa(_write_chartqa(Path(tmp), rows)))
        self.assertEqual([c.native_id for c in candidates], ["train-00000-of-00001:0"])
        self.assertEqual(candidates[0].images, (CHART_IMAGE,))
        self.assertEqual(candidates[0].family_key, hashlib.sha256(CHART_IMAGE).hexdigest())

    def test_answer_types_and_gold(self):
        rows = [
            _chart_row("Is it higher?", ["Yes."]),          # mcq, gold A
            _chart_row("Is it higher?", ["No"]),            # mcq, gold B
            _chart_row("What is the rate?", ["12.5."]),     # decimal
            _chart_row("What is the change?", ["-3"]),      # decimal
            _chart_row("Which country?", ["West Germany"]),  # short_text
            _chart_row("Explain it?", ["one, two"]),        # comma -> skipped
            _chart_row("Explain it?", ["a b c d e"]),       # 5 words -> skipped
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidates = list(hf_docs.iter_chartqa(_write_chartqa(Path(tmp), rows)))
        self.assertEqual(
            [(c.answer_type, c.gold, c.choices, c.accept) for c in candidates],
            [("mcq", "A", ("Yes", "No"), {}),
             ("mcq", "B", ("Yes", "No"), {}),
             ("decimal", "12.5", (), {"rel_tol": "0.05"}),
             ("decimal", "-3", (), {"rel_tol": "0.05"}),
             ("short_text", "West Germany", (), {})],
        )

    def test_year_answers_match_exactly(self):
        rows = [
            _chart_row("In which year was the share highest?", ["2019"]),  # year read -> exact
            _chart_row("When did sales peak?", ["1998"]),                  # year read -> exact
            _chart_row("What is the value in 2015?", ["2000"]),            # a value -> relaxed
            _chart_row("How many years did it grow?", ["12"]),             # not a 4-digit year -> relaxed
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidates = list(hf_docs.iter_chartqa(_write_chartqa(Path(tmp), rows)))
        self.assertEqual([c.accept for c in candidates], [{}, {}, {"rel_tol": "0.05"}, {"rel_tol": "0.05"}])

    def test_subject_keywords_in_priority_order(self):
        rows = [
            _chart_row("What is the gdp of the brand?", ["1"]),  # Marketing before Economics
            _chart_row("What was the inflation rate?", ["1"]),   # Economics
            _chart_row("How many people answered?", ["1"]),      # Sociology
            _chart_row("What is the value?", ["1"]),             # Manage
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidates = list(hf_docs.iter_chartqa(_write_chartqa(Path(tmp), rows)))
        self.assertEqual([c.subject for c in candidates], ["Marketing", "Economics", "Sociology", "Manage"])


class TatdqaTest(unittest.TestCase):
    def _run(self, entries: list[dict], members: dict[str, bytes]) -> list:
        with tempfile.TemporaryDirectory() as tmp:
            return list(hf_docs.iter_tatdqa(_write_tatdqa(Path(tmp), entries, members)))

    def test_arithmetic_and_count_gold_is_plain_decimal(self):
        questions = [
            _tat_q("q1", "What is the total?", 27178.5),
            _tat_q("q2", "How many items?", 3, answer_type="count"),
            _tat_q("q3", "Big number?", 1e20),
            _tat_q("q4", "Small number?", 0.25),
            _tat_q("q5", "Span?", ["x"], answer_type="span"),
            _tat_q("q6", "String answer?", "5"),
            _tat_q("q7", "Bool answer?", True),
            _tat_q("q8", "List answer?", [5]),
        ]
        candidates = self._run([_tat_entry("doc1", 1, questions)], {"train/doc1_1.png": PAGE_PNG})
        self.assertEqual([c.gold for c in candidates], ["27178.5", "3", "100000000000000000000", "0.25"])
        self.assertEqual({c.answer_type for c in candidates}, {"decimal"})
        self.assertEqual({c.accept["rel_tol"] for c in candidates}, {"0.01"})

    def test_scale_hint_subject_and_derivation_not_read(self):
        questions = [
            _tat_q("q1", " What was the stock return? ", 0.5, scale="percent"),
            _tat_q("q2", "What was the total?", 7, scale="thousand"),
        ]
        candidates = self._run([_tat_entry("doc1", 1, questions)], {"train/doc1_1.png": PAGE_PNG})
        self.assertEqual([c.question for c in candidates], [
            "What was the stock return?\nAnswer scale should be: percent.",
            "What was the total?\nAnswer scale should be: thousand.",
        ])
        self.assertEqual([c.subject for c in candidates], ["Finance", "Accounting"])
        self.assertTrue(all("SECRET" not in c.question for c in candidates))

    def test_page_png_member_is_required_and_ids_are_stable(self):
        entries = [
            _tat_entry("doc1", 1, [_tat_q("q1", "What is the total?", 2)]),
            _tat_entry("doc2", 2, [_tat_q("q2", "What is the total?", 2)]),  # doc2_2.png absent
        ]
        members = {"train/doc1_1.png": PAGE_PNG, "train/doc2_1.png": b"other-page", "train/doc1.pdf": b"%PDF"}
        candidates = self._run(entries, members)
        self.assertEqual([(c.native_id, c.family_key, c.images) for c in candidates],
                         [("doc1:q1", "doc1.pdf", (PAGE_PNG,))])


class RegistryTest(unittest.TestCase):
    def test_iterators_and_specs_cover_all_sources(self):
        self.assertEqual(set(hf_docs.ITERATORS), {"hf-chartqa", "hf-tatdqa", "hf-tatdqa-pdf"})
        self.assertEqual(set(hf_docs.SPECS), {"hf-chartqa", "hf-tatdqa", "hf-tatdqa-pdf"})


if __name__ == "__main__":
    unittest.main()
