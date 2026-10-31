"""Pinned HF ChartQA and TAT-DQA train rows -> verifier-only Candidate records.

ChartQA gold is the upstream human-written label (human_or_machine == 0 rows only). TAT-DQA gold is
the upstream numeric answer of arithmetic/count questions; the derivation, facts and block mapping
fields are never read. Rows with a missing image or failing a data-quality check are skipped
silently; programming errors propagate. No upstream derivation or rationale text is copied into a
Candidate.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq

from .hf_common import Candidate, SourceSpec, mcq_from_answer

SPECS: dict[str, SourceSpec] = {
    "hf-chartqa": SourceSpec(
        "hf-chartqa", "HuggingFaceM4/ChartQA", "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
        "GPL-3.0", "HUMAN", "ChartQA human-written questions (AMT) over Statista/Pew/OWID/OECD charts",
        "chart_bar",
    ),
    "hf-tatdqa": SourceSpec(
        "hf-tatdqa", "next-tat/TAT-DQA", "bc46208b22f0bfe47152c98fd0938d238ce868c2",
        "CC-BY-4.0", "HUMAN", "TAT-DQA financial annotators (Zhu et al. 2022)", "document",
    ),
}

_CHARTQA_COLUMNS = ["image", "query", "label", "human_or_machine"]
# Checked in order; the first subject with a keyword at a word start in the lowercased query wins.
_CHARTQA_RULES = (
    ("Marketing", ("brand", "sales", "market", "advertis", "consumer", "social media", "smartphone",
                   "app", "online", "retail", "customer")),
    ("Economics", ("gdp", "inflation", "unemployment", "economy", "price", "trade", "export", "import",
                   "debt", "interest rate", "wage")),
    ("Sociology", ("people", "adult", "american", "survey", "opinion", "population", "voter", "public",
                   "women", "men", "religio")),
)
_NUMBER = re.compile(r"-?\d+(\.\d+)?")
# Year answers are axis reads and must match exactly: 5% of 2019 would accept 1918..2120.
_YEAR = re.compile(r"(1[5-9]|20)\d\d")
_YEAR_QUESTION = re.compile(r"\b(year|when)\b")
_TATDQA_TYPES = ("arithmetic", "count")
_TATDQA_FINANCE = re.compile(r"(?i)(stock|share|dividend|return|interest|investment|cash flow|market)")


def _image_bytes(image: dict | None) -> bytes | None:
    raw = image.get("bytes") if image else None
    return raw if isinstance(raw, bytes) and raw else None


def _train_rows(repo_dir: Path, columns: list[str]) -> Iterator[tuple[Path, int, dict]]:
    files = sorted(repo_dir.glob("data/train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train parquet files under {repo_dir / 'data'}")
    for path in files:
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=columns):
            for row in batch.to_pylist():
                yield path, index, row
                index += 1


def _chartqa_candidate(row: dict, native_id: str) -> Candidate | None:
    raw = _image_bytes(row["image"])
    query, labels = row["query"], row["label"]
    if row["human_or_machine"] != 0 or raw is None or not isinstance(query, str) or not query.strip():
        return None
    if not isinstance(labels, list) or len(labels) != 1 or not isinstance(labels[0], str):
        return None
    answer = labels[0].strip().removesuffix(".").strip()
    if not answer:
        return None
    question = query.strip()
    lowered = question.lower()
    subject = next((name for name, words in _CHARTQA_RULES
                    if any(re.search(rf"\b{re.escape(w)}", lowered) for w in words)), "Manage")
    if answer.lower() in ("yes", "no"):
        question, choices, gold = mcq_from_answer(question, ["Yes", "No"], 0 if answer.lower() == "yes" else 1)
        answer_type, accept = "mcq", {}
    elif _NUMBER.fullmatch(answer):
        exact = _YEAR.fullmatch(answer) and _YEAR_QUESTION.search(lowered)
        choices, gold, answer_type, accept = (), answer, "decimal", {} if exact else {"rel_tol": "0.05"}
    elif len(answer.split()) <= 4 and "," not in answer:
        choices, gold, answer_type, accept = (), answer, "short_text", {}
    else:
        return None
    return Candidate(source="hf-chartqa", native_id=native_id, subject=subject, question=question,
                     choices=choices, answer_type=answer_type, gold=gold, accept=accept,
                     images=(raw,), family_key=hashlib.sha256(raw).hexdigest())


def iter_chartqa(repo_dir: Path) -> Iterator[Candidate]:
    for path, index, row in _train_rows(repo_dir, _CHARTQA_COLUMNS):
        candidate = _chartqa_candidate(row, f"{path.stem}:{index}")
        if candidate is not None:
            yield candidate


def _tatdqa_candidate(doc_uid: str, source: str, q: dict, raw: bytes) -> Candidate | None:
    # family_key is the source report, not the page uid: 2207 train pages come from 172 reports.
    question, answer, scale = q["question"], q["answer"], q["scale"]
    if q["answer_type"] not in _TATDQA_TYPES or not raw or not isinstance(question, str) or not question.strip():
        return None
    # type() keeps bool out (bool is an int subclass) and rejects non-finite floats.
    if type(answer) is int:
        gold = str(answer)
    elif type(answer) is float and math.isfinite(answer):
        gold = format(Decimal(repr(answer)), "f")  # plain notation, never 1e+20
    else:
        return None
    base = question.strip()
    text = f"{base}\nAnswer scale should be: {scale}." if scale else base
    subject = "Finance" if _TATDQA_FINANCE.search(base) else "Accounting"
    return Candidate(source="hf-tatdqa", native_id=f"{doc_uid}:{q['uid']}", subject=subject, question=text,
                     choices=(), answer_type="decimal", gold=gold, accept={"rel_tol": "0.01"},
                     images=(raw,), family_key=source)


def iter_tatdqa(repo_dir: Path) -> Iterator[Candidate]:
    with (repo_dir / "tatdqa_dataset_train.json").open(encoding="utf-8") as handle:
        entries = json.load(handle)
    with zipfile.ZipFile(repo_dir / "tatdqa_docs_train.zip") as archive:
        members = set(archive.namelist())
        for entry in entries:
            doc = entry["doc"]
            member = f"train/{doc['uid']}_{doc['page']}.png"  # page PNG only; .pdf/.json members unused
            if member not in members:
                continue
            raw = archive.read(member)
            for q in entry["questions"]:
                candidate = _tatdqa_candidate(doc["uid"], doc["source"], q, raw)
                if candidate is not None:
                    yield candidate


ITERATORS = {
    "hf-chartqa": iter_chartqa,
    "hf-tatdqa": iter_tatdqa,
}
