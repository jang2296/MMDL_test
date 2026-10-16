"""Native Geometry3K/PGPS9K train records, never predicted parses or solution targets.

The input is an explicit loader envelope: ``native`` is the original problem
object, ``diagram_annotation`` is Geometry3K's separate logic_form.json object,
and ``provenance`` holds loader evidence, not invented native fields. PGPS9K's
JSON key belongs in provenance.native_id; its native semantic clauses stay private.

Provenance requires pinned source_revision, upstream_split, native_id,
image_sha256, HUMAN label/annotation provenance and false model/benchmark taint.
PGPS additionally requires a complete declared Geometry3K parent list, origin,
and the complete 601-test-ID image-hash index. This checks exact image hashes
and declared parents only: source-loader authenticity, crop/semantic exclusion,
rights and RL approval remain separate gates. No downloads or file I/O occur.

Primary schemas: InterGPS annotation_tool/data_collection/data_examples/11/{data,
logic_form}.json and PGPS datasets/{dataset,preprossing,utils}.py at pins below.
PGPS clause examples: IJCAI-2023 paper 0376, Fig.2 (numeric lengths/angles).
"""
from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

from ..answers import number
from ..schema import ActorItem, AnswerType, Lineage, Mode, Split, VerifierItem, to_json

GEOMETRY3K_REVISION = "99e6b5293c7bb4a5eaf2a370e6b213c79ab33ebb"
PGPS_CODE_REVISION = "f18368b289c10f45a4ed8b27e857251f31300fd6"
PGPS_DATA_REVISION = "7dfd9dad4e455344b751b87340cd0ecf8798f4db"
GEOMETRY3K_URL = "https://github.com/lupantech/InterGPS"
PGPS_URL = "https://huggingface.co/datasets/PeijieWang/PGPS9K"
PGPS_TERMS_URL = "https://nlpr.ia.ac.cn/databases/CASIA-PGPS9K/index.html"
_TEST_IDS = frozenset(str(i) for i in range(2401, 3002))
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_POINT = r"[A-Z](?:[0-9]|')*"


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _envelope(record: dict, source: str, image_sha: str) -> tuple[dict, dict]:
    if type(record) is not dict or type(record.get("native")) is not dict or type(record.get("provenance")) is not dict:
        raise ValueError("geometry: native record and explicit loader provenance required")
    native, provenance = record["native"], record["provenance"]
    revision = GEOMETRY3K_REVISION if source == "geometry3k" else PGPS_DATA_REVISION
    if (provenance.get("source_revision") != revision or provenance.get("upstream_split") != "train"
            or provenance.get("label_provenance") != "HUMAN"
            or provenance.get("annotation_provenance") != "HUMAN"
            or provenance.get("model_touched") is not False
            or provenance.get("benchmark_derived") is not False
            or not isinstance(image_sha, str) or not _SHA.fullmatch(image_sha)
            or provenance.get("image_sha256") != image_sha):
        raise ValueError("geometry: pinned human train provenance and image binding required")
    pid = provenance.get("native_id")
    if not isinstance(pid, str) or not re.fullmatch(r"[A-Za-z0-9_.:@-]{1,200}", pid):
        raise ValueError("geometry: original safe JSON key/ID required")
    return native, provenance


def _numeric_text(value: object) -> str:
    if type(value) is str:
        text = value.strip()
    elif type(value) in (int, float):
        text = str(value)
    else:
        raise ValueError("geometry: native numeric annotation required")
    if number(text) is None:
        raise ValueError("geometry: unsupported or nonfinite native numeric annotation")
    return text


def _precision(provenance: dict) -> int:
    # Never infer rounding from the digits of the reference answer.
    policy = provenance.get("numeric_precision")
    if (type(policy) is not dict or set(policy) != {"round_to", "basis"}
            or type(policy["round_to"]) is not int or not 0 <= policy["round_to"] <= 12
            or policy["basis"] not in ("native_choice_precision", "explicit_question_precision")):
        raise ValueError("geometry: independent declared numeric precision required")
    return policy["round_to"]


def _answer(native: dict, provenance: dict, *, g3k: bool) -> tuple[tuple[str, ...], AnswerType, str, dict]:
    original = native.get("choices")
    if original is not None and (type(original) is not list or len(original) > 10):
        raise ValueError("geometry: unsupported native choices")
    choices: tuple[str, ...] = ()
    if original:
        if len(original) < 2:
            raise ValueError("geometry: at least two original choices required")
        if g3k:
            if any(not isinstance(v, str) or not v.strip() for v in original):
                raise ValueError("geometry: original choice strings required")
            choices = tuple(original)
            answer = native.get("answer")
            if not isinstance(answer, str) or not re.fullmatch(r"[A-J]", answer) or ord(answer) - 65 >= len(choices):
                raise ValueError("geometry: native answer letter missing or outside choices")
        else:
            numeric_choices = [_numeric_text(value) for value in original]
            choices = tuple(value if isinstance(value, str) else text
                            for value, text in zip(original, numeric_choices, strict=True))
            gold = number(_numeric_text(native.get("answer")))
            matches = [i for i, choice in enumerate(choices) if number(choice) == gold]
            if len(matches) != 1:
                raise ValueError("geometry: answer must match exactly one original numeric choice")
            answer = chr(65 + matches[0])
        if len(set(choices)) != len(choices):
            raise ValueError("geometry: duplicate native choices")
        return choices, "mcq", answer, {}
    answer = _numeric_text(native.get("answer"))
    precision = _precision(provenance)
    value = number(answer)
    assert value is not None
    # Source decimal answers only: converting an irrational expression or repeating
    # fraction into a new rounded target would invent reference supervision.
    try:
        decimal = Decimal(answer)
    except InvalidOperation as exc:
        raise ValueError("geometry: native decimal answer required") from exc
    if not decimal.is_finite():
        raise ValueError("geometry: native decimal answer required")
    return (), "decimal", answer, {"round_to": precision}


def _slot(slots: dict[str, str], kind: str, labels: list[str], value: str) -> None:
    parsed = number(value)
    if parsed is None:
        return
    if parsed < 0 or (kind == "angle" and parsed > 360):
        raise ValueError("geometry: invalid native length/angle")
    if kind == "len" and len(labels) == 2:
        labels = sorted(labels)
    elif kind == "angle" and len(labels) == 3:
        labels = min(labels, list(reversed(labels)))
    key = f"{kind}[{''.join(labels)}]"
    normalized = str(parsed)
    if key in slots and slots[key] != normalized:
        raise ValueError("geometry: conflicting native diagram annotations")
    slots[key] = normalized


def _logic_slots(logic: object) -> dict[str, str]:
    if type(logic) is not list or any(not isinstance(v, str) for v in logic):
        raise ValueError("geometry: native diagram_logic_form list required")
    slots: dict[str, str] = {}
    # Only literal numerical annotations. Symbolic expressions and text-derived
    # formulas are not evaluated or promoted into visually verified observations.
    for clause in logic:
        compact = re.sub(r"\s+", "", clause)
        match = re.fullmatch(r"Equals\((LengthOf\(Line\([^()]+\)\)|MeasureOf\(Angle\([^()]+\)\)),([^()]+)\)", compact)
        if not match:
            reverse = re.fullmatch(r"Equals\(([^()]+),(LengthOf\(Line\([^()]+\)\)|MeasureOf\(Angle\([^()]+\)\))\)", compact)
            if reverse:
                expression, value = reverse[2], reverse[1]
            else:
                continue
        else:
            expression, value = match[1], match[2]
        labels = expression[expression.index("(", expression.index("(") + 1) + 1:-2].split(",")
        kind = "len" if expression.startswith("LengthOf") else "angle"
        expected = (1, 2) if kind == "len" else (1, 3)
        if (len(labels) not in expected or any(not re.fullmatch(rf"{_POINT}|[a-z]|[0-9]+", label) for label in labels)):
            continue
        _slot(slots, kind, labels, value)
    return dict(sorted(slots.items()))


def _pgps_slots(clauses: object) -> dict[str, str]:
    if type(clauses) is not list or any(not isinstance(v, str) for v in clauses):
        raise ValueError("geometry: original parsing_sem_seqs list required")
    slots: dict[str, str] = {}
    for clause in clauses:
        # Confirmed native templates: EC = 2 and m \\angle STR = 60 (paper Fig.2).
        length = re.fullmatch(rf"\s*({_POINT})\s*({_POINT})\s*=\s*(.+?)\s*", clause)
        angle = re.fullmatch(r"\s*(?:m\s*)?(?:\\angle|∠)\s*([A-Z0-9' ]+)\s*=\s*(.+?)\s*", clause)
        if length:
            _slot(slots, "len", [length[1], length[2]], length[3])
        elif angle:
            label = re.sub(r"\s+", "", angle[1])
            labels = re.findall(_POINT, label)
            if len(labels) != 3:
                labels = [label] if re.fullmatch(r"[0-9]+|[A-Z]", label) else []
            if labels:
                _slot(slots, "angle", labels, angle[2])
    return dict(sorted(slots.items()))


def _records(source: str, question: object, native: dict, provenance: dict, slots: dict[str, str],
             split: Split, image_rel: str, image_sha: str, mode: Mode,
             parents: tuple[str, ...]) -> tuple[ActorItem, VerifierItem, Lineage]:
    choices, kind, gold, accept = _answer(native, provenance, g3k=source == "geometry3k")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("geometry: native question required")
    pid = f"{source}-rl:{provenance['native_id']}"
    actor = ActorItem(pid, source, "geometry", mode, question, choices, kind, (image_rel,), (image_sha,))
    # Native annotations do not declare a complete causally necessary slot set.
    verifier = VerifierItem(pid, kind, gold, accept, choices, slots,
                            dict.fromkeys(slots, "number"), (),
                            tuple(f"{k} = {v}" for k, v in slots.items()), ())
    revision = GEOMETRY3K_REVISION if source == "geometry3k" else PGPS_DATA_REVISION
    terms = "MIT-repository; textbook-image-rights-unreviewed" if source == "geometry3k" else "CASIA-PGPS9K-academic-agreement-required"
    lineage = Lineage(pid, source, revision, terms, "HUMAN", f"native-diagram:{source}:{image_sha}",
                      "Geometry3K-textbooks" if source == "geometry3k" else provenance["upstream_origin"],
                      None, None, "native-geometry-annotation", image_sha[:16], split,
                      (f"{source}@{revision}:train:{provenance['native_id']}", *parents))
    for item in (actor, verifier, lineage):
        to_json(item)
    return actor, verifier, lineage


def convert_geometry3k(record: dict, split: Split, image_rel: str, image_sha: str,
                       mode: Mode) -> tuple[ActorItem, VerifierItem, Lineage]:
    native, provenance = _envelope(record, "geometry3k", image_sha)
    if (native.get("data_type") != "train" or type(native.get("id")) is not int
            or not 0 <= native["id"] < 2101 or str(native["id"]) != provenance["native_id"]):
        raise ValueError("geometry: Geometry3K original train IDs/data_type only")
    diagram = record.get("diagram_annotation")
    if type(diagram) is not dict:
        raise ValueError("geometry: original logic_form.json sidecar required")
    slots = _logic_slots(diagram.get("diagram_logic_form"))
    proof = _hash({"id": native["id"], "data_type": native["data_type"],
                   "problem_text": native.get("problem_text"), "choices": native.get("choices"),
                   "answer": native.get("answer"), "diagram_logic_form": diagram.get("diagram_logic_form")})
    return _records("geometry3k", native.get("problem_text"), native, provenance, slots,
                    split, image_rel, image_sha, mode, (f"normalized-input-sha256:{proof}",))


def convert_pgps9k(record: dict, split: Split, image_rel: str, image_sha: str,
                  mode: Mode) -> tuple[ActorItem, VerifierItem, Lineage]:
    native, provenance = _envelope(record, "pgps9k", image_sha)
    if "id" in native and native["id"] != provenance["native_id"]:
        raise ValueError("geometry: native PGPS JSON key binding mismatch")
    diagram = native.get("diagram")
    if (not isinstance(diagram, str) or not diagram or "\\" in diagram
            or PurePosixPath(diagram).is_absolute() or ":" in diagram
            or any(part in ("", ".", "..") for part in diagram.split("/"))):
        raise ValueError("geometry: original safe PGPS diagram path required")
    parents = provenance.get("geometry3k_parent_ids")
    test_images = provenance.get("geometry3k_test_image_hashes")
    if (provenance.get("parentage_complete") is not True or type(parents) is not list
            or any(not isinstance(pid, str) or not re.fullmatch(r"0|[1-9][0-9]*", pid)
                   or not 0 <= int(pid) < 3002 for pid in parents) or len(set(parents)) != len(parents)
            or type(test_images) is not dict or set(test_images) != _TEST_IDS
            or any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in test_images.values())
            or provenance.get("geometry3k_revision") != GEOMETRY3K_REVISION):
        raise ValueError("geometry: complete Geometry3K parentage and 601-test image index required")
    if _TEST_IDS.intersection(parents) or image_sha in test_images.values():
        raise ValueError("geometry: Geometry3K test parent or diagram excluded")
    origin = provenance.get("upstream_origin")
    if (origin not in ("Geometry3K", "PGPS9K-textbooks")
            or (origin == "Geometry3K") != bool(parents)):
        raise ValueError("geometry: unknown or inconsistent upstream origin")
    # Also exclude Geometry3K validation parents; other benchmark evaluation splits
    # must never become training merely through a second dataset's train split.
    if any(int(pid) >= 2101 for pid in parents):
        raise ValueError("geometry: Geometry3K nontrain parent excluded")
    structural = native.get("parsing_stru_seqs")
    if type(structural) is not list or any(not isinstance(v, str) for v in structural):
        raise ValueError("geometry: original structural clause list required")
    slots = _pgps_slots(native.get("parsing_sem_seqs"))
    proof = _hash({key: native.get(key) for key in ("diagram", "text", "answer", "choices",
                                                   "parsing_sem_seqs", "parsing_stru_seqs")})
    lineage_parents = (f"normalized-input-sha256:{proof}", f"geometry3k-test-index-sha256:{_hash(test_images)}",
                       *(f"geometry3k@{GEOMETRY3K_REVISION}:train:{pid}" for pid in parents))
    return _records("pgps9k", native.get("text"), native, provenance, slots,
                    split, image_rel, image_sha, mode, lineage_parents)
