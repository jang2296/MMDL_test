"""AI similarity review tooling for the HF-pool exclusion scan (sample -> sheets -> compile -> check).

Works only from the answer-free review packet written by hf_pool_exclusion.assemble:
  sample   label-blind, hash-ordered per-source item sample (the packet carries no gold)
  sheets   one contact sheet (PNG) + text sidecar (TXT) per sampled item chunk of <= 12 benchmark records
  compile  reviewers' per-pair verdict files -> adjudication decisions (+ summary); unsampled items are
           excluded conservatively as UNRESOLVED (family quarantine)
  check    dry-run adjudication.apply on the assembled fresh root; writes nothing into build-reviewed
The scanner, its thresholds and adjudication.py are untouched. AI review stays human_verified=false and
nothing here grants training permission.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import string
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import adjudication
from .hf_pool_exclusion import _desc, _sha_file, _snapshot

SAMPLE_VERSION = "hf-pool-review-sample-v1"
PACKET_FORMAT = "hf-pool-exclusion-review-packet-v1"
VERDICTS = ("DISTINCT", "DUPLICATE", "UNRESOLVED")
_PRIORITY = ("DISTINCT", "UNRESOLVED", "DUPLICATE")  # weakest -> strongest
# Unreviewed decisions cite the small sample file (it binds packet_sha256 and lists every reviewed item):
# adjudication re-hashes each decision's evidence, and the 39 MB packet x ~30k decisions costs ~17 min per derive.
NOT_REVIEWED = "NOT_REVIEWED_IN_PASS: excluded conservatively (family quarantine)"
NO_RECORD = "NO_BENCHMARK_RECORD_LINKED: nothing to compare, excluded conservatively"
MAX_PER_SHEET = 12
_SPLIT_RANK = {"train": 0, "dev": 0, "sealed": 1, "style_sealed": 1}
_W, _PAD, _LABEL_H, _POOL_BOX, _BENCH_BOX = 1320, 10, 30, 640, 320
_BENCH_TEXT_MAX, _REASON_MAX = 600, 1500


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=1).encode() + b"\n"


def _write_new(path: Path, data: bytes) -> dict:
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise ValueError(f"output {path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _desc(path)


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {type(exc).__name__}") from exc
    if type(value) is not dict:
        raise ValueError(f"{path}: JSON object required")
    return value


def load_packet(path: Path) -> tuple[dict, str]:
    packet = _load_json(path)
    if (packet.get("format") != PACKET_FORMAT or type(packet.get("candidates")) is not list
            or type(packet.get("items")) is not dict or type(packet.get("benchmark_records")) is not dict
            or type(packet.get("review_template")) is not dict):
        raise ValueError("review packet: unexpected format")
    return packet, _sha_file(Path(path))


def _item_candidates(packet: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in packet["candidates"]:
        if row["problem_id"] not in packet["items"]:
            raise ValueError(f"review packet: candidate item {row['problem_id']} missing from items")
        out[row["problem_id"]].append(row)
    return out


# ---------------------------------------------------------------- sample

def make_sample(packet: dict, packet_sha256: str, per_source: int, seed: int) -> dict:
    """Label-blind: only problem_id, source and split are read. Train/dev first, then sealed variants."""
    if per_source < 1:
        raise ValueError("sample: --per-source must be >= 1")
    by_source: dict[str, list[str]] = defaultdict(list)
    for pid in _item_candidates(packet):
        by_source[packet["items"][pid]["source"]].append(pid)
    chosen: dict[str, list[str]] = {}
    for source, pids in sorted(by_source.items()):
        pids.sort(key=lambda p: (_SPLIT_RANK.get(packet["items"][p]["split"], 2),
                                 hashlib.sha256(f"{seed}:{p}".encode()).hexdigest()))
        chosen[source] = pids[:per_source]
    return {"version": SAMPLE_VERSION, "packet_sha256": packet_sha256, "seed": seed, "per_source": per_source,
            "problem_ids": [p for pids in chosen.values() for p in pids],
            "counts": {s: {"flagged": len(by_source[s]), "sampled": len(chosen[s])} for s in chosen}}


def _load_sample(path: Path, packet: dict, packet_sha256: str) -> dict:
    sample = _load_json(path)
    ids = sample.get("problem_ids")
    if (sample.get("version") != SAMPLE_VERSION or sample.get("packet_sha256") != packet_sha256
            or type(ids) is not list or len(set(ids)) != len(ids) or any(p not in packet["items"] for p in ids)):
        raise ValueError("sample: does not belong to this packet")
    return sample


# ---------------------------------------------------------------- sheets

def _links(rows: list[dict]) -> dict[str, list[str]]:
    """benchmark record key -> sorted unique descriptions of the detector hits that linked it."""
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cand = row["candidate"]
        ref = cand.get("ref")
        parts = [f"{row['stage']}/{row['reason']}"]
        if "hamming" in cand:
            parts.append(f"hamming={cand['hamming']}")
        if "cosine" in cand:
            parts.append(f"cosine={cand['cosine']:.4f}")
        if "field" in cand:
            parts.append(f"field={cand['field']}")
        if isinstance(ref, dict) and "low_information" in ref:
            parts.append(f"low_information={ref['low_information']}")
        for key in row["benchmark_records"]:
            out[key].add(" ".join(parts))
    return {key: sorted(v) for key, v in out.items()}


@lru_cache(maxsize=256)
def _thumb(path: str, box: int) -> Image.Image:
    if not Path(path).is_file():
        raise ValueError(f"sheets: missing image {path}")
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    rgb = Image.new("RGB", rgba.size, "white")  # transparent backgrounds show white, not palette black
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    rgb.thumbnail((box, box), Image.Resampling.LANCZOS)  # shrink only, never upscale
    return rgb


def _tiles(paths: list[str], box: int) -> list[Image.Image]:
    if not paths:
        raise ValueError("sheets: record without images")
    box = min(box, (_W - 2 * _PAD - (len(paths) - 1) * _PAD) // len(paths))
    return [_thumb(p, box) for p in paths]


def _cell(label: str, paths: list[str], box: int) -> dict:
    tiles = _tiles(paths, box)
    return {"label": label, "tiles": tiles, "w": sum(t.width for t in tiles) + _PAD * (len(tiles) - 1),
            "h": _LABEL_H + max(t.height for t in tiles)}


def render_sheet(header: str, pool_paths: list[str], records: list[tuple[str, list[str]]]) -> Image.Image:
    """Pool image(s) top-left (<=640 box), then B-label cells (<=320 box) wrapped into rows."""
    font = ImageFont.load_default(size=20)
    pool = _cell(header, pool_paths, _POOL_BOX)
    placed: list[tuple[int, int, dict]] = [(_PAD, _PAD, pool)]
    y, x, row_h = _PAD + pool["h"] + _PAD, _PAD, 0
    for label, paths in records:
        cell = _cell(label, paths, _BENCH_BOX)
        if x > _PAD and x + cell["w"] > _W - _PAD:
            y, x, row_h = y + row_h + _PAD, _PAD, 0
        placed.append((x, y, cell))
        x, row_h = x + cell["w"] + _PAD, max(row_h, cell["h"])
    canvas = Image.new("RGB", (_W, y + row_h + _PAD), "white")
    draw = ImageDraw.Draw(canvas)
    for px, py, cell in placed:
        draw.text((px, py), cell["label"], fill="black", font=font)
        draw.rectangle((px - 1, py + _LABEL_H - 1, px + cell["w"], py + cell["h"]), outline="gray")
        tx = px
        for tile in cell["tiles"]:
            canvas.paste(tile, (tx, py + _LABEL_H))
            tx += tile.width + _PAD
    return canvas


def _choices(choices: list[str]) -> list[str]:
    letters = string.ascii_uppercase
    return [f"  ({letters[i] if i < len(letters) else i}) {text}" for i, text in enumerate(choices)]


def _sheet_text(pid: str, item: dict, labelled: list[tuple[str, str]], records: dict, links: dict) -> str:
    lines = [f"POOL ITEM {pid}  source={item['source']} split={item['split']}",
             f"Question: {item['question']}", "Choices:", *_choices(item["choices"])]
    for label, key in labelled:
        rec = records[key]
        question = " ".join(q.strip() for q in rec["question"])[:_BENCH_TEXT_MAX]
        lines += ["", f"--- {label} {key} ---", "Linked by:", *(f"  {x}" for x in links[key]),
                  f"Question: {question}", "Choices:", *_choices(rec["choices"])]
    return "\n".join(lines) + "\n"


def make_sheets(packet: dict, sample: dict, output: Path) -> dict:
    output = Path(output).absolute()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output {output} already exists; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)
    by_item = _item_candidates(packet)
    index: dict[str, dict] = {}
    for n, pid in enumerate(sample["problem_ids"], 1):
        item, links = packet["items"][pid], _links(by_item[pid])
        keys = sorted(links)
        chunks = [keys[i:i + MAX_PER_SHEET] for i in range(0, len(keys), MAX_PER_SHEET)] or [[]]
        if len(chunks) > len(string.ascii_lowercase):
            raise ValueError(f"sheets: {pid} links {len(keys)} records; too many sheets")
        for c_i, chunk in enumerate(chunks):
            stem = f"{n:04d}-{string.ascii_lowercase[c_i]}"
            labelled = [(f"B{c_i * MAX_PER_SHEET + j + 1}", key) for j, key in enumerate(chunk)]
            header = f"POOL {pid} [{item['source']}/{item['split']}] sheet {stem}"
            image = render_sheet(header, item["image_paths"],
                                 [(label, packet["benchmark_records"][key]["image_paths"]) for label, key in labelled])
            png = output / f"{stem}.png"
            image.save(png)
            txt = output / f"{stem}.txt"
            txt.write_text(_sheet_text(pid, item, labelled, packet["benchmark_records"], links), encoding="utf-8")
            index[stem] = {"problem_id": pid, "png": png.name, "png_sha256": _sha_file(png),
                           "txt": txt.name, "txt_sha256": _sha_file(txt), "labels": dict(labelled)}
    _write_new(output / "index.json", _json_bytes(index))
    return {"items": len(sample["problem_ids"]), "sheets": len(index),
            "pairs": sum(len(s["labels"]) for s in index.values())}


# ---------------------------------------------------------------- compile

def _stamp(value: str) -> str:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("compile: --reviewed-at-utc must be an ISO UTC timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0) or not value.endswith("Z"):
        raise ValueError("compile: --reviewed-at-utc must be UTC ending in Z")
    return value


def _sheet_evidence(index: dict, index_dir: Path) -> dict[str, list[dict]]:
    """problem_id -> png/txt descriptors, each re-hashed against the index."""
    out: dict[str, list[dict]] = defaultdict(list)
    for stem, meta in index.items():
        for kind in ("png", "txt"):
            path = index_dir / meta[kind]
            if not path.is_file() or _sha_file(path) != meta[f"{kind}_sha256"]:
                raise ValueError(f"compile: sheet {stem} {kind} missing or changed since rendering")
            out[meta["problem_id"]].append(_desc(path))
    return out


def _read_verdicts(verdict_dir: Path, sample: dict, index: dict) -> tuple[dict, dict]:
    """(pid, record key) -> {verdict, reason, label}; pid -> verdict-file descriptors."""
    pairs: dict[tuple[str, str], dict] = {}
    files: dict[str, dict[str, dict]] = defaultdict(dict)
    paths = sorted(Path(verdict_dir).glob("*.json"))
    if not paths:
        raise ValueError(f"compile: no verdict files in {verdict_dir}")
    for path in paths:
        doc = _load_json(path)
        pid, stems, rows = doc.get("problem_id"), doc.get("sheets"), doc.get("pairs")
        if not isinstance(pid, str) or pid not in sample["problem_ids"] or type(stems) is not list or type(rows) is not list:
            raise ValueError(f"compile: {path.name}: sampled problem_id, sheets and pairs required")
        labels: dict[str, str] = {}
        for stem in stems:
            if stem not in index or index[stem]["problem_id"] != pid:
                raise ValueError(f"compile: {path.name}: unknown sheet {stem!r} for {pid}")
            labels.update(index[stem]["labels"])
        for row in rows:
            label, key, verdict, reason = (row.get(k) if type(row) is dict else None
                                           for k in ("label", "record", "verdict", "reason"))
            if not isinstance(label, str) or label not in labels or labels[label] != key:
                raise ValueError(f"compile: {path.name}: unknown label/record {label!r} -> {key!r}")
            if verdict not in VERDICTS:
                raise ValueError(f"compile: {path.name}: illegal verdict {verdict!r} for {label}")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"compile: {path.name}: non-empty reason required for {label}")
            if (pid, key) in pairs:
                raise ValueError(f"compile: {path.name}: second verdict for {label} of {pid}")
            pairs[(pid, key)] = {"verdict": verdict, "reason": reason.strip(), "label": label}
        files[pid][str(path)] = _desc(path)
    return pairs, {pid: list(v.values()) for pid, v in files.items()}


def _decide(rows: list[tuple[str, str, str]]) -> tuple[str, str]:
    """rows = (label, verdict, reason) per linked record -> DUPLICATE > UNRESOLVED > DISTINCT."""
    verdict = max((v for _, v, _ in rows), key=_PRIORITY.index)
    return verdict, " | ".join(f"{label}={v}: {reason}" for label, v, reason in rows)[:_REASON_MAX]


def compile_decisions(packet: dict, sample_desc: dict, sample: dict, index: dict, index_dir: Path,
                      verdict_dir: Path, reviewer_id: str, reviewed_at_utc: str) -> tuple[dict, dict]:
    if not reviewer_id.strip():
        raise ValueError("compile: --reviewer-id required")
    reviewed_at_utc = _stamp(reviewed_at_utc)
    sampled = set(sample["problem_ids"])
    by_item = _item_candidates(packet)
    for row in packet["candidates"]:
        if adjudication.candidate_id(row["candidate"]) != row["candidate_id"]:
            raise ValueError(f"compile: packet candidate_id mismatch for {row['candidate_id']}")
    evidence = _sheet_evidence(index, index_dir)
    pairs, verdict_files = _read_verdicts(verdict_dir, sample, index)
    for pid in sampled:
        want = {key for row in by_item[pid] for key in row["benchmark_records"]}
        have = {key for p, key in pairs if p == pid}
        if pid not in evidence:
            raise ValueError(f"compile: no sheets rendered for sampled item {pid}")
        mapped = {key for s in index.values() if s["problem_id"] == pid for key in s["labels"].values()}
        if mapped != want:
            raise ValueError(f"compile: sheet index for {pid} does not match the packet's linked records")
        if have != want:
            raise ValueError(f"compile: {pid}: {len(want - have)} linked records without a verdict "
                             f"(first {sorted(want - have)[:3]})")
    decisions, seen = [], set()
    for row in packet["candidates"]:
        cid, pid = row["candidate_id"], row["problem_id"]
        if cid in seen:
            continue
        seen.add(cid)
        if pid not in sampled:
            decisions.append({"candidate_id": cid, "verdict": "UNRESOLVED", "reason": NOT_REVIEWED,
                              "evidence": [sample_desc]})
        elif not row["benchmark_records"]:
            decisions.append({"candidate_id": cid, "verdict": "UNRESOLVED", "reason": NO_RECORD,
                              "evidence": [*evidence[pid], *verdict_files.get(pid, [])]})
        else:
            verdict, reason = _decide([(pairs[(pid, k)]["label"], pairs[(pid, k)]["verdict"], pairs[(pid, k)]["reason"])
                                       for k in row["benchmark_records"]])
            decisions.append({"candidate_id": cid, "verdict": verdict, "reason": reason,
                              "evidence": [*evidence[pid], *verdict_files[pid]]})
    review = copy.deepcopy(packet["review_template"])
    review["provenance"] = {"kind": "AI_DIRECT_SIMILARITY_REVIEW", "human_verified": False,
                            "reviewer_id": reviewer_id.strip(), "reviewed_at_utc": reviewed_at_utc}
    review["decisions"] = decisions
    return review, _summary(packet, sampled, decisions, pairs)


def _summary(packet: dict, sampled: set[str], decisions: list[dict], pairs: dict) -> dict:
    verdict = {d["candidate_id"]: d["verdict"] for d in decisions}
    unique = {row["candidate_id"]: row for row in packet["candidates"]}
    per_item: dict[str, str] = {}
    for cid, row in unique.items():
        if row["problem_id"] in sampled:
            per_item[row["problem_id"]] = max(per_item.get(row["problem_id"], "DISTINCT"), verdict[cid],
                                              key=_PRIORITY.index)
    per_source: dict[str, Counter] = defaultdict(Counter)
    for pid, v in per_item.items():
        per_source[packet["items"][pid]["source"]][v] += 1
    reasons: dict[str, Counter] = defaultdict(Counter)
    for cid, row in unique.items():
        if row["problem_id"] in sampled:
            reasons[f"{row['stage']}/{row['reason']}"][verdict[cid]] += 1
    return {"candidates": dict(Counter(verdict.values())),
            "reviewed_candidates": sum(r["problem_id"] in sampled for r in unique.values()),
            "reviewed_pairs": dict(Counter(p["verdict"] for p in pairs.values())),
            "per_source_items": {s: {"reviewed": sum(c.values()), **{v: c[v] for v in VERDICTS}}
                                 for s, c in sorted(per_source.items())},
            "per_reason": {r: {"candidates": sum(c.values()), **{v: c[v] for v in VERDICTS},
                               "duplicate_rate": round(c["DUPLICATE"] / sum(c.values()), 4)}
                           for r, c in sorted(reasons.items())}}


# ---------------------------------------------------------------- check

def check_decisions(decisions: Path, assemble_dir: Path) -> dict:
    """Dry run of hf_pool_exclusion.run_apply: nothing is written anywhere."""
    assemble_dir = Path(assemble_dir).absolute()
    items, files, _ = _snapshot(assemble_dir / "build-reviewed")
    reviewed = adjudication.apply(_desc(assemble_dir / "raw-fullscan/receipt_v2.json"), _desc(decisions), items)
    adjudication.validate(reviewed, items)
    if reviewed["input_files"] != files:
        raise ValueError("check: reviewed receipt is not a complete binding of the fresh root")
    adj, allowed = reviewed["adjudication"], set(reviewed["allowlist"])
    per_source: dict[str, dict] = defaultdict(lambda: {"items": 0, "allowlisted": 0})
    for pid, item in items.items():
        row = per_source[item["actor"].source]
        row["items"] += 1
        row["allowlisted"] += pid in allowed
    return {"status": "CHECK_PASSED_DRY_RUN", "counts": adj["counts"], "allowlist": len(allowed),
            "items": len(items), "blacklisted_problem_ids": len(adj["blacklisted_problem_ids"]),
            "unresolved_problem_ids": len(adj["unresolved_problem_ids"]),
            "released_candidates": len(adj["released_candidate_ids"]),
            "per_source": dict(sorted(per_source.items()))}


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fine_tuning.notebook_rl.hf_pool_review", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--packet", type=Path, required=True)
    s.add_argument("--per-source", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--output", type=Path, required=True)
    h = sub.add_parser("sheets")
    for name in ("packet", "sample", "output"):
        h.add_argument(f"--{name}", type=Path, required=True)
    c = sub.add_parser("compile")
    for name in ("packet", "sample", "sheets", "verdicts", "output"):
        c.add_argument(f"--{name}", type=Path, required=True)
    c.add_argument("--reviewer-id", required=True)
    c.add_argument("--reviewed-at-utc", required=True)
    k = sub.add_parser("check")
    k.add_argument("--decisions", type=Path, required=True)
    k.add_argument("--assemble-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "sample":
            packet, digest = load_packet(args.packet)
            sample = make_sample(packet, digest, args.per_source, args.seed)
            _write_new(args.output / "sample.json", _json_bytes(sample))
            result = {"status": "SAMPLE_WRITTEN", "items": len(sample["problem_ids"]), "counts": sample["counts"]}
        elif args.command == "sheets":
            packet, digest = load_packet(args.packet)
            result = {"status": "SHEETS_WRITTEN", **make_sheets(packet, _load_sample(args.sample, packet, digest), args.output)}
        elif args.command == "compile":
            packet, digest = load_packet(args.packet)
            sample = _load_sample(args.sample, packet, digest)
            review, summary = compile_decisions(packet, _desc(args.sample), sample, _load_json(args.sheets),
                                                args.sheets.absolute().parent, args.verdicts, args.reviewer_id,
                                                args.reviewed_at_utc)
            _write_new(args.output, _json_bytes(review))
            _write_new(Path(f"{args.output}.summary.json"), _json_bytes(summary))
            result = {"status": "DECISIONS_WRITTEN", "decisions": len(review["decisions"]),
                      "candidates": summary["candidates"], "reviewed_candidates": summary["reviewed_candidates"],
                      "summary": f"{args.output}.summary.json"}
        else:
            result = check_decisions(args.decisions, args.assemble_dir)
    except ValueError as exc:
        print(json.dumps({"status": "REFUSED", "stage": args.command, "error": str(exc)}), flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
