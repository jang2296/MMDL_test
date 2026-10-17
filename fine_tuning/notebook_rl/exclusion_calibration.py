"""Private, deterministic copy-detector calibration material, never training data.

Positive queries are four transformations of actual MMMU-val/Pro shared bytes.
Negative queries are newly rendered program-authored worlds, not model labels.
This bounded calibration measures copy retrieval, not semantic non-overlap proof.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import contamination as c
from . import vault
from .build_exclusion_index import _private_root, _write_once
from .dataset import generate_pool
from .schema import load_protocol

SEED = 2026101701
TRANSFORMS = ("jpeg80", "resize_half", "center80", "white_border25")


def transform(raw: bytes, name: str) -> bytes:
    if name not in TRANSFORMS:
        raise ValueError("unknown calibration transformation")
    with Image.open(io.BytesIO(raw)) as original:
        image = original.convert("RGB")
    w, h = image.size
    output = io.BytesIO()
    if name == "jpeg80":
        image.save(output, format="JPEG", quality=80, optimize=False)
        return output.getvalue()
    if name == "resize_half":
        image = image.resize((max(3, w // 2), max(3, h // 2)), Image.Resampling.LANCZOS)
    elif name == "center80":
        image = image.crop((w // 10, h // 10, w - w // 10, h - h // 10))
    elif name == "white_border25":
        image = ImageOps.expand(image, border=max(1, min(w, h) // 4), fill="white")
    else:
        raise ValueError("unknown calibration transformation")
    image.save(output, format="PNG", compress_level=6)
    return output.getvalue()


def _image(path: Path, sha: str) -> bytes:
    raw = vault._read(path)
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("calibration image hash changed")
    return raw


def prepare(index_path: Path, raw_images: Path, output: Path) -> dict:
    index = c.load_index(index_path)
    if not index["full_coverage"]:
        raise ValueError("full native benchmark index required")
    output = _private_root(output, create=True)
    vault._safe_root(output / "images", create=True)
    source_sets: dict[str, set[str]] = {"val": set(), "pro": set()}
    for record in index["data"]["records"]:
        ref = record["ref"]
        if ref["dataset_id"] == "MMMU/MMMU" and ref["split"] == "validation":
            source_sets["val"].update(record["image_sha256"])
        if ref["dataset_id"] == "MMMU/MMMU_Pro" and ref["configuration"] == "standard (10 options)":
            source_sets["pro"].update(record["image_sha256"])
    shared = sorted(source_sets["val"] & source_sets["pro"])
    if len(shared) != 632:
        raise ValueError("expected 632 actual shared native image bytes")
    images, fingerprints, positives, negatives = {}, {}, [], []

    def retain(raw, path=None):
        sha = hashlib.sha256(raw).hexdigest()
        destination = output / "images" / sha if path is None else path
        if path is None:
            _write_once(destination, raw)
        images[sha] = str(destination.absolute())
        if sha not in fingerprints:
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                fingerprints[sha] = c.image_fingerprints(image)
        return sha

    for sha in shared:
        path = raw_images / sha
        raw = _image(path, sha)
        retain(raw, path)
        for name in TRANSFORMS:
            query = retain(transform(raw, name))
            positives.append({"reference": sha, "query": query, "transform": name,
                              "role": "calibration" if int(sha[:8], 16) % 2 == 0 else "validation"})
    # No benchmark text or answer is passed to these deterministic generators.
    # Fresh seed + explicit source identities; labels are not exported at all.
    generator_sources = {}
    for name in ("graph", "chart", "music", "circuit"):
        pool = generate_pool(name, 32, SEED, "dev", mode_mix=0)
        for n, problem in enumerate(pool):
            generator_sources[name] = problem.lineage.source_revision
            sha = retain(problem.image_bytes[0])
            if sha in index["raw_hashes"] or fingerprints[sha]["pixels_sha256"] in index["pixel_hashes"]:
                raise ValueError("generated calibration negative duplicates benchmark bytes/pixels")
            negatives.append({"query": sha, "source": name,
                              "structure_hash": problem.lineage.structure_hash,
                              "role": "calibration" if n < 16 else "validation"})
    value = {"format": "notebook-rl-copy-calibration-v1", "status": "PREPARED_NOT_MEASURED",
             "purpose": "EXCLUSION_CALIBRATION_ONLY", "training_allowed": False,
             "public_upload_allowed": False, "benchmark_derived": True,
             "human_verified": False, "seed": SEED,
             "index_sha256": index["sha256"], "index_file_sha256": index["file_sha256"],
             "producer_sha256": hashlib.sha256(vault._read(Path(__file__))).hexdigest(),
             "fingerprint_producer_sha256": hashlib.sha256(vault._read(Path(c.__file__))).hexdigest(),
             "generator_sources": generator_sources, "reference_sha256": shared,
             "images": images, "fingerprints": fingerprints,
             "positives": positives, "negatives": negatives,
             "limitations": ["Program-source negatives, not independent human duplicate annotations",
                 "632-reference retrieval calibration; full 12168-image scan can have more false positives",
                 "No Pro-vision screenshot calibration or cross-modal semantic calibration in this smoke",
                 "Thresholds do not certify absence of semantic overlap or pretraining contamination"]}
    _write_once(output / "material.json", c._json(value))
    _write_once(output / "image_manifest.json", c._json(images))
    return {"material": str(output / "material.json"), "positives": len(positives),
            "negatives": len(negatives), "images": len(images), "training_allowed": False}


def threshold_for_fpr(scores: list[float], target: float = .01) -> float:
    if not scores or not 0 <= target < 1 or any(not math.isfinite(v) or not -1 <= v <= 1 for v in scores):
        raise ValueError("finite cosine calibration scores required")
    ordered = sorted(scores)
    allowed = math.floor(len(scores) * target)
    boundary = ordered[len(scores) - allowed - 1]
    threshold = max(.000001, math.nextafter(boundary, math.inf))
    if threshold > 1:
        raise ValueError("cannot meet fixed false-positive target")
    return threshold


def _rate(positive: list[float], negative: list[float], threshold: float, examples: dict) -> dict:
    if not positive or not negative:
        raise ValueError("positive and negative calibration coverage required")
    return {"positive_count": len(positive), "negative_count": len(negative),
            "tpr": sum(v >= threshold for v in positive) / len(positive),
            "fpr": sum(v >= threshold for v in negative) / len(negative),
            "examples_sha256": c._digest(examples)}


def measure(material_path: Path, sscd: dict, siglip: dict, output: Path, *, engines: dict) -> dict:
    """Vectors are actual cache exports SHA-bound to every calibration image."""
    raw = vault._read(material_path)
    data = json.loads(raw, object_pairs_hook=vault._unique)
    if (data.get("format") != "notebook-rl-copy-calibration-v1"
            or data.get("training_allowed") is not False or data.get("seed") != SEED):
        raise ValueError("unsupported calibration material")
    producer_sha = hashlib.sha256(vault._read(Path(__file__))).hexdigest()
    fingerprint_sha = hashlib.sha256(vault._read(Path(c.__file__))).hexdigest()
    if (data.get("producer_sha256") != producer_sha
            or data.get("fingerprint_producer_sha256") != fingerprint_sha):
        raise ValueError("calibration material producer/fingerprint code changed")
    if (set(engines) != {"sscd", "siglip"}
            or engines["sscd"].get("engine_id") != "sscd_disc_mixup"
            or engines["siglip"].get("engine_id") != "google/siglip-base-patch16-224"
            or any(not isinstance(v.get("artifact_sha256"), str) or len(v["artifact_sha256"]) != 64
                   for v in engines.values())):
        raise ValueError("actual pinned calibration engine identities required")
    for sha, path in data["images"].items():
        _image(Path(path), sha)
    bindings = {key: key for key in data["images"]}
    vectors = {}
    for name, values in (("sscd", sscd), ("siglip", siglip)):
        keys, values = c._vectors(values, set(bindings), bindings)
        c.cosine_similarities(values, values[:1])
        vectors[name] = dict(zip(keys, values, strict=True))
    refs = data["reference_sha256"]
    scores: dict[str, dict[str, list[float]]] = {
        name: {"positive": [], "negative": []} for name in ("dhash", "tiles", "sscd", "siglip")}
    fprints = data["fingerprints"]

    def hamming(left, right, *, tiles):
        variants = c._VARIANTS if tiles else ("full",)
        kinds = ("dhash", "phash") if tiles else ("dhash",)
        return min((int(fprints[left]["variants"][a][kind], 16) ^
                    int(fprints[right]["variants"][b][kind], 16)).bit_count()
                   for a in variants for b in variants for kind in kinds)

    for row in data["positives"]:
        q, ref = row["query"], row["reference"]
        for name in ("dhash", "tiles"):
            scores[name]["positive"].append(-hamming(q, ref, tiles=name == "tiles"))
        for name in vectors:
            scores[name]["positive"].append(float(c.cosine_similarities([vectors[name][q]], [vectors[name][ref]])[0, 0]))
    for row in data["negatives"]:
        q = row["query"]
        for name in ("dhash", "tiles"):
            scores[name]["negative"].append(-min(hamming(q, ref, tiles=name == "tiles") for ref in refs))
        for name in vectors:
            scores[name]["negative"].append(float(np.max(c.cosine_similarities([vectors[name][q]], [vectors[name][ref] for ref in refs]))))
    calibration_negatives = [s for row, s in zip(data["negatives"], scores["siglip"]["negative"], strict=True)
                             if row["role"] == "calibration"]
    thresholds = {"dhash": -3, "tiles": -3, "sscd": .75,
                  "siglip": threshold_for_fpr(calibration_negatives)}
    per_engine, validation = {}, {}
    for name, values in scores.items():
        per_engine[name] = _rate(values["positive"], values["negative"], thresholds[name],
                                 {"material_sha256": hashlib.sha256(raw).hexdigest(), "scores": values})
        subset = {kind: [s for row, s in zip(data["positives" if kind == "positive" else "negatives"], values[kind], strict=True)
                         if row["role"] == "validation"] for kind in values}
        validation[name] = _rate(subset["positive"], subset["negative"], thresholds[name], subset)
    protocol = load_protocol(Path(__file__).with_name("protocol_v1.yaml"))
    calibration = {"frozen_before_scan": True, "protocol_sha256": protocol["sha256"],
                   "index_sha256": data["index_sha256"],
                   "positive_count": len(data["positives"]), "negative_count": len(data["negatives"]),
                   "tpr": per_engine["siglip"]["tpr"], "fpr": per_engine["siglip"]["fpr"],
                   "per_engine": per_engine, "validation": validation,
                   "thresholds": {"perceptual_hamming": 3, "sscd_exclude": .95,
                                  "sscd_review": .75, "siglip_review": thresholds["siglip"]},
                   "threshold_policy": "siglip-calibration-negative-query-FPR-0.01-v1",
                   "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                   "chronology": "after read-only partial perceptual diagnosis; before full candidate scan",
                   "engines": engines,
                   "embedding_exports_sha256": {"sscd": c._digest(sscd), "siglip": c._digest(siglip)},
                   "material_sha256": hashlib.sha256(raw).hexdigest(),
                   "producer_sha256": producer_sha,
                   "fingerprint_producer_sha256": fingerprint_sha,
                   "limitations": data["limitations"], "training_allowed": False}
    output = _private_root(output, create=True)
    _write_once(output / "scores.json", c._json(scores))
    _write_once(output / "calibration.json", c._json(calibration))
    return calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--raw-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.index, args.raw_images, args.output)))


if __name__ == "__main__":
    main()
