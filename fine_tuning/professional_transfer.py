"""Private ChartQA-human source-holdout preparation; HF payloads stay on RunPod.

This evaluates real published charts, not canonical-label or fullG7 completion.
Selection reads only query/image/human provenance; labels enter the separate
gold ledger only after the selected identities have been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image

from . import audit
from .evaluate import digest
from .modules import BASE_ID, BASE_REVISION
from .sources import download_exact


LOCK = Path(__file__).with_name("chartqa_transfer_source_v1.json")
PROMPT = ('Question: {question}\nReturn exactly one JSON object with one key, '
          '"answer", whose value is a single string or number. '
          'Do not include reasoning, alternative answers or other keys.')


def training_index(specs):
    """Validate actual trained datasets, including the original frozen sources.

    Reading a historical approval here does not authorize new training with
    changed code. The trainer's current-source guard remains untouched.
    """
    if not isinstance(specs, list) or not specs:
        raise ValueError("Complete trained-dataset evidence required")
    pixels, dhashes, texts, ancestors, evidence = set(), set(), set(), set(), []
    for spec in specs:
        records_path, manifest_path, metadata_path = (
            Path(spec[key]) for key in ("records", "manifest", "checkpoint_metadata"))
        manifest = json.loads(manifest_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        if audit.sha(metadata_path) != spec["checkpoint_metadata_sha256"]:
            raise ValueError("Closed training metadata changed")
        if (manifest.get("status") != "APPROVED" or manifest.get("training_allowed") is not True
                or manifest.get("benchmark_derived") is not False
                or manifest.get("contamination_audit", {}).get("status") != "PASS"
                or metadata.get("epoch_exposure_complete") is not True
                or metadata.get("base_id") != BASE_ID or metadata.get("base_revision") != BASE_REVISION
                or metadata.get("manifest_sha256") != audit.sha(manifest_path)
                or metadata.get("records_sha256") != audit.sha(records_path)
                or manifest.get("records_sha256") != audit.sha(records_path)):
            raise ValueError("Actual clean trained-dataset closure required")
        # Ancestry is inside the checkpoint-bound manifest, not a replaceable registry.
        sources = manifest["sources"]
        for name in ("train.py", "modules.py"):
            actual = audit.sha(Path(spec["source_root"]) / name)
            if actual != metadata.get("source_sha256", {}).get(name) or actual != manifest.get("source_sha256", {}).get(name):
                raise ValueError("Original frozen training source mismatch")
        rows = audit.read_rows(records_path)
        if (not rows or type(manifest.get("record_count")) is not int
                or len(rows) != manifest.get("record_count") or len(rows) != metadata.get("planned_records")
                or len(rows) != metadata.get("exposure", {}).get("records")):
            raise ValueError("Trained record coverage mismatch")
        audit.check_splits(rows)
        known = set(manifest.get("known_parent_ids", []))
        verified, source_ancestors = set(), set()
        for row in rows:
            audit.check_lineage(row, sources)
            if row.get("split") != "train" or row.get("training_allowed") is not True or not set(row["lineage"]).issubset(known):
                raise ValueError("Invalid trained split/permission/ancestry")
            pending = list(row["lineage"])
            while pending:
                parent = pending.pop()
                if parent not in source_ancestors:
                    source_ancestors.add(parent)
                    source = sources[parent]
                    if "chartqa" in (parent + " " + str(source.get("repo_id", ""))).casefold():
                        raise ValueError("ChartQA is not held out from this training ancestry")
                    pending.extend(source["parents"])
            texts.add(audit.text_hash(row["user_text"]))
            if not isinstance(row.get("image_paths"), list) or not row["image_paths"]:
                raise ValueError("Image-based trained evidence required")
            for name in row["image_paths"]:
                if name in verified:
                    continue
                path = audit.image_path(spec["image_root"], name)
                if manifest.get("files", {}).get(name) != audit.sha(path):
                    raise ValueError("Trained image changed")
                with Image.open(path) as image:
                    fingerprint = audit.image_fingerprint(image)
                pixels.add(fingerprint["pixels_sha256"])
                dhashes.add(fingerprint["dhash"])
                verified.add(name)
        if verified != set(manifest.get("files", {})):
            raise ValueError("Trained image inventory coverage mismatch")
        ancestors.update(source_ancestors)
        evidence.append({"manifest_sha256": audit.sha(manifest_path),
            "records_sha256": audit.sha(records_path), "checkpoint_metadata_sha256": audit.sha(metadata_path),
            "records": len(rows), "images": len(verified)})
    return {"pixels": pixels, "dhashes": dhashes, "texts": texts, "ancestors": ancestors, "evidence": evidence}


def observation(row, source_index):
    """Gold-blind projection. The caller must not use labels to replace a row."""
    origin = row["human_or_machine"]
    if type(origin) is not int or origin not in (0, 1):
        raise ValueError("Pinned human/machine class-label schema required")
    if origin != 0:
        return None
    query, entry = row["query"], row["image"]
    if not isinstance(query, str) or not query.strip() or not isinstance(entry, dict):
        raise ValueError("Nonempty source question/image required")
    raw = entry.get("bytes")
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("Embedded source image bytes required; no path/download fallback")
    with Image.open(io.BytesIO(raw)) as image:
        fingerprint = audit.image_fingerprint(image)
    identity = digest([query, fingerprint["pixels_sha256"]])
    return {"id": "chartqa-human-" + identity, "query": query, "source_index": source_index,
        "family_id": "chartqa-chart-" + fingerprint["pixels_sha256"],
        "source_bytes_sha256": hashlib.sha256(raw).hexdigest(), **fingerprint}


def select(observations, index, count=60, seed=3407):
    if type(count) is not int or count <= 0 or type(seed) is not int:
        raise ValueError("Positive count and fixed integer seed required")
    if len({row["id"] for row in observations}) != len(observations):
        raise ValueError("Duplicate source query/image identities")
    rejected, families = [], {}
    for row in sorted(observations, key=lambda item: digest([seed, "chartqa-human-v1", item["id"]])):
        reason = "TRAIN_QUESTION_MATCH" if audit.text_hash(row["query"]) in index["texts"] else None
        if row["pixels_sha256"] in index["pixels"]:
            reason = "TRAIN_IMAGE_PIXEL_MATCH"
        elif any((int(row["dhash"], 16) ^ int(value, 16)).bit_count() <= 3 for value in index["dhashes"]):
            reason = "TRAIN_IMAGE_NEAR_MATCH"
        if reason:
            rejected.append({"id": row["id"], "reason": reason})
        elif row["family_id"] not in families:
            families[row["family_id"]] = row
    chosen = list(families.values())[:count]
    if len(chosen) != count:
        raise ValueError("Insufficient disjoint human chart families; do not shrink or replace by scores")
    return chosen, rejected


def freeze(rows, index, output, source_receipt, count=60, seed=3407):
    """Pure offline builder also exercised with independently authored fixtures."""
    observations = [value for number, row in enumerate(rows) if (value := observation(row, number)) is not None]
    chosen, rejected = select(observations, index, count, seed)
    # Gold is inspected only for these preselected rows; invalid gold aborts.
    labels = {}
    for item in chosen:
        answers = rows[item["source_index"]]["label"]
        if not isinstance(answers, list) or not answers or any(not isinstance(value, str) or not value.strip() for value in answers):
            raise ValueError("Selected source gold invalid; no score-based replacement")
        labels[item["id"]] = answers
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    inputs, gold, images = [], [], {}
    for item in chosen:
        name = "images/" + item["pixels_sha256"] + ".png"
        with Image.open(io.BytesIO(rows[item["source_index"]]["image"]["bytes"])) as image:
            rgb = image.convert("RGB")
            rgb.save(out / name)
            rgb.close()
        images[name] = audit.sha(out / name)
        inputs.append({"id": item["id"], "subject": "ChartQA_Human", "question_type": "open",
            "question": item["query"], "options": [], "user_text": PROMPT.format(question=item["query"]),
            "image_paths": [name], "benchmark_derived": False, "training_allowed": False,
            "public_upload_allowed": False})
        gold.append({"id": item["id"], "family_id": item["family_id"], "root_id": item["id"],
            "metric_kind": "chartqa_human_exact", "target": json.dumps({"answers": labels[item["id"]]}),
            "training_allowed": False, "public_upload_allowed": False})
    audit.write_rows(out / "inputs.jsonl", inputs)
    audit.write_rows(out / "gold.jsonl", gold)
    audit.write_rows(out / "selection.jsonl", chosen)
    audit.write_rows(out / "excluded.jsonl", rejected)
    manifest = {"status": "COHORT_FROZEN", "count": count, "ids": [item["id"] for item in chosen],
        "inputs_sha256": audit.sha(out / "inputs.jsonl"), "gold_sha256": audit.sha(out / "gold.jsonl"),
        "selection_sha256": audit.sha(out / "selection.jsonl"), "excluded_sha256": audit.sha(out / "excluded.jsonl"),
        "images": images, "prompt_sha256": digest(PROMPT), "seed": seed,
        "source": source_receipt, "training_evidence": index["evidence"],
        "held_out_training_ancestors": sorted(index["ancestors"]),
        "evaluation_domain": "PUBLISHED_CHARTQA_HUMAN_SOURCE_TRANSFER",
        "selection_policy": "Gold-blind SHA order; one human question per exact RGB chart family; train question/pixel/dHash<=3 excluded",
        "metric_policy": "single terminal JSON scalar; casefold/whitespace exact or finite Decimal exact; no5percent/unit/substring rescue",
        "limits": "No publisher/latent-document/semantic/crop duplicate certification; no unseen-pretraining claim; not fullG7 or official ChartQA leaderboard",
        "builder_source_sha256": audit.sha(__file__), "training_allowed": False,
        "public_upload_allowed": False, "benchmark_derived": False}
    audit.dump(out / "manifest.json", manifest)
    return manifest


def prepare(task_root, training_evidence, output, count=60, seed=3407):
    if not os.environ.get("RUNPOD_POD_ID"):
        raise ValueError("HF payload acquisition and professional source preparation are RunPod-only")
    if Path(output).exists():
        raise ValueError("Cohort output already exists; no download or overwrite")
    lock = json.loads(LOCK.read_text())
    index = training_index(json.loads(Path(training_evidence).read_text())["datasets"])
    root = Path(task_root)
    receipt_path = root / "chartqa-human-downloads.json"
    paths = download_exact(lock["repo_id"], lock["revision"], [lock["file"]["path"]],
                           root / "hf-cache" / "hub", receipt_path)
    path = paths[0]
    if path.stat().st_size != lock["file"]["bytes"] or audit.sha(path) != lock["file"]["sha256"]:
        raise ValueError("Frozen ChartQA source file differs")
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    if len(rows) != lock["released_split_records"]:
        raise ValueError("Frozen ChartQA test record coverage differs")
    return freeze(rows, index, output, {"lock_sha256": audit.sha(LOCK), "download_receipt_sha256": audit.sha(receipt_path), **lock}, count, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("task-root", "training-evidence", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    result = prepare(args.task_root, args.training_evidence, args.output, args.count, args.seed)
    print(json.dumps({"status": result["status"], "count": result["count"], "GPU_evaluated": False}))


if __name__ == "__main__":
    main()
