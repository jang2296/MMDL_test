"""Audit the real V8 ledger against pinned Parquet; separate inference and gold."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import shutil
from pathlib import Path

MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATA_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
STRESS = [f"validation_{s}_{n}" for s, n in [
    ("Music", 3), ("Music", 15), ("Music", 25), ("Music", 26),
    ("Computer_Science", 2), ("Computer_Science", 19), ("Accounting", 1),
    ("Accounting", 19), ("Biology", 6), ("Materials", 6),
    ("Physics", 27), ("Public_Health", 14),
]]
PRIMARY_SUBJECTS = {"Music", "Electronics", "Materials", "Computer_Science", "Biology", "Accounting"}
CONTROLS = {"Physics", "Public_Health", "Diagnostics_and_Laboratory_Medicine"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_model_snapshot(snapshot, manifest):
    """Verify real local bytes, not only a revision-shaped directory name."""
    root = Path(snapshot).resolve(strict=True)
    model = json.loads(Path(manifest).read_text())
    if (model["repo_id"], model["revision"], model["dtype"], model["processor_revision"]) != (
            "Qwen/Qwen3-VL-4B-Instruct", MODEL_REVISION, "bfloat16", MODEL_REVISION):
        raise ValueError("Pinned model manifest mismatch")
    for item in model["files"]:
        name = Path(item["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Unsafe model manifest path")
        path = root / name
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"Actual model snapshot bytes differ: {name}")
    return digest({"repo_id": model["repo_id"], "revision": model["revision"], "files": model["files"]})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def select_samples(rows, seed=3407):
    """Outcome-stratified hash ranks; control pilot picks are nested in expansion."""
    by_subject = {}
    for row in rows:
        by_subject.setdefault(row["subject"], {True: [], False: []})[row["correct"]].append(row["id"])
    if len(by_subject) != 30 or len({r["id"] for r in rows}) != 900:
        raise ValueError("Sampling requires all 900 unique validation items")
    expansion, pilot, strata = [], [], []
    for subject, groups in sorted(by_subject.items()):
        for correct, ids in groups.items():
            if len(ids) < 4:
                raise ValueError(f"Insufficient stratum: {subject}/{correct}")
            ranked = sorted(ids, key=lambda sid: (digest([seed, "sample-v1", sid]), sid))
            chosen = ranked[:4]
            expansion.extend(chosen)
            count = 4 if subject in PRIMARY_SUBJECTS else 2 if subject in CONTROLS else 0
            pilot.extend(chosen[:count])
            strata.extend({"sample_id": sid, "subject": subject, "historical_correct": correct,
                           "rank": i + 1, "population_size": len(ids),
                           "inclusion_probability_expansion": 1.0 if subject == "Music" else 4 / len(ids),
                           "inclusion_probability_pilot": count / len(ids),
                           "in_pilot": sid in chosen[:count]} for i, sid in enumerate(ranked))
    expansion.extend(r["id"] for r in rows if r["subject"] == "Music" and r["id"] not in expansion)
    if len(pilot) != 60 or len(expansion) != 262 or not set(pilot) <= set(expansion):
        raise ValueError("Nested selection counts differ from the plan")
    if not set(STRESS) <= {r["id"] for r in rows}:
        raise ValueError("Missing fixed stress case")
    return {"seed": seed, "policy": "SHA256 canonical [seed,sample-v1,id], within subject/old correctness",
            "stress_ids": STRESS, "pilot_ids": sorted(pilot), "expansion_ids": sorted(expansion),
            "strata": strata, "warning": "Balanced diagnostic samples are NOT MMMU population accuracy"}


def image_role(field, question, options, hint, explanation):
    marker = rf"<image\s+{int(field.split('_')[1])}\s*>"
    q = bool(re.search(marker, question + "\n" + (hint or "")))
    o = any(re.search(marker, text) for text in options)
    if q and o:
        return "question_and_option"
    if q:
        return "question"
    if o:
        return "option"
    return "gold_explanation" if re.search(marker, explanation or "") else "unreferenced"


def file_snapshot(root, exclude_git=True):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not (exclude_git and ".git" in p.relative_to(root).parts)}


def ingest(source_dir, data_root, output_dir, model_manifest, public_root=None):
    import pyarrow.parquet as pq  # type: ignore[import-untyped]  # pyarrow ships no typing stubs
    from PIL import Image

    source_dir, data_root, output_dir = map(Path, (source_dir, data_root, output_dir))
    if output_dir.exists():
        raise FileExistsError("Use a new preparation directory; never overwrite an existing audit")
    model = json.loads(Path(model_manifest).read_text())
    if (model.get("repo_id"), model.get("revision"), model.get("dtype"), model.get("processor_revision")) != (
            "Qwen/Qwen3-VL-4B-Instruct", MODEL_REVISION, "bfloat16", MODEL_REVISION):
        raise ValueError("Model manifest mismatch; refusing automatic replacement")
    dataset = json.loads((data_root / "manifest.json").read_text())
    if (dataset.get("repo_id"), dataset.get("revision")) != ("MMMU/MMMU", DATA_REVISION):
        raise ValueError("Dataset revision mismatch; refusing automatic replacement")
    rows = read_rows(source_dir / "predictions.jsonl")
    records = {r["id"]: r for r in rows}
    if len(rows) != len(records) or len(rows) != 900 or sum(r["correct"] for r in rows) != 558:
        raise ValueError("Expected immutable V8 baseline: 900 unique rows, 558 correct")
    origin = json.loads((source_dir / "source.json").read_text())
    source_run = Path(origin["source_run"])
    if not source_run.is_dir() or sha(source_run / "predictions.jsonl") != origin["source_predictions_sha256"]:
        raise ValueError("Original inference ledger missing or hash mismatch")
    original = {r["id"]: r for r in read_rows(source_run / "predictions.jsonl")}
    if set(original) != set(records) or any(original[sid]["raw_response"] != r["raw_response"]
                                           for sid, r in records.items()):
        raise ValueError("Stored V8 response differs from original inference")
    output_dir.mkdir(parents=True)
    (output_dir / "assets/images").mkdir(parents=True)
    files = {f["path"]: f for f in dataset["files"]}
    inputs, gold, audits, queue = [], [], [], []
    pixel_verified = 0
    for subject in sorted({r["subject"] for r in rows}):
        relative = f"{subject}/validation-00000-of-00001.parquet"
        path = data_root / relative
        if path.stat().st_size != files[relative]["bytes"] or sha(path) != files[relative]["sha256"]:
            raise ValueError(f"Parquet hash mismatch: {subject}")
        source_rows = pq.read_table(path).to_pylist()
        if len(source_rows) != 30:
            raise ValueError(f"Wrong subject count: {subject}")
        for item in source_rows:
            sid = item["id"]
            old = records[sid]
            options = ast.literal_eval(item["options"]) if isinstance(item["options"], str) else item["options"]
            for key in ("question", "question_type", "answer"):
                if item[key] != old[key]:
                    raise ValueError(f"Source mismatch: {sid}/{key}")
            if options != old["options"]:
                raise ValueError(f"Option order mismatch: {sid}")
            hint = item.get("hint") if isinstance(item.get("hint"), str) else None
            refs = old["image_references"]
            if len(refs) != len(old["images"]):
                raise ValueError(f"Historical image mapping mismatch: {sid}")
            images, excluded, all_hashes = [], [], []
            for ref, relative_image in zip(refs, old["images"], strict=True):
                image_path = (source_run / relative_image).resolve()
                if not image_path.is_relative_to(source_run.parents[1].resolve()):
                    raise ValueError("Image path escapes recovered artifacts")
                image_hash = sha(image_path)
                if image_path.stem != image_hash:
                    raise ValueError(f"Historical image content hash mismatch: {sid}")
                field = ref["field"]
                raw = item[field]
                if not raw or not raw.get("bytes"):
                    raise ValueError(f"Source image bytes unavailable: {sid}/{field}")
                with Image.open(io.BytesIO(raw["bytes"])) as image, Image.open(image_path) as recovered:
                    rgb = image.convert("RGB")
                    if rgb.size != recovered.size or rgb.tobytes() != recovered.convert("RGB").tobytes():
                        raise ValueError(f"Recovered image differs from pinned source pixels: {sid}/{field}")
                    pixel_verified += 1
                role = image_role(field, item["question"], options, hint, item.get("explanation"))
                all_hashes.append(image_hash)
                metadata = {"field": field, "marker": ref["marker"], "role": role,
                            "order": ref["content_index"], "path": f"assets/images/{image_hash}.png",
                            "sha256": image_hash, "source_bytes_sha256": hashlib.sha256(raw["bytes"]).hexdigest(),
                            "original_size": ref["original_size"]}
                if role in {"gold_explanation", "unreferenced"}:
                    excluded.append(metadata)
                    continue
                target = output_dir / metadata["path"]
                if not target.exists():
                    shutil.copyfile(image_path, target)
                images.append(metadata)
            # Record legacy inputs without copying excluded assets into new model input storage.
            tags = ast.literal_eval(item["img_type"]) if isinstance(item.get("img_type"), str) else item.get("img_type", [])
            sample = {"sample_id": sid, "subject": subject, "question": item["question"],
                      "question_type": item["question_type"], "options": options, "hint": hint,
                      "images": images, "family_id": None, "skill_tags": tags,
                      "provenance": "BENCHMARK_DERIVED", "training_allowed": False}
            inputs.append(sample)
            gold.append({"sample_id": sid, "answer": item["answer"], "historical_correct": old["correct"],
                         "historical_parsed_answer": old["parsed_answer"], "historical_finish_reason": old["finish_reason"],
                         "subject": subject, "provenance": "BENCHMARK_DERIVED", "training_allowed": False})
            audits.append({"sample_id": sid, "source_record_sha256": old["_record_sha256"],
                           "historical_images": all_hashes, "diagnostic_images": [im["sha256"] for im in images],
                           "excluded_images": excluded, "historical_processed_sizes": old["processed_image_sizes"],
                           "historical_image_grid_thw": old["image_grid_thw"], "historical_input_tokens": old["input_tokens"],
                           "input_role_status": "VALID" if images else "BLOCKED_NO_REFERENCED_IMAGE",
                           "legacy_role_mismatch": bool(excluded), "provenance": "BENCHMARK_DERIVED", "training_allowed": False})
            for level in ("O0", "O1"):
                queue.append({"sample_id": sid, "level": level, "annotation_version": "1",
                              "facts": None, "source_image_hashes": [im["sha256"] for im in images],
                              "evidence_locations": [], "completeness_status": "INCOMPLETE",
                              "target_leakage_status": "AWAITING_REVIEW", "annotator_exposure_status": "UNKNOWN",
                              "reviewer_status": "AWAITING_REVIEW", "annotator_id": None, "reviewer_id": None,
                              "approved": False, "provenance": "BENCHMARK_DERIVED", "training_allowed": False})
    # Connected components group shared images and exact shared question/hint; no gold in grouping.
    parent = {r["sample_id"]: r["sample_id"] for r in inputs}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    seen = {}
    for sample in inputs:
        features = ["image:" + im["sha256"] for im in sample["images"]]
        text = re.sub(r"<image\s+\d+>", "<image>", sample["question"] + "\n" + (sample["hint"] or ""))
        features.append("text:" + digest(text))
        for feature in features:
            sid = sample["sample_id"]
            if feature in seen:
                a, b = sorted([find(sid), find(seen[feature])])
                parent[b] = a
            seen[feature] = sid
    for sample in inputs:
        sample["family_id"] = "family-" + digest(find(sample["sample_id"]))[:16]
    for name, content in [("samples.jsonl", inputs), ("gold.jsonl", gold), ("input_audit.jsonl", audits),
                          ("oracle_review_queue.jsonl", queue)]:
        write_rows(output_dir / name, content)
    write_json(output_dir / "sample_selection.json", select_samples(rows))
    frozen = {"v8_source": str(source_dir.resolve()), "original_run": str(source_run.resolve()),
              "v8_files": file_snapshot(source_dir),
              "original_predictions_sha256": sha(source_run / "predictions.jsonl"),
              "model_manifest_sha256": sha(model_manifest), "dataset_manifest_sha256": sha(data_root / "manifest.json"),
              "public_files": file_snapshot(public_root) if public_root else {},
              "public_root": str(Path(public_root).resolve()) if public_root else None}
    write_json(output_dir / "frozen_sources.json", frozen)
    receipt = {"status": "AUDITED", "source_records": 900, "historical_correct": 558,
               "historical_accuracy": .62, "source_parquet_pixel_comparisons": pixel_verified,
               "excluded_image_occurrences": sum(len(a["excluded_images"]) for a in audits),
               "referenced_image_occurrences": sum(len(s["images"]) for s in inputs),
               "no_referenced_image_ids": [a["sample_id"] for a in audits if a["input_role_status"] != "VALID"],
               "approved_oracles": 0, "oracle_status": "BLOCKED_PENDING_INDEPENDENT_HUMAN_REVIEW",
               "files": {name: sha(output_dir / name) for name in ["samples.jsonl", "gold.jsonl", "input_audit.jsonl",
                                                                  "oracle_review_queue.jsonl", "sample_selection.json"]},
               "provenance": "BENCHMARK_DERIVED", "training_allowed": False}
    write_json(output_dir / "receipt.json", receipt)
    return receipt


def verify_frozen(prepared_dir):
    frozen = json.loads((Path(prepared_dir) / "frozen_sources.json").read_text())
    if file_snapshot(frozen["v8_source"]) != frozen["v8_files"]:
        raise ValueError("Immutable V8 files changed")
    if sha(Path(frozen["original_run"]) / "predictions.jsonl") != frozen["original_predictions_sha256"]:
        raise ValueError("Original predictions changed")
    if frozen["public_root"] and file_snapshot(frozen["public_root"]) != frozen["public_files"]:
        raise ValueError("Public MMDL checkout changed")
    return {"v8_unchanged": True, "original_predictions_unchanged": True, "MMDL_unchanged": True}
