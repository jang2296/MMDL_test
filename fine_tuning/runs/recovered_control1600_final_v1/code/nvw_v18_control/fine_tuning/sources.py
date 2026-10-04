"""Metadata locally; exact approved external downloads only on an owned RunPod.

No source labels or benchmark files are used as generator inputs. TRACE's
answer-dependent annotations remain target-only and are not perception labels.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path

from .audit import dump, image_fingerprint, model_input_record, sha, shingles, text_hash, write_rows


SOURCES = {
    "trace": ("maveryn/trace", "dataset-v1"),
    "raven_hf": ("HuggingFaceM4/RAVEN", "main"),
    "raven_nvidia": ("nvidia/Nemotron-Image-Training-v3", "main"),
    "arc_synthetic": ("Fraser/arc-agi-synthetic", "main"),
    "clevr_smoke": ("maujim/CLEVR_sample_500", "main"),
    "regular": ("zihuwang/ReGuLaR", "main"),
}


def inspect(output):
    from huggingface_hub import HfApi
    api, rows = HfApi(), {}
    for name, (repo, revision) in SOURCES.items():
        try:
            info = api.dataset_info(repo, revision=revision, files_metadata=True)
            rows[name] = {
                "repo_id": repo, "requested_revision": revision, "revision": info.sha,
                "card": info.card_data.to_dict() if info.card_data else {},
                "files": [{"path": x.rfilename, "bytes": x.size,
                    "lfs_sha256": x.lfs.sha256 if x.lfs else None} for x in info.siblings],
                "training_allowed": False, "rights_status": "PENDING_FILE_AND_ASSET_REVIEW",
                "metadata_only_no_payload_download": True,
            }
        except Exception as exc:
            rows[name] = {"repo_id": repo, "requested_revision": revision,
                          "error": f"{type(exc).__name__}: {exc}", "training_allowed": False}
    dump(output, {"status": "METADATA_INSPECTED", "sources": rows})
    return rows


def download_exact(repo, revision, files, cache, destination):
    from huggingface_hub import HfApi, hf_hub_download
    if not os.environ.get("RUNPOD_POD_ID"):
        raise ValueError("HF payload downloads are allowed on RunPod only")
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision or len(revision) != 40:
        raise ValueError("Exact dataset revision required")
    remote = {x.rfilename: x for x in info.siblings}
    records, paths = [], []
    for name in files:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or name not in remote:
            raise ValueError("Unapproved source file")
        path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision,
                                    cache_dir=str(cache)))
        checksum = sha(path)
        if path.stat().st_size != remote[name].size:
            raise ValueError("Downloaded dataset size mismatch")
        if remote[name].lfs and remote[name].lfs.sha256 != checksum:
            raise ValueError("Downloaded dataset LFS hash mismatch")
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": checksum})
        paths.append(path)
    dump(destination, {"repo_id": repo, "revision": revision, "files": records,
                       "training_allowed": False, "purpose": "audit_before_training"})
    return paths


def benchmark_index(paths, coverage, output):
    import pyarrow.parquet as pq
    from PIL import Image
    text, pixels, bytes_hashes, dhashes, long_text = set(), set(), set(), set(), []
    if not isinstance(coverage, dict) or not isinstance(coverage.get("files"), list) or not coverage["files"]:
        raise ValueError("Typed exact requested benchmark file coverage required")
    requested = {}
    for entry in coverage["files"]:
        path = str(Path(entry["path"]).resolve(strict=True))
        if (path in requested or not isinstance(entry.get("records"), int) or entry["records"] < 0
                or entry.get("dataset_id") not in {"MMMU/MMMU", "MMMU/MMMU_Pro"}
                or not entry.get("split") or not entry.get("configuration")
                or not re.fullmatch(r"[0-9a-f]{40}", entry.get("revision", ""))):
            raise ValueError("Invalid/duplicate exact benchmark coverage entry")
        requested[path] = entry
    resolved = [str(Path(path).resolve(strict=True)) for path in paths]
    if len(set(resolved)) != len(resolved) or set(resolved) != set(requested):
        raise ValueError("Requested benchmark file coverage mismatch")
    if coverage.get("record_count") != sum(entry["records"] for entry in requested.values()):
        raise ValueError("Requested benchmark record coverage mismatch")
    count, per_file, image_count = 0, {}, 0
    for path in paths:
        requested_entry = requested[str(Path(path).resolve())]
        if requested_entry.get("sha256") and sha(path) != requested_entry["sha256"]:
            raise ValueError("Requested benchmark file bytes differ")
        before = count
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                count += 1
                options = row.get("options") or []
                if isinstance(options, str):
                    try:
                        options = ast.literal_eval(options)
                    except (ValueError, SyntaxError):
                        options = [options]
                if not isinstance(options, (list, tuple)) or any(not isinstance(value, str) for value in options):
                    raise ValueError("Benchmark options schema invalid")
                question = row.get("question")
                vision_only = "vision" in requested_entry["configuration"].casefold()
                if not isinstance(question, str) or not question.strip():
                    if vision_only and question in (None, ""):
                        question = ""
                    else:
                        raise ValueError("Benchmark question schema missing/invalid")
                for value in (question, question+"\n"+"\n".join(map(str, options))):
                    text.add(text_hash(value))
                s = shingles(question)
                if len(s) >= 10:
                    long_text.append(sorted(s))
                entries = [v for k, v in row.items()
                           if re.fullmatch(r"image(?:_?\d+)?|images", k) and v is not None]
                if not entries:
                    raise ValueError("Benchmark visual record has no image bytes")
                before_images = image_count
                for entry in entries:
                    for image_entry in entry if isinstance(entry, list) else [entry]:
                        if (not isinstance(image_entry, dict)
                                or not isinstance(image_entry.get("bytes"), (bytes, bytearray))
                                or not image_entry["bytes"]):
                            raise ValueError("Benchmark image bytes missing; coverage is incomplete")
                        raw = image_entry["bytes"]
                        image_count += 1
                        bytes_hashes.add(hashlib.sha256(raw).hexdigest())
                        with Image.open(io.BytesIO(raw)) as image:
                            fp = image_fingerprint(image)
                        pixels.add(fp["pixels_sha256"])
                        dhashes.add(fp["dhash"])
                if image_count == before_images:
                    raise ValueError("Benchmark image list is empty")
        observed = count-before
        requested_entry = requested[str(Path(path).resolve())]
        if observed != requested_entry["records"]:
            raise ValueError("Requested benchmark per-file record coverage mismatch")
        per_file[str(Path(path).resolve())] = observed
    if count != coverage["record_count"]:
        raise ValueError("Requested benchmark total record coverage mismatch")
    value = {"status": "COMPLETE", "coverage": coverage, "record_count": count,
        "coverage_verified": True, "per_file_record_counts": per_file, "image_count": image_count,
        "source_files": {str(Path(p).resolve()): sha(p) for p in paths}, "text_sha256": sorted(text),
        "pixels_sha256": sorted(pixels), "image_bytes_sha256": sorted(bytes_hashes),
        "dhash": sorted(dhashes), "long_text_shingles": long_text,
        "training_allowed": False, "purpose": "exclusion_fingerprints_only",
        "crop_and_semantic_similarity_complete": False}
    dump(output, value)
    return {k: value[k] for k in ("status", "coverage", "record_count")}


def typed_trace_answer(value):
    answer = json.loads(value) if isinstance(value, str) else value
    if not isinstance(answer, dict) or set(answer) != {"type", "value"}:
        raise ValueError("TRACE typed answer requires exactly type/value")
    kind, result = answer["type"], answer["value"]
    if not isinstance(kind, str):
        raise ValueError("TRACE answer type must be a string")
    valid = {"integer": isinstance(result, int) and not isinstance(result, bool),
             "number": isinstance(result, (int, float)) and not isinstance(result, bool)
                       and math.isfinite(result),
             "boolean": isinstance(result, bool),
             "string": isinstance(result, str) and bool(result.strip()),
             "option_letter": isinstance(result, str) and bool(re.fullmatch("[A-Z]", result))}
    if kind not in valid or not valid[kind]:
        raise ValueError("TRACE answer value violates its declared type")
    return answer


def trace_family_groups(rows, seed):
    """Union actual task/scene/grammar identifiers before any split/augmentation."""
    parents = {}
    def find(value):
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    for row in rows:
        tokens = row["grouping_identifiers"]
        for token in tokens[1:]:
            parents[find(token)] = find(tokens[0])
    components = {}
    for token in list(parents):
        components.setdefault(find(token), []).append(token)
    for row in rows:
        tokens = sorted(components.get(find(row["grouping_identifiers"][0]), row["grouping_identifiers"]))
        family = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        row["family_id"] = "trace-family-"+family
        bucket = hashlib.sha256(f"{seed}:TRACE-FAMILY:{family}".encode()).digest()[0]
        row["split"] = "train" if bucket < 205 else "dev" if bucket < 230 else "holdout"


def trace_records(paths, output, seed=3407, train_limit=9000):
    import pyarrow.parquet as pq
    from PIL import Image
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    images = root/"images"
    images.mkdir()
    rows, skipped, seen = [], [], set()
    for path in paths:
        source_hash = sha(path)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for item in batch.to_pylist():
                sid, task = item.get("instance_id"), item.get("task")
                if (not isinstance(sid, str) or not sid.strip() or not isinstance(task, str)
                        or not task.strip() or sid in seen):
                    raise ValueError("Missing or duplicate TRACE source identity")
                seen.add(sid)
                answer = typed_trace_answer(item["answer_gt"])
                entries = item.get("images")
                if not isinstance(entries, list) or not entries or len(entries) > 2:
                    skipped.append({"id": sid, "reason": "UNSUPPORTED_IMAGE_COUNT"})
                    continue
                prompt = item.get("prompt_answer")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("TRACE image prompt is absent")
                if re.search(r"<image[^>]*>", prompt) and any(
                        marker != "<image>" for marker in re.findall(r"<image[^>]*>", prompt)):
                    raise ValueError("Unsupported TRACE image placeholder")
                if prompt.count("<image>") not in (0, len(entries)):
                    raise ValueError("TRACE image/prompt placeholder count mismatch")
                identity = hashlib.sha256(json.dumps(["trace", sid]).encode()).hexdigest()
                image_paths = []
                for i, entry in enumerate(entries):
                    if not isinstance(entry, dict) or not isinstance(entry.get("bytes"), (bytes, bytearray)) or not entry["bytes"]:
                        raise ValueError("TRACE image bytes absent")
                    name = f"images/{identity}_{i}.png"
                    with Image.open(io.BytesIO(entry["bytes"])) as image:
                        with (root/name).open("xb") as destination:
                            image.convert("RGB").save(destination, format="PNG")
                    image_paths.append(name)
                annotation = json.loads(item["annotation_gt"])
                reward = json.loads(item["reward_contract"])
                if not isinstance(annotation, dict) or not isinstance(reward, dict):
                    raise ValueError("TRACE annotation/reward contract must be objects")
                declared_type = reward.get("answer_type")
                if declared_type is not None and declared_type != answer["type"]:
                    raise ValueError("TRACE reward/answer type disagreement")
                grouping = ["task:"+task]
                metadata = item.get("metadata", {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                if not isinstance(metadata, dict):
                    raise ValueError("TRACE metadata must be an object")
                for key in ("scene", "scene_id", "scene_grammar", "scene_grammar_id", "grammar", "grammar_id"):
                    value = item.get(key, metadata.get(key))
                    if value is not None:
                        if not isinstance(value, (str, int)) or not str(value).strip():
                            raise ValueError("Invalid TRACE scene/grammar identifier")
                        grouping.append(key+":"+str(value))
                # The original answer format remains intact. Never inject its
                # answer-dependent annotation into the user message.
                target = json.dumps({"answer": answer["value"]}, ensure_ascii=False)
                record = {"id": "trace-"+identity, "root_id": "trace-"+identity,
                    "grouping_identifiers": grouping, "source_instance_id": sid,
                    "source_kind": "PUBLIC_INDEPENDENT_SYNTHETIC", "source_id": "trace",
                    "lineage": ["trace"], "benchmark_derived": False,
                    "training_allowed": False, "quality_status": "SOURCE_GOLD",
                    "supervised_target_identifiable": True, "answer_identifiable": True,
                    "user_text": prompt, "target_text": target, "image_paths": image_paths,
                    "source_prompt_answer": prompt,
                    "source_conversations": item.get("conversations", item.get("messages")),
                    "domain": item["domain"], "task": task,
                    "typed_answer": answer, "annotation_type": annotation.get("type"),
                    "annotation_role": "ANSWER_DEPENDENT_NOT_INPUT", "reward_contract": reward,
                    "original_split": "train", "record_source_file": source_hash}
                rows.append(model_input_record(record))
    trace_family_groups(rows, seed)
    rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
    train = [r for r in rows if r["split"] == "train"][:train_limit]
    remaining = [r for r in rows if r["split"] != "train"]
    write_rows(root/"records.jsonl", train+remaining)
    dump(root/"conversion.json", {"status": "CONVERTED_NOT_APPROVED", "train_count": len(train),
        "dev_count": sum(r["split"] == "dev" for r in remaining),
        "holdout_count": sum(r["split"] == "holdout" for r in remaining),
        "domains": sorted({r["domain"] for r in rows}), "skipped": skipped,
        "split_policy": "connected registered task/scene/grammar families; original train only",
        "actual_grouping_dimensions": sorted({token.split(":", 1)[0] for r in rows for token in r["grouping_identifiers"]}),
        "generalization_claim": "registered-family disjointness only; task-only metadata cannot establish unseen scene/grammar/domain",
        "independent_solver_verified": False, "trace_sidecar_reused": False})
    return train+remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    metadata = sub.add_parser("inspect")
    metadata.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "inspect":
        result = inspect(args.output)
        print(json.dumps({k: {f: v.get(f) for f in ("repo_id", "revision", "error")} for k,v in result.items()}))


if __name__ == "__main__":
    main()
