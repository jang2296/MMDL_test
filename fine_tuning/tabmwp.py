"""Teacher-free TabMWP preparation; never grants training approval or fetches images."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .audit import check_splits, dump, model_input_record, read_rows, sha, write_rows

REVISION = "5a1a52214521590b075f545b76a4f5ce666345e3"
REPOSITORY = "https://github.com/lupantech/PromptPG"
RAW_URL = f"https://raw.githubusercontent.com/lupantech/PromptPG/{REVISION}/"
SOURCE_ID = f"tabmwp-{REVISION}"
DATASET_LICENSE = "CC-BY-NC-SA-4.0"
SEED = 3407
COUNTS = {"train": 1600, "dev": 200, "test": 200}
ARTIFACTS = {
    "README.md": (24202, "a2c9f0f5b6ed546ae27c91f3720483e8a4c29e5e"),
    "LICENSE.md": (1062, "2816d4a8a9e72eafa012bf838c9a4c205d9a167b"),
    "data/tabmwp/problems_train.json": (25733404, "bebca41ffd8124b196781ecd2ee36488961e0d0a"),
    "data/tabmwp/problems_dev.json": (8533741, "5506b356bc3e45b0aa8d245dc3f927a67da3f50f"),
    "data/tabmwp/problems_test.json": (8520863, "eb3a92279b57aa84ce7bb7d3051c7b9749a1aadc"),
    "data/tabmwp/splits.json": (411685, "dae553b2006b94b08d59858c0a4f435f192de7db"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _verify_blob(data, name):
    size, expected = ARTIFACTS[name]
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if len(data) != size or actual != expected:
        raise ValueError(f"Official source bytes differ: {name}")


def acquire(source_root):
    """Download only six small, pinned source documents into a new private root."""
    root = Path(source_root)
    if root.exists():
        raise FileExistsError(root)
    # Verify everything before creating the destination; partial acquisition is not a receipt.
    payloads = {}
    for name, (size, _) in ARTIFACTS.items():
        with urllib.request.urlopen(RAW_URL + name, timeout=60) as response:
            data = response.read(size + 1)
        _verify_blob(data, name)
        payloads[name] = data
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, data in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        files[name] = {"sha256": sha(path), "bytes": len(data),
                       "git_blob_sha1": ARTIFACTS[name][1], "url": RAW_URL + name}
    receipt = {"source_id": SOURCE_ID, "repository": REPOSITORY, "revision": REVISION,
               "dataset_license": DATASET_LICENSE, "repository_code_license": "MIT",
               "license_evidence": "README.md", "files": files,
               "rights_approved": False, "training_allowed": False,
               "public_upload_allowed": False, "images_downloaded": False}
    dump(root / "source_receipt.json", receipt)
    return receipt


def load_source(source_root):
    """Require the immutable official blob identities, not a self-asserted receipt."""
    root = Path(source_root)
    receipt = json.loads((root / "source_receipt.json").read_text(encoding="utf-8"))
    if (receipt.get("source_id") != SOURCE_ID or receipt.get("revision") != REVISION
            or receipt.get("dataset_license") != DATASET_LICENSE
            or set(receipt.get("files", {})) != set(ARTIFACTS)):
        raise ValueError("Unrecognized TabMWP source receipt")
    for name in ARTIFACTS:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source path is not a private regular artifact")
        _verify_blob(path.read_bytes(), name)
        if receipt["files"][name].get("sha256") != sha(path):
            raise ValueError(f"Source receipt hash differs: {name}")
    splits = json.loads((root / "data/tabmwp/splits.json").read_text(encoding="utf-8"))
    if set(splits) != set(COUNTS):
        raise ValueError("All three official splits required")
    problems, seen = {}, set()
    for split in COUNTS:
        rows = json.loads((root / f"data/tabmwp/problems_{split}.json").read_text(encoding="utf-8"))
        ids = [str(value) for value in splits[split]]
        if len(ids) != len(set(ids)) or set(rows) != set(ids) or seen.intersection(ids):
            raise ValueError("Official split coverage or unique IDs differ")
        if any(not pid.isdecimal() or row.get("split") != split for pid, row in rows.items()):
            raise ValueError("Invalid native ID/split")
        problems[split] = rows
        seen.update(ids)
    return problems, receipt


def table_facts(problem):
    """Preserve column/cell order; require agreement of both native table forms."""
    columns, title, text = (problem.get(k) for k in ("table_for_pd", "table_title", "table"))
    if (not isinstance(columns, dict) or not columns or title is not None and not isinstance(title, str)
            or not isinstance(text, str) or any(not isinstance(k, str) for k in columns)
            or any(not isinstance(values, list) or not values for values in columns.values())
            or any(not isinstance(cell, str) for values in columns.values() for cell in values)):
        raise ValueError("Malformed native table")
    headers = list(columns)
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("Nonrectangular native table")
    rows = [list(row) for row in zip(*columns.values(), strict=True)]
    if (type(problem.get("row_num")) is not int or type(problem.get("column_num")) is not int
            or problem["row_num"] not in (len(rows), len(rows) + 1)
            or problem["column_num"] != len(headers)):
        raise ValueError("Native table dimensions disagree")
    # Native headerless tables have synthetic pandas keys, not visible header facts.
    visible_headers = headers if problem["row_num"] == len(rows) + 1 else None
    parsed = [[cell.strip() for cell in line.split("|")] for line in text.splitlines()]
    expected = [[cell.strip() for cell in row] for row in (
        [headers, *rows] if visible_headers is not None else rows)]
    if parsed != expected:
        raise ValueError("Native table representations disagree")
    return {"title": title, "headers": visible_headers, "rows": rows}


def family_id(facts):
    # Titles/questions/answers do not distinguish otherwise identical tables.
    ordered = {"headers": None if facts["headers"] is None else [cell.strip() for cell in facts["headers"]],
               "rows": [[cell.strip() for cell in row] for row in facts["rows"]]}
    return "tabmwp-table-" + digest(ordered)


def paired_records(pid, problem, facts, native_split):
    """Same source image/question/ID; only assistant supervision changes."""
    question, answer, choices = (problem.get(k) for k in ("question", "answer", "choices"))
    if (not isinstance(pid, str) or not pid.isdecimal()
            or not isinstance(question, str) or not question.strip()
            or not isinstance(answer, str) or not answer.strip()
            or native_split not in COUNTS or problem.get("split") != native_split):
        raise ValueError("Invalid native question, answer, ID, or split")
    user_text = "Question: " + question
    if problem.get("ques_type") == "multi_choice":
        if (not isinstance(choices, list) or not 2 <= len(choices) <= 26
                or any(not isinstance(c, str) or not c.strip() for c in choices)
                or len(set(choices)) != len(choices) or choices.count(answer) != 1):
            raise ValueError("Invalid native multiple-choice annotation")
        user_text += "\nOptions:\n" + "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
        user_text += "\nPlease select the correct answer from the options above."
    elif problem.get("ques_type") != "free_text" or choices is not None:
        raise ValueError("Unknown native question type")
    split = "holdout" if native_split == "test" else native_split
    common = {"id": "tabmwp-" + pid, "root_id": "tabmwp-" + pid,
              "family_id": family_id(facts), "split": split,
              "evaluation_role": "sealed" if native_split == "test" else split,
              "source_id": SOURCE_ID, "source_revision": REVISION,
              "source_native_id": pid, "source_native_split": native_split,
              "native_question_type": problem["ques_type"],
              "source_kind": "SOURCE_ANNOTATED", "lineage": [SOURCE_ID], "parents": [],
              "dataset_license": DATASET_LICENSE, "benchmark_derived": False,
              "training_allowed": False, "public_upload_allowed": False,
              "quality_status": "SOURCE_GOLD", "table_parity": "PASS",
              "pixel_semantics_verified": False, "supervised_target_identifiable": False,
              "image_paths": [f"tables/{pid}.png"], "user_text": user_text,
              "image_source_url": RAW_URL + f"data/tabmwp/tables/{pid}.png",
              "teacher_generated": False, "native_solution_used": False}
    targets = {"A": {"answer": answer}, "B": {"facts": facts, "answer": answer}}
    return {variant: model_input_record(dict(common, target_text=json.dumps(
        target, ensure_ascii=False, separators=(",", ":")))) for variant, target in targets.items()}


def plan(problems, excluded_train_families=()):
    """Scan all original splits before selecting, excluding cross-split families."""
    if set(problems) != set(COUNTS):
        raise ValueError("All native splits required for family exclusion")
    tables, families, quarantine, seen = {}, defaultdict(set), [], set()
    for split in COUNTS:
        for pid, problem in problems[split].items():
            if pid in seen:
                raise ValueError("Native ID overlaps across splits")
            seen.add(pid)
            try:
                facts = table_facts(problem)
            except ValueError as error:
                quarantine.append({"native_id": pid, "native_split": split, "reason": str(error)})
                continue
            tables[pid] = facts
            families[family_id(facts)].add(split)
    leaking = {family for family, splits in families.items() if len(splits) > 1}
    excluded = set(excluded_train_families)
    if any(families.get(family) != {"train"} for family in excluded):
        raise ValueError("Candidate exclusions must name train-only native table families")
    variants = {variant: {split: [] for split in ("train", "dev", "sealed")} for variant in ("A", "B")}
    selected_ids, candidates = {}, {}
    for split, count in COUNTS.items():
        valid = {}
        for pid, problem in problems[split].items():
            if pid not in tables:
                continue
            family = family_id(tables[pid])
            if family in leaking:
                quarantine.append({"native_id": pid, "native_split": split,
                                   "family_id": family, "reason": "ALL_SPLIT_TABLE_FAMILY_OVERLAP"})
                continue
            if split == "train" and family in excluded:
                quarantine.append({"native_id": pid, "native_split": split,
                                   "family_id": family, "reason": "PRETRAINING_CONTAMINATION_CANDIDATE_FAMILY_EXCLUSION"})
                continue
            try:
                valid[pid] = paired_records(pid, problem, tables[pid], split)
            except ValueError as error:
                quarantine.append({"native_id": pid, "native_split": split, "reason": str(error)})
        candidates[split] = len(valid)
        if len(valid) < count:
            raise ValueError(f"Insufficient clean {split} candidates: {len(valid)} < {count}")
        chosen = sorted(valid, key=lambda pid: (digest([SEED, pid]), pid))[:count]
        role = "sealed" if split == "test" else split
        selected_ids[role] = ["tabmwp-" + pid for pid in chosen]
        for variant in variants:
            variants[variant][role] = [valid[pid][variant] for pid in chosen]
    for groups in variants.values():
        check_splits([row for rows in groups.values() for row in rows])
    return variants, {"seed": SEED, "selected_ids": selected_ids,
                      "candidate_counts": candidates, "native_counts": {k: len(v) for k, v in problems.items()},
                      "cross_split_family_count": len(leaking),
                      "family_definition": "ordered native headers/cells, boundary whitespace stripped; title excluded",
                      "quarantine": sorted(quarantine, key=lambda row: (row["native_split"], row["native_id"]))}


def exclusion_policy(source_root, problems, receipt_path):
    """Validate a historical candidate scan, never convert it into an audit approval."""
    source_root, receipt_path = Path(source_root).resolve(), Path(receipt_path)
    scan = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (scan.get("status") != "CANDIDATE_ONLY_SCAN_COMPLETE_NOT_APPROVAL"
            or scan.get("source_id") != SOURCE_ID or scan.get("tabmwp_revision") != REVISION
            or any(scan.get(key) is not False for key in ("training_allowed", "approval_executed", "gpu_executed", "processor_executed", "sample_replacements"))
            or scan.get("training_executed", False) is not False
            or scan.get("condition") != "A" or scan.get("inputs_unchanged_before_after") is not True):
        raise ValueError("Exclusions require a frozen pre-training candidate-only scan")
    original, selected = plan(problems)
    original_ids = selected["selected_ids"]["train"]
    scanned = scan.get("scanned_ids", [])
    candidates = scan.get("candidates", [])
    if (not isinstance(scanned, list) or len(scanned) != 1600 or len(set(scanned)) != 1600
            or set(scanned) != set(original_ids) or scan.get("scanned_record_count") != 1600
            or not isinstance(candidates, list) or not candidates):
        raise ValueError("Scan must cover the original deterministic native1600 train selection")
    reasons, combinations, candidate_ids = Counter(), Counter(), []
    allowed = {"EXACT_BENCHMARK_TEXT", "NEAR_BENCHMARK_TEXT", "EXACT_BENCHMARK_IMAGE", "NEAR_IMAGE_REVIEW_REQUIRED"}
    for candidate in candidates:
        values = candidate.get("reasons", [])
        if (candidate.get("id") not in original_ids or not isinstance(values, list) or not values
                or any(value not in allowed for value in values) or values != sorted(set(values))):
            raise ValueError("Unknown, dev/sealed, or malformed contamination candidate")
        candidate_ids.append(candidate["id"])
        reasons.update(values)
        combinations["+".join(values)] += 1
    if (len(set(candidate_ids)) != len(candidate_ids) or scan.get("candidate_record_count") != len(candidates)
            or any(type(scan.get(key)) is not int for key in ("scanned_record_count", "candidate_record_count", "no_candidate_record_count"))
            or scan.get("no_candidate_record_count") != 1600 - len(candidates)
            or scan.get("reason_counts") != dict(reasons) or scan.get("reason_combination_counts") != dict(combinations)):
        raise ValueError("Candidate counts or reason arithmetic differs")
    bindings = scan.get("source_and_index_sha256", {})
    if (not isinstance(bindings, dict) or not bindings
            or any(not Path(name).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", str(value))
                   for name, value in bindings.items())):
        raise ValueError("Absolute original source/index byte bindings required")
    producers = [name for name in bindings if Path(name).name == "tabmwp.py"]
    if len(producers) != 1:
        raise ValueError("Historical converter producer identity missing")
    historical_producer = producers[0]
    for name, value in bindings.items():
        if name != historical_producer and (Path(name).is_symlink() or sha(name) != value):
            raise ValueError("Historical scan source/index bytes changed")
    if any(str(source_root / name) not in bindings for name in ["source_receipt.json", *ARTIFACTS]):
        raise ValueError("Pinned native source byte coverage missing")
    index_paths = [name for name in bindings if "index" in Path(name).name and Path(name).suffix == ".json"]
    manifests = [Path(name) for name in bindings if Path(name).name == "manifest.json"]
    if len(index_paths) != 1 or len(manifests) != 1:
        raise ValueError("Exactly one original prepared manifest and benchmark index required")
    old_path = manifests[0]
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if (old.get("preparation_hash") != digest({k: v for k, v in old.items() if k != "preparation_hash"})
            or old.get("preparation_hash") != scan.get("tabmwp_preparation_hash")
            or old.get("source_id") != SOURCE_ID or old.get("revision") != REVISION
            or old.get("training_allowed") is not False or old.get("selection") != {k: v for k, v in selected.items() if k != "quarantine"}):
        raise ValueError("Scan must bind the original unchanged default preparation")
    for variant, groups in original.items():
        for role, rows in groups.items():
            path = old_path.parent / f"{variant}/{role}.jsonl"
            if str(path) not in bindings or old.get("files", {}).get(f"{variant}/{role}.jsonl", {}).get("sha256") != bindings[str(path)] or read_rows(path) != rows:
                raise ValueError("Original paired native proposal bytes differ")
    trial_path = Path(scan.get("trial_root", "")) / "trial_manifest.json"
    proposal_path = trial_path.parent / "A/proposed.jsonl"
    if str(trial_path) not in bindings or str(proposal_path) not in bindings:
        raise ValueError("Original scanned trial/proposal bindings missing")
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    if (trial.get("trial_hash") != digest({k: v for k, v in trial.items() if k != "trial_hash"})
            or trial.get("trial_hash") != scan.get("tabmwp_trial_hash")
            or trial.get("tabmwp_preparation_hash") != old["preparation_hash"]
            or trial.get("proposals", {}).get("A", {}).get("sha256") != bindings[str(proposal_path)]):
        raise ValueError("Original trial identity differs from candidate scan")
    proposal = [row for row in read_rows(proposal_path) if row.get("source_id") == SOURCE_ID]
    expected = {row["id"]: row for row in original["A"]["train"]}
    keys = ("user_text", "target_text", "image_paths", "family_id", "root_id", "source_revision")
    if ([row["id"] for row in proposal] != scanned
            or any(any(row.get(key) != expected[row["id"]][key] for key in keys) for row in proposal)):
        raise ValueError("Scanned IDs/inputs differ from the bound original trial")
    families = sorted({expected[sid]["family_id"] for sid in candidate_ids})
    excluded_ids = []
    for pid, problem in problems["train"].items():
        try:
            facts = table_facts(problem)
        except ValueError:
            continue
        if family_id(facts) in families:
            excluded_ids.append("tabmwp-" + pid)
    return {"policy": "PRETRAINING_CONSERVATIVE_CANDIDATE_FAMILY_EXCLUSION_V1",
            "reason": "Exclude candidate table families before a new fixed-quota preparation; not duplicate proof or approval",
            "scan_sha256": sha(receipt_path), "candidate_ids": sorted(candidate_ids),
            "candidate_reasons": {row["id"]: row["reasons"] for row in candidates},
            "excluded_train_ids": sorted(excluded_ids), "excluded_train_families": families,
            "original_preparation_hash": old["preparation_hash"], "original_manifest_sha256": bindings[str(old_path)],
            "benchmark_index_sha256": bindings[index_paths[0]],
            "historical_converter_sha256": bindings[historical_producer], "historical_converter_path": historical_producer,
            "source_and_index_sha256": bindings,
            "training_allowed": False, "approval_executed": False, "model_outcomes_used": False}


def prepare(source_root, output_root, exclusion_receipt=None):
    """Freeze paired proposals, not approved training inputs; images remain pending."""
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(output)
    source_root = Path(source_root)
    problems, receipt = load_source(source_root)
    policy = exclusion_policy(source_root, problems, exclusion_receipt) if exclusion_receipt is not None else None
    variants, selection = plan(problems, policy["excluded_train_families"] if policy else ())
    if policy:
        if exclusion_policy(source_root, problems, exclusion_receipt) != policy:
            raise ValueError("Candidate exclusion evidence changed during preparation")
        selection["train_candidate_exclusion"] = policy
    # Bind source bytes again before output, in case an external writer changed them.
    _, current = load_source(source_root)
    if current != receipt:
        raise ValueError("Source receipt changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    if policy:
        with (output / "exclusion_receipt.json").open("xb") as stream:
            stream.write(Path(exclusion_receipt).read_bytes())
        if sha(output / "exclusion_receipt.json") != policy["scan_sha256"]:
            raise ValueError("Candidate scan changed before frozen copy")
    files = {}
    for variant, groups in variants.items():
        for split, rows in groups.items():
            name = f"{variant}/{split}.jsonl"
            write_rows(output / name, rows)
            files[name] = {"sha256": sha(output / name), "count": len(rows)}
    dump(output / "quarantine.json", selection.pop("quarantine"))
    manifest = {"status": "PREPARED_PENDING_GATES", "training_allowed": False,
                "benchmark_derived": False, "public_upload_allowed": False,
                "source_id": SOURCE_ID, "revision": REVISION, "dataset_license": DATASET_LICENSE,
                "source_receipt_sha256": sha(source_root / "source_receipt.json"),
                "source_files": receipt["files"], "files": files, "selection": selection,
                "target_definitions": {"A": "native answer only, JSON answer field",
                                       "B": "native ordered visible table facts plus identical JSON answer"},
                "quarantine_sha256": sha(output / "quarantine.json"),
                "pending_gates": ["SOURCE_RIGHTS_AND_ANCESTRY_REGISTRATION",
                                  "PINNED_IMAGE_ACQUISITION_HASH_AND_PIXEL_AUDIT",
                                  "ALL_MMMU_AND_MMMU_PRO_SPLIT_CONTAMINATION_AUDIT",
                                  "REAL_PINNED_PROCESSOR_AND_ASSISTANT_MASK_PROOF",
                                  "A_B_ASSISTANT_TARGET_TOKEN_EXPOSURE_AND_TRAINING_PROTOCOL"],
                "verification_limit": "Native table JSON/text parity is not image-pixel truth or answer-solver verification",
                "teacher_generated": False, "native_solution_used": False}
    manifest["preparation_hash"] = digest(manifest)
    dump(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--exclusion-receipt", type=Path)
    args = parser.parse_args()
    if args.exclusion_receipt is not None and args.output_root is None:
        parser.error("--exclusion-receipt requires a new --output-root")
    if args.acquire:
        acquire(args.source_root)
    if args.output_root:
        result = prepare(args.source_root, args.output_root, args.exclusion_receipt)
        print(json.dumps({"status": result["status"], "preparation_hash": result["preparation_hash"]}))
    elif not args.acquire:
        parser.error("Specify --acquire or --output-root")


if __name__ == "__main__":
    main()
