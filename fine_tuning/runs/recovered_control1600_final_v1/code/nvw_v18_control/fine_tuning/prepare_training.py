"""Private CPU-only preparation; no HF downloads, inference, or training.

python -m fine_tuning.prepare_training index --coverage ... --output ...
python -m fine_tuning.prepare_training smoke --nvw-root ... --source-registry ... \
    --benchmark-index ... --model-path ... --output ...
python -m fine_tuning.prepare_training pilot --nvw-root ... --trace-downloads ... \
    --trace-cache-root ... --source-registry ... --benchmark-index ... \
    --model-path ... --output ...
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from . import audit, nvw, sources

POLICY_PATH = Path(__file__).with_name("mixture_policy_v1.json")


def policy():
    return json.loads(POLICY_PATH.read_text())


def registry(path):
    value = json.loads(Path(path).read_text())
    value = value.get("sources", value)
    if not isinstance(value, dict) or not value:
        raise ValueError("Actual nonempty source-rights registry required")
    return value


def order(rows, seed=3407):
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:COMMON:{row['id']}".encode()).hexdigest())


def build_exclusion_index(coverage_path, output):
    coverage = json.loads(Path(coverage_path).read_text())
    paths = [Path(entry["path"]) for entry in coverage["files"]]
    return sources.benchmark_index(paths, coverage, output)


def trace_paths(downloads_path, cache_root):
    """Resolve only receipt-listed existing cache files and verify all exact bytes."""
    receipt = json.loads(Path(downloads_path).read_text())
    if (receipt.get("repo_id") != "maveryn/trace" or not isinstance(receipt.get("files"), list)
            or not receipt["files"] or not isinstance(receipt.get("revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", receipt["revision"])):
        raise ValueError("Exact closed TRACE download receipt required")
    cache_root = Path(cache_root).resolve(strict=True)
    snapshot = cache_root / "datasets--maveryn--trace" / "snapshots" / receipt["revision"]
    paths, names = [], set()
    for entry in receipt["files"]:
        relative = Path(entry["path"])
        train_path = (relative.parts[:2] == ("data", "train") and len(relative.parts) == 3
                      and relative.suffix == ".parquet")
        legacy_train_path = bool(re.fullmatch(r"data/train(?:-\d+-of-\d+)?\.parquet", relative.as_posix()))
        if (relative.is_absolute() or ".." in relative.parts
                or not (train_path or legacy_train_path)
                or entry["path"] in names):
            raise ValueError("Duplicate/traversing/non-Parquet TRACE receipt path")
        names.add(entry["path"])
        candidate = Path(entry.get("local_path", entry.get("cache_path", snapshot / relative)))
        path = candidate.resolve(strict=True)
        if not path.is_relative_to(cache_root) or not path.is_file():
            raise ValueError("TRACE cache path escapes exact cache root")
        if path.stat().st_size != entry["bytes"] or audit.sha(path) != entry["sha256"]:
            raise ValueError("TRACE cached file differs from download receipt")
        paths.append(path)
    return paths


def joint_counts(rows):
    counts = Counter((row["nvw_track"], row["view"]) for row in rows if "nvw_track" in row)
    return {track: {view: counts[track, view] for view in ("qa", "structure", "grounded")}
            for track in ("NVW-K", "NVW-I", "NVW-S")}


def verified_canonical_roots(rows, path):
    """Root IDs are world fingerprints, not caller-controlled episode labels."""
    episodes = audit.read_rows(path)
    canonical = {}
    for episode in episodes:
        if (not isinstance(episode.get("world"), dict) or not isinstance(episode.get("family_id"), str)
                or episode.get("root_id") != nvw.digest([episode["family_id"], episode["world"]])
                or episode["root_id"] in canonical):
            raise ValueError("Canonical root fingerprint mismatch or duplicate")
        canonical[episode["root_id"]] = episode
    if any(row.get("root_id") not in canonical
           or row.get("family_id") != canonical[row["root_id"]]["family_id"]
           for row in rows if row.get("source_id") == "nvw-independent-procedural-v1"):
        raise ValueError("Own-NVW record root/family differs from canonical world")
    return canonical


def processor_contract(receipt):
    return {key: receipt.get(key) for key in ("processor_revision", "processor_class", "chat_template_sha256",
                                            "sequence_budget", "episode_pixels", "producer_sha256")}


def validate_joint_quotas(matrix, view_quotas, track_quotas=None):
    if set(matrix) != {"NVW-K", "NVW-I", "NVW-S"}:
        raise ValueError("All three canonical NVW tracks require explicit joint quotas")
    for row in matrix.values():
        if set(row) != {"qa", "structure", "grounded"} or any(type(v) is not int or v < 0 for v in row.values()):
            raise ValueError("Invalid track/view quota table")
    columns = {view: sum(row[view] for row in matrix.values()) for view in ("qa", "structure", "grounded")}
    if columns != view_quotas:
        raise ValueError("Track/view quota marginals disagree")
    if track_quotas is not None and {track: sum(values.values()) for track, values in matrix.items()} != track_quotas:
        raise ValueError("Canonical track quota marginals disagree")


def select_nvw(nvw_root, quotas=None, seed=3407, joint_quotas=None):
    nvw_root = Path(nvw_root)
    verification = nvw.verify_dataset(nvw_root)
    quotas = quotas or policy()["nvw_views"]
    matrix = joint_quotas or policy()["nvw_joint_views"]
    validate_joint_quotas(matrix, quotas, policy()["nvw_tracks"] if joint_quotas is None else None)
    rows = audit.read_rows(nvw_root / "prototype_train.jsonl")
    # Check the entire supplied split universe before selection.
    universe = []
    for name in ("prototype_train.jsonl", "prototype_development.jsonl", "prototype_sealed_holdout.jsonl"):
        if (nvw_root/name).exists():
            universe.extend(audit.read_rows(nvw_root/name))
    audit.check_splits(universe)
    canonical_path = nvw_root/"canonical_private.jsonl"
    tracks = {episode["root_id"]: episode["track"] for episode in audit.read_rows(canonical_path)}
    canonical_hash = audit.sha(canonical_path)
    enriched = []
    for row in rows:
        track = tracks.get(row["root_id"])
        if track not in matrix:
            raise ValueError("Canonical NVW track absent or invalid")
        enriched.append(dict(row, nvw_track=track, nvw_track_source_sha256=canonical_hash))
    eligible = [row for row in enriched if row["split"] == "train"
                and (row["view"] != "qa" or row.get("answer_identifiable") is True)]
    available_joint = joint_counts(eligible)
    selected = []
    for track, by_view in matrix.items():
        for view, count in by_view.items():
            candidates = [row for row in eligible if row["nvw_track"] == track and row["view"] == view]
            if len(candidates) < count:
                raise ValueError(f"NVW quota shortfall: {track}/{view} requires {count}, available {len(candidates)}; no cross-track borrowing")
            selected.extend(order(candidates, seed)[:count])
    audit.check_splits(selected)
    if any(count > 3 for count in Counter(row["root_id"] for row in selected).values()):
        raise ValueError("NVW selection exceeds three views per root")
    return order(selected, seed), {"independent_verification": verification,
                                  "available_views": {view: sum(row[view] for row in available_joint.values()) for view in quotas},
                                  "available_joint_views": available_joint, "required_joint_views": matrix,
                                  "selected_joint_views": joint_counts(selected),
                                  "selected_tracks": dict(Counter(row["nvw_track"] for row in selected)),
                                  "selected_views": dict(Counter(row["view"] for row in selected)),
                                  "canonical_track_source_sha256": canonical_hash,
                                  "source_manifest_sha256": audit.sha(nvw_root/"manifest.json")}


def copy_image(source_root, destination_root, name):
    """Reuse regular assets by exact hash; exclusive-create every new destination."""
    original = audit.image_path(source_root, name)
    destination_root = Path(destination_root).resolve(strict=True)
    target = destination_root / name
    if not target.resolve().is_relative_to(destination_root):
        raise ValueError("Destination image path escapes mixture root")
    for parent in target.parents:
        if parent == destination_root:
            break
        if parent.is_symlink():
            raise ValueError("Mixture image parent is a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = audit.sha(original)
    if target.exists() or target.is_symlink():
        existing = audit.image_path(destination_root, name)
        if audit.sha(existing) != expected:
            raise ValueError("Image name collision with different source bytes")
        return expected
    with original.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    if audit.sha(target) != expected:
        raise ValueError("Copied image bytes differ from verified source")
    return expected


def freeze_approved(raw_path, image_root, source_registry, benchmark_index, model_path, output,
                    expected_records, expected_roots=None, stage="pilot", selection=None):
    output = Path(output)
    # Rights failures are diagnosed before expensive actual preprocessing.
    registered = registry(source_registry)
    raw = audit.read_rows(raw_path)
    audit.check_splits(raw)
    for row in raw:
        audit.check_lineage(row, registered)
        audit.check_target_identifiability(row, image_root)
    _, receipt = audit.preflight_records(raw_path, image_root, model_path, output/"processor_preflight.json")
    approved_path = output/"approved"
    approved = audit.approve(raw_path, image_root, registered, benchmark_index, approved_path,
                             input_preflight_receipt=output/"processor_preflight.json")
    rows = audit.read_rows(approved_path/"records.jsonl")
    roots = len({row["root_id"] for row in rows})
    ready = len(rows) == expected_records and (expected_roots is None or roots == expected_roots)
    approved_joint = joint_counts(rows)
    if stage == "pilot":
        required_joint = (selection or {}).get("required_joint_views")
        if not isinstance(required_joint, dict):
            raise ValueError("Pilot must declare its canonical track/view quota table")
        required_views = {view: sum(values[view] for values in required_joint.values())
                          for view in ("qa", "structure", "grounded")}
        validate_joint_quotas(required_joint, required_views)
        if expected_records == policy()["pilot_records"] and required_joint != policy()["nvw_joint_views"]:
            raise ValueError("10K pilot joint quotas differ from the frozen track/view policy")
        ready = ready and approved_joint == required_joint
    report = {"status": "READY" if ready else "INCOMPLETE_QUOTA", "stage": stage,
              "expected_records": expected_records, "approved_records": len(rows), "approved_roots": roots,
              "quarantined_records": approved["contamination_audit"]["quarantined"],
              "approved_nvw_joint_views": approved_joint,
              "approved_nvw_tracks": {track: sum(values.values()) for track, values in approved_joint.items()},
              "selection": selection or {}, "mixture_policy": policy(),
              "mixture_policy_sha256": audit.sha(POLICY_PATH),
              "model_input_receipt_sha256": audit.sha(output/"processor_preflight.json"),
              "processor_revision": receipt["processor_revision"],
              "training_allowed": ready, "public_upload_allowed": False}
    if expected_roots is not None:
        report["expected_roots"] = expected_roots
    audit.dump(output/"preparation_receipt.json", report)
    if not ready:
        raise ValueError("Approved quota shortfall; quarantined/unapproved records will not fill it")
    ordered_ids = [row["id"] for row in rows]
    common = dict(approved, stage=stage, preparation_receipt_sha256=audit.sha(output/"preparation_receipt.json"),
                  mixture_policy=policy(), mixture_policy_sha256=audit.sha(POLICY_PATH),
                  record_order_sha256=hashlib.sha256(json.dumps(ordered_ids).encode()).hexdigest(),
                  module_arms=policy()["module_arms"], same_records_order_and_images_for_all_arms=True,
                  actual_arm_execution_verified=False,
                  nvw_joint_views=approved_joint,
                  nvw_tracks=report["approved_nvw_tracks"],
                  records_path=str((approved_path/"records.jsonl").resolve()))
    audit.dump(output / ("smoke_manifest.json" if stage == "smoke" else "common_manifest.json"), common)
    return report


def prepare_smoke(nvw_root, source_registry, benchmark_index, model_path, output):
    nvw_root, output = Path(nvw_root), Path(output)
    verification = nvw.verify_dataset(nvw_root)
    rows = audit.read_rows(nvw_root/"smoke_train.jsonl")
    expected = policy()
    if (len(rows) != expected["smoke_views"] or len({r["root_id"] for r in rows}) != expected["smoke_roots"]
            or Counter(row["view"] for row in rows) != {"qa": 120, "structure": 120, "grounded": 120}
            or any(row["split"] != "train" for row in rows)):
        raise ValueError("Exact 120-root/360-view train-only smoke contract required")
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output/"proposed_records.jsonl"
    audit.write_rows(raw_path, order(rows, expected["seed"]))
    return freeze_approved(raw_path, nvw_root, source_registry, benchmark_index, model_path, output,
                           expected["smoke_views"], expected["smoke_roots"], "smoke",
                           {"independent_verification": verification,
                            "source_manifest_sha256": audit.sha(nvw_root/"manifest.json")})


def prepare_pilot(nvw_root, trace_downloads, trace_cache_root, source_registry,
                   benchmark_index, model_path, output):
    nvw_root, output = Path(nvw_root), Path(output)
    if output.exists():
        raise FileExistsError("Refuse to overwrite an existing mixture preparation")
    expected = policy()
    selected, selection = select_nvw(nvw_root, expected["nvw_views"], expected["seed"])
    paths = trace_paths(trace_downloads, trace_cache_root)
    # TRACE conversion owns the common regular image root; NVW images are added
    # with their original relative names. No duplicate TRACE image tree is needed.
    trace = sources.trace_records(paths, output, expected["seed"], expected["trace_records"])
    trace_train = [row for row in trace if row["split"] == "train"]
    if len(trace_train) != expected["trace_records"]:
        audit.dump(output/"shortfall.json", {"status": "INCOMPLETE_TRACE_QUOTA",
                    "expected": expected["trace_records"], "available": len(trace_train), "training_allowed": False})
        raise ValueError("TRACE conversion cannot satisfy 9000 approved-candidate records")
    roots = {row["root_id"] for row in selected}
    canonical = [row for row in audit.read_rows(nvw_root/"canonical_private.jsonl") if row["root_id"] in roots]
    if len(canonical) != len(roots):
        raise ValueError("Selected NVW roots lack exact canonical episode ownership")
    audit.write_rows(output/"canonical_private.jsonl", canonical)
    for row in selected:
        for name in row["image_paths"]:
            copy_image(nvw_root, output, name)
    rows = order(selected + trace_train, expected["seed"])
    audit.check_splits(rows + [row for row in trace if row["split"] != "train"])
    raw_path = output/"proposed_records.jsonl"
    audit.write_rows(raw_path, rows)
    selection.update({"trace_conversion_sha256": audit.sha(output/"conversion.json"),
                      "trace_downloads_sha256": audit.sha(trace_downloads),
                      "nvw_records": len(selected), "trace_records": len(trace_train),
                      "scoped_substitution": expected["substitution"],
                      "source_assets": "original sources preserved; regular immutable mixture copies"})
    return freeze_approved(raw_path, output, source_registry, benchmark_index, model_path, output,
                           expected["pilot_records"], stage="pilot", selection=selection)


def complete_pilot(parent, parent_source_root, trace_downloads, trace_cache_root, source_registry,
                   benchmark_index, model_path, output):
    """Reapprove parent + delta into approved/records.jsonl, with assets/manifest at output root."""
    parent, parent_source_root, output = Path(parent), Path(parent_source_root), Path(output)
    expected, registered = policy(), registry(source_registry)
    if output.exists() or (parent/"common_manifest.json").exists():
        raise ValueError("Require an incomplete immutable parent and a new completion directory")
    paths = ("proposed_records.jsonl", "processor_preflight.json", "preparation_receipt.json",
             "approved/manifest.json", "approved/records.jsonl", "approved/quarantine.jsonl",
             "records.jsonl", "conversion.json", "canonical_private.jsonl")
    frozen = {name: audit.sha(parent/name) for name in paths}
    preparation = json.loads((parent/"preparation_receipt.json").read_text())
    manifest = json.loads((parent/"approved/manifest.json").read_text())
    preflight = json.loads((parent/"processor_preflight.json").read_text())
    raw, approved = audit.read_rows(parent/"proposed_records.jsonl"), audit.read_rows(parent/"approved/records.jsonl")
    quarantined = audit.read_rows(parent/"approved/quarantine.jsonl")
    source_root = Path(__file__).parent
    old_sources = {name: audit.sha(parent_source_root/name) for name in ("train.py", "modules.py")}
    current_sources = {name: audit.sha(source_root/name) for name in old_sources}
    def collator(root):
        text = (root/"train.py").read_text()
        tree = ast.parse(text)
        definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AssistantCollator"]
        if len(definitions) != 1:
            raise ValueError("Exact original/current AssistantCollator source required")
        return {"ast": ast.dump(definitions[0], include_attributes=False),
                "source_sha256": hashlib.sha256(ast.get_source_segment(text, definitions[0]).encode()).hexdigest()}
    if (manifest.get("source_sha256") != old_sources or old_sources["modules.py"] != current_sources["modules.py"]
            or audit.sha(parent_source_root/"audit.py") != audit.sha(source_root/"audit.py")
            or audit.sha(parent_source_root/"sources.py") != audit.sha(source_root/"sources.py")
            or collator(parent_source_root) != collator(source_root)):
        raise ValueError("Parent source hashes or actual processor/collator implementation differ")
    if (preparation.get("status") != "INCOMPLETE_QUOTA" or preparation.get("stage") != "pilot"
            or preparation.get("mixture_policy_sha256") != audit.sha(POLICY_PATH)
            or preparation.get("expected_records") != expected["pilot_records"]
            or preparation.get("approved_records") != len(approved)
            or preparation.get("model_input_receipt_sha256") != frozen["processor_preflight.json"]
            or manifest.get("status") != "APPROVED" or manifest.get("sources") != registered
            or manifest.get("records_sha256") != frozen["approved/records.jsonl"]
            or manifest.get("record_count") != len(approved)
            or manifest.get("model_input_audit", {}).get("receipt_sha256") != frozen["processor_preflight.json"]
            or manifest.get("contamination_audit", {}).get("index_sha256") != audit.sha(benchmark_index)
            or preparation.get("selection", {}).get("trace_conversion_sha256") != frozen["conversion.json"]
            or preparation.get("selection", {}).get("trace_downloads_sha256") != audit.sha(trace_downloads)
            or preflight.get("records_sha256") != frozen["proposed_records.jsonl"]
            or preflight.get("producer_sha256") != audit.sha(source_root/"audit.py")
            or len(raw) != expected["pilot_records"] or joint_counts(approved) != expected["nvw_joint_views"]):
        raise ValueError("Corrupted or mismatched closed parent preparation evidence")
    raw_by_id = {row["id"]: row for row in raw}
    approved_ids, rejected_ids = {row["id"] for row in approved}, {row["id"] for row in quarantined}
    if (len(raw_by_id) != len(raw) or len(approved_ids) != len(approved)
            or len(rejected_ids) != len(quarantined) or approved_ids & rejected_ids
            or approved_ids | rejected_ids != set(raw_by_id)
            or preparation.get("quarantined_records") != len(rejected_ids)
            or manifest.get("contamination_audit", {}).get("quarantined") != len(rejected_ids)
            or any(raw_by_id[sid].get("source_id") != "trace" or "nvw_track" in raw_by_id[sid] for sid in rejected_ids)
            or not 0 < len(rejected_ids) < expected["trace_records"]):
        raise ValueError("Only excluded TRACE quarantine may create the pilot shortfall")
    for row in approved:
        original = dict(audit.model_input_record(raw_by_id[row["id"]]), training_allowed=True,
                        processor_receipt_sha256=frozen["processor_preflight.json"])
        if row != original:
            raise ValueError("Parent approved record differs from its processor-bound original")
    canonical = audit.read_rows(parent/"canonical_private.jsonl")
    nvw_roots = {row["root_id"] for row in raw if "nvw_track" in row}
    if (len(canonical) != len(nvw_roots) or {episode["root_id"] for episode in canonical} != nvw_roots
            or any(not isinstance(episode.get("image"), str) or not isinstance(episode.get("image_sha256"), str)
                   or episode.get("image_sha256") != manifest.get("files", {}).get(episode.get("image"))
                   or episode.get("image_sha256") != audit.sha(audit.image_path(parent, episode["image"]))
                   for episode in canonical)):
        raise ValueError("Parent canonical roots/images do not belong to the approved mixture")
    output.mkdir(parents=True, exist_ok=False)
    parent_approval = audit.approve(parent/"proposed_records.jsonl", parent, registered, benchmark_index,
                                   output/"parent_reapproved", parent/"processor_preflight.json")
    reused = audit.read_rows(output/"parent_reapproved/records.jsonl")
    if reused != approved or audit.read_rows(output/"parent_reapproved/quarantine.jsonl") != quarantined:
        raise ValueError("Parent reapproval changed records or quarantine; no quota rescue")
    approved_images = dict(parent_approval["files"])
    cached = trace_paths(trace_downloads, trace_cache_root)
    converted = sources.trace_records(cached, output/"trace_candidates", expected["seed"], train_limit=10**9)
    converted_by_id = {row["id"]: row for row in converted}
    parent_conversion = audit.read_rows(parent/"records.jsonl")
    if any(converted_by_id.get(row["id"]) != row for row in parent_conversion):
        raise ValueError("TRACE conversion identity/family split changed")
    used_roots = {row["root_id"] for row in raw}
    candidates = [row for row in converted if row["split"] == "train" and row["id"] not in raw_by_id
                  and row["root_id"] not in used_roots and row.get("source_id") == "trace"
                  and row.get("original_split") == "train" and "nvw_track" not in row]
    remaining, delta_rows, delta_evidence = len(rejected_ids), [], []
    while remaining and candidates:
        selected, candidates = candidates[:remaining], candidates[remaining:]
        delta_root = output/f"delta_{len(delta_evidence):03d}"
        additions, delta_manifest, delta_receipt, evidence = approve_delta(
            selected, output/"trace_candidates", registered, benchmark_index, model_path, delta_root)
        if processor_contract(delta_receipt) != processor_contract(preflight):
            raise ValueError("Replacement processor contract differs from closed parent")
        for name, digest in delta_manifest["files"].items():
            if name in approved_images and approved_images[name] != digest:
                raise ValueError("Approved parent/delta image-name collision")
            approved_images[name] = digest
        delta_rows.extend(additions)
        remaining -= len(additions)
        delta_evidence.append(evidence)
    if remaining:
        audit.dump(output/"completion_receipt.json", {"status": "INCOMPLETE_REPLACEMENT_QUOTA",
            "remaining": remaining, "deltas": delta_evidence, "training_allowed": False})
        raise ValueError("Unused approved TRACE candidates cannot satisfy the exact replacement quota")
    rows = order(reused + delta_rows, expected["seed"])
    universe = {row["id"]: row for row in converted if row["split"] != "train"}
    audit.check_splits(rows + list(universe.values()))
    if (len(rows) != expected["pilot_records"] or joint_counts(rows) != expected["nvw_joint_views"]
            or len([row for row in rows if row.get("source_id") == "trace"]) != expected["trace_records"]
            or any(row["id"] in rejected_ids or row["split"] != "train" for row in rows)):
        raise ValueError("Completed pilot total/joint quotas/splits differ")
    file_sources = {}
    for image_root, subset in ((parent, reused), (output/"trace_candidates", delta_rows)):
        for row in subset:
            for name in row["image_paths"]:
                file_sources.setdefault((str(image_root), name), image_root)
    files = {name: copy_image(image_root, output, name) for (_, name), image_root in file_sources.items()}
    if files != approved_images:
        raise ValueError("Copied images differ from closed parent/delta approval hashes")
    copy_image(parent, output, "canonical_private.jsonl")
    records_path = output/"approved/records.jsonl"
    audit.write_rows(records_path, rows)
    if any(audit.sha(parent/name) != digest for name, digest in frozen.items()):
        raise ValueError("Closed parent changed during completion")
    if any(audit.sha(source_root/name) != digest for name, digest in current_sources.items()):
        raise ValueError("Training source changed during completion")
    receipt = {"status": "READY", "stage": "pilot", "approved_records": len(rows),
        "parent_files_sha256": frozen, "parent_quarantine_excluded": sorted(rejected_ids),
        "parent_approved_reused": len(reused), "new_approved_trace": len(delta_rows), "deltas": delta_evidence,
        "source_binding": {"parent": old_sources, "current": current_sources,
                           "AssistantCollator_AST_equal": True,
                           "AssistantCollator_source_sha256": collator(source_root)["source_sha256"]},
        "canonical_private_sha256": audit.sha(output/"canonical_private.jsonl"),
        "training_allowed": True, "public_upload_allowed": False}
    audit.dump(output/"completion_receipt.json", receipt)
    common = dict(parent_approval, records_sha256=audit.sha(records_path), record_count=len(rows),
        files=files, image_root=str(output.resolve()), source_sha256=current_sources,
        stage="pilot", preparation_receipt_sha256=audit.sha(output/"completion_receipt.json"),
        canonical_private_sha256=receipt["canonical_private_sha256"],
        contamination_audit={**parent_approval["contamination_audit"],
            "quarantined": len(rejected_ids)+sum(item["quarantined"] for item in delta_evidence),
            "composite_approval_receipt_sha256": audit.sha(output/"completion_receipt.json")},
        model_input_audit={"status": "PASS", "kind": "COMPOSITE_PROCESSOR_RECEIPTS",
            "parent_receipt_sha256": frozen["processor_preflight.json"],
            "delta_receipts_sha256": [item["processor_receipt_sha256"] for item in delta_evidence],
            "parent_approved_records": len(reused), "delta_approved_records": len(delta_rows),
            "processor_revision": preflight["processor_revision"], "sequence_budget": preflight["sequence_budget"],
            "episode_pixels": preflight["episode_pixels"], "full_preflight_reexecuted": False, "record_count": len(rows)},
        mixture_policy=expected, mixture_policy_sha256=audit.sha(POLICY_PATH),
        record_order_sha256=hashlib.sha256(json.dumps([row["id"] for row in rows]).encode()).hexdigest(),
        module_arms=expected["module_arms"], same_records_order_and_images_for_all_arms=True,
        actual_arm_execution_verified=False, nvw_joint_views=joint_counts(rows),
        nvw_tracks={track: sum(values.values()) for track, values in joint_counts(rows).items()},
        records_path=str(records_path.resolve()))
    audit.dump(output/"common_manifest.json", common)
    return receipt


def approve_delta(rows, image_root, registered, benchmark_index, model_path, output):
    """The same real processor, rights and contamination gate for either replacement source."""
    output = Path(output)
    output.mkdir()
    proposed = output/"proposed_records.jsonl"
    audit.write_rows(proposed, rows)
    for row in rows:
        audit.check_lineage(row, registered)
        audit.check_target_identifiability(row, image_root)
    _, processor = audit.preflight_records(proposed, image_root, model_path, output/"processor_preflight.json")
    approved = audit.approve(proposed, image_root, registered, benchmark_index, output/"approved",
                             output/"processor_preflight.json")
    additions = audit.read_rows(output/"approved/records.jsonl")
    evidence = {"proposed_sha256": audit.sha(proposed),
        "processor_receipt_sha256": audit.sha(output/"processor_preflight.json"),
        "approved_manifest_sha256": audit.sha(output/"approved/manifest.json"),
        "approved_records_sha256": approved["records_sha256"], "accepted": len(additions),
        "quarantined": approved["contamination_audit"]["quarantined"]}
    return additions, approved, processor, evidence


def select_approved_subset(rows, count, seed=3407):
    """Keep approved bytes and fixed source/track/view ratios; never select by score."""
    if type(count) is not int or not 400 <= count < 10000 or count % 40:
        raise ValueError("A reduced common pilot requires 400..9960 records, divisible by 40")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Approved parent contains duplicate IDs")
    if any(row.get("source_id") not in {"trace", "nvw-independent-procedural-v1"} for row in rows):
        raise ValueError("Unexpected source in approved parent")
    trace = order([row for row in rows if row["source_id"] == "trace"], seed)[:count*9//10]
    i = count//40
    i_structure = (i+3)//4
    s_structure = i-i//2-i_structure
    matrix = {
        "NVW-K": {"qa": i, "structure": i//2, "grounded": (i+1)//2},
        "NVW-I": {"qa": i//2, "structure": i_structure, "grounded": i-i//2-i_structure},
        "NVW-S": {"qa": (i+1)//2, "structure": s_structure, "grounded": i-(i+1)//2-s_structure},
    }
    selected = list(trace)
    for track, by_view in matrix.items():
        for view, quota in by_view.items():
            candidates = [row for row in rows if row.get("source_id") == "nvw-independent-procedural-v1"
                          and row.get("nvw_track") == track and row.get("view") == view]
            if len(candidates) < quota:
                raise ValueError("Approved track/view shortfall; no borrowing or regeneration")
            selected.extend(order(candidates, seed)[:quota])
    if len(selected) != count or joint_counts(selected) != matrix:
        raise ValueError("Reduced pilot source/track/view count mismatch")
    audit.check_splits(selected)
    return order(selected, seed)


def closed_pilot(parent):
    """Verify the immutable full parent and its actual approved component chain."""
    from .train import approved_records
    from .modules import BASE_REVISION

    parent = Path(parent)
    parent_records, parent_manifest = parent/"approved/records.jsonl", parent/"common_manifest.json"
    frozen = {name: audit.sha(parent/name) for name in
              ("approved/records.jsonl", "common_manifest.json", "canonical_private.jsonl", "completion_receipt.json")}
    rows, manifest = approved_records(parent_records, parent_manifest, parent)
    completed = json.loads((parent/"completion_receipt.json").read_text())
    inputs = manifest.get("model_input_audit", {})
    if completed.get("kind") == "ROOT_DISJOINT_FULL_PILOT":
        return closed_disjoint_components(parent, rows, manifest, completed, frozen)
    if (completed.get("status") != "READY" or completed.get("stage") != "pilot"
            or completed.get("approved_records") != len(rows)
            or completed.get("parent_approved_reused", 0)+completed.get("new_approved_trace", 0) != len(rows)
            or completed.get("canonical_private_sha256") != frozen["canonical_private.jsonl"]
            or completed.get("source_binding", {}).get("current") != manifest.get("source_sha256")
            or manifest.get("preparation_receipt_sha256") != frozen["completion_receipt.json"]
            or manifest.get("contamination_audit", {}).get("composite_approval_receipt_sha256") != frozen["completion_receipt.json"]
            or inputs.get("status") != "PASS" or inputs.get("kind") != "COMPOSITE_PROCESSOR_RECEIPTS"
            or inputs.get("record_count") != len(rows) or inputs.get("processor_revision") != BASE_REVISION
            or inputs.get("sequence_budget") != 4096 or inputs.get("episode_pixels") != 1048576
            or inputs.get("parent_approved_records") != completed.get("parent_approved_reused")
            or inputs.get("delta_approved_records") != completed.get("new_approved_trace")
            or inputs.get("parent_receipt_sha256") != completed.get("parent_files_sha256", {}).get("processor_preflight.json")
            or inputs.get("delta_receipts_sha256") != [d["processor_receipt_sha256"] for d in completed.get("deltas", [])]
            or any(row["id"] in completed.get("parent_quarantine_excluded", []) for row in rows)):
        raise ValueError("Parent processor/completion provenance is not a closed matching PASS")
    component = "parent_reapproved/records.jsonl"
    component_files = {component: completed.get("parent_files_sha256", {}).get("approved/records.jsonl")}
    if audit.sha(audit.image_path(parent, component)) != component_files[component]:
        raise ValueError("Closed reapproved parent component bytes changed")
    reused = audit.read_rows(parent/component)
    additions = []
    if len(reused) != completed.get("parent_approved_reused"):
        raise ValueError("Closed approved components count mismatch")
    for index, delta in enumerate(completed.get("deltas", [])):
        folder = parent/f"delta_{index:03d}"
        for name, key in (("proposed_records.jsonl", "proposed_sha256"),
                          ("processor_preflight.json", "processor_receipt_sha256"),
                          ("approved/manifest.json", "approved_manifest_sha256"),
                          ("approved/records.jsonl", "approved_records_sha256")):
            if audit.sha(audit.image_path(folder, name)) != delta[key]:
                raise ValueError("Closed delta approval/input receipt bytes changed")
            component_files[f"delta_{index:03d}/{name}"] = delta[key]
        delta_rows = audit.read_rows(folder/"approved/records.jsonl")
        if len(delta_rows) != delta.get("accepted"):
            raise ValueError("Closed approved components count mismatch")
        additions.extend(delta_rows)
    if (len(additions) != completed.get("new_approved_trace")
            or order(reused + additions, policy()["seed"]) != rows):
        raise ValueError("Final parent rows differ from closed approved components")
    frozen.update(component_files)
    if (len(rows) != policy()["pilot_records"] or manifest.get("stage") != "pilot"
            or manifest.get("same_records_order_and_images_for_all_arms") is not True
            or manifest.get("module_arms") != policy()["module_arms"]
            or joint_counts(rows) != policy()["nvw_joint_views"]
            or Counter(row["source_id"] for row in rows) !=
               {"trace": 9000, "nvw-independent-procedural-v1": 1000}
            or manifest.get("canonical_private_sha256") != frozen["canonical_private.jsonl"]):
        raise ValueError("Reduced pilot requires the closed exact approved 10K parent")
    return rows, manifest, completed, frozen, component_files


def closed_disjoint_components(parent, rows, manifest, completed, frozen):
    """Retain reconstruction checks when a repaired full parent is later subsetted."""
    from .modules import BASE_REVISION

    inputs = manifest.get("model_input_audit", {})
    exclusion = completed.get("smoke_exclusion", {})
    original_files = completed.get("parent_files_sha256", {})
    sources = {name: completed.get("source_sha256", {}).get(name) for name in ("train.py", "modules.py")}
    if (completed.get("status") != "READY" or completed.get("stage") != "pilot"
            or completed.get("approved_records") != 10000 or len(rows) != 10000
            or completed.get("records_sha256") != frozen["approved/records.jsonl"]
            or completed.get("canonical_private_sha256") != frozen["canonical_private.jsonl"]
            or manifest.get("canonical_private_sha256") != frozen["canonical_private.jsonl"]
            or manifest.get("source_sha256") != sources
            or manifest.get("preparation_receipt_sha256") != frozen["completion_receipt.json"]
            or manifest.get("contamination_audit", {}).get("composite_approval_receipt_sha256") != frozen["completion_receipt.json"]
            or manifest.get("smoke_exclusion") != exclusion
            or inputs.get("status") != "PASS" or inputs.get("kind") != "INHERITED_PARENT_AND_NVW_DELTA"
            or inputs.get("record_count") != len(rows) or inputs.get("processor_revision") != BASE_REVISION
            or inputs.get("sequence_budget") != 4096 or inputs.get("episode_pixels") != 1048576
            or inputs.get("parent_manifest_sha256") != original_files.get("common_manifest.json")
            or inputs.get("parent_approved_records") != completed.get("parent_approved_reused")
            or inputs.get("delta_approved_records") != completed.get("new_approved_nvw")
            or inputs.get("delta_receipts_sha256") != [d["processor_receipt_sha256"] for d in completed.get("deltas", [])]
            or manifest.get("stage") != "pilot" or manifest.get("module_arms") != policy()["module_arms"]
            or manifest.get("same_records_order_and_images_for_all_arms") is not True
            or joint_counts(rows) != policy()["nvw_joint_views"]
            or Counter(row["source_id"] for row in rows) != {"trace": 9000, "nvw-independent-procedural-v1": 1000}):
        raise ValueError("Root-disjoint full parent processor/completion provenance mismatch")
    components = {"parent_evidence/"+name: digest for name, digest in original_files.items()}
    components.update({exclusion["evidence_path"]: exclusion["records_sha256"],
                       exclusion["manifest_evidence_path"]: exclusion["manifest_sha256"],
                       exclusion["canonical_evidence_path"]: exclusion["canonical_sha256"]})
    additions = []
    for index, delta in enumerate(completed.get("deltas", [])):
        folder = f"delta_{index:03d}"
        for name, key in (("proposed_records.jsonl", "proposed_sha256"),
                          ("processor_preflight.json", "processor_receipt_sha256"),
                          ("approved/manifest.json", "approved_manifest_sha256"),
                          ("approved/records.jsonl", "approved_records_sha256")):
            components[f"{folder}/{name}"] = delta[key]
        group = audit.read_rows(parent/folder/"approved/records.jsonl")
        if len(group) != delta["accepted"]:
            raise ValueError("Root-disjoint approved delta count changed")
        additions.extend(group)
    for name, digest in components.items():
        if audit.sha(audit.image_path(parent, name)) != digest:
            raise ValueError("Root-disjoint approved component bytes changed")
    original = audit.read_rows(parent/"parent_evidence/approved/records.jsonl")
    smoke_rows = audit.read_rows(parent/exclusion["evidence_path"])
    verified_canonical_roots(rows, parent/"canonical_private.jsonl")
    verified_canonical_roots(original, parent/"parent_evidence/canonical_private.jsonl")
    verified_canonical_roots(smoke_rows, parent/exclusion["canonical_evidence_path"])
    retained, observed, _path, _smoke = smoke_exclusion(original, parent/exclusion["evidence_path"])
    if (len(retained) != completed["parent_approved_reused"] or len(additions) != completed["new_approved_nvw"]
            or observed["records_sha256"] != exclusion["records_sha256"]
            or len(original)-len(retained) != exclusion["excluded_parent_records"]
            or order(retained + additions) != rows
            or {row["root_id"] for row in rows} & {row["root_id"] for row in smoke_rows}):
        raise ValueError("Root-disjoint final rows differ from closed approved components")
    frozen.update(components)
    return rows, manifest, completed, frozen, components


def smoke_exclusion(rows, smoke_records):
    """Exclude roots only; never admit a record from the supplied smoke ledger."""
    smoke_path = Path(smoke_records)
    smoke_path = audit.image_path(smoke_path.parent, smoke_path.name)
    smoke_hash = audit.sha(smoke_path)
    smoke_rows = audit.read_rows(smoke_path)
    if not smoke_rows or any(not isinstance(row, dict)
            or row.get("source_id") != "nvw-independent-procedural-v1"
            or row.get("source_kind") != "INDEPENDENT_SYNTHETIC"
            or row.get("lineage") != ["independent-nvw-v1"] or row.get("parents") != []
            or row.get("split") != "train" or row.get("training_allowed") is not True
            or row.get("benchmark_derived") is not False
            or row.get("view") not in {"qa", "structure", "grounded"}
            or any(not isinstance(row.get(key), str) or not row[key].strip()
                   for key in ("id", "root_id", "family_id", "user_text", "target_text"))
            or not isinstance(row.get("image_paths"), list) or not row["image_paths"]
            or any(not isinstance(name, str) or not name or Path(name).is_absolute()
                   or ".." in Path(name).parts for name in row["image_paths"])
            for row in smoke_rows):
        raise ValueError("Smoke exclusion requires valid approved own-NVW train rows")
    audit.check_splits(smoke_rows)
    smoke_roots = {row["root_id"] for row in smoke_rows}
    eligible = [row for row in rows if row["root_id"] not in smoke_roots]
    excluded = [row for row in rows if row["root_id"] in smoke_roots]
    exclusion = {"records_sha256": smoke_hash, "record_count": len(smoke_rows),
        "root_count": len(smoke_roots), "excluded_parent_records": len(excluded),
        "excluded_parent_roots": len({row["root_id"] for row in excluded}),
        "remaining_parent_records": len(eligible),
        "root_ids_sha256": hashlib.sha256(json.dumps(sorted(smoke_roots)).encode()).hexdigest(),
        "evidence_path": "smoke_exclusion/"+smoke_path.name}
    return eligible, exclusion, smoke_path, smoke_rows


def subset_pilot(parent, output, count, reason, exclude_smoke_records=None):
    """New common subset of the approved 10K; keep the full parent and protocol unchanged."""
    parent, output = Path(parent), Path(output)
    if output.exists() or not isinstance(reason, str) or not reason.strip():
        raise ValueError("New output and explicit scale-change reason required")
    rows, manifest, _completed, frozen, component_files = closed_pilot(parent)
    eligible, exclusion = rows, None
    if exclude_smoke_records is not None:
        eligible, exclusion, smoke_path, _smoke_rows = smoke_exclusion(rows, exclude_smoke_records)
        smoke_hash = exclusion["records_sha256"]
    selected = select_approved_subset(eligible, count, policy()["seed"])
    output.mkdir(parents=True, exist_ok=False)
    evidence_root = output/"component_evidence"
    evidence_root.mkdir()
    for name, digest in component_files.items():
        if copy_image(parent, evidence_root, name) != digest:
            raise ValueError("Copied approved component bytes changed")
    if exclusion is not None:
        exclusion_root = output/"smoke_exclusion"
        exclusion_root.mkdir()
        if copy_image(smoke_path.parent, exclusion_root, smoke_path.name) != smoke_hash:
            raise ValueError("Smoke exclusion ledger changed during subset creation")
    files = {name: copy_image(parent, output, name) for name in
             sorted({name for row in selected for name in row["image_paths"]})}
    if any(digest != manifest["files"].get(name) for name, digest in files.items()):
        raise ValueError("Subset image differs from closed parent approval")
    copy_image(parent, output, "canonical_private.jsonl")
    records = output/"approved/records.jsonl"
    audit.write_rows(records, selected)
    tracks = {track: sum(values.values()) for track, values in joint_counts(selected).items()}
    scale = {"status": "APPROVED_COMMON_SUBSET", "original_records": len(rows),
        "selected_records": count, "parent_manifest_sha256": frozen["common_manifest.json"],
        "parent_records_sha256": frozen["approved/records.jsonl"],
        "selection_policy": "Per-source and canonical track/view SHA256 COMMON order, independent of evaluation outcomes",
        "selection_seed": policy()["seed"], "reason": reason,
        "source_counts": {"trace": count*9//10, "nvw": count//10}, "nvw_track_counts": tracks,
        "original_10K_execution_completed": False}
    if exclusion is not None:
        scale["smoke_exclusion"] = exclusion
        scale["selection_policy"] += "; exclude every supplied smoke root before quota selection"
    audit.dump(output/"subset_receipt.json", {"status": "READY", "pilot_scale": scale,
        "records_sha256": audit.sha(records), "parent_files_sha256": frozen,
        "component_files_sha256": component_files,
        "smoke_exclusion": exclusion,
        "actual_processor_reexecuted": False, "training_allowed": True, "public_upload_allowed": False})
    common = dict(manifest, records_path=str(records.resolve()), records_sha256=audit.sha(records),
        record_count=count, image_root=str(output.resolve()), files=files, pilot_scale=scale,
        record_order_sha256=hashlib.sha256(json.dumps([row["id"] for row in selected]).encode()).hexdigest(),
        nvw_tracks=tracks, nvw_joint_views=joint_counts(selected),
        preparation_receipt_sha256=audit.sha(output/"subset_receipt.json"), actual_arm_execution_verified=False,
        model_input_audit={**manifest["model_input_audit"], "kind": "INHERITED_APPROVED_SUBSET",
            "parent_record_count": len(rows), "record_count": count,
            "parent_manifest_sha256": frozen["common_manifest.json"], "processor_reexecuted": False})
    if any(audit.sha(parent/name) != digest for name, digest in frozen.items()):
        raise ValueError("Closed approved parent changed during subset creation")
    if exclusion is not None and audit.sha(smoke_path) != smoke_hash:
        raise ValueError("Smoke exclusion ledger changed during subset creation")
    audit.dump(output/"common_manifest.json", common)
    return scale


def disjoint_full_pilot(parent, nvw_root, smoke_records, smoke_manifest, source_registry,
                        benchmark_index, model_path, output):
    """Replace smoke-overlapping NVW only; approve a NEW full 10K, never expand a reduced run."""
    from .modules import BASE_REVISION

    parent, nvw_root, output = Path(parent), Path(nvw_root), Path(output)
    if output.exists() or output.resolve().is_relative_to(parent.resolve()):
        raise ValueError("A new output outside the immutable full parent is required")
    rows, manifest, completed, frozen, _components = closed_pilot(parent)
    parent_canonical = verified_canonical_roots(rows, parent/"canonical_private.jsonl")
    smoke_manifest = Path(smoke_manifest)
    smoke_manifest = audit.image_path(smoke_manifest.parent, smoke_manifest.name)
    smoke_manifest_hash = audit.sha(smoke_manifest)
    smoke_path = Path(smoke_records)
    if smoke_path.name in {smoke_manifest.name, "canonical_private.jsonl"} or smoke_manifest.name == "canonical_private.jsonl":
        raise ValueError("Smoke record and manifest evidence filenames must be distinct")
    smoke_approval = json.loads(smoke_manifest.read_text())
    smoke_canonical_path = audit.image_path(smoke_approval["image_root"], "canonical_private.jsonl")
    smoke_canonical_hash = audit.sha(smoke_canonical_path)
    verified_canonical_roots(audit.read_rows(smoke_path), smoke_canonical_path)
    reused, exclusion, smoke_path, smoke_rows = smoke_exclusion(rows, smoke_records)
    smoke_inputs = smoke_approval.get("model_input_audit", {})
    smoke_roots = {row["root_id"] for row in smoke_rows}
    if (len(smoke_rows) != policy()["smoke_views"] or len(smoke_roots) != policy()["smoke_roots"]
            or any({row["view"] for row in smoke_rows if row["root_id"] == root}
                   != {"qa", "structure", "grounded"} for root in smoke_roots)
            or smoke_approval.get("status") != "APPROVED" or smoke_approval.get("stage") != "smoke"
            or smoke_approval.get("training_allowed") is not True or smoke_approval.get("benchmark_derived") is not False
            or smoke_approval.get("records_sha256") != exclusion["records_sha256"]
            or smoke_approval.get("record_count") != len(smoke_rows)
            or smoke_approval.get("source_sha256") != manifest["source_sha256"]
            or smoke_approval.get("contamination_audit", {}).get("status") != "PASS"
            or smoke_inputs.get("status") != "PASS" or smoke_inputs.get("record_count") != len(smoke_rows)
            or smoke_inputs.get("processor_revision") != BASE_REVISION
            or smoke_inputs.get("sequence_budget") != 4096 or smoke_inputs.get("episode_pixels") != 1048576
            or not re.fullmatch(r"[0-9a-f]{64}", str(smoke_inputs.get("receipt_sha256", "")))):
        raise ValueError("Exact approved 120-root smoke ledger/manifest binding required")
    removed = [row for row in rows if row["root_id"] in smoke_roots]
    if any(row["source_id"] != "nvw-independent-procedural-v1" for row in removed):
        raise ValueError("Smoke root removal may replace only own NVW, never TRACE")
    registered = registry(source_registry)
    if registered != manifest.get("sources") or audit.sha(benchmark_index) != manifest["contamination_audit"]["index_sha256"]:
        raise ValueError("Replacement rights registry/exclusion index differs from closed parent")
    pool_files = {name: audit.sha(audit.image_path(nvw_root, name)) for name in
                  ("manifest.json", "canonical_private.jsonl", "prototype_train.jsonl",
                   "prototype_development.jsonl", "prototype_sealed_holdout.jsonl")}
    verified = nvw.verify_dataset(nvw_root)
    pool_manifest = json.loads((nvw_root/"manifest.json").read_text())
    if pool_manifest.get("generator_sha256") != registered.get("independent-nvw-v1", {}).get("generator_sha256"):
        raise ValueError("NVW pool generator is not the rights-approved source")
    universe = [row for name in ("prototype_train.jsonl", "prototype_development.jsonl", "prototype_sealed_holdout.jsonl")
                for row in audit.read_rows(nvw_root/name)]
    canonical = verified_canonical_roots(universe, nvw_root/"canonical_private.jsonl")
    audit.check_splits(universe)
    # complete_pilot already required each TRACE receipt to equal the original
    # actual parent receipt. Use a closed representative, not a new measurement.
    inherited_names = [name for name in frozen if name.endswith("/processor_preflight.json")]
    if not inherited_names:
        raise ValueError("Closed representative inherited actual processor receipt required")
    inherited_name = inherited_names[0]
    inherited = json.loads(audit.image_path(parent, inherited_name).read_text())
    inherited_proposed = str(Path(inherited_name).with_name("proposed_records.jsonl"))
    if (inherited.get("status") != "PASS" or inherited.get("kind") != "ACTUAL_PROCESSOR_PREFLIGHT"
            or inherited.get("truncation") is not False or not inherited.get("chat_template_sha256")
            or inherited.get("processor_revision") != BASE_REVISION or inherited.get("processor_class") != "Qwen3VLProcessor"
            or inherited.get("sequence_budget") != 4096 or inherited.get("episode_pixels") != 1048576
            or inherited.get("producer_sha256") != audit.sha(Path(__file__).with_name("audit.py"))
            or inherited.get("records_sha256") != frozen.get(inherited_proposed)):
        raise ValueError("Invalid closed representative inherited actual processor receipt")
    excluded_roots = {row["root_id"] for row in rows} | smoke_roots
    candidates = []
    for row in universe:
        if row["split"] != "train" or row["root_id"] in excluded_roots or (row["view"] == "qa" and row.get("answer_identifiable") is not True):
            continue
        if (row.get("source_id") != "nvw-independent-procedural-v1" or row.get("source_kind") != "INDEPENDENT_SYNTHETIC"
                or row.get("lineage") != ["independent-nvw-v1"] or row.get("parents") != []):
            raise ValueError("Only unused independent own-NVW train candidates may fill the deficit")
        candidates.append(dict(row, nvw_track=canonical[row["root_id"]]["track"],
                               nvw_track_source_sha256=pool_files["canonical_private.jsonl"]))
    deficit = joint_counts(removed)
    available = joint_counts(candidates)
    if any(available[t][v] < q for t, values in deficit.items() for v, q in values.items()):
        raise ValueError("Unused own-NVW train pool joint quota shortfall; no source/track borrowing")
    audit.check_splits(reused + [row for row in universe if row["split"] != "train"])
    output.mkdir(parents=True, exist_ok=False)
    exclusion.update(manifest_sha256=smoke_manifest_hash,
        manifest_evidence_path="smoke_exclusion/"+smoke_manifest.name,
        canonical_sha256=smoke_canonical_hash, canonical_evidence_path="smoke_exclusion/canonical_private.jsonl",
        excluded_parent_ids=[row["id"] for row in removed],
        exclusion_basis="shared own-NVW semantic root IDs; not an assertion of identical rendered inputs")
    draft = {"status": "PENDING_ACTUAL_PROCESSOR_AND_APPROVAL", "stage": "pilot", "target_records": 10000,
        "replacement_joint_views": deficit, "smoke_exclusion": exclusion,
        "pool_files_sha256": pool_files, "pool_verification": verified,
        "source_sha256": {n: audit.sha(Path(__file__).with_name(n)) for n in
                          ("prepare_training.py", "audit.py", "nvw.py", "train.py", "modules.py")},
        "source_registry_sha256": audit.sha(source_registry), "benchmark_index_sha256": audit.sha(benchmark_index),
        "representative_inherited_processor": {"path": inherited_name, "sha256": frozen[inherited_name],
                                               "contract": processor_contract(inherited), "reexecuted": False},
        "training_allowed": False, "public_upload_allowed": False}
    audit.dump(output/"candidate_preparation.json", draft)
    if not Path(model_path).is_dir() or Path(model_path).name != BASE_REVISION:
        raise ValueError("Actual pinned CPU processor snapshot is unavailable; candidates remain unapproved")
    additions, deltas, delta_files = [], [], {}
    while any(q for values in deficit.values() for q in values.values()):
        selected = []
        for track, values in deficit.items():
            for view, quota in values.items():
                choices = order([row for row in candidates if row["nvw_track"] == track and row["view"] == view])
                if len(choices) < quota:
                    audit.dump(output/"completion_receipt.json", {**draft, "status": "INCOMPLETE_REPLACEMENT_QUOTA",
                        "remaining_joint_views": deficit, "deltas": deltas})
                    raise ValueError("Approved NVW replacement quota shortfall; quarantine is never waived")
                selected.extend(choices[:quota])
        tried_roots = {row["root_id"] for row in selected}
        candidates = [row for row in candidates if row["root_id"] not in tried_roots]
        accepted, approval, processor, evidence = approve_delta(order(selected), nvw_root, registered,
            benchmark_index, model_path, output/f"delta_{len(deltas):03d}")
        if processor_contract(processor) != processor_contract(inherited):
            raise ValueError("Replacement processor contract differs from closed parent")
        additions.extend(accepted)
        deltas.append(evidence)
        for name, digest in approval["files"].items():
            if name in delta_files and delta_files[name] != digest:
                raise ValueError("NVW delta image-name collision")
            delta_files[name] = digest
        accepted_joint = joint_counts(accepted)
        deficit = {t: {v: q-accepted_joint[t][v] for v, q in values.items()} for t, values in deficit.items()}
    final = order(reused + additions)
    audit.check_splits(final + [row for row in universe if row["split"] != "train"])
    if (len(final) != 10000 or joint_counts(final) != policy()["nvw_joint_views"]
            or Counter(row["source_id"] for row in final) != {"trace": 9000, "nvw-independent-procedural-v1": 1000}
            or {row["root_id"] for row in final} & smoke_roots):
        raise ValueError("Full 10K source/joint/root-disjoint contract failed")
    files = {}
    for image_root, group, approved_files in ((parent, reused, manifest["files"]), (nvw_root, additions, delta_files)):
        for name in sorted({name for row in group for name in row["image_paths"]}):
            digest = copy_image(image_root, output, name)
            if digest != approved_files.get(name) or (name in files and files[name] != digest):
                raise ValueError("Full pilot images differ from closed approval")
            files[name] = digest
    reused_roots = {row["root_id"] for row in reused if row["source_id"] != "trace"}
    new_roots = {row["root_id"] for row in additions}
    final_canonical = [parent_canonical[root] for root in sorted(reused_roots)] + [canonical[root] for root in sorted(new_roots)]
    if any(episode["image_sha256"] != files.get(episode["image"]) for episode in final_canonical):
        raise ValueError("Canonical episode image ownership differs from full approval")
    audit.write_rows(output/"canonical_private.jsonl", final_canonical)
    records = output/"approved/records.jsonl"
    audit.write_rows(records, final)
    evidence_root = output/"parent_evidence"
    evidence_root.mkdir()
    for name, digest in frozen.items():
        if copy_image(parent, evidence_root, name) != digest:
            raise ValueError("Closed parent evidence changed during completion")
    smoke_root = output/"smoke_exclusion"
    smoke_root.mkdir()
    for source, digest in ((smoke_path, exclusion["records_sha256"]), (smoke_manifest, smoke_manifest_hash),
                           (smoke_canonical_path, smoke_canonical_hash)):
        if copy_image(source.parent, smoke_root, source.name) != digest:
            raise ValueError("Smoke exclusion ledger/manifest changed during completion")
    receipt = {**draft, "status": "READY", "kind": "ROOT_DISJOINT_FULL_PILOT", "approved_records": len(final),
        "records_sha256": audit.sha(records), "parent_files_sha256": frozen,
        "parent_approved_reused": len(reused), "new_approved_nvw": len(additions), "deltas": deltas,
        "canonical_private_sha256": audit.sha(output/"canonical_private.jsonl"),
        "training_allowed": True, "original_10K_execution_completed": False}
    if (any(audit.sha(parent/name) != digest for name, digest in frozen.items())
            or any(audit.sha(nvw_root/name) != digest for name, digest in pool_files.items())
            or any(audit.sha(Path(__file__).with_name(name)) != digest for name, digest in draft["source_sha256"].items())
            or audit.sha(source_registry) != draft["source_registry_sha256"]
            or audit.sha(benchmark_index) != draft["benchmark_index_sha256"]
            or audit.sha(smoke_path) != exclusion["records_sha256"] or audit.sha(smoke_manifest) != smoke_manifest_hash
            or audit.sha(smoke_canonical_path) != smoke_canonical_hash):
        raise ValueError("Immutable parent/pool/smoke/source changed during completion")
    audit.dump(output/"completion_receipt.json", receipt)
    audit.dump(output/"common_manifest.json", dict(manifest, records_path=str(records.resolve()),
        records_sha256=audit.sha(records), record_count=len(final), image_root=str(output.resolve()), files=files,
        canonical_private_sha256=receipt["canonical_private_sha256"],
        preparation_receipt_sha256=audit.sha(output/"completion_receipt.json"), actual_arm_execution_verified=False,
        record_order_sha256=hashlib.sha256(json.dumps([row["id"] for row in final]).encode()).hexdigest(),
        smoke_exclusion=exclusion,
        contamination_audit={**manifest["contamination_audit"],
            "quarantined": manifest["contamination_audit"].get("quarantined", 0)+sum(d["quarantined"] for d in deltas),
            "composite_approval_receipt_sha256": audit.sha(output/"completion_receipt.json")},
        model_input_audit={"status": "PASS", "kind": "INHERITED_PARENT_AND_NVW_DELTA",
            "parent_manifest_sha256": frozen["common_manifest.json"], "parent_approved_records": len(reused),
            "delta_approved_records": len(additions), "delta_receipts_sha256": [d["processor_receipt_sha256"] for d in deltas],
            "record_count": len(final), "processor_revision": BASE_REVISION, "sequence_budget": 4096,
            "episode_pixels": 1048576, "full_preflight_reexecuted": False}))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    index = sub.add_parser("index")
    index.add_argument("--coverage", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    for stage in ("smoke", "pilot"):
        command = sub.add_parser(stage)
        for name in ("nvw-root", "source-registry", "benchmark-index", "model-path", "output"):
            command.add_argument("--"+name, type=Path, required=True)
        if stage == "pilot":
            command.add_argument("--trace-downloads", type=Path, required=True)
            command.add_argument("--trace-cache-root", type=Path, required=True)
    complete = sub.add_parser("complete-pilot")
    for name in ("parent", "parent-source-root", "trace-downloads", "trace-cache-root", "source-registry",
                 "benchmark-index", "model-path", "output"):
        complete.add_argument("--"+name, type=Path, required=True)
    subset = sub.add_parser("subset-pilot")
    subset.add_argument("--parent", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--count", type=int, required=True)
    subset.add_argument("--reason", required=True)
    subset.add_argument("--exclude-smoke-records", type=Path)
    disjoint = sub.add_parser("disjoint-full-pilot")
    for name in ("parent", "nvw-root", "smoke-records", "smoke-manifest", "source-registry",
                 "benchmark-index", "model-path", "output"):
        disjoint.add_argument("--"+name, type=Path, required=True)
    args = vars(parser.parse_args())
    stage = args.pop("stage")
    operation = {"index": build_exclusion_index, "smoke": prepare_smoke, "pilot": prepare_pilot,
                 "complete-pilot": complete_pilot, "subset-pilot": subset_pilot,
                 "disjoint-full-pilot": disjoint_full_pilot}[stage]
    if stage == "index":
        args["coverage_path"] = args.pop("coverage")
    print(json.dumps(operation(**args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
