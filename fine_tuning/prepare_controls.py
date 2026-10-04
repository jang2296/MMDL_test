"""Prepare the frozen NVW 0% control on Pod; no downloads, inference or training."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import audit, prepare_training
from .train import approved_records


def select_trace_control(parent_rows, reference_rows, seed=3407):
    """Replace only the approved reference's NVW quota, independently of scores."""
    if any(row.get("split") != "train" for rows in (parent_rows, reference_rows) for row in rows):
        raise ValueError("Mixture control requires only existing train-split records")
    if reference_rows != prepare_training.select_approved_subset(parent_rows, len(reference_rows), seed):
        raise ValueError("Reference differs from the frozen 10% parent selection")
    common = [row for row in reference_rows if row["source_id"] == "trace"]
    roots = {row["root_id"] for row in reference_rows}
    # Treatments share training families; train/dev/holdout separation is unchanged.
    candidates = [row for row in parent_rows if row["source_id"] == "trace"
                  and row["root_id"] not in roots]
    required = len(reference_rows)-len(common)
    if len(candidates) < required:
        raise ValueError("Insufficient distinct approved TRACE replacements")
    control = prepare_training.order(common+prepare_training.order(candidates, seed)[:required], seed)
    audit.check_splits(control)
    return control


def prepare(parent, reference, output):
    parent, reference, output = Path(parent).resolve(strict=True), Path(reference).resolve(strict=True), Path(output)
    if (output.exists() or output.is_symlink()
            or any(output.resolve().is_relative_to(root) for root in (parent, reference))):
        raise ValueError("A new control output outside both immutable input directories is required")
    rows, manifest, completed, frozen, _components = prepare_training.closed_pilot(parent)
    reference_files = {name: audit.sha(audit.image_path(reference, name)) for name in
                       ("approved/records.jsonl", "common_manifest.json", "subset_receipt.json")}
    reference_rows, reference_manifest = approved_records(
        reference/"approved/records.jsonl", reference/"common_manifest.json", reference)
    reference_receipt = json.loads((reference/"subset_receipt.json").read_text())
    if (reference_manifest.get("preparation_receipt_sha256") != reference_files["subset_receipt.json"]
            or reference_receipt.get("status") != "READY"
            or reference_receipt.get("records_sha256") != reference_files["approved/records.jsonl"]
            or reference_receipt.get("parent_files_sha256") != frozen
            or reference_receipt.get("pilot_scale") != reference_manifest.get("pilot_scale")
            or reference_manifest.get("source_sha256") != manifest.get("source_sha256")
            or any(reference_manifest["files"].get(name) != manifest["files"].get(name)
                   for row in reference_rows for name in row["image_paths"])):
        raise ValueError("Reference approval does not bind the same immutable full parent")
    eligible = rows
    exclusion = reference_receipt.get("smoke_exclusion")
    if exclusion is not None:
        if not isinstance(exclusion, dict):
            raise ValueError("Malformed reference smoke exclusion")
        name = exclusion.get("evidence_path")
        smoke_path = audit.image_path(reference, name)
        eligible, observed, _path, smoke_rows = prepare_training.smoke_exclusion(rows, smoke_path)
        if observed != exclusion or reference_manifest["pilot_scale"].get("smoke_exclusion") != exclusion:
            raise ValueError("Reference smoke exclusion differs from its immutable ledger")
        for row in smoke_rows:
            audit.check_lineage(row, manifest["sources"])
        reference_files[name] = audit.sha(smoke_path)
    elif completed.get("kind") != "ROOT_DISJOINT_FULL_PILOT":
        raise ValueError("A closed smoke-root-disjoint parent or verified reference smoke exclusion is required")
    seed = prepare_training.policy()["seed"]
    control = select_trace_control(eligible, reference_rows, seed)
    audit.check_splits(control+[
        row for row in reference_rows if row["source_id"] != "trace"])
    for row in control:
        audit.check_lineage(row, manifest["sources"])
    output.mkdir(parents=True, exist_ok=False)
    files = {name: prepare_training.copy_image(parent, output, name) for name in
             sorted({name for row in control for name in row["image_paths"]})}
    if any(digest != manifest["files"].get(name) for name, digest in files.items()):
        raise ValueError("Control image differs from its closed approval")
    for folder, source, frozen_files in (("parent_evidence", parent, frozen),
                                         ("reference_evidence", reference, reference_files)):
        evidence_root = output/folder
        evidence_root.mkdir()
        for name, digest in frozen_files.items():
            if prepare_training.copy_image(source, evidence_root, name) != digest:
                raise ValueError("Approval evidence changed during copying")
    records = output/"approved/records.jsonl"
    audit.write_rows(records, control)
    common = [row for row in reference_rows if row["source_id"] == "trace"]
    receipt = {
        "status": "DATA_PREPARATION_COMPLETED", "kind": "TRACE_NVW_0_VS_10_PERCENT_CONTROL",
        "selection_seed": seed, "record_count_per_condition": len(control),
        "common_unchanged_TRACE_records": len(common), "replacement_TRACE_records": len(control)-len(common),
        "replacement_roots_disjoint_from_reference": True,
        "replacement_train_families_may_overlap_reference": True,
        "control_source_counts": dict(Counter(row["source_id"] for row in control)),
        "reference_source_counts": dict(Counter(row["source_id"] for row in reference_rows)),
        "common_records_sha256": hashlib.sha256(json.dumps(common, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "records_sha256": audit.sha(records), "parent_files_sha256": frozen,
        "reference": str(reference), "reference_files_sha256": reference_files,
        "reference_smoke_exclusion": exclusion,
        "actual_processor_reexecuted": False, "GPU_training_executed": False,
        "professional_bridge_tested": False, "fixed_randomized_mapping_tested": False,
        "source_sha256": audit.sha(__file__), "training_allowed": True, "public_upload_allowed": False,
        "limitations": ["TRACE/NVW mixture substitution, not professional transfer or plan A0-A3.",
                        "Shared training families between treatments do not establish new-family generalization.",
                        "Equal record counts do not imply equal input/target token exposure.",
                        "Choose one module setting before training both mixture conditions."],
    }
    if (any(audit.sha(parent/name) != digest for name, digest in frozen.items())
            or any(audit.sha(reference/name) != digest for name, digest in reference_files.items())):
        raise ValueError("Immutable parent or reference changed during control preparation")
    audit.dump(output/"control_receipt.json", receipt)
    policy = {"kind": receipt["kind"], "nvw_fraction": 0, "record_count": len(control), "seed": seed}
    audit.dump(output/"mixture_policy.json", policy)
    common_manifest = dict(manifest,
        dataset_id="nvw-private-trace-only-control-v1", stage="nvw_mixture_control",
        records_path=str(records.resolve()), records_sha256=audit.sha(records), record_count=len(control),
        image_root=str(output.resolve()), files=files, actual_arm_execution_verified=False,
        record_order_sha256=hashlib.sha256(json.dumps([row["id"] for row in control]).encode()).hexdigest(),
        mixture_policy=policy, mixture_policy_sha256=audit.sha(output/"mixture_policy.json"),
        nvw_tracks={track: 0 for track in ("NVW-K", "NVW-I", "NVW-S")},
        nvw_joint_views=prepare_training.joint_counts(control),
        preparation_receipt_sha256=audit.sha(output/"control_receipt.json"),
        model_input_audit={"status": "PASS", "kind": "INHERITED_APPROVED_TRACE_CONTROL",
            "record_count": len(control), "parent_record_count": len(rows),
            "parent_manifest_sha256": frozen["common_manifest.json"], "processor_reexecuted": False,
            **{key: manifest["model_input_audit"][key] for key in
               ("processor_revision", "sequence_budget", "episode_pixels")}})
    for key in ("canonical_private_sha256", "pilot_scale", "smoke_exclusion"):
        common_manifest.pop(key, None)
    audit.dump(output/"common_manifest.json", common_manifest)
    approved_records(records, output/"common_manifest.json", output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "reference", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
