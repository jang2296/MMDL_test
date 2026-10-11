"""Freeze a private TabMWP A/B trial; CPU approval is a separate reviewed action."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from . import audit, prepare_training, tabmwp, tabmwp_images


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unchanged(bound):
    if any(not Path(name).is_file() or audit.sha(name) != expected for name, expected in bound.items()):
        raise ValueError("Frozen input changed during preparation/approval")


def native_bundle(prepared_root, image_root):
    """Replay native conversion and every pinned image, including dev/sealed images."""
    prepared, images = Path(prepared_root), Path(image_root)
    paths, bound, preparation_hash = tabmwp_images.selection(prepared)
    identity = read_json(images / "acquisition.json")
    expected = {"revision": tabmwp.REVISION, "preparation_hash": preparation_hash,
                "selected_paths": paths, "source_and_prepared_sha256": bound}
    if identity != expected:
        raise ValueError("Native image acquisition identity differs")
    receipts = sorted(images.glob("receipt_*.json"))
    if not receipts:
        raise ValueError("Missing native COMPLETE image receipt")
    receipt_path = receipts[-1]
    receipt = read_json(receipt_path)
    if (receipt.get("status") != "COMPLETE" or receipt.get("revision") != tabmwp.REVISION
            or receipt.get("preparation_hash") != preparation_hash
            or receipt.get("tree_sha1") != tabmwp_images.TREE_SHA
            or receipt.get("selection_sha256") != tabmwp.digest(paths)
            or receipt.get("expected_count") != 2000 or receipt.get("verified_count") != 2000
            or receipt.get("errors") or receipt.get("changed_sources")
            or set(receipt.get("files", {})) != set(paths)
            or any(receipt.get(k) is not False for k in (
                "rights_approved", "training_allowed", "public_upload_allowed",
                "human_semantic_certification", "pixel_semantics_verified"))):
        raise ValueError("Native image receipt is incomplete or overclaims approval")
    for name, key in (("tree.json", "tree_metadata_sha256"), ("commit.json", "commit_metadata_sha256")):
        if audit.sha(audit.image_path(images, name)) != receipt.get(key):
            raise ValueError("Frozen native Git metadata differs")
    entries = tabmwp_images.verify_tree(read_json(images / "commit.json"), read_json(images / "tree.json"))
    actual_paths = {p.relative_to(images).as_posix() for p in images.rglob("*.png")}
    if actual_paths != set(paths):
        raise ValueError("Native image payload coverage differs")
    for relative, selected in paths.items():
        image = audit.image_path(images, relative)
        proof = tabmwp_images.verify_png(image.read_bytes(), entries["data/tabmwp/" + relative])
        recorded = receipt["files"][relative]
        if (any(recorded.get(k) != v for k, v in proof.items())
                or any(recorded.get(k) != v for k, v in selected.items())
                or recorded.get("url") != tabmwp.RAW_URL + "data/tabmwp/" + relative):
            raise ValueError("Native image receipt/payload mismatch")
        bound[str(image)] = proof["sha256"]
    if receipt.get("total_verified_bytes") != sum(v["bytes"] for v in receipt["files"].values()):
        raise ValueError("Native image byte total differs")
    for name in ("acquisition.json", "tree.json", "commit.json", receipt_path.name):
        bound[str(images / name)] = audit.sha(images / name)
    variants = {variant: {role: audit.read_rows(prepared / f"{variant}/{role}.jsonl")
                          for role in ("train", "dev", "sealed")} for variant in ("A", "B")}
    return variants, bound, receipt_path


def retention_bundle(root):
    """Require the exact closed historical approval chain; no fallback to raw data."""
    root = Path(root)
    rows, manifest, _completion, frozen, _components = prepare_training.closed_pilot(root)
    selected = prepare_training.select_approved_subset(rows, 400, tabmwp.SEED)
    if (len(selected) != 400 or Counter(r.get("source_id") for r in selected) !=
            {"trace": 360, "nvw-independent-procedural-v1": 40}
            or any(r.get("split") != "train" or r.get("training_allowed") is not True
                   or r.get("benchmark_derived") is not False for r in selected)):
        raise ValueError("Retention must be the historical approved train-only 360/40 subset")
    bound = {str(root / name): value for name, value in frozen.items()}
    for row in selected:
        audit.check_lineage(row, manifest["sources"])
        audit.check_target_identifiability(row, root)
        for name in row["image_paths"]:
            path = audit.image_path(root, name)
            value = audit.sha(path)
            if manifest["files"].get(name) != value:
                raise ValueError("Retention image differs from closed approval")
            bound[str(path)] = value
    return selected, manifest, bound


def pair_check(a, b, retention_ids):
    """The only variant difference is native assistant target supervision."""
    if len(a) != 2000 or len(b) != 2000 or len(set(retention_ids)) != 400:
        raise ValueError("Exact native1600 + retention400 pair required")
    if [r["id"] for r in a] != [r["id"] for r in b] or len({r["id"] for r in a}) != 2000:
        raise ValueError("A/B IDs/order differ or repeat")
    observed = set()
    for left, right in zip(a, b, strict=True):
        if left.get("split") != "train" or right.get("split") != "train":
            raise ValueError("Dev/sealed records cannot enter training proposals")
        if left["id"] in retention_ids:
            if left != right:
                raise ValueError("Retention supervision/metadata must remain identical A/B")
            observed.add(left["id"])
        else:
            if (left.get("source_id") != tabmwp.SOURCE_ID or left.get("native_solution_used") is not False
                    or left.get("teacher_generated") is not False
                    or {k: v for k, v in left.items() if k != "target_text"} !=
                       {k: v for k, v in right.items() if k != "target_text"}):
                raise ValueError("Native A/B inputs or provenance differ")
            ta, tb = json.loads(left["target_text"]), json.loads(right["target_text"])
            if set(ta) != {"answer"} or set(tb) != {"facts", "answer"} or ta["answer"] != tb["answer"]:
                raise ValueError("Native A/B answer or target contract differs")
    if observed != set(retention_ids):
        raise ValueError("Retention ID coverage differs")
    audit.check_splits(a)
    audit.check_splits(b)


def contract():
    parent = Path(__file__).with_name("protocol.json")
    pixels = audit.training_input_contract()
    return {"arm": "FM", "learning_rate": 1e-5,
            "module_selection_basis": "Historical G6 external development, not new MMMU outcomes",
            "each_variant_starts_from": "F0", "dtype": "bfloat16", "quantization": None,
            "epochs": 1, "micro_batch_size": 1, "gradient_accumulation": 16,
            "master_seed": tabmwp.SEED, "records_per_variant": 2000, "expected_updates": 125,
            "parent_protocol_sha256": audit.sha(parent), "input_contract": pixels,
            "order_policy": "Same frozen record order; existing trainer Random(3407).shuffle once",
            "target_token_normalization": "Existing trainer sums assistant token loss per update",
            "automatic_fallback": False, "training_execution_authorized": False}


def prepare(prepared_root, image_root, retention_root, output_root):
    """New proposals/registry only; all originals and historical approvals stay immutable."""
    from .train import order_hash

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(output)
    native, native_bound, image_receipt = native_bundle(prepared_root, image_root)
    retention, parent_manifest, retention_bound = retention_bundle(retention_root)
    retention_ids = [r["id"] for r in retention]
    registry = dict(parent_manifest["sources"])
    if tabmwp.SOURCE_ID in registry:
        raise ValueError("TabMWP source cannot replace a historical registry entry")
    registry[tabmwp.SOURCE_ID] = {
        "source_known": True, "rights_approved": False, "benchmark_derived": False, "parents": [],
        "revision": tabmwp.REVISION, "source_url": tabmwp.REPOSITORY + "/tree/" + tabmwp.REVISION,
        "license": tabmwp.DATASET_LICENSE, "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "attribution": "TabMWP / PromptPG, Pan Lu et al.; underlying educational tables from IXL",
        "rights_scope_candidate": "Private noncommercial academic research; preserve attribution and share-alike notices",
        "source_receipt_sha256": audit.sha(Path(prepared_root).parent / "source_receipt.json"),
        "license_evidence_sha256": audit.sha(Path(prepared_root).parent / "README.md"),
        "public_upload_allowed": False,
        "annotation_basis": "Pinned upstream native table/question/answer, ordered JSON/text parity",
        "approval_limits": "Source-trust candidate only; no solver proof, teacher solution, full human or pixel-semantic certification"}
    variants = {}
    for variant in ("A", "B"):
        rows = [dict(row, supervised_target_identifiable=True, answer_identifiable=True,
                     annotation_basis="UPSTREAM_NATIVE_SOURCE_GOLD",
                     annotation_basis_review_required=True) for row in native[variant]["train"]]
        variants[variant] = prepare_training.order(rows + retention, tabmwp.SEED)
        # Source train/dev/holdout family exclusion was replayed by native_bundle.
        audit.check_splits(variants[variant] + native[variant]["dev"] + native[variant]["sealed"])
    pair_check(variants["A"], variants["B"], retention_ids)
    bound = {**native_bound, **retention_bound}
    unchanged(bound)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for source_root, rows in ((image_root, native["A"]["train"]), (retention_root, retention)):
        for row in rows:
            for name in row["image_paths"]:
                value = prepare_training.copy_image(source_root, output, name)
                if name in files and files[name] != value:
                    raise ValueError("Native/retention image collision")
                files[name] = value
    proposals = {}
    for variant, rows in variants.items():
        name = f"{variant}/proposed.jsonl"
        audit.write_rows(output / name, rows)
        proposals[variant] = {"path": name, "sha256": audit.sha(output / name), "count": len(rows)}
    audit.dump(output / "source_registry_candidate.json", {"sources": registry})
    training = contract()
    ids = [row["id"] for row in variants["A"]]
    shuffled = list(ids)
    random.Random(tabmwp.SEED).shuffle(shuffled)
    unchanged(bound)
    manifest = {"status": "PREPARED_PENDING_GATES", "training_allowed": False,
                "public_upload_allowed": False, "benchmark_derived": False,
                "proposals": proposals, "files": files, "record_ids": ids,
                "record_order_sha256": order_hash(ids), "consumption_order_sha256": order_hash(shuffled),
                "order_hash_algorithm": "fine_tuning.train.order_hash: SHA256(json.dumps(ids).encode())",
                "retention_ids": retention_ids, "retention_count": 400, "native_count": 1600,
                "native_image_receipt_sha256": audit.sha(image_receipt),
                "tabmwp_preparation_hash": read_json(Path(prepared_root) / "manifest.json")["preparation_hash"],
                "source_registry_candidate_sha256": audit.sha(output / "source_registry_candidate.json"),
                "frozen_inputs_sha256": bound,
                "source_sha256": {name: audit.sha(Path(__file__).with_name(name))
                                  for name in ("train.py", "modules.py", "tabmwp_trial.py")},
                "required_benchmark_coverage": parent_manifest["contamination_audit"]["coverage"],
                "training_contract": training,
                "native_annotation_trust": "Review candidate, not human/pixel-semantic certification",
                "native_teacher_generated": False, "native_solution_used": False,
                "retention_policy": "Historical approved rows preserved exactly; no new teacher annotations",
                "pending_gates": ["MAIN_REVIEW_OF_SOURCE_RIGHTS_AND_ANNOTATION_TRUST",
                                  "BOTH_ACTUAL_PINNED_PROCESSOR_AND_ASSISTANT_MASK_PROOFS",
                                  "COMPLETE_EXPECTED_BENCHMARK_EXCLUSION_FOR_BOTH_VARIANTS",
                                  "EXACT_2000_APPROVED_A_B_PARITY_AND_TARGET_TOKEN_EXPOSURE",
                                  "EXPLICIT_TRAINING_EXECUTION_AUTHORITY"]}
    manifest["trial_hash"] = tabmwp.digest(manifest)
    audit.dump(output / "trial_manifest.json", manifest)
    return manifest


def validate_trial(root):
    """Fail closed before any processor import or approval output."""
    from .train import order_hash

    root = Path(root)
    manifest = read_json(root / "trial_manifest.json")
    hashed = dict(manifest)
    expected = hashed.pop("trial_hash", None)
    if (expected != tabmwp.digest(hashed) or manifest.get("status") != "PREPARED_PENDING_GATES"
            or manifest.get("training_allowed") is not False or manifest.get("training_contract") != contract()):
        raise ValueError("Trial preparation/protocol identity differs")
    for name, expected in manifest["source_sha256"].items():
        if audit.sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("Trial/trainer/module source changed")
    unchanged(manifest["frozen_inputs_sha256"])
    if audit.sha(root / "source_registry_candidate.json") != manifest["source_registry_candidate_sha256"]:
        raise ValueError("Candidate registry differs")
    variants = {}
    for variant, proposal in manifest["proposals"].items():
        path = audit.image_path(root, proposal["path"])
        if audit.sha(path) != proposal["sha256"]:
            raise ValueError("Proposal bytes differ")
        variants[variant] = audit.read_rows(path)
    pair_check(variants["A"], variants["B"], manifest["retention_ids"])
    if manifest["record_ids"] != [r["id"] for r in variants["A"]]:
        raise ValueError("Trial order differs")
    shuffled = list(manifest["record_ids"])
    random.Random(tabmwp.SEED).shuffle(shuffled)
    if (manifest.get("record_order_sha256") != order_hash(manifest["record_ids"])
            or manifest.get("consumption_order_sha256") != order_hash(shuffled)):
        raise ValueError("Trial order hashes differ from the actual trainer")
    expected_files = {name for row in variants["A"] for name in row["image_paths"]}
    if expected_files != set(manifest["files"]):
        raise ValueError("Trial image coverage differs")
    for name, expected in manifest["files"].items():
        if audit.sha(audit.image_path(root, name)) != expected:
            raise ValueError("Trial image bytes differ")
    return variants, manifest


def approve_cpu(root, reviewed_registry, benchmark_index, model_path, review_path, output_root):
    """Explicit future CPU-only approval stage; preparation never calls this function."""
    root, output = Path(root), Path(output_root)
    if output.exists():
        raise FileExistsError(output)
    variants, manifest = validate_trial(root)
    review = read_json(review_path)
    required = {"status": "APPROVED_FOR_CPU_DATA_AUDIT",
                "trial_manifest_sha256": audit.sha(root / "trial_manifest.json"),
                "source_registry_sha256": audit.sha(reviewed_registry),
                "benchmark_index_sha256": audit.sha(benchmark_index),
                "native_source_annotation_trust_approved": True,
                "training_execution_authorized": False}
    if any(review.get(k) != v for k, v in required.items()):
        raise ValueError("Exact main review required before CPU approval")
    registry = prepare_training.registry(reviewed_registry)
    candidate = prepare_training.registry(root / "source_registry_candidate.json")
    expected_registry = json.loads(json.dumps(candidate))
    expected_registry[tabmwp.SOURCE_ID]["rights_approved"] = True
    if registry != expected_registry:
        raise ValueError("Reviewed registry may only approve the exact native source candidate")
    index = read_json(benchmark_index)
    def coverage_identity(coverage):
        return sorted(tabmwp.digest({k: v for k, v in entry.items() if k != "path"})
                      for entry in coverage.get("files", []))
    if (index.get("status") != "COMPLETE" or index.get("coverage_verified") is not True
            or coverage_identity(index.get("coverage", {})) != coverage_identity(manifest["required_benchmark_coverage"])):
        raise ValueError("Complete exact expected MMMU/MMMU-Pro exclusion coverage required")
    for rows in variants.values():
        for row in rows:
            audit.check_lineage(row, registry)
            audit.check_target_identifiability(row, root)
    frozen = {str(Path(p)): audit.sha(p) for p in (reviewed_registry, benchmark_index, review_path)}
    frozen[str(root / "trial_manifest.json")] = audit.sha(root / "trial_manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    results, processors, evidence = {}, {}, {}
    for variant in ("A", "B"):
        rows, _approval, processor, proof = prepare_training.approve_delta(
            variants[variant], root, registry, benchmark_index, model_path, output / variant)
        results[variant], processors[variant], evidence[variant] = rows, processor, proof
    # Each audit can quarantine items; no partial dataset is ready for this trial.
    exact = all([r["id"] for r in results[v]] == manifest["record_ids"] for v in results)
    if exact:
        comparable = {v: [dict(r, processor_receipt_sha256="VARIANT_SPECIFIC") for r in rows]
                      for v, rows in results.items()}
        pair_check(comparable["A"], comparable["B"], manifest["retention_ids"])
        for sid in manifest["record_ids"]:
            left, right = (processors[v]["records"][sid] for v in ("A", "B"))
            if (left["image_grid_thw"] != right["image_grid_thw"]
                    or left["tensor_sha256"]["pixel_values"] != right["tensor_sha256"]["pixel_values"]
                    or sid in manifest["retention_ids"] and left != right):
                raise ValueError("Actual A/B image or retention processor proofs differ")
    unchanged(frozen)
    validate_trial(root)
    tokens = {v: sum(p["assistant_tokens"] for p in processor["records"].values())
              for v, processor in processors.items()}
    report = {"status": "DATA_APPROVED" if exact else "INCOMPLETE_QUARANTINE",
              "training_allowed": exact, "training_execution_authorized": False,
              "training_executed": False, "public_upload_allowed": False,
              "trial_manifest_sha256": frozen[str(root / "trial_manifest.json")],
              "review_sha256": audit.sha(review_path), "evidence": evidence,
              "assistant_tokens": tokens,
              "target_token_exposure_matched": len(set(tokens.values())) == 1,
              "target_token_limit": "A/B have matched records/updates, not matched assistant token exposure"}
    audit.dump(output / "cpu_approval_receipt.json", report)
    if not exact:
        raise ValueError("Exact A/B2000 approval shortfall; evidence preserved, trial not ready")
    for variant in ("A", "B"):
        training_manifest = read_json(output / variant / "approved/manifest.json")
        training_manifest.update(
            tabmwp_condition=variant, tabmwp_preparation_hash=manifest["tabmwp_preparation_hash"],
            tabmwp_trial_hash=manifest["trial_hash"],
            tabmwp_trial_manifest_sha256=frozen[str(root / "trial_manifest.json")],
            tabmwp_trial_receipt_sha256=audit.sha(output / "cpu_approval_receipt.json"),
            paired_policy="Identical2000IDs/order/images; native1600answer-onlyA vs nativefacts+sameanswerB; identical historical400retention",
            training_contract=manifest["training_contract"],
            record_order_sha256=manifest["record_order_sha256"],
            consumption_order_sha256=manifest["consumption_order_sha256"])
        audit.dump(output / variant / "training_manifest.json", training_manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in ("prepared-root", "image-root", "retention-root", "output-root"):
        prepare_parser.add_argument("--" + name, required=True, type=Path)
    approve_parser = commands.add_parser("approve-cpu")
    for name in ("root", "reviewed-registry", "benchmark-index", "model-path", "review-path", "output-root"):
        approve_parser.add_argument("--" + name, required=True, type=Path)
    args = vars(parser.parse_args())
    stage = args.pop("stage")
    result = prepare(**args) if stage == "prepare" else approve_cpu(**args)
    print(json.dumps({"status": result["status"], "training_allowed": result["training_allowed"]}))


if __name__ == "__main__":
    main()
