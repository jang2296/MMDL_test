"""D1–D8 data evidence for an exclusion-reviewed mixed pool under approval stage ai_reviewed_local.

Sibling of smoke_data_evidence (whose program-only graph checks stay unchanged). D1/D2/D4 rerun
project generators only; HUMAN and external PROGRAM items are covered by the per-source AI review
record (human_verified=false). The real full exclusion receipt is required and the existing
approval gate code is rerun before anything is written. Nothing here is a training approval.
Quarantined families may stay in the build but are never sampled, regenerated or eligible.
A pool_freeze vault (contamination/union_receipt.json) is accepted the same way: each shard receipt is
re-checked against its own shard, its registry/evidence come per shard ({"shards": [descriptors]}), and the
allowlist is the vault items allowlisted in their own shard. A pool without generator items has empty D1/D2/D4.

Runs without torch/pyarrow (e.g. the notebook-data env that has the matplotlib/rdkit/music21
generator dependencies): `python -m fine_tuning.notebook_rl.pool_data_evidence review-sample|produce`.
"""
from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image

from . import approval, contamination, pool_freeze, smoke_data_evidence as smoke
from .cli import artifact_path
from .dataset import SPLITS, _bytes, _write
from .generators import _NAMES, get_generator
from .schema import load_protocol
from .vault import _read, _safe_root, read_actors, read_lineage, read_verifier

_HERE = Path(__file__).resolve().parent
VARIATION_METHOD = "same-template-natural-variation-v1"
INTERVENTION_METHOD = "label-changing-intervention-v1"
_LIMITS = ["AI review is fallible; human_verified=false.",
           "Python guards are not an OS or hostile same-user isolation boundary.",
           "D4 natural-variation pairs change the whole image with identical non-image inputs; "
           "they show text-only answers are impossible, not which visual detail matters.",
           "HUMAN and external PROGRAM labels are not re-derived by any solver.",
           "Quarantined (non-allowlisted) items are outside D1/D2/D3/D4 and never eligible; "
           "a source with a failing AI verdict or a short sample is blocked, not repaired."]
_ACCEPTANCE_LIMIT = (f"{approval.ACCEPTANCE_RULE} (AI review v2): a source with failures stays eligible only with "
                     f">= min({approval.ACCEPTANCE_SAMPLE}, n) reviewed and <= {approval.ACCEPTANCE_MAX_FAILED} failures; "
                     "its failed items are ineligible, its unreviewed items are not individually checked.")


def _generator(source: str):
    for name in sorted(_NAMES):
        try:
            module = get_generator(name)
        except ModuleNotFoundError:
            continue
        if module.GENERATOR_VERSION == source:
            return module
    raise ValueError(f"pool evidence: no importable generator for {source}")


def _inputs(root: Path):
    root = _safe_root(root)
    items, files, manifests = {}, {}, {}
    for split in SPLITS:
        part = root / split
        manifests[split] = smoke._json(part / "manifest.yaml")
        actors, lineages, verifiers = read_actors(part), read_lineage(part), read_verifier(part)
        if {actor.problem_id for actor in actors} != set(lineages) or set(lineages) != set(verifiers):
            raise ValueError("pool evidence: mismatched vault identities")
        for name in ("manifest.yaml", "actor_inputs.jsonl", "lineage.jsonl", "verifier_private.jsonl"):
            files[str(part / name)] = smoke._sha(_read(part / name))
        for actor in actors:
            lineage = lineages[actor.problem_id]
            if (actor.problem_id in items or lineage.split != split
                    or lineage.benchmark_derived or lineage.model_touched):
                raise ValueError("pool evidence: forbidden or duplicated candidate")
            approval.label_class(lineage)
            images = tuple(_read(root / "images" / rel) for rel in actor.image_paths)
            if tuple(smoke._sha(raw) for raw in images) != actor.image_sha256:
                raise ValueError("pool evidence: original image changed")
            for rel, digest in zip(actor.image_paths, actor.image_sha256, strict=True):
                files[str(root / "images" / rel)] = digest
            items[actor.problem_id] = (actor, verifiers[actor.problem_id], lineage, images)
    if not items:
        raise ValueError("pool evidence: nonempty pool required")
    return items, files, manifests


def review_sample(root: Path, *, seed: int = 3407, count: int = approval.AI_LOCAL_SAMPLE) -> dict:
    """What the AI review must cover: per source, the first min(count, n) exclusion-allowlisted items in
    seed-keyed hash order (count 20; 60 for a v2 source with a failure), and every manifest leak warning on
    an allowlisted item. Quarantined items are never sampled."""
    root = _safe_root(Path(root).absolute())
    items, files, manifests = _inputs(root)
    _, _, allowlist = _exclusion(root, files, items)
    groups: dict[str, list[str]] = defaultdict(list)
    for pid in sorted(allowlist):
        groups[items[pid][0].source].append(pid)
    return {"sources": {source: approval.ai_review_order(pids, seed)[:min(count, len(pids))]
                        for source, pids in sorted(groups.items())},
            "leak_warnings": [row for manifest in manifests.values() for row in manifest["leak_review"]
                              if row["problem_id"] in allowlist]}


def measure_variation(module, example, output: Path, *, count: int = 100, limit: int = 50) -> dict:
    """Same template and identical non-image inputs, different generated image; every pair kept on disk."""
    actor, verifier, lineage, images = example
    start = int(actor.problem_id.split(":")[-2])
    style = "shift" if lineage.split == "style_sealed" else "default"
    fixed = {k: v for k, v in asdict(actor).items() if k not in ("problem_id", "image_paths", "image_sha256")}
    rows: list[dict] = []
    for index in range(start + 1, start + 1 + count * limit):
        if len(rows) == count:
            break
        try:
            p = module.generate(index, lineage.generator_seed, lineage.split, actor.mode, style)
            module.self_check(p)
        except ValueError:
            continue
        fields = {k: v for k, v in asdict(p.actor).items() if k not in ("problem_id", "image_paths", "image_sha256")}
        if p.lineage.template_id != lineage.template_id or fields != fixed:
            continue
        row = {"attempt": len(rows), "index": index, "problem_id": p.actor.problem_id,
               "source": lineage.source, "template": lineage.template_id, "before": verifier.gold,
               "after": p.verifier.gold, "question_unchanged": True, "nonimage_inputs_unchanged": True,
               "before_images_sha256": list(actor.image_sha256), "after_images_sha256": list(p.actor.image_sha256),
               "status": "PASS" if p.verifier.gold != verifier.gold and p.image_bytes != images else "FAIL"}
        _write(output / "attempts" / f"{lineage.source}-{lineage.template_id}-{len(rows):03d}.json", _bytes(row) + b"\n")
        rows.append(row)
    return {"pairs": len(rows), "answer_changed": sum(row["status"] == "PASS" for row in rows),
            "question_unchanged": True, "nonimage_inputs_unchanged": True, "method": VARIATION_METHOD,
            "attempts_sha256": smoke._hash(rows), "attempts_recorded": len(rows)}


def _inventory(items: dict) -> dict:
    """Family identities (structure, image, pixel) of `_inputs` items, as the exclusion scan keys them."""
    inventory = {}
    for pid, row in items.items():
        pixels = []
        for raw in row[3]:
            with Image.open(io.BytesIO(raw)) as image:
                pixels.append(contamination.audit.image_fingerprint(image)["pixels_sha256"])
        inventory[pid] = {"structure_hash": row[2].structure_hash, "image_sha256": list(row[0].image_sha256),
                          "pixel_sha256": pixels}
    return inventory


def _receipt(root: Path, receipt_descriptor: dict, files: dict, items: dict) -> tuple[dict, set[str]]:
    """The bound reviewed receipt and its re-derived allowlist (quarantined families stay, never eligible)."""
    receipt = smoke._bound_json(receipt_descriptor)
    if (receipt.get("version") != contamination.VERSION
            or receipt.get("status") != "EXCLUSION_CHECKED_WITH_LIMITS" or receipt.get("blockers") != []
            or receipt.get("training_allowed") is not False or receipt.get("approval_granted") is not False
            or receipt.get("partial_name") is not None
            or receipt.get("receipt_sha256") != smoke._hash({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            or set(receipt.get("steps", {})) != set(contamination._STEPS)
            or any(step.get("status") != "COMPLETE" for step in receipt["steps"].values())
            or receipt.get("input_files") != files or receipt.get("input_files_sha256") != smoke._hash(files)
            or smoke._json(root / "manifest.yaml").get("exclusion_receipt_sha256") != receipt_descriptor["file_sha256"]):
        raise ValueError("pool evidence: actual complete exclusion receipt required")
    from .adjudication import validate as reviewed_candidates
    inventory = _inventory(items)
    quarantine = contamination.family_quarantine(inventory, reviewed_candidates(receipt, inventory))
    if (receipt.get("quarantined_ids") != quarantine["quarantined_ids"]
            or receipt.get("allowlist") != quarantine["allowlist"] or not quarantine["allowlist"]):
        raise ValueError("pool evidence: reviewed allowlist mismatch or no allowlisted item")
    return receipt, set(quarantine["allowlist"])


def _union(root: Path, items: dict) -> tuple[dict, dict, set[str]]:
    """A pool_freeze vault: each shard receipt re-checked by `_receipt` against its own shard's items; every vault
    item must equal its shard record, be allowlisted there and share no family with an item quarantined in any
    shard. Returns (union descriptor, union, allowlist)."""
    union, _ = pool_freeze.read_union(root)
    if pool_freeze.verify(Path(union["dataset_manifest"]["path"]), root):
        raise ValueError("pool evidence: vault bytes differ from the frozen dataset manifest")
    shards = []
    for entry in union["shards"]:
        shard_root = _safe_root(Path(entry["root"]))
        shard_items, shard_files, _ = _inputs(shard_root)
        receipt, allowed = _receipt(shard_root, entry["receipt"], shard_files, shard_items)
        if (type(receipt.get("adjudication")) is not dict
                or smoke._sha(_read(shard_root / "manifest.yaml")) != entry["build_manifest_sha256"]
                or smoke._hash(smoke._bound_json(entry["source_registry"])) != receipt.get("source_registry_sha256")
                or smoke._hash(smoke._bound_json(entry["evidence"])) != receipt.get("evidence_sha256")):
            raise ValueError("pool evidence: union shard receipt is unreviewed or not bound to its registry/evidence")
        shards.append((shard_items, allowed))
    cross = pool_freeze.cross_shard_allowlist([(_inventory(shard_items), allowed) for shard_items, allowed in shards])
    for pid, row in items.items():
        shard_items, allowed = shards[union["items"][pid]["shard"]]
        if pid not in allowed or pid not in cross or shard_items.get(pid) != row:
            raise ValueError(f"pool evidence: {pid} is not an allowlisted record of its shard")
    return smoke._descriptor(root / "contamination" / pool_freeze.UNION_FILE), union, set(items)


def _exclusion(root: Path, files: dict, items: dict) -> tuple[dict, dict, set[str]]:
    """(receipt descriptor, receipt, allowlist): the build's own reviewed receipt, or a pool_freeze union."""
    if (root / "contamination" / pool_freeze.UNION_FILE).exists():
        return _union(root, items)
    descriptor = smoke._descriptor(root / "contamination/receipt_v2.json")
    return (descriptor, *_receipt(root, descriptor, files, items))


def produce(root: Path, review_descriptor: dict, benchmark: Path, output: Path, *,
            index: dict, source_registry: dict, evidence: dict) -> dict:
    """Measure D1–D8, rerun the approval gate checks in process, then write proofs and the data request."""
    root = _safe_root(Path(root).absolute())
    output = artifact_path(Path(output).absolute())
    if output.exists():
        raise FileExistsError("pool evidence: preserve existing evidence")
    items, files, manifests = _inputs(root)
    receipt_descriptor, receipt, allowlist = _exclusion(root, files, items)
    if receipt.get("version") == pool_freeze.UNION_VERSION:
        # Per-shard registries/evidence, exactly as the union names them (each re-checked by _union).
        registry: dict = {}
        if (source_registry != {"shards": [entry["source_registry"] for entry in receipt["shards"]]}
                or evidence != {"shards": [entry["evidence"] for entry in receipt["shards"]]}):
            raise ValueError("pool evidence: a union vault needs its shards' registry/evidence descriptors, in order")
    else:
        registry = (smoke._bound_json(source_registry) if set(source_registry) == {"path", "file_sha256"}
                    else source_registry)
        if smoke._hash(registry) != receipt.get("source_registry_sha256"):
            raise ValueError("pool evidence: source registry differs from the exclusion receipt")
    review = smoke._bound_json(review_descriptor)
    output.mkdir(mode=0o700)
    checks: list[dict] = []
    templates: dict[str, tuple] = {}
    # Quarantined items are never eligible: D1/D2 and the D4 template examples use allowlisted items only.
    for pid, (actor, verifier, lineage, images) in sorted(items.items()):
        if pid not in allowlist or approval.label_class(lineage) != "GENERATOR":
            continue
        module = _generator(lineage.source)
        style = "shift" if lineage.split == "style_sealed" else "default"
        p = module.generate(int(pid.split(":")[-2]), lineage.generator_seed, lineage.split, actor.mode, style)
        module.self_check(p)  # rebuilds the generator's independent solutions
        if lineage.source in smoke._SOURCES:
            before, independent = smoke._solvers(verifier.accept["graph_world"], lineage.template_id)
            solved = before == independent == verifier.gold
        else:
            solved = p.verifier.gold == verifier.gold
        checks.append({"problem_id": pid, "solver_match": solved, "repeat_match": (
            p.actor, p.verifier, replace(p.lineage, source_revision=smoke._revision(module)), p.image_bytes)
            == (actor, verifier, lineage, images)})
        templates.setdefault(lineage.source + ":" + lineage.template_id, (actor, verifier, lineage, images))
    d4 = {}
    for key, example in sorted(templates.items()):
        source, template = example[2].source, example[2].template_id
        d4[key] = (smoke.measure_interventions(source, template, example, output) | {"method": INTERVENTION_METHOD}
                   if source in smoke._SOURCES else measure_variation(_generator(source), example, output))
    leaks = {}
    for key, extract in (("structure_cross_split", lambda row: [row[2].structure_hash]),
                         ("image_cross_split", lambda row: row[0].image_sha256)):
        owners = defaultdict(set)
        for row in items.values():
            for value in extract(row):
                owners[value].add(row[2].split)
        leaks[key] = sum(len(splits) > 1 for splits in owners.values())
    warnings = {split: manifest["leak_review"] for split, manifest in manifests.items()}
    skipped = approval.is_skipped_review(review)
    provenance = {"reviewer_id": review.get("reviewer_id"),
                  "kind": approval.SOURCE_REVIEW_SKIPPED_KIND if skipped else approval.AI_LOCAL_REVIEW_KIND,
                  "method": approval.SOURCE_REVIEW_SKIPPED if skipped else approval.AI_LOCAL_REVIEW_METHOD,
                  "reviewed_at_utc": review.get("reviewed_at_utc"), "human_verified": False, "labels_modified": False}
    ids_sha = smoke._hash(sorted(row["problem_id"] for row in checks))
    gates = {
        "D1": {"checked_ids_sha256": ids_sha, "items_checked": len(checks),
               "repeat_mismatches": sum(not row["repeat_match"] for row in checks)},
        "D2": {"checked_ids_sha256": ids_sha, "items_checked": len(checks),
               "solver_mismatches": sum(not row["solver_match"] for row in checks), "independent_solvers": True},
        "D3": {"sampling_seed": 3407, "review": review_descriptor}, "D4": {"templates": d4}, "D5": leaks,
        "D6": {"exclusion_receipt_sha256": receipt_descriptor["file_sha256"]},
        "D7": {"reviewed_actor_files": {p: s for p, s in files.items() if Path(p).name == "actor_inputs.jsonl"},
               "warnings_sha256": smoke._hash(warnings),
               "unresolved": sum(not row.get("leak_free") for row in review.get("leak_warnings", []) if type(row) is dict)},
        "D8": smoke.measure_isolation(root, Path(benchmark))["measurements"]}
    protocol = load_protocol(_HERE / "protocol_v1.yaml")
    binding = {"build_manifest_sha256": smoke._sha(_read(root / "manifest.yaml")),
               "exclusion_receipt_sha256": receipt_descriptor["file_sha256"], "input_files_sha256": smoke._hash(files),
               "protocol_sha256": protocol["sha256"],
               **{key: receipt[key] for key in ("index_sha256", "index_file_sha256", "source_registry_sha256", "evidence_sha256")}}
    scope = {"purpose": approval.AI_LOCAL_PURPOSE, "protocol_sha256": protocol["sha256"],
             "code_sha256": approval.code_hashes(), "input_files_sha256": smoke._hash(files),
             "producer": smoke._descriptor(Path(__file__))}
    proofs = {key: {"measurements": value, **({"provenance": provenance} if key in ("D3", "D7") else {})}
              for key, value in gates.items()}
    # Reuse the real approval checks; no failed measurement becomes PASS.
    approval._measurement_scope(scope, binding, protocol, {}, ai_local=True)
    as_items = {pid: {"actor": row[0], "verifier": row[1], "lineage": row[2]} for pid, row in items.items()}
    blocked, quality = approval._data_gates(proofs, as_items, manifests, files, registry,
                                            receipt_descriptor["file_sha256"], 3407, ai_review=review, allowlist=allowlist)
    eligibility = approval.ai_local_eligibility({pid: row[0].source for pid, row in items.items()}, allowlist,
                                                blocked or {}, quality,
                                                approval.SOURCE_REVIEW_SKIPPED if skipped else None)
    if any(smoke._sha(_read(Path(path))) != digest for path, digest in files.items()):
        raise ValueError("pool evidence: inputs changed during measurement")
    report: dict = {"format": "notebook-rl-data-gates-v1", "status": "COMPLETE", "binding": binding, "scope": scope,
                    "gates": {}}
    for key, proof in proofs.items():
        proof.update(gate=key, status="PASS", binding_sha256=smoke._hash(binding), scope_sha256=smoke._hash(scope),
                     producer_sha256=scope["producer"]["file_sha256"], records=checks if key in ("D1", "D2") else None,
                     limitations=([*_LIMITS, f"source_review={approval.SOURCE_REVIEW_SKIPPED}: no per-source sample "
                                   "review; leak-warned items excluded before selection."] if skipped else
                                  _LIMITS if quality is None else [*_LIMITS, _ACCEPTANCE_LIMIT]))
        path = output / (key + ".json")
        _write(path, _bytes(proof) + b"\n")
        report["gates"][key] = {"status": "PASS", "artifact": smoke._descriptor(path)}
    _write(output / "data_report.json", _bytes(report) + b"\n")
    request = {"build_root": str(root), "index": index, "source_registry": source_registry, "evidence": evidence,
               "data_report": smoke._descriptor(output / "data_report.json"), "expected_measurement_scope": scope}
    _write(output / "data_request.json", _bytes(request) + b"\n")
    return {"status": "AI_LOCAL_DATA_EVIDENCE_WRITTEN_NOT_APPROVAL", "training_allowed": False,
            "human_verified": False, "data_request": smoke._descriptor(output / "data_request.json"),
            "eligibility": {key: value for key, value in eligibility.items() if key != "eligible_ids"}
            | {"eligible_count": len(eligibility["eligible_ids"])}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("review-sample")
    sample.add_argument("--build", type=Path, required=True)
    sample.add_argument("--count", type=int, default=approval.AI_LOCAL_SAMPLE,
                        help=f"items per source (min(count, n)); {approval.ACCEPTANCE_SAMPLE} for an acceptance review")
    run = sub.add_parser("produce")
    for name in ("build", "review", "benchmark-probe", "index", "out"):
        run.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-registry", "evidence"):
        run.add_argument(f"--{name}", type=Path, nargs="+", required=True,
                         help="one file; for a pool_freeze vault one per shard, in union-receipt order")
    args = parser.parse_args(argv)
    if args.command == "review-sample":
        print(json.dumps(review_sample(args.build, count=args.count), indent=1))
        return 0
    union = (args.build / "contamination" / pool_freeze.UNION_FILE).exists()

    def bound(paths: list[Path]) -> dict:
        if union:
            return {"shards": [smoke._descriptor(path) for path in paths]}
        if len(paths) != 1:
            parser.error("--source-registry/--evidence take one file unless the build is a pool_freeze union vault")
        return smoke._descriptor(paths[0])
    result = produce(args.build, smoke._descriptor(args.review), args.benchmark_probe, args.out,
                     index=smoke._descriptor(args.index), source_registry=bound(args.source_registry),
                     evidence=bound(args.evidence))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
