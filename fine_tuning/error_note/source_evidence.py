"""Prepare private, source-bound review inputs and answer rules; never training targets.

Only the supplied frozen train order is used. Cached annotations remain raw references:
their availability does not certify a generated note, and no review is performed here.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from fine_tuning.notebook_rl.answers import compare_training
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.vault import _read, _safe_root, read_actors, read_lineage, read_verifier

from .answer_contract import VERSION as RULES_VERSION
from .audit import digest, text_sha256
from .collect_v3 import file_sha
from .order import read_order

VERSION = "note-source-provenance-v1"
_REPOS = {"hf-scienceqa": "derek-thomas/ScienceQA", "hf-geometry3k": "hiyouga/geometry3k",
          "hf-chebi-role": "liupf/ChEBI-20-MM", "hf-slake": "BoKelvin/SLAKE"}
_COLUMNS = {"hf-scienceqa": ["question", "choices", "answer", "hint", "lecture", "solution", "skill"],
            "hf-geometry3k": ["problem", "answer"]}


def _json(path: Path) -> dict:
    return json.loads(_read(path))


def _checked_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("source evidence: unsafe relative path")
    path = root / rel
    _safe_root(path.parent)
    if path.is_symlink() or not path.is_file():
        raise ValueError("source evidence: regular source/image file required")
    return path


def _native_rows(raw_root: Path, selected: dict[str, dict]) -> tuple[dict, dict]:
    """Read only the four available native annotation sources, with download hash/revision checks."""
    download = raw_root / "download_manifest.jsonl"
    entries = [json.loads(line) for line in _read(download).splitlines() if line.strip()]
    native, files = {}, {}
    for source in sorted({row["source"] for row in selected.values()} & _REPOS.keys()):
        wanted = {row["native_id"] for row in selected.values() if row["source"] == source}
        revisions = {row["source_revision"] for row in selected.values() if row["source"] == source}
        if len(revisions) != 1:
            raise ValueError("source evidence: mixed source revisions")
        revision = next(iter(revisions))
        repo = _REPOS[source]
        repo_root = raw_root / repo.replace("/", "__")
        pattern = "data/train-*.parquet" if source in _COLUMNS else "train.csv" if source == "hf-chebi-role" else "train.json"
        paths = sorted(repo_root.glob(pattern))
        if not paths:
            continue
        if len(paths) != 1:
            raise ValueError("source evidence: ambiguous native row numbering")
        path = paths[0]
        relative = path.relative_to(raw_root).as_posix()
        _checked_file(raw_root, relative)
        expected = {row["sha256"] for row in entries if row["repo"] == repo and row["revision"] == revision
                    and row["path"] == path.relative_to(repo_root).as_posix()}
        sha = file_sha(path)
        if expected != {sha}:
            raise ValueError("source evidence: cached source hash/revision mismatch")
        files[relative] = sha
        rows: Iterable[dict]
        if source in _COLUMNS:
            import pyarrow.parquet as pq

            table = pq.ParquetFile(path)
            columns = [column for column in _COLUMNS[source] if column in table.schema_arrow.names]
            rows = (row for batch in table.iter_batches(columns=columns) for row in batch.to_pylist())
        elif source == "hf-chebi-role":
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        else:
            rows = json.loads(_read(path))
        for index, row in enumerate(rows):
            key = (f"cid:{int(row['CID'])}" if source == "hf-chebi-role" else
                   f"train:{row['qid']}" if source == "hf-slake" else f"train:{index}")
            if key not in wanted:
                continue
            if (source, key) in native:
                raise ValueError("source evidence: duplicate native id")
            native[source, key] = (row, relative, sha)
        if not all((source, key) in native for key in wanted):
            raise ValueError("source evidence: native id absent from pinned train source")
    return native, {"download_manifest_sha256": file_sha(download), "files": files}


def _annotations(actor, item, frozen: dict, native: dict) -> tuple[list[dict], str]:
    entry = native.get((actor.source, frozen["native_id"]))
    if entry is None:
        return [], "SOURCE_CACHE_UNAVAILABLE" if actor.source in _REPOS else "LABEL_ONLY"
    row, relative, sha = entry
    source = actor.source
    scope = "LABEL_ONLY"
    if source == "hf-scienceqa":
        hint = row.get("hint") or ""
        question = (f"Context: {hint}\n" if hint.strip() else "") + row["question"]
        if (question != actor.question or tuple(row["choices"]) != actor.choices
                or "ABCDEFGHIJ"[row["answer"]] != item.gold):
            raise ValueError("source evidence: ScienceQA native binding mismatch")
        names = _COLUMNS[source]
        if row.get("lecture") or row.get("solution"):
            scope = "SOURCE_EXPLANATION_FOR_REVIEW"
    elif source == "hf-geometry3k":
        if (row["problem"].strip().removeprefix("<image>").strip() != actor.question
                or row["answer"].strip() != item.gold):
            raise ValueError("source evidence: Geometry3K native binding mismatch")
        names = _COLUMNS[source]
    elif source == "hf-chebi-role":
        from fine_tuning.notebook_rl.adapters.hf_chem import parse_roles

        if actor.choices[ord(item.gold) - ord("A")] not in parse_roles(row["description"]):
            raise ValueError("source evidence: ChEBI native role binding mismatch")
        names, scope = ["CID", "description", "SMILES"], "CURATED_ROLE_DESCRIPTION_FOR_REVIEW"
    else:
        from fine_tuning.notebook_rl.adapters.hf_medical import _qa_answer

        if _qa_answer(row["question"], row["answer"]) != (actor.question, actor.choices, item.answer_type, item.gold):
            raise ValueError("source evidence: SLAKE native binding mismatch")
        names = ["qid", "question", "answer", "modality", "location", "triple"]
        triple = row.get("triple")
        concrete = (isinstance(triple, list) and len(triple) == 3 and
                    all(isinstance(value, str) and value.strip() and value not in ("_", "vhead", "ktail")
                        for value in triple))
        scope = "SOURCE_TRIPLE_FOR_REVIEW" if concrete else "SOURCE_METADATA_ONLY_PLACEHOLDER_TRIPLE"
    references = [{"field": name, "value": row[name], "sha256": digest(row[name]),
                   "location": f"raw:{relative}#{frozen['native_id']}:{name}", "file_sha256": sha}
                  for name in names if name in row and row[name] not in (None, "", [])]
    return references, scope


def _rule(actor, item, lineage, frozen: dict, references: list[dict], binding: dict) -> tuple[dict | None, str]:
    if item.answer_type == "mcq":
        return None, "MCQ_UNCHANGED"
    kind, accept = item.answer_type, dict(item.accept)
    if kind not in ("int", "decimal", "fraction", "unit_number", "short_text"):
        return None, "UNSUPPORTED_TYPE_EXACT_FALLBACK"
    unit = "UNSUPPORTED_NO_EXPLICIT_NATIVE_UNIT" if actor.source == "hf-geometry3k" else "NO_UNIT_EXTENSION"
    problem = next((ref for ref in references if ref["field"] == "problem"), None)
    if (actor.source == "hf-geometry3k" and problem is not None and
            re.search(r"\bin degrees\b|\bdegree measure\b", actor.question, re.I) and
            re.search(r"\b(angle|measure|degrees?)\b|∠", actor.question, re.I)):
        kind = "unit_number"
        accept.update(unit_aliases=["°", "degree", "degrees"], require_unit=False)
        unit = "EXPLICIT_NATIVE_DEGREES"
    policy = {"gold_sha256": text_sha256(item.gold), "original_accept": item.accept,
              "answer_type": kind, "accept": accept, "unit_evidence": problem if unit == "EXPLICIT_NATIVE_DEGREES" else None,
              "adapter_code_sha256": frozen["adapter_code_sha256"]}
    rule = {key: binding[key] for key in ("actor_sha256", "image_sha256", "source", "source_revision", "split", "native_id")}
    rule.update(answer_type=kind, accept=accept, provenance={
        "label_provenance": lineage.label_provenance, "field": "verifier.gold + adapter.accept + native.unit_evidence",
        "location": f"vault:train/verifier_private.jsonl#{actor.problem_id};adapter:{frozen['adapter_code_sha256']}",
        "sha256": digest(policy), "human_verified": False,
        "review_scope": "Original label and frozen adapter numeric policy; unit aliases only from explicit native degrees; no note review"})
    if compare_training(item.gold, item, rule) is not True:
        return None, "UNSUPPORTED_GOLD_EXACT_FALLBACK"
    return rule, unit


def prepare(*, build_root: Path, manifest: Path, order: Path, limit: int, raw_root: Path, output: Path) -> dict:
    """Write a new private directory; input and output hashes are portable, with no timestamps."""
    if type(limit) is not int or limit < 1:
        raise ValueError("source evidence: positive limit required")
    if output.exists():
        raise ValueError("source evidence: refusing to overwrite output")
    build_root, raw_root = _safe_root(build_root), _safe_root(raw_root)
    frozen, manifest_sha = _json(manifest), file_sha(manifest)
    vault_manifest = _json(build_root / "manifest.yaml")
    vault_sha = file_sha(build_root / "manifest.yaml")
    freeze_binding = _json(build_root / "freeze_binding.json")
    if (frozen.get("version") != "pool-freeze-manifest-v1" or
            freeze_binding.get("version") != "pool-freeze-vault-binding-v1" or
            freeze_binding.get("dataset_manifest_sha256") != manifest_sha or
            freeze_binding.get("vault_manifest_sha256") != vault_sha):
        raise ValueError("source evidence: frozen manifest/vault binding mismatch")
    train = build_root / "train"
    train_sha = file_sha(train / "manifest.yaml")
    if vault_manifest["split_manifests"]["train"] != train_sha:
        raise ValueError("source evidence: train split binding mismatch")
    actors = {actor.problem_id: actor for actor in read_actors(train)}
    items, lineages = read_verifier(train), read_lineage(train)
    fixed_order = read_order(order)
    if limit > len(fixed_order["items"]):
        raise ValueError("source evidence: limit exceeds fixed order")
    manifest_items = {row["problem_id"]: row for row in frozen["items"]}
    if len(manifest_items) != len(frozen["items"]):
        raise ValueError("source evidence: duplicate frozen id")
    selected = {}
    for entry in fixed_order["items"][:limit]:
        pid = entry["problem_id"]
        if pid not in manifest_items or pid not in actors or pid not in items or pid not in lineages:
            raise ValueError("source evidence: selected id absent from frozen train vault")
        row, actor, lineage = manifest_items[pid], actors[pid], lineages[pid]
        if (row["status"] != "INCLUDED" or row["split"] != "train" or lineage.split != "train" or
                row["source"] != actor.source or entry["source"] != actor.source or
                row["source_revision"] != lineage.source_revision or row["subject"] != entry["subject"] or
                row["image_sha256"] != list(actor.image_sha256) or row["image_paths"] != list(actor.image_paths) or
                not isinstance(row["native_id"], str) or not row["native_id"].strip()):
            raise ValueError("source evidence: selected train identity mismatch")
        for name, record in (("actor", actor), ("verifier", items[pid]), ("lineage", lineage)):
            if row["record_sha256"][name] != text_sha256(to_json(record)):
                raise ValueError("source evidence: frozen record hash mismatch")
        for relative, sha in zip(actor.image_paths, actor.image_sha256, strict=True):
            if file_sha(_checked_file(build_root / "images", relative)) != sha:
                raise ValueError("source evidence: image hash mismatch")
        selected[pid] = row
    native, cache = _native_rows(raw_root, selected)
    rules, sources = {}, {}
    statuses: Counter[str] = Counter()
    availability: Counter[str] = Counter()
    for pid, row in selected.items():
        actor, item, lineage = actors[pid], items[pid], lineages[pid]
        references, scope = _annotations(actor, item, row, native)
        binding = {"actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                   "source": actor.source, "source_revision": lineage.source_revision, "split": "train", "native_id": row["native_id"]}
        rule, status = _rule(actor, item, lineage, row, references, binding)
        if rule is not None:
            rules[pid] = rule
        statuses[status] += 1
        availability[scope] += 1
        sources[pid] = binding | {"label_provenance": lineage.label_provenance, "human_verified": False,
            "vault_manifest_sha256": train_sha, "dataset_manifest_sha256": manifest_sha,
            "record_sha256": row["record_sha256"], "adapter_code_sha256": row["adapter_code_sha256"],
            "label": {"value": item.gold, "sha256": text_sha256(item.gold),
                      "location": f"vault:train/verifier_private.jsonl#{pid}:gold"},
            "adapter_policy": {"accept": item.accept, "sha256": digest(item.accept),
                               "provenance": "Frozen adapter policy, not source-authored tolerance"},
            "annotations": references, "content_verification_available": scope in (
                "SOURCE_EXPLANATION_FOR_REVIEW", "CURATED_ROLE_DESCRIPTION_FOR_REVIEW", "SOURCE_TRIPLE_FOR_REVIEW"),
            "content_verification_scope": scope, "note_content_status": "UNKNOWN_REVIEW_REQUIRED",
            "reference_use": "PRIVATE_REVIEW_INPUT_ONLY_NOT_TRAINING_TARGET", "unit_policy": status,
            "leak_aliases": [], "acronym_candidates": []}
    documents = {"rules.json": {"version": RULES_VERSION, "vault_manifest_sha256": train_sha, "items": rules},
                 "source-provenance.json": {"version": VERSION, "items": sources}}
    payloads = {name: json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=1) + "\n"
                for name, value in documents.items()}
    summary = {"version": "source-evidence-preparation-v1", "count": len(selected), "rules": len(rules),
               "selected_ids_sha256": digest(list(selected)), "order_sha256": file_sha(order),
               "dataset_manifest_sha256": manifest_sha, "vault_manifest_sha256": vault_sha,
               "train_manifest_sha256": train_sha, "cache": cache,
               "answer_policy": dict(statuses), "source_availability": dict(availability),
               "output_sha256": {name: text_sha256(text) for name, text in payloads.items()},
               "human_verified": False, "training_permission_granted": False}
    _safe_root(output.parent)
    output.mkdir(mode=0o700)
    for name, text in (payloads | {"summary.json": json.dumps(summary, sort_keys=True, indent=1) + "\n"}).items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(text)
        (output / name).chmod(0o600)
    return summary


def prepare_dev(*, build_root: Path, manifest: Path, output: Path) -> dict:
    """Pin all frozen dev non-MCQ policies for evaluation only; never extend training rules."""
    if output.exists():
        raise ValueError("dev evidence: refusing to overwrite output")
    build_root = _safe_root(build_root)
    frozen, manifest_sha = _json(manifest), file_sha(manifest)
    vault_sha = file_sha(build_root / "manifest.yaml")
    binding = _json(build_root / "freeze_binding.json")
    if (frozen.get("version") != "pool-freeze-manifest-v1" or
            binding.get("version") != "pool-freeze-vault-binding-v1" or
            binding.get("dataset_manifest_sha256") != manifest_sha or
            binding.get("vault_manifest_sha256") != vault_sha):
        raise ValueError("dev evidence: frozen manifest/vault binding mismatch")
    dev_root = build_root / "dev"
    dev_sha = file_sha(dev_root / "manifest.yaml")
    if _json(build_root / "manifest.yaml")["split_manifests"]["dev"] != dev_sha:
        raise ValueError("dev evidence: dev split binding mismatch")
    actors = {actor.problem_id: actor for actor in read_actors(dev_root)}
    items, lineages = read_verifier(dev_root), read_lineage(dev_root)
    rows = {row["problem_id"]: row for row in frozen["items"]}
    if len(rows) != len(frozen["items"]):
        raise ValueError("dev evidence: duplicate frozen id")
    selected = {pid: row for pid, row in rows.items() if row["status"] == "HELD_OUT_DEV"}
    if not selected or set(selected) != set(actors) or set(selected) != set(items) or set(selected) != set(lineages):
        raise ValueError("dev evidence: dev vault must exactly cover all held-out dev ids")
    rules = {}
    for pid in sorted(selected):
        row, actor, item, lineage = selected[pid], actors[pid], items[pid], lineages[pid]
        if (row["split"] != "dev" or lineage.split != "dev" or row["source"] != actor.source or
                lineage.source != actor.source or row["source_revision"] != lineage.source_revision or
                row["image_sha256"] != list(actor.image_sha256) or row["image_paths"] != list(actor.image_paths) or
                not isinstance(row["native_id"], str) or not row["native_id"].strip()):
            raise ValueError("dev evidence: frozen dev identity mismatch")
        for name, record in (("actor", actor), ("verifier", item), ("lineage", lineage)):
            if row["record_sha256"][name] != text_sha256(to_json(record)):
                raise ValueError("dev evidence: frozen record hash mismatch")
        for relative, sha in zip(actor.image_paths, actor.image_sha256, strict=True):
            if file_sha(_checked_file(build_root / "images", relative)) != sha:
                raise ValueError("dev evidence: image hash mismatch")
        if item.answer_type == "mcq":
            continue
        kind, accept = item.answer_type, dict(item.accept)
        range_rule = {"answer_type": "range", "accept": {}}
        if kind == "short_text" and not accept and compare_training(item.gold, item, range_rule) is True:
            kind = "range"
        rule = {"actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                "source": actor.source, "source_revision": lineage.source_revision, "split": "dev",
                "native_id": row["native_id"], "answer_type": kind, "accept": accept,
                "provenance": {"label_provenance": lineage.label_provenance,
                    "field": "verifier.gold + frozen adapter.accept",
                    "location": f"vault:dev/verifier_private.jsonl#{pid};adapter:{row['adapter_code_sha256']}",
                    "sha256": digest({"gold": item.gold, "answer_type": item.answer_type, "accept": item.accept,
                                      "adapter_code_sha256": row["adapter_code_sha256"]}),
                    "human_verified": False,
                    "review_scope": "Evaluation only: frozen native label syntax; numeric tolerance is adapter policy, not source annotation"}}
        if compare_training(item.gold, item, rule) is not True:
            raise ValueError("dev evidence: unsupported frozen answer policy")
        rules[pid] = rule
    document = {"version": "external-dev-answer-rules-v1", "vault_manifest_sha256": dev_sha, "items": rules}
    text = json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=1) + "\n"
    summary = {"version": "external-dev-evidence-preparation-v1", "count": len(selected), "rules": len(rules),
               "selected_ids_sha256": digest(sorted(selected)), "dataset_manifest_sha256": manifest_sha,
               "vault_manifest_sha256": vault_sha, "dev_manifest_sha256": dev_sha,
               "answer_types": dict(Counter(rule["answer_type"] for rule in rules.values())),
               "output_sha256": {"rules.json": text_sha256(text)}, "human_verified": False,
               "training_allowed": False}
    _safe_root(output.parent)
    output.mkdir(mode=0o700)
    for name, value in {"rules.json": text, "summary.json": json.dumps(summary, sort_keys=True, indent=1) + "\n"}.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(value)
        (output / name).chmod(0o600)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in ("build-root", "manifest", "order", "raw-root", "output"):
        prepare_parser.add_argument(f"--{name}", type=Path, required=True)
    prepare_parser.add_argument("--limit", type=int, required=True)
    dev_parser = commands.add_parser("prepare-dev")
    for name in ("build-root", "manifest", "output"):
        dev_parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-dev":
        summary = prepare_dev(build_root=args.build_root, manifest=args.manifest, output=args.output)
    else:
        summary = prepare(build_root=args.build_root, manifest=args.manifest, order=args.order,
                          limit=args.limit, raw_root=args.raw_root, output=args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
