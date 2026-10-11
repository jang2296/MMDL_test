"""Frozen, benchmark-derived selection and gold-blind P0 input interventions."""

from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from experiments.bottleneck_diagnostic.ingest import read_rows, sha, write_json, write_rows
from experiments.bottleneck_diagnostic.input_builder import canonical_hash, stage_seed, validate_sample
from experiments.bottleneck_diagnostic.metrics import CLUSTERS, cluster
from experiments.bottleneck_diagnostic.oracle_validation import validate_oracle
from mmdl.evaluation.prompt import build_prompt

MODES = ("I", "T", "IT", "IR", "TR")
REVIEW_POLICY = "ALLOW_AI_REVIEWED_DIAGNOSTIC"
EXCLUDED = {
    "validation_Energy_and_Power_20": "QUESTION_DERIVED_O0",
    "validation_Math_16": "TARGET_REVEALED",
}
PREFERRED_IDS = {
    f"validation_{subject}_{number}" for subject, number in (
        ("Music", 18), ("Music", 2), ("Geography", 27),
        ("Mechanical_Engineering", 10), ("Biology", 27), ("Accounting", 22),
        ("Energy_and_Power", 26), ("Energy_and_Power", 3),
    )
}


def build_request(sample: dict[str, Any], mode: str,
                  oracle: dict[str, Any] | None = None,
                  rule: dict[str, Any] | None = None) -> dict[str, Any]:
    """One independent user message; P0 and original image order remain intact."""
    validate_sample(sample)
    if mode not in MODES or not sample["images"]:
        raise ValueError("Unknown mode or missing actual sample images")
    if sample["sample_id"] in EXCLUDED:
        raise ValueError("Sample is excluded from input recovery")
    text = build_prompt(sample, Path(__file__).resolve().parents[2])
    verdict: dict[str, str] = {}
    if mode in {"T", "IT", "TR"}:
        verdict = validate_oracle(oracle, sample, "O0", REVIEW_POLICY)
        if verdict["status"] != "VALID":
            raise ValueError(f"Illegal O0 oracle: {verdict['reason']}")
        assert oracle is not None
        facts = oracle["facts"] if isinstance(oracle["facts"], str) else json.dumps(
            oracle["facts"], ensure_ascii=False, sort_keys=True,
        )
        text = f"Provided visual information (O0):\n{facts}\n\n{text}"
    elif oracle is not None:
        raise ValueError("Oracle is only allowed in T/IT/TR")
    if mode in {"IR", "TR"}:
        if (not isinstance(rule, dict)
                or set(rule) != {"subject", "version", "text", "sources", "approved"}
                or rule["subject"] != sample["subject"] or rule["approved"] is not True
                or any(not isinstance(rule[key], str) or not rule[key].strip()
                       for key in ("subject", "version", "text"))
                or not isinstance(rule["sources"], list) or not rule["sources"]
                or any(not isinstance(url, str) or urlparse(url).scheme not in {"https", "http"}
                       or not urlparse(url).hostname for url in rule["sources"])):
            raise ValueError("Rule must be approved, subject-fixed and carry version/text/source URLs")
        text = f"Provided subject rule ({rule['subject']}):\n{rule['text']}\n\n{text}"
    elif rule is not None:
        raise ValueError("Rule is only allowed in IR/TR")
    images = sample["images"] if mode in {"I", "IT", "IR"} else []
    messages = [{"role": "user", "content": [
        *[{"type": "image", "image": item["path"]} for item in images],
        {"type": "text", "text": text},
    ]}]
    result = {
        "sample_id": sample["sample_id"], "request_id": f"{sample['sample_id']}::{mode}",
        "subject": sample["subject"], "mode": mode, "seed": stage_seed(3407, sample["sample_id"], mode),
        "max_new_tokens": 8192, "messages": messages, "images": [dict(im) for im in images],
        "sample_hash": canonical_hash(sample), "prompt_hash": canonical_hash(messages),
        "source_image_hashes": [im["sha256"] for im in sample["images"]],
        "oracle_hash": oracle["oracle_hash"] if oracle is not None else None,
        "oracle_review_policy": REVIEW_POLICY, "verification_kind": verdict.get("verification_kind"),
        "oracle_review": {key: oracle[key] for key in (
            "annotation_version", "annotator_id", "reviewer_id", "reviewer_status",
            "annotator_exposure_status", "completeness_status", "target_leakage_status",
        )} if oracle is not None else None,
        "rule": dict(rule) if rule is not None else None,
        "rule_hash": canonical_hash(rule) if rule is not None else None,
        "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
        "public_upload_allowed": False,
    }
    result["cache_key"] = canonical_hash(result)
    return result


def _select_cluster(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bounded subject allocation: six per outcome, at most four per subject."""
    by_id = {row["sample_id"]: row for row in candidates}
    ranked = sorted(by_id, key=lambda sid: (canonical_hash([3407, "input-recovery-v1", sid]), sid))
    ranks = {sid: index for index, sid in enumerate(ranked)}

    def cost(ids: tuple[str, ...]) -> tuple[Any, ...]:
        return (-sum(sid in PREFERRED_IDS for sid in ids), sum(ranks[sid] for sid in ids), ids)

    states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
    for subject in sorted({row["subject"] for row in candidates}):
        groups = {correct: sorted(
            [r["sample_id"] for r in candidates
             if r["subject"] == subject and r["historical_correct"] is correct],
            key=lambda sid: (sid not in PREFERRED_IDS, ranks[sid], sid),
        ) for correct in (True, False)}
        following: dict[tuple[int, int], tuple[str, ...]] = {}
        for (right, wrong), chosen in states.items():
            for nright in range(min(4, len(groups[True]), 6 - right) + 1):
                for nwrong in range(min(4 - nright, len(groups[False]), 6 - wrong) + 1):
                    ids = tuple(sorted((*chosen, *groups[True][:nright], *groups[False][:nwrong])))
                    key = (right + nright, wrong + nwrong)
                    if key not in following or cost(ids) < cost(following[key]):
                        following[key] = ids
        states = following
    if (6, 6) not in states:
        raise ValueError("Infeasible 12-item cluster with 6/6 historical outcomes and subject cap 4")
    return [by_id[sid] for sid in states[6, 6]]


def prepare(source_root: str | Path, output_root: str | Path,
            baseline_predictions_path: str | Path, *,
            g6_predictions_paths: dict[str, str | Path] | None = None) -> dict[str, Any]:
    """Create a new frozen 48-item bundle; never modify historical sources."""
    source_root = Path(source_root).resolve(strict=True)
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Preparation requires a new output directory")
    resolved_output = output_root.resolve()
    if resolved_output.is_relative_to(source_root) or source_root.is_relative_to(resolved_output):
        raise ValueError("Output must be separate from historical source root")
    sources: dict[str, dict[str, Any]] = {}

    def load(name: str, path: Path) -> list[dict[str, Any]]:
        path = path.resolve(strict=True)
        before = sha(path)
        rows = read_rows(path)
        if sha(path) != before:
            raise ValueError(f"Source changed during read: {name}")
        sources[name] = {"path": str(path), "sha256": before, "bytes": path.stat().st_size}
        return rows

    def index(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
        result = {row[key]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError(f"Duplicate {key} in historical source")
        return result

    samples = index(load("samples", source_root / "prepared/samples.jsonl"), "sample_id")
    gold = index(load("gold", source_root / "prepared/gold.jsonl"), "sample_id")
    baseline = index(load("baseline_predictions", Path(baseline_predictions_path)), "id")
    if (len(baseline) != 900 or any(type(row.get("correct")) is not bool for row in baseline.values())
            or sum(row["correct"] for row in baseline.values()) != 558
            or set(samples) != set(gold) or set(samples) != set(baseline)):
        raise ValueError("Expected complete V8 900-item/558-correct sources with identical IDs")
    for sid, sample in samples.items():
        validate_sample(sample)
        answer = gold[sid]
        if (type(answer.get("historical_correct")) is not bool
                or answer["historical_correct"] != baseline[sid]["correct"]
                or answer.get("answer") != baseline[sid].get("answer")
                or answer.get("subject") != sample["subject"]
                or baseline[sid].get("subject") != sample["subject"]
                or any(baseline[sid].get(key) != sample[key]
                       for key in ("question", "question_type", "options"))
                or answer.get("provenance") != "BENCHMARK_DERIVED"
                or answer.get("training_allowed") is not False):
            raise ValueError(f"Historical gold/baseline mismatch: {sid}")
    oracles: dict[str, dict[str, Any]] = {}
    for name in ("pilot_reviewed", "coverage_reviewed", "expansion_reviewed"):
        for oracle in load(name, source_root / "oracles" / f"{name}.jsonl"):
            sid = oracle["sample_id"]
            if oracle.get("level") != "O0" or sid not in samples:
                continue
            if sid in oracles and oracle != oracles[sid]:
                raise ValueError(f"Conflicting historical O0 records: {sid}")
            oracles[sid] = oracle
    candidates = []
    exclusions = dict(EXCLUDED)
    for sid, sample in samples.items():
        if cluster(sample["subject"]) not in CLUSTERS or sid in EXCLUDED:
            continue
        verdict = validate_oracle(oracles.get(sid), sample, "O0", REVIEW_POLICY)
        if verdict["status"] != "VALID" or not sample["images"]:
            exclusions[sid] = "NO_REFERENCED_IMAGE" if not sample["images"] else verdict["reason"]
            continue
        candidates.append({"sample_id": sid, "subject": sample["subject"],
                           "historical_correct": gold[sid]["historical_correct"]})
    strata = [row for group in CLUSTERS for row in _select_cluster(
        [row for row in candidates if cluster(row["subject"]) == group],
    )]
    selected_ids = [row["sample_id"] for row in strata]
    g6: dict[str, dict[str, Any]] = {}
    if g6_predictions_paths:
        if set(g6_predictions_paths) != {"F0", "FL", "FM", "FV", "FML"}:
            raise ValueError("G6 history requires all five explicit old ledgers")
        g6 = {arm: index(load(f"g6_{arm}", Path(path)), "id")
              for arm, path in sorted(g6_predictions_paths.items())}
        identities = [set(rows) for rows in g6.values()]
        if len(identities[0]) != 60 or any(ids != identities[0] for ids in identities):
            raise ValueError("Historical G6 ledgers must cover the same 60 items")
        if any(type(row.get("correct")) is not bool for rows in g6.values() for row in rows.values()):
            raise ValueError("Historical G6 correctness must be boolean")
    for row in strata:
        sid = row["sample_id"]
        row["cluster"] = cluster(row["subject"])
        row["g6_correctness"] = {arm: rows[sid]["correct"] for arm, rows in g6.items()
                                 if sid in rows}
        row["g6_kept_by"] = [arm for arm, value in row["g6_correctness"].items()
                             if arm != "F0" and value and row["g6_correctness"]["F0"]]
        row["g6_lost_by"] = [arm for arm, value in row["g6_correctness"].items()
                             if arm != "F0" and not value and row["g6_correctness"]["F0"]]
    subset: list[str] = []
    for group in CLUSTERS:
        members = [row for row in strata if row["cluster"] == group]
        combinations = [rows for rows in itertools.combinations(members, 3)
                        if {row["historical_correct"] for row in rows} == {True, False}]
        chosen = min(combinations, key=lambda rows: (
            -sum(any(row[field] for row in rows) for field in ("g6_kept_by", "g6_lost_by")),
            -sum(row["sample_id"] in PREFERRED_IDS for row in rows),
            canonical_hash([3407, "input-recovery-subset-v1", [row["sample_id"] for row in rows]]),
        ))
        subset.extend(row["sample_id"] for row in chosen)
    selected_samples = [samples[sid] for sid in selected_ids]
    assets: dict[str, str] = {}
    for sample in selected_samples:
        for image in sample["images"]:
            path = (source_root / "prepared" / image["path"]).resolve(strict=True)
            if not path.is_relative_to(source_root / "prepared") or sha(path) != image["sha256"]:
                raise ValueError("Selected image escapes source root or differs from declared SHA256")
            if image["path"] in assets and assets[image["path"]] != image["sha256"]:
                raise ValueError("Conflicting image paths")
            assets[image["path"]] = image["sha256"]
    for record in sources.values():
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("Historical source changed before preparation")
    selection = {
        "seed": 3407, "selection_policy": "Fixed preferred G6 overlap, then canonical SHA256 rank;"
        " six correct/six wrong per cluster, subject cap four; only historical results used",
        "selected_ids": selected_ids, "subset_ids": subset, "strata": strata,
        "preferred_missing": sorted(PREFERRED_IDS - set(selected_ids)),
        "preferred_limit": "Preferences never relax oracle validity, 6/6 strata or subject cap",
        "exclusions": exclusions, "source_files": sources, "image_files": assets,
        "oracle_hashes": {sid: oracles[sid]["oracle_hash"] for sid in selected_ids},
        "sample_hashes": {sid: canonical_hash(samples[sid]) for sid in selected_ids},
        "gold_hashes": {sid: canonical_hash(gold[sid]) for sid in selected_ids},
        "oracle_review_policy": REVIEW_POLICY,
        "verification_kind": "AI_REVIEWED_NOT_HUMAN_VERIFIED",
        "g6_history_scope": "Old 4K/P0 outcomes only; no prediction of new 8K outcomes"
        if g6 else "G6 history not supplied; no keep/loss claims",
        "g6_subset_limitations": {group: {
            field: not any(row[field] for row in strata if row["cluster"] == group)
            for field in ("g6_kept_by", "g6_lost_by")
        } for group in CLUSTERS},
        "warning": "Balanced diagnostic selection is not MMMU population accuracy",
        "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
        "public_upload_allowed": False,
    }
    selection["selection_hash"] = canonical_hash(selection)
    output_root.mkdir(parents=True, exist_ok=False)
    for name, digest in assets.items():
        target = output_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / "prepared" / name, target)
        if sha(target) != digest:
            raise ValueError("Copied image hash mismatch")
    write_rows(output_root / "samples.jsonl", selected_samples)
    write_rows(output_root / "oracles.jsonl", [oracles[sid] for sid in selected_ids])
    write_rows(output_root / "gold.jsonl", [gold[sid] for sid in selected_ids])
    for record in sources.values():
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("Historical source changed during preparation; bundle is not approved")
    write_json(output_root / "selection.json", selection)
    return selection
