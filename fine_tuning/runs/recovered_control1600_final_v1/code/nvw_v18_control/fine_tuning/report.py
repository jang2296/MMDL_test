"""Private saved-result report; no model calls, training, or public baseline replacement."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION, SUBJECTS

from .audit import dump, read_rows, sha
from .run_experiment import (ARMS, PAIRED_INPUT_FIELDS, assert_same_exposure, closed_controller,
                             exposure_matches, pilot_scale_evidence, reuse_pilot, reuse_pilot_evaluations)

LIMITS = [
    "MMMU 60문항(30과목×2) 진단이며 전체 900문항 평가가 아니다.",
    "4096-token 진단은 기존 62.00%/32768-token V8과 직접 비교할 수 없다.",
    "공통 pilot은 실제 승인된 source 혼합만 측정하며 전문 분야 학습 혼합의 완료가 아니다.",
    "외부 QA/구조/상태는 서로 다른 지표이며 전체 MMMU 정확도로 합산하지 않는다.",
    "출처 또는 내부 모듈의 인과 책임을 인증하지 않는다. CPU mock 검사는 GPU 실행 증거가 아니다.",
    "50K 확장과 계획된 전체 ablation의 달성을 주장하지 않는다.",
]


def load_evaluation(path: Path, baseline: bool = False) -> tuple[dict, list[dict]]:
    summary = json.loads((path / "summary.json").read_text())
    rows = read_rows(path / "predictions.jsonl")
    done = [row for row in rows if row.get("status") == "COMPLETED"]
    full = len(done) == len(rows) and bool(rows)
    correct = sum(row.get("correct") is True for row in done)
    if (summary.get("model_id") != MODEL_ID or summary.get("model_revision") != MODEL_REVISION
            or summary.get("predictions_sha256") != sha(path / "predictions.jsonl")
            or summary.get("denominator") != len(rows) or len(rows) != 60
            or len({row["id"] for row in rows}) != len(rows)
            or summary.get("completed") != len(done) or summary.get("correct") != correct
            or any(type(row.get("correct")) is not bool for row in done)
            or any(row.get("status") not in {"COMPLETED", "BUDGET_STOP"} for row in rows)
            or any(row.get("correct") is not None for row in rows if row["status"] == "BUDGET_STOP")
            or any(row["status"] == "COMPLETED" for row in rows[len(done):])
            or summary.get("status") != ("EVALUATED" if full else "BUDGET_STOP")
            or summary.get("accuracy") != (correct / len(rows) if full else None)
            or not summary.get("cohort_sha256") or not summary.get("protocol_sha256")
            or baseline and (not full or summary.get("checkpoint") != "F0-frozen")):
        raise ValueError(f"Invalid saved evaluation identity/hash/coverage/arithmetic: {path}")
    return summary, rows


def compare_inputs(reference: list[dict], rows: list[dict]) -> None:
    if [row["id"] for row in reference] != [row["id"] for row in rows]:
        raise ValueError("Paired evaluation ID order differs")
    for before, after in zip(reference, rows, strict=True):
        for field in (*PAIRED_INPUT_FIELDS, "subject"):
            if (any(field not in row or row[field] is None for row in (before, after))
                    or json.dumps(before[field], sort_keys=True, allow_nan=False)
                    != json.dumps(after[field], sort_keys=True, allow_nan=False)):
                raise ValueError(f"Paired observed field differs or is missing: {after['id']}/{field}")


def paired_counts(reference: list[dict], rows: list[dict]) -> dict:
    counts = Counter({"both_correct": 0, "both_wrong": 0, "gain": 0, "loss": 0})
    for before, after in zip(reference, rows, strict=True):
        if type(before.get("correct")) is not bool or type(after.get("correct")) is not bool:
            raise ValueError("Only completed boolean correctness supports paired counts")
        counts["both_correct" if before["correct"] and after["correct"] else
               "both_wrong" if not before["correct"] and not after["correct"] else
               "gain" if after["correct"] else "loss"] += 1
    return dict(counts, count=len(rows), net=counts["gain"]-counts["loss"],
                f0_correct=sum(row["correct"] for row in reference),
                ft_correct=sum(row["correct"] for row in rows))


def external_groups(reference: list[dict], rows: list[dict]) -> dict:
    groups = {}
    for kind in sorted({row["metric_kind"] for row in rows}):
        if any(before.get("metric_kind") != after.get("metric_kind")
               for before, after in zip(reference, rows, strict=True)):
            raise ValueError("Paired external metric kinds differ")
        selected = [i for i, row in enumerate(rows) if row["metric_kind"] == kind]
        before, after = [reference[i] for i in selected], [rows[i] for i in selected]
        item = paired_counts(before, after)
        for label, values in (("f0", before), ("ft", after)):
            metrics = {}
            for field in ("structure_f1", "supported_states_exact", "rules_exact"):
                observed = [row[field] for row in values if field in row]
                if observed:
                    if any(type(value) not in {int, float, bool} or not math.isfinite(value)
                           or not 0 <= value <= 1 for value in observed):
                        raise ValueError(f"Invalid stored metric: {field}")
                    metrics[field] = {"mean": sum(observed)/len(observed), "observed": len(observed),
                                      "missing": len(values)-len(observed)}
            boxes = [row for row in values if "bbox_hits" in row and "bbox_denominator" in row]
            if boxes:
                hits, denominator = sum(row["bbox_hits"] for row in boxes), sum(row["bbox_denominator"] for row in boxes)
                if any(type(row[key]) is not int or not 0 <= row["bbox_hits"] <= row["bbox_denominator"]
                       for row in boxes for key in ("bbox_hits", "bbox_denominator")):
                    raise ValueError("Invalid stored bbox counts")
                metrics["bbox"] = {"hits": hits, "denominator": denominator,
                                    "rate": hits/denominator if denominator else None,
                                    "observed": len(boxes), "missing": len(values)-len(boxes)}
            item[label] = metrics
        groups[kind] = item
    return groups


def frozen_pilot_scale(controller: Path) -> dict:
    path = controller/"pilot_input_freeze.json"
    if not path.exists():
        return {"original_records": 10000, "actual_records": 10000, "scaled": False,
                "source_counts": {}, "legacy_default": True}
    frozen = json.loads(path.read_text())
    scale = frozen.get("scale", {"original_records": 10000, "actual_records": 10000, "scaled": False})
    actual, original = scale.get("actual_records"), scale.get("original_records")
    if (type(actual) is not int or type(original) is not int or not 0 < actual <= original
            or scale.get("scaled") is not (actual < original)):
        raise ValueError("Invalid frozen pilot scale")
    if actual < original:
        declared = scale.get("pilot_scale", {})
        if (actual < 400 or actual % 40 or scale.get("explicit_override") is not True
                or declared.get("selected_records") != actual or declared.get("original_records") != original
                or scale.get("source_counts") != {"trace": actual*9//10, "nvw": actual//10}
                or scale.get("nvw_tracks") != {"NVW-K": actual//20, "NVW-I": actual//40, "NVW-S": actual//40}
                or scale.get("nvw_view_counts") != {"qa": actual//20, "structure": actual//40, "grounded": actual//40}
                or scale.get("original_10K_execution_completed") is not False):
            raise ValueError("Reduced report requires explicit frozen approved scale/quota evidence")
    return dict(scale, freeze_sha256=sha(path), protocol_sha256=frozen.get("protocol_sha256"),
                records_sha256=frozen.get("pilot_records_sha256"),
                manifest_sha256=frozen.get("pilot_manifest_sha256"))


def checkpoint_evidence(controller: Path, arm: str, journal: list[dict],
                        scale: dict | None = None) -> tuple[dict, dict | None]:
    checkpoint = controller / f"pilot_{arm}" / "checkpoint"
    metadata_path = checkpoint / "metadata.json"
    if not metadata_path.exists():
        return {"status": "未完成", "reason": "Pilot checkpoint missing"}, None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("arm") != arm or metadata.get("base_id") != MODEL_ID or metadata.get("base_revision") != MODEL_REVISION:
        raise ValueError("Checkpoint arm/base identity mismatch")
    files = metadata.get("files", {})
    for name, digest in files.items():
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint.resolve()) or sha(path) != digest:
            raise ValueError("Checkpoint payload hash mismatch")
    reload_verified = any(event.get("stage") == f"pilot_{arm}_reload" and event.get("status") == "EXITED"
                          and event.get("returncode") == 0 for event in journal)
    scale = scale or {"actual_records": 10000}
    if any(scale.get(key) and metadata.get(key) != scale[key] for key in ("records_sha256", "manifest_sha256")):
        raise ValueError("Checkpoint does not match frozen common pilot data")
    complete = (exposure_matches(metadata) and metadata.get("planned_records") == scale["actual_records"]
                and metadata.get("termination") == "EPOCH_COMPLETE" and reload_verified and bool(files))
    return {"status": "COMPLETED" if complete else "未完成", "path": str(checkpoint.resolve()),
            "metadata_sha256": sha(metadata_path),
            "checkpoint_files": files, "exposure": metadata.get("exposure"),
            "planned_records": metadata.get("planned_records"), "termination": metadata.get("termination"),
            "consumed_order_sha256": metadata.get("consumed_order_sha256"),
            "fresh_reload_verified": reload_verified}, metadata if complete else None


def verified_reuse(controller: Path, journal: list[dict], baselines: dict) -> tuple[dict, dict, dict]:
    """Recheck original absolute artifacts; receipts never stand in for parent evidence."""
    training_path, evaluation_path = controller/"pilot_reuse.json", controller/"pilot_evaluation_reuse.json"
    if not training_path.exists():
        if evaluation_path.exists():
            raise ValueError("Evaluation reuse requires a validated training reference")
        return {}, {}, {}
    freeze = json.loads((controller/"pilot_input_freeze.json").read_text())

    def frozen_path(digest: str) -> Path:
        paths = [Path(name) for name, value in freeze["files"].items() if value == digest]
        if len(paths) != 1 or not paths[0].is_absolute() or sha(paths[0]) != digest:
            raise ValueError("Reuse requires one hash-bound original absolute input path")
        return paths[0]

    def receipt(path: Path, kind: str) -> dict:
        saved = json.loads(path.read_text())
        events = [event for event in journal if event.get("stage") == path.stem]
        if (saved.get("status") != "PASS" or saved.get("kind") != kind
                or not Path(saved.get("parent", "")).is_absolute()
                or not events or events[-1].get("status") != "PASS"
                or events[-1].get("receipt_sha256") != sha(path)):
            raise ValueError("Reuse receipt identity/current journal hash mismatch")
        return saved

    records = frozen_path(freeze["pilot_records_sha256"])
    manifest = frozen_path(freeze["pilot_manifest_sha256"])
    protocol = frozen_path(freeze["protocol_sha256"])
    if str(manifest) != freeze.get("pilot_manifest") or not Path(freeze["image_root"]).is_absolute():
        raise ValueError("Reuse manifest/image-root reference mismatch")
    config, rows = json.loads(protocol.read_text()), read_rows(records)
    scale = freeze["scale"]
    actual_scale = pilot_scale_evidence(rows, json.loads(manifest.read_text()), config["training"]["pilot_records"],
        scale["actual_records"] if scale.get("explicit_override") else None, 360, config["master_seed"])
    if actual_scale != scale or freeze["seed"] != config["master_seed"]:
        raise ValueError("Reuse frozen scale/seed differs from actual common data")
    saved = receipt(training_path, "CLOSED_PILOT_TRAIN_REUSE")
    parent = Path(saved["parent"])
    parent_freeze = json.loads((parent/"pilot_input_freeze.json").read_text())
    if parent_freeze.get("scale") != scale:
        raise ValueError("Reused training parent has a different frozen pilot scale")
    _pilots, _files, expected = reuse_pilot(parent, rows, records, manifest, config["training"],
        config["master_seed"], protocol, Path(freeze["image_root"]))
    if saved != expected:
        raise ValueError("Training reuse receipt differs from closed parent evidence")
    _status, parent_journal, _frozen = closed_controller(parent)
    checkpoints = {arm: Path(value["checkpoint"]) for arm, value in saved["arms"].items()}
    training = {arm: (parent, parent_journal) for arm in checkpoints}
    evidence = {"training": {"receipt_sha256": sha(training_path), "parent": str(parent),
                             "training_repeated": False, "arms": saved["arms"]}}
    evaluations: dict = {}
    if evaluation_path.exists():
        saved = receipt(evaluation_path, "CLOSED_PILOT_EVALUATION_REUSE")
        cohorts = {name: frozen_path(summary["cohort_sha256"]) for name, (_path, summary, _rows) in baselines.items()}
        evaluations, _files, expected = reuse_pilot_evaluations(Path(saved["parent"]), checkpoints,
            {name: value[0] for name, value in baselines.items()},
            {name: path.parent for name, path in cohorts.items()}, protocol)
        if saved != expected:
            raise ValueError("Evaluation reuse receipt differs from closed paired parent evidence")
        evidence["evaluation"] = {"receipt_sha256": sha(evaluation_path), "parent": saved["parent"],
                                  "evaluation_repeated": False, "arms": saved["arms"]}
    return training, evaluations, evidence


def build_report(controller: Path, baseline_mmmu: Path, baseline_external: Path, output: Path) -> dict:
    baselines = {name: (path, *load_evaluation(path, baseline=True))
                 for name, path in (("mmmu", baseline_mmmu), ("external", baseline_external))}
    if (Counter(row["subject"] for row in baselines["mmmu"][2]) != Counter({name: 2 for name in SUBJECTS})
            or baselines["mmmu"][1]["protocol_sha256"] != baselines["external"][1]["protocol_sha256"]):
        raise ValueError("F0 must use one protocol and exactly two samples per MMMU subject")
    journal = read_rows(controller / "journal.jsonl") if (controller / "journal.jsonl").exists() else []
    scale = frozen_pilot_scale(controller)
    reused_training, reused_evaluations, reuse_evidence = verified_reuse(controller, journal, baselines)
    if scale.get("protocol_sha256") and scale["protocol_sha256"] != baselines["mmmu"][1]["protocol_sha256"]:
        raise ValueError("Training scale must retain the frozen F0 inference protocol")
    limits = list(LIMITS)
    counts = scale.get("source_counts", {})
    if counts:
        limits.append(f"실제 공통 pilot 목표 {scale['actual_records']} records: " +
                      ", ".join(f"{name}={count}" for name,count in sorted(counts.items())) + ".")
    if scale["scaled"]:
        limits.append(f"원래 {scale['original_records']}-record pilot은 수행하지 않았으며 축소 공통 pilot만 비교한다.")
    result: dict = {"status": "PRIVATE_SAVED_RESULT_REPORT", "training_allowed": False, "public_upload_allowed": False,
              "limits": limits, "training_scale": scale, "controller_status": json.loads((controller / "status.json").read_text())
              if (controller / "status.json").exists() else {"status": "未完成"}, "baselines": {}, "arms": {},
              "reuse": reuse_evidence}
    for name, (path, summary, _rows) in baselines.items():
        if summary.get("benchmark_derived") is not (name == "mmmu"):
            raise ValueError("Evaluation benchmark/external identity mismatch")
        result["baselines"][name] = {"summary_sha256": sha(path / "summary.json"),
                                    "predictions_sha256": summary["predictions_sha256"],
                                    "correct": summary["correct"], "count": summary["denominator"]}
    completed_metadata = {}
    for arm in ARMS:
        training_root, training_journal = reused_training.get(arm, (controller, journal))
        if arm in reused_training and (controller/f"pilot_{arm}").exists():
            raise ValueError("Reused arm also has a new local training artifact")
        evidence, metadata = checkpoint_evidence(training_root, arm, training_journal, scale)
        evidence.update(training_reused=arm in reused_training, training_controller=str(training_root.resolve()))
        item: dict = {"checkpoint": evidence}
        if metadata is not None:
            completed_metadata[arm] = metadata
        for name, (baseline_path, baseline_summary, reference) in baselines.items():
            path = reused_evaluations.get((arm, name), controller / f"pilot_{arm}_{name}")
            if (arm, name) in reused_evaluations and (controller/f"pilot_{arm}_{name}").exists():
                raise ValueError("Reused evaluation also has a new local result artifact")
            if metadata is None or not (path / "summary.json").exists():
                item[name] = {"status": "未完成", "accuracy": None, "reason": "Pilot exposure/reload or evaluation missing"}
                continue
            summary, rows = load_evaluation(path)
            if (summary["cohort_sha256"] != baseline_summary["cohort_sha256"]
                    or summary["protocol_sha256"] != baseline_summary["protocol_sha256"]
                    or summary.get("benchmark_derived") is not (name == "mmmu")
                    or summary.get("checkpoint_metadata_sha256") != evidence["metadata_sha256"]
                    or Path(summary.get("checkpoint", "")).resolve() != Path(evidence["path"])):
                raise ValueError("Pilot/F0 cohort/protocol/checkpoint metadata mismatch")
            if summary["status"] != "EVALUATED":
                item[name] = {"status": "未完成", "accuracy": None, "completed": summary["completed"],
                              "count": summary["denominator"], "predictions_sha256": summary["predictions_sha256"]}
                continue
            compare_inputs(reference, rows)
            receipt_path = path / "input_equivalence.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                expected = {"baseline_summary_sha256": sha(baseline_path / "summary.json"),
                            "baseline_predictions_sha256": baseline_summary["predictions_sha256"],
                            "pilot_summary_sha256": sha(path / "summary.json"),
                            "pilot_predictions_sha256": summary["predictions_sha256"]}
                if receipt.get("status") != "PASS" or any(receipt.get(key) != value for key, value in expected.items()):
                    raise ValueError("Input-equivalence receipt hash mismatch")
            scored = {"status": "COMPLETED", "paired": paired_counts(reference, rows),
                      "path": str(path.resolve()), "evaluation_reused": (arm, name) in reused_evaluations,
                      "summary_sha256": sha(path / "summary.json"), "predictions_sha256": summary["predictions_sha256"],
                      "actual_inputs_equal": True}
            if name == "mmmu":
                scored["accuracy"] = summary["accuracy"]
                scored["subjects"] = {subject: paired_counts([row for row in reference if row["subject"] == subject],
                    [row for row in rows if row["subject"] == subject]) for subject in SUBJECTS}
            else:
                scored["metric_kind_counts"] = dict(Counter(row["metric_kind"] for row in rows))
                scored["groups"] = external_groups(reference, rows)
                scored["mixed_metric_accuracy_is_not_MMMU"] = True
            item[name] = scored
        item["status"] = "COMPLETED" if all(item[name]["status"] == "COMPLETED" for name in baselines) else "未完成"
        result["arms"][arm] = item
    if completed_metadata:
        assert_same_exposure(completed_metadata)
    result["original_10K_execution_completed"] = (scale["actual_records"] == 10000
        and all(item["status"] == "COMPLETED" for item in result["arms"].values()))
    lines = ["# 비공개 pilot 결과", "", *("- "+limit for limit in limits), "",
             "| Arm | Pilot | MMMU60 | External60 |", "|---|---|---|---|"]
    for arm, item in result["arms"].items():
        mmmu = item["mmmu"]
        label = f"{mmmu['paired']['ft_correct']}/60; net {mmmu['paired']['net']:+d}" if mmmu["status"] == "COMPLETED" else "未完成"
        pilot_label = item['checkpoint']['status'] + (" (closed parent reused; no new training)"
                                                     if item['checkpoint'].get('training_reused') else "")
        lines.append(f"| {arm} | {pilot_label} | {label} | {item['external']['status']} |")
        for name in baselines:
            value = item[name]
            if value["status"] == "COMPLETED":
                paired = value["paired"]
                lines.extend(["", f"## {arm} {name}", "",
                    f"Both correct {paired['both_correct']}; both wrong {paired['both_wrong']}; "
                    f"gain {paired['gain']}; loss {paired['loss']}; net {paired['net']:+d}.", ""])
                if value["evaluation_reused"]:
                    lines.extend([f"Closed parent evaluation reused; no new inference: `{value['path']}`.", ""])
                if name == "mmmu":
                    lines.extend(["| Subject | F0 | FT | Gain | Loss | Net |", "|---|---|---|---|---|---|"])
                    for subject, counts in value["subjects"].items():
                        lines.append(f"| {subject} | {counts['f0_correct']}/2 | {counts['ft_correct']}/2 | "
                                     f"{counts['gain']} | {counts['loss']} | {counts['net']:+d} |")
                else:
                    lines.extend(["| Metric kind | Count | F0 exact | FT exact | Gain | Loss |",
                                  "|---|---|---|---|---|---|"])
                    for kind, counts in value["groups"].items():
                        lines.append(f"| {kind} | {counts['count']} | {counts['f0_correct']} | "
                                     f"{counts['ft_correct']} | {counts['gain']} | {counts['loss']} |")
                    for kind, counts in value["groups"].items():
                        lines.extend(["", f"Stored {kind} metrics (coverage included):", "",
                                      "```json", json.dumps({key: counts[key] for key in ("f0", "ft")},
                                                            ensure_ascii=False, sort_keys=True), "```", ""])
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "report.json", result)
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("controller", "baseline-mmmu", "baseline-external", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    build_report(args.controller, args.baseline_mmmu, args.baseline_external, args.output)


if __name__ == "__main__":
    main()
