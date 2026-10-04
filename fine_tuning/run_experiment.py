"""Owned-Pod subprocess controller; frozen sources/data and explicit GPU/balance cutoff."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .audit import dump, read_rows, sha

ARMS = ("FL", "FM", "FV", "FML")
PAIRED_INPUT_FIELDS = ("input_sha256", "prompt_token_ids", "image_grid_thw",
                       "processed_image_sizes", "seed")


def utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC deadlines require explicit timezone")
    return parsed.astimezone(dt.timezone.utc)


def effective_deadline(gpu_cutoff: dt.datetime, billing_start: dt.datetime,
                       balance_usd: float, reserve_usd: float, hourly_usd: float) -> dt.datetime:
    if not all(math.isfinite(v) for v in (balance_usd, reserve_usd, hourly_usd)):
        raise ValueError("Nonfinite financial budget")
    if hourly_usd <= 0 or reserve_usd < 0 or balance_usd <= reserve_usd:
        raise ValueError("Positive rate and balance above recovery reserve required")
    financial = billing_start + dt.timedelta(seconds=(balance_usd-reserve_usd)/hourly_usd*3600)
    return min(gpu_cutoff, financial)


def source_snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.py"))
            if not {"runs", "generated", "__pycache__"} & set(p.relative_to(root).parts)}


def assert_source_frozen(root: Path, frozen: dict) -> None:
    if source_snapshot(root) != frozen:
        raise ValueError("Source changed after controller freeze")


def exposure_matches(metadata: dict) -> bool:
    return (metadata.get("epoch_exposure_complete") is True
            and metadata.get("exposure", {}).get("records") == metadata.get("planned_records")
            and metadata.get("consumed_order_sha256") == metadata.get("record_order_sha256")
            and metadata.get("exposure", {}).get("updates", 0) > 0)


def assert_same_exposure(arms: dict) -> None:
    compared = [(m["consumed_order_sha256"], m["exposure"]) for m in arms.values()]
    if any(value != compared[0] for value in compared[1:]):
        raise ValueError("Arms differ in actual IDs/order/token/image/update exposure")


def pilot_estimate(smokes: dict, pilot_records: int, evaluation_seconds: float,
                   margin: float = 2.0, reused_training=(), reused_evaluations=(),
                   evaluation_by_cohort=None) -> float:
    if not math.isfinite(margin) or margin < 1 or pilot_records < 1 or set(smokes) != set(ARMS):
        raise ValueError("Complete four-arm measured smoke required")
    if set(reused_training)-set(ARMS) or any(arm not in ARMS or name not in {"mmmu", "external"}
                                           for arm, name in reused_evaluations):
        raise ValueError("Unknown reused pilot stage")
    total = 0.0
    for arm in ARMS:
        metadata = smokes[arm]
        if not exposure_matches(metadata) or metadata["exposure"]["records"] <= 0:
            raise ValueError("Incomplete smoke cannot predict a complete pilot")
        if arm not in reused_training:
            total += metadata["seconds"] / metadata["exposure"]["records"] * pilot_records
    evaluation = 4 * evaluation_seconds
    if reused_evaluations:
        if not evaluation_by_cohort or set(evaluation_by_cohort) != {"mmmu", "external"}:
            raise ValueError("Measured per-cohort cost required to subtract closed evaluations")
        evaluation -= sum(evaluation_by_cohort[name] for _arm, name in reused_evaluations)
    return margin * total + evaluation + 150 * (4-len(set(reused_training)))


def pilot_scale_evidence(rows: list[dict], manifest: dict, original_records: int,
                         explicit_records: int | None, smoke_records: int, seed: int) -> dict:
    """An explicit approved common subset changes training exposure, never inference."""
    actual = original_records if explicit_records is None else explicit_records
    if len(rows) != actual:
        raise ValueError("Common pilot quota mismatch")
    if explicit_records is not None and (type(actual) is not int or not smoke_records <= actual <= original_records):
        raise ValueError("Explicit pilot size must cover smoke and not exceed the original pilot")
    evidence = {"original_records": original_records, "actual_records": actual,
                "explicit_override": explicit_records is not None, "scaled": actual < original_records}
    if actual == original_records:
        return dict(evidence, source_counts=manifest.get("source_counts", {}),
                    nvw_tracks=manifest.get("nvw_tracks", {}), nvw_joint_views=manifest.get("nvw_joint_views", {}))
    if actual < 400 or actual % 40:
        raise ValueError("Reduced common pilot requires N>=400 and N divisible by 40")
    scale = manifest.get("pilot_scale", {})
    if (scale.get("original_records") != original_records or scale.get("selected_records") != actual
            or scale.get("selection_seed") != seed
            or any(not isinstance(scale.get(key), str) or not scale[key].strip()
                   for key in ("selection_policy", "reason"))
            or not re.fullmatch(r"[0-9a-f]{64}", str(scale.get("parent_manifest_sha256", "")))):
        raise ValueError("Explicit subset requires declared immutable scale provenance")
    source_names = {"trace": "trace", "nvw-independent-procedural-v1": "nvw"}
    if any(row.get("source_id") not in source_names for row in rows):
        raise ValueError("Reduced pilot contains an undeclared source")
    sources = dict(Counter(source_names[row["source_id"]] for row in rows))
    tracks = dict(Counter(row.get("nvw_track") for row in rows if source_names[row["source_id"]] == "nvw"))
    view_counts = dict(Counter(row.get("view") for row in rows if source_names[row["source_id"]] == "nvw"))
    pairs = Counter((row.get("nvw_track"), row.get("view")) for row in rows
                    if source_names[row["source_id"]] == "nvw")
    views = {track: {view: pairs[track, view] for view in ("qa", "structure", "grounded")}
             for track in ("NVW-K", "NVW-I", "NVW-S")}
    if (sources != {"trace": actual*9//10, "nvw": actual//10}
            or tracks != {"NVW-K": actual//20, "NVW-I": actual//40, "NVW-S": actual//40}
            or scale.get("source_counts") != sources or scale.get("nvw_track_counts") != tracks
            or manifest.get("nvw_tracks") != tracks or manifest.get("nvw_joint_views") != views
            or view_counts != {"qa": actual//20, "structure": actual//40, "grounded": actual//40}):
        raise ValueError("Reduced pilot actual source/track/view marginals differ from declared quotas")
    return dict(evidence, pilot_scale=scale, source_counts=sources, nvw_tracks=tracks,
                nvw_joint_views=views, nvw_view_counts=view_counts, original_10K_execution_completed=False)


def closed_controller(parent: Path, evaluation_only: bool = False) -> tuple[dict, list, dict]:
    status_path, journal_path = parent/"status.json", parent/"journal.jsonl"
    status, journal = json.loads(status_path.read_text()), read_rows(journal_path)
    terminal = {"BLOCKED", "BUDGET_STOP", "SMOKE_COMPLETE_PILOT_BUDGET_STOP", "FOUR_ARM_PILOT_EVALUATED",
                "ADDITIONAL_FL_NVW0_VS10_PILOT_COMPLETED", "BOUNDED_GRAPH_A3_GPU_SMOKE_COMPLETED",
                "BOUNDED_GRAPH_FIXED_MODULE_A0_A3_EVALUATED"}
    if evaluation_only:
        terminal.add("FL_1600_PAIRED_EVALUATIONS_COMPLETED")
    if (status.get("status") not in terminal or not journal or journal[-1].get("stage") != "controller"
            or journal[-1].get("status") != status["status"]):
        raise ValueError("Reuse requires a closed parent controller and terminal journal")
    frozen = {str(path.resolve()): sha(path) for path in (status_path, journal_path, parent/"source_freeze.json")}
    return status, journal, frozen


def reuse_training(parent: Path, rows: list[dict], records: Path, manifest: Path,
                   cfg: dict, seed: int, phase: str) -> tuple[dict, dict, dict]:
    """Verify closed train + fresh reload; partial/unclosed arms never supply weights."""
    from .modules import ARMS as arm_groups, BASE_ID, BASE_REVISION
    from .train import order_hash

    status, journal, frozen = closed_controller(parent)
    current_sources = {name: sha(Path(__file__).with_name(name)) for name in ("train.py", "modules.py")}
    source_freeze = json.loads((parent/"source_freeze.json").read_text())["sources"]
    if any(source_freeze.get(name) != digest for name, digest in current_sources.items()):
        raise ValueError("Reused smoke training source differs from current source")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    expected_ids = [rows[i]["id"] for i in order]
    budgets = {key: cfg[key] for key in ("min_pixels", "max_pixels", "episode_pixels")}
    budgets.update(sequence_budget=cfg["max_sequence_tokens"], accumulation=cfg["gradient_accumulation"])
    smokes, evidence = {}, {}
    for arm in ARMS:
        arm_root, checkpoint = parent/f"{phase}_{arm}", parent/f"{phase}_{arm}/checkpoint"
        if phase == "pilot" and not arm_root.exists() and not any(
                event.get("stage", "").startswith(f"pilot_{arm}_") for event in journal):
            continue
        metadata_path = checkpoint/"metadata.json"
        metadata = json.loads(metadata_path.read_text())
        for step in ("train", "reload"):
            events = [event for event in journal if event.get("stage") == f"{phase}_{arm}_{step}"]
            if not events or events[-1].get("status") != "EXITED" or events[-1].get("returncode") != 0:
                raise ValueError(f"Reused smoke {arm}/{step} did not close successfully")
        if (not exposure_matches(metadata) or metadata.get("termination") != "EPOCH_COMPLETE"
                or metadata.get("arm") != arm or metadata.get("base_id") != BASE_ID
                or metadata.get("base_revision") != BASE_REVISION or metadata.get("seed") != seed
                or metadata.get("records_sha256") != sha(records) or metadata.get("manifest_sha256") != sha(manifest)
                or metadata.get("source_sha256") != current_sources
                or not math.isfinite(metadata.get("seconds", 0)) or metadata.get("seconds", 0) <= 0
                or metadata.get("planned_records") != len(rows)
                or metadata["exposure"].get("images") != sum(len(row["image_paths"]) for row in rows)
                or metadata["exposure"]["updates"] != math.ceil(len(rows)/cfg["gradient_accumulation"])
                or metadata.get("consumed_order_sha256") != order_hash(expected_ids)
                or any(metadata.get("budgets", {}).get(key) != value for key, value in budgets.items())):
            raise ValueError(f"Reused smoke {arm} identity/source/exposure/budgets mismatch")
        consumed_path, updates_path = arm_root/"consumed_ids.jsonl", arm_root/"updates.jsonl"
        consumed, updates = read_rows(consumed_path), read_rows(updates_path)
        if (sha(consumed_path) != metadata.get("consumed_ids_sha256")
                or [row["id"] for row in consumed] != expected_ids
                or len(updates) != metadata["exposure"]["updates"] or not updates):
            raise ValueError(f"Reused smoke {arm} consumed/update ledger mismatch")
        for update in updates:
            groups = set(arm_groups[arm])
            if (any(set(update.get(key, {})) != groups
                    or any(not math.isfinite(value) or value <= 0 for value in update[key].values())
                    for key in ("gradient_l2", "sampled_update_l1"))
                    or update.get("frozen_sampled_checks", 0) <= 0
                    or not 0 < update.get("peak_allocated_bytes", 0) <= update.get("peak_reserved_bytes", 0)):
                raise ValueError(f"Reused smoke {arm} lacks real gradient/update/memory evidence")
        if phase == "pilot":
            if (metadata.get("epochs") != 1 or metadata.get("base_precision") != "BF16"
                    or metadata.get("trainable_master_precision") != "FP32" or metadata.get("budgets", {}).get("max_steps") != 0
                    or [row.get("position") for row in consumed] != list(range(len(rows)))
                    or any(type(row.get(key)) is not int or row[key] <= 0 for row in consumed
                           for key in ("input_tokens", "target_tokens"))
                    or any(sum(row[key] for row in consumed) != metadata["exposure"].get(key)
                           for key in ("input_tokens", "target_tokens"))):
                raise ValueError(f"Reused pilot {arm} full consumed/token/precision evidence mismatch")
            for update_index, update in enumerate(updates, 1):
                prefix = expected_ids[:min(update_index*cfg["gradient_accumulation"], len(rows))]
                if (update.get("step") != update_index or update.get("consumed_records") != len(prefix)
                        or update.get("consumed_order_sha256") != order_hash(prefix)
                        or update.get("loss_tokens") != sum(row["target_tokens"] for row in
                            consumed[(update_index-1)*cfg["gradient_accumulation"]:update_index*cfg["gradient_accumulation"]])):
                    raise ValueError(f"Reused pilot {arm} update exposure evidence mismatch")
            starts = [event for event in journal if event.get("stage") == f"pilot_{arm}_reload"
                      and event.get("status") == "STARTING"]
            command = starts[-1].get("command", []) if starts else []
            if ("--verify-checkpoint" not in command
                    or command.index("--verify-checkpoint")+1 >= len(command)
                    or Path(command[command.index("--verify-checkpoint")+1]).resolve() != checkpoint.resolve()):
                raise ValueError(f"Reused pilot {arm} lacks fresh-process checkpoint reload binding")
        files = metadata.get("files", {})
        trainable = metadata.get("inventory", {}).get("trainable", [])
        if (not files or not metadata.get("state_sha256")
                or metadata.get("inventory", {}).get("arm") != arm
                or not trainable or set(metadata["state_sha256"]) != {item["name"] for item in trainable}
                or {item["group"] for item in trainable} != set(arm_groups[arm])
                or ("M" in arm_groups[arm] and "mergers.pt" not in files)
                or (any(group != "M" for group in arm_groups[arm])
                    and not {"adapter/adapter_config.json", "adapter/adapter_model.safetensors"} <= set(files))):
            raise ValueError(f"Reused smoke {arm} checkpoint payload identity incomplete")
        for name, digest in files.items():
            path = (checkpoint/name).resolve()
            if not path.is_relative_to(checkpoint.resolve()) or sha(path) != digest:
                raise ValueError(f"Reused smoke {arm} checkpoint payload changed")
            frozen[str(path)] = digest
        for path in (metadata_path, consumed_path, updates_path):
            frozen[str(path.resolve())] = sha(path)
        smokes[arm] = metadata
        evidence[arm] = {"metadata_sha256": sha(metadata_path), "records": len(rows),
                         "updates": len(updates), "fresh_reload_verified": True,
                         "training_root": str(arm_root.resolve()), "checkpoint": str(checkpoint.resolve())}
    if phase == "pilot" and not smokes:
        raise ValueError("No completed pilot checkpoint is available for reuse")
    assert_same_exposure(smokes)
    receipt = {"status": "PASS", "kind": "CLOSED_FOUR_ARM_SMOKE_REUSE" if phase == "smoke" else "CLOSED_PILOT_TRAIN_REUSE",
               "parent": str(parent.resolve()),
               "parent_status": status["status"], "files": frozen, "arms": evidence,
               "training_source_sha256": current_sources, "pilot_starts_from": "F0"}
    return smokes, frozen, receipt


def reuse_smoke(parent: Path, rows: list[dict], records: Path, manifest: Path,
                cfg: dict, seed: int) -> tuple[dict, dict, dict]:
    """Reuse G4 evidence, never initialize a pilot with smoke-trained weights."""
    if len(rows) != 360:
        raise ValueError("Smoke reuse requires 360 rows")
    return reuse_training(parent, rows, records, manifest, cfg, seed, "smoke")


def reuse_pilot(parent: Path, rows: list[dict], records: Path, manifest: Path,
                cfg: dict, seed: int, protocol: Path, image_root: Path) -> tuple[dict, dict, dict]:
    freeze_path = parent/"pilot_input_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get("protocol_sha256") != sha(protocol) or freeze.get("seed") != seed
            or freeze.get("pilot_records_sha256") != sha(records)
            or freeze.get("pilot_manifest_sha256") != sha(manifest)
            or Path(freeze.get("image_root", "")).resolve() != image_root.resolve()):
        raise ValueError("Reused pilot protocol/data/seed/image-root binding mismatch")
    pilots, frozen, receipt = reuse_training(parent, rows, records, manifest, cfg, seed, "pilot")
    frozen[str(freeze_path.resolve())] = sha(freeze_path)
    receipt.update(files=frozen, protocol_sha256=sha(protocol), training_repeated=False,
                   missing_arms_start_from="F0")
    return pilots, frozen, receipt


class Controller:
    def __init__(self, output: Path, deadline: dt.datetime, source_root: Path):
        output.mkdir(parents=True, exist_ok=False)
        self.output, self.deadline, self.source_root = output, deadline, source_root
        self.sources = source_snapshot(source_root)
        dump(output / "source_freeze.json", {"sources": self.sources, "deadline_utc": deadline.isoformat()})
        self.stage_seconds: dict[str, float] = {}

    def remaining(self) -> float:
        return (self.deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()

    def event(self, **value):
        value["utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        with (self.output / "journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(value), flush=True)

    def stage(self, name: str, command: list[str], maximum_seconds: float | None = None) -> None:
        assert_source_frozen(self.source_root, self.sources)
        remaining = self.remaining()
        if remaining <= 120:
            raise TimeoutError("GPU/financial deadline reached; no new stage")
        allowed = min(remaining-20, maximum_seconds or remaining-20)
        self.event(stage=name, status="STARTING", command=command, timeout_seconds=allowed)
        started = time.monotonic()
        with (self.output / (name + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, stdin=subprocess.DEVNULL)
            self.event(stage=name, status="RUNNING", pid=process.pid)
            try:
                stage_deadline = started + allowed
                while True:
                    try:
                        result = process.wait(timeout=min(600, max(0, stage_deadline-time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= stage_deadline:
                            raise
                        try:
                            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu",
                                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
                            probe = {"gpu":gpu.stdout.strip(), "gpu_probe_returncode":gpu.returncode}
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            probe = {"gpu_probe_error":f"{type(exc).__name__}: {exc}"}
                        self.event(stage=name, status="HEALTH_POLL", pid=process.pid,
                                   process_live=process.poll() is None,
                                   **probe,
                                   elapsed_seconds=time.monotonic()-started)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
                self.event(stage=name, status="HARD_DEADLINE_STOP", seconds=time.monotonic()-started)
                raise TimeoutError("Owned subprocess terminated at its explicit deadline") from None
        elapsed = time.monotonic()-started
        self.stage_seconds[name] = elapsed
        self.event(stage=name, status="EXITED", returncode=result, seconds=elapsed)
        if result != 0:
            raise RuntimeError(f"Stage {name} failed with exit {result}; see {name}.log")


def cohort_gate(cohort: Path, expected_count: int | None = None) -> dict:
    manifest = json.loads((cohort / "manifest.json").read_text())
    if manifest.get("status") != "COHORT_FROZEN" or manifest.get("training_allowed") is not False:
        raise ValueError(f"Missing frozen evaluation gate: {cohort}")
    for name, key in (("inputs.jsonl", "inputs_sha256"), ("gold.jsonl", "gold_sha256")):
        if sha(cohort/name) != manifest.get(key):
            raise ValueError("Evaluation cohort bytes changed")
    if expected_count is not None and manifest.get("count") != expected_count:
        raise ValueError("Wrong paired evaluation cohort count")
    return manifest


def evaluation_gate(output: Path, cohort: Path, protocol: Path, baseline: bool = False) -> dict:
    from .modules import BASE_ID, BASE_REVISION

    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((cohort / "manifest.json").read_text())
    if (summary.get("status") != "EVALUATED" or summary.get("cohort_sha256") != sha(cohort / "manifest.json")
            or summary.get("protocol_sha256") != sha(protocol)
            or summary.get("denominator") != manifest["count"] or summary.get("completed") != manifest["count"]
            or summary.get("predictions_sha256") != sha(output / "predictions.jsonl")
            or summary.get("model_id") != BASE_ID or summary.get("model_revision") != BASE_REVISION
            or baseline and summary.get("checkpoint") != "F0-frozen"):
        raise ValueError("Evaluation is incomplete or does not match immutable F0/cohort/protocol")
    expected = [json.loads(line)["id"] for line in (cohort / "inputs.jsonl").read_text().splitlines()]
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    if ([row["id"] for row in predictions] != expected or len(set(expected)) != len(expected)
            or any(row.get("status") != "COMPLETED" for row in predictions)):
        raise ValueError("Evaluation ID order/coverage/status mismatch")
    return summary


def paired_input_gate(output: Path, baseline: Path, cohort: Path, protocol: Path) -> dict:
    evaluation_gate(baseline, cohort, protocol, baseline=True)
    evaluation_gate(output, cohort, protocol)
    references = read_rows(baseline / "predictions.jsonl")
    predictions = read_rows(output / "predictions.jsonl")
    observed = []
    for reference, prediction in zip(references, predictions, strict=True):
        for field in PAIRED_INPUT_FIELDS:
            if any(field not in row or row[field] is None for row in (reference, prediction)):
                raise ValueError(f"Missing paired observed input: {prediction['id']}/{field}")
            if (json.dumps(reference[field], sort_keys=True, allow_nan=False)
                    != json.dumps(prediction[field], sort_keys=True, allow_nan=False)):
                raise ValueError(f"Paired observed input differs: {prediction['id']}/{field}")
        observed.append({"id": prediction["id"], **{key: prediction[key] for key in PAIRED_INPUT_FIELDS}})
    return {"status": "PASS", "kind": "PAIRED_ACTUAL_INPUT_EQUIVALENCE", "count": len(observed),
            "fields": list(PAIRED_INPUT_FIELDS), "baseline_result": str(baseline.resolve()),
            "baseline_summary_sha256": sha(baseline / "summary.json"),
            "baseline_predictions_sha256": sha(baseline / "predictions.jsonl"),
            "pilot_summary_sha256": sha(output / "summary.json"),
            "pilot_predictions_sha256": sha(output / "predictions.jsonl"),
            "cohort_sha256": sha(cohort / "manifest.json"), "protocol_sha256": sha(protocol),
            "observed_inputs_sha256": hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest(),
            "training_allowed": False, "public_upload_allowed": False}


def reuse_pilot_evaluations(parent: Path, checkpoints: dict, baselines: dict,
                           cohorts: dict, protocol: Path) -> tuple[dict, dict, dict]:
    """Reference closed exact-checkpoint results; never modify or copy old outputs."""
    status, journal, frozen = closed_controller(parent, evaluation_only=True)
    source_freeze = json.loads((parent/"source_freeze.json").read_text())["sources"]
    results: dict[tuple[str, str], Path] = {}
    evidence: dict[str, dict] = {}
    for arm in ARMS:
        for name, cohort in cohorts.items():
            stage = f"pilot_{arm}_{name}"
            output = parent/stage
            events = [event for event in journal if event.get("stage") == stage]
            if not output.exists() and not events:
                continue
            if arm not in checkpoints:
                raise ValueError("Pilot evaluation reuse requires a validated reused checkpoint")
            if not events or events[-1].get("status") != "EXITED" or events[-1].get("returncode") != 0:
                raise ValueError(f"Reused pilot evaluation {stage} did not close successfully")
            cohort_gate(cohort, 60)
            summary = evaluation_gate(output, cohort, protocol)
            checkpoint = checkpoints[arm]
            if (Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve()
                    or summary.get("checkpoint_metadata_sha256") != sha(checkpoint/"metadata.json")
                    or summary.get("benchmark_derived") is not (name == "mmmu")
                    or summary.get("evaluator_source_sha256") != source_freeze.get("evaluate.py")
                    or source_freeze.get("evaluate.py") != sha(Path(__file__).with_name("evaluate.py"))
                    or not math.isfinite(summary.get("seconds", 0)) or summary.get("seconds", 0) <= 0):
                raise ValueError(f"Reused pilot evaluation {stage} checkpoint/source mismatch")
            paired = paired_input_gate(output, baselines[name], cohort, protocol)
            paired.update(arm=arm, cohort=name)
            for filename in ("summary.json", "predictions.jsonl"):
                frozen[str((output/filename).resolve())] = sha(output/filename)
            frozen[str((checkpoint/"metadata.json").resolve())] = sha(checkpoint/"metadata.json")
            results[arm, name] = output
            evidence.setdefault(arm, {})[name] = {"output": str(output.resolve()),
                "checkpoint": str(checkpoint.resolve()), "checkpoint_metadata_sha256": sha(checkpoint/"metadata.json"),
                "summary_sha256": sha(output/"summary.json"), "predictions_sha256": sha(output/"predictions.jsonl"),
                "paired_input_equivalence": paired, "evaluation_repeated": False}
    if not results:
        raise ValueError("No completed pilot evaluation is available for reuse")
    return results, frozen, {"status": "PASS", "kind": "CLOSED_PILOT_EVALUATION_REUSE",
        "parent": str(parent.resolve()), "parent_status": status["status"], "files": frozen, "arms": evidence}


def run(args, controller: Controller) -> dict:
    from .train import approved_records

    protocol = json.loads(args.protocol.read_text())
    cfg = protocol["training"]
    pilot_image_root = args.pilot_image_root or args.image_root
    smoke_rows, smoke_manifest = approved_records(args.smoke_records, args.smoke_manifest, args.image_root)
    if len({r["root_id"] for r in smoke_rows}) != 120:
        raise ValueError("Smoke must contain exactly 120 independently verified roots")
    override = getattr(args, "pilot_record_count", None)
    if override is not None and (type(override) is not int or not max(400, len(smoke_rows)) <= override <= cfg["pilot_records"]
                                 or override < cfg["pilot_records"] and override % 40):
        raise ValueError("Explicit pilot count requires smoke<=N<=original; subset N>=400 and divisible by40")
    cohort_gate(args.mmmu_cohort, 60)
    cohort_gate(args.external_cohort)
    help_result = subprocess.run([sys.executable, "-m", "fine_tuning.evaluate", "--help"],
                                 capture_output=True, text=True, timeout=30)
    if help_result.returncode or "run-external" not in help_result.stdout:
        raise ValueError("Missing evaluate run-external command; no external success inferred")
    immutable = {str(p.resolve()): sha(p) for p in (args.smoke_records, args.smoke_manifest,
        args.protocol, args.mmmu_cohort / "manifest.json",
        args.external_cohort / "manifest.json")}
    dump(controller.output / "input_freeze.json", {"files": immutable,
        "smoke_records_sha256": smoke_manifest["records_sha256"],
        "pilot_gate": "DEFERRED_UNTIL_SMOKE_RELOAD_VERIFIED", "seed": protocol["master_seed"],
        "smoke_image_root": str(args.image_root.resolve()), "pilot_image_root": str(pilot_image_root.resolve())})

    def frozen():
        if any(sha(Path(path)) != digest for path, digest in immutable.items()):
            raise ValueError("Immutable data/protocol changed")

    def train(phase: str, arm: str, records: Path, manifest: Path) -> dict:
        frozen()
        output = controller.output / f"{phase}_{arm}"
        image_root = pilot_image_root if phase == "pilot" else args.image_root
        command = [sys.executable, "-m", "fine_tuning.train", "--model-path", str(args.model_path),
                   "--records", str(records), "--manifest", str(manifest), "--image-root", str(image_root),
                   "--output", str(output), "--arm", arm, "--seed", str(protocol["master_seed"]),
                   "--budget-seconds", str(max(1, controller.remaining()-120))]
        for flag, value in (("min-pixels", cfg["min_pixels"]), ("max-pixels", cfg["max_pixels"]),
                            ("episode-pixels", cfg["episode_pixels"]), ("sequence-budget", cfg["max_sequence_tokens"]),
                            ("accumulation", cfg["gradient_accumulation"])):
            command.extend(["--"+flag, str(value)])
        controller.stage(f"{phase}_{arm}_train", command)
        metadata = json.loads((output / "checkpoint" / "metadata.json").read_text())
        if not exposure_matches(metadata):
            raise TimeoutError(f"{phase}/{arm} did not complete its frozen exposure; checkpoint retained")
        controller.stage(f"{phase}_{arm}_reload", [sys.executable, "-m", "fine_tuning.train",
            "--model-path", str(args.model_path), "--verify-checkpoint", str(output / "checkpoint")], 600)
        return metadata

    baselines: dict[str, Path] = {}
    reused_evaluations: dict[tuple[str, str], Path] = {}

    def evaluate(phase: str, arm: str, checkpoint: Path | None) -> None:
        frozen()
        for name, subcommand, cohort in (("mmmu", "run", args.mmmu_cohort),
                                        ("external", "run-external", args.external_cohort)):
            reused = args.baseline_mmmu if name == "mmmu" else args.baseline_external
            if arm == "F0" and reused:
                summary = evaluation_gate(reused, cohort, args.protocol, baseline=True)
                controller.stage_seconds[f"{phase}_{arm}_{name}"] = summary["seconds"]
                controller.event(stage=f"{phase}_{arm}_{name}", status="CLOSED_BASELINE_REUSED",
                                 result=str(reused), summary_sha256=sha(reused / "summary.json"))
                baselines[name] = reused
                for filename in ("summary.json", "predictions.jsonl"):
                    immutable[str((reused / filename).resolve())] = sha(reused / filename)
                continue
            if phase == "pilot" and (arm, name) in reused_evaluations:
                output = reused_evaluations[arm, name]
                receipt = paired_input_gate(output, baselines[name], cohort, args.protocol)
                receipt.update(arm=arm, cohort=name)
                receipt_path = controller.output/f"{phase}_{arm}_{name}_input_equivalence.json"
                dump(receipt_path, receipt)
                controller.event(stage=f"{phase}_{arm}_{name}", status="CLOSED_PILOT_EVALUATION_REUSED",
                    result=str(output.resolve()), checkpoint=str(checkpoint),
                    summary_sha256=sha(output/"summary.json"), receipt_sha256=sha(receipt_path), no_gpu_redo=True)
                continue
            output = controller.output / f"{phase}_{arm}_{name}"
            command = [sys.executable, "-m", "fine_tuning.evaluate", subcommand, "--model-path", str(args.model_path),
                "--cohort", str(cohort), "--output", str(output), "--protocol", str(args.protocol),
                "--budget-seconds", str(max(1, int(controller.remaining()-120)))]
            if checkpoint:
                command.extend(["--checkpoint", str(checkpoint)])
            controller.stage(f"{phase}_{arm}_{name}", command)
            evaluation_gate(output, cohort, args.protocol)
            frozen()
            if arm == "F0":
                baselines[name] = output
                for filename in ("summary.json", "predictions.jsonl"):
                    immutable[str((output / filename).resolve())] = sha(output / filename)
            else:
                receipt = paired_input_gate(output, baselines[name], cohort, args.protocol)
                receipt.update(arm=arm, cohort=name)
                dump(output / "input_equivalence.json", receipt)
                controller.event(stage=f"{phase}_{arm}_{name}_input_gate", status="PASS",
                                 receipt_sha256=sha(output / "input_equivalence.json"))

    evaluate("baseline", "F0", None)
    smokes: dict[str, dict] = {}
    reused_smoke = getattr(args, "reuse_smoke_controller", None)
    if reused_smoke:
        smokes, reused_files, receipt = reuse_smoke(reused_smoke, smoke_rows, args.smoke_records,
                                                   args.smoke_manifest, cfg, protocol["master_seed"])
        immutable.update(reused_files)
        dump(controller.output/"smoke_reuse.json", receipt)
        controller.event(stage="smoke_reuse", status="PASS", parent=str(reused_smoke),
                         receipt_sha256=sha(controller.output/"smoke_reuse.json"), no_gpu_redo=True)
    else:
        for arm in ARMS:
            smokes[arm] = train("smoke", arm, args.smoke_records, args.smoke_manifest)
    assert_same_exposure(smokes)
    # G4 smoke is the update/memory/save/reload gate, not another scored cohort run.
    eval_seconds = sum(controller.stage_seconds[f"baseline_F0_{name}"] for name in ("mmmu", "external"))
    # Smoke has its own approval: an unfinished common pilot must not prevent it.
    try:
        pilot_rows, pilot_manifest = approved_records(args.pilot_records, args.pilot_manifest, pilot_image_root)
        scale = pilot_scale_evidence(pilot_rows, pilot_manifest, cfg["pilot_records"],
            getattr(args, "pilot_record_count", None), len(smoke_rows), protocol["master_seed"])
        if {r["root_id"] for r in smoke_rows} & {r["root_id"] for r in pilot_rows}:
            raise ValueError("Smoke and pilot roots overlap")
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "BLOCKED", "smoke_completed": True, "pilot_started": False,
                "reason": f"Common pilot approval missing or invalid: {exc}"}
    for path in (args.pilot_records, args.pilot_manifest):
        immutable[str(path.resolve())] = sha(path)
    dump(controller.output / "pilot_input_freeze.json", {"files": immutable,
        "pilot_records_sha256": pilot_manifest["records_sha256"], "seed": protocol["master_seed"],
        "image_root": str(pilot_image_root.resolve()), "pilot_manifest": str(args.pilot_manifest.resolve()),
        "pilot_manifest_sha256": sha(args.pilot_manifest), "scale": scale,
        "protocol_sha256": sha(args.protocol)})
    pilots: dict[str, dict] = {}
    checkpoints: dict[str, Path] = {}
    pilot_parent = getattr(args, "reuse_pilot_controller", None)
    if pilot_parent:
        pilots, reused_files, receipt = reuse_pilot(pilot_parent, pilot_rows, args.pilot_records,
            args.pilot_manifest, cfg, protocol["master_seed"], args.protocol, pilot_image_root)
        immutable.update(reused_files)
        checkpoints = {arm: Path(receipt["arms"][arm]["checkpoint"]) for arm in pilots}
        dump(controller.output/"pilot_reuse.json", receipt)
        controller.event(stage="pilot_reuse", status="PASS", parent=str(pilot_parent),
            receipt_sha256=sha(controller.output/"pilot_reuse.json"), no_gpu_redo=True)
    evaluation_parent = getattr(args, "reuse_pilot_evaluation_controller", None)
    if evaluation_parent:
        if not checkpoints:
            raise ValueError("Pilot evaluation reuse requires validated reused training checkpoints")
        reused_evaluations, reused_files, receipt = reuse_pilot_evaluations(evaluation_parent, checkpoints,
            baselines, {"mmmu": args.mmmu_cohort, "external": args.external_cohort}, args.protocol)
        immutable.update(reused_files)
        dump(controller.output/"pilot_evaluation_reuse.json", receipt)
        controller.event(stage="pilot_evaluation_reuse", status="PASS", parent=str(evaluation_parent),
            receipt_sha256=sha(controller.output/"pilot_evaluation_reuse.json"), no_gpu_redo=True)
    measured_cohorts = {name: controller.stage_seconds[f"baseline_F0_{name}"] for name in ("mmmu", "external")}
    forecast_margin = getattr(args, "training_forecast_margin", 2.0)
    estimate = pilot_estimate(smokes, len(pilot_rows), eval_seconds, margin=forecast_margin, reused_training=pilots,
        reused_evaluations=reused_evaluations, evaluation_by_cohort=measured_cohorts)
    controller.event(stage="pilot_budget_gate", status="ESTIMATED", required_seconds=estimate,
        remaining_seconds=controller.remaining(), reused_training=list(pilots),
        reused_evaluations=[list(stage) for stage in reused_evaluations],
        training_forecast_margin=forecast_margin,
        estimate_policy=f"{forecast_margin:g}x smoke per-record training + measured cohort evaluations +150s per reload; subtract verified closed stages only")
    if estimate >= controller.remaining()-120:
        return {"status": "SMOKE_COMPLETE_PILOT_BUDGET_STOP", "pilot_started": False,
                "required_seconds_estimate": estimate, "remaining_seconds": controller.remaining()}
    for arm in ARMS:
        if arm not in pilots:
            pilots[arm] = train("pilot", arm, args.pilot_records, args.pilot_manifest)
            checkpoints[arm] = controller.output / f"pilot_{arm}" / "checkpoint"
        else:
            controller.event(stage=f"pilot_{arm}_train_reload", status="CLOSED_PILOT_TRAIN_REUSED",
                checkpoint=str(checkpoints[arm]), metadata_sha256=sha(checkpoints[arm]/"metadata.json"), no_gpu_redo=True)
        evaluate("pilot", arm, checkpoints[arm])
    assert_same_exposure(pilots)
    return {"status": "FOUR_ARM_PILOT_EVALUATED", "pilot_started": True,
            "pilot_records": len(pilot_rows), "original_pilot_records": cfg["pilot_records"],
            "original_10K_execution_completed": len(pilot_rows) == 10000,
            "historical_v8_changed": False, "holdout_semantic_certification": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "smoke-records", "smoke-manifest", "pilot-records", "pilot-manifest",
                 "image-root", "mmmu-cohort", "external-cohort", "protocol", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--owned-pod-id", required=True)
    parser.add_argument("--pilot-image-root", type=Path, help="Pilot image root; defaults to --image-root")
    parser.add_argument("--pilot-record-count", type=int, help="Explicit approved common subset; default original protocol count")
    parser.add_argument("--reuse-smoke-controller", type=Path, help="Closed four-arm G4 controller; reuse evidence, not trained initialization")
    parser.add_argument("--reuse-pilot-controller", type=Path, help="Closed controller; reuse only verified completed pilot train+fresh reload")
    parser.add_argument("--reuse-pilot-evaluation-controller", type=Path, help="Closed exact-checkpoint paired evaluation controller; requires reused training")
    parser.add_argument("--baseline-mmmu", type=Path, help="Reuse already closed exact F0-frozen result")
    parser.add_argument("--baseline-external", type=Path)
    parser.add_argument("--gpu-cutoff-utc", required=True)
    parser.add_argument("--billing-start-utc", required=True)
    parser.add_argument("--balance-usd", type=float, required=True)
    parser.add_argument("--reserve-usd", type=float, required=True)
    parser.add_argument("--hourly-usd", type=float, required=True, help="All billed hourly charges, including storage")
    parser.add_argument("--training-forecast-margin", type=float, default=2.0,
                        help="Explicit operating forecast multiplier >=1; does not change training or hard deadline")
    args = parser.parse_args()
    if not math.isfinite(args.training_forecast_margin) or args.training_forecast_margin < 1:
        parser.error("Training forecast margin must be finite and >=1")
    if os.environ.get("RUNPOD_POD_ID") != args.owned_pod_id:
        parser.error("Exact owned RunPod identity required; no local GPU execution")
    deadline = effective_deadline(utc(args.gpu_cutoff_utc), utc(args.billing_start_utc),
                                  args.balance_usd, args.reserve_usd, args.hourly_usd)
    controller = Controller(args.output, deadline, Path(__file__).parent)
    try:
        result = run(args, controller)
    except TimeoutError as exc:
        result = {"status": "BUDGET_STOP", "reason": str(exc)}
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        result = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
    result.update(deadline_utc=deadline.isoformat(), stages=controller.stage_seconds,
                  sources_sha256=hashlib.sha256(json.dumps(controller.sources, sort_keys=True).encode()).hexdigest())
    dump(args.output / "status.json", result)
    controller.event(stage="controller", **result)
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
