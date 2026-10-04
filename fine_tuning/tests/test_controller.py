"""Budget, immutable-source, exposure and owned-subprocess checks; CPU only."""
import datetime as dt
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fine_tuning.audit import sha
from fine_tuning.modules import BASE_ID, BASE_REVISION
from fine_tuning.run_experiment import (ARMS, Controller, assert_same_exposure, assert_source_frozen,
    effective_deadline, evaluation_gate, exposure_matches, paired_input_gate, pilot_estimate,
    pilot_scale_evidence, reuse_pilot, reuse_pilot_evaluations, reuse_smoke, run, source_snapshot, utc)
from fine_tuning.train import order_hash, stop_for_budget


class ControllerTests(unittest.TestCase):
    def pilot_reuse_fixture(self, root):
        import fine_tuning.train as training

        parent = root/"parent"
        checkpoint = parent/"pilot_FL/checkpoint"
        (checkpoint/"adapter").mkdir(parents=True)
        rows = [{"id": str(i), "root_id": str(i), "image_paths": ["image.png"]} for i in range(1600)]
        records, manifest, protocol = root/"records.jsonl", root/"manifest.json", root/"protocol.json"
        records.write_text("".join(json.dumps(row)+"\n" for row in rows))
        manifest.write_text('{}')
        cfg = {"min_pixels": 256, "max_pixels": 512, "episode_pixels": 1024,
               "max_sequence_tokens": 4096, "gradient_accumulation": 16}
        protocol.write_text(json.dumps({"training": cfg, "master_seed": 3407}))
        sources = {name: sha(Path(training.__file__).with_name(name)) for name in ("train.py", "modules.py", "evaluate.py")}
        (parent/"source_freeze.json").write_text(json.dumps({"sources": sources}))
        (parent/"pilot_input_freeze.json").write_text(json.dumps({"protocol_sha256": sha(protocol), "seed": 3407,
            "pilot_records_sha256": sha(records), "pilot_manifest_sha256": sha(manifest), "image_root": str(root)}))
        indices = list(range(len(rows)))
        random.Random(3407).shuffle(indices)
        ids = [rows[i]["id"] for i in indices]
        consumed = [{"id": sid, "position": i, "input_tokens": 2, "target_tokens": 1} for i,sid in enumerate(ids)]
        consumed_path, updates_path = checkpoint.parent/"consumed_ids.jsonl", checkpoint.parent/"updates.jsonl"
        consumed_path.write_text("".join(json.dumps(row)+"\n" for row in consumed))
        updates = [{"step": i, "consumed_records": i*16, "consumed_order_sha256": order_hash(ids[:i*16]),
            "loss_tokens": 16, "gradient_l2": {"L": 1.0}, "sampled_update_l1": {"L": 1.0},
            "frozen_sampled_checks": 9, "peak_allocated_bytes": 1, "peak_reserved_bytes": 2} for i in range(1,101)]
        updates_path.write_text("".join(json.dumps(row)+"\n" for row in updates))
        files = {}
        for name in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
            path = checkpoint/name
            path.write_bytes(b"CPU fixture, not model weights")
            files[name] = sha(path)
        metadata = {"arm": "FL", "base_id": BASE_ID, "base_revision": BASE_REVISION, "seed": 3407,
            "epoch_exposure_complete": True, "termination": "EPOCH_COMPLETE", "epochs": 1,
            "base_precision": "BF16", "trainable_master_precision": "FP32", "planned_records": 1600,
            "exposure": {"records": 1600, "images": 1600, "updates": 100, "input_tokens": 3200, "target_tokens": 1600},
            "record_order_sha256": order_hash(ids), "consumed_order_sha256": order_hash(ids), "seconds": 300,
            "source_sha256": {n: sources[n] for n in ("train.py", "modules.py")}, "records_sha256": sha(records),
            "manifest_sha256": sha(manifest), "consumed_ids_sha256": sha(consumed_path),
            "budgets": {"min_pixels": 256, "max_pixels": 512, "episode_pixels": 1024,
                        "sequence_budget": 4096, "accumulation": 16, "max_steps": 0},
            "inventory": {"arm": "FL", "trainable": [{"name": "L", "group": "L"}]},
            "state_sha256": {"L": "b"*64}, "files": files}
        (checkpoint/"metadata.json").write_text(json.dumps(metadata))
        journal = [{"stage": "pilot_FL_train", "status": "EXITED", "returncode": 0},
            {"stage": "pilot_FL_reload", "status": "STARTING", "command": ["python", "-m", "fine_tuning.train",
                "--verify-checkpoint", str(checkpoint)]},
            {"stage": "pilot_FL_reload", "status": "EXITED", "returncode": 0},
            {"stage": "controller", "status": "BLOCKED"}]
        (parent/"journal.jsonl").write_text("".join(json.dumps(event)+"\n" for event in journal))
        (parent/"status.json").write_text(json.dumps({"status": "BLOCKED", "reason": "Evaluation prompt path missing"}))
        return parent, rows, records, manifest, cfg, protocol, checkpoint

    def test_closed_pilot_reuse_requires_exact_full_training_reload_and_payload(self):
        for bad in (None, "seed", "records", "manifest", "source", "budget", "payload", "reload",
                    "partial", "open", "no_checkpoint", "partial_arm", "tokens", "update", "protocol", "reload_binding"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                parent, rows, records, manifest, cfg, protocol, checkpoint = self.pilot_reuse_fixture(root)
                metadata_path = checkpoint/"metadata.json"
                metadata = json.loads(metadata_path.read_text())
                journal_path = parent/"journal.jsonl"
                journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
                if bad == "seed":
                    metadata["seed"] = 17
                elif bad in {"records", "manifest", "protocol"}:
                    {"records": records, "manifest": manifest, "protocol": protocol}[bad].write_text('changed')
                elif bad == "source":
                    metadata["source_sha256"]["train.py"] = "wrong"
                elif bad == "budget":
                    metadata["budgets"]["accumulation"] = 8
                elif bad == "payload":
                    (checkpoint/"adapter/adapter_model.safetensors").write_bytes(b"changed")
                elif bad == "reload":
                    journal[-2]["returncode"] = 1
                elif bad == "partial":
                    metadata["epoch_exposure_complete"] = False
                elif bad == "open":
                    (parent/"status.json").write_text('{"status":"RUNNING"}')
                elif bad == "partial_arm":
                    journal.insert(-1, {"stage": "pilot_FM_train", "status": "STARTING"})
                elif bad == "tokens":
                    metadata["exposure"]["input_tokens"] += 1
                elif bad == "update":
                    path = checkpoint.parent/"updates.jsonl"
                    values = [json.loads(line) for line in path.read_text().splitlines()]
                    values[-1]["consumed_order_sha256"] = "wrong"
                    path.write_text("".join(json.dumps(row)+"\n" for row in values))
                elif bad == "reload_binding":
                    journal[1]["command"][-1] = str(root/"other-checkpoint")
                metadata_path.write_text(json.dumps(metadata))
                journal_path.write_text("".join(json.dumps(event)+"\n" for event in journal))
                if bad == "no_checkpoint":
                    metadata_path.unlink()
                before = {str(p): sha(p) for p in parent.rglob("*") if p.is_file()}
                if bad:
                    with self.assertRaises((ValueError, OSError)):
                        reuse_pilot(parent, rows, records, manifest, cfg, 3407, protocol, root)
                else:
                    pilots, frozen, receipt = reuse_pilot(parent, rows, records, manifest, cfg, 3407, protocol, root)
                    self.assertEqual(set(pilots), {"FL"})
                    self.assertEqual(receipt["arms"]["FL"]["checkpoint"], str(checkpoint.resolve()))
                    self.assertFalse(receipt["training_repeated"])
                    self.assertEqual(receipt["missing_arms_start_from"], "F0")
                    self.assertIn(str(metadata_path.resolve()), frozen)
                self.assertEqual(before, {str(p): sha(p) for p in parent.rglob("*") if p.is_file()})

    def test_closed_pilot_evaluation_reuse_requires_checkpoint_hash_and_paired_actual_inputs(self):
        for bad in (None, "checkpoint", "metadata", "input", "seed", "source", "stale_source", "partial", "open", "exit", "no_checkpoint"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                parent, _rows, _records, _manifest, _cfg, protocol, checkpoint = self.pilot_reuse_fixture(root)
                output = root/"evaluations"
                output.mkdir()
                sources = json.loads((parent/"source_freeze.json").read_text())["sources"]
                if bad == "stale_source":
                    sources["evaluate.py"] = "old-evaluator-matching-its-own-freeze"
                (output/"source_freeze.json").write_text(json.dumps({"sources": sources}))
                cohorts, baselines, journal = {}, {}, []
                for name in ("mmmu", "external"):
                    cohort, baseline, evaluated = root/name, root/(name+"_F0"), output/f"pilot_FL_{name}"
                    for path in (cohort, baseline, evaluated):
                        path.mkdir()
                    inputs = [{"id": str(i)} for i in range(60)]
                    (cohort/"inputs.jsonl").write_text("".join(json.dumps(row)+"\n" for row in inputs))
                    (cohort/"gold.jsonl").write_text('{}\n')
                    (cohort/"manifest.json").write_text(json.dumps({"status": "COHORT_FROZEN", "training_allowed": False,
                        "count": 60, "inputs_sha256": sha(cohort/"inputs.jsonl"), "gold_sha256": sha(cohort/"gold.jsonl")}))
                    predictions = [dict(row, status="COMPLETED", input_sha256=row["id"], prompt_token_ids=[1,2],
                        image_grid_thw=[[1,1,1]], processed_image_sizes=[[32,32]], seed=3407) for row in inputs]
                    for destination in (baseline, evaluated):
                        values = [dict(row) for row in predictions]
                        if destination == evaluated and name == "mmmu" and bad in {"input", "seed"}:
                            values[0]["prompt_token_ids" if bad == "input" else "seed"] = [99] if bad == "input" else 17
                        (destination/"predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in values))
                        summary = {"status": "EVALUATED", "model_id": BASE_ID, "model_revision": BASE_REVISION,
                            "cohort_sha256": sha(cohort/"manifest.json"), "protocol_sha256": sha(protocol),
                            "denominator": 60, "completed": 60, "predictions_sha256": sha(destination/"predictions.jsonl"),
                            "checkpoint": "F0-frozen" if destination == baseline else str(checkpoint),
                            "checkpoint_metadata_sha256": None if destination == baseline else sha(checkpoint/"metadata.json"),
                            "evaluator_source_sha256": sources["evaluate.py"], "benchmark_derived": name == "mmmu", "seconds": 10}
                        if destination == evaluated and name == "mmmu":
                            if bad == "checkpoint":
                                summary["checkpoint"] = str(root/"wrong-checkpoint")
                            elif bad == "metadata":
                                summary["checkpoint_metadata_sha256"] = "wrong"
                            elif bad == "source":
                                summary["evaluator_source_sha256"] = "wrong"
                            elif bad == "partial":
                                summary.update(status="BUDGET_STOP", completed=59)
                        (destination/"summary.json").write_text(json.dumps(summary))
                    journal.append({"stage": f"pilot_FL_{name}", "status": "EXITED", "returncode": 1 if bad == "exit" else 0})
                    cohorts[name], baselines[name] = cohort, baseline
                status = "FL_1600_PAIRED_EVALUATIONS_COMPLETED"
                journal.append({"stage": "controller", "status": status})
                (output/"status.json").write_text(json.dumps({"status": "RUNNING" if bad == "open" else status}))
                (output/"journal.jsonl").write_text("".join(json.dumps(event)+"\n" for event in journal))
                before = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
                checkpoints = {} if bad == "no_checkpoint" else {"FL": checkpoint}
                if bad:
                    with self.assertRaises(ValueError):
                        reuse_pilot_evaluations(output, checkpoints, baselines, cohorts, protocol)
                else:
                    reused, frozen, receipt = reuse_pilot_evaluations(output, checkpoints, baselines, cohorts, protocol)
                    self.assertEqual(set(reused), {("FL", "mmmu"), ("FL", "external")})
                    self.assertEqual(receipt["arms"]["FL"]["mmmu"]["paired_input_equivalence"]["count"], 60)
                    self.assertIn(str((checkpoint/"metadata.json").resolve()), frozen)
                self.assertEqual(before, {str(p): sha(p) for p in root.rglob("*") if p.is_file()})

    def test_explicit_scale_requires_actual_declared_quota_and_provenance(self):
        rows = [{"source_id": "trace"} for _ in range(360)]
        joint = {"NVW-K": {"qa": 10, "structure": 5, "grounded": 5},
                 "NVW-I": {"qa": 5, "structure": 3, "grounded": 2},
                 "NVW-S": {"qa": 5, "structure": 2, "grounded": 3}}
        rows += [{"source_id": "nvw-independent-procedural-v1", "nvw_track": track, "view": view}
                 for track, values in joint.items() for view,count in values.items() for _ in range(count)]
        tracks = {"NVW-K": 20, "NVW-I": 10, "NVW-S": 10}
        manifest = {"nvw_tracks": tracks, "nvw_joint_views": joint, "pilot_scale": {
            "original_records": 10000, "selected_records": 400, "selection_seed": 3407,
            "selection_policy": "source/track-stratified ID hash, no outcomes", "reason": "approved budget scale",
            "parent_manifest_sha256": "a"*64, "source_counts": {"trace": 360, "nvw": 40}, "nvw_track_counts": tracks}}
        scale = pilot_scale_evidence(rows, manifest, 10000, 400, 360, 3407)
        self.assertTrue(scale["scaled"])
        self.assertEqual(scale["nvw_joint_views"], joint)
        self.assertEqual(scale["nvw_view_counts"], {"qa": 20, "structure": 10, "grounded": 10})
        self.assertFalse(scale["original_10K_execution_completed"])
        with self.assertRaises(ValueError):
            pilot_scale_evidence(rows, manifest, 10000, None, 360, 3407)
        for key, value in (("source_counts", {"trace": 359, "nvw": 41}),
                           ("parent_manifest_sha256", "unverified"), ("selection_seed", 17)):
            changed = dict(manifest, pilot_scale=dict(manifest["pilot_scale"], **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot_scale_evidence(rows, changed, 10000, 400, 360, 3407)
        with self.assertRaises(ValueError):
            pilot_scale_evidence([dict(row, nvw_track="wrong") for row in rows], manifest, 10000, 400, 360, 3407)
        with self.assertRaises(ValueError):
            pilot_scale_evidence(rows, dict(manifest, nvw_joint_views={"structure": 40}), 10000, 400, 360, 3407)
        with self.assertRaises(ValueError):
            pilot_scale_evidence(rows, dict(manifest, nvw_joint_views={"qa": 40}), 10000, 400, 360, 3407)
        all_qa = [dict(row, view="qa") if "nvw_track" in row else row for row in rows]
        all_qa_joint = {track: {"qa": count, "structure": 0, "grounded": 0} for track,count in tracks.items()}
        with self.assertRaisesRegex(ValueError, "marginals"):
            pilot_scale_evidence(all_qa, dict(manifest, nvw_joint_views=all_qa_joint), 10000, 400, 360, 3407)

    def test_closed_smoke_reuse_checks_all_four_arms_and_payloads_without_model_calls(self):
        import fine_tuning.train as training

        for bad in (None, "open", "reload", "source", "payload", "order", "budget"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rows = [{"id": str(i), "root_id": str(i//3), "image_paths": ["image.png"]} for i in range(360)]
                records, manifest = root/"records.jsonl", root/"manifest.json"
                records.write_text("".join(json.dumps(row)+"\n" for row in rows))
                manifest.write_text('{}')
                sources = {name: sha(Path(training.__file__).with_name(name)) for name in ("train.py", "modules.py")}
                (root/"source_freeze.json").write_text(json.dumps({"sources": sources}))
                cfg = {"min_pixels": 256, "max_pixels": 512, "episode_pixels": 1024,
                       "max_sequence_tokens": 4096, "gradient_accumulation": 16}
                indices = list(range(360))
                random.Random(3407).shuffle(indices)
                ids = [rows[i]["id"] for i in indices]
                journal = []
                for arm, groups in (("FL", ("L",)), ("FM", ("M",)), ("FV", ("V",)), ("FML", ("M", "L"))):
                    arm_root, checkpoint = root/f"smoke_{arm}", root/f"smoke_{arm}/checkpoint"
                    checkpoint.mkdir(parents=True)
                    files = {}
                    names = (["mergers.pt"] if "M" in groups else []) + (
                        ["adapter/adapter_config.json", "adapter/adapter_model.safetensors"] if any(g != "M" for g in groups) else [])
                    for name in names:
                        path = checkpoint/name
                        path.parent.mkdir(exist_ok=True)
                        path.write_bytes(b"synthetic CPU payload, not weights")
                        files[name] = sha(path)
                    consumed = arm_root/"consumed_ids.jsonl"
                    consumed.write_text("".join(json.dumps({"id": sid})+"\n" for sid in ids))
                    update = {"gradient_l2": {g: 1.0 for g in groups}, "sampled_update_l1": {g: 1.0 for g in groups},
                              "frozen_sampled_checks": 9, "peak_allocated_bytes": 1, "peak_reserved_bytes": 2}
                    (arm_root/"updates.jsonl").write_text((json.dumps(update)+"\n")*23)
                    metadata = {"arm": arm, "base_id": BASE_ID, "base_revision": BASE_REVISION,
                        "seed": 3407, "epoch_exposure_complete": True, "termination": "EPOCH_COMPLETE",
                        "planned_records": 360, "exposure": {"records": 360, "images": 360, "updates": 23},
                        "record_order_sha256": order_hash(ids), "consumed_order_sha256": order_hash(ids),
                        "source_sha256": sources, "records_sha256": sha(records), "manifest_sha256": sha(manifest),
                        "consumed_ids_sha256": sha(consumed), "seconds": 300,
                        "budgets": {"min_pixels": 256, "max_pixels": 512, "episode_pixels": 1024,
                                    "sequence_budget": 4096, "accumulation": 16},
                        "inventory": {"arm": arm, "trainable": [{"name": g, "group": g} for g in groups]},
                        "files": files, "state_sha256": {g: "b"*64 for g in groups}}
                    if bad == "source" and arm == "FM":
                        metadata["source_sha256"] = dict(sources, **{"train.py": "wrong"})
                    if bad == "order" and arm == "FM":
                        metadata["consumed_order_sha256"] = "wrong"
                    if bad == "budget" and arm == "FM":
                        metadata["budgets"]["accumulation"] = 8
                    (checkpoint/"metadata.json").write_text(json.dumps(metadata))
                    for step in ("train", "reload"):
                        journal.append({"stage": f"smoke_{arm}_{step}", "status": "EXITED",
                                        "returncode": 1 if bad == "reload" and arm == "FM" and step == "reload" else 0})
                journal.append({"stage": "controller", "status": "BLOCKED"})
                (root/"journal.jsonl").write_text("".join(json.dumps(event)+"\n" for event in journal))
                (root/"status.json").write_text(json.dumps({"status": "RUNNING" if bad == "open" else "BLOCKED"}))
                if bad == "payload":
                    (root/"smoke_FM/checkpoint/mergers.pt").write_bytes(b"changed")
                if bad:
                    with self.assertRaises(ValueError):
                        reuse_smoke(root, rows, records, manifest, cfg, 3407)
                else:
                    smokes, frozen, receipt = reuse_smoke(root, rows, records, manifest, cfg, 3407)
                    self.assertEqual(set(smokes), set(ARMS))
                    self.assertEqual(receipt["pilot_starts_from"], "F0")
                    self.assertIn(str((root/"smoke_FM/checkpoint/mergers.pt").resolve()), frozen)

    def test_health_probe_failure_does_not_terminate_a_live_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"source"
            source.mkdir()
            controller = Controller(root/"output", dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1), source)
            process = Mock(pid=123)
            process.wait.side_effect = [subprocess.TimeoutExpired("stage", 600), 0]
            process.poll.return_value = None
            with patch("fine_tuning.run_experiment.subprocess.Popen", return_value=process), \
                    patch("fine_tuning.run_experiment.subprocess.run", side_effect=FileNotFoundError("probe absent")), \
                    patch("fine_tuning.run_experiment.time.monotonic", return_value=0), \
                    patch("fine_tuning.run_experiment.terminate_group") as cleanup:
                controller.stage("live", ["stage"], 1200)
            cleanup.assert_called_once_with(process, grace_seconds=15)
            events = [json.loads(line) for line in (root/"output/journal.jsonl").read_text().splitlines()]
            health = next(e for e in events if e["status"] == "HEALTH_POLL")
            self.assertTrue(health["process_live"])
            self.assertIn("FileNotFoundError", health["gpu_probe_error"])
            self.assertEqual(events[-1]["status"], "EXITED")

    def test_budget_boundaries_and_order_hash(self):
        self.assertFalse(stop_for_budget(10, None, 30))
        self.assertFalse(stop_for_budget(10, 1000, 30))
        self.assertTrue(stop_for_budget(850, 1000, 30))
        self.assertTrue(stop_for_budget(10, 100, 0))
        self.assertNotEqual(order_hash(["a", "b"]), order_hash(["b", "a"]))

    def test_financial_and_explicit_utc_cutoff(self):
        start = utc("2026-10-03T15:37:58Z")
        cap = utc("2026-10-04T03:00:00Z")
        self.assertEqual(effective_deadline(cap, start, 8.78, 1, .49), cap)
        self.assertEqual(effective_deadline(cap, start, 1.49, 1, .49), start + dt.timedelta(hours=1))
        with self.assertRaises(ValueError):
            utc("2026-10-04T03:00:00")
        with self.assertRaises(ValueError):
            effective_deadline(cap, start, float("nan"), 1, .49)

    def test_exposure_and_smoke_estimate_fail_closed(self):
        metadata = {"epoch_exposure_complete": True, "planned_records": 120,
                    "exposure": {"records": 120, "updates": 8}, "record_order_sha256": "same",
                    "consumed_order_sha256": "same", "seconds": 240}
        self.assertTrue(exposure_matches(metadata))
        self.assertEqual(pilot_estimate({arm: metadata for arm in ARMS}, 10000, 50), 160800)
        self.assertEqual(pilot_estimate({arm: metadata for arm in ARMS}, 10000, 50, margin=1.5), 120800)
        for margin in (0.99, float("nan"), float("inf"), float("-inf")):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                pilot_estimate({arm: metadata for arm in ARMS}, 10000, 50, margin=margin)
        self.assertEqual(pilot_estimate({arm: metadata for arm in ARMS}, 10000, 50,
            reused_training={"FL"}, reused_evaluations={("FL", "mmmu"), ("FL", "external")},
            evaluation_by_cohort={"mmmu": 30, "external": 20}), 120600)
        # A closed train+reload does not remove the still-required evaluations.
        self.assertEqual(pilot_estimate({arm: metadata for arm in ARMS}, 10000, 50,
                                      reused_training={"FL"}), 120650)
        with self.assertRaises(ValueError):
            pilot_estimate({arm: metadata for arm in ARMS}, 10000, 50, reused_evaluations={("FL", "mmmu")})
        metadata["epoch_exposure_complete"] = False
        self.assertFalse(exposure_matches(metadata))
        with self.assertRaises(ValueError):
            pilot_estimate({arm: metadata for arm in ARMS}, 10000, 50)
        assert_same_exposure({"FL": metadata, "FM": metadata})
        with self.assertRaises(ValueError):
            assert_same_exposure({"FL": metadata, "FM": dict(metadata, consumed_order_sha256="different")})

    def test_subprocess_success_failure_timeout_and_source_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "one.py").write_text("original")
            frozen = source_snapshot(source)
            controller = Controller(root / "output", dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5), source)
            controller.stage("okay", [sys.executable, "-c", "print('okay')"], 5)
            with self.assertRaises(RuntimeError):
                controller.stage("fail", [sys.executable, "-c", "raise SystemExit(3)"], 5)
            with self.assertRaises(TimeoutError):
                controller.stage("timeout", [sys.executable, "-c", "import time;time.sleep(10)"], .1)
            events = [json.loads(line) for line in (root / "output" / "journal.jsonl").read_text().splitlines()]
            self.assertTrue(any(e["status"] == "HARD_DEADLINE_STOP" for e in events))
            (source / "one.py").write_text("changed")
            with self.assertRaises(ValueError):
                assert_source_frozen(source, frozen)

    def test_closed_f0_evaluation_identity_order_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort, output = root / "cohort", root / "output"
            cohort.mkdir()
            output.mkdir()
            (cohort / "manifest.json").write_text(json.dumps({"count": 2}))
            (cohort / "inputs.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')
            protocol = root / "protocol.json"
            protocol.write_text('{}')
            predictions = output / "predictions.jsonl"
            predictions.write_text('{"id":"a","status":"COMPLETED"}\n{"id":"b","status":"COMPLETED"}\n')
            summary = {"status": "EVALUATED", "model_id": BASE_ID, "model_revision": BASE_REVISION,
                       "cohort_sha256": sha(cohort / "manifest.json"), "protocol_sha256": sha(protocol),
                       "denominator": 2, "completed": 2, "predictions_sha256": sha(predictions),
                       "checkpoint": "F0-frozen"}
            summary_path = output / "summary.json"
            summary_path.write_text(json.dumps(summary))
            evaluation_gate(output, cohort, protocol, baseline=True)
            summary["model_revision"] = "wrong"
            summary_path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                evaluation_gate(output, cohort, protocol, baseline=True)
            summary["model_revision"] = BASE_REVISION
            predictions.write_text('{"id":"b","status":"COMPLETED"}\n{"id":"a","status":"COMPLETED"}\n')
            summary["predictions_sha256"] = sha(predictions)
            summary_path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                evaluation_gate(output, cohort, protocol, baseline=True)

    def test_separate_pilot_root_and_no_scored_smoke_stages(self):
        for separate_root, reuse_fl, reuse_eval in ((False, False, False), (True, False, False),
                                                   (True, True, False), (True, True, True)):
            with self.subTest(separate_root=separate_root, reuse_fl=reuse_fl, reuse_eval=reuse_eval), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                controller = Controller(root / "output", dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1), source)
                cfg = {"pilot_records": 2, "min_pixels": 256, "max_pixels": 512,
                       "episode_pixels": 1024, "max_sequence_tokens": 4096, "gradient_accumulation": 16}
                protocol = root / "protocol.json"
                protocol.write_text(json.dumps({"training": cfg, "master_seed": 3407}))
                paths = {}
                for key in ("smoke_records", "smoke_manifest", "pilot_records", "pilot_manifest"):
                    paths[key] = root / key
                    paths[key].write_text('{}')
                for key in ("mmmu_cohort", "external_cohort"):
                    paths[key] = root / key
                    paths[key].mkdir()
                    (paths[key] / "manifest.json").write_text('{}')
                args = SimpleNamespace(**paths, protocol=protocol, model_path=root / "model",
                    image_root=root / "smoke_images", pilot_image_root=root / "pilot_images" if separate_root else None,
                    baseline_mmmu=None, baseline_external=None,
                    reuse_pilot_controller=root/"closed-parent" if reuse_fl else None,
                    reuse_pilot_evaluation_controller=root/"closed-eval" if reuse_eval else None)
                expected_pilot_root = args.pilot_image_root or args.image_root
                smoke_rows = [{"root_id": str(i)} for i in range(120)]
                pilot_rows = [{"root_id": "pilot1"}, {"root_id": "pilot2"}]
                stages = []
                reused_metadata = {"epoch_exposure_complete": True, "planned_records": 2,
                    "exposure": {"records": 2, "updates": 1}, "record_order_sha256": "pilot",
                    "consumed_order_sha256": "pilot", "seconds": 1}
                old_checkpoint = root/"closed-parent/pilot_FL/checkpoint"
                if reuse_fl:
                    old_checkpoint.mkdir(parents=True)
                    (old_checkpoint/"metadata.json").write_text(json.dumps(reused_metadata))
                reused_results = {}
                if reuse_eval:
                    for name in ("mmmu", "external"):
                        old_output = root/f"closed-eval/pilot_FL_{name}"
                        old_output.mkdir(parents=True)
                        (old_output/"summary.json").write_text('{}')
                        (old_output/"predictions.jsonl").write_text('')
                        reused_results["FL", name] = old_output

                def fake_stage(name, command, maximum_seconds=None):
                    stages.append(name)
                    controller.stage_seconds[name] = 10
                    if name.endswith("_train"):
                        self.assertNotIn("--checkpoint", command)
                        phase = name.split("_")[0]
                        self.assertEqual(Path(command[command.index("--image-root")+1]),
                                         expected_pilot_root if phase == "pilot" else args.image_root)
                        checkpoint = Path(command[command.index("--output")+1]) / "checkpoint"
                        checkpoint.mkdir(parents=True)
                        count = 360 if phase == "smoke" else 2
                        metadata = {"epoch_exposure_complete": True, "planned_records": count,
                            "exposure": {"records": count, "updates": 1}, "record_order_sha256": phase,
                            "consumed_order_sha256": phase, "seconds": 1}
                        (checkpoint / "metadata.json").write_text(json.dumps(metadata))
                    elif name.endswith(("_mmmu", "_external")):
                        if name.startswith("pilot_FL_") and reuse_fl:
                            self.assertEqual(Path(command[command.index("--checkpoint")+1]), old_checkpoint)
                        output = Path(command[command.index("--output")+1])
                        output.mkdir()
                        (output / "summary.json").write_text('{}')
                        (output / "predictions.jsonl").write_text('')

                with patch("fine_tuning.train.approved_records", side_effect=[
                        (smoke_rows, {"records_sha256": "smoke"}), (pilot_rows, {"records_sha256": "pilot"})]) as approved, \
                        patch("fine_tuning.run_experiment.cohort_gate"), \
                        patch("fine_tuning.run_experiment.evaluation_gate"), \
                        patch("fine_tuning.run_experiment.paired_input_gate", return_value={"status": "PASS"}) as paired, \
                        patch("fine_tuning.run_experiment.reuse_pilot", return_value=({"FL": reused_metadata}, {},
                            {"arms": {"FL": {"checkpoint": str(old_checkpoint)}}})) as reused_training, \
                        patch("fine_tuning.run_experiment.reuse_pilot_evaluations", return_value=(reused_results, {},
                            {"arms": {"FL": {"mmmu": {}, "external": {}}}})) as reused_evaluation, \
                        patch("fine_tuning.run_experiment.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="run-external")), \
                        patch.object(controller, "stage", side_effect=fake_stage):
                    result = run(args, controller)
                self.assertEqual(result["status"], "FOUR_ARM_PILOT_EVALUATED")
                self.assertEqual(approved.call_args_list[0].args[-1], args.image_root)
                self.assertEqual(approved.call_args_list[1].args[-1], expected_pilot_root)
                self.assertEqual([n for n in stages if n.startswith("smoke_")],
                                 [f"smoke_{arm}_{step}" for arm in ARMS for step in ("train", "reload")])
                self.assertEqual(sum(n.endswith(("_mmmu", "_external")) for n in stages), 8 if reuse_eval else 10)
                self.assertEqual(reused_training.call_count, int(reuse_fl))
                self.assertEqual(reused_evaluation.call_count, int(reuse_eval))
                if reuse_fl:
                    self.assertNotIn("pilot_FL_train", stages)
                    self.assertNotIn("pilot_FL_reload", stages)
                    self.assertFalse((controller.output/"pilot_FL").exists())
                    self.assertTrue((controller.output/"pilot_reuse.json").exists())
                for arm in ("FM", "FV", "FML"):
                    self.assertIn(f"pilot_{arm}_train", stages)
                    self.assertIn(f"pilot_{arm}_reload", stages)
                self.assertEqual(paired.call_count, 8)
                for call in paired.call_args_list:
                    output, baseline, cohort, used_protocol = call.args
                    name = output.name.rsplit("_", 1)[-1]
                    self.assertEqual(baseline, controller.output / f"baseline_F0_{name}")
                    self.assertEqual(cohort, paths[f"{name}_cohort"])
                    self.assertEqual(used_protocol, protocol)
                    receipt_path = (controller.output/f"pilot_FL_{name}_input_equivalence.json"
                                    if reuse_eval and output.parent.name == "closed-eval" else output/"input_equivalence.json")
                    self.assertEqual(json.loads(receipt_path.read_text())["cohort"], name)

    def test_paired_actual_inputs_fail_closed_on_missing_mismatch_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort, baseline, pilot = root / "cohort", root / "baseline", root / "pilot"
            for path in (cohort, baseline, pilot):
                path.mkdir()
            (cohort / "manifest.json").write_text(json.dumps({"count": 2}))
            (cohort / "inputs.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')
            protocol = root / "protocol.json"
            protocol.write_text('{}')
            rows = [{"id": sid, "status": "COMPLETED", "input_sha256": sid,
                     "prompt_token_ids": [1, 2], "image_grid_thw": [[1, 2, 2]],
                     "processed_image_sizes": [[32, 32]], "seed": 3407} for sid in ("a", "b")]

            def save(output, values):
                predictions = output / "predictions.jsonl"
                predictions.write_text("".join(json.dumps(row)+"\n" for row in values))
                (output / "summary.json").write_text(json.dumps({"status": "EVALUATED",
                    "model_id": BASE_ID, "model_revision": BASE_REVISION,
                    "cohort_sha256": sha(cohort / "manifest.json"), "protocol_sha256": sha(protocol),
                    "denominator": 2, "completed": 2, "predictions_sha256": sha(predictions),
                    "checkpoint": "F0-frozen" if output == baseline else "pilot",
                    "evaluator_source_sha256": output.name}))

            save(baseline, rows)
            original = sha(baseline / "predictions.jsonl")
            save(pilot, rows)
            receipt = paired_input_gate(pilot, baseline, cohort, protocol)
            self.assertEqual((receipt["status"], receipt["count"]), ("PASS", 2))
            for field in ("input_sha256", "prompt_token_ids", "image_grid_thw", "processed_image_sizes", "seed"):
                with self.subTest(field=field):
                    changed = [dict(row) for row in rows]
                    changed[0][field] = "different"
                    save(pilot, changed)
                    with self.assertRaisesRegex(ValueError, f"Paired observed input differs: a/{field}"):
                        paired_input_gate(pilot, baseline, cohort, protocol)
            for missing_output in (baseline, pilot):
                with self.subTest(missing_output=missing_output):
                    save(baseline, rows)
                    save(pilot, rows)
                    changed = [dict(row) for row in rows]
                    del changed[0]["prompt_token_ids"]
                    save(missing_output, changed)
                    with self.assertRaisesRegex(ValueError, "Missing paired observed input: a/prompt_token_ids"):
                        paired_input_gate(pilot, baseline, cohort, protocol)
            save(baseline, rows)
            save(pilot, rows[::-1])
            with self.assertRaisesRegex(ValueError, "Evaluation ID order/coverage/status mismatch"):
                paired_input_gate(pilot, baseline, cohort, protocol)
            self.assertEqual(sha(baseline / "predictions.jsonl"), original)


if __name__ == "__main__":
    unittest.main()
