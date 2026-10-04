"""Hand-calculated private report checks; synthetic CPU ledgers are not GPU evidence."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION, SUBJECTS
from fine_tuning.audit import sha
from fine_tuning.report import build_report, compare_inputs, external_groups, load_evaluation, paired_counts
from fine_tuning.run_experiment import pilot_scale_evidence, reuse_pilot, reuse_pilot_evaluations
from fine_tuning.tests import test_controller as controller_fixtures


class ReportTests(unittest.TestCase):
    def test_closed_reuse_report_follows_original_references_and_rejects_invalid_evidence(self):
        for bad in (None, "metadata", "hash", "open", "partial", "ref", "journal", "kind",
                    "evaluation_open", "evaluation_partial", "evaluation_ref", "input", "evaluator", "scale"):
            with self.subTest(bad=bad), TemporaryDirectory() as directory:
                root = Path(directory)
                parent, training_rows, records, manifest, cfg, protocol, checkpoint = controller_fixtures.ControllerTests().pilot_reuse_fixture(root)
                # The controller fixture supplies real ledger contracts, not actual GPU weights.
                joint = {"NVW-K": {"qa": 40, "structure": 20, "grounded": 20},
                         "NVW-I": {"qa": 20, "structure": 10, "grounded": 10},
                         "NVW-S": {"qa": 20, "structure": 10, "grounded": 10}}
                labels = [{"source_id": "trace"} for _ in range(1440)] + [
                    {"source_id": "nvw-independent-procedural-v1", "nvw_track": track, "view": view}
                    for track, views in joint.items() for view, count in views.items() for _ in range(count)]
                for row, label in zip(training_rows, labels, strict=True):
                    row.update(label)
                records.write_text("".join(json.dumps(row)+"\n" for row in training_rows))
                tracks = {"NVW-K": 80, "NVW-I": 40, "NVW-S": 40}
                manifest_value = {"nvw_tracks": tracks, "nvw_joint_views": joint, "pilot_scale": {
                    "original_records": 10000, "selected_records": 1600, "selection_seed": 3407,
                    "selection_policy": "source/track hash, no outcomes", "reason": "explicit budget reduction",
                    "parent_manifest_sha256": "a"*64, "source_counts": {"trace": 1440, "nvw": 160},
                    "nvw_track_counts": tracks}}
                manifest.write_text(json.dumps(manifest_value))
                cfg["pilot_records"] = 10000
                protocol.write_text(json.dumps({"training": cfg, "master_seed": 3407}))
                scale = pilot_scale_evidence(training_rows, manifest_value, 10000, 1600, 360, 3407)
                metadata_path = checkpoint/"metadata.json"
                metadata = json.loads(metadata_path.read_text())
                metadata.update(records_sha256=sha(records), manifest_sha256=sha(manifest))
                metadata_path.write_text(json.dumps(metadata))
                freeze = {"protocol_sha256": sha(protocol), "seed": 3407, "scale": scale,
                    "pilot_records_sha256": sha(records), "pilot_manifest_sha256": sha(manifest),
                    "pilot_manifest": str(manifest), "image_root": str(root),
                    "files": {str(path): sha(path) for path in (records, manifest, protocol)}}
                (parent/"pilot_input_freeze.json").write_text(json.dumps(freeze))
                controller, evaluations = root/"controller", root/"evaluations"
                controller.mkdir()
                evaluations.mkdir()
                sources = json.loads((parent/"source_freeze.json").read_text())["sources"]
                (evaluations/"source_freeze.json").write_text(json.dumps({"sources": sources}))
                baselines, cohorts, evaluation_journal = {}, {}, []
                rows = [{"id": f"{subject}_{i}", "subject": subject, "status": "COMPLETED", "correct": i == 0,
                    "seed": index, "input_sha256": str(index), "prompt_token_ids": [index], "image_grid_thw": [],
                    "processed_image_sizes": [], "metric_kind": "qa"}
                    for index, (subject, i) in enumerate((subject, i) for subject in SUBJECTS for i in range(2))]
                for name in ("mmmu", "external"):
                    cohort, baseline, evaluated = root/name, root/(name+"_F0"), evaluations/f"pilot_FL_{name}"
                    for path in (cohort, baseline, evaluated):
                        path.mkdir()
                    (cohort/"inputs.jsonl").write_text("".join(json.dumps({"id": row["id"]})+"\n" for row in rows))
                    (cohort/"gold.jsonl").write_text('{}\n')
                    (cohort/"manifest.json").write_text(json.dumps({"status": "COHORT_FROZEN", "training_allowed": False,
                        "count": 60, "inputs_sha256": sha(cohort/"inputs.jsonl"), "gold_sha256": sha(cohort/"gold.jsonl")}))
                    freeze["files"][str(cohort/"manifest.json")] = sha(cohort/"manifest.json")
                    # Make the two cohort identities distinct while retaining a paired synthetic row ledger.
                    value = json.loads((cohort/"manifest.json").read_text())
                    value["fixture_cohort"] = name
                    (cohort/"manifest.json").write_text(json.dumps(value))
                    freeze["files"][str(cohort/"manifest.json")] = sha(cohort/"manifest.json")
                    for path in (baseline, evaluated):
                        (path/"predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
                        summary = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "status": "EVALUATED",
                            "predictions_sha256": sha(path/"predictions.jsonl"), "denominator": 60, "completed": 60,
                            "correct": 30, "accuracy": .5, "checkpoint": "F0-frozen" if path == baseline else str(checkpoint),
                            "checkpoint_metadata_sha256": None if path == baseline else sha(metadata_path),
                            "cohort_sha256": sha(cohort/"manifest.json"), "protocol_sha256": sha(protocol),
                            "benchmark_derived": name == "mmmu", "evaluator_source_sha256": sources["evaluate.py"], "seconds": 10}
                        (path/"summary.json").write_text(json.dumps(summary))
                    evaluation_journal.append({"stage": f"pilot_FL_{name}", "status": "EXITED", "returncode": 0})
                    baselines[name], cohorts[name] = baseline, cohort
                terminal = "FL_1600_PAIRED_EVALUATIONS_COMPLETED"
                evaluation_journal.append({"stage": "controller", "status": terminal})
                (evaluations/"journal.jsonl").write_text("".join(json.dumps(row)+"\n" for row in evaluation_journal))
                (evaluations/"status.json").write_text(json.dumps({"status": terminal}))
                (controller/"pilot_input_freeze.json").write_text(json.dumps(freeze))
                _pilots, _files, training_receipt = reuse_pilot(parent, training_rows, records, manifest, cfg, 3407, protocol, root)
                _outputs, _files, evaluation_receipt = reuse_pilot_evaluations(evaluations, {"FL": checkpoint}, baselines, cohorts, protocol)
                if bad == "metadata":
                    metadata["seed"] = 17
                    metadata_path.write_text(json.dumps(metadata))
                elif bad == "hash":
                    (checkpoint/"adapter/adapter_model.safetensors").write_bytes(b"changed payload")
                elif bad == "open":
                    (parent/"status.json").write_text('{"status":"RUNNING"}')
                elif bad == "partial":
                    metadata["epoch_exposure_complete"] = False
                    metadata_path.write_text(json.dumps(metadata))
                elif bad == "ref":
                    training_receipt["arms"]["FL"]["checkpoint"] = str(root/"wrong_checkpoint")
                elif bad == "kind":
                    training_receipt["kind"] = "UNVERIFIED"
                elif bad == "evaluation_open":
                    (evaluations/"status.json").write_text('{"status":"RUNNING"}')
                elif bad == "evaluation_ref":
                    evaluation_receipt["arms"]["FL"]["mmmu"]["output"] = str(root/"wrong_output")
                elif bad in {"evaluation_partial", "evaluator", "input"}:
                    path = evaluations/"pilot_FL_mmmu"
                    summary = json.loads((path/"summary.json").read_text())
                    if bad == "evaluation_partial":
                        summary.update(status="BUDGET_STOP", completed=59, accuracy=None)
                    elif bad == "evaluator":
                        summary["evaluator_source_sha256"] = "wrong"
                    else:
                        changed = [dict(row) for row in rows]
                        changed[0]["prompt_token_ids"] = [999]
                        (path/"predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in changed))
                        summary["predictions_sha256"] = sha(path/"predictions.jsonl")
                    (path/"summary.json").write_text(json.dumps(summary))
                elif bad == "scale":
                    freeze["scale"]["pilot_scale"]["reason"] = "different scale provenance"
                    (controller/"pilot_input_freeze.json").write_text(json.dumps(freeze))
                current_journal = []
                for name, value in (("pilot_reuse", training_receipt), ("pilot_evaluation_reuse", evaluation_receipt)):
                    path = controller/(name+".json")
                    path.write_text(json.dumps(value))
                    current_journal.append({"stage": name, "status": "PASS", "receipt_sha256": sha(path)})
                if bad == "journal":
                    current_journal[0]["receipt_sha256"] = "wrong"
                (controller/"journal.jsonl").write_text("".join(json.dumps(row)+"\n" for row in current_journal))
                before = {str(path): sha(path) for path in root.rglob("*") if path.is_file()}
                if bad:
                    with self.assertRaises(ValueError):
                        build_report(controller, baselines["mmmu"], baselines["external"], root/"report")
                    self.assertFalse((root/"report").exists())
                else:
                    result = build_report(controller, baselines["mmmu"], baselines["external"], root/"report")
                    fl = result["arms"]["FL"]
                    self.assertEqual(fl["status"], "COMPLETED")
                    self.assertTrue(fl["checkpoint"]["training_reused"])
                    self.assertEqual(fl["checkpoint"]["path"], str(checkpoint))
                    self.assertTrue(fl["checkpoint"]["fresh_reload_verified"])
                    self.assertTrue(fl["mmmu"]["evaluation_reused"])
                    self.assertEqual(fl["mmmu"]["path"], str(evaluations/"pilot_FL_mmmu"))
                    self.assertEqual(fl["mmmu"]["paired"]["net"], 0)
                    self.assertTrue(all(result["arms"][arm]["status"] == "未完成" for arm in ("FM", "FV", "FML")))
                    self.assertFalse(result["original_10K_execution_completed"])
                    self.assertIn("no new training", (root/"report/report.md").read_text())
                self.assertEqual(before, {name: sha(Path(name)) for name in before})

    def test_paired_four_cells_and_observed_input_gate(self):
        reference = [{"id": str(i), "correct": value, "subject": "Accounting", "seed": i,
                      "input_sha256": str(i), "prompt_token_ids": [i], "image_grid_thw": [],
                      "processed_image_sizes": []} for i, value in enumerate((True, False, False, True))]
        rows = [dict(row, correct=value) for row, value in zip(reference, (True, False, True, False), strict=True)]
        compare_inputs(reference, rows)
        counts = paired_counts(reference, rows)
        self.assertEqual({key: counts[key] for key in ("both_correct", "both_wrong", "gain", "loss", "net")},
                         {"both_correct": 1, "both_wrong": 1, "gain": 1, "loss": 1, "net": 0})
        rows[0]["prompt_token_ids"] = [999]
        with self.assertRaises(ValueError):
            compare_inputs(reference, rows)
        del rows[0]["prompt_token_ids"]
        with self.assertRaises(ValueError):
            compare_inputs(reference, rows)

    def test_external_metric_coverage_and_box_weighting(self):
        rows = [{"metric_kind": "structure", "correct": True, "structure_f1": .5,
                 "bbox_hits": 1, "bbox_denominator": 2},
                {"metric_kind": "structure", "correct": False, "bbox_hits": 3, "bbox_denominator": 3},
                {"metric_kind": "grounded", "correct": True, "supported_states_exact": False, "rules_exact": True}]
        groups = external_groups(rows, rows)
        self.assertEqual(groups["structure"]["count"], 2)
        self.assertEqual(groups["structure"]["ft"]["structure_f1"], {"mean": .5, "observed": 1, "missing": 1})
        self.assertEqual(groups["structure"]["ft"]["bbox"]["rate"], .8)
        self.assertEqual(groups["grounded"]["ft"]["supported_states_exact"]["mean"], 0)

    def test_missing_pilots_and_partial_ledgers_never_claim_accuracy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = root / "controller"
            controller.mkdir()
            rows = [{"id": f"{subject}_{i}", "subject": subject, "status": "COMPLETED", "correct": i == 0}
                    for subject in SUBJECTS for i in range(2)]
            for i, row in enumerate(rows):
                row.update(seed=i, input_sha256=str(i), prompt_token_ids=[i], image_grid_thw=[],
                           processed_image_sizes=[], metric_kind="qa")
            baseline_paths = []
            for name in ("mmmu", "external"):
                path = root / name
                path.mkdir()
                (path / "predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
                summary = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "status": "EVALUATED",
                    "predictions_sha256": sha(path / "predictions.jsonl"), "denominator": 60,
                    "completed": 60, "correct": 30, "accuracy": .5, "checkpoint": "F0-frozen",
                    "cohort_sha256": name, "protocol_sha256": "same", "benchmark_derived": name == "mmmu"}
                (path / "summary.json").write_text(json.dumps(summary))
                baseline_paths.append(path)
            before = [sha(path / "predictions.jsonl") for path in baseline_paths]
            report = build_report(controller, *baseline_paths, root / "report")
            self.assertTrue(all(item["status"] == "未完成" for item in report["arms"].values()))
            self.assertTrue(all(item["mmmu"]["accuracy"] is None for item in report["arms"].values()))
            self.assertEqual(before, [sha(path / "predictions.jsonl") for path in baseline_paths])
            checkpoint = controller / "pilot_FL" / "checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter.bin").write_bytes(b"synthetic CPU fixture, not real weights")
            metadata = {"arm": "FL", "base_id": MODEL_ID, "base_revision": MODEL_REVISION,
                        "files": {"adapter.bin": sha(checkpoint / "adapter.bin")},
                        "epoch_exposure_complete": True, "termination": "EPOCH_COMPLETE",
                        "planned_records": 10000, "exposure": {"records": 10000, "updates": 625},
                        "record_order_sha256": "order", "consumed_order_sha256": "order"}
            (checkpoint / "metadata.json").write_text(json.dumps(metadata))
            (controller / "journal.jsonl").write_text(json.dumps({"stage": "pilot_FL_reload", "status": "EXITED", "returncode": 0})+"\n")
            for name, baseline in zip(("mmmu", "external"), baseline_paths, strict=True):
                pilot = controller / f"pilot_FL_{name}"
                pilot.mkdir()
                (pilot / "predictions.jsonl").write_bytes((baseline / "predictions.jsonl").read_bytes())
                pilot_summary = json.loads((baseline / "summary.json").read_text())
                pilot_summary.update(checkpoint=str(checkpoint), checkpoint_metadata_sha256=sha(checkpoint / "metadata.json"))
                (pilot / "summary.json").write_text(json.dumps(pilot_summary))
            completed = build_report(controller, *baseline_paths, root / "completed_report")
            self.assertEqual(completed["arms"]["FL"]["status"], "COMPLETED")
            self.assertEqual(completed["arms"]["FL"]["mmmu"]["paired"]["net"], 0)
            self.assertEqual(len(completed["arms"]["FL"]["mmmu"]["subjects"]), 30)
            self.assertNotIn("accuracy", completed["arms"]["FL"]["external"])
            metadata.update(planned_records=400, exposure={"records": 400, "updates": 25},
                            records_sha256="c"*64, manifest_sha256="d"*64)
            (checkpoint / "metadata.json").write_text(json.dumps(metadata))
            scale = {"original_records": 10000, "actual_records": 400, "scaled": True,
                     "explicit_override": True, "original_10K_execution_completed": False,
                     "source_counts": {"trace": 360, "nvw": 40},
                     "nvw_tracks": {"NVW-K": 20, "NVW-I": 10, "NVW-S": 10},
                     "nvw_view_counts": {"qa": 20, "structure": 10, "grounded": 10},
                     "pilot_scale": {"original_records": 10000, "selected_records": 400}}
            freeze = {"scale": scale, "protocol_sha256": "same", "pilot_records_sha256": "c"*64,
                      "pilot_manifest_sha256": "d"*64}
            (controller / "pilot_input_freeze.json").write_text(json.dumps(freeze))
            for name in ("mmmu", "external"):
                pilot = controller / f"pilot_FL_{name}"
                saved = json.loads((pilot / "summary.json").read_text())
                saved["checkpoint_metadata_sha256"] = sha(checkpoint / "metadata.json")
                (pilot / "summary.json").write_text(json.dumps(saved))
            reduced = build_report(controller, *baseline_paths, root / "reduced_report")
            self.assertEqual(reduced["arms"]["FL"]["checkpoint"]["planned_records"], 400)
            self.assertEqual(reduced["arms"]["FL"]["status"], "COMPLETED")
            self.assertFalse(reduced["original_10K_execution_completed"])
            self.assertEqual(reduced["training_scale"]["source_counts"], {"trace": 360, "nvw": 40})
            self.assertTrue(any("10000-record pilot" in limit for limit in reduced["limits"]))
            freeze["scale"]["explicit_override"] = False
            (controller / "pilot_input_freeze.json").write_text(json.dumps(freeze))
            with self.assertRaisesRegex(ValueError, "explicit frozen"):
                build_report(controller, *baseline_paths, root / "invalid_reduced_report")
            path = baseline_paths[0]
            rows[-1] = {"id": rows[-1]["id"], "status": "BUDGET_STOP", "correct": None}
            (path / "predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            summary.update(status="BUDGET_STOP", completed=59, correct=30, accuracy=None,
                           predictions_sha256=sha(path / "predictions.jsonl"))
            (path / "summary.json").write_text(json.dumps(summary))
            self.assertIsNone(load_evaluation(path)[0]["accuracy"])
            with self.assertRaises(ValueError):
                load_evaluation(path, baseline=True)
            (path / "predictions.jsonl").write_text('tampered')
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                load_evaluation(path)


if __name__ == "__main__":
    unittest.main()
