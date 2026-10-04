"""Small CPU contract checks; fixture receipts are not GPU or source certification."""
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fine_tuning import audit, prepare_controls, prepare_training, run_control
from fine_tuning.modules import BASE_REVISION
from fine_tuning.tests import test_controller as existing, test_prepare_controls as data_fixtures


class ControlRunnerTests(unittest.TestCase):
    def test_fresh_training_requires_full_exposure_updates_frozen_and_payload_proof(self):
        for bad in (None, "tokens", "updates", "frozen", "module", "partial", "payload", "traversal"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                parent, rows, _records, manifest, cfg, _protocol, checkpoint = existing.ControllerTests().pilot_reuse_fixture(Path(tmp))
                # The standalone manifest layout is different from the borrowed reuse fixture.
                approved = manifest.parent/"approved/records.jsonl"
                approved.parent.mkdir()
                approved.write_bytes((manifest.parent/"records.jsonl").read_bytes())
                metadata_path = checkpoint/"metadata.json"
                metadata = json.loads(metadata_path.read_text())
                reference = json.loads(metadata_path.read_text())
                metadata["exposure"]["visual_tokens"] = 1600
                updates = audit.read_rows(checkpoint.parent/"updates.jsonl")
                if bad == "tokens":
                    metadata["exposure"]["target_tokens"] += 1
                elif bad == "updates":
                    updates[0]["loss_tokens"] += 1
                elif bad == "frozen":
                    updates[0]["frozen_sampled_checks"] = 0
                elif bad == "module":
                    metadata["inventory"]["arm"] = "FM"
                elif bad == "partial":
                    metadata["epoch_exposure_complete"] = False
                elif bad == "payload":
                    (checkpoint/"adapter/adapter_model.safetensors").write_bytes(b"changed")
                elif bad == "traversal":
                    metadata["files"]["../../outside"] = "a"*64
                metadata_path.write_text(json.dumps(metadata))
                (checkpoint.parent/"updates.jsonl").write_text("".join(json.dumps(row)+"\n" for row in updates))
                if bad:
                    with self.assertRaises(ValueError):
                        run_control.verify_training(parent/"pilot_FL", rows, manifest, cfg, reference)
                else:
                    result = run_control.verify_training(parent/"pilot_FL", rows, manifest, cfg, reference)
                    self.assertEqual(result["exposure"]["updates"], 100)

    def lineage_fixture(self, root):
        parent, reference, control, smoke = (root/name for name in ("parent", "reference", "control", "smoke"))
        smoke_images = root/"prototype_images"
        (parent/"images").mkdir(parents=True)
        (parent/"images/query.png").write_bytes(b"synthetic image bytes; no processor")
        rows = data_fixtures.fixture_rows()
        sources = {source: {"benchmark_derived": False, "rights_approved": True, "source_known": True, "parents": []}
                   for source in ("trace", "nvw-independent-procedural-v1", "independent-nvw-v1")}
        manifest = {"status": "APPROVED", "training_allowed": True, "benchmark_derived": False,
            "record_count": len(rows), "known_parent_ids": list(sources), "sources": sources,
            "source_sha256": {name: audit.sha(Path(run_control.__file__).with_name(name)) for name in ("train.py", "modules.py")},
            "files": {"images/query.png": audit.sha(parent/"images/query.png")}, "contamination_audit": {"status": "PASS"},
            "model_input_audit": {"status": "PASS", "record_count": len(rows), "processor_revision": BASE_REVISION,
                                  "sequence_budget": 4096, "episode_pixels": 1048576}}
        audit.write_rows(parent/"approved/records.jsonl", rows)
        manifest["records_sha256"] = audit.sha(parent/"approved/records.jsonl")
        audit.dump(parent/"common_manifest.json", manifest)
        audit.dump(parent/"completion_receipt.json", {"fixture": True})
        audit.dump(parent/"canonical_private.jsonl", {"fixture": True})
        frozen = {name: audit.sha(parent/name) for name in ("approved/records.jsonl", "common_manifest.json", "completion_receipt.json", "canonical_private.jsonl")}
        smoke_rows = []
        for i in range(120):
            for view in ("qa", "structure", "grounded"):
                smoke_rows.append(dict(rows[0], id=f"smoke-{i}-{view}", root_id=f"smoke-{i}", family_id=f"smoke-{i}",
                    source_id="nvw-independent-procedural-v1", source_kind="INDEPENDENT_SYNTHETIC",
                    lineage=["independent-nvw-v1"], parents=[], view=view))
        audit.write_rows(smoke/"approved/records.jsonl", smoke_rows)
        smoke_manifest = dict(manifest, record_count=360, records_sha256=audit.sha(smoke/"approved/records.jsonl"))
        audit.dump(smoke/"smoke_manifest.json", smoke_manifest)
        (smoke_images/"images").mkdir(parents=True)
        (smoke_images/"images/query.png").write_bytes((parent/"images/query.png").read_bytes())
        evidence = (rows, manifest, {"kind": "EARLIER_CLOSED_PARENT"}, frozen, {})
        with patch.object(prepare_training, "closed_pilot", return_value=evidence):
            prepare_training.subset_pilot(parent, reference, 1600, "CPU fixture", smoke/"approved/records.jsonl")
            prepare_controls.prepare(parent, reference, control)
        return parent, reference, control, smoke, smoke_images, evidence

    def test_control_requires_exact_parent_preparation_smoke_and_image_bindings(self):
        for bad in (None, "parent", "copied", "smoke", "policy", "image", "approval", "wrong_smoke_root", "smoke_image"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                parent, reference, control, smoke, smoke_images, evidence = self.lineage_fixture(Path(tmp))
                if bad == "parent":
                    receipt = json.loads((control/"control_receipt.json").read_text())
                    receipt["parent_files_sha256"]["canonical_private.jsonl"] = "a"*64
                    (control/"control_receipt.json").write_text(json.dumps(receipt))
                    manifest = json.loads((control/"common_manifest.json").read_text())
                    manifest["preparation_receipt_sha256"] = audit.sha(control/"control_receipt.json")
                    (control/"common_manifest.json").write_text(json.dumps(manifest))
                elif bad == "copied":
                    (control/"parent_evidence/canonical_private.jsonl").write_bytes(b"changed")
                elif bad == "smoke":
                    (reference/"smoke_exclusion/records.jsonl").write_bytes(b"changed")
                elif bad == "policy":
                    (control/"mixture_policy.json").write_text(json.dumps({"nvw_fraction": 0.1}))
                elif bad == "image":
                    (control/"images/query.png").write_bytes(b"changed")
                elif bad == "approval":
                    manifest = json.loads((control/"common_manifest.json").read_text())
                    manifest["known_parent_ids"].append("unapproved")
                    (control/"common_manifest.json").write_text(json.dumps(manifest))
                elif bad == "wrong_smoke_root":
                    smoke_images = smoke
                elif bad == "smoke_image":
                    (smoke_images/"images/query.png").write_bytes(b"changed")
                with patch.object(prepare_training, "closed_pilot", return_value=evidence):
                    if bad:
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            run_control.control_inputs(parent, reference, control, smoke, smoke_images)
                    else:
                        rows, reference_rows, smoke_rows, frozen = run_control.control_inputs(parent, reference, control, smoke, smoke_images)
                        self.assertEqual((len(rows), len(reference_rows), len(smoke_rows)), (1600, 1600, 360))
                        self.assertEqual(frozen[str((smoke_images/"images/query.png").resolve())],
                                         audit.sha(smoke_images/"images/query.png"))
                        self.assertFalse((smoke/"images").exists())

    def test_preflight_budget_and_reference_gates_never_start_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root/"protocol.json"
            protocol.write_bytes(Path(run_control.__file__).with_name("protocol.json").read_bytes())
            reference = root/"reference"
            reference.mkdir()
            audit.dump(reference/"common_manifest.json", {})
            model = root/"model"
            model.mkdir()
            audit.dump(model/"config.json", {})
            for name in ("mmmu", "external"):
                audit.dump(root/name/"manifest.json", {})
                audit.dump(root/name/"inputs.jsonl", {})
                audit.dump(root/name/"gold.jsonl", {})
                audit.dump(root/f"baseline_{name}"/"summary.json", {})
                audit.dump(root/f"baseline_{name}"/"predictions.jsonl", {})
                audit.dump(root/f"reference_{name}"/"summary.json", {"seconds": 1})
            args = SimpleNamespace(parent=root, reference=reference, control=root, smoke_root=root, smoke_image_root=root/"images",
                smoke_controller=root, reference_training_controller=root, reference_evaluation_controller=root,
                previous_controller=root, protocol=protocol, model_path=model,
                mmmu_cohort=root/"mmmu", external_cohort=root/"external",
                baseline_mmmu=root/"baseline_mmmu", baseline_external=root/"baseline_external")
            controller = Mock(output=root/"output")
            controller.output.mkdir()
            controller.remaining.return_value = 120
            results = {("FL", n): root/f"reference_{n}" for n in ("mmmu", "external")}
            with patch.object(run_control, "control_inputs", return_value=([], [], [], {})), \
                    patch.object(run_control.runner, "closed_controller", return_value=({}, [], {})), \
                    patch.object(run_control.runner, "pilot_scale_evidence"), \
                    patch.object(run_control.runner, "reuse_smoke", return_value=({}, {}, {})), \
                    patch.object(run_control.runner, "reuse_pilot", return_value=({"FL": {"seconds": 1, "exposure": {"records": 1600}}}, {}, {"arms": {"FL": {"checkpoint": str(root)}}})), \
                    patch.object(run_control.runner, "reuse_pilot_evaluations", return_value=(results, {}, {})), \
                    patch.object(run_control.runner, "cohort_gate"), patch.object(run_control.runner, "evaluation_gate"):
                self.assertEqual(run_control.run(args, controller)["status"], "BUDGET_STOP")
                controller.stage.assert_not_called()
                results.pop(("FL", "external"))
                with self.assertRaisesRegex(ValueError, "Both closed"):
                    run_control.run(args, controller)
                controller.stage.assert_not_called()
                results["FL", "external"] = root/"reference_external"
                controller.output = root/"complete_output"
                controller.output.mkdir()
                controller.remaining.return_value = 1000
                metadata = {"epoch_exposure_complete": True, "planned_records": 1600,
                    "exposure": {"records": 1600, "updates": 100}, "record_order_sha256": "same",
                    "consumed_order_sha256": "same", "files": {"adapter.bin": audit.sha(protocol)}}
                stages = []

                def fake_stage(name, command, maximum=None):
                    stages.append(name)
                    if name.endswith("train"):
                        output = Path(command[command.index("--output")+1])
                        audit.dump(output/"checkpoint/metadata.json", metadata)
                        (output/"checkpoint/adapter.bin").write_bytes(protocol.read_bytes())
                        audit.dump(output/"consumed_ids.jsonl", {})
                        audit.dump(output/"updates.jsonl", {})
                    elif name.endswith(("mmmu", "external")):
                        output = Path(command[command.index("--output")+1])
                        checkpoint = controller.output/"pilot_FL/checkpoint"
                        audit.dump(output/"summary.json", {"status": "EVALUATED", "checkpoint": str(checkpoint),
                            "checkpoint_metadata_sha256": audit.sha(checkpoint/"metadata.json"),
                            "evaluator_source_sha256": audit.sha(Path(run_control.__file__).with_name("evaluate.py")),
                            "benchmark_derived": name.endswith("mmmu")})
                        audit.dump(output/"predictions.jsonl", {})

                predictions = [dict(id=str(i), correct=True, subject="fixture", input_sha256=str(i),
                    prompt_token_ids=[1], image_grid_thw=[[1, 1, 1]], processed_image_sizes=[[32, 32]],
                    seed=3407, metric_kind="qa") for i in range(60)]
                with patch.object(controller, "stage", side_effect=fake_stage), \
                        patch.object(run_control, "require_idle_gpu") as idle, \
                        patch.object(run_control, "verify_training", return_value=metadata), \
                        patch.object(run_control.runner, "paired_input_gate", return_value={"status": "PASS"}), \
                        patch.object(run_control, "load_evaluation", return_value=({}, predictions)):
                    completed = run_control.run(args, controller)
                    self.assertEqual(completed["status"], "ADDITIONAL_FL_NVW0_VS10_PILOT_COMPLETED")
                    self.assertEqual(stages, ["pilot_FL_train", "pilot_FL_reload", "pilot_FL_mmmu", "pilot_FL_external"])
                    self.assertEqual(idle.call_count, 4)
                    report = json.loads((controller.output/"report.json").read_text())
                    self.assertFalse(report["original_G7_completed"])
                    self.assertFalse(report["both_conditions_preregistered"])
                    self.assertEqual(report["comparisons"]["mmmu"]["helper_condition_labels"]["f0"], "trained_NVW10_FL_reference")
                    self.assertTrue(report["comparisons"]["external"]["mixed_metrics_not_MMMU"])

    def test_financial_deadline_and_input_mismatch_helpers_remain_strict(self):
        start = dt.datetime(2026, 10, 3, tzinfo=dt.timezone.utc)
        deadline = run_control.runner.effective_deadline(start+dt.timedelta(days=1), start, 8, 1, 0.54)
        self.assertAlmostEqual((deadline-start).total_seconds(), 7/0.54*3600, places=5)
        with self.assertRaises(ValueError):
            run_control.runner.effective_deadline(deadline, start, 1, 1, 0.54)
        with self.assertRaises(ValueError):
            run_control.compare_inputs([{"id": "one"}], [{"id": "one"}])

    def test_gpu_idle_requires_successful_empty_compute_process_list(self):
        for code, output in ((0, ""), (0, "5810\n"), (1, "")):
            with self.subTest(code=code, output=output), patch.object(run_control.subprocess, "run",
                    return_value=SimpleNamespace(returncode=code, stdout=output)) as probe:
                if code or output:
                    with self.assertRaisesRegex(ValueError, "GPU idle"):
                        run_control.require_idle_gpu()
                else:
                    run_control.require_idle_gpu()
                self.assertEqual(probe.call_args.kwargs["timeout"], 15)


if __name__ == "__main__":
    unittest.main()
