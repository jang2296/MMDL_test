"""Self-authored stage/authority fixtures; no genuine GPU, dataset or checkpoint."""
import copy
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning import tabmwp_runpod as runner
from mmdl.runtime.artifacts import digest, sha256_file, write_json


class TabMWPControllerTests(unittest.TestCase):
    def fixture(self, root):
        model = root / "runtime/model"
        model.mkdir(parents=True)
        args = SimpleNamespace(runtime_root=root / "runtime", trial_root=root / "trial",
                               approval_root=root / "approval", cohort_root=root / "external",
                               diagnostic_status=root / "diagnostic_status.json", authorization=root / "authority.json",
                               commit="a" * 40, owned_pod_id="own-fixture")
        write_json(args.runtime_root / "setup/model.json", {
            "status": "MODEL_HASH_VERIFIED", "code_commit": args.commit,
            "code_repository": "jang2296/MMDL_test", "model_path": str(model)})
        diagnostic_outputs = root / "original_diagnostic_outputs"
        diagnostic_outputs.mkdir()
        diagnostic_execution = {"code_commit": "b" * 40, "owned_pod_id": args.owned_pod_id,
                                "outputs": str(diagnostic_outputs)}
        write_json(args.diagnostic_status, {"status": "DIAGNOSTIC_INFERENCE_COMPLETED",
                   "actual_cells": 336, "training_executed": False, **diagnostic_execution})
        trial = {"training_contract": runner.contract(), "retention_ids": [],
                 "source_sha256": {name: sha256_file(Path(runner.__file__).with_name(name))
                                   for name in ("train.py", "modules.py", "tabmwp_trial.py")},
                 "record_order_sha256": "fixture-order", "consumption_order_sha256": "fixture-consumed"}
        trial["trial_hash"] = digest(trial)
        write_json(args.trial_root / "trial_manifest.json", trial)
        receipt = {"status": "DATA_APPROVED", "training_allowed": True,
                   "training_execution_authorized": False,
                   "trial_manifest_sha256": sha256_file(args.trial_root / "trial_manifest.json")}
        write_json(args.approval_root / "cpu_approval_receipt.json", receipt)
        cohort = {"tabmwp_preparation_hash": "fixture-preparation",
                  "approved_training_records": {"record_count": 2000, "sha256": "records-A"}}
        write_json(args.cohort_root / "manifest.json", cohort)
        (args.cohort_root / "samples.jsonl").write_text("self-authored fixture\n")
        manifests = {}
        for candidate in ("A", "B"):
            manifest = {"tabmwp_condition": candidate, "tabmwp_trial_hash": trial["trial_hash"],
                        "tabmwp_trial_manifest_sha256": receipt["trial_manifest_sha256"],
                        "tabmwp_trial_receipt_sha256": sha256_file(args.approval_root / "cpu_approval_receipt.json"),
                        "tabmwp_preparation_hash": cohort["tabmwp_preparation_hash"],
                        "training_contract": runner.contract(), "records_sha256": "records-" + candidate,
                        "record_order_sha256": trial["record_order_sha256"],
                        "consumption_order_sha256": trial["consumption_order_sha256"]}
            write_json(args.approval_root / candidate / "training_manifest.json", manifest)
            (args.approval_root / candidate / "approved").mkdir()
            (args.approval_root / candidate / "approved/records.jsonl").write_text("self-authored\n")
            manifests[candidate] = manifest
        binding = {"trial_manifest_sha256": receipt["trial_manifest_sha256"],
                   "cpu_approval_receipt_sha256": sha256_file(args.approval_root / "cpu_approval_receipt.json"),
                   "diagnostic_status_sha256": sha256_file(args.diagnostic_status),
                   "diagnostic_execution": diagnostic_execution,
                   "external_manifest_sha256": sha256_file(args.cohort_root / "manifest.json"),
                   "training_manifests_sha256": {v: sha256_file(args.approval_root / v / "training_manifest.json")
                                                 for v in ("A", "B")}}
        authority = {"status": "APPROVED_FOR_BOUNDED_TABMWP_AB", "training_execution_authorized": True,
                     "code_commit": args.commit, "owned_pod_id": args.owned_pod_id,
                     "training_contract": runner.contract(), "input_binding": binding,
                     "teacher_generated_supervision_allowed": False, "automatic_expansion_allowed": False}
        write_json(args.authorization, authority)
        return args, model, manifests, binding, authority, cohort

    def test_authority_cpu_only_wrong_pod_or_changed_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, model, manifests, binding, authority, cohort = self.fixture(Path(directory))
            def approved(_records, path, _images):
                return ([{"id": str(i)} for i in range(2000)], manifests[path.parent.name])
            with patch.object(runner, "model_identity"), patch.object(runner, "pair_check"), \
                    patch("fine_tuning.train.approved_records", side_effect=approved), \
                    patch("fine_tuning.tabmwp_evaluate.load_cohort", return_value=(cohort, {})):
                self.assertEqual(runner.input_gate(args), (model, manifests, binding))
                for key, value in (("training_execution_authorized", False), ("owned_pod_id", "other"),
                                   ("teacher_generated_supervision_allowed", True), ("input_binding", {})):
                    changed = copy.deepcopy(authority)
                    changed[key] = value
                    write_json(args.authorization, changed)
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        runner.input_gate(args)
                write_json(args.authorization, authority)
                from mmdl.runtime.artifacts import read_json
                diagnosis = read_json(args.diagnostic_status)
                for key, value in (("owned_pod_id", "other-pod"), ("code_commit", "invalid"),
                                   ("outputs", str(Path(directory) / "missing"))):
                    write_json(args.diagnostic_status, diagnosis | {key: value})
                    with self.subTest(diagnostic_key=key), self.assertRaises((ValueError, FileNotFoundError)):
                        runner.input_gate(args)
                write_json(args.diagnostic_status, {"status": "INCOMPLETE", "actual_cells": 1})
                with self.assertRaises(ValueError):
                    runner.input_gate(args)

    def controller(self, root, args, manifests, binding, incomplete=None, mutate=False):
        class Fake:
            deadline = datetime.now(timezone.utc) + timedelta(hours=1)
            output = root / "controller"
            def remaining(self):
                return 3600
            def stage(self, name, command, maximum_seconds=None):
                calls.append((name, command))
                if name.endswith("_external"):
                    output = Path(command[command.index("--output") + 1])
                    write_json(output / "status.json", {"status": "COMPLETED", "completed": 1 if incomplete == name else 132})
                    if mutate:
                        args.authorization.write_text("changed")
                elif name.endswith("_train"):
                    candidate = name[0]
                    output = Path(command[command.index("--output") + 1])
                    metadata = {"termination": "BUDGET_STOP" if incomplete == name else "EPOCH_COMPLETE",
                                "arm": "FM", "seed": 3407, "epochs": 1, "budgets": {"accumulation": 16},
                                "base_precision": "BF16", "trainable_master_precision": "FP32",
                                "epoch_exposure_complete": True, "planned_records": 2000,
                                "exposure": {"records": 2000, "updates": 125, "target_tokens": 20 if candidate == "A" else 50},
                                "consumed_order_sha256": "fixture-consumed", "record_order_sha256": "fixture-consumed",
                                "manifest_sha256": binding["training_manifests_sha256"][candidate],
                                "records_sha256": manifests[candidate]["records_sha256"]}
                    write_json(output / "checkpoint/metadata.json", metadata)
        calls = []
        return Fake(), calls

    def test_fresh_base_for_both_variants_complete_396_and_no_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, model, manifests, binding, _authority, _cohort = self.fixture(root)
            fake, calls = self.controller(root, args, manifests, binding)
            with patch.object(runner, "input_gate", return_value=(model, manifests, binding)):
                result = runner.run(args, fake)
            self.assertEqual([name for name, _ in calls], ["F0_external", "A_train", "A_reload", "A_export",
                                                          "A_external", "B_train", "B_reload", "B_export", "B_external"])
            for name, command in calls:
                if name.endswith("_train"):
                    self.assertEqual(command[command.index("--model-path") + 1], str(model))
                    self.assertEqual(command[command.index("--arm") + 1], "FM")
                if name.endswith("_external"):
                    self.assertEqual(command[command.index("--slots") + 1], "4")
            self.assertEqual(result["external_actual_cells"], 396)
            self.assertFalse(result["automatic_expansion_allowed"])
            self.assertEqual(result["target_token_exposure"], {"A": 20, "B": 50})

    def test_incomplete_or_mutated_inputs_block_later_stages(self):
        for incomplete, mutate, error in (("F0_external", False, ValueError),
                                          ("A_train", False, TimeoutError), (None, True, ValueError)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, model, manifests, binding, _authority, _cohort = self.fixture(root)
                fake, calls = self.controller(root, args, manifests, binding, incomplete, mutate)
                with patch.object(runner, "input_gate", return_value=(model, manifests, binding)), self.assertRaises(error):
                    runner.run(args, fake)
                self.assertNotIn("B_train", [name for name, _ in calls])


if __name__ == "__main__":
    unittest.main()
