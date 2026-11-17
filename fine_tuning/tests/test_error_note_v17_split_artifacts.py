"""Small CPU fixtures for split identity and tamper checks; no model execution."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fine_tuning.error_note import campaign_v16b as campaign
from fine_tuning.error_note import prep_v16b as prep
from fine_tuning.error_note import v17_artifacts as artifacts
from fine_tuning.error_note import v17_preflight as pf
from fine_tuning.error_note.audit import digest


class SplitArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.output = self.root / "campaign"
        self.export = self.root / "export"
        self.output.mkdir()
        self.export.mkdir()
        for path in ("src/mmdl/evaluation/engine.py", "prompts/mmmu_mcq_v1.txt", "configs/eval/mmmu_val_v8.yaml",
                     "fine_tuning/notebook_rl/eval_configs/mmmu_val_v8_b2_seed3407.yaml", "third_party/mmmu/eval_utils.py",
                     "src/mmdl/runtime/contracts.py"):
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture\n")
        source = self.repo / "fine_tuning/notebook_rl/proof_eval.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('FORMAT = "fixture"\ndef load_preregistration(path):\n    return path\n')
        self.source = {"base_commit": "a" * 40, "actual_files": {"notebook_rl/proof_eval.py": prep.file_sha(source)}}
        self.plan = prep.seal(self.output / "campaign-plan.json", {
            "source": self.source, "local_sequential": False, "total_slots": 3,
            "protocol": {"version": "error-note-v17", "sha256": "d" * 64, "prompts_sha256": "b" * 64}})
        self.identity = {"code_commit": "a" * 40, "source_sha256": digest(self.source["actual_files"]),
                         "campaign_plan_sha256": self.plan["sha256"], "protocol_sha256": "d" * 64}
        cost = prep.seal(self.output / "cost.json", {"version": "v17-update-cost-receipt-v1",
            **self.identity, "status": "CALIBRATION_LOW_COVERAGE", "optimizer_steps": 0,
            "coverage": {"actual_a1_count": 0}, "measurement": None, "actual_abc_status": "UNMEASURED_NO_ADMITTED_A1"})
        functional = prep.seal(self.output / "functional.json", {"version": pf.VERSION, "status": "PASS",
            **self.identity, "optimizer_steps": 0, "production_parents_consumed": 0, "local_smoke": False,
            "capacity": {"version": "v17-capacity-v1", "model_parameter_update": False,
                "report": {"version": "v17-update-memory-v1", "optimizer_steps": 0, "fixture_only": True,
                           "action_tokens": 8192, "measurement": "actual_cuda_bf16_shared_backward_only"}},
            "f0_transport": {"pass": True}, "coexistence": [{"pass": True}], "update_cost_sha256": cost["sha256"],
            "update_cost_receipt": artifacts._ref(self.output / "cost.json")})
        self.args = NS(output=self.output, code_commit="a" * 40, functional_receipt=self.output / "functional.json",
            functional_receipt_sha256=prep.file_sha(self.output / "functional.json"), export_output=self.export,
            evaluation_decision=self.output / "decision.json", artifact_output=self.output / "training-artifacts.json")
        self.functional = functional
        checkpoint = self.output / "checkpoint-3"
        checkpoint.mkdir()
        self.metadata = {"protocol": self.plan["protocol"], "step": 3, "binding": campaign._binding(self.plan)}
        (checkpoint / "metadata.json").write_text(json.dumps(self.metadata))
        (checkpoint / "weights.pt").write_bytes(b"small fixture")
        self.ready = prep.seal(self.output / "export-ready.json", {"checkpoint": str(checkpoint),
            "metadata_descriptor": artifacts._ref(checkpoint / "metadata.json"), "plan_sha256": self.plan["sha256"]})
        prep.seal(self.output / "reload-3.json", {"version": "v16b-reload-v1", "plan_sha256": self.plan["sha256"],
            "checkpoint": self.ready["metadata_descriptor"], "runtime_equal": True, "greedy_equal": True,
            "probe_max_abs_difference": 0, "producer_pid": 1, "reload_pid": 2})
        admission = prep.seal(self.output / "evaluation-admission.json", {"plan_sha256": self.plan["sha256"],
            "checkpoint": self.ready["metadata_descriptor"], "decision": {"status": "EVALUATION_CANDIDATE"}})
        prep.seal(self.args.evaluation_decision, {"decision": "PROCEED", "authorized_by": "user",
            "plan_sha256": self.plan["sha256"], "admission_sha256": admission["sha256"]})
        (self.export / "model").mkdir()
        weights = self.export / "model/adapter.safetensors"
        weights.write_bytes(b"export fixture")
        (self.export / "model_ref.json").write_text(json.dumps({"kind": "unmerged-lora", "dtype": "bfloat16",
            "code_commit": "a" * 40, "export": {"binding": campaign._binding(self.plan),
                "campaign": {"plan_sha256": self.plan["sha256"]}}, "files": [{"path": weights.name,
                    "bytes": weights.stat().st_size, "sha256": prep.file_sha(weights)}]}))
        self.patches = [patch.object(artifacts, "ROOT", self.repo), patch.object(campaign, "_plan_check"),
            patch.object(campaign.single, "source_manifest", return_value=self.source),
            patch.object(campaign, "require_complete", return_value=self.ready),
            patch("fine_tuning.notebook_rl.checkpoint.read_metadata", return_value=self.metadata),
            patch.object(artifacts.subprocess, "run", return_value=NS(stdout="a" * 40))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def create(self):
        return artifacts.create_training_artifacts(self.args, self.plan)

    def test_training_needs_no_evaluation_calibration_and_is_immutable(self):
        bound = pf.bind_training(self.args, self.plan)
        self.assertEqual(bound["status"], "BOUND")
        self.assertNotIn("calibration", bound)
        self.assertEqual(pf.bind_training(self.args, self.plan), bound)
        self.args.functional_receipt_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            pf.bind_training(self.args, self.plan)

    def test_missing_training_seal_and_unverified_reload_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "training-artifact-seal"):
            pf.bind_evaluation(self.args, self.plan)
        path = self.output / "reload-3.json"
        reload = prep.read_sealed(path)
        path.unlink()
        prep.seal(path, {k: v for k, v in reload.items() if k != "sha256"} | {"reload_pid": 1})
        with self.assertRaisesRegex(ValueError, "fresh-process"):
            self.create()

    def test_complete_and_export_dispatch_use_training_binding(self):
        args = ["--output", str(self.output), "--version", "error-note-v17", "--execution-phase", "training",
                "--code-commit", "a" * 40, "--functional-receipt", str(self.args.functional_receipt),
                "--functional-receipt-sha256", self.args.functional_receipt_sha256]
        with patch.object(pf, "bind_execution", side_effect=AssertionError("legacy binding used")), \
                patch("fine_tuning.notebook_rl.proof_eval.export_for_eval") as export:
            self.assertEqual(campaign.main(["require-complete", *args]), 0)
            self.assertEqual(campaign.main(["export", *args, "--base", str(self.root / "base"),
                                           "--export-output", str(self.export)]), 0)
            export.assert_called_once()

    def test_seal_roundtrip_preserves_training_identity(self):
        sealed = self.create()
        self.assertEqual(artifacts.verify_training_artifacts(self.args.artifact_output), sealed)
        self.assertEqual(sealed["training"], self.identity)
        self.assertEqual(self.create(), sealed)
        self.assertEqual(sealed["export"]["model_ref"], artifacts._ref(self.export / "model_ref.json"))

    def test_every_bound_artifact_rejects_one_byte_change(self):
        self.create()
        for path in (self.output / "checkpoint-3/weights.pt", self.export / "model/adapter.safetensors",
                     self.export / "model_ref.json", self.output / "reload-3.json", self.output / "cost.json",
                     self.repo / "fine_tuning/notebook_rl/proof_eval.py"):
            original = path.read_bytes()
            with self.subTest(path=path):
                path.write_bytes(original + b"x")
                with self.assertRaises(ValueError):
                    artifacts.verify_training_artifacts(self.args.artifact_output)
                path.write_bytes(original)

    def test_added_checkpoint_file_and_symlink_escape_are_rejected(self):
        self.create()
        extra = self.output / "checkpoint-3/extra"
        extra.write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            artifacts.verify_training_artifacts(self.args.artifact_output)
        extra.unlink()
        extra.symlink_to(self.args.functional_receipt)
        with self.assertRaisesRegex(ValueError, "symlink"):
            artifacts.verify_training_artifacts(self.args.artifact_output)

    def test_relabeling_or_path_escape_is_rejected_even_after_resealing(self):
        sealed = self.create()
        for change in ("identity", "path"):
            changed = deepcopy(sealed)
            if change == "identity":
                changed["training"]["source_sha256"] = "e" * 64
            else:
                changed["training_code"]["files"][0]["path"] = "../outside.py"
                changed["training_code"]["files_sha256"] = digest(changed["training_code"]["files"])
            changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
            self.args.artifact_output.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                artifacts.verify_training_artifacts(self.args.artifact_output)

    def test_protected_real_b2_config_scoring_and_prereg_changes(self):
        before = artifacts.protected_evaluation(self.repo)
        for filename in ("fine_tuning/notebook_rl/eval_configs/mmmu_val_v8_b2_seed3407.yaml",
                         "third_party/mmmu/eval_utils.py", "fine_tuning/notebook_rl/proof_eval.py",
                         "src/mmdl/runtime/contracts.py"):
            path = self.repo / filename
            original = path.read_text()
            with self.subTest(filename=filename):
                path.write_text(original.replace("fixture", "changed"))
                self.assertNotEqual(artifacts.protected_evaluation(self.repo), before)
                path.write_text(original)

    def test_engine_binding_checks_final_model_evaluator_and_environment(self):
        from mmdl.evaluation import engine as evaluation

        sealed = self.create()
        self.args.training_artifact_seal = self.args.artifact_output
        self.args.engine_validation_receipt = self.output / "engine-validation.json"
        environment = {"python": "fixture", "platform": {}, "torch": {"cuda_runtime": "fixture"},
                       "nvidia_smi": {"gpus": []}}
        validation = {"version": "v17-eval-engine-validation-v1", "status": "PASS",
            "training_artifact_sha256": prep.file_sha(self.args.artifact_output),
            "model_ref_sha256": sealed["export"]["model_ref"]["file_sha256"],
            "evaluation_commit": "b" * 40, "source_sha256": "f" * 64,
            "environment_sha256": digest(evaluation.environment_identity(environment))}
        self.args.code_commit = "b" * 40
        with patch.object(artifacts, "evaluation_identity", return_value={"code_commit": "b" * 40, "source_sha256": "f" * 64}), \
                patch("mmdl.runtime.environment.collect_environment", return_value=environment):
            for key in ("model_ref_sha256", "evaluation_commit", "source_sha256", "environment_sha256"):
                invalid = validation | {key: "0" * len(validation[key])}
                self.args.engine_validation_receipt.write_text(json.dumps(invalid | {"sha256": digest(invalid)}))
                self.args.engine_validation_receipt_sha256 = prep.file_sha(self.args.engine_validation_receipt)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    pf.bind_evaluation(self.args, self.plan)
            self.args.engine_validation_receipt.write_text(json.dumps(validation | {"sha256": digest(validation)}))
            self.args.engine_validation_receipt_sha256 = prep.file_sha(self.args.engine_validation_receipt)
            result = pf.bind_evaluation(self.args, self.plan)
            self.assertEqual(result["training"]["code_commit"], "a" * 40)
            self.assertEqual(result["evaluation"]["code_commit"], "b" * 40)


if __name__ == "__main__":
    unittest.main()
