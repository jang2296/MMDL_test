"""CPU artifact/consent checks; authored bytes are not model execution or user consent."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fine_tuning.error_note import campaign_v16b as campaign, v17_artifacts as artifacts, v17_eval as ev
from fine_tuning.error_note import v17_preflight as pf
from fine_tuning.error_note.prep_v16b import read_sealed, seal
from mmdl.runtime.artifacts import digest, write_json


class AdmissionTests(unittest.TestCase):
    def fixture(self, *, degraded=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        repo, output, exported = root / "repo", root / "campaign", root / "export"
        code = repo / "fine_tuning"
        code.mkdir(parents=True)
        source = code / "source.py"
        source.write_text('VALUE = "CPU fixture"\n')
        output.mkdir()
        checkpoint = output / "checkpoint-3"
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(b"CPU fixture")
        (exported / "model").mkdir(parents=True)
        weights = exported / "model/adapter.safetensors"
        weights.write_bytes(b"CPU fixture")
        source_identity = {"base_commit": "a" * 40, "actual_files": {"source.py": artifacts.file_sha(source)}}
        plan = seal(output / "campaign-plan.json", {"version": "single-campaign-v18", "source": source_identity,
            "local_sequential": False, "total_slots": 3, "model_path": str(root / "base"),
            "protocol": {"version": "error-note-v18", "sha256": "d" * 64, "prompts_sha256": "b" * 64}})
        metadata = {"protocol": plan["protocol"], "step": 3, "binding": campaign._binding(plan)}
        write_json(checkpoint / "metadata.json", metadata)
        descriptor = artifacts._ref(checkpoint / "metadata.json")
        seal(output / "campaign-state.json", {"status": "COMPLETE", "plan_sha256": plan["sha256"],
            "stop_reasons": [], "checkpoint": descriptor, "cursor": {"slot": 3, "phase": "final"}})
        seal(output / "terminal-closure.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor,
            "status": "DRAINED_AND_SETTLED", "pending_requests": 0, "unconsumed_rows": 0})
        comparison = {"complete": True, "overall": {"net_change": -3 if degraded else 0, "p_value": .01 if degraded else 1.}}
        effect_comparison = {"complete": True, "eligible": False, "overall": {"net_change": 0, "p_value": 1.}}
        monitor = seal(output / "monitor-final.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor, "comparison": comparison})
        effect = seal(output / "effect-final.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor, "comparison": effect_comparison})
        seal(output / "evaluation-admission.json", {"version": "v17-evaluation-admission-v1", "plan_sha256": plan["sha256"],
            "checkpoint": descriptor, "monitor_sha256": monitor["sha256"], "effect_sha256": effect["sha256"],
            "decision": ev.evaluation_gate(comparison, effect_comparison)})
        seal(output / "reload-3.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor,
            "runtime_equal": True, "greedy_equal": True, "probe_max_abs_difference": 0, "producer_pid": 1, "reload_pid": 2})
        write_json(exported / "model_ref.json", {"kind": "unmerged-lora", "dtype": "bfloat16", "code_commit": "a" * 40,
            "export": {"binding": campaign._binding(plan), "campaign": {"plan_sha256": plan["sha256"]},
                       "training_protocol": plan["protocol"]},
            "files": [{"path": weights.relative_to(exported / "model").as_posix(), "bytes": weights.stat().st_size,
                       "sha256": artifacts.file_sha(weights)}]})
        for name in ("functional.json", "cost.json"):
            seal(output / name, {"fixture": name})
        identity = {"code_commit": "a" * 40, "source_sha256": digest(source_identity["actual_files"]),
                    "campaign_plan_sha256": plan["sha256"], "protocol_sha256": plan["protocol"]["sha256"]}
        binding = {**identity, "production_admitted": True, "functional_receipt": artifacts._ref(output / "functional.json"),
                   "update_cost_receipt": artifacts._ref(output / "cost.json")}
        for patcher in (patch.object(artifacts, "ROOT", repo), patch.object(campaign, "_plan_check"),
                        patch.object(pf, "bind_training", return_value=binding),
                        patch("fine_tuning.notebook_rl.checkpoint.read_metadata", return_value=metadata),
                        patch.object(artifacts.subprocess, "run", return_value=NS(stdout="a" * 40)),
                        patch.object(artifacts, "protected_evaluation", return_value={"sha256": "f" * 64})):
            patcher.start()
            self.addCleanup(patcher.stop)
        args = NS(output=output, export_output=exported, evaluation_decision=None, artifact_output=output / "training-artifacts.json")
        sealed = artifacts.create_training_artifacts(args, plan)
        evaluator = {key: identity[key] for key in ("code_commit", "source_sha256")}
        return args, plan, sealed, evaluator, repo

    def authorize(self, args, repo, evaluator, *, text="CPU fixture explicit evaluation approval"):
        output = args.output / "training-artifacts-verified.json"
        with patch.object(artifacts, "evaluation_identity", return_value=evaluator):
            argv = ["verify", "--receipt", str(args.artifact_output), "--repo", str(repo), "--commit", "a" * 40,
                    "--output", str(output), "--expected-protocol", "error-note-v18"]
            if text is not None:
                argv += ["--authorization", text]
            artifacts.main(argv)
        return output

    def test_degraded_complete_training_closes_exports_and_seals_without_consent(self):
        args, plan, sealed, evaluator, repo = self.fixture(degraded=True)
        self.assertEqual(campaign.require_complete(args.output)["status"], "COMPLETE")
        self.assertEqual(artifacts.verify_training_artifacts(args.artifact_output), sealed)
        self.assertNotIn("evaluation_decision", sealed)
        self.assertEqual(read_sealed(Path(sealed["evaluation_admission"]["path"]))["decision"]["status"], "DEGRADED")
        argv = ["--output", str(args.output), "--version", "error-note-v18", "--execution-phase", "training"]
        with patch("builtins.print"), patch("fine_tuning.notebook_rl.proof_eval.export_for_eval") as exported:
            self.assertEqual(campaign.main(["require-complete", *argv]), 0)
            self.assertEqual(campaign.main(["export", *argv, "--base", plan["model_path"],
                "--export-output", str(args.export_output), "--code-commit", "a" * 40]), 0)
            exported.assert_called_once()
            self.assertEqual(campaign.main(["training-artifact-seal", *argv,
                "--export-output", str(args.export_output), "--artifact-output", str(args.artifact_output)]), 0)
        with self.assertRaisesRegex(ValueError, "DEGRADED"):
            self.authorize(args, repo, evaluator)

    def test_engine_refuses_missing_later_consent_before_device_or_model_checks(self):
        from fine_tuning.error_note import v17_eval_engine as engine
        from fine_tuning.notebook_rl import proof_eval as pe

        args, plan, sealed, evaluator, _ = self.fixture()
        hardware = args.output / "hardware.json"
        write_json(hardware, {})
        run = NS(output=args.output / "engine.json", max_seconds=1, training_seal=args.artifact_output,
            model_ref=Path(sealed["export"]["model_ref"]["path"]), base_path=Path(plan["model_path"]),
            eval_protocol=pe.VARIANTS[pe.FINAL_VARIANT_V8][0], hardware=hardware, code_commit="a" * 40,
            source_hash=None, artifact_verification=None)
        model = {"kind": "unmerged-lora", "export": {"training_protocol": plan["protocol"]}}
        with patch.object(pe, "_lora_model_ref", return_value=(model, {})), \
             patch.object(engine, "source_sha256", return_value=evaluator["source_sha256"]), \
             patch("mmdl.evaluation.engine.git_commit", return_value="a" * 40), \
             patch("mmdl.runtime.environment.collect_environment") as device:
            with self.assertRaisesRegex(ValueError, "artifact-verification"):
                engine.validate(run)
            device.assert_not_called()

    def test_review_required_seal_needs_later_explicit_authorization_and_rechecks_every_binding(self):
        args, plan, sealed, evaluator, repo = self.fixture()
        with self.assertRaisesRegex(ValueError, "authorization required"):
            self.authorize(args, repo, evaluator, text=None)
        path = self.authorize(args, repo, evaluator)
        original = path.read_bytes()
        verified = artifacts.verify_evaluation_authorization(path, args.artifact_output, evaluator)
        decision = verified["evaluation_decision"]
        self.assertFalse(decision["pre_authorized"])
        self.assertEqual(decision["plan_sha256"], plan["sha256"])
        self.assertEqual(decision["admission_sha256"], read_sealed(Path(sealed["evaluation_admission"]["path"]))["sha256"])
        for key, value in (("plan_sha256", "0" * 64), ("model_ref", {"path": "/other", "file_sha256": "0" * 64}),
                           ("admission_sha256", "0" * 64), ("pre_authorized", True)):
            changed = deepcopy(verified)
            changed["evaluation_decision"][key] = value
            changed["evaluation_decision"] = ev.seal(changed["evaluation_decision"])
            write_json(path, ev.seal(changed))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "authorization belongs"):
                artifacts.verify_evaluation_authorization(path, args.artifact_output, evaluator)
        path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "authorization belongs"):
            artifacts.verify_evaluation_authorization(path, args.artifact_output, evaluator | {"code_commit": "b" * 40})

    def test_resealed_admission_cannot_change_its_checkpoint_or_gate(self):
        args, _, sealed, _, _ = self.fixture()
        admission_path = Path(sealed["evaluation_admission"]["path"])
        original = read_sealed(admission_path)
        for key, value in (("checkpoint", {"path": "/other", "file_sha256": "0" * 64}),
                           ("decision", {"status": "EVALUATION_CANDIDATE"})):
            changed = ev.seal(original | {key: value})
            write_json(admission_path, changed)
            tampered = deepcopy(sealed)
            tampered["evaluation_admission"] = artifacts._ref(admission_path)
            write_json(args.artifact_output, ev.seal(tampered))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "admission/monitor/effect/checkpoint"):
                artifacts.verify_training_artifacts(args.artifact_output)


if __name__ == "__main__":
    unittest.main()
