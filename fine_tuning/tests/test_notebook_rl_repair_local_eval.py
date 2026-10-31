"""Scripted evaluator controller boundaries; no GPU or native metric correctness claim."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fine_tuning.notebook_rl import checkpoint, local_eval as runner, local_ladder_generation as generation
from fine_tuning.notebook_rl import local_ladder_recovery as recovery, local_profile, local_train
from fine_tuning.tests import test_notebook_rl_local_eval as legacy
from fine_tuning.tests.test_notebook_rl_repair_replay import probe_fixture


class RepairLocalEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = legacy.NativeLadderRuntimeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        case = self.fixture
        case.protocol = json.loads(json.dumps(local_profile.build_protocol(repair_notebook=True, context_limit=24576)))
        self.minimum = {"launch": {"protocol": case.protocol}, "files": [], "repair_functional_verified": True,
                        "S_gradient_verified": True, "repair_learning_verified": True, "benefit_verified": False}
        case.contract.update(stage="FIXED_TEN_FROM_FRESH_F0", admission_policy=local_train.REPAIR_ADMISSION,
            training_budget={"version": local_train.TRAINING_BUDGET_VERSION, "max_seconds": 14400},
            max_optimizer_updates=5, no_signal_policy="record; consume the fixed pair; no replacement or forced update",
            same_generated_notebook_checks_and_kl=True,
            initial_checkpoint="standalone NF4 F0 with newly seeded adapters, not minimal-trained weights",
            authority=case.protocol["local_authorization"], parent_gate=self.minimum,
            deterministic_execution={}, matched_notebook_arm=None,
            repair_protocol_sha256=case.protocol["sha256"], repair_objective_sha256=case.protocol["repair_objective_sha256"])
        case.launch.update(protocol=case.protocol, execution_contract_sha256=checkpoint.digest(case.contract))
        case.launch["binding"]["protocol_sha256"] = case.protocol["sha256"]
        case.metadata.update(protocol=case.protocol, cursor={"position": 10, "sync_pending": False})
        case.metadata["inventory"]["trainable"] = []
        for index, actor in enumerate(case.training):
            actor.image_paths = ["image"] * (2 if index == 1 else 1)
        ids = [case.training[0].problem_id, case.training[1].problem_id]
        probes = probe_fixture(ids)
        case.expected["forward_probes"] = probes
        case.files[str(case.capture / "checkpoint/metadata.json")] = case.metadata
        case.files[str(case.reload.parent / "forward_probes.json")] = probes
        case.files[str(case.reload.parent / "sync.json")] = [{"request_id": "local-sync:" + pid,
            "policy_version": 2, "mean_abs_delta": 0., "max_abs_delta": 0.} for pid in ids]
        self.proof: dict[str, Any] = {"sha256": "a" * 64, "S_gradient_verified": False, "paths": {name: "file-hash" for name in (
            "launch.json", "result.json", "reload_expected.json", "initialization.json", "repair_update_evidence.json", "checkpoint/runtime.pt")}}
        case.restored.update(execution_contract=case.contract, execution_contract_sha256=checkpoint.digest(case.contract),
            capture_binding_sha256=checkpoint.digest(case.launch["binding"]), repair_protocol_sha256=case.protocol["sha256"],
            repair_replay_sha256=self.proof["sha256"], repair_update_evidence_sha256="file-hash",
            fresh_reload_verified=True, repair_functional_verified=True, S_gradient_verified=False,
            repair_learning_verified=False, benefit_verified=False)
        self.enterContext(patch.object(local_train, "training_parent_gate", return_value=self.minimum))
        self.replay = self.enterContext(patch.object(local_train, "_repair_training_replay", return_value=self.proof))
        self.enterContext(patch.object(runner, "read", side_effect=lambda path: copy.deepcopy(case.files[str(path)])))
        self.snapshot = self.enterContext(patch("fine_tuning.notebook_rl.repair_replay.read_snapshot",
            side_effect=lambda path: (copy.deepcopy(case.files[str(path)]), "file-hash")))
        self.templates = generation.circuit_evaluation_prompts(generation.CIRCUIT_REPAIR_PROMPT_POLICY)
        self.enterContext(patch.object(local_profile, "local_prompts", return_value=case.templates))
        self.protocol = recovery.protocol(self.templates, case.protocol["sha256"], context_limit=24576)
        case.gate["protocol_sha256"] = runner.digest(self.protocol)
        case.sources.return_value = {"approval": {"fixture_only": "source"}}
        self.record = case.recovery_record()
        case.gate["protocol_sha256"] = runner.digest(self.protocol)
        self.record["protocol"] = self.protocol
        self.record["protocol_sha256"] = runner.digest(self.protocol)
        self.record["repair_metrics"] = {"final_claims": {"known_issue_count": 0, "known_issue_coverage": None}}
        for key in ("note_count", "note_verified_count", "note_verified_fraction"):
            self.record[key] = None
        self.record["generation"]["protocol"] = generation.protocol(self.templates, case.protocol["sha256"], context_limit=24576)
        for row in self.record["generation"]["rows"]:
            row["context_limit"] = 24576
        self.diagnose = self.enterContext(patch.object(recovery, "diagnose", side_effect=lambda *args, **kwargs: copy.deepcopy(self.record)))

    def evaluate(self, name="evaluation", *, condition="FT", arm="notebook", policy=None, with_recovery=True):
        return runner.evaluate_ladder(self.fixture.capture, self.fixture.cohort, self.fixture.receipt, self.fixture.root / name,
            condition=condition, arm=arm, reload_result=self.fixture.reload,
            minimal_capture=self.fixture.root / "minimum", minimal_reload=self.fixture.root / "minimum-reload",
            with_recovery=with_recovery, evaluation_prompt_policy=policy or generation.CIRCUIT_REPAIR_PROMPT_POLICY)

    def test_exact_repair_policy_and_recovery_required_before_outputs_or_model(self) -> None:
        for index, (policy, enabled) in enumerate(((None, True), (generation.CIRCUIT_PROMPT_POLICY, True),
                                                  (generation.CIRCUIT_REPAIR_PROMPT_POLICY, False))):
            output = self.fixture.root / f"invalid{index}"
            with self.subTest(policy=policy, recovery=enabled), self.assertRaises(ValueError):
                runner.evaluate_ladder(self.fixture.capture, self.fixture.cohort, self.fixture.receipt, output,
                    condition="FT", arm="notebook", reload_result=self.fixture.reload,
                    minimal_capture=self.fixture.root / "minimum", minimal_reload=self.fixture.root / "minimum-reload",
                    with_recovery=enabled, evaluation_prompt_policy=policy)
            self.assertFalse(output.exists())
        self.fixture.launch["protocol"]["context_limit"] = 16384
        with self.assertRaises(ValueError):
            self.evaluate("bad-context")
        self.assertFalse((self.fixture.root / "bad-context").exists())
        self.fixture.load_ft.assert_not_called()
        self.fixture.engine.assert_not_called()

    def test_s_false_training_can_evaluate_with_distinct_metrics_and_full_runtime_bindings(self) -> None:
        result = self.evaluate()
        self.assertEqual(result["evaluation_protocol"], recovery.REPAIR_EVALUATOR_VERSION)
        self.assertEqual(result["repair_metrics"], self.record["repair_metrics"])
        self.assertIsNone(result["note_count"])
        self.assertFalse(result["repair_source_S_gradient_verified"])
        self.assertEqual(result["repair_training_replay_sha256"], self.proof["sha256"])
        self.assertIn(str(self.fixture.capture / "checkpoint/runtime.pt"), result["runtime_input_files"])
        self.assertEqual(self.replay.call_args.kwargs["parameter_inventory"], [])

    def test_wrong_reload_link_boolean_step_and_stale_replay_stop_before_loading(self) -> None:
        for index, mutation in enumerate(("link", "bool", "source", "fresh")):
            original, proof = copy.deepcopy(self.fixture.restored), copy.deepcopy(self.proof)
            if mutation == "link":
                self.fixture.restored["repair_replay_sha256"] = "foreign"
            elif mutation == "bool":
                self.fixture.restored["step"] = True
            elif mutation == "fresh":
                self.fixture.restored["fresh_reload_verified"] = False
            else:
                self.proof["paths"]["result.json"] = "changed-version"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.evaluate(f"bad-binding{index}")
            self.assertFalse((self.fixture.root / f"bad-binding{index}").exists())
            self.fixture.restored.clear()
            self.fixture.restored.update(original)
            self.proof.clear()
            self.proof.update(proof)
        self.fixture.load_ft.assert_not_called()

    def test_r0_explicit_matched_reload_files_are_rechecked_during_inference(self) -> None:
        case = self.fixture
        source = case.root / "matched/checkpoint/runtime.pt"
        initialization = case.root / "matched/initialization.json"
        matched = {"initialization": {"path": str(initialization), "sha256": "file-hash"},
            "fresh_reload": {"path": str(case.root / "matched-reload"), "files": {"result.json": "file-hash"}},
            "repair_runtime_files": {str(source): "file-hash"}}
        case.contract.update(arm="answer-only", notebook_reward_enabled=False, matched_notebook_arm=matched)
        case.captured["arm"] = "answer-only"
        case.launch["execution_contract_sha256"] = checkpoint.digest(case.contract)
        case.restored["execution_contract_sha256"] = checkpoint.digest(case.contract)
        case.files[str(initialization)] = case.files[str(case.capture / "initialization.json")]
        case.matched.return_value = matched
        valid = self.evaluate("control-valid", arm="answer-only")
        self.assertIn(str(source), valid["runtime_input_files"])
        changed = False

        def recover(*args, **kwargs):
            nonlocal changed
            changed = True
            return copy.deepcopy(self.record)

        case.file_hash.side_effect = lambda path: "changed" if changed and Path(path) == source else "file-hash"
        self.diagnose.side_effect = recover
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.evaluate("control-mutation", arm="answer-only")
        self.assertEqual(case.matched.call_args.kwargs["matched_reload"], case.root / "matched-reload")
        self.assertEqual(case.matched.call_args.kwargs["admission"], case.contract["admission_policy"])
        self.assertIsNone(case.matched.call_args.kwargs["historical_code_sha256"])
        summary = json.loads((case.root / "control-mutation/result.json").read_text())
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertIsNone(summary["repair_metrics"])

    def test_f0_ft_comparison_preserves_exact_repair_metrics_without_old_note_claim(self) -> None:
        self.fixture.parameters.return_value = self.fixture.initial
        self.record["policy_version"] = self.record["generation"]["policy_version"] = 0
        before = self.evaluate("before", condition="F0")
        self.fixture.parameters.return_value = self.fixture.final
        self.record["policy_version"] = self.record["generation"]["policy_version"] = 2
        after = self.evaluate("after")
        paired = {"repair_metrics": {"left": before["repair_metrics"], "right": after["repair_metrics"]}}
        with patch.object(recovery, "validate_paired_records", return_value=paired):
            result = runner.compare(before, after)
        self.assertEqual(result["repair_metrics"], paired["repair_metrics"])
        self.assertFalse(result["general_performance_improvement_verified"])
        after["repair_metrics"] = {"foreign": True}
        with self.assertRaises(ValueError):
            runner.compare(before, after)

    def test_cli_accepts_only_explicit_repair_policy(self) -> None:
        with patch.object(runner, "evaluate_ladder", return_value={"status": "COMPLETE_NATIVE_LADDER_RECOVERY"}) as call, \
                patch("builtins.print"):
            runner.main(["evaluate-ladder", "--capture", "capture", "--cohort", "cohort", "--data-gate", "gate",
                "--reload-result", "reload", "--output", "out", "--condition", "FT", "--arm", "notebook",
                "--with-recovery", "--evaluation-prompt-policy", generation.CIRCUIT_REPAIR_PROMPT_POLICY])
        self.assertTrue(call.call_args.kwargs["with_recovery"])
        self.assertEqual(call.call_args.kwargs["evaluation_prompt_policy"], generation.CIRCUIT_REPAIR_PROMPT_POLICY)


class FunctionalRepairLocalEvalTests(RepairLocalEvalTests):
    def setUp(self) -> None:
        super().setUp()
        self.minimum.update(admission_policy=local_train.REPAIR_FUNCTIONAL_EXPLORATION,
                            S_gradient_verified=False, repair_learning_verified=False)
        case = self.fixture
        case.contract["admission_policy"] = local_train.REPAIR_FUNCTIONAL_EXPLORATION
        case.launch["execution_contract_sha256"] = checkpoint.digest(case.contract)
        case.restored["execution_contract_sha256"] = checkpoint.digest(case.contract)


if __name__ == "__main__":
    unittest.main()
