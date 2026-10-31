"""CPU caller propagation only; mechanical exploration does not certify notebook learning."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import checkpoint, local_eval, local_profile, local_run, local_train
from fine_tuning.notebook_rl.local_train import training_binding as real_training_binding
from fine_tuning.tests import test_notebook_rl_local_eval as evaluation
from fine_tuning.tests import test_notebook_rl_local_train as training


POLICY = local_run.VERIFIED_UPDATE_EXPLORATION


def protocol():
    return local_profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)


def parent_proof(current):
    return {"launch": {"protocol": current}, "verified_update_exploration_admitted": True,
            "strict_notebook_correctness_verified": False, "notebook_reward_learning_verified": False,
            "relationship_learning_verified": False}


class VerifiedUpdateCallerTests(unittest.TestCase):
    def test_fixed_ten_cli_propagates_admission_to_both_arms(self):
        required = ["train", "--preparation", "/unused/preparation", "--model-path", "/unused/base",
                    "--cache-check", "/unused/cache", "--minimal", "/unused/minimal",
                    "--minimal-reload", "/unused/reload", "--output", "/unused/output"]
        for arm in ("notebook", "answer-only"):
            extra = [] if arm == "notebook" else ["--match-initialization", "/unused/notebook/initialization.json"]
            with self.subTest(arm=arm), patch.object(local_train, "train", return_value={
                    "status": "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD"}) as called, patch("builtins.print"):
                self.assertEqual(local_train.main(required + ["--arm", arm, "--admission", POLICY, *extra]), 0)
                self.assertEqual(called.call_args.kwargs["admission"], POLICY)
                self.assertEqual(called.call_args.kwargs["arm"], arm)
                self.assertEqual(called.call_args.kwargs["max_seconds"], 14400)

    def test_fixed_ten_and_control_forward_retained_signal_kwarg_only_when_opted_in(self):
        for admission, arm in ((POLICY, "notebook"), (POLICY, "answer-only"),
                               (local_train.STRICT_ADMISSION, "notebook")):
            fixture = training.IntermediateCheckpointTests()
            fixture.setUp()
            try:
                if admission == POLICY:
                    fixture.protocol.clear()
                    fixture.protocol.update(protocol())
                parent = parent_proof(fixture.protocol)
                parent["launch"].update(model_path=str(fixture.model), cache_check=str(fixture.cache))
                match = None
                if arm == "answer-only":
                    match = fixture.root / "notebook/initialization.json"
                    match.parent.mkdir()
                    match.write_text(json.dumps({"trainable_sha256": local_run.parameter_hashes(fixture.bundle),
                        "base_sha256": fixture.bundle.base_sha256,
                        "quantization": fixture.bundle.inventory["quantization"],
                        "seed": fixture.protocol["local_budget"]["seed"],
                        "model_cache_metadata_sha256": checkpoint.file_hash(fixture.model / "nf4_base.json")}))
                with self.subTest(admission=admission, arm=arm), \
                        patch.object(local_train, "training_parent_gate", return_value=parent) as gate, \
                        patch.object(local_train, "matched_notebook_arm", return_value={"fixture_only": True}) as matched, \
                        patch("fine_tuning.notebook_rl.updates.bind_notebook_objective"), \
                        patch("fine_tuning.notebook_rl.updates.notebook_objective_scores", return_value={}), \
                        patch("fine_tuning.notebook_rl.updates.optimizer_step", side_effect=fixture.update) as update, \
                        patch.object(local_train, "retained_update_signal", return_value={"fixture_only": True}) as retained:
                    result = local_train.train(fixture.root / "preparation.json", fixture.model, fixture.cache,
                        fixture.root / "minimum", fixture.root / "reload", fixture.output,
                        arm=arm, match_initialization=match, admission=admission)
                gate.assert_called_once_with(fixture.root / "minimum", fixture.root / "reload", admission=admission)
                self.assertEqual(update.call_args.kwargs["notebook_reward_enabled"], arm == "notebook")
                if admission == POLICY:
                    self.assertIs(update.call_args.kwargs["require_retained_learning_signal"], True)
                    retained.assert_called_once()
                else:
                    self.assertNotIn("require_retained_learning_signal", update.call_args.kwargs)
                    retained.assert_not_called()
                if match:
                    self.assertEqual(matched.call_args.kwargs, {
                        "admission": admission, "parent": parent, "max_seconds": 14400})
                else:
                    matched.assert_not_called()
                launch = json.loads((fixture.output / "launch.json").read_text())
                self.assertEqual(launch["execution_contract"]["admission_policy"], admission)
                self.assertEqual(launch["execution_contract"]["parent_gate"], json.loads(json.dumps(parent)))
                self.assertEqual(len(launch["execution_contract"]["pair_ids"]), 5)
                self.assertEqual(result["optimizer_updates"], 1)
                self.assertFalse(result["self_improvement_verified"])
                self.assertFalse(result["notebook_gradient_contribution_verified"])
            finally:
                fixture.doCleanups()

    def minimum_fixture(self, root, *, explicit):
        current, prepared = protocol(), {"fixture_only": "prepared", "image_root": str(root / "images")}
        actors = [SimpleNamespace(problem_id="single"), SimpleNamespace(problem_id="pair")]
        contract = local_run.minimum_execution_contract(POLICY, current) if explicit else None
        launch = {"pid": 900001, "protocol": json.loads(json.dumps(current)), "standalone_base": False,
            "preparation": str(root / "preparation.json"), "model_path": str(root / "base"),
            "problem_ids": [actor.problem_id for actor in actors],
            "binding": local_run.minimum_binding(prepared, current, contract)}
        if explicit:
            launch.update(execution_contract=contract, execution_contract_sha256=checkpoint.digest(contract))
        saved = {"status": "UPDATED_SAVED_PENDING_FRESH_RELOAD", "optimizer_updates": 1,
                 "attempted_ids": launch["problem_ids"], "discarded_ids": []}
        expected = {"trainable_sha256": {"L": {"weight": "hash"}}, "step": 1,
                    "frozen_base_sha256": "base", "quantization": {"fixture_only": True}, "forward_probes": {}}
        for name, row in (("launch.json", launch), ("result.json", saved), ("reload_expected.json", expected)):
            (root / name).write_text(json.dumps(row))
        return launch, prepared, actors, expected

    def test_minimum_reload_echoes_explicit_contract_and_preserves_legacy_none_binding(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "capture"
                root.mkdir()
                launch, prepared, actors, expected = self.minimum_fixture(root, explicit=explicit)
                cursor = {"epoch": 0, "position": 2, "discarded_problem_ids": [],
                          "problem_ids": launch["problem_ids"], "sync_pending": False}
                bundle = SimpleNamespace(base_sha256="base", inventory={"quantization": {"fixture_only": True}})
                with ExitStack() as stack:
                    def mocked(name, **kwargs):
                        return stack.enter_context(patch(name, **kwargs))
                    mocked("psutil.pid_exists", return_value=False)
                    mocked("fine_tuning.notebook_rl.local_run.read_preparation", return_value=(prepared, {}, actors))
                    mocked("fine_tuning.notebook_rl.local_run.minimal_actors", return_value=actors)
                    mocked("fine_tuning.notebook_rl.local_run._install_actor_guards")
                    resumed = mocked("fine_tuning.notebook_rl.checkpoint.load_resume", return_value=(
                        bundle, SimpleNamespace(state={1: True}), SimpleNamespace(last_epoch=1), 1, cursor))
                    mocked("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged")
                    mocked("fine_tuning.notebook_rl.local_run.parameter_hashes", return_value=expected["trainable_sha256"])
                    mocked("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine", return_value=object())
                    mocked("fine_tuning.notebook_rl.local_run.load_images", return_value=[])
                    mocked("fine_tuning.notebook_rl.local_run.make_probes", return_value=[])
                    mocked("fine_tuning.notebook_rl.local_run.probe_evidence", return_value={})
                    mocked("fine_tuning.notebook_rl.trainer.synchronize", return_value={})
                    mocked("torch.cuda.max_memory_allocated", return_value=0)
                    result = local_run.reload_capture(root, root.parent / "restored")
                self.assertEqual(resumed.call_args.kwargs["binding"], launch["binding"])
                if explicit:
                    self.assertEqual(result["execution_contract"], launch["execution_contract"])
                    self.assertEqual(result["execution_contract_sha256"], launch["execution_contract_sha256"])
                    self.assertEqual(result["capture_binding_sha256"], checkpoint.digest(launch["binding"]))
                else:
                    self.assertEqual(launch["binding"], local_run._binding(prepared, protocol()))
                    self.assertNotIn("execution_contract", result)
                    self.assertNotIn("capture_binding_sha256", result)
                self.assertFalse(result["self_improvement_verified"])

    def test_missing_or_mutated_minimum_launch_contract_rejects_before_model(self):
        for mutation in ("missing_contract", "missing_digest", "missing_both", "wrong_digest", "rehashed_limit"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "capture"
                root.mkdir()
                launch, prepared, actors, _ = self.minimum_fixture(root, explicit=True)
                if mutation in ("missing_contract", "missing_both"):
                    del launch["execution_contract"]
                if mutation in ("missing_digest", "missing_both"):
                    del launch["execution_contract_sha256"]
                if mutation == "wrong_digest":
                    launch["execution_contract_sha256"] = "0" * 64
                if mutation == "rehashed_limit":
                    launch["execution_contract"]["max_optimizer_updates"] = 2
                    launch["execution_contract_sha256"] = checkpoint.digest(launch["execution_contract"])
                (root / "launch.json").write_text(json.dumps(launch))
                with patch("psutil.pid_exists", return_value=False), \
                        patch.object(local_run, "read_preparation", return_value=(prepared, {}, actors)), \
                        patch.object(local_run, "minimal_actors", return_value=actors), \
                        patch.object(checkpoint, "load_resume") as resume, \
                        patch("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy") as load, \
                        patch("fine_tuning.notebook_rl.quantized_policy.build_quantized_policy") as build:
                    with self.assertRaises(ValueError):
                        local_run.reload_capture(root, root.parent / "rejected")
                    resume.assert_not_called()
                    load.assert_not_called()
                    build.assert_not_called()
                self.assertFalse((root.parent / "rejected").exists())

    def native_fixture(self):
        from fine_tuning.notebook_rl import local_ladder_generation as generation

        fixture = evaluation.NativeLadderRuntimeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol.clear()
        fixture.protocol.update(json.loads(json.dumps(protocol())))
        parent = parent_proof(fixture.protocol)
        fixture.contract.update(admission_policy=POLICY, parent_gate=parent)
        fixture.launch["execution_contract_sha256"] = checkpoint.digest(fixture.contract)
        fixture.launch["binding"] = real_training_binding(fixture.prepared, fixture.protocol, fixture.contract)
        fixture.launch["binding"]["code_sha256"] = "historical"
        templates = generation.circuit_evaluation_prompts()
        current = generation.protocol(templates, fixture.protocol["sha256"], context_limit=24576)
        fixture.gate["protocol_sha256"] = local_eval.digest(current)
        fixture.generation["protocol"] = current
        for row in fixture.generation["rows"]:
            row["context_limit"] = 24576
        return fixture, parent, templates

    def test_native_evaluation_and_control_propagate_mechanical_parent_without_learning_claims(self):
        from fine_tuning.notebook_rl import local_ladder_generation as generation

        fixture, parent, templates = self.native_fixture()
        for arm in ("notebook", "answer-only"):
            fixture.captured["arm"] = fixture.contract["arm"] = arm
            fixture.contract["notebook_reward_enabled"] = arm == "notebook"
            if arm == "answer-only":
                path = fixture.root / "notebook/initialization.json"
                fixture.files[str(path)] = fixture.files[str(fixture.capture / "initialization.json")]
                fixture.contract["matched_notebook_arm"] = {"initialization": {"path": str(path)}}
                fixture.matched.return_value = fixture.contract["matched_notebook_arm"]
            fixture.launch["execution_contract_sha256"] = checkpoint.digest(fixture.contract)
            fixture.launch["binding"] = real_training_binding(fixture.prepared, fixture.protocol, fixture.contract)
            fixture.launch["binding"]["code_sha256"] = "historical"
            with self.subTest(arm=arm), patch.object(local_train, "training_parent_gate", return_value=parent) as gate, \
                    patch.object(local_train, "training_binding", wraps=real_training_binding) as binding:
                result = local_eval.evaluate_ladder(fixture.capture, fixture.cohort, fixture.receipt,
                    fixture.root / ("explore-" + arm), condition="FT", arm=arm, reload_result=fixture.reload,
                    minimal_capture=fixture.root / "minimal", minimal_reload=fixture.root / "minimal-reload",
                    evaluation_prompt_policy=generation.CIRCUIT_PROMPT_POLICY)
            gate.assert_called_once_with(fixture.root / "minimal", fixture.root / "minimal-reload", admission=POLICY)
            self.assertEqual(binding.call_count, 1)
            self.assertEqual(binding.call_args.args[0], fixture.prepared)
            self.assertEqual(binding.call_args.args[1], local_profile.restore_protocol(fixture.protocol))
            self.assertEqual(binding.call_args.args[2], fixture.contract)
            self.assertEqual(fixture.generate.call_args.args[3], templates)
            saved = json.loads((fixture.root / ("explore-" + arm) / "launch.json").read_text())
            self.assertEqual(saved["minimal_parent_gate"], parent)
            self.assertFalse(saved["minimal_parent_gate"]["strict_notebook_correctness_verified"])
            self.assertFalse(saved["minimal_parent_gate"]["relationship_learning_verified"])
            self.assertIsNone(result["accuracy"])
            self.assertEqual(result["training_updates"], 0)
            if arm == "answer-only":
                self.assertEqual(fixture.matched.call_args.kwargs["admission"], POLICY)
                self.assertEqual(fixture.matched.call_args.kwargs["parent"], parent)

    def test_native_missing_or_changed_launch_contract_and_parent_stop_before_model(self):
        fixture, parent, _ = self.native_fixture()
        original = copy.deepcopy(fixture.launch)
        for mutation in ("missing_contract", "missing_digest", "wrong_digest", "changed_parent"):
            fixture.launch.clear()
            fixture.launch.update(copy.deepcopy(original))
            if mutation == "missing_contract":
                del fixture.launch["execution_contract"]
            elif mutation == "missing_digest":
                del fixture.launch["execution_contract_sha256"]
            elif mutation == "wrong_digest":
                fixture.launch["execution_contract_sha256"] = "0" * 64
            fresh = parent | {"changed": True} if mutation == "changed_parent" else parent
            with self.subTest(mutation=mutation), patch.object(local_train, "training_parent_gate", return_value=fresh):
                with self.assertRaises(ValueError):
                    fixture.run_native("rejected-" + mutation)
            fixture.load_f0.assert_not_called()
            fixture.load_ft.assert_not_called()
            fixture.validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
