"""CPU controller fixtures and tiny updates; no pretrained learning/reload claim."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import checkpoint, local_profile, local_run, local_train
from fine_tuning.notebook_rl.local_data import ACTOR_MODE_POLICY
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_retained_signal as retained
from fine_tuning.tests import test_notebook_rl_synthesis_credit_admission as synthesis
from fine_tuning.tests.test_notebook_rl_schema import records


POLICY = local_run.VERIFIED_UPDATE_EXPLORATION


class VerifiedUpdatePolicyTests(unittest.TestCase):
    def protocol(self):
        return local_profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)

    def tiny_update(self):
        fixture = retained.RetainedSignalTests()
        values = fixture.fixture()
        before = local_run.parameter_hashes(values[0])
        update = fixture.run_step(values, require_retained_learning_signal=True)
        after = local_run.parameter_hashes(values[0])
        self.assertEqual(update["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual(values[4].last_epoch, 1)
        return update, before, after

    def test_policy_requires_exact_current_profile_and_changes_only_execution_binding(self):
        protocol = self.protocol()
        contract = local_run.minimum_execution_contract(POLICY, protocol)
        assert contract is not None
        self.assertEqual(contract, {"version": "minimum-execution-contract-v1", "policy": POLICY,
            "stage": "MINIMAL_TWO_PROBLEM_ONE_UPDATE", "max_optimizer_updates": 1,
            "require_retained_learning_signal": True})
        with patch.object(local_run.approval, "code_hashes", return_value={"fixture": "source"}):
            old = local_run._binding({"input": "fixed"}, protocol)
            self.assertEqual(local_run.minimum_binding({"input": "fixed"}, protocol), old)
            self.assertEqual(local_run.minimum_binding({"input": "fixed"}, protocol, None), old)
            new = local_run.minimum_binding({"input": "fixed"}, protocol, contract)
            self.assertNotEqual(old["data_sha256"], new["data_sha256"])
            self.assertEqual({k: v for k, v in old.items() if k != "data_sha256"},
                             {k: v for k, v in new.items() if k != "data_sha256"})
            for changed in (contract | {"max_optimizer_updates": 2},
                            contract | {"require_retained_learning_signal": False}, contract | {"extra": True}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    local_run.minimum_binding({}, protocol, changed)
        self.assertIsNone(local_run.minimum_execution_contract(None, {}))
        self.assertEqual(local_train.execution_admission({"admission_policy": POLICY}, protocol), POLICY)
        for incompatible in (local_profile.build_protocol(),
                local_profile.build_protocol(verified_note_synthesis=True, context_limit=24576),
                local_profile.build_protocol(verified_note_fact_credit=True, context_limit=24576)):
            with self.subTest(profile=incompatible["protocol_id"]), self.assertRaises(ValueError):
                local_run.minimum_execution_contract(POLICY, incompatible)
        with self.assertRaises(ValueError):
            local_run.minimum_execution_contract(POLICY + "-forged", protocol)

    def test_retained_evidence_accepts_actual_tiny_answer_update_and_rejects_forgery(self):
        update, before, after = self.tiny_update()
        self.assertTrue(local_run.verify_changes(before, after, update)["vision_off_only_observed"])
        self.assertEqual(local_run.retained_update_signal(update, self.protocol()), update["retained_learning_signal"])
        for mutation in ("missing", "false", "zero", "bool", "count", "duplicate", "unknown", "mis", "basis", "nan"):
            changed = copy.deepcopy(update)
            signal = changed["retained_learning_signal"]
            if mutation == "missing":
                del changed["retained_learning_signal"]
            elif mutation == "false":
                signal["policy_gradient_present"] = False
            elif mutation in ("zero", "bool"):
                signal["nonzero_policy_gradient_tokens"] = 0 if mutation == "zero" else True
            elif mutation == "count":
                signal["retained_action_count"] += 1
            elif mutation == "duplicate":
                signal["signal_sample_ids"] *= 2
            elif mutation == "unknown":
                signal["signal_sample_ids"] = ["absent"]
            elif mutation == "mis":
                changed["mean_log_ratio_by_sample"][signal["signal_sample_ids"][0]] = 99.
            elif mutation == "basis":
                signal["basis"] = "KL-only"
            else:
                changed["mean_log_ratio_by_sample"][signal["signal_sample_ids"][0]] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                local_run.retained_update_signal(changed, self.protocol())

    def test_capture_fact_signal_reaches_only_opt_in_updater_and_keeps_zero_update_failure(self):
        fixture = synthesis.partial_fixture(facts_only=True)
        self.addCleanup(fixture.doCleanups)
        self.assertFalse(local_run.error_relationship_signal([fixture.tree], fixture.protocol["notebook_objective"],
                                                             fixture.engine.tokenizer))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        build = root / "build"
        build.mkdir()
        (build / "manifest.yaml").write_text(json.dumps({"split_manifests": {"train": "fixture"}}))
        fixture.verifier.metadata = {"vault_manifest_sha256": "fixture"}
        returns = {
            "fine_tuning.notebook_rl.local_run.read_preparation": ({"build_root": str(build)}, {}, [fixture.actor]),
            "fine_tuning.notebook_rl.local_run.minimal_actors": [fixture.actor],
            "fine_tuning.notebook_rl.local_run._binding": {},
            "fine_tuning.notebook_rl.local_run._install_actor_guards": [],
            "fine_tuning.notebook_rl.local_run.load_images": fixture.images[fixture.actor.problem_id],
            "fine_tuning.notebook_rl.local_run.make_probes": [],
            "fine_tuning.notebook_rl.local_run.parameter_hashes": {},
            "fine_tuning.notebook_rl.local_calibrate.verify_original": {},
            "fine_tuning.notebook_rl.quantized_policy.build_quantized_policy": SimpleNamespace(inventory={"fixture_only": True}),
            "fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine": fixture.engine,
            "fine_tuning.notebook_rl.checkpoint.make_optimizer": (object(), object()),
            "fine_tuning.notebook_rl.trainer.synchronize": {},
            "fine_tuning.notebook_rl.rollout.build_tree": fixture.tree,
            "fine_tuning.notebook_rl.notebook_learning.append_rewarded_notebooks": None,
            "torch.cuda.reset_peak_memory_stats": None,
            "torch.cuda.max_memory_allocated": 0,
            "torch.cuda.max_memory_reserved": 0,
        }
        for index, (policy, eligible) in enumerate(((None, True), (POLICY, True), (POLICY, False))):
            fixture.tree.notebook.pop("learning_objective", None)
            output = root / str(index)
            with ExitStack() as stack:
                for target, value in returns.items():
                    stack.enter_context(patch(target, return_value=value))
                stack.enter_context(patch("fine_tuning.notebook_rl.verifier_service.VerifierService",
                                           return_value=nullcontext(fixture.verifier)))
                stack.enter_context(patch("builtins.print"))
                if not eligible:
                    stack.enter_context(patch("fine_tuning.notebook_rl.updates.partition_learning_trees",
                                               return_value=([], [], [fixture.actor.problem_id])))
                optimizer = stack.enter_context(patch("fine_tuning.notebook_rl.updates.optimizer_step",
                    return_value={"status": "NO_RETAINED_LEARNING_SIGNAL", "step_before": 0, "step_after": 0}))
                result = local_run.capture(root / "preparation", root / "model", output, standalone=False,
                    verified_note_synthesis_credit=True, context_limit=24576, execution_policy=policy)
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertFalse(result["notebook_gradient_contribution_verified"])
            self.assertFalse((output / "checkpoint").exists())
            self.assertFalse((output / "reload_expected.json").exists())
            if policy is None:
                self.assertEqual(result["status"], "NO_ERROR_RELATIONSHIP_LEARNING_SIGNAL")
                optimizer.assert_not_called()
                self.assertNotIn("execution_contract", json.loads((output / "launch.json").read_text()))
            elif eligible:
                self.assertEqual(result["status"], "NO_RETAINED_LEARNING_SIGNAL")
                self.assertIs(optimizer.call_args.kwargs["require_retained_learning_signal"], True)
            else:
                self.assertEqual(result["status"], "NO_LEARNING_SIGNAL")
                optimizer.assert_not_called()

    def admission_fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        capture, reload = root / "capture", root / "reload"

        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return {"path": str(path), "file_sha256": checkpoint.file_hash(path)}

        protocol = self.protocol()
        actor = records()[0]
        actors = [actor, replace(actor, problem_id="fake:pair:train:2", image_paths=("images/a.png", "images/b.png"),
                                 image_sha256=("a" * 64, "b" * 64))]
        build = root / "build"
        save(build / "manifest.yaml", {})
        prepared = {"format": "notebook-rl-local-data-v1", "status": "LOCAL_DATA_REVALIDATED",
            "local_protocol_sha256": protocol["sha256"], "build_root": str(build),
            "actor_projection": {"policy": ACTOR_MODE_POLICY, "changed_ids": []},
            "actors": save(root / "actors.json", [json.loads(to_json(a)) for a in actors]),
            "execution_actors": save(root / "execution.json", [json.loads(to_json(a)) for a in actors]),
            "data_request": save(root / "request.json", {"build_root": str(build)}),
            "validated_data": save(root / "checked.json", {"status": "VALIDATED_FOR_MEASUREMENT", "blockers": [],
                "measurement_allowed": True, "training_allowed": False,
                "code_sha256": local_train.approval.code_hashes(),
                "input_binding": {"build_manifest_sha256": checkpoint.file_hash(build / "manifest.yaml")},
                "actors": [json.loads(to_json(a)) for a in actors]})}
        preparation = root / "preparation.json"
        save(preparation, prepared)
        contract = local_run.minimum_execution_contract(POLICY, protocol)
        launch = {"protocol": json.loads(json.dumps(protocol)), "standalone_base": True, "pid": 111222,
            "purpose": "MINIMAL_TWO_PROBLEM_ONE_UPDATE", "problem_ids": [a.problem_id for a in actors],
            "preparation": str(preparation), "binding": local_run.minimum_binding(prepared, protocol, contract),
            "execution_contract": contract, "execution_contract_sha256": checkpoint.digest(contract)}
        update, before, after = self.tiny_update()
        result = {"status": "UPDATED_SAVED_PENDING_FRESH_RELOAD", "optimizer_updates": 1,
            "original_access_guard": {"denied_original_access_attempts": 0},
            "retained_learning_signal": update["retained_learning_signal"], "notebook_signal_groups": 0,
            "notebook_gradient_contribution_verified": False, **local_run.verify_changes(before, after, update)}
        restored = {"status": "FRESH_PROCESS_RELOAD_VERIFIED", "capture_pid": launch["pid"], "pid": 111333, "step": 1,
            "original_access_guard": {"denied_original_access_attempts": 0}, "execution_contract": contract,
            "execution_contract_sha256": checkpoint.digest(contract), "capture_binding_sha256": checkpoint.digest(launch["binding"]),
            **{key: True for key in ("all_trainable_hashes_match", "optimizer_restored",
                                     "actual_forward_and_generation_match", "frozen_quantized_base_unchanged")}}
        expected = {"step": 1, "trainable_sha256": after, "frozen_base_sha256": "fixture-base",
                    "quantization": {}, "forward_probes": {}}
        metadata = {"step": 1, "state_sha256": {n: h for values in after.values() for n, h in values.items()},
                    "base_sha256": "fixture-base", "inventory": {"quantization": {}}}
        rows = {capture / "launch.json": launch, capture / "result.json": result, capture / "update.json": update,
            capture / "parameter_changes.json": {"before": before, "after": after},
            capture / "reload_expected.json": expected, capture / "updated_forward_probes.json": {},
            reload / "result.json": restored, reload / "forward_probes.json": {}, reload / "sync.json": {}}
        for path, value in rows.items():
            save(path, value)
        for name in ("metadata.json", "metadata.sha256", "adapter.safetensors", "mergers.safetensors", "runtime.pt"):
            save(capture / "checkpoint" / name, {"fixture_only": True})
        return capture, reload, rows, metadata, save

    def test_actual_tiny_answer_update_can_admit_without_notebook_certification(self):
        capture, reload, _, metadata, _ = self.admission_fixture()
        with patch("psutil.pid_exists", return_value=False), patch.object(checkpoint, "read_metadata", return_value=metadata), \
                patch.object(local_train, "notebook_learning_gate", side_effect=AssertionError("no notebook certification")):
            gate = local_train.training_parent_gate(capture, reload, admission=POLICY)
        self.assertTrue(gate["verified_update_exploration_admitted"])
        self.assertTrue(gate["retained_policy_coefficient_signal_verified"])
        self.assertFalse(gate["notebook_reward_learning_verified"])
        self.assertFalse(gate["strict_notebook_correctness_verified"])

    def test_historical_minimum_source_is_derived_from_bound_validation_and_proof_stays_stable(self):
        capture, reload, rows, metadata, _ = self.admission_fixture()
        binding = rows[capture / "launch.json"]["binding"]
        with patch("psutil.pid_exists", return_value=False), \
                patch.object(checkpoint, "read_metadata", return_value=metadata) as checked:
            proofs = []
            for current_sources in ({"current.py": "1" * 64}, {"newer.py": "2" * 64}):
                self.assertNotEqual(checkpoint.digest(current_sources), binding["code_sha256"])
                with patch.object(local_train.approval, "code_hashes", return_value=current_sources):
                    proofs.append(local_train.training_parent_gate(capture, reload, admission=POLICY))
                checked.assert_called_with(capture / "checkpoint", binding=binding)
        self.assertEqual(proofs[0], proofs[1])
        self.assertTrue(proofs[0]["verified_update_exploration_admitted"])
        self.assertFalse(proofs[0]["notebook_reward_learning_verified"])

    def test_historical_minimum_rejects_changed_source_mapping_and_descriptor(self):
        capture, reload, rows, metadata, save = self.admission_fixture()
        launch = rows[capture / "launch.json"]
        preparation = Path(launch["preparation"])
        prepared = json.loads(preparation.read_text())
        validation = Path(prepared["validated_data"]["path"])
        original = json.loads(validation.read_text())
        mutations = ("unbound_bytes", "descriptor", "bound_wrong_map", "empty_map", "scalar", "invalid_hash")
        with patch("psutil.pid_exists", return_value=False), patch.object(checkpoint, "read_metadata", return_value=metadata):
            for mutation in mutations:
                changed = copy.deepcopy(prepared)
                sources = ({"old.py": "f" * 64} if mutation not in ("empty_map", "scalar", "invalid_hash")
                           else {} if mutation == "empty_map" else "not-a-map" if mutation == "scalar"
                           else {"old.py": "invalid"})
                descriptor = save(validation, original | {"code_sha256": sources})
                if mutation == "descriptor":
                    changed["validated_data"]["file_sha256"] = "0" * 64
                elif mutation != "unbound_bytes":
                    changed["validated_data"] = descriptor
                save(preparation, changed)
                changed_binding = local_run.minimum_binding(changed, launch["protocol"], launch["execution_contract"])
                changed_binding["code_sha256"] = launch["binding"]["code_sha256"]
                save(capture / "launch.json", launch | {"binding": changed_binding})
                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                        ValueError, "artifact bytes differ|source mapping required|source/preparation/execution binding differs"):
                    local_train.training_parent_gate(capture, reload, admission=POLICY)
                save(validation, original)
                save(preparation, prepared)
                save(capture / "launch.json", launch)

    def test_current_preparation_still_rejects_stale_sources_before_validation(self):
        capture, _, rows, _, save = self.admission_fixture()
        launch = rows[capture / "launch.json"]
        preparation = Path(launch["preparation"])
        prepared = json.loads(preparation.read_text())
        save(preparation, prepared | {"profile": local_profile.PROFILE, "cloud_allowed": False,
                                     "image_root": str(Path(prepared["build_root"]) / "images")})
        with patch.object(local_run.approval, "code_hashes", return_value={"current.py": "3" * 64}), \
                patch.object(local_run.approval, "validate_for_gate_data") as validation:
            with self.assertRaisesRegex(ValueError, "stale source or incomplete input validation"):
                local_run.read_preparation(preparation, launch["protocol"])
            validation.assert_not_called()

    def test_new_admission_refuses_live_zero_incomplete_mutated_and_mismatched_reload(self):
        capture, reload, rows, metadata, save = self.admission_fixture()
        mutations: tuple[tuple[Path, str, object], ...] = ((capture / "result.json", "optimizer_updates", 0),
            (capture / "result.json", "status", "INCOMPLETE"),
            (reload / "result.json", "optimizer_restored", False),
            (reload / "result.json", "capture_binding_sha256", "0" * 64),
            (reload / "result.json", "execution_contract", {}),
            (capture / "launch.json", "execution_contract_sha256", "0" * 64),
            (capture / "launch.json", "execution_contract", None),
            (capture / "launch.json", "binding", rows[capture / "launch.json"]["binding"] | {"code_sha256": "0" * 64}),
            (capture / "launch.json", "problem_ids", list(reversed(rows[capture / "launch.json"]["problem_ids"]))),
            (capture / "update.json", "retained_learning_signal", {}))
        with patch.object(checkpoint, "read_metadata", return_value=metadata):
            for path, key, value in mutations:
                save(path, rows[path] | {key: value})
                with self.subTest(key=key, value=value), patch("psutil.pid_exists", return_value=False), self.assertRaises(ValueError):
                    local_train.training_parent_gate(capture, reload, admission=POLICY)
                save(path, rows[path])
            for live_pid in (111222, 111333):
                with self.subTest(live_pid=live_pid), patch("psutil.pid_exists", side_effect=lambda p: p == live_pid), \
                        self.assertRaises(ValueError):
                    local_train.training_parent_gate(capture, reload, admission=POLICY)


if __name__ == "__main__":
    unittest.main()
