"""CPU controller fixtures; no real model, update, reload, or improvement evidence."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

from fine_tuning.notebook_rl import checkpoint, local_profile, local_run
from fine_tuning.notebook_rl.schema import ProblemTree
from fine_tuning.tests.test_notebook_rl_schema import records


class RepairLocalRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.build = self.root / "build"
        self.build.mkdir()
        (self.build / "manifest.yaml").write_text(json.dumps({"split_manifests": {"train": "fixture"}}))
        actor = records()[0]
        self.actors = [actor, replace(actor, problem_id=actor.problem_id + ":multi",
            image_paths=("images/a.png", "images/b.png"), image_sha256=("a" * 64, "b" * 64))]
        self.trees = {a.problem_id: ProblemTree(a.problem_id, 0, [], "VISION_OFF", {}, {}) for a in self.actors}
        self.receipts = {a.problem_id: self.receipt(a.problem_id) for a in self.actors}
        self.engine: Any = SimpleNamespace(tokenizer=object())
        self.verifier: Any = SimpleNamespace(metadata={"vault_manifest_sha256": "fixture"})
        self.images = [SimpleNamespace(close=lambda: None)]

    @staticmethod
    def receipt(pid: str, *, admitted: bool = True, signal: bool = True,
                admission: str = "VISION_OFF", reasons: tuple[str, ...] = ()) -> dict:
        value = {"version": "repair-tree-credit-v1", "problem_id": pid, "admitted": admitted,
            "has_signal": signal, "admission": admission, "admission_reasons": list(reasons),
            "signal_components": {"answer": False, "online": signal, "final": False}, "groups": []}
        return value | {"sha256": checkpoint.digest(value)}

    def run_capture(self, *, update: dict | None = None, actual_prepare: bool = False, **options: Any) -> tuple[dict, dict]:
        output = self.root / "capture"
        calls: dict[str, Any] = {}

        def collect(actor, images, engine, verifier, prompts, cfg, version, route, **kwargs):
            calls.setdefault("collect", []).append(kwargs)
            kwargs["on_raw"]({"status": "STARTED", "source": actor.problem_id})
            return self.trees[actor.problem_id]

        def append(trees, images, engine, verifier, prompts, cfg, **kwargs):
            calls.setdefault("append", []).append(kwargs)
            kwargs["on_raw"]({"status": "CLOSED", "source": trees[0].problem_id})

        def prepare(tree, actor, verifier, tokenizer, **kwargs):
            calls.setdefault("prepare", []).append((tree, actor, verifier, tokenizer, kwargs))
            return copy.deepcopy(self.receipts[tree.problem_id])

        def evidence(value, protocol, **kwargs):
            calls["evidence"] = kwargs
            if "retained_repair_signal" not in value:
                raise ValueError("actual retained repair update/source receipts required")
            return {"S_gradient_verified": value["repair_aggregate_gradient"]["S"], "repair_learning_verified": False}

        module = ModuleType("fine_tuning.notebook_rl.repair_evidence")
        module.repair_update_evidence = evidence  # type: ignore[attr-defined]
        bundle = SimpleNamespace(inventory={"quantization": {"type": "fixture"}, "trainable": []}, base_sha256="frozen")
        before = {"L": {"l": "old"}, "M": {"m": "old"}, "V": {"v": "frozen"}}
        after = {"L": {"l": "new"}, "M": {"m": "new"}, "V": {"v": "frozen"}}
        returns = {"local_run.read_preparation": ({"build_root": str(self.build)}, {}, self.actors),
            "local_run._binding": {}, "local_run._install_actor_guards": [],
            "local_run.load_images": self.images, "local_run.make_probes": [],
            "local_calibrate.verify_original": {}, "quantized_policy.build_quantized_policy": bundle,
            "shared_policy_engine.SharedPolicyRolloutEngine": self.engine,
            "checkpoint.make_optimizer": (object(), object()), "checkpoint.capture_rng": {},
            "verifier_service.VerifierService": nullcontext(self.verifier), "trainer.synchronize": {},
            "local_run.probe_evidence": []}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {module.__name__: module}))
            calls["journal"] = stack.enter_context(patch("fine_tuning.notebook_rl.repair_replay.validate_journal"))
            for target, value in returns.items():
                stack.enter_context(patch("fine_tuning.notebook_rl." + target, return_value=value))
            for target in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch("torch.cuda." + target, return_value=0))
            for target in ("updates.partition_learning_trees", "updates.notebook_objective_scores",
                           "updates.notebook_signal_groups", "local_run.error_relationship_signal",
                           "local_run.assessment_signal"):
                stack.enter_context(patch("fine_tuning.notebook_rl." + target, side_effect=AssertionError("legacy gate")))
            stack.enter_context(patch("fine_tuning.notebook_rl.rollout.build_tree", side_effect=collect))
            stack.enter_context(patch("fine_tuning.notebook_rl.notebook_learning.append_rewarded_notebooks", side_effect=append))
            if not actual_prepare:
                stack.enter_context(patch("fine_tuning.notebook_rl.repair_training.prepare_tree_credit", side_effect=prepare))
            stack.enter_context(patch.object(local_run, "parameter_hashes", side_effect=[before, after]))
            stack.enter_context(patch("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged"))
            calls["save"] = stack.enter_context(patch.object(checkpoint, "save"))
            calls["optimizer"] = stack.enter_context(patch("fine_tuning.notebook_rl.updates.optimizer_step",
                return_value=update or {"status": "NO_MATCHED_ACTIONS"}))
            stack.enter_context(patch("builtins.print"))
            result = local_run.capture(self.root / "preparation", self.root / "model", output,
                standalone=False, repair_notebook=True, context_limit=24576, **options)
        return result, calls

    def test_cli_explicit_repair_prepare_minimal_and_legacy_omission(self) -> None:
        prepare = ["prepare", "--previous-request", "/private/request", "--selection", "/private/selection",
                   "--output", "/private/unused"]
        minimal = ["minimal", "--preparation", "/private/preparation", "--model-path", "/private/model",
                   "--output", "/private/unused"]
        flags = ["--repair-notebook", "--context-limit", "24576"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                return_value={"status": "LOCAL_DATA_REVALIDATED"}) as called, patch("builtins.print"):
            self.assertEqual(local_run.main(prepare + flags), 0)
            self.assertEqual(called.call_args.args[2], local_profile.build_protocol(repair_notebook=True, context_limit=24576))
        for extra in (flags, []):
            with patch.object(local_run, "capture") as called:
                self.assertEqual(local_run.main(minimal + extra), 0)
            if extra:
                self.assertIs(called.call_args.kwargs["repair_notebook"], True)
            else:
                self.assertNotIn("repair_notebook", called.call_args.kwargs)
        for command in (prepare, minimal):
            changes = [["--repair-notebook"], flags + ["--pair-draft-synthesis"],
                       flags + ["--whole-reference-notebook"], flags + ["--verified-note-assessment-credit"],
                       flags + ["--verified-note-fact-credit"]]
            if command is minimal:
                changes.append(flags + ["--execution-policy", local_run.VERIFIED_UPDATE_EXPLORATION])
            for extra in changes:
                with self.subTest(stage=command[0], extra=extra), patch("sys.stderr"), \
                        patch.object(Path, "mkdir") as mkdir, self.assertRaises(SystemExit):
                    local_run.main(command + extra)
                mkdir.assert_not_called()

    def test_direct_configuration_rejected_before_output_or_model(self) -> None:
        for options in ({"context_limit": 16384}, {"verified_note_fact_credit": True},
                        {"pair_draft_synthesis": True}, {"repair_notebook": 1},
                        {"execution_policy": local_run.VERIFIED_UPDATE_EXPLORATION}):
            with self.subTest(options=options), patch.object(local_run, "read_preparation") as read, \
                    patch.object(Path, "mkdir") as mkdir, \
                    patch("fine_tuning.notebook_rl.quantized_policy.build_quantized_policy") as model, \
                    patch("torch.cuda.reset_peak_memory_stats") as cuda, self.assertRaises(ValueError):
                local_run.capture(self.root / "preparation", self.root / "model", self.root / "capture",
                    standalone=False, **({"repair_notebook": True, "context_limit": 24576} | options))
            read.assert_not_called()
            mkdir.assert_not_called()
            model.assert_not_called()
            cuda.assert_not_called()

    def test_online_only_eligibility_uses_raw_hooks_fresh_receipts_and_preserves_links(self) -> None:
        result, calls = self.run_capture()
        self.assertEqual(result["status"], "NO_MATCHED_ACTIONS")
        self.assertEqual(result["notebook_signal_groups"], 0)
        self.assertFalse(result["repair_functional_verified"])
        self.assertFalse(result["S_gradient_verified"])
        self.assertEqual(len(calls["prepare"]), 2)
        optimizer = calls["optimizer"]
        self.assertEqual(optimizer.call_args.args[3], list(self.trees.values()))
        self.assertEqual(optimizer.call_args.kwargs["observation_actors"], {a.problem_id: a for a in self.actors})
        self.assertIs(optimizer.call_args.kwargs["objective_verifier"], self.verifier)
        self.assertIs(optimizer.call_args.kwargs["require_retained_learning_signal"], True)
        for kwargs in calls["append"]:
            self.assertIn("on_raw", kwargs)
            self.assertFalse({"on_observation", "on_draft", "on_page"} & kwargs.keys())
        for tree, actor, verifier, tokenizer, kwargs in calls["prepare"]:
            self.assertIs(tree, self.trees[actor.problem_id])
            self.assertIs(verifier, self.verifier)
            self.assertIs(tokenizer, self.engine.tokenizer)
            self.assertIs(kwargs["preparation"], self.engine)
            self.assertEqual(kwargs["master_seed"], 3407)
        output = self.root / "capture"
        rows = [json.loads(path.read_text()) for path in sorted(output.glob("raw-*.json"))]
        self.assertEqual([row["kind"] for row in rows], ["REPAIR_RAW", "REPAIR_RAW", "TREE"] * 2)
        index = json.loads((output / "repair_preparation_index.json").read_text())
        self.assertEqual(len(index["trees"]), 2)
        self.assertEqual([row["kind"] for row in index["raw_records"]], [row["kind"] for row in rows])
        for link in index["raw_records"]:
            self.assertEqual(checkpoint.file_hash(output / link["file"]), link["sha256"])
        self.assertEqual(result["repair_preparation_index_sha256"], checkpoint.file_hash(output / "repair_preparation_index.json"))
        for link in index["trees"]:
            self.assertEqual(checkpoint.file_hash(output / link["tree_file"]), link["tree_file_sha256"])
            self.assertEqual(checkpoint.file_hash(output / link["preparation_file"]), link["preparation_file_sha256"])
            self.assertEqual(json.loads((output / link["preparation_file"]).read_text())["sha256"], link["preparation_receipt_sha256"])
        calls["save"].assert_not_called()

    def test_actual_authored_online_no_success_preparation_reaches_optimizer(self) -> None:
        from fine_tuning.tests.test_notebook_rl_repair_training import RepairTrainingTests

        fixture = RepairTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.fixture.cfg = replace(fixture.fixture.cfg, master_seed=3407)
        tree, engine = fixture.online()
        fixture.attach(tree, engine)
        self.actors, self.trees, self.engine, self.verifier = [fixture.actor], {tree.problem_id: tree}, engine, fixture.verifier
        self.images = fixture.images
        self.verifier.metadata = {"vault_manifest_sha256": "fixture"}
        with patch.object(local_run, "minimal_actors", return_value=self.actors):
            result, calls = self.run_capture(actual_prepare=True)
        self.assertEqual(result["status"], "NO_MATCHED_ACTIONS")
        receipt = json.loads((self.root / "capture" / "repair-preparation-0000.json").read_text())
        self.assertTrue(receipt["admitted"])
        self.assertEqual(receipt["signal_components"], {"answer": False, "online": True, "final": False})
        self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "NO_CORRECT_ANSWER")
        calls["optimizer"].assert_called_once()

    def test_unresolved_incomplete_and_signal_free_keep_raw_without_update(self) -> None:
        first, second = [actor.problem_id for actor in self.actors]
        self.receipts[first] = self.receipt(first, admitted=False, signal=False, admission="UNRESOLVED",
                                          reasons=("UNRESOLVED_ROUTING", "INCOMPLETE_FINAL_COLLECTION"))
        self.receipts[second] = self.receipt(second, signal=False)
        result, calls = self.run_capture()
        self.assertEqual(result["status"], "UNRESOLVED_ROUTING")
        self.assertEqual(result["unresolved_ids"], [first])
        self.assertEqual(result["incomplete_ids"], [first])
        self.assertEqual(result["signal_free_ids"], [second])
        self.assertEqual(result["discarded_ids"], [first, second])
        self.assertEqual(len(list((self.root / "capture").glob("raw-*.json"))), 6)
        self.assertTrue((self.root / "capture" / "repair_preparation_index.json").is_file())
        calls["optimizer"].assert_not_called()

    def test_disk_low_stops_before_the_checkpoint_write_and_records_the_launch_host(self) -> None:
        update = {"status": "UPDATED_UNSYNCHRONIZED", "step_after": 1,
            "routes": {actor.problem_id: False for actor in self.actors}, "grad_norm": {"L": 1., "M": 1., "V": None},
            "repair_preparation": self.receipts, "repair_learning_verified": False,
            "retained_repair_signal": {"required": True, "policy_gradient_present": True, "nonzero_policy_gradient_tokens": 1},
            "repair_aggregate_gradient": {"S": True, "benefit_verified": False}}
        with patch("shutil.disk_usage", return_value=SimpleNamespace(total=10, used=9, free=1)):
            result, calls = self.run_capture(update=update, min_free_gib=1.0)
        self.assertEqual(result["status"], "DISK_LOW")
        self.assertEqual(result["disk_low"]["free_bytes"], 1)
        calls["save"].assert_not_called()
        output = self.root / "capture"
        self.assertFalse((output / "reload_expected.json").exists())
        self.assertEqual(json.loads((output / "result.json").read_text())["status"], "DISK_LOW")
        self.assertEqual(json.loads((output / "launch.json").read_text())["host"], local_run.host_identity())

    def test_save_receipts_keep_functional_false_and_separate_gradient_S(self) -> None:
        update = {"status": "UPDATED_UNSYNCHRONIZED", "step_after": 1,
            "routes": {actor.problem_id: False for actor in self.actors}, "grad_norm": {"L": 1., "M": 1., "V": None},
            "repair_preparation": self.receipts, "repair_learning_verified": False,
            "retained_repair_signal": {"required": True, "policy_gradient_present": True, "nonzero_policy_gradient_tokens": 1},
            "repair_aggregate_gradient": {"S": True, "benefit_verified": False}}
        result, calls = self.run_capture(update=update)
        self.assertEqual(result["status"], "UPDATED_SAVED_PENDING_FRESH_RELOAD")
        self.assertEqual(result["optimizer_updates"], 1)
        self.assertTrue(result["S_gradient_verified"])
        self.assertFalse(result["repair_functional_verified"])
        self.assertFalse(result["repair_learning_verified"])
        self.assertFalse(result["notebook_gradient_contribution_verified"])
        self.assertNotIn("notebook_combined_objective_gradient_observed", result)
        calls["save"].assert_called_once()
        output = self.root / "capture"
        evidence = json.loads((output / "repair_update_evidence.json").read_text())
        expected = json.loads((output / "reload_expected.json").read_text())
        self.assertEqual(evidence["update_file_sha256"], checkpoint.file_hash(output / "update.json"))
        self.assertEqual(evidence["preparation_index_sha256"], result["repair_preparation_index_sha256"])
        self.assertEqual(expected["repair_update_evidence_sha256"], checkpoint.file_hash(output / "repair_update_evidence.json"))
        self.assertEqual(expected["repair_protocol_sha256"], local_profile.build_protocol(repair_notebook=True, context_limit=24576)["sha256"])

    def closed_reload(self) -> tuple[Path, dict, dict]:
        capture = self.root / "closed"
        capture.mkdir()
        protocol = local_profile.build_protocol(repair_notebook=True, context_limit=24576)
        binding = {"code_sha256": checkpoint.digest({})}
        values = {"launch.json": {"pid": 123, "protocol": protocol, "binding": binding,
                "preparation": "/private/preparation", "model_path": "/private/model", "standalone_base": False,
                "problem_ids": [actor.problem_id for actor in self.actors]},
            "result.json": {"status": "UPDATED_SAVED_PENDING_FRESH_RELOAD", "optimizer_updates": 1,
                "discarded_ids": [], "attempted_ids": [actor.problem_id for actor in self.actors]},
            "reload_expected.json": {"trainable_sha256": {}, "step": 1, "frozen_base_sha256": "frozen",
                "quantization": {}, "forward_probes": {"fixture": "forward"}}, "repair_update_evidence.json": {}}
        for name, value in values.items():
            (capture / name).write_text(json.dumps(value))
        return capture, protocol, binding

    def reload_patches(self, stack: ExitStack, capture: Path, binding: dict, *, signal: bool = False) -> dict:
        paths = {path.name: checkpoint.file_hash(path) for path in capture.iterdir()}
        report = {"sha256": "r" * 64, "S_gradient_verified": signal, "paths": paths,
                  "repair_functional_verified": False}
        stack.enter_context(patch("psutil.pid_exists", return_value=False))
        returns = {"local_run.read_preparation": ({"build_root": str(self.build), "image_root": str(self.build / "images")}, {}, self.actors),
            "local_run.minimum_binding": binding, "local_run._install_actor_guards": [],
            "approval.code_hashes": {}, "checkpoint.read_metadata": {"inventory": {"trainable": []}},
            "repair_replay.replay_capture": report, "local_run.parameter_hashes": {},
            "local_run.load_images": [SimpleNamespace(close=lambda: None)], "local_run.make_probes": [],
            "shared_policy_engine.SharedPolicyRolloutEngine": self.engine, "trainer.synchronize": {},
            "local_run.probe_evidence": {"fixture": "forward"}}
        for target, value in returns.items():
            stack.enter_context(patch("fine_tuning.notebook_rl." + target, return_value=value))
        stack.enter_context(patch("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged"))
        stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
        return report

    def test_reload_rejects_independent_replay_failure_before_output_or_model(self) -> None:
        capture, _, binding = self.closed_reload()
        with ExitStack() as stack:
            self.reload_patches(stack, capture, binding)
            stack.enter_context(patch("fine_tuning.notebook_rl.repair_replay.replay_capture",
                side_effect=ValueError("independent original source replay differs")))
            mkdir = stack.enter_context(patch.object(Path, "mkdir"))
            load = stack.enter_context(patch.object(checkpoint, "load_resume"))
            with self.assertRaisesRegex(ValueError, "independent original source replay differs"):
                local_run.reload_capture(capture, self.root / "reload")
        mkdir.assert_not_called()
        load.assert_not_called()

    def test_only_completed_fresh_restore_and_probes_certify_F_with_independent_S(self) -> None:
        capture, protocol, binding = self.closed_reload()
        for signal in (False, True):
            with self.subTest(signal=signal), ExitStack() as stack:
                report = self.reload_patches(stack, capture, binding, signal=signal)
                bundle = SimpleNamespace(inventory={"quantization": {}}, base_sha256="frozen")
                cursor = {"epoch": 0, "position": 2, "discarded_problem_ids": [],
                          "problem_ids": [actor.problem_id for actor in self.actors], "sync_pending": False}
                load = stack.enter_context(patch.object(checkpoint, "load_resume", return_value=(bundle,
                    SimpleNamespace(state={"fixture": True}), SimpleNamespace(last_epoch=1), 1, cursor)))
                result = local_run.reload_capture(capture, self.root / f"reload-{signal}")
            load.assert_called_once()
            self.assertTrue(result["repair_functional_verified"])
            self.assertTrue(result["fresh_reload_verified"])
            self.assertIs(result["S_gradient_verified"], signal)
            self.assertIs(result["repair_learning_verified"], signal)
            self.assertFalse(result["benefit_verified"])
            self.assertEqual(result["capture_pid"], 123)
            self.assertNotEqual(result["pid"], result["capture_pid"])
            self.assertEqual(result["repair_replay_sha256"], report["sha256"])
            self.assertEqual(result["repair_protocol_sha256"], protocol["sha256"])
            self.assertEqual(result["capture_binding_sha256"], checkpoint.digest(binding))

    def test_reloaded_source_mutation_never_certifies_F(self) -> None:
        capture, _, binding = self.closed_reload()
        with ExitStack() as stack:
            self.reload_patches(stack, capture, binding, signal=True)
            bundle = SimpleNamespace(inventory={"quantization": {}}, base_sha256="frozen")
            cursor = {"epoch": 0, "position": 2, "discarded_problem_ids": [],
                      "problem_ids": [actor.problem_id for actor in self.actors], "sync_pending": False}
            stack.enter_context(patch.object(checkpoint, "load_resume", return_value=(bundle,
                SimpleNamespace(state={"fixture": True}), SimpleNamespace(last_epoch=1), 1, cursor)))

            def changed(*args):
                (capture / "repair_update_evidence.json").write_text('{"changed":true}')
                return {"fixture": "forward"}

            stack.enter_context(patch.object(local_run, "probe_evidence", side_effect=changed))
            with self.assertRaisesRegex(ValueError, "original repair evidence changed"):
                local_run.reload_capture(capture, self.root / "reload")
        self.assertFalse((self.root / "reload" / "result.json").exists())

    def test_replay_cannot_replace_cached_reload_records_with_rehashed_versions(self) -> None:
        capture, _, binding = self.closed_reload()
        for name in ("launch.json", "result.json", "reload_expected.json"):
            raw = (capture / name).read_bytes()
            with self.subTest(name=name), ExitStack() as stack:
                report = self.reload_patches(stack, capture, binding, signal=True)

                def changed(*args, **kwargs):
                    value = json.loads(raw)
                    value["changed_after_cached_read"] = True
                    if name == "reload_expected.json":
                        value["forward_probes"] = {"changed": "forward"}
                    (capture / name).write_text(json.dumps(value))
                    report["paths"][name] = checkpoint.file_hash(capture / name)
                    return report

                stack.enter_context(patch("fine_tuning.notebook_rl.repair_replay.replay_capture", side_effect=changed))
                load = stack.enter_context(patch.object(checkpoint, "load_resume"))
                mkdir = stack.enter_context(patch.object(Path, "mkdir"))
                with self.assertRaisesRegex(ValueError, "cached repair records differ"):
                    local_run.reload_capture(capture, self.root / "reload")
            load.assert_not_called()
            mkdir.assert_not_called()
            (capture / name).write_bytes(raw)

    def test_update_without_retained_source_receipt_fails_with_durable_raw(self) -> None:
        update = {"status": "UPDATED_UNSYNCHRONIZED", "repair_learning_verified": False,
                  "repair_aggregate_gradient": {"S": True}}
        with self.assertRaisesRegex(ValueError, "actual retained repair update/source receipts"):
            self.run_capture(update=update)
        output = self.root / "capture"
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(result["repair_functional_verified"])
        self.assertFalse(result["S_gradient_verified"])
        self.assertEqual(json.loads((output / "update.json").read_text()), update)
        self.assertEqual(len(list(output.glob("raw-*.json"))), 6)
        self.assertTrue((output / "repair_preparation_index.json").is_file())


if __name__ == "__main__":
    unittest.main()
