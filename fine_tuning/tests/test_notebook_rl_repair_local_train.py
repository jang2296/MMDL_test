"""Repair admission and scripted fixed-ten integration with real tiny checkpoint payloads."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fine_tuning.notebook_rl import checkpoint, local_train as training
from fine_tuning.notebook_rl.local_profile import build_protocol
from fine_tuning.tests import test_notebook_rl_local_train as legacy
from fine_tuning.tests import test_notebook_rl_repair_replay as replay_fixture
from fine_tuning.tests.test_notebook_rl_repair_replay import probe_fixture


def forward_probes(problem_ids=("a", "b")):
    return probe_fixture(list(problem_ids))


def sync_probes(problem_ids=("a", "b"), step=1):
    return [{"request_id": "local-sync:" + pid, "policy_version": step, "mean_abs_delta": 0., "max_abs_delta": 0.}
            for pid in problem_ids]


class RepairAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = build_protocol(repair_notebook=True, context_limit=24576)
        self.actors = [SimpleNamespace(problem_id=pid, image_paths=["image"] * count)
                       for pid, count in (("a", 1), ("b", 2))]

    def test_only_exact_explicit_repair_admission_accepts_canonical_objective(self) -> None:
        contract = {"admission_policy": training.REPAIR_ADMISSION}
        self.assertEqual(training.execution_admission(contract, self.protocol), training.REPAIR_ADMISSION)
        for policy in (None, training.STRICT_ADMISSION, training.ASSESSMENT_CREDIT_ADMISSION):
            with self.assertRaises(ValueError):
                training.execution_admission({} if policy is None else {"admission_policy": policy}, self.protocol)
        for field in ("repair_objective_sha256", "repair_retry_contract", "sha256"):
            changed = copy.deepcopy(self.protocol)
            changed[field] = "foreign"
            with self.subTest(field=field), self.assertRaises(ValueError):
                training.execution_admission(contract, changed)
        with self.assertRaises(ValueError):
            training.execution_admission(contract, build_protocol())

    def test_functional_policy_is_explicit_canonical_and_hash_distinct(self) -> None:
        policy = training.REPAIR_FUNCTIONAL_EXPLORATION
        self.assertEqual(training.execution_admission({"admission_policy": policy}, self.protocol), policy)
        self.assertNotEqual(checkpoint.digest({"admission_policy": policy}),
                            checkpoint.digest({"admission_policy": training.REPAIR_ADMISSION}))
        for field in ("profile", "repair_objective_sha256", "repair_retry_contract", "sha256"):
            changed = copy.deepcopy(self.protocol)
            changed[field] = "foreign"
            with self.subTest(field=field), self.assertRaises(ValueError):
                training.execution_admission({"admission_policy": policy}, changed)
        changed = copy.deepcopy(self.protocol)
        changed["local_budget"]["seed"] += 1
        changed["sha256"] = checkpoint.digest({key: value for key, value in changed.items() if key != "sha256"})
        with self.assertRaises(ValueError):
            training.execution_admission({"admission_policy": policy}, changed)
        with self.assertRaises(ValueError):
            training.execution_admission({"admission_policy": policy}, build_protocol())

    def test_both_arms_require_same_minimum_s_before_model_loading(self) -> None:
        for arm in ("notebook", "answer-only"):
            options: dict[str, Any] = {} if arm == "notebook" else {"match_initialization": Path("initialization.json"),
                                                                     "matched_reload": Path("matched-reload")}
            with patch.object(training, "training_parent_gate", side_effect=ValueError("minimum S false")) as gate, \
                    patch("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy") as model:
                with self.assertRaisesRegex(ValueError, "minimum S"):
                    training.train(Path("prep"), Path("base"), Path("cache"), Path("minimum"), Path("reload"),
                        Path("output"), arm=arm, admission=training.REPAIR_ADMISSION, **options)
                gate.assert_called_once_with(Path("minimum"), Path("reload"), admission=training.REPAIR_ADMISSION)
                model.assert_not_called()

    def test_control_requires_explicit_matched_reload_before_parent_or_outputs(self) -> None:
        with patch.object(training, "training_parent_gate") as gate:
            with self.assertRaisesRegex(ValueError, "matched reload"):
                training.train(Path("prep"), Path("base"), Path("cache"), Path("minimum"), Path("reload"),
                    Path("output"), arm="answer-only", match_initialization=Path("initialization.json"),
                    admission=training.REPAIR_ADMISSION)
            gate.assert_not_called()

    def test_saved_probe_cardinality_request_identity_and_integer_policy_are_strict(self) -> None:
        sync = sync_probes()
        training._repair_probe_binding(forward_probes(), sync, 1, self.actors)
        for mutation in ("one", "images", "generated", "bool"):
            probes, reports = forward_probes(), copy.deepcopy(sync)
            if mutation == "one":
                probes["inputs"].pop()
            elif mutation == "images":
                probes["inputs"][1]["processed_image_sizes"].pop()
            elif mutation == "generated":
                probes["generated"][0]["completions"].append({})
            else:
                reports[0]["policy_version"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                training._repair_probe_binding(probes, reports, 1, self.actors)

    def test_cli_forwards_only_explicit_repair_admission_and_matched_reload(self) -> None:
        with patch.object(training, "train", return_value={"status": "NO_OPTIMIZER_UPDATES"}) as run, patch("builtins.print"):
            code = training.main(["train", "--preparation", "prep", "--model-path", "base", "--cache-check", "cache",
                "--output", "out", "--arm", "answer-only", "--match-initialization", "initialization.json",
                "--matched-reload", "fresh", "--admission", training.REPAIR_ADMISSION])
        self.assertEqual(code, 2)
        self.assertEqual(run.call_args.kwargs["matched_reload"], Path("fresh"))
        self.assertEqual(run.call_args.kwargs["admission"], training.REPAIR_ADMISSION)

    def test_functional_cli_forwards_exact_policy_to_both_arms(self) -> None:
        for arm in ("notebook", "answer-only"):
            extra = [] if arm == "notebook" else ["--match-initialization", "initialization.json", "--matched-reload", "fresh"]
            with self.subTest(arm=arm), patch.object(training, "train", return_value={"status": "NO_OPTIMIZER_UPDATES"}) as run, \
                    patch("builtins.print"):
                code = training.main(["train", "--preparation", "prep", "--model-path", "base", "--cache-check", "cache",
                    "--output", "out", "--arm", arm, "--admission", training.REPAIR_FUNCTIONAL_EXPLORATION, *extra])
            self.assertEqual(code, 2)
            self.assertEqual(run.call_args.kwargs["admission"], training.REPAIR_FUNCTIONAL_EXPLORATION)

    def test_functional_minimum_replays_f_without_claiming_s_and_retains_integrity_gates(self) -> None:
        from fine_tuning.notebook_rl import repair_replay

        fixture = replay_fixture.RepairReplayTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        actors, _, receipts, prepared, binding, evidence, module = fixture.compose()
        capture, reload = fixture.root, fixture.root / "reload"
        reload.mkdir()
        launch = training._read(capture / "launch.json") | {"standalone_base": True,
            "purpose": "MINIMAL_TWO_PROBLEM_ONE_UPDATE", "preparation": str(capture / "preparation.json")}
        fixture.write(capture, "launch.json", launch)
        result = training._read(capture / "result.json") | {"original_access_guard": {"denied_original_access_attempts": 0}}
        fixture.write(capture, "result.json", result)
        self.enterContext(patch.dict(sys.modules, {module.__name__: module}))
        self.enterContext(patch("fine_tuning.notebook_rl.local_run.minimum_binding", return_value=binding))
        self.enterContext(patch.object(repair_replay, "replay_trees", return_value=receipts))
        inputs = self.enterContext(patch.object(training, "read_preparation", return_value=(prepared, {}, actors)))
        alive = self.enterContext(patch("psutil.pid_exists", return_value=False))
        proof = repair_replay.replay_capture(capture, protocol=fixture.protocol, prepared=prepared,
                                            actors=actors, parameter_inventory=[])
        expected = training._read(capture / "reload_expected.json")
        restored = {"status": "FRESH_PROCESS_RELOAD_VERIFIED", "pid": 124, "capture_pid": 123, "step": 1,
            "original_access_guard": {"denied_original_access_attempts": 0},
            **{key: True for key in ("all_trainable_hashes_match", "optimizer_restored", "actual_forward_and_generation_match",
                                    "frozen_quantized_base_unchanged", "repair_functional_verified")},
            "S_gradient_verified": False, "repair_learning_verified": False, "benefit_verified": False,
            "repair_replay_sha256": proof["sha256"],
            "repair_update_evidence_sha256": checkpoint.file_hash(capture / "repair_update_evidence.json"),
            "capture_binding_sha256": checkpoint.digest(binding), "repair_protocol_sha256": fixture.protocol["sha256"]}
        fixture.write(reload, "result.json", restored)
        fixture.write(reload, "forward_probes.json", expected["forward_probes"])
        fixture.write(reload, "sync.json", sync_probes([actor.problem_id for actor in actors]))

        def gate():
            return training.training_parent_gate(capture, reload, admission=training.REPAIR_FUNCTIONAL_EXPLORATION)

        accepted = gate()
        self.assertTrue(accepted["repair_functional_verified"])
        self.assertFalse(accepted["S_gradient_verified"] or accepted["repair_learning_verified"]
                         or accepted["strict_notebook_correctness_verified"] or accepted["benefit_verified"])
        self.assertFalse(evidence.call_args.kwargs["require_s"])
        self.assertEqual(len([name for name in accepted["repair_replay"]["paths"] if name.startswith("checkpoint/")]), 5)
        with self.assertRaises(ValueError):
            training.training_parent_gate(capture, reload, admission=training.REPAIR_ADMISSION)
        mutations: list[tuple[Path, str, Any]] = [(reload / "result.json", key, value) for key, value in (
            ("repair_functional_verified", False), ("S_gradient_verified", True), ("repair_learning_verified", True),
            ("S_gradient_verified", 0), ("repair_replay_sha256", "foreign"), ("optimizer_restored", False),
            ("actual_forward_and_generation_match", False), ("pid", True))]
        mutations += [(capture / "result.json", "optimizer_updates", 0),
                      (capture / "launch.json", "problem_ids", ["foreign", actors[1].problem_id]),
                      (capture / "launch.json", "binding", binding | {"code_sha256": "c" * 64})]
        for path, key, value in mutations:
            original = path.read_bytes()
            fixture.write(path.parent, path.name, training._read(path) | {key: value})
            with self.subTest(file=path.name, key=key), self.assertRaises(ValueError):
                gate()
            path.write_bytes(original)
        for relative in ("raw-0000.json", "checkpoint/metadata.json", "checkpoint/metadata.sha256",
                         "checkpoint/adapter.safetensors", "checkpoint/mergers.safetensors", "checkpoint/runtime.pt"):
            path = capture / relative
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(file=relative), self.assertRaises(ValueError):
                gate()
            path.write_bytes(original)
        original = (reload / "result.json").read_bytes()
        (reload / "result.json").unlink()
        with self.assertRaises(ValueError):
            gate()
        (reload / "result.json").write_bytes(original)
        alive.return_value = True
        with self.assertRaises(ValueError):
            gate()
        alive.return_value = False
        inputs.return_value = (prepared, {}, [replace(actors[0], problem_id="foreign"), actors[1]])
        with self.assertRaises(ValueError):
            gate()
        inputs.return_value = (prepared, {}, actors)
        evidence.return_value["S_gradient_verified"] = True
        envelope = training._read(capture / "repair_update_evidence.json")
        envelope["validation"] = evidence.return_value
        evidence_sha = fixture.write(capture, "repair_update_evidence.json", envelope)
        fixture.write(capture, "result.json", result | {"S_gradient_verified": True,
            "repair_update_evidence_sha256": evidence_sha})
        fixture.write(capture, "reload_expected.json", expected | {"repair_update_evidence_sha256": evidence_sha})
        observed = repair_replay.replay_capture(capture, protocol=fixture.protocol, prepared=prepared,
                                               actors=actors, parameter_inventory=[])
        fixture.write(reload, "result.json", restored | {"S_gradient_verified": True, "repair_learning_verified": True,
            "repair_update_evidence_sha256": evidence_sha, "repair_replay_sha256": observed["sha256"]})
        self.assertTrue(gate()["S_gradient_verified"])
        self.assertTrue(training.training_parent_gate(capture, reload, admission=training.REPAIR_ADMISSION)["S_gradient_verified"])
        self.assertTrue(evidence.call_args.kwargs["require_s"])

    def test_minimum_gate_replays_s_and_rejects_bool_pid_count_or_changed_reload_binding(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        capture, reload = root / "capture", root / "reload"
        capture.mkdir()
        reload.mkdir()
        evidence = capture / "repair_update_evidence.json"
        evidence.write_text("{}")
        launch = {"protocol": self.protocol, "pid": 10001, "problem_ids": ["a", "b"],
                  "preparation": str(root / "preparation.json"), "binding": {"actual": "fixture"}}
        original = {"optimizer_updates": 1}
        restored = {"pid": 10002, "capture_pid": 10001, "step": 1, "repair_functional_verified": True,
            "repair_learning_verified": True, "S_gradient_verified": True, "benefit_verified": False,
            "repair_replay_sha256": "replay", "repair_update_evidence_sha256": checkpoint.file_hash(evidence),
            "capture_binding_sha256": checkpoint.digest(launch["binding"]), "repair_protocol_sha256": self.protocol["sha256"]}
        expected = {"step": 1, "forward_probes": forward_probes()}
        replay = {"sha256": "replay", "S_gradient_verified": True, "paths": {}}
        for mutation in (None, "count", "pid", "step", "binding", "S", "parent_mutation", "reload_mutation"):
            result, fresh = copy.deepcopy(original), copy.deepcopy(restored)
            summary = copy.deepcopy(replay)
            if mutation == "count":
                result["optimizer_updates"] = True
            elif mutation == "pid":
                fresh["pid"] = True
            elif mutation == "step":
                fresh["step"] = True
            elif mutation == "binding":
                fresh["capture_binding_sha256"] = "foreign"
            elif mutation == "S":
                summary["S_gradient_verified"] = False
            values = {capture / "launch.json": launch, capture / "result.json": result,
                      reload / "result.json": fresh, capture / "reload_expected.json": expected,
                      reload / "sync.json": sync_probes(), reload / "forward_probes.json": forward_probes()}
            for path, value in values.items():
                path.write_text(json.dumps(value))
            summary["paths"] = {name: checkpoint.file_hash(capture / name)
                                for name in ("launch.json", "result.json", "reload_expected.json")}

            def functional(*args, **kwargs):
                if mutation == "parent_mutation":
                    (capture / "launch.json").write_text(json.dumps(launch | {"pid": 10003}))
                return {"launch": launch, "files": []}

            def semantic(*args, **kwargs):
                if mutation == "reload_mutation":
                    (reload / "result.json").write_text(json.dumps(fresh | {"step": 2}))
                return summary

            with patch.object(training, "parent_gate", side_effect=functional) as parent, \
                    patch.object(training, "read_preparation", return_value=({}, {}, self.actors)), \
                    patch.object(training.checkpoint, "read_metadata", return_value={"inventory": {"trainable": []}}), \
                    patch("fine_tuning.notebook_rl.repair_replay.replay_capture", side_effect=semantic) as replayed, \
                    patch("psutil.pid_exists", return_value=False):
                if mutation is None:
                    gate = training.training_parent_gate(capture, reload, admission=training.REPAIR_ADMISSION)
                    self.assertTrue(gate["repair_learning_verified"])
                    self.assertFalse(gate["strict_notebook_correctness_verified"])
                    self.assertTrue(replayed.call_args.kwargs["require_s"])
                else:
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        training.training_parent_gate(capture, reload, admission=training.REPAIR_ADMISSION)
                parent.assert_called_once_with(capture.absolute(), reload.absolute(), require_notebook_learning=False)


class RepairFixedTenTests(unittest.TestCase):
    admission = training.REPAIR_ADMISSION

    def setUp(self) -> None:
        self.fixture = legacy.IntermediateCheckpointTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.protocol = build_protocol(repair_notebook=True, context_limit=24576)
        for index, actor in enumerate(self.fixture.actors):
            actor.image_paths = ["image"] * (2 if index == 1 else 1)
        self.fixture.stop_after = 99
        self.preparations: list[tuple[str, bool]] = []
        self.evidence_calls: list[dict] = []
        self.signal_after = 0
        self.all_zero = False
        self.parent = {"launch": {"protocol": self.fixture.protocol, "model_path": str(self.fixture.model),
            "cache_check": str(self.fixture.cache)}, "repair_functional_verified": True,
            "S_gradient_verified": True, "repair_learning_verified": True, "files": []}
        self.enterContext(patch.object(training, "training_parent_gate", return_value=self.parent))
        self.enterContext(patch("fine_tuning.notebook_rl.updates.partition_learning_trees", side_effect=AssertionError("legacy partition")))
        self.enterContext(patch("fine_tuning.notebook_rl.repair_training.prepare_tree_credit", side_effect=self.prepare))
        self.enterContext(patch("fine_tuning.notebook_rl.repair_evidence.repair_update_evidence", side_effect=self.evidence))

        def engine(*args, **kwargs):
            return SimpleNamespace(policy_version=0, tokenizer=object())

        def build(actor, images, engine, verifier, prompts, cfg, version, rule, **options):
            from fine_tuning.notebook_rl.schema import ProblemTree
            self.assertTrue(self.fixture.moments_parked)
            self.assertTrue(callable(options["on_raw"]))
            options["on_raw"]({"problem_id": actor.problem_id, "completion": "authored fixture raw"})
            return ProblemTree(actor.problem_id, version, [], "VISION_OFF", {}, {})

        self.enterContext(patch("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine", side_effect=engine))
        self.enterContext(patch("fine_tuning.notebook_rl.rollout.build_tree", side_effect=build))

    def prepare(self, tree, actor, verifier, tokenizer, *, notebook_reward_enabled=True, **kwargs):
        self.assertFalse(self.fixture.moments_parked)
        self.preparations.append((actor.problem_id, notebook_reward_enabled))
        signal = not self.all_zero and int(actor.problem_id) >= self.signal_after
        value = {"admitted": True, "has_signal": signal, "admission": "VISION_OFF", "admission_reasons": [],
                 "groups": [], "scripted_boundary_only": True, "notebook_reward_enabled": notebook_reward_enabled}
        return value | {"sha256": checkpoint.digest(value)}

    def evidence(self, update, protocol, **kwargs):
        self.evidence_calls.append(kwargs)
        self.assertFalse(kwargs["require_s"])
        self.assertEqual(set(kwargs["preparations"]), {tree.problem_id for tree in kwargs["trees"]})
        return {"version": "repair-update-evidence-v1", "S_gradient_verified": False,
                "functional_verified": False, "repair_learning_verified": False, "benefit_verified": False}

    def run_training(self, **options):
        return self.fixture.run_training(admission=self.admission, **options)

    def test_skipped_pair_consumed_then_real_tiny_update_checkpoint_preserves_f_s_distinction(self) -> None:
        self.signal_after = 2
        self.fixture.stop_after = 1
        result = self.run_training()
        self.assertEqual((result["status"], result["optimizer_updates"]), ("BUDGET_STOP", 1))
        self.assertEqual(result["attempted_ids"], ["0", "1", "2", "3"])
        self.assertEqual(result["discarded_ids"], ["0", "1"])
        self.assertFalse(result["repair_functional_verified"] or result["repair_learning_verified"] or result["S_gradient_verified"])
        committed = result["last_committed_checkpoint"]
        metadata = checkpoint.read_metadata(Path(committed["path"]), binding=json.loads(
            (self.fixture.output / "launch.json").read_text())["binding"])
        self.assertEqual(metadata["cursor"]["position"], 4)
        self.assertEqual(metadata["cursor"]["consumed_problem_ids"], ["0", "1", "2", "3"])
        self.assertEqual(committed["pair_index"], 1)
        index = json.loads((self.fixture.output / "repair_preparation_index.json").read_text())
        self.assertEqual([row["problem_id"] for row in index["trees"]], result["attempted_ids"])
        self.assertEqual([row["kind"] for row in index["raw_records"]], ["REPAIR_RAW", "TREE"] * 4)
        self.assertTrue(all(enabled for _, enabled in self.preparations))

    def test_zero_updates_preserve_all_ten_and_do_not_fabricate_checkpoint_or_f(self) -> None:
        self.all_zero = True
        result = self.run_training()
        self.assertEqual((result["status"], result["optimizer_updates"]), ("NO_OPTIMIZER_UPDATES", 0))
        self.assertEqual(result["attempted_ids"], [str(index) for index in range(10)])
        self.assertEqual(result["signal_free_ids"], result["attempted_ids"])
        self.assertFalse(list(self.fixture.output.glob("checkpoint*")))
        self.assertFalse(self.evidence_calls)
        self.assertFalse(self.fixture.moments_parked)
        self.assertEqual(self.fixture.parking_steps, [0] * 10)
        self.assertFalse(result["repair_functional_verified"])
        self.assertEqual(len(list(self.fixture.output.glob("pair-*.json"))), 5)

    def test_r0_keeps_generation_and_uses_explicit_reload_with_zero_applied_repair_control(self) -> None:
        self.fixture.stop_after = 1
        initialization = self.fixture.root / "matched/initialization.json"
        initialization.parent.mkdir()
        initial = {"trainable_sha256": training.parameter_hashes(self.fixture.bundle),
            "base_sha256": self.fixture.bundle.base_sha256, "quantization": self.fixture.bundle.inventory["quantization"],
            "seed": 3407, "model_cache_metadata_sha256": checkpoint.file_hash(self.fixture.model / "nf4_base.json")}
        initialization.write_text(json.dumps(initial))
        reload = self.fixture.root / "matched-reload"
        with patch.object(training, "matched_notebook_arm", return_value={"initialization": {
                "sha256": checkpoint.file_hash(initialization)}}) as matched:
            result = training.train(self.fixture.root / "preparation.json", self.fixture.model, self.fixture.cache,
                self.fixture.root / "minimum", self.fixture.root / "reload", self.fixture.output,
                arm="answer-only", admission=self.admission,
                match_initialization=initialization, matched_reload=reload)
        self.assertEqual((result["arm"], result["optimizer_updates"]), ("answer-only", 1))
        self.assertEqual(matched.call_args.kwargs["matched_reload"], reload)
        self.assertTrue(self.preparations)
        self.assertTrue(all(not enabled for _, enabled in self.preparations))
        self.assertTrue(all(call["control"] for call in self.evidence_calls))
        self.assertFalse(result["S_gradient_verified"] or result["repair_learning_verified"])
        index = json.loads((self.fixture.output / "repair_preparation_index.json").read_text())
        self.assertTrue(index["control"])
        self.assertEqual([row["kind"] for row in index["raw_records"]], ["REPAIR_RAW", "TREE"] * 2)

    def test_complete_fixed_pair_replay_accepts_s_false_and_rejects_boolean_step_or_foreign_pair(self) -> None:
        def sync(bundle, preparation, engine, probes, protocol, step, *, push):
            engine.policy_version = step
            return sync_probes(("0", "1"), step)

        probes = forward_probes(("0", "1"))
        with patch("fine_tuning.notebook_rl.trainer.synchronize", side_effect=sync), \
                patch.object(training, "probe_evidence", return_value=probes):
            result = self.run_training()
        self.assertEqual(result["optimizer_updates"], 5)
        self.assertFalse(result["S_gradient_verified"] or result["repair_functional_verified"])
        capture = self.fixture.output
        launch = json.loads((capture / "launch.json").read_text())
        index = json.loads((capture / "repair_preparation_index.json").read_text())
        fresh = {row["problem_id"]: json.loads((capture / row["preparation_file"]).read_text()) for row in index["trees"]}
        metadata = checkpoint.read_metadata(capture / "checkpoint", binding=launch["binding"])
        with patch("fine_tuning.notebook_rl.repair_replay.replay_trees", return_value=fresh), \
                patch("fine_tuning.notebook_rl.repair_replay.validate_journal", return_value={row["file"]: row["sha256"]
                                                                                         for row in index["raw_records"]}):
            proof = training._repair_training_replay(capture, result, launch["execution_contract"], self.fixture.protocol,
                                                     parameter_inventory=metadata["inventory"]["trainable"])
            self.assertEqual(proof["optimizer_updates"], 5)
            self.assertFalse(proof["S_gradient_verified"] or proof["vjp_remeasured"])
            update_file = capture / "pair-00.json"
            original = update_file.read_text()
            for mutation in ("step", "pair"):
                value = json.loads(original)
                if mutation == "step":
                    value["step_after"] = True
                else:
                    value["attempted_ids"] = ["2", "3"]
                update_file.write_text(json.dumps(value))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    training._repair_training_replay(capture, result, launch["execution_contract"], self.fixture.protocol,
                                                     parameter_inventory=metadata["inventory"]["trainable"])
                update_file.write_text(original)
            result_file = capture / "result.json"
            original_result = result_file.read_text()

            def mutate_preparation(*args):
                result_file.write_text(json.dumps(result | {"changed_between_parse_and_hash": True}))
                return self.fixture.prepared, {}, self.fixture.actors

            with patch.object(training, "read_preparation", side_effect=mutate_preparation), self.assertRaises(ValueError):
                training._repair_training_replay(capture, result, launch["execution_contract"], self.fixture.protocol,
                                                 parameter_inventory=metadata["inventory"]["trainable"])
            result_file.write_text(original_result)

    def test_real_checkpoint_weight_and_runtime_replacements_cannot_reuse_fresh_reload(self) -> None:
        def sync(bundle, preparation, engine, probes, protocol, step, *, push):
            engine.policy_version = step
            return sync_probes(("0", "1"), step)

        probes = forward_probes(("0", "1"))
        with patch("fine_tuning.notebook_rl.trainer.synchronize", side_effect=sync), \
                patch.object(training, "probe_evidence", return_value=probes):
            result = self.run_training()
        capture = self.fixture.output
        launch = json.loads((capture / "launch.json").read_text())
        index = json.loads((capture / "repair_preparation_index.json").read_text())
        fresh = {row["problem_id"]: json.loads((capture / row["preparation_file"]).read_text()) for row in index["trees"]}
        metadata = checkpoint.read_metadata(capture / "checkpoint", binding=launch["binding"])
        reload = self.fixture.root / "fresh-reload"
        reload.mkdir()
        with patch("fine_tuning.notebook_rl.repair_replay.replay_trees", return_value=fresh), \
                patch("fine_tuning.notebook_rl.repair_replay.validate_journal", return_value={row["file"]: row["sha256"]
                                                                                         for row in index["raw_records"]}), \
                patch("psutil.pid_exists", return_value=False):
            proof = training._repair_training_replay(capture, result, launch["execution_contract"], self.fixture.protocol,
                                                     parameter_inventory=metadata["inventory"]["trainable"])
            self.assertTrue(all(f"{name}/{filename}" in proof["paths"]
                for name in [*(f"checkpoint-pair-{number:02d}" for number in range(5)), "checkpoint"]
                for filename in ("metadata.json", "metadata.sha256", "adapter.safetensors", "mergers.safetensors", "runtime.pt")))
            restored = {"status": "LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED", "pid": launch["pid"] + 100000,
                "capture_pid": launch["pid"], "step": 5, "fresh_reload_verified": True,
                "all_trainable_hashes_match": True, "optimizer_restored": True, "actual_forward_and_generation_match": True,
                "frozen_quantized_base_unchanged": True, "repair_functional_verified": True,
                "repair_replay_sha256": proof["sha256"], "S_gradient_verified": False,
                "repair_learning_verified": False, "benefit_verified": False,
                "repair_update_evidence_sha256": checkpoint.file_hash(capture / "repair_update_evidence.json"),
                "repair_protocol_sha256": self.fixture.protocol["sha256"], "capture_binding_sha256": checkpoint.digest(launch["binding"])}
            (reload / "result.json").write_text(json.dumps(restored))
            (reload / "forward_probes.json").write_text(json.dumps(probes))
            (reload / "sync.json").write_text(json.dumps(sync_probes(("0", "1"), 5)))

            def matched(admission=self.admission):
                return training.matched_notebook_arm(capture / "initialization.json", self.fixture.prepared,
                    self.fixture.protocol, self.fixture.model, self.fixture.cache, training.fixed_pairs(self.fixture.actors),
                    admission=admission, parent=launch["execution_contract"]["parent_gate"],
                    max_seconds=14400, matched_reload=reload)

            matched_receipt = matched()
            self.assertEqual(matched_receipt["repair_replay_sha256"], proof["sha256"])
            self.assertEqual(matched_receipt["repair_runtime_files"][str(capture / "checkpoint/runtime.pt")],
                             checkpoint.file_hash(capture / "checkpoint/runtime.pt"))
            other = next(policy for policy in training.REPAIR_ADMISSIONS if policy != self.admission)
            with self.assertRaisesRegex(ValueError, "admission and parent"):
                matched(other)
            for destination, source, names in (("checkpoint-pair-00", "checkpoint-pair-04", ("adapter.safetensors", "mergers.safetensors")),
                                                ("checkpoint", "checkpoint-pair-00", ("adapter.safetensors", "mergers.safetensors")),
                                                ("checkpoint", "checkpoint-pair-00", ("runtime.pt",))):
                target, other = capture / destination, capture / source
                saved = {name: (target / name).read_bytes() for name in (*names, "metadata.json", "metadata.sha256")}
                current = json.loads(saved["metadata.json"])
                foreign = checkpoint.read_metadata(other, binding=launch["binding"])
                for name in names:
                    (target / name).write_bytes((other / name).read_bytes())
                    current["files"][name] = checkpoint.file_hash(target / name)
                if "runtime.pt" not in names:
                    current["state_sha256"] = foreign["state_sha256"]
                (target / "metadata.json").write_text(json.dumps(current))
                (target / "metadata.sha256").write_text(checkpoint.file_hash(target / "metadata.json") + "\n")
                with self.subTest(destination=destination, names=names), self.assertRaises(ValueError):
                    matched()
                for name, raw in saved.items():
                    (target / name).write_bytes(raw)


class FunctionalRepairFixedTenTests(RepairFixedTenTests):
    admission = training.REPAIR_FUNCTIONAL_EXPLORATION

    def setUp(self) -> None:
        super().setUp()
        self.parent.update(admission_policy=self.admission, S_gradient_verified=False, repair_learning_verified=False)

    def test_functional_fixed_ten_reload_restores_real_tiny_payload_and_keeps_s_false(self) -> None:
        self.fixture.prepared["image_root"] = str(self.fixture.data / "images")
        parent_capture, parent_reload = self.fixture.root / "minimum", self.fixture.root / "reload"
        parent_capture.mkdir()
        parent_reload.mkdir()
        paths = [parent_capture / "launch.json", parent_reload / "sync.json"]
        for path in paths:
            path.write_text("{}")
        self.parent["files"] = [{"path": str(path), "sha256": checkpoint.file_hash(path)} for path in paths]
        probes = forward_probes(("0", "1"))

        def sync(bundle, preparation, engine, requests, protocol, step, *, push):
            engine.policy_version = step
            return sync_probes(("0", "1"), step)

        original_load = checkpoint.load_resume

        def cpu_load(model_path, saved, protocol, *, binding, **kwargs):
            def factory(*args):
                model = legacy.bundle()
                model.inventory["quantization"] = {"fixture_only": True}
                return model
            return original_load(model_path, saved, protocol, binding=binding, device="cpu", policy_factory=factory)

        with patch("fine_tuning.notebook_rl.trainer.synchronize", side_effect=sync), \
                patch.object(training, "probe_evidence", return_value=probes):
            result = self.run_training()
            capture = self.fixture.output
            launch = training._read(capture / "launch.json")
            launch["pid"] += 100000
            (capture / "launch.json").write_text(json.dumps(launch))
            index = training._read(capture / "repair_preparation_index.json")
            fresh = {row["problem_id"]: training._read(capture / row["preparation_file"]) for row in index["trees"]}
            with patch("fine_tuning.notebook_rl.repair_replay.replay_trees", return_value=fresh), \
                    patch("fine_tuning.notebook_rl.repair_replay.validate_journal", return_value={row["file"]: row["sha256"]
                                                                                             for row in index["raw_records"]}), \
                    patch.object(training, "training_parent_gate", return_value=launch["execution_contract"]["parent_gate"]) as parent, \
                    patch.object(checkpoint, "load_resume", side_effect=cpu_load) as resumed, \
                    patch("psutil.pid_exists", return_value=False):
                restored = training.reload_training(capture, self.fixture.root / "fresh")
        self.assertEqual((result["optimizer_updates"], restored["step"]), (5, 5))
        self.assertEqual(restored["execution_contract"]["admission_policy"], self.admission)
        self.assertFalse(restored["execution_contract"]["parent_gate"]["S_gradient_verified"])
        self.assertEqual(restored["execution_contract_sha256"], checkpoint.digest(launch["execution_contract"]))
        self.assertTrue(restored["repair_functional_verified"] and restored["optimizer_restored"])
        self.assertFalse(restored["S_gradient_verified"] or restored["repair_learning_verified"] or restored["benefit_verified"])
        resumed.assert_called_once()
        parent.assert_called_once_with(parent_capture, parent_reload, admission=self.admission)


if __name__ == "__main__":
    unittest.main()
