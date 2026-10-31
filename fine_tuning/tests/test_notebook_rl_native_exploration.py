"""Bounded collection and lifecycle fixtures; these never load pretrained weights or CUDA."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl import checkpoint, local_profile, local_train, updates
from fine_tuning.tests import test_notebook_rl_local_train as training
from fine_tuning.tests import test_notebook_rl_local_eval as evaluation


POLICY = local_train.FIXED_TEN_NATIVE_EXPLORATION


def protocol():
    return local_profile.build_protocol(verified_note_synthesis_credit=True,
                                       pair_action_notebook=True, context_limit=24576)


def descriptor(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "file_sha256": checkpoint.file_hash(path)}


def collection_inputs(root, current, prepared, actors):
    original = descriptor(root / "original-selection.json", {"problem_ids": [a.problem_id for a in actors]})
    prepared.update(previous_selection=original, selection=descriptor(root / "selection.json", {
        "previous_selection": original, "problem_ids": [a.problem_id for a in actors],
        "local_protocol_sha256": current["sha256"]}))


class NativeCollectionAdmissionTests(unittest.TestCase):
    def test_policy_is_exact_and_older_defaults_remain_strict(self):
        current = protocol()
        self.assertEqual(local_train.execution_admission({}, current), local_train.STRICT_ADMISSION)
        self.assertEqual(local_train.execution_admission({"admission_policy": POLICY}, current), POLICY)
        for changed in (local_profile.build_protocol(),
                        current | {"notebook_generation_policy": "verified-pair-action-notebook-v1"},
                        current | {"notebook_update_requirement": None},
                        current | {"context_limit": 16384}):
            with self.subTest(changed=changed["protocol_id"]), self.assertRaises(ValueError):
                local_train.execution_admission({"admission_policy": POLICY}, changed)

    def test_missing_minimum_is_rejected_for_every_older_policy(self):
        for admission in (local_train.STRICT_ADMISSION, local_train.EXPLORATORY_ADMISSION,
                          local_train.SYNTHESIS_CREDIT_ADMISSION, "verified-update-exploration-v1"):
            with self.subTest(admission=admission), patch.object(local_train, "training_parent_gate") as gate:
                with self.assertRaisesRegex(ValueError, "minimum and reload paths required"):
                    local_train.train(Path("/unused"), Path("/unused"), Path("/unused"),
                                      None, None, Path("/unused"), arm="notebook", admission=admission)
                gate.assert_not_called()

    def test_new_collection_rejects_answer_control_and_implicit_parent(self):
        for arm, minimal, reload in (("answer-only", None, None), ("notebook", Path("/old"), None)):
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "control is unsupported|matched initialization"):
                local_train.train(Path("/unused"), Path("/unused"), Path("/unused"), minimal, reload,
                    Path("/unused"), arm=arm, admission=POLICY)
        with patch.object(local_train, "training_parent_gate", return_value={"launch": {"protocol": protocol()}}):
            with self.assertRaisesRegex(ValueError, "no matched answer-only control"):
                local_train.train(Path("/unused"), Path("/unused"), Path("/unused"), Path("/minimum"),
                    Path("/reload"), Path("/unused"), arm="answer-only", match_initialization=Path("/match"),
                    admission=local_train.SYNTHESIS_CREDIT_ADMISSION)

    def test_original_order_cohort_and_descriptor_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root, current = Path(directory), protocol()
            actors = [type("Actor", (), {"problem_id": str(i)})() for i in range(10)]
            prepared: dict = {}
            collection_inputs(root, current, prepared, actors)
            gate = local_train.native_collection_gate(prepared, current, {"cache": "checked"}, actors)
            self.assertTrue(gate["collection_allowed"])
            self.assertFalse(gate["native_relationship_learning_verified"])
            self.assertFalse(gate["strict_notebook_correctness_verified"])
            self.assertNotIn("launch", gate)
            for changed in (actors[::-1], actors[:-1], actors[:-1] + actors[:1]):
                with self.subTest(ids=[a.problem_id for a in changed]), self.assertRaises(ValueError):
                    local_train.native_collection_gate(prepared, current, {}, changed)
            prepared["previous_selection"] = {"problem_ids": list(range(10)), "gold": "inline-private"}
            with self.assertRaisesRegex(ValueError, "file descriptors"):
                local_train.native_collection_gate(prepared, current, {}, actors)

    def test_cli_accepts_only_explicit_new_policy_without_minimum_paths(self):
        args = ["train", "--preparation", "/unused/preparation", "--model-path", "/unused/model",
                "--cache-check", "/unused/cache", "--output", "/unused/output", "--arm", "notebook",
                "--admission", POLICY]
        with patch.object(local_train, "train", return_value={"status": "NO_OPTIMIZER_UPDATES"}) as called, \
                patch("builtins.print"):
            self.assertEqual(local_train.main(args), 2)
        self.assertEqual(called.call_args.args[3:5], (None, None))
        self.assertEqual(called.call_args.kwargs["admission"], POLICY)

    def test_copied_native_proof_cannot_be_assigned_to_another_original_pair(self):
        from fine_tuning.notebook_rl.schema import Group, ProblemTree, Sample, to_json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = [[str(i), str(i+1)] for i in range(0, 10, 2)]
            for i in range(10):
                pid, group_id, version = str(i), f"group-{i}", min(i // 2, 2)
                samples = [Sample(f"sample-{i}" if ordinal == 0 else f"sample-{i}:{ordinal}",
                    pid, group_id, "DIAG", 1, None, [3], [1], [-.5], "stop", "PERSISTED", 3407 + ordinal, version)
                    for ordinal in range(4)]
                tree = ProblemTree(pid, version, [Group(group_id, "DIAG", pid, "a" * 64, samples, True, None)],
                                   "VISION_OFF", {}, {})
                descriptor(root / f"raw-{i:04d}.json", {"kind": "TREE", "raw": json.loads(to_json(tree))})
            records: list[dict] = []
            scores = {f"sample-{i}" if ordinal == 0 else f"sample-{i}:{ordinal}": {"fixture_sample": f"{i}:{ordinal}"}
                      for i in range(10) for ordinal in range(4)}
            for i, ids in enumerate(pairs):
                sid = f"sample-{i*2}"
                full_group = [{"sample_id": sid if ordinal == 0 else f"{sid}:{ordinal}",
                    "token_ids_sha256": checkpoint.digest([1]), "text_sha256": checkpoint.digest("PERSISTED"),
                    "score_sha256": checkpoint.digest(scores[sid if ordinal == 0 else f"{sid}:{ordinal}"])}
                    for ordinal in range(4)]
                credit = {"token_ids_sha256": checkpoint.digest([1]), "text_sha256": checkpoint.digest("PERSISTED"),
                    "score_sha256": checkpoint.digest(scores[sid]), "partial_token_advantages": [.1],
                    "full_group": full_group, "full_group_sha256": checkpoint.digest(full_group)}
                record = {"status": "UPDATED_UNSYNCHRONIZED" if i < 2 else "NO_LEARNING_SIGNAL",
                    "attempted_ids": ids, "step_before": i, "step_after": i+1,
                    "routes": {ids[0]: False}, "mean_log_ratio_by_sample": {sid: 0.},
                    "synthesis_token_credit": {sid: credit},
                    "retained_native_relationship_signal": {"signal_sample_ids": [sid]},
                    "notebook_backward": [{"sample_id": sid, "group_id": f"group-{i*2}",
                                           "action_tokens": 1, "vision_on": False}]}
                records.append(record)
                descriptor(root / f"pair-{i:02d}.json", record)
            with patch("fine_tuning.notebook_rl.updates.notebook_objective_scores", return_value=scores), \
                    patch("fine_tuning.notebook_rl.episode_notebook.validate_update_binding"), \
                    patch.object(local_train, "native_relationship_update_evidence") as native:
                local_train.validate_native_updates(root, {"optimizer_updates": 2}, {"pair_ids": pairs}, protocol())
                self.assertEqual(native.call_count, 2)
                records[1]["retained_native_relationship_signal"] = records[0]["retained_native_relationship_signal"]
                descriptor(root / "pair-01.json", records[1])
                with self.assertRaisesRegex(ValueError, "another original TREE/pair"):
                    local_train.validate_native_updates(root, {"optimizer_updates": 2}, {"pair_ids": pairs}, protocol())
                self.assertEqual(native.call_count, 3)
                records[1]["retained_native_relationship_signal"] = {"signal_sample_ids": ["sample-2"]}
                records[1]["synthesis_token_credit"] = {"sample-2": records[0]["synthesis_token_credit"]["sample-0"]}
                descriptor(root / "pair-01.json", records[1])
                with self.assertRaisesRegex(ValueError, "original group tokens/scores"):
                    local_train.validate_native_updates(root, {"optimizer_updates": 2}, {"pair_ids": pairs}, protocol())


class NativeCollectionLifecycleTests(unittest.TestCase):
    def fixture(self):
        fixture = training.IntermediateCheckpointTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol.clear()
        fixture.protocol.update(protocol())
        fixture.stop_after = 99
        fixture.prepared["image_root"] = str(fixture.data / "images")
        collection_inputs(fixture.root, fixture.protocol, fixture.prepared, fixture.actors)
        (fixture.root / "inputs").mkdir()
        (fixture.root / "inputs/preparation.json").write_text(json.dumps(fixture.prepared))
        self.enterContext(patch.object(local_train.approval, "code_hashes", return_value={"fixture.py": "a" * 64}))
        self.enterContext(patch("fine_tuning.notebook_rl.updates.bind_notebook_objective"))
        self.enterContext(patch("fine_tuning.notebook_rl.updates.notebook_objective_scores", return_value={}))
        self.enterContext(patch("fine_tuning.notebook_rl.episode_notebook.validate_update_binding"))
        self.enterContext(patch.object(local_train, "retained_update_signal", return_value={"fixture_only": True}))

        def append(trees, *_args, **_kwargs):
            for tree in trees:
                tree.notebook["rewarded_notebook"] = {
                    "notebook_generation_policy": fixture.protocol["notebook_generation_policy"],
                    "notebook_draft_policy": fixture.protocol.get("notebook_draft_policy")}

        self.enterContext(patch("fine_tuning.notebook_rl.notebook_learning.append_rewarded_notebooks", side_effect=append))
        self.enterContext(patch.object(local_train, "probe_evidence", return_value={"fixture_only": True}))
        return fixture

    def run_collection(self, fixture, **kwargs):
        return local_train.train(fixture.root / "inputs/preparation.json", fixture.model, fixture.cache,
                                None, None, fixture.output, arm="notebook", admission=POLICY, **kwargs)

    def test_no_signal_pairs_reach_later_original_pairs_and_native_gate_precedes_counted_commits(self):
        fixture = self.fixture()
        calls = []

        def update(*args, **kwargs):
            ids = [tree.problem_id for tree in args[3]]
            calls.append(ids)
            self.assertEqual(args[6]["notebook_update_requirement"], updates.NATIVE_RELATIONSHIP_REQUIREMENT)
            self.assertTrue(kwargs["require_retained_learning_signal"])
            if len(calls) <= 2:
                return {"status": ("NO_RETAINED_NATIVE_RELATIONSHIP_SIGNAL" if len(calls) == 1
                                   else "NO_NATIVE_RELATIONSHIP_GRADIENT")}
            return fixture.update(*args, **kwargs)

        with patch("fine_tuning.notebook_rl.updates.optimizer_step", side_effect=update), \
                patch.object(local_train, "training_parent_gate", side_effect=AssertionError("no fabricated parent")), \
                patch.object(local_train, "native_relationship_update_evidence", return_value={"fixture_only": True}) as native:
            result = self.run_collection(fixture)
        self.assertEqual(result["attempted_ids"], [str(i) for i in range(10)])
        self.assertEqual(calls, [[str(i)] for i in range(0, 10, 2)])
        self.assertEqual(result["optimizer_updates"], 3)
        self.assertEqual(native.call_count, 3)
        self.assertFalse((fixture.output / "checkpoint-pair-00").exists())
        self.assertFalse((fixture.output / "checkpoint-pair-01").exists())
        launch = json.loads((fixture.output / "launch.json").read_text())
        self.assertIsNone(launch["execution_contract"]["parent_gate"])
        self.assertEqual(launch["execution_contract"]["pair_ids"], [[str(i), str(i+1)] for i in range(0, 10, 2)])
        self.assertFalse(result["self_improvement_verified"])
        self.assertFalse(result["accurate_whole_notebook_verified"])
        self.assertFalse(result["strict_notebook_correctness_verified"])
        metadata = checkpoint.read_metadata(fixture.output / "checkpoint", binding=launch["binding"])
        self.assertEqual((metadata["step"], metadata["cursor"]["position"]), (3, 10))

    def test_budget_and_failed_native_evidence_cannot_create_a_commit(self):
        fixture = self.fixture()
        with patch.object(local_train, "native_relationship_update_evidence", side_effect=ValueError("missing native VJP")):
            with self.assertRaisesRegex(ValueError, "native VJP"):
                self.run_collection(fixture)
        saved = json.loads((fixture.output / "result.json").read_text())
        self.assertEqual(saved["optimizer_updates"], 0)
        self.assertFalse(list(fixture.output.glob("checkpoint*")))

    def test_collection_budget_stops_before_later_pair_without_replacement(self):
        fixture = self.fixture()
        fixture.no_signal = True
        result = self.run_collection(fixture, max_seconds=1)
        self.assertEqual((result["status"], result["optimizer_updates"]), ("BUDGET_STOP", 0))
        self.assertEqual(result["attempted_ids"], ["0", "1"])
        self.assertFalse(list(fixture.output.glob("checkpoint*")))

    def test_real_tiny_payload_reload_binds_collection_contract_and_revalidates_data_cache(self):
        fixture = self.fixture()
        initial = copy.deepcopy(fixture.bundle)
        with patch.object(local_train, "native_relationship_update_evidence", return_value={"fixture_only": True}) as native:
            self.assertEqual(self.run_collection(fixture)["optimizer_updates"], 5)
            launch_path = fixture.output / "launch.json"
            launch = json.loads(launch_path.read_text())
            launch["pid"] = 900001
            launch_path.write_text(json.dumps(launch))
            real_load = checkpoint.load_resume

            def resume(*args, **kwargs):
                self.assertEqual(kwargs["device"], "cuda:0")
                return real_load(*args, **(kwargs | {"device": "cpu",
                    "policy_factory": lambda *_: copy.deepcopy(initial)}))

            with patch("psutil.pid_exists", return_value=False), \
                    patch.object(checkpoint, "load_resume", side_effect=resume) as load, \
                    patch.object(local_train, "training_parent_gate", side_effect=AssertionError("no minimum parent")), \
                    patch.object(local_train, "cache_readiness", return_value={"original_model_root": str(fixture.root / "original")}) as cache, \
                    patch.object(local_train, "read_preparation", return_value=(fixture.prepared, {}, fixture.actors)) as prepared:
                restored = local_train.reload_training(fixture.output, fixture.root / "fresh-reload")
            cache.assert_called_once()
            prepared.assert_called_once()
            self.assertEqual(load.call_count, 1)
            self.assertEqual(native.call_count, 10)
        self.assertEqual(restored["status"], "LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED")
        self.assertEqual(restored["execution_contract"], launch["execution_contract"])
        self.assertEqual(restored["capture_binding_sha256"], checkpoint.digest(launch["binding"]))
        self.assertFalse(restored["self_improvement_verified"])
        with self.assertRaisesRegex(ValueError, "collection execution contract"):
            local_train.validate_native_collection(launch["execution_contract"], fixture.prepared,
                fixture.protocol, {"original_model_root": "changed"}, fixture.actors)


class NativeCollectionEvaluationTests(unittest.TestCase):
    """Verify consumer delegation and fail-closed ordering without running model inference."""
    def fixture(self):
        from fine_tuning.notebook_rl import local_ladder_generation

        fixture = evaluation.NativeLadderRuntimeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol.clear()
        fixture.protocol.update(json.loads(json.dumps(protocol())))
        fixture.contract.update(admission_policy=POLICY, parent_gate=None,
                                collection_gate={"collection_allowed": True, "fixture_only": True,
                                                 "code_sha256": fixture.launch["binding"]["code_sha256"]})
        fixture.launch["execution_contract_sha256"] = checkpoint.digest(fixture.contract)
        fixture.launch["binding"]["protocol_sha256"] = fixture.protocol["sha256"]
        fixture.restored.update(execution_contract=fixture.contract,
            execution_contract_sha256=fixture.launch["execution_contract_sha256"],
            capture_binding_sha256=checkpoint.digest(fixture.launch["binding"]))
        fixture.expected["step"] = fixture.metadata["step"] = fixture.captured["optimizer_updates"]
        current = local_ladder_generation.protocol(fixture.templates, fixture.protocol["sha256"], context_limit=24576)
        fixture.gate["protocol_sha256"] = local_ladder_generation._digest(current)
        fixture.generation["protocol"] = current
        for row in fixture.generation["rows"]:
            row["context_limit"] = 24576
        return fixture

    def evaluate(self, fixture, name="collection-eval", **kwargs):
        return evaluation.runner.evaluate_ladder(fixture.capture, fixture.cohort, fixture.receipt,
            fixture.root / name, condition="FT", arm="notebook", reload_result=fixture.reload, **kwargs)

    def test_native_evaluation_revalidates_collection_inputs_cache_contract_and_commits(self):
        from fine_tuning.notebook_rl import local_run

        fixture = self.fixture()
        with patch.object(local_run, "read_preparation", return_value=(fixture.prepared, {}, fixture.training)) as prepared, \
                patch.object(local_train, "validate_native_collection", return_value=fixture.contract["collection_gate"]) as gate, \
                patch.object(local_train, "validate_native_updates") as commits, \
                patch.object(local_train, "training_parent_gate", side_effect=AssertionError("no fabricated parent")):
            result = self.evaluate(fixture)
        canonical = local_profile.restore_protocol(fixture.protocol)
        prepared.assert_called_once_with(fixture.preparation, canonical)
        self.assertEqual(gate.call_args.args[:3], (fixture.contract, fixture.prepared, canonical))
        commits.assert_called_once_with(fixture.capture, fixture.captured, fixture.contract, canonical)
        self.assertEqual(result["status"], "COMPLETE_NATIVE_LADDER_GENERATION")
        launched = json.loads((fixture.root / "collection-eval/launch.json").read_text())
        self.assertEqual(launched["native_collection_gate"], fixture.contract["collection_gate"])
        self.assertNotIn("minimal_parent_gate", launched)

    def test_changed_collection_data_or_reload_contract_cannot_load_model(self):
        from fine_tuning.notebook_rl import local_run

        fixture = self.fixture()
        with patch.object(local_run, "read_preparation", side_effect=ValueError("stale original data/source")):
            with self.assertRaisesRegex(ValueError, "stale original"):
                self.evaluate(fixture)
        fixture.load_ft.assert_not_called()
        fixture.restored["capture_binding_sha256"] = "changed"
        with patch.object(local_run, "read_preparation", return_value=(fixture.prepared, {}, fixture.training)), \
                patch.object(local_train, "validate_native_collection", return_value=fixture.contract["collection_gate"]), \
                patch.object(local_train, "validate_native_updates"):
            with self.assertRaisesRegex(ValueError, "reload execution binding"):
                self.evaluate(fixture)
        fixture.load_ft.assert_not_called()

    def test_old_policy_still_requires_minimum_and_native_control_is_unsupported(self):
        fixture = self.fixture()
        fixture.contract["admission_policy"] = local_train.STRICT_ADMISSION
        fixture.launch["execution_contract_sha256"] = checkpoint.digest(fixture.contract)
        with self.assertRaisesRegex(ValueError, "minimum capture and reload required"):
            self.evaluate(fixture)
        fixture.contract["admission_policy"] = POLICY
        fixture.launch["execution_contract_sha256"] = checkpoint.digest(fixture.contract)
        with self.assertRaisesRegex(ValueError, "control is unsupported"):
            evaluation.runner.evaluate_ladder(fixture.capture, fixture.cohort, fixture.receipt,
                fixture.root / "unsupported", condition="FT", arm="answer-only", reload_result=fixture.reload)
        fixture.load_ft.assert_not_called()


if __name__ == "__main__":
    unittest.main()
