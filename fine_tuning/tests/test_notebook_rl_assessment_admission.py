"""Authored CPU assessment admission/replay and caller contracts, never Qwen success."""
import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import checkpoint, local_eval, local_profile, local_run, local_train, updates
from fine_tuning.notebook_rl.local_data import ACTOR_MODE_POLICY
from fine_tuning.notebook_rl.schema import ProblemTree, to_json
from fine_tuning.modules import tensor_sha256
from fine_tuning.tests import test_notebook_rl_local_eval as evaluation
from fine_tuning.tests import test_notebook_rl_local_train as training
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration
from fine_tuning.tests import test_notebook_rl_assessment_profile as profile
from fine_tuning.tests import test_notebook_rl_whole_reference_training_episode as whole


POLICY = local_train.ASSESSMENT_CREDIT_ADMISSION


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "file_sha256": checkpoint.file_hash(path)}


class AssessmentAdmissionTests(unittest.TestCase):
    def setUp(self):
        case = profile.AssessmentLifecycleTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.case, self.f = case, case.f
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "raw-0000.json"
        save(self.path, {"kind": "TREE", "raw": json.loads(to_json(self.f.tree))})
        self.values = self.f.optimizer_fixture()
        bundle, optimizer, scheduler, preparation, original, reference = self.values
        bundle.model["M"] = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        bundle.params["M"] = [bundle.model["M"]]
        bundle.inventory["trainable"] = [{"name": key, "group": key, "shape": list(bundle.model[key].shape),
            "dtype": str(bundle.model[key].dtype), "numel": bundle.model[key].numel()} for key in ("L", "M", "V")]
        optimizer, scheduler = checkpoint.make_optimizer(bundle, self.f.protocol)
        note_id = self.f.tree.groups[-1].samples[0].sample_id

        def logprobs(bundle, inputs, *args, **kwargs):
            return original(bundle, inputs, *args, **kwargs) + bundle.model["M"] * (inputs["sample"].sample_id == note_id)

        self.values = bundle, optimizer, scheduler, preparation, logprobs, reference
        self.before = {key: {key: tensor_sha256(bundle.model[key].reshape(-1))} for key in ("L", "M", "V")}
        self.update = self.f.step(self.values)
        self.after = {key: {key: tensor_sha256(bundle.model[key].reshape(-1))} for key in ("L", "M", "V")}
        scores = updates.notebook_objective_scores(self.f.tree, self.f.protocol["notebook_objective"])
        self.result = {"notebook_signal_groups": updates.notebook_signal_groups(self.f.tree, scores)}

    def replay(self, update=None):
        return local_train.fact_credit_learning_gate(self.root, self.result,
            self.update if update is None else update, self.f.protocol,
            actors={self.f.actor.problem_id: self.f.actor}, verifier=self.f.verifier,
            tokenizer=self.f.engine.tokenizer, admission=POLICY)

    def test_real_strict_zero_assessment_update_replays_without_notebook_success(self):
        self.assertEqual(self.update["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertNotEqual(self.before, self.after)
        self.assertEqual(self.replay(), [self.path])
        self.assertTrue(all(s.reward == 0 for s in self.f.tree.groups[-1].samples))
        with self.assertRaisesRegex(ValueError, "strict notebook admission is unsupported"):
            local_train.notebook_learning_gate(self.root, self.result, self.update, self.f.protocol)

    def test_other_channels_zero_mis_nonfinite_and_off_vision_cannot_replace_assessment_vjp(self):
        for kind in ("answer", "fact", "relation", "KL", "zero", "mis", "nonfinite", "vision", "credit"):
            update = copy.deepcopy(self.update)
            for row in update["notebook_backward"]:
                measured = row.get("assessment_gradient", {})
                if measured.get("measured") is not True:
                    continue
                if kind in ("answer", "fact", "relation", "KL"):
                    row["assessment_gradient"] = {"scope": "native_assessment_policy_loss_only", "measured": False}
                elif kind == "zero":
                    measured.update(group_norms=dict.fromkeys(("L", "M", "V"), 0.), nonzero_groups=[])
                elif kind == "mis":
                    update["mean_log_ratio_by_sample"][row["sample_id"]] = 999.
                elif kind == "nonfinite":
                    measured["group_norms"]["L"] = float("nan")
                elif kind == "vision":
                    measured["group_norms"]["V"] = 1.
                    measured["nonzero_groups"] = ["L", "V"]
                else:
                    update["synthesis_token_credit"][row["sample_id"]]["assessment_token_advantages"][0] += 1.
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.replay(update)

    def test_original_full_n4_credit_and_raw_token_identity_are_required(self):
        raw = json.loads(self.path.read_text())
        for kind in ("reduced_n4", "tokens", "score", "foreign"):
            changed, update = copy.deepcopy(raw), copy.deepcopy(self.update)
            if kind == "reduced_n4":
                changed["raw"]["groups"][-1]["samples"].pop()
            elif kind == "tokens":
                changed["raw"]["groups"][-1]["samples"][0]["completion_token_ids"][0] += 1
            elif kind == "score":
                next(iter(update["synthesis_token_credit"].values()))["full_group"][0]["score_sha256"] = "a" * 64
            else:
                update["routes"] = {"foreign-pair": False}
            save(self.path, changed)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.replay(update)
        save(self.path, raw)

    def test_aggregate_requires_exact_signed_membership_and_current_requirement(self):
        for kind in ("missing", "duplicate", "foreign", "mis", "route", "mask", "weight", "norm",
                     "vector", "inventory", "zero", "positive_only", "old_requirement", "missing_receipt"):
            update = copy.deepcopy(self.update)
            aggregate = update["assessment_aggregate_gradient"]
            member = aggregate["members"][0]
            if kind == "missing":
                aggregate["members"].pop()
            elif kind == "duplicate":
                aggregate["members"].append(copy.deepcopy(member))
            elif kind == "foreign":
                member["sample_id"] = "foreign"
            elif kind == "mis":
                member["retained"], member["disposition"] = False, "MIS_EXCLUDED"
            elif kind == "route":
                member["vision_on"] = not member["vision_on"]
            elif kind == "mask":
                member["mask_sha256"] = "a" * 64
            elif kind in ("weight", "norm"):
                member[kind] *= 2
            elif kind == "vector":
                member["token_advantages_sha256"] = "a" * 64
            elif kind == "inventory":
                aggregate["parameter_inventory"].pop()
            elif kind == "zero":
                aggregate.update(group_norms=dict.fromkeys(("L", "M", "V"), 0.), nonzero_groups=[])
            elif kind == "positive_only":
                aggregate["positive_only"] = True
            elif kind == "old_requirement":
                update["notebook_update_requirement"] = "verified-native-assessment-gradient-v1"
            else:
                update.pop("assessment_aggregate_gradient")
            # Matching hashes cannot make omitted or foreign actions part of the original n4.
            aggregate["members_sha256"] = checkpoint.digest(aggregate["members"])
            aggregate["parameter_inventory_sha256"] = checkpoint.digest(aggregate["parameter_inventory"])
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.replay(update)

    def test_checkpoint_parameter_inventory_rejects_resealed_parameter_name(self):
        update = copy.deepcopy(self.update)
        aggregate = update["assessment_aggregate_gradient"]
        aggregate["parameter_inventory"][0]["name"] = "foreign.parameter"
        aggregate["parameter_inventory_sha256"] = checkpoint.digest(aggregate["parameter_inventory"])
        with self.assertRaises(ValueError):
            local_run.assessment_update_evidence(update, self.f.protocol,
                parameter_inventory=self.values[0].inventory["trainable"])

    def parent_inputs(self):
        f, root = self.f, self.root
        second = replace(f.actor, problem_id="fixture:second", image_paths=("images/a.png", "images/b.png"),
                         image_sha256=("a" * 64, "b" * 64))
        actors = [f.actor, second]
        build = root / "build"
        save(build / "manifest.yaml", {"split_manifests": {"train": "fixture-vault"}})
        actor_rows = [json.loads(to_json(actor)) for actor in actors]
        prepared = {"format": "notebook-rl-local-data-v1", "status": "LOCAL_DATA_REVALIDATED",
            "local_protocol_sha256": f.protocol["sha256"], "build_root": str(build),
            "actors": save(root / "actors.json", actor_rows), "execution_actors": save(root / "execution.json", actor_rows),
            "actor_projection": {"policy": ACTOR_MODE_POLICY, "changed_ids": []},
            "data_request": save(root / "request.json", {"build_root": str(build), "index": {"path": str(root / "benchmark/index")}}),
            "validated_data": save(root / "checked.json", {"status": "VALIDATED_FOR_MEASUREMENT", "blockers": [],
                "measurement_allowed": True, "training_allowed": False, "actors": actor_rows,
                "code_sha256": {"historical-minimum.py": "a" * 64},
                "input_binding": {"build_manifest_sha256": checkpoint.file_hash(build / "manifest.yaml")}})}
        preparation = save(root / "preparation.json", prepared)
        launch = {"protocol": json.loads(json.dumps(f.protocol)), "standalone_base": True, "pid": 111222,
            "purpose": "MINIMAL_TWO_PROBLEM_ONE_UPDATE", "problem_ids": [a.problem_id for a in actors],
            "model_path": str(root / "unused-model"), "preparation": preparation["path"],
            "binding": local_run.minimum_binding(prepared, f.protocol, None) | {
                "code_sha256": checkpoint.digest({"historical-minimum.py": "a" * 64})}}
        expected = {"step": 1, "trainable_sha256": self.after, "frozen_base_sha256": "fixture-base",
                    "quantization": {}, "forward_probes": {}}
        metadata = {"step": 1, "state_sha256": {n: h for values in self.after.values() for n, h in values.items()},
                    "base_sha256": "fixture-base", "inventory": {"quantization": {},
                        "trainable": self.values[0].inventory["trainable"]}}
        result = self.result | {"status": "UPDATED_SAVED_PENDING_FRESH_RELOAD", "optimizer_updates": 1,
            "attempted_ids": launch["problem_ids"], "original_access_guard": {"denied_original_access_attempts": 0},
            **local_run.verify_changes(self.before, self.after, self.update),
            **local_run.assessment_update_evidence(self.update, f.protocol)}
        restored = {"status": "FRESH_PROCESS_RELOAD_VERIFIED", "capture_pid": launch["pid"], "pid": 111333, "step": 1,
            "original_access_guard": {"denied_original_access_attempts": 0}, **{key: True for key in (
                "all_trainable_hashes_match", "optimizer_restored", "actual_forward_and_generation_match",
                "frozen_quantized_base_unchanged")}}
        for name, value in (("launch.json", launch), ("result.json", result), ("update.json", self.update),
                            ("parameter_changes.json", {"before": self.before, "after": self.after}),
                            ("reload_expected.json", expected), ("updated_forward_probes.json", {}),
                            ("reload/result.json", restored), ("reload/forward_probes.json", {}), ("reload/sync.json", {})):
            save(root / name, value)
        for name in ("metadata.json", "metadata.sha256", "adapter.safetensors", "mergers.safetensors", "runtime.pt"):
            save(root / "checkpoint" / name, {"fixture_metadata_only": True})
        save(root / "raw-0001.json", {"kind": "TREE", "raw": json.loads(to_json(ProblemTree(second.problem_id, 0, [], "VISION_OFF", {}, {})))})
        return launch, prepared, restored, metadata

    def test_historical_minimum_requires_actual_parent_gate_and_exact_two_tree_reload_identity(self):
        launch, prepared, restored, metadata = self.parent_inputs()
        f = self.f
        f.verifier.metadata = {"vault_manifest_sha256": "fixture-vault"}
        with patch("psutil.pid_exists", return_value=False), patch.object(checkpoint, "read_metadata", return_value=metadata), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=f.engine.tokenizer), \
                patch("fine_tuning.notebook_rl.verifier_service.VerifierService", return_value=nullcontext(f.verifier)), \
                patch.object(local_train, "read_preparation", side_effect=AssertionError("historical source is immutable")), \
                patch.object(local_train, "fact_credit_learning_gate", return_value=[self.path]) as replay:
            gate = local_train.training_parent_gate(self.root, self.root / "reload", admission=POLICY)
            self.assertTrue(gate["assessment_learning_verified"])
            self.assertTrue(gate["assessment_only_gradient_verified"])
            self.assertFalse(gate["notebook_reward_learning_verified"])
            self.assertFalse(gate["accurate_whole_notebook_verified"])
            self.assertEqual(replay.call_args.kwargs["admission"], POLICY)
            for kind in ("reload", "live_reload", "source", "preparation", "ids", "tree_policy"):
                changed_launch, changed_prepared, changed_reload = copy.deepcopy(launch), copy.deepcopy(prepared), copy.deepcopy(restored)
                if kind == "reload":
                    changed_reload["actual_forward_and_generation_match"] = False
                elif kind == "live_reload":
                    changed_reload["pid"] = launch["pid"]
                elif kind == "source":
                    changed_launch["binding"]["code_sha256"] = "0" * 64
                elif kind == "preparation":
                    changed_prepared["local_protocol_sha256"] = "0" * 64
                elif kind == "ids":
                    changed_launch["problem_ids"] = launch["problem_ids"][::-1]
                else:
                    raw = json.loads(self.path.read_text())
                    raw["raw"]["policy_version"] = 1
                    save(self.path, raw)
                save(self.root / "launch.json", changed_launch)
                save(self.root / "preparation.json", changed_prepared)
                save(self.root / "reload/result.json", changed_reload)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    local_train.training_parent_gate(self.root, self.root / "reload", admission=POLICY)

    def test_answer_only_retains_actions_without_assessment_commit_or_credit(self):
        case = whole.WholeReferenceTrainingEpisodeTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        f = case.f
        f.protocol, f.prompts, f.cfg = self.f.protocol, self.f.prompts, self.f.cfg
        f.engine.tokenizer = profile.AssessmentTokenizer()
        samples = integration.episode.solution_samples(f.tree)
        samples[1].text = integration.response(f.item.slots, wrong=False)
        for index, sample in enumerate(samples):
            parent = samples[0] if index in (2, 3) else samples[2] if index in (4, 5) else None
            sample.prompt_token_ids = ([1, 2] if parent is None else
                parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2])
            sample.completion_token_ids = f.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.result = f.verifier.grade(f.actor.problem_id, sample.text, sample.finish_reason, f.actor.mode)
            sample.reward = sample.result.reward_answer
        case.append(["{}"] * 4)
        updates.bind_notebook_objective(f.tree, f.actor, f.verifier, f.protocol["notebook_objective"])
        bundle, optimizer, scheduler, preparation, logprobs, reference = f.optimizer_fixture()

        def shifted_reference(bundle, inputs, *args, **kwargs):
            return reference(bundle, inputs, *args, **kwargs) + (.005 if inputs["sample"].kind == "DIAG" else 0.)

        control = f.step((bundle, optimizer, scheduler, preparation, logprobs, shifted_reference), notebook_reward_enabled=False)
        self.assertEqual(control["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertGreater(control["kl_mean"], 0.)
        local_train.assessment_control_update_evidence(control, f.protocol, [f.tree])
        discarded = copy.deepcopy(f.tree)
        discarded.problem_id = "discarded-no-answer-signal"
        discarded.groups[-1].group_id = "discarded-diag"
        local_train.assessment_control_update_evidence(control, f.protocol, [f.tree, discarded])
        self.assertNotIn("discarded-diag", control["effective_rewards"])
        self.assertGreater(len(control["mean_log_ratio_by_sample"]), 4)
        control_root = self.root / "control"
        path = control_root / "raw-0000.json"
        save(path, {"kind": "TREE", "raw": json.loads(to_json(f.tree))})
        scores = updates.notebook_objective_scores(f.tree, f.protocol["notebook_objective"])

        def replay(record):
            return local_train.fact_credit_learning_gate(control_root,
                {"notebook_signal_groups": updates.notebook_signal_groups(f.tree, scores)}, record, f.protocol,
                actors={f.actor.problem_id: f.actor}, verifier=f.verifier, tokenizer=f.engine.tokenizer,
                admission=POLICY, tree_ids={f.tree.problem_id}, control=True)

        self.assertEqual(replay(control), [path])
        for kind in ("missing_diag_mis", "nonfinite_mis", "forged_receipt", "answer_reward"):
            changed = copy.deepcopy(control)
            if kind == "missing_diag_mis":
                for sample in f.tree.groups[-1].samples:
                    changed["mean_log_ratio_by_sample"].pop(sample.sample_id)
            elif kind == "nonfinite_mis":
                changed["mean_log_ratio_by_sample"][f.tree.groups[0].samples[0].sample_id] = float("nan")
            elif kind == "forged_receipt":
                changed["synthesis_regrade_receipts"] = {"unrelated-tree": {"forged": True}}
            else:
                changed["effective_rewards"][f.tree.groups[0].group_id] = [1.] * len(f.tree.groups[0].samples)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                replay(changed)
        for key, value in (("assessment_commit_required", True), ("assessment_learning_verified", True),
                           ("notebook_backward", self.update["notebook_backward"]),
                           ("synthesis_token_credit", self.update["synthesis_token_credit"]),
                           ("effective_rewards", self.update["effective_rewards"]),
                           ("assessment_aggregate_gradient", {}),
                           ("assessment_aggregate_gradient", self.update["assessment_aggregate_gradient"])):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "zero DIAG reward"):
                local_train.assessment_control_update_evidence(control | {key: value}, f.protocol, [f.tree])

    def test_pair_replay_keeps_unrouted_no_recovery_tree_and_rejects_foreign_pair_proof(self):
        original_native = integration.native

        def second_native():
            actor, item, verifier = original_native()
            actor, item = replace(actor, problem_id="fixture:second"), replace(item, problem_id="fixture:second")
            verifier.actors, verifier.items = {actor.problem_id: actor}, {actor.problem_id: item}
            return actor, item, verifier

        second = whole.WholeReferenceTrainingEpisodeTests()
        with patch.object(integration, "native", side_effect=second_native):
            second.setUp()
        self.addCleanup(second.doCleanups)
        f = second.f
        f.protocol, f.prompts = self.f.protocol, self.f.prompts
        f.cfg, f.engine.tokenizer = self.f.cfg, self.f.engine.tokenizer
        samples = integration.episode.solution_samples(f.tree)
        for index, sample in enumerate(samples):
            sample.text = integration.response(f.item.slots)
            parent = samples[0] if index in (2, 3) else samples[2] if index in (4, 5) else None
            sample.prompt_token_ids = ([1, 2] if parent is None else
                parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2])
            sample.completion_token_ids = f.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.result = f.verifier.grade(f.actor.problem_id, sample.text, sample.finish_reason, f.actor.mode)
            sample.reward = sample.result.reward_answer
        second.append()
        updates.bind_notebook_objective(f.tree, f.actor, f.verifier, f.protocol["notebook_objective"])
        self.assertEqual(f.tree.notebook["rewarded_notebook"]["status"], "NO_VERIFIED_RECOVERY")
        pairs = [[self.f.actor.problem_id, f.actor.problem_id], *[[f"unused:{i}", f"unused:{i+1}"] for i in range(2, 10, 2)]]
        save(self.root / "raw-0001.json", {"kind": "TREE", "raw": json.loads(to_json(f.tree))})
        for index, pid in enumerate([pid for pair in pairs[1:] for pid in pair], 2):
            save(self.root / f"raw-{index:04d}.json", {"kind": "TREE", "raw": json.loads(to_json(ProblemTree(pid, 1, [], "VISION_OFF", {}, {})))})
        update = self.update | {"attempted_ids": pairs[0]}
        save(self.root / "pair-00.json", update)
        for index, pair in enumerate(pairs[1:], 1):
            save(self.root / f"pair-{index:02d}.json", {"status": "NO_LEARNING_SIGNAL", "attempted_ids": pair, "step": 1})
        build = self.root / "pair-build"
        save(build / "manifest.yaml", {"split_manifests": {"train": "fixture-vault"}})
        actors = save(self.root / "inputs-actors.json", [json.loads(to_json(a)) for a in (self.f.actor, f.actor)])
        prepared = {"actors": actors, "execution_actors": actors, "actor_projection": {"policy": ACTOR_MODE_POLICY, "changed_ids": []},
            "build_root": str(build), "data_request": save(self.root / "inputs-request.json", {"index": {"path": str(self.root / "benchmark/index")}})}
        save(self.root / "launch.json", {"preparation": save(self.root / "inputs-preparation.json", prepared)["path"], "model_path": "unused"})
        self.f.verifier.actors.update(f.verifier.actors)
        self.f.verifier.items.update(f.verifier.items)
        self.f.verifier.metadata = {"vault_manifest_sha256": "fixture-vault"}
        contract = {"admission_policy": POLICY, "arm": "notebook", "notebook_reward_enabled": True, "pair_ids": pairs}
        result = {"optimizer_updates": 1, "assessment_learning_verified": True, "assessment_only_gradient_verified": True,
            "strict_notebook_correctness_verified": False, "accurate_whole_notebook_verified": False,
            "visual_improvement_verified": False, "self_improvement_verified": False}
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=self.f.engine.tokenizer), \
                patch("fine_tuning.notebook_rl.verifier_service.VerifierService", side_effect=lambda *a, **k: nullcontext(self.f.verifier)):
            local_train.validate_native_updates(self.root, result, contract, self.f.protocol)
            changed = copy.deepcopy(update)
            changed["retained_assessment_signal"]["signal_sample_ids"] = ["foreign-pair-note"]
            save(self.root / "pair-00.json", changed)
            with self.assertRaisesRegex(ValueError, "another original TREE/pair"):
                local_train.validate_native_updates(self.root, result, contract, self.f.protocol)


class AssessmentCallerTests(unittest.TestCase):
    def training_fixture(self):
        fixture = training.IntermediateCheckpointTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol.clear()
        fixture.protocol.update(local_profile.build_protocol(**profile.OPTIONS))
        parent = {"launch": {"protocol": json.loads(json.dumps(fixture.protocol)), "model_path": str(fixture.model), "cache_check": str(fixture.cache)},
                  "files": [{"path": str(fixture.root / "minimum/launch.json")}, {"path": str(fixture.root / "reload/sync.json")}]}
        self.enterContext(patch.object(local_train, "training_parent_gate", return_value=parent))
        self.enterContext(patch.object(updates, "bind_notebook_objective"))
        self.enterContext(patch.object(updates, "notebook_objective_scores", return_value={}))
        self.enterContext(patch.object(local_train, "probe_evidence", return_value={"fixture_only": True}))
        self.enterContext(patch("fine_tuning.notebook_rl.notebook_learning.append_rewarded_notebooks", side_effect=lambda trees, *a, **k:
            trees[0].notebook.update(rewarded_notebook={"notebook_generation_policy": fixture.protocol["notebook_generation_policy"],
                                                     "notebook_draft_policy": fixture.protocol["notebook_draft_policy"]})))
        return fixture

    def test_cli_explicit_policy_and_missing_minimum_remain_fail_closed(self):
        args = ["train", "--preparation", "/unused/prep", "--model-path", "/unused/model", "--cache-check", "/unused/cache",
                "--minimal", "/unused/min", "--minimal-reload", "/unused/reload", "--output", "/unused/out",
                "--arm", "notebook", "--admission", POLICY]
        with patch.object(local_train, "train", return_value={"status": "NO_OPTIMIZER_UPDATES"}) as train, patch("builtins.print"):
            self.assertEqual(local_train.main(args), 2)
            self.assertEqual(train.call_args.kwargs["admission"], POLICY)
        with patch.object(local_train, "parent_gate", side_effect=ValueError("real minimum/reload required")) as parent:
            with self.assertRaisesRegex(ValueError, "real minimum/reload"):
                local_train.training_parent_gate(Path("/unused/min"), Path("/unused/reload"), admission=POLICY)
            parent.assert_called_once()

    def test_train_dispatches_assessment_and_preserves_real_tiny_checkpoint(self):
        fixture = self.training_fixture()
        with patch.object(local_train, "native_relationship_update_evidence", side_effect=AssertionError("native-only gate")), \
                patch.object(local_run, "assessment_update_evidence", return_value={
                    "assessment_learning_verified": True, "assessment_only_gradient_verified": True}) as assessment:
            result = fixture.run_training(admission=POLICY)
        self.assertEqual((result["status"], result["optimizer_updates"]), ("BUDGET_STOP", 1))
        self.assertEqual(assessment.call_count, 1)
        launch = json.loads((fixture.output / "launch.json").read_text())
        metadata = checkpoint.read_metadata(Path(result["last_committed_checkpoint"]["path"]), binding=launch["binding"])
        self.assertEqual((metadata["step"], metadata["cursor"]["position"]), (1, 2))
        self.assertTrue(result["assessment_learning_verified"])
        self.assertFalse(result["accurate_whole_notebook_verified"])

    def test_no_eligible_pair_keeps_step_zero_and_does_not_force_an_update(self):
        fixture = self.training_fixture()
        fixture.no_signal = True
        with patch.object(local_run, "assessment_update_evidence", side_effect=AssertionError("no committed assessment")):
            result = fixture.run_training(admission=POLICY)
        self.assertEqual((result["status"], result["optimizer_updates"]), ("BUDGET_STOP", 0))
        pair = json.loads((fixture.output / "pair-00.json").read_text())
        self.assertEqual(pair, {"status": "NO_LEARNING_SIGNAL", "attempted_ids": ["0", "1"],
            "signal_free_ids": ["0", "1"], "unresolved_ids": [], "discarded_ids": ["0", "1"], "step": 0})
        self.assertFalse(list(fixture.output.glob("checkpoint*")))

    def test_fixed_ten_reload_revalidates_assessment_pairs_and_binds_saved_contract(self):
        fixture = self.training_fixture()
        initial = copy.deepcopy(fixture.bundle)
        fixture.stop_after = 99
        fixture.prepared["image_root"] = str(fixture.data / "images")
        with patch.object(local_run, "assessment_update_evidence", return_value={
                "assessment_learning_verified": True, "assessment_only_gradient_verified": True}):
            result = fixture.run_training(admission=POLICY)
        self.assertEqual((result["status"], result["optimizer_updates"]), ("LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD", 5))
        launch = json.loads((fixture.output / "launch.json").read_text())
        launch["pid"] = 900001
        save(fixture.output / "launch.json", launch)
        real_load = checkpoint.load_resume

        def resume(*args, **kwargs):
            self.assertEqual(kwargs["device"], "cuda:0")
            return real_load(*args, **(kwargs | {"device": "cpu", "policy_factory": lambda *_: copy.deepcopy(initial)}))

        with patch("psutil.pid_exists", return_value=False), patch.object(checkpoint, "load_resume", side_effect=resume), \
                patch.object(local_train, "validate_native_updates") as pairs:
            restored = local_train.reload_training(fixture.output, fixture.root / "fresh-reload")
        pairs.assert_called_once_with(fixture.output, result, launch["execution_contract"], fixture.protocol,
            parameter_inventory=initial.inventory["trainable"])
        self.assertEqual(restored["capture_binding_sha256"], checkpoint.digest(launch["binding"]))
        self.assertEqual(restored["execution_contract"], launch["execution_contract"])
        self.assertTrue(restored["assessment_only_gradient_verified"])
        self.assertFalse(restored["accurate_whole_notebook_verified"])
        fixture.bundle.model.load_state_dict(initial.model.state_dict())

        def control_step(*args, **kwargs):
            self.assertIs(kwargs["notebook_reward_enabled"], False)
            return fixture.update(*args, **kwargs) | {"notebook_update_requirement": updates.ASSESSMENT_REQUIREMENT,
                "notebook_objective": fixture.protocol["notebook_objective"], "notebook_reward_enabled": False,
                "assessment_commit_required": False, "assessment_learning_verified": False, "effective_rewards": {}}

        control_output = fixture.root / "answer-only"
        with patch.object(updates, "optimizer_step", side_effect=control_step), \
                patch.object(local_train, "validate_native_updates") as pairs:
            control = local_train.train(fixture.root / "preparation.json", fixture.model, fixture.cache,
                fixture.root / "minimum", fixture.root / "reload", control_output,
                arm="answer-only", admission=POLICY, match_initialization=fixture.output / "initialization.json")
            self.assertEqual(control["optimizer_updates"], 5)
            control_launch = json.loads((control_output / "launch.json").read_text())
            control_launch["pid"] = 900002
            save(control_output / "launch.json", control_launch)
            with patch("psutil.pid_exists", return_value=False), patch.object(checkpoint, "load_resume", side_effect=resume):
                control_reload = local_train.reload_training(control_output, fixture.root / "answer-reload")
        self.assertEqual(pairs.call_count, 2)
        self.assertFalse(control_reload["assessment_learning_verified"])
        self.assertFalse(control_reload["assessment_only_gradient_verified"])
        self.assertEqual(json.loads((control_output / "initialization.json").read_text()),
                         json.loads((fixture.output / "initialization.json").read_text()))

    def test_matching_control_replays_notebook_pairs_and_preserves_parent_budget_and_initialization(self):
        fixture = training.HistoricalMatchedNotebookTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol = local_profile.build_protocol(**profile.OPTIONS)
        parent = {"assessment_learning_verified": True, "strict_notebook_correctness_verified": False}
        fixture.contract.update(admission_policy=POLICY, parent_gate=parent)
        fixture.launch["protocol"] = fixture.metadata["protocol"] = json.loads(json.dumps(fixture.protocol))
        fixture.metadata["inventory"]["trainable"] = [{"name": "L.weight", "group": "L", "shape": [1],
            "dtype": "torch.float32", "numel": 1}]
        fixture.metadata["resolved_protocol_sha256"] = checkpoint.digest(fixture.protocol)
        fixture.launch["execution_contract_sha256"] = checkpoint.digest(fixture.contract)
        fixture.launch["binding"] = fixture.metadata["binding"] = local_train.training_binding(fixture.prepared, fixture.protocol, fixture.contract)
        fixture.write_json(fixture.path.parent / "launch.json", fixture.launch)
        fixture.write_metadata()
        with patch.object(local_train, "validate_native_updates") as pairs:
            fixture.match(admission=POLICY, parent=parent, max_seconds=14400)
            pairs.assert_called_once_with(fixture.path.parent, fixture.result, fixture.contract, fixture.protocol,
                parameter_inventory=fixture.metadata["inventory"]["trainable"])
            with self.assertRaisesRegex(ValueError, "time budget"):
                fixture.match(admission=POLICY, parent=parent, max_seconds=60000)
            with self.assertRaisesRegex(ValueError, "admission and parent"):
                fixture.match(admission=POLICY, parent={"assessment_learning_verified": False})

    def test_native_evaluation_replays_assessment_pairs_for_both_arms_and_binds_reload(self):
        from fine_tuning.notebook_rl import local_ladder_generation as generation

        for arm in ("notebook", "answer-only"):
            fixture = evaluation.NativeLadderRuntimeTests()
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
            fixture.protocol.clear()
            fixture.protocol.update(json.loads(json.dumps(local_profile.build_protocol(**profile.OPTIONS))))
            fixture.metadata["inventory"]["trainable"] = [{"name": "L.weight", "group": "L", "shape": [1],
                "dtype": "torch.float32", "numel": 1}]
            fixture.launch["binding"]["protocol_sha256"] = fixture.protocol["sha256"]
            parent = {"launch": {"protocol": fixture.protocol}, "assessment_learning_verified": True,
                      "strict_notebook_correctness_verified": False}
            fixture.contract.update(admission_policy=POLICY, parent_gate=parent, arm=arm, notebook_reward_enabled=arm == "notebook")
            fixture.captured["arm"] = arm
            fixture.launch["execution_contract_sha256"] = checkpoint.digest(fixture.contract)
            fixture.restored.update(execution_contract=fixture.contract,
                execution_contract_sha256=checkpoint.digest(fixture.contract), capture_binding_sha256=checkpoint.digest(fixture.launch["binding"]))
            current = generation.protocol(fixture.templates, fixture.protocol["sha256"], context_limit=24576)
            fixture.gate["protocol_sha256"] = local_eval.digest(current)
            fixture.generation["protocol"] = current
            for row in fixture.generation["rows"]:
                row["context_limit"] = 24576
            if arm == "answer-only":
                fixture.contract["matched_notebook_arm"] = {"initialization": {"path": str(fixture.capture / "initialization.json")}}
                fixture.launch["execution_contract_sha256"] = checkpoint.digest(fixture.contract)
                fixture.restored["execution_contract_sha256"] = checkpoint.digest(fixture.contract)
            with patch.object(local_train, "training_parent_gate", return_value=parent), \
                    patch.object(local_train, "validate_native_updates") as pairs, \
                    patch.object(local_train, "matched_notebook_arm", return_value=fixture.contract.get("matched_notebook_arm")):
                result = local_eval.evaluate_ladder(fixture.capture, fixture.cohort, fixture.receipt, fixture.root / "assessment-eval",
                    condition="FT", arm=arm, reload_result=fixture.reload,
                    minimal_capture=fixture.root / "minimal", minimal_reload=fixture.root / "minimal-reload")
                self.assertEqual(result["status"], "COMPLETE_NATIVE_LADDER_GENERATION")
                pairs.assert_called_once()
                self.assertEqual(pairs.call_args.args[:3], (fixture.capture, fixture.captured, fixture.contract))
                self.assertEqual(json.loads(json.dumps(pairs.call_args.args[3])), fixture.protocol)
                self.assertEqual(pairs.call_args.kwargs,
                    {"parameter_inventory": fixture.metadata["inventory"]["trainable"]})
                fixture.restored["capture_binding_sha256"] = "changed"
                with self.assertRaisesRegex(ValueError, "assessment reload execution"):
                    local_eval.evaluate_ladder(fixture.capture, fixture.cohort, fixture.receipt, fixture.root / "changed-eval",
                        condition="FT", arm=arm, reload_result=fixture.reload,
                        minimal_capture=fixture.root / "minimal", minimal_reload=fixture.root / "minimal-reload")


if __name__ == "__main__":
    unittest.main()
