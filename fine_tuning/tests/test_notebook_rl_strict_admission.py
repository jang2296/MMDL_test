"""CPU route admission checks; authored outputs are not model-learning evidence."""
import copy
import json
import unittest
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from fine_tuning.notebook_rl.episode_notebook import (
    FEEDBACK_POLICY, STRICT_ROUTE_RULE, audit_episode, strict_routing, validate_update_binding,
)
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.local_run import capture
from fine_tuning.notebook_rl.local_train import eligible_trees
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import SlotReport, to_json
from fine_tuning.notebook_rl.updates import (
    has_learning_signal, optimizer_step, partition_learning_trees, route_flags,
)
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests import test_notebook_rl_episode_notebook as fixtures

RIGHT, WRONG = fixtures.RIGHT, fixtures.WRONG


class StrictAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EpisodeNotebookTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def record(self, texts, *, strict=True, plain=False, problem_id=None, notebook="none"):
        fixture = self.fixture
        if problem_id is not None:
            actor, item, lineage, *_ = fixtures.records()
            fixture.actor = replace(actor, problem_id=problem_id)
            vault = fixture.root / problem_id
            write_split(vault, [fixture.actor], [replace(item, problem_id=problem_id)],
                        [replace(lineage, problem_id=problem_id)])
            fixture.verifier = Verifier(vault)
        if plain:
            fixture.actor = replace(fixture.actor, mode="plain")
        fixture.prompts = fixture.prompts | ({"notebook_route_rule": STRICT_ROUTE_RULE,
            "notebook_feedback_policy": FEEDBACK_POLICY} if strict else {})
        record, engine = fixture.tree([texts, [RIGHT] * 2])
        if notebook != "none":
            audit = audit_episode(record, fixture.actor, fixture.verifier,
                                  routing_rule=STRICT_ROUTE_RULE)
            valid = json.dumps(fixtures.authored_note(record, audit))
            if notebook == "mixed":
                engine.replies.append([valid, "{}", "{}", "{}"])
            elif notebook == "constant":
                engine.replies.append([valid] * 4)
            else:
                self.assertEqual(notebook, "failed")
                engine.replies.extend([["{}"] * 4] * 3)
            fixture.append(record, engine)
        else:
            fixture.append(record, engine, should_stop=lambda: True)
        return record

    def rehash(self, record):
        note = record.notebook["rewarded_notebook"]
        note["episode_audit_sha256"] = _hash(note["episode_audit"])

    def test_missing_plain_partial_zero_required_and_conflict_cannot_certify_off(self):
        record = self.record(["Final answer: 7"] * 2)
        self.assertEqual(strict_routing(record)["admission"], "UNRESOLVED")
        validate_update_binding(record)  # Unresolved raw records remain valid to save.
        self.assertFalse(has_learning_signal(record, include_notebooks=True))
        for slots in (None, SlotReport(0, 1, 0, (), 0, 1, ()),
                      SlotReport(1, 2, 1, (), 1, 2, ()),
                      SlotReport(0, 0, 0, (), 0, 0, ()),
                      SlotReport(1, 1, 1, (), 1, 1, ("cell[a][b]",))):
            changed = copy.deepcopy(record)
            for sample in changed.groups[0].samples:
                sample.result = replace(sample.result, slots=slots)
            self.assertEqual(strict_routing(changed)["admission"], "UNRESOLVED")
        plain = self.record([RIGHT] * 2, plain=True)
        self.assertEqual(strict_routing(plain)["admission"], "UNRESOLVED")

    def test_complete_correct_observations_allow_off_even_with_wrong_answer(self):
        record = self.record([RIGHT, RIGHT.replace("Final answer: 7", "Final answer: 8")])
        self.assertEqual(strict_routing(record)["admission"], "VISION_OFF")
        self.assertTrue(has_learning_signal(record))
        self.assertEqual(route_flags([record], "MOD-RV", 3407, 0), {record.problem_id: False})

    def test_wrong_required_observation_allows_on_even_with_missing_other_attempt(self):
        record = self.record(["Final answer: 7", WRONG])
        self.assertEqual(strict_routing(record)["admission"], "VISION_ON")
        self.assertEqual(route_flags([record], "MOD-RV", 3407, 0), {record.problem_id: True})
        self.assertTrue(has_learning_signal(record))

    def test_rehashed_audit_flags_cannot_replace_actual_slot_evidence(self):
        record = self.record(["Final answer: 7"] * 2)
        for key, value in (("admission", "VISION_OFF"), ("all_required_observations_verified", True),
                           ("complete_observation_attempts", [1, 2]), ("visual_attempts", [1]),
                           ("complete_episode", False), ("module_causality_certified", True),
                           ("model_claims_used_for_routing", True)):
            changed = copy.deepcopy(record)
            changed.notebook["rewarded_notebook"]["episode_audit"][key] = value
            self.rehash(changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "admission differs"):
                validate_update_binding(changed)
        changed = copy.deepcopy(record)
        changed.notebook["rewarded_notebook"]["episode_audit"]["attempts"][0]["wrong_slots"] = ["fake"]
        self.rehash(changed)
        with self.assertRaisesRegex(ValueError, "wrong observations differ"):
            validate_update_binding(changed)

    def test_strict_rejects_forced_random_and_unresolved_routes_legacy_unchanged(self):
        unresolved = self.record(["Final answer: 7"] * 2)
        for arm in ("MOD-RV", "MOD-FML", "MOD-FVML", "MOD-RR"):
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "resolved MOD-RV"):
                route_flags([unresolved], arm, 3407, 0)
        resolved = self.record([RIGHT] * 2)
        for arm in ("MOD-FML", "MOD-FVML", "MOD-RR"):
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "resolved MOD-RV"):
                route_flags([resolved], arm, 3407, 0)
        self.fixture.prompts.pop("notebook_route_rule")
        self.fixture.prompts.pop("notebook_feedback_policy")
        legacy = self.record(["Final answer: 7"] * 2, strict=False)
        self.assertNotIn("admission", legacy.notebook["rewarded_notebook"]["episode_audit"])
        self.assertEqual(route_flags([legacy], "MOD-FVML", 3407, 0), {legacy.problem_id: True})
        self.assertEqual(route_flags([legacy], "MOD-FML", 3407, 0), {legacy.problem_id: False})
        self.assertEqual(route_flags([legacy], "MOD-RR", 3407, 0), {legacy.problem_id: False})

    def test_optimizer_rejects_unresolved_missing_binding_and_health_override_before_step(self):
        protocol = build_protocol()
        # Keep the pre-reread fixture's admission boundary; v13 has separate tests.
        protocol["protocol_id"] = "local-nf4-full-history-notebook-v12"
        for key in tuple(protocol):
            if key.startswith("notebook_observation_"):
                del protocol[key]
        unresolved = self.record(["Final answer: 7"] * 2)
        resolved = self.record([RIGHT, WRONG])
        for record, arm, health, message in (
                (unresolved, "MOD-RV", False, "signal-free"),
                (resolved, "MOD-FML", False, "without vision health override"),
                (resolved, "MOD-FVML", False, "without vision health override"),
                (resolved, "MOD-RR", False, "without vision health override"),
                (resolved, "MOD-RV", True, "without vision health override")):
            opt = Mock()
            with self.subTest(arm=arm, health=health), self.assertRaisesRegex(ValueError, message):
                optimizer_step(SimpleNamespace(inventory={"arm": arm}), opt,
                    SimpleNamespace(last_epoch=0), [record], {record.problem_id: [object()]},
                    SimpleNamespace(policy_version=0, tokenizer=None), protocol, "ALG-NB3", step=0,
                    seed=3407, vision_disabled=health)
            opt.step.assert_not_called()
        del resolved.notebook["rewarded_notebook"]["episode_audit"]["routing_rule"]
        with self.assertRaisesRegex(ValueError, "strict admission binding missing"):
            optimizer_step(SimpleNamespace(inventory={"arm": "MOD-RV"}), Mock(),
                SimpleNamespace(last_epoch=0), [resolved], {resolved.problem_id: []},
                SimpleNamespace(policy_version=0), protocol, "ALG-NB3", step=0, seed=3407)

    def test_strict_tree_cannot_use_legacy_protocol_to_disable_vision(self):
        record = self.record([RIGHT, WRONG])
        protocol = build_protocol() | {"notebook_route_rule": "all-observed-required-slots-v1"}
        for health in (False, True):
            optimizer = Mock()
            with self.subTest(health=health), self.assertRaisesRegex(ValueError, "tree/protocol routing admission"):
                optimizer_step(SimpleNamespace(inventory={"arm": "MOD-RV"}), optimizer,
                    SimpleNamespace(last_epoch=0), [record], {record.problem_id: []},
                    SimpleNamespace(policy_version=0), protocol, "ALG-NB3", step=0, seed=3407,
                    vision_disabled=health)
            optimizer.step.assert_not_called()

    def test_partition_keeps_fixed_raw_attempts_separates_unresolved_and_signal_free(self):
        records = [self.record(["Final answer: 7"] * 2, problem_id="unresolved"),
                   self.record([RIGHT] * 2, problem_id="constant"),
                   self.record([RIGHT, WRONG], problem_id="eligible")]
        original = [to_json(record) for record in records]
        eligible, unresolved, free = partition_learning_trees(records, include_notebooks=True)
        self.assertEqual(eligible, [records[2]])
        self.assertEqual(unresolved, [records[0].problem_id])
        self.assertEqual(free, [records[1].problem_id])
        self.assertEqual([to_json(record) for record in records], original)
        self.assertEqual(eligible_trees(records, "notebook"), eligible)
        self.assertEqual(eligible_trees(records, "answer-only"), eligible)
        self.assertEqual(partition_learning_trees(records[:1]), ([], unresolved, []))
        self.assertEqual(partition_learning_trees([]), ([], [], []))

    def test_minimal_controller_records_unresolved_or_mixed_without_replacement(self):
        for case in ("unresolved", "mixed", "signal-free", "answer-only", "failed-notes", "constant-notes"):
            first = self.record(["Final answer: 7"] * 2 if case != "signal-free" else [RIGHT] * 2,
                                problem_id=case + "-first")
            second = self.record([RIGHT, WRONG] if case in ("mixed", "answer-only", "failed-notes", "constant-notes") else
                                 ["Final answer: 7"] * 2 if case == "unresolved" else [RIGHT] * 2,
                                 problem_id=case + "-second",
                                 notebook={"mixed": "mixed", "failed-notes": "failed",
                                           "constant-notes": "constant"}.get(case, "none"))
            trees = [first, second]
            before = [to_json(tree) for tree in trees]
            actors = [SimpleNamespace(problem_id=t.problem_id) for t in trees]
            data = self.fixture.root / (case + "-data")
            data.mkdir()
            (data / "manifest.yaml").write_text(json.dumps({"split_manifests": {"train": "vault"}}))
            bundle = SimpleNamespace(inventory={"quantization": {}}, base_sha256="base")
            verifier = MagicMock()
            verifier.__enter__.return_value.metadata = {"vault_manifest_sha256": "vault"}
            update = {"status": "UPDATED_UNSYNCHRONIZED", "step_after": 1, "notebook_backward": []}
            writes = {}
            mocks = {
                "fine_tuning.notebook_rl.local_run.cache_readiness": {"original_model_root": str(data)},
                "fine_tuning.notebook_rl.local_run.read_preparation": ({"build_root": str(data)}, {}, actors),
                "fine_tuning.notebook_rl.local_run.minimal_actors": actors,
                "fine_tuning.notebook_rl.local_run._binding": {},
                "fine_tuning.notebook_rl.local_run._install_actor_guards": [],
                "fine_tuning.notebook_rl.local_run.load_images": [Mock()],
                "fine_tuning.notebook_rl.local_run.make_probes": [],
                "fine_tuning.notebook_rl.local_run.parameter_hashes": {},
                "fine_tuning.notebook_rl.local_run.verify_changes": {},
                "fine_tuning.notebook_rl.local_run.probe_evidence": {},
                "fine_tuning.notebook_rl.quantized_policy.load_quantized_policy": bundle,
                "fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged": None,
                "fine_tuning.notebook_rl.quantized_cache.block_original_access":
                    {"denied_original_access_attempts": 0},
                "fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine": Mock(),
                "fine_tuning.notebook_rl.checkpoint.make_optimizer": (Mock(), Mock()),
                "fine_tuning.notebook_rl.checkpoint.capture_rng": {},
                "fine_tuning.notebook_rl.trainer.synchronize": {},
                "fine_tuning.notebook_rl.verifier_service.VerifierService": verifier,
                "fine_tuning.notebook_rl.notebook_learning.append_rewarded_notebooks": {},
                "torch.cuda.reset_peak_memory_stats": None,
                "torch.cuda.max_memory_allocated": 0,
                "torch.cuda.max_memory_reserved": 0,
            }
            with self.subTest(case=case), ExitStack() as stack:
                for name, value in mocks.items():
                    stack.enter_context(patch(name, return_value=value))
                build = stack.enter_context(patch("fine_tuning.notebook_rl.rollout.build_tree", side_effect=trees))
                optimizer = stack.enter_context(patch("fine_tuning.notebook_rl.updates.optimizer_step", return_value=update))
                saved = stack.enter_context(patch("fine_tuning.notebook_rl.checkpoint.save"))
                stack.enter_context(patch("fine_tuning.notebook_rl.local_run.write_json",
                    side_effect=lambda path, value: writes.update({path.name: copy.deepcopy(value)})))
                stack.enter_context(patch("builtins.print"))
                result = capture(data / "prepared", self.fixture.root / "model",
                    self.fixture.root / (case + "-output"), standalone=True, cache_check=data / "cache")
            self.assertEqual(build.call_count, 2)
            self.assertEqual(result["attempted_ids"], [t.problem_id for t in trees])
            self.assertEqual([to_json(t) for t in trees], before)
            self.assertEqual(sum(v["kind"] == "TREE" for k, v in writes.items() if k.startswith("raw-")), 2)
            if case == "mixed":
                self.assertEqual(result["unresolved_ids"], [first.problem_id])
                self.assertEqual(result["signal_free_ids"], [])
                self.assertEqual(result["discarded_ids"], [first.problem_id])
                self.assertEqual(optimizer.call_args.args[3], [second])
                self.assertEqual(saved.call_args.args[4]["discarded_problem_ids"], result["discarded_ids"])
                self.assertEqual(saved.call_args.args[4]["position"], 2)
                self.assertEqual(result["notebook_signal_groups"], 1)
            elif case in ("answer-only", "failed-notes", "constant-notes"):
                self.assertEqual(result["status"], "NO_NOTEBOOK_LEARNING_SIGNAL")
                self.assertEqual(result["optimizer_updates"], 0)
                self.assertEqual(result["notebook_signal_groups"], 0)
                self.assertFalse(result["notebook_gradient_contribution_verified"])
                self.assertEqual(result["discarded_ids"], [first.problem_id])
                optimizer.assert_not_called()
                saved.assert_not_called()
            else:
                self.assertEqual(result["status"], "UNRESOLVED_ROUTING" if case == "unresolved" else "NO_LEARNING_SIGNAL")
                self.assertEqual(result["optimizer_updates"], 0)
                self.assertEqual(result["discarded_ids"], result["attempted_ids"])
                self.assertEqual(result["signal_free_ids"], [] if case == "unresolved" else result["attempted_ids"])
                optimizer.assert_not_called()
                saved.assert_not_called()

    def test_protocol_binds_stricter_identity_and_rejects_changed_rule(self):
        protocol = build_protocol()
        self.assertEqual(protocol["protocol_id"], "local-nf4-full-history-notebook-v13")
        self.assertEqual(local_prompts(protocol)["notebook_route_rule"], STRICT_ROUTE_RULE)
        with self.assertRaisesRegex(ValueError, "strict routing admission"):
            local_prompts(protocol | {"notebook_route_rule": "all-observed-required-slots-v1"})


if __name__ == "__main__":
    unittest.main()
