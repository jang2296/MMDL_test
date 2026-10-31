"""Authored historical replay fixtures, not accurate-notebook or GPU evidence."""
import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl import checkpoint, episode_notebook, local_train, updates
from fine_tuning.notebook_rl.answers import VERSION as ANSWER_VERSION
from fine_tuning.notebook_rl.local_data import ACTOR_MODE_POLICY
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.vault import read_lineage, write_split
from fine_tuning.notebook_rl.verifier import VERSION as CURRENT_VERSION
from fine_tuning.notebook_rl.verifier_service import VerifierService
from fine_tuning.tests import test_notebook_rl_local_train as fixtures

OLD_VERSION = ANSWER_VERSION + "+native-graph-relations-v1"


class HistoricalVerifierTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExploratoryAdmissionTests()
        with patch("fine_tuning.notebook_rl.verifier.VERSION", OLD_VERSION):
            self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.record = self.fixture.fixture
        self.tree, self.actor = self.record.tree, self.record.actor
        self.root, self.path = self.fixture.root, self.fixture.path
        self.verifier = self.record.verifier

    def audit(self, tree=None, **kwargs):
        return episode_notebook.audit_episode(tree or self.tree, self.actor, self.verifier,
            routing_rule=episode_notebook.STRICT_ROUTE_RULE, **kwargs)

    def score(self, tree=None, **kwargs):
        return updates._score_notebook_objective(tree or self.tree, self.actor,
            self.verifier, self.record.objective, **kwargs)

    def gate(self, **kwargs):
        return local_train.fact_credit_learning_gate(self.root, self.fixture.result,
            self.fixture.update, self.fixture.protocol, actors={self.tree.problem_id: self.actor},
            verifier=self.verifier, tokenizer=self.record.engine.tokenizer, **kwargs)

    def test_default_rejects_and_explicit_replay_preserves_whole_objective_and_credit(self):
        before = to_json(self.tree), copy.deepcopy(self.fixture.update), self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "stored outcome differs"):
            self.audit()
        with self.assertRaisesRegex(ValueError, "stored outcome differs"):
            self.score()
        self.assertEqual(self.audit(replay_relation_v1=True),
                         self.tree.notebook["rewarded_notebook"]["episode_audit"])
        self.assertEqual(self.score(replay_relation_v1=True), self.tree.notebook["learning_objective"])
        self.assertEqual(self.gate(replay_relation_v1=True), [self.path])
        self.assertEqual((to_json(self.tree), self.fixture.update, self.path.read_bytes()), before)
        self.assertTrue(all(sample.reward == 0 for sample in self.record.group.samples))

    def test_service_cause_uses_fresh_issued_result_and_never_forged_historical_result(self):
        with VerifierService(self.root / "atoms", forbidden_roots=[]) as verifier:
            with patch.object(verifier, "cause", wraps=verifier.cause) as cause:
                checked = episode_notebook.audit_episode(self.tree, self.actor, verifier,
                    routing_rule=episode_notebook.STRICT_ROUTE_RULE, replay_relation_v1=True)
            self.assertEqual(checked, self.tree.notebook["rewarded_notebook"]["episode_audit"])
            self.assertTrue(cause.call_args_list)
            self.assertTrue(all(call.args[1].verifier_version == CURRENT_VERSION
                                for call in cause.call_args_list))

    def test_unknown_mixed_and_future_versions_reject(self):
        first = episode_notebook.solution_samples(self.tree)[0]
        for version in (CURRENT_VERSION, "unknown", ANSWER_VERSION + "+native-graph-relations-v2",
                        ANSWER_VERSION + "+native-graph-relations-v4"):
            changed = copy.deepcopy(self.tree)
            episode_notebook.solution_samples(changed)[0].result = replace(first.result, verifier_version=version)
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "historical verifier transition"):
                self.audit(changed, replay_relation_v1=True)
        with patch("fine_tuning.notebook_rl.verifier.VERSION", ANSWER_VERSION + "+native-graph-relations-v4"):
            with self.assertRaisesRegex(ValueError, "historical verifier transition"):
                self.audit(replay_relation_v1=True)

    def test_every_solution_field_and_false_child_claim_are_compared(self):
        samples = episode_notebook.solution_samples(self.tree)
        for index in (0, len(samples) - 1):
            original = samples[index].result
            mutations = [replace(original, status="INCORRECT" if original.status == "CORRECT" else "CORRECT",
                                 reward_answer=1.0 - original.reward_answer),
                replace(original, reward_answer=1.0 - original.reward_answer),
                replace(original, extracted="999"),
                replace(original, slots=replace(original.slots, required_correct=0)),
                replace(original, slots=replace(original.slots, false_relations=("edge[a,b]",))),
                replace(original, slots=replace(original.slots, unverified_relations=("edge[a,b]",)))]
            for result in mutations:
                changed = copy.deepcopy(self.tree)
                episode_notebook.solution_samples(changed)[index].result = result
                with self.subTest(index=index, result=result), self.assertRaises(ValueError):
                    self.audit(changed, replay_relation_v1=True)

    def test_diag_projection_slot_drift_and_missing_result_reject(self):
        for missing in (False, True):
            changed = copy.deepcopy(self.tree)
            sample = next(s for g in changed.groups if g.kind == "DIAG" for s in g.samples)
            sample.result = (None if missing else replace(sample.result,
                slots=replace(sample.result.slots, all_total=sample.result.slots.all_total + 1)))
            changed.notebook["rewarded_notebook"]["notebook_groups_sha256"] = _hash(
                [json.loads(to_json(group)) for group in changed.groups if group.kind == "DIAG"])
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError,
                    "stored outcome differs|reinspection requires"):
                self.score(changed, replay_relation_v1=True)

    def test_invalid_diag_without_claims_remains_none(self):
        from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleEngine

        fixture = self.record.fixture
        with patch("fine_tuning.notebook_rl.verifier.VERSION", OLD_VERSION):
            tree, _ = fixture.tree()
            engine = ReversibleEngine([["Observed facts:\ncell[a][b] = 7\nSolution:"],
                                       ["{}"] * fixture.cfg.n_diag])
            fixture.append(tree, engine, should_stop=lambda: bool(tree.notebook.get("model_notes")))
        self.assertTrue(all(sample.result is None for group in tree.groups if group.kind == "DIAG"
                            for sample in group.samples))
        before = to_json(tree)
        self.score(tree, replay_relation_v1=True)
        self.assertEqual(to_json(tree), before)

    def test_no_diag_tree_outside_update_routes_is_freshly_regraded_and_bound(self):
        fixture = self.record.fixture
        actor = replace(self.actor, problem_id=self.actor.problem_id + ":no-note")
        item = self.verifier.items[self.actor.problem_id]
        self.verifier.items[actor.problem_id] = replace(item, problem_id=actor.problem_id)
        fixture.actor = actor
        with patch("fine_tuning.notebook_rl.verifier.VERSION", OLD_VERSION):
            tree, engine = fixture.tree()
            fixture.append(tree, engine, should_stop=lambda: True)
            updates.bind_notebook_objective(tree, actor, self.verifier, self.record.objective)
        self.assertFalse(any(group.kind == "DIAG" for group in tree.groups))
        path = self.root / "raw-0001.json"
        path.write_text(json.dumps({"kind": "TREE", "raw": json.loads(to_json(tree))}))
        actors = {self.actor.problem_id: self.actor, actor.problem_id: actor}
        with patch.object(self.verifier, "grade", wraps=self.verifier.grade) as grade:
            paths = local_train.fact_credit_learning_gate(self.root, self.fixture.result,
                self.fixture.update, self.fixture.protocol, actors=actors, verifier=self.verifier,
                tokenizer=self.record.engine.tokenizer, replay_relation_v1=True)
        self.assertEqual(paths, [self.path, path])
        self.assertEqual({call.args[1] for call in grade.call_args_list if call.args[0] == actor.problem_id},
                         {sample.text for sample in episode_notebook.solution_samples(tree)})
        with patch.object(self.verifier, "grade", wraps=self.verifier.grade) as grade:
            original_grade = grade._mock_wraps

            def drift(problem_id, *args):
                result = original_grade(problem_id, *args)
                return replace(result, extracted="999") if problem_id == actor.problem_id else result

            grade.side_effect = drift
            with self.assertRaisesRegex(ValueError, "stored outcome differs"):
                local_train.fact_credit_learning_gate(self.root, self.fixture.result,
                    self.fixture.update, self.fixture.protocol, actors=actors, verifier=self.verifier,
                    tokenizer=self.record.engine.tokenizer, replay_relation_v1=True)

    def parent_inputs(self):
        def save(name, value):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return {"path": str(path), "file_sha256": checkpoint.file_hash(path)}

        root = self.root / "parent-build"
        write_split(root / "train", [self.actor], list(self.verifier.items.values()),
                    list(read_lineage(self.root / "atoms").values()))
        request = save("request.json", {"build_root": str(root),
            "index": {"path": str(self.root / "benchmark" / "index.json")}})
        actors = save("actors.json", [json.loads(to_json(self.actor))])
        save("result.json", self.fixture.result)
        save("update.json", self.fixture.update)
        save("parent-build/manifest.yaml", {"split_manifests": {"train":
            checkpoint.file_hash(root / "train" / "manifest.yaml")}})
        checked = save("checked.json", {"status": "VALIDATED_FOR_MEASUREMENT", "blockers": [],
            "measurement_allowed": True, "training_allowed": False,
            "input_binding": {"build_manifest_sha256": checkpoint.file_hash(root / "manifest.yaml")},
            "actors": [json.loads(to_json(self.actor))]})
        prepared = {"format": "notebook-rl-local-data-v1", "status": "LOCAL_DATA_REVALIDATED",
            "local_protocol_sha256": self.fixture.protocol["sha256"], "build_root": str(root),
            "data_request": request, "validated_data": checked, "actors": actors, "execution_actors": actors,
            "actor_projection": {"policy": ACTOR_MODE_POLICY, "changed_ids": []}}
        preparation = save("preparation.json", prepared)
        return {"launch": {"protocol": json.loads(json.dumps(self.fixture.protocol)),
            "preparation": preparation["path"], "problem_ids": [self.actor.problem_id],
            "model_path": str(self.root / "unused-model"),
            "binding": {"data_sha256": checkpoint.digest(prepared), "code_sha256": "a" * 64,
                "protocol_sha256": self.fixture.protocol["sha256"],
                "prompts_sha256": self.fixture.protocol["prompts_sha256"]}},
            "notebook_reward_learning_verified": False, "files": []}

    def test_parent_proof_is_deterministic_explicit_and_requires_historical_binding(self):
        prerequisite = self.parent_inputs()
        before = self.path.read_bytes(), copy.deepcopy(prerequisite)
        with patch.object(local_train, "parent_gate", return_value=prerequisite), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=self.record.engine.tokenizer):
            first = local_train.training_parent_gate(self.root, self.root / "reload",
                admission=local_train.EXPLORATORY_ADMISSION)
            second = local_train.training_parent_gate(self.root, self.root / "reload",
                admission=local_train.EXPLORATORY_ADMISSION)
            self.assertEqual(first, second)
            proof = first["historical_verifier_replay"]
            self.assertEqual(proof["version"], "native-graph-relations-v1-to-v3-semantic-replay-v1")
            self.assertEqual((proof["from_verifier_version"], proof["to_verifier_version"]),
                             (OLD_VERSION, CURRENT_VERSION))
            self.assertEqual(proof["current_verifier_code_sha256"]["verifier.py"],
                checkpoint.file_hash(Path(episode_notebook.__file__).with_name("verifier.py")))
            self.assertEqual(proof["tree_ids"], [self.tree.problem_id])
            self.assertFalse(first["strict_notebook_correctness_verified"])
            self.assertEqual((self.path.read_bytes(), prerequisite), before)
            prerequisite["launch"]["binding"]["code_sha256"] = checkpoint.digest(local_train.approval.code_hashes())
            with self.assertRaisesRegex(ValueError, "homogeneous historical"):
                local_train.training_parent_gate(self.root, self.root / "reload",
                    admission=local_train.EXPLORATORY_ADMISSION)

    def test_parent_rejects_mixed_versions_and_missing_launch_tree_before_service(self):
        prerequisite = self.parent_inputs()
        for mutation in ("mixed", "missing-tree"):
            raw = json.loads(self.path.read_text())
            if mutation == "mixed":
                raw["raw"]["groups"][0]["samples"][0]["result"]["verifier_version"] = CURRENT_VERSION
            else:
                prerequisite["launch"]["problem_ids"].append("missing")
            self.path.write_text(json.dumps(raw))
            with self.subTest(mutation=mutation), \
                    patch.object(local_train, "parent_gate", return_value=prerequisite), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=self.record.engine.tokenizer), \
                    patch("fine_tuning.notebook_rl.verifier_service.VerifierService") as service:
                with self.assertRaisesRegex(ValueError, "homogeneous historical|minimal input validation differs"):
                    local_train.training_parent_gate(self.root, self.root / "reload",
                        admission=local_train.EXPLORATORY_ADMISSION)
                service.assert_not_called()
            self.path.write_text(json.dumps(self.fixture.raw))


if __name__ == "__main__":
    unittest.main()
