"""Fixed cohort/control admission tests, not model-learning evidence."""
import copy
import json
import os
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fine_tuning.notebook_rl.local_profile import build_protocol
from fine_tuning.notebook_rl.notebook_learning import NOTEBOOK_REWARD_VERSION
from fine_tuning.notebook_rl.local_train import (
    EXPLORATORY_ADMISSION, STRICT_ADMISSION, TRAINING_BUDGET_VERSION, eligible_trees,
    execution_admission, execution_max_seconds, fact_credit_learning_gate, fixed_pairs,
    main, matched_notebook_arm, notebook_learning_gate,
    parent_gate, reload_training, train, training_binding, training_parent_gate,
)
from fine_tuning.tests.test_notebook_rl_policy import bundle
from fine_tuning.tests.test_notebook_rl_updates import tree


class LocalTrainTests(unittest.TestCase):
    def notebook_evidence(self):
        from fine_tuning.notebook_rl.episode_notebook import STRICT_ROUTE_RULE, audit_episode
        from fine_tuning.notebook_rl.schema import to_json
        from fine_tuning.tests.test_notebook_rl_episode_notebook import EpisodeNotebookTests, authored_note
        from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleEngine

        fixture = EpisodeNotebookTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.prompts["notebook_route_rule"] = STRICT_ROUTE_RULE
        fixture.prompts["notebook_feedback_policy"] = build_protocol()["notebook_feedback_policy"]
        fixture.prompts["notebook_observation_policy"] = build_protocol()["notebook_observation_policy"]
        record, _ = fixture.tree()
        note = authored_note(record, audit_episode(record, fixture.actor, fixture.verifier))
        engine = ReversibleEngine([["Observed facts:\ncell[a][b] = 8\nSolution:"],
                                  [json.dumps(note), "{}", json.dumps(note), "{}"]])
        fixture.append(record, engine)
        group = record.groups[-1]
        rows = [{"sample_id": sample.sample_id, "group_id": group.group_id, "advantage": .5,
                 "reward_gradient_isolated": True, "vision_on": True,
                 "action_tokens": len(sample.completion_token_ids),
                 "gradient_before": {g: [None] for g in ("L", "M", "V")},
                 "gradient_after": {"L": ["a"*64], "M": [None], "V": [None]},
                 "changed_groups": ["L"]} for sample in group.samples if sample.reward == 1]
        update = {"notebook_reward_enabled": True, "notebook_backward": rows, "step_before": 0,
                  "routes": {record.problem_id: True},
                  "effective_rewards": {group.group_id: [s.reward for s in group.samples]}}
        result = {"notebook_signal_groups": 1, "notebook_gradient_contribution_verified": True}
        raw = {"kind": "TREE", "raw": json.loads(to_json(record))}
        path = fixture.root / "raw-0000.json"
        path.write_text(json.dumps(raw))
        return fixture.root, path, raw, result, update

    def test_valid_note_is_bound_to_positive_isolated_backward(self):
        root, path, _, result, update = self.notebook_evidence()
        self.assertEqual(notebook_learning_gate(root, result, update, build_protocol()), [path])

    def test_legacy_notebook_evidence_cannot_admit_strict_v5_training(self):
        from fine_tuning.notebook_rl.rollout import _hash

        root, path, raw, result, update = self.notebook_evidence()
        note = raw["raw"]["notebook"]["rewarded_notebook"]
        audit = note["episode_audit"]
        audit["routing_rule"] = "all-observed-required-slots-v1"
        del audit["admission"]
        note["episode_audit_sha256"] = _hash(audit)
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "strict notebook routing admission"):
            notebook_learning_gate(root, result, update, build_protocol())

    def test_no_note_signal_or_contribution_cannot_admit_training(self):
        root, _, _, result, update = self.notebook_evidence()
        for key, value in (("notebook_signal_groups", 0), ("notebook_signal_groups", True),
                           ("notebook_gradient_contribution_verified", False)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "notebook reward-gradient"):
                notebook_learning_gate(root, result | {key: value}, update, build_protocol())

    def test_kl_only_negative_or_claimed_gradient_does_not_pass(self):
        root, _, _, result, update = self.notebook_evidence()
        for key, value in (("advantage", -.5),
                           ("advantage", float("nan")), ("action_tokens", 0), ("action_tokens", 1),
                           ("changed_groups", ["V"]), ("vision_on", False),
                           ("gradient_after", {g: [None] for g in ("L", "M", "V")})):
            changed = copy.deepcopy(update)
            changed["notebook_backward"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "notebook reward-gradient"):
                notebook_learning_gate(root, result, changed, build_protocol())
        changed = copy.deepcopy(update)
        changed["notebook_backward"][0]["reward_gradient_isolated"] = False
        self.assertEqual(len(notebook_learning_gate(root, result, changed, build_protocol())), 1)
        for row in changed["notebook_backward"]:
            row["reward_gradient_isolated"] = False
        with self.assertRaisesRegex(ValueError, "notebook reward-gradient"):
            notebook_learning_gate(root, result, changed, build_protocol())
        changed = copy.deepcopy(update)
        changed["notebook_backward"] = []
        with self.assertRaisesRegex(ValueError, "notebook reward-gradient"):
            notebook_learning_gate(root, result, changed, build_protocol())

    def test_other_policy_route_and_invalid_gradient_hash_cannot_supply_evidence(self):
        root, _, _, result, update = self.notebook_evidence()
        for key, value in (("step_before", 1), ("routes", dict.fromkeys(update["routes"], False))):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "policy or route differs"):
                notebook_learning_gate(root, result, update | {key: value}, build_protocol())
        changed = copy.deepcopy(update)
        changed["notebook_backward"][0]["gradient_after"]["L"] = ["not-a-tensor-hash"]
        with self.assertRaisesRegex(ValueError, "incomplete notebook gradient"):
            notebook_learning_gate(root, result, changed, build_protocol())

    def test_changed_note_checks_rewards_or_duplicate_evidence_is_rejected(self):
        root, path, raw, result, update = self.notebook_evidence()
        for key, value in (("text", "another note"), ("reward", 0), ("reward_version", "old"),
                           ("eligible_for_rl_loss", False),
                           ("checks", {k: 1 for k in raw["raw"]["notebook"]["model_notes"][0]["checks"]})):
            changed = copy.deepcopy(raw)
            changed["raw"]["notebook"]["model_notes"][0][key] = value
            path.write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "notebook reward-gradient|source binding"):
                notebook_learning_gate(root, result, update, build_protocol())
        path.write_text(json.dumps(raw))
        altered = copy.deepcopy(update)
        altered["effective_rewards"][next(iter(altered["effective_rewards"]))] = [1., 1., 1., 1.]
        with self.assertRaisesRegex(ValueError, "effective rewards differ"):
            notebook_learning_gate(root, result, altered, build_protocol())
        altered = copy.deepcopy(update)
        altered["notebook_backward"].append(altered["notebook_backward"][0])
        with self.assertRaisesRegex(ValueError, "duplicate notebook backward"):
            notebook_learning_gate(root, result, altered, build_protocol())
        (root / "raw-0001.json").write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "duplicate notebook tree"):
            notebook_learning_gate(root, result, update, build_protocol())

    def test_diagnostic_parent_does_not_require_note_learning_but_training_does(self):
        protocol = json.loads(json.dumps(build_protocol()))
        launch = {"protocol": protocol, "standalone_base": True, "pid": 123,
                  "purpose": "MINIMAL_TWO_PROBLEM_ONE_UPDATE", "problem_ids": ["a", "b"], "binding": {}}
        result = {"status": "UPDATED_SAVED_PENDING_FRESH_RELOAD", "optimizer_updates": 1,
                  "original_access_guard": {"denied_original_access_attempts": 0}}
        restored = {"status": "FRESH_PROCESS_RELOAD_VERIFIED", "capture_pid": 123, "pid": 456, "step": 1,
                    "original_access_guard": {"denied_original_access_attempts": 0},
                    **{key: True for key in ("all_trainable_hashes_match", "optimizer_restored",
                       "actual_forward_and_generation_match", "frozen_quantized_base_unchanged")}}
        expected = {"step": 1, "trainable_sha256": {}, "frozen_base_sha256": "base",
                    "quantization": {}, "forward_probes": {}}
        metadata = {"step": 1, "state_sha256": {}, "base_sha256": "base", "inventory": {"quantization": {}}}
        values = [launch, result, restored, expected, {}, {"before": {}, "after": {}}, {}, {}]
        for require_note in (False, True):
            with patch("fine_tuning.notebook_rl.local_train._read", side_effect=values), \
                    patch("psutil.pid_exists", return_value=False), \
                    patch("fine_tuning.notebook_rl.local_train.verify_changes", return_value={}), \
                    patch("fine_tuning.notebook_rl.checkpoint.read_metadata", return_value=metadata), \
                    patch("fine_tuning.notebook_rl.checkpoint.file_hash", return_value="hash"), \
                    patch("fine_tuning.notebook_rl.local_train.notebook_learning_gate", return_value=[Path("/tmp/tree")]) as notes:
                gate = parent_gate(Path("/tmp/capture"), Path("/tmp/reload"), require_notebook_learning=require_note)
                self.assertIs(gate["notebook_reward_learning_verified"], require_note)
                self.assertEqual(notes.call_count, int(require_note))
                self.assertEqual(len(gate["files"]), 14 + int(require_note))

    def test_both_training_arms_require_notebook_evidence_before_loading_a_model(self):
        for arm in ("notebook", "answer-only"):
            with patch("fine_tuning.notebook_rl.local_train.parent_gate", side_effect=ValueError("note gate")) as gate, \
                    patch("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy") as model:
                with self.assertRaisesRegex(ValueError, "note gate"):
                    train(Path("/tmp/prepared"), Path("/tmp/base"), Path("/tmp/cache"),
                          Path("/tmp/minimum"), Path("/tmp/reload"), Path("/tmp/output"), arm=arm,
                          match_initialization=Path("/tmp/initialization.json") if arm == "answer-only" else None)
                gate.assert_called_once_with(Path("/tmp/minimum"), Path("/tmp/reload"), require_notebook_learning=True)
                model.assert_not_called()

    def test_checkpoint_binding_changes_when_the_reward_control_or_cohort_changes(self):
        prepared, protocol = {"actors": "fixed-ten"}, build_protocol()
        contract = {"arm": "notebook", "pair_ids": [["a", "b"]]}
        first = training_binding(prepared, protocol, contract)
        control = training_binding(prepared, protocol, contract | {"arm": "answer-only"})
        self.assertNotEqual(first["data_sha256"], control["data_sha256"])
        self.assertNotEqual(first["data_sha256"], training_binding(
            {"actors": "different-ten"}, protocol, contract)["data_sha256"])
        self.assertEqual(first["protocol_sha256"], control["protocol_sha256"])

    def test_context_variant_changes_checkpoint_identity_without_changing_training_budget(self):
        prepared = {"actors": "fixed-ten"}
        contract = {"arm": "notebook", "pair_ids": [["a", "b"]]}
        short = build_protocol(verified_note_fact_credit=True)
        long = build_protocol(verified_note_fact_credit=True, context_limit=24576)
        first = training_binding(prepared, short, contract)
        second = training_binding(prepared, long, contract)
        self.assertNotEqual(first["protocol_sha256"], second["protocol_sha256"])
        self.assertEqual(first["data_sha256"], second["data_sha256"])
        self.assertEqual(execution_max_seconds(contract, short), execution_max_seconds(contract, long))

    def test_control_rejects_unrelated_initialization_before_checkpoint_read(self):
        prepared, protocol = {"actors": "fixed-ten"}, build_protocol()
        with patch("fine_tuning.notebook_rl.local_train._read", side_effect=[
                {"execution_contract": {"arm": "notebook"}}, {"status": "FAILED"}]), \
                patch("fine_tuning.notebook_rl.checkpoint.read_metadata") as metadata:
            with self.assertRaisesRegex(ValueError, "control must match"):
                matched_notebook_arm(Path("/tmp/other/initialization.json"), prepared, protocol,
                    Path("/tmp/base"), Path("/tmp/cache"), [])
            metadata.assert_not_called()

    def test_control_rejects_replaced_closed_initialization(self):
        prepared, protocol = {"actors": "fixed-ten"}, build_protocol()
        result = {"status": "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD", "arm": "notebook",
                  "initialization_sha256": "closed-original"}
        launch = {"execution_contract": {"arm": "notebook"}}
        with patch("fine_tuning.notebook_rl.local_train._read", side_effect=[launch, result]), \
                patch("fine_tuning.notebook_rl.checkpoint.file_hash", return_value="replaced-current"), \
                patch("fine_tuning.notebook_rl.checkpoint.read_metadata") as metadata:
            with self.assertRaisesRegex(ValueError, "control must match"):
                matched_notebook_arm(Path("/tmp/replaced/initialization.json"), prepared, protocol,
                    Path("/tmp/base"), Path("/tmp/cache"), [])
            metadata.assert_not_called()

    def test_fixed_ten_consumed_in_order_without_replacement(self):
        actors = [SimpleNamespace(problem_id=str(i)) for i in range(10)]
        pairs = fixed_pairs(actors)
        self.assertEqual([a for pair in pairs for a in pair], actors)
        self.assertEqual([len(p) for p in pairs], [2]*5)
        with self.assertRaisesRegex(ValueError, "ten unique"):
            fixed_pairs(actors[:-1])
        with self.assertRaisesRegex(ValueError, "ten unique"):
            fixed_pairs(actors[:-1] + [actors[0]])

    def test_note_only_variance_is_not_an_answer_control_signal(self):
        record = tree(bundle())
        note = copy.deepcopy(record.groups[0])
        note.group_id += ":note"
        note.kind = "DIAG"
        for sample in note.samples:
            sample.kind = "DIAG"
            sample.group_id = note.group_id
            sample.sample_id += ":note"
            sample.diag_reward, sample.result = sample.reward, None
        for sample in record.groups[0].samples:
            sample.reward = 0.
        record.groups.append(note)
        record.notebook["rewarded_notebook"] = {"status": "GENERATED", "group_id": note.group_id,
            "reward_version": NOTEBOOK_REWARD_VERSION}
        original = copy.deepcopy(record)
        self.assertEqual(eligible_trees([record], "notebook"), [record])
        self.assertEqual(eligible_trees([record], "answer-only"), [])
        self.assertEqual(record, original)
        record.notebook["rewarded_notebook"]["reward_version"] = "historical-unmatched-notebook-reward"
        self.assertEqual(eligible_trees([record], "notebook"), [])
        with self.assertRaisesRegex(ValueError, "explicit"):
            eligible_trees([record], "pick-a-winning-arm")

    def test_parent_gate_rejects_a_live_or_incomplete_minimal_run(self):
        protocol = json.loads(json.dumps(build_protocol()))
        launch = {"protocol": protocol, "standalone_base": True, "pid": os.getpid(),
            "purpose": "MINIMAL_TWO_PROBLEM_ONE_UPDATE", "problem_ids": ["single", "pair"]}
        for live in (True, False):
            values = [launch, {"status": "FAILED", "optimizer_updates": 0}, {}, {}, {}, {}]
            with patch("fine_tuning.notebook_rl.local_train._read", side_effect=values), \
                    patch("psutil.pid_exists", return_value=live), \
                    patch("fine_tuning.notebook_rl.checkpoint.read_metadata") as metadata:
                with self.assertRaisesRegex(ValueError, "completed real minimal"):
                    parent_gate(Path("/tmp/capture"), Path("/tmp/reload"))
                metadata.assert_not_called()


class HistoricalMatchedNotebookTests(unittest.TestCase):
    """Real files/bindings/checksums; byte fixtures are not learned model evidence."""

    def setUp(self):
        from fine_tuning.notebook_rl import checkpoint

        self.checkpoint = checkpoint
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "notebook/initialization.json"
        self.path.parent.mkdir()
        self.path.write_text('{"fixture_only":true}')
        self.prepared, self.protocol = {"actors": "fixed-ten-fixture"}, build_protocol()
        self.pairs = fixed_pairs([SimpleNamespace(problem_id=str(i)) for i in range(10)])
        self.contract = {"arm": "notebook", "notebook_reward_enabled": True,
                         "pair_ids": [[a.problem_id for a in pair] for pair in self.pairs]}
        self.current_binding = training_binding(self.prepared, self.protocol, self.contract)
        self.historical_code = checkpoint.digest({"fixture_historical_training_code": "v1"})
        self.assertNotEqual(self.current_binding["code_sha256"], self.historical_code)
        binding = self.current_binding | {"code_sha256": self.historical_code}
        self.model, self.cache = self.root / "base", self.root / "cache"
        self.launch = {"execution_contract": self.contract,
            "execution_contract_sha256": checkpoint.digest(self.contract), "binding": binding,
            "protocol": json.loads(json.dumps(self.protocol)), "preparation_sha256": checkpoint.digest(self.prepared),
            "model_path": str(self.model), "cache_check": str(self.cache)}
        self.result = {"status": "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD", "arm": "notebook",
            "optimizer_updates": 2, "initialization_sha256": checkpoint.file_hash(self.path)}
        self.write_json(self.path.parent / "launch.json", self.launch)
        self.write_json(self.path.parent / "result.json", self.result)
        self.ckpt = self.path.parent / "checkpoint"
        self.ckpt.mkdir()
        for name in ("adapter.safetensors", "mergers.safetensors", "runtime.pt"):
            (self.ckpt / name).write_bytes(b"checksum fixture only; not a model payload")
        self.metadata = {"format": checkpoint.FORMAT, "base_id": checkpoint.BASE_ID,
            "base_revision": checkpoint.BASE_REVISION, "arm": "MOD-RV", "binding": binding,
            "protocol": json.loads(json.dumps(self.protocol)), "resolved_protocol_sha256": checkpoint.digest(self.protocol),
            "inventory": {"fixture_only": True}, "base_sha256": "d" * 64, "state_sha256": {},
            "optimizer_names": [], "step": 2, "cursor": {"epoch": 0, "position": 10, "discarded_problem_ids": []},
            "vision_on": True, "files": {path.name: checkpoint.file_hash(path) for path in self.ckpt.iterdir()}}
        self.write_metadata()

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, sort_keys=True))

    def write_metadata(self):
        self.write_json(self.ckpt / "metadata.json", self.metadata)
        (self.ckpt / "metadata.sha256").write_text(self.checkpoint.file_hash(self.ckpt / "metadata.json"))

    def match(self, **kwargs):
        return matched_notebook_arm(self.path, self.prepared, self.protocol, self.model, self.cache, self.pairs, **kwargs)

    def test_historical_evaluation_keeps_training_default_strict_and_files_unchanged(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "control must match"):
            self.match()
        receipt = self.match(historical_code_sha256=self.historical_code)
        self.assertEqual(receipt, {"initialization": {"path": str(self.path),
            "sha256": self.checkpoint.file_hash(self.path)},
            "launch_sha256": self.checkpoint.file_hash(self.path.parent / "launch.json"),
            "result_sha256": self.checkpoint.file_hash(self.path.parent / "result.json")})
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        for value in ("0" * 64, "malformed", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.match(historical_code_sha256=value)

    def test_historical_mode_preserves_data_metadata_and_stored_receipt_checks(self):
        receipt = self.match(historical_code_sha256=self.historical_code)
        with self.assertRaisesRegex(ValueError, "control must match"):
            matched_notebook_arm(self.path, {"actors": "different-ten"}, self.protocol,
                self.model, self.cache, self.pairs, historical_code_sha256=self.historical_code)
        self.launch["fixture_note"] = "changed after the control stored its matching receipt"
        self.write_json(self.path.parent / "launch.json", self.launch)
        self.assertNotEqual(receipt, self.match(historical_code_sha256=self.historical_code))
        (self.ckpt / "adapter.safetensors").write_bytes(b"corrupted checksum fixture")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.match(historical_code_sha256=self.historical_code)

    def test_current_training_binding_still_works_and_metadata_protocol_must_match_capture(self):
        self.launch["binding"] = self.metadata["binding"] = self.current_binding
        self.write_json(self.path.parent / "launch.json", self.launch)
        self.write_metadata()
        self.assertEqual(self.match()["launch_sha256"], self.checkpoint.file_hash(self.path.parent / "launch.json"))
        self.metadata["protocol"]["fixture_protocol_change"] = True
        self.metadata["resolved_protocol_sha256"] = self.checkpoint.digest(self.metadata["protocol"])
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "inconsistent update evidence"):
            self.match()

    def test_control_requires_same_admission_and_parent_proof(self):
        self.contract["parent_gate"] = {"fixture_proof": "original"}
        with self.assertRaisesRegex(ValueError, "admission and parent"):
            self.match(historical_code_sha256=self.historical_code, parent={"fixture_proof": "changed"})
        with self.assertRaisesRegex(ValueError, "admission and parent"):
            self.match(historical_code_sha256=self.historical_code, admission=EXPLORATORY_ADMISSION)

    def test_matched_control_cannot_cross_context_even_with_verified_historical_source(self):
        for limit, other_limit in ((16384, 24576), (24576, 16384)):
            with self.subTest(context_limit=limit):
                self.protocol = build_protocol(verified_note_fact_credit=True, context_limit=limit)
                self.launch["protocol"] = self.metadata["protocol"] = json.loads(json.dumps(self.protocol))
                self.launch["binding"] = self.metadata["binding"] = training_binding(
                    self.prepared, self.protocol, self.contract) | {"code_sha256": self.historical_code}
                self.metadata["resolved_protocol_sha256"] = self.checkpoint.digest(self.protocol)
                self.write_json(self.path.parent / "launch.json", self.launch)
                self.write_metadata()
                self.match(historical_code_sha256=self.historical_code)
                before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
                changed = build_protocol(verified_note_fact_credit=True, context_limit=other_limit)
                with patch("fine_tuning.notebook_rl.checkpoint.read_metadata") as metadata:
                    with self.assertRaisesRegex(ValueError, "control must match"):
                        matched_notebook_arm(self.path, self.prepared, changed, self.model, self.cache,
                            self.pairs, historical_code_sha256=self.historical_code)
                    metadata.assert_not_called()
                self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_control_requires_requested_budget_and_accepts_historical_default(self):
        self.match(historical_code_sha256=self.historical_code, max_seconds=14400)
        with self.assertRaisesRegex(ValueError, "training time budget"):
            self.match(historical_code_sha256=self.historical_code, max_seconds=60000)
        self.contract["training_budget"] = {"version": TRAINING_BUDGET_VERSION, "max_seconds": 60000}
        self.launch["execution_contract_sha256"] = self.checkpoint.digest(self.contract)
        self.launch["binding"] = self.metadata["binding"] = training_binding(
            self.prepared, self.protocol, self.contract) | {"code_sha256": self.historical_code}
        self.write_json(self.path.parent / "launch.json", self.launch)
        self.write_metadata()
        self.match(historical_code_sha256=self.historical_code, max_seconds=60000)
        with self.assertRaisesRegex(ValueError, "training time budget"):
            self.match(historical_code_sha256=self.historical_code, max_seconds=14400)


class ExecutionBudgetTests(unittest.TestCase):
    protocol = {"local_budget": {"max_seconds_per_run": 14400}}

    def test_legacy_default_explicit_bounds_and_contract_binding(self):
        self.assertEqual(execution_max_seconds({}, self.protocol), 14400)
        for seconds in (1, 14400, 60000, 86400):
            contract = {"training_budget": {"version": TRAINING_BUDGET_VERSION, "max_seconds": seconds}}
            self.assertEqual(execution_max_seconds(contract, self.protocol), seconds)
        with patch("fine_tuning.notebook_rl.local_train._binding", return_value={"fixture": "binding"}):
            first = training_binding({}, self.protocol, {"training_budget": {
                "version": TRAINING_BUDGET_VERSION, "max_seconds": 14400}})
            second = training_binding({}, self.protocol, {"training_budget": {
                "version": TRAINING_BUDGET_VERSION, "max_seconds": 60000}})
        self.assertNotEqual(first["data_sha256"], second["data_sha256"])
        self.assertEqual(self.protocol["local_budget"]["max_seconds_per_run"], 14400)

    def test_bad_types_ranges_versions_and_extra_keys_reject_before_binding(self):
        invalid = [{"version": TRAINING_BUDGET_VERSION, "max_seconds": value}
                   for value in (0, -1, 86401, True, False, 14400., "14400", None)]
        invalid += [None, {}, {"version": "unknown", "max_seconds": 14400},
                    {"version": TRAINING_BUDGET_VERSION, "max_seconds": 14400, "automatic_extension": True}]
        for budget in invalid:
            with self.subTest(budget=budget), patch("fine_tuning.notebook_rl.local_train._binding") as binding:
                with self.assertRaisesRegex(ValueError, "versioned training budget"):
                    training_binding({}, self.protocol, {"training_budget": budget})
                binding.assert_not_called()

    def test_cli_default_and_explicit_budget_reach_train(self):
        arguments = ["train", "--preparation", "/tmp/preparation", "--model-path", "/tmp/base",
            "--cache-check", "/tmp/cache", "--minimal", "/tmp/minimum", "--minimal-reload", "/tmp/reload",
            "--output", "/tmp/output", "--arm", "notebook"]
        for flags, seconds in (([], 14400), (["--max-seconds", "60000"], 60000)):
            with patch("fine_tuning.notebook_rl.local_train.train", return_value={
                    "status": "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD"}) as run, patch("builtins.print"):
                self.assertEqual(main(arguments + flags), 0)
                self.assertEqual(run.call_args.kwargs["max_seconds"], seconds)
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            main(arguments + ["--max-seconds", "3.5"])


class ExploratoryAdmissionTests(unittest.TestCase):
    def setUp(self):
        from fine_tuning.notebook_rl import updates
        from fine_tuning.notebook_rl.fact_credit import group_credit
        from fine_tuning.notebook_rl.rollout import _hash
        from fine_tuning.notebook_rl.schema import to_json
        from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleEngine
        from fine_tuning.tests.test_notebook_rl_notebook_objective import NotebookObjectiveTests

        fixture = NotebookObjectiveTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol = build_protocol(verified_note_fact_credit=True)
        fixture.objective = fixture.protocol["notebook_objective"]
        fixture.fixture.prompts.update({key: fixture.protocol[key] for key in
            ("notebook_route_rule", "notebook_feedback_policy", "notebook_observation_policy")})
        partial, wrong = copy.deepcopy(fixture.note), copy.deepcopy(fixture.note)
        partial["corrected_facts"] = {"cell[a][b]": "7"}
        wrong["corrected_facts"] = {"cell[a][b]": "999", "cell[c][d]": "999"}
        fixture.engine = ReversibleEngine([["Observed facts:\ncell[a][b] = 7\ncell[c][d] = 5\nSolution:"],
                                          [json.dumps(partial), json.dumps(wrong)] * 2])
        groups = []
        fixture.fixture.append(fixture.tree, fixture.engine, on_group=groups.append,
                               should_stop=lambda: bool(groups))
        fixture.group = fixture.tree.groups[-1]
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier, fixture.objective)
        self.fixture = fixture
        self.protocol = fixture.protocol
        self.root = fixture.fixture.root
        self.path = self.root / "raw-0000.json"
        self.raw = {"kind": "TREE", "raw": json.loads(to_json(fixture.tree))}
        self.path.write_text(json.dumps(self.raw))
        scores = updates.notebook_objective_scores(fixture.tree, fixture.objective)
        credits = group_credit(fixture.group.samples, scores, fixture.engine.tokenizer)
        self.update = {"notebook_reward_enabled": True, "notebook_objective": fixture.objective,
            "notebook_objective_receipts": {fixture.tree.problem_id: fixture.tree.notebook["learning_objective"]},
            "step_before": 0, "routes": {fixture.tree.problem_id: True},
            "effective_rewards": {fixture.group.group_id: [scores[s.sample_id]["effective_reward"]
                                                          for s in fixture.group.samples]},
            "fact_token_credit": credits,
            "mean_log_ratio_by_sample": {s.sample_id: 0.0 for s in fixture.group.samples},
            "notebook_backward": []}
        for sample in fixture.group.samples:
            credit, score = credits[sample.sample_id], scores[sample.sample_id]
            self.update["notebook_backward"].append({"sample_id": sample.sample_id,
                "group_id": fixture.group.group_id, "advantage": credit["strict_advantage"],
                "strict_reward": sample.reward, "atomic_reward": score["atomic_reward"],
                "effective_reward": score["effective_reward"], "objective_score_sha256": _hash(score),
                "fact_credit_sha256": _hash(credit),
                "nonzero_partial_tokens": sum(v != 0 for v in credit["partial_token_advantages"]),
                "action_tokens": len(sample.completion_token_ids), "vision_on": True,
                "gradient_before": {g: [None] for g in ("L", "M", "V")},
                "gradient_after": {"L": ["a" * 64], "M": [None], "V": [None]},
                "changed_groups": ["L"], "reward_gradient_isolated": True})
        self.result = {"notebook_signal_groups": 1, "notebook_gradient_contribution_verified": True}

    def gate(self):
        fixture = self.fixture
        return fact_credit_learning_gate(self.root, self.result, self.update, self.protocol,
            actors={fixture.tree.problem_id: fixture.actor}, verifier=fixture.verifier,
            tokenizer=fixture.engine.tokenizer)

    def test_partial_fact_learning_admits_exploration_without_strict_success(self):
        self.assertTrue(all(s.reward == 0 for s in self.fixture.group.samples))
        self.assertEqual(self.gate(), [self.path])
        with self.assertRaisesRegex(ValueError, "positive notebook reward-gradient"):
            notebook_learning_gate(self.root, self.result, self.update, self.protocol)

    def test_explicit_version_is_v16_only_and_default_stays_strict(self):
        self.assertEqual(execution_admission({}, self.protocol), STRICT_ADMISSION)
        self.assertEqual(execution_admission({"admission_policy": EXPLORATORY_ADMISSION}, self.protocol),
                         EXPLORATORY_ADMISSION)
        for value in (None, "unknown", True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "admission policy"):
                execution_admission({"admission_policy": value}, self.protocol)
        with self.assertRaisesRegex(ValueError, "admission policy"):
            execution_admission({"admission_policy": EXPLORATORY_ADMISSION}, build_protocol())

    def test_answer_kl_and_negative_only_evidence_cannot_admit(self):
        original = copy.deepcopy(self.update)
        for mutation in ("answer-only", "kl-only", "negative-only", "missing-backward", "fake-gradient"):
            self.update = copy.deepcopy(original)
            if mutation == "answer-only":
                self.update["notebook_reward_enabled"] = False
            elif mutation == "missing-backward":
                self.update["notebook_backward"] = []
            else:
                for row in self.update["notebook_backward"]:
                    if mutation == "kl-only":
                        row["reward_gradient_isolated"] = False
                    elif mutation == "fake-gradient":
                        row["gradient_after"]["L"] = ["invalid-hash"]
                    elif any(v > 0 for v in self.update["fact_token_credit"][row["sample_id"]]["partial_token_advantages"]):
                        self.update["mean_log_ratio_by_sample"][row["sample_id"]] = 1.0
                if mutation == "negative-only":
                    self.update["notebook_backward"] = [r for r in self.update["notebook_backward"]
                        if self.update["mean_log_ratio_by_sample"][r["sample_id"]] == 0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.gate()

    def test_full_group_means_survive_mis_and_forged_receipts_reject(self):
        from fine_tuning.notebook_rl.rollout import _hash

        rejected = self.fixture.group.samples[1].sample_id
        self.update["mean_log_ratio_by_sample"][rejected] = 1.0
        self.update["notebook_backward"] = [r for r in self.update["notebook_backward"] if r["sample_id"] != rejected]
        self.assertEqual(self.gate(), [self.path])
        original = copy.deepcopy(self.update)
        for mutation in ("credit", "score", "retention", "duplicate"):
            self.update = copy.deepcopy(original)
            if mutation == "credit":
                next(iter(self.update["fact_token_credit"].values()))["full_group_size"] -= 1
            elif mutation == "score":
                fresh_raw = copy.deepcopy(self.raw)
                receipt = fresh_raw["raw"]["notebook"]["learning_objective"]
                sample = self.fixture.group.samples[0].sample_id
                receipt["samples"][sample]["fact_rewards"]["cell[a][b]"] = 0
                receipt["sha256"] = _hash({k: v for k, v in receipt.items() if k != "sha256"})
                self.path.write_text(json.dumps(fresh_raw))
                self.update["notebook_objective_receipts"][self.fixture.tree.problem_id] = receipt
            elif mutation == "retention":
                self.update["mean_log_ratio_by_sample"][rejected] = 0
            else:
                self.update["notebook_backward"].append(self.update["notebook_backward"][0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.gate()
            self.path.write_text(json.dumps(self.raw))

    def test_mis_fp32_inclusive_boundaries_match_production(self):
        import torch

        threshold = torch.tensor(self.protocol["loss"]["mismatch"]["delta_seq"], dtype=torch.float32)
        sample_id = self.fixture.group.samples[0].sample_id
        for sign in (1, -1):
            self.update["mean_log_ratio_by_sample"][sample_id] = float(sign * threshold)
            self.assertEqual(self.gate(), [self.path])
            outside = torch.nextafter(threshold, torch.tensor(float("inf")))
            self.update["mean_log_ratio_by_sample"][sample_id] = float(sign * outside)
            with self.assertRaisesRegex(ValueError, "MIS-rejected"):
                self.gate()

    def test_parent_dispatch_keeps_real_update_and_reload_gate_mandatory(self):
        for admission, require in ((STRICT_ADMISSION, True), (EXPLORATORY_ADMISSION, False)):
            with patch("fine_tuning.notebook_rl.local_train.parent_gate", side_effect=ValueError("real update required")) as gate:
                with self.assertRaisesRegex(ValueError, "real update required"):
                    training_parent_gate(self.root, self.root / "reload", admission=admission)
                gate.assert_called_once_with(self.root, self.root / "reload", **(
                    {"require_notebook_learning": True} if require else {}))

    def test_exploratory_parent_uses_fresh_verifier_and_rejects_other_vault_identity(self):
        from fine_tuning.notebook_rl import checkpoint
        from fine_tuning.notebook_rl.local_data import ACTOR_MODE_POLICY
        from fine_tuning.notebook_rl.schema import to_json
        from fine_tuning.notebook_rl.vault import read_lineage, write_split
        from fine_tuning.notebook_rl.verifier_service import VerifierService

        def save(name, value):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return {"path": str(path), "file_sha256": checkpoint.file_hash(path)}

        root = self.root / "parent-build"
        actor = self.fixture.actor
        write_split(root / "train", [actor], list(self.fixture.verifier.items.values()),
                    list(read_lineage(self.root / "atoms").values()))
        request = save("request.json", {"build_root": str(root),
            "index": {"path": str(self.root / "benchmark" / "index.json")}})
        actors = save("actors.json", [json.loads(to_json(actor))])
        save("result.json", self.result)
        save("update.json", self.update)
        for mismatch in (False, True):
            save("parent-build/manifest.yaml", {"split_manifests": {"train":
                "0" * 64 if mismatch else checkpoint.file_hash(root / "train" / "manifest.yaml")}})
            checked = save("checked.json", {"status": "VALIDATED_FOR_MEASUREMENT", "blockers": [],
                "measurement_allowed": True, "training_allowed": False,
                "input_binding": {"build_manifest_sha256": checkpoint.file_hash(root / "manifest.yaml")},
                "actors": [json.loads(to_json(actor))]})
            prepared = {"format": "notebook-rl-local-data-v1", "status": "LOCAL_DATA_REVALIDATED",
                "local_protocol_sha256": self.protocol["sha256"], "build_root": str(root),
                "data_request": request, "validated_data": checked, "actors": actors, "execution_actors": actors,
                "actor_projection": {"policy": ACTOR_MODE_POLICY, "changed_ids": []}}
            preparation = save("preparation.json", prepared)
            prerequisite = {"launch": {"protocol": json.loads(json.dumps(self.protocol)),
                "preparation": preparation["path"], "problem_ids": [actor.problem_id],
                "model_path": str(self.root / "unused-model"),
                "binding": {"data_sha256": checkpoint.digest(prepared),
                    "protocol_sha256": self.protocol["sha256"], "prompts_sha256": self.protocol["prompts_sha256"]}},
                "notebook_reward_learning_verified": False, "files": []}
            services = []

            def start_verifier(*args, **kwargs):
                service = VerifierService(*args, **kwargs)
                services.append(service)
                self.addCleanup(service.close)
                return service

            with self.subTest(mismatch=mismatch), \
                    patch("fine_tuning.notebook_rl.local_train.parent_gate", return_value=prerequisite) as parent, \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=self.fixture.engine.tokenizer) as tokenizer, \
                    patch("fine_tuning.notebook_rl.verifier_service.VerifierService", side_effect=start_verifier) as service_call, \
                    patch("fine_tuning.notebook_rl.local_train._install_actor_guards") as guards, \
                    patch("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy") as model:
                if mismatch:
                    with self.assertRaisesRegex(ValueError, "minimal verifier identity differs"):
                        training_parent_gate(self.root, self.root / "reload", admission=EXPLORATORY_ADMISSION)
                else:
                    gate = training_parent_gate(self.root, self.root / "reload", admission=EXPLORATORY_ADMISSION)
                    self.assertTrue(gate["positive_fact_credit_and_isolated_note_gradient_verified"])
                    self.assertFalse(gate["strict_notebook_correctness_verified"])
                    self.assertFalse(gate["notebook_reward_learning_verified"])
                    self.assertEqual(gate["admission_policy"], EXPLORATORY_ADMISSION)
                    self.assertIn({"path": str(self.path), "sha256": checkpoint.file_hash(self.path)}, gate["files"])
                parent.assert_called_once_with(self.root, self.root / "reload")
                tokenizer.assert_called_once_with(prerequisite["launch"]["model_path"],
                                                  local_files_only=True, trust_remote_code=False)
                forbidden = [self.root / "benchmark", Path.home() / "mmdl-data/evaluation"]
                service_call.assert_called_once_with(root / "train", forbidden_roots=forbidden, timeout=60)
                self.assertNotEqual(services[0].metadata["pid"], os.getpid())
                self.assertTrue(services[0].metadata["python_guards_installed"])
                self.assertIsNotNone(services[0]._process.poll())
                guards.assert_not_called()
                model.assert_not_called()

    def test_reload_rejects_unknown_or_tampered_admission_before_model_load(self):
        protocol = json.loads(json.dumps(build_protocol(verified_note_fact_credit=True)))
        for contract in ({"admission_policy": "unknown"}, {"admission_policy": EXPLORATORY_ADMISSION}):
            launch = {"protocol": protocol, "execution_contract": contract, "execution_contract_sha256": "tampered"}
            with patch("fine_tuning.notebook_rl.local_train._read", side_effect=[launch, {}, {}]), \
                    patch("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy") as model:
                with self.assertRaisesRegex(ValueError, "admission policy|execution contract changed"):
                    reload_training(self.root, self.root / "reload")
                model.assert_not_called()

    def test_reload_requires_exact_optimizer_collection_policy_before_model_load(self):
        from fine_tuning.notebook_rl import checkpoint

        protocol = build_protocol()
        launch = {"protocol": protocol, "execution_contract": {},
                  "execution_contract_sha256": checkpoint.digest({})}
        for policy in (None, "legacy", True, {"version": checkpoint.OPTIMIZER_PARKING_POLICY}):
            changed = launch if policy is None else launch | {"optimizer_state_collection_policy": policy}
            with self.subTest(policy=policy), \
                    patch("fine_tuning.notebook_rl.local_train._read", side_effect=[changed, {}]), \
                    patch.object(checkpoint, "load_resume") as load:
                with self.assertRaisesRegex(ValueError, "optimizer state collection policy differs"):
                    reload_training(self.root, self.root / "reload")
                load.assert_not_called()


class IntermediateCheckpointTests(unittest.TestCase):
    """Scripted CPU rollouts with real tiny optimizer/checkpoint payloads, not NF4 evidence."""

    def setUp(self):
        from fine_tuning.notebook_rl import checkpoint
        from fine_tuning.notebook_rl.policy import assert_base_unchanged
        from fine_tuning.notebook_rl.schema import ProblemTree

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.model = self.root / "data", self.root / "base"
        self.data.mkdir()
        self.model.mkdir()
        (self.data / "manifest.yaml").write_text(json.dumps({"split_manifests": {"train": "fixture-vault"}}))
        (self.model / "nf4_base.json").write_text('{"fixture_only":true}')
        self.output, self.cache = self.root / "output", self.root / "cache"
        self.protocol = build_protocol()
        self.bundle = bundle()
        self.bundle.inventory["quantization"] = {"fixture_only": True}
        self.actors = [SimpleNamespace(problem_id=str(i)) for i in range(10)]
        self.prepared = {"build_root": str(self.data)}
        parent = {"launch": {"protocol": self.protocol, "model_path": str(self.model), "cache_check": str(self.cache)}}
        self.now, self.stop_after, self.fail_sync_step, self.no_signal = 0., 1, None, False
        self.moments_parked = False
        self.parking_steps: list[int] = []
        self.collection_events: list[str] = []
        stack = self.enterContext(ExitStack())

        def patched(name, **kwargs):
            return stack.enter_context(patch(name, **kwargs))

        @contextmanager
        def park(model, optimizer, scheduler, protocol, step):
            self.assertIs(model, self.bundle)
            self.assertIs(scheduler.optimizer, optimizer)
            self.assertEqual(protocol, self.protocol)
            self.assertEqual(scheduler.last_epoch, step)
            self.assertFalse(self.moments_parked)
            self.moments_parked = True
            self.parking_steps.append(step)
            self.collection_events.append("park")

            def restore():
                if self.moments_parked:
                    self.collection_events.append("early-restore")
                self.moments_parked = False

            try:
                yield restore
            finally:
                self.moments_parked = False
                self.collection_events.append("restore")

        def build(actor, _images, _engine, _verifier, _prompts, _cfg, version, _rule, **_):
            self.assertTrue(self.moments_parked)
            self.collection_events.append("build:" + actor.problem_id)
            return ProblemTree(actor.problem_id, version, [], "VISION_OFF", {}, {})

        original_save = checkpoint.save

        def save(*args, **kwargs):
            self.assertFalse(self.moments_parked)
            self.collection_events.append("save")
            return original_save(*args, **kwargs)

        patched("fine_tuning.notebook_rl.checkpoint.park_adam_moments", side_effect=park)
        patched("fine_tuning.notebook_rl.checkpoint.save", side_effect=save)
        patched("fine_tuning.notebook_rl.local_train.training_parent_gate", return_value=parent)
        patched("fine_tuning.notebook_rl.local_train.cache_readiness",
                return_value={"original_model_root": str(self.root / "original")})
        patched("fine_tuning.notebook_rl.local_train.read_preparation", return_value=(self.prepared, {}, self.actors))
        patched("fine_tuning.notebook_rl.local_train.local_prompts", return_value={})
        patched("fine_tuning.notebook_rl.local_train._install_actor_guards", return_value=[])
        patched("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={"fixture_only": True})
        patched("fine_tuning.notebook_rl.quantized_cache.block_original_access",
                return_value={"denied_original_access_attempts": 0})
        patched("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy", return_value=self.bundle)
        patched("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged", side_effect=assert_base_unchanged)
        patched("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine",
                return_value=SimpleNamespace(policy_version=0))
        patched("fine_tuning.notebook_rl.local_train.load_images", side_effect=lambda *_: [MagicMock()])
        service = patched("fine_tuning.notebook_rl.verifier_service.VerifierService")
        service.return_value.__enter__.return_value.metadata = {"vault_manifest_sha256": "fixture-vault"}
        patched("fine_tuning.notebook_rl.local_train.make_probes", return_value=[])
        patched("fine_tuning.notebook_rl.rollout.RolloutConfig.from_protocol", return_value=object())
        patched("fine_tuning.notebook_rl.rollout.build_tree", side_effect=build)
        patched("fine_tuning.notebook_rl.notebook_learning.append_rewarded_notebooks", side_effect=self.collect_notebooks)
        patched("fine_tuning.notebook_rl.updates.partition_learning_trees", side_effect=self.partition)
        patched("fine_tuning.notebook_rl.updates.optimizer_step", side_effect=self.update)
        patched("fine_tuning.notebook_rl.trainer.synchronize", side_effect=self.synchronize)
        patched("fine_tuning.notebook_rl.local_train.time.monotonic", side_effect=lambda: self.now)
        for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
            patched("torch.cuda." + name, return_value=0)
        patched("builtins.print")

    def collect_notebooks(self, trees, *_args, **_kwargs):
        self.assertTrue(self.moments_parked)
        self.collection_events.append("notes:" + trees[0].problem_id)

    def partition(self, trees, **_):
        self.assertFalse(self.moments_parked)
        if self.no_signal:
            self.now = self.protocol["local_budget"]["max_seconds_per_run"] + 1
            return [], [], [t.problem_id for t in trees]
        return trees[:1], [], [trees[1].problem_id]

    def update(self, model, optimizer, scheduler, trees, *_args, step, before_optimizer_step, **_kwargs):
        # Moments stay parked through forward/backward and return only right before the commit.
        self.assertTrue(self.moments_parked)
        model.route.apply(False)
        sum(p.sum() for group in ("L", "M") for p in model.params[group]).backward()
        before_optimizer_step()
        self.assertFalse(self.moments_parked)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step + 1 == self.stop_after:
            self.now = self.protocol["local_budget"]["max_seconds_per_run"] + 1
        return {"status": "UPDATED_UNSYNCHRONIZED", "step_before": step, "step_after": step + 1,
            "notebook_backward": [], "routes": {t.problem_id: False for t in trees},
            "grad_norm": {"L": 1., "M": 1., "V": None}}

    def synchronize(self, _model, _preparation, engine, _probes, _protocol, step, *, push):
        self.assertFalse(self.moments_parked)
        if push and step == self.fail_sync_step:
            raise RuntimeError("fixture sync validation failed")
        engine.policy_version = step
        return [{"fixture_only": True, "step": step}]

    def run_training(self, **kwargs):
        return train(self.root / "preparation.json", self.model, self.cache,
            self.root / "minimum", self.root / "reload", self.output, arm="notebook", **kwargs)

    def test_update_then_timeout_preserves_real_payload_and_partial_cursor(self):
        import torch
        from fine_tuning.notebook_rl import checkpoint

        result = self.run_training()
        self.assertEqual((result["status"], result["optimizer_updates"]), ("BUDGET_STOP", 1))
        committed = result["last_committed_checkpoint"]
        path = Path(committed["path"])
        launch = json.loads((self.output / "launch.json").read_text())
        self.assertEqual(launch["optimizer_state_collection_policy"], checkpoint.OPTIMIZER_PARKING_POLICY)
        self.assertEqual(self.collection_events, ["park", "build:0", "notes:0", "restore",
                                                "park", "build:1", "notes:1", "restore",
                                                "park", "early-restore", "restore", "save"])
        self.assertEqual(launch["execution_contract"]["training_budget"], {
            "version": TRAINING_BUDGET_VERSION, "max_seconds": 14400})
        metadata = checkpoint.read_metadata(path, binding=launch["binding"])
        self.assertEqual(metadata["cursor"], {"epoch": 0, "position": 2, "discarded_problem_ids": ["1"],
            "problem_ids": [str(i) for i in range(10)], "consumed_problem_ids": ["0", "1"], "sync_pending": False})
        self.assertEqual(metadata["step"], 1)
        runtime = torch.load(path / "runtime.pt", weights_only=True)
        self.assertTrue(runtime["optimizer"]["state"])
        self.assertEqual(runtime["scheduler"]["last_epoch"], 1)
        self.assertEqual(set(runtime["rng"]), {"python", "numpy", "torch", "cuda"})
        self.assertEqual(committed["metadata_sha256"], checkpoint.file_hash(path / "metadata.json"))
        self.assertEqual(json.loads((self.output / "checkpoint-pair-00.json").read_text()), committed)
        self.assertEqual(json.loads((self.output / "result.json").read_text())["last_committed_checkpoint"], committed)
        self.assertFalse(committed["fixed_ten_training_complete"])
        self.assertFalse(committed["resume_supported"])
        self.assertFalse((self.output / "checkpoint").exists())
        self.assertFalse((self.output / "reload_expected.json").exists())
        launch["pid"] = 12345
        with patch("fine_tuning.notebook_rl.local_train._read", side_effect=[launch, result]) as read, \
                patch("psutil.pid_exists", return_value=False), \
                patch("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy") as load:
            with self.assertRaisesRegex(ValueError, "closed current-code fixed-ten"):
                reload_training(self.output, self.root / "partial-reload")
            self.assertEqual(read.call_count, 2)
            load.assert_not_called()

    def test_commits_are_unique_and_earlier_cursors_do_not_mutate(self):
        self.stop_after = 99
        calls = 0

        def partition(trees, **kwargs):
            nonlocal calls
            calls += 1
            self.no_signal = calls > 2
            return self.partition(trees, **kwargs)

        with patch("fine_tuning.notebook_rl.updates.partition_learning_trees", side_effect=partition):
            result = self.run_training()
        first = json.loads((self.output / "checkpoint-pair-00/metadata.json").read_text())
        second = json.loads((self.output / "checkpoint-pair-01/metadata.json").read_text())
        self.assertEqual(first["cursor"]["consumed_problem_ids"], ["0", "1"])
        self.assertEqual(first["cursor"]["discarded_problem_ids"], ["1"])
        self.assertEqual(second["cursor"]["consumed_problem_ids"], ["0", "1", "2", "3"])
        self.assertEqual(second["cursor"]["discarded_problem_ids"], ["1", "3"])
        self.assertEqual((first["step"], second["step"]), (1, 2))
        self.assertEqual(result["last_committed_checkpoint"]["pair_index"], 1)
        self.assertEqual(result["discarded_ids"], ["1", "3", "4", "5"])
        self.assertEqual(result["last_committed_checkpoint"]["cursor"], second["cursor"])
        self.assertEqual(self.parking_steps, [0, 0, 0, 1, 1, 1, 2, 2])

    def test_no_signal_creates_neither_update_nor_checkpoint(self):
        self.no_signal = True
        result = self.run_training()
        self.assertEqual((result["status"], result["optimizer_updates"]), ("BUDGET_STOP", 0))
        self.assertNotIn("last_committed_checkpoint", result)
        self.assertFalse(list(self.output.glob("checkpoint*")))
        self.assertFalse(self.moments_parked)
        self.assertEqual(self.parking_steps, [0, 0])
        self.assertEqual(self.collection_events[-1], "restore")

    def test_collection_deadline_restores_before_budget_return(self):
        def collect(*args, **kwargs):
            self.collect_notebooks(*args, **kwargs)
            self.now = self.protocol["local_budget"]["max_seconds_per_run"] + 1

        with patch("fine_tuning.notebook_rl.notebook_learning.append_rewarded_notebooks", side_effect=collect):
            result = self.run_training()
        self.assertEqual((result["status"], result["optimizer_updates"]), ("BUDGET_STOP", 0))
        self.assertFalse(self.moments_parked)
        self.assertEqual(self.collection_events[-1], "restore")
        self.assertFalse(list(self.output.glob("checkpoint*")))

    def test_collection_exception_restores_before_failure_record_without_update(self):
        for phase in ("build_tree", "append_rewarded_notebooks"):
            self.output = self.root / phase
            self.collection_events.clear()

            def fail(*_args, **_kwargs):
                self.assertTrue(self.moments_parked)
                raise RuntimeError("fixture collection failure")

            target = "rollout" if phase == "build_tree" else "notebook_learning"
            with self.subTest(phase=phase), patch("fine_tuning.notebook_rl." + target + "." + phase, side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "fixture collection failure"):
                    self.run_training()
            self.assertFalse(self.moments_parked)
            self.assertEqual(self.collection_events[-1], "restore")
            result = json.loads((self.output / "result.json").read_text())
            self.assertEqual((result["status"], result["optimizer_updates"]), ("FAILED", 0))
            self.assertFalse(list(self.output.glob("checkpoint*")))

    def test_failed_second_sync_preserves_first_commit_without_new_checkpoint(self):
        self.stop_after, self.fail_sync_step = 2, 2
        with self.assertRaisesRegex(RuntimeError, "sync validation failed"):
            self.run_training()
        result = json.loads((self.output / "result.json").read_text())
        self.assertEqual((result["status"], result["optimizer_updates"]), ("FAILED", 2))
        self.assertEqual(result["last_committed_checkpoint"]["step"], 1)
        self.assertTrue((self.output / "checkpoint-pair-00/metadata.sha256").is_file())
        self.assertFalse((self.output / "checkpoint-pair-01").exists())
        self.assertFalse((self.output / "checkpoint-pair-01.json").exists())

    def test_explicit_training_budget_controls_deadline_and_is_saved_without_protocol_mutation(self):
        from fine_tuning.notebook_rl.checkpoint import digest

        self.stop_after = 99
        protocol_before = copy.deepcopy(self.protocol)

        def update(*args, step, **kwargs):
            value = self.update(*args, step=step, **kwargs)
            self.now = 15000 if step == 0 else 60001
            return value

        with patch("fine_tuning.notebook_rl.updates.optimizer_step", side_effect=update):
            result = self.run_training(max_seconds=60000)
        self.assertEqual((result["status"], result["optimizer_updates"]), ("BUDGET_STOP", 2))
        launch = json.loads((self.output / "launch.json").read_text())
        self.assertEqual(launch["execution_contract"]["training_budget"], {
            "version": TRAINING_BUDGET_VERSION, "max_seconds": 60000})
        self.assertEqual(launch["execution_contract_sha256"], digest(launch["execution_contract"]))
        self.assertEqual(self.protocol, protocol_before)
        self.assertEqual(launch["protocol"], json.loads(json.dumps(protocol_before)))


class ResumeParentHostTests(unittest.TestCase):
    """A resume parent from another boot is closed even if its PID number is live (or equal) on this host."""

    def parent(self, root: Path, **record) -> tuple[Path, dict, dict, dict]:
        from fine_tuning.notebook_rl.local_run import host_identity

        protocol, contract = {"protocol_id": "fixture"}, {"arm": "notebook"}
        binding = {"protocol_sha256": "a", "prompts_sha256": "b", "code_sha256": "c"}
        new = {"pid": os.getpid(), "host": host_identity(), "protocol": protocol, "binding": binding,
               "preparation_sha256": "p"}
        old = {"pid": os.getpid(), "protocol": protocol, "execution_contract": contract, "binding": binding,
               "preparation_sha256": "p", **record}
        (root / "launch.json").write_text(json.dumps(old))
        return root, protocol, new, contract

    def test_parent_from_another_boot_is_accepted_despite_a_live_equal_pid(self):
        from fine_tuning.notebook_rl.local_run import host_identity
        from fine_tuning.notebook_rl.local_train import _resume_parent

        with tempfile.TemporaryDirectory() as directory, patch("psutil.pid_exists", return_value=True):
            parent, protocol, launch, contract = self.parent(
                Path(directory), host={**host_identity(), "boot_id": "previous-boot"})
            old, _ = _resume_parent(parent, protocol, launch, contract)
            self.assertEqual(old["pid"], launch["pid"])

    def test_same_host_live_or_same_process_parent_and_legacy_live_parent_are_refused(self):
        from fine_tuning.notebook_rl.local_run import host_identity
        from fine_tuning.notebook_rl.local_train import _resume_parent

        for label, record, alive in (("same process", {"host": host_identity()}, False),
                                     ("same host, live pid", {"host": host_identity(), "pid": os.getpid() + 100000}, True),
                                     ("legacy, live pid", {"pid": os.getpid() + 100000}, True),
                                     ("legacy, same pid", {}, False)):
            with self.subTest(label), tempfile.TemporaryDirectory() as directory, \
                    patch("psutil.pid_exists", return_value=alive):
                parent, protocol, launch, contract = self.parent(Path(directory), **record)
                with self.assertRaisesRegex(ValueError, "closed segment"):
                    _resume_parent(parent, protocol, launch, contract)

    def test_legacy_dead_parent_is_still_accepted(self):
        from fine_tuning.notebook_rl.local_train import _resume_parent

        with tempfile.TemporaryDirectory() as directory, patch("psutil.pid_exists", return_value=False):
            parent, protocol, launch, contract = self.parent(Path(directory), pid=os.getpid() + 100000)
            self.assertEqual(_resume_parent(parent, protocol, launch, contract)[0]["pid"], os.getpid() + 100000)


if __name__ == "__main__":
    unittest.main()
