"""Saved CPU update replay and tamper boundaries; authored fixtures, not Qwen benefit."""
from __future__ import annotations

import copy
import json
import unittest
from typing import Any
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import repair_evidence, repair_training
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
from fine_tuning.tests import test_notebook_rl_repair_update as fixture


class RepairEvidenceTests(unittest.TestCase):
    f: fixture.RepairUpdateTests
    tree: ProblemTree
    engine: Any
    update: dict
    receipts: dict
    inventory: list[dict]

    @classmethod
    def setUpClass(cls):
        cls.f = fixture.RepairUpdateTests()
        cls.f.setUp()
        cls.addClassCleanup(cls.f.doCleanups)
        cls.tree, cls.engine = cls.f.online()
        cls.update, _, _ = cls.f.step(cls.tree, cls.engine)
        cls.receipts = {cls.tree.problem_id: cls.f.f.prepare(cls.tree, cls.engine)}
        cls.inventory = [{key: value for key, value in row.items() if key != "index"}
                         for row in cls.update["repair_aggregate_gradient"]["parameter_inventory"]]

    def verify(self, update=None, **options):
        return repair_evidence.repair_update_evidence(self.update if update is None else update,
            self.f.protocol, trees=options.pop("trees", [self.tree]),
            preparations=options.pop("preparations", self.receipts),
            parameter_inventory=options.pop("parameter_inventory", self.inventory), **options)

    def test_actual_tiny_update_and_json_roundtrip_verify_s_without_claiming_f_or_benefit(self):
        before = _hash(self.update)
        result = self.verify(require_s=True)
        clone = json.loads(json.dumps(self.update))
        trees = [from_json(ProblemTree, to_json(self.tree))]
        self.assertEqual(result, self.verify(clone, trees=trees, require_s=True))
        self.assertTrue(result["S_gradient_verified"])
        for key in ("repair_learning_verified", "functional_verified", "benefit_verified"):
            self.assertIs(result[key], False)
        self.assertIs(result["scope"]["VJP_remeasured"], False)
        self.assertEqual(before, _hash(self.update))

    def test_rehashed_journal_tampering_is_rejected(self):
        changes = (
            lambda rows: rows.pop(), lambda rows: rows.reverse(),
            lambda rows: rows[0].update(old_logprobs=[.5] * len(rows[0]["old_logprobs"])),
            lambda rows: rows[0].update(retained=False), lambda rows: rows[0].update(vision_on=True),
            lambda rows: rows[0].update(policy_version=False), lambda rows: rows[0].update(norm=True),
            lambda rows: rows[0].update(sample_sha256="a" * 64),
            lambda rows: rows[0].update(logprob_dtype="torch.int32"),
            lambda rows: rows[0]["old_logprobs"].__setitem__(0, float("nan")),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                update = copy.deepcopy(self.update)
                change(update["repair_actions"])
                try:
                    update["repair_actions_sha256"] = _hash(update["repair_actions"])
                except ValueError:
                    pass
                with self.assertRaises(ValueError):
                    self.verify(update)

    def test_recorded_gradient_membership_norm_and_s_tampering_is_rejected(self):
        changes = (
            lambda agg: agg["members"].pop(),
            lambda agg: agg["members"][0].update(positive_verified_tokens=True),
            lambda agg: agg["members"][0].update(disposition="MIS_EXCLUDED"),
            lambda agg: agg["combined"]["positive"].update(L=999.),
            lambda agg: agg["combined"]["signed"].update(V=1.),
            lambda agg: agg["channels"]["online"]["positive"].update(L=True),
            lambda agg: agg["parameter_inventory"][0].update(index=False),
            lambda agg: agg.update(S=False), lambda agg: agg.update(benefit_verified=True),
            lambda agg: agg.update(scope="answer_or_kl_gradient"),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                update = copy.deepcopy(self.update)
                agg = update["repair_aggregate_gradient"]
                change(agg)
                agg["members_sha256"] = _hash(agg["members"])
                agg["parameter_inventory_sha256"] = _hash(agg["parameter_inventory"])
                with self.assertRaises(ValueError):
                    self.verify(update)

    def test_fresh_sources_parameter_inventory_steps_and_denominators_are_required(self):
        for key, value in (("step_before", False), ("step_after", True), ("n_problems", True),
                           ("no_action_samples", False), ("masked_fraction", 1.),
                           ("notebook_reward_enabled", False), ("repair_learning_verified", True),
                           ("max_current_old_difference", .01)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(self.update | {key: value})
        modified = from_json(ProblemTree, to_json(self.tree))
        modified.groups[0].samples[0].text += " altered"
        with self.assertRaises(ValueError):
            self.verify(trees=[modified])
        inventory = copy.deepcopy(self.inventory)
        inventory[0]["name"] += "_other"
        with self.assertRaises(ValueError):
            self.verify(parameter_inventory=inventory)
        receipts = copy.deepcopy(self.receipts)
        receipt = receipts[self.tree.problem_id]
        receipt["groups"][1]["original_group_size"] = 1
        receipt["sha256"] = _hash({key: value for key, value in receipt.items() if key != "sha256"})
        with self.assertRaises(ValueError):
            self.verify(self.update | {"repair_preparation": receipts}, preparations=receipts)

    def test_mis_excluded_positive_cannot_be_reintroduced_by_stored_s(self):
        f = fixture.RepairUpdateTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        tree, engine = f.online()
        f.shifted = {tree.groups[1].samples[0].sample_id}
        update, _, _ = f.step(tree, engine)
        receipts = {tree.problem_id: f.f.prepare(tree, engine)}
        result = repair_evidence.repair_update_evidence(update, f.protocol, trees=[tree],
            preparations=receipts, parameter_inventory=self.inventory)
        self.assertFalse(result["S_gradient_verified"])
        with self.assertRaises(ValueError):
            repair_evidence.repair_update_evidence(update, f.protocol, trees=[tree],
                preparations=receipts, parameter_inventory=self.inventory, require_s=True)

    def test_r0_preserves_original_n_and_has_no_repair_channel(self):
        f = fixture.RepairUpdateTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        tree, engine = f.answers()
        update, _, _ = f.step(tree, engine, notebook_reward_enabled=False)
        receipts = {tree.problem_id: f.f.prepare(tree, engine, notebook_reward_enabled=False)}
        result = repair_evidence.repair_update_evidence(update, f.protocol, trees=[tree],
            preparations=receipts, parameter_inventory=self.inventory, control=True)
        self.assertFalse(result["S_gradient_verified"])
        self.assertEqual(update["repair_aggregate_gradient"]["members"], [])
        with self.assertRaises(ValueError):
            repair_evidence.repair_update_evidence(update, f.protocol, trees=[tree],
                preparations=receipts, parameter_inventory=self.inventory, control=False)

    def test_stored_mis_mean_cannot_cross_threshold_within_roundoff_allowance(self):
        f = fixture.RepairUpdateTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        tree, engine = f.online()
        attach = f.attach_inputs
        target = tree.groups[1].samples[0].sample_id

        def attach_float32(current_tree, current_engine):
            forward, reference = attach(current_tree, current_engine)

            def shifted(bundle, inputs, completion_len, *, chunk):
                value = forward(bundle, inputs, completion_len, chunk=chunk).float()
                return value + (.0499999 if inputs["sample"].sample_id == target else 0.)

            def ref32(bundle, inputs, completion_len, *, chunk):
                return reference(bundle, inputs, completion_len, chunk=chunk).float()

            return shifted, ref32

        with patch.object(f, "attach_inputs", side_effect=attach_float32):
            update, _, _ = f.step(tree, engine)
        receipts = {tree.problem_id: f.f.prepare(tree, engine)}
        kwargs: dict[str, Any] = dict(trees=[tree], preparations=receipts,
                                      parameter_inventory=self.inventory, require_s=True)
        repair_evidence.repair_update_evidence(update, f.protocol, **kwargs)
        self.assertTrue(next(row for row in update["repair_actions"] if row["sample_id"] == target)["retained"])
        threshold = torch.tensor(.05, dtype=torch.float32)
        excluded = torch.nextafter(threshold, torch.tensor(float("inf"))).item()
        saved = update["mean_log_ratio_by_sample"][target]
        self.assertLess(abs(excluded - saved), 8 * torch.finfo(torch.float32).eps)
        for bad in (excluded, saved + 1e-12):
            changed = copy.deepcopy(update)
            changed["mean_log_ratio_by_sample"][target] = bad
            with self.subTest(mean=bad), self.assertRaisesRegex(ValueError, "stored MIS mean"):
                repair_evidence.repair_update_evidence(changed, f.protocol, **kwargs)

    def test_replayed_preparation_remains_independent_of_optimizer_record(self):
        fresh = repair_training.prepare_tree_credit(self.tree, self.f.f.actor, self.f.f.verifier,
            self.engine.tokenizer, preparation=self.engine, images=self.f.f.images, master_seed=3407)
        self.assertEqual(fresh, self.receipts[self.tree.problem_id])
        self.assertFalse(torch.cuda.is_initialized())

    def test_malformed_final_sibling_is_present_in_action_and_zero_credit_membership(self):
        f = fixture.RepairUpdateTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        tree, engine = f.f.online(wrong_answer=False)
        valid = f.f.fixture.final(tree)
        f.f.attach(tree, engine, [valid, valid, valid, "{"])
        update, _, _ = f.step(tree, engine)
        receipts = {tree.problem_id: f.f.prepare(tree, engine)}
        result = repair_evidence.repair_update_evidence(update, f.protocol, trees=[tree],
            preparations=receipts, parameter_inventory=self.inventory)
        self.assertTrue(result["S_gradient_verified"])
        final = receipts[tree.problem_id]["groups"][-1]
        self.assertEqual(final["original_group_size"], 4)
        self.assertIsNotNone(final["samples"][3]["parser_error_code"])
        malformed_id = tree.groups[-1].samples[3].sample_id
        action = next(row for row in update["repair_actions"] if row["sample_id"] == malformed_id)
        member = next(row for row in update["repair_aggregate_gradient"]["members"] if row["sample_id"] == malformed_id)
        self.assertEqual(action["norm"], 4 * 8192)
        self.assertEqual((member["disposition"], member["positive_verified_tokens"]), ("ZERO_VECTOR", 0))


if __name__ == "__main__":
    unittest.main()
