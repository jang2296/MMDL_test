"""CPU tiny-model checks for preparation, RL, HOLD, flushes and NF4 reload gates."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import warnings

import torch

from fine_tuning.error_note import protocol as proto, rl, single, train_v16b as train, train_v4
from fine_tuning.notebook_rl import checkpoint
from fine_tuning.tests.test_error_note_train import tiny_factory
from fine_tuning.tests.test_error_note_v16b_loss import contract as row_contract, row
from fine_tuning.tests.test_error_note_v16b_protocol import protocol


class V16bJointUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.protocol = protocol()
        self.bundle = tiny_factory(protocol=self.protocol)
        self.optimizer, self.scheduler = checkpoint.make_optimizer(self.bundle, self.protocol)
        self.cursor = {"slot": 0, "updates": 0, "consumed_problem_ids": [], "consumed_prep_rows": [],
                       "policy_sha256": rl.policy_sha256(self.bundle)}

    def contract(self, rows, *, phase, slot=None, ids=None):
        slot = self.cursor["slot"] if slot is None else slot
        version = slot if phase == "prep" else slot - self.protocol["training_contract"]["prep_updates"]
        memberships = row_contract(self.bundle, rows, version=version, phase=phase)
        return proto.v16b_contract_for_update(self.protocol, slot, phase=phase,
            policy_sha256=rl.policy_sha256(self.bundle),
            expected_problem_ids=memberships["expected_problem_ids"] if ids is None else ids,
            expected_obs_problem_ids=memberships["expected_obs_problem_ids"],
            note_problem_ids=memberships["note_problem_ids"], resolve_problem_ids=memberships["resolve_problem_ids"],
            baseline_receipts=memberships["baseline_receipts"], reward_values=memberships["reward_values"])

    def update(self, rows, contract):
        # Zero-signal slots deliberately consume LR without an optimizer step.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of .*lr_scheduler.step", category=UserWarning)
            return single.run_joint_single_update(self.bundle, self.optimizer, self.scheduler, rows, contract,
                cursor=self.cursor, block_ids=contract["expected_problem_ids"], allow_update=True, token_budget=64)

    def assert_gradients_cleared(self):
        self.assertTrue(all(parameter.grad is None for values in self.bundle.params.values() for parameter in values))

    def test_self_and_human_same_prep_parent_update_then_rl_starts_at_zero(self):
        rows = [row(self.bundle, index=0, kind=kind, version=0) for kind in ("PREP_SELF", "PREP_HUMAN")]
        before = rl.policy_sha256(self.bundle)
        initial_epoch = self.scheduler.last_epoch
        with patch.object(self.optimizer, "step", wraps=self.optimizer.step) as step:
            result = self.update(rows, self.contract(rows, phase="prep"))
        self.assertEqual(step.call_count, 1)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(set(result["stats"]["update_gradient_norms"]), set(self.bundle.params))
        self.assertGreater(result["stats"]["update_gradient_norms"]["L"], 0.)
        self.assertNotEqual(rl.policy_sha256(self.bundle), before)
        self.assertEqual(self.cursor["consumed_prep_rows"], ["PREP_SELF:problem-0", "PREP_HUMAN:problem-0"])
        self.assertEqual(self.cursor["consumed_problem_ids"], [])
        self.assertEqual((self.cursor["slot"], self.cursor["phase"], self.cursor["rl_block"],
                          self.cursor["policy_version"]), (1, "rl", 0, 0))
        self.assertEqual(self.scheduler.last_epoch, initial_epoch + 1)
        self.assert_gradients_cleared()
        rl_rows = [row(self.bundle, index=1, version=0)]
        contract = self.contract(rl_rows, phase="rl")
        self.assertEqual((contract["global_slot"], contract["block_id"]), (1, 0))
        result = self.update(rl_rows, contract)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(self.cursor["consumed_problem_ids"], ["problem-1"])
        self.assertEqual((self.cursor["slot"], self.cursor["rl_block"], self.cursor["updates"]), (2, 1, 2))

    def test_duplicate_same_prep_row_is_rejected_without_consuming_schedule(self):
        original = row(self.bundle, kind="PREP_SELF", version=0)
        rows = [original, copy.deepcopy(original)]
        before = copy.deepcopy(self.cursor)
        epoch = self.scheduler.last_epoch
        with self.assertRaisesRegex(ValueError, "repeated preparation row"):
            self.update(rows, self.contract(rows, phase="prep"))
        self.assertEqual(self.cursor, before)
        self.assertEqual(self.scheduler.last_epoch, epoch)
        self.assertFalse(self.optimizer.state)

    def test_consumed_prep_row_cannot_repeat_in_a_later_prep_slot(self):
        self.protocol = protocol(prep_updates=2)
        self.optimizer, self.scheduler = checkpoint.make_optimizer(self.bundle, self.protocol)
        rows = [row(self.bundle, kind="PREP_SELF", version=0)]
        self.update(rows, self.contract(rows, phase="prep"))
        before = copy.deepcopy(self.cursor)
        with self.assertRaisesRegex(ValueError, "repeated preparation row"):
            self.update(rows, self.contract(rows, phase="prep"))
        self.assertEqual(self.cursor, before)

    def test_repeated_rl_parent_is_rejected_after_its_first_committed_update(self):
        self.cursor["slot"] = 1
        rows = [row(self.bundle, index=7, version=0)]
        self.update(rows, self.contract(rows, phase="rl"))
        before = copy.deepcopy(self.cursor)
        epoch = self.scheduler.last_epoch
        repeated = [row(self.bundle, index=7, version=1)]
        with self.assertRaisesRegex(ValueError, "duplicate RL parent"):
            self.update(repeated, self.contract(repeated, phase="rl"))
        self.assertEqual(self.cursor, before)
        self.assertEqual(self.scheduler.last_epoch, epoch)

    def test_hold_discards_note_and_rl_gradients_and_consumes_only_lr_slot(self):
        self.cursor["slot"] = 4  # preparation 1 + RL version 3
        rows = [row(self.bundle, index=2, kind="N", version=3), row(self.bundle, index=3, version=3)]
        rows[1]["generation"]["sampling_logprobs"] = [value - 1. for value in rows[1]["generation"]["sampling_logprobs"]]
        before = rl.policy_sha256(self.bundle)
        epoch = self.scheduler.last_epoch
        with patch.object(self.optimizer, "step", wraps=self.optimizer.step) as step:
            result = self.update(rows, self.contract(rows, phase="rl"))
        self.assertEqual(step.call_count, 0)
        self.assertTrue(result["stats"]["hold"])
        self.assertEqual(result["stats"]["hold_reason"], "IS_MEAN")
        self.assertGreater(result["stats"]["sft"]["N"]["weighted_loss"], 0.)
        self.assertTrue(any(value and value > 0 for value in result["stats"]["note_gradient_norms"].values()))
        self.assertFalse(result["stats"]["note_gradient_retained"])
        self.assertFalse(result["stats"]["task_signal"])
        self.assertTrue(all(value is None for value in result["stats"]["update_gradient_norms"].values()))
        self.assertEqual((self.cursor["slot"], self.cursor["updates"]), (5, 0))
        self.assertEqual(self.scheduler.last_epoch, epoch + 1)
        self.assertEqual(rl.policy_sha256(self.bundle), before)
        self.assertFalse(self.optimizer.state)
        self.assert_gradients_cleared()

    def test_external_hold_discards_incoming_gradients_without_backward(self):
        self.cursor["slot"] = 1
        rows = [row(self.bundle, version=0)]
        contract = self.contract(rows, phase="rl") | {"external_hold": "MIXED_VERSION"}
        before = rl.policy_sha256(self.bundle)
        for values in self.bundle.params.values():
            for parameter in values:
                parameter.grad = torch.ones_like(parameter)
        epoch = self.scheduler.last_epoch
        with patch.object(train_v4, "backward_single_block", side_effect=AssertionError("held block entered backward")):
            result = self.update(rows, contract)
        self.assertTrue(result["stats"]["hold"])
        self.assertEqual(self.cursor["updates"], 0)
        self.assertEqual(self.scheduler.last_epoch, epoch + 1)
        self.assertEqual(rl.policy_sha256(self.bundle), before)
        self.assert_gradients_cleared()

    def test_anchor_gradient_without_task_signal_cannot_decay_parameters(self):
        with torch.no_grad():
            for parameter in self.bundle.params["L"]:
                parameter.add_(.2)
        self.cursor.update(slot=1, policy_sha256=rl.policy_sha256(self.bundle))
        # A negative row whose tokens are all masked still enters the loss (so it is anchored) but gives no task signal.
        rows = [row(self.bundle, advantage=-1., version=0)]
        before = rl.policy_sha256(self.bundle)
        epoch = self.scheduler.last_epoch
        with patch.object(self.optimizer, "step", wraps=self.optimizer.step) as step, \
                patch.object(rl, "negative_token_mask", lambda value: torch.zeros_like(value, dtype=torch.bool)):
            result = self.update(rows, self.contract(rows, phase="rl"))
        self.assertGreater(result["stats"]["rl"]["anchor"], 0.)
        self.assertTrue(result["stats"]["has_gradient"])
        self.assertFalse(result["stats"]["task_signal"])
        self.assertEqual(step.call_count, 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(self.scheduler.last_epoch, epoch + 1)
        self.assertEqual(rl.policy_sha256(self.bundle), before)
        self.assertFalse(self.optimizer.state)
        self.assert_gradients_cleared()

    def test_anchor_covers_only_rl_rows_that_entered_the_pg_loss(self):
        for advantage, anchored in ((0., False), (1., True)):
            with self.subTest(advantage=advantage):
                with torch.no_grad():
                    for parameter in self.bundle.params["L"]:
                        parameter.add_(.2)
                rows = [row(self.bundle, advantage=advantage, version=0)]
                stats = train.backward_v16b_block(self.bundle, rows, row_contract(self.bundle, rows, version=0))
                self.bundle.model.zero_grad(set_to_none=True)
                self.assertEqual(stats["rl"]["anchor"] > 0., anchored)
                self.assertEqual(stats["task_signal"], anchored)

    def test_terminal_endoftext_is_an_end_token_not_a_control_token_error(self):
        from PIL import Image
        from fine_tuning.error_note.contract import ROUTE_FULL
        from fine_tuning.tests.test_error_note_train import tiny_processor

        processor = tiny_processor()
        tokenizer = processor.tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": tokenizer.additional_special_tokens + ["<|endoftext|>"]})
        eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        actions = tokenizer.encode("a b", add_special_tokens=False) + [eos]
        images, image = [Image.new("RGB", (8, 8))], {"min_pixels": 64, "max_pixels": 64}
        for kind in ("C", "N"):
            with self.subTest(kind=kind):
                raw = {"schema": train.ROW_SCHEMA, "kind": kind, "problem_id": "p", "route": ROUTE_FULL,
                       "prompt": "question", "target": "ab", "verified_byte_spans": [[0, 2]],
                       "generation": {"text": "ab", "completion_token_ids": actions, "finish_reason": "stop"}}
                encoded = train.encode_v16b_row(processor, self.bundle.model.config, raw, images, image=image)
                self.assertEqual(encoded["loss_mask"].tolist(), [True, True, False])
        with self.assertRaisesRegex(ValueError, "nonterminal end token"):
            train.encode_v16b_row(processor, self.bundle.model.config,
                                  raw | {"generation": raw["generation"] | {"completion_token_ids": [eos] + actions}},
                                  images, image=image)

    def test_only_three_final_empty_flush_slots_are_accepted(self):
        prep = self.protocol["training_contract"]["prep_updates"]
        self.cursor["slot"] = prep + 187
        with self.assertRaisesRegex(ValueError, "invalid flush"):
            self.update([], self.contract([], phase="rl"))
        self.cursor["slot"] = prep + 188
        epoch = self.scheduler.last_epoch
        before = rl.policy_sha256(self.bundle)
        for index in range(3):
            result = self.update([], self.contract([], phase="rl"))
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual(self.cursor["rl_block"], 189 + index)
            self.assertTrue(self.cursor["flush"])
        self.assertEqual(self.scheduler.last_epoch, epoch + 3)
        self.assertEqual(self.cursor["slot"], self.protocol["training_contract"]["total_slots"])
        self.assertEqual(rl.policy_sha256(self.bundle), before)
        self.assertFalse(self.optimizer.state)
        with self.assertRaisesRegex(ValueError, "invalid phase, slot"):
            self.contract([], phase="rl")

    def test_update_requires_explicit_optin(self):
        rows = [row(self.bundle, kind="PREP_SELF", version=0)]
        contract = self.contract(rows, phase="prep")
        with self.assertRaisesRegex(ValueError, "opt-in"):
            single.run_joint_single_update(self.bundle, self.optimizer, self.scheduler, rows, contract,
                cursor=self.cursor, block_ids=contract["expected_problem_ids"])
        self.assertEqual(self.cursor["slot"], 0)

    def test_nf4_reload_requires_smoke_and_explicit_factory_before_checkpoint_access(self):
        smoke = protocol(local_smoke=True)
        legacy = proto.plain_protocol_single("nf4", 219,
            contracts={key: "0" * 64 for key in proto.SINGLE_CONTRACT_HASHES})
        factory = Mock(name="cached_nf4_factory")
        with patch.object(checkpoint, "load_resume", side_effect=AssertionError("unsafe checkpoint access")):
            for sealed, provided in ((smoke, None), (legacy, None), (legacy, factory)):
                with self.subTest(version=sealed["version"], factory=provided is not None), self.assertRaisesRegex(ValueError, "explicit cache factory"):
                    single.load_single_continuation(Path("unused-model"), Path("unused-checkpoint"), sealed,
                        binding={}, policy_factory=provided)
        saved = {"position": 0, "single": self.cursor}
        with patch.object(checkpoint, "load_resume", return_value=(self.bundle, self.optimizer, self.scheduler, 0, saved)) as resume:
            restored = single.load_single_continuation(Path("unused-model"), Path("unused-checkpoint"), smoke,
                binding={"source": "fixture"}, policy_factory=factory)
        self.assertIs(restored[0], self.bundle)
        self.assertIs(resume.call_args.kwargs["policy_factory"], factory)
        self.assertEqual(restored[3], self.cursor)


if __name__ == "__main__":
    unittest.main()
