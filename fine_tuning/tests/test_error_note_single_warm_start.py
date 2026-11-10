"""SINGLE v11c warm start on the authored tiny CPU Qwen; never real checkpoints, pod data or CUDA."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from fine_tuning.error_note import audit, single, single_campaign as campaign
from fine_tuning.error_note.protocol import _digest
from fine_tuning.notebook_rl import checkpoint
from fine_tuning.tests import test_error_note_single_campaign as base
from fine_tuning.tests.test_error_note_train import tiny_factory

setUpModule, tearDownModule = base.setUpModule, base.tearDownModule


class WarmStartTests(base.CampaignCase):
    raw_rows = base.TrainCampaignTests.raw_rows
    real_closed = base.TrainCampaignTests.real_closed
    encode = base.TrainCampaignTests.encode
    execute_train = base.TrainCampaignTests.execute_train

    def setUp(self):
        super().setUp()
        self.a_out, self.b_out = self.out / "a", self.out / "b"
        self.plan = self.a_plan = base.plan_for(48, "train")  # three 16-item blocks
        with patch.object(single, "run_single_block", side_effect=self.real_closed):
            first = self.execute_train(output=self.a_out)
        reloaded = self.execute_train(output=self.a_out, reload_only=True)
        self.assertEqual((first["cursor"]["slot"], first["cursor"]["updates"], reloaded["status"]), (1, 1, "RELOADED"))
        self.receipt = self.out / "reload-1.json"
        self.receipt.write_text(json.dumps({"status": "RELOADED", "slot": 1, "updates": 1}))

    def body(self):
        """The stopped plan with a new code identity: only the protocol source hash and its sha change."""
        body = copy.deepcopy({key: value for key, value in self.a_plan.items() if key != "sha256"})
        body["source"] = {"authored_fixture": True, "warm_start_code": "v11c"}
        protocol = body["protocol"]
        protocol["training_contract"]["hashes"]["source"] = audit.digest(body["source"])
        protocol["sha256"] = _digest({key: value for key, value in protocol.items() if key != "sha256"})
        return body

    def record(self, body, **overrides):
        paths = {"plan": self.a_out / "campaign-plan.json", "checkpoint": self.a_out / "checkpoint-1/metadata.json",
                 "reload": self.receipt}
        arguments = {f"{name}_{kind}": value for name, path in paths.items()
                     for kind, value in (("path", path), ("sha256", single.file_sha(path)))}
        return campaign.warm_start_record(body, **(arguments | overrides))

    def warm_plan(self, body=None):
        body = body or self.body()
        body["warm_start"] = self.record(body)
        return body | {"sha256": audit.digest(body)}

    def test_warm_start_continues_weights_moments_scheduler_and_cursor_without_checkpoint0(self):
        self.plan = self.b_plan = self.warm_plan()
        with patch.object(single, "run_single_block", side_effect=self.real_closed):
            result = self.execute_train(output=self.b_out)
        self.assertEqual((result["status"], result["cursor"]["slot"], result["cursor"]["updates"]), ("BOUNDED_STOP", 2, 2))
        self.assertFalse((self.b_out / "checkpoint-0").exists())
        self.assertEqual(sorted(p.name for p in self.b_out.glob("checkpoint-*")), ["checkpoint-1", "checkpoint-2"])
        runtime = torch.load(self.b_out / "checkpoint-2/runtime.pt", weights_only=True)
        self.assertEqual(runtime["scheduler"]["last_epoch"], 2)
        self.assertTrue(runtime["optimizer"]["state"])
        self.assertTrue(all(float(entry["step"]) == 2 for entry in runtime["optimizer"]["state"].values()))
        # Same tensors as a native resume of the source campaign for that block.
        self.plan = self.a_plan
        with patch.object(single, "run_single_block", side_effect=self.real_closed):
            self.assertEqual(self.execute_train(output=self.a_out)["cursor"]["slot"], 2)
        native = checkpoint.read_metadata(self.a_out / "checkpoint-2", binding=campaign._binding(self.a_plan))
        warm = checkpoint.read_metadata(self.b_out / "checkpoint-2", binding=campaign._binding(self.b_plan))
        self.assertEqual(warm["state_sha256"], native["state_sha256"])
        self.plan = self.b_plan
        restored = self.execute_train(output=self.b_out, reload_only=True)
        self.assertEqual((restored["status"], restored["cursor"]), ("RELOADED", result["cursor"]))
        state = single._state(self.b_out / "campaign-state.json")
        self.assertEqual(state["warm_start"], self.b_plan["warm_start"]["checkpoint"])
        self.assertEqual(state["pilot_checkpoint"], campaign._descriptor(self.b_out / "checkpoint-1/metadata.json"))
        campaign._direct_gate(self.b_plan, state, True, None)
        state["checkpoint"] = {"path": "a later checkpoint", "file_sha256": "e" * 64}
        campaign._direct_gate(self.b_plan, state, True, None)  # the main-gate receipt stays fixed across blocks

    def test_refusals(self):
        body = self.body()
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.record(body, checkpoint_sha256="0" * 64)
        for receipt in ({"status": "FAILED", "slot": 1, "updates": 1}, {"status": "RELOADED", "slot": 2, "updates": 2}):
            self.receipt.write_text(json.dumps(receipt))
            with self.subTest(receipt=receipt), self.assertRaisesRegex(ValueError, "reload receipt"):
                self.record(body)
        self.receipt.write_text(json.dumps({"status": "RELOADED", "slot": 1, "updates": 1}))
        for change in (lambda b: b["items"][0].update(subject="Chemistry"),
                       lambda b: b.update(blocks=[b["blocks"][1], b["blocks"][0], b["blocks"][2]]),
                       lambda b: b["protocol"]["training_contract"].update(master_seed=1)):
            changed = copy.deepcopy(body)
            change(changed)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "same items, blocks and protocol"):
                self.record(changed)
        with self.assertRaisesRegex(ValueError, "warm start needs"):
            campaign._prepare(SimpleNamespace(warm_start_plan=Path("plan.json"), warm_start_plan_sha256="a" * 64))
        tampered = self.warm_plan()
        tampered["warm_start"]["cursor_sha256"] = "0" * 64
        self.plan = base.reseal(tampered)
        with self.assertRaisesRegex(ValueError, "warm-start checkpoint cursor changed"):
            self.execute_train(output=self.b_out)
        self.assertFalse(any(self.b_out.glob("checkpoint-*")))
        tampered["warm_start"]["slot"] = 0
        with self.assertRaisesRegex(ValueError, "invalid warm-start record"):
            campaign._plan_check(base.reseal(tampered))

    def test_crash_after_warm_checkpoint_save_recovers_without_the_source_checkpoint(self):
        self.plan = self.warm_plan()
        actual_state = single._state

        def interrupted(path, value=None):
            if path.name == "campaign-state.json" and value is not None:
                raise RuntimeError("authored warm pointer crash")
            return actual_state(path, value)

        with patch.object(single, "_state", side_effect=interrupted), self.assertRaisesRegex(RuntimeError, "warm pointer"):
            self.execute_train(output=self.b_out)
        self.assertTrue((self.b_out / "checkpoint-1/metadata.json").exists())
        self.assertFalse((self.b_out / "campaign-state.json").exists())
        (self.a_out / "checkpoint-1").rename(self.a_out / "moved")  # the source can no longer be loaded
        factory = Mock(wraps=tiny_factory)
        with patch.object(single, "save_single_continuation", side_effect=AssertionError("duplicate checkpoint")):
            result = self.execute_train(output=self.b_out, policy_factory=factory, reload_only=True)
        self.assertEqual((result["status"], result["cursor"]["slot"], factory.call_count), ("RELOADED", 1, 1))
        state = single._state(self.b_out / "campaign-state.json")
        self.assertEqual((state["status"], state["warm_start"]), ("READY", self.plan["warm_start"]["checkpoint"]))

    def test_plan_without_warm_start_is_unchanged(self):
        body = {key: value for key, value in self.a_plan.items() if key != "sha256"}
        self.assertNotIn("warm_start", self.a_plan)
        self.assertEqual(self.a_plan["sha256"], audit.digest(body))
        campaign._plan_check(self.a_plan)
        self.assertTrue((self.a_out / "checkpoint-0/metadata.json").exists())
        state = single._state(self.a_out / "campaign-state.json")
        self.assertFalse({"warm_start", "pilot_checkpoint"} & set(state))
        args = campaign.build_parser().parse_args(["prepare", *[f"--{name}={name}" for name in (
            "build-root", "order", "frozen-approval", "source-provenance", "training-rules", "note-reference-supplement",
            "model-path", "output")], *[f"--{name}-sha256=x" for name in (
            "order", "frozen-approval", "source-provenance", "training-rules", "note-reference-supplement",
            "pilot-ids", "dev-ids", "retention-ids")]])
        self.assertEqual([getattr(args, name) for name in campaign.WARM_START_ARGS], [None] * 6)


if __name__ == "__main__":
    unittest.main()
