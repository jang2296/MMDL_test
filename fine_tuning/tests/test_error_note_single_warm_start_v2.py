"""SINGLE v15 warm start v2 (sealed v14 -> v15 transition) on the authored tiny CPU Qwen.

The real-checkpoint test is opt-in (env MMDL_V14_CAMPAIGN = .../pod-v14-e3odtf0md92kr1/final/extract/single-v14); it hashes
the stored checkpoint-41 but never loads a model or touches CUDA.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from fine_tuning.error_note import audit, single, single_campaign as campaign
from fine_tuning.error_note.protocol import (SINGLE_ROUTES_V15, VERSION_SINGLE_V13, VERSION_SINGLE_V14, VERSION_SINGLE_V15,
                                            plain_protocol_single)
from fine_tuning.notebook_rl import checkpoint
from fine_tuning.tests import test_error_note_single_campaign as base, test_error_note_single_warm_start as v1

setUpModule, tearDownModule = base.setUpModule, base.tearDownModule
V14_PROTOCOL_SHA256 = "6a861b4ff7022b4616b954b958f4374d5385eae3bf124a340ccb3b7409c6e71e"


def protocol(body: dict, version: str, schedule_steps: int | None) -> dict:
    """The fixture protocol (base.plan_for's pins) bound to the body's own code source."""
    contracts = {key: "a" * 64 for key in ("dataset", "order", "data_admission")} | {"source": audit.digest(body["source"])}
    steps = {} if schedule_steps is None else {"schedule_steps": schedule_steps}
    return plain_protocol_single("bf16", len(body["blocks"]), contracts=contracts, token_scale=1024., version=version,
                                 **steps)


class WarmStartV2Tests(base.CampaignCase):
    raw_rows = base.TrainCampaignTests.raw_rows
    real_closed = base.TrainCampaignTests.real_closed
    encode = base.TrainCampaignTests.encode
    execute_train = base.TrainCampaignTests.execute_train
    record = v1.WarmStartTests.record
    warm_plan = v1.WarmStartTests.warm_plan

    def setUp(self):
        super().setUp()
        self.a_out, self.b_out = self.out / "a", self.out / "b"
        # A stopped v14 campaign: three 16-item blocks, a 3-step LR schedule, one committed block, a reload receipt.
        self.plan = self.a_plan = base.plan_for(48, "train", execution_max_blocks=3, version=VERSION_SINGLE_V14)
        with patch.object(single, "run_single_block", side_effect=self.real_closed):
            first = self.execute_train(output=self.a_out)
        reloaded = self.execute_train(output=self.a_out, reload_only=True)
        self.assertEqual((first["cursor"]["slot"], first["cursor"]["updates"], reloaded["status"]), (1, 1, "RELOADED"))
        self.receipt = self.out / "reload-1.json"
        self.receipt.write_text(json.dumps({"status": "RELOADED", "slot": 1, "updates": 1}))

    def body(self, version=VERSION_SINGLE_V15, schedule_steps=3):
        body = copy.deepcopy({key: value for key, value in self.a_plan.items() if key != "sha256"})
        body["source"] = {"authored_fixture": True, "warm_start_code": "v15"}
        body["protocol"] = protocol(body, version, schedule_steps)
        return body

    def test_v15_warm_start_continues_the_v14_schedule(self):
        b_plan = self.warm_plan()
        warm = b_plan["warm_start"]
        self.assertEqual((warm["version"], warm["decision"], warm["transition"], warm["slot"], warm["updates"]),
                         (campaign.WARM_START_VERSION_V2, campaign.WARM_START_DECISION_V2,
                          campaign.WARM_START_TRANSITION_V2, 1, 1))
        # The carry-over needs no optimizer code: v15 keeps v14's optimizer block (the LambdaLR stores no lambda).
        self.assertEqual(b_plan["protocol"]["optimizer"], self.a_plan["protocol"]["optimizer"])
        campaign._plan_check(b_plan)
        self.plan = self.b_plan = b_plan
        with patch.object(single, "run_single_block", side_effect=self.real_closed):
            result = self.execute_train(output=self.b_out)
        self.assertEqual((result["status"], result["cursor"]["slot"], result["cursor"]["updates"]), ("BOUNDED_STOP", 2, 2))
        self.assertEqual(sorted(p.name for p in self.b_out.glob("checkpoint-*")), ["checkpoint-1", "checkpoint-2"])
        multiplier = checkpoint._schedule(self.a_plan["protocol"]["optimizer"])
        for slot in (1, 2):
            metadata = checkpoint.read_metadata(self.b_out / f"checkpoint-{slot}", binding=campaign._binding(b_plan))
            self.assertEqual(metadata["protocol"]["version"], VERSION_SINGLE_V15)
            runtime = torch.load(self.b_out / f"checkpoint-{slot}/runtime.pt", weights_only=True)
            scheduler = runtime["scheduler"]
            groups = runtime["optimizer"]["param_groups"]
            self.assertEqual(scheduler["last_epoch"], slot)
            self.assertEqual(scheduler["base_lrs"], [b_plan["protocol"]["optimizer"]["lr"][g["name"]] for g in groups])
            self.assertEqual(scheduler["_last_lr"], [lr * multiplier(slot) for lr in scheduler["base_lrs"]])
            self.assertEqual({float(entry["step"]) for entry in runtime["optimizer"]["state"].values()}, {float(slot)})
        restored = self.execute_train(output=self.b_out, reload_only=True)
        self.assertEqual((restored["status"], restored["cursor"]), ("RELOADED", result["cursor"]))

    def test_probe_block_then_a_fresh_call_finishes_the_schedule(self):
        """The v15 pod flow: one block after the warm start (the block-41 probe job), then a new call for the rest."""
        self.plan = self.b_plan = self.warm_plan()
        with patch.object(single, "run_single_block", side_effect=self.real_closed):
            probe = self.execute_train(output=self.b_out)
            self.assertEqual((probe["status"], probe["cursor"]["slot"], probe["pending"]), ("BOUNDED_STOP", 2, None))
            rest = self.execute_train(output=self.b_out)
        self.assertEqual((rest["status"], rest["cursor"]["slot"], rest["cursor"]["updates"]), ("COMPLETE", 3, 3))
        # The warm anchor (checkpoint-1) is pruned like any grandparent; the final checkpoint still reloads.
        self.assertEqual(sorted(p.name for p in self.b_out.glob("checkpoint-*")), ["checkpoint-2", "checkpoint-3"])
        restored = self.execute_train(output=self.b_out, reload_only=True)
        self.assertEqual((restored["status"], restored["cursor"]), ("RELOADED", rest["cursor"]))
        scheduler = torch.load(self.b_out / "checkpoint-3/runtime.pt", weights_only=True)["scheduler"]
        multiplier = checkpoint._schedule(self.a_plan["protocol"]["optimizer"])  # the v14 schedule, step 3 of 3
        self.assertEqual((scheduler["last_epoch"], scheduler["_last_lr"]),
                         (3, [lr * multiplier(3) for lr in scheduler["base_lrs"]]))

    def test_same_version_still_seals_a_v1_record(self):
        warm = self.warm_plan(self.body(VERSION_SINGLE_V14))["warm_start"]
        self.assertEqual((warm["version"], warm["decision"]), (campaign.WARM_START_VERSION, campaign.WARM_START_DECISION))
        self.assertNotIn("transition", warm)

    def test_refusals(self):
        v13 = base.plan_for(48, "train")
        v13["protocol"] = protocol(v13, VERSION_SINGLE_V13, None)
        v13_path = self.out / "v13-plan.json"
        v13_path.write_text(json.dumps(base.reseal(v13)))
        with self.assertRaisesRegex(ValueError, "no sealed protocol transition"):
            self.record(self.body(), plan_path=v13_path, plan_sha256=single.file_sha(v13_path))
        routes = self.body(VERSION_SINGLE_V14)
        routes["protocol"]["training_contract"]["routes"] = dict(SINGLE_ROUTES_V15)
        steps = self.body(schedule_steps=2)
        limit = self.body(schedule_steps=2)
        limit["execution_max_blocks"] = 2
        items, blocks = self.body(), self.body()
        items["items"][0].update(subject="Chemistry")
        blocks["blocks"] = [blocks["blocks"][1], blocks["blocks"][0], blocks["blocks"][2]]
        for name, body in (("v14 routes", routes), ("total_steps", steps), ("execution_max_blocks", limit),
                           ("items", items), ("blocks", blocks)):
            with self.subTest(name), self.assertRaisesRegex(ValueError, "same items, blocks and protocol"):
                self.record(body)
        good = self.warm_plan()
        for name, change in (("transition", lambda w: w["transition"]["protocol_changes"].append("optimizer")),
                             ("no transition", lambda w: w.pop("transition")),
                             ("v1 decision", lambda w: w.update(decision=campaign.WARM_START_DECISION)),
                             ("v1 version", lambda w: w.update(version=campaign.WARM_START_VERSION))):
            tampered = copy.deepcopy(good)
            change(tampered["warm_start"])
            with self.subTest(name), self.assertRaisesRegex(ValueError, "invalid warm-start record"):
                campaign._plan_check(base.reseal(tampered))
        v14_body = self.body(VERSION_SINGLE_V14)  # a v2 record on a non-v15 plan
        v14_body["warm_start"] = good["warm_start"]
        with self.assertRaisesRegex(ValueError, "invalid warm-start record"):
            campaign._plan_check(base.reseal(v14_body))

    @unittest.skipUnless(os.environ.get("MMDL_V14_CAMPAIGN"), "opt-in: MMDL_V14_CAMPAIGN=<pod-v14 extract>/single-v14")
    def test_real_v14_checkpoint41_builds_v15_warm_plan(self):
        root = Path(os.environ["MMDL_V14_CAMPAIGN"])
        plan_path, metadata_path = root / "campaign/campaign-plan.json", root / "campaign/checkpoint-41/metadata.json"
        reload_path = root / "reload-41.json"
        old = single._read(plan_path)
        contract = old["protocol"]["training_contract"]
        rebuilt = plain_protocol_single("bf16", 219, contracts=contract["hashes"], note_reference=contract["note_reference"],
                                        version=VERSION_SINGLE_V14, schedule_steps=64)
        self.assertEqual((rebuilt, rebuilt["sha256"]), (old["protocol"], V14_PROTOCOL_SHA256))
        body = copy.deepcopy({key: value for key, value in old.items() if key != "sha256"})
        body["source"] = single.source_manifest()
        body["generation"]["code_commit"] = "0" * 40  # a new code identity; model_files_sha256 stays
        body["protocol"] = plain_protocol_single("bf16", 219, contracts=contract["hashes"] | {
            "source": audit.digest(body["source"])}, note_reference=contract["note_reference"], version=VERSION_SINGLE_V15,
            schedule_steps=64)
        body["warm_start"] = campaign.warm_start_record(body, plan_path=plan_path, plan_sha256=single.file_sha(plan_path),
            checkpoint_path=metadata_path, checkpoint_sha256=single.file_sha(metadata_path), reload_path=reload_path,
            reload_sha256=single.file_sha(reload_path))
        plan = body | {"sha256": audit.digest(body)}
        campaign._plan_check(plan)
        warm = plan["warm_start"]
        self.assertEqual((warm["version"], warm["slot"], warm["updates"], warm["source_binding"]["protocol_sha256"]),
                         (campaign.WARM_START_VERSION_V2, 41, 41, V14_PROTOCOL_SHA256))
        runtime = torch.load(metadata_path.parent / "runtime.pt", map_location="cpu", weights_only=True, mmap=True)
        scheduler, optimizer = runtime["scheduler"], runtime["optimizer"]
        multiplier = checkpoint._schedule(plan["protocol"]["optimizer"])
        self.assertEqual(scheduler["last_epoch"], 41)
        self.assertEqual(scheduler["base_lrs"], [plan["protocol"]["optimizer"]["lr"][g["name"]]
                                                 for g in optimizer["param_groups"]])
        self.assertEqual(scheduler["_last_lr"], [lr * multiplier(41) for lr in scheduler["base_lrs"]])
        self.assertEqual((multiplier(63), multiplier(64)), (1 / 54, 0.0))
        steps = {g["name"]: {float(optimizer["state"][i]["step"]) for i in g["params"]} for g in optimizer["param_groups"]}
        self.assertEqual(steps, {"L": {41.}, "M": {41.}, "V": {12.}})


if __name__ == "__main__":
    unittest.main()
