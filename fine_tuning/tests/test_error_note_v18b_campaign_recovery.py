"""Public campaign recovery options and explicitly selected partial-evaluation boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fine_tuning.error_note import campaign_v16b as campaign
from fine_tuning.error_note.audit import digest
from fine_tuning.error_note.collect import write_json
from fine_tuning.error_note.prep_v16b import seal
from fine_tuning.tests.test_error_note_v18b_recovery import descriptor


class CampaignRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir(exist_ok=True)
        plan = {"protocol": {"version": "error-note-v18b"}, "blocks": [["p0"], ["p1"], ["p2"]],
            "execution_settings": {"monitor_stop_action": "evaluate_checkpoint", "evaluation_strategy": "training_cutoff"},
            "local_sequential": False, "total_slots": 7, "prep_batches": [[]], "version": "single-campaign-v18b"}
        plan = seal(root / "campaign-plan.json", plan)
        snapshot = root / "runtime.json"
        snapshot.write_text("{}")
        probe = root / "probe.json"
        probe.write_text("{}")

        def checkpoint(slot, consumed):
            transaction = seal(root / "transactions" / f"slot-{slot - 1}.json", {"plan_sha256": plan["sha256"], "slot": slot - 1})
            cursor = {"slot": slot, "updates": slot, "phase": "rl", "policy_version": slot - 1,
                "consumed_problem_ids": consumed, "consumed_row_ids": [], "last_transaction_sha256": transaction["sha256"]}
            path = root / f"checkpoint-{slot}-rl"
            path.mkdir()
            write_json(path / "metadata.json", {"cursor": {"single": cursor}, "step": slot, "protocol": plan["protocol"]})
            return {"cursor": cursor, "checkpoint": descriptor(path / "metadata.json"),
                    "runtime": descriptor(snapshot), "probe": {"json": descriptor(probe)}, "guard": {}}

        normal, current = checkpoint(2, ["p0"]), checkpoint(3, ["p0", "p1"])
        body = {"plan_sha256": plan["sha256"], "status": "STOPPED", "stop_reasons": ["MONITOR_STOP"],
                **current, "v18b_normal_state": normal}
        state = seal(root / "campaign-state.json", body)
        return plan, state

    def test_monitor_uses_previous_healthy_and_cutoff_uses_latest_durable_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, state = self.fixture(root)
            selected = campaign.partial_evaluation_state(plan, state, root)
            self.assertEqual(selected["checkpoint"], state["v18b_normal_state"]["checkpoint"])
            self.assertEqual(selected["evaluation_selection"]["consumed_items"], 1)
            cutoff = deepcopy(state)
            cutoff["stop_reasons"] = ["TRAINING_CUTOFF"]
            selected = campaign.partial_evaluation_state(plan, cutoff, root)
            self.assertEqual(selected["checkpoint"], state["checkpoint"])
            self.assertEqual(selected["evaluation_selection"]["consumed_items"], 2)
            self.assertEqual(selected["status"], "PARTIAL_EVALUATION_CANDIDATE")

    def test_zero_main_wrong_trace_and_unapproved_stop_never_become_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, state = self.fixture(root)
            zero = deepcopy(state)
            zero["v18b_normal_state"]["cursor"]["consumed_problem_ids"] = []
            self.assertIsNone(campaign.partial_evaluation_state(plan, zero, root))
            for reason in ("STOP_REQUESTED", "RUN_DEADLINE", "WATCHDOG_STOP"):
                self.assertIsNone(campaign.partial_evaluation_state(plan, state | {"stop_reasons": [reason]}, root))
            stopped = deepcopy(plan)
            stopped["execution_settings"]["monitor_stop_action"] = "stop"
            self.assertIsNone(campaign.partial_evaluation_state(stopped, state, root))
            changed = deepcopy(state)
            changed["v18b_normal_state"]["cursor"]["last_transaction_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "trace"):
                campaign.partial_evaluation_state(plan, changed, root)

    def test_cutoff_after_all_first_solves_exports_each_settlement_boundary_as_partial(self):
        for slot in range(4, 8):
            with self.subTest(slot=slot), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan, state = self.fixture(root)
                transaction = seal(root / "transactions" / f"slot-{slot - 1}.json",
                                   {"plan_sha256": plan["sha256"], "slot": slot - 1})
                cursor = state["cursor"] | {"slot": slot, "updates": slot, "phase": "stopped",
                    "consumed_problem_ids": ["p0", "p1", "p2"], "last_transaction_sha256": transaction["sha256"]}
                path = root / f"checkpoint-{slot}-stopped"
                path.mkdir()
                metadata = {"cursor": {"single": cursor}, "step": slot, "protocol": plan["protocol"]}
                write_json(path / "metadata.json", metadata)
                body = {key: value for key, value in state.items() if key != "sha256"}
                body.update(cursor=cursor, checkpoint=descriptor(path / "metadata.json"), stop_reasons=["TRAINING_CUTOFF"])
                state = body | {"sha256": digest(body)}
                write_json(root / "campaign-state.json", state)
                selected = campaign.partial_evaluation_state(plan, state, root)
                self.assertEqual(selected["evaluation_selection"]["consumed_items"], 3)
                self.assertEqual(selected["status"], "PARTIAL_EVALUATION_CANDIDATE")
                write_json(root / "evaluation-checkpoint-state.json", selected)
                for name in ("monitor-final", "effect-final", "evaluation-admission"):
                    seal(root / (name + ".json"), {"plan_sha256": plan["sha256"], "checkpoint": selected["checkpoint"],
                        "evaluation_selection": selected["evaluation_selection"]})
                with patch.object(campaign, "_plan_check"):
                    self.assertEqual(campaign.require_complete(root)["status"], "PARTIAL_EVALUATION_CANDIDATE")
                    self.assertEqual(campaign.export_metadata(metadata, plan, path)["stage"], "partial")
                for wrong_slot in (3, 8):
                    wrong = deepcopy(state)
                    wrong["cursor"]["slot"] = wrong_slot
                    self.assertIsNone(campaign.partial_evaluation_state(plan, wrong, root))
                wrong = deepcopy(state)
                wrong["v18b_normal_state"] = {key: value for key, value in state.items()
                                               if key in ("cursor", "checkpoint", "runtime", "probe", "guard")}
                wrong["stop_reasons"] = ["MONITOR_STOP"]
                self.assertIsNone(campaign.partial_evaluation_state(plan, wrong, root))

    def test_e8_may_select_actual_postprep_checkpoint_as_visibly_preparation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, state = self.fixture(root)
            transaction = seal(root / "transactions/slot-0.json", {"plan_sha256": plan["sha256"], "slot": 0, "contract": {"phase": "prep"}})
            normal = deepcopy(state["v18b_normal_state"])
            normal["cursor"].update(slot=1, updates=1, policy_version=0, consumed_problem_ids=[],
                                    last_transaction_sha256=transaction["sha256"])
            normal["runtime"] = None
            path = root / "checkpoint-1-prep-end"
            path.mkdir()
            write_json(path / "metadata.json", {"cursor": {"single": normal["cursor"]}, "step": 1, "protocol": plan["protocol"]})
            normal["checkpoint"] = descriptor(path / "metadata.json")
            write_json(root / "prep-checkpoint.json", normal["checkpoint"])
            body = {key: value for key, value in state.items() if key != "sha256"}
            body["v18b_normal_state"] = normal
            state = body | {"sha256": digest(body)}
            write_json(root / "campaign-state.json", state)
            selected = campaign.partial_evaluation_state(plan, state, root)
            self.assertEqual(selected["evaluation_selection"]["consumed_items"], 0)
            self.assertTrue(selected["evaluation_selection"]["preparation_only"])
            self.assertIsNone(selected["runtime"])
            write_json(root / "evaluation-checkpoint-state.json", selected)
            for name in ("monitor-final", "effect-final", "evaluation-admission"):
                seal(root / (name + ".json"), {"plan_sha256": plan["sha256"], "checkpoint": selected["checkpoint"],
                    "evaluation_selection": selected["evaluation_selection"]})
            with patch.object(campaign, "_plan_check"):
                self.assertEqual(campaign.require_complete(root)["status"], "PARTIAL_EVALUATION_CANDIDATE")
                exported = campaign.export_metadata(json.loads((path / "metadata.json").read_text()), plan, path)
            self.assertTrue(exported["evaluation_selection"]["preparation_only"])
            self.assertEqual(exported["stage"], "partial")
            cutoff = deepcopy(state)
            cutoff.update(normal, stop_reasons=["TRAINING_CUTOFF"])
            self.assertIsNone(campaign.partial_evaluation_state(plan, cutoff, root))
            write_json(root / "prep-checkpoint.json", state["checkpoint"])
            with self.assertRaisesRegex(ValueError, "completed prep checkpoint"):
                campaign.partial_evaluation_state(plan, state, root)

    def test_public_require_and_export_preserve_partial_status_and_selected_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, state = self.fixture(root)
            selected = campaign.partial_evaluation_state(plan, state, root)
            write_json(root / "evaluation-checkpoint-state.json", selected)
            for name in ("monitor-final", "effect-final", "evaluation-admission"):
                seal(root / (name + ".json"), {"plan_sha256": plan["sha256"],
                    "checkpoint": selected["checkpoint"], "evaluation_selection": selected["evaluation_selection"]})
            with patch.object(campaign, "_plan_check"):
                ready = campaign.require_complete(root)
                self.assertEqual(ready["status"], "PARTIAL_EVALUATION_CANDIDATE")
                path = Path(selected["checkpoint"]["path"]).parent
                metadata = json.loads((path / "metadata.json").read_text())
                exported = campaign.export_metadata(metadata, plan, path)
            self.assertEqual(exported["stage"], "partial")
            self.assertEqual(exported["evaluation_selection"], selected["evaluation_selection"])
            self.assertNotEqual(ready["metadata_descriptor"], state["checkpoint"])

    def test_main_slot_backup_boundaries_ignore_preparation_offset_and_allow_local_second_block(self):
        from fine_tuning.error_note.v16b_runtime import V16bScheduler, run_v16b
        from fine_tuning.tests.test_error_note_v16b_runtime import fixture
        sampler, items, plan, callbacks, _, _, saved = fixture(count=34)
        build, update = callbacks.rows, callbacks.update
        callbacks.rows = lambda kind, item, record: [row | {"row_id": item["problem_id"] + ":" + kind}
                                                     for row in build(kind, item, record)]
        def consume(v, rows, contract):
            return update(v, rows, contract) | {"consumed_row_ids": [row["row_id"] for row in rows]}
        callbacks.update = consume
        callbacks.checkpoint_due = lambda v: v == 1
        protocol = {"version": "error-note-v18b", "prep_updates": 15}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = V16bScheduler(sampler, plan=plan, protocol=protocol, items=items, grade=None,
                verify=None, out=root, callbacks=callbacks, local_sequential=True)
            result = run_v16b(None, None, None, sampler, plan, out=root, protocol=protocol,
                items=items, callbacks=callbacks, runtime=runtime, local_sequential=True)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual([entry[0] + 1 for entry in saved], [2, 8, 16, 24, 32, 37])

    def test_eta_tracks_current_slots_and_recent_actual_wall_time_without_double_charging(self):
        report = campaign.eta_v18b(list(range(1, 81)), done=80, total=191)
        self.assertEqual(report["done"], 80)
        self.assertEqual(report["recent_slots"], 16)
        self.assertEqual(report["mean_block_seconds"], 72.5)
        self.assertEqual(report["remaining_seconds"], 72.5 * 111)
        self.assertEqual(report["elapsed_main_seconds"], 3240)
        self.assertEqual(report["separate_checkpoint_or_monitor_charge_seconds"], 0)
        self.assertFalse(report["evaluation_added_to_training_seconds"])
        self.assertFalse(report["initial_estimate"])

    def test_public_cli_freezes_settings_backup_and_bounds_local_crash_option(self):
        args = campaign.build_parser().parse_args(["prepare", "--output", "/tmp/campaign", "--version", "error-note-v18b",
            "--execution-settings", "/tmp/settings.json", "--backup-destination", "/tmp/backups",
            "--backup-transfer-command", '["receiver", "{archive}"]'])
        self.assertEqual(args.execution_settings, Path("/tmp/settings.json"))
        self.assertEqual(args.backup_destination, Path("/tmp/backups"))
        with self.assertRaisesRegex(ValueError, "local two-block"):
            campaign.main(["advance", "--output", "/tmp/campaign", "--checkpoint-after-main-blocks", "2"])
        with patch.object(campaign, "data_api"), tempfile.TemporaryDirectory() as directory:
            args = campaign.build_parser().parse_args(["prepare", "--output", str(Path(directory) / "campaign"),
                "--version", "error-note-v18b", "--data-root", directory, "--prep-root", directory, "--model-path", directory])
            with self.assertRaisesRegex(ValueError, "production requires"):
                campaign.prepare(args)


if __name__ == "__main__":
    unittest.main()
