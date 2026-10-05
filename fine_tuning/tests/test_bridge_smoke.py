"""CPU guards and synthetic journal checks; never GPU memory/update certification."""
import json
import random
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fine_tuning import audit, run_bridge_smoke, run_control
from fine_tuning.train import order_hash
from fine_tuning.tests import test_controller as fixtures


@contextmanager
def historical_processor_fixture():
    """CPU fixture only: no processor execution or current-source reapproval."""
    real_sha, producer = audit.sha, Path(audit.__file__).resolve()
    def fixture_sha(path):
        if Path(path).resolve() == producer:
            return "bba0ed1b202df81ab61a27760fec4921b31d92508a06141abfac5160d3fb6aae"
        return real_sha(path)
    with patch.object(audit, "sha", side_effect=fixture_sha):
        yield


class BridgeSmokeTests(unittest.TestCase):
    def test_historical_fixture_rejects_changed_producer_and_binds_all84_inputs(self):
        root = Path(run_bridge_smoke.__file__).parent
        approval = root/"runs/recovered_graph_pool32_child_approval_v2/proof"
        pool, protocol = root/"generated/graph_cs_bridge_split_pool32_v2", root/"bridge_training_protocol_v1.json"
        with self.assertRaisesRegex(ValueError, "approval/source/processor identity mismatch"):
            run_bridge_smoke.inputs(approval, pool, protocol)
        with historical_processor_fixture():
            rows, _manifest, _records, contract, frozen = run_bridge_smoke.inputs(approval, pool, protocol)
            self.assertEqual((len(rows), len({r["root_id"] for r in rows})), (84, 28))
            self.assertEqual({r["split"] for r in rows}, {"train"})
            self.assertEqual(contract["episode_pixels"], 3145728)
            self.assertTrue(all(audit.sha(Path(p)) == digest for p, digest in frozen.items()))
            with tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary)/"proof"
                shutil.copytree(approval, copied)
                receipt = copied/"A3_processor.json"
                receipt.write_bytes(receipt.read_bytes()+b" ")
                with self.assertRaisesRegex(ValueError, "approval/source/processor identity mismatch"):
                    run_bridge_smoke.inputs(copied, pool, protocol)

    def test_complete84_journal_tail_is_checked_and_control_remains1600(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, rows, records, manifest, cfg, _protocol, checkpoint = fixtures.ControllerTests().pilot_reuse_fixture(Path(tmp))
            rows = rows[:84]
            records.write_text("".join(json.dumps(row)+"\n" for row in rows))
            indices = list(range(84))
            random.Random(3407).shuffle(indices)
            ids = [rows[i]["id"] for i in indices]
            consumed = [{"id": sid, "position": i, "input_tokens": 2, "target_tokens": 1} for i,sid in enumerate(ids)]
            (checkpoint.parent/"consumed_ids.jsonl").write_text("".join(json.dumps(r)+"\n" for r in consumed))
            updates = [{"step": step, "consumed_records": min(step*16,84),
                "consumed_order_sha256": order_hash(ids[:min(step*16,84)]), "loss_tokens": min(step*16,84)-(step-1)*16,
                "gradient_l2": {"L": 1.}, "sampled_update_l1": {"L": 1.}, "frozen_sampled_checks": 9,
                "peak_allocated_bytes": 1, "peak_reserved_bytes": 2} for step in range(1,7)]
            (checkpoint.parent/"updates.jsonl").write_text("".join(json.dumps(r)+"\n" for r in updates))
            metadata = json.loads((checkpoint/"metadata.json").read_text())
            contract = {"fixture": "explicit child"}
            metadata.update(planned_records=84, exposure={"records":84,"images":84,"updates":6,
                "input_tokens":168,"target_tokens":84,"visual_tokens":84}, training_input_contract=contract,
                records_sha256=audit.sha(records), record_order_sha256=order_hash(ids), consumed_order_sha256=order_hash(ids),
                consumed_ids_sha256=audit.sha(checkpoint.parent/"consumed_ids.jsonl"))
            (checkpoint/"metadata.json").write_text(json.dumps(metadata))
            result = run_control.verify_fl_training(parent/"pilot_FL", rows, records, manifest, cfg, metadata, contract)
            self.assertEqual(result["exposure"]["updates"], 6)
            with self.assertRaises(ValueError):
                run_control.verify_training(parent/"pilot_FL", rows, manifest, cfg, metadata)
            with self.assertRaises(ValueError):
                run_control.verify_fl_training(parent/"pilot_FL", rows, records, manifest, cfg, metadata, {"wrong":True})
            updates[-1]["loss_tokens"] = 16
            (checkpoint.parent/"updates.jsonl").write_text("".join(json.dumps(r)+"\n" for r in updates))
            with self.assertRaises(ValueError):
                run_control.verify_fl_training(parent/"pilot_FL", rows, records, manifest, cfg, metadata, contract)

    def test_live_parent_or_insufficient_budget_cannot_start_gpu(self):
        args = SimpleNamespace(previous_pid=1, previous_birth="1")
        controller = Mock()
        with patch.object(run_bridge_smoke, "process_gone", return_value=False):
            with self.assertRaises(ValueError):
                run_bridge_smoke.run(args, controller)
        controller.stage.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(previous_pid=1, previous_birth="1", previous_controller=root/"closed",
                approval=root/"approval", image_root=root/"images", input_protocol=root/"child",
                reference_checkpoint=root/"reference", model_path=root/"model")
            audit.dump(root/"model/config.json", {"vision_config":{"spatial_merge_size":2}})
            audit.dump(root/"reference/metadata.json", {"arm":"FL", "base_revision":run_bridge_smoke.BASE_REVISION,
                "epoch_exposure_complete":True, "planned_records":84, "exposure":{"records":84,"updates":6},
                "consumed_order_sha256":"same", "record_order_sha256":"same"})
            controller.output = root/"result"
            controller.remaining.return_value = 1019
            with patch.object(run_bridge_smoke, "process_gone", return_value=True), \
                 patch.object(run_bridge_smoke.runner, "closed_controller", return_value=({},[],{})), \
                 patch.object(run_bridge_smoke, "inputs", return_value=([{}]*84,root/"manifest",root/"records",{},{})), \
                 patch.object(run_bridge_smoke, "require_idle_gpu") as idle:
                result = run_bridge_smoke.run(args, controller)
            self.assertEqual(result["status"], "BUDGET_STOP")
            self.assertFalse(result["GPU_training_started"])
            idle.assert_not_called()
            controller.stage.assert_not_called()
            controller.output = root/"post_check_budget_stop"
            controller.remaining.side_effect = [1100, 749]
            contract = {"min_pixels":262144,"max_pixels":1048576,"episode_pixels":3145728,"sequence_budget":4096}
            with patch.object(run_bridge_smoke, "process_gone", return_value=True), \
                 patch.object(run_bridge_smoke.runner, "closed_controller", return_value=({},[],{})), \
                 patch.object(run_bridge_smoke, "inputs", return_value=([{}]*84,root/"manifest",root/"records",contract,{})), \
                 patch.object(run_bridge_smoke, "require_idle_gpu") as idle:
                with self.assertRaises(TimeoutError):
                    run_bridge_smoke.run(args, controller)
            idle.assert_called_once()
            controller.stage.assert_not_called()

    def test_closed_control_status_requires_matching_terminal_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = "ADDITIONAL_FL_NVW0_VS10_PILOT_COMPLETED"
            audit.dump(root/"status.json", {"status":state})
            audit.dump(root/"source_freeze.json", {})
            audit.write_rows(root/"journal.jsonl", [{"stage":"controller","status":state}])
            self.assertEqual(run_bridge_smoke.runner.closed_controller(root)[0]["status"], state)
            (root/"journal.jsonl").write_text(json.dumps({"stage":"pilot_FL_external","status":"RUNNING"})+"\n")
            with self.assertRaises(ValueError):
                run_bridge_smoke.runner.closed_controller(root)


if __name__ == "__main__":
    unittest.main()
