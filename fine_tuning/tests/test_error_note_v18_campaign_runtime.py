"""Short CPU checks for production dispatch, delayed closure and scoped telemetry."""
import json
from pathlib import Path
import tempfile
import unittest

from fine_tuning.error_note import campaign_v16b as campaign, guard_v16b as guards, prep_v16b as prep
from fine_tuning.error_note.audit import digest
from fine_tuning.error_note.v16b_runtime import (OBS_TASK_V18, V16bScheduler,
    observation_baseline_v18, run_v16b)
from fine_tuning.tests.test_error_note_v16b_runtime import fixture


class BaselineTests(unittest.TestCase):
    def test_production_requests_bind_seed_for_every_v18_generation_stage(self):
        import hashlib
        from fine_tuning.error_note.protocol import TRAIN_SAMPLING_V16B

        item = {"problem_id": "fixture", "image_paths": [], "image_sha256": []}
        for suffix in ("prep", "prep-obs", "monitor", "effect", "t1", "obs", "note", "resolve"):
            stage = f"v18-{suffix}"
            with self.subTest(stage=stage):
                request = prep.make_request(item, "fixed question", stage, cap=16, sampling=TRAIN_SAMPLING_V16B,
                                            protocol_sha256="a" * 64, policy_sha256="b" * 64)
                expected = hashlib.sha256(f"3407:fixture:{stage}".encode()).digest()
                self.assertEqual(request["seed"], int.from_bytes(expected[:8], "big") % (2**31))
                self.assertEqual(request["stage"], stage)

    def test_first_binary_advantage_and_task_isolation(self):
        receipt = observation_baseline_v18("geometry", [], history_cutoff=-2)
        self.assertEqual((1 - receipt["baseline"], 0 - receipt["baseline"]), (.5, -.5))
        self.assertEqual(receipt["initial"], .5)
        self.assertEqual(receipt["sha256"], digest({k: v for k, v in receipt.items() if k != "sha256"}))
        base = {"kind": "OBS", "layer": "geometry", "committed": True, "pg": True,
                "task": OBS_TASK_V18, "block": 0, "reward": 1.}
        history = [base, base | {"task": "v17-list-f1", "reward": .3}, base | {"pg": False, "reward": None},
                   base | {"block": 2}, base | {"layer": "graph"}]
        actual = observation_baseline_v18("geometry", history, history_cutoff=0)
        self.assertEqual((actual["baseline"], actual["window_count"]), (5 / 9, 1))


class ClosureTests(unittest.TestCase):
    def execute(self, root):
        sampler, items, plan, callbacks, _, calls, saved = fixture(count=2)
        build = callbacks.rows
        callbacks.rows = lambda kind, item, record: [row | {"row_id": item["problem_id"] + ":" + kind}
                                                     for row in build(kind, item, record)]
        update = callbacks.update
        def consume(v, rows, contract):
            stats = update(v, rows, contract)
            stats.update(consumed_row_ids=[r["row_id"] for r in rows],
                         note_update_alignment={"gradient_measured": False}, policy_mask_counts={})
            return stats
        callbacks.update = consume
        runtime = V16bScheduler(sampler, plan=plan, protocol={"version": "error-note-v18"}, items=items,
            grade=None, verify=None, out=root, local_sequential=True, callbacks=callbacks)
        result = run_v16b(None, None, None, sampler, plan, out=root, protocol={"version": "error-note-v18"},
                         items=items, callbacks=callbacks, runtime=runtime, local_sequential=True)
        return runtime, result, calls, saved

    def test_last_parent_note_and_resolve_consumed_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, result, calls, saved = self.execute(Path(tmp))
            self.assertEqual(result["status"], "COMPLETE")
            closure = runtime.terminal_closure([r["row_id"] for _, rows, _, _ in calls for r in rows])
            self.assertEqual((closure["pending_requests"], closure["unconsumed_rows"]), (0, 0))
            self.assertEqual(closure["accepted_rows"], 8)
            self.assertEqual(calls[-1][1][0]["row_id"], "p1:RESOLVE")
            self.assertEqual(calls[-1][1][0]["generation"]["policy_version"], 3)
            self.assertFalse(saved[-1][2])
            timing = json.loads((Path(tmp) / "update-timing-4.json").read_text())["timing"]
            self.assertEqual(timing["note_update_alignment"], {"gradient_measured": False})
            self.assertEqual(timing["schema"], "v18-update-timing-v1")
            with self.assertRaisesRegex(RuntimeError, "cursor"):
                runtime.terminal_closure([])
            runtime.row_settlements.pop("4")
            with self.assertRaisesRegex(RuntimeError, "settlement"):
                runtime.terminal_closure()

    def test_missing_consumption_fails_and_hold_has_explicit_discard(self):
        sampler, items, plan, callbacks, *_ = fixture(1)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = V16bScheduler(sampler, plan=plan, protocol={"version": "error-note-v18"}, items=items,
                grade=None, verify=None, out=Path(tmp), local_sequential=True, callbacks=callbacks)
            rows = [{"row_id": "p0:N", "generation": {"policy_version": 1}}]
            with self.assertRaisesRegex(RuntimeError, "telemetry"):
                runtime.settle_rows(0, rows, {})
            with self.assertRaisesRegex(RuntimeError, "neither consumed"):
                runtime.settle_rows(0, rows, {"consumed_row_ids": []})
            runtime.settle_rows(0, rows, {"consumed_row_ids": [], "hold": True, "hold_reason": "REPLAY"})
            self.assertEqual(runtime.row_settlements["0"]["discarded_row_ids"], ["p0:N"])


class PortableGuardTests(unittest.TestCase):
    def test_fresh_materialization_does_not_inherit_accepted_receipt(self):
        prior = {"binding": "EXACT_NATIVE_ID_IMAGE_QUESTION_CHOICES_GOLD", "status": "CORRECT",
                 "completion_tokens": 100, "finish_reason": "stop"}
        items = [{"problem_id": str(i), "source": "s", "layer": "geometry", "f0_prior": prior} for i in range(3000)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = root / "sidecar.json"
            sidecar.write_text(json.dumps({"items": {}, "guard": {"items": items, "extra_priors": {}}}))
            staged = guards.materialize_guard_inputs_v18(sidecar, root / "guard")
            manifest = Path(staged["manifest"])
            result = guards.falsealarm_receipt_v18(manifest, trials=2, extra_priors=Path(staged["extra_priors"]))
            self.assertFalse(result["inherited_acceptance"])
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertFalse(guards.receipt_passes_v18(result, manifest.read_bytes()))
            good = result | {"trials": 2000, "status": "PASS_SCOPED", "measured_status": "PASS", "overall_upper95": .01}
            self.assertTrue(guards.receipt_passes_v18(good, manifest.read_bytes()))
            self.assertFalse(guards.receipt_passes_v18(good | {"inherited_acceptance": True}, manifest.read_bytes()))

    def test_normal_campaign_prepare_dispatches_v18_protocol_and_data(self):
        import contextlib
        from unittest.mock import patch
        from fine_tuning.tests.test_error_note_v17_campaign import fixture as campaign_fixture
        with tempfile.TemporaryDirectory() as tmp:
            args, native, patches = campaign_fixture(Path(tmp))
            args.version = "error-note-v18"
            def load(root, split):
                return [item for item in native if item["problem_id"].startswith(split)]
            with contextlib.ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                stack.enter_context(patch("fine_tuning.error_note.v18_data.validate", return_value={"effect_eligible": False}))
                stack.enter_context(patch("fine_tuning.error_note.v18_data.load_items", side_effect=load))
                plan = campaign.prepare(args)
            self.assertEqual(plan["version"], "single-campaign-v18")
            self.assertEqual(plan["protocol"]["version"], "error-note-v18")
            self.assertEqual(plan["total_slots"], 8)
            campaign._plan_check(plan)

    def test_operator_risk_acceptance_preserves_failed_measurement(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            receipt = {"version": "v18-guard-falsealarm-v1", "inherited_acceptance": False,
                "manifest_sha256": guards.hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "overall_upper95": .03, "status": "FAIL", "measured_status": "FAIL", "trials": 2000,
                "unmeasured_rules": ["IS_DRIFT", "TAIL_RATE"],
                "method": "fixed-position-stratified-fill-bootstrap-v1"}
            target = root / "receipt.json"
            with patch.object(guards, "falsealarm_receipt_v18", return_value=receipt):
                result = guards.main(["--version", "error-note-v18", "--manifest", str(manifest),
                    "--output", str(target), "--risk-acceptance", "Operator explicitly accepts fixed scoped risk"])
            self.assertEqual(result, 0)
            saved = json.loads(target.read_text())
            self.assertEqual((saved["status"], saved["measured_status"], saved["overall_upper95"]),
                             ("RISK_ACCEPTED", "FAIL", .03))
            target = root / "passing-receipt.json"
            passed = receipt | {"status": "PASS_SCOPED", "measured_status": "PASS", "overall_upper95": .01}
            with patch.object(guards, "falsealarm_receipt_v18", return_value=passed):
                result = guards.main(["--version", "error-note-v18", "--manifest", str(manifest),
                    "--output", str(target), "--risk-acceptance", "Applies only if needed"])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(target.read_text())["status"], "PASS_SCOPED")

    def test_version_parsers_are_explicit(self):
        args = campaign.build_parser().parse_args(["prepare", "--output", "/tmp/cpu", "--version", "error-note-v18"])
        self.assertEqual(args.version, "error-note-v18")
        args = prep.build_parser().parse_args(["select", "--data-root", "/tmp/data", "--output", "/tmp/cpu", "--version", "error-note-v18"])
        self.assertEqual(args.version, "error-note-v18")


class ReplayTests(unittest.TestCase):
    def test_real_row_helpers_keep_sampled_archive_immutable(self):
        from copy import deepcopy
        from fine_tuning.error_note.v18_replay import replay_parent
        from fine_tuning.tests.test_error_note_v18_rows import map_record
        from fine_tuning.tests.test_error_note_v17_rows import TOKENIZER, verifier

        native = map_record()
        item = {k: v for k, v in native.items() if k != "t1"}
        item["evaluation"] = {"question": native["question"], "answer_type": "text", "choices": [], "gold": "Iowa"}
        item["v17_private_verifier"] = verifier(item, [])
        before = deepcopy(native)
        report, admitted = replay_parent(item, native, tokenizer=TOKENIZER,
            grade=lambda item, output: ("INCORRECT", output["text"]),
            grade_v8=lambda item, output: ("INCORRECT", None))
        self.assertEqual(native, before)
        self.assertEqual(report["policy"]["reason"], "FORMAT_ONLY_CERTIFIED")
        self.assertTrue(report["note_requested"])
        self.assertFalse(admitted)
        self.assertEqual((report["generated_answers"], report["gpu_updates"]), (0, 0))

    def test_archive_directory_rejects_duplicates_mismatch_and_changed_plan(self):
        from fine_tuning.error_note.v18_replay import read_archive
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = root / "runtime" / "items"
            items.mkdir(parents=True)
            plan = {"items": [{"problem_id": "parent"}]}
            plan["sha256"] = digest(plan)
            (root / "campaign-plan.json").write_text(json.dumps(plan))
            (items / "native.json").write_text(json.dumps({"problem_id": "parent"}))
            saved, records = read_archive(root)
            self.assertEqual((saved, list(records)), (plan, ["parent"]))
            (items / "duplicate.json").write_text(json.dumps({"problem_id": "parent"}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_archive(root)
            (items / "duplicate.json").unlink()
            plan["items"][0]["problem_id"] = "changed"
            (root / "campaign-plan.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "seal"):
                read_archive(root)


class SupplyReportTests(unittest.TestCase):
    def test_fixed_512_and_remainder_keep_real_note_gradient_and_c_use_separate(self):
        items = [{"problem_id": str(i), "source": "s", "v17_private_verifier": {"repair_catalog_eligible": True}}
                 for i in range(513)]
        record = {"t1": {}, "t1_ledger": {"claims": [{"verdict": "CONTRADICTED"}]},
                  "note": {}, "note_receipt": {"n_accepted": True, "verified_repair_count": 1, "note_condition": "self_found"},
                  "resolve_receipt": {"accepted": True, "corrected_use_byte_spans": [[0, 1]]}}
        stats = {"note_parent_gradient_norms": {"512": {"L": .5}}, "verified_repair_parent_ids": ["512"],
                 "note_update_alignment": {"parameter_delta_dot": -.1}}
        updates = [{"global_slot": 10, "result": {"optimizer_steps": 1, "stats": stats}}]
        result = campaign.v18_supply_report({"512": record}, items, updates, ["512:N", "512:C"])
        self.assertEqual([r["counts"]["parents"] for r in result["cohorts"]], [512, 1])
        last = result["cohorts"][1]["counts"]
        self.assertEqual((last["n_accepted"], last["n_gradient_measured"], last["n_gradient_nonzero"],
                          last["n_saved_update_linked"], last["c_repair_used_and_consumed"]), (1, 1, 1, 1, 1))
        self.assertEqual(result["joint_note_update_alignment"][0]["measurement"], stats["note_update_alignment"])
        self.assertFalse(result["effect_measured"])


class PublicReceiptTests(unittest.TestCase):
    def test_exact_allowlist_and_metadata_only_replay_evidence(self):
        from scripts.sync_test_repo import AUTHORED_JSON_PATHS
        name = "fine_tuning/error_note/v18_replay_receipt.json"
        self.assertIn(name, AUTHORED_JSON_PATHS)
        root = Path(__file__).resolve().parents[2]
        raw = (root / name).read_text()
        body = json.loads(raw)
        self.assertEqual(body["sha256"], digest({k: v for k, v in body.items() if k != "sha256"}))
        for private in ('"question"', '"raw_response"', '"gold"', '"problem_id"', '/home/', '/workspace/'):
            self.assertNotIn(private, raw)
        self.assertTrue(body["full_campaign"])
        self.assertEqual(body["aggregate_counts"]["ALL"]["parents"], 3000)
        self.assertEqual((body["aggregate_counts"]["ALL"]["N"], body["aggregate_counts"]["ALL"]["C"]), (0, 0))
        self.assertTrue(body["historical_reaudit"]["all_equal"])
        self.assertEqual((body["generated_answers"], body["gpu_updates"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
