"""External monitor and supply guards use CPU records, never benchmark targets."""
from copy import deepcopy
import unittest

from fine_tuning.error_note.guard_v16b import SignalGateV18B
from fine_tuning.error_note import v18b_monitor as m


def record(pid, length=100, correct=True):
    return {"problem_id": pid, "prompt": "same", "output": {"status": "CORRECT" if correct else "WRONG",
        "finish_reason": "stop", "generation": {"completion_token_ids": [1] * length}}}


class MonitorTests(unittest.TestCase):
    def test_published_observation_baseline_uses_only_consumed_graded_graph_rows(self):
        def observation(truth):
            return {"obs_score": {"pg": True, "ledger": {"claims": [{"kind": "EDGE_PRESENT",
                    "original_value": True, "verdict": "VERIFIED" if truth else "CONTRADICTED"}]}}}
        records = {"first": observation(False), "next": observation(True)}
        items = [{"problem_id": pid, "source": "hf-gvlqa"} for pid in records]
        summary = m.supply_report(records, items, [], ["first:OBS", "next:RL"], slot=1)
        baseline = summary["graph_observation_baseline"]
        self.assertEqual((baseline["graded"], baseline["correct"], baseline["always_false_correct"]), (1, 0, 1))

    def fixture(self):
        items = [{"problem_id": str(i), "source": "hf-gvlqa" if i < 64 else "hf-scienceqa",
                  "training_allowed": False, "benchmark_derived": False} for i in range(115)]
        base = [record(x["problem_id"]) for x in items]
        return items, base

    def test_b2_drift_detects_sampler_collapse_including_scienceqa(self):
        items, base = self.fixture()
        collapsed = [record(x["problem_id"], length=7 if x["source"] == "hf-gvlqa" else 100) for x in items]
        report = m.monitor_comparison(base, base, collapsed, items)
        self.assertEqual(report["status"], "CONTINUE")
        self.assertEqual(report["reasons"], [])
        self.assertEqual(report["warnings"], ["B2_SHORT_DRIFT"])
        self.assertEqual(report["excluded_ceiling_n"], 0)
        self.assertEqual(report["false_alarm_calibration"], "UNMEASURED_SINGLE_F0")
        only_science = [record(x["problem_id"], length=7 if x["source"] == "hf-scienceqa" else 100) for x in items]
        science = m.short_monitor_comparison(base, base, only_science, items)
        self.assertEqual(science["status"], "CONTINUE")
        self.assertEqual(science["warnings"], ["B2_SHORT_DRIFT"])
        self.assertEqual([i for i in range(20) if m.monitor_due(i)], [8, 16])

    def test_short_drift_continues_with_insignificant_loss_or_accuracy_gain(self):
        items, base = self.fixture()
        for i in range(24, 44):
            base[i]["output"]["status"] = "WRONG"
        for recovered, lost in ((8, 14), (14, 8)):
            with self.subTest(recovered=recovered, lost=lost):
                current = deepcopy(base)
                for i in range(64):
                    current[i]["output"]["generation"]["completion_token_ids"] = [1] * 7
                for i in range(lost):
                    current[i]["output"]["status"] = "WRONG"
                for i in range(24, 24 + recovered):
                    current[i]["output"]["status"] = "CORRECT"
                report = m.monitor_comparison(base, base, current, items)
                self.assertEqual(report["status"], "CONTINUE")
                self.assertEqual(report["against_f0"]["net_change"], recovered - lost)
                self.assertGreater(report["against_f0"]["p_value"], .05)
                self.assertEqual(report["warnings"], ["B2_SHORT_DRIFT"])
                self.assertFalse(m.monitor_stop_decision(report, "evaluate_checkpoint", {}, None)["evaluate_requested"])

    def test_significant_accuracy_degradation_still_stops_with_or_without_short_drift(self):
        items, base = self.fixture()
        for short in (False, True):
            with self.subTest(short=short):
                current = deepcopy(base)
                for i in range(16):
                    current[i]["output"]["status"] = "WRONG"
                if short:
                    for i in range(64):
                        current[i]["output"]["generation"]["completion_token_ids"] = [1] * 7
                report = m.monitor_comparison(base, base, current, items)
                self.assertEqual(report["status"], "STOP_REQUESTED")
                self.assertEqual(report["reasons"], ["MONITOR_DEGRADED"])
                self.assertEqual(report["warnings"], ["B2_SHORT_DRIFT"] if short else [])

    def test_request_changes_and_training_leak_are_rejected(self):
        items, base = self.fixture()
        changed = deepcopy(base)
        changed[0]["prompt"] = "different"
        with self.assertRaisesRegex(ValueError, "REQUEST_CHANGED"):
            m.monitor_comparison(base, base, changed, items)
        items[0]["training_allowed"] = True
        with self.assertRaisesRegex(ValueError, "TRAINING_LEAK"):
            m.monitor_comparison(base, base, base, items)

    def test_short_diagnostic_never_stops_for_accuracy_and_censors_length_limit(self):
        items, base = self.fixture()
        current = deepcopy(base)
        for index, row in enumerate(current):
            row["output"]["status"] = "WRONG"
            row["output"]["generation"]["completion_token_ids"] = [1] * (31 if index < 40 else 32)
        report = m.short_monitor_comparison(base, base, current, items)
        self.assertEqual(report["status"], "CONTINUE")
        self.assertEqual(report["reasons"], [])
        self.assertEqual(report["against_f0"]["candidate_natural_short"], 40)
        self.assertEqual(report["thresholds"]["stop_rule"], "NONE")
        for row in current[:40]:
            row["output"]["finish_reason"] = "length"
        censored = m.short_monitor_comparison(base, base, current, items)
        self.assertEqual(censored["against_f0"]["candidate_natural_short"], 0)
        self.assertEqual(censored["against_f0"]["candidate_truncated"], 40)
        current[0]["output"]["generation"]["max_tokens"] = 32768
        with self.assertRaisesRegex(ValueError, "CAP_CHANGED"):
            m.short_monitor_comparison(base, base, current, items)

    def test_scienceqa_short_collapse_warns_even_when_264_item_average_dilutes_it(self):
        items = [{"problem_id": str(i), "source": "hf-scienceqa" if i < 51 else "other",
                  "training_allowed": False, "benchmark_derived": False} for i in range(264)]
        base = [record(str(i), length=7 if i < 4 else 64) for i in range(264)]
        current = [record(str(i), length=7 if i < 43 else 64) for i in range(264)]
        report = m.short_monitor_comparison(base, base, current, items)
        self.assertLess(report["against_f0"]["short_delta"], .15)
        self.assertEqual(report["warnings"], ["B2_SHORT_DRIFT"])
        self.assertEqual(report["short_drift_sources"], ["hf-scienceqa"])
        self.assertEqual(report["status"], "CONTINUE")

    def test_stop_uses_previous_normal_boundary_or_terminates(self):
        stopped = {"status": "STOP_REQUESTED"}
        decision = m.monitor_stop_decision(stopped, "evaluate_checkpoint", {"id": "bad"}, {"id": "normal"})
        self.assertEqual(decision["checkpoint"], {"id": "normal"})
        self.assertFalse(decision["auto_evaluate"])
        self.assertIsNone(m.monitor_stop_decision(stopped, "stop", {}, None)["checkpoint"])
        with self.assertRaisesRegex(ValueError, "PREVIOUS_NORMAL"):
            m.monitor_stop_decision(stopped, "evaluate_checkpoint", {}, None)

    def test_projection_retains_b2_scenario_and_uncertainty_without_launch(self):
        items, base = self.fixture()
        short = [record(x["problem_id"], length=50) for x in items]
        kwargs = dict(reference_mmmu_tokens=4120000, throughput_tokens_per_second=177, hourly_usd=1.42,
                      elapsed_seconds=3600, remaining_train_seconds=20000, bootstrap_trials=100)
        value = m.project_mmmu_lengths(base, short, items, **kwargs)
        self.assertEqual(value["length_ratio"], .5)
        self.assertEqual(value["scenarios"]["upper"]["tokens"], 4120000)
        self.assertFalse(value["auto_launch"])
        split = m.project_mmmu_lengths(base, short, items, evaluation_pods=2, second_pod_overhead_seconds=2000, **kwargs)
        self.assertGreater(split["scenarios"]["upper"]["estimated_usd"], value["scenarios"]["upper"]["estimated_usd"])
        self.assertLess(split["scenarios"]["upper"]["total_seconds"], value["scenarios"]["upper"]["total_seconds"])

    def test_delayed_fixed_gate_does_not_assign_other_parents_tokens(self):
        order = [[str(i)] for i in range(40)]
        gate = SignalGateV18B(order, [str(i) for i in range(40)])
        stats = {"optimizer_step": True, "n_content_gradient_finite_nonzero": True,
            "note_gradient_retained": True, "n_verified_content_tokens": 1000,
            "n_parent_evidence": [{"problem_id": "39", "row_id": "39:N", "supervised_tokens": 5,
                                   "gradient_finite_nonzero": True, "note_gradient_retained": True}]}
        gate.consume(stats)
        gate.mark_checkpoint("checkpoint-one")
        report = gate.evaluate(34, [str(i) for i in range(32)], {"OBS": [str(i) for i in range(8)]})
        self.assertEqual(report["verified_content_tokens"], 0)
        self.assertTrue(report["early_zero_diagnostic"])
        self.assertFalse(report["stop_requested"])
        self.assertTrue(gate.evaluate(34, [str(i) for i in range(32)], {"OBS": [str(i) for i in range(8)]},
                                     contract_failure_reproduced=True)["stop_requested"])
        gate = SignalGateV18B(order, [str(i) for i in range(40)])
        self.assertFalse(gate.evaluate(2, ["0"], {"T1": ["0"]})["early_zero_diagnostic"])
        stats["n_parent_evidence"][0].update(problem_id="0", row_id="0:N", supervised_tokens=7)
        gate.consume(stats)
        self.assertEqual(gate.evaluate(34, [str(i) for i in range(32)], ["0"])["verified_content_tokens"], 0)
        gate.mark_checkpoint("checkpoint-two")
        self.assertEqual(gate.evaluate(34, [str(i) for i in range(32)], ["0"])["verified_content_tokens"], 7)
        resumed = SignalGateV18B(order, [str(i) for i in range(40)], gate.state)
        self.assertEqual(resumed.state["saved_rows"]["0:N"]["checkpoint_id"], "checkpoint-two")


    def test_public_guard_materialization_v18b_keeps_new_rules_unmeasured_and_rejects_old_waiver(self):
        import hashlib
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from fine_tuning.error_note import guard_v16b as guards
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            legacy = {"version": "v17-guard-falsealarm-v1", "manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
                "status": "PASS_SCOPED", "measured_status": "PASS", "overall_upper95": .01, "trials": 2000,
                "unmeasured_rules": ["IS_DRIFT", "TAIL_RATE"], "method": "fixed-position-stratified-fill-bootstrap-v1"}
            target = root / "receipt.json"
            with patch("fine_tuning.error_note.v18b_data.validate") as validate, \
                    patch.object(guards, "materialize_guard_inputs_v18", return_value={"manifest": str(manifest),
                        "extra_priors": str(root / "extra.json")}) as materialize, \
                    patch.object(guards, "falsealarm_receipt_v17", return_value=legacy):
                self.assertEqual(guards.main(["--version", "error-note-v18b", "--materialize-data-root", str(root),
                    "--output", str(target)]), 0)
                validate.assert_called_once_with(root)
                materialize.assert_called_once()
            receipt = json.loads(target.read_text())
            self.assertIn("B2_SHORT_DRIFT", receipt["unmeasured_rules"])
            self.assertIn("MONITOR_DEGRADED", receipt["unmeasured_rules"])
            self.assertEqual(receipt["external_b2_actions"],
                             {"MONITOR_DEGRADED": "STOP_REQUESTED", "B2_SHORT_DRIFT": "WARNING"})
            self.assertFalse(receipt["inherited_acceptance"])
            self.assertTrue(guards.receipt_passes_v18b(receipt, b"{}"))
            self.assertFalse(guards.receipt_passes_v18b(receipt | {"version": "v18-guard-falsealarm-v1"}, b"{}"))


    def test_actual_campaign_prepare_dispatches_smoke_and_normal_v18b_gate(self):
        import contextlib
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from fine_tuning.error_note import campaign_v16b as campaign, v18b_data
        from fine_tuning.error_note.prep_v16b import seal
        from fine_tuning.tests.test_error_note_v17_campaign import fixture
        from fine_tuning.tests.test_error_note_v18b_eval import settings
        for smoke in (True, False):
            with self.subTest(local_smoke=smoke), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args, native, patches = fixture(root, smoke=smoke)
                args.version = "error-note-v18b"
                for i, item in enumerate(native[:8]):
                    item["layer"] = "graph" if i < 6 else "scene"
                if not smoke:
                    args.execution_settings = root / "execution.json"
                    seal(args.execution_settings, {"version": "v18b-execution-settings-v1", "settings": settings()})
                    args.backup_destination = root / "backups"
                    args.backup_transfer_command = ["cpu-fixture-never-executed", "{archive}"]
                with contextlib.ExitStack() as stack:
                    for p in patches:
                        stack.enter_context(p)
                    stack.enter_context(patch.object(v18b_data, "validate", return_value={"reasoning_monitor_count": 64}))
                    stack.enter_context(patch.object(v18b_data, "load_items", side_effect=lambda root, split:
                        [row for row in native if row["problem_id"].startswith(split)]))
                    result = campaign.prepare(args)
                self.assertEqual(result["version"], "single-campaign-v18b")
                self.assertEqual(result["protocol"]["version"], "error-note-v18b")
                self.assertEqual(len(result["signal_gate"]["parents"]), 8 if smoke else 32)
                campaign._plan_check(result)


if __name__ == "__main__":
    unittest.main()
