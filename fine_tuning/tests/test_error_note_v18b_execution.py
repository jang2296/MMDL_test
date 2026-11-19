"""The operator's financial, generation and training choices reach the real CLI."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fine_tuning.error_note import collect, plan, v18_data_range, v18b_execution as policy
from fine_tuning.notebook_rl.pod.chain import check_spec
from fine_tuning.tests.test_error_note_v17_split_plan import split_params


def settings():
    return {"run_id": "private-v18b", "first_billed_at": "2026-11-17T00:00:00Z",
            "total_timeout_seconds": 14 * 3600, "recovery_reserve_seconds": 600,
            "post_eval_reserve_seconds": 1200, "evaluation_strategy": "one_pod",
            "training_cutoff_seconds": None, "monitor_stop_action": "stop", "reuse_f0_raw": False,
            "billing": [{"first_billed_at": "2026-11-17T00:00:00Z", "usd_per_hour": 1.40}]}


class ExecutionTests(unittest.TestCase):
    def test_outer_chain_runs_the_command_after_an_advisory_target(self):
        from fine_tuning.notebook_rl.pod import chain
        from fine_tuning.notebook_rl.pod.common import sha256_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            completed = root / "completed"
            decisions = settings() | {"limit_action": "warn", "budget_limit_usd": None}
            value = {"version": chain.SPEC_VERSION, "meta": {"protocol_version": "error-note-v18b",
                     "execution_settings": decisions}, "steps": [{"name": "work", "argv": [sys.executable,
                     "-c", "from pathlib import Path;Path(__import__('sys').argv[1]).write_text('done')", str(completed)],
                     "expect_files": [str(completed)]}]}
            spec_path.write_text(json.dumps(value))
            with patch.dict("os.environ", MMDL_RUN_DIR=directory), \
                    patch.object(policy, "remaining", return_value=-1):
                self.assertEqual(chain.run(spec_path, sha256_file(spec_path)), 0)
            self.assertEqual(completed.read_text(), "done")

    def test_one_clock_across_training_second_pod_and_recovery(self):
        decisions = settings()
        start = policy._epoch(decisions["first_billed_at"])
        self.assertEqual(policy.remaining(decisions, now=start+600, reserve_seconds=600), 14*3600-1200)
        decisions["evaluation_strategy"] = "two_pod"
        decisions["billing"].append({"first_billed_at": "2026-11-17T05:00:00Z", "usd_per_hour": 1.40})
        deadline = policy.deadline_epoch(decisions)
        self.assertLess(deadline, start+14*3600)
        self.assertAlmostEqual(policy.projected_cost(decisions, deadline), 22, places=6)
        self.assertLess(policy.remaining(decisions, now=deadline, reserve_seconds=600), 0)
        with self.assertRaises(ValueError):
            policy.validate_settings(decisions | {"training_cutoff_seconds": 100})

    def test_completion_mode_warns_after_time_and_money_targets(self):
        decisions = settings() | {"limit_action": "warn", "budget_limit_usd": 22.,
                                  "total_timeout_seconds": 48 * 3600}
        start = policy._epoch(decisions["first_billed_at"])
        self.assertEqual(policy.validate_settings(decisions), decisions)
        self.assertFalse(policy.enforce_limits(decisions))
        self.assertGreater(policy.projected_cost(decisions, start + 24 * 3600), 22)
        self.assertEqual(policy.deadline_epoch(decisions), start + 48 * 3600)
        self.assertLess(policy.remaining(decisions, now=start + 49 * 3600), 0)
        self.assertFalse(policy.enforce_limits(decisions))
        self.assertEqual(policy.planning_status(decisions, now=start + 49 * 3600)["warnings"],
                         ["PLANNING_TIME_TARGET_EXCEEDED", "PLANNING_COST_TARGET_EXCEEDED"])
        with self.assertRaises(ValueError):
            policy.validate_settings(decisions | {"evaluation_strategy": "training_cutoff", "training_cutoff_seconds": 600})

    def test_public_train_chain_rebuilds_data_and_admits_in_same_process(self):
        p = split_params(Path("/private/train"), "v17-train")
        for name in (*plan.V17_TRAIN_HASHED,):
            del p[name + "_path"], p[name + "_sha256"]
        del p["functional_receipt_path"], p["bundle_archive_path"]
        p.update(version=plan.PARAMS_VERSION_V18B_SPLIT, execution_settings=settings(),
                 cache_dir="/private/cache", backup_destination="/offhost/backup")
        with self.assertRaisesRegex(ValueError, "off-pod backup receiver"):
            plan.plan("v18b-train", p)
        p["backup_transfer_command"] = ["/private/receive", "{archive}", "{bundle_sha256}"]
        spec = check_spec(plan.plan("v18b-train", p))
        steps = {s["name"]: s["argv"] for s in spec["steps"]}
        self.assertIn("fine_tuning.error_note.v18b_data", steps["prepare-data"])
        self.assertNotIn("bf16-functional-calibration", steps)
        self.assertIn("--execution-settings", steps["campaign-advance"])
        self.assertNotIn("--allow-f0-raw-reuse", steps["prep-generate"])
        self.assertEqual(spec["meta"]["execution_settings"], settings())
        p["execution_settings"]["reuse_f0_raw"] = True
        with self.assertRaises(ValueError):
            plan.plan("v18b-train", p)
        for role in ("source", "evidence"):
            p["f0_reuse_" + role + "_path"] = "/private/" + role
            p["f0_reuse_" + role + "_sha256"] = "a"*64
        self.assertIn("--allow-f0-raw-reuse", next(s["argv"] for s in plan.plan("v18b-train", p)["steps"]
                                                    if s["name"] == "prep-generate"))
        p["execution_settings"].update(limit_action="warn", budget_limit_usd=None, total_timeout_seconds=86400)
        spec = plan.plan("v18b-train", p)
        self.assertEqual(spec["meta"]["financial_mode"], "measured_balance_reserve")
        self.assertIsNone(spec["meta"]["total_timeout_seconds"])
        self.assertEqual(spec["meta"]["planning_target_seconds"], 86400)

    def test_only_held_out_b2_may_use_the_short_diagnostic_contract(self):
        request = {"problem_id": "held", "request_id": "id", "stage": "v18b-b2-short-monitor",
                   "block_id": "v18b-b2-short-monitor", "seed": 23, "max_tokens": 64,
                   "messages": [], "images": [], "evaluation_monitor": True,
                   "protocol_version": "error-note-v18b", "sampling": copy.copy(collect.SINGLE_SAMPLING)}
        self.assertEqual(collect.single_request_binding(request)["max_tokens"], 64)
        for change in ({"stage": "T1"}, {"max_tokens": 32768}, {"sampling": {}},
                       {"protocol_version": "error-note-v18"}):
            with self.assertRaises(ValueError):
                collect.single_request_binding(request | change)

    def test_range_timeout_has_one_retry_budget(self):
        source = {"url": "https://example.invalid/archive", "archive_bytes": 4096, "etag": '"fixture"'}
        meter = {"requests": 0, "received_bytes": 0}
        with patch.object(v18_data_range.urllib.request, "urlopen", side_effect=TimeoutError), \
                patch.object(v18_data_range.time, "sleep") as delay, self.assertRaises(TimeoutError):
            v18_data_range._range(source, 100, 32, meter)
        self.assertEqual((meter["requests"], delay.call_count), (5, 4))
