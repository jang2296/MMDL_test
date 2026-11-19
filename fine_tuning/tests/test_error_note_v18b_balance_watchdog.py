"""Completion runs stop on a measured reserve, including recharge and unavailable telemetry."""
from __future__ import annotations

from contextlib import ExitStack
import datetime as dt
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from fine_tuning import budget
from fine_tuning.notebook_rl.pod import job, runner
from fine_tuning.tests.test_notebook_rl_pod import approved_job, prep_job, seal


class MeasuredBalanceTests(unittest.TestCase):
    def test_only_fresh_matched_balance_observations_are_trusted(self):
        instant = dt.datetime.now(dt.timezone.utc)
        value = {"version": "v18b-balance-observation-v1", "owned_pod_id": "owned",
                 "observed_at": instant.isoformat(), "balance_usd": 8.}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "balance.json"
            path.write_text(json.dumps(value))
            self.assertEqual(budget.measured_balance(str(path), "owned", now=instant), 8.)
            for changed in ({"owned_pod_id": "someone-else"}, {"balance_usd": False},
                            {"observed_at": (instant - dt.timedelta(minutes=11)).isoformat()},
                            {"observed_at": (instant + dt.timedelta(minutes=2)).isoformat()}):
                path.write_text(json.dumps(value | changed))
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    budget.measured_balance(str(path), "owned", now=instant)

    def exercise(self, balances, waits, *, technical_seconds=0):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            receipt = Path(directory) / "reserve.json"
            result = Path(directory) / "result.json"
            child = Mock(pid=12345)
            child.wait.side_effect = waits
            stack.enter_context(patch.object(budget, "measured_balance", side_effect=[20., *balances]))
            stack.enter_context(patch.object(budget.subprocess, "Popen", return_value=child))
            stack.enter_context(patch.object(budget, "terminate_group"))
            stack.enter_context(patch.dict("os.environ", RUNPOD_POD_ID="owned"))
            stamp = dt.datetime.now(dt.timezone.utc).isoformat()
            argv = ["budget", "--owned-pod-id", "owned", "--gpu-cutoff-utc", stamp,
                    "--billing-start-utc", stamp, "--balance-usd", "30", "--reserve-usd", "1.5",
                    "--hourly-usd", "1.43", "--max-seconds", str(technical_seconds), "--financial-mode", "measured_balance_reserve",
                    "--balance-observation", str(Path(directory) / "observation.json"),
                    "--reserve-stop-receipt", str(receipt), "--result-file", str(result), "--", "fixture"]
            stack.enter_context(patch("sys.argv", argv))
            if technical_seconds:
                stack.enter_context(patch.object(budget.time, "monotonic", side_effect=[0., 10., 10., 10.]))
            code = budget.main()
            return code, json.loads(result.read_text()), receipt.exists()

    def test_a_measured_reserve_stops_even_after_recharge(self):
        code, result, receipt = self.exercise([3., 20., 1.5],
            [subprocess.TimeoutExpired("fixture", 60), subprocess.TimeoutExpired("fixture", 60)])
        self.assertEqual(code, 124)
        self.assertEqual(result["measured_balance_usd"], 1.5)
        self.assertEqual(result["required_action"], "POD_STOP_REQUIRED")
        self.assertTrue(receipt)

    def test_temporary_telemetry_failure_does_not_stop_training(self):
        code, result, receipt = self.exercise([OSError("offline"), 20.], [subprocess.TimeoutExpired("fixture", 60), 0])
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "EXITED")
        self.assertIsNone(result["deadline_utc"])
        self.assertFalse(receipt)

    def test_installation_technical_bound_is_not_a_financial_stop(self):
        code, result, receipt = self.exercise([], [], technical_seconds=1)
        self.assertEqual(code, 124)
        self.assertEqual(result["status"], "TECHNICAL_TIMEOUT")
        self.assertFalse(receipt)

    def test_preparation_uses_the_same_actual_balance_channel(self):
        value = prep_job(financial_mode="measured_balance_reserve", repo_root="/root/MMDL_test")
        value["financial"].update(reserve_usd=1.5, balance_snapshot_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        value["approval"]["approved_sha256"] = "SEAL"
        self.assertEqual(job.validate_job(seal(value))["version"], "pod-prep-v1")
        self.assertEqual(len(job.financial_argv(value)), 20)
        self.assertIn("/workspace/jobs/balance-latest.json", job.financial_argv(value))

    def test_runner_can_finish_unbounded_main_under_external_watchdog(self):
        import sys
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(runner.run_logged([sys.executable, "-c", "pass"], Path(directory) / "run.log", 0), 0)

    def test_only_measured_reserve_job_can_remove_the_main_time_limit(self):
        value = approved_job(None, financial_mode="measured_balance_reserve", repo_root="/root/MMDL_test",
                             stage="v18-main", max_main_seconds=0)
        del value["data"]
        value["financial"]["reserve_usd"] = 1.5
        value["financial"]["balance_snapshot_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        value["approval"]["approved_sha256"] = "SEAL"
        self.assertEqual(job.validate_job(seal(value))["max_main_seconds"], 0)
        flags = job.financial_argv(value)
        self.assertEqual(len(flags), 20)
        self.assertIn("measured_balance_reserve", flags)
        value["financial_mode"] = "balance_reserve_only"
        value["approval"]["approved_sha256"] = "SEAL"
        with self.assertRaisesRegex(job.JobError, "max_main_seconds"):
            job.validate_job(seal(value))
