"""CPU contracts for split-job clones and the snapshot-based reserve watchdog."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fine_tuning import budget
from fine_tuning.notebook_rl.pod import chain, job, verify


def approved_split(now, **changes):
    value = {
        "version": "pod-job-v1", "commit": "a" * 40,
        "repo_root": "/root/MMDL_test_eval", "financial_mode": "balance_reserve_only",
        "volume_root": "/workspace/vol", "run_dir": "/workspace/vol/runs/eval",
        "stage": "eval-course", "smoke": [sys.executable, "-c", "pass"],
        "main": [sys.executable, "-c", "pass"], "max_main_seconds": 7200,
        "min_free_gib": 0.001,
        "financial": {"owned_pod_id": "owned-pod", "gpu_cutoff_utc": now.isoformat(),
                      "billing_start_utc": (now - dt.timedelta(hours=5)).isoformat(),
                      "balance_snapshot_utc": now.isoformat(), "balance_usd": 2.5,
                      "reserve_usd": 1.5, "hourly_usd": 1.0},
        "approval": {"approved_by_user": True, "approved_utc": now.isoformat(),
                     "approval_note": "fixture approval", "approved_sha256": ""},
    }
    value.update(changes)
    value["approval"]["approved_sha256"] = job.content_sha256(value)
    return value


class SplitPodTests(unittest.TestCase):
    def setUp(self):
        self.now = dt.datetime.now(dt.timezone.utc)

    def test_insufficient_estimated_funding_is_reported_without_shrinking_technical_bound(self):
        value = approved_split(self.now)
        job.validate_job(value, now=self.now)
        fit = job.budget_fit(value, self.now)
        self.assertFalse(fit["fits"])
        self.assertEqual(7200, value["max_main_seconds"])
        self.assertEqual(self.now + dt.timedelta(hours=1), dt.datetime.fromisoformat(fit["deadline_utc"]))
        self.assertEqual(22, len(job.financial_argv(value)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.json"
            path.write_text(json.dumps(value))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(0, job.main([str(path)]))
            self.assertFalse(json.loads(output.getvalue())["budget"]["fits"])

    def test_legacy_jobs_still_refuse_a_bound_that_does_not_fit(self):
        value = approved_split(self.now)
        del value["financial_mode"], value["repo_root"], value["financial"]["balance_snapshot_utc"]
        value["financial"].update(billing_start_utc=self.now.isoformat(), reserve_usd=1.0)
        value["approval"]["approved_sha256"] = job.content_sha256(value)
        self.assertEqual(12, len(job.financial_argv(value)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.json"
            path.write_text(json.dumps(value))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(2, job.main([str(path)]))

    def test_split_mode_requires_measured_snapshot_fixed_reserve_and_eval_clone(self):
        for field, change in (
            ("repo_root", "/root/MMDL_test"), ("financial_mode", "unknown"),
            ("reserve_usd", 1.0), ("balance_snapshot_utc", (self.now + dt.timedelta(hours=1)).isoformat()),
        ):
            value = approved_split(self.now)
            target = value["financial"] if field in value["financial"] else value
            target[field] = change
            value["approval"]["approved_sha256"] = job.content_sha256(value)
            with self.assertRaises(job.JobError, msg=field):
                job.validate_job(value, now=self.now)
        value = approved_split(self.now)
        del value["financial"]["balance_snapshot_utc"]
        with self.assertRaises(job.JobError):
            job.validate_job(value, now=self.now)

    def test_repo_identity_binds_start_and_verify_to_the_executing_clone(self):
        value = approved_split(self.now)
        with patch.object(job.subprocess, "check_output", side_effect=["/root/MMDL_test_eval\n", "a" * 40]):
            job.check_repo_root(value, "/root/MMDL_test_eval")
        with self.assertRaisesRegex(job.JobError, "executing"):
            job.check_repo_root(value, "/root/MMDL_test")
        with patch.object(job.subprocess, "check_output", side_effect=["/root/MMDL_test_eval", "b" * 40]):
            with self.assertRaisesRegex(job.JobError, "commit"):
                job.check_repo_root(value, "/root/MMDL_test_eval")
        with tempfile.TemporaryDirectory() as tmp:
            result = verify.run_checks(repo_root=tmp, commit="a" * 40, volume_root=tmp,
                                       run_dir=tmp, data=None, min_free_gib=0.001,
                                       stage="eval-course", job=value)
        clone = next(item for item in result["checks"] if item["name"] == "clone")
        self.assertFalse(clone["ok"])
        self.assertIn("executing", clone["detail"])

    def test_start_dry_run_refuses_the_training_clone_for_an_eval_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.json"
            path.write_text(json.dumps(approved_split(self.now)))
            script = Path(__file__).resolve().parents[1] / "notebook_rl/pod/start.sh"
            result = subprocess.run(["bash", str(script), "--job", str(path)],
                                    env={**os.environ, "MMDL_POD_PYTHON": sys.executable},
                                    capture_output=True, text=True, timeout=10)
        self.assertEqual(2, result.returncode)
        self.assertIn("executing start.sh/verify clone", result.stderr)

    def test_topup_receipts_are_explicit_linked_and_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = approved_split(self.now - dt.timedelta(minutes=1), run_dir=tmp)
            first = job.extend_financial(value, balance_snapshot_utc=self.now.isoformat(), balance_usd=5,
                                         topup_reference="actual account top-up receipt", authorization="user authorized top-up",
                                         confirmed_by="claude", now=self.now)
            first_bytes = first.read_bytes()
            second = job.extend_financial(value, balance_snapshot_utc=(self.now + dt.timedelta(seconds=1)).isoformat(),
                                          balance_usd=8, topup_reference="second actual top-up",
                                          authorization="explicit extension", confirmed_by="user", now=self.now)
            a, b = json.loads(first_bytes), json.loads(second.read_text())
            self.assertEqual(a["sha256"], b["previous"])
            self.assertEqual(first_bytes, first.read_bytes())
            kwargs = dict(root_sha256=job.content_sha256(value), owned_pod_id="owned-pod",
                          snapshot=budget.utc(value["financial"]["balance_snapshot_utc"]),
                          balance_usd=2.5, hourly_usd=1.0, now=self.now)
            deadline, _, hashes = budget.balance_extensions(first.parent, **kwargs)
            self.assertEqual(b["new_deadline_utc"], deadline.isoformat())
            second.unlink()
            with self.assertRaisesRegex(ValueError, "removed or replaced"):
                budget.balance_extensions(first.parent, seen=hashes, **kwargs)

    def test_topup_rejects_wrong_pod_price_authorization_and_unmeasured_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = approved_split(self.now - dt.timedelta(minutes=1), run_dir=tmp)
            path = job.extend_financial(value, balance_snapshot_utc=self.now.isoformat(), balance_usd=5,
                                        topup_reference="actual receipt", authorization="authorized",
                                        confirmed_by="claude", now=self.now)
            receipt = json.loads(path.read_text())
            old_snapshot = budget.utc(value["financial"]["balance_snapshot_utc"])
            old_deadline = budget.reserve_deadline(old_snapshot, 2.5, 1.5, 1.0, self.now)
            for changes in ({"owned_pod_id": "another-pod"}, {"hourly_usd": 0.5},
                            {"authorization": ""}, {"confirmed_by": "automatic"},
                            {"balance_snapshot_utc": (self.now + dt.timedelta(hours=1)).isoformat()}):
                bad = receipt | changes
                bad["sha256"] = budget.balance_extension_sha256(bad)
                with self.assertRaises(ValueError, msg=str(changes)):
                    budget.check_balance_extension(bad, previous=job.content_sha256(value),
                                                   deadline=old_deadline, snapshot=old_snapshot,
                                                   owned_pod_id="owned-pod", hourly_usd=1.0, now=self.now)
            with self.assertRaisesRegex(job.JobError, "five minutes"):
                job.extend_financial(value, balance_snapshot_utc=(self.now - dt.timedelta(minutes=6)).isoformat(),
                                     balance_usd=8, topup_reference="receipt", authorization="authorized",
                                     confirmed_by="claude", now=self.now)

    def watchdog(self, root, *, balance, command):
        now = dt.datetime.now(dt.timezone.utc)
        argv = [sys.executable, "-m", "fine_tuning.budget", "--owned-pod-id", "owned-pod",
                "--gpu-cutoff-utc", (now - dt.timedelta(hours=1)).isoformat(),
                "--billing-start-utc", (now - dt.timedelta(hours=5)).isoformat(),
                "--balance-usd", str(balance), "--reserve-usd", "1.5", "--hourly-usd", "1",
                "--financial-mode", "balance_reserve_only", "--balance-snapshot-utc", now.isoformat(),
                "--extension-receipts", str(root / "extensions"), "--extension-root-sha256", "a" * 64,
                "--reserve-stop-receipt", str(root / "reserve-stop.json"), "--max-seconds", "0.001",
                "--result-file", str(root / "result.json"), "--", *command]
        return subprocess.run(argv, capture_output=True, text=True, timeout=10,
                              env={**os.environ, "RUNPOD_POD_ID": "owned-pod"})

    def test_reserve_watchdog_ignores_estimated_cost_and_old_gpu_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.watchdog(root, balance=2, command=[sys.executable, "-c", "import time;time.sleep(.05)"])
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertEqual("EXITED", json.loads((root / "result.json").read_text())["status"])


    def test_running_watchdog_adopts_an_explicit_topup_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = approved_split(self.now - dt.timedelta(minutes=1), run_dir=tmp)
            argv = ["budget", *job.financial_argv(value), "--max-seconds", "0.001",
                    "--result-file", str(Path(tmp) / "result.json"), "--", "unused-command"]
            process = Mock(pid=123)

            def wait(*, timeout):
                if not (Path(tmp) / "financial_extensions").exists():
                    job.extend_financial(value, balance_snapshot_utc=self.now.isoformat(), balance_usd=5,
                                         topup_reference="actual top-up", authorization="explicit extension",
                                         confirmed_by="claude", now=self.now)
                    raise subprocess.TimeoutExpired("unused-command", timeout)
                return 0

            process.wait.side_effect = wait
            with patch.object(sys, "argv", argv), patch.dict(os.environ, RUNPOD_POD_ID="owned-pod"), \
                    patch.object(budget.subprocess, "Popen", return_value=process), \
                    patch.object(budget, "terminate_group"), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, budget.main())
            result = json.loads((Path(tmp) / "result.json").read_text())
            receipt = json.loads((Path(tmp) / "financial_extensions/000001.json").read_text())
            self.assertEqual([receipt["sha256"]], result["extension_sha256"])
            self.assertEqual(receipt["new_deadline_utc"], result["deadline_utc"])

    def test_reserve_hit_stops_before_start_and_leaves_owned_pod_action_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.watchdog(root, balance=1.5,
                                   command=[sys.executable, "-c", f"open({str(root / 'unexpected')!r}, 'w').close()"])
            self.assertEqual(124, result.returncode, result.stdout + result.stderr)
            self.assertFalse((root / "unexpected").exists())
            receipt = json.loads((root / "reserve-stop.json").read_text())
            self.assertEqual("POD_STOP_REQUIRED", receipt["required_action"])
            self.assertEqual("RECOVERY_REQUIRED", receipt["recovery_status"])
            self.assertEqual("owned-pod", receipt["owned_pod_id"])
            self.assertFalse(receipt["automatic_pod_stop"])

    def test_measured_continuation_uses_a_separate_log_and_keeps_technical_timeouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {"version": chain.SPEC_VERSION, "steps": [
                {"name": "ok", "argv": [sys.executable, "-c", "pass"], "expect_files": [], "timeout_seconds": 2}],
                "meta": {"run_dir": tmp}}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            digest = chain.sha256_file(path)
            with patch.dict(os.environ, MMDL_RUN_DIR=tmp), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, chain.run(path, digest))
                self.assertEqual(0, chain.run(path, digest, log_name="eval_continuation_log.jsonl"))
                with self.assertRaisesRegex(chain.ChainError, "safe basename"):
                    chain.run(path, digest, log_name="../escape.jsonl")
                with self.assertRaisesRegex(chain.ChainError, "exists"):
                    chain.run(path, digest, log_name="eval_continuation_log.jsonl")
            self.assertTrue((root / chain.LOG_NAME).is_file())
            records = chain.verify_log(root / "eval_continuation_log.jsonl", digest)
            self.assertEqual(2, next(r for r in records if r["event"] == "STEP_START")["timeout_seconds"])


if __name__ == "__main__":
    unittest.main()
