"""Fresh launch guards; CPU fixtures are not GPU or data-approval evidence."""
import json
import datetime as dt
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.runpod_g6 import checkout, expected_shortfall, run


class FreshLaunchTests(unittest.TestCase):
    def test_checkout_and_quarantine_are_fail_closed(self):
        valid = "a" * 40
        responses = [SimpleNamespace(stdout=value) for value in
                     (valid, "", "https://github.com/jang2296/MMDL_test.git")]
        with patch("fine_tuning.runpod_g6.subprocess.run", side_effect=responses):
            checkout(Path("/repo"), valid)
        for outputs in ((valid, " M code.py", "unused"),
                        (valid, "", "https://github.com/jang2296/MMDL.git"),
                        ("b" * 40, "", "unused")):
            with patch("fine_tuning.runpod_g6.subprocess.run", side_effect=[
                    SimpleNamespace(stdout=value) for value in outputs]):
                with self.assertRaises(ValueError):
                    checkout(Path("/repo"), valid)
        with self.assertRaises(ValueError):
            checkout(Path("/repo"), "main")
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            receipt = {"status": "INCOMPLETE_QUOTA", "stage": "pilot",
                       "expected_records": 10000, "quarantined_records": 25}
            path = raw / "preparation_receipt.json"
            path.write_text(json.dumps(receipt))
            self.assertEqual(expected_shortfall(raw), receipt)
            for changed in ({"status": "READY"}, {"expected_records": 1600},
                            {"quarantined_records": 0}, {"quarantined_records": True}):
                path.write_text(json.dumps(receipt | changed))
                with self.assertRaises(ValueError):
                    expected_shortfall(raw)

    def test_fresh_plan_preserves_quota_and_never_reuses_old_weights(self):
        for quarantine in (True, False):
            with self.subTest(quarantine=quarantine), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repo, runtime = root / "repo", root / "runtime"
                (repo / "manifests/models").mkdir(parents=True)
                (runtime / "setup").mkdir(parents=True)
                model = runtime / "model"
                model.mkdir()
                (model / "config.json").write_text("{}")
                from fine_tuning.audit import sha
                manifest = {"files": [{"path": "config.json", "sha256": sha(model / "config.json")}]}
                (repo / "manifests/models/baseline.json").write_text(json.dumps(manifest))
                setup = {"status": "MODEL_HASH_VERIFIED", "code_commit": "a" * 40,
                         "code_repository": "jang2296/MMDL_test", "model_path": str(model), "manifest": manifest}
                (runtime / "setup/model.json").write_text(json.dumps(setup))
                args = SimpleNamespace(runtime_root=runtime, commit="a" * 40, owned_pod_id="test-owned",
                                       gpu_cutoff_utc="2026-10-05T00:00:00Z",
                                       billing_start_utc="2026-10-04T12:00:00Z",
                                       balance_usd=10, reserve_usd=1, hourly_usd=0.54)
                commands = {}
                class CpuController:
                    deadline = dt.datetime(2026, 10, 5, tzinfo=dt.timezone.utc)
                    def stage(self, name, command, **kwargs):
                        commands[name] = command
                        if name == "full10k":
                            self.last_grace = kwargs["termination_grace_seconds"]
                        if name == "approve_pilot" and quarantine:
                            raw = runtime / "training/pilot_raw"
                            raw.mkdir(parents=True)
                            (raw / "preparation_receipt.json").write_text(json.dumps({
                                "status": "INCOMPLETE_QUOTA", "stage": "pilot",
                                "expected_records": 10000, "quarantined_records": 25}))
                            raise RuntimeError("CPU fixture: recorded quarantine shortfall")
                        if name == "full10k":
                            experiment = runtime / "runs/g6_full10k"
                            experiment.mkdir(parents=True)
                            (experiment / "status.json").write_text('{"status":"BUDGET_STOP"}')
                    def event(self, **value):
                        self.last_event = value
                controller = CpuController()
                self.assertEqual(run(args, controller, repo)["status"], "BUDGET_STOP")
                self.assertEqual(controller.last_grace, 45)
                self.assertIn("complete_pilot", commands)
                self.assertIn("disjoint_pilot", commands)
                self.assertEqual(commands["generate_pilot_pool"][-6:],
                                 ["--per-family", "600", "--smoke-roots", "0", "--seed", "3407"])
                self.assertIn(str(repo / "fine_tuning/protocol.json"), commands["full10k"])
                cut_index = commands["full10k"].index("--gpu-cutoff-utc") + 1
                self.assertEqual(commands["full10k"][cut_index], "2026-10-04T23:58:00+00:00")
                self.assertFalse(any("reuse" in value or "pilot-record-count" in value
                                     for value in commands["full10k"]))
                self.assertNotIn("report", commands)


if __name__ == "__main__":
    unittest.main()
