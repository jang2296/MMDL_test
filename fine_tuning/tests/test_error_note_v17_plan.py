"""The main plan is sealed only after a separate measured calibration stage."""
import copy
from contextlib import redirect_stderr
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning.error_note import plan as planner
from fine_tuning.error_note.v17_eval import calibration_receipt, calibration_timeout, seal
from fine_tuning.modules import BASE_REVISION
from fine_tuning.notebook_rl.pod.chain import check_spec
from fine_tuning.tests.test_error_note_v16b_plan import params as legacy_params


def params(root: Path):
    value = {"version": planner.PARAMS_VERSION_V17, "commit": "a" * 40,
        "gpu_type": "NVIDIA A100-SXM4-80GB", "hourly_usd": 1.39,
        "timeouts": {"cpu": 600, "reload": 1800, "export": 1800, "compass": 1800, "calibrate": 1800},
        "long_tail_allowance_seconds": 900, "long_tail_basis": "measured single tail window",
        "funding": {"balance_usd": 50, "balance_observed_at": "2026-11-17T00:00:00Z",
            "first_bill_at": "2026-11-17T00:00:00Z", "storage_usd_per_hour": .01,
            "recovery_reserve_usd": 1.5, "recovery_seconds": 600,
            "other_post_train_seconds": 1800, "next_train_seconds": 3600,
            "funding_deadline_utc": None}}
    value.update({name: str(root / name) for name in planner.V17_PATHS})
    value.update({name + "_sha256": "b" * 64 for name in planner.V17_HASHED})
    value.update(repo_root="/repo", model_path=f"/cache/{BASE_REVISION}",
                 run_dir=str(root), out_root=str(root / "output"))
    return value


def measured_receipts(value):
    receipt = calibration_receipt(seal({"version": "v17-eval-probe-v1", "slots": 8, "items": 16,
        "wall_seconds": 300, "generated_tokens": 120000, "sixty_second_rates": [390, 410, 400, 400, 400],
        "dtype": "bfloat16", "backend": "vllm-0.11.0", "config_sha256": "c" * 64,
        "source_sha256": "d" * 64, "code_commit": value["commit"], "environment_sha256": "e" * 64,
        "requests_sha256": value["performance_requests_sha256"],
        "max_observed_context_tokens": 32768, "cumulative_preemptions": 0,
        "disjoint_receipt_sha256": "f" * 64, "load_finalize_allowance_seconds": 60,
        "long_tail_allowance_seconds": 900, "long_tail_basis": "measured single tail window"}))
    for name, body in (("calibration", receipt), ("functional_receipt", {
            "version": "v17-bf16-preflight-v1", "status": "PASS", "code_commit": value["commit"]})):
        path = Path(value[name + "_path"])
        path.write_text(json.dumps(body), encoding="utf-8")
        value[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return receipt


class V17PlanTests(unittest.TestCase):
    def test_calibration_requires_clean_source_and_doctor_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            value = params(Path(directory))
            spec = planner.plan("v17-calibration", value)
        check_spec(spec)
        steps = {item["name"]: item for item in spec["steps"]}
        names = list(steps)
        self.assertLess(names.index("verify-published-source"), names.index("doctor"))
        self.assertLess(names.index("doctor-content-gate"), names.index("prep-generate"))
        self.assertNotIn("stage-data", steps)
        self.assertLess(names.index("pin-v17-bundle-manifest"), names.index("verify-data"))
        self.assertLess(names.index("verify-data"), names.index("pin-v17-performance-requests"))
        self.assertLess(names.index("pin-v17-performance-requests"), names.index("prep-generate"))
        self.assertIn("--output", steps["doctor"]["argv"])
        self.assertIn("--required-gib", steps["doctor"]["argv"])
        self.assertIn("--version", steps["prep-generate"]["argv"])
        self.assertIn("error-note-v17", steps["prep-generate"]["argv"])
        self.assertNotIn("campaign-advance", steps)
        self.assertNotIn("evaluate-900", steps)
        self.assertIn("eval-throughput-calibration", steps)
        self.assertIsNone(spec["meta"]["evaluation_watchdog_seconds"])

    def test_generated_commands_match_live_parsers_and_required_runtime_flags(self):
        from fine_tuning.error_note import campaign_v16b, prep_v16b, v17_eval
        from fine_tuning.notebook_rl import proof_eval

        with tempfile.TemporaryDirectory() as directory:
            value = params(Path(directory))
            value["funding"]["funding_deadline_utc"] = "2026-11-18T00:00:00Z"
            calibration = planner.plan("v17-calibration", value)
            measured_receipts(value)
            main = planner.plan("v17", value)
        for spec in (calibration, main):
            for step in spec["steps"]:
                argv = step["argv"]
                if "fine_tuning.error_note.prep_v16b" in argv:
                    parsed = prep_v16b.build_parser().parse_args(argv[argv.index("fine_tuning.error_note.prep_v16b") + 1:])
                    self.assertEqual(parsed.version, "error-note-v17")
                if "fine_tuning.error_note.campaign_v16b" in argv:
                    parsed = campaign_v16b.build_parser().parse_args(
                        argv[argv.index("fine_tuning.error_note.campaign_v16b") + 1:])
                    self.assertEqual(parsed.version, "error-note-v17")
                    if parsed.command != "prepare":
                        self.assertEqual(parsed.code_commit, value["commit"])
                    if spec is main:
                        self.assertEqual(str(parsed.calibration), value["calibration_path"])
                        self.assertEqual(parsed.calibration_sha256, value["calibration_sha256"])
                        self.assertEqual(str(parsed.functional_receipt), value["functional_receipt_path"])
                        self.assertEqual(parsed.functional_receipt_sha256, value["functional_receipt_sha256"])
                    if parsed.command == "preflight":
                        self.assertTrue(parsed.allow_model_run)
                        self.assertIsNotNone(parsed.eval_protocol)
                        self.assertEqual(argv[:3], ["/usr/bin/env", "MMDL_EVAL_SLOTS=8",
                                                  "VLLM_ENABLE_V1_MULTIPROCESSING=0"])
        probe = next(step["argv"] for step in calibration["steps"] if step["name"] == "eval-throughput-calibration")
        self.assertEqual(probe[:3], ["/usr/bin/env", "MMDL_EVAL_SLOTS=8",
                                    "VLLM_ENABLE_V1_MULTIPROCESSING=0"])
        with patch.object(v17_eval, "probe", side_effect=RuntimeError("probe parser accepted")):
            with self.assertRaisesRegex(RuntimeError, "probe parser accepted"):
                v17_eval.main(probe[probe.index("fine_tuning.error_note.v17_eval") + 1:])
        evaluate = next(step["argv"] for step in main["steps"] if step["name"] == "evaluate-900")
        forwarded = evaluate[evaluate.index("--") + 1:]
        outer = evaluate[evaluate.index("fine_tuning.error_note.v17_eval") + 1:]
        self.assertEqual(outer[outer.index("--recovery-reserve-seconds") + 1], "600")
        self.assertIn("--funding-deadline-epoch", outer)
        with patch.object(v17_eval, "run_evaluation", side_effect=RuntimeError("wrapper parser accepted")):
            with self.assertRaisesRegex(RuntimeError, "wrapper parser accepted"):
                v17_eval.main(outer)
        self.assertEqual(forwarded[forwarded.index("--base-path") + 1], value["model_path"])
        run_args = dict(prereg_path=Path("/unused/prereg"), ledger=Path("/unused/ledger"),
                        calibration_path=Path("/unused/calibration"), preflight=Path("/unused/preflight"))
        with patch.object(proof_eval, "load_preregistration", side_effect=RuntimeError("eval parser accepted")):
            with self.assertRaisesRegex(RuntimeError, "eval parser accepted"):
                v17_eval.run_evaluation(forwarded, **run_args)
            missing = list(forwarded)
            index = missing.index("--base-path")
            del missing[index:index + 2]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                v17_eval.run_evaluation(missing, **run_args)

    def test_main_requires_pinned_prior_receipts_and_shares_measured_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            value = params(Path(directory))
            calibration = measured_receipts(value)
            spec = planner.plan("v17", value)
        check_spec(spec)
        steps = {item["name"]: item for item in spec["steps"]}
        watchdog = calibration_timeout(calibration)
        self.assertEqual(steps["evaluate-900"]["timeout_seconds"], watchdog)
        self.assertEqual(spec["meta"]["evaluation_watchdog_seconds"], watchdog)
        self.assertEqual(spec["meta"]["funding"]["evaluation_watchdog_seconds"], watchdog)
        self.assertEqual(spec["meta"]["funding"]["initial_snapshot_decision"]["evaluation_seconds"], watchdog)
        self.assertEqual(spec["meta"]["funding"]["initial_snapshot_decision"]["status"], "CONTINUE")
        self.assertTrue(spec["meta"]["funding"]["snapshot_only"])
        self.assertAlmostEqual(spec["meta"]["funding"]["post_train_hours"], (watchdog + 2400) / 3600)
        self.assertEqual(steps["evaluate-900"]["argv"].count("--sample-id"), 900)
        self.assertIn("MMDL_EVAL_SLOTS=8", steps["evaluate-900"]["argv"])
        self.assertIn("--preflight", steps["evaluate-900"]["argv"])
        self.assertIn("v12", steps["preregister-v17"]["argv"])
        self.assertIn("--calibration", steps["preregister-v17"]["argv"])
        self.assertNotIn("timeout_seconds", steps["campaign-advance"])
        self.assertNotIn("prep-generate", steps)
        self.assertIsNone(spec["meta"]["total_timeout_seconds"])
        self.assertIsNone(spec["meta"]["training_elapsed_cutoff"])
        self.assertEqual(spec["meta"]["training_counts"], {"main": 3000, "prep_candidates": 512, "development": 200})
        self.assertEqual(spec["meta"]["protocol_contract"]["eval_adapter_serving"], "unmerged-lora")

    def test_main_records_standing_decision_before_completion_and_passes_it_on(self):
        from fine_tuning.error_note import campaign_v16b, v17_eval
        from fine_tuning.error_note.prep_v16b import seal

        with tempfile.TemporaryDirectory() as directory:
            value = params(Path(directory))
            measured_receipts(value)
            spec = planner.plan("v17", value)
            names = [item["name"] for item in spec["steps"]]
            self.assertEqual(names.index("evaluation-decision") + 1, names.index("require-complete"))
            steps = {item["name"]: item for item in spec["steps"]}
            decision = steps["evaluation-decision"]["expect_files"][0]
            for name in ("require-complete", "export", "final-adapter-preflight"):
                argv = steps[name]["argv"]
                parsed = campaign_v16b.build_parser().parse_args(
                    argv[argv.index("fine_tuning.error_note.campaign_v16b") + 1:])
                self.assertEqual(str(parsed.evaluation_decision), decision)
            decide = steps["evaluation-decision"]["argv"]
            self.assertIn("DEGRADED is never pre-authorized", decide[decide.index("--authorization") + 1])
            out = Path(directory) / "campaign"
            out.mkdir()
            plan_body = seal(out / "campaign-plan.json", {"version": "x"})
            overall = {"net_change": -2, "p_value": .5}
            admission = seal(out / "evaluation-admission.json", {"plan_sha256": plan_body["sha256"],
                "decision": {"status": "EVAL_REVIEW_REQUIRED", "development": {"overall": overall},
                             "effect": {"eligible": False}}})
            receipt = Path(directory) / "decision.json"
            self.assertEqual(v17_eval.main(["decide", "--campaign-output", str(out), "--authorization", "user",
                                            "--output", str(receipt)]), 0)
            body = json.loads(receipt.read_text())
            self.assertEqual((body["decision"], body["authorized_by"], body["admission_sha256"]),
                             ("PROCEED", "user", admission["sha256"]))
            self.assertEqual(v17_eval.main(["decide", "--campaign-output", str(out), "--authorization", "user",
                                            "--output", str(receipt)]), 0)  # resumed chain: same receipt
            degraded = Path(directory) / "degraded"
            degraded.mkdir()
            seal(degraded / "campaign-plan.json", {"version": "x"})
            seal(degraded / "evaluation-admission.json", {"plan_sha256": plan_body["sha256"],
                "decision": {"status": "DEGRADED", "development": {"overall": overall}}})
            refused = Path(directory) / "refused.json"
            with redirect_stderr(io.StringIO()):
                self.assertEqual(v17_eval.main(["decide", "--campaign-output", str(degraded),
                                                "--authorization", "user", "--output", str(refused)]), 3)
            self.assertFalse(refused.exists())

    def test_missing_or_replaced_receipts_and_fixed_eval_timeout_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            value = params(Path(directory))
            measured_receipts(value)
            for extra in ({"calibration_sha256": "0" * 64}, {"functional_receipt_sha256": "0" * 64},
                          {"commit": "f" * 40}, {"model_path": "/model/latest"}):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    planner.plan("v17", value | extra)
            altered = copy.deepcopy(value)
            altered["timeouts"]["evaluate"] = 23400
            with self.assertRaises(ValueError):
                planner.plan("v17", altered)
            Path(value["calibration_path"]).unlink()
            with self.assertRaises(OSError):
                planner.plan("v17", value)

    def test_main_refuses_balance_snapshot_below_calibrated_post_training_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            value = params(Path(directory))
            measured_receipts(value)
            value["funding"]["balance_usd"] = 1
            with self.assertRaisesRegex(ValueError, "funding snapshot"):
                planner.plan("v17", value)

    def test_source_gate_checks_publication_and_never_contacts_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            output = Path(directory) / "receipts/source.json"
            responses = [str(root), "a" * 40, "", "https://github.com/jang2296/MMDL_test.git",
                         "refs/remotes/origin/feature"]
            with patch.object(planner.subprocess, "run") as run:
                run.side_effect = [type("Result", (), {"stdout": text})() for text in responses]
                receipt = planner.verify_source_v17(root, "a" * 40, output)
                commands = [call.args[0] for call in run.call_args_list]
            self.assertFalse(receipt["network_checked"])
            self.assertTrue(output.is_file())
            self.assertFalse(any("fetch" in command or "push" in command for command in commands))
            with self.assertRaises(ValueError):
                planner.verify_source_v17(root, "a" * 40, root / "bad.json")

    def test_legacy_plan_output_bytes_are_unchanged(self):
        value = planner.plan("v16b", legacy_params())
        actual = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(actual, "315df37a2dde26465d2de247a606b142efbb87a4ded374fd2c2301fb1fdb3e04")
