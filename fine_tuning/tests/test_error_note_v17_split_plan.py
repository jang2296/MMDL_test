"""CPU contracts for the independent v17 training/evaluation chains."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fine_tuning.error_note import plan as planner
from fine_tuning.modules import BASE_REVISION
from fine_tuning.notebook_rl.pod.chain import check_spec


def split_params(root: Path, stage: str) -> dict:
    training = stage == "v17-train"
    value = {"version": planner.PARAMS_VERSION_V17_SPLIT, "commit": "a" * 40,
             "python": "/usr/bin/python3", "repo_root": "/repo-train" if training else "/repo-eval",
             "model_path": f"/cache/{BASE_REVISION}", "out_root": str(root / "output"),
             "run_dir": str(root), "gpu_type": "NVIDIA A100-SXM4-80GB", "hourly_usd": 1.39,
             "funding": {"balance_usd": .1, "balance_observed_at": "2026-11-17T00:00:00Z",
                         "first_bill_at": "2026-11-17T00:00:00Z", "storage_usd_per_hour": .01,
                         "recovery_reserve_usd": 1.5, "recovery_seconds": 600},
             "timeouts": {name: 600 for name in
                          (("cpu", "calibrate", "reload", "export") if training else
                           ("cpu", "calibrate", "validate", "compass"))}}
    paths = planner.V17_TRAIN_PATHS if training else planner.V17_EVAL_PATHS
    hashes = planner.V17_TRAIN_HASHED if training else planner.V17_EVAL_HASHED
    value.update({name: str(root / name) for name in paths})
    value.update({name + "_sha256": "b" * 64 for name in hashes})
    if training:
        value.update(training_estimate_seconds=26208,
                     training_estimate_basis="v16b measured RL 7.28 hours; estimate only")
    else:
        value["timeouts"]["validate"] = 20000
        value.update(long_tail_allowance_seconds=900, eval_engine_mode="cuda_graph",
                     long_tail_basis="CPU fixture long-tail assumption")
    return value


class Parsed(BaseException):
    def __init__(self, value):
        self.value = value


def parse_only(module, argv):
    original = argparse.ArgumentParser.parse_args

    def stop(parser, args=None, namespace=None):
        raise Parsed(original(parser, args, namespace))

    with patch.object(argparse.ArgumentParser, "parse_args", stop):
        try:
            module.main(argv)
        except Parsed as result:
            return result.value
    raise AssertionError("command did not reach its real argument parser")


class V17SplitPlanTests(unittest.TestCase):
    def test_training_is_one_complete_chain_without_evaluation_inputs(self):
        p = split_params(Path("/private/train"), "v17-train")
        spec = check_spec(planner.plan("v17-train", p))
        names = [item["name"] for item in spec["steps"]]
        self.assertEqual(names[-7:], ["bf16-functional-calibration", "campaign-advance", "evaluation-decision",
                                     "require-complete", "reload", "export", "training-artifact-seal"])
        self.assertLess(names.index("bf16-functional-calibration"), names.index("campaign-advance"))
        rendered = json.dumps(spec)
        self.assertNotIn("eval-throughput-calibration", rendered)
        self.assertNotIn("--calibration", rendered)
        self.assertNotIn("eval_engine", rendered)
        self.assertEqual(spec["meta"]["funding"]["initial_snapshot_decision"], "RECHARGE_REQUESTED")
        self.assertFalse(spec["meta"]["funding"]["estimated_cost_stops_execution"])
        self.assertIsNone(spec["meta"]["training_elapsed_cutoff"])
        self.assertEqual(spec["meta"]["training_estimate_seconds"], 26208)
        self.assertNotIn("timeout_seconds", next(i for i in spec["steps"] if i["name"] == "campaign-advance"))

    def test_evaluation_seals_only_after_final_model_validation(self):
        p = split_params(Path("/private/eval"), "v17-eval")
        p["commit"] = "c" * 40
        spec = check_spec(planner.plan("v17-eval", p))
        names = [item["name"] for item in spec["steps"]]
        self.assertLess(names.index("verify-training-artifacts"), names.index("eval-engine-validation"))
        self.assertEqual(names[-1], "finalize-and-run-evaluation")
        self.assertNotIn("evaluate-900", names)
        self.assertIsNone(spec["meta"]["evaluation_watchdog_seconds"])
        self.assertNotIn("timeout_seconds", spec["steps"][-1])
        command = next(i for i in spec["steps"] if i["name"] == "eval-engine-validation")["argv"]
        self.assertEqual(command[command.index("--mode") + 1], "auto")
        self.assertEqual(command[command.index("--baseline-raw") + 1], p["b2_lengths_path"])
        self.assertEqual(command[command.index("--model-ref") + 1], p["export_root"] + "/model_ref.json")

    def test_split_validation_rejects_unbound_estimates_and_wrong_reserve(self):
        for stage, key in (("v17-train", "training_estimate_basis"), ("v17-eval", "long_tail_basis")):
            p = split_params(Path("/private/run"), stage)
            p[key] = ""
            with self.assertRaisesRegex(ValueError, "explicit"):
                planner.plan(stage, p)
            p = split_params(Path("/private/run"), stage)
            p["funding"]["recovery_reserve_usd"] = 0
            with self.assertRaisesRegex(ValueError, "1.5 USD"):
                planner.plan(stage, p)

    def test_every_generated_module_command_uses_real_parser(self):
        from fine_tuning.error_note import v17_eval
        validation = {"evaluation_watchdog_seconds": 12345, "slots": 7,
                      "estimated_evaluation_seconds": 4567, "selected_mode": "cuda_graph"}
        with tempfile.TemporaryDirectory() as directory:
            p = split_params(Path(directory), "v17-eval")
            validation_path = Path(p["out_root"]) / "engine-validation.json"
            validation_path.parent.mkdir(parents=True)
            validation_path.write_text("{}")
            with patch("fine_tuning.error_note.v17_eval_engine.load_engine_validation", return_value=validation):
                continuation = planner._evaluation_continuation(p, validation_path)
            specs = [planner.plan("v17-eval", p), continuation,
                     planner.plan("v17-train", split_params(Path(directory), "v17-train"))]
        covered = set()

        def check(argv):
            if "-m" not in argv:
                return
            index = argv.index("-m")
            name, args = argv[index + 1], argv[index + 2:]
            parsed = parse_only(importlib.import_module(name), args)
            covered.add(name)
            if name == planner.CHAIN and args[0] in ("capture", "with-sha256"):
                check(args[args.index("--") + 1:])
            if name == "fine_tuning.error_note.campaign_v16b":
                self.assertEqual(parsed.execution_phase, "training")
                self.assertIsNone(parsed.calibration)
            if name == "fine_tuning.error_note.v17_eval" and args[0] == "run":
                forwarded = args[args.index("--") + 1:]
                original = argparse.ArgumentParser.parse_args

                def stop(parser, args=None, namespace=None):
                    raise Parsed(original(parser, args, namespace))

                with patch.object(argparse.ArgumentParser, "parse_args", stop), self.assertRaises(Parsed):
                    v17_eval.run_evaluation(forwarded, prereg_path=Path("/unused/prereg"),
                        ledger=Path("/unused/ledger"), calibration_path=Path("/unused/calibration"),
                        preflight=Path("/unused/preflight"))

        for spec in specs:
            for item in spec["steps"]:
                with self.subTest(step=item["name"]):
                    check(item["argv"])
        self.assertIn("fine_tuning.error_note.v17_eval_engine", covered)
        self.assertIn("fine_tuning.error_note.v17_artifacts", covered)
        self.assertIn("fine_tuning.notebook_rl.proof_eval", covered)

    def test_runtime_continuation_uses_measured_values_and_is_write_once(self):
        from fine_tuning.notebook_rl.pod.common import sha256_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = split_params(root, "v17-eval")
            artifact = Path(p["training_artifacts_path"])
            artifact.write_text("{}")
            p["training_artifacts_sha256"] = sha256_file(artifact)
            validation_path = Path(p["out_root"]) / "engine-validation.json"
            validation_path.parent.mkdir(parents=True)
            validation_path.write_text("{}")
            validation = {"evaluation_watchdog_seconds": 12345, "slots": 7,
                          "estimated_evaluation_seconds": 4567, "selected_mode": "cuda_graph"}
            with patch("fine_tuning.error_note.v17_eval_engine.load_engine_validation", return_value=validation), \
                 patch("fine_tuning.error_note.v17_artifacts.verify_training_artifacts"), \
                 patch("fine_tuning.notebook_rl.pod.chain.run", return_value=0) as run:
                result = planner.finalize_evaluation(p, validation_path, Path(p["out_root"]) / "continuation", execute=True)
                spec = json.loads(Path(result["spec"]).read_text())
                sealed = json.loads(Path(result["seal"]).read_text())
                self.assertEqual(sealed["spec_sha256"], sha256_file(result["spec"]))
                evaluate = next(i for i in spec["steps"] if i["name"] == "evaluate-900")
                self.assertEqual(evaluate["timeout_seconds"], 12345)
                self.assertIn("MMDL_EVAL_SLOTS=7", evaluate["argv"])
                self.assertEqual(spec["meta"]["estimated_evaluation_seconds"], 4567)
                self.assertEqual(run.call_args.kwargs["log_name"], "eval_continuation_log.jsonl")
                planner.finalize_evaluation(p, validation_path, Path(p["out_root"]) / "continuation")
                validation["slots"] = 6
                with self.assertRaisesRegex(ValueError, "different sealed"):
                    planner.finalize_evaluation(p, validation_path, Path(p["out_root"]) / "continuation")

    def test_legacy_v17_spec_bytes_unchanged(self):
        from fine_tuning.tests.test_error_note_v17_plan import measured_receipts, params
        expected = {"v17-calibration": "9237a72441c5f3736aa64bb20cebbf88a704264890a41db9f0c15066a748f5b1",
                    "v17": "e1240153e9668fee000990668741b8320a2669f63d066fdd6c084ae959420a93"}
        with tempfile.TemporaryDirectory() as directory:
            measured = params(Path(directory))
            receipt = measured_receipts(measured)
            p = params(Path("/private/v17"))
            specs = {"v17-calibration": planner.plan("v17-calibration", p)}
            for key in ("calibration_sha256", "functional_receipt_sha256"):
                p[key] = measured[key]
            with patch.object(planner, "_v17_calibration", return_value=receipt):
                specs["v17"] = planner.plan("v17", p)
            for stage, spec in specs.items():
                encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"))
                self.assertEqual(hashlib.sha256(encoded.encode()).hexdigest(), expected[stage])

    def test_cli_accepts_both_split_stages_and_runtime_finalizer(self):
        for stage in ("v17-train", "v17-eval"):
            parsed = parse_only(planner, ["plan", "--stage", stage, "--params", "/unused", "--output", "/unused"])
            self.assertEqual(parsed.stage, stage)
        parsed = parse_only(planner, ["finalize-evaluation", "--params-json", "{}", "--engine-validation",
                                     "/unused", "--output-dir", "/unused", "--run"])
        self.assertTrue(parsed.run)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_only(planner, ["finalize-evaluation", "--params-json", "{}"])


if __name__ == "__main__":
    unittest.main()
