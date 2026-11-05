"""CPU-only single-response stage plans (single-pilot, single-main) and the chain capture receipt helper."""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from fine_tuning.error_note import evaluate, single_campaign
from fine_tuning.error_note import plan as planner
from fine_tuning.notebook_rl.pod import chain, job


PY = "/workspace/volume/venv/bin/python"
OUT = "/workspace/private/single"
CAMPAIGN_DIR = f"{OUT}/campaign"
EXPORT_FINAL = "/workspace/volume/eval-data/proof/export-single-final"
CAMPAIGN_FLAGS = (("--campaign-mode", "train"), ("--precision", "bf16"), ("--output", CAMPAIGN_DIR),
                  ("--max-num-seqs", "16"), ("--max-num-batched-tokens", "393216"), ("--token-budget", "24000"))

BASE = {
    "version": "error-note-plan-params-single-v1", "commit": "a" * 40,
    "python": PY, "repo_root": "/workspace/volume/clone",
    "build_root": "/workspace/private/build", "out_root": OUT,
    "model_path": "/workspace/volume/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "volume_root": "/workspace/volume",
    "order_path": "/workspace/private/order.json", "order_sha256": "b" * 64,
    "frozen_approval_path": "/workspace/private/frozen-approval.json", "frozen_approval_sha256": "c" * 64,
    "source_provenance_path": "/workspace/private/source-provenance.json", "source_provenance_sha256": "d" * 64,
    "training_rules_path": "/workspace/private/training-rules.json", "training_rules_sha256": "e" * 64,
    "pilot_ids_sha256": "f" * 64, "dev_ids_sha256": "1" * 64, "retention_ids_sha256": "2" * 64,
    "max_num_seqs": 16, "max_num_batched_tokens": 393216, "token_budget": 24000,
    "engine": {"name": "vllm", "kv_cache_gib": 16, "gpu_memory_utilization": 0.65, "sleep": True},
}
PILOT_EXTRA = {
    "dev_manifest_path": "/workspace/private/dev/manifest.json", "dev_manifest_sha256": "3" * 64,
    "dev_vault_root": "/workspace/private/dev/vault",
    "dev_rules_path": "/workspace/private/dev/rules.json", "dev_rules_sha256": "4" * 64,
    "timeouts": {"cpu": 11, "advance_first": 22, "advance_pilot": 33, "reload": 44, "export": 55, "evaluate": 66},
}
MAIN_EXTRA = {
    "pilot_gate_path": "/workspace/private/gate.json", "pilot_gate_sha256": "5" * 64,
    "timeouts": {"cpu": 77, "advance_main": 88, "reload": 99, "export": 111},
}
PILOT_NAMES = ["pin-order", "pin-frozen-approval", "pin-source-provenance", "pin-training-rules",
               "pin-dev-manifest", "pin-dev-rules", "advance-1", "reload-1", "advance-pilot", "reload-19",
               "export-19", "dev-prereg", "dev-f0", "dev-f0-grade", "dev-hyb", "dev-hyb-grade", "dev-analyze"]
MAIN_NAMES = ["pin-order", "pin-frozen-approval", "pin-source-provenance", "pin-training-rules", "pin-pilot-gate",
              "advance-main", "reload-219", "export-219"]
# Timeout kind per step; every step not listed here is a cpu step (pins, dev-prereg, grades, dev-analyze).
PILOT_KIND = {"advance-1": "advance_first", "advance-pilot": "advance_pilot", "reload-1": "reload",
              "reload-19": "reload", "export-19": "export", "dev-f0": "evaluate", "dev-hyb": "evaluate"}


def params(stage: str, **changes: object) -> dict:
    value = copy.deepcopy(BASE)
    value.update(copy.deepcopy(PILOT_EXTRA if stage == "single-pilot" else MAIN_EXTRA))
    value.update(changes)
    return value


def value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class SingleStagePlanTests(unittest.TestCase):
    def assert_pins(self, steps: dict, p: dict, keys: tuple[str, ...]) -> None:
        for key in keys:
            name = f"pin-{key.replace('_', '-')}"
            self.assertEqual(steps[name]["argv"][3:],
                             ["pin", "--file", p[f"{key}_path"], "--sha256", p[f"{key}_sha256"]], name)

    def test_pilot_steps(self) -> None:
        spec = planner.plan("single-pilot", params("single-pilot"))
        self.assertIs(chain.check_spec(spec), spec)
        self.assertEqual(spec["meta"]["job_stage"], "single-pilot")
        self.assertEqual([step["name"] for step in spec["steps"]], PILOT_NAMES)
        steps = {step["name"]: step for step in spec["steps"]}
        self.assert_pins(steps, params("single-pilot"),
                         ("order", "frozen_approval", "source_provenance", "training_rules", "dev_manifest", "dev_rules"))
        for name, blocks in (("advance-1", "1"), ("advance-pilot", "18")):
            argv = steps[name]["argv"]
            self.assertEqual(argv[2], planner.CAMPAIGN, name)
            for flag in ("--allow-update", "--allow-save", "--allow-model-run"):
                self.assertIn(flag, argv, name)
            self.assertEqual(value_after(argv, "--max-blocks"), blocks, name)
            for flag, value in CAMPAIGN_FLAGS:
                self.assertEqual(value_after(argv, flag), value, name)
        self.assertEqual(steps["advance-pilot"]["expect_files"], [f"{CAMPAIGN_DIR}/checkpoint-19/metadata.json"])
        export = steps["export-19"]["argv"]
        self.assertEqual(value_after(export, "--checkpoint"), f"{CAMPAIGN_DIR}/checkpoint-19")
        self.assertEqual(value_after(export, "--binding"), f"{CAMPAIGN_DIR}/campaign-plan.json")
        timeouts = PILOT_EXTRA["timeouts"]
        for step in spec["steps"]:
            self.assertEqual(step["timeout_seconds"], timeouts[PILOT_KIND.get(step["name"], "cpu")], step["name"])

    def test_main_steps(self) -> None:
        spec = planner.plan("single-main", params("single-main"))
        self.assertIs(chain.check_spec(spec), spec)
        self.assertEqual(spec["meta"]["job_stage"], "single-main")
        self.assertEqual([step["name"] for step in spec["steps"]], MAIN_NAMES)
        steps = {step["name"]: step for step in spec["steps"]}
        p = params("single-main")
        self.assert_pins(steps, p, ("order", "frozen_approval", "source_provenance", "training_rules", "pilot_gate"))
        advance = steps["advance-main"]["argv"]
        self.assertIn("--continue-main", advance)
        self.assertEqual(value_after(advance, "--pilot-gate"), p["pilot_gate_path"])
        self.assertEqual(value_after(advance, "--pilot-gate-sha256"), p["pilot_gate_sha256"])
        self.assertEqual(value_after(advance, "--max-blocks"), "200")
        for flag, value in CAMPAIGN_FLAGS:
            self.assertEqual(value_after(advance, flag), value)
        export = steps["export-219"]["argv"]
        self.assertEqual(value_after(export, "--output"), EXPORT_FINAL)
        self.assertEqual(value_after(export, "--checkpoint"), f"{CAMPAIGN_DIR}/checkpoint-219")
        for step in spec["steps"]:
            self.assertFalse(any("error_note.evaluate" in arg or "run-eval" in arg for arg in step["argv"]), step["name"])
        kinds = {"advance-main": "advance_main", "reload-219": "reload", "export-219": "export"}
        for step in spec["steps"]:
            self.assertEqual(step["timeout_seconds"], p["timeouts"][kinds.get(step["name"], "cpu")], step["name"])

    def test_reload_steps_capture_receipt(self) -> None:
        pilot = {step["name"]: step for step in planner.plan("single-pilot", params("single-pilot"))["steps"]}
        main = {step["name"]: step for step in planner.plan("single-main", params("single-main"))["steps"]}
        for steps, slot in ((pilot, 1), (pilot, 19), (main, 219)):
            step = steps[f"reload-{slot}"]
            receipt = f"{OUT}/reload-{slot}.json"
            argv = step["argv"]
            self.assertEqual(argv[3:6], ["capture", "--output", receipt], step["name"])
            self.assertEqual(argv[6], "--", step["name"])
            self.assertEqual(step["expect_files"], [receipt], step["name"])
            # the wrapped command is `python -m single_campaign reload ...`
            self.assertEqual(argv[7:9], [PY, "-m"], step["name"])
            self.assertEqual(argv[9], planner.CAMPAIGN, step["name"])
            self.assertEqual(argv[10], "reload", step["name"])

    def test_generated_argv_parse_with_real_clis(self) -> None:
        counts = {"campaign": 0, "reload": 0, "evaluate": 0}
        for stage in planner.SINGLE_STAGES:
            for step in planner.plan(stage, params(stage))["steps"]:
                argv = step["argv"]
                with self.subTest(stage=stage, step=step["name"]):
                    if argv[2] == planner.CAMPAIGN:
                        single_campaign.build_parser().parse_args(argv[3:])
                        counts["campaign"] += 1
                    elif argv[2] == planner.EVALUATE:
                        evaluate.build_parser().parse_args(argv[3:])
                        counts["evaluate"] += 1
                    elif argv[2] == planner.CHAIN and argv[3] == "capture":
                        wrapped = argv[7:]
                        self.assertEqual(wrapped[2], planner.CAMPAIGN)
                        single_campaign.build_parser().parse_args(wrapped[3:])
                        counts["reload"] += 1
        self.assertEqual(counts, {"campaign": 3, "reload": 3, "evaluate": 6})

    def test_refusals(self) -> None:
        for stage in planner.SINGLE_STAGES:
            base = params(stage)
            for key in base:
                with self.subTest(stage=stage, missing=key), self.assertRaises(ValueError):
                    planner.plan(stage, {name: value for name, value in base.items() if name != key})
            for key in [name for name in base if name.endswith("_sha256")]:
                with self.subTest(stage=stage, malformed=key), self.assertRaises(ValueError):
                    planner.plan(stage, params(stage, **{key: "bad"}))
            private = [name for name in base
                       if name in ("build_root", "out_root", "dev_vault_root")
                       or (name.endswith("_path") and name != "model_path")]
            for key in private:
                with self.subTest(stage=stage, public=key), self.assertRaisesRegex(ValueError, "outside repo_root"):
                    planner.plan(stage, params(stage, **{key: f"{base['repo_root']}/private.json"}))
            with self.subTest(stage=stage, case="max_num_seqs"), self.assertRaises(ValueError):
                planner.plan(stage, params(stage, max_num_seqs=17))
            timeouts = base["timeouts"]
            for changed in ({**timeouts, "collect": 1},
                            {name: value for name, value in timeouts.items() if name != "cpu"}):
                with self.subTest(stage=stage, timeouts=sorted(changed)), self.assertRaises(ValueError):
                    planner.plan(stage, params(stage, timeouts=changed))
            with self.subTest(stage=stage, case="model_path"), self.assertRaises(ValueError):
                planner.plan(stage, params(stage, model_path="/workspace/volume/snapshots/main"))
        with self.assertRaises(ValueError):  # the pilot-only dev inputs are unknown keys for single-main
            planner.plan("single-main", params("single-main", dev_manifest_path="/workspace/private/dev/manifest.json"))

    def test_job_stages_require_training_data(self) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        for stage in planner.SINGLE_STAGES:
            value = {"version": job.VERSION, "commit": "a" * 40, "volume_root": "/workspace/volume",
                     "run_dir": "/workspace/run", "stage": stage, "smoke": ["python", "smoke"], "main": ["python", "main"],
                     "max_main_seconds": 600, "min_free_gib": 1,
                     "financial": {"owned_pod_id": "fixture", "gpu_cutoff_utc": (now + dt.timedelta(hours=1)).isoformat(),
                                   "billing_start_utc": now.isoformat(), "balance_usd": 10, "reserve_usd": 1, "hourly_usd": 1},
                     "data": {"bundle_manifest": "/workspace/bundle.json", "bundle_manifest_sha256": "b" * 64,
                              "dataset_manifest": "/private/manifest.json", "dataset_manifest_sha256": "c" * 64,
                              "vault_root": "/private/vault"},
                     "approval": {"approved_by_user": True, "approved_utc": now.isoformat(), "approval_note": "fixture",
                                  "approved_sha256": ""}}
            value["approval"]["approved_sha256"] = job.content_sha256(value)
            self.assertIs(job.validate_job(value, now=now), value, stage)
            del value["data"]
            with self.assertRaises(job.JobError, msg=stage):
                job.validate_job(value, now=now)

    def test_deterministic_cli_plan_preserves_existing_output(self) -> None:
        stage = "single-pilot"
        first = planner.plan(stage, params(stage))
        changed = planner.plan(stage, params(stage, dev_rules_sha256="6" * 64))
        self.assertNotEqual(first["meta"]["params_sha256"], changed["meta"]["params_sha256"])
        self.assertEqual(first, planner.plan(stage, params(stage)))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            root = Path(tmp)
            source, output = root / "params.json", root / "steps.json"
            source.write_text(json.dumps(params(stage)), encoding="utf-8")
            argv = ["plan", "--stage", stage, "--params", str(source), "--output", str(output)]
            self.assertEqual(planner.main(argv), 0)
            original = output.read_bytes()
            self.assertEqual(planner.main(argv), 2)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(json.loads(original), first)


class ChainCaptureTests(unittest.TestCase):
    def test_capture_writes_once_and_propagates_exit_code(self) -> None:
        printed = "print('{\"status\": \"RELOADED\", \"slot\": 1}')"
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            receipt = Path(tmp) / "reload-1.json"
            argv = ["capture", "--output", str(receipt), "--", sys.executable, "-c", printed]
            self.assertEqual(chain.main(argv), 0)
            self.assertEqual(receipt.read_text(encoding="utf-8"), '{"status": "RELOADED", "slot": 1}\n')
            before = receipt.read_bytes()
            self.assertEqual(chain.main(argv), 2)  # write-once: an existing receipt is refused, not replaced
            self.assertEqual(receipt.read_bytes(), before)
            failed = Path(tmp) / "exit-3.json"
            self.assertEqual(chain.main(["capture", "--output", str(failed), "--", sys.executable, "-c",
                                         "import sys; sys.exit(3)"]), 3)
            self.assertEqual(failed.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
