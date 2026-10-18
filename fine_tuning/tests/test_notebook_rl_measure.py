"""Candidate launcher boundaries; mocked GPU setup is not actual GPU evidence."""
import copy
import datetime as dt
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fine_tuning.notebook_rl import launch, measure
from fine_tuning.notebook_rl.dataset import _bytes, _sha
from fine_tuning.notebook_rl.schema import load_protocol


class MeasurementLaunchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = load_protocol(launch.HERE/"protocol_v1.yaml")
        self.job = {"format": measure.FORMAT, "build_root": str(self.root/"data"),
            "model_path": str(self.root/self.cfg["initial_checkpoint"]["revision"]),
            "output": str(self.root/"run"), "seed": 3407, "commit": "a"*40,
            "deadline_utc": "2030-01-01T00:00:00+00:00", "forbidden_roots": [str(self.root/"benchmark")],
            "expected_measurement_scope": {}, "runtime_scope": {
                "purpose": "FROZEN_WEIGHT_MEASUREMENT_ONLY", "runtime_id": "owned-pod",
                "environment_lock_sha256": "a"*64, "device": dict(type="cuda", index=0,
                    uuid="GPU-example", name="fixture GPU", driver_version="fixture", total_memory_bytes=48*1024**3)}}
        for key in measure.DESCRIPTORS:
            self.job[key] = {"path": str(self.root/(key+".json")), "file_sha256": "b"*64}
        self.checked = {"status": "VALIDATED_FOR_MEASUREMENT", "measurement_allowed": True,
            "training_allowed": False, "blockers": [], "input_binding": {}, "code_sha256": {"fixture.py": "a"*64}}
        self.engine = SimpleNamespace(policy_version=0,
            llm=SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=Mock()))))

    def test_job_shape_does_not_grant_measurement_or_allocate(self):
        measure.validate_job(self.job, self.cfg)
        checked = measure.preflight(self.job, self.cfg)
        self.assertEqual(checked["status"], "BLOCKED_NOT_CHECKED")
        self.assertFalse(checked["training_allowed"])
        self.assertFalse(Path(self.job["output"]).exists())

    def test_no_subset_caps_training_inline_private_or_output_alias(self):
        for changes in ({"subset": 1}, {"max_new_tokens": 100}, {"arm": "MOD-RV"},
                        {"seed": True}, {"evidence": {"secret": "payload"}},
                        {"output": str(launch.ROOT/"MMDL"/"never-write")},
                        {"output": str(self.root)}, {"output": str(self.root/"data"/"output")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                measure.validate_job(self.job | changes, self.cfg)
        wrong = copy.deepcopy(self.job)
        wrong["runtime_scope"]["purpose"] = "TRAINING"
        with self.assertRaises(ValueError):
            measure.validate_job(wrong, self.cfg)

    def test_check_and_blocked_execute_do_not_reach_gpu(self):
        path = self.root/"job.json"
        path.write_bytes(_bytes(self.job))
        for mode in ("--check", "--execute"):
            with patch.object(measure, "execute") as execute, redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(measure.main(["--job", str(path), mode]), 2)
                self.assertFalse(json.loads(stdout.getvalue())["gpu_runtime_checked"])
                execute.assert_not_called()

    def test_unapproved_or_changed_data_cannot_reach_runtime(self):
        with patch.object(launch, "runtime_check") as runtime:
            for checked in ({}, self.checked | {"training_allowed": True}, self.checked | {"measurement_allowed": False}):
                with self.assertRaisesRegex(ValueError, "eligibility"):
                    measure.execute(self.job, self.cfg, checked)
            with patch.object(measure, "preflight", return_value={}), self.assertRaisesRegex(ValueError, "changed"):
                measure.execute(self.job, self.cfg, self.checked)
            runtime.assert_not_called()

    def configured(self, callback):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(measure, "preflight", return_value=self.checked))
        runtime = stack.enter_context(patch.object(launch, "runtime_check", return_value={"deadline_timestamp": 9999999999.}))
        stack.enter_context(patch("fine_tuning.notebook_rl.guards.install_guards"))
        stack.enter_context(patch("fine_tuning.notebook_rl.guards.prepare_vllm_schema_dependency", return_value={"fixture": True}))
        factory = stack.enter_context(patch("fine_tuning.notebook_rl.engines.VLLMRolloutEngine", return_value=self.engine))
        core = stack.enter_context(patch("fine_tuning.notebook_rl.phase_a.measure_candidates", side_effect=callback))
        return runtime, factory, core

    @staticmethod
    def closed_core(root, output, **kwargs):
        output.mkdir()
        value = {"status": "MEASUREMENTS_COMPLETED_NOT_PHASE_A_APPROVAL"}
        (output/"summary.json").write_bytes(_bytes(value))
        return value

    def test_only_fixed_original_engine_no_training_and_owned_shutdown(self):
        runtime, factory, core = self.configured(self.closed_core)
        result = measure.execute(self.job, self.cfg, self.checked)
        factory.assert_called_once_with(Path(self.job["model_path"]), self.cfg["image"],
            self.cfg["engine"]["vllm"]["kv_cache_memory_bytes"], 16, 16384)
        self.assertEqual(runtime.call_args.args[0]["expected_gpu_scope"], self.job["runtime_scope"])
        self.assertEqual(core.call_args.kwargs["seed"], 3407)
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["phase_a_completed"])
        self.assertFalse(result["gpu_gates_certified"])
        self.assertFalse(result["pod_billing_stopped"])
        self.engine.llm.llm_engine.engine_core.shutdown.assert_called_once()
        root = Path(self.job["output"])
        self.assertEqual(result["candidate_summary_sha256"], _sha((root/"candidates"/"summary.json").read_bytes()))
        self.assertEqual(result, launch.read_json(root/"result.json"))

    def test_failure_and_interrupt_preserve_receipt_and_close_engine(self):
        for n, exception in enumerate((ValueError("fixture"), KeyboardInterrupt())):
            with self.subTest(exception=type(exception).__name__):
                self.job["output"] = str(self.root/f"failure-{n}")
                self.engine.llm.llm_engine.engine_core.shutdown.reset_mock()
                self.configured(Mock(side_effect=exception))
                with self.assertRaises(type(exception)):
                    measure.execute(self.job, self.cfg, self.checked)
                result = launch.read_json(Path(self.job["output"])/"result.json")
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertEqual(result["error_type"], type(exception).__name__)
                self.assertEqual(result["engine_shutdown"], "COMPLETED")
                self.engine.llm.llm_engine.engine_core.shutdown.assert_called_once()

    def test_missing_closed_ledger_and_shutdown_failure_cannot_pass(self):
        self.configured(lambda *args, **kwargs: {"status": "MEASUREMENTS_COMPLETED_NOT_PHASE_A_APPROVAL"})
        self.engine.llm.llm_engine.engine_core.shutdown.side_effect = RuntimeError("fixture shutdown")
        with self.assertRaises(RuntimeError):
            measure.execute(self.job, self.cfg, self.checked)
        result = launch.read_json(Path(self.job["output"])/"result.json")
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["engine_shutdown"], "FAILED")

    def test_changed_policy_cannot_be_measured_as_f0(self):
        _, _, core = self.configured(self.closed_core)
        self.engine.policy_version = 1
        with self.assertRaisesRegex(ValueError, "original F0"):
            measure.execute(self.job, self.cfg, self.checked)
        core.assert_not_called()
        self.engine.llm.llm_engine.engine_core.shutdown.assert_called_once()

    def test_runtime_memory_minimum_keeps_training_two_copies(self):
        import torch

        model = Path(self.job["model_path"])
        model.mkdir()
        weight = model/"model.safetensors"
        weight.write_bytes(b"fixture")
        lock = self.root/self.cfg["environment"]["lock"]
        lock.parent.mkdir(parents=True)
        lock.write_text("# no installed packages in this authored runtime fixture\n")
        manifest = self.root/self.cfg["initial_checkpoint"]["manifest"]
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(_bytes({"files": [{"path": weight.name, "bytes": weight.stat().st_size,
                                              "sha256": _sha(weight.read_bytes())}]}))
        scope = copy.deepcopy(self.job["runtime_scope"])
        scope["environment_lock_sha256"] = _sha(lock.read_bytes())
        now = time.time()
        def iso(seconds):
            return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).isoformat()
        command = ["python", "-m", "fine_tuning.budget", "--owned-pod-id", "owned-pod",
            "--gpu-cutoff-utc", iso(now+7200), "--billing-start-utc", iso(now-10),
            "--balance-usd", "2.3", "--reserve-usd", "1", "--hourly-usd", ".5", "--max-seconds", "7200"]
        watcher = SimpleNamespace(pid=77, cmdline=lambda: command, create_time=lambda: now-10)
        psutil = SimpleNamespace(Process=lambda: SimpleNamespace(parents=lambda: [watcher]))
        with ExitStack() as stack:
            stack.enter_context(patch.object(launch, "ROOT", self.root))
            stack.enter_context(patch.object(launch, "import_module", return_value=psutil))
            stack.enter_context(patch.object(launch.subprocess, "run"))
            output = stack.enter_context(patch.object(launch.subprocess, "check_output"))
            stack.enter_context(patch.dict(os.environ, {"RUNPOD_POD_ID": "owned-pod", "MMDL_BUDGET_WATCHDOG_PID": "77"}))
            for name, value in (("is_available", True), ("is_bf16_supported", True), ("set_device", None),
                                ("get_device_properties", SimpleNamespace(total_memory=48*1024**3)),
                                ("mem_get_info", (48*1024**3, 48*1024**3))):
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            stack.enter_context(patch.object(torch, "ones", return_value=torch.ones((8, 8))))
            from fine_tuning.notebook_rl import shared_smoke
            for format_, copies in ((measure.FORMAT, 1), ("notebook-rl-training-job-v1", 2), (shared_smoke.FORMAT, 1)):
                output.side_effect = ["git@github.com:jang2296/MMDL_test.git", "a"*40, "",
                                      "GPU-example, fixture GPU, fixture"]
                actual_scope = scope | {"purpose": shared_smoke.PURPOSE} if format_ == shared_smoke.FORMAT else scope
                result = launch.runtime_check(self.job | {"format": format_, "expected_gpu_scope": actual_scope,
                    "deadline_utc": iso(now+3600)}, self.cfg)
                self.assertEqual(result["weight_copies"], copies)
                self.assertEqual(result["minimum_weights_and_kv_bytes"],
                    shared_smoke.memory_floor(self.cfg, weight.stat().st_size) if format_ == shared_smoke.FORMAT else
                    copies*weight.stat().st_size+self.cfg["engine"]["vllm"]["kv_cache_memory_bytes"])
                self.assertFalse(result["memory_fit_certified"])


if __name__ == "__main__":
    unittest.main()
