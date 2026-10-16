"""Bounded launcher wiring and real tiny update; never pretrained CUDA evidence."""
import copy
import io
import json
import subprocess
import sys
import time
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from fine_tuning.notebook_rl import gate_session as session
from fine_tuning.notebook_rl import gate_throughput, launch
from fine_tuning.notebook_rl.dataset import _bytes, _sha
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_gate_throughput as throughput_fixture
from fine_tuning.tests import test_notebook_rl_measure as measurement_fixture
from fine_tuning.tests.test_notebook_rl_schema import records


class GateSessionTests(unittest.TestCase):
    def setUp(self):
        fixture = measurement_fixture.MeasurementLaunchTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.cfg = fixture.root, fixture.cfg
        self.job = fixture.job | {"format": session.FORMAT, "arm": "MOD-FML",
            "runtime_scope": fixture.job["runtime_scope"] | {"purpose": session.PURPOSE}}
        self.actors = [replace(records()[0], problem_id=str(n),
            image_paths=tuple(f"{i}.png" for i in range(n)), image_sha256=tuple("a"*64 for _ in range(n))) for n in (1, 2)]
        rows = [{"id": f"probe-{a.problem_id}", "actor": json.loads(to_json(a)), "completion_ids": [10, 11, 2]}
                for a in self.actors]
        path = self.root/"probes.json"
        raw = _bytes({"probes": rows})
        path.write_bytes(raw)
        self.job["sync_probes"] = {"path": str(path), "file_sha256": _sha(raw)}
        self.job["probe_scope"] = {"probes": [{"id": r["id"], "image_count": len(r["actor"]["image_paths"]),
            "input_sha256": _sha(_bytes(r))} for r in rows]}
        self.checked = fixture.checked | {"actors": self.actors, "structures": {"1": "a"*64, "2": "b"*64}}

    def task(self, stage="capture", previous=None):
        value = {"stage": stage, "job": self.job, "validated_sha256": session._validation_hash(self.checked),
                 "capture_handoff": previous}
        path = self.root/"task.json"
        path.write_bytes(_bytes(value))
        return {"path": str(path), "file_sha256": _sha(path.read_bytes())}

    def test_real_actor_shape_has_canonical_validation_hash_and_bound_probes(self):
        session.validate_job(self.job, self.cfg)
        wires = self.checked | {"actors": [json.loads(to_json(a)) for a in self.actors]}
        self.assertEqual(session._validation_hash(self.checked), session._validation_hash(wires))
        with patch.object(session.approval, "validate_for_measurement", return_value=self.checked):
            self.assertEqual(session.preflight(self.job, self.cfg), self.checked)
        self.assertEqual(set(session._probe_actors(self.job, self.checked)), {"probe-1", "probe-2"})

    def test_cannot_override_bounded_job_or_use_frozen_only_authority(self):
        for change in ({"algorithm": "ALG-T1"}, {"max_steps": 30}, {"seed": True},
                       {"arm": "MOD-RS"}, {"output": str(self.root)},
                       {"runtime_scope": self.job["runtime_scope"] | {"purpose": "FROZEN_WEIGHT_MEASUREMENT_ONLY"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                session.validate_job(self.job | change, self.cfg)

    def test_probe_must_be_eligible_exact_actor_and_hash_bound(self):
        with self.assertRaisesRegex(ValueError, "outside eligible"):
            session._probe_actors(self.job, self.checked | {"actors": self.actors[:1]})
        changed = copy.deepcopy(self.job)
        changed["probe_scope"]["probes"][0]["input_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "bound single/two"):
            session._probe_actors(changed, self.checked)
        Path(self.job["sync_probes"]["path"]).write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "descriptor"):
            session._probe_actors(self.job, self.checked)

    def test_actual_missing_data_blocks_check_and_execute_before_gpu_or_child(self):
        path = self.root/"job.json"
        path.write_bytes(_bytes(self.job))
        for flag in ("--check", "--execute"):
            with patch.object(launch, "runtime_check") as runtime, patch.object(session, "_run_child") as child, \
                    redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(session.main(["--job", str(path), flag]), 2)
                result = json.loads(stdout.getvalue())
                self.assertFalse(result["gpu_runtime_checked"])
                runtime.assert_not_called()
                child.assert_not_called()
        self.assertFalse(Path(self.job["output"]).exists())

    def test_worker_revalidates_before_runtime_and_refuses_changed_source(self):
        descriptor = self.task()
        for checked in ({}, self.checked | {"structures": {}}):
            with patch.object(session, "preflight", return_value=checked), patch.object(launch, "runtime_check") as runtime:
                with self.assertRaisesRegex(ValueError, "changed before child"):
                    session.worker(descriptor, self.cfg)
                runtime.assert_not_called()

    def test_worker_preserves_failure_receipt_not_success_or_billing_stop(self):
        descriptor = self.task()
        with patch.object(session, "preflight", return_value=self.checked), \
                patch.object(launch, "runtime_check", return_value={"device": {"index": 0}}) as runtime, \
                patch.object(session, "_capture", side_effect=RuntimeError("authored failure")):
            with self.assertRaisesRegex(RuntimeError, "authored failure"):
                session.worker(descriptor, self.cfg)
        self.assertEqual(runtime.call_args.args[0]["format"], session.FORMAT)  # Two-copy floor, never frozen-only.
        result = launch.read_json(Path(self.job["output"])/"result.json")
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["gpu_gates_certified"])
        self.assertFalse(result["pod_billing_stopped"])

    def test_model_free_parent_cannot_accept_unapproved_or_changed_inputs(self):
        with patch.object(session, "_run_child") as child:
            with self.assertRaisesRegex(ValueError, "eligibility"):
                session.execute(self.job, self.cfg, {})
            with patch.object(session, "preflight", return_value={}):
                with self.assertRaisesRegex(ValueError, "eligibility"):
                    session.execute(self.job, self.cfg, self.checked)
            child.assert_not_called()

    def test_parent_retains_same_handles_saved_step1_and_no_approval(self):
        handles = [object(), object()]
        calls = []
        def child(task, path, deadline):
            calls.append(task)
            target = Path(task["job"]["output"])
            target.mkdir()
            if task["stage"] == "capture":
                value = {"status": "CAPTURE_COMPLETED", "capture": str(target/"capture"),
                    "shared_sha256": "b"*64, "saved_step": 1, "live_final_step": 2}
                (target/"result.json").write_bytes(_bytes(value))
            return handles[len(calls)-1]
        # Explicit orchestration fixture, not native model/GPU evidence.
        with patch.object(session, "preflight", return_value=self.checked), \
                patch.object(session, "_run_child", side_effect=child), \
                patch("fine_tuning.notebook_rl.gate_checkpoint.compare_checkpoint",
                    return_value={"native_execution": True, "passed": True}) as compare:
            result = session.execute(self.job, self.cfg, self.checked)
        self.assertEqual([t["stage"] for t in calls], ["capture", "replay"])
        self.assertIs(compare.call_args.args[2], handles[0])
        self.assertIs(compare.call_args.args[3], handles[1])
        self.assertEqual(result["saved_checkpoint_step"], 1)
        self.assertEqual(result["capture_optimizer_steps"], 2)
        self.assertEqual(result["replay_optimizer_steps"], 1)
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["phase_a_completed"])
        self.assertFalse(result["gpu_gates_certified"])

    def test_failed_capture_never_starts_replay_and_keeps_failure_receipt(self):
        with patch.object(session, "preflight", return_value=self.checked), \
                patch.object(session, "_run_child", side_effect=RuntimeError("capture failed")) as child:
            with self.assertRaises(RuntimeError):
                session.execute(self.job, self.cfg, self.checked)
            child.assert_called_once()
        result = launch.read_json(Path(self.job["output"])/"result.json")
        self.assertEqual(result["status"], "INCOMPLETE")

    def test_actual_child_cli_uses_fixed_module_and_preserves_private_failure(self):
        path = self.root/"real-task.json"
        with patch.object(session, "__name__", "__main__"), self.assertRaisesRegex(ValueError, "child failed"):
            session._run_child({"invalid": True}, path, time.time()+30)
        text = path.with_suffix(".log").read_text()
        self.assertIn("exact private stage task required", text)
        self.assertNotIn("Error while finding module specification for '__main__'", text)
        self.assertTrue(path.exists())

    def test_actual_owned_child_is_terminated_on_timeout(self):
        actual_popen, owned = subprocess.Popen, []
        def sleeping_child(*args, **kwargs):
            child = actual_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            owned.append(child)
            return child
        with patch.object(session.subprocess, "Popen", side_effect=sleeping_child):
            with self.assertRaises(subprocess.TimeoutExpired):
                session._run_child({}, self.root/"timeout.json", time.time()+.2)
        self.assertEqual(len(owned), 1)
        self.assertIsNotNone(owned[0].poll())

    def test_capture_wiring_uses_new_version1_batch_and_preserves_first_update_on_failure(self):
        # Dependency fixtures exercise ordering/cleanup, not GPU/model identity.
        from fine_tuning.notebook_rl import batches, checkpoint, gate_inputs
        from fine_tuning.notebook_rl import engines, policy, verifier_service

        root = Path(self.job["build_root"])
        root.mkdir()
        (root/"manifest.yaml").write_bytes(_bytes({"split_manifests": {"train": "f"*64}}))
        for name, status in (("complete", "COLLECTED"), ("stopped", "BUDGET_STOP")):
            with self.subTest(status=status), ExitStack() as stack:
                output = self.root/name
                output.mkdir()
                b = SimpleNamespace(model=SimpleNamespace(to=Mock(), config=object()), processor=object())
                scheduler = SimpleNamespace(last_epoch=0)
                engine = SimpleNamespace(policy_version=0, llm=SimpleNamespace(llm_engine=SimpleNamespace(
                    engine_core=SimpleNamespace(shutdown=Mock()))))
                preparation = SimpleNamespace(policy_version=0)
                verifier = SimpleNamespace(metadata={"vault_manifest_sha256": "f"*64})
                events = []
                image = Image.new("RGB", (2, 2))
                self.addCleanup(image.close)
                batch = batches.CollectedBatch([], {"1": [image]}, 1, ["1"], [], None, .1, status)
                @contextmanager
                def service(*args, **kwargs):
                    yield verifier
                def shared(*args, **kwargs):
                    events.append("shared0-1")
                    scheduler.last_epoch = engine.policy_version = preparation.policy_version = 1
                    return {"G11": {}, "G13": {}}
                def save(*args, **kwargs):
                    self.assertEqual(args[3], 1)
                    events.append("save-shared1")
                    args[6].mkdir()
                    (args[6]/"metadata.json").write_bytes(b"{}")
                def collect(*args, **kwargs):
                    self.assertEqual(engine.policy_version, 1)
                    events.append("collect-version1")
                    return batch
                def packet(path, **kwargs):
                    self.assertEqual(kwargs["step"], 1)
                    self.assertEqual(preparation.policy_version, 1)
                    self.assertIs(kwargs["images"]["1"][0], image)
                    self.assertIn("NOT_TRAINING_RESUME", kwargs["cursor"]["scope"])
                    events.append("freeze-version1")
                    return {"path": str(path), "file_sha256": "a"*64}
                def capture(*args, **kwargs):
                    self.assertEqual(scheduler.last_epoch, 1)
                    events.append("capture1-2")
                    scheduler.last_epoch = 2
                    return {"native_execution": True, "step": 1}
                for owner, attr, value in ((policy, "build_policy", Mock(return_value=b)),
                        (checkpoint, "make_optimizer", Mock(return_value=(object(), scheduler))),
                        (checkpoint, "save", Mock(side_effect=save)),
                        (engines, "VLLMRolloutEngine", Mock(return_value=engine)),
                        (engines, "InputPreparation", Mock(return_value=preparation)),
                        (launch, "load_probes", Mock(return_value=[])),
                        (verifier_service, "VerifierService", service),
                        (session, "_shared_step", shared), (batches, "collect_batch", collect),
                        (gate_inputs, "save_inputs", packet), (gate_inputs, "capture_from_inputs", capture)):
                    stack.enter_context(patch.object(owner, attr, value))
                stack.enter_context(patch("fine_tuning.notebook_rl.guards.prepare_vllm_schema_dependency", return_value={}))
                args = (self.job, self.cfg, self.checked, {"device": {"index": 0}, "deadline_timestamp": time.time()+60},
                        output, {"fixture": "binding"}, lambda row: None)
                if status == "COLLECTED":
                    result = session._capture(*args)
                    self.assertEqual(result["saved_step"], 1)
                    self.assertEqual(result["live_final_step"], 2)
                    self.assertEqual(events, ["shared0-1", "save-shared1", "collect-version1", "freeze-version1", "capture1-2"])
                else:
                    with self.assertRaisesRegex(ValueError, "batch incomplete"):
                        session._capture(*args)
                    self.assertEqual(events, ["shared0-1", "save-shared1", "collect-version1"])
                    self.assertTrue((output/"shared_checkpoint/metadata.json").exists())
                engine.llm.llm_engine.engine_core.shutdown.assert_called_once()
                with self.assertRaises(ValueError):
                    image.tobytes()

    def test_replay_installs_all_split_private_guards_before_native_reload(self):
        from fine_tuning.notebook_rl import gate_inputs, guards
        from fine_tuning.notebook_rl.dataset import SPLITS

        binding = launch.checkpoint_binding(self.checked, self.cfg, self.job["commit"])
        previous = {"status": "CAPTURE_COMPLETED", "binding": binding, "saved_step": 1, "live_final_step": 2,
            "capture": str(self.root/"capture"), "input_packet": {"path": str(self.root/"packet"), "file_sha256": "c"*64}}
        path = self.root/"capture-handoff.json"
        path.write_bytes(_bytes(previous))
        task = self.task("replay", {"path": str(path), "file_sha256": _sha(path.read_bytes())})
        calls = []
        def guarded(**kwargs):
            calls.append("guards")
            blocked = set(kwargs["vault_files"])
            self.assertTrue({Path(self.job["build_root"])/s/n for s in SPLITS
                for n in ("verifier_private.jsonl", "lineage.jsonl")} <= blocked)
        def replay(*args, **kwargs):
            self.assertEqual(calls, ["guards"])
            calls.append("replay")
            self.assertEqual(args[3], previous["input_packet"])
            return {"native_execution": True, "step": 1}
        with patch.object(session, "preflight", return_value=self.checked), \
                patch.object(launch, "runtime_check", return_value={"device": {"index": 0}}), \
                patch.object(guards, "prepare_vllm_schema_dependency", return_value={}), \
                patch.object(guards, "install_guards", side_effect=guarded), \
                patch.object(gate_inputs, "replay_from_inputs", side_effect=replay):
            result = session.worker(task, self.cfg)
        self.assertEqual(calls, ["guards", "replay"])
        self.assertEqual(result["status"], "REPLAY_COMPLETED")
        self.assertFalse(result["training_allowed"])

    def test_shared_gates_have_one_actual_tiny_update_and_reject_cpu_as_native(self):
        fixture = throughput_fixture.ThroughputTests()
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(fixture.doCleanups)
        args, journal, _ = fixture.fixture()
        observed = []
        @contextmanager
        def guard(*values, **kwargs):
            observed.append(("begin", args["scheduler"].last_epoch))
            report = {"status": "OBSERVING", "native_execution": False}
            yield report
            observed.append(("end", args["scheduler"].last_epoch))
            report["status"] = "MEASURED_ONE_RL_UPDATE"
        params = {k: v for k, v in args.items() if k not in ("algorithm", "step", "seed")}
        with patch("fine_tuning.notebook_rl.gate_guards.measure_guards", side_effect=guard), \
                patch.object(gate_throughput, "measure_step", side_effect=gate_throughput._measure_step), \
                patch("torch.nn.functional.cross_entropy", side_effect=AssertionError("SFT forbidden")):
            with self.assertRaisesRegex(ValueError, "native finalized"):
                session._shared_step(**params, job=self.job)
        self.assertEqual(observed, [("begin", 0), ("end", 1)])
        self.assertEqual(args["engine"].pushes, [1])
        self.assertEqual(args["scheduler"].last_epoch, 1)
        self.assertTrue(journal[-1]["weights_updated"])
        self.assertFalse(journal[-1]["native_execution"])


if __name__ == "__main__":
    unittest.main()
