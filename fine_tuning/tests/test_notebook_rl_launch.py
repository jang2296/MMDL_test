"""Launch trust boundaries; no cloud mutation or GPU execution in these fixtures."""
import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from PIL import Image

from fine_tuning.notebook_rl import launch
from fine_tuning.notebook_rl.dataset import _bytes, _sha
from fine_tuning.notebook_rl.prompts import load_prompts
from fine_tuning.notebook_rl.schema import load_protocol
from fine_tuning.tests.test_notebook_rl_schema import records


class LaunchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = load_protocol(launch.HERE/"protocol_v1.yaml")
        self.job: dict[str, Any] = {"format": "notebook-rl-training-job-v1", "build_root": str(self.root/"data"),
            "selection_path": str(self.root/"selection.json"), "phase_a_report": str(self.root/"phase.json"),
            "model_path": str(self.root/self.cfg["initial_checkpoint"]["revision"]),
            "arm": "MOD-RV", "algorithm": "ALG-NB+D", "seed": 3407, "output": str(self.root/"run"),
            "deadline_utc": "2030-01-01T00:00:00+00:00", "max_steps": 3, "resume": None,
            "commit": "a"*40, "forbidden_roots": [str(self.root/"benchmark")],
            "expected_gpu_scope": {"arm": "MOD-RV", "runtime_id": "authored-pod"}}
        for key in launch.DESCRIPTORS:
            self.job[key] = {"path": str(self.root/(key+".json")), "file_sha256": "b"*64}

    def test_valid_job_shape_is_not_training_permission(self):
        launch.validate_job(self.job, self.cfg)
        result = launch.preflight(self.job, self.cfg)
        self.assertFalse(result["training_allowed"])
        self.assertEqual(result["status"], "BLOCKED_NOT_CHECKED")
        self.assertFalse(Path(self.job["output"]).exists())

    def test_no_inline_private_evidence_or_wrong_job_fields(self):
        for key, value in (("evidence", {"raw_benchmark_text": "not-allowed"}), ("seed", True),
                           ("arm", "MOD-RS"), ("max_steps", 0), ("forbidden_roots", []),
                           ("output", str(launch.ROOT/"MMDL"/"do-not-write"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.validate_job(self.job | {key: value}, self.cfg)
        with self.assertRaises(ValueError):
            launch.validate_job(self.job | {"training_allowed": True}, self.cfg)

    def test_descriptor_bytes_and_unique_json(self):
        path = self.root/"report.json"
        path.write_text('{"status":"fixture"}')
        desc = {"path": str(path), "file_sha256": _sha(path.read_bytes())}
        self.assertEqual(launch.read_descriptor(desc), {"status": "fixture"})
        path.write_text('{"status":"changed"}')
        with self.assertRaisesRegex(ValueError, "artifact bytes"):
            launch.read_descriptor(desc)
        path.write_text('{"status":0,"status":1}')
        with self.assertRaises(ValueError):
            launch.read_json(path)

    def test_cli_check_and_execute_do_not_allocate_when_blocked(self):
        path = self.root/"job.json"
        path.write_text(json.dumps(self.job))
        for mode in ("--check", "--execute"):
            with self.subTest(mode=mode), patch.object(launch, "execute") as execute, redirect_stdout(io.StringIO()) as out:
                self.assertEqual(launch.main(["--job", str(path), mode]), 2)
                self.assertFalse(json.loads(out.getvalue())["training_executed"])
                execute.assert_not_called()

    def test_approval_cannot_be_bypassed_by_direct_execute(self):
        with patch.object(launch, "runtime_check") as runtime, self.assertRaisesRegex(ValueError, "unapproved"):
            launch.execute(self.job, self.cfg, {"status": "BLOCKED_NOT_CHECKED"})
        runtime.assert_not_called()
        approved = {"status": "VALIDATED_FOR_TRAINING", "training_allowed": True}
        with patch.object(launch, "preflight", return_value={}), patch.object(launch, "runtime_check") as runtime, \
                self.assertRaisesRegex(ValueError, "approval changed"):
            launch.execute(self.job, self.cfg, approved)
        runtime.assert_not_called()

    def test_wrong_checkout_and_missing_watchdog_fail_before_model_load(self):
        with patch.object(launch.subprocess, "check_output", return_value="git@github.com:jang2296/MMDL.git"), \
                self.assertRaisesRegex(ValueError, "MMDL_test"):
            launch.runtime_check(self.job, self.cfg)
        answers = ["git@github.com:jang2296/MMDL_test.git", "a"*40, ""]
        with patch.object(launch.subprocess, "check_output", side_effect=answers), \
                patch.dict(os.environ, {"RUNPOD_POD_ID": "authored-pod", "MMDL_BUDGET_WATCHDOG_PID": "0"}), \
                self.assertRaisesRegex(ValueError, "live ancestor"):
            launch.runtime_check(self.job, self.cfg)

    def test_f0_numbers_without_closed_raw_ledger_fail_before_gpu_allocation(self):
        approved = {"status": "VALIDATED_FOR_TRAINING", "training_allowed": True,
                    "input_binding": {}, "code_sha256": {"launch.py": "a"*64}}
        path = self.root/"f0_report.json"
        path.write_bytes(_bytes({"status": "COMPLETED", "step": 0,
            "input_binding": launch.checkpoint_binding(approved, self.cfg, self.job["commit"]),
            "dev_first_attempt_acc": .5, "grounding_probe": .8, "len_median": 128}))
        self.job["f0_report"] = {"path": str(path), "file_sha256": _sha(path.read_bytes())}
        with patch.object(launch, "preflight", return_value=approved), patch.object(launch, "runtime_check") as runtime, \
                self.assertRaises(ValueError):
            launch.execute(self.job, self.cfg, approved)
        runtime.assert_not_called()

    def test_checkpoint_binding_stable_across_gpu_receipt_but_not_input_changes(self):
        approved = {"code_sha256": {"launch.py": "a"*64}, "input_binding": {
            "selection_sha256": "b"*64, "gpu_report_sha256": "c"*64, "data_report_sha256": "d"*64}}
        before = launch.checkpoint_binding(approved, self.cfg, self.job["commit"])
        changed = copy.deepcopy(approved)
        changed["input_binding"]["gpu_report_sha256"] = "e"*64
        self.assertEqual(before, launch.checkpoint_binding(changed, self.cfg, self.job["commit"]))
        changed["input_binding"]["selection_sha256"] = "f"*64
        self.assertNotEqual(before, launch.checkpoint_binding(changed, self.cfg, self.job["commit"]))

    def test_sync_probes_bind_actual_images_token_ids_and_measured_scope(self):
        image = self.root/"a.png"
        Image.new("RGB", (16, 16), "white").save(image)
        actor = replace(records()[0], mode="plain", image_paths=("a.png",), image_sha256=(_sha(image.read_bytes()),))
        rows = []
        for n in (1, 2):
            candidate = replace(actor, image_paths=("a.png",)*n, image_sha256=actor.image_sha256*n)
            rows.append({"id": f"p{n}", "actor": asdict(candidate), "completion_ids": [3, 4]})
        path = self.root/"probes.json"
        path.write_bytes(_bytes({"probes": rows}))
        desc = {"path": str(path), "file_sha256": _sha(path.read_bytes())}
        scope = {"probes": [{"id": row["id"], "image_count": i+1, "input_sha256": _sha(_bytes(row))}
                             for i, row in enumerate(rows)]}
        prompts = load_prompts(launch.HERE/"prompts_v1.yaml")
        with ExitStack() as stack:
            probes = launch.load_probes(desc, scope, self.root, prompts, stack)
            self.assertEqual([len(p.request.images) for p in probes], [1, 2])
            self.assertEqual(probes[0].completion_ids, [3, 4])
        scope["probes"][0]["input_sha256"] = "f"*64
        with ExitStack() as stack, self.assertRaisesRegex(ValueError, "differs"):
            launch.load_probes(desc, scope, self.root, prompts, stack)

    def test_live_inventory_precedes_engine_and_owned_engine_always_closes(self):
        # Authored approval/runtime doubles exercise orchestration only, never GPU permission.
        data = Path(self.job["build_root"])
        data.mkdir()
        (data/"manifest.yaml").write_bytes(_bytes({"split_manifests": {"train": "authored-vault"}}))
        phase = {"status": "PHASE_A_COMPLETED", "policy_version": 0, "health_reference": {"len_median": 80}}
        phase_path = Path(self.job["phase_a_report"])
        phase_path.write_bytes(_bytes(phase))
        approved = {"status": "VALIDATED_FOR_TRAINING", "training_allowed": True, "actors": [], "structures": {},
                    "code_sha256": {"launch.py": "a"*64},
                    "input_binding": {"phase_a_report_sha256": _sha(phase_path.read_bytes())}}
        f0 = {"status": "VALIDATED_F0_REFERENCE", "f0_report_sha256": self.job["f0_report"]["file_sha256"],
              "dev_first_attempt_acc": .5, "grounding_probe": .7}
        for failure in (None, "inventory", "preparation", "training"):
            with self.subTest(failure=failure), ExitStack() as stack:
                job = self.job | {"output": str(self.root / f"run-{failure}")}
                bundle = SimpleNamespace(model=Mock(), inventory={"arm": "MOD-RV"}, processor=Mock())
                shutdown = Mock()
                engine = SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(
                    engine_core=SimpleNamespace(shutdown=shutdown))))
                verifier = Mock()
                verifier.__enter__ = Mock(return_value=SimpleNamespace(metadata={"vault_manifest_sha256": "authored-vault"}))
                verifier.__exit__ = Mock(return_value=False)
                def mock(name, **kwargs):
                    return stack.enter_context(patch(name, **kwargs))
                mock("fine_tuning.notebook_rl.launch.preflight", return_value=approved)
                mock("fine_tuning.notebook_rl.launch.runtime_check",
                     return_value={"device": {"index": 0}, "deadline_timestamp": 1e20})
                mock("fine_tuning.notebook_rl.monitor.validate_f0_reference", return_value=f0)
                mock("fine_tuning.notebook_rl.guards.prepare_vllm_schema_dependency", return_value={"authored": True})
                mock("fine_tuning.notebook_rl.guards.install_guards")
                mock("fine_tuning.notebook_rl.policy.build_policy", return_value=bundle)
                mock("fine_tuning.notebook_rl.checkpoint.make_optimizer", return_value=(Mock(), Mock()))
                mock("fine_tuning.notebook_rl.checkpoint.capture_rng", return_value={"authored": True})
                mock("fine_tuning.notebook_rl.checkpoint.restore_rng")
                inventory = mock("fine_tuning.notebook_rl.gpu_gates.measure_inventory", return_value={"authored": True},
                                 side_effect=ValueError("inventory failure") if failure == "inventory" else None)
                create_engine = mock("fine_tuning.notebook_rl.engines.VLLMRolloutEngine", return_value=engine)
                mock("fine_tuning.notebook_rl.engines.InputPreparation", return_value=Mock(),
                     side_effect=ValueError("preparation failure") if failure == "preparation" else None)
                mock("fine_tuning.notebook_rl.launch.load_probes", return_value=[])
                mock("fine_tuning.notebook_rl.verifier_service.VerifierService", return_value=verifier)
                run = mock("fine_tuning.notebook_rl.trainer.run_loop", return_value={"status": "STEP_LIMIT"},
                           side_effect=RuntimeError("training failure") if failure == "training" else None)
                if failure:
                    with self.assertRaises((ValueError, RuntimeError)):
                        launch.execute(job, self.cfg, approved)
                else:
                    self.assertEqual(launch.execute(job, self.cfg, approved), {"status": "STEP_LIMIT"})
                inventory.assert_called_once_with(bundle, self.cfg)
                if failure == "inventory":
                    create_engine.assert_not_called()
                    shutdown.assert_not_called()
                else:
                    create_engine.assert_called_once()
                    shutdown.assert_called_once()
                    saved = json.loads((Path(job["output"])/"policy_inventory.json").read_bytes())
                    self.assertEqual(saved["measurements"], {"authored": True})
                    self.assertFalse(saved["training_complete"])
                if failure in ("inventory", "preparation"):
                    run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
