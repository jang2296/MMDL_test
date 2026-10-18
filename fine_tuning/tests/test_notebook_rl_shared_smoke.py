"""24GB execution boundaries; authored orchestration is not 4B/GPU evidence."""
import copy
import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl import checkpoint, launch, shared_smoke as smoke
from fine_tuning.notebook_rl.batches import CollectedBatch
from fine_tuning.notebook_rl.dataset import _bytes
from fine_tuning.notebook_rl.schema import from_json, ProblemTree, to_json
from fine_tuning.tests import test_notebook_rl_gate_session as gates
from fine_tuning.tests import test_notebook_rl_checkpoint as ck
from fine_tuning.tests.test_notebook_rl_updates import tree
from fine_tuning.tests.test_notebook_rl_policy import bundle


class SharedSmokeTests(unittest.TestCase):
    def setUp(self):
        fixture = gates.GateSessionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.cfg = fixture.root, fixture.cfg
        self.job = fixture.job | {"format": smoke.FORMAT, "arm": "MOD-RV", "algorithm": "ALG-NB3",
            "runtime_scope": fixture.job["runtime_scope"] | {"purpose": smoke.PURPOSE},
            "problem_ids": [str(n) for n in range(1, 11)]}
        actors = fixture.actors + [replace(fixture.actors[0], problem_id=str(n)) for n in range(3, 11)]
        self.checked = fixture.checked | {"actors": actors,
            "structures": {a.problem_id: f"{i:064x}" for i, a in enumerate(actors)}}

    def test_original_protocol_and_memory_floor(self):
        self.assertEqual(self.cfg["sha256"], "653fd442b9184384f1e028ad7269c8d8022cbe59b06c26f90afacd8c82d3daea")
        self.assertEqual(smoke.memory_floor(self.cfg, 8_875_719_344), 11_187_632_816)
        for value in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                smoke.memory_floor(self.cfg, value)

    def test_ten_eligible_unique_structures_and_actual_bound_probes(self):
        before = copy.deepcopy(self.cfg)
        with patch.object(smoke.approval, "validate_for_gate_data", return_value=self.checked):
            self.assertEqual(smoke.preflight(self.job, self.cfg), self.checked)
            for updates in ({"structures": dict.fromkeys(self.job["problem_ids"], "a"*64)},
                            {"actors": self.checked["actors"][:-1]}):
                with patch.object(smoke.approval, "validate_for_gate_data", return_value=self.checked | updates):
                    with self.assertRaises(ValueError):
                        smoke.preflight(self.job, self.cfg)
        self.assertEqual(before, self.cfg)
        for changes in ({"problem_ids": self.job["problem_ids"][:-1]}, {"arm": "MOD-FML"},
                        {"algorithm": "ALG-T1"}, {"runtime_scope": None},
                        {"problem_ids": ["1"]*10}, {"max_steps": 20},
                        {"output": str(launch.ROOT/"MMDL"/"blocked")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                smoke.validate_job(self.job | changes, self.cfg)

    def test_blocked_data_never_allocates_or_creates_output(self):
        path = self.root/"job.json"
        path.write_bytes(_bytes(self.job))
        blocked = self.checked | {"status": "BLOCKED_NOT_CHECKED", "measurement_allowed": False,
                                  "blockers": ["authored"]}
        for mode in ("--check", "--execute"):
            with patch.object(smoke, "preflight", return_value=blocked), patch.object(smoke, "execute") as execute, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(smoke.main(["--job", str(path), mode]), 2)
                execute.assert_not_called()
        with patch.object(launch, "runtime_check") as runtime, self.assertRaises(ValueError):
            smoke.execute(self.job, self.cfg, blocked)
        runtime.assert_not_called()
        self.assertFalse(Path(self.job["output"]).exists())

    def test_notes_preserved_for_discarded_trees_and_budget_stops(self):
        # Tree schema is real; the note text below is an authored control fixture.
        t = tree(bundle(), "1", version=0, route="VISION_OFF")
        batch = CollectedBatch([], {}, 1, ["1"], [{"problem_id": "1", "tree": to_json(t)}], None, 0., "EXHAUSTED")
        image = Image.new("RGB", (8, 8))
        def note(t, *_args, **_kwargs):
            t.notebook["final_notebook"] = {"status": "BUDGET_STOP"}
            return t.notebook["final_notebook"]
        with patch.object(smoke, "load_images", return_value=[image]), \
                patch("fine_tuning.notebook_rl.rollout.append_final_notebook", side_effect=note):
            statuses = smoke._final_notes(batch, {"1": self.checked["actors"][0]}, self.root, None, {}, None, lambda: True)
        self.assertEqual(statuses, {"1": "BUDGET_STOP"})
        self.assertEqual(batch.status, "BUDGET_STOP")
        self.assertEqual(from_json(ProblemTree, batch.discarded[0]["tree"]).notebook["final_notebook"]["status"], "BUDGET_STOP")
        with self.assertRaises(ValueError):
            image.getpixel((0, 0))

    def run_capture(self, *, failure=False, signal=True):
        # Actual tiny Adam update + real checkpoint.save; all GPU/rollout controls are authored.
        import torch
        self.checked["code_sha256"] = smoke.approval.code_hashes()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model, manifest, cfg, _ = ck.fixture(Path(tmp.name))
        cfg["initial_checkpoint"]["manifest"] = str(manifest)
        bundle = ck.fresh_tiny(model, "MOD-RV", cfg)
        output = self.root/f"capture-{failure}-{signal}"
        job = self.job | {"model_path": str(model), "output": str(output)}
        data = Path(job["build_root"])
        data.mkdir(exist_ok=True)
        (data/"manifest.yaml").write_bytes(_bytes({"split_manifests": {"train": "authored"}}))
        engine = SimpleNamespace(policy_version=0, _synchronized=True)
        def collect(actors, structures, order, cursor, *_args, **_kwargs):
            pid = order[0]
            return CollectedBatch([tree(bundle, pid, version=0)] if signal else [],
                {pid: [Image.new("RGB", (8, 8))]} if signal else {}, 10, order, [], None, 0., "EXHAUSTED")
        def update(b, opt, sched, *_args, **_kwargs):
            ck.action_step(b, opt, sched)
            return {"status": "UPDATED_UNSYNCHRONIZED", "step_after": 1}
        def sync(_b, _p, _e, _probes, _cfg, step, **kwargs):
            if step and failure:
                raise RuntimeError("authored sync failure")
            engine.policy_version = step
            return []
        stack = ExitStack()
        self.addCleanup(stack.close)
        for target, kwargs in (
            ("fine_tuning.notebook_rl.shared_smoke.preflight", {"return_value": self.checked}),
            ("fine_tuning.notebook_rl.launch.runtime_check", {"return_value": {"device": {"index": 0}, "deadline_timestamp": 1e20}}),
            ("fine_tuning.notebook_rl.guards.install_guards", {}),
            ("fine_tuning.notebook_rl.policy.build_policy", {"return_value": bundle}),
            ("fine_tuning.notebook_rl.gpu_gates.measure_inventory", {"return_value": {"authored": True}}),
            ("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine", {"return_value": engine}),
            ("fine_tuning.notebook_rl.prompts.load_prompts", {"return_value": {"sha256": cfg["prompts_sha256"]}}),
            ("fine_tuning.notebook_rl.launch.load_probes", {"return_value": []}),
            ("fine_tuning.notebook_rl.verifier_service.VerifierService", {"return_value": __import__("contextlib").nullcontext(SimpleNamespace(metadata={"vault_manifest_sha256": "authored"}))}),
            ("fine_tuning.notebook_rl.batches.collect_batch", {"side_effect": collect}),
            ("fine_tuning.notebook_rl.shared_smoke._final_notes", {"return_value": {}}),
            ("fine_tuning.notebook_rl.updates.optimizer_step", {"side_effect": update}),
            ("fine_tuning.notebook_rl.trainer.synchronize", {"side_effect": sync}),
        ):
            stack.enter_context(patch(target, **kwargs))
        stack.enter_context(patch.object(bundle.model, "to", return_value=bundle.model))
        for name in ("reset_peak_memory_stats", "empty_cache", "max_memory_allocated", "max_memory_reserved"):
            stack.enter_context(patch.object(torch.cuda, name, return_value=0))
        if failure:
            with self.assertRaisesRegex(RuntimeError, "sync failure"):
                smoke.execute(job, cfg, self.checked)
            self.assertEqual(launch.read_json(output/"result.json")["status"], "FAILED")
            self.assertFalse((output/"checkpoint").exists())
            return None
        result = smoke.execute(job, cfg, self.checked)
        return job, cfg, result, bundle

    def test_capture_real_tiny_checkpoint_complete_cursor_and_reload_boundaries(self):
        import os
        job, cfg, result, bundle = self.run_capture()
        self.assertEqual(result["status"], "CAPTURED_PENDING_FRESH_RELOAD")
        self.assertFalse(result["fresh_reload_verified"])
        metadata = checkpoint.read_metadata(Path(result["checkpoint"]), binding=result["binding"])
        self.assertEqual(metadata["step"], 1)
        # The capture fixture is still alive: calling the reload API is not a fresh process.
        with self.assertRaisesRegex(ValueError, "process"):
            smoke.reload_check(job, cfg)
        closed = launch.read_json(Path(job["output"])/"launch.json")
        closed["capture_pid"] = os.getpid()+1_000_000
        Path(job["output"], "launch.json").write_bytes(_bytes(closed))
        _, _, _, step, cursor = checkpoint.load_resume(Path(job["model_path"]), Path(result["checkpoint"]), cfg,
            binding=result["binding"], _test_policy_factory=ck.fresh_tiny, device="cpu")
        self.assertFalse(cursor["sync_pending"])
        self.assertEqual(cursor["position"], 10)
        # Reload boundary checks use a CPU-return double; not fresh-process/GPU evidence.
        with patch("psutil.pid_exists", return_value=False), patch.object(checkpoint, "load_resume",
                return_value=(bundle, None, None, step, cursor)):
            value = smoke.reload_check(job, cfg)
            self.assertTrue(value["fresh_reload_verified"])
        expected_path = Path(job["output"])/"reload_expected.json"
        expected = launch.read_json(expected_path)
        expected["parameter_sha256"] = {}
        expected_path.write_bytes(_bytes(expected))
        with patch("psutil.pid_exists", return_value=False), patch.object(checkpoint, "load_resume",
                return_value=(bundle, None, None, step, cursor)), self.assertRaisesRegex(ValueError, "differ"):
            smoke.reload_check(job, cfg)

    def test_failed_sync_or_no_learning_signal_is_not_completion(self):
        self.run_capture(failure=True)
        _, _, result, _ = self.run_capture(signal=False)
        self.assertEqual(result["status"], "NO_LEARNING_SIGNAL")
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertNotIn("checkpoint", result)


if __name__ == "__main__":
    unittest.main()
