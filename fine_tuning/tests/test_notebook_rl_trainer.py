"""Actual tiny-Qwen updates on controlled actions; not pretrained 4B self-correction."""
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl.batches import CollectedBatch
from fine_tuning.notebook_rl.checkpoint import load_resume, make_optimizer, read_metadata
from fine_tuning.notebook_rl.engines import RolloutRequest
from fine_tuning.notebook_rl.policy import assert_base_unchanged, reference_logprobs
from fine_tuning.notebook_rl.trainer import SyncProbe, batch_statistics, grounding_disabled, run_loop
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests.test_notebook_rl_checkpoint import fixture, fresh_tiny
from fine_tuning.tests.test_notebook_rl_policy import inputs
from fine_tuning.tests.test_notebook_rl_schema import records
from fine_tuning.tests.test_notebook_rl_updates import Preparation, tree


class SyncFixture(Preparation):
    """Checks the control protocol; it is NOT a real vLLM weight-cache test."""
    def __init__(self, version=0, fail_push=False):
        self.policy_version, self.pushes, self.probes = version, [], []
        self.fail_push = fail_push

    def push_weights(self, named, version):
        self.pushes.append(version)
        if self.fail_push:
            raise RuntimeError("fixture push failed")
        weights = dict(named)
        assert weights and all(torch.isfinite(v).all() for v in weights.values())
        assert version == self.policy_version + 1
        self.policy_version = version

    def validate_sync(self, probes, threshold):
        assert len(probes) == 2 and sorted(len(p[0].images) for p in probes) == [1, 2]
        assert 0 < threshold <= .02
        self.probes.append(self.policy_version)
        return [{"request_id": p[0].request_id, "policy_version": self.policy_version,
                 "mean_abs_delta": 0, "test_fixture_not_vllm": True} for p in probes]


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model_path, _, self.cfg, self.binding = fixture(self.root)
        self.cfg["batch"] = {"problems_per_step": 1, "max_attempted_problems_per_step": 3, "epochs": 1}
        # These controlled actions are intentionally 3 tokens, not natural reasoning.
        self.cfg["health"]["short_response_max_fraction"] = 1.0
        self.b = fresh_tiny(self.model_path, "MOD-RV", self.cfg)
        self.opt, self.sched = make_optimizer(self.b, self.cfg)
        actor = records()[0]
        self.actors = [replace(actor, problem_id=f"p{i}") for i in range(4)]
        self.structures = {a.problem_id: hashlib.sha256(a.problem_id.encode()).hexdigest() for a in self.actors}
        self.engine = SyncFixture()
        self.probes = []
        for n in (1, 2):
            images = [Image.new("RGB", (16, 16), "white") for _ in range(n)]
            for im in images:
                self.addCleanup(im.close)
            req = RolloutRequest(f"probe{n}", [4, 3, 3, 3, 3, 5, 9], None, images, 1, 3, [3407])
            self.probes.append(SyncProbe(req, [10, 11, 2]))

    def collect(self, actors, structures, order, cursor, loader, engine, *args, **kwargs):
        pid = order[cursor]
        row = tree(self.b, pid, version=engine.policy_version, route="VISION_ON")
        kwargs["on_tree"](row)
        return CollectedBatch([row], {pid: loader(actors[pid])}, cursor+1, [pid], [], None, .01,
                              "EXHAUSTED" if cursor+1 == len(order) else "COLLECTED")

    def run_training(self, *, output="run", maximum=3, cursor=None, clock=lambda: 0):
        return run_loop(bundle=self.b, optimizer=self.opt, scheduler=self.sched, engine=self.engine,
            preparation=self.engine, verifier=None, actors=self.actors, structures=self.structures,
            image_loader=lambda a: [Image.new("RGB", (16, 16), "white")], prompts={}, protocol=self.cfg,
            algorithm="ALG-T1", seed=3407, output=self.root/output, binding=self.binding,
            f0_reference={"len_median": 3}, probes=self.probes, deadline_timestamp=1000,
            max_steps=maximum, cursor=cursor, clock=clock)

    def test_three_actual_updates_order_raw_preservation_and_bound_checkpoint(self):
        before = {g: [tensor_sha256(p) for p in ps] for g, ps in self.b.params.items()}
        reference = reference_logprobs(self.b, inputs(), 3).clone()
        with patch("fine_tuning.notebook_rl.trainer.collect_batch", side_effect=self.collect), \
                patch("torch.nn.functional.cross_entropy", side_effect=AssertionError("SFT forbidden")):
            result = self.run_training()
        self.assertEqual(result["status"], "STEP_LIMIT")
        self.assertEqual(result["step"], 3)
        self.assertFalse(result["model_improvement_verified"])
        self.assertEqual(self.engine.pushes, [1, 2, 3])
        self.assertEqual(self.engine.probes, [0, 1, 2, 3])
        self.assertEqual(len(list((self.root/"run/closed_trees").glob("*.json"))), 3)
        rows = [json.loads(line) for line in (self.root/"run/metrics.jsonl").read_text().splitlines()]
        self.assertEqual([r["step"] for r in rows], [1, 2, 3])
        meta = read_metadata(Path(result["last_checkpoint"]), binding=self.binding)
        self.assertEqual(meta["cursor"]["history"], rows)
        self.assertEqual(meta["cursor"]["position"], 3)
        self.assertFalse(meta["cursor"]["sync_pending"])
        for g in ("L", "M", "V"):
            self.assertNotEqual(before[g], [tensor_sha256(p) for p in self.b.params[g]])
        self.assertTrue(torch.equal(reference, reference_logprobs(self.b, inputs(), 3)))
        assert_base_unchanged(self.b)

    def test_resume_next_actual_step_matches_uninterrupted_and_consumes_no_old_ids(self):
        with patch("fine_tuning.notebook_rl.trainer.collect_batch", side_effect=self.collect):
            first = self.run_training(maximum=2)
            checkpoint_dir = Path(first["last_checkpoint"])
            cursor = read_metadata(checkpoint_dir, binding=self.binding)["cursor"]
            live = self.run_training(output="live", maximum=3, cursor=cursor)
            expected = {n: tensor_sha256(p) for n, p in self.b.model.named_parameters()}
            self.b, self.opt, self.sched, step, restored = load_resume(self.model_path, checkpoint_dir,
                self.cfg, binding=self.binding, _test_policy_factory=fresh_tiny)
            self.engine = SyncFixture(version=step)
            resumed = self.run_training(output="resumed", maximum=3, cursor=restored)
        self.assertEqual(live["status"], resumed["status"])
        self.assertEqual(expected, {n: tensor_sha256(p) for n, p in self.b.model.named_parameters()})
        batch = json.loads((self.root/"resumed/batches/000003.json").read_text())
        self.assertEqual(batch["cursor_before"], 2)
        self.assertEqual(batch["cursor_after"], 3)

    def test_budget_before_first_call_saves_f0_without_updates(self):
        with patch("fine_tuning.notebook_rl.trainer.collect_batch") as called:
            result = self.run_training(clock=lambda: 881)
        self.assertEqual(result["status"], "BUDGET_STOP")
        self.assertEqual(result["step"], 0)
        self.assertEqual(result["consumed"], 0)
        self.assertTrue(Path(result["last_checkpoint"]).is_dir())
        called.assert_not_called()

    def test_sync_failure_never_claims_completed_step_and_blocks_resume(self):
        self.engine.fail_push = True
        with patch("fine_tuning.notebook_rl.trainer.collect_batch", side_effect=self.collect):
            result = self.run_training()
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(result["sync_pending"])
        self.assertEqual(result["step"], 1)
        self.assertFalse((self.root/"run/metrics.jsonl").exists())
        self.assertTrue((self.root/"run/batches/000001.update.json").exists())
        meta = read_metadata(Path(result["last_checkpoint"]), binding=self.binding)
        with self.assertRaisesRegex(ValueError, "resume cursor"):
            self.run_training(output="rejected", cursor=meta["cursor"])

    def test_health_stop_saved_and_never_epoch_success(self):
        self.cfg["health"]["short_response_max_fraction"] = .2
        with patch("fine_tuning.notebook_rl.trainer.collect_batch", side_effect=self.collect):
            result = self.run_training()
        self.assertEqual(result["status"], "HEALTH_STOP")
        self.assertEqual(result["stop_reasons"], ["LENGTH_COLLAPSE"])
        self.assertEqual(result["step"], 1)
        self.assertFalse(result["full_epoch_completed"])
        cursor = read_metadata(Path(result["last_checkpoint"]), binding=self.binding)["cursor"]
        self.assertEqual(cursor["stop_reasons"], ["LENGTH_COLLAPSE"])
        with self.assertRaisesRegex(ValueError, "resume cursor"):
            self.run_training(output="health_resume", cursor=cursor)
        self.assertEqual(self.sched.last_epoch, 1)

    def test_budget_after_collection_replays_untrained_batch_on_resume(self):
        calls = 0
        def clock():
            nonlocal calls
            calls += 1
            return 881 if calls >= 3 else 0
        with patch("fine_tuning.notebook_rl.trainer.collect_batch", side_effect=self.collect):
            first = self.run_training(clock=clock)
            cursor = read_metadata(Path(first["last_checkpoint"]), binding=self.binding)["cursor"]
            self.assertEqual(first["status"], "BUDGET_STOP")
            self.assertEqual(first["step"], 0)
            self.assertEqual(cursor["position"], 0)
            self.assertEqual(len(cursor["untrained_problem_ids"]), 1)
            resumed = self.run_training(output="resumed_budget", maximum=4, cursor=cursor)
        self.assertEqual(resumed["status"], "EPOCH_COMPLETED")
        self.assertEqual(resumed["step"], 4)
        batch = json.loads((self.root/"resumed_budget/batches/000002.json").read_text())
        self.assertEqual(batch["trees"][0]["problem_id"], cursor["untrained_problem_ids"][0])
        final = read_metadata(Path(resumed["last_checkpoint"]), binding=self.binding)["cursor"]
        self.assertEqual(final["untrained_problem_ids"], [])

    def test_discarded_unverifiable_results_are_not_hidden_from_health_checks(self):
        def discarded(*args, **kwargs):
            batch = self.collect(*args, **kwargs)
            bad = copy.deepcopy(batch.trees[0])
            for sample in bad.groups[0].samples:
                sample.result = replace(sample.result, status="UNVERIFIABLE", reward_answer=0.)
            batch.discarded.append({"problem_id": "discarded", "reason": "NO_SOLUTION_ADVANTAGE", "tree": to_json(bad)})
            self.assertEqual(batch_statistics(batch)["unverifiable_fraction"], .5)
            return batch
        with patch("fine_tuning.notebook_rl.trainer.collect_batch", side_effect=discarded):
            result = self.run_training()
        self.assertEqual(result["stop_reasons"], ["VERIFIER"])
        self.assertEqual(result["step"], 0)
        self.assertFalse(self.opt.state)

    def test_vision_health_requires_two_real_decreases_and_valid_reference(self):
        self.assertFalse(grounding_disabled([{"grounding_probe": .7}], {"grounding_probe": .8}))
        self.assertTrue(grounding_disabled([{"grounding_probe": .7}, {"grounding_probe": None},
                                           {"grounding_probe": .6}], {"grounding_probe": .8}))
        self.assertFalse(grounding_disabled([{"grounding_probe": .7}, {"grounding_probe": .8}], {"grounding_probe": .8}))
        with self.assertRaises(ValueError):
            grounding_disabled([{"grounding_probe": .7}], {})

    def test_invalid_run_or_resume_rejected_before_output(self):
        with self.assertRaisesRegex(ValueError, "every 12"):
            self.run_training(maximum=12)
        with self.assertRaisesRegex(ValueError, "measured|resume cursor"):
            self.run_training(cursor={})
        self.assertFalse((self.root/"run").exists())


if __name__ == "__main__":
    unittest.main()
