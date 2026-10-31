"""Actual CPU repair collection with scripted trainer boundaries; no model training."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from PIL import Image

from fine_tuning.notebook_rl import trainer
from fine_tuning.notebook_rl.batches import collect_batch
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.repair_contract import ORIGIN_VERSION, public_origin_inventory
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import ProblemTree, Sample, from_json, to_json
from fine_tuning.tests import test_notebook_rl_repair_notebook as notebook_fixture
from fine_tuning.tests.test_notebook_rl_reasoning_trace import fixture_trace, response
from fine_tuning.tests.test_notebook_rl_repair_credit import joint
from fine_tuning.tests.test_notebook_rl_repair_training import OriginalEngine


class RepairFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = notebook_fixture.RepairNotebookTests()
        self.fixture.setUp()
        self.actor, self.verifier = self.fixture.actor, self.fixture.verifier
        self.protocol = build_protocol(repair_notebook=True, context_limit=24576)
        self.prompts = local_prompts(self.protocol)
        self.actors = {self.actor.problem_id: self.actor}
        self.structures = {self.actor.problem_id: hashlib.sha256(self.actor.problem_id.encode()).hexdigest()}
        self.events: list[dict] = []
        self.loaded: list = []

    def loader(self, actor):
        images = [Image.new("RGB", (16, 16), "white") for _ in actor.image_paths]
        for image in images:
            self.addCleanup(image.close)
        self.loaded.extend(images)
        return images

    def online_engine(self, *, malformed=False, visual=False):
        trace = fixture_trace(self.fixture.problem)
        raw = response(self.fixture.problem, {**trace, "cost": trace["cost"] - 1})
        raw = raw.rsplit("Final answer:", 1)[0] + "Final answer: 999"
        if visual:
            lines = raw.splitlines()
            index = next(index for index, line in enumerate(lines) if " = " in line)
            lines[index] = lines[index].split(" = ")[0] + " = 999"
            raw = "\n".join(lines)
        group_id = f"{self.actor.problem_id}:v0:T1:{_hash('root')[:16]}"
        parent_id = min((f"{group_id}:{index}" for index in range(4)),
                        key=lambda value: hashlib.sha256(f"{self.actor.problem_id}:0:{value}".encode()).digest())
        parent = Sample(parent_id, self.actor.problem_id, group_id, "T1", 1, None,
                        [1], [2], [-.1], "stop", raw, 0, 0)
        inventory = public_origin_inventory([parent], contract_version=ORIGIN_VERSION,
                                            response_contracts={parent_id: None}, parent_sample_id=parent_id)
        origin = next(row.ref_id for row in inventory.origins if row.rule == "COST_SUM")
        positive = joint(self.fixture.wrong, origin, proposed={"field": "cost", "value": trace["cost"]})
        unknown = joint(self.fixture.wrong, origin, intent="UNKNOWN")
        replies = [[raw] * 4, [positive, "ErrorCheck: {" if malformed else unknown]]
        empty = 'ErrorCheck: {"version":"repair-check-v1","claims":[],"uncertain_refs":[]}\nRevisedSolution:\n'
        replies.append([empty + self.fixture.wrong] * 2)
        return OriginalEngine(replies)

    def collect(self, engine, **options):
        return collect_batch(self.actors, self.structures, list(self.actors), 0, self.loader,
            engine, self.verifier, self.prompts, self.protocol, "ALG-NB3", should_stop=lambda: False,
            on_raw=options.pop("on_raw", self.events.append), **options)


class RepairBatchTests(RepairFixture):
    def test_all_wrong_online_signal_survives_legacy_gate_and_is_saved_after_append(self) -> None:
        closed = []
        engine = self.online_engine()

        def preserve(tree):
            self.assertEqual(self.events[-1]["purpose"], "REPAIR_FINAL_CLOSED")
            self.assertIn("rewarded_notebook", tree.notebook)
            closed.append(to_json(tree))
            tree.groups.clear()

        with patch("fine_tuning.notebook_rl.batches.has_learning_signal", side_effect=AssertionError("legacy gate")):
            batch = self.collect(engine, on_tree=preserve)
        self.assertEqual((batch.status, batch.cursor, len(batch.trees)), ("EXHAUSTED", 1, 1))
        self.assertEqual([len(group.samples) for group in batch.trees[0].groups], [4, 2, 2])
        self.assertEqual(closed, [to_json(batch.trees[0])])
        assert batch.diagnosis is not None
        receipt = batch.diagnosis["repair_preparation"][self.actor.problem_id]
        self.assertTrue(receipt["admitted"] and receipt["has_signal"])
        self.assertEqual(receipt["signal_components"], {"answer": False, "online": True, "final": False})
        self.assertEqual(batch.trees[0].notebook["rewarded_notebook"]["status"], "NO_CORRECT_ANSWER")
        self.assertEqual([request.n for request in engine.requests], [4, 2, 2])

    def test_complete_malformed_sibling_remains_intact_and_valid_sibling_continues(self) -> None:
        batch = self.collect(self.online_engine(malformed=True, visual=True))
        self.assertEqual(len(batch.trees), 1)
        group = batch.trees[0].groups[1]
        self.assertEqual((group.usable, group.reason, len(group.samples)), (True, None, 2))
        self.assertIsNone(group.samples[1].result)
        self.assertEqual(trainer.batch_statistics(batch)["missing_result_fraction"], 1 / 8)
        third = batch.trees[0].groups[2]
        self.assertEqual((third.kind, len(third.samples)), ("T3", 2))
        self.assertEqual(third.samples[0].parent_sample_id, group.samples[0].sample_id)
        assert batch.diagnosis is not None
        prepared = batch.diagnosis["repair_preparation"][self.actor.problem_id]
        self.assertTrue(prepared["admitted"])
        self.assertEqual(prepared["groups"][1]["original_group_size"], 2)
        self.assertFalse(any(prepared["groups"][1]["samples"][1]["applied_online_coefficients"]))

    def test_signal_free_discard_preserves_complete_final_n4_and_raw_before_tree(self) -> None:
        engine = OriginalEngine([[self.fixture.right] * 4, ["{"] * 4])
        closed = []
        batch = self.collect(engine, on_tree=lambda tree: closed.append(to_json(tree)))
        self.assertFalse(batch.trees)
        row = batch.discarded[0]
        self.assertEqual((row["reason"], row["admission_reasons"]), ("NO_REPAIR_POLICY_GRADIENT", []))
        tree = from_json(ProblemTree, row["tree"])
        self.assertEqual((tree.groups[-1].usable, len(tree.groups[-1].samples)), (True, 4))
        self.assertEqual(closed, [row["tree"]])
        self.assertEqual(sum(event["purpose"] == "REPAIR_FINAL_COMPLETION" for event in self.events), 4)
        self.assertEqual(self.events[-1]["purpose"], "REPAIR_FINAL_CLOSED")
        with self.assertRaises(ValueError):
            self.loaded[0].getpixel((0, 0))

    def test_unresolved_discard_preserves_fresh_admission_reasons(self) -> None:
        raw = "Solution:\nNo structured observations.\nFinal answer: " + self.fixture.problem.verifier.gold
        batch = self.collect(OriginalEngine([[raw] * 4, ["{"] * 4]))
        self.assertEqual(batch.discarded[0]["reason"], "REPAIR_NOT_ADMITTED")
        self.assertIn("UNRESOLVED_ROUTING", batch.discarded[0]["admission_reasons"])

    def test_protocol_callback_and_seed_fail_before_loading_without_fallback(self) -> None:
        for mutation in ("callback", "seed", "algorithm", "profile", "prompt", "attempts"):
            protocol, prompts = copy.deepcopy(self.protocol), copy.deepcopy(self.prompts)
            options: dict[str, Any] = {"on_raw": self.events.append, "seed": 3407}
            algorithm = "ALG-NB3"
            if mutation == "callback":
                options["on_raw"] = None
            elif mutation == "seed":
                options["seed"] = 1
            elif mutation == "algorithm":
                algorithm = "ALG-T1"
            elif mutation == "profile":
                protocol["protocol_id"] = "old"
            elif mutation == "prompt":
                prompts["retry"] = "modified"
            else:
                protocol["batch"]["max_attempted_problems_per_step"] = 6
            loader = Mock(side_effect=AssertionError("loaded before validation"))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                collect_batch(self.actors, self.structures, list(self.actors), 0, loader,
                    self.online_engine(), self.verifier, prompts, protocol, algorithm,
                    should_stop=lambda: False, **options)
            loader.assert_not_called()

    def test_callback_failure_and_generation_interrupt_keep_first_raw_and_close_images(self) -> None:
        for failure in (OSError("writer failed"), KeyboardInterrupt("later generation interrupted")):
            engine = self.online_engine()
            original = engine.sample
            events = []

            def write(event):
                events.append(event)
                if isinstance(failure, OSError):
                    raise failure

            def sample(requests, *, on_completion):
                def completed(*args):
                    on_completion(*args)
                    raise failure
                return original(requests, on_completion=completed)

            with patch.object(engine, "sample", side_effect=sample), self.assertRaises(type(failure)):
                self.collect(engine, on_raw=write)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["purpose"], "NATIVE_SOLUTION_COMPLETION")
            with self.assertRaises(ValueError):
                self.loaded[-1].getpixel((0, 0))


class RepairTrainerTests(RepairFixture):
    def setUp(self) -> None:
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "run"
        self.bundle = SimpleNamespace(inventory={"arm": "MOD-RV"}, params={"V": [object()]})
        self.optimizer = SimpleNamespace(zero_grad=Mock())
        self.scheduler = SimpleNamespace(last_epoch=0)
        self.engine = self.online_engine()
        self.saved: list[dict] = []
        self.updated: list[dict] = []

    def run_training(self, *, status="UPDATED_UNSYNCHRONIZED", **changes):
        def sync(bundle, preparation, engine, probes, protocol, step, *, push):
            engine.policy_version = step
            return [{"scripted_cpu_sync": True, "step": step}]

        def save(bundle, optimizer, scheduler, step, cursor, rng, destination, **kwargs):
            self.saved.append(copy.deepcopy(cursor))

        def update(bundle, optimizer, scheduler, trees, images, preparation, protocol, algorithm, **kwargs):
            self.updated.append(kwargs)
            self.assertIs(kwargs["objective_verifier"], self.verifier)
            self.assertEqual(kwargs["observation_actors"], self.actors)
            self.assertFalse(kwargs["vision_disabled"])
            if status != "UPDATED_UNSYNCHRONIZED":
                return {"status": status, "step_before": 0, "step_after": 0}
            self.scheduler.last_epoch += 1
            from fine_tuning.notebook_rl.repair_training import prepare_tree_credit
            receipts = {tree.problem_id: prepare_tree_credit(tree, self.actors[tree.problem_id], self.verifier,
                preparation.tokenizer, preparation=preparation, images=images[tree.problem_id], master_seed=3407)
                for tree in trees}
            groups = [group for tree in trees for group in tree.groups]
            return {"status": status, "step_after": 1, "n_groups_by_kind": {kind: sum(group.kind == kind for group in groups)
                for kind in ("T1", "T2", "T3", "DIAG")}, "repair_preparation": receipts,
                "effective_rewards": {group.group_id: [row.reward for row in group.samples] for group in groups},
                "retained_repair_signal": {"policy_gradient_present": True, "scripted_cpu_boundary": True},
                "repair_aggregate_gradient": {"scripted_cpu_boundary": True, "actual_gradient_verified": False},
                "kl_mean": 0., "masked_fraction": 0., "mean_abs_lp_diff": 0., "grad_norm": dict.fromkeys(("L", "M", "V"), 0.),
                "routes": {tree.problem_id: tree.route == "VISION_ON" for tree in trees}, "seconds": dict.fromkeys(("old", "ref", "train"), .01)}

        options = dict(bundle=self.bundle, optimizer=self.optimizer, scheduler=self.scheduler, engine=self.engine,
            preparation=self.engine, verifier=self.verifier, actors=list(self.actors.values()), structures=self.structures,
            image_loader=self.loader, prompts=self.prompts, protocol=self.protocol, algorithm="ALG-NB3", seed=3407,
            output=self.output, binding={}, f0_reference={"len_median": 100}, probes=[], deadline_timestamp=1000,
            max_steps=1, clock=lambda: 0)
        options.update(changes)
        with patch.object(trainer, "synchronize", side_effect=sync), patch.object(trainer, "optimizer_step", side_effect=update), \
                patch.object(trainer.checkpoint, "save", side_effect=save), patch.object(trainer.checkpoint, "capture_rng", return_value={}), \
                patch.object(trainer, "assert_base_unchanged"):
            return trainer.run_loop(**options)

    def test_raw_write_once_and_updater_receipt_metrics_are_preserved(self) -> None:
        original = self.verifier.grade

        def grade(*args, **kwargs):
            paths = list((self.output / "raw").glob("*.json"))
            self.assertTrue(paths)
            self.assertEqual(json.loads(paths[0].read_text())["event"], "REPAIR_RAW")
            return original(*args, **kwargs)

        with patch.object(self.verifier, "grade", side_effect=grade):
            result = self.run_training()
        self.assertEqual((result["status"], result["step"]), ("EPOCH_COMPLETED", 1))
        raw = [json.loads(path.read_text()) for path in sorted((self.output / "raw").glob("*.json"))]
        self.assertEqual(len(raw), 11)
        self.assertEqual(sum(row["record"]["purpose"] == "RETRY_FRESH_OBSERVATION" for row in raw), 2)
        self.assertTrue(all(row["event"] == "REPAIR_RAW" for row in raw))
        self.assertEqual(raw[0]["record"]["purpose"], "NATIVE_SOLUTION_COMPLETION")
        update = json.loads((self.output / "batches/000001.update.json").read_text())
        self.assertTrue(update["retained_repair_signal"]["policy_gradient_present"])
        record = json.loads((self.output / "metrics.jsonl").read_text())
        self.assertEqual(record["degenerate_fraction"], 1.)
        self.assertTrue(record["answer_degeneracy_does_not_imply_zero_repair_signal"])
        self.assertEqual(record["repair_signal_components"], {"answer": False, "online": True, "final": False})
        self.assertFalse(result["vision_health_disabled"])

    def test_no_update_statuses_do_not_increment_step_or_emit_update_metrics(self) -> None:
        for index, status in enumerate(("NO_ACTIONS", "NO_LEARNING_SIGNAL", "NO_ADMITTED_REPAIR_TREES", "NO_MATCHED_ACTIONS")):
            self.engine = self.online_engine()
            result = self.run_training(status=status, output=self.output.parent / f"empty{index}")
            self.assertEqual(result["step"], 0)
            self.assertEqual(self.scheduler.last_epoch, 0)
            self.assertFalse((self.output.parent / f"empty{index}/metrics.jsonl").exists())

    def test_wrong_profile_arm_and_unbounded_run_fail_before_artifacts(self) -> None:
        with self.assertRaises(ValueError):
            self.run_training(max_steps=6)
        self.assertFalse(self.output.exists())
        self.bundle.inventory["arm"] = "MOD-LM"
        with self.assertRaises(ValueError):
            self.run_training()
        self.assertFalse(self.output.exists())

    def test_vision_health_request_stops_without_disabling_modules_and_resume_rejects_override(self) -> None:
        with patch.object(trainer, "grounding_disabled", side_effect=[False, True]):
            result = self.run_training()
        self.assertEqual((result["status"], result["stop_reasons"]), ("HEALTH_STOP", ["VISION_HEALTH_REVIEW_REQUIRED"]))
        self.assertFalse(result["vision_health_disabled"])
        cursor = copy.deepcopy(self.saved[-1])
        cursor.update(vision_health_disabled=True, stop_reasons=[])
        with self.assertRaisesRegex(ValueError, "disabled Vision"):
            self.run_training(cursor=cursor, max_steps=2, output=self.output.parent / "resume")
        self.assertFalse((self.output.parent / "resume").exists())

    def test_partial_completion_is_persisted_when_later_generation_interrupts(self) -> None:
        original = self.engine.sample

        def sample(requests, *, on_completion):
            def complete(*args):
                on_completion(*args)
                raise KeyboardInterrupt("after durable first completion")
            return original(requests, on_completion=complete)

        with patch.object(self.engine, "sample", side_effect=sample):
            result = self.run_training()
        self.assertEqual((result["status"], result["step"]), ("FAILED", 0))
        paths = list((self.output / "raw").glob("*.json"))
        self.assertEqual(len(paths), 1)
        self.assertEqual(json.loads(paths[0].read_text())["record"]["index"], 0)
        self.assertFalse(self.updated)


if __name__ == "__main__":
    unittest.main()
