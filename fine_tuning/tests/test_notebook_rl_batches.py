"""Synthetic collection tests. They are not GPU learning or data approval evidence."""
import hashlib
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl.batches import collect_batch, family_order
from fine_tuning.notebook_rl.schema import Group, ProblemTree
from fine_tuning.tests.test_notebook_rl_policy import config
from fine_tuning.tests.test_notebook_rl_schema import records


class BatchTests(unittest.TestCase):
    def setup_data(self, n):
        actor = records()[0]
        actors = {str(i): replace(actor, problem_id=str(i), family="table" if i % 2 else "graph") for i in range(n)}
        structures = {key: hashlib.sha256(key.encode()).hexdigest() for key in actors}
        protocol = config()
        protocol["batch"] = {"problems_per_step": 2, "max_attempted_problems_per_step": 6}
        return actors, structures, list(actors), protocol

    def run_batch(self, n=8, *, callback=None, algorithm="ALG-T1", structures=None, stop=lambda: False):
        actors, default_structures, order, protocol = self.setup_data(n)
        return collect_batch(actors, structures or default_structures, order, 0,
            lambda actor: [Image.new("RGB", (32, 16), (0, 127, 255))], SimpleNamespace(policy_version=0),
            None, {}, protocol, algorithm, should_stop=stop)

    def scripted(self, actor, images, *args, **kwargs):
        source = records()[5]
        samples = [replace(source, sample_id=actor.problem_id+str(i), problem_id=actor.problem_id,
                           group_id=actor.problem_id, kind="T1", turn=1, reward=float(i)) for i in range(2)]
        group = Group(actor.problem_id, "T1", actor.problem_id, "a"*64, samples, True, None)
        return ProblemTree(actor.problem_id, 0, [group], "VISION_OFF", {}, {})

    def test_family_shuffle_stable_and_round_robin(self):
        actors, _, _, _ = self.setup_data(8)
        first = family_order(list(actors.values()), 3407)
        self.assertEqual(first, family_order(list(reversed(list(actors.values()))), 3407))
        self.assertEqual(len(first), 8)
        self.assertTrue(all(actors[first[i]].family != actors[first[i+1]].family for i in range(7)))

    def test_bounded_skips_never_repeat_same_epoch_and_keep_raw_discard(self):
        def no_signal(*args, **kwargs):
            row = self.scripted(*args, **kwargs)
            for sample in row.groups[0].samples:
                sample.reward = 0.
            return row
        with patch("fine_tuning.notebook_rl.batches.build_tree", side_effect=no_signal):
            batch = self.run_batch()
        self.assertEqual(batch.status, "ATTEMPT_LIMIT")
        self.assertEqual(batch.cursor, 6)
        self.assertEqual(batch.attempted, [str(i) for i in range(6)])
        self.assertEqual(len(batch.discarded), 6)
        self.assertTrue(all("tree" in row for row in batch.discarded))
        self.assertFalse(batch.trees)

    def test_structure_uniqueness_and_partial_exhaustion(self):
        # Native generators use the documented 16-hex structure identity, not a file SHA.
        structures = {str(i): "f"*16 for i in range(3)}
        with patch("fine_tuning.notebook_rl.batches.build_tree", side_effect=self.scripted):
            batch = self.run_batch(n=3, structures=structures)
        self.assertEqual(batch.status, "EXHAUSTED")
        self.assertEqual(len(batch.trees), 1)
        self.assertEqual([r["reason"] for r in batch.discarded], ["STRUCTURE_ALREADY_IN_STEP"]*2)
        for images in batch.images.values():
            for image in images:
                image.close()

    def test_blind_is_gray_before_sampling_not_only_before_gradient(self):
        observed = []
        def blind(actor, images, *args, **kwargs):
            observed.append((images[0].size, images[0].getextrema()))
            return self.scripted(actor, images, *args)
        with patch("fine_tuning.notebook_rl.batches.build_tree", side_effect=blind):
            batch = self.run_batch(algorithm="CTL-BLIND")
        self.assertEqual(observed, [((32, 16), ((128, 128),)*3)]*2)
        self.assertTrue(all(t.notebook["input_condition"] == "BLIND_SAMPLED" for t in batch.trees))
        for images in batch.images.values():
            for image in images:
                image.close()

    def test_stop_before_request_does_not_consume_cursor_or_make_calls(self):
        with patch("fine_tuning.notebook_rl.batches.build_tree") as called:
            batch = self.run_batch(stop=lambda: True)
        self.assertEqual(batch.status, "BUDGET_STOP")
        self.assertEqual(batch.cursor, 0)
        called.assert_not_called()

    def test_final_notebook_budget_stop_propagates_without_losing_closed_tree(self):
        for signal in (False, True):
            def stopped(*args, **kwargs):
                tree = self.scripted(*args, **kwargs)
                tree.notebook["final_notebook"] = {"status": "BUDGET_STOP", "used_for_loss": False}
                if not signal:
                    for sample in tree.groups[0].samples:
                        sample.reward = 0.
                return tree
            with patch("fine_tuning.notebook_rl.batches.build_tree", side_effect=stopped):
                batch = self.run_batch()
            self.assertEqual(batch.status, "BUDGET_STOP")
            self.assertEqual(batch.cursor, 1)
            if signal:
                self.assertEqual(batch.trees[0].notebook["final_notebook"]["status"], "BUDGET_STOP")
                for values in batch.images.values():
                    for image in values:
                        image.close()
            else:
                self.assertIn('"final_notebook"', batch.discarded[0]["tree"])


if __name__ == "__main__":
    unittest.main()
