"""Real tiny CPU updates with empty notebook credit; never a notebook-learning claim."""
import unittest
from unittest.mock import patch

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl import synthesis_credit, updates
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.tests import test_notebook_rl_synthesis_credit_updates as fixtures


def hashes(bundle):
    return {group: [tensor_sha256(parameter) for parameter in parameters]
            for group, parameters in bundle.params.items()}


class EmptySynthesisCreditTests(unittest.TestCase):
    def no_recovery(self, *, answer_signal=True):
        fixture = fixtures.SynthesisCreditUpdateTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for index, sample in enumerate(solution_samples(fixture.tree)):
            sample.text = fixtures.response(fixture.fixture.item.slots, wrong=not (answer_signal and index == 1))
            sample.completion_token_ids = fixture.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.result = fixture.verifier.grade(fixture.tree.problem_id, sample.text, "stop", "obs")
            sample.reward = float(sample.result.status == "CORRECT")
        fixture.fixture.append()
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier, fixture.objective)
        self.assertEqual(fixture.tree.notebook["rewarded_notebook"]["status"], "NO_VERIFIED_RECOVERY")
        self.assertFalse(any(group.kind == "DIAG" for group in fixture.tree.groups))
        self.assertEqual(fixture.tree.notebook["learning_objective"]["samples"], {})
        return fixture

    def test_no_diag_answer_policy_gradient_commits_without_inventing_credit(self):
        fixture = self.no_recovery()
        values = fixture.optimizer_fixture()
        bundle, optimizer, scheduler = values[:3]
        before = hashes(bundle)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = fixture.step(values, require_retained_learning_signal=True)
        step.assert_called_once()
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual((result["step_before"], result["step_after"], scheduler.last_epoch), (0, 1, 1))
        self.assertTrue(optimizer.state)
        self.assertTrue(result["retained_learning_signal"]["policy_gradient_present"])
        self.assertNotEqual(before["L"], hashes(bundle)["L"])
        self.assertEqual(before["V"], hashes(bundle)["V"])
        self.assertEqual(result["n_groups_by_kind"]["DIAG"], 0)
        self.assertEqual(result["notebook_backward"], [])
        self.assertNotIn("synthesis_token_credit", result)
        self.assertEqual(result["synthesis_regrade_receipts"][fixture.tree.problem_id]["samples"], {})
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))

    def test_absent_original_credit_still_rejects_nonempty_fresh_credit_before_step(self):
        fixture = fixtures.SynthesisCreditUpdateTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.prepare(solution_signal=True)
        values = fixture.optimizer_fixture()
        bundle, optimizer, scheduler = values[:3]
        before = hashes(bundle)
        original = synthesis_credit.group_credit
        returned = []

        def omit_original(*args, **kwargs):
            fresh = original(*args, **kwargs)
            returned.append(fresh)
            return {} if len(returned) == 1 else fresh

        with patch.object(synthesis_credit, "group_credit", side_effect=omit_original), \
                patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(ValueError, "synthesis token credit changed before optimizer"):
                fixture.step(values, require_retained_learning_signal=True)
        self.assertEqual(len(returned), 2)
        self.assertTrue(returned[1])
        step.assert_not_called()
        self.assertEqual((scheduler.last_epoch, optimizer.state), (0, {}))
        self.assertEqual(before, hashes(bundle))
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))

    def test_no_diag_all_zero_answers_remain_signal_free_and_cannot_force_update(self):
        fixture = self.no_recovery(answer_signal=False)
        selected, unresolved, signal_free = updates.partition_learning_trees(
            [fixture.tree], include_notebooks=True, notebook_objective=fixture.objective)
        self.assertEqual((selected, unresolved, signal_free), ([], [], [fixture.tree.problem_id]))
        values = fixture.optimizer_fixture()
        bundle, optimizer, scheduler = values[:3]
        before = hashes(bundle)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(ValueError, "dynamic sampling must exclude signal-free"):
                fixture.step(values, require_retained_learning_signal=True)
        step.assert_not_called()
        self.assertEqual((scheduler.last_epoch, optimizer.state), (0, {}))
        self.assertEqual(before, hashes(bundle))
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))


if __name__ == "__main__":
    unittest.main()
