"""Tiny authored-action checks of the reward control; no pretrained-model evidence."""
import copy
import unittest

from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.notebook_learning import NOTEBOOK_REWARD_VERSION
from fine_tuning.notebook_rl.updates import effective_rewards, optimizer_step
from fine_tuning.tests.test_notebook_rl_policy import bundle, config
from fine_tuning.tests.test_notebook_rl_updates import Preparation, tree


def add_note(record):
    note = copy.deepcopy(record.groups[0])
    note.group_id += ":note"
    note.kind = "DIAG"
    for sample in note.samples:
        sample.kind = "DIAG"
        sample.group_id = note.group_id
        sample.sample_id += ":note"
        sample.diag_reward, sample.result = sample.reward, None
    record.groups.append(note)
    record.notebook["rewarded_notebook"] = {"status": "GENERATED", "group_id": note.group_id,
        "reward_version": NOTEBOOK_REWARD_VERSION}


class ControlTests(unittest.TestCase):
    def test_control_zeroes_note_reward_but_keeps_raw_notes_and_real_answer_update(self):
        b, protocol = bundle(), config()
        protocol.update(profile="local-nf4-notebook-rl-v1", notebook_reward=NOTEBOOK_REWARD_VERSION)
        record = tree(b, route="VISION_OFF")
        add_note(record)
        original = copy.deepcopy(record)
        opt, scheduler = make_optimizer(b, protocol)
        result = optimizer_step(b, opt, scheduler, [record], {"p": [object()]}, Preparation(),
            protocol, "ALG-T1", step=0, seed=3407, notebook_reward_enabled=False)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertFalse(result["notebook_reward_enabled"])
        self.assertEqual(result["n_groups_by_kind"]["DIAG"], 1)
        self.assertEqual(result["effective_rewards"][record.groups[-1].group_id], [0., 0.])
        self.assertEqual(result["effective_rewards"][record.groups[0].group_id], [1., 0.])
        self.assertEqual(record, original)
        self.assertGreater(result["grad_norm"]["M"], 0)
        self.assertIsNone(result["grad_norm"]["V"])

    def test_note_only_variance_cannot_force_a_control_update(self):
        b, protocol = bundle(), config()
        protocol.update(profile="local-nf4-notebook-rl-v1", notebook_reward=NOTEBOOK_REWARD_VERSION)
        record = tree(b)
        add_note(record)
        for sample in record.groups[0].samples:
            sample.reward = 0.
        opt, scheduler = make_optimizer(b, protocol)
        with self.assertRaisesRegex(ValueError, "signal-free"):
            optimizer_step(b, opt, scheduler, [record], {"p": [object()]}, Preparation(),
                protocol, "ALG-T1", step=0, seed=3407, notebook_reward_enabled=False)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertFalse(opt.state)

    def test_reward_control_does_not_change_observed_group_sizes(self):
        record = tree(bundle())
        add_note(record)
        actual = effective_rewards(record.groups, "ALG-T1", 3407, 0)
        controlled = effective_rewards(record.groups, "ALG-T1", 3407, 0, notebook_reward_enabled=False)
        self.assertEqual({k: len(v) for k, v in actual.items()}, {k: len(v) for k, v in controlled.items()})
        self.assertEqual(actual[record.groups[0].group_id], controlled[record.groups[0].group_id])


if __name__ == "__main__":
    unittest.main()
