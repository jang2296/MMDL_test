"""Transaction-local pair audits preserve raw-label credit; CPU fixtures only."""
import copy
import json
from typing import Any
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import pair_action_notebook as action, synthesis_credit as credit, updates
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleTokenizer
from fine_tuning.tests.test_notebook_rl_pair_action_integration import pair_action_fixture


class PairBatchValidationTests(unittest.TestCase):
    fixture: Any
    generation_counts: tuple[int, int]

    @classmethod
    def setUpClass(cls):
        owner = unittest.TestCase()
        cls.addClassCleanup(owner.doCleanups)
        cls.fixture = pair_action_fixture(owner, tokenizer=ReversibleTokenizer(), with_notebooks=False)
        fixture = cls.fixture
        observed = "Observed facts:\n" + "\n".join(f"{key} = {value}" for key, value in fixture.item.slots.items()) + "\nSolution:"
        fixture.engine.replies = [[observed], ["PERSISTED", "UNRESOLVED"] * 2, ["PERSISTED", "UNRESOLVED"] * 2]
        with patch.object(synthesis, "audit_episode", wraps=synthesis.audit_episode) as synth, \
                patch.object(episode, "audit_episode", wraps=episode.audit_episode) as source:
            action.append([fixture.tree], fixture.images, fixture.engine, fixture.verifier,
                fixture.prompts, fixture.cfg, actors={fixture.actor.problem_id: fixture.actor},
                should_stop=lambda: sum(group.kind == "DIAG" for group in fixture.tree.groups) >= 2)
        cls.generation_counts = synth.call_count, source.call_count

    def tree(self, count=2):
        tree = copy.deepcopy(self.fixture.tree)
        groups = [group for group in tree.groups if group.kind == "DIAG"][:count]
        tree.groups = [group for group in tree.groups if group.kind != "DIAG"] + groups
        record = tree.notebook["rewarded_notebook"]
        record["group_ids"] = [group.group_id for group in groups]
        record["group_id"] = groups[-1].group_id
        record["target_per_group"] = {key: value for key, value in record["target_per_group"].items()
                                      if key in record["group_ids"]}
        record["notebook_groups_sha256"] = action._group_hash(tree)
        ids = {sample.sample_id for group in groups for sample in group.samples}
        tree.notebook["model_notes"] = [row for row in tree.notebook["model_notes"] if row["sample_id"] in ids]
        return tree

    def score(self, tree):
        return updates._score_notebook_objective(tree, self.fixture.actor, self.fixture.verifier,
                                                 self.fixture.protocol["notebook_objective"])

    def test_scores_receipts_group_credit_and_masks_equal_public_per_sample_scoring(self):
        tree = self.tree()
        record = tree.notebook["rewarded_notebook"]
        expected = {sample.sample_id: credit.score_note(sample.text, sample.finish_reason, tree,
            record["synthesis_audit"], self.fixture.verifier, actor=self.fixture.actor, sample_id=sample.sample_id)
            for group in tree.groups if group.kind == "DIAG" for sample in group.samples}
        receipt = self.score(tree)
        self.assertEqual(json.dumps(receipt["samples"], sort_keys=True), json.dumps(expected, sort_keys=True))
        self.assertEqual(receipt["sha256"], _hash({key: value for key, value in receipt.items() if key != "sha256"}))
        for group in tree.groups:
            if group.kind == "DIAG":
                self.assertEqual(credit.group_credit(group.samples, receipt["samples"], self.fixture.engine.tokenizer),
                                 credit.group_credit(group.samples, expected, self.fixture.engine.tokenizer))

    def test_audit_counts_are_constant_across_four_and_eight_labels_and_replay(self):
        for count in (1, 2):
            tree = self.tree(count)
            with self.subTest(labels=4 * count), \
                    patch.object(synthesis, "audit_episode", wraps=synthesis.audit_episode) as synth, \
                    patch.object(episode, "audit_episode", wraps=episode.audit_episode) as source, \
                    patch.object(action, "validate_update_binding", wraps=action.validate_update_binding) as binding:
                self.assertEqual(len(self.score(tree)["samples"]), count * 4)
                self.assertEqual((synth.call_count, source.call_count, binding.call_count), (1, 2, 1))
            with self.subTest(replay_labels=4 * count), \
                    patch.object(synthesis, "audit_episode", wraps=synthesis.audit_episode) as synth, \
                    patch.object(episode, "audit_episode", wraps=episode.audit_episode) as source:
                receipt = episode.regrade_synthesis(tree, self.fixture.actor, self.fixture.verifier,
                                                    self.fixture.engine.tokenizer)
                self.assertEqual(len(receipt["samples"]), count * 4)
                self.assertEqual((synth.call_count, source.call_count), (1, 2))
        self.assertEqual(self.generation_counts, (4, 6))

    def test_every_new_invocation_rejects_mutated_source_or_sample_bindings(self):
        tree = self.tree(1)
        expected = self.score(tree)
        for mutation in ("source", "ancestry", "operation", "target", "policy", "prompt", "image", "raw", "tokens"):
            changed = copy.deepcopy(tree)
            record = changed.notebook["rewarded_notebook"]
            group = next(group for group in changed.groups if group.kind == "DIAG")
            entry = record["target_per_group"][group.group_id]
            if mutation == "source":
                episode.solution_samples(changed)[0].text += " altered original"
            elif mutation == "ancestry":
                episode.solution_samples(changed)[-1].parent_sample_id = episode.solution_samples(changed)[1].sample_id
            elif mutation == "operation":
                episode.solution_samples(changed)[0].text += '\nTrace: {"states":["q0","q999"]}'
            elif mutation == "target":
                entry["target"]["later"]["locus"] = "changed"
            elif mutation == "policy":
                changed.policy_version += 1
            elif mutation == "prompt":
                entry["messages"][0]["content"].append({"type": "text", "text": "changed"})
            elif mutation == "image":
                entry["images"]["image_sha256"][0] = "0" * 64
            elif mutation == "raw":
                group.samples[0].text = "UNRESOLVED"
            else:
                group.samples[0].completion_token_ids[0] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.score(changed)
            with self.subTest(boundary_mutation=mutation), self.assertRaises(ValueError):
                episode.regrade_synthesis(changed, self.fixture.actor, self.fixture.verifier,
                                         self.fixture.engine.tokenizer)
        self.assertEqual(self.score(tree), expected)

    def test_public_entry_points_keep_fresh_audit_and_membership_checks(self):
        tree = self.tree(1)
        sample = next(group for group in tree.groups if group.kind == "DIAG").samples[0]
        audit = tree.notebook["rewarded_notebook"]["synthesis_audit"]
        with patch.object(synthesis, "audit_episode", wraps=synthesis.audit_episode) as fresh:
            report = synthesis.grade_note(sample.text, sample.finish_reason, tree, audit,
                                          self.fixture.verifier, actor=self.fixture.actor,
                                          contract=synthesis.BOUNDED_VERSION)
            self.assertEqual(fresh.call_count, 1)
            self.assertEqual(report, synthesis._grade_note_audited(sample.text, sample.finish_reason,
                             tree, audit, self.fixture.verifier, contract=synthesis.BOUNDED_VERSION))
        with self.assertRaises(ValueError):
            synthesis.grade_note(sample.text, sample.finish_reason, tree, audit | {"complete_episode": False},
                                 self.fixture.verifier, actor=self.fixture.actor, contract=synthesis.BOUNDED_VERSION)
        with patch.object(action, "validate_update_binding", wraps=action.validate_update_binding) as binding:
            target = action.bound_target(tree, sample.sample_id, sample.text, sample.finish_reason)
            self.assertEqual(binding.call_count, 1)
            self.assertEqual(target, action._bound_target_checked(tree, sample.sample_id, sample.text, sample.finish_reason))
        sample.text = "UNRESOLVED"
        with self.assertRaises(ValueError):
            action.bound_target(tree, sample.sample_id, sample.text, sample.finish_reason)

    def test_batch_keeps_exact_false_replay_contract(self):
        value: Any
        for value in (True, None, 0, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                updates._score_notebook_objective(self.tree(1), self.fixture.actor, self.fixture.verifier,
                    self.fixture.protocol["notebook_objective"], replay_relation_v1=value)

    def test_optimizer_observer_mutations_require_fresh_scoring_and_block_commit(self):
        for mutation in ("target", "scores"):
            tree = self.tree()
            fixture = self.fixture
            objective = fixture.protocol["notebook_objective"]
            updates.bind_notebook_objective(tree, fixture.actor, fixture.verifier, objective)
            receipt = tree.notebook["learning_objective"]
            positive = next(sample_id for sample_id, row in receipt["samples"].items()
                            if any(row["relationship_rewards"].values()))
            events = []

            def progress(event):
                if event["stage"] != "optimizer":
                    return
                events.append(event)
                if mutation == "target":
                    tree.notebook["rewarded_notebook"]["target_plan"][0]["earlier"]["attempt"] = 99
                else:
                    row = receipt["samples"][positive]
                    row["relationship_rewards"] = dict.fromkeys(row["relationship_rewards"], 0)
                    row.update(relationship_later_answer_correct={}, atomic_reward=0., effective_reward=0.)
                    receipt["samples"][positive] = credit._seal({key: value for key, value in row.items()
                                                               if key != "score_sha256"})
                    receipt["sha256"] = _hash({key: value for key, value in receipt.items() if key != "sha256"})
                    updates.notebook_objective_scores(tree, objective)

            with self.subTest(mutation=mutation), patch.object(fixture, "tree", tree):
                values = fixture.optimizer_fixture()
                before = {key: value.detach().clone() for key, value in values[0].model.items()}
                with patch.object(values[1], "step", wraps=values[1].step) as step, \
                        patch.object(credit, "_score_pair_notes", wraps=credit._score_pair_notes) as batch:
                    with self.assertRaises(ValueError):
                        fixture.step(values, on_progress=progress)
                self.assertEqual(len(events), 1)
                self.assertEqual(batch.call_count, 1 if mutation == "target" else 2)
                step.assert_not_called()
                self.assertEqual((values[2].last_epoch, values[1].state), (0, {}))
                self.assertTrue(all(torch.equal(before[key], value.detach()) for key, value in values[0].model.items()))
                self.assertTrue(all(value.grad is None for value in values[0].model.values()))


if __name__ == "__main__":
    unittest.main()
