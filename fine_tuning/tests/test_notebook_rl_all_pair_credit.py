"""Authored all-pair CPU credit checks; no real-model learning claim."""
import copy
import json
from typing import Any
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_pair_probe as probe
from fine_tuning.notebook_rl import notebook_synthesis as synthesis, pair_action_notebook as action, paired_notebook
from fine_tuning.notebook_rl import synthesis_credit as credit
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleTokenizer
from fine_tuning.tests.test_notebook_rl_pair_action_integration import pair_action_fixture


class AllPairCreditTests(unittest.TestCase):
    fixture: Any
    groups: dict[str, Any]
    scores: dict[str, dict]

    @classmethod
    def setUpClass(cls):
        owner = unittest.TestCase()
        cls.addClassCleanup(owner.doCleanups)
        identities = [("OBSERVATION", "start"), ("OBSERVATION", "accepting"), ("TRACE", "states[1:3]")]
        stop_after = 0

        def scripted_targets(samples, reread):
            nonlocal stop_after
            plan = action._plan(samples, reread)[0]
            targets = [next(target for target in plan if target["earlier"]["attempt"] == 1
                            and target["later"]["attempt"] == 3
                            and (target["earlier"]["kind"], target["earlier"]["locus"]) == identity)
                       for identity in identities]
            stop_after = max(plan.index(target) for target in targets) + 1
            return targets

        append = action.append

        def bounded_append(trees, *args, **kwargs):
            return append(trees, *args, **(kwargs | {"should_stop": lambda: sum(
                group.kind == "DIAG" for group in trees[0].groups) >= stop_after}))

        replies = [["PERSISTED", "UNRESOLVED"] * 2, ["PERSISTED"] * 4,
                   [" \tPERSISTED\r\n", "CORRECTED", "UNRESOLVED", ("PERSISTED", "length")]]
        with patch.object(probe, "select_targets", side_effect=scripted_targets), \
                patch.object(action, "append", side_effect=bounded_append):
            cls.fixture = pair_action_fixture(owner, replies, ReversibleTokenizer())
        record = cls.fixture.tree.notebook["rewarded_notebook"]
        cls.groups = {target["earlier"]["locus"]: group for group in cls.fixture.tree.groups if group.kind == "DIAG"
                      and (target := record["target_per_group"][group.group_id]["target"])["earlier"]["attempt"] == 1
                      and target["later"]["attempt"] == 3
                      and (target["earlier"]["kind"], target["earlier"]["locus"]) in identities}
        cls.scores = {sample.sample_id: credit.score_note(sample.text, sample.finish_reason, cls.fixture.tree,
            record["synthesis_audit"], cls.fixture.verifier, actor=cls.fixture.actor, sample_id=sample.sample_id)
            for group in cls.groups.values() for sample in group.samples}

    @staticmethod
    def reseal(row):
        binding = row["pair_action_binding"]
        binding["target_plan_sha256"] = _hash(binding["target_plan"])
        row["pair_action_binding_sha256"] = credit.digest({"binding": binding,
            "solution_sha256": row["solution_sha256"], "transcript_sha256": row["transcript_sha256"]})
        return credit._seal({key: value for key, value in row.items() if key != "score_sha256"})

    def mapped(self, locus):
        group = self.groups[locus]
        return credit.group_credit(group.samples, self.scores, self.fixture.engine.tokenizer)

    def test_stable_correct_pair_earns_zero_and_stable_wrong_pair_keeps_verified_credit(self):
        samples = episode.solution_samples(self.fixture.tree)
        for locus, value, expected in (("accepting", "q3", [0, 0, 0, 0]), ("start", "q999", [1, 0, 1, 0])):
            with self.subTest(locus=locus):
                self.assertTrue(all(f"{locus} = {value}" in samples[index].text for index in (0, 2)))
                group = self.groups[locus]
                rows = [self.scores[sample.sample_id] for sample in group.samples]
                target = rows[0]["pair_action_binding"]["target_group"]["target"]
                origin = credit._origin(target)
                self.assertEqual(target["raw_values"], [])
                self.assertEqual([row["relationship_rewards"].get(origin, 0) for row in rows], expected)
                self.assertTrue(all(row["strict_reward"] == 0 and row["fact_rewards"] == {} for row in rows))
                mapped = self.mapped(locus)
                self.assertEqual(credit.has_signal(group.samples, self.scores), locus == "start")
                if locus == "accepting":
                    self.assertTrue(all(row["selected_lesson_indices"] == {} for row in rows))
                    self.assertTrue(all(not any(row["partial_token_advantages"]) for row in mapped.values()))

    def test_trace_uses_existing_native_truth_and_original_label_token_masks(self):
        group = self.groups["states[1:3]"]
        rows = [self.scores[sample.sample_id] for sample in group.samples]
        target = rows[0]["pair_action_binding"]["target_group"]["target"]
        origin = credit._origin(target)
        self.assertEqual((target["channel"], target["earlier"]["kind"], target["raw_values"]), ("NATIVE", "TRACE", []))
        self.assertEqual([row["relationship_rewards"][origin] for row in rows], [1, 0, 0, 0])
        self.assertEqual(rows[0]["relationship_credit_types"][origin], credit.NATIVE_ERROR_RELATIONSHIP)
        mapped = self.mapped("states[1:3]")
        sample = group.samples[0]
        entry = mapped[sample.sample_id]["relationships"][0]
        self.assertEqual((entry["group_mean"], entry["advantage"]), (.25, .75))
        self.assertEqual(entry["token_indices"], list(range(2, 11)))
        vector = mapped[sample.sample_id]["relationship_token_advantages"]
        self.assertTrue(all(value == 0 for index, value in enumerate(vector) if index not in range(2, 11)))
        self.assertAlmostEqual(sum(vector), credit.RELATIONSHIP_WEIGHT * .75 / len(rows[0]["original_issue_ids"]))
        self.assertEqual(mapped[group.samples[-1].sample_id]["mapping_status"], "NO_MAPPED_UNITS")
        self.assertEqual(credit.group_credit(group.samples, json.loads(json.dumps(self.scores)),
                                            self.fixture.engine.tokenizer), mapped)
        with self.assertRaisesRegex(ValueError, "original complete"):
            credit.group_credit(group.samples[:3], self.scores, self.fixture.engine.tokenizer)

    def test_resealed_cross_kind_locus_and_old_value_filter_receipts_are_rejected(self):
        sample = self.groups["start"].samples[0]
        for mutation in ("kind", "locus", "raw_values", "version", "span", "tokens", "source"):
            row = copy.deepcopy(self.scores[sample.sample_id])
            binding = row["pair_action_binding"]
            target = binding["target_group"]["target"]
            if mutation == "kind":
                target["later"]["kind"] = "TRACE"
            elif mutation == "locus":
                target["later"]["locus"] = "accepting"
            elif mutation == "raw_values":
                target["raw_values"] = ["q999", "q0"]
            elif mutation == "version":
                row["notebook_generation_policy"] = "verified-pair-action-notebook-v1"
            elif mutation == "span":
                row["relationship_spans"][credit._origin(target)] = [0, 1]
            elif mutation == "tokens":
                binding["completion_token_ids_sha256"] = "0" * 64
            else:
                row["source_sha256"] = "0" * 64
            target["pair_id"] = paired_notebook._pair_id(target["earlier"], target["later"])
            binding["target_plan"][binding["target_group"]["target_index"] - 1] = copy.deepcopy(target)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                credit.validate_score(self.reseal(row), sample)

    def test_fresh_scoring_rejects_changed_source_ancestry_and_trace_operation(self):
        original = self.fixture.tree
        for mutation in ("source", "ancestry", "operation"):
            tree = copy.deepcopy(original)
            samples = episode.solution_samples(tree)
            if mutation == "source":
                samples[0].text += "\nforged source"
            elif mutation == "ancestry":
                samples[2].parent_sample_id = samples[1].sample_id
            else:
                record = tree.notebook["rewarded_notebook"]
                gid = self.groups["states[1:3]"].group_id
                entry = record["target_per_group"][gid]
                entry["target"]["later"]["locus"] = "states[2:4]"
                entry["target"]["pair_id"] = paired_notebook._pair_id(entry["target"]["earlier"], entry["target"]["later"])
                record["target_plan"][entry["target_index"] - 1] = copy.deepcopy(entry["target"])
                record["target_plan_sha256"] = _hash(record["target_plan"])
            sample = self.groups["states[1:3]"].samples[0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit = synthesis.audit_episode(tree, self.fixture.actor, self.fixture.verifier,
                                                contract=synthesis.BOUNDED_VERSION)
                credit.score_note(sample.text, sample.finish_reason, tree, audit, self.fixture.verifier,
                                  actor=self.fixture.actor, sample_id=sample.sample_id)


if __name__ == "__main__":
    unittest.main()
