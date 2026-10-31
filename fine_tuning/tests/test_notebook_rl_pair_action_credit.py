"""Actual fixture label tokens test partial credit, not pretrained model learning."""
import copy
import json
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import pair_action_notebook as action, synthesis_credit as credit
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests import test_notebook_rl_paired_credit as previous_credit
from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleTokenizer
from fine_tuning.tests.test_notebook_rl_pair_action_integration import pair_action_fixture, scripted_groups


class PairActionCreditTests(unittest.TestCase):
    def fixture(self, replies=None, *, characters=True):
        return pair_action_fixture(self, replies, ReversibleTokenizer() if characters else None)

    @staticmethod
    def groups(fixture):
        return scripted_groups(fixture)

    def scores(self, fixture, group):
        audit = fixture.tree.notebook["rewarded_notebook"]["synthesis_audit"]
        scores = {sample.sample_id: credit.score_note(sample.text, sample.finish_reason, fixture.tree,
            audit, fixture.verifier, actor=fixture.actor, sample_id=sample.sample_id) for sample in group.samples}
        return scores, credit.group_credit(group.samples, scores, fixture.engine.tokenizer)

    @staticmethod
    def reseal(row):
        binding = row["pair_action_binding"]
        row["pair_action_binding_sha256"] = credit.digest({"binding": binding,
            "solution_sha256": row["solution_sha256"], "transcript_sha256": row["transcript_sha256"]})
        return credit._seal({key: value for key, value in row.items() if key != "score_sha256"})

    def test_native_and_output_use_existing_truth_and_only_original_generated_label_tokens(self):
        fixture = self.fixture()
        for group, expected_type in zip(self.groups(fixture)[1:],
                (credit.NATIVE_ERROR_RELATIONSHIP, credit.OUTPUT_FINALIZATION), strict=True):
            with patch.object(synthesis, "_lesson", wraps=synthesis._lesson) as verify:
                scores, mapped = self.scores(fixture, group)
            target = fixture.tree.notebook["rewarded_notebook"]["target_per_group"][group.group_id]["target"]
            origin = credit._origin(target)
            first = group.samples[0]
            row, output = scores[first.sample_id], mapped[first.sample_id]
            self.assertEqual(verify.call_args_list[0].args[0], {"rule": target["earlier"]["rule"],
                "earlier": target["earlier"], "later": target["later"], "relation": first.text})
            self.assertEqual(row["strict_reward"], 0.)
            self.assertEqual(row["pair_action_binding"]["generation_contract"], action.contract())
            self.assertFalse(row["json_complete"])
            self.assertEqual(row["fact_rewards"], {})
            self.assertEqual(row["relationship_credit_types"][origin], expected_type)
            self.assertEqual(row["relationship_rewards"][origin], 1)
            self.assertEqual(row["atomic_reward"], credit.RELATIONSHIP_WEIGHT / len(row["original_issue_ids"]))
            entry = output["relationships"][0]
            self.assertEqual((entry["group_mean"], entry["advantage"]), (.5, .5))
            self.assertEqual(entry["token_indices"], list(range(len(first.text))))
            self.assertEqual(output["relationship_token_advantages"][-1], 0.)
            self.assertAlmostEqual(sum(output["relationship_token_advantages"]),
                                   credit.RELATIONSHIP_WEIGHT * .5 / len(row["original_issue_ids"]))
            self.assertEqual(output["full_group_size"], 4)
            self.assertTrue(credit.has_signal(group.samples, scores))
            if expected_type == credit.OUTPUT_FINALIZATION:
                self.assertEqual(row["relationship_later_answer_correct"], {origin: False})
            else:
                self.assertEqual(row["relationship_later_answer_correct"], {})
            self.assertEqual(credit.group_credit(group.samples, json.loads(json.dumps(scores)),
                                               fixture.engine.tokenizer), mapped)

    def test_uniform_verified_and_uniform_unresolved_groups_have_zero_policy_credit(self):
        for replies in ([['UNRESOLVED'] * 4, ['PERSISTED'] * 4, ['FINALIZED'] * 4],
                        [['UNRESOLVED'] * 4] * 3):
            fixture = self.fixture(replies)
            for group in self.groups(fixture):
                scores, output = self.scores(fixture, group)
                self.assertFalse(credit.has_signal(group.samples, scores))
                self.assertTrue(all(value == 0 for row in output.values() for value in row["partial_token_advantages"]))

    def test_unresolved_is_mapped_sampled_abstention_with_zero_reward(self):
        fixture = self.fixture()
        group = self.groups(fixture)[1]
        scores, output = self.scores(fixture, group)
        sample = group.samples[1]
        row = scores[sample.sample_id]
        origin = next(iter(row["relationship_spans"]))
        self.assertEqual(row["relationship_rewards"][origin], 0)
        self.assertEqual(row["relationship_relations"][origin], "UNRESOLVED")
        self.assertEqual(output[sample.sample_id]["relationships"][0]["advantage"], -.5)
        self.assertLess(sum(output[sample.sample_id]["relationship_token_advantages"]), 0.)

    def test_public_target_membership_does_not_invent_an_original_issue(self):
        fixture = self.fixture([["PERSISTED"] * 4, ["UNRESOLVED"] * 4, ["UNRESOLVED"] * 4])
        group = self.groups(fixture)[0]
        scores, output = self.scores(fixture, group)
        self.assertTrue(all(not row["selected_lesson_indices"] and not row["relationship_spans"]
                            and not any(row["relationship_rewards"].values()) for row in scores.values()))
        self.assertTrue(all(row["mapping_status"] == "NO_MAPPED_UNITS" for row in output.values()))

    def test_invalid_truncated_and_non_ascii_padded_labels_have_no_mapped_credit(self):
        invalid = ['{"relation":"PERSISTED"}', "PERSISTED extra", ("PERSISTED", "length"), "\u00a0PERSISTED"]
        fixture = self.fixture([["UNRESOLVED"] * 4, invalid, ["UNRESOLVED"] * 4])
        scores, output = self.scores(fixture, self.groups(fixture)[1])
        self.assertTrue(all(row["atomic_reward"] == 0 and not row["relationship_spans"] for row in scores.values()))
        self.assertTrue(all(not any(row["partial_token_advantages"]) for row in output.values()))

    def test_ascii_outer_whitespace_is_excluded_and_crossing_tokens_receive_no_credit(self):
        replies = [["UNRESOLVED"] * 4, [" \tPERSISTED\r\n", "UNRESOLVED"] * 2, ["UNRESOLVED"] * 4]
        fixture = self.fixture(replies)
        group = self.groups(fixture)[1]
        scores, output = self.scores(fixture, group)
        sample = group.samples[0]
        span = next(iter(scores[sample.sample_id]["relationship_spans"].values()))
        self.assertEqual(sample.text[slice(*span)], "PERSISTED")
        entry = output[sample.sample_id]["relationships"][0]
        self.assertEqual(entry["token_indices"], list(range(2, 11)))
        chunked = self.fixture([["UNRESOLVED"] * 4,
            [" PERSISTED ", "UNRESOLVED"] * 2, ["UNRESOLVED"] * 4], characters=False)
        group = self.groups(chunked)[1]
        _, output = self.scores(chunked, group)
        entry = output[group.samples[0].sample_id]["relationships"][0]
        self.assertEqual(entry["token_indices"], [])
        self.assertEqual(entry["coefficient"], 0.)

    def test_sample_id_raw_text_and_finish_are_required_to_resolve_actual_bound_target(self):
        fixture = self.fixture()
        sample = self.groups(fixture)[1].samples[0]
        args = (sample.text, sample.finish_reason, fixture.tree,
                fixture.tree.notebook["rewarded_notebook"]["synthesis_audit"], fixture.verifier)
        with self.assertRaisesRegex(ValueError, "actual sample ID"):
            credit.score_note(*args, actor=fixture.actor)
        for text, finish, identity in (("CORRECTED", "stop", sample.sample_id),
                (sample.text, "length", sample.sample_id), (sample.text, "stop", "unknown")):
            with self.subTest(text=text, finish=finish, identity=identity), self.assertRaises(ValueError):
                credit.score_note(text, finish, fixture.tree, args[3], fixture.verifier,
                                  actor=fixture.actor, sample_id=identity)

    def test_receipt_swaps_raw_tokens_plan_target_and_spans_are_rejected(self):
        fixture = self.fixture()
        native, output_group = self.groups(fixture)[1:]
        scores, _ = self.scores(fixture, native)
        output_scores, _ = self.scores(fixture, output_group)
        sample = native.samples[0]
        original = scores[sample.sample_id]
        for mutation in ("sample", "tokens", "plan", "target", "span", "prompt", "policy"):
            row = copy.deepcopy(original)
            binding = row["pair_action_binding"]
            if mutation == "sample":
                binding["sample_id"] = native.samples[1].sample_id
            elif mutation == "tokens":
                binding["completion_token_ids_sha256"] = "0" * 64
            elif mutation == "plan":
                binding["target_plan_sha256"] = "0" * 64
            elif mutation == "target":
                other = output_scores[output_group.samples[0].sample_id]["pair_action_binding"]["target_group"]["target"]
                binding["target_group"]["target"] = other
                binding["target_plan"][binding["target_group"]["target_index"] - 1] = other
                binding["target_plan_sha256"] = _hash(binding["target_plan"])
            elif mutation == "span":
                row["relationship_spans"][next(iter(row["relationship_spans"]))] = [0, 1]
            elif mutation == "prompt":
                binding["target_group"]["prompt_sha256"] = "0" * 64
            else:
                row["notebook_generation_policy"] = "unknown"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                credit.validate_score(self.reseal(row), sample)

    def test_legacy_paired_receipt_bytes_ignore_the_new_optional_sample_identity(self):
        previous = previous_credit.PairedCreditTests()
        previous.setUp()
        self.addCleanup(previous.doCleanups)
        raw = json.dumps(previous.note())
        args = (raw, "stop", previous.tree, previous.audit, previous.verifier)
        before = credit.score_note(*args, actor=previous.actor)
        after = credit.score_note(*args, actor=previous.actor, sample_id="unused-legacy-identity")
        self.assertEqual(json.dumps(before, sort_keys=True), json.dumps(after, sort_keys=True))
        self.assertFalse(any(key in after for key in credit._PAIR_ACTION_FIELDS))
        self.assertEqual(len(episode.solution_samples(previous.tree)), 6)

    def test_private_truth_mutation_changes_credit_without_changing_public_targets_or_prompts(self):
        fixture = self.fixture()
        group = self.groups(fixture)[1]
        before, _ = self.scores(fixture, group)
        sample = group.samples[0]
        record = fixture.tree.notebook["rewarded_notebook"]
        public = copy.deepcopy((record["target_plan"], record["target_per_group"]))
        self.assertGreater(before[sample.sample_id]["atomic_reward"], 0.)
        fixture.item.slots["start"] = "q999"
        for original in episode.solution_samples(fixture.tree):
            original.result = fixture.verifier.grade(original.problem_id, original.text, original.finish_reason,
                                                       fixture.actor.mode)
            original.reward = float(original.result.status == "CORRECT")
        original_audit = episode.audit_episode(fixture.tree, fixture.actor, fixture.verifier,
                                              routing_rule=episode.STRICT_ROUTE_RULE)
        fresh = synthesis.audit_episode(fixture.tree, fixture.actor, fixture.verifier, contract=synthesis.BOUNDED_VERSION)
        record.update(episode_audit=original_audit, episode_audit_sha256=_hash(original_audit),
                      synthesis_audit=fresh, synthesis_audit_sha256=_hash(fresh))
        for row in fixture.tree.notebook["model_notes"]:
            row.update(action.grade_note(row["text"], row["note_finish_reason"], fixture.tree, fresh,
                                         fixture.verifier, actor=fixture.actor, contract=synthesis.BOUNDED_VERSION))
        after, _ = self.scores(fixture, group)
        self.assertEqual((record["target_plan"], record["target_per_group"]), public)
        self.assertEqual(after[sample.sample_id]["atomic_reward"], 0.)
        self.assertEqual(after[sample.sample_id]["relationship_spans"], {})


if __name__ == "__main__":
    unittest.main()
