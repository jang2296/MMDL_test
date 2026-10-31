"""CPU same-history reference plumbing; authored replies prove no model improvement."""
import copy
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from fine_tuning.notebook_rl import local_profile as profile
from fine_tuning.notebook_rl import notebook_whole_trained_probe as probe, whole_reference_notebook as reference
from fine_tuning.notebook_rl.engines import Completion
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as native
from fine_tuning.tests import test_notebook_rl_whole_reference_training_episode as training


class WholeReferenceTrainedProbeTests(unittest.TestCase):
    def fixture(self):
        authored = training.WholeReferenceTrainingEpisodeTests()
        authored.setUp()
        self.addCleanup(authored.doCleanups)
        authored.append(["{}"] * 4)
        f = authored.f
        f.engine.policy_version, f.engine.context_limit = 1, 24576
        f.engine.requests.clear()
        first_actor = replace(f.actor, problem_id="fixed-first-no-recovery")
        f.verifier.actors[first_actor.problem_id] = first_actor
        f.verifier.items[first_actor.problem_id] = replace(f.item, problem_id=first_actor.problem_id)
        first = native.tree_from(first_actor, f.verifier, [native.response(f.item.slots)] * 6)
        return f, [first_actor, f.actor], [first, f.tree], authored.good

    def collect(self, f, actors, trees, replies):
        f.engine.replies = [replies]
        events = []
        rows, skipped = probe.collect(actors, trees, f.images, f.prompts, f.engine, f.verifier,
            lambda *row: events.append(row), time.monotonic() + 60, protocol=f.protocol)
        return rows, skipped, events

    def test_reference_contract_is_explicit_and_legacy_draft_contract_stays_distinct(self):
        legacy = profile.build_protocol(verified_note_synthesis_credit=True,
            pair_draft_synthesis=True, context_limit=24576)
        current = profile.build_protocol(verified_note_synthesis_credit=True,
            pair_draft_synthesis=True, whole_reference_notebook=True, context_limit=24576)
        old, new = probe.contract(legacy), probe.contract(current)
        self.assertEqual(old["protocol_id"], probe.VERSION)
        self.assertNotIn("diagnostic_only", old)
        self.assertNotIn("notebook_generation_policy", old)
        self.assertEqual(new["protocol_id"], probe.REFERENCE_VERSION)
        self.assertNotEqual(new["protocol_id"], old["protocol_id"])
        self.assertEqual(new["notebook_generation_policy"], reference.VERSION)
        self.assertEqual(new["notebook_reference_contract"], reference.contract())
        self.assertEqual(new["notebook_reference_contract_sha256"], _hash(reference.contract()))
        self.assertTrue(new["diagnostic_only"])
        self.assertEqual((new["n"], new["max_new_tokens"], new["context_limit"]), (4, 8192, 24576))
        for key in ("training_allowed", "draft_regeneration", "heldout", "notebook_reward_effect_isolated",
                    "self_improvement_verified", "feedback", "retries"):
            self.assertIs(new[key], False)
        for mutation in ("protocol_id", "notebook_generation_policy", "notebook_reference_contract"):
            changed = copy.deepcopy(current)
            changed[mutation] = "stale"
            changed["sha256"] = profile.digest({key: value for key, value in changed.items() if key != "sha256"})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.contract(changed)

    def test_observed_policy_one_raw_samples_score_without_mutating_or_relabeling_history(self):
        f, actors, trees, good = self.fixture()
        originals = [to_json(tree) for tree in trees]
        note, baseline = f.tree.notebook["rewarded_notebook"], f.tree.groups[-1]
        with patch.object(probe.credit, "score_note", wraps=probe.credit.score_note) as scorer:
            rows, skipped, events = self.collect(f, actors, trees, [good, "{}", "not JSON", (good, "length")])
        self.assertEqual([to_json(tree) for tree in trees], originals)
        self.assertEqual(skipped, [{"problem_id": actors[0].problem_id, "reason": "NO_VERIFIED_RECOVERY"}])
        self.assertEqual(len(f.engine.requests), 1)
        request = f.engine.requests[0]
        self.assertEqual(request.request_id, f"{f.actor.problem_id}:{probe.REFERENCE_VERSION}")
        self.assertEqual(request.messages, note["notebook_messages"])
        self.assertIs(request.images, f.images[f.actor.problem_id])
        self.assertEqual(request.seeds, [sample.seed for sample in baseline.samples])
        self.assertEqual((request.n, request.max_new_tokens), (4, 8192))
        self.assertEqual([event for event, _ in events], ["CALL_STARTED", "CALL_GENERATED", "CALL_COMPLETED"])
        self.assertEqual(events[0][1]["prompt_token_ids"], baseline.samples[0].prompt_token_ids)
        self.assertEqual(len(events[0][1]["baseline_replay"]["samples"]), 4)
        row = rows[0]
        self.assertEqual((row["before_totals"]["strict_verified_notes"], row["after_totals"]["strict_verified_notes"]), (0, 1))
        self.assertEqual((row["history_policy_version"], row["generation_policy_version"], row["optimizer_updates"]), (0, 1, 0))
        self.assertFalse(row["heldout"])
        self.assertEqual(scorer.call_count, 8)
        source_ids = {sample.sample_id for group in f.tree.groups for sample in group.samples}
        observed = []
        for call in scorer.call_args_list[:4]:
            self.assertNotIn("observed_sample", call.kwargs)
            self.assertIn(call.kwargs["sample_id"], source_ids)
        for call, checked in zip(scorer.call_args_list[4:], row["samples"], strict=True):
            actual = call.kwargs["observed_sample"]
            observed.append(actual)
            self.assertIs(call.args[2], f.tree)
            self.assertEqual(call.kwargs["sample_id"], actual.sample_id)
            self.assertNotIn(actual.sample_id, source_ids)
            self.assertEqual(actual.policy_version, 1)
            self.assertEqual(actual.prompt_token_ids, baseline.samples[0].prompt_token_ids)
            self.assertEqual(f.engine.tokenizer.decode(actual.completion_token_ids), actual.text)
            self.assertEqual(checked["text"], actual.text)
            self.assertIs(checked["eligible_for_rl_loss"], False)
            self.assertIs(checked["used_for_routing"], False)
            strict, score = checked["strict_check"], checked["score"]
            self.assertEqual(strict["note_text_sha256"], reference._sha(actual.text))
            self.assertEqual(score["raw_text_sha256"], reference._sha(actual.text))
            self.assertEqual(score["solution_sha256"], note["synthesis_audit"]["solution_sha256"])
            self.assertEqual(score["whole_reference_sample_binding"]["prompt_token_ids_sha256"],
                             probe.credit.digest(actual.prompt_token_ids))
            self.assertEqual(score["whole_reference_sample_binding"]["completion_token_ids_sha256"],
                             probe.credit.digest(actual.completion_token_ids))
            self.assertTrue(score["diagnostic_only"])
            with self.assertRaises(ValueError):
                probe.credit.validate_score(score, actual)
        strict = row["samples"][0]["strict_check"]
        self.assertNotEqual(strict["note_text_sha256"], strict["rendered_strict_report"]["note_text_sha256"])
        self.assertEqual(strict["reward"], strict["rendered_strict_report"]["reward"])
        with self.assertRaises(ValueError):
            probe.credit.group_credit(observed, {sample["sample_id"]: sample["score"] for sample in row["samples"]},
                                     f.engine.tokenizer)

    def test_observed_token_decode_is_checked_before_diagnostic_scoring(self):
        f, actors, trees, good = self.fixture()
        originals, sample = [to_json(tree) for tree in trees], f.engine.sample

        def changed(requests):
            outputs = sample(requests)
            value = outputs[0].completions[0]
            outputs[0].completions[0] = Completion(value.token_ids, value.logprobs,
                value.finish_reason, value.text + " changed")
            return outputs

        with patch.object(f.engine, "sample", side_effect=changed), \
                patch.object(probe.credit, "score_note", wraps=probe.credit.score_note) as scorer:
            with self.assertRaisesRegex(ValueError, "token/text binding"):
                self.collect(f, actors, trees, [good] * 4)
            self.assertEqual(scorer.call_count, 4)
            self.assertTrue(all("observed_sample" not in call.kwargs for call in scorer.call_args_list))
        self.assertEqual([to_json(tree) for tree in trees], originals)

    def test_reference_history_generation_contract_and_saved_seed_are_checked_before_sampling(self):
        for mutation in ("generation", "contract", "seed"):
            with self.subTest(mutation=mutation):
                f, actors, trees, good = self.fixture()
                if mutation == "generation":
                    f.tree.notebook["rewarded_notebook"]["notebook_generation_policy"] = "whole-history-reference-assessments-v1"
                elif mutation == "contract":
                    f.tree.notebook["rewarded_notebook"]["notebook_reference_contract"]["judgments"] = "changed"
                else:
                    f.tree.groups[-1].samples[0].seed += 1
                with self.assertRaises(ValueError):
                    self.collect(f, actors, trees, [good] * 4)
                self.assertEqual(f.engine.requests, [])


if __name__ == "__main__":
    unittest.main()
