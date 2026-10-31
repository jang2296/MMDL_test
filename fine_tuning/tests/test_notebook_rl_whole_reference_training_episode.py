"""Authored CPU reference actions through production plumbing; no model-success claim."""
import copy
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, local_profile
from fine_tuning.notebook_rl import notebook_pair_drafts as drafts, updates
from fine_tuning.notebook_rl import whole_reference_notebook as reference
from fine_tuning.notebook_rl.notebook_learning import append_rewarded_notebooks
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash, build_tree
from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration
from fine_tuning.tests.test_notebook_rl_whole_reference_notebook import compact


class WholeReferenceTrainingEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.f = integration.SynthesisIntegrationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        f.protocol = local_profile.build_protocol(verified_note_synthesis_credit=True,
            pair_draft_synthesis=True, whole_reference_notebook=True, context_limit=24576)
        f.prompts = local_profile.local_prompts(f.protocol)
        f.cfg = RolloutConfig.from_protocol(f.protocol, "ALG-NB3", master_seed=3407)
        self.good = json.dumps(compact(episode.solution_samples(f.tree), f.note))
        original_prepare = f.engine.prepare

        def prepare(request):
            actual = original_prepare(request)
            actual.pixels = {"pixel_values": torch.zeros((1, 2))}
            actual.images, actual.image_spans = request.images, ()
            return actual

        f.engine.prepare = prepare

    def append(self, final=None, **kwargs):
        f = self.f
        observation = "Observed facts:\n" + "\n".join(
            f"{key} = {value}" for key, value in f.item.slots.items()) + "\nSolution:"
        self.raw_drafts = ["CORRECTED", "{broken", ("PERSISTED", "length"), "UNRESOLVED"]
        f.engine.replies = [[observation], self.raw_drafts,
            [self.good, "{}", self.good, "not JSON"] if final is None else final]
        return append_rewarded_notebooks([f.tree], f.images, f.engine, f.verifier, f.prompts, f.cfg,
            actors={f.actor.problem_id: f.actor}, **kwargs)

    def validate(self, tree=None):
        f = self.f
        episode.validate_update_binding(f.tree if tree is None else tree, tokenizer=f.engine.tokenizer,
            preparation=f.engine, images=f.images[f.actor.problem_id], observation_actor=f.actor,
            observation_master_seed=3407, generation_policy=reference.VERSION, draft_policy=drafts.VERSION)

    def regrade(self, tree=None):
        f = self.f
        return episode.regrade_synthesis(f.tree if tree is None else tree, f.actor, f.verifier, f.engine.tokenizer)

    @staticmethod
    def reseal(tree):
        note = tree.notebook["rewarded_notebook"]
        note["notebook_groups_sha256"] = _hash([
            json.loads(to_json(group)) for group in tree.groups if group.kind == "DIAG"])

    def test_actual_raw_actions_complete_n4_original_history_and_replay(self):
        f = self.f
        before = to_json(f.tree)
        report = self.append()
        record = f.tree.notebook["rewarded_notebook"]
        self.assertEqual(report["generated_groups"], 1)
        self.assertEqual(record["coverage"], "COMPLETE")
        self.assertNotIn("partial_artifacts", record)
        self.assertEqual((len(record["rounds"]), len(f.engine.requests)), (1, 3))
        self.assertEqual(record["stop_reason"], "VALID_NOTE")
        self.assertEqual([row["text"] for row in record["pair_drafts"]["calls"][0]["completions"]],
                         ["CORRECTED", "{broken", "PERSISTED", "UNRESOLVED"])
        originals = episode.solution_samples(f.tree)
        self.assertEqual([to_json(row) for row in originals], [to_json(row) for row in episode.solution_samples(
            from_json(ProblemTree, before))])
        request, group = f.engine.requests[-1], f.tree.groups[-1]
        self.assertEqual((request.n, request.max_new_tokens, f.cfg.context_limit), (4, 8192, 24576))
        self.assertIs(request.images, f.images[f.actor.problem_id])
        self.assertEqual([row.text for row in group.samples], [self.good, "{}", self.good, "not JSON"])
        self.assertEqual([row.reward for row in group.samples], [1., 0., 1., 0.])
        for sample, checked in zip(group.samples, f.tree.notebook["model_notes"], strict=True):
            self.assertEqual(sample.text, f.engine.tokenizer.decode(sample.completion_token_ids))
            self.assertEqual(sample.sampler_logprobs, [-.1] * len(sample.completion_token_ids))
            self.assertEqual(checked["note_text_sha256"], hashlib.sha256(sample.text.encode()).hexdigest())
            self.assertEqual(checked["rendered_strict_report"]["reward"], sample.reward)
        row = f.tree.notebook["model_notes"][0]
        self.assertNotEqual(row["note_text_sha256"], row["rendered_strict_report"]["note_text_sha256"])
        self.validate()
        restored = from_json(ProblemTree, to_json(f.tree))
        self.validate(restored)
        self.assertEqual(len(self.regrade(restored)["samples"]), 4)

    def test_shared_text_matches_diagnostic_and_excludes_private_judgments(self):
        from fine_tuning.notebook_rl import notebook_whole_reference_probe as diagnostic
        from fine_tuning.notebook_rl import notebook_whole_synthesis_probe as baseline

        self.append()
        f = self.f
        record = f.tree.notebook["rewarded_notebook"]
        samples = episode.solution_samples(f.tree)
        expected = reference.generation_text(f.actor, samples, record["notebook_schema"],
            record["synthesis_audit"]["success_attempt"], record["fresh_observation"], record["pair_drafts"])
        messages, _, _ = diagnostic._messages(record["notebook_messages"], f.actor,
            record["notebook_schema"], samples, record, record["synthesis_audit"]["success_attempt"])
        self.assertEqual(reference.SCOPE_GRAMMAR, baseline.SCOPE_GRAMMAR)
        self.assertEqual(messages[0]["content"][-1]["text"], expected)
        self.assertEqual(f.engine.requests[-1].messages[0]["content"][-1]["text"], expected)
        private = [SimpleNamespace(text=row.text, sample_id=row.sample_id, parent_sample_id=row.parent_sample_id,
            finish_reason=row.finish_reason, result="PRIVATE_RESULT", reward="PRIVATE_REWARD", gold="PRIVATE_GOLD")
            for row in samples]
        self.assertEqual(expected, reference.generation_text(f.actor, private, record["notebook_schema"],
            record["synthesis_audit"]["success_attempt"], record["fresh_observation"], record["pair_drafts"]))
        for hidden in ("PRIVATE_RESULT", "PRIVATE_REWARD", "PRIVATE_GOLD", '"checks":', '"issues":'):
            self.assertNotIn(hidden, expected)
        transcript = json.loads(expected.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        for original, row in zip(samples, transcript, strict=True):
            self.assertEqual(original.text, "".join(text for _, text in row["source_lines"]))
            self.assertEqual(original.finish_reason, row["finish_reason"])

    def test_malformed_and_length_outputs_zero_reward_without_repairs(self):
        self.append(["{}", "not JSON", (self.good, "length"), "[]"])
        f = self.f
        record = f.tree.notebook["rewarded_notebook"]
        self.assertEqual((len(record["rounds"]), len(f.engine.requests)), (1, 3))
        self.assertEqual(record["stop_reason"], "NOTEBOOK_ROUND_LIMIT")
        self.assertEqual([sample.reward for sample in f.tree.groups[-1].samples], [0.] * 4)
        self.assertTrue(all(row["expanded_text_sha256"] is None for row in f.tree.notebook["model_notes"]))
        self.validate()
        self.assertEqual(len(self.regrade()["samples"]), 4)

    def test_contract_raw_rendered_history_draft_and_image_tampering_fail_closed(self):
        self.append()
        f = self.f
        for mutation in ("contract", "raw", "tokens", "rendered", "reward", "history", "finish", "draft", "actor", "inventory", "partial", "seed", "sample_id", "order"):
            with self.subTest(mutation=mutation):
                tree = copy.deepcopy(f.tree)
                record, row = tree.notebook["rewarded_notebook"], tree.notebook["model_notes"][0]
                sample = tree.groups[-1].samples[0]
                if mutation == "contract":
                    record["notebook_reference_contract"]["judgments"] = "host supplied"
                    record["notebook_reference_contract_sha256"] = _hash(record["notebook_reference_contract"])
                elif mutation == "raw":
                    sample.text = row["text"] = "{}"
                    row["note_text_sha256"] = hashlib.sha256(b"{}").hexdigest()
                elif mutation == "tokens":
                    sample.completion_token_ids[-2] += 1
                elif mutation == "rendered":
                    row["rendered_strict_report"]["bounded_claims_verified"] = False
                elif mutation == "reward":
                    sample.reward = sample.diag_reward = row["reward"] = 0.
                elif mutation == "history":
                    episode.solution_samples(tree)[0].text += "\nchanged"
                elif mutation == "finish":
                    episode.solution_samples(tree)[0].finish_reason = "length"
                elif mutation == "draft":
                    draft = record["pair_drafts"]
                    draft["calls"][0]["completions"][0]["text"] = "changed"
                    draft["sha256"] = _hash({key: value for key, value in draft.items() if key != "sha256"})
                    record["pair_drafts_sha256"] = _hash(draft)
                elif mutation == "actor":
                    record["notebook_actor"]["question"] += "changed"
                elif mutation == "inventory":
                    record["reference_inventory"]["assessment_refs"].pop()
                    record["reference_inventory_sha256"] = _hash(record["reference_inventory"])
                elif mutation == "seed":
                    sample.seed += 1
                elif mutation == "sample_id":
                    sample.sample_id += "changed"
                elif mutation == "order":
                    tree.groups[-1].samples.reverse()
                else:
                    record["partial_artifacts"] = []
                self.reseal(tree)
                with self.assertRaises(ValueError):
                    self.regrade(tree)
        f.images[f.actor.problem_id][0].putpixel((0, 0), (1, 2, 3))
        with self.assertRaises(ValueError):
            self.validate()

    def test_new_binding_cannot_be_attached_to_historical_canonical_or_partial_modes(self):
        f = self.f
        for policy in (None, "whole-history-reference-assessments-v1", "verified-pair-action-notebook-v1"):
            prompts = copy.deepcopy(f.prompts)
            if policy is None:
                del prompts["notebook_generation_policy"]
            else:
                prompts["notebook_generation_policy"] = policy
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                append_rewarded_notebooks([copy.deepcopy(f.tree)], f.images, f.engine, f.verifier,
                    prompts, f.cfg, actors={f.actor.problem_id: f.actor})
        self.assertEqual(f.engine.requests, [])
        self.append()
        for policy in (None, "whole-history-reference-assessments-v1"):
            tree = copy.deepcopy(f.tree)
            if policy is None:
                del tree.notebook["rewarded_notebook"]["notebook_generation_policy"]
            else:
                tree.notebook["rewarded_notebook"]["notebook_generation_policy"] = policy
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.validate(tree)

    def test_final_context_block_preserves_originals_and_all_drafts(self):
        self.f.engine.blocked = "EPISODE_NOTEBOOK"
        self.assertEqual(self.append()["generated_groups"], 0)
        record = self.f.tree.notebook["rewarded_notebook"]
        self.assertEqual(record["status"], "FINAL_NOTE_BLOCKED")
        self.assertEqual(record["pair_drafts"]["status"], "COMPLETE")
        self.assertEqual(record["group_ids"], [])
        self.assertNotIn("model_notes", self.f.tree.notebook)
        self.validate()
        self.assertEqual(self.regrade()["samples"], {})

    def test_draft_context_block_preserves_reread_and_blocked_stage_without_actions(self):
        self.f.engine.blocked = "PAIR_DRAFT"
        self.assertEqual(self.append()["generated_groups"], 0)
        record = self.f.tree.notebook["rewarded_notebook"]
        self.assertEqual(record["status"], "DRAFT_BLOCKED")
        self.assertEqual(record["pair_drafts"]["status"], "CONTEXT_BLOCKED")
        self.assertEqual(record["group_ids"], [])
        self.validate()
        self.assertEqual(self.regrade()["samples"], {})

    def test_reread_context_block_preserves_originals_without_draft_or_note_actions(self):
        self.f.engine.blocked = "FRESH_OBSERVATION"
        self.assertEqual(self.append()["generated_groups"], 0)
        record = self.f.tree.notebook["rewarded_notebook"]
        self.assertEqual(record["status"], "CONTEXT_BLOCKED")
        self.assertNotIn("fresh_observation", record)
        self.assertNotIn("pair_drafts", record)
        self.validate()
        self.assertEqual(self.regrade()["samples"], {})

    def test_budget_stop_before_generation_preserves_complete_audited_history(self):
        self.assertEqual(self.append(should_stop=lambda: True)["generated_groups"], 0)
        self.assertEqual(self.f.tree.notebook["rewarded_notebook"]["status"], "BUDGET_STOP")
        self.assertEqual(self.f.engine.requests, [])
        self.validate()
        self.assertEqual(self.regrade()["samples"], {})

    def test_no_recovery_keeps_raw_history_without_constructing_a_reference_prompt(self):
        f = self.f
        samples = episode.solution_samples(f.tree)
        for sample in samples[4:]:
            sample.text = samples[0].text
            sample.completion_token_ids = f.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.result = f.verifier.grade(sample.problem_id, sample.text, "stop", f.actor.mode)
            sample.reward = 0.
        before = [to_json(sample) for sample in samples]
        with patch.object(reference, "instruction", wraps=reference.instruction) as instruction:
            self.assertEqual(self.append()["generated_groups"], 0)
            self.validate()
            self.assertEqual(self.regrade()["samples"], {})
        instruction.assert_not_called()
        self.assertEqual(f.tree.notebook["rewarded_notebook"]["status"], "NO_VERIFIED_RECOVERY")
        self.assertEqual(f.engine.requests, [])
        self.assertEqual(before, [to_json(sample) for sample in episode.solution_samples(f.tree)])

    def test_original_input_context_block_keeps_actual_skip_without_invented_inventory(self):
        f = self.f
        f.engine.blocked = ":T1"
        f.tree = build_tree(f.actor, f.images[f.actor.problem_id], f.engine, f.verifier,
            f.prompts, f.cfg, 0, "verified-any-visual-v1")
        f.engine.blocked = None
        skipped = copy.deepcopy(episode.solution_skips(f.tree))
        self.assertTrue(skipped)
        self.assertEqual(skipped[0]["status"], "CONTEXT_BLOCKED")
        self.assertEqual(episode.solution_samples(f.tree), [])
        with patch.object(reference, "instruction", wraps=reference.instruction) as instruction:
            self.assertEqual(self.append()["generated_groups"], 0)
            self.validate()
            self.assertEqual(self.regrade()["samples"], {})
        instruction.assert_not_called()
        record = f.tree.notebook["rewarded_notebook"]
        self.assertEqual(record["status"], "INCOMPLETE_EPISODE")
        self.assertIsNone(record["reference_inventory"])
        self.assertEqual(skipped, episode.solution_skips(f.tree))
        self.assertEqual(f.engine.requests, [])

    def test_actual_differentiable_fixture_commits_only_verified_native_signal(self):
        self.append()
        f = self.f
        updates.bind_notebook_objective(f.tree, f.actor, f.verifier, f.protocol["notebook_objective"])
        fixture = f.optimizer_fixture()
        result = f.step(fixture)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertTrue(result["native_relationship_commit_required"])
        self.assertTrue(torch.count_nonzero(fixture[0].model["L"]))
        self.assertEqual(fixture[2].last_epoch, 1)
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))

    def test_native_zero_with_fact_signal_never_reaches_optimizer(self):
        note = json.loads(self.good)
        for row in note["assessments"]:
            row[3] = None
        self.append([json.dumps(note), "{}", json.dumps(note), "{}"])
        f = self.f
        updates.bind_notebook_objective(f.tree, f.actor, f.verifier, f.protocol["notebook_objective"])
        fixture = f.optimizer_fixture()
        initial_epoch = fixture[2].last_epoch
        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
            result = f.step(fixture)
        self.assertEqual(result["status"], "NO_RETAINED_NATIVE_RELATIONSHIP_SIGNAL")
        step.assert_not_called()
        self.assertEqual(fixture[1].state, {})
        self.assertEqual(fixture[2].last_epoch, initial_epoch)
        self.assertTrue(all(torch.count_nonzero(parameter) == 0 and parameter.grad is None
                            for parameter in fixture[0].model.parameters()))


if __name__ == "__main__":
    unittest.main()
