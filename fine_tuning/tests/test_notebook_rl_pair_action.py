"""Fresh scripted pair actions and their binding, not model learning evidence."""
import copy
from dataclasses import replace
import json
import unittest

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_pair_probe as probe
from fine_tuning.notebook_rl import notebook_synthesis as synthesis, pair_action_notebook as action, paired_notebook
from fine_tuning.notebook_rl.local_profile import local_prompts
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as native
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration
from fine_tuning.tests.test_notebook_rl_pair_action_integration import pair_action_fixture, pair_action_protocol, scripted_groups


class PairActionTests(unittest.TestCase):
    def fresh(self, *, unknown=False, no_recovery=False):
        fixture = integration.SynthesisIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol = pair_action_protocol()
        fixture.prompts = local_prompts(fixture.protocol)
        fixture.cfg = RolloutConfig.from_protocol(fixture.protocol, "ALG-NB3", master_seed=3407)
        samples = episode.solution_samples(fixture.tree)
        for index, sample in enumerate(samples):
            if no_recovery:
                sample.text = native.response(fixture.item.slots)
            if unknown:
                sample.text = "Solution:" + sample.text.split("Solution:", 1)[1]
            sample.completion_token_ids = fixture.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.result = fixture.verifier.grade(sample.problem_id, sample.text, sample.finish_reason, fixture.actor.mode)
            sample.reward = float(sample.result.status == "CORRECT")
            if sample.parent_sample_id is not None:
                parent = next(value for value in samples if value.sample_id == sample.parent_sample_id)
                sample.prompt_token_ids = parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2]
        original_prepare = fixture.engine.prepare

        def prepare(request):
            prepared = original_prepare(request)
            prepared.pixels = {"pixel_values": torch.zeros((1, 2))}
            prepared.images, prepared.image_spans = request.images, ()
            return prepared

        fixture.engine.prepare = prepare
        return fixture

    @staticmethod
    def append(fixture, *, reread=None, should_stop=None):
        observed = reread or "Observed facts:\n" + "\n".join(
            f"{key} = {value}" for key, value in fixture.item.slots.items()) + "\nSolution:"
        plan, _ = action._plan(episode.solution_samples(fixture.tree), None)
        fixture.engine.replies = [[observed], *[["UNRESOLVED"] * 4 for _ in plan]]
        return action.append([fixture.tree], fixture.images, fixture.engine, fixture.verifier,
            fixture.prompts, fixture.cfg, actors={fixture.actor.problem_id: fixture.actor}, should_stop=should_stop)

    def test_versioned_prompt_discloses_all_public_pairs_partial_coverage_and_normalization(self):
        contract = action.contract()
        self.assertEqual(action.VERSION, "verified-pair-action-notebook-v2")
        self.assertNotEqual(action.prompt_contract(), probe.PROMPT)
        self.assertIn("TRACE operation", action.prompt_contract())
        self.assertIn("ALL public authored ancestor comparisons", action.prompt_contract())
        self.assertEqual(contract["coverage"], "PARTIAL")
        self.assertIn("isolated errors", contract["unexamined_coverage"])
        self.assertFalse(contract["full_notebook_verified"])
        self.assertIn("cap128", contract["normalization"])
        self.assertFalse(contract["strict_reward_enabled"])
        self.assertEqual(contract["relationship_token_unit"], action.TOKEN_UNIT)
        self.assertEqual((contract["n"], contract["max_new_tokens"], contract["context_limit"]), (4, 128, 24576))

    def test_all_fresh_groups_keep_original_histories_images_reread_tokens_and_target_conditions(self):
        fixture = pair_action_fixture(self, all_targets=True)
        record = fixture.tree.notebook["rewarded_notebook"]
        self.assertEqual(record["rounds"], [])
        self.assertEqual(record["stop_reason"], "PAIR_ACTION_TARGET_LIMIT")
        groups = [group for group in fixture.tree.groups if group.kind == "DIAG"]
        self.assertEqual(len(groups), len(record["target_plan"]))
        originals = episode.solution_samples(fixture.tree)
        expected_seeds = groups[0].samples
        for index, (group, request) in enumerate(zip(groups, fixture.engine.requests[1:], strict=True), 1):
            entry = record["target_per_group"][group.group_id]
            self.assertEqual(entry["target_index"], index)
            self.assertEqual(entry["target"], record["target_plan"][index - 1])
            self.assertEqual(request.seeds, [sample.seed for sample in expected_seeds])
            self.assertIs(request.images, fixture.images[fixture.actor.problem_id])
            text = request.messages[0]["content"][-1]["text"]
            transcript = json.loads(text.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
            self.assertEqual(len(transcript), len(originals))
            for row, original in zip(transcript, originals, strict=True):
                self.assertEqual("".join(line for _, line in row["source_lines"]), original.text)
            self.assertEqual(text.count("<target_pair>"), 1)
            self.assertNotIn("<public_pairs>", text)
            self.assertNotIn('"checks":', text)
            self.assertIn("UNVERIFIED", text)
            for sample in group.samples:
                self.assertEqual(fixture.engine.tokenizer.decode(sample.completion_token_ids), sample.text)
                if sample is group.samples[0] and index in (1, len(groups)):
                    self.assertEqual(action.bound_target(fixture.tree, sample.sample_id, sample.text, sample.finish_reason), entry["target"])
                self.assertIsNone(sample.result)
                self.assertEqual(sample.reward, 0.)
        self.assertEqual([sample.text for group in scripted_groups(fixture) for sample in group.samples],
            ["UNRESOLVED"] * 4 + ["PERSISTED", "UNRESOLVED"] * 2 + ["FINALIZED", "CORRECTED"] * 2)
        action.validate_update_binding(fixture.tree, tokenizer=fixture.engine.tokenizer, preparation=fixture.engine,
            images=fixture.images[fixture.actor.problem_id], observation_actor=fixture.actor)

    def test_resealed_public_plan_target_swaps_and_raw_action_tampering_are_rejected(self):
        fixture = pair_action_fixture(self)
        original = fixture.tree
        first = next(group for group in original.groups if group.kind == "DIAG")
        for mutation in ("target", "plan", "old_plan", "messages", "text", "tokens", "seed", "finish", "contract"):
            tree = copy.deepcopy(original)
            note = tree.notebook["rewarded_notebook"]
            group = next(group for group in tree.groups if group.kind == "DIAG")
            sample = group.samples[0]
            if mutation == "target":
                note["target_per_group"][first.group_id]["target"] = note["target_plan"][1]
            elif mutation == "plan":
                note["target_plan"].reverse()
                note["target_plan_sha256"] = _hash(note["target_plan"])
            elif mutation == "old_plan":
                note["target_plan"] = probe.select_targets(episode.solution_samples(tree), note["fresh_observation"])
                note["target_plan_sha256"] = _hash(note["target_plan"])
            elif mutation == "messages":
                note["target_per_group"][first.group_id]["messages"] = []
                note["target_per_group"][first.group_id]["messages_sha256"] = _hash([])
            elif mutation == "text":
                sample.text = "FINALIZED"
            elif mutation == "tokens":
                sample.completion_token_ids[0] += 10000
            elif mutation == "seed":
                sample.seed += 1
            elif mutation == "finish":
                sample.finish_reason = "length"
            else:
                note["notebook_generation_contract"]["strict_reward_enabled"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                action.validate_update_binding(tree, tokenizer=fixture.engine.tokenizer)
        with self.assertRaisesRegex(ValueError, "membership differs"):
            action.bound_target(original, first.samples[0].sample_id, "FINALIZED", "stop")

    def test_context_blockage_and_budget_stop_preserve_originals_and_independent_route(self):
        for blocked in ("FRESH_OBSERVATION", "PAIR_ACTION", None):
            fixture = self.fresh()
            before = [to_json(sample) for sample in episode.solution_samples(fixture.tree)]
            fixture.engine.blocked = blocked
            report = self.append(fixture, should_stop=(lambda: True) if blocked is None else None)
            note = fixture.tree.notebook["rewarded_notebook"]
            self.assertEqual(report["generated_groups"], 0)
            self.assertEqual(note["status"], "BUDGET_STOP" if blocked is None else "CONTEXT_BLOCKED")
            self.assertTrue(note["routing_finalized"])
            self.assertEqual(before, [to_json(sample) for sample in episode.solution_samples(fixture.tree)])
            action.validate_update_binding(fixture.tree, tokenizer=fixture.engine.tokenizer)

    def test_unverified_reread_neither_adds_targets_nor_blocks_original_ancestor_pairs(self):
        fixture = self.fresh()
        expected, _ = action._plan(episode.solution_samples(fixture.tree), None)
        report = self.append(fixture, reread="Observed facts:\nunknown = first\nunknown = second\nSolution:")
        note = fixture.tree.notebook["rewarded_notebook"]
        self.assertEqual(report["generated_groups"], len(expected))
        self.assertEqual(note["status"], "GENERATED")
        self.assertIsNone(note["public_target_error"])
        self.assertEqual(note["target_plan"], expected)
        self.assertIs(note["fresh_observation"]["verified"], False)
        action.validate_update_binding(fixture.tree, tokenizer=fixture.engine.tokenizer)

    def test_plan_preserves_all_public_stable_slots_trace_operations_and_retry_branches_without_gold(self):
        fixture = self.fresh()
        samples = episode.solution_samples(fixture.tree)
        inventory = paired_notebook.public_pairs(samples)
        plan, blocked = action._plan(samples, None)
        self.assertIsNone(blocked)
        self.assertEqual([row["pair_id"] for row in plan], list(inventory))
        self.assertEqual([{side: row[side] for side in ("earlier", "later")} for row in plan], list(inventory.values()))
        self.assertEqual({row["earlier"]["kind"] for row in plan}, {"OBSERVATION", "TRACE", "OUTPUT"})
        self.assertTrue(all(row["channel"] == ("OUTPUT" if row["earlier"]["kind"] == "OUTPUT" else "NATIVE")
                            and row["raw_values"] == [] for row in plan))
        stable = [(row["earlier"]["attempt"], row["later"]["attempt"]) for row in plan
                  if row["earlier"]["locus"] == "accepting"]
        self.assertIn((3, 5), stable)
        self.assertIn((3, 6), stable)
        self.assertTrue(all(row["earlier"]["attempt"] < row["later"]["attempt"] for row in plan))
        changed = [replace(sample, result=None, reward=999., diag_reward=999., seed=0) for sample in samples]
        fixture.verifier.items[fixture.actor.problem_id] = replace(fixture.item, gold="PRIVATE_CHANGED_GOLD")
        self.assertEqual(action._plan(changed, {"verified": True, "completion": {"text": "private-independent"}}),
                         (plan, None))
        orphaned = [replace(sample, parent_sample_id=None) for sample in samples]
        self.assertEqual(action._plan(orphaned, None), ([], "pair action: authored ancestor pairs required"))

    def test_unknown_observations_never_become_healthy_vision_and_no_recovery_creates_no_notes(self):
        unknown = self.fresh(unknown=True)
        self.append(unknown)
        note = unknown.tree.notebook["rewarded_notebook"]
        self.assertEqual(note["episode_audit"]["admission"], "UNRESOLVED")
        self.assertFalse(note["episode_audit"]["all_required_observations_verified"])
        self.assertEqual({target["earlier"]["kind"] for target in note["target_plan"]}, {"TRACE", "OUTPUT"})
        unresolved = self.fresh(no_recovery=True)
        self.append(unresolved)
        self.assertEqual(unresolved.tree.notebook["rewarded_notebook"]["status"], "NO_VERIFIED_RECOVERY")
        self.assertEqual(unresolved.engine.requests, [])

    def test_legacy_strict_assessment_is_preserved_but_applied_strict_reward_is_disabled(self):
        fixture = integration.SynthesisIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        checked = action.grade_note(fixture.good, "stop", fixture.tree, fixture.audit, fixture.verifier,
                                    actor=fixture.actor, contract=synthesis.BOUNDED_VERSION)
        self.assertEqual(checked["legacy_strict_reward"], 1.)
        self.assertEqual(checked["reward"], 0.)
        self.assertFalse(checked["strict_reward_enabled"])
        self.assertEqual(checked["notebook_generation_policy"], action.VERSION)

    def test_historical_notes_wrong_policy_and_new_images_cannot_be_rebound_as_fresh_actions(self):
        fixture = pair_action_fixture(self)
        with self.assertRaisesRegex(ValueError, "fresh same-policy"):
            action.append([fixture.tree], fixture.images, fixture.engine, fixture.verifier,
                fixture.prompts, fixture.cfg, actors={fixture.actor.problem_id: fixture.actor})
        with self.assertRaisesRegex(ValueError, "execution binding"):
            action.validate_update_binding(fixture.tree, generation_policy="historical-policy")
        original_prepare = fixture.engine.prepare

        def different_pixels(request):
            prepared = original_prepare(request)
            if "PAIR_ACTION" in request.request_id:
                prepared.pixels["pixel_values"] = torch.ones((1, 2))
            return prepared

        fixture.engine.prepare = different_pixels
        with self.assertRaisesRegex(ValueError, "original image tensors"):
            action.validate_update_binding(fixture.tree, preparation=fixture.engine,
                images=fixture.images[fixture.actor.problem_id], observation_actor=fixture.actor)
