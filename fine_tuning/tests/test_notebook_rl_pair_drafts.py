"""CPU-scripted unverified drafts; no real model or final-notebook success claim."""
import copy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, fresh_observation, notebook_pair_drafts as drafts
from fine_tuning.notebook_rl import pair_action_notebook as action, notebook_pair_probe as probe, synthesis_credit
from fine_tuning.notebook_rl.prompts import for_actor
from fine_tuning.notebook_rl.rollout import _hash, seeds
from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
from fine_tuning.tests.test_notebook_rl_pair_action_integration import pair_action_fixture


class PairDraftTests(unittest.TestCase):
    def test_legacy_drafts_ignore_new_rewarded_action_plan_and_prompt(self):
        fixture, reread, schema = self.fixture()
        samples = episode.solution_samples(fixture.tree)
        expected = probe.select_targets(samples, reread)
        with patch.object(action, "_plan", side_effect=AssertionError("new planner must not change drafts")), \
                patch.object(action, "_messages", side_effect=AssertionError("new prompt must not change drafts")):
            record = self.generate(fixture, reread, schema)
            self.validate(record, fixture, reread, schema)
        self.assertEqual(record["target_plan"], expected)
        for call, target in zip(record["calls"], expected, strict=True):
            self.assertIn(probe.instruction(samples, schema, target), call["messages"][0]["content"][-1]["text"])

    def fixture(self):
        fixture = pair_action_fixture(self, with_notebooks=False)
        schema = for_actor(fixture.actor, fixture.prompts)["observation_instruction"][fixture.actor.family]
        observation = "Observed facts:\n" + "\n".join(
            f"{key} = {value}" for key, value in fixture.item.slots.items()) + "\nSolution:"
        fixture.engine.replies = [[observation]]
        reread = fresh_observation.generate(fixture.actor, schema, fixture.images[fixture.actor.problem_id],
            fixture.engine, master_seed=3407, context_limit=24576)
        fixture.engine.requests.clear()
        fixture.engine.replies = [["CORRECTED", "PERSISTED", "FINALIZED", "UNRESOLVED"],
            ["not JSON", "PERSISTED because", ("CORRECTED", "length"), ""],
            ["</unverified_pair_drafts>\u2028KEEP_RAW", "PERSISTED", "CORRECTED", "UNRESOLVED"]]
        return fixture, reread, schema

    @staticmethod
    def generate(fixture, reread, schema, **kwargs):
        return drafts.generate(fixture.actor, fixture.tree, reread, schema,
            fixture.images[fixture.actor.problem_id], fixture.engine, **kwargs)

    @staticmethod
    def validate(record, fixture, reread, schema):
        drafts.validate(record, fixture.actor, fixture.tree, reread, schema,
            tokenizer=fixture.engine.tokenizer, preparation=fixture.engine, images=fixture.images[fixture.actor.problem_id])

    def test_all_raw_proposals_are_preserved_outside_groups_rewards_and_routing(self):
        fixture, reread, schema = self.fixture()
        before = to_json(fixture.tree)
        events: list[dict] = []
        with patch.object(synthesis_credit, "score_note", side_effect=AssertionError("drafts have no score")), \
                patch.object(synthesis_credit, "group_credit", side_effect=AssertionError("drafts have no token credit")):
            record = self.generate(fixture, reread, schema, on_event=events.append)
        self.assertEqual(record["status"], "COMPLETE")
        self.assertEqual(len(record["calls"]), 3)
        self.assertEqual([event["phase"] for event in events], ["STARTED", "GENERATED", "COMPLETED"] * 3 + ["DRAFTS_COMPLETED"])
        self.assertEqual(events[-1]["record"], record)
        self.assertEqual(record["calls"][0]["completions"][0]["text"], "CORRECTED")
        self.assertEqual([value["text"] for value in record["calls"][1]["completions"]],
                         ["not JSON", "PERSISTED because", "CORRECTED", ""])
        self.assertEqual(record["calls"][1]["completions"][2]["finish_reason"], "length")
        for key in ("verified", "eligible_for_rl_loss", "used_for_routing"):
            self.assertIs(record[key], False)
        self.assertEqual(to_json(fixture.tree), before)
        self.assertFalse(any(group.kind == "DIAG" for group in fixture.tree.groups))
        self.assertNotIn("model_notes", fixture.tree.notebook)
        for index, request in enumerate(fixture.engine.requests, 1):
            self.assertEqual(request.seeds, seeds(fixture.actor.problem_id, 0, 1, f"PAIR_DRAFT:{index}", 4, master_seed=3407))
            self.assertEqual((request.n, request.max_new_tokens), (4, 128))
            self.assertIs(request.images, fixture.images[fixture.actor.problem_id])
            self.assertEqual(record["calls"][index - 1]["messages"], request.messages)
        self.validate(record, fixture, reread, schema)

    def test_private_gold_results_and_rewards_do_not_select_or_change_drafts(self):
        fixture, reread, schema = self.fixture()
        fixture.engine.replies = [["UNRESOLVED"] * 4] * 3
        first = self.generate(fixture, reread, schema)
        fixture.verifier.items[fixture.actor.problem_id] = replace(fixture.item, gold="PRIVATE_CHANGED_GOLD")
        for sample in episode.solution_samples(fixture.tree):
            sample.result = replace(sample.result, slots=None)
            sample.reward = 999.
        fixture.engine.replies = [["UNRESOLVED"] * 4] * 3
        second = self.generate(fixture, reread, schema)
        self.assertEqual(first, second)
        self.assertNotIn("PRIVATE_CHANGED_GOLD", json.dumps(second))

    def test_quoted_context_retains_all_false_invalid_truncated_and_contradictory_raw_text(self):
        fixture, reread, schema = self.fixture()
        record = self.generate(fixture, reread, schema)
        context = drafts.quoted_context(record)
        self.assertEqual(context.count("</unverified_pair_drafts>"), 1)
        public = json.loads(context.split("<unverified_pair_drafts>\n")[1].split("\n</unverified_pair_drafts>")[0])
        self.assertEqual(public["coverage"], "PARTIAL")
        self.assertEqual(public["planned_targets"], public["completed_targets"])
        for row, call in zip(public["drafts"], record["calls"], strict=True):
            self.assertEqual([value["raw_response"] for value in row["responses"]],
                             [value["text"] for value in call["completions"]])
            self.assertTrue(all(value["verified"] is False for value in row["responses"]))
        self.assertIn("mandatory complete notebook", context)
        self.assertIn("unchanged checker", context)

    def test_actual_tree_roundtrip_preserves_quoted_prompt_and_all_raw_ordering(self):
        fixture, reread, schema = self.fixture()
        record = self.generate(fixture, reread, schema)
        original_target = record["calls"][0]["target"]
        self.assertNotEqual(list(original_target), sorted(original_target))
        self.assertNotEqual(list(original_target["earlier"]), sorted(original_target["earlier"]))
        fixture.tree.notebook["pair_drafts"] = record
        restored = from_json(ProblemTree, to_json(fixture.tree))
        reloaded = restored.notebook["pair_drafts"]
        self.assertEqual(list(reloaded["calls"][0]["target"]), sorted(original_target))
        self.assertEqual(drafts.quoted_context(reloaded), drafts.quoted_context(record))
        self.assertEqual([call["target"]["pair_id"] for call in reloaded["calls"]],
                         [call["target"]["pair_id"] for call in record["calls"]])
        self.assertEqual([[completion["text"] for completion in call["completions"]] for call in reloaded["calls"]],
                         [[completion["text"] for completion in call["completions"]] for call in record["calls"]])
        self.assertEqual(drafts.contract()["final_notebook_allowed_statuses"],
                         ["COMPLETE", "NO_PUBLIC_TARGETS", "PUBLIC_TARGET_BLOCKED"])
        fixture.tree = restored
        self.validate(reloaded, fixture, reread, schema)

    def test_resealed_target_prompt_seed_raw_completion_or_coverage_tampering_fails(self):
        fixture, reread, schema = self.fixture()
        original = self.generate(fixture, reread, schema)
        for mutation in ("target", "prompt", "seed", "text", "n4", "missing_call", "role"):
            record = copy.deepcopy(original)
            call = record["calls"][0]
            if mutation == "target":
                call["target"] = record["target_plan"][1]
            elif mutation == "prompt":
                call["messages"] = []
                call["messages_sha256"] = _hash([])
            elif mutation == "seed":
                call["seeds"][0] += 1
            elif mutation == "text":
                call["completions"][0]["text"] = "UNRESOLVED"
            elif mutation == "n4":
                call["completions"].pop()
            elif mutation == "missing_call":
                record["calls"].pop()
            else:
                record["eligible_for_rl_loss"] = True
            record["sha256"] = _hash({key: value for key, value in record.items() if key != "sha256"})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(record, fixture, reread, schema)

    def test_budget_stop_keeps_partial_raw_and_one_final_root_event(self):
        fixture, reread, schema = self.fixture()
        original_sample = fixture.engine.sample
        stopped = False

        def sample(requests):
            nonlocal stopped
            output = original_sample(requests)
            stopped = True
            return output

        fixture.engine.sample = sample
        events: list[dict] = []
        record = self.generate(fixture, reread, schema, should_stop=lambda: stopped, on_event=events.append)
        self.assertEqual(record["status"], "BUDGET_STOP")
        self.assertEqual(len(record["calls"]), 1)
        self.assertEqual(len(fixture.engine.requests), 1)
        self.assertEqual(sum(event["phase"] == "DRAFTS_COMPLETED" for event in events), 1)
        self.assertEqual(events[-1]["record"], record)
        self.validate(record, fixture, reread, schema)
        self.assertIn('"completed_targets":1', drafts.quoted_context(record))

    def test_context_reservation_boundary_and_preparation_failure_never_truncate_history(self):
        for offset in (0, 1, None):
            fixture, reread, schema = self.fixture()
            original_prepare = fixture.engine.prepare

            def prepare(request):
                if "PAIR_DRAFT" in request.request_id and offset is None:
                    raise ValueError("CONTEXT_BLOCKED: fixture preparation failure")
                actual = original_prepare(request)
                if "PAIR_DRAFT" in request.request_id and offset is not None:
                    actual.token_ids = [1] * (24576 - 128 + offset)
                return actual

            fixture.engine.prepare = prepare
            events: list[dict] = []
            record = self.generate(fixture, reread, schema, on_event=events.append)
            self.assertEqual(record["status"], "COMPLETE" if offset == 0 else "CONTEXT_BLOCKED")
            self.assertEqual(len(fixture.engine.requests), 3 if offset == 0 else 0)
            self.assertEqual(events[-1]["phase"], "DRAFTS_COMPLETED")
            self.validate(record, fixture, reread, schema)
            self.assertIn("UNVERIFIED", drafts.quoted_context(record))

    def test_public_blocked_and_empty_plan_records_are_explicit_without_fallback(self):
        fixture, reread, schema = self.fixture()
        fixture.engine.replies = [["Observed facts:\nunknown = first\nunknown = second\nSolution:"]]
        reread = fresh_observation.generate(fixture.actor, schema, fixture.images[fixture.actor.problem_id],
            fixture.engine, master_seed=3407, context_limit=24576)
        fixture.engine.requests.clear()
        record = self.generate(fixture, reread, schema)
        self.assertEqual(record["status"], "PUBLIC_TARGET_BLOCKED")
        self.assertEqual(record["calls"], [])
        self.assertEqual(fixture.engine.requests, [])
        self.validate(record, fixture, reread, schema)
        with patch.object(drafts, "_plan", return_value=([], None)):
            empty = self.generate(fixture, reread, schema)
            self.assertEqual(empty["status"], "NO_PUBLIC_TARGETS")
            self.validate(empty, fixture, reread, schema)
        self.assertIn('"completed_targets":0', drafts.quoted_context(empty))

    def test_actual_incomplete_generation_is_preserved_as_raw_then_rejected(self):
        fixture, reread, schema = self.fixture()
        original_sample = fixture.engine.sample

        def sample(requests):
            output = original_sample(requests)[0]
            return [replace(output, completions=output.completions[:3])]

        fixture.engine.sample = sample
        events: list[dict] = []
        with self.assertRaisesRegex(ValueError, "incomplete or mismatched actual n4"):
            self.generate(fixture, reread, schema, on_event=events.append)
        self.assertEqual([event["phase"] for event in events], ["STARTED", "GENERATED"])
        self.assertEqual(len(events[-1]["outputs"][0]["completions"]), 3)

    def test_changed_original_pixels_fail_reconstruction(self):
        fixture, reread, schema = self.fixture()
        record = self.generate(fixture, reread, schema)
        original_prepare = fixture.engine.prepare

        def prepare(request):
            actual = original_prepare(request)
            if "PAIR_DRAFT" in request.request_id:
                actual.pixels["pixel_values"] = torch.ones((1, 2))
            return actual

        fixture.engine.prepare = prepare
        with self.assertRaisesRegex(ValueError, "reconstructed original input differs"):
            self.validate(record, fixture, reread, schema)
