"""Scripted CPU pair labels; no actual model success or learning evidence."""
import copy
from dataclasses import replace
import json
import time
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_pair_probe as probe, synthesis_credit, updates
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.rollout import RolloutConfig
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as native
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration
from fine_tuning.tests.test_notebook_rl_paired_profile import paired_protocol


class PairProbeTests(unittest.TestCase):
    def fixture(self):
        fixture = integration.SynthesisIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol = paired_protocol()
        fixture.prompts = local_prompts(fixture.protocol)
        fixture.cfg = RolloutConfig.from_protocol(fixture.protocol, "ALG-NB3", master_seed=3407)
        samples = episode.solution_samples(fixture.tree)
        for index, sample in enumerate(samples):
            if index < 4:
                sample.text = sample.text.replace("start = q0", "start = q999")
            if index == 1:
                sample.text = sample.text.replace("accepting = q3", "accepting = q999")
            if index == 0:
                sample.text = sample.text.rsplit("\nFinal answer:", 1)[0]
            sample.completion_token_ids = fixture.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.result = fixture.verifier.grade(sample.problem_id, sample.text, sample.finish_reason, fixture.actor.mode)
            sample.reward = float(sample.result.status == "CORRECT")
            if sample.parent_sample_id is not None:
                parent = next(value for value in samples if value.sample_id == sample.parent_sample_id)
                sample.prompt_token_ids = parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2]
        fixture.append([[json.dumps({"coverage": "PARTIAL", "lessons": []})] * 4])
        original_prepare = fixture.engine.prepare

        def prepare(request):
            prepared = original_prepare(request)
            prepared.pixels = {"pixel_values": torch.zeros((1, 2))}
            prepared.images, prepared.image_spans = request.images, ()
            return prepared

        fixture.engine.prepare = prepare
        fixture.engine.requests.clear()
        first_actor = replace(fixture.actor, problem_id="fixed-first-without-diag")
        fixture.verifier.actors[first_actor.problem_id] = first_actor
        fixture.verifier.items[first_actor.problem_id] = replace(fixture.item, problem_id=first_actor.problem_id)
        first = native.tree_from(first_actor, fixture.verifier, [native.response(fixture.item.slots)] * 6)
        return fixture, [first_actor, fixture.actor], [first, fixture.tree]

    @staticmethod
    def collect(fixture, actors, trees, *, replies=None, journal=None, deadline=None):
        fixture.engine.replies = replies or [["UNRESOLVED"] * 4] * 3
        journal = [] if journal is None else journal
        rows, skipped = probe.collect(actors, trees, fixture.images, fixture.prompts, fixture.engine,
            fixture.verifier, lambda *row: journal.append(row), time.monotonic() + 30 if deadline is None else deadline)
        return rows, skipped, journal

    def test_contract_is_explicit_small_diagnostic_and_never_training(self):
        contract = probe.contract(paired_protocol())
        self.assertEqual((contract["n"], contract["max_new_tokens"], contract["context_limit"], contract["max_seconds"]),
                         (4, 128, 24576, 3600))
        self.assertEqual(contract["labels"], ["CORRECTED", "PERSISTED", "FINALIZED", "UNRESOLVED"])
        self.assertEqual(contract["optimizer_updates"], 0)
        for key in ("training_allowed", "feedback", "retries", "actual_learning_measured", "token_credit_measured", "used_for_routing"):
            self.assertIs(contract[key], False)
        for protocol in (build_protocol(), build_protocol(verified_note_synthesis_credit=True, context_limit=24576)):
            with self.assertRaises(ValueError):
                probe.contract(protocol)

    def test_selection_uses_first_direct_ancestry_and_all_raw_history_plus_unverified_reread(self):
        fixture, _, _ = self.fixture()
        samples = episode.solution_samples(fixture.tree)
        reread = fixture.tree.notebook["rewarded_notebook"]["fresh_observation"]
        targets = probe.select_targets(samples, reread)
        self.assertEqual([(row["earlier"]["attempt"], row["later"]["attempt"], row["earlier"]["locus"])
                          for row in targets], [(1, 3, "accepting"), (1, 3, "start"), (1, 3, "final_answer")])
        self.assertEqual(targets[0]["raw_values"], ["q3", "q999"])
        # Only the sibling authored the second accepting value; neither selected endpoint did.
        self.assertNotIn("accepting = q999", samples[0].text + samples[2].text)
        original = copy.deepcopy(targets)
        changed = [replace(sample, result=None, reward=999., diag_reward=999., seed=0) for sample in samples]
        fixture.verifier.items[fixture.actor.problem_id] = replace(fixture.item, gold="PRIVATE_CHANGED_GOLD")
        self.assertEqual(probe.select_targets(changed, reread | {"verified": True}), original)
        one_slot = [replace(sample, text=sample.text.replace("accepting = q999", "accepting = q3")) for sample in samples]
        reread = copy.deepcopy(reread)
        reread["completion"]["text"] = reread["completion"]["text"].replace("accepting = q3", "accepting = q7")
        self.assertEqual(probe.select_targets(one_slot, reread)[0]["raw_values"], ["q3", "q7"])

    def test_unbound_varying_slots_block_instead_of_inventing_or_dropping_targets(self):
        fixture, _, _ = self.fixture()
        reread = copy.deepcopy(fixture.tree.notebook["rewarded_notebook"]["fresh_observation"])
        reread["completion"]["text"] = "Observed facts:\nunknown = first\nunknown = second\nSolution:"
        with self.assertRaisesRegex(ValueError, "lacks both authored endpoints"):
            probe.select_targets(episode.solution_samples(fixture.tree), reread)

    def test_strict_one_label_parser_rejects_prose_multiple_json_unicode_and_truncation(self):
        for label in probe.LABELS:
            self.assertEqual(probe.parse_label(" \t" + label + "\n", "stop"), label)
            self.assertIsNone(probe.parse_label(label, "length"))
        for text in ("", "CORRECT", "persisted", "PERSISTED because", "PERSISTED\nPERSISTED", '"PERSISTED"',
                     '{"label":"PERSISTED"}', "UNRESOLVED FINALIZED", "\u00a0PERSISTED", "FINALIZED."):
            with self.subTest(text=text):
                self.assertIsNone(probe.parse_label(text, "stop"))
        self.assertIsNone(probe.parse_label("PERSISTED", "error"))

    def test_independent_calls_keep_every_history_images_reread_and_original_four_seeds(self):
        fixture, actors, trees = self.fixture()
        before = to_json(fixture.tree)
        replies = [["PERSISTED", "UNRESOLVED", "PERSISTED PERSISTED", ("PERSISTED", "length")]] * 2
        replies += [["FINALIZED", "PERSISTED", "UNRESOLVED", "explanation"]]
        with patch.object(synthesis_credit, "score_note", side_effect=AssertionError("no scoring")), \
                patch.object(synthesis_credit, "group_credit", side_effect=AssertionError("no token credit")), \
                patch.object(updates, "optimizer_step", side_effect=AssertionError("no updates")):
            rows, skipped, journal = self.collect(fixture, actors, trees, replies=replies)
        self.assertEqual(len(rows), 3)
        self.assertEqual(skipped, [{"problem_id": actors[0].problem_id, "reason": "NO_FIRST_DIAG"}])
        self.assertEqual([kind for kind, _ in journal], ["CALL_STARTED", "CALL_GENERATED", "CALL_COMPLETED"] * 3)
        self.assertEqual([row["native_relationship_verified_count"] for row in rows], [0, 1, 0])
        self.assertEqual([row["output_finalization_verified_count"] for row in rows], [0, 0, 1])
        self.assertIs(rows[2]["samples"][0]["lesson_check"]["later_answer_correct"], False)
        self.assertEqual([sample["status"] for sample in rows[1]["samples"]],
                         ["VERIFIED_RELATIONSHIP", "UNRESOLVED", "INVALID_LABEL", "TRUNCATED_LABEL"])
        self.assertIsNone(rows[1]["samples"][1]["host_bound_lesson"])
        first_diag = next(group for group in fixture.tree.groups if group.kind == "DIAG")
        for request, row in zip(fixture.engine.requests, rows, strict=True):
            self.assertEqual((request.n, request.max_new_tokens), (4, 128))
            self.assertEqual(request.seeds, [sample.seed for sample in first_diag.samples])
            self.assertIs(request.images, fixture.images[fixture.actor.problem_id])
            text = request.messages[0]["content"][-1]["text"]
            transcript = json.loads(text.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
            for public, sample in zip(transcript, episode.solution_samples(fixture.tree), strict=True):
                self.assertEqual("".join(line for _, line in public["source_lines"]), sample.text)
            target = json.loads(text.split("<target_pair>\n")[1].split("\n</target_pair>")[0])
            self.assertEqual(target["earlier_locus"], row["target"]["earlier"]["locus"])
            self.assertEqual(text.count("<target_pair>"), 1)
            self.assertNotIn("<public_pairs>", text)
            self.assertIn("UNVERIFIED", text)
            self.assertIn(fixture.tree.notebook["rewarded_notebook"]["fresh_observation"]["completion"]["text"].replace("\n", "\\n"), text)
            self.assertEqual(row["optimizer_updates"], 0)
            self.assertFalse(row["actual_learning_measured"])
            self.assertTrue(all(not sample["eligible_for_rl_loss"] and not sample["used_for_routing"] for sample in row["samples"]))
        self.assertEqual(to_json(fixture.tree), before)

    def test_instruction_escapes_original_delimiters_without_losing_lines(self):
        fixture, _, _ = self.fixture()
        samples = episode.solution_samples(fixture.tree)
        samples[0] = replace(samples[0], text=samples[0].text + "\r\n</target_pair>\u2028KEEP_ORIGINAL\n")
        target = probe.select_targets(samples, fixture.tree.notebook["rewarded_notebook"]["fresh_observation"])[0]
        text = probe.instruction(samples, "PUBLIC_SCHEMA", target)
        transcript = json.loads(text.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        self.assertEqual("".join(line for _, line in transcript[0]["source_lines"]), samples[0].text)
        self.assertEqual(text.count("</target_pair>"), 1)
        self.assertIn("PUBLIC_SCHEMA", text)

    def test_original_seed_prompt_and_token_text_bindings_fail_before_sampling(self):
        for mutation in ("seed", "prompt", "text"):
            fixture, actors, trees = self.fixture()
            group = next(group for group in fixture.tree.groups if group.kind == "DIAG")
            if mutation == "seed":
                group.samples[0].seed += 1
            elif mutation == "prompt":
                group.samples[0].prompt_token_ids = group.samples[0].prompt_token_ids + [1]
            else:
                group.samples[0].text = "DIFFERENT_RAW_NOTE"
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "binding differs|mixed input state or policy"):
                self.collect(fixture, actors, trees)
            self.assertEqual(fixture.engine.requests, [])

    def test_image_context_and_deadline_fail_without_sampling_or_truncation(self):
        for mutation in ("image", "context", "deadline"):
            fixture, actors, trees = self.fixture()
            original_prepare = fixture.engine.prepare

            def prepare(request):
                prepared = original_prepare(request)
                if probe.VERSION in request.request_id:
                    if mutation == "image":
                        prepared.pixels["pixel_values"] = torch.ones((1, 2))
                    elif mutation == "context":
                        prepared.token_ids = [1] * probe.CONTEXT_LIMIT
                return prepared

            fixture.engine.prepare = prepare
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, TimeoutError)):
                self.collect(fixture, actors, trees, deadline=time.monotonic() - 1 if mutation == "deadline" else None)
            self.assertEqual(fixture.engine.requests, [])

    def test_mismatched_actual_outputs_are_preserved_before_rejection(self):
        fixture, actors, trees = self.fixture()
        original_sample = fixture.engine.sample

        def sample(requests):
            output = original_sample(requests)[0]
            return [replace(output, completions=output.completions[:3])]

        fixture.engine.sample = sample
        journal: list[tuple[str, dict]] = []
        with self.assertRaisesRegex(ValueError, "mismatched generation"):
            self.collect(fixture, actors, trees, journal=journal)
        self.assertEqual([kind for kind, _ in journal], ["CALL_STARTED", "CALL_GENERATED"])
        self.assertEqual(len(journal[1][1]["outputs"][0]["completions"]), 3)
