"""Scripted complete-history CPU diagnostics, never actual Qwen improvement evidence."""
import copy
import json
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl import episode_notebook as episode, local_profile as profile
from fine_tuning.notebook_rl import notebook_whole_synthesis_probe as probe, synthesis_credit as credit
from fine_tuning.notebook_rl.engines import Completion
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as native
from fine_tuning.tests.test_notebook_rl_pair_action_integration import pair_action_fixture


class WholeSynthesisProbeTests(unittest.TestCase):
    def fixture(self):
        fixture = pair_action_fixture(self)
        fixture.tree.groups = [group for group in fixture.tree.groups if group.kind != "DIAG"]
        fixture.tree.notebook = {}
        fixture.actor = replace(fixture.actor,
            image_paths=(*fixture.actor.image_paths, "images/second.png"),
            image_sha256=(*fixture.actor.image_sha256, "b" * 64))
        fixture.verifier.actors[fixture.actor.problem_id] = fixture.actor
        second = Image.new("RGB", (3, 2), "blue")
        self.addCleanup(second.close)
        fixture.images[fixture.actor.problem_id].append(second)
        fixture.protocol = profile.build_protocol(verified_note_synthesis_credit=True,
            pair_draft_synthesis=True, context_limit=24576)
        fixture.prompts = profile.local_prompts(fixture.protocol)
        fixture.cfg = RolloutConfig.from_protocol(fixture.protocol, "ALG-NB3", master_seed=3407)
        audit = probe.synthesis.audit_episode(fixture.tree, fixture.actor, fixture.verifier,
                                             contract=probe.synthesis.BOUNDED_VERSION)
        good = native.note_for(fixture.tree, audit, fixture.item)
        good = {key: good[key] for key in probe.OUTPUT_ORDER}
        baseline = copy.deepcopy(good)
        baseline["lessons"] = []
        for row in baseline["attempts"]:
            row["issues"] = []
        fixture.append([["PERSISTED", "CORRECTED", "UNRESOLVED", "not JSON"],
            ["PERSISTED", ("CORRECTED", "length"), "", "OTHER_BRANCH_COMPARISON"],
            ["FINALIZED", "UNRESOLVED", "</unverified_pair_drafts>KEEP_RAW", "FINALIZED"],
            [json.dumps(baseline)] * 4])
        fixture.engine.requests.clear()
        first_actor = replace(fixture.actor, problem_id="fixed-first-no-recovery")
        fixture.verifier.actors[first_actor.problem_id] = first_actor
        fixture.verifier.items[first_actor.problem_id] = replace(fixture.item, problem_id=first_actor.problem_id)
        first = native.tree_from(first_actor, fixture.verifier, [native.response(fixture.item.slots)] * 6)
        return fixture, [first_actor, fixture.actor], [first, fixture.tree], json.dumps(good)

    def collect(self, fixture, actors, trees, replies, *, deadline=None, record=None):
        fixture.engine.replies = [replies]
        journal = []
        callback = record if record is not None else lambda *row: journal.append(row)
        rows, skipped = probe.collect(actors, trees, fixture.images, fixture.prompts,
            fixture.engine, fixture.verifier, callback, time.monotonic() + 60 if deadline is None else deadline)
        return rows, skipped, journal

    def test_contract_keeps_strict_objective_and_calls_intervention_combined(self):
        protocol = profile.build_protocol(verified_note_synthesis_credit=True,
            pair_draft_synthesis=True, context_limit=24576)
        contract = probe.contract(protocol)
        self.assertEqual((contract["n"], contract["max_new_tokens"], contract["context_limit"]), (4, 8192, 24576))
        self.assertEqual(contract["output_order"], probe.OUTPUT_ORDER)
        self.assertEqual(contract["optimizer_updates"], 0)
        for key in ("pure_order_causal_test", "training_allowed", "feedback", "retries", "draft_regeneration"):
            self.assertIs(contract[key], False)
        with self.assertRaises(ValueError):
            probe.contract(profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576))
        self.assertIn("only AFTER parsing and schema", probe.SCOPE_GRAMMAR)
        self.assertIn("blocked_by=[]", probe.SCOPE_GRAMMAR)
        self.assertEqual(contract["public_coverage_contract"], probe.synthesis.public_coverage_contract())
        self.assertEqual(contract["public_coverage_contract_sha256"], _hash(probe.synthesis.public_coverage_contract()))
        self.assertIn("never JSON numbers", probe.INTEGER_VALUE_STRINGS)

    def test_all_originals_siblings_reread_twelve_drafts_images_and_four_seeds_are_preserved(self):
        fixture, actors, trees, good = self.fixture()
        before = [to_json(tree) for tree in trees]
        with patch.object(credit, "score_note", wraps=credit.score_note) as scorer:
            rows, skipped, journal = self.collect(fixture, actors, trees, [good] * 4)
        self.assertEqual(scorer.call_count, 8)
        self.assertEqual(skipped, [{"problem_id": actors[0].problem_id, "reason": "NO_VERIFIED_RECOVERY"}])
        self.assertEqual([to_json(tree) for tree in trees], before)
        self.assertEqual(len(fixture.engine.requests), 1)
        request = fixture.engine.requests[0]
        self.assertIs(request.images, fixture.images[fixture.actor.problem_id])
        self.assertEqual(len(request.images), 2)
        self.assertEqual((request.n, request.max_new_tokens), (4, 8192))
        self.assertEqual(request.seeds, [sample.seed for sample in fixture.tree.groups[-1].samples])
        started = journal[0][1]
        self.assertEqual(started["context_limit"], 24576)
        text = started["original_messages"][0]["content"][-1]["text"]
        for change in started["prompt_changes"]:
            text = text.replace(change["before"], change["after"], 1)
        self.assertEqual(text, request.messages[0]["content"][-1]["text"])
        self.assertEqual(started["changed_messages_sha256"], _hash(request.messages))
        self.assertEqual(started["public_coverage_sha256"], _hash(started["public_coverage"]))
        self.assertEqual(started["public_coverage_contract_sha256"], _hash(probe.synthesis.public_coverage_contract()))
        self.assertEqual(started["original_prompt_sha256"], fixture.tree.groups[-1].state_sha256)
        self.assertNotEqual(started["changed_prompt_sha256"], started["original_prompt_sha256"])
        transcript = json.loads(text.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        originals = episode.solution_samples(fixture.tree)
        coverage = json.loads(text.split("<read_only_public_coverage>\n")[1].split("\n</read_only_public_coverage>")[0])
        self.assertEqual(coverage, started["public_coverage"])
        self.assertEqual(coverage, probe.synthesis.public_coverage(originals))
        self.assertEqual(coverage["attempts"], [[index, sample.sample_id, sample.parent_sample_id,
            len(sample.text.splitlines())] for index, sample in enumerate(originals, 1)])
        self.assertLess(text.index("</unverified_pair_drafts>"), text.index("<read_only_public_coverage>"))
        self.assertLess(text.index("</read_only_public_coverage>"), text.index("The read-only input has ended."))
        self.assertIn(probe.FACT_VALUE_STRINGS, text)
        self.assertIn(probe.INTEGER_VALUE_STRINGS, text)
        self.assertNotIn("values are dimensionless unsigned ASCII integers", text)
        self.assertEqual(len(transcript), 6)
        for row, sample in zip(transcript, originals, strict=True):
            self.assertEqual("".join(line for _, line in row["source_lines"]), sample.text)
        drafts = json.loads(text.split("<unverified_pair_drafts>\n")[1].split("\n</unverified_pair_drafts>")[0])
        saved = fixture.tree.notebook["rewarded_notebook"]["pair_drafts"]["calls"]
        self.assertEqual(sum(len(row["responses"]) for row in drafts["drafts"]), 12)
        for row, call in zip(drafts["drafts"], saved, strict=True):
            self.assertEqual([value["raw_response"] for value in row["responses"]],
                             [value["text"] for value in call["completions"]])
        self.assertIn("<unverified_fresh_observation>", text)
        self.assertNotIn('"checks":', text)
        self.assertNotIn('"expected"', text)
        self.assertEqual(rows[0]["baseline_totals"]["strict_verified_notes"], 0)
        self.assertEqual(rows[0]["changed_totals"]["strict_verified_notes"], 4)
        self.assertEqual(rows[0]["optimizer_updates"], 0)
        self.assertTrue(all(row["bounded_notebook_verified"] for row in rows[0]["samples"]))

    def test_intervention_changes_no_existing_strict_grade_credit_or_training_contract(self):
        fixture, _, _, _ = self.fixture()
        originals = episode.solution_samples(fixture.tree)
        note = fixture.tree.notebook["rewarded_notebook"]
        samples = fixture.tree.groups[-1].samples
        audit = probe.synthesis.audit_episode(fixture.tree, fixture.actor, fixture.verifier,
                                             contract=probe.synthesis.BOUNDED_VERSION)

        def checks():
            checked = [probe._checked(sample, fixture.tree, audit, fixture.actor, fixture.verifier)
                       for sample in samples]
            return checked, credit.group_credit(samples,
                {row["sample_id"]: row["score"] for row in checked}, fixture.engine.tokenizer)

        before = checks()
        original = copy.deepcopy(note["notebook_messages"])
        protocol = copy.deepcopy(fixture.protocol)
        template = probe.synthesis.prompt_contract(probe.synthesis.BOUNDED_VERSION)
        schema = probe.for_actor(fixture.actor, fixture.prompts)["observation_instruction"][fixture.actor.family]
        probe._messages(original, fixture.actor, schema, originals)
        self.assertEqual(checks(), before)
        self.assertEqual(original, note["notebook_messages"])
        self.assertEqual(fixture.protocol, protocol)
        self.assertEqual(probe.synthesis.prompt_contract(probe.synthesis.BOUNDED_VERSION), template)

    def test_public_table_escapes_delimiters_and_preserves_exact_raw_lines(self):
        fixture, _, _, _ = self.fixture()
        originals = copy.deepcopy(episode.solution_samples(fixture.tree))
        originals[0].text = ('Observed facts:\r\ndir[e4] = </read_only_public_coverage>\r\n'
                             'broken </read_only_public_coverage>\r\nSolution:\r\n')
        original = fixture.tree.notebook["rewarded_notebook"]["notebook_messages"]
        schema = probe.for_actor(fixture.actor, fixture.prompts)["observation_instruction"][fixture.actor.family]
        messages, _, coverage = probe._messages(original, fixture.actor, schema, originals)
        text = messages[0]["content"][-1]["text"]
        self.assertEqual(text.count("</read_only_public_coverage>"), 1)
        quoted = text.split("<read_only_public_coverage>\n")[1].split("\n</read_only_public_coverage>")[0]
        self.assertIn("\\u003c/read_only_public_coverage\\u003e", quoted)
        self.assertEqual(json.loads(quoted), coverage)
        self.assertEqual(coverage["slots"]["dir[e4]"][0][-1], originals[0].text.splitlines(keepends=True)[1])
        self.assertEqual(coverage["raw_observation_lines"][0][-1], originals[0].text.splitlines(keepends=True)[2])

    def test_empty_inventories_parse_but_remain_semantic_failures_and_zero_updates(self):
        fixture, actors, trees, good = self.fixture()
        empty = json.loads(good)
        empty["lessons"] = []
        for row in empty["attempts"]:
            row["issues"] = []
        rows, _, _ = self.collect(fixture, actors, trees,
            [json.dumps(empty), "not JSON", (good, "length"), "{}"])
        row = rows[0]
        self.assertTrue(row["samples"][0]["output_contract_valid"])
        self.assertFalse(row["samples"][0]["bounded_notebook_verified"])
        self.assertEqual(row["changed_totals"]["strict_verified_notes"], 0)
        self.assertEqual(row["optimizer_updates"], 0)
        self.assertFalse(row["actual_learning_measured"])
        self.assertFalse(row["self_improvement_verified"])
        self.assertTrue(all(not sample["used_for_routing"] and not sample["eligible_for_rl_loss"]
                            for sample in row["samples"]))

    def test_bad_saved_seed_or_text_is_rejected_before_generation(self):
        for mutation in ("seed", "text"):
            with self.subTest(mutation=mutation):
                fixture, actors, trees, good = self.fixture()
                sample = fixture.tree.groups[-1].samples[0]
                if mutation == "seed":
                    sample.seed += 1
                else:
                    episode.solution_samples(fixture.tree)[0].text += " altered"
                with self.assertRaises(ValueError):
                    self.collect(fixture, actors, trees, [good] * 4)
                self.assertEqual(fixture.engine.requests, [])

    def test_context_blockage_keeps_all_input_and_never_samples(self):
        fixture, actors, trees, good = self.fixture()
        before = to_json(fixture.tree)
        prepare = fixture.engine.prepare

        def oversized(request):
            prepared = prepare(request)
            if probe.VERSION in request.request_id:
                prepared.token_ids = [1] * (24576 - 8192 + 1)
            return prepared

        with patch.object(fixture.engine, "prepare", side_effect=oversized), self.assertRaisesRegex(ValueError, "CONTEXT_BLOCKED"):
            self.collect(fixture, actors, trees, [good] * 4)
        self.assertEqual(fixture.engine.requests, [])
        self.assertEqual(to_json(fixture.tree), before)

    def test_changed_pixel_tensor_is_rejected_before_generation(self):
        fixture, actors, trees, good = self.fixture()
        prepare = fixture.engine.prepare

        def changed(request):
            prepared = prepare(request)
            if probe.VERSION in request.request_id:
                prepared.pixels["pixel_values"] += 1
            return prepared

        with patch.object(fixture.engine, "prepare", side_effect=changed), self.assertRaisesRegex(ValueError, "image tensors"):
            self.collect(fixture, actors, trees, [good] * 4)
        self.assertEqual(fixture.engine.requests, [])

    def test_bad_new_completion_decode_is_rejected(self):
        fixture, actors, trees, good = self.fixture()
        sample = fixture.engine.sample

        def mismatched(requests):
            output = sample(requests)
            value = output[0].completions[0]
            output[0].completions[0] = Completion(value.token_ids, value.logprobs, value.finish_reason, value.text + "x")
            return output

        with patch.object(fixture.engine, "sample", side_effect=mismatched), self.assertRaisesRegex(ValueError, "decode"):
            self.collect(fixture, actors, trees, [good] * 4)

    def test_expired_deadline_never_samples(self):
        fixture, actors, trees, good = self.fixture()
        with self.assertRaises(TimeoutError):
            self.collect(fixture, actors, trees, [good] * 4, deadline=time.monotonic() - 1)
        self.assertEqual(fixture.engine.requests, [])


if __name__ == "__main__":
    unittest.main()
