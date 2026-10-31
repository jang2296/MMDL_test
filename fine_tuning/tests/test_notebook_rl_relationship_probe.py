"""Scripted CPU contracts, not actual model proposals, learning, or improvement."""
import copy
import json
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_relationship_probe as probe
from fine_tuning.notebook_rl import notebook_synthesis as synthesis, synthesis_credit as credit
from fine_tuning.notebook_rl.local_profile import build_protocol
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as native
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration


class RelationshipProbeTests(unittest.TestCase):
    def fixture(self, *, missing_final=False, mcq=False):
        fixture = integration.SynthesisIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        if mcq:
            fixture.actor = replace(fixture.actor, choices=("accept", "reject"), answer_format="mcq")
            fixture.item = replace(fixture.item, choices=fixture.actor.choices, answer_type="mcq", gold="A")
            fixture.verifier.actors[fixture.actor.problem_id] = fixture.actor
            fixture.verifier.items[fixture.actor.problem_id] = fixture.item
            for sample in episode.solution_samples(fixture.tree):
                sample.text = sample.text.replace("Final answer: reject", "Final answer: B").replace("Final answer: accept", "Final answer: A")
                sample.completion_token_ids = fixture.engine.tokenizer.encode(sample.text) + [2]
                sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
                sample.result = fixture.verifier.grade(sample.problem_id, sample.text, "stop", fixture.actor.mode)
                sample.reward = float(sample.result.status == "CORRECT")
        if missing_final:
            for sample in episode.solution_samples(fixture.tree)[:4]:
                sample.text = sample.text.rsplit("\nFinal answer:", 1)[0]
                sample.completion_token_ids = fixture.engine.tokenizer.encode(sample.text) + [2]
                sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
                sample.result = fixture.verifier.grade(sample.problem_id, sample.text, "stop", fixture.actor.mode)
                sample.reward = 0.
        if missing_final or mcq:
            samples = episode.solution_samples(fixture.tree)
            for index, sample in enumerate(samples):
                if sample.parent_sample_id is not None:
                    parent = next(s for s in samples if s.sample_id == sample.parent_sample_id)
                    sample.prompt_token_ids = parent.prompt_token_ids + parent.completion_token_ids + [900 + index//2]
        audit = synthesis.audit_episode(fixture.tree, fixture.actor, fixture.verifier, contract=synthesis.BOUNDED_VERSION)
        correct = native.note_for(fixture.tree, audit, fixture.item)
        baseline = copy.deepcopy(correct)
        baseline["lessons"] = []
        fixture.append([[json.dumps(baseline)] * 4, *([[json.dumps(correct)] * 4] * (2 if mcq else 1))])
        original_prepare = fixture.engine.prepare

        def prepare(request):
            prepared = original_prepare(request)
            prepared.pixels = {"pixel_values": torch.zeros((1, 2))}
            prepared.images = request.images
            prepared.image_spans = ()
            return prepared

        fixture.engine.prepare = prepare
        fixture.engine.requests.clear()
        first_actor = replace(fixture.actor, problem_id="unrecovered-fixed-first")
        fixture.verifier.actors[first_actor.problem_id] = first_actor
        fixture.verifier.items[first_actor.problem_id] = replace(fixture.item, problem_id=first_actor.problem_id)
        wrong = native.response(fixture.item.slots)
        if mcq:
            wrong = wrong.replace("Final answer: reject", "Final answer: B")
        first = native.tree_from(first_actor, fixture.verifier, [wrong] * 6)
        return fixture, [first_actor, fixture.actor], [first, fixture.tree], correct

    def collect(self, fixture, actors, trees, replies, *, deadline=None, comparisons=False):
        fixture.engine.replies = [replies]
        journal = []
        rows, skipped = probe.collect(actors, trees, fixture.images, fixture.prompts, fixture.engine,
            fixture.verifier, lambda *row: journal.append(row), time.monotonic()+30 if deadline is None else deadline,
            comparisons=comparisons)
        return rows, skipped, journal

    def test_distinct_contract_has_only_relation_output_and_no_training(self):
        protocol = build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        contract = probe.contract(protocol)
        self.assertEqual((contract["n"], contract["max_new_tokens"], contract["context_limit"]), (4, 8192, 24576))
        for key in ("training_allowed", "feedback", "retries", "actual_learning_measured", "full_notebook_verified"):
            self.assertIs(contract[key], False)
        self.assertEqual(contract["optimizer_updates"], 0)
        self.assertNotIn("EXACTLY corrected_facts", contract["prompt"])
        self.assertNotIn("success_attempt", contract["prompt"])
        self.assertIn("lessons=[] is an honest", contract["prompt"])
        self.assertEqual(probe.NATIVE_RULES, synthesis.PROMPT.split("Native rules:", 1)[1].split("No extra claim fields", 1)[0])
        with self.assertRaises(ValueError):
            probe.contract(build_protocol())

    def test_original_all_history_reread_images_seeds_and_all_new_outputs_are_freshly_scored(self):
        fixture, actors, trees, correct = self.fixture()
        lesson = next(row for row in correct["lessons"] if row["relation"] == "CORRECTED")
        raw = json.dumps({"lessons": [lesson]})
        before = to_json(fixture.tree)
        with patch.object(credit, "score_note", wraps=credit.score_note) as scorer:
            rows, skipped, journal = self.collect(fixture, actors, trees, [raw, "not JSON", (raw, "length"), '{"lessons":[]}'])
        self.assertEqual(scorer.call_count, 4)
        self.assertEqual([call.args[0] for call in scorer.call_args_list], [raw, "not JSON", raw, '{"lessons":[]}'])
        self.assertEqual(skipped, [{"problem_id": actors[0].problem_id, "reason": "NO_VERIFIED_RECOVERY"}])
        self.assertEqual(len(fixture.engine.requests), 1)
        request = fixture.engine.requests[0]
        first_diag = next(g for g in fixture.tree.groups if g.kind == "DIAG")
        self.assertEqual(request.seeds, [s.seed for s in first_diag.samples])
        self.assertEqual((request.n, request.max_new_tokens), (4, 8192))
        self.assertIs(request.images, fixture.images[fixture.actor.problem_id])
        text = request.messages[0]["content"][-1]["text"]
        actual = json.loads(text.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        samples = episode.solution_samples(fixture.tree)
        self.assertEqual(len(actual), len(samples))
        for row, sample in zip(actual, samples, strict=True):
            self.assertEqual("".join(line for _, line in row["source_lines"]), sample.text)
        pairs = json.loads(text.split("<public_retry_pairs>\n")[1].split("\n</public_retry_pairs>")[0])
        self.assertEqual(pairs, [[row["parent"], row["attempt"]] for row in episode.public_transcript(samples) if row["parent"] is not None])
        self.assertIn("UNVERIFIED", text)
        self.assertIn(fixture.tree.notebook["rewarded_notebook"]["fresh_observation"]["completion"]["text"].replace("\n", "\\n"), text)
        self.assertEqual([kind for kind, _ in journal], ["CALL_STARTED", "CALL_GENERATED", "CALL_COMPLETED"])
        self.assertEqual([row["status"] for row in rows[0]["samples"]], ["PROPOSAL", "INVALID_JSON", "LENGTH", "PROPOSAL"])
        self.assertTrue(all(row["score"]["strict_reward"] == 0 for row in rows[0]["samples"]))
        self.assertEqual(rows[0]["samples"][1]["text"], "not JSON")
        self.assertEqual(sum(rows[0]["samples"][2]["score"]["relationship_rewards"].values()), 0)
        self.assertTrue(rows[0]["potential_centered_token_signal"])
        self.assertFalse(rows[0]["actual_learning_measured"])
        self.assertEqual(to_json(fixture.tree), before)

    def test_wrong_kind_rule_and_evidence_receive_no_verified_relationship_credit(self):
        fixture, actors, trees, correct = self.fixture()
        lesson = next(row for row in correct["lessons"] if row["relation"] == "CORRECTED")
        bad = [copy.deepcopy(lesson) for _ in range(3)]
        bad[0]["earlier"]["kind"] = "OUTPUT"
        bad[1]["rule"] = "WRONG_RULE"
        bad[2]["later"]["evidence_ranges"] = [[999, 999]]
        rows, _, _ = self.collect(fixture, actors, trees,
            [json.dumps({"lessons": [item]}) for item in [*bad, lesson]])
        units = [row["verified_credit_units"][credit.NATIVE_ERROR_RELATIONSHIP] for row in rows[0]["samples"]]
        self.assertEqual(units, [0, 0, 0, 1])
        self.assertTrue(rows[0]["potential_centered_token_signal"])

    def test_mcq_context_keeps_question_choices_and_original_control_without_solve_instruction(self):
        fixture, actors, trees, _ = self.fixture(mcq=True)
        original_control = copy.deepcopy(fixture.tree.notebook["rewarded_notebook"]["notebook_messages"])
        control_text = original_control[0]["content"][-1]["text"]
        self.assertIn("Please select the correct answer from the options above.\n", control_text)
        self.collect(fixture, actors, trees, ['{"lessons":[]}']*4)
        text = fixture.engine.requests[0].messages[0]["content"][-1]["text"]
        self.assertIn("Question: " + fixture.actor.question + "\nOptions:\nA. accept\nB. reject\n", text)
        self.assertNotIn("Please select the correct answer from the options above.", text)
        self.assertIn("UNVERIFIED", text)
        self.assertEqual(fixture.tree.notebook["rewarded_notebook"]["notebook_messages"], original_control)

    def test_zero_identical_valid_and_divergent_credit_use_complete_original_n4(self):
        for condition in ("zero", "identical", "divergent"):
            with self.subTest(condition=condition):
                fixture, actors, trees, correct = self.fixture()
                lesson = next(row for row in correct["lessons"] if row["relation"] == "CORRECTED")
                good, empty = json.dumps({"lessons": [lesson]}), '{"lessons":[]}'
                replies = [empty]*4 if condition == "zero" else [good]*4 if condition == "identical" else [good, empty, empty, empty]
                rows, _, _ = self.collect(fixture, actors, trees, replies)
                row = rows[0]
                self.assertEqual(row["same_unit_rewards_across_all_four"], condition != "divergent")
                self.assertEqual(row["potential_centered_token_signal"], condition == "divergent")
                self.assertEqual(row["absolute_relationship_units"], 0 if condition == "zero" else 4 if condition == "identical" else 1)
                self.assertTrue(all(item["full_group_size"] == 4 for item in row["actual_group_credit"].values()))
                self.assertTrue(row["proposal_credit_comparison_eligible"])

    def test_verified_other_branch_comparison_remains_noncredit(self):
        fixture, actors, trees, correct = self.fixture()
        lesson = copy.deepcopy(next(row for row in correct["lessons"] if row["earlier"]["attempt"] == 1))
        lesson["relation"] = "OTHER_BRANCH_COMPARISON"
        lesson["later"] = lesson["earlier"] | {"attempt": 2}
        rows, _, _ = self.collect(fixture, actors, trees, [json.dumps({"lessons": [lesson]})]*4)
        self.assertTrue(all(row["lesson_checks"][0]["verified"] for row in rows[0]["samples"]))
        self.assertEqual(rows[0]["absolute_relationship_units"], 0)
        self.assertFalse(rows[0]["potential_centered_token_signal"])

    def test_output_finalization_and_native_error_units_remain_separate(self):
        fixture, actors, trees, correct = self.fixture(missing_final=True)
        source = episode.solution_samples(fixture.tree)
        earlier = {"attempt": 1, "kind": "OUTPUT", "rule": "FINALIZATION", "locus": "final_answer",
                   "evidence_ranges": [[1, len(source[0].text.splitlines())]]}
        last = len(source[4].text.splitlines())
        final = {"rule": "FINALIZATION", "relation": "FINALIZED", "earlier": earlier,
                 "later": earlier | {"attempt": 5, "evidence_ranges": [[last, last]]}}
        native_lesson = next(row for row in correct["lessons"] if row["relation"] == "CORRECTED")
        rows, _, _ = self.collect(fixture, actors, trees,
            [json.dumps({"lessons": [final]}), json.dumps({"lessons": [native_lesson]}), '{"lessons":[]}', '{"lessons":[]}'])
        self.assertEqual(rows[0]["samples"][0]["verified_credit_units"], {credit.NATIVE_ERROR_RELATIONSHIP: 0, credit.OUTPUT_FINALIZATION: 1})
        self.assertEqual(rows[0]["samples"][1]["verified_credit_units"], {credit.NATIVE_ERROR_RELATIONSHIP: 1, credit.OUTPUT_FINALIZATION: 0})

    def test_full_notebook_violation_keeps_all_fresh_actual_scores_without_rewriting(self):
        fixture, actors, trees, correct = self.fixture()
        rows, _, journal = self.collect(fixture, actors, trees,
            [json.dumps(correct), '{"lessons":[]}', '{"lessons":[]}', '{"lessons":[]}'])
        self.assertEqual(len(journal[1][1]["completions"]), 4)
        self.assertEqual(rows[0]["samples"][0]["score"]["strict_reward"], 1.)
        self.assertTrue(rows[0]["samples"][0]["score"]["fact_rewards"])
        self.assertEqual(rows[0]["status"], "OUTPUT_CONTRACT_VIOLATION")
        self.assertFalse(rows[0]["proposal_credit_comparison_eligible"])
        self.assertEqual(len(rows[0]["actual_group_credit"]), 4)
        self.assertFalse(rows[0]["potential_centered_token_signal"])

    def test_deadline_and_original_seed_token_image_policy_binding_fail_before_generation(self):
        for mutation in ("deadline", "seed", "token", "image", "policy", "budget"):
            with self.subTest(mutation=mutation):
                fixture, actors, trees, _ = self.fixture()
                first = next(g for g in fixture.tree.groups if g.kind == "DIAG")
                if mutation == "seed":
                    first.samples[0].seed += 1
                if mutation == "token":
                    first.samples[0].prompt_token_ids = [99]
                if mutation == "policy":
                    fixture.engine.policy_version = 1
                original = fixture.engine.prepare

                def prepare(request):
                    item = original(request)
                    if probe.VERSION in request.request_id:
                        if mutation == "image":
                            item.pixels["pixel_values"] += 1
                        if mutation == "budget":
                            item.token_ids = [99] * 24576
                    return item

                fixture.engine.prepare = prepare
                with self.assertRaises((ValueError, TimeoutError)):
                    self.collect(fixture, actors, trees, ['{"lessons":[]}']*4,
                        deadline=time.monotonic()-1 if mutation == "deadline" else None)
                self.assertEqual(fixture.engine.requests, [])


class RelationshipComparisonTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = native.native()
        self.originals = episode.solution_samples(native.tree_from(self.actor, self.verifier,
            [native.response(self.item.slots)] * 4 + [native.response(self.item.slots, wrong=False)] * 2))

    def note(self, originals=None, lessons=None):
        originals = self.originals if originals is None else originals
        lessons = [] if lessons is None else lessons
        rows = []
        for public in episode.public_transcript(originals):
            if public["parent"] is None:
                continue
            parent, child = public["parent"], public["attempt"]
            for channel in ("NATIVE", "OUTPUT"):
                indices = [index for index, lesson in enumerate(lessons)
                    if (lesson["earlier"]["attempt"], lesson["later"]["attempt"]) == (parent, child)
                    and (lesson["relation"] == "FINALIZED") == (channel == "OUTPUT")]
                rows.append({"parent": parent, "child": child, "channel": channel,
                    "status": "PROPOSED" if indices else "UNKNOWN", "comparison": "I cannot establish the relationship.",
                    "earlier_evidence_ranges": [[1, 1]] if originals[parent - 1].text else [],
                    "later_evidence_ranges": [[1, 1]] if originals[child - 1].text else [], "lesson_indices": indices})
        return {"comparisons": rows, "lessons": lessons}

    def test_distinct_optional_contract_retains_v1_and_explicitly_limits_pair_scope(self):
        protocol = build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        old, new = probe.contract(protocol), probe.contract(protocol, comparisons=True)
        self.assertEqual(old["protocol_id"], probe.VERSION)
        self.assertEqual(old["prompt"], probe.PROMPT)
        self.assertNotIn("comparison_reward", old)
        self.assertEqual(new["protocol_id"], probe.COMPARISONS_VERSION)
        self.assertNotEqual(old["prompt_sha256"], new["prompt_sha256"])
        self.assertEqual(new["comparison_reward"], 0)
        self.assertFalse(new["comparison_semantics_verified"])
        self.assertFalse(new["comparison_status_used_for_routing"])
        self.assertIn("EXACTLY two keys, comparisons and lessons", new["prompt"])
        self.assertIn("listed direct parent-child pair", new["prompt"])
        self.assertNotIn("OTHER_BRANCH_COMPARISON requires", new["prompt"])
        self.assertNotIn("later.attempt null", new["prompt"])
        for key in ("n", "max_new_tokens", "context_limit", "max_seconds", "optimizer_updates", "seeds"):
            self.assertEqual(old[key], new[key])

    def test_public_pair_coverage_order_citations_and_exact_types_are_required(self):
        valid = self.note()
        self.assertTrue(probe._comparisons_valid(valid, self.originals))
        mutations = {
            "missed_pair": lambda note: note["comparisons"].pop(),
            "duplicate_pair": lambda note: note["comparisons"].__setitem__(2, copy.deepcopy(note["comparisons"][0])),
            "wrong_branch": lambda note: note["comparisons"][0].update(parent=2),
            "wrong_order": lambda note: note["comparisons"].reverse(),
            "wrong_channel": lambda note: note["comparisons"][0].update(channel="OUTPUT"),
            "invalid_citation": lambda note: note["comparisons"][0].update(earlier_evidence_ranges=[[999, 999]]),
            "overlapping_citation": lambda note: note["comparisons"][0].update(earlier_evidence_ranges=[[1, 1], [1, 1]]),
            "boolean_citation": lambda note: note["comparisons"][0].update(earlier_evidence_ranges=[[True, 1]]),
            "empty_citation": lambda note: note["comparisons"][0].update(earlier_evidence_ranges=[]),
            "boolean_parent": lambda note: note["comparisons"][0].update(parent=True),
            "boolean_index": lambda note: note["comparisons"][0].update(status="PROPOSED", lesson_indices=[True]),
            "missing_proposal": lambda note: note["comparisons"][0].update(status="PROPOSED"),
            "empty_comparison": lambda note: note["comparisons"][0].update(comparison=" \n"),
            "unknown_status": lambda note: note["comparisons"][0].update(status="VERIFIED"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                note = copy.deepcopy(valid)
                mutate(note)
                self.assertFalse(probe._comparisons_valid(note, self.originals))
        self.assertFalse(probe._comparisons_valid({"comparisons": [], "lessons": []}, self.originals))

    def test_comparisons_must_precede_lessons_in_original_json(self):
        note = self.note()
        for ordered, expected in ((note, True), ({"lessons": note["lessons"], "comparisons": note["comparisons"]}, False)):
            with self.subTest(order=list(ordered)):
                parsed, _, _ = credit._json_units(json.dumps(ordered))
                self.assertEqual(probe._comparisons_valid(parsed, self.originals), expected)
        self.assertIn("both lists IN THAT ORDER", probe.COMPARISONS_PROMPT)

    def test_empty_original_has_no_invented_line_and_uncertainty_is_unverified(self):
        self.originals[0].text = ""
        note = self.note()
        note["comparisons"][0]["status"] = "NO_SUPPORTED_ERROR"
        self.assertTrue(probe._comparisons_valid(note, self.originals))
        self.assertEqual(note["comparisons"][0]["earlier_evidence_ranges"], [])
        note["comparisons"][0]["earlier_evidence_ranges"] = [[1, 1]]
        self.assertFalse(probe._comparisons_valid(note, self.originals))

    def test_proposed_lesson_indices_bind_channels_endpoints_and_original_citations(self):
        tree = native.tree_from(self.actor, self.verifier, [sample.text for sample in self.originals])
        audit = synthesis.audit_episode(tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        lesson = next(row for row in native.note_for(tree, audit, self.item)["lessons"]
            if row["earlier"]["attempt"] == 3 and row["relation"] == "CORRECTED")
        valid = self.note(lessons=[lesson])
        self.assertTrue(probe._comparisons_valid(valid, self.originals))
        selected = next(index for index, row in enumerate(valid["comparisons"]) if row["lesson_indices"])
        mutations = {
            "boolean_index": lambda note: note["comparisons"][selected].update(lesson_indices=[False]),
            "out_of_range": lambda note: note["comparisons"][selected].update(lesson_indices=[1]),
            "negative_index": lambda note: note["comparisons"][selected].update(lesson_indices=[-1]),
            "duplicate_index": lambda note: note["comparisons"][selected].update(lesson_indices=[0, 0]),
            "wrong_endpoint": lambda note: note["lessons"][0]["later"].update(attempt=6),
            "wrong_channel_endpoint": lambda note: note["lessons"][0]["earlier"].update(kind="OUTPUT"),
            "wrong_citation": lambda note: note["lessons"][0]["earlier"].update(evidence_ranges=[[999, 999]]),
            "unreferenced_lesson": lambda note: note["comparisons"][selected].update(status="UNKNOWN", lesson_indices=[]),
            "uncertain_with_reference": lambda note: note["comparisons"][selected].update(status="UNKNOWN"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                note = copy.deepcopy(valid)
                mutate(note)
                self.assertFalse(probe._comparisons_valid(note, self.originals))

    def test_raw_lesson_credit_excludes_comparisons_and_keeps_all_four_outputs(self):
        case = RelationshipProbeTests()
        self.addCleanup(case.doCleanups)
        fixture, actors, trees, correct = case.fixture()
        originals = episode.solution_samples(fixture.tree)
        lesson = next(row for row in correct["lessons"] if row["earlier"]["attempt"] == 3
                      and row["relation"] == "CORRECTED")
        good = self.note(originals, [lesson])
        # A repeated lesson-looking object in comparison prose must not become a credit unit.
        for row in good["comparisons"]:
            row["comparison"] = json.dumps(lesson)
        empty = self.note(originals)
        invalid = copy.deepcopy(empty)
        invalid["comparisons"].pop()
        replies = [json.dumps(good), json.dumps(empty), json.dumps(invalid), "not JSON"]
        before = to_json(fixture.tree)
        with patch.object(credit, "score_note", wraps=credit.score_note) as scorer:
            rows, _, _ = case.collect(fixture, actors, trees, replies, comparisons=True)
        self.assertEqual([call.args[0] for call in scorer.call_args_list], replies)
        self.assertEqual(len(fixture.engine.requests), 1)
        self.assertIn(probe.COMPARISONS_VERSION, fixture.engine.requests[0].request_id)
        result = rows[0]
        self.assertEqual([row["output_contract_valid"] for row in result["samples"]], [True, True, False, False])
        self.assertEqual(result["absolute_relationship_units"], 1)
        self.assertFalse(result["proposal_credit_comparison_eligible"])
        self.assertFalse(result["actual_learning_measured"])
        self.assertEqual(to_json(fixture.tree), before)
        for row in result["samples"]:
            self.assertEqual(row["score"]["strict_reward"], 0)
            self.assertEqual(row["score"]["fact_rewards"], {})
            self.assertEqual(row["comparison_reward"], 0)
            self.assertFalse(row["comparison_semantics_verified"])
        mapped = result["actual_group_credit"][result["samples"][0]["sample_id"]]
        self.assertEqual(mapped["full_group_size"], 4)
        self.assertTrue(all(mean in (0, .25) for mean in mapped["relationship_group_means"].values()))
        raw = replies[0]
        for start, end in result["samples"][0]["score"]["relationship_spans"].values():
            self.assertGreater(start, raw.index('"lessons"'))
            self.assertEqual(json.loads(raw[start:end]), lesson)
        self.assertFalse(any(mapped["relationship_token_advantages"][:len(fixture.engine.tokenizer.encode(
            raw[:raw.index('"lessons"')])) - 1]))
