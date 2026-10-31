"""Authored unit-credit fixtures, not pretrained model learning or diagnosis certificates."""
import copy
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import synthesis_credit as credit
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures
from fine_tuning.tests.test_notebook_rl_updates import Characters


def partial_fixture(*, persisted=True):
    """Complete literal on-policy fixture; callers must register fixture.doCleanups."""
    fixture = integration.SynthesisIntegrationTests("test_full_history_generation_uses_original_images_and_independent_route")
    fixture.setUp()
    if persisted:
        for sample in episode.solution_samples(fixture.tree)[4:]:
            sample.text = fixtures.response(fixture.item.slots, wrong=True, answer="accept")
            sample.completion_token_ids = fixture.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.result = fixture.verifier.grade(sample.problem_id, sample.text, "stop", fixture.actor.mode)
            sample.reward = float(sample.result.status == "CORRECT")
    audit = synthesis.audit_episode(fixture.tree, fixture.actor, fixture.verifier, contract=synthesis.BOUNDED_VERSION)
    note = fixtures.note_for(fixture.tree, audit, fixture.item)
    note["attempts"], note["corrected_facts"] = [], {}
    fixture.partial_note = json.dumps(note, separators=(",", ":"))
    fixture.good = fixture.partial_note
    fixture.append(rounds=[[fixture.partial_note, "{}", "{}", "{}"] for _ in range(3)])
    return fixture


class SynthesisCreditTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        texts = [fixtures.response(self.item.slots)] * 4 + [fixtures.response(self.item.slots, wrong=True, answer="accept")] * 2
        self.tree = fixtures.tree_from(self.actor, self.verifier, texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.note = fixtures.note_for(self.tree, self.audit, self.item)
        self.lesson = next(row for row in self.note["lessons"] if row["relation"] == "PERSISTED")

    def score(self, note, *, text=None, finish="stop", audit=None, strict=None):
        raw = json.dumps(note, separators=(",", ":")) if text is None else text
        return credit.score_note(raw, finish, self.tree, audit or self.audit, self.verifier,
                                 strict_report=strict, actor=self.actor)

    def group(self, notes, *, texts=None, tokenizer=None):
        samples, scores = [], {}
        for index, note in enumerate(notes):
            text = json.dumps(note, separators=(",", ":")) if texts is None else texts[index]
            row = self.score(note, text=text)
            sample = SimpleNamespace(sample_id=f"s{index}", group_id="authored-group", kind="DIAG",
                text=text, finish_reason="stop", reward=row["strict_reward"],
                completion_token_ids=[ord(char) for char in text])
            samples.append(sample)
            scores[sample.sample_id] = row
        return samples, scores, credit.group_credit(samples, scores, tokenizer or Characters())

    def test_strict_zero_verified_persisted_and_required_fact_are_independent(self):
        note = {"corrected_facts": {"start": "q0"}, "lessons": [self.lesson]}
        row = self.score(note)
        origin = credit.issue_id(self.lesson["earlier"]["attempt"], self.lesson["earlier"])
        self.assertEqual(row["strict_reward"], 0.)
        self.assertEqual(row["fact_rewards"], {"start": 1})
        self.assertEqual(row["relationship_rewards"][origin], 1)
        expected = .0625 / len(self.item.slots) + .0625 / len(row["original_issue_ids"])
        self.assertEqual(row["atomic_reward"], expected)
        self.assertEqual(row["effective_reward"], expected)
        self.assertFalse(row["lesson_semantics_verified"])
        self.assertFalse(row["full_diagnosis"])
        self.assertFalse(row["all_reasoning_verified"])
        self.assertEqual(self.score({"corrected_facts": "malformed", "lessons": [self.lesson]})["relationship_rewards"][origin], 1)
        self.assertEqual(self.score({"corrected_facts": {"start": "q0"}, "lessons": "malformed"})["fact_rewards"], {"start": 1})

    def test_fresh_audit_and_strict_report_cannot_be_forged(self):
        bad = copy.deepcopy(self.audit)
        bad["attempts"][0]["issues"] = []
        with self.assertRaisesRegex(ValueError, "source audit"):
            self.score({}, audit=bad)
        strict = synthesis.grade_note("{}", "stop", self.tree, self.audit, self.verifier,
                                      actor=self.actor, contract=synthesis.BOUNDED_VERSION)
        strict["reward"] = 1.
        with self.assertRaisesRegex(ValueError, "strict report"):
            self.score({}, strict=strict)
        with self.assertRaises(ValueError):
            credit.score_note("{}", "stop", self.tree, self.audit, self.verifier,
                               actor=self.actor, replay_relation_v1=True)

    def test_first_submitted_origin_is_selected_before_grading_not_best_duplicate(self):
        wrong = copy.deepcopy(self.lesson)
        wrong["later"]["evidence_ranges"] = []
        row = self.score({"lessons": [wrong, self.lesson]})
        origin = credit.issue_id(self.lesson["earlier"]["attempt"], self.lesson["earlier"])
        self.assertEqual(row["selected_lesson_indices"], {origin: 0})
        self.assertEqual(row["relationship_rewards"][origin], 0)
        reverse = self.score({"lessons": [self.lesson, wrong]})
        self.assertEqual(reverse["relationship_rewards"][origin], 1)
        reordered = self.score({"lessons": ["unmatched", self.lesson]})
        self.assertEqual(reordered["selected_lesson_indices"], {origin: 1})
        self.assertEqual(reordered["relationship_rewards"][origin], 1)
        extra = copy.deepcopy(self.lesson)
        extra["private invented field"] = "unsupported"
        mixed = self.score({"lessons": [extra, self.lesson]})
        self.assertEqual(mixed["relationship_rewards"][origin], 0)
        self.assertNotIn(origin, mixed["relationship_spans"])

    def test_wrong_direction_sibling_and_missing_evidence_do_not_earn_relationship_credit(self):
        variants = []
        for mutate in (lambda row: row["later"].update(attempt=2),
                       lambda row: row["later"].update(locus="states[0:2]"),
                       lambda row: row["later"].update(evidence_ranges=[]),
                       lambda row: row.update(relation="CORRECTED"),
                       lambda row: row["earlier"].update(kind="OBSERVATION")):
            bad = copy.deepcopy(self.lesson)
            mutate(bad)
            variants.append(bad)
        for lesson in variants:
            self.assertEqual(sum(self.score({"lessons": [lesson]})["relationship_rewards"].values()), 0)
        comparison = next(row for row in self.note["lessons"] if row["relation"] == "OTHER_BRANCH_COMPARISON")
        self.assertEqual(sum(self.score({"lessons": [comparison]})["relationship_rewards"].values()), 0)

    def test_malformed_json_collisions_and_invalid_fact_values_never_coerce(self):
        for text in ('{"corrected_facts":{"start":"q0","start":"q1"}}',
                     '{"corrected_facts":{"start":"q0"},"extra":NaN}', '{"lessons":[]', '[]'):
            row = self.score({}, text=text)
            self.assertFalse(row["json_complete"])
            self.assertEqual(row["atomic_reward"], 0.)
        row = self.score({"corrected_facts": {"start": "q0", " START ": "q0"}, "lessons": [self.lesson]})
        self.assertEqual(row["fact_rewards"], {})
        self.assertEqual(sum(row["relationship_rewards"].values()), 1)
        row = self.score({"corrected_facts": {"start": "q0", "accepting": 5, "unknown": "q0"}})
        self.assertEqual(row["fact_rewards"], {"start": 1})
        self.assertEqual(set(row["fact_spans"]), {"start"})
        row = self.score({"corrected_facts": {"start": "q0"}, "lessons": [self.lesson]}, finish="length")
        self.assertEqual(row["atomic_reward"], 0.)
        full = copy.deepcopy(self.note)
        full["lessons"][0]["rule"] = float("nan")
        self.assertEqual(self.score(full)["atomic_reward"], 0.)

    def test_fixed_issue_denominator_counts_earlier_unit_only(self):
        row = self.score({"lessons": [self.lesson]})
        self.assertEqual(len(row["original_issue_ids"]), sum(len(attempt["issues"]) for attempt in self.audit["attempts"]))
        self.assertEqual(sum(row["relationship_rewards"].values()), 1)
        self.assertEqual(row["atomic_reward"], .0625 / len(row["original_issue_ids"]))

    def test_full_n4_means_include_malformed_and_missing_without_mis_recentering(self):
        samples, scores, out = self.group([{"lessons": [self.lesson]}, {}, {}, {}],
                                         texts=[json.dumps({"lessons": [self.lesson]}), "{}", "invalid JSON", "{}"])
        self.assertTrue(credit.has_signal(samples, scores))
        relation = out["s0"]["relationships"][0]
        self.assertEqual((relation["reward"], relation["group_mean"], relation["advantage"]), (1, .25, .75))
        self.assertAlmostEqual(sum(out["s0"]["relationship_token_advantages"]), .0625 * .75 / len(scores["s0"]["original_issue_ids"]))
        self.assertEqual(out["s0"]["strict_advantage"], 0.)
        self.assertEqual(out["s0"]["full_group_size"], 4)
        with self.assertRaisesRegex(ValueError, "original complete"):
            credit.group_credit(samples[:2], scores, Characters())
        self.assertTrue(all(not any(out[key]["partial_token_advantages"]) for key in ("s1", "s2", "s3")))

    def test_unit_vectors_can_have_signal_when_scalar_totals_match(self):
        notes = [{"corrected_facts": {"start": "q0", "accepting": "q0"}},
                 {"corrected_facts": {"start": "q3", "accepting": "q3"}}] * 2
        samples, scores, out = self.group(notes)
        self.assertEqual(len({row["effective_reward"] for row in scores.values()}), 1)
        self.assertTrue(credit.has_signal(samples, scores))
        self.assertTrue(any(out["s0"]["fact_token_advantages"]))
        self.assertFalse(any(out["s0"]["relationship_token_advantages"]))
        same, same_scores, same_out = self.group([notes[0]] * 4)
        self.assertFalse(credit.has_signal(same, same_scores))
        self.assertTrue(all(not any(row["partial_token_advantages"]) for row in same_out.values()))

    def test_decoder_spans_bind_escaped_repeated_literals_and_only_first_lesson(self):
        lesson = json.dumps(self.lesson, separators=(",", ":"))
        text = '{"other":"q0","corrected_facts":{"START":"\\u00710"},"lessons":[' + lesson + ',' + lesson + ']}'
        samples, scores, out = self.group([{}, {}, {}, {}], texts=[text, "{}", "{}", "{}"])
        row = scores["s0"]
        start, end = row["fact_spans"]["start"]
        self.assertEqual(text[start:end], '"\\u00710"')
        origin = next(iter(row["relationship_spans"]))
        start, end = row["relationship_spans"][origin]
        self.assertEqual(text[start:end], lesson)
        self.assertEqual(row["selected_lesson_indices"][origin], 0)
        allowed = set(out["s0"]["facts"][0]["token_indices"]) | set(out["s0"]["relationships"][0]["token_indices"])
        self.assertTrue(all(value == 0 for index, value in enumerate(out["s0"]["partial_token_advantages"]) if index not in allowed))
        credit.validate_score(row, samples[0])

    def test_fact_entry_order_is_stable_after_sorted_json_score_roundtrip(self):
        notes = [{"corrected_facts": {"start": "q0", "accepting": "q3"}}, {}, {}, {}]
        samples, scores, expected = self.group(notes)
        restored = json.loads(json.dumps(scores, sort_keys=True))
        self.assertEqual(credit.group_credit(samples, restored, Characters()), expected)
        self.assertEqual([entry["slot"] for entry in expected["s0"]["facts"]], ["accepting", "start"])
        self.assertEqual(expected["s0"]["facts"][0]["literal_span"], scores["s0"]["fact_spans"]["accepting"])
        for name in ("fact_spans", "relationship_spans"):
            bad = copy.deepcopy(restored)
            bad["s0"][name]["unbound"] = [0, 1]
            bad["s0"] = credit._seal({key: value for key, value in bad["s0"].items() if key != "score_sha256"})
            with self.subTest(component=name), self.assertRaises(ValueError):
                credit.group_credit(samples, bad, Characters())

    def test_relationship_length_changes_mapping_count_not_total_credit(self):
        compact = json.dumps({"lessons": [self.lesson]}, separators=(",", ":"))
        padded = json.dumps({"lessons": [self.lesson]}, indent=4)
        _, _, out = self.group([{}, {}, {}, {}], texts=[compact, padded, "{}", "{}"])
        a, b = out["s0"]["relationships"][0], out["s1"]["relationships"][0]
        self.assertNotEqual(len(a["token_indices"]), len(b["token_indices"]))
        self.assertAlmostEqual(sum(out["s0"]["relationship_token_advantages"]), sum(out["s1"]["relationship_token_advantages"]))

    def test_unstable_prefix_abstains_and_full_decode_mismatch_rejects(self):
        notes = [{"lessons": [self.lesson]}, {}, {}, {}]
        unstable = SimpleNamespace(decode=lambda ids, **kw: "?" if len(ids) == 1 else "".join(chr(index) for index in ids))
        _, _, out = self.group(notes, tokenizer=unstable)
        self.assertTrue(all(row["mapping_status"] == "UNSTABLE_TOKEN_PREFIX" for row in out.values()))
        self.assertTrue(all(not any(row["partial_token_advantages"]) for row in out.values()))
        with self.assertRaisesRegex(ValueError, "tokens and text differ"):
            self.group(notes, tokenizer=SimpleNamespace(decode=lambda ids, **kw: "different"))

    def test_validate_rejects_changed_arithmetic_text_strict_and_literal_spans(self):
        samples, scores, _ = self.group([{"lessons": [self.lesson]}] * 4)
        original = scores["s0"]
        for mutation in ("atomic", "span", "selection", "hash", "denominator"):
            row = copy.deepcopy(original)
            if mutation == "atomic":
                row["atomic_reward"] = .5
            elif mutation == "span":
                row["relationship_spans"][next(iter(row["relationship_spans"]))] = [0, 1]
            elif mutation == "selection":
                row["selected_lesson_indices"][next(iter(row["selected_lesson_indices"]))] = 3
            elif mutation == "hash":
                row["text_sha256"] = "0" * 64
            else:
                row["required_fact_count"] = True
            row = credit._seal({key: value for key, value in row.items() if key != "score_sha256"})
            with self.assertRaises(ValueError):
                credit.validate_score(row, samples[0])
        with self.assertRaises(ValueError):
            credit.validate_score(original, SimpleNamespace(**(vars(samples[0]) | {"reward": 1.})))

    def test_zero_issue_denominator_does_not_redistribute_relationship_weight(self):
        self.actor, self.item, self.verifier = fixtures.graph_native("shortest")
        trace = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 4}
        texts = [fixtures.graph_response(self.item.slots, trace, "8")] * 4 + [fixtures.graph_response(self.item.slots, trace, "4")] * 2
        self.tree = fixtures.tree_from(self.actor, self.verifier, texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        row = self.score({"corrected_facts": self.item.slots})
        self.assertEqual(row["original_issue_ids"], [])
        self.assertEqual(row["relationship_rewards"], {})
        self.assertEqual(row["atomic_reward"], .0625)

    def test_zero_required_fact_denominator_does_not_redistribute_its_weight(self):
        normal = self.verifier.grade
        def no_required_facts(problem_id, text, finish_reason, mode):
            result = normal(problem_id, text, finish_reason, mode)
            return replace(result, slots=replace(result.slots, required=0)) if text == "Observed facts:\nSolution:\n" else result
        with patch.object(self.verifier, "grade", side_effect=no_required_facts):
            row = self.score({"lessons": [self.lesson]})
        self.assertEqual(row["required_fact_count"], 0)
        self.assertEqual(row["fact_rewards"], {})
        self.assertEqual(row["atomic_reward"], .0625 / len(row["original_issue_ids"]))

    def test_strict_report_and_its_semantic_certificates_are_unchanged(self):
        text = json.dumps(self.note, separators=(",", ":"))
        strict = synthesis.grade_note(text, "stop", self.tree, self.audit, self.verifier,
                                     actor=self.actor, contract=synthesis.BOUNDED_VERSION)
        original = copy.deepcopy(strict)
        row = self.score(self.note, strict=strict)
        self.assertEqual(row["strict_reward"], strict["reward"])
        self.assertEqual(row["strict_reward"], 1.)
        self.assertEqual(strict, original)
        self.assertLessEqual(row["effective_reward"], 1.125)

    def test_stable_crossing_tokens_are_unmapped_without_recentering(self):
        samples, scores, _ = self.group([{"lessons": [self.lesson]}, {}, {}, {}])
        texts = {index: sample.text for index, sample in enumerate(samples)}
        for index, sample in enumerate(samples):
            sample.completion_token_ids = [index]
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "".join(texts[index] for index in ids))
        out = credit.group_credit(samples, scores, tokenizer)
        self.assertEqual(out["s0"]["relationships"][0]["group_mean"], .25)
        self.assertEqual(out["s0"]["mapping_status"], "NO_MAPPED_UNITS")
        self.assertTrue(all(not any(row["partial_token_advantages"]) for row in out.values()))

    def test_complete_bound_fixture_keeps_strict_zero_and_maps_partial_relationship_tokens(self):
        fixture = partial_fixture()
        self.addCleanup(fixture.doCleanups)
        episode.regrade_synthesis(fixture.tree, fixture.actor, fixture.verifier, fixture.engine.tokenizer)
        group = next(group for group in fixture.tree.groups if group.kind == "DIAG")
        audit = fixture.tree.notebook["rewarded_notebook"]["synthesis_audit"]
        scores = {sample.sample_id: credit.score_note(sample.text, sample.finish_reason,
            fixture.tree, audit, fixture.verifier, actor=fixture.actor) for sample in group.samples}
        self.assertTrue(all(row["strict_reward"] == 0 for row in scores.values()))
        out = credit.group_credit(group.samples, scores, fixture.engine.tokenizer)
        self.assertTrue(any(entry["reward"] == 1 and entry["advantage"] > 0 and entry["token_indices"]
                            and entry["relation"] == "PERSISTED"
                            for row in out.values() for entry in row["relationships"]))


class OutputFinalizationCreditTests(unittest.TestCase):
    score = SynthesisCreditTests.score
    group = SynthesisCreditTests.group

    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        self.prepare()

    def prepare(self, *, wrong_final=False, native_errors=False):
        missing = fixtures.response(self.item.slots, wrong=native_errors).rsplit("\nFinal answer:", 1)[0]
        later = fixtures.response(self.item.slots, wrong=False, answer="reject" if wrong_final else "accept")
        texts = [missing] * 4 + [later, fixtures.response(self.item.slots, wrong=False)]
        self.tree = fixtures.tree_from(self.actor, self.verifier, texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        earlier = {"attempt": 1, "kind": "OUTPUT", "rule": "FINALIZATION", "locus": "final_answer",
                   "evidence_ranges": [[1, len(texts[0].splitlines())]]}
        last = len(texts[4].splitlines())
        later_endpoint = {"attempt": 5, "kind": "OUTPUT", "rule": "FINALIZATION", "locus": "final_answer",
                          "evidence_ranges": [[last, last]]}
        self.lesson = {"rule": "FINALIZATION", "earlier": earlier, "later": later_endpoint, "relation": "FINALIZED"}
        self.origin = credit.issue_id(1, earlier)

    def test_fixed_output_denominator_includes_unrepaired_siblings_and_empty_lessons_get_zero(self):
        row = self.score({"lessons": [self.lesson]})
        expected = [credit.issue_id(index, self.lesson["earlier"]) for index in (1, 2, 3, 4)]
        self.assertEqual(row["original_issue_ids"], expected)
        self.assertTrue(all(not attempt["issues"] for attempt in self.audit["attempts"]))
        self.assertEqual(row["relationship_credit_types"], dict.fromkeys(expected, credit.OUTPUT_FINALIZATION))
        self.assertEqual(sum(row["relationship_rewards"].values()), 1)
        self.assertEqual(row["atomic_reward"], .0625 / 4)
        self.assertEqual(row["relationship_later_answer_correct"], {self.origin: True})
        self.assertEqual(row["strict_reward"], 0.)
        empty = self.score({"lessons": []})
        self.assertEqual(empty["original_issue_ids"], expected)
        self.assertEqual(sum(empty["relationship_rewards"].values()), 0)
        self.assertEqual(empty["atomic_reward"], 0.)
        self.assertEqual(credit.VERSION, "verified-synthesis-unit-token-credit-v2")
        self.assertEqual(credit.contract()["positive_relationships"][credit.OUTPUT_FINALIZATION], ["FINALIZED"])
        self.assertFalse(credit.contract()["output_checks_are_mandatory_native_issues"])

    def test_wrong_terminal_answer_can_finalize_presence_without_correct_answer_credit(self):
        self.prepare(wrong_final=True)
        row = self.score({"lessons": [self.lesson]})
        self.assertEqual(self.tree.groups[-1].samples[0].result.status, "INCORRECT")
        self.assertEqual(row["relationship_rewards"][self.origin], 1)
        self.assertEqual(row["relationship_later_answer_correct"], {self.origin: False})
        _, _, out = self.group([{"lessons": [self.lesson]}, {}, {}, {}])
        entry = out["s0"]["relationships"][0]
        self.assertEqual(entry["credit_type"], credit.OUTPUT_FINALIZATION)
        self.assertFalse(entry["later_answer_correct"])
        self.assertEqual(entry["reward"], 1)

    def test_first_duplicate_and_invalid_output_pairs_never_select_a_later_good_claim(self):
        wrong = copy.deepcopy(self.lesson)
        wrong["later"]["attempt"] = 2
        row = self.score({"lessons": [wrong, self.lesson]})
        self.assertEqual(row["selected_lesson_indices"], {self.origin: 0})
        self.assertEqual(row["relationship_rewards"][self.origin], 0)
        self.assertEqual(self.score({"lessons": [self.lesson, wrong]})["relationship_rewards"][self.origin], 1)
        for mutate in (lambda lesson: lesson.update(relation="CORRECTED"),
                       lambda lesson: lesson.update(relation="PERSISTED"),
                       lambda lesson: lesson["later"].update(kind="TRACE"),
                       lambda lesson: lesson["later"].update(rule="DFA_TRANSITION"),
                       lambda lesson: lesson["later"].update(locus="states[1]"),
                       lambda lesson: lesson["earlier"].update(evidence_ranges=[[1, 1]])):
            bad = copy.deepcopy(self.lesson)
            mutate(bad)
            self.assertEqual(sum(self.score({"lessons": [bad]})["relationship_rewards"].values()), 0)

    def test_native_and_output_units_share_one_fixed_relationship_budget(self):
        self.prepare(native_errors=True)
        native = next(lesson for lesson in fixtures.note_for(self.tree, self.audit, self.item)["lessons"]
                      if lesson["relation"] == "CORRECTED")
        row = self.score({"lessons": [native, self.lesson]})
        native_count = sum(len(attempt["issues"]) for attempt in self.audit["attempts"])
        self.assertGreater(native_count, 0)
        self.assertEqual(len(row["original_issue_ids"]), native_count + 4)
        self.assertEqual(sum(row["relationship_rewards"].values()), 2)
        self.assertEqual(row["atomic_reward"], .0625 * 2 / (native_count + 4))
        _, _, out = self.group([{"lessons": [native, self.lesson]}, {}, {}, {}])
        self.assertEqual([entry["credit_type"] for entry in out["s0"]["relationships"]],
                         [credit.NATIVE_ERROR_RELATIONSHIP, credit.OUTPUT_FINALIZATION])

    def test_mixed_credit_is_identical_after_actual_problem_tree_serialization(self):
        from fine_tuning.notebook_rl import updates
        from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
        from fine_tuning.tests.test_notebook_rl_synthesis_credit_admission import output_fixture

        fixture = output_fixture(mixed=True)
        self.addCleanup(fixture.doCleanups)
        objective = fixture.protocol["notebook_objective"]
        scores = updates.notebook_objective_scores(fixture.tree, objective)
        expected = {sample_id: row for group in fixture.tree.groups if group.kind == "DIAG"
            for sample_id, row in credit.group_credit(group.samples, scores, fixture.engine.tokenizer).items()}
        restored = from_json(ProblemTree, to_json(fixture.tree))
        restored_scores = updates.notebook_objective_scores(restored, objective)
        actual = {sample_id: row for group in restored.groups if group.kind == "DIAG"
            for sample_id, row in credit.group_credit(group.samples, restored_scores, fixture.engine.tokenizer).items()}
        self.assertEqual(actual, expected)
        self.assertEqual(episode._hash(actual), episode._hash(expected))
        for sample_id, row in actual.items():
            self.assertEqual([entry["issue_id"] for entry in row["relationships"]],
                [origin for origin in row["original_issue_ids"] if origin in restored_scores[sample_id]["relationship_spans"]])

    def test_typed_output_receipts_reject_resealed_forged_types_ids_and_answer_flags(self):
        samples, scores, _ = self.group([{"lessons": [self.lesson]}, {}, {}, {}])
        for mutation in ("type", "missing_type", "id", "relation", "answer_type", "missing_answer"):
            row = copy.deepcopy(scores["s0"])
            if mutation == "type":
                row["relationship_credit_types"][self.origin] = credit.NATIVE_ERROR_RELATIONSHIP
            elif mutation == "missing_type":
                del row["relationship_credit_types"][self.origin]
            elif mutation == "id":
                row["original_issue_ids"][0] = json.dumps([1, "OUTPUT", "OTHER", "final_answer"], separators=(",", ":"))
            elif mutation == "relation":
                row["relationship_relations"][self.origin] = "PERSISTED"
            elif mutation == "answer_type":
                row["relationship_later_answer_correct"][self.origin] = 1
            else:
                row["relationship_later_answer_correct"] = {}
            row = credit._seal({key: value for key, value in row.items() if key != "score_sha256"})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                credit.validate_score(row, samples[0])

    def test_equal_n4_has_no_policy_advantage_and_crossing_tokens_never_gain_credit(self):
        notes = [{"lessons": [self.lesson]}] * 4
        samples, scores, out = self.group(notes)
        self.assertFalse(credit.has_signal(samples, scores))
        self.assertTrue(all(not any(row["partial_token_advantages"]) for row in out.values()))
        samples, scores, out = self.group([notes[0], {}, {}, {}])
        entry = out["s0"]["relationships"][0]
        self.assertEqual((entry["reward"], entry["group_mean"], entry["advantage"]), (1, .25, .75))
        span = entry["literal_span"]
        self.assertEqual(samples[0].text[span[0]:span[1]], json.dumps(self.lesson, separators=(",", ":")))
        self.assertAlmostEqual(sum(out["s0"]["relationship_token_advantages"]), .0625 * .75 / 4)
        texts = {index: sample.text for index, sample in enumerate(samples)}
        for index, sample in enumerate(samples):
            sample.completion_token_ids = [index]
        crossing = SimpleNamespace(decode=lambda ids, **kwargs: "".join(texts[index] for index in ids))
        out = credit.group_credit(samples, scores, crossing)
        self.assertEqual(out["s0"]["relationships"][0]["token_indices"], [])
        self.assertEqual(out["s0"]["relationships"][0]["group_mean"], .25)
        self.assertTrue(all(not any(row["relationship_token_advantages"]) for row in out.values()))


if __name__ == "__main__":
    unittest.main()
