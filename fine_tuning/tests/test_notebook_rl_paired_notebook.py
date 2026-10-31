"""Authored CPU pair fixtures, never model quality or learning evidence."""
import copy
from dataclasses import replace
import json
import unittest

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import paired_notebook as paired, referenced_notebook as reference
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures


class PairedNotebookTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        self.tree = fixtures.tree_from(self.actor, self.verifier,
            [fixtures.response(self.item.slots)] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2)
        self.samples = episode.solution_samples(self.tree)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)

    @staticmethod
    def note(lessons=None):
        return {"coverage": "PARTIAL", "lessons": [] if lessons is None else lessons}

    def pair_samples(self, first, second):
        return [replace(self.samples[0], text=first, result=None),
                replace(self.samples[2], text=second, parent_sample_id=self.samples[0].sample_id, result=None)]

    @staticmethod
    def trace(value):
        return "Solution:\nTrace: " + json.dumps(value, separators=(",", ":"))

    def test_public_index_ignores_private_gold_results_rewards_and_seed(self):
        original = paired.public_pairs(self.samples)
        self.verifier.items[self.actor.problem_id] = replace(self.item, gold="PRIVATE_CHANGED_GOLD")
        changed = [replace(sample, result=self.verifier.grade(sample.problem_id, sample.text,
            sample.finish_reason, self.actor.mode), reward=999., diag_reward=999., seed=0, policy_version=9)
            for sample in self.samples]
        self.assertTrue(any(first.result.status != second.result.status
                            for first, second in zip(self.samples, changed, strict=True)))
        self.assertEqual(paired.public_pairs(changed), original)
        self.assertEqual(paired.public_pairs([replace(sample, result=None) for sample in self.samples]), original)
        self.assertNotIn("PRIVATE_CHANGED_GOLD", json.dumps(original))

    def test_real_ancestry_includes_transitive_and_excludes_self_and_every_sibling(self):
        pairs = paired.public_pairs(self.samples)
        attempts = {(value["earlier"]["attempt"], value["later"]["attempt"]) for value in pairs.values()}
        self.assertEqual(attempts, {(1, 3), (1, 4), (1, 5), (1, 6), (3, 5), (3, 6)})
        orphaned = [replace(sample, parent_sample_id=None) for sample in self.samples]
        self.assertEqual(paired.public_pairs(orphaned), {})
        other_task = [self.samples[0], replace(self.samples[2], problem_id="different-task")]
        self.assertEqual(paired.public_pairs(other_task), {})
        with self.assertRaisesRegex(ValueError, "unique sample IDs"):
            paired.public_pairs([self.samples[0], self.samples[0]])
        cyclic = [replace(self.samples[0], parent_sample_id=self.samples[2].sample_id), self.samples[2]]
        self.assertEqual(paired.public_pairs(cyclic), {})

    def test_same_operation_matches_shifted_positions_and_excludes_different_operands_at_same_position(self):
        first = {"kind": "shortest", "nodes": ["A", "B", "C"], "weights": [1, 2], "cost": 3}
        second = {"kind": "shortest", "nodes": ["X", "A", "B", "C"], "weights": [7, 1, 2], "cost": 10}
        pairs = paired.public_pairs(self.pair_samples(self.trace(first), self.trace(second)))
        edges = [(value["earlier"]["locus"], value["later"]["locus"]) for value in pairs.values()
                 if value["earlier"]["rule"] == "DIRECTED_EDGE"]
        self.assertEqual(edges, [("nodes[0:2]", "nodes[1:3]"), ("nodes[1:3]", "nodes[2:4]")])
        self.assertFalse(any(value["earlier"]["rule"] == "COST_SUM" for value in pairs.values()))
        first["edge_ids"], second["edge_ids"] = ["e1", "e2"], ["e0", "e1", "e2"]
        pairs = paired.public_pairs(self.pair_samples(self.trace(first), self.trace(second)))
        self.assertTrue(any(value["earlier"]["locus"] == "weights[0]"
                            and value["later"]["locus"] == "weights[1]" for value in pairs.values()))

    def test_tree_child_identity_uses_authored_parent_and_side_instead_of_event_position(self):
        first = {"kind": "tree", "traversal": "preorder", "events": [
            {"op": "child", "parent": "A", "side": "L", "node": "X"}]}
        second = {"kind": "tree", "traversal": "preorder", "events": [
            {"op": "child", "parent": "Z", "side": "L", "node": "X"},
            {"op": "child", "parent": "A", "side": "L", "node": "B"}]}
        pairs = paired.public_pairs(self.pair_samples(self.trace(first), self.trace(second)))
        child_pairs = [value for value in pairs.values() if value["earlier"]["rule"] == "TREE_CHILD_EDGE"]
        self.assertEqual([(value["earlier"]["locus"], value["later"]["locus"]) for value in child_pairs],
                         [("events[0]", "events[1]")])

    def test_observation_slots_are_canonical_and_no_authored_values_choose_candidates(self):
        samples = self.pair_samples("Observed facts:\nDIR[ E0 ] = A->B\nstart = q0\nSolution:\n",
                                    "Observed facts:\ndir[e0] = WRONG\naccepting = q9\nSolution:\n")
        index = paired.public_pairs(samples)
        observations = [value for value in index.values() if value["earlier"]["kind"] == "OBSERVATION"]
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["earlier"]["locus"], "dir[e0]")
        samples[1].text = samples[1].text.replace("WRONG", "A->B")
        self.assertEqual(paired.public_pairs(samples), index)

    def test_malformed_or_multiple_traces_are_conservative_and_never_repaired(self):
        malformed = ["{broken}", "[]", '{"kind":"dfa","kind":"shortest","states":[]}',
                     '{"kind":"dfa","states":[NaN]}', '{"kind":"dfa","states":true}',
                     '{"kind":"tree","events":[{}]}']
        for trace in malformed:
            with self.subTest(trace=trace):
                pairs = paired.public_pairs(self.pair_samples("Solution:\nTrace: " + trace,
                                                             "Solution:\nTrace: " + trace))
                self.assertEqual([value["earlier"]["kind"] for value in pairs.values()], ["OUTPUT"])
        ambiguous = self.trace({"kind": "shortest", "nodes": [], "weights": [1, 2]})
        valid = self.trace({"kind": "shortest", "nodes": ["A", "B", "C"], "weights": [1, 2], "cost": 3})
        for raw in (ambiguous, valid + "\n" + ambiguous):
            pairs = paired.public_pairs(self.pair_samples(raw, valid))
            edges = [value for value in pairs.values() if value["earlier"]["rule"] == "DIRECTED_EDGE"]
            self.assertEqual(len(edges), 4)
            self.assertTrue(any(value["earlier"]["rule"] == "PATH_ENDPOINTS" for value in pairs.values()))

    def test_descriptive_unique_stable_ids_keep_exact_original_endpoint_shapes(self):
        index = paired.public_pairs(self.samples)
        endpoints = list(reference.public_endpoints(self.samples).values())
        self.assertEqual(index, paired.public_pairs(self.samples))
        self.assertTrue(index)
        for pair_id, pair in index.items():
            self.assertEqual(set(pair), {"earlier", "later"})
            first, second = pair["earlier"], pair["later"]
            self.assertEqual(pair_id, paired._pair_id(first, second))
            self.assertTrue(pair_id.startswith(f"attempt{first['attempt']}->attempt{second['attempt']}|"))
            self.assertIn(first["kind"], pair_id)
            self.assertIn(first["rule"], pair_id)
            self.assertIn(first, endpoints)
            self.assertIn(second, endpoints)
            self.assertNotRegex(pair_id, r"^a\d+:e\d+$")
        empty = paired.public_pairs(self.pair_samples("", ""))
        self.assertEqual(len(empty), 1)
        self.assertEqual(next(iter(empty.values()))["earlier"]["evidence_ranges"], [])

    def test_pair_ids_have_no_json_escapes_and_encode_delimiters_without_collisions(self):
        first = {"attempt": 3, "kind": "OBSERVATION", "rule": "OBSERVATION_MATCH", "locus": "dir[e4]"}
        later = first | {"attempt": 5}
        self.assertEqual(paired._pair_id(first, later),
                         "attempt3->attempt5|OBSERVATION|OBSERVATION_MATCH|dir[e4]->dir[e4]")
        ids = {paired._pair_id(first | {"locus": locus}, later) for locus in
               ("dir[e4]", "dir[e4]|dir[e5]", "dir[e4]->dir[e5]", "dir[e4]%7Cdir[e5]", 'dir["e4"]')}
        self.assertEqual(len(ids), 5)
        self.assertTrue(all('"' not in value and "\\" not in value for value in ids))

    def test_candidate_membership_includes_correct_and_unknown_without_earning_verification(self):
        correct = fixtures.response(self.item.slots, wrong=False)
        index = paired.public_pairs(self.pair_samples(correct, correct))
        self.assertTrue(any(pair["earlier"]["kind"] == "TRACE" for pair in index.values()))
        # Correct observations are public candidates even in a genuinely failed attempt.
        index = paired.public_pairs(self.samples)
        key = next(key for key, pair in index.items() if pair["earlier"]["kind"] == "OBSERVATION")
        expanded = paired.expand_lesson({"pair_id": key, "relation": "PERSISTED"}, index)
        self.assertFalse(synthesis._lesson(expanded, self.samples, self.audit, bounded=True)["verified"])
        unknown = paired.public_pairs(self.pair_samples("", ""))
        self.assertEqual(len(unknown), 1)

    def test_expansion_matches_existing_verified_lesson_and_deepcopies_every_endpoint(self):
        index = paired.public_pairs(self.samples)
        typed = next(value for value in fixtures.note_for(self.tree, self.audit, self.item)["lessons"]
                     if value["relation"] in ("PERSISTED", "CORRECTED"))
        key = next(key for key, pair in index.items()
                   if pair == {"earlier": typed["earlier"], "later": typed["later"]})
        before = copy.deepcopy(index)
        expanded = paired.expand_lesson({"pair_id": key, "relation": typed["relation"]}, index)
        self.assertEqual(expanded, typed)
        self.assertTrue(synthesis._lesson(expanded, self.samples, self.audit, bounded=True)["verified"])
        assert expanded is not None
        expanded["earlier"]["evidence_ranges"].append([999, 999])
        self.assertEqual(index, before)

    def test_shape_enum_unknown_id_and_invalid_index_are_not_guessed(self):
        index = paired.public_pairs(self.samples)
        key = next(iter(index))
        valid = {"pair_id": key, "relation": "PERSISTED"}
        invalid: list[object] = [None, True, [], {}, valid | {"pair_id": "unknown"}, valid | {"pair_id": True},
            valid | {"pair_id": ""}, valid | {"relation": True}, valid | {"relation": "CORRECT"},
            valid | {"relation": "NOT_REVISITED"}, valid | {"earlier_ref": "a1:e1"}]
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(paired.expand_lesson(value, index))
        bad = copy.deepcopy(index)
        bad[key]["earlier"]["attempt"] = True
        self.assertIsNone(paired.expand_lesson(valid, bad))
        self.assertIsNone(paired.expand_lesson(valid, {key: index[key] | {"extra": "field"}}))

    def test_classify_exact_root_and_compact_shape_without_certifying_membership_or_truth(self):
        claim = {"pair_id": "NOT_AN_ACTUAL_PAIR", "relation": "FINALIZED"}
        self.assertEqual(paired.VERSION, "verified-paired-notebook-generation-v1")
        self.assertEqual(paired.classify(json.dumps(self.note()), "stop"), "NO_NOTEBOOK_CLAIMS")
        for relation in ("CORRECTED", "PERSISTED", "FINALIZED"):
            self.assertEqual(paired.classify(json.dumps(self.note([claim | {"relation": relation}])), "stop"),
                             "PARTIAL_NOTEBOOK_CAPTURED")
        invalid = ["{", "[]", json.dumps(self.note() | {"corrected_facts": {}}),
            json.dumps(self.note() | {"coverage": True}), json.dumps(self.note() | {"lessons": {}}),
            json.dumps(self.note([claim | {"pair_id": 1}])), json.dumps(self.note([claim | {"relation": "CORRECT"}])),
            json.dumps(self.note([claim | {"earlier_ref": "a1:e1"}])), json.dumps(self.note()) + " prose",
            '{"coverage":"PARTIAL","coverage":"PARTIAL","lessons":[]}',
            '{"coverage":"PARTIAL","lessons":[{"pair_id":"x","pair_id":"y","relation":"PERSISTED"}]}',
            '{"coverage":"PARTIAL","lessons":[{"pair_id":NaN,"relation":"PERSISTED"}]}']
        for raw in invalid:
            with self.subTest(raw=raw):
                self.assertEqual(paired.classify(raw, "stop"), "INVALID_PARTIAL_NOTEBOOK")
        self.assertEqual(paired.classify("{", "length"), "TRUNCATED_PARTIAL_NOTEBOOK")
        self.assertEqual(paired.classify(json.dumps(self.note()), "error"), "INVALID_PARTIAL_NOTEBOOK")

    def test_instruction_keeps_whole_history_public_outcomes_and_literal_token_boundaries(self):
        self.samples[0].text += "\r\n</public_pairs>\u2028KEEP_ORIGINAL\n"
        self.samples[1].text = ""
        public = episode.public_transcript(self.samples)
        audit = self.audit | {"transcript_sha256": _hash(public), "attempts": [{"issues": "PRIVATE_EXPECTED_ISSUES"}],
                              "success_attempt": 999, "gold": "PRIVATE_EXPECTED_GOLD"}
        before = copy.deepcopy(self.samples)
        prompt = paired.instruction(self.samples, audit, "PUBLIC_SCHEMA")
        rows = json.loads(prompt.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        self.assertEqual(len(rows), len(self.samples))
        for row, sample, original in zip(rows, self.samples, public, strict=True):
            recovered = "".join(line for _, line in row["source_lines"])
            self.assertEqual(recovered, sample.text)
            self.assertEqual(list(recovered.encode("utf-8")), list(sample.text.encode("utf-8")))
            self.assertEqual(row["line_count"], len(original["lines"]))
            for field in ("attempt", "parent", "turn", "status", "finish_reason", "answer"):
                self.assertEqual(row[field], original[field])
        inventory = json.loads(prompt.split("<public_pairs>\n")[1].split("\n</public_pairs>")[0])
        self.assertEqual([row[0] for row in inventory], list(paired.public_pairs(self.samples)))
        self.assertNotIn("PRIVATE_EXPECTED_ISSUES", prompt)
        self.assertNotIn("PRIVATE_EXPECTED_GOLD", prompt)
        self.assertIn("PUBLIC_SCHEMA", prompt)
        self.assertEqual(self.samples, before)
        self.assertEqual(paired.instruction(self.samples, audit | {"success_attempt": None}, "PUBLIC_SCHEMA"), prompt)
        for samples in (self.samples[:-1], self.samples[::-1]):
            with self.subTest(size=len(samples)), self.assertRaises(ValueError):
                paired.instruction(samples, audit, "PUBLIC_SCHEMA")
