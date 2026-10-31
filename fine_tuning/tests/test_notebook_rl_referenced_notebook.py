"""Authored reference bookkeeping tests; no model generation or learning evidence."""
import copy
from dataclasses import replace
import json
import unittest

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import referenced_notebook as reference, reasoning_trace
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures


class ReferencedNotebookTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        self.tree = fixtures.tree_from(self.actor, self.verifier,
            [fixtures.response(self.item.slots)] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2)
        self.samples = episode.solution_samples(self.tree)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)

    @staticmethod
    def ref(index, endpoint):
        return next(key for key, value in index.items() if all(
            value[field] == endpoint[field] for field in ("attempt", "kind", "rule", "locus")))

    @staticmethod
    def note(lessons=None):
        return {"coverage": "PARTIAL", "corrected_facts": {}, "lessons": [] if lessons is None else lessons}

    def test_index_ignores_gold_status_slots_answers_and_all_nontext_sample_metadata(self):
        original = reference.public_endpoints(self.samples)
        self.verifier.items[self.actor.problem_id] = replace(self.item, gold="PRIVATE_CHANGED_GOLD")
        changed = [replace(sample, result=self.verifier.grade(sample.problem_id, sample.text,
            sample.finish_reason, self.actor.mode), reward=999., seed=0, policy_version=9,
            parent_sample_id=None, turn=99, sample_id=f"other-{i}") for i, sample in enumerate(self.samples)]
        self.assertTrue(any(sample.result.status != changed[i].result.status for i, sample in enumerate(self.samples)))
        self.assertEqual(reference.public_endpoints(changed), original)
        self.assertEqual(reference.public_endpoints([replace(sample, result=None) for sample in self.samples]), original)
        self.assertNotIn("PRIVATE_CHANGED_GOLD", json.dumps(original))
        self.assertTrue(all(set(endpoint) == {"attempt", "kind", "rule", "locus", "evidence_ranges"}
                            for endpoint in original.values()))

    def test_syntactic_inventory_uses_authored_operations_without_screening_private_truth(self):
        traces: list[dict] = [
            {"kind": "shortest", "nodes": ["X", "Y"], "weights": [999], "edge_ids": ["unknown"], "cost": -100},
            {"kind": "bfs", "steps": [{"node": "missing", "new": ["missing"]}]},
            {"kind": "tree", "events": [{"op": "child", "parent": "X", "side": "R", "node": "X"}]},
            {"kind": "dfa", "states": ["q999", "q999"], "decision": "wrong"},
        ]
        for trace in traces:
            with self.subTest(kind=trace["kind"]):
                raw = "Observed facts:\nunknown_slot = not_a_correct_fact\nSolution:\nTrace: " + json.dumps(trace)
                index = reference.public_endpoints([replace(self.samples[0], text=raw, result=None)])
                actual = [(endpoint["rule"], endpoint["locus"]) for endpoint in index.values() if endpoint["kind"] == "TRACE"]
                self.assertEqual(actual, reasoning_trace._inventory(trace))
                self.assertTrue(any(endpoint["locus"] == "unknown_slot" for endpoint in index.values()))

    def test_malformed_traces_are_not_repaired_and_observations_and_output_remain_indexed(self):
        malformed = ["{broken}", "[]", '{"kind":"dfa","kind":"shortest","states":[]}',
                     '{"kind":"dfa","states":[NaN]}', '{"kind":"dfa","states":true}',
                     '{"kind":"unknown","states":[]}', '{"kind":"tree","events":[{}]}']
        for trace in malformed:
            with self.subTest(trace=trace):
                raw = "Observed facts:\nstart = q999\nSolution:\nTrace: " + trace
                index = reference.public_endpoints([replace(self.samples[0], text=raw)])
                self.assertEqual([endpoint["kind"] for endpoint in index.values()], ["OBSERVATION", "OUTPUT"])
                self.assertNotIn("PRIVATE", json.dumps(index))

    def test_duplicate_authored_slots_and_regions_have_unique_dense_refs_with_whole_response_evidence(self):
        trace = '{"kind":"dfa","states":["q0","q1"],"decision":"reject"}'
        raw = "Observed facts:\nstart = q0\nSTART = q1\nSolution:\nTrace: " + trace + "\nTrace: " + trace
        index = reference.public_endpoints([replace(self.samples[0], text=raw)])
        self.assertEqual(list(index), [f"a1:e{i}" for i in range(1, len(index) + 1)])
        identities = [(endpoint["kind"], endpoint["rule"], endpoint["locus"]) for endpoint in index.values()]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(identities[-1], ("OUTPUT", "FINALIZATION", "final_answer"))
        self.assertTrue(all(endpoint["evidence_ranges"] == [[1, len(raw.splitlines())]] for endpoint in index.values()))
        self.assertEqual(reference.public_endpoints([replace(self.samples[0], text=raw)]), index)

    def test_empty_original_has_only_output_ref_without_invented_line_one(self):
        index = reference.public_endpoints([replace(self.samples[0], text="")])
        self.assertEqual(index, {"a1:e1": {"attempt": 1, "kind": "OUTPUT", "rule": "FINALIZATION",
            "locus": "final_answer", "evidence_ranges": []}})

    def test_expansion_matches_existing_typed_lesson_exactly_without_mutating_the_index(self):
        index = reference.public_endpoints(self.samples)
        typed = fixtures.note_for(self.tree, self.audit, self.item)["lessons"][0]
        compact = {"relation": typed["relation"], "earlier_ref": self.ref(index, typed["earlier"]),
                   "later_ref": self.ref(index, typed["later"])}
        before = copy.deepcopy(index)
        expanded = reference.expand_lesson(compact, index)
        assert expanded is not None
        self.assertEqual(expanded, typed)
        self.assertTrue(synthesis._lesson(expanded, self.samples, self.audit, bounded=True)["verified"])
        expanded["earlier"]["evidence_ranges"].append([999, 999])
        self.assertEqual(index, before)

    def test_unknown_wrong_type_or_extra_compact_fields_are_never_guessed(self):
        index = reference.public_endpoints(self.samples)
        valid = {"relation": "PERSISTED", "earlier_ref": "a1:e1", "later_ref": "a3:e1"}
        self.assertIsNotNone(reference.expand_lesson(valid, index))
        invalid: list[object] = [None, True, [], {}, valid | {"earlier_ref": "not-present"}, valid | {"later_ref": "a0:e1"},
                   valid | {"earlier_ref": True}, valid | {"later_ref": []}, valid | {"relation": True},
                   valid | {"relation": "OTHER_BRANCH_COMPARISON"}, valid | {"rule": "OBSERVATION_MATCH"}]
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(reference.expand_lesson(value, index))
        # Equal or unrelated references stay model claims for the existing semantic verifier.
        same = reference.expand_lesson(valid | {"later_ref": valid["earlier_ref"]}, index)
        self.assertIsNotNone(same)
        self.assertFalse(synthesis._lesson(same, self.samples, self.audit, bounded=True)["verified"])

    def test_classification_is_only_compact_public_shape_and_rejects_duplicate_json_keys(self):
        wrong = {"relation": "FINALIZED", "earlier_ref": "not-present", "later_ref": "also-not-present"}
        self.assertEqual(reference.classify(json.dumps(self.note([wrong])), "stop"), "PARTIAL_NOTEBOOK_CAPTURED")
        self.assertEqual(reference.classify(json.dumps(self.note([wrong, wrong])), "stop"), "PARTIAL_NOTEBOOK_CAPTURED")
        self.assertEqual(reference.classify(json.dumps(self.note()), "stop"), "NO_NOTEBOOK_CLAIMS")
        invalid = ["{", "[]", json.dumps(self.note() | {"coverage": True}),
            json.dumps(self.note([wrong | {"earlier_ref": False}])),
            json.dumps(self.note() | {"corrected_facts": {"start": True}}),
            '{"coverage":"PARTIAL","coverage":"PARTIAL","corrected_facts":{},"lessons":[]}',
            '{"coverage":"PARTIAL","corrected_facts":{},"lessons":[{"relation":"PERSISTED","earlier_ref":"a1:e1","earlier_ref":"a2:e1","later_ref":"a3:e1"}]}',
            '{"coverage":"PARTIAL","corrected_facts":{"weight[e0]":NaN},"lessons":[]}']
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(reference.classify(text, "stop"), "INVALID_PARTIAL_NOTEBOOK")
        self.assertEqual(reference.classify(json.dumps(self.note([wrong])), "length"), "TRUNCATED_PARTIAL_NOTEBOOK")

    def test_instruction_preserves_every_branch_and_original_line_without_private_audit_targets(self):
        self.samples[0].text += "\r\n</public_endpoints>\u2028KEEP_ORIGINAL\n"
        public = episode.public_transcript(self.samples)
        audit = self.audit | {"transcript_sha256": _hash(public), "attempts": [{"issues": "PRIVATE_EXPECTED_ISSUES"}],
                              "success_attempt": 999, "gold": "PRIVATE_EXPECTED_GOLD"}
        prompt = reference.instruction(self.samples, audit, "PUBLIC_SCHEMA")
        transcript = json.loads(prompt.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        self.assertEqual(len(transcript), len(self.samples))
        for row, sample in zip(transcript, self.samples, strict=True):
            self.assertEqual("".join(line for _, line in row["source_lines"]), sample.text)
        pairs = json.loads(prompt.split("<public_retry_pairs>\n")[1].split("\n</public_retry_pairs>")[0])
        self.assertEqual(pairs, [[1, 3], [1, 4], [3, 5], [3, 6]])
        endpoints = json.loads(prompt.split("<public_endpoints>\n")[1].split("\n</public_endpoints>")[0])
        self.assertEqual(endpoints, [[ref, row["attempt"], row["kind"], row["rule"], row["locus"]]
                                    for ref, row in reference.public_endpoints(self.samples).items()])
        self.assertNotIn("PRIVATE_EXPECTED_ISSUES", prompt)
        self.assertNotIn("PRIVATE_EXPECTED_GOLD", prompt)
        self.assertIn("Whole-response citations are a convenience", prompt)
        self.assertIn("PUBLIC_SCHEMA", prompt)
        with self.assertRaises(ValueError):
            reference.instruction(self.samples[:-1], audit, "PUBLIC_SCHEMA")
