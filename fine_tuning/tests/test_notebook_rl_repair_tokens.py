"""Tiny authored CPU fixtures for original character/token ownership only."""
from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from typing import Any

from fine_tuning.notebook_rl.generators import circuit, graph, graph_pair
from fine_tuning.notebook_rl.reasoning_trace import _inventory
from fine_tuning.notebook_rl.repair_contract import RawSpan, RepairContractError, VERSION, parse_retry_sections
from fine_tuning.notebook_rl.repair_credit import operation_patch
from fine_tuning.notebook_rl.repair_tokens import emitted_patch_spans, span_token_ownership
from fine_tuning.tests.test_notebook_rl_circuit_trace import fixture_trace as circuit_fixture, response as circuit_response
from fine_tuning.tests.test_notebook_rl_reasoning_trace import fixture_trace, response
from fine_tuning.tests.test_notebook_rl_repair_contract import ChunkTokenizer, UnstableTokenizer
from fine_tuning.tests.test_notebook_rl_repair_verifier import joint


def sections(raw: str):
    return parse_retry_sections(raw, contract_version=VERSION)


class RepairTokenTests(unittest.TestCase):
    def test_every_supported_native_and_circuit_projection_maps_original_typed_literals(self) -> None:
        problems = [graph.generate(index, 3407, "train", "obs") for index in (0, 1, 2, 3, 7, 11)]
        problems.append(graph_pair.generate(0, 3407, "train", "obs"))
        problems.extend(circuit.generate(index, 20261021, "sealed", "obs") for index in (2, 5))
        seen = set()
        for problem in problems:
            is_circuit = problem.actor.source == circuit.GENERATOR_VERSION
            trace = circuit_fixture(problem) if is_circuit else fixture_trace(problem)
            body = circuit_response(problem, trace) if is_circuit else response(problem, trace)
            raw = joint(body)
            bound = sections(raw)
            for rule, locus in _inventory(trace):
                patch = operation_patch(trace, rule, locus)
                spans = emitted_patch_spans(raw, bound, kind="TRACE", rule=rule, locus=locus)
                with self.subTest(kind=trace["kind"], rule=rule, locus=locus):
                    if patch is None:
                        self.assertIsNone(spans)
                        continue
                    self.assertTrue(spans)
                    assert spans is not None
                    values = [json.loads(raw[span.start:span.end]) for span in spans]
                    actual = values if rule in ("DIRECTED_EDGE", "DFA_TRANSITION") else values[0]
                    self.assertEqual(json.dumps(actual, sort_keys=True), json.dumps(patch["value"], sort_keys=True))
                    self.assertTrue(all(bound.solution_span.start <= span.start < span.end <= len(raw) for span in spans))
                    seen.add(rule)
        self.assertTrue({"DIRECTED_EDGE", "EDGE_WEIGHT", "COST_SUM", "BFS_FIFO", "BFS_NEW_NEIGHBORS",
                         "TREE_CHILD_EDGE", "TREE_EVENT_ORDER", "DFA_TRANSITION", "DFA_ACCEPTANCE",
                         "CIRCUIT_NODE_VOLTAGE", "CIRCUIT_OHM", "CIRCUIT_KCL_SUM", "CIRCUIT_INPUT_OHM",
                         "CIRCUIT_QUERY_VALUE"} <= seen)

    def test_reordered_escaped_json_crlf_and_quoted_old_trace_keep_exact_positions(self) -> None:
        literal = r'{ "cost" : 3, "weights" : [ 2 , 3 ], "no\u0064es" : [ "\u0041" , "B" , "H" ], "edge_ids" : null, "kind" : "shortest" }'
        body = "Observed facts:\r\n dir[e4] = H→G  \t\r\nSolution:\r\n\tTrace:\t" + literal + " \t\r\nFinal answer: 3"
        raw, bound = joint(body, newline="\r\n"), None
        bound = sections(raw)
        self.assertNotEqual(len(raw[:bound.solution_span.start].encode()), bound.solution_span.start)
        pair = emitted_patch_spans(raw, bound, kind="TRACE", rule="DIRECTED_EDGE", locus="nodes[0:2]")
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual([raw[span.start:span.end] for span in pair], [r'"\u0041"', '"B"'])
        weight = emitted_patch_spans(raw, bound, kind="TRACE", rule="EDGE_WEIGHT", locus="weights[1]")
        self.assertIsNotNone(weight)
        assert weight is not None
        self.assertEqual(raw[weight[0].start:weight[0].end], "3")
        self.assertLess(weight[0].start, raw.index('"no\\u0064es"'))
        cost = emitted_patch_spans(raw, bound, kind="TRACE", rule="COST_SUM", locus="cost")
        assert cost is not None
        self.assertEqual(raw[cost[0].start:cost[0].end], "3")
        self.assertNotEqual(weight, cost)
        observation = emitted_patch_spans(raw, bound, kind="OBSERVATION", rule="OBSERVATION_MATCH", locus="dir[e4]")
        assert observation is not None
        self.assertEqual(raw[observation[0].start:observation[0].end], "H→G")
        self.assertGreater(observation[0].start, bound.solution_span.start)

    def test_whole_fields_rows_and_two_item_patches_do_not_pick_equal_values_elsewhere(self) -> None:
        trace = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 2}
        literal = json.dumps(trace, separators=(",", ":"))
        raw = joint("Solution:\nTrace: " + literal + "\nFinal answer: 2")
        bound = sections(raw)
        for rule in ("PATH_ENDPOINTS", "MINIMALITY"):
            spans = emitted_patch_spans(raw, bound, kind="TRACE", rule=rule, locus="nodes")
            assert spans is not None
            self.assertEqual(raw[spans[0].start:spans[0].end], '["A","B","H"]')
        weight = emitted_patch_spans(raw, bound, kind="TRACE", rule="EDGE_WEIGHT", locus="weights[1]")
        assert weight is not None
        expected = raw.index('"weights":[2,2]') + len('"weights":[2,')
        self.assertEqual(weight, (RawSpan(expected, expected + 1),))
        bfs = {"kind": "bfs", "steps": [{"node": "A", "new": ["B"]}, {"node": "B", "new": []}]}
        raw = joint("Solution:\nTrace: " + json.dumps(bfs) + "\nFinal answer: A,B")
        for rule, locus in (("BFS_FIFO", "steps[0].node"), ("BFS_NEW_NEIGHBORS", "steps[0].new")):
            spans = emitted_patch_spans(raw, sections(raw), kind="TRACE", rule=rule, locus=locus)
            assert spans is not None
            self.assertEqual(json.loads(raw[spans[0].start:spans[0].end]), bfs["steps"][0])

    def test_circuit_input_and_query_own_whole_field_not_one_scalar(self) -> None:
        problem = circuit.generate(2, 20261021, "sealed", "obs")
        trace = circuit_fixture(problem)
        raw = joint(circuit_response(problem, trace))
        for rule, locus, field in (("CIRCUIT_INPUT_CURRENT", "input[1]", "input"),
                                  ("CIRCUIT_INPUT_VOLTAGE_LINK", "input[0]", "input"),
                                  ("CIRCUIT_QUERY_VALUE", "query", "query")):
            spans = emitted_patch_spans(raw, sections(raw), kind="TRACE", rule=rule, locus=locus)
            assert spans is not None
            self.assertEqual(len(spans), 1)
            self.assertEqual(json.loads(raw[spans[0].start:spans[0].end]), trace[field])

    def test_ambiguous_observations_and_malformed_or_unsupported_native_abstain(self) -> None:
        trace = {"kind": "dfa", "states": ["q0", "q1", "q2", "q2", "q3"], "decision": "accept"}
        line = "Trace: " + json.dumps(trace)
        for observations in ("dir[e4] = H->G\ndir[e4] = H->G\n", "dir[e4] = ???\n", "dir[e4] = H->G or G->H\n"):
            raw = joint("Observed facts:\n" + observations + "Solution:\n" + line + "\nFinal answer: accept")
            self.assertIsNone(emitted_patch_spans(raw, sections(raw), kind="OBSERVATION",
                                                rule="OBSERVATION_MATCH", locus="dir[e4]"))
        for native in ("```\n" + line + "\n```", line + "\n" + line,
                       'Trace: {"kind":"dfa","states":[true],"decision":"accept"}', "Trace: {}", "No trace"):
            raw = joint("Solution:\n" + native + "\nFinal answer: accept")
            self.assertIsNone(emitted_patch_spans(raw, sections(raw), kind="TRACE", rule="DFA_START", locus="states[0]"))
        raw = joint("Solution:\n" + line + "\nFinal answer: accept")
        for kind, rule, locus in (("OUTPUT", "FINALIZATION", "final_answer"),
                                 ("TRACE", "UNSUPPORTED", "states[0]"), ("TRACE", "DFA_START", "states[99]")):
            self.assertIsNone(emitted_patch_spans(raw, sections(raw), kind=kind, rule=rule, locus=locus))

    def test_raw_section_hash_or_bounds_mismatch_is_integrity_failure_before_abstention(self) -> None:
        raw = joint("Solution:\nFinal answer: 7")
        bound = sections(raw)
        for forged in (replace(bound, raw_sha="0" * 64),
                       replace(bound, solution_span=RawSpan(bound.solution_span.start + 1, len(raw)))):
            with self.assertRaises(RepairContractError):
                emitted_patch_spans(raw, forged, kind="UNSUPPORTED", rule="unsupported", locus="unsupported")
        with self.assertRaises(RepairContractError):
            emitted_patch_spans(raw.replace("7", "8"), bound, kind="TRACE", rule="DFA_START", locus="states[0]")

    def test_token_boundaries_eos_and_overlap_union_preserve_original_indices(self) -> None:
        raw = "AabcdefZ"
        tokenizer = ChunkTokenizer(["Aab", "c", "de", "fZ", ""])
        ids = [0, 1, 2, 3, 4]
        record = span_token_ownership(raw, ids, tokenizer, (RawSpan(1, 7),))
        self.assertEqual(record["status"], "MAPPED_WITH_EXCLUDED_BOUNDARIES")
        self.assertEqual(record["token_indices"], [1, 2])
        self.assertEqual(record["excluded_crossing_token_indices"], [0, 3])
        self.assertEqual(record["raw_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(record["token_ids_sha256"], hashlib.sha256(b"[0,1,2,3,4]").hexdigest())
        self.assertEqual(ids, [0, 1, 2, 3, 4])
        raw, tokenizer = "abcdef", ChunkTokenizer([*"abcdef", ""])
        record = span_token_ownership(raw, list(range(7)), tokenizer,
                                      (RawSpan(1, 4), RawSpan(2, 5), RawSpan(1, 4)))
        self.assertEqual(record["spans"], [[1, 5]])
        self.assertEqual(record["token_indices"], [1, 2, 3, 4])
        touching = span_token_ownership("abc", [0], ChunkTokenizer(["abc"]), (RawSpan(0, 1), RawSpan(1, 3)))
        self.assertEqual(touching["token_indices"], [0])
        gap = span_token_ownership('"A","B"', [0], ChunkTokenizer(['"A","B"']), (RawSpan(0, 3), RawSpan(4, 7)))
        self.assertEqual(gap["token_indices"], [])
        self.assertEqual(gap["excluded_crossing_token_indices"], [0])

    def test_unstable_prefix_empty_spans_and_invalid_token_or_span_inputs(self) -> None:
        record = span_token_ownership("α", [0, 1], UnstableTokenizer(["α"]), (RawSpan(0, 1),))
        self.assertEqual(record["status"], "UNSTABLE_TOKEN_PREFIX")
        self.assertEqual(record["token_indices"], [])
        empty = span_token_ownership("abc", [0], ChunkTokenizer(["abc"]), ())
        self.assertEqual(empty["status"], "NO_SPANS")
        self.assertEqual(empty["token_indices"], [])
        zero = span_token_ownership("abc", [0], ChunkTokenizer(["abc"]), (RawSpan(1, 1),))
        self.assertEqual(zero["status"], "NO_SPANS")
        malformed: tuple[Any, ...] = ([RawSpan(0, 1)], (RawSpan(0, 4),), (object(),))
        for spans in malformed:
            with self.subTest(spans=spans), self.assertRaises(RepairContractError):
                span_token_ownership("abc", [0], ChunkTokenizer(["abc"]), spans)
        forged = object.__new__(RawSpan)
        object.__setattr__(forged, "start", False)
        object.__setattr__(forged, "end", 1)
        with self.assertRaises(RepairContractError):
            span_token_ownership("abc", [0], ChunkTokenizer(["abc"]), (forged,))
        invalid_ids: tuple[Any, ...] = ([True], [-1], (0,))
        for ids in invalid_ids:
            with self.subTest(ids=ids), self.assertRaises(RepairContractError):
                span_token_ownership("abc", ids, ChunkTokenizer(["abc"]), ())
        for spans in ((), (RawSpan(0, 1),)):
            with self.assertRaisesRegex(RepairContractError, "TOKEN_TEXT_MISMATCH"):
                span_token_ownership("different", [0], ChunkTokenizer(["abc"]), spans)

    def test_inconsistent_full_decode_and_mutated_prefix_slice_are_integrity_failures(self) -> None:
        class Inconsistent:
            def __init__(self) -> None:
                self.full_calls = 0

            def decode(self, ids, **kwargs):
                if len(ids) == 2:
                    self.full_calls += 1
                    return "ABCD" if self.full_calls == 1 else "AB"
                return "A"

        class Mutating:
            def decode(self, ids, **kwargs):
                if len(ids) == 1:
                    ids[0] = 9
                return "AB" if len(ids) == 2 else "A"

        with self.assertRaisesRegex(RepairContractError, "TOKEN_TEXT_MISMATCH"):
            span_token_ownership("ABCD", [0, 1], Inconsistent(), (RawSpan(0, 4),))
        original_ids = [0, 1]
        with self.assertRaisesRegex(RepairContractError, "TOKEN_IDS"):
            span_token_ownership("AB", original_ids, Mutating(), (RawSpan(0, 2),))
        self.assertEqual(original_ids, [0, 1])
        stable = span_token_ownership("AB", [0, 1], ChunkTokenizer(["A", "B"]), (RawSpan(0, 2),))
        self.assertEqual(stable["token_indices"], [0, 1])
        unstable = span_token_ownership("α", [0, 1], UnstableTokenizer(["α"]), (RawSpan(0, 1),))
        self.assertEqual((unstable["status"], unstable["token_indices"]), ("UNSTABLE_TOKEN_PREFIX", []))


if __name__ == "__main__":
    unittest.main()
