"""Authored observation cases, not benchmark-derived labels."""
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl.observations import (
    compare_slots, native_absent_relations, normalize_slot, observation_spans, parse_observations, slot_id,
    required_observations_verified, verified_wrong_slots,
)
from fine_tuning.notebook_rl.schema import VerifierItem, to_json


def item():
    return VerifierItem("fixture", "int", "1", {}, (),
                        {"bar[Mar]": "4", "wire[R1]": "N0-N1", "note[1.1]": "C#4"},
                        {"bar[Mar]": "number", "wire[R1]": "relation", "note[1.1]": "pitch"},
                        ("bar[Mar]", "wire[R1]"), (), ())


def native_graph(kind="bfs"):
    slots = {"edge[A,B]": "1", "edge[D,H]": "1", "edge[D,E]": "1"}
    world = {"kind": kind, "nodes": list("ABCDEFGH"),
             "positions": {node: index for index, node in enumerate("ABEDCFGH")},
             "edges": [["A", "B", 1], ["D", "H", 1], ["D", "E", 1]]}
    return VerifierItem("gen-graph-v1:graph:train:fixture", "list", "A,B", {"graph_world": world},
                        (), slots, dict.fromkeys(slots, "number"), tuple(slots), (), ())


def native_tree():
    world = {"kind": "tree", "root": "D", "children": {
        "D": {"L": "B", "R": "F"}, "B": {"L": "C", "R": "G"},
        "F": {"L": "A"}, "A": {"R": "E"}}}
    slots = {"root": "D", "child[D,L]": "B", "child[D,R]": "F", "child[B,L]": "C",
             "child[B,R]": "G", "child[F,L]": "A", "child[A,R]": "E"}
    return VerifierItem("gen-graph-v1:graph:train:tree-fixture", "list", "D,B,C,G,F,A,E",
                        {"graph_world": world}, (), slots, dict.fromkeys(slots, "relation"), tuple(slots), (), ())


class ObservationTests(unittest.TestCase):
    def test_bounds_only(self):
        text = "bar[Mar] = 99\nObserved facts:\n- bar[Mar] = 4\nSolution:\nbar[Mar] = 99\nFinal answer: 1"
        self.assertEqual(parse_observations(text), {"bar[mar]": ["4"]})
        for text in ("bar[Mar] = 4", "Observed facts:\nbar[Mar] = 4", "Solution:\nbar[Mar] = 4"):
            self.assertEqual(parse_observations(text), {})

    def test_assignment_offsets_do_not_include_identical_solution_prose(self):
        text = "한글\nObserved facts:\nbar[Mar] = 8\nSolution:\nbar[Mar] = 8\n"
        spans = observation_spans(text)
        self.assertEqual(len(spans), 1)
        start, end, key, value = spans[0]
        self.assertEqual((text[start:end], key, value), ("bar[Mar] = 8\n", "bar[mar]", "8"))
        self.assertLessEqual(end, text.index("Solution:"))

    def test_key_variants(self):
        self.assertEqual(slot_id('Cell[ " Row A " ][ \' COL B\']'), "cell[rowa][colb]")
        parsed = parse_observations("Observed facts:\nRINGS = 2\n- cell[ a ][b] = 3\nSolution:")
        self.assertEqual(parsed, {"rings": ["2"], "cell[a][b]": ["3"]})

    def test_crlf_headers_and_repeated_blocks_keep_original_offsets(self):
        text = "한글\nObserved facts:\nbar[Mar] = 4\nSolution:\nbar[Mar] = 99\nObserved facts:\nbar[Mar] = 8\nSolution:\n"
        for newline in ("\n", "\r\n"):
            raw = text.replace("\n", newline)
            spans = observation_spans(raw)
            self.assertEqual([raw[start:end] for start, end, _, _ in spans],
                             ["bar[Mar] = 4" + newline, "bar[Mar] = 8" + newline])
            self.assertEqual(parse_observations(raw), {"bar[mar]": ["4", "8"]})
            self.assertEqual(compare_slots(parse_observations(raw), item()).conflicting, ("bar[mar]",))

    def test_numeric_observations(self):
        for value in ("4", "4.0", "4 kN", "4kN", "4.00 V", "4e0", "8/2"):
            self.assertEqual(normalize_slot(value, "number"), "4")
        self.assertEqual(normalize_slot("1/2 ohm", "number"), "1/2")
        for value in ("4 or 8", "4@2", "4,8", "nan", "four", "4 - 8"):
            self.assertIsNone(normalize_slot(value, "number"))

    def test_pitch(self):
        for value in ("C#4", "C♯4", "c sharp 4", "Ｃ＃４"):
            self.assertEqual(normalize_slot(value, "pitch"), "C#4")
        for value in ("Bb3", "B♭3", "b flat 3"):
            self.assertEqual(normalize_slot(value, "pitch"), "Bb3")
        self.assertIsNone(normalize_slot("C#", "pitch"))
        self.assertNotEqual(normalize_slot("Db4", "pitch"), normalize_slot("C#4", "pitch"))

    def test_complete_and_order_independent(self):
        report = compare_slots({"note[1.1]": ["C♯4"], "wire[r1]": ["N1-N0"], "bar[mar]": ["4.0"]}, item())
        self.assertEqual((report.parsed, report.required_correct, report.all_correct), (3, 2, 3))
        self.assertEqual(report.wrong_required, ())
        to_json(report)

    def test_identical_duplicates_not_conflicts(self):
        report = compare_slots({"bar[mar]": ["4", "4.0", "8/2"]}, item())
        self.assertEqual((report.parsed, report.required_correct, report.conflicting), (1, 1, ()))

    def test_conflicting_duplicate_not_cherry_picked(self):
        report = compare_slots({"bar[mar]": ["4", "8"]}, item())
        self.assertEqual((report.required_correct, report.wrong_required, report.conflicting),
                         (0, ("bar[mar]",), ("bar[mar]",)))

    def test_missing_unknown_invalid_distinct(self):
        report = compare_slots({"invented": ["99"]}, item())
        self.assertEqual((report.parsed, report.wrong_required), (0, ()))
        report = compare_slots({"bar[mar]": ["unreadable"]}, item())
        self.assertEqual((report.parsed, report.required_correct, report.wrong_required), (1, 0, ()))
        self.assertFalse(required_observations_verified(report))

    def test_directed_values_keep_order(self):
        q = replace(item(), slots={"dir[e1]": "A->B", "trans[q0,0]": "q1"},
                    slot_types={"dir[e1]": "relation", "trans[q0,0]": "relation"}, required_slots=("dir[e1]",))
        self.assertEqual(compare_slots({"dir[e1]": ["B->A"]}, q).wrong_required, ("dir[e1]",))
        self.assertEqual(compare_slots({"dir[e1]": ["a → b"], "trans[q0,0]": ["Q1"]}, q).all_correct, 2)

    def test_multiple_blocks_preserve_conflicts(self):
        text = "Observed facts:\nbar[Mar] = 4\nSolution:\nfirst\nObserved facts:\nbar[Mar] = 8\nSolution:"
        self.assertEqual(compare_slots(parse_observations(text), item()).conflicting, ("bar[mar]",))

    def test_quoted_relations_preserve_direction_states_and_original_spans(self):
        facts = {"dir[e0]": "Node_A->Node_D", "trans[q0,0]": "state_final", "start": "state_0",
                 "root": "root_node", "child[F,L]": "leaf_1"}
        q = replace(item(), slots=facts, slot_types=dict.fromkeys(facts, "relation"), required_slots=tuple(facts))
        for quote in ('"', "'"):
            with self.subTest(quote=quote):
                text = "한글\r\nObserved facts:\r\n" + "".join(
                    f"{key} = {quote}{value}{quote}\r\n" for key, value in facts.items()) + "Solution:\r\n"
                original = text
                spans = observation_spans(text)
                parsed = parse_observations(text)
                report = compare_slots(parsed, q)
                self.assertTrue(required_observations_verified(report))
                self.assertEqual(verified_wrong_slots(report), ())
                self.assertEqual([text[start:end] for start, end, _, _ in spans],
                                 [f"{key} = {quote}{value}{quote}\r\n" for key, value in facts.items()])
                self.assertEqual([value for _, _, _, value in spans],
                                 [quote + value + quote for value in facts.values()])
                self.assertEqual(text, original)
                self.assertEqual(compare_slots(parsed, replace(q, gold="unrelated", answer_type="short_text")), report)
                reversed_report = compare_slots(parsed | {"dir[e0]": [quote + "Node_D->Node_A" + quote]}, q)
                self.assertEqual(verified_wrong_slots(reversed_report), ("dir[e0]",))
        self.assertEqual(normalize_slot('"N1 - N0"', "relation"), "n0-n1")

    def test_malformed_native_relations_remain_unknown_and_cannot_certify_complete_observations(self):
        cases = {"dir[e0]": ("A->D", "A -> D or D -> A", "A D", "A-D", "A,D"),
                 "trans[q0,0]": ("q1", "q0 or q1", "q 1", "q0->q1"),
                 "start": ("q0", "q0 or q1", "q 0"),
                 "root": ("F", "F or A", "F A"),
                 "child[F,L]": ("A", "A or F", "A-F")}
        for key, (expected, *malformed) in cases.items():
            q = replace(item(), slots={key: expected}, slot_types={key: "relation"}, required_slots=(key,))
            malformed += ["?", "unreadable?", '"' + expected, expected + '"',
                          '"' + expected + "'", '"\"' + expected + '\""', "'\"" + expected + "\"'",
                          '"' + expected + ' prose"', '“' + expected + '”', '＂' + expected + '＂']
            for value in malformed:
                with self.subTest(key=key, value=value):
                    report = compare_slots({key: [value]}, q)
                    self.assertEqual((report.parsed, report.required_correct, verified_wrong_slots(report)), (1, 0, ()))
                    self.assertFalse(required_observations_verified(report))
        self.assertIsNone(normalize_slot('"N0 or N1"', "relation"))
        for key in ("trans[q0,0]", "start", "root", "child[F,L]"):
            q = replace(item(), slots={key: "known_node"}, slot_types={key: "relation"}, required_slots=(key,))
            self.assertEqual(verified_wrong_slots(compare_slots({key: ["unknown"]}, q)), (slot_id(key),))

    def test_unknown_duplicate_cannot_hide_wrong_or_certify_correct_relation(self):
        q = replace(item(), slots={"dir[e0]": "A->D"}, slot_types={"dir[e0]": "relation"}, required_slots=("dir[e0]",))
        for known, wrong in (('"A->D"', ()), ('"D->A"', ("dir[e0]",))):
            for values in ([known, "unreadable?"], ["unreadable?", known]):
                with self.subTest(values=values):
                    report = compare_slots({"dir[e0]": values}, q)
                    self.assertEqual((report.parsed, report.required_correct, report.conflicting), (1, 0, ("dir[e0]",)))
                    self.assertEqual(verified_wrong_slots(report), wrong)
                    self.assertFalse(required_observations_verified(report))

    def test_bad_reference_rejected(self):
        for q in (replace(item(), slots={"bar[X]": "1", "bar[x]": "2"},
                          slot_types={"bar[X]": "number", "bar[x]": "number"}, required_slots=()),
                  replace(item(), slots=item().slots | {"bar[Mar]": "not numeric"})):
            with self.assertRaises(ValueError):
                compare_slots({}, q)


class NativeRelationTests(unittest.TestCase):
    def test_reversed_and_invented_edges_preserve_registered_coverage(self):
        for kind in ("bfs", "shortest"):
            q = native_graph(kind)
            positive = {key: [value] for key, value in q.slots.items()}
            baseline = compare_slots(positive, q)
            report = compare_slots(positive | {"edge[H,D]": ["1"], "edge[E,D]": ["1"],
                                                "edge[A,H]": ["2"]}, q)
            self.assertEqual(report.false_relations, ("edge[a,h]", "edge[e,d]", "edge[h,d]"))
            self.assertEqual(report.unverified_relations, ())
            self.assertEqual(replace(report, false_relations=()), baseline)
            self.assertEqual(report.wrong_required, ())
            self.assertEqual((report.parsed, report.required_correct, report.all_correct), (3, 3, 3))
            self.assertEqual(verified_wrong_slots(report), report.false_relations)
            to_json(report)

    def test_registered_wrong_weights_still_use_wrong_required(self):
        report = compare_slots({"edge[A,B]": ["2"], "edge[D,H]": ["1"], "edge[H,D]": ["1"]},
                               native_graph())
        self.assertEqual((report.wrong_required, report.false_relations),
                         (("edge[a,b]",), ("edge[h,d]",)))
        self.assertEqual(verified_wrong_slots(report), ("edge[a,b]", "edge[h,d]"))
        self.assertEqual((report.parsed, report.required_correct), (2, 1))

    def test_duplicates_cannot_hide_false_or_registered_wrong_assertions(self):
        for values, conflict in ((["1", "1.0"], False), (["1", "0"], True),
                                  (["1", "2"], True), (["1", "unreadable"], True)):
            with self.subTest(values=values):
                report = compare_slots({"edge[H,D]": values, "EDGE[ H, D ]": ["1"],
                                        "edge[A,B]": ["1", "2"]}, native_graph())
                self.assertEqual(report.false_relations, ("edge[h,d]",))
                self.assertEqual(report.unverified_relations, ())
                self.assertEqual(report.wrong_required, ("edge[a,b]",))
                self.assertEqual("edge[h,d]" in report.conflicting, conflict)
                self.assertIn("edge[a,b]", report.conflicting)
                self.assertEqual(report.parsed, 1)
                to_json(report)

    def test_native_looking_unknown_claims_are_only_unverified(self):
        for key, values in (("edge[Z,D]", ["1"]), ("edge[H][D]", ["1"]),
                            ("edge[H,D,E]", ["1"]), ("edge[H]", ["1"])):
            with self.subTest(key=key):
                report = compare_slots({key: values}, native_graph())
                self.assertEqual(report.unverified_relations, (slot_id(key),))
                self.assertEqual((report.false_relations, verified_wrong_slots(report)), ((), ()))
        for value in ("0", "-1", "nan", "inf", "1 or 2", "1V", "1,2", "1/0", "1e1001", ""):
            with self.subTest(value=value):
                report = compare_slots({"edge[H,D]": [value]}, native_graph())
                self.assertEqual(report.unverified_relations, ("edge[h,d]",))
                self.assertEqual(report.false_relations, ())

    def test_missing_edges_unknown_keys_and_prose_are_not_false_claims(self):
        parsed = parse_observations("Observed facts:\ninvented = 99\nH -> D\nSolution:\nedge[H,D] = 1")
        report = compare_slots(parsed | {"edge[H,D]": []}, native_graph())
        self.assertEqual((report.parsed, report.required_correct), (0, 0))
        self.assertEqual((report.false_relations, report.unverified_relations), ((), ()))

    def test_closed_world_requires_complete_matching_native_metadata(self):
        q = native_graph()
        world = q.accept["graph_world"]
        incomplete = {key: value for key, value in q.slots.items() if key != "edge[D,H]"}
        candidates = [replace(q, problem_id="other:graph:fixture"),
                      replace(q, problem_id="gen-graph-pair-v1:graph:fixture"),
                      replace(q, accept={}), replace(q, accept={"graph_world": {}}),
                      replace(q, accept={"graph_world": world | {"positions": {}}}),
                      replace(q, accept={"graph_world": world | {"edges": [["A", "Z", 1]]}}),
                      replace(q, accept={"graph_world": world | {"kind": "dfa"}}),
                      replace(q, slots={}, slot_types={}, required_slots=()),
                      replace(q, slots=incomplete, slot_types=dict.fromkeys(incomplete, "number"),
                              required_slots=tuple(incomplete)),
                      replace(q, slot_types=dict.fromkeys(q.slots, "text")),
                      replace(q, slots=q.slots | {"edge[A,B]": "2"})]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                report = compare_slots({"edge[H,D]": ["1"]}, candidate)
                self.assertEqual(report.false_relations, ())
                self.assertEqual(report.unverified_relations, ("edge[h,d]",))

    def test_relations_do_not_depend_on_gold_or_answer_type(self):
        q = native_graph()
        report = compare_slots({"edge[H,D]": ["1"]}, q)
        self.assertEqual(compare_slots({"edge[H,D]": ["1"]},
                                       replace(q, gold="unrelated", answer_type="short_text")), report)


class NativeTreeRelationTests(unittest.TestCase):
    def test_false_children_preserve_registered_counts_and_sparse_evidence(self):
        q = native_tree()
        facts = {key: [value] for key, value in q.slots.items()}
        baseline = compare_slots(facts, q)
        self.assertEqual((baseline.parsed, baseline.required_correct, baseline.all_correct), (7, 7, 7))
        self.assertEqual((baseline.false_relations, baseline.unverified_relations), ((), ()))
        for observed in ({}, facts):
            report = compare_slots(observed | {"child[F,R]": ["E"]}, q)
            self.assertEqual(report.false_relations, ("child[f,r]",))
            self.assertEqual(replace(report, false_relations=()), compare_slots(observed, q))
        registered = compare_slots({"child[F,L]": ["E"]}, q)
        self.assertEqual((registered.wrong_required, registered.false_relations), (("child[f,l]",), ()))

    def test_normalized_duplicate_and_null_claims_cannot_erase_false_child(self):
        for values, conflict in ((["E", "e"], False), (["E", "null"], True),
                                  (["E", "unknown"], True), (["E", "X"], True), (["E", "A"], True)):
            with self.subTest(values=values):
                report = compare_slots({"child[F,R]": values, "CHILD[ 'Ｆ', \"Ｒ\" ]": ["Ｅ"]}, native_tree())
                self.assertEqual(report.false_relations, ("child[f,r]",))
                self.assertEqual(report.unverified_relations, ())
                self.assertEqual(report.conflicting, ("child[f,r]",) if conflict else ())
                self.assertEqual((report.parsed, report.required_correct), (0, 0))
                to_json(report)
        parsed = parse_observations("Observed facts:\nCHILD[ 'Ｆ', \"Ｒ\" ] = Ｅ\nSolution:")
        self.assertEqual(compare_slots(parsed, native_tree()).false_relations, ("child[f,r]",))

    def test_unknown_malformed_prose_and_null_child_claims_are_unverified(self):
        for key in ("child[X,R]", "child[F,X]", "child[F][R]", "child[F,R,E]", "child[F]"):
            with self.subTest(key=key):
                report = compare_slots({key: ["E"]}, native_tree())
                self.assertEqual((report.false_relations, report.unverified_relations), ((), (slot_id(key),)))
        # Absence declarations stay unverified; they do not add correct facts to the denominator.
        for value in ("X", "unknown", "null", "None", "none", "E or A", "E->A", "E,A", "E-A", "", "1"):
            with self.subTest(value=value):
                report = compare_slots({"child[F,R]": [value]}, native_tree())
                self.assertEqual((report.false_relations, report.unverified_relations), ((), ("child[f,r]",)))
                self.assertEqual((report.parsed, report.required_correct, report.required), (0, 0, 7))
        parsed = parse_observations("Observed facts:\nF has right child E\nSolution:\nchild[F,R] = E")
        self.assertEqual(compare_slots(parsed | {"child[F,R]": []}, native_tree()), compare_slots({}, native_tree()))

    def test_tree_absence_proof_requires_complete_native_world_and_slot_types(self):
        q = native_tree()
        world = q.accept["graph_world"]
        incomplete = {key: value for key, value in q.slots.items() if key != "child[F,L]"}
        candidates = [replace(q, problem_id="other:graph:fixture"),
            replace(q, problem_id="gen-graph-pair-v1:graph:fixture"), replace(q, accept={}),
            replace(q, accept={"graph_world": {}}),
            replace(q, accept={"graph_world": world | {"children": {}}}),
            replace(q, accept={"graph_world": world | {"kind": "dfa"}}),
            replace(q, accept={"graph_world": world | {"root": "X"}}),
            replace(q, accept={"graph_world": world | {"children": world["children"] | {"F": {"R": "A"}}}}),
            replace(q, slots={}, slot_types={}, required_slots=()),
            replace(q, slots=incomplete, slot_types=dict.fromkeys(incomplete, "relation"), required_slots=tuple(incomplete)),
            replace(q, slot_types=dict.fromkeys(q.slots, "text")), replace(q, slots=q.slots | {"child[F,L]": "E"})]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                report = compare_slots({"child[F,R]": ["E"]}, candidate)
                self.assertEqual((report.false_relations, report.unverified_relations), ((), ("child[f,r]",)))
        for candidate, key, value in ((q, "edge[a,b]", "1"), (native_graph(), "child[f,r]", "E")):
            report = compare_slots({key: [value]}, candidate)
            self.assertEqual((report.false_relations, report.unverified_relations), ((), (key,)))

    def test_tree_proof_rejects_registered_malformed_and_unknown_keys_without_gold(self):
        q = native_tree()
        candidates = ("child[f,r]", "child[f,l]", "child[x,r]", "child[f,x]", "CHILD[F,R]", "invented")
        self.assertEqual(native_absent_relations(candidates, q), ("child[f,r]",))
        report = compare_slots({"child[F,R]": ["E"]}, q)
        unrelated_gold = replace(q, gold="unrelated", answer_type="short_text")
        self.assertEqual(compare_slots({"child[F,R]": ["E"]}, unrelated_gold), report)
        self.assertEqual(native_absent_relations(candidates, unrelated_gold), ("child[f,r]",))


if __name__ == "__main__":
    unittest.main()
