"""Program-authored fixtures exercise verifier logic, never model quality or training."""
import copy
import hashlib
import json
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl.generators import graph, graph_pair
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.reasoning_trace import (
    VERSION, check_trace, instruction_for_question, native_integer_value, native_scalar_line, validate_trace_report,
)


def fixture_trace(problem):
    world = problem.verifier.accept["graph_world"]
    kind = world["kind"]
    if kind == "shortest":
        nodes = graph.nx.shortest_path(graph._network(world), "A", "H", weight="weight")
        indexes = [next(i for i, edge in enumerate(world["edges"]) if edge[:2] == [a, b])
                   for a, b in zip(nodes, nodes[1:])]
        weights = [world["edges"][i][2] for i in indexes]
        return {"kind": kind, "nodes": nodes, "weights": weights, "cost": sum(weights),
                "edge_ids": [f"e{i}" for i in indexes] if problem.actor.source == graph_pair.GENERATOR_VERSION else None}
    if kind == "bfs":
        queue, seen, steps = ["A"], {"A"}, []
        while queue:
            node = queue.pop(0)
            new = sorted(b for a, b, _ in world["edges"] if a == node and b not in seen)
            steps.append({"node": node, "new": new})
            seen.update(new)
            queue.extend(new)
        return {"kind": kind, "steps": steps}
    if kind == "dfa":
        states = [world["start"]]
        for symbol in "0110":
            states.append(world["trans"][states[-1]][symbol])
        return {"kind": kind, "states": states,
                "decision": "accept" if states[-1] in world["accepting"] else "reject"}
    traversal = problem.lineage.template_id.rsplit("-", 1)[1]
    events = []

    def visit(node):
        events.append({"op": "enter", "node": node})
        if traversal == "preorder":
            events.append({"op": "emit", "node": node})
        for side in ("L", "R"):
            child = world["children"].get(node, {}).get(side)
            if child is not None:
                events.append({"op": "child", "parent": node, "side": side, "node": child})
                visit(child)
            if side == "L" and traversal == "inorder":
                events.append({"op": "emit", "node": node})
        if traversal == "postorder":
            events.append({"op": "emit", "node": node})
        events.append({"op": "exit", "node": node})

    visit(world["root"])
    return {"kind": kind, "traversal": traversal, "events": events}


def response(problem, trace):
    facts = "\n".join(f"{key} = {value}" for key, value in problem.verifier.slots.items())
    return f"Observed facts:\n{facts}\nSolution:\nTrace: {json.dumps(trace)}\nFinal answer: {problem.verifier.gold}"


def refresh_wire_summary(report):
    """Keep forged wire summaries consistent, so structural checks must catch the corruption."""
    rows = report["checks"]
    report["issues"] = [row for row in rows if row["status"] != "COMPLIANT"]
    report["status"] = ("VIOLATED" if any(row["status"] == "VIOLATED" for row in rows)
                        else "UNVERIFIABLE" if report["issues"] else "COMPLIANT")
    report["coverage"]["checked"] = sorted({r["rule"] for r in rows if r["status"] != "UNVERIFIABLE"})
    report["coverage"]["unverified"] = sorted({r["rule"] for r in rows if r["status"] == "UNVERIFIABLE"})
    report["coverage"]["complete"] = not report["coverage"]["unverified"]


class NativeReasoningTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = [graph.generate(i, 3407, "train", "obs") for i in (0, 1, 2, 3, 7, 11)]
        cls.pair = graph_pair.generate(0, 3407, "train", "obs")

    def audit(self, problem, trace, finish="stop", text=None):
        text = response(problem, trace) if text is None else text
        report = check_trace(actor=problem.actor, item=problem.verifier, text=text, finish_reason=finish)
        validate_trace_report(report, text, finish)
        return report

    def test_all_native_families_and_tree_traversals(self):
        for problem in [*self.problems, self.pair]:
            with self.subTest(template=problem.lineage.template_id):
                report = self.audit(problem, fixture_trace(problem))
                self.assertEqual(report["status"], "COMPLIANT")
                self.assertTrue(report["coverage"]["complete"])
                self.assertEqual(report["issues"], [])
                self.assertIn(VERSION, instruction_for_question(problem.actor.question, problem.actor.source))

    def test_instruction_field_specifications_require_json_strings_integers_and_null(self):
        problems = [graph.generate(i, 3407, "train", "obs") for i in (0, 1, 2, 3, 7, 11)]
        pair = graph_pair.generate(0, 3407, "train", "obs")
        for problem in [*problems, pair]:
            trace = fixture_trace(problem)
            instruction = instruction_for_question(problem.actor.question, problem.actor.source)
            with self.subTest(template=problem.lineage.template_id):
                self.assertIn("FIELD SPECIFICATIONS (descriptions, not output JSON):", instruction)
                self.assertIn("Within the Trace JSON, all node/state labels and edge IDs must be "
                              "double-quoted JSON strings.", instruction)
                self.assertIn("Numeric trace fields must be JSON integers, not strings, decimals, or booleans.", instruction)
                self.assertIn("Use actual labels and values; do not copy descriptions or placeholders.", instruction)
                self.assertNotIn("{", instruction)
                self.assertNotIn("}", instruction)
                self.assertIn('kind: "' + trace["kind"] + '"', instruction)
                if trace["kind"] == "shortest":
                    ids = ("array of edge-ID strings in path order" if problem is pair else
                           'the JSON null value (unquoted, not "null")')
                    self.assertIn("edge_ids: " + ids, instruction)
                    self.assertIn("weights: array of integer weights", instruction)
                    self.assertIn("cost: integer sum", instruction)
                elif trace["kind"] == "bfs":
                    self.assertIn("exactly node (dequeued node-label string) and new (array", instruction)
                elif trace["kind"] == "tree":
                    self.assertIn('traversal: "' + trace["traversal"] + '"', instruction)
                    self.assertIn('op "enter", "emit", or "exit"', instruction)
                    self.assertIn('op, parent, side, and node for op "child"', instruction)
                else:
                    self.assertIn("states: array of state-label strings", instruction)
                    self.assertIn('decision: the JSON string "accept" or "reject"', instruction)

    def test_unquoted_dfa_state_labels_remain_unverifiable_without_normalization(self):
        problem = graph.generate(2, 3407, "train", "obs")
        trace = fixture_trace(problem)
        payload = json.dumps(trace)
        unquoted = payload
        for state in set(trace["states"]):
            unquoted = unquoted.replace(json.dumps(state), state)
        text = response(problem, trace).replace(payload, unquoted)
        self.assertEqual(self.audit(problem, trace)["status"], "COMPLIANT")
        report = self.audit(problem, trace, text=text)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertIsNone(report["trace"])
        self.assertEqual([(row["rule"], row["status"]) for row in report["checks"]],
                         [("TRACE_FORMAT", "UNVERIFIABLE")])
        self.assertEqual(report["text_sha256"], hashlib.sha256(text.encode()).hexdigest())

    def test_correct_observation_reversed_when_used_even_with_correct_final(self):
        trace = fixture_trace(self.pair)
        trace["nodes"][1], trace["nodes"][2] = trace["nodes"][2], trace["nodes"][1]
        report = self.audit(self.pair, trace)
        self.assertEqual(report["status"], "VIOLATED")
        errors = [row for row in report["issues"] if row["rule"] == "DIRECTED_EDGE"]
        self.assertTrue(errors)
        self.assertTrue(all(ref["status"] == "MATCH" for row in errors for ref in row["observation_refs"]))
        self.assertTrue(any(ref["slot"].startswith("dir[") for row in errors for ref in row["observation_refs"]))

    def test_truncation_keeps_positive_trace_error_and_missing_final_is_unverified(self):
        trace = fixture_trace(self.pair)
        trace["edge_ids"][0] = "e99"
        raw = response(self.pair, trace).split("\nFinal answer:")[0]
        report = self.audit(self.pair, trace, "length", raw)
        self.assertEqual(report["status"], "VIOLATED")
        self.assertIn("RESPONSE_SCOPE", report["coverage"]["unverified"])
        self.assertFalse(report["coverage"]["complete"])

    def test_fabricated_edge_id_false_applied_weight_and_false_cost(self):
        for change, rule in (("id", "DIRECTED_EDGE"), ("weight", "EDGE_WEIGHT"), ("cost", "COST_SUM")):
            trace = fixture_trace(self.pair)
            if change == "id":
                trace["edge_ids"][0] = "e99"
            elif change == "weight":
                trace["weights"][0] += 1
                trace["cost"] += 1
            else:
                trace["cost"] += 1
            report = self.audit(self.pair, trace)
            self.assertTrue(any(r["rule"] == rule and r["status"] == "VIOLATED" for r in report["issues"]))

    def test_invalid_directed_walk_is_not_repaired(self):
        # Repeating a real edge cannot be silently removed to claim the optimum.
        problem = self.problems[0]
        trace = fixture_trace(problem)
        trace["nodes"].insert(1, trace["nodes"][0])
        trace["weights"].insert(0, 1)
        trace["cost"] += 1
        report = self.audit(problem, trace)
        self.assertEqual(report["status"], "VIOLATED")
        self.assertIn("MINIMALITY", report["coverage"]["unverified"])

    def test_real_nonminimal_alternate_path_is_rejected(self):
        problem = graph.generate(0, 40, "train", "obs")
        world = problem.verifier.accept["graph_world"]
        network = graph._network(world)
        paths = list(graph.nx.all_simple_paths(network, "A", "H"))
        nodes = max(paths, key=lambda path: sum(network[a][b]["weight"] for a, b in zip(path, path[1:])))
        weights = [network[a][b]["weight"] for a, b in zip(nodes, nodes[1:])]
        self.assertGreater(sum(weights), graph._dijkstra(world))
        report = self.audit(problem, {"kind": "shortest", "nodes": nodes,
            "edge_ids": None, "weights": weights, "cost": sum(weights)})
        self.assertEqual([row["rule"] for row in report["issues"]], ["MINIMALITY"])
        self.assertTrue(report["coverage"]["complete"])

    def test_native_shortest_requires_null_ids_pair_requires_real_ids(self):
        for problem in (self.problems[0], self.pair):
            trace = fixture_trace(problem)
            trace["edge_ids"] = [] if trace["edge_ids"] is None else None
            self.assertEqual(self.audit(problem, trace)["status"], "UNVERIFIABLE")
            trace["edge_ids"] = "null"
            self.assertEqual(self.audit(problem, trace)["status"], "UNVERIFIABLE")

    def test_bfs_fifo_direction_discovery_sort_and_completion(self):
        problem = self.problems[1]
        for change, rule in (("fifo", "BFS_FIFO"), ("new", "BFS_NEW_NEIGHBORS"), ("missing", "BFS_COMPLETENESS")):
            trace = fixture_trace(problem)
            if change == "fifo":
                trace["steps"][1]["node"] = "A"
            elif change == "new":
                trace["steps"][0]["new"].append("A")
            else:
                trace["steps"].pop()
            report = self.audit(problem, trace)
            self.assertTrue(any(row["rule"] == rule and row["status"] == "VIOLATED" for row in report["issues"]))
        # Find a native branching fixture so alphabetical enqueue order has a real counterexample.
        for seed in range(10):
            branch = graph.generate(1, seed, "train", "obs")
            trace = fixture_trace(branch)
            step = next((s for s in trace["steps"] if len(s["new"]) > 1), None)
            if step is not None:
                step["new"].reverse()
                self.assertEqual(self.audit(branch, trace)["status"], "VIOLATED")
                break
        else:
            self.fail("native branching fixture not found")

    def test_tree_false_root_child_stack_emit_order_and_incomplete(self):
        for problem in self.problems[3:]:
            for change in ("root", "child", "emit", "exit", "missing"):
                trace = fixture_trace(problem)
                events = trace["events"]
                if change == "root":
                    events[0]["node"] = "Z"
                elif change == "child":
                    next(e for e in events if e["op"] == "child")["node"] = "Z"
                elif change == "missing":
                    events.pop()
                else:
                    at = next(i for i, e in enumerate(events[1:], 1) if e["op"] == change)
                    events[at], events[at - 1] = events[at - 1], events[at]
                self.assertEqual(self.audit(problem, trace)["status"], "VIOLATED", (change, problem.lineage.template_id))

    def test_dfa_start_each_symbol_fixed_order_complete_word_and_acceptance(self):
        problem = self.problems[2]
        for change, rule in (("start", "DFA_START"), ("missing", "DFA_COMPLETE_WORD"),
                             ("order", "DFA_TRANSITION"), ("decision", "DFA_ACCEPTANCE")):
            trace = fixture_trace(problem)
            if change == "start":
                trace["states"][0] = "q3"
            elif change == "missing":
                trace["states"].pop(2)
            elif change == "order":
                trace["states"][2] = next(q for q in ("q0", "q1", "q2", "q3") if q != trace["states"][2])
            else:
                trace["decision"] = "reject" if trace["decision"] == "accept" else "accept"
            report = self.audit(problem, trace)
            self.assertTrue(any(row["rule"] == rule and row["status"] == "VIOLATED" for row in report["issues"]), change)

    def test_dfa_acceptance_unknown_names_exact_failed_prerequisite_without_using_gold(self):
        problem = self.problems[2]
        for states in (["q0"], ["q0"] * 4, ["q0"] * 6, ["q9"] * 4, ["q0"] * 4 + ["q9"]):
            trace = {"kind": "dfa", "states": states, "decision": "reject"}
            text = response(problem, trace)
            report = self.audit(problem, trace, text=text)
            acceptance = report["checks"][-1]
            cause = ({"rule": "DFA_COMPLETE_WORD", "locus": "states"} if len(states) != 5
                     else {"rule": "DFA_TRANSITION", "locus": "states[3:5]"})
            with self.subTest(states=states):
                self.assertEqual((acceptance["status"], acceptance["blocked_by"]), ("UNVERIFIABLE", [cause]))
                self.assertTrue(any(row["rule"] == cause["rule"] and row["locus"] == cause["locus"]
                                    and row["status"] == "VIOLATED" for row in report["checks"][:-1]))
                self.assertFalse(report["coverage"]["complete"])
                self.assertEqual(report, check_trace(actor=problem.actor,
                    item=replace(problem.verifier, gold="PRIVATE-CHANGED-GOLD"), text=text, finish_reason="stop"))
                self.assertEqual(report["trace"]["line"], "Trace: " + json.dumps(trace))
                for key in ("gold", "expected", "reference_trace", "replacement_trace", "graph_world"):
                    self.assertNotIn('"' + key + '"', json.dumps(report))

    def test_dfa_known_terminal_acceptance_does_not_certify_failed_transitions(self):
        problem = self.problems[2]
        trace = fixture_trace(problem)
        trace["states"][0] = trace["states"][3] = "q9"
        for valid in (True, False):
            if not valid:
                trace["decision"] = "reject" if trace["decision"] == "accept" else "accept"
            report = self.audit(problem, trace)
            last = next(row for row in report["checks"] if row["locus"] == "states[3:5]")
            self.assertEqual(last["status"], "VIOLATED")
            self.assertEqual((report["checks"][-1]["status"], report["checks"][-1]["blocked_by"]),
                             ("COMPLIANT" if valid else "VIOLATED", []))
            bad = copy.deepcopy(report)
            bad["checks"][-1].update(status="UNVERIFIABLE", blocked_by=[{
                "rule": "DFA_TRANSITION", "locus": "states[3:5]"}])
            refresh_wire_summary(bad)
            with self.assertRaises(ValueError):
                validate_trace_report(bad, response(problem, trace), "stop")

    def test_dfa_acceptance_wire_rejects_missing_wrong_unproven_or_definite_dependencies(self):
        problem = self.problems[2]
        for states in (["q9"] * 4, ["q9", "q9", "q0", "q0", "q9"]):
            trace = {"kind": "dfa", "states": states, "decision": "reject"}
            text = response(problem, trace)
            report = self.audit(problem, trace)
            for mutation in ("missing", "empty", "start", "earlier", "self", "duplicate", "private",
                             "unproven", "unknown_cause", "definite", "other_cause"):
                bad = copy.deepcopy(report)
                acceptance = bad["checks"][-1]
                cause = acceptance["blocked_by"][0]
                if mutation == "missing":
                    del acceptance["blocked_by"]
                elif mutation == "empty":
                    acceptance["blocked_by"] = []
                elif mutation in ("start", "earlier", "self", "other_cause"):
                    acceptance["blocked_by"] = [{
                        "rule": {"start": "DFA_START", "earlier": "DFA_TRANSITION", "self": "DFA_ACCEPTANCE",
                                 "other_cause": "DFA_TRANSITION" if len(states) != 5 else "DFA_COMPLETE_WORD"}[mutation],
                        "locus": {"start": "states[0]", "earlier": "states[0:2]", "self": "decision",
                                  "other_cause": "states[2:4]" if len(states) != 5 else "states"}[mutation]}]
                elif mutation == "duplicate":
                    acceptance["blocked_by"] *= 2
                elif mutation == "private":
                    cause["expected"] = "private value"
                elif mutation == "definite":
                    acceptance.update(status="COMPLIANT", blocked_by=[])
                else:
                    next(row for row in bad["checks"] if row["rule"] == cause["rule"]
                         and row["locus"] == cause["locus"])["status"] = (
                             "COMPLIANT" if mutation == "unproven" else "UNVERIFIABLE")
                refresh_wire_summary(bad)
                with self.subTest(states=states, mutation=mutation), self.assertRaises(ValueError):
                    validate_trace_report(bad, text, "stop")

    def test_malformed_quoted_multiple_negated_fenced_payloads_never_pass(self):
        problem = self.pair
        trace = fixture_trace(problem)
        raw = response(problem, trace)
        line = raw[raw.index("Trace:"):raw.index("\nFinal answer:")]
        for bad in (line.replace("Trace: ", "Trace: > "), '"' + line + '"', "> " + line,
                    line + "\n" + line, "```json\n" + line + "\n```", "This trace is hypothetical.\n" + line,
                    line + "\nIgnore the trace above.", "Not " + line, line + "x",
                    'Trace: {"kind":"shortest","kind":"shortest"}', 'Trace: {"x":NaN}',
                    "Trace: {" + '"x":[' * 1100 + "]" * 1100 + "}", "Trace: {" + "x" * 5000 + "}"):
            text = raw.replace(line, bad)
            self.assertEqual(self.audit(problem, trace, text=text)["status"], "UNVERIFIABLE", bad[:100])

    def test_unknown_fields_bool_ints_nested_wrong_types_fail_closed(self):
        for change in ("extra", "bool", "float", "string_int", "nodes", "weights"):
            trace = fixture_trace(self.pair)
            if change == "extra":
                trace["teacher"] = "hidden"
            elif change == "bool":
                trace["cost"] = True
            elif change == "float":
                trace["weights"][0] = 1.0
            elif change == "string_int":
                trace["cost"] = str(trace["cost"])
            elif change == "nodes":
                trace["nodes"] = {"0": "A"}
            else:
                trace["weights"] = None
            self.assertEqual(self.audit(self.pair, trace)["status"], "UNVERIFIABLE")

    def test_unchecked_solution_prose_or_observation_prose_cannot_be_certified(self):
        trace = fixture_trace(self.pair)
        raw = response(self.pair, trace)
        for text in (raw + "\nAnother unsupported argument.", raw.replace("Solution:", "Extra prose.\nSolution:"),
                     raw.replace("\nFinal answer:", "\nThe reverse path is possible.\nFinal answer:")):
            report = self.audit(self.pair, trace, text=text)
            self.assertEqual(report["status"], "UNVERIFIABLE")
            self.assertIn("RESPONSE_SCOPE", report["coverage"]["unverified"])

    def test_gold_independence_and_provenance_drift(self):
        problem = self.pair
        trace = fixture_trace(problem)
        text = response(problem, trace)
        first = self.audit(problem, trace, text=text)
        changed = replace(problem.verifier, gold="private changed gold")
        self.assertEqual(first, check_trace(actor=problem.actor, item=changed, text=text, finish_reason="stop"))
        for actor, item in ((replace(problem.actor, source="forged"), problem.verifier),
                            (replace(problem.actor, question=problem.actor.question + "wrong"), problem.verifier),
                            (replace(problem.actor, problem_id="foreign"), problem.verifier),
                            (problem.actor, replace(problem.verifier, slots={**problem.verifier.slots, "fake": "1"})),
                            (problem.actor, replace(problem.verifier, slot_types=dict.fromkeys(problem.verifier.slots, "text")))):
            report = check_trace(actor=actor, item=item, text=text, finish_reason="stop")
            self.assertEqual(report["status"], "UNVERIFIABLE")
            validate_trace_report(report, text, "stop")
        with self.assertRaises(ValueError):
            instruction_for_question("What about another word?", "gen-graph-v1")

    def test_original_offsets_hashes_and_no_solution_observation_alias(self):
        trace = fixture_trace(self.pair)
        raw = response(self.pair, trace)
        raw = raw.replace("\n", "\r\n")
        report = self.audit(self.pair, trace, text=raw)
        self.assertEqual(report["text_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        for row in report["checks"]:
            for ref in row["observation_refs"]:
                start, end = ref["span"]
                self.assertLess(end, raw.index("Solution:"))
                self.assertEqual(hashlib.sha256(raw[start:end].encode()).hexdigest(), ref["sha256"])

    def test_horizontal_header_and_trace_padding_preserves_original_evidence(self):
        trace = fixture_trace(self.pair)
        plain = response(self.pair, trace)
        for ending in ("\n", "\r\n"):
            for padding in (" ", "\t", " \t"):
                for prefix, delimiter, suffix in (("", " ", ""), ("\t ", "\t ", " \t"), ("", "", "\t")):
                    with self.subTest(ending=repr(ending), padding=repr(padding), prefix=repr(prefix),
                                      delimiter=repr(delimiter), suffix=repr(suffix)):
                        raw = plain.replace("Observed facts:\n", f"Observed facts:{padding}\n")
                        raw = raw.replace("Solution:\n", f"Solution:{padding}\n \t\n")
                        raw = raw.replace("Trace: ", f"{prefix}Trace:{delimiter}")
                        raw = raw.replace("\nFinal answer:", f"{suffix}\nFinal answer:").replace("\n", ending)
                        report = self.audit(self.pair, trace, text=raw)
                        self.assertEqual(report["status"], "COMPLIANT")
                        start, end = report["trace"]["span"]
                        line = prefix + "Trace:" + delimiter + json.dumps(trace) + suffix
                        self.assertEqual(raw[start:end], line)
                        self.assertEqual(report["trace"]["line"], line)
                        self.assertEqual(report["trace"]["sha256"], hashlib.sha256(line.encode()).hexdigest())
                        self.assertEqual(report["text_sha256"], hashlib.sha256(raw.encode()).hexdigest())
                        refs = [ref for row in report["checks"] for ref in row["observation_refs"]]
                        self.assertTrue(refs)
                        for ref in refs:
                            start, end = ref["span"]
                            self.assertLess(end, raw.index("Solution:"))
                            self.assertEqual(ref["sha256"], hashlib.sha256(raw[start:end].encode()).hexdigest())

    def test_horizontal_header_padding_does_not_erase_bfs_violations(self):
        problem = self.problems[1]
        trace = fixture_trace(problem)
        trace["steps"][0]["new"] = ["A"]
        raw = response(problem, trace).replace("Observed facts:\n", "Observed facts: \n")
        raw = raw.replace("Solution:\n", "Solution: \n")
        report = self.audit(problem, trace, text=raw)
        self.assertEqual(report["status"], "VIOLATED")
        self.assertEqual(report["checks"][0]["status"], "COMPLIANT")
        self.assertIn("BFS_NEW_NEIGHBORS", report["coverage"]["checked"])
        self.assertNotIn("TRACE_FORMAT", report["coverage"]["unverified"])

    def test_padded_headers_keep_split_json_and_nested_duplicate_keys_invalid(self):
        problem = self.problems[1]
        trace = fixture_trace(problem)
        raw = response(problem, trace).replace("Observed facts:\n", "Observed facts: \n")
        raw = raw.replace("Solution:\n", "Solution: \n")
        for malformed in (raw.replace('"new": [', '"new": [\n', 1),
                          raw.replace('{"node": "A",', '{"node": "A", "node": "A",', 1)):
            report = self.audit(problem, trace, text=malformed)
            self.assertIsNone(report["trace"])
            self.assertEqual([(row["rule"], row["status"]) for row in report["checks"]],
                             [("TRACE_FORMAT", "UNVERIFIABLE")])

    def test_header_or_trace_prose_controls_and_unicode_padding_remain_blocking(self):
        trace = fixture_trace(self.pair)
        plain = response(self.pair, trace)
        for padding in (" extra prose", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029",
                        "\r ", "\x00", "\x1b", "\x7f", "\u200b", "\u00a0"):
            for raw in (plain.replace("Observed facts:\n", f"Observed facts:{padding}\n"),
                        plain.replace("Solution:\n", f"Solution:{padding}\n"),
                        plain.replace("Trace: ", f"Trace:{padding} "),
                        plain.replace("Trace: ", f"{padding}Trace: "),
                        plain.replace("\nFinal answer:", f"{padding}\nFinal answer:")):
                with self.subTest(padding=repr(padding), text=repr(raw[:80])):
                    report = self.audit(self.pair, trace, text=raw)
                    self.assertEqual(report["status"], "UNVERIFIABLE")
                    self.assertFalse(report["coverage"]["complete"])

    def test_padded_trace_wire_rejects_trimmed_or_shifted_evidence(self):
        trace = fixture_trace(self.pair)
        raw = response(self.pair, trace).replace("Solution:\nTrace: ", "Solution: \t\r\n\tTrace:\t ")
        raw = raw.replace("\nFinal answer:", " \t\r\nFinal answer:")
        report = self.audit(self.pair, trace, text=raw)
        self.assertEqual(report["status"], "COMPLIANT")
        for trim_start, trim_end in ((1, 0), (0, 1), (1, 2)):
            bad = copy.deepcopy(report)
            start, end = bad["trace"]["span"]
            start, end = start + trim_start, end - trim_end
            bad["trace"] = {"span": [start, end], "line": raw[start:end],
                            "sha256": hashlib.sha256(raw[start:end].encode()).hexdigest()}
            with self.assertRaises(ValueError):
                validate_trace_report(bad, raw, "stop")

    def test_duplicate_or_oversized_observation_evidence_is_bounded_unverified(self):
        trace = fixture_trace(self.pair)
        raw = response(self.pair, trace)
        assignment = next(iter(self.pair.verifier.slots)) + " = 1\n"
        for copies in (1, 1000):
            text = raw.replace("Observed facts:\n", "Observed facts:\n" + assignment * copies)
            report = self.audit(self.pair, trace, text=text)
            self.assertEqual(report["status"], "UNVERIFIABLE")
            self.assertLess(len(json.dumps(report)), 65536)

    def test_report_wire_rejects_forged_shapes_unbound_hashes_nonfinite_and_private_loci(self):
        trace = fixture_trace(self.pair)
        text = response(self.pair, trace)
        report = self.audit(self.pair, trace)
        mutations = [lambda r: r.update(extra="hidden"), lambda r: r.update(text_sha256="0" * 64),
                     lambda r: r.update(finish_reason="length"), lambda r: r.update(trace=None),
                     lambda r: r["checks"][0].update(locus="private expected value"),
                     lambda r: r["checks"][0].update(observation_refs=[{"value": float("nan")}]),
                     lambda r: r["trace"].update(span=[False, 1]),
                     lambda r: r["coverage"].update(complete=1)]
        for mutate in mutations:
            bad = copy.deepcopy(report)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_trace_report(bad, text, "stop")
        bad = copy.deepcopy(report)
        ref = next(row["observation_refs"][0] for row in bad["checks"] if row["observation_refs"])
        ref["span"] = [text.index("Solution:"), len(text)]
        with self.assertRaises(ValueError):
            validate_trace_report(bad, text, "stop")

    def test_v2_missing_edge_weight_and_minimality_name_exact_failed_prerequisite(self):
        trace = fixture_trace(self.pair)
        trace["edge_ids"][0] = "e99"
        report = self.audit(self.pair, trace)
        self.assertEqual(report["version"], "native-reasoning-trace-v4")
        weight = next(row for row in report["checks"] if row["locus"] == "weights[0]")
        minimum = next(row for row in report["checks"] if row["rule"] == "MINIMALITY")
        expected = [{"rule": "DIRECTED_EDGE", "locus": "nodes[0:2]"}]
        self.assertEqual(weight["status"], "UNVERIFIABLE")
        self.assertEqual(weight["blocked_by"], expected)
        self.assertEqual(minimum["blocked_by"], expected)
        self.assertTrue(all(row["blocked_by"] == [] for row in report["checks"] if row["status"] != "UNVERIFIABLE"))

    def test_v2_minimality_accounts_for_all_and_only_failed_path_prerequisites(self):
        trace = fixture_trace(self.pair)
        trace["nodes"][0] = "G"
        trace["weights"][0] += 1
        trace["cost"] += 2
        report = self.audit(self.pair, trace)
        failed = [{"rule": row["rule"], "locus": row["locus"]} for row in report["checks"]
                  if row["status"] == "VIOLATED"]
        self.assertEqual({row["rule"] for row in failed}, {"PATH_ENDPOINTS", "DIRECTED_EDGE", "EDGE_WEIGHT", "COST_SUM"})
        self.assertEqual(report["checks"][-1]["blocked_by"], failed)
        text = response(self.pair, trace)
        bad = copy.deepcopy(report)
        bad["checks"][-1]["blocked_by"].pop()
        refresh_wire_summary(bad)
        with self.assertRaises(ValueError):
            validate_trace_report(bad, text, "stop")

    def test_v2_earlier_violation_of_unrelated_rule_cannot_block_weight(self):
        trace = fixture_trace(self.pair)
        trace["nodes"][0] = "G"
        trace["edge_ids"][0] = "e99"
        text = response(self.pair, trace)
        report = self.audit(self.pair, trace)
        self.assertEqual(next(row for row in report["checks"] if row["rule"] == "PATH_ENDPOINTS")["status"], "VIOLATED")
        bad = copy.deepcopy(report)
        weight = next(row for row in bad["checks"] if row["locus"] == "weights[0]")
        weight["blocked_by"] = [{"rule": "PATH_ENDPOINTS", "locus": "nodes"}]
        refresh_wire_summary(bad)
        with self.assertRaises(ValueError):
            validate_trace_report(bad, text, "stop")

    def test_v2_bfs_entire_submitted_suffix_is_explicitly_unverified(self):
        problem = self.problems[1]
        for failure in ("fifo", "neighbors"):
            trace = fixture_trace(problem)
            trace["steps"].extend([{"node": "Z", "new": ["A"]}, {"node": "B", "new": []}])
            if failure == "fifo":
                trace["steps"][1]["node"] = "A"
                first = {"rule": "BFS_FIFO", "locus": "steps[1].node"}
            else:
                trace["steps"][1]["new"] = ["A"]
                first = {"rule": "BFS_NEW_NEIGHBORS", "locus": "steps[1].new"}
            report = self.audit(problem, trace)
            identities = [(row["rule"], row["locus"]) for row in report["checks"]]
            first_index = identities.index((first["rule"], first["locus"]))
            suffix = report["checks"][first_index + 1:]
            self.assertTrue(suffix)
            self.assertTrue(all(row["status"] == "UNVERIFIABLE" and row["blocked_by"] == [first]
                                and row["observation_refs"] == [] for row in suffix))
            for i in range(2, len(trace["steps"])):
                self.assertIn(("BFS_FIFO", f"steps[{i}].node"), identities)
                self.assertIn(("BFS_NEW_NEIGHBORS", f"steps[{i}].new"), identities)
            self.assertEqual(identities[-1], ("BFS_COMPLETENESS", "steps"))
            self.assertFalse(report["coverage"]["complete"])

    def test_v2_bfs_missing_duplicate_or_out_of_range_region_rejected(self):
        problem = self.problems[1]
        trace = fixture_trace(problem)
        trace["steps"][1]["node"] = "A"
        text = response(problem, trace)
        report = self.audit(problem, trace)
        for mutation in ("missing", "duplicate", "outside"):
            bad = copy.deepcopy(report)
            at = next(i for i, row in enumerate(bad["checks"]) if row["locus"] == "steps[2].new")
            if mutation == "missing":
                bad["checks"].pop(at)
            elif mutation == "duplicate":
                bad["checks"].insert(at, copy.deepcopy(bad["checks"][at]))
            else:
                bad["checks"][at]["locus"] = "steps[31].new"
            refresh_wire_summary(bad)
            with self.assertRaises(ValueError):
                validate_trace_report(bad, text, "stop")

    def test_v2_dependencies_reject_missing_nonviolated_forward_self_duplicate_and_private_refs(self):
        problem = self.problems[1]
        trace = fixture_trace(problem)
        trace["steps"][1]["node"] = "A"
        text = response(problem, trace)
        report = self.audit(problem, trace)
        mutations = [lambda row: row.pop("blocked_by"), lambda row: row.update(blocked_by=[]),
                     lambda row: row.update(blocked_by=[{"rule": "BFS_FIFO", "locus": "steps[0].node"}]),
                     lambda row: row.update(blocked_by=[{"rule": "BFS_FIFO", "locus": "steps[3].node"}]),
                     lambda row: row.update(blocked_by=[{"rule": row["rule"], "locus": row["locus"]}]),
                     lambda row: row["blocked_by"].append(copy.deepcopy(row["blocked_by"][0])),
                     lambda row: row["blocked_by"][0].update(expected="private value"),
                     lambda row: row["blocked_by"][0].update(locus="private expected value")]
        for mutate in mutations:
            bad = copy.deepcopy(report)
            row = next(r for r in bad["checks"] if r["locus"] == "steps[2].new")
            mutate(row)
            refresh_wire_summary(bad)
            with self.assertRaises(ValueError):
                validate_trace_report(bad, text, "stop")
        bad = copy.deepcopy(report)
        next(row for row in bad["checks"] if row["locus"] == "steps[1].node")["status"] = "COMPLIANT"
        refresh_wire_summary(bad)
        with self.assertRaises(ValueError):
            validate_trace_report(bad, text, "stop")

    def test_v2_response_prose_stays_separate_from_algorithm_dependencies(self):
        for problem in (self.pair, self.problems[1]):
            trace = fixture_trace(problem)
            text = response(problem, trace) + "\nUnchecked substantive explanation."
            report = self.audit(problem, trace, text=text)
            self.assertEqual(report["status"], "UNVERIFIABLE")
            self.assertEqual([(row["rule"], row["blocked_by"]) for row in report["issues"]], [("RESPONSE_SCOPE", [])])
            dependent_rule = "MINIMALITY" if trace["kind"] == "shortest" else "BFS_COMPLETENESS"
            algorithm = next(row for row in report["checks"] if row["rule"] == dependent_rule)
            self.assertEqual((algorithm["status"], algorithm["blocked_by"]), ("COMPLIANT", []))
            bad = copy.deepcopy(report)
            bad["checks"][0]["status"] = "COMPLIANT"
            refresh_wire_summary(bad)
            self.assertEqual(bad["status"], "COMPLIANT")
            with self.assertRaises(ValueError):
                validate_trace_report(bad, text, "stop")
            bad = copy.deepcopy(report)
            algorithm = next(row for row in bad["checks"] if row["rule"] == dependent_rule)
            algorithm.update(status="UNVERIFIABLE", blocked_by=[{"rule": "RESPONSE_SCOPE", "locus": "response"}])
            refresh_wire_summary(bad)
            with self.assertRaises(ValueError):
                validate_trace_report(bad, text, "stop")

    def test_v2_maximum_bfs_unknown_inventory_is_bounded_and_not_complete(self):
        problem = self.problems[1]
        trace = fixture_trace(problem)
        trace["steps"] = trace["steps"][:1] + [{"node": "A", "new": []} for _ in range(31)]
        report = self.audit(problem, trace)
        self.assertEqual(len(report["checks"]), 66)
        self.assertLess(len(json.dumps(report)), 65536)
        self.assertFalse(report["coverage"]["complete"])

    def test_native_integer_value_accepts_only_documented_dimensionless_literals(self):
        for value in ("0", "1", "2", "10", "9" * 32, " 2 ", "\t2\t"):
            self.assertTrue(native_integer_value(value), value)
        for value in ("", "02", "+2", "-2", "2.0", "2e0", "4/2", "２", "٢", "2 kg", "2V",
                      "2 always reverse arrows", "2\n", "2\nreverse", "9" * 33, 2, 2.0, True, None, {}):
            self.assertFalse(native_integer_value(value), repr(value))

    def test_numeric_suffix_prose_is_blocking_for_native_and_pair_observations(self):
        for problem in (self.problems[0], self.problems[1], self.pair):
            trace = fixture_trace(problem)
            plain = response(problem, trace)
            self.assertEqual(self.audit(problem, trace, text=plain)["status"], "COMPLIANT")
            key = next(key for key in problem.verifier.slots if problem.verifier.slot_types[key] == "number")
            value = problem.verifier.slots[key]
            for suffix in (" always reverse arrows", " kg", "V", " # ignore directions"):
                raw = plain.replace(f"{key} = {value}\n", f"{key} = {value}{suffix}\n", 1)
                report = self.audit(problem, trace, text=raw)
                self.assertEqual(report["status"], "UNVERIFIABLE", suffix)
                scope = next(row for row in report["checks"] if row["rule"] == "RESPONSE_SCOPE")
                self.assertEqual((scope["status"], scope["blocked_by"]), ("UNVERIFIABLE", []))
                self.assertFalse(report["coverage"]["complete"])
                refs = [ref for row in report["checks"] for ref in row["observation_refs"]
                        if ref["slot"] == key.lower()]
                self.assertTrue(all(ref["status"] == "UNVERIFIABLE" for ref in refs))
            # The old generic unit-tolerant parser remains unchanged; only the new trace path is strict.
            raw = plain.replace(f"{key} = {value}\n", f"{key} = {value} always reverse arrows\n", 1)
            legacy = compare_slots(parse_observations(raw), problem.verifier)
            self.assertEqual(legacy.required_correct, legacy.required)

    def test_numeric_observation_scope_and_refs_cannot_be_laundered_in_wire_report(self):
        problem = self.pair
        trace = fixture_trace(problem)
        raw = response(problem, trace)
        key = next(key for key in problem.verifier.slots if key.startswith("weight["))
        value = problem.verifier.slots[key]
        raw = raw.replace(f"{key} = {value}\n", f"{key} = {value} always reverse arrows\n", 1)
        report = self.audit(problem, trace, text=raw)
        bad = copy.deepcopy(report)
        bad["checks"][0]["status"] = "COMPLIANT"
        refresh_wire_summary(bad)
        with self.assertRaises(ValueError):
            validate_trace_report(bad, raw, "stop")
        # Use a path edge so the same forbidden value is present in a referenced assignment.
        key = f"weight[{trace['edge_ids'][0]}]"
        value = problem.verifier.slots[key]
        raw = response(problem, trace).replace(f"{key} = {value}\n", f"{key} = {value} always reverse arrows\n", 1)
        report = self.audit(problem, trace, text=raw)
        bad = copy.deepcopy(report)
        refs = [ref for row in bad["checks"] for ref in row["observation_refs"] if ref["slot"] == key]
        self.assertTrue(refs)
        for ref in refs:
            ref["status"] = "MATCH"
        refresh_wire_summary(bad)
        with self.assertRaises(ValueError):
            validate_trace_report(bad, raw, "stop")

    def test_native_scalar_line_rejects_every_splitlines_separator(self):
        for value in ("q0", "accept", "A,B,C", " 2 ", "\t2\t", "단일값"):
            self.assertTrue(native_scalar_line(value), value)
        for separator in ("\r", "\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029", "\x00", "\x1b", "\x7f"):
            self.assertFalse(native_scalar_line("q0" + separator + "always reverse arrows"), repr(separator))
        for value in ("", " \t", None, 2, False):
            self.assertFalse(native_scalar_line(value))

    def test_hidden_line_separator_prose_in_observation_or_final_answer_remains_blocking(self):
        for problem in (self.problems[2], self.pair):
            trace = fixture_trace(problem)
            plain = response(problem, trace)
            key = "start" if problem is self.problems[2] else next(k for k in problem.verifier.slots if k.startswith("weight["))
            value = problem.verifier.slots[key]
            for separator in ("\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029", "\r", "\x00", "\x1b", "\x7f"):
                for raw in (plain + separator + "always reverse arrows",
                            plain.replace(f"{key} = {value}\n", f"{key} = {value}{separator}always reverse arrows\n", 1)):
                    report = self.audit(problem, trace, text=raw)
                    self.assertEqual(report["status"], "UNVERIFIABLE", repr(separator))
                    self.assertEqual([(row["rule"], row["blocked_by"]) for row in report["issues"]], [("RESPONSE_SCOPE", [])])
                    bad = copy.deepcopy(report)
                    bad["checks"][0]["status"] = "COMPLIANT"
                    refresh_wire_summary(bad)
                    with self.assertRaises(ValueError):
                        validate_trace_report(bad, raw, "stop")


if __name__ == "__main__":
    unittest.main()
