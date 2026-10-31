"""Authored native transcripts test rewards, never model learning or improvement."""
import copy
import json
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.notebook_rl.generators import graph
from fine_tuning.notebook_rl.schema import ActorItem, Group, ProblemTree, Sample, VerifierItem, to_json
from fine_tuning.notebook_rl.verifier import Verifier


def native():
    world = {"kind": "dfa", "start": "q0", "accepting": ["q3"], "trans": {
        "q0": {"0": "q1", "1": "q0"}, "q1": {"0": "q1", "1": "q2"},
        "q2": {"0": "q3", "1": "q2"}, "q3": {"0": "q3", "1": "q0"}}}
    problem_id = "gen-graph-v1:graph:train:fixture"
    actor = ActorItem(problem_id, graph.GENERATOR_VERSION, "graph", "obs", graph._question("dfa", "postorder"),
                      (), "short_text", ("images/fixture.png",), ("a" * 64,))
    facts = graph._facts(world)
    item = VerifierItem(problem_id, "short_text", "accept", {"graph_world": world}, (), facts,
                        graph._slot_types(facts), tuple(facts), (), ())
    verifier = Verifier.__new__(Verifier)
    verifier.actors, verifier.items = {problem_id: actor}, {problem_id: item}
    return actor, item, verifier


def response(facts, *, wrong=True, answer=None, extra=""):
    trace = {"kind": "dfa", "states": ["q0", "q1", "q0", "q0", "q1"] if wrong
             else ["q0", "q1", "q2", "q2", "q3"], "decision": "reject" if wrong else "accept"}
    return ("Observed facts:\n" + "\n".join(f"{key} = {value}" for key, value in facts.items())
            + "\nSolution:\nTrace: " + json.dumps(trace, separators=(",", ":")) + "\n" + extra
            + "Final answer: " + (answer or ("reject" if wrong else "accept")))


def graph_native(kind):
    layout = "ABDECFGH" if kind == "bfs" else "ABDECHFG"
    world = {"kind": kind, "nodes": list("ABCDEFGH"), "positions": {node: i for i, node in enumerate(layout)},
             "edges": [["A", "B", 1], ["A", "C", 1], ["B", "D", 1]] if kind == "bfs"
             else [["A", "B", 2], ["A", "C", 4], ["B", "H", 2], ["C", "H", 4]]}
    problem_id = f"gen-graph-v1:graph:train:{kind}-fixture"
    answer_format = "list" if kind == "bfs" else "int"
    actor = ActorItem(problem_id, graph.GENERATOR_VERSION, "graph", "obs", graph._question(kind, "postorder"),
                      (), answer_format, ("images/fixture.png",), ("a" * 64,))
    facts = graph._facts(world)
    policy = {"ordered": True, "element_type": "short_text"} if kind == "bfs" else {}
    item = VerifierItem(problem_id, answer_format, "A,B,C,D" if kind == "bfs" else "4",
                        {"graph_world": world, **policy}, (), facts, graph._slot_types(facts), tuple(facts), (), ())
    verifier = Verifier.__new__(Verifier)
    verifier.actors, verifier.items = {problem_id: actor}, {problem_id: item}
    return actor, item, verifier


def graph_response(facts, trace, answer):
    return ("Observed facts:\n" + "\n".join(f"{key} = {value}" for key, value in facts.items())
            + "\nSolution:\nTrace: " + json.dumps(trace, separators=(",", ":")) + "\nFinal answer: " + answer)


def tree_from(actor, verifier, texts, *, finishes=None):
    samples = []
    for i, text in enumerate(texts):
        turn, kind, parent_index = (1, "T1", None) if i < 2 else (2, "T2", 0) if i < 4 else (3, "T3", 2)
        parent = samples[parent_index] if parent_index is not None else None
        prompt = [1, 2] if parent is None else parent.prompt_token_ids + parent.completion_token_ids + [900 + i // 2]
        finish = finishes[i] if finishes else "stop"
        sample = Sample(f"sample-{i+1}", actor.problem_id, kind, kind, turn,
                        parent.sample_id if parent else None, prompt, [100 + i], [-0.1], finish,
                        text, 3407 + i, 0, verifier.grade(actor.problem_id, text, finish, actor.mode))
        samples.append(sample)
    groups = [Group(kind, kind, actor.problem_id, "b" * 64, samples[i:i+2], True, None)
              for i, kind in ((0, "T1"), (2, "T2"), (4, "T3"))]
    return ProblemTree(actor.problem_id, 0, groups, "VISION_OFF", {}, {})


def endpoint(attempt, check, samples):
    return {"attempt": attempt, **{key: check[key] for key in ("kind", "rule", "locus")},
            "evidence_ranges": [[1, len(samples[attempt - 1].text.splitlines())]]
            if samples[attempt - 1].text else []}


def note_for(tree, audit, item):
    samples = solution_samples(tree)
    note = {"corrected_facts": item.slots, "attempts": [], "success_attempt": audit["success_attempt"],
            "corrected_answer": audit["attempts"][audit["success_attempt"] - 1]["answer"], "lessons": []}
    for attempt in audit["attempts"]:
        index = attempt["attempt"]
        note["attempts"].append({"attempt": index, "outcome": attempt["outcome"], "answer": attempt["answer"],
            "evidence_ranges": [[1, len(samples[index - 1].text.splitlines())]] if samples[index - 1].text else [],
            "issues": [{key: value for key, value in endpoint(index, issue, samples).items() if key != "attempt"}
                       for issue in attempt["issues"]]})
        if audit["version"] == synthesis.BOUNDED_VERSION:
            note["attempts"][-1]["unverified"] = [{
                **{key: value for key, value in endpoint(index, unknown, samples).items() if key != "attempt"},
                "blocked_by": unknown["blocked_by"]} for unknown in attempt["unverified"]]
        for issue in attempt["issues"]:
            candidates = [(row["attempt"], check) for row in audit["attempts"][index:]
                          for check in row["checks"] if check["identity"] == issue["identity"]
                          and check["status"] in ("VIOLATED", "COMPLIANT")
                          and not any(synthesis._key(unknown) == synthesis._key(check)
                                      for unknown in row["unverified"])]
            if candidates:
                descendants = [(later, target) for later, target in candidates
                               if synthesis._descendant(index, later, samples)]
                repairs = [(later, target) for later, target in descendants if target["status"] == "COMPLIANT"]
                later, target = (repairs or descendants or candidates)[0]
                relation = ("OTHER_BRANCH_COMPARISON" if not synthesis._descendant(index, later, samples)
                            else "PERSISTED" if target["status"] == "VIOLATED" else "CORRECTED")
                end = endpoint(later, target, samples)
            else:
                operation_unknowns = [(row["attempt"], check) for row in audit["attempts"][index:]
                                      for check in row["unverified"] if check["identity"] == issue["identity"]
                                      and synthesis._eligible_unknown(check, row["checks"], accounted=True)]
                scope_unknowns = [(row["attempt"], check) for row in audit["attempts"][index:]
                                  for check in row["unverified"] if synthesis._scope_unknown(check)]
                if audit["version"] == synthesis.BOUNDED_VERSION and (operation_unknowns or scope_unknowns):
                    later, target = (operation_unknowns or scope_unknowns)[0]
                    relation = "UNVERIFIABLE_LATER" if operation_unknowns else "SCOPE_UNVERIFIABLE_LATER"
                    end = endpoint(later, target, samples) | {"blocked_by": target["blocked_by"]}
                else:
                    relation = "NOT_REVISITED"
                    later = len(samples) if index < len(samples) else None
                    end = {"attempt": later, "kind": "ABSENCE", "rule": None, "locus": None,
                           "evidence_ranges": [[1, len(samples[-1].text.splitlines())]] if later else []}
            note["lessons"].append({"rule": issue["rule"], "earlier": endpoint(index, issue, samples),
                                    "later": end, "relation": relation})
    return copy.deepcopy(note)


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = native()
        self.texts = [response(self.item.slots)] * 4 + [response(self.item.slots, wrong=False)] * 2
        self.tree = tree_from(self.actor, self.verifier, self.texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier)
        self.note = note_for(self.tree, self.audit, self.item)

    def grade(self, note=None, *, tree=None, audit=None):
        return synthesis.grade_note(json.dumps(self.note if note is None else note), "stop", tree or self.tree,
                                    audit or self.audit, self.verifier, actor=self.actor)

    def test_full_synthesis_verifies_every_branch_and_preserves_raw(self):
        before = to_json(self.tree), copy.deepcopy(self.audit), copy.deepcopy(self.note)
        checked = self.grade()
        self.assertEqual(checked["reward"], 1.)
        self.assertTrue(checked["lesson_semantics_verified"])
        self.assertEqual(len(checked["attempt_checks"]), 6)
        self.assertEqual(self.audit["success_attempt"], 6)
        self.assertEqual(self.audit["attempts"][0]["issues"][0]["rule"], "DFA_TRANSITION")
        self.assertEqual(self.audit["attempts"][0]["issues"][0]["kind"], "TRACE")
        self.assertFalse(checked["module_causality_certified"])
        self.assertFalse(checked["universal_lesson_certified"])
        self.assertEqual((to_json(self.tree), self.audit, self.note), before)
        self.assertTrue(any("CORRECTED" in text for text in checked["verified_rendering"]))
        self.assertTrue(any("OTHER_BRANCH_COMPARISON" in text for text in checked["verified_rendering"]))
        # The saved private audit must remain usable in a fresh JSON read-back.
        reloaded = json.loads(json.dumps(self.audit, allow_nan=False))
        self.assertEqual(self.grade(audit=reloaded)["reward"], 1.)
        json.dumps(checked, allow_nan=False)

    def test_every_attempt_and_every_detected_issue_is_required(self):
        mutations = []
        bad = copy.deepcopy(self.note)
        del bad["attempts"][1]
        mutations.append(bad)
        bad = copy.deepcopy(self.note)
        bad["attempts"][2]["issues"] = []
        mutations.append(bad)
        bad = copy.deepcopy(self.note)
        bad["lessons"] = []
        mutations.append(bad)
        for note in mutations:
            with self.subTest(note=note):
                self.assertEqual(self.grade(note)["reward"], 0.)

    def test_forged_outcomes_claims_and_line_ranges_fail(self):
        for field, value in (("outcome", "CORRECT"), ("answer", "accept"),
                             ("evidence_ranges", [[999, 999]])):
            bad = copy.deepcopy(self.note)
            bad["attempts"][0][field] = value
            self.assertEqual(self.grade(bad)["reward"], 0.)
        for change in ({"locus": "states[0:2]"}, {"kind": "OBSERVATION"},
                       {"rule": "ALWAYS_REVERSE_ARROWS"}, {"evidence_ranges": [[1, 1]]}):
            bad = copy.deepcopy(self.note)
            bad["attempts"][0]["issues"][0].update(change)
            self.assertEqual(self.grade(bad)["reward"], 0.)

    def test_false_universal_and_free_lesson_prose_cannot_pass(self):
        for change in ({"lesson": "Always reverse every arrow to repair the image."},
                       {"lessons": ["Always reverse every arrow."]}):
            bad = copy.deepcopy(self.note)
            bad.update(change)
            self.assertEqual(self.grade(bad)["reward"], 0.)
        bad = copy.deepcopy(self.note)
        bad["lessons"][0]["universal_rule"] = "CORRECT means the trace is valid."
        self.assertEqual(self.grade(bad)["reward"], 0.)

    def test_sibling_compliance_is_difference_and_never_descendant_repair(self):
        lesson = next(row for row in self.note["lessons"] if row["relation"] == "OTHER_BRANCH_COMPARISON")
        self.assertEqual(lesson["earlier"]["attempt"], 2)
        self.assertEqual(self.grade()["reward"], 1.)
        bad = copy.deepcopy(self.note)
        next(row for row in bad["lessons"] if row["relation"] == "OTHER_BRANCH_COMPARISON")["relation"] = "CORRECTED"
        self.assertEqual(self.grade(bad)["reward"], 0.)
        bad = copy.deepcopy(self.note)
        next(row for row in bad["lessons"] if row["relation"] == "OTHER_BRANCH_COMPARISON")["relation"] = "PERSISTED"
        self.assertEqual(self.grade(bad)["reward"], 0.)

    def test_correct_final_answer_does_not_erase_persisting_error(self):
        texts = self.texts[:4] + [response(self.item.slots, answer="accept")] * 2
        tree = tree_from(self.actor, self.verifier, texts)
        audit = synthesis.audit_episode(tree, self.actor, self.verifier)
        note = note_for(tree, audit, self.item)
        checked = self.grade(note, tree=tree, audit=audit)
        self.assertEqual(checked["reward"], 1.)
        self.assertEqual(audit["attempts"][-1]["outcome"], "CORRECT")
        self.assertEqual(len(audit["attempts"][-1]["issues"]), 1)
        self.assertEqual(note["lessons"][-1]["relation"], "NOT_REVISITED")
        self.assertIsNone(note["lessons"][-1]["later"]["attempt"])
        bad = copy.deepcopy(note)
        bad["attempts"][-1]["issues"] = []
        self.assertEqual(self.grade(bad, tree=tree, audit=audit)["reward"], 0.)
        bad = copy.deepcopy(note)
        bad["lessons"][0]["relation"] = "CORRECTED"
        self.assertEqual(self.grade(bad, tree=tree, audit=audit)["reward"], 0.)

    def test_truncated_response_retains_real_trace_and_observation_errors(self):
        facts = self.item.slots | {"trans[q1,1]": "q0"}
        texts = self.texts.copy()
        texts[0] = response(facts)
        tree = tree_from(self.actor, self.verifier, texts, finishes=["length"] + ["stop"] * 5)
        audit = synthesis.audit_episode(tree, self.actor, self.verifier)
        first = audit["attempts"][0]
        self.assertEqual(first["outcome"], "TRUNCATED")
        self.assertEqual({issue["kind"] for issue in first["issues"]}, {"TRACE", "OBSERVATION"})
        self.assertNotIn("cause", first)
        note = note_for(tree, audit, self.item)
        checked = self.grade(note, tree=tree, audit=audit)
        self.assertEqual(checked["reward"], 1.)
        bad = copy.deepcopy(note)
        bad["attempts"][0]["issues"] = [issue for issue in bad["attempts"][0]["issues"] if issue["kind"] != "TRACE"]
        self.assertEqual(self.grade(bad, tree=tree, audit=audit)["reward"], 0.)

    def test_correct_observation_wrong_application_is_not_misread(self):
        first = self.audit["attempts"][0]
        self.assertTrue(all(check["status"] == "COMPLIANT" for check in first["checks"] if check["kind"] == "OBSERVATION"))
        self.assertEqual([issue["kind"] for issue in first["issues"]], ["TRACE"])
        bad = copy.deepcopy(self.note)
        bad["attempts"][0]["issues"][0].update(kind="OBSERVATION", rule="OBSERVATION_MATCH", locus="trans[q1,1]")
        self.assertEqual(self.grade(bad)["reward"], 0.)

    def test_absent_trace_and_extra_substantive_prose_are_incomplete(self):
        for text in ("Observed facts:\nSolution:\nFinal answer: reject",
                     response(self.item.slots, extra="Actually I reverse every state transition.\n")):
            texts = self.texts.copy()
            texts[0] = text
            tree = tree_from(self.actor, self.verifier, texts)
            audit = synthesis.audit_episode(tree, self.actor, self.verifier)
            note = note_for(tree, audit, self.item)
            self.assertFalse(audit["solution_trace_complete"])
            self.assertTrue(audit["attempts"][0]["unverified"])
            checked = self.grade(note, tree=tree, audit=audit)
            self.assertEqual(checked["reward"], 0.)
            self.assertFalse(checked["checks"]["solution_trace_complete"])
            self.assertFalse(checked["lesson_semantics_verified"])

    def test_gold_changes_only_answer_outcome_not_trace_or_observation_extraction(self):
        first = self.audit["attempts"][0]
        self.verifier.items[self.actor.problem_id] = replace(self.item, gold="reject")
        changed_trace = self.verifier.trace(self.actor.problem_id, self.texts[0], "stop")
        self.assertEqual(changed_trace, first["trace"])
        self.assertEqual(synthesis._observation_checks(solution_samples(self.tree)[0], self.verifier),
                         [check for check in first["checks"] if check["kind"] == "OBSERVATION"])
        with self.assertRaisesRegex(ValueError, "stored outcome"):
            synthesis.audit_episode(self.tree, self.actor, self.verifier)

    def test_fresh_reaudit_rejects_forged_report_and_broken_ancestry(self):
        forged = copy.deepcopy(self.audit)
        forged["attempts"][0]["issues"] = []
        with self.assertRaisesRegex(ValueError, "freshly audited"):
            self.grade(audit=forged)
        tree = copy.deepcopy(self.tree)
        tree.groups[-1].samples[0].prompt_token_ids = [10, 11]
        tree.groups[-1].samples[1].prompt_token_ids = [10, 11]
        with self.assertRaisesRegex(ValueError, "ancestry"):
            synthesis.audit_episode(tree, self.actor, self.verifier)

    def test_prompt_preserves_every_raw_line_and_keeps_private_audit_out(self):
        audit = copy.deepcopy(self.audit)
        audit["private_secret"] = "PRIVATE_REFERENCE_SENTINEL"
        prompt = synthesis.instruction(solution_samples(self.tree), audit, "PUBLIC_SLOT_SCHEMA")
        self.assertNotIn("PRIVATE_REFERENCE_SENTINEL", prompt)
        self.assertNotIn("text_sha256", prompt)
        rows = json.loads(prompt.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
        for row, sample in zip(rows, solution_samples(self.tree), strict=True):
            self.assertEqual("".join(line for _, line in row["source_lines"]), sample.text)
            self.assertEqual(row["answer"], sample.result.extracted)
            self.assertEqual(row["status"], sample.result.status)
        self.assertEqual([row["parent"] for row in rows], [None, None, 1, 1, 3, 3])

    def test_same_index_different_operands_never_proves_repair(self):
        first = {"kind": "dfa", "states": ["q0", "q1", "q0", "q0", "q1"]}
        second = {"kind": "dfa", "states": ["q0", "q2", "q0", "q0", "q1"]}
        self.assertNotEqual(synthesis._identity(first, "DFA_TRANSITION", "states[1:3]"),
                            synthesis._identity(second, "DFA_TRANSITION", "states[1:3]"))
        bfs1 = {"kind": "bfs", "steps": [{"node": "A", "new": ["B"]}, {"node": "B", "new": []}]}
        bfs2 = {"kind": "bfs", "steps": [{"node": "A", "new": ["C"]}, {"node": "B", "new": []}]}
        self.assertNotEqual(synthesis._identity(bfs1, "BFS_NEW_NEIGHBORS", "steps[1].new"),
                            synthesis._identity(bfs2, "BFS_NEW_NEIGHBORS", "steps[1].new"))
        tree1 = {"kind": "tree", "events": [{"op": "child", "parent": "A", "side": "L", "node": "B"}]}
        tree2 = {"kind": "tree", "events": [{"op": "child", "parent": "A", "side": "R", "node": "B"}]}
        self.assertNotEqual(synthesis._identity(tree1, "TREE_CHILD_EDGE", "events[0]"),
                            synthesis._identity(tree2, "TREE_CHILD_EDGE", "events[0]"))
        shortest1 = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None}
        shortest2 = {"kind": "shortest", "nodes": ["A", "C", "H"], "edge_ids": None}
        shortest3 = {"kind": "shortest", "nodes": ["C", "A", "B", "H"], "edge_ids": None}
        self.assertNotEqual(synthesis._identity(shortest1, "DIRECTED_EDGE", "nodes[0:2]"),
                            synthesis._identity(shortest2, "DIRECTED_EDGE", "nodes[0:2]"))
        self.assertEqual(synthesis._identity(shortest1, "DIRECTED_EDGE", "nodes[0:2]"),
                         synthesis._identity(shortest3, "DIRECTED_EDGE", "nodes[1:3]"))

    def test_compliant_same_index_from_different_state_cannot_be_claimed_as_correction(self):
        changed = response(self.item.slots, answer="accept").replace(
            '["q0","q1","q0","q0","q1"]', '["q0","q0","q0","q0","q1"]')
        tree = tree_from(self.actor, self.verifier, self.texts[:4] + [changed] * 2)
        audit = synthesis.audit_episode(tree, self.actor, self.verifier)
        note = note_for(tree, audit, self.item)
        target = next(check for check in audit["attempts"][4]["checks"]
                      if check["rule"] == "DFA_TRANSITION" and check["locus"] == "states[1:3]")
        self.assertEqual(target["status"], "COMPLIANT")
        origin = audit["attempts"][0]["issues"][0]
        forged = {"rule": origin["rule"], "earlier": endpoint(1, origin, solution_samples(tree)),
                  "later": endpoint(5, target, solution_samples(tree)), "relation": "CORRECTED"}
        note["lessons"].append(forged)
        checked = self.grade(note, tree=tree, audit=audit)
        self.assertEqual(checked["reward"], 0.)
        self.assertFalse(checked["lesson_checks"][-1]["verified"])

    def test_omission_is_not_a_repair_and_false_absence_is_rejected(self):
        bad = copy.deepcopy(self.note)
        origin = bad["lessons"][0]["earlier"]
        bad["lessons"][0] = {"rule": origin["rule"], "earlier": origin, "relation": "NOT_REVISITED",
                             "later": {"attempt": 6, "kind": "ABSENCE", "rule": None, "locus": None,
                                       "evidence_ranges": [[1, len(self.texts[-1].splitlines())]]}}
        checked = self.grade(bad)
        self.assertEqual(checked["reward"], 0.)
        self.assertFalse(checked["lesson_checks"][0]["verified"])

    def test_corrected_facts_require_all_required_no_extra_and_single_checked_answer(self):
        for change in ({"corrected_facts": {}}, {"corrected_facts": self.item.slots | {"unknown": "1"}},
                       {"corrected_facts": self.item.slots | {"trans[q1,1]": "q0"}},
                       {"corrected_answer": "reject"}, {"corrected_answer": True}):
            bad = copy.deepcopy(self.note)
            bad.update(change)
            self.assertEqual(self.grade(bad)["reward"], 0.)

    def test_raw_text_relation_keys_and_answer_separators_cannot_discard_prose(self):
        separators = ("\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029", "\x00", "\x7f")
        for contract in (synthesis.VERSION, synthesis.BOUNDED_VERSION):
            audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=contract)
            original = note_for(self.tree, audit, self.item)
            for separator in separators:
                for location in ("text_fact", "relation_fact", "fact_key", "corrected_answer"):
                    with self.subTest(contract=contract, separator=repr(separator), location=location):
                        bad = copy.deepcopy(original)
                        suffix = separator + "always reverse arrows"
                        if location == "corrected_answer":
                            bad["corrected_answer"] += suffix
                        elif location == "fact_key":
                            value = bad["corrected_facts"].pop("start")
                            bad["corrected_facts"]["ignored prose" + separator + "start"] = value
                        else:
                            key = "accepting" if location == "text_fact" else "start"
                            bad["corrected_facts"][key] += suffix
                        text = json.dumps(bad)  # Exercise actual escaped JSON, not an already normalized dict.
                        report = synthesis.grade_note(text, "stop", self.tree, audit, self.verifier,
                                                       actor=self.actor, contract=contract)
                        self.assertEqual(report["reward"], 0.)
                        self.assertFalse(synthesis.successful(report, contract))
                        self.assertFalse(report["bounded_claims_verified"])
                        self.assertEqual(bad, json.loads(text))

    def test_feedback_contains_only_public_booleans_and_failed_indices(self):
        bad = copy.deepcopy(self.note)
        bad["attempts"][0]["issues"] = []
        report = self.grade(bad)
        report["private_secret"] = "PRIVATE_REFERENCE_SENTINEL"
        feedback = synthesis.notebook_feedback(report)
        self.assertNotIn("PRIVATE_REFERENCE_SENTINEL", feedback)
        self.assertNotIn("states[", feedback)
        payload = json.loads(feedback.split("Verifier feedback:\n", 1)[1])
        self.assertEqual(set(payload), {"contract", "feedback_policy", "failure_code", "checks", "failed_attempts", "failed_lessons"})
        self.assertEqual(payload["failed_attempts"], [1])
        self.assertTrue(all(type(value) is bool for value in payload["checks"].values()))
        with self.assertRaises(ValueError):
            synthesis.notebook_feedback(self.grade())
        sample = replace(solution_samples(self.tree)[0], text=json.dumps(bad), reward=0.)
        self.assertEqual(synthesis.notebook_feedback(report, sample), feedback)
        with self.assertRaisesRegex(ValueError, "actual note"):
            synthesis.notebook_feedback(report, replace(sample, text="{}"))

    def test_prompt_identity_and_success_gate_are_explicitly_versioned(self):
        self.assertTrue(synthesis.successful(self.grade(), synthesis.VERSION))
        self.assertFalse(synthesis.successful(self.grade(), synthesis.BOUNDED_VERSION))
        self.assertNotEqual(synthesis.prompt_contract(synthesis.VERSION), synthesis.prompt_contract(synthesis.BOUNDED_VERSION))
        self.assertIn("{transcript}", synthesis.prompt_contract(synthesis.BOUNDED_VERSION))
        with self.assertRaises(ValueError):
            synthesis.prompt_contract("unversioned")


class BoundedSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = graph_native("bfs")
        wrong = {"kind": "bfs", "steps": [{"node": "A", "new": ["C", "B"]},
                 {"node": "C", "new": []}, {"node": "B", "new": ["D"]}, {"node": "D", "new": []}]}
        right = {"kind": "bfs", "steps": [{"node": "A", "new": ["B", "C"]},
                 {"node": "B", "new": ["D"]}, {"node": "C", "new": []}, {"node": "D", "new": []}]}
        texts = [graph_response(self.item.slots, wrong, "A,C,B,D")] * 4 + [graph_response(self.item.slots, right, "A,B,C,D")] * 2
        self.tree = tree_from(self.actor, self.verifier, texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.note = note_for(self.tree, self.audit, self.item)

    def grade(self, note=None):
        return synthesis.grade_note(json.dumps(self.note if note is None else note), "stop", self.tree, self.audit,
                                    self.verifier, actor=self.actor, contract=synthesis.BOUNDED_VERSION)

    def test_bounded_inventory_can_pass_while_full_reasoning_stays_unverified(self):
        self.assertFalse(self.audit["solution_trace_complete"])
        self.assertTrue(self.audit["unknown_regions_accountable"])
        self.assertEqual(len(self.audit["attempts"][0]["unverified"]), 7)
        report = self.grade()
        self.assertEqual(report["reward"], 1.)
        self.assertTrue(synthesis.successful(report, synthesis.BOUNDED_VERSION))
        self.assertTrue(report["bounded_claims_verified"])
        self.assertFalse(report["solution_trace_complete"])
        self.assertFalse(report["all_reasoning_verified"])
        self.assertFalse(report["full_diagnosis"])
        self.assertFalse(report["lesson_semantics_verified"])
        forged = copy.deepcopy(report)
        forged["full_diagnosis"] = True
        self.assertFalse(synthesis.successful(forged, synthesis.BOUNDED_VERSION))
        strict_audit = synthesis.audit_episode(self.tree, self.actor, self.verifier)
        strict_note = copy.deepcopy(self.note)
        for attempt in strict_note["attempts"]:
            del attempt["unverified"]
        strict = synthesis.grade_note(json.dumps(strict_note), "stop", self.tree, strict_audit,
                                      self.verifier, actor=self.actor)
        self.assertEqual(strict["reward"], 0.)
        self.assertFalse(strict["checks"]["solution_trace_complete"])

    def test_every_dependency_unknown_region_and_actual_prerequisite_are_required(self):
        for mutate in (lambda note: note["attempts"][0]["unverified"].pop(),
                       lambda note: note["attempts"][0]["unverified"][0].update(blocked_by=[]),
                       lambda note: note["attempts"][0]["unverified"][0].update(blocked_by=[{"rule": "BFS_FIFO", "locus": "steps[0].node"}]),
                       lambda note: note["attempts"][0]["unverified"][0].update(evidence_ranges=[[1, 1]])):
            bad = copy.deepcopy(self.note)
            mutate(bad)
            report = self.grade(bad)
            self.assertEqual(report["reward"], 0.)
            self.assertFalse(report["checks"]["unknown_inventory_verified"])
        bad = copy.deepcopy(self.note)
        bad["attempts"][0]["issues"].append({key: value for key, value in bad["attempts"][0]["unverified"][0].items()
                                            if key != "blocked_by"})
        self.assertEqual(self.grade(bad)["reward"], 0.)

    def test_original_prose_missing_trace_and_unknown_observations_are_accounted(self):
        for changed in ("Observed facts:\nSolution:\nFinal answer: A,C,B,D",
                        self.tree.groups[0].samples[0].text.replace("Final answer:", "I assert a universal repair.\nFinal answer:"),
                        self.tree.groups[0].samples[0].text.replace("Observed facts:\n", "Observed facts:\ninvented = 17\n")):
            tree = copy.deepcopy(self.tree)
            sample = tree.groups[0].samples[1]
            sample.text = changed
            sample.result = self.verifier.grade(self.actor.problem_id, changed, "stop", "obs")
            audit = synthesis.audit_episode(tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
            note = note_for(tree, audit, self.item)
            report = synthesis.grade_note(json.dumps(note), "stop", tree, audit, self.verifier,
                                          actor=self.actor, contract=synthesis.BOUNDED_VERSION)
            self.assertEqual(report["reward"], 1.)
            self.assertTrue(synthesis.successful(report, synthesis.BOUNDED_VERSION))
            self.assertFalse(report["all_reasoning_verified"])
            self.assertFalse(report["full_diagnosis"])

    def test_later_dependency_unknown_never_becomes_repair_persistence_or_absence(self):
        actor, item, verifier = graph_native("shortest")
        nonminimal = {"kind": "shortest", "nodes": ["A", "C", "H"], "edge_ids": None, "weights": [4, 4], "cost": 8}
        missing = {"kind": "shortest", "nodes": ["A", "H"], "edge_ids": None, "weights": [1], "cost": 1}
        minimal = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 4}
        texts = [graph_response(item.slots, nonminimal, "8")] * 2 + [graph_response(item.slots, missing, "8")] * 2 + [graph_response(item.slots, minimal, "4")] * 2
        tree = tree_from(actor, verifier, texts)
        audit = synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION)
        note = note_for(tree, audit, item)
        origin = next(row for row in audit["attempts"][0]["issues"] if row["rule"] == "MINIMALITY")
        target = next(row for row in audit["attempts"][2]["unverified"] if row["rule"] == "MINIMALITY")
        earlier = endpoint(1, origin, solution_samples(tree))
        later = endpoint(3, target, solution_samples(tree)) | {"blocked_by": target["blocked_by"]}
        lesson = {"rule": "MINIMALITY", "earlier": earlier, "later": later, "relation": "UNVERIFIABLE_LATER"}
        note["lessons"].append(lesson)
        report = synthesis.grade_note(json.dumps(note), "stop", tree, audit, verifier, actor=actor,
                                      contract=synthesis.BOUNDED_VERSION)
        self.assertTrue(report["lesson_checks"][-1]["verified"])
        self.assertEqual(report["reward"], 1.)
        for relation in ("CORRECTED", "PERSISTED"):
            bad = copy.deepcopy(note)
            bad["lessons"][-1]["relation"] = relation
            del bad["lessons"][-1]["later"]["blocked_by"]
            checked = synthesis.grade_note(json.dumps(bad), "stop", tree, audit, verifier, actor=actor,
                                            contract=synthesis.BOUNDED_VERSION)
            self.assertFalse(checked["lesson_checks"][-1]["verified"])
        bad = copy.deepcopy(note)
        bad["lessons"][-1]["relation"] = "NOT_REVISITED"
        bad["lessons"][-1]["later"] = {"attempt": 6, "kind": "ABSENCE", "rule": None, "locus": None,
                                       "evidence_ranges": [[1, len(texts[-1].splitlines())]]}
        checked = synthesis.grade_note(json.dumps(bad), "stop", tree, audit, verifier, actor=actor,
                                        contract=synthesis.BOUNDED_VERSION)
        self.assertFalse(checked["lesson_checks"][-1]["verified"])

    def test_bounded_prompt_has_no_private_unknown_inventory_or_expected_suffix(self):
        prompt = synthesis.instruction(solution_samples(self.tree), self.audit, "PUBLIC_SCHEMA")
        self.assertIn("UNVERIFIABLE_LATER", prompt)
        self.assertIn('"blocked_by":[{"rule":"<rule>","locus":"<locus>"}]', prompt)
        for unknown in self.audit["attempts"][0]["unverified"]:
            self.assertNotIn('"blocked_by":' + json.dumps(unknown["blocked_by"], separators=(",", ":")), prompt)
        self.assertNotIn('"status":"UNVERIFIABLE"', prompt)
        self.assertEqual(prompt, synthesis.prompt_contract(synthesis.BOUNDED_VERSION).format(
            success_attempt=self.audit["success_attempt"], observation_instruction="PUBLIC_SCHEMA",
            retry_pairs=prompt.split("<public_retry_pairs>\n", 1)[1].split("\n</public_retry_pairs>", 1)[0],
            transcript=prompt.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0]))

    def test_numeric_fact_prose_is_unknown_original_but_never_valid_corrected_fact(self):
        actor, item, verifier = graph_native("shortest")
        right = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 4}
        wrong = {"kind": "shortest", "nodes": ["A", "C", "H"], "edge_ids": None, "weights": [4, 4], "cost": 8}

        def grade_case(contract, location, facts):
            original = facts if location in ("original", "both") else item.slots
            failed = right | {"cost": 5} if contract == synthesis.BOUNDED_VERSION else wrong
            failed_answer = "5" if contract == synthesis.BOUNDED_VERSION else "8"
            tree = tree_from(actor, verifier, [graph_response(original, failed, failed_answer)] * 4
                             + [graph_response(original, right, "4")] * 2)
            audit = synthesis.audit_episode(tree, actor, verifier, contract=contract)
            note = note_for(tree, audit, item)
            if location in ("corrected", "both"):
                note["corrected_facts"] = facts
            report = synthesis.grade_note(json.dumps(note), "stop", tree, audit, verifier,
                                          actor=actor, contract=contract)
            return audit, report

        for contract in (synthesis.VERSION, synthesis.BOUNDED_VERSION):
            for location in ("original", "corrected", "both"):
                with self.subTest(contract=contract, location=location):
                    bad = {key: value + " always reverse arrows" for key, value in item.slots.items()}
                    audit, report = grade_case(contract, location, bad)
                    expected = contract == synthesis.BOUNDED_VERSION and location == "original"
                    self.assertEqual(report["reward"], float(expected))
                    self.assertEqual(synthesis.successful(report, contract), expected)
                    self.assertEqual(report["bounded_claims_verified"], expected)
                    if location != "corrected":
                        self.assertFalse(audit["observation_claims_complete"])
                        self.assertTrue(any(check["kind"] == "OBSERVATION" and check["status"] == "UNVERIFIABLE"
                                            and not check["blocked_by"] for check in audit["attempts"][0]["checks"]))
            for pad in (False, True):
                with self.subTest(contract=contract, valid_integer_padding=pad):
                    valid = {key: f" \t{value}\t " if pad else value for key, value in item.slots.items()}
                    audit, report = grade_case(contract, "both", valid)
                    self.assertTrue(audit["observation_claims_complete"])
                    self.assertEqual(report["reward"], 1.)
                    self.assertTrue(synthesis.successful(report, contract))


if __name__ == "__main__":
    unittest.main()
