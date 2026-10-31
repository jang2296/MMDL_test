"""Real verifier with authored transcripts; accounting is not model success or semantic repair."""
import copy
import hashlib
import json
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.notebook_rl.notebook_learning import _first_attempt_lines
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures


class AccountedUnknownTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        self.wrong = fixtures.response(self.item.slots)
        self.right = fixtures.response(self.item.slots, wrong=False)

    def prepare(self, texts, *, contract=synthesis.BOUNDED_VERSION):
        tree = fixtures.tree_from(self.actor, self.verifier, texts)
        audit = synthesis.audit_episode(tree, self.actor, self.verifier, contract=contract)
        return tree, audit, fixtures.note_for(tree, audit, self.item)

    def grade(self, tree, audit, note):
        return synthesis.grade_note(json.dumps(note), "stop", tree, audit, self.verifier,
                                    actor=self.actor, contract=audit["version"])

    def test_root_unknowns_bind_every_original_line_without_semantic_certification(self):
        cases = (("Observed facts:\nSolution:\nTrace: {\nFinal answer: reject", "TRACE_FORMAT"),
                 (self.wrong.replace('"states":["q0","q1","q0","q0","q1"]', '"states":7'), "TRACE_SCHEMA"),
                 (self.wrong.replace("Final answer:", "Always reverse every arrow.\nFinal answer:"), "RESPONSE_SCOPE"))
        for text, rule in cases:
            with self.subTest(rule=rule):
                tree, audit, note = self.prepare([text] * 4 + [self.right] * 2)
                before = to_json(tree), copy.deepcopy(audit)
                unknown = next(row for row in audit["attempts"][0]["unverified"] if row["rule"] == rule)
                lines = _first_attempt_lines(text)
                self.assertEqual(unknown["blocked_by"], [])
                self.assertEqual(unknown["refs"], [{"span": [line["start"], line["end"]],
                    "sha256": hashlib.sha256(line["text"].encode()).hexdigest()} for line in lines])
                self.assertEqual(audit["attempts"][0]["text_sha256"], hashlib.sha256(text.encode()).hexdigest())
                checked = self.grade(tree, audit, note)
                self.assertEqual(checked["reward"], 1.)
                self.assertTrue(checked["checks"]["unknown_inventory_verified"])
                self.assertFalse(checked["solution_trace_complete"])
                self.assertFalse(checked["all_reasoning_verified"])
                self.assertFalse(checked["full_diagnosis"])
                self.assertFalse(checked["lesson_semantics_verified"])
                self.assertEqual((to_json(tree), audit), before)
                unknown_claim = next(row for row in note["attempts"][0]["unverified"] if row["rule"] == rule)
                unknown_claim["evidence_ranges"] = [[len(lines), len(lines)]]
                self.assertEqual(self.grade(tree, audit, note)["reward"], 0.)

    def test_empty_original_response_has_no_invented_line_one(self):
        tree, audit, note = self.prepare([""] * 4 + [self.right] * 2)
        self.assertEqual(audit["attempts"][0]["checks"][0]["refs"], [])
        self.assertEqual(audit["attempts"][0]["text_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(note["attempts"][0]["evidence_ranges"], [])
        self.assertEqual(note["attempts"][0]["unverified"][0]["evidence_ranges"], [])
        self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)
        for field in ("attempt", "unknown"):
            bad = copy.deepcopy(note)
            claim = bad["attempts"][0]
            (claim if field == "attempt" else claim["unverified"][0])["evidence_ranges"] = [[1, 1]]
            self.assertEqual(self.grade(tree, audit, bad)["reward"], 0.)

    def test_nonempty_unknown_cannot_use_empty_evidence(self):
        tree, audit, note = self.prepare(["malformed"] * 4 + [self.right] * 2)
        note["attempts"][0]["unverified"][0]["evidence_ranges"] = []
        self.assertEqual(self.grade(tree, audit, note)["reward"], 0.)

    def test_unknown_inventory_rejects_missing_duplicate_extra_or_forged_dependencies(self):
        tree, audit, note = self.prepare(["malformed"] * 4 + [self.right] * 2)
        for mutation in ("missing", "duplicate", "extra", "dependency", "as_issue"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(note)
                unknown = bad["attempts"][0]["unverified"]
                if mutation == "missing":
                    unknown.clear()
                elif mutation == "duplicate":
                    unknown.append(copy.deepcopy(unknown[0]))
                elif mutation == "extra":
                    unknown.append(unknown[0] | {"rule": "TRACE_SCHEMA", "locus": "trace"})
                elif mutation == "dependency":
                    unknown[0]["blocked_by"] = [{"rule": "DFA_TRANSITION", "locus": "states[0:2]"}]
                else:
                    bad["attempts"][0]["issues"].append({key: value for key, value in unknown[0].items()
                                                          if key != "blocked_by"})
                self.assertEqual(self.grade(tree, audit, bad)["reward"], 0.)

    def test_all_original_unknown_observation_occurrences_are_required(self):
        text = self.wrong.replace("Solution:", "invented = 17\ninvented = 18\nSolution:")
        tree, audit, note = self.prepare([text] * 4 + [self.right] * 2)
        unknown = next(row for row in audit["attempts"][0]["unverified"] if row["kind"] == "OBSERVATION")
        self.assertEqual(unknown["rule"], "OBSERVATION_MATCH")
        self.assertEqual(len(unknown["refs"]), 2)
        self.assertFalse(audit["observation_claims_complete"])
        checked = self.grade(tree, audit, note)
        self.assertEqual(checked["reward"], 1.)
        self.assertTrue(checked["checks"]["observation_claims_accounted"])
        claim = next(row for row in note["attempts"][0]["unverified"] if row["kind"] == "OBSERVATION")
        first_line = next(line["line_id"] for line in _first_attempt_lines(text) if line["text"] == "invented = 17\n")
        claim["evidence_ranges"] = [[first_line, first_line]]
        self.assertEqual(self.grade(tree, audit, note)["reward"], 0.)

    def test_known_observation_issue_survives_malformed_trace_and_cannot_be_omitted(self):
        wrong_facts = self.item.slots | {"trans[q1,1]": "q0"}
        text = fixtures.response(wrong_facts).replace('Trace: {"kind"', 'Trace: {broken"kind"')
        tree, audit, note = self.prepare([text] * 4 + [self.right] * 2)
        self.assertEqual([row["kind"] for row in audit["attempts"][0]["issues"]], ["OBSERVATION"])
        self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)
        note["attempts"][0]["issues"] = []
        self.assertEqual(self.grade(tree, audit, note)["reward"], 0.)

    def numeric_response(self, values, answer):
        trace = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 4}
        text = fixtures.graph_response(self.item.slots, trace, answer)
        return text.replace("edge[A,B] = 2", "\n".join(f"edge[A,B] = {value}" for value in values))

    def test_quoted_and_unknown_relations_keep_known_and_unknown_evidence(self):
        cases = ((['"q1"'], "COMPLIANT", False), (["'q0'"], "VIOLATED", False),
                 (["unreadable?"], None, True), (["q0 or q1"], None, True),
                 (['"q1'], None, True), (['"\"q1\""'], None, True),
                 (['"q1"', "unreadable?"], "COMPLIANT", True),
                 (["unreadable?", '"q0"'], "VIOLATED", True))
        for values, known_status, unknown_present in cases:
            with self.subTest(values=values):
                text = self.wrong.replace("trans[q0,0] = q1", "\n".join(f"trans[q0,0] = {value}" for value in values))
                tree, audit, note = self.prepare([text] * 4 + [self.right] * 2)
                before = to_json(tree)
                checks = [row for row in audit["attempts"][0]["checks"]
                          if row["kind"] == "OBSERVATION" and row["locus"] == "trans[q0,0]"]
                known = [row for row in checks if row["status"] != "UNVERIFIABLE"]
                unknown = [row for row in checks if row["status"] == "UNVERIFIABLE"]
                self.assertEqual([row["status"] for row in known], [known_status] if known_status else [])
                self.assertEqual(len(unknown), int(unknown_present))
                for check in checks:
                    for ref in check["refs"]:
                        start, end = ref["span"]
                        self.assertIn(text[start:end], [f"trans[q0,0] = {value}\n" for value in values])
                        self.assertEqual(ref["sha256"], hashlib.sha256(text[start:end].encode()).hexdigest())
                if unknown:
                    self.assertEqual(len(unknown[0]["refs"]), len(values))
                observation_issues = [row for row in audit["attempts"][0]["issues"] if row["kind"] == "OBSERVATION"]
                self.assertEqual(len(observation_issues), int(known_status == "VIOLATED"))
                if known_status != "VIOLATED":
                    self.assertFalse(any(row["earlier"]["kind"] == "OBSERVATION" for row in note["lessons"]))
                self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)
                self.assertEqual(to_json(tree), before)

    def test_mixed_later_relation_never_proves_correction_or_persistence(self):
        wrong = self.wrong.replace("trans[q0,0] = q1", 'trans[q0,0] = "q0"')
        for later_value in ('"q1"', '"q0"'):
            with self.subTest(later_value=later_value):
                later = self.right.replace("trans[q0,0] = q1",
                                           f"trans[q0,0] = {later_value}\ntrans[q0,0] = unreadable?")
                tree, audit, note = self.prepare([wrong] * 4 + [later] * 2)
                lesson = next(row for row in note["lessons"] if row["earlier"]["attempt"] == 4
                              and row["earlier"]["kind"] == "OBSERVATION")
                self.assertEqual(lesson["relation"], "UNVERIFIABLE_LATER")
                self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)
                for relation in ("CORRECTED", "PERSISTED", "OTHER_BRANCH_COMPARISON"):
                    bad = copy.deepcopy(note)
                    changed = next(row for row in bad["lessons"] if row["earlier"] == lesson["earlier"])
                    changed["relation"] = relation
                    del changed["later"]["blocked_by"]
                    self.assertEqual(self.grade(tree, audit, bad)["reward"], 0.)

    def test_duplicate_numeric_occurrences_preserve_known_and_unknown_evidence_separately(self):
        self.actor, self.item, self.verifier = fixtures.graph_native("shortest")
        cases = ((["4", "2 units"], "VIOLATED", True), (["2 units", "4"], "VIOLATED", True),
                 (["2", "2 units"], "COMPLIANT", True), (["2 units", "2"], "COMPLIANT", True),
                 (["2", "4", "2 units"], "VIOLATED", True),
                 (["2 units", "4", "2"], "VIOLATED", True),
                 (["2 units", "4 units"], None, True),
                 (["4", "4"], "VIOLATED", False), (["2", "2"], "COMPLIANT", False))
        for values, known_status, unknown_present in cases:
            with self.subTest(values=values):
                text = self.numeric_response(values, "8")
                tree, audit, note = self.prepare([text] * 4 + [self.numeric_response(["2"], "4")] * 2)
                checks = [row for row in audit["attempts"][0]["checks"]
                          if row["kind"] == "OBSERVATION" and row["locus"] == "edge[a,b]"]
                known = [row for row in checks if row["status"] != "UNVERIFIABLE"]
                unknown = [row for row in checks if row["status"] == "UNVERIFIABLE"]
                self.assertEqual([row["status"] for row in known], [known_status] if known_status else [])
                self.assertEqual(len(unknown), int(unknown_present))
                if known:
                    self.assertEqual(len(known[0]["refs"]), sum(value.isascii() and value.isdigit() for value in values))
                if unknown:
                    self.assertEqual(len(unknown[0]["refs"]), len(values))
                    bad = copy.deepcopy(note)
                    bad["attempts"][0]["unverified"] = [row for row in bad["attempts"][0]["unverified"]
                                                        if row["kind"] != "OBSERVATION"]
                    self.assertEqual(self.grade(tree, audit, bad)["reward"], 0.)
                    bad = copy.deepcopy(note)
                    claim = next(row for row in bad["attempts"][0]["unverified"] if row["kind"] == "OBSERVATION")
                    first = next(row["line_id"] for row in _first_attempt_lines(text)
                                 if row["text"].startswith("edge[A,B] = "))
                    claim["evidence_ranges"] = [[first, first]]
                    self.assertEqual(self.grade(tree, audit, bad)["reward"], 0.)
                self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)
                if unknown:
                    strict_tree, strict_audit, strict_note = self.prepare(texts=[text] * 4
                        + [self.numeric_response(["2"], "4")] * 2, contract=synthesis.VERSION)
                    self.assertEqual(self.grade(strict_tree, strict_audit, strict_note)["reward"], 0.)
                if known_status == "VIOLATED":
                    bad = copy.deepcopy(note)
                    for attempt in bad["attempts"]:
                        attempt["issues"] = []
                    bad["lessons"] = []
                    checked = self.grade(tree, audit, bad)
                    self.assertEqual(checked["reward"], 0.)
                    self.assertFalse(checked["checks"]["all_attempts_verified"])
                    self.assertFalse(checked["checks"]["all_issues_synthesized"])

    def test_mixed_later_slot_never_proves_correction_or_persistence(self):
        self.actor, self.item, self.verifier = fixtures.graph_native("shortest")
        for later_value in ("2", "4"):
            with self.subTest(later_value=later_value):
                texts = [self.numeric_response(["4"], "8")] * 4 + [self.numeric_response([later_value, "2 units"], "4")] * 2
                tree, audit, note = self.prepare(texts)
                lesson = next(row for row in note["lessons"] if row["earlier"]["attempt"] == 4)
                self.assertEqual(lesson["relation"], "UNVERIFIABLE_LATER")
                self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)
                for relation in ("CORRECTED", "PERSISTED", "OTHER_BRANCH_COMPARISON"):
                    bad = copy.deepcopy(note)
                    changed = next(row for row in bad["lessons"] if row["earlier"]["attempt"] == 4)
                    changed["relation"] = relation
                    del changed["later"]["blocked_by"]
                    self.assertEqual(self.grade(tree, audit, bad)["reward"], 0.)

    def test_same_slot_unknown_later_is_undecidable_not_correction_or_absence(self):
        self.actor, self.item, self.verifier = fixtures.graph_native("shortest")
        trace = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 4}
        wrong = fixtures.graph_response(self.item.slots | {"edge[A,B]": "4"}, trace, "8")
        later = fixtures.graph_response(self.item.slots | {"edge[A,B]": "2 units"}, trace, "4")
        tree, audit, note = self.prepare([wrong] * 4 + [later] * 2)
        lesson = next(row for row in note["lessons"] if row["earlier"]["attempt"] == 4
                      and row["earlier"]["kind"] == "OBSERVATION")
        self.assertEqual(lesson["relation"], "UNVERIFIABLE_LATER")
        self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)
        for relation in ("CORRECTED", "PERSISTED", "SCOPE_UNVERIFIABLE_LATER", "NOT_REVISITED"):
            bad = copy.deepcopy(note)
            changed = next(row for row in bad["lessons"] if row["earlier"] == lesson["earlier"])
            changed["relation"] = relation
            if relation == "NOT_REVISITED":
                changed["later"] = {"attempt": 6, "kind": "ABSENCE", "rule": None, "locus": None,
                                    "evidence_ranges": [[1, len(later.splitlines())]]}
            elif relation in {"CORRECTED", "PERSISTED"}:
                del changed["later"]["blocked_by"]
            self.assertEqual(self.grade(tree, audit, bad)["reward"], 0.)

    def test_corrected_facts_and_final_answer_remain_strict(self):
        tree, audit, note = self.prepare(["malformed"] * 4 + [self.right] * 2)
        for change in ("fact_missing", "fact_wrong", "answer_wrong"):
            bad = copy.deepcopy(note)
            if change == "fact_missing":
                bad["corrected_facts"].pop("start")
            elif change == "fact_wrong":
                bad["corrected_facts"]["start"] = "q1"
            else:
                bad["corrected_answer"] = "reject"
            self.assertEqual(self.grade(tree, audit, bad)["reward"], 0.)

    def test_unknown_native_task_contract_is_a_hard_block(self):
        self.actor = replace(self.actor, question=self.actor.question + " unsupported")
        self.verifier.actors[self.actor.problem_id] = self.actor
        tree, audit, note = self.prepare([self.wrong] * 4 + [self.right] * 2)
        self.assertEqual(audit["attempts"][0]["unverified"][0]["rule"], "NATIVE_CONTRACT")
        self.assertFalse(audit["unknown_regions_accountable"])
        self.assertEqual(self.grade(tree, audit, note)["reward"], 0.)

    def test_forged_fresh_audit_or_absent_correct_retry_never_admits(self):
        tree, audit, note = self.prepare(["malformed"] * 4 + [self.right] * 2)
        forged = copy.deepcopy(audit)
        forged["attempts"][0]["text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "freshly audited"):
            self.grade(tree, forged, note)
        tree = fixtures.tree_from(self.actor, self.verifier, ["malformed"] * 6)
        audit = synthesis.audit_episode(tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        with self.assertRaisesRegex(ValueError, "freshly audited recovery"):
            self.grade(tree, audit, note)

    def test_strict_contract_retains_unknown_veto_and_exact_gate(self):
        for text in ("", "malformed", self.wrong.replace("Solution:", "invented = 17\nSolution:")):
            tree, audit, note = self.prepare([text] * 4 + [self.right] * 2, contract=synthesis.VERSION)
            checked = self.grade(tree, audit, note)
            self.assertEqual(checked["reward"], 0.)
            self.assertFalse(synthesis.successful(checked, synthesis.VERSION))
            self.assertNotIn("unknown_regions_accountable", audit)
        self.assertEqual(synthesis.BOUNDED_VERSION, "local-whole-history-bounded-synthesis-v4")

    def test_dfa_acceptance_requires_the_exact_proven_dependency_and_original_evidence(self):
        for states, dependency in ((["q0"], {"rule": "DFA_COMPLETE_WORD", "locus": "states"}),
                                   (["q0", "q1", "q2"], {"rule": "DFA_COMPLETE_WORD", "locus": "states"}),
                                   (["q0", "q1", "q2", "q2", "q3", "q3"],
                                    {"rule": "DFA_COMPLETE_WORD", "locus": "states"}),
                                   (["q0", "q1", "q2", "q2", "q9"],
                                    {"rule": "DFA_TRANSITION", "locus": "states[3:5]"})):
            with self.subTest(states=states):
                trace = {"kind": "dfa", "states": states, "decision": "reject"}
                text = fixtures.graph_response(self.item.slots, trace, "reject")
                tree, audit, note = self.prepare([text] * 4 + [self.right] * 2)
                acceptance = next(row for row in audit["attempts"][0]["unverified"] if row["rule"] == "DFA_ACCEPTANCE")
                self.assertEqual(acceptance["blocked_by"], [dependency])
                self.assertTrue(any(row["rule"] == dependency["rule"] and row["locus"] == dependency["locus"]
                                    for row in audit["attempts"][0]["issues"]))
                self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)
                for mutation in ("missing", "other_locus", "unrelated", "evidence", "omit_failure"):
                    bad = copy.deepcopy(note)
                    claim = next(row for row in bad["attempts"][0]["unverified"] if row["rule"] == "DFA_ACCEPTANCE")
                    if mutation == "missing":
                        claim["blocked_by"] = []
                    elif mutation == "other_locus":
                        claim["blocked_by"] = [dependency | {"locus": "states[0:2]"}]
                    elif mutation == "unrelated":
                        claim["blocked_by"] = [{"rule": "DFA_START", "locus": "states[0]"}]
                    elif mutation == "evidence":
                        claim["evidence_ranges"] = [[1, 1]]
                    else:
                        bad["attempts"][0]["issues"] = [row for row in bad["attempts"][0]["issues"]
                                                          if row["rule"] != dependency["rule"]]
                    self.assertEqual(self.grade(tree, audit, bad)["reward"], 0.)
                strict_tree, strict_audit, strict_note = self.prepare([text] * 4 + [self.right] * 2,
                                                                     contract=synthesis.VERSION)
                self.assertEqual(self.grade(strict_tree, strict_audit, strict_note)["reward"], 0.)

    def test_scope_unknown_later_is_not_absence_repair_or_same_operation(self):
        self.actor, self.item, self.verifier = fixtures.graph_native("shortest")
        wrong = {"kind": "shortest", "nodes": ["A", "D", "H"], "edge_ids": None, "weights": [1, 1], "cost": 2}
        right = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 4}
        correct = fixtures.graph_response(self.item.slots, right, "4")
        texts = [fixtures.graph_response(self.item.slots, wrong, "2")] * 4 + ["Final answer: 4", correct]
        tree, audit, note = self.prepare(texts)
        self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)
        lesson = next(row for row in note["lessons"] if row["earlier"]["attempt"] == 4)
        self.assertEqual(lesson["relation"], "SCOPE_UNVERIFIABLE_LATER")
        self.assertEqual(lesson["later"]["rule"], "TRACE_FORMAT")
        for relation in ("CORRECTED", "PERSISTED", "OTHER_BRANCH_COMPARISON", "UNVERIFIABLE_LATER", "NOT_REVISITED"):
            bad = copy.deepcopy(note)
            changed = next(row for row in bad["lessons"] if row["earlier"]["attempt"] == 4)
            changed["relation"] = relation
            if relation == "NOT_REVISITED":
                changed["later"] = {"attempt": 6, "kind": "ABSENCE", "rule": None, "locus": None,
                                    "evidence_ranges": [[1, len(correct.splitlines())]]}
            elif relation != "UNVERIFIABLE_LATER":
                del changed["later"]["blocked_by"]
            checked = self.grade(tree, audit, bad)
            self.assertEqual(checked["reward"], 0.)
        bad = copy.deepcopy(note)
        changed = next(row for row in bad["lessons"] if row["earlier"]["attempt"] == 4)
        changed["later"] = fixtures.endpoint(6, audit["attempts"][5]["checks"][0], solution_samples(tree)) | {"blocked_by": []}
        self.assertEqual(self.grade(tree, audit, bad)["reward"], 0.)

    def test_unrelated_unknown_observation_does_not_claim_revisit_or_block_absence(self):
        self.actor, self.item, self.verifier = fixtures.graph_native("shortest")
        right = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 4}
        facts = self.item.slots | {"edge[D,H]": "1"}
        earlier = fixtures.graph_response(facts, right, "8")
        later = fixtures.graph_response(self.item.slots | {"invented": "17"}, right, "4")
        tree, audit, note = self.prepare([earlier] * 4 + [later] * 2)
        issue = next(row for row in audit["attempts"][3]["issues"] if row["locus"] == "edge[d,h]")
        lesson = next(row for row in note["lessons"] if row["earlier"]["attempt"] == 4
                      and row["earlier"]["locus"] == issue["locus"])
        self.assertEqual(lesson["relation"], "NOT_REVISITED")
        self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)

    def test_empty_later_scope_is_bound_with_no_fabricated_evidence(self):
        self.actor, self.item, self.verifier = fixtures.graph_native("shortest")
        wrong = {"kind": "shortest", "nodes": ["A", "D", "H"], "edge_ids": None, "weights": [1, 1], "cost": 2}
        right = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 4}
        tree, audit, note = self.prepare([fixtures.graph_response(self.item.slots, wrong, "2")] * 4
                                        + ["", fixtures.graph_response(self.item.slots, right, "4")])
        lesson = next(row for row in note["lessons"] if row["earlier"]["attempt"] == 4)
        self.assertEqual(lesson["relation"], "SCOPE_UNVERIFIABLE_LATER")
        self.assertEqual(lesson["later"]["attempt"], 5)
        self.assertEqual(lesson["later"]["evidence_ranges"], [])
        self.assertEqual(self.grade(tree, audit, note)["reward"], 1.)
        lesson["later"]["evidence_ranges"] = [[1, 1]]
        self.assertEqual(self.grade(tree, audit, note)["reward"], 0.)


if __name__ == "__main__":
    unittest.main()
