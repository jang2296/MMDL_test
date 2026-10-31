"""Authored native-world counterexamples, not actual model or learning success."""
import copy
import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

from fine_tuning.notebook_rl.attribution import attempt_cause, problem_route
from fine_tuning.notebook_rl.episode_notebook import (
    FEEDBACK_POLICY, REINSPECTION_INSTRUCTION, STRICT_ROUTE_RULE, audit_episode,
    grade_note, notebook_feedback, strict_routing, validate_update_binding,
)
from fine_tuning.notebook_rl.observations import slot_id
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import DiagTarget
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.notebook_rl.verifier_service import (
    MAX_SLOTS, VerifierService, VerifierServiceError, _Rejected, _result,
)
from fine_tuning.tests import test_notebook_rl_episode_notebook as fixtures
from fine_tuning.tests.test_notebook_rl_observations import native_graph, native_tree


class NativeRelationIntegrationTests(unittest.TestCase):
    false_key, false_value = "edge[H,D]", "1"
    registered_key, unknown_key, other_false_key = "edge[A,B]", "edge[X,D]", "edge[H,A]"

    def native_item(self):
        return native_graph()

    def setUp(self):
        self.fixture = fixtures.EpisodeNotebookTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        actor, _, lineage, *_ = fixtures.records()
        self.item = replace(self.native_item(), answer_type="int", gold="7")
        self.fixture.actor = replace(actor, problem_id=self.item.problem_id, source="gen-graph-v1", family="graph")
        self.root = self.fixture.root / "native"
        write_split(self.root, [self.fixture.actor], [self.item], [replace(lineage,
                    problem_id=self.item.problem_id, source="gen-graph-v1")])
        self.fixture.verifier = Verifier(self.root)
        self.fixture.prompts.update(notebook_route_rule=STRICT_ROUTE_RULE, notebook_feedback_policy=FEEDBACK_POLICY)
        self.facts = "\n".join(f"{k} = {v}" for k, v in self.item.slots.items())
        self.false_slot = slot_id(self.false_key)
        self.false_claim = f"{self.false_key} = {self.false_value}"

    def text(self, extra="", answer="8"):
        return f"Observed facts:\n{self.facts}\n{extra}\nSolution:\nFinal answer: {answer}"

    def tree(self, extra=None, *, correct_sibling=False):
        extra = self.false_claim if extra is None else extra
        first = [self.text(extra, "7" if correct_sibling else "8"), self.text(extra)]
        tree, engine = self.fixture.tree([first, [self.text(answer="7")] * 2])
        audit = audit_episode(tree, self.fixture.actor, self.fixture.verifier, routing_rule=STRICT_ROUTE_RULE)
        note = fixtures.authored_note(tree, audit)
        note["corrected_facts"] = dict(self.item.slots)
        return tree, engine, audit, note

    def test_false_edges_reach_fresh_service_cause_target_and_route_without_private_values(self):
        with VerifierService(self.root, forbidden_roots=[]) as verifier:
            result = verifier.grade(self.item.problem_id, self.text(self.false_claim), "stop", "obs")
            self.assertEqual(result.slots.wrong_required, ())
            self.assertEqual(result.slots.false_relations, (self.false_slot,))
            self.assertEqual(verifier.cause(self.item.problem_id, result), "VISUAL_RELATION")
            target = verifier.diagnosis_target(self.item.problem_id, result)
            self.assertEqual(target, DiagTarget("VISUAL", (self.false_slot,)))
            route, evidence = verifier.route(self.item.problem_id, [result], "verified-any-visual-v1")
            self.assertEqual(route, "VISION_ON")
            self.assertEqual(evidence["wrong_required"], [])
            self.assertEqual(evidence["false_relations"], [{"result_index": 0, "wrong_slots": [self.false_slot]}])
            self.assertEqual(verifier.grade_diagnosis(self.item.problem_id,
                f"Cause: VISUAL_RELATION\nWrong item: {self.false_slot}\nLesson: Check the relation.", target), 1.25)
            for key in ("gold", "graph_world", "oracle_fact_lines", self.registered_key):
                self.assertNotIn(key, json.dumps(evidence))
            forged = replace(result, slots=replace(result.slots, false_relations=(slot_id(self.other_false_key),)))
            with self.assertRaisesRegex(VerifierServiceError, "UNISSUED_RESULT"):
                verifier.cause(self.item.problem_id, forged)
            with self.assertRaisesRegex(VerifierServiceError, "UNISSUED_TARGET"):
                verifier.grade_diagnosis(self.item.problem_id, "anything",
                                         DiagTarget("VISUAL", (slot_id(self.other_false_key),)))
            graph = Path(__file__).parents[1] / "notebook_rl/generators/graph.py"
            self.assertEqual(verifier.metadata["code_sha256"]["generators/graph.py"],
                             hashlib.sha256(graph.read_bytes()).hexdigest())

    def test_unregistered_false_proof_cannot_be_forged_from_registered_or_unknown_keys(self):
        result = self.fixture.verifier.grade(self.item.problem_id, self.text(), "stop", "obs")
        registered = slot_id(self.registered_key)
        for key in (registered, slot_id(self.unknown_key), self.false_slot + "[extra]", "invented"):
            forged = replace(result, slots=replace(result.slots, false_relations=(key,)))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "native-world proof"):
                problem_route([forged], self.item, "verified-any-visual-v1")
            if key != registered:
                with self.assertRaises(ValueError):
                    self.fixture.verifier.grade_diagnosis(self.item.problem_id, "anything", DiagTarget("VISUAL", (key,)))
        # Registered observations can legitimately be diagnosis targets for wrong values.
        self.assertEqual(self.fixture.verifier.grade_diagnosis(self.item.problem_id,
            f"Cause: VISUAL_MISREAD\nWrong item: {registered}\nLesson: Recheck its value.",
            DiagTarget("VISUAL", (self.registered_key,))), 1.25)

    def test_known_false_relation_has_evidence_without_coverage_but_unknown_is_not_visual(self):
        verifier = self.fixture.verifier
        sparse = verifier.grade(self.item.problem_id,
            f"Observed facts:\n{self.false_claim}\nSolution:\nFinal answer: 8", "stop", "obs")
        self.assertEqual(sparse.slots.parsed, 0)
        self.assertEqual(attempt_cause(sparse, self.item), "VISUAL_RELATION")
        for extra in (f"{self.unknown_key} = {self.false_value}",
                      f"{self.false_key} = unreadable", f"{self.false_key} = 0"):
            tree, _, audit, _ = self.tree(extra)
            self.assertEqual(tree.groups[0].samples[0].result.slots.required_correct, len(self.item.required_slots))
            self.assertEqual(audit["admission"], "UNRESOLVED")
            self.assertEqual(audit["attempts"][0]["cause"], "UNKNOWN")
            self.assertFalse(audit["all_required_observations_verified"])
        tree, _, audit, _ = self.tree("")
        self.assertEqual(audit["admission"], "VISION_OFF")
        self.assertTrue(strict_routing(tree)["all_required_observations_verified"])

    def test_all_history_false_relation_is_required_even_for_correct_or_truncated_answer(self):
        tree, _, audit, _ = self.tree(correct_sibling=True)
        self.assertEqual(audit["attempts"][0]["cause"], "ANSWER_CORRECT")
        self.assertEqual(audit["attempts"][0]["wrong_slots"], [self.false_slot])
        first = tree.groups[0].samples[0]
        first.finish_reason = "length"
        first.result = self.fixture.verifier.grade(tree.problem_id, first.text, "length", "obs")
        checked = audit_episode(tree, self.fixture.actor, self.fixture.verifier, routing_rule=STRICT_ROUTE_RULE)
        self.assertEqual(checked["attempts"][0]["cause"], "FINALIZATION")
        self.assertEqual(checked["admission"], "VISION_ON")
        self.assertFalse(checked["module_causality_certified"])

    def test_note_needs_false_edge_claim_and_its_original_evidence_line(self):
        tree, _, audit, note = self.tree()
        def grade(value):
            return grade_note(json.dumps(value), "stop", tree, audit, self.fixture.verifier)
        self.assertEqual(grade(note)["reward"], 1)
        missing = copy.deepcopy(note)
        missing["attempts"][0]["wrong_slots"] = []
        self.assertFalse(grade(missing)["attempt_checks"][0]["wrong_slots"])
        self.assertEqual(grade(missing)["reward"], 0)
        uncited = copy.deepcopy(note)
        last = len(tree.groups[0].samples[0].text.splitlines())
        uncited["attempts"][0]["line_ranges"] = [[last, last]]
        self.assertFalse(grade(uncited)["attempt_checks"][0]["visual_evidence"])
        self.assertEqual(grade(uncited)["reward"], 0)

    def test_corrected_facts_false_edge_rejected_and_only_submitted_key_in_feedback(self):
        tree, engine, audit, note = self.tree()
        note["corrected_facts"][self.false_key] = self.false_value
        bad = json.dumps(note)
        checked = grade_note(bad, "stop", tree, audit, self.fixture.verifier, feedback_policy=FEEDBACK_POLICY)
        self.assertTrue(checked["checks"]["answer_verified"])
        self.assertFalse(checked["checks"]["registered_facts_verified"])
        self.assertEqual(checked["reinspection_slots"], [self.false_slot])
        self.assertEqual(checked["reward"], 0)
        engine.replies.extend([[bad] * 4] * 3)
        self.fixture.append(tree, engine)
        validate_update_binding(tree, tokenizer=engine.tokenizer, feedback_policy=FEEDBACK_POLICY)
        row = tree.notebook["model_notes"][0]
        sample = next(g.samples[0] for g in tree.groups if g.kind == "DIAG")
        feedback = notebook_feedback(row, sample)
        public = json.loads(feedback[len(REINSPECTION_INSTRUCTION):])
        self.assertEqual(public["reinspection_slots"], [self.false_slot])
        self.assertNotIn(self.registered_key, feedback)
        self.assertNotIn("graph_world", feedback)
        changed = copy.deepcopy(tree)
        saved = changed.notebook["rewarded_notebook"]
        saved["episode_audit"]["attempts"][0]["wrong_slots"] = []
        saved["episode_audit_sha256"] = _hash(saved["episode_audit"])
        with self.assertRaisesRegex(ValueError, "wrong observations differ"):
            validate_update_binding(changed)

    def test_false_and_unverified_names_are_bounded_at_service_boundary(self):
        result = self.fixture.verifier.grade(self.item.problem_id, self.text(), "stop", "obs")
        for field in ("false_relations", "unverified_relations"):
            for names in (("x" * 257,), tuple(f"edge[x{i},y]" for i in range(MAX_SLOTS + 1))):
                value = replace(result, slots=replace(result.slots, **{field: names}))
                with self.subTest(field=field), self.assertRaises(_Rejected):
                    _result(value)


class NativeTreeRelationIntegrationTests(NativeRelationIntegrationTests):
    false_key, false_value = "child[F,R]", "E"
    registered_key, unknown_key, other_false_key = "child[F,L]", "child[X,R]", "child[A,L]"

    def native_item(self):
        return native_tree()


if __name__ == "__main__":
    unittest.main()
