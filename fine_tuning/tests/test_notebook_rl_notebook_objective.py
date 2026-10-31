"""Authored CPU atom fixtures only; no actual learned notebook success claim."""
import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as notebook
from fine_tuning.notebook_rl import notebook_objective as objective
from fine_tuning.notebook_rl.local_eval import digest
from fine_tuning.notebook_rl.observations import slot_id
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.notebook_rl.verifier_service import VerifierService
from fine_tuning.tests import test_notebook_rl_episode_notebook as fixtures
from fine_tuning.tests import test_notebook_rl_native_relations as native_fixtures

A, B = "cell[a][b]", "cell[c][d]"
WRONG = f"Observed facts:\n{A} = 8\n{B} = 5\nSolution:\nFinal answer: 8"
RIGHT = f"Observed facts:\n{A} = 7\n{B} = 5\nSolution:\nFinal answer: 7"


class NotebookObjectiveTests(unittest.TestCase):
    def test_fact_credit_retains_submitted_value_checks_not_reference_targets(self):
        note = copy.deepcopy(self.note)
        note["corrected_facts"][B] = "999"
        result = self.score(note, objective_version=objective.FACT_CREDIT_VERSION)
        self.assertEqual(result["fact_rewards"], {A: 1, B: 0})
        self.assertEqual(result["required_slot_count"], 2)
        self.assertEqual((result["numerator"], result["denominator"]), (6, 48))
        self.assertEqual(result["effective_reward"], .03125)
        self.assertEqual(result["strict_reward"], 0)
        invalid = self.score(text="not JSON", objective_version=objective.FACT_CREDIT_VERSION)
        self.assertEqual((invalid["fact_rewards"], invalid["required_slot_count"]), ({}, 2))
        self.assertEqual(self.score(objective_version=objective.FACT_CREDIT_VERSION)["effective_reward"], 1.0625)

    def test_fact_credit_aliases_preserve_values_raw_strict_report_and_tree(self):
        original_tree = to_json(self.tree)
        for alias in ("CELL[A][B]", "cell[ a ][ b ]", "cell['a'][\"b\"]", "cell[ａ][ｂ]"):
            for escaped in (False, True):
                with self.subTest(alias=alias, escaped=escaped):
                    note = copy.deepcopy(self.note)
                    note["corrected_facts"] = {alias: " 7 ", B: "5"}
                    original_note = copy.deepcopy(note)
                    text = json.dumps(note, ensure_ascii=escaped)
                    strict = notebook.grade_note(text, "stop", self.tree, self.audit, self.verifier)
                    original_strict = copy.deepcopy(strict)
                    with patch.object(self.verifier, "grade", wraps=self.verifier.grade) as grade:
                        result = self.score(text=text, strict_report=strict,
                                            objective_version=objective.FACT_CREDIT_VERSION)
                    self.assertEqual(result["fact_rewards"], {A: 1, B: 1})
                    self.assertEqual(result["effective_reward"], strict["reward"] + .0625)
                    self.assertEqual(result["strict_report_sha256"], digest(strict))
                    self.assertEqual(result["text_sha256"], digest(text))
                    self.assertEqual(note, original_note)
                    self.assertEqual(strict, original_strict)
                    self.assertIn(f"Observed facts:\n{A} =  7 \nSolution:\n",
                                  [call.args[1] for call in grade.call_args_list])
                    for version in (objective.VERSION, objective.COMPONENT_VERSION):
                        canonical_only = self.score(text=text, objective_version=version)
                        self.assertEqual(canonical_only["atomic_gate"], "INVALID_SLOT_IDS")
                        self.assertEqual(canonical_only["atomic_reward"], 0)
        self.assertEqual(to_json(self.tree), original_tree)

    def test_fact_credit_aliases_keep_wrong_unknown_null_and_collision_guards(self):
        for value in ("7", "999"):
            note = copy.deepcopy(self.note)
            note["corrected_facts"] = {"CELL[ A ][ B ]": value, B: "5"}
            result = self.score(note, objective_version=objective.FACT_CREDIT_VERSION)
            self.assertEqual(result["fact_rewards"], {A: int(value == "7"), B: 1})
        for facts in ({A: "7", "CELL[A][B]": "7", B: "5"},
                      {A: "7", "CELL[A][B]": "999", B: "5"},
                      {"CELL[A][B]": "7", B: "5", "unknown": "1"},
                      {"CELL[A][B]": None, B: "5"},
                      {"ＣＥＬＬ[a][b]": "7", B: "5"}):
            with self.subTest(facts=facts):
                note = copy.deepcopy(self.note)
                note["corrected_facts"] = facts
                result = self.score(note, objective_version=objective.FACT_CREDIT_VERSION)
                self.assertEqual(result["fact_rewards"], {})
                self.assertEqual(result["atomic_reward"], 0)

    def test_optional_registered_facts_do_not_receive_required_fact_credit(self):
        _, item, lineage, *_ = fixtures.records()
        item = replace(item, slots={A: "7", B: "5"}, slot_types={A: "number", B: "number"},
                       required_slots=(A,))
        root = self.fixture.root / "optional-fact"
        write_split(root, [self.actor], [item], [lineage])
        self.verifier = self.fixture.verifier = Verifier(root)
        self.tree, _ = self.fixture.tree([[WRONG] * 2, [WRONG] * 2, [RIGHT] * 2])
        self.audit, self.note = self.notebook(self.tree)
        for optional_value in ("5", "999"):
            with self.subTest(optional_value=optional_value):
                self.note["corrected_facts"][B] = optional_value
                result = self.score(objective_version=objective.FACT_CREDIT_VERSION)
                self.assertEqual(result["fact_rewards"], {A: 1})
                self.assertEqual(result["required_slot_count"], 1)
                self.assertEqual((result["numerator"], result["denominator"]), (6, 24))
                self.assertEqual(result["atomic_reward"], .0625)

    def setUp(self):
        self.fixture = fixtures.EpisodeNotebookTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.actor, item, lineage, *_ = fixtures.records()
        self.root = self.fixture.root / "atoms"
        item = replace(item, slots={A: "7", B: "5"}, slot_types={A: "number", B: "number"},
                       required_slots=(A, B))
        write_split(self.root, [self.actor], [item], [lineage])
        self.verifier = self.fixture.verifier = Verifier(self.root)
        self.tree, _ = self.fixture.tree([[WRONG] * 2, [WRONG] * 2, [RIGHT] * 2])
        self.audit, self.note = self.notebook(self.tree)

    def notebook(self, tree):
        audit = notebook.audit_episode(tree, self.actor, self.verifier, routing_rule=notebook.STRICT_ROUTE_RULE)
        note = fixtures.authored_note(tree, audit)
        note["corrected_facts"] = {A: "7", B: "5"}
        return audit, note

    def score(self, note=None, **kwargs):
        text = kwargs.pop("text", json.dumps(self.note if note is None else note))
        return objective.score_note(text, kwargs.pop("finish_reason", "stop"),
            kwargs.pop("tree", self.tree), kwargs.pop("audit", self.audit),
            kwargs.pop("verifier", self.verifier), actor=self.actor, **kwargs)

    def test_perfect_positive_fixed_denominator_strict_record_and_tree_unchanged(self):
        text = json.dumps(self.note)
        strict = notebook.grade_note(text, "stop", self.tree, self.audit, self.verifier)
        old_strict, old_tree = copy.deepcopy(strict), to_json(self.tree)
        result = self.score(strict_report=strict)
        self.assertEqual((result["numerator"], result["denominator"]), (12, 12))
        self.assertEqual((result["strict_reward"], result["atomic_reward"], result["effective_reward"]), (1., .25, 1.25))
        self.assertEqual(strict, old_strict)
        self.assertEqual(to_json(self.tree), old_tree)
        self.assertEqual(result["strict_report_sha256"], digest(strict))
        self.assertEqual(result["text_sha256"], digest(text))
        self.assertTrue(result["objective_independently_verified"])
        self.assertFalse(result["lesson_semantics_verified"])
        json.dumps(result, allow_nan=False)

    def test_one_false_positive_wrong_slot_loses_only_that_atom(self):
        note = copy.deepcopy(self.note)
        note["attempts"][0]["wrong_slots"] = [A, B]
        result = self.score(note)
        self.assertEqual((result["numerator"], result["denominator"], result["strict_reward"]), (11, 12, 0))
        self.assertAlmostEqual(result["effective_reward"], .25 * 11 / 12)

    def test_wrong_slot_aliases_keep_all_objective_rewards_and_source_binding(self):
        original = to_json(self.tree)
        for version in (objective.VERSION, objective.COMPONENT_VERSION, objective.FACT_CREDIT_VERSION):
            baseline = self.score(objective_version=version)
            for alias in ("CELL[A][B]", "cell[ a ][ b ]", "cell['a'][\"b\"]", "cell[ａ][ｂ]"):
                with self.subTest(version=version, alias=alias):
                    note = copy.deepcopy(self.note)
                    note["attempts"][2]["wrong_slots"] = [alias]
                    text = json.dumps(note, ensure_ascii=False)
                    result = self.score(text=text, objective_version=version)
                    for key in ("strict_reward", "numerator", "denominator", "effective_reward", "atomic_gate"):
                        self.assertEqual(result[key], baseline[key])
                    self.assertEqual(result["text_sha256"], digest(text))
                    self.assertEqual(note["attempts"][2]["wrong_slots"], [alias])
        self.assertEqual(to_json(self.tree), original)

    def test_wrong_slots_validate_normalized_order_and_multiplicity_for_every_objective(self):
        wrong_both = WRONG.replace(f"{B} = 5", f"{B} = 6")
        tree, _ = self.fixture.tree([[wrong_both] * 2, [RIGHT] * 2])
        audit, valid = self.notebook(tree)
        valid["attempts"][0]["wrong_slots"] = ["cell[ a ][ b ]", "CELL[C][D]"]
        for version in (objective.VERSION, objective.COMPONENT_VERSION, objective.FACT_CREDIT_VERSION):
            with self.subTest(version=version):
                self.assertEqual(self.score(valid, tree=tree, audit=audit, objective_version=version)["strict_reward"], 1.)
            for wrong in ([A, A, B], [A, "CELL[A][B]", B], ["CELL[C][D]", "CELL[A][B]"]):
                with self.subTest(version=version, wrong=wrong):
                    note = copy.deepcopy(valid)
                    note["attempts"][0]["wrong_slots"] = wrong
                    result = self.score(note, tree=tree, audit=audit, objective_version=version)
                    self.assertEqual(result["strict_reward"], 0.)
                    self.assertEqual(result["numerator"], 0)
                    self.assertEqual(result["atomic_gate"], "INVALID_SLOT_IDS")
            for invalid in ([A], [A, "cell[d][c]"], [None], [1], [True], A):
                with self.subTest(version=version, wrong=invalid):
                    note = copy.deepcopy(valid)
                    note["attempts"][0]["wrong_slots"] = invalid
                    result = self.score(note, tree=tree, audit=audit, objective_version=version)
                    self.assertEqual(result["strict_reward"], 0.)

    def test_missing_and_wrong_final_slots_keep_full_denominator(self):
        for facts in ({A: "7"}, {A: "999", B: "5"}):
            with self.subTest(facts=facts):
                note = copy.deepcopy(self.note)
                note["corrected_facts"] = facts
                result = self.score(note)
                self.assertEqual((result["numerator"], result["denominator"]), (6, 12))
                self.assertEqual(result["effective_reward"], .125)
        note = copy.deepcopy(self.note)
        note["corrected_facts"]["unknown"] = "1"
        result = self.score(note)
        self.assertEqual((result["numerator"], result["denominator"], result["effective_reward"]), (0, 12, 0))
        self.assertEqual(result["atomic_gate"], "FACT_CONFLICT_OR_UNKNOWN")

    def test_answer_alone_incorrect_answer_or_absent_attempt_evidence_gets_no_atoms(self):
        for change in ("facts", "answer", "history", "cause", "citation", "no_attempts"):
            with self.subTest(change=change):
                note = copy.deepcopy(self.note)
                if change == "facts":
                    note["corrected_facts"] = {}
                elif change == "answer":
                    note["corrected_answer"] = "8"
                elif change == "history":
                    for claim in note["attempts"]:
                        claim["answer"] = "999"
                elif change == "cause":
                    for claim in note["attempts"]:
                        claim["cause"] = "UNKNOWN"
                elif change == "citation":
                    for claim, sample in zip(note["attempts"], notebook.solution_samples(self.tree), strict=True):
                        last = len(sample.text.splitlines())
                        claim["line_ranges"] = [[last, last]]
                else:
                    note["attempts"] = []
                result = self.score(note)
                self.assertEqual((result["numerator"], result["denominator"], result["effective_reward"]), (0, 12, 0))

    def test_noncanonical_fact_ids_duplicates_and_invalid_ranges_fail_closed(self):
        for change in ("alias", "duplicate_alias", "duplicate_wrong", "ranges"):
            with self.subTest(change=change):
                note = copy.deepcopy(self.note)
                if change == "alias":
                    note["corrected_facts"] = {"CELL[a][b]": "7", B: "5"}
                elif change == "duplicate_alias":
                    note["corrected_facts"]["CELL[a][b]"] = "7"
                elif change == "duplicate_wrong":
                    note["attempts"][0]["wrong_slots"] = [A, A]
                else:
                    note["attempts"][0]["line_ranges"] = [[1, 2], [2, 3]]
                result = self.score(note)
                self.assertEqual(result["numerator"], 0)
                self.assertEqual(result["atomic_reward"], 0)
                # An old strict acceptance is still the unchanged strict term, never atomic credit.
                self.assertEqual(result["effective_reward"], result["strict_reward"])
        text = json.dumps(self.note)
        for invalid in (text.replace('"corrected_answer": "7"', '"corrected_answer": NaN'),
                        text.replace('"corrected_facts": {', '"corrected_facts": {"x":"1","x":"2",'),
                        "not JSON"):
            with self.subTest(invalid=invalid[:60]):
                self.assertEqual(self.score(text=invalid)["numerator"], 0)
        self.assertEqual(self.score(finish_reason="length")["numerator"], 0)

    def test_source_missing_or_conflicting_slot_cannot_gain_an_atom(self):
        for change in ("missing", "conflicting"):
            with self.subTest(change=change):
                tree = copy.deepcopy(self.tree)
                sample = tree.groups[0].samples[0]
                if change == "missing":
                    sample.text = sample.text.replace(f"{B} = 5\n", "")
                else:
                    sample.text = sample.text.replace(f"{A} = 8\n", f"{A} = 8\n{A} = 9\n")
                sample.result = self.verifier.grade(tree.problem_id, sample.text, sample.finish_reason, "obs")
                audit, note = self.notebook(tree)
                result = self.score(note, tree=tree, audit=audit)
                self.assertEqual((result["numerator"], result["denominator"]), (11, 12))

    def test_unknown_source_or_forged_strict_report_is_rejected(self):
        forged = copy.deepcopy(self.audit)
        forged["attempts"][0]["wrong_slots"] = []
        with self.assertRaisesRegex(ValueError, "source audit differs"):
            self.score(audit=forged)
        changed_tree = copy.deepcopy(self.tree)
        changed_tree.groups[0].samples[0].text = "Final answer: 99"
        with self.assertRaisesRegex(ValueError, "fresh verifier"):
            self.score(tree=changed_tree)
        strict = notebook.grade_note(json.dumps(self.note), "stop", self.tree, self.audit, self.verifier)
        strict["reward"] = 0.
        with self.assertRaisesRegex(ValueError, "strict report differs"):
            self.score(strict_report=strict)

    def test_singleton_cache_is_within_note_and_all_same_rewards_stay_identical(self):
        with patch.object(self.verifier, "grade", wraps=self.verifier.grade) as grade:
            result = self.score()
        for key, value in ((A, "7"), (B, "5")):
            query = f"Observed facts:\n{key} = {value}\nSolution:\n"
            self.assertEqual(sum(call.args[1] == query for call in grade.call_args_list), 1)
        self.assertEqual(result["denominator"], 12)
        note = copy.deepcopy(self.note)
        note["corrected_facts"] = {A: "7"}
        self.assertEqual([self.score(note)["effective_reward"] for _ in range(4)], [.125] * 4)

    def test_false_relations_never_add_required_atoms_and_corrected_false_edges_reject(self):
        fixture = native_fixtures.NativeRelationIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        tree, _, audit, note = fixture.tree()
        note["corrected_facts"] = {slot_id(key): value for key, value in note["corrected_facts"].items()}
        def score(version=objective.VERSION):
            return objective.score_note(json.dumps(note), "stop", tree, audit, fixture.fixture.verifier,
                                        actor=fixture.fixture.actor, objective_version=version)
        result = score()
        self.assertEqual((result["numerator"], result["denominator"]), (12, 12))
        original_facts = dict(note["corrected_facts"])
        for key, value in (("EDGE[ H , D ]", "1"), ("edge[h,d]", "null"), ("edge[x,d]", "1")):
            note["corrected_facts"] = original_facts | {key: value}
            for version in (objective.VERSION, objective.COMPONENT_VERSION, objective.FACT_CREDIT_VERSION):
                with self.subTest(key=key, value=value, version=version):
                    self.assertEqual(score(version)["numerator"], 0)
                    self.assertEqual(score(version)["atomic_reward"], 0)

    def test_verifier_service_only_interface_and_feedback_policy_are_supported(self):
        self.tree.notebook["rewarded_notebook"] = {"feedback_policy": notebook.FEEDBACK_POLICY}
        text = json.dumps(self.note)
        strict = notebook.grade_note(text, "stop", self.tree, self.audit, self.verifier,
                                     feedback_policy=notebook.FEEDBACK_POLICY)
        with VerifierService(self.root, forbidden_roots=[], timeout=60) as verifier:
            result = self.score(verifier=verifier, strict_report=strict)
        self.assertEqual(result["effective_reward"], 1.25)

    def test_effective_helper_rejects_invalid_counts_and_keeps_strict_priority(self):
        self.assertEqual(objective.contract()["atomic_weight"], .25)
        self.assertLessEqual(objective.effective_reward(0, 12, 12), .25)
        self.assertGreaterEqual(objective.effective_reward(1, 0, 12), 1.)
        for values in ((True, 1, 2), (.5, 1, 2), (float("nan"), 1, 2), (0, 3, 2),
                       (0, -1, 2), (0, 0, -1), (0, True, 2), (0, 1, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                objective.effective_reward(*values)

    def test_components_preserve_independent_correctness_without_strict_certification(self):
        note = copy.deepcopy(self.note)
        for claim in note["attempts"]:
            claim["cause"] = "UNKNOWN"
        original = to_json(self.tree)
        result = self.score(note, objective_version=objective.COMPONENT_VERSION)
        self.assertEqual(result["strict_reward"], 0.)
        self.assertEqual(result["component_counts"], {"required_facts": 2, "history_with_conclusion": 6,
            "cause_with_history": 0, "error_locations_with_evidence": 6})
        self.assertEqual((result["numerator"], result["denominator"]), (36, 48))
        self.assertEqual(result["effective_reward"], .1875)
        self.assertEqual(to_json(self.tree), original)
        self.assertEqual(self.score(note)["effective_reward"], 0.)  # V14 remains unchanged.

    def test_components_do_not_reward_empty_error_list_when_actual_error_exists(self):
        note = copy.deepcopy(self.note)
        for claim in note["attempts"]:
            claim["wrong_slots"] = []
        result = self.score(note, objective_version=objective.COMPONENT_VERSION)
        self.assertEqual(result["component_counts"]["error_locations_with_evidence"], 2)
        self.assertEqual(result["component_denominators"]["error_locations_with_evidence"], 6)
        self.assertEqual(result["strict_reward"], 0.)

    def test_component_errors_lose_only_the_claimed_credit_and_fixed_denominators(self):
        note = copy.deepcopy(self.note)
        note["corrected_facts"] = {A: "999"}
        for claim, sample in zip(note["attempts"], notebook.solution_samples(self.tree), strict=True):
            last = len(sample.text.splitlines())
            claim["line_ranges"] = [[last, last]]
        note["attempts"][0]["answer"] = "999"
        result = self.score(note, objective_version=objective.COMPONENT_VERSION)
        self.assertEqual(result["component_counts"], {"required_facts": 0, "history_with_conclusion": 5,
            "cause_with_history": 5, "error_locations_with_evidence": 2})
        self.assertEqual((result["numerator"], result["denominator"]), (24, 48))
        self.assertEqual(result["effective_reward"], .125)
        note["corrected_facts"]["unknown"] = "1"
        self.assertEqual(self.score(note, objective_version=objective.COMPONENT_VERSION)["atomic_reward"], 0.)

    def test_component_protocol_uses_no_variance_fabrication_or_wrong_answer_rescue(self):
        self.assertEqual(objective.contract(objective.COMPONENT_VERSION)["component_weight"], .25)
        self.assertEqual([self.score(objective_version=objective.COMPONENT_VERSION)["effective_reward"]
                          for _ in range(4)], [1.25] * 4)
        for change in ("answer", "format", "truncated"):
            note = copy.deepcopy(self.note)
            kwargs = {"objective_version": objective.COMPONENT_VERSION}
            if change == "answer":
                note["corrected_answer"] = "8"
            elif change == "format":
                note["attempts"] = []
            else:
                kwargs["finish_reason"] = "length"
            self.assertEqual(self.score(note, **kwargs)["effective_reward"], 0.)
        with self.assertRaises(ValueError):
            self.score(objective_version="made-up")


if __name__ == "__main__":
    unittest.main()
