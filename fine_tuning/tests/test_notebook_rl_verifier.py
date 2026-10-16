"""Authored reward cases, isolated from real benchmark data."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fine_tuning.notebook_rl.schema import DiagTarget, VerifierResult
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import Verifier, answer_reward, diagnosis_fields
from fine_tuning.tests.test_notebook_rl_schema import records


class VerifierTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        actor, verifier, lineage, *_ = records()
        self.item = verifier
        self.root = Path(temp.name) / "vault"
        write_split(self.root, [actor], [verifier], [lineage])
        self.verifier = Verifier(self.root)

    def grade(self, text, finish="stop", mode="plain"):
        return self.verifier.grade(self.item.problem_id, text, finish, mode)

    def test_every_answer_status(self):
        cases = (("Final answer: 7", "CORRECT", 1.0), ("Final answer: 8", "INCORRECT", 0),
                 ("The answer might be 7", "INVALID_NO_FINAL", 0),
                 ("Final answer: 7 or 8", "INVALID_MULTIPLE", 0),
                 ("Final answer: bananas", "UNVERIFIABLE", 0),
                 ("Final answer: 7,8", "INVALID_MULTIPLE", 0))
        for text, status, reward in cases:
            result = self.grade(text)
            self.assertEqual((result.status, result.reward_answer), (status, reward))
        self.assertEqual(self.grade("Final answer: 7", "length").status, "TRUNCATED")
        with self.assertRaises(ValueError):
            self.grade("Final answer: 7", mode="unsupported")

    def test_observations_are_recorded_even_for_correct_or_invalid_answers(self):
        prefix = "Observed facts:\ncell[a][b] = 8\nSolution:\n"
        for suffix in ("Final answer: 7", "Final answer: 8", "No final answer"):
            result = self.grade(prefix + suffix, mode="obs")
            self.assertEqual(result.slots.wrong_required, ("cell[a][b]",))
            self.assertIsNone(self.grade(prefix + suffix).slots)

    def test_no_exec_and_no_gold_rescue(self):
        self.assertEqual(self.grade("I calculate 7.\nFinal answer: 8").status, "INCORRECT")
        self.assertEqual(self.grade("Final answer: __import__('os').system('false')").status, "UNVERIFIABLE")
        self.assertFalse((self.root / "unexpected").exists())

    def test_mcq_conflict_and_fraction_unit(self):
        q = replace(self.item, answer_type="mcq", gold="B", choices=("70", "7", "8"))
        self.verifier.items[q.problem_id] = q
        self.assertEqual(self.grade("Final answer: B. 70").status, "INVALID_MULTIPLE")
        self.assertEqual(self.grade("Final answer: B. 7").status, "CORRECT")
        self.verifier.items[q.problem_id] = replace(q, answer_type="unit_number", gold="23/12 A",
            choices=(), accept={"unit_aliases": ["A"], "require_unit": True})
        self.assertEqual(self.grade("Final answer: 23/12 A").status, "CORRECT")
        self.assertEqual(self.grade("Final answer: 23/12 V").status, "INCORRECT")

    def test_visual_reward_requires_graded_final_answer(self):
        facts = "Observed facts:\ncell[a][b] = 7\nSolution:\n"
        self.assertEqual(answer_reward(self.grade(facts, mode="obs"), visual_lambda=.1), 0)
        self.assertEqual(answer_reward(self.grade(facts + "Final answer: 8", mode="obs"), visual_lambda=.1), .1)
        self.assertEqual(answer_reward(self.grade(facts + "Final answer: 7", mode="obs"), visual_lambda=.1), 1.1)
        self.assertEqual(answer_reward(self.grade(facts + "Final answer: 7", "length", "obs"), visual_lambda=.1), 0)
        for status in ("UNVERIFIABLE", "SYSTEM_ERROR", "CONTEXT_BLOCKED"):
            with self.assertRaises(ValueError):
                answer_reward(VerifierResult(status, None, 0.0, None, "fixture"))
        for coefficient in (True, -.1, .2, float("nan")):
            with self.assertRaises(ValueError):
                answer_reward(self.grade("Final answer: 7"), visual_lambda=coefficient)

    def test_diagnosis_exact_format_class_and_slot(self):
        target = DiagTarget("VISUAL", ("cell[a][b]",))
        note = "Cause: VISUAL_MISREAD\nWrong item: cell[a][b]\nLesson: Recheck the cell."
        def score(text, t=target):
            return self.verifier.grade_diagnosis(self.item.problem_id, text, t)
        self.assertEqual(score(note), 1.25)
        self.assertEqual(score(note.replace("cell[a][b]", "cell[c][d]")), 1)
        self.assertEqual(score(note.replace("VISUAL_MISREAD", "NON_VISUAL")), 0)
        self.assertEqual(score(note + "\nCause: NON_VISUAL"), 0)
        self.assertEqual(score(note.replace("Recheck the cell.", "")), 0)
        self.assertEqual(score("Cause: FINALIZATION\nWrong item: none\nLesson: Give one answer.",
                               DiagTarget("FINALIZATION", ())), 1)
        self.assertIsNone(diagnosis_fields("I think VISUAL_MISREAD"))
        with self.assertRaises(ValueError):
            score(note, DiagTarget("VISUAL", ("unregistered",)))


if __name__ == "__main__":
    unittest.main()
