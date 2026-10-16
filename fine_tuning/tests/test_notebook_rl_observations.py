"""Authored observation cases, not benchmark-derived labels."""
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl.observations import compare_slots, normalize_slot, parse_observations, slot_id
from fine_tuning.notebook_rl.schema import VerifierItem, to_json


def item():
    return VerifierItem("fixture", "int", "1", {}, (),
                        {"bar[Mar]": "4", "wire[R1]": "N0-N1", "note[1.1]": "C#4"},
                        {"bar[Mar]": "number", "wire[R1]": "relation", "note[1.1]": "pitch"},
                        ("bar[Mar]", "wire[R1]"), (), ())


class ObservationTests(unittest.TestCase):
    def test_bounds_only(self):
        text = "bar[Mar] = 99\nObserved facts:\n- bar[Mar] = 4\nSolution:\nbar[Mar] = 99\nFinal answer: 1"
        self.assertEqual(parse_observations(text), {"bar[mar]": ["4"]})
        for text in ("bar[Mar] = 4", "Observed facts:\nbar[Mar] = 4", "Solution:\nbar[Mar] = 4"):
            self.assertEqual(parse_observations(text), {})

    def test_key_variants(self):
        self.assertEqual(slot_id('Cell[ " Row A " ][ \' COL B\']'), "cell[rowa][colb]")
        parsed = parse_observations("Observed facts:\nRINGS = 2\n- cell[ a ][b] = 3\nSolution:")
        self.assertEqual(parsed, {"rings": ["2"], "cell[a][b]": ["3"]})

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
        self.assertEqual((report.parsed, report.wrong_required), (1, ("bar[mar]",)))

    def test_directed_values_keep_order(self):
        q = replace(item(), slots={"dir[e1]": "A->B", "trans[q0,0]": "q1"},
                    slot_types={"dir[e1]": "relation", "trans[q0,0]": "relation"}, required_slots=("dir[e1]",))
        self.assertEqual(compare_slots({"dir[e1]": ["B->A"]}, q).wrong_required, ("dir[e1]",))
        self.assertEqual(compare_slots({"dir[e1]": ["a → b"], "trans[q0,0]": ["Q1"]}, q).all_correct, 2)

    def test_multiple_blocks_preserve_conflicts(self):
        text = "Observed facts:\nbar[Mar] = 4\nSolution:\nfirst\nObserved facts:\nbar[Mar] = 8\nSolution:"
        self.assertEqual(compare_slots(parse_observations(text), item()).conflicting, ("bar[mar]",))

    def test_bad_reference_rejected(self):
        for q in (replace(item(), slots={"bar[X]": "1", "bar[x]": "2"},
                          slot_types={"bar[X]": "number", "bar[x]": "number"}, required_slots=()),
                  replace(item(), slots=item().slots | {"bar[Mar]": "not numeric"})):
            with self.assertRaises(ValueError):
                compare_slots({}, q)


if __name__ == "__main__":
    unittest.main()
