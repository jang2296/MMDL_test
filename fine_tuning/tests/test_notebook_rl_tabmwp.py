"""Authored fixture only; native solution fields are deliberately unreadable."""
import unittest

from fine_tuning.notebook_rl.adapters.tabmwp_rl import convert
from fine_tuning.notebook_rl.answers import compare


class NoSolution(dict):
    def __getitem__(self, key):
        if key == "solution":
            raise AssertionError("native solution read")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "solution":
            raise AssertionError("native solution read")
        return super().get(key, default)


def fixture(**updates):
    return NoSolution(dict(split="train", question="How many units are in the East row?", answer="17",
                           ques_type="free_text", choices=None, ans_type="integer_number",
                           table_for_pd={"Region": ["East", "West"], "Units": ["17", "23"]},
                           table_title="Regional inventory", table="Region | Units\nEast | 17\nWest | 23",
                           row_num=3, column_num=2, solution=object()) | updates)


def records(native=None, **kwargs):
    return convert(fixture() if native is None else native, "123", kwargs.get("split", "train"),
                   "tabmwp/123.png", "a" * 64, "obs")


class TabMWPTests(unittest.TestCase):
    def test_separate_records_no_solution(self):
        a, v, lineage = records()
        self.assertNotIn("gold", vars(a))
        self.assertEqual(v.gold, "17")
        self.assertTrue(compare("17", v))
        self.assertEqual(v.slots["cell[East][Units]"], "17")
        self.assertEqual(v.required_slots, ())
        self.assertEqual((lineage.label_provenance, lineage.upstream_origin), ("HUMAN", "IXL"))
        self.assertFalse(lineage.benchmark_derived)
        self.assertFalse(lineage.model_touched)

    def test_upstream_holdouts_never_train(self):
        for source_split in ("dev", "test", "validation", None):
            with self.assertRaises(ValueError):
                records(fixture(split=source_split))
        self.assertEqual(records(split="sealed")[2].split, "sealed")

    def test_mcq_gold_label_original_order(self):
        a, v, _ = records(fixture(ques_type="multi_choice", choices=["23", "17", "12"]))
        self.assertEqual((a.choices, v.gold), (("23", "17", "12"), "B"))
        self.assertTrue(compare("B", v))
        with self.assertRaises(ValueError):
            records(fixture(ques_type="multi_choice", choices=["17", "17"]))

    def test_numeric_precision_and_fraction(self):
        v = records(fixture(ans_type="decimal_number", answer="1.25"))[1]
        self.assertEqual(v.accept, {"round_to": 2})
        self.assertTrue(compare("1.2501", v))
        v = records(fixture(ans_type="decimal_number", answer="1/4"))[1]
        self.assertEqual(v.answer_type, "fraction")
        self.assertTrue(compare("1/4", v))
        self.assertFalse(compare("0.25", v))

    def test_text_boolean(self):
        for answer, alias in (("yes", "true"), ("false", "no")):
            v = records(fixture(ans_type="boolean_text", answer=answer))[1]
            self.assertTrue(compare(alias, v))
        self.assertTrue(compare("East", records(fixture(ans_type="extractive_text", answer="East"))[1]))

    def test_table_family_ignores_question_answer(self):
        self.assertEqual(records()[2].structure_hash,
                         records(fixture(question="A different question?", answer="23"))[2].structure_hash)

    def test_malformed_inputs_rejected(self):
        for values in ({"ans_type": "unknown"}, {"answer": "17kg"}, {"row_num": 99},
                       {"choices": []}, {"table": "inconsistent"},
                       {"table_for_pd": {"Region": ["East", "East"], "Units": ["17", "23"]},
                        "table": "Region | Units\nEast | 17\nEast | 23"}):
            with self.assertRaises(ValueError):
                records(fixture(**values))


if __name__ == "__main__":
    unittest.main()
