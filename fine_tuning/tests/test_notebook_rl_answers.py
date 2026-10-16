"""Authored answer-reward cases; no benchmark examples, weights, or network."""
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl.answers import compare, extract_final, is_ambiguous, mcq_choice, number
from fine_tuning.notebook_rl.schema import VerifierItem


def item(kind="int", gold="53", accept=None, choices=()):
    return VerifierItem("fixture", kind, gold, accept or {}, choices, {}, {}, (), (), ())


class AnswerTests(unittest.TestCase):
    def test_extraction_wrappers(self):
        for value in ("53", "**53**", "$53$", r"\boxed{53}", r"$\boxed{53}$", "53.", "**53.**"):
            with self.subTest(value=value):
                result = extract_final("Solution: 2 + 2\nFinal answer: " + value, "stop")
                self.assertEqual((result.status, result.answer), ("OK", "53"))

    def test_final_line_not_reasoning_or_next_line(self):
        self.assertFalse(compare(extract_final("The value 53 appears here.\nFinal answer: 35", "stop").answer, item()))
        for text in ("53", "Error check: I re-checked carefully", "Final answer:\n53", "Final answer:  "):
            self.assertEqual(extract_final(text, "stop").status, "INVALID_NO_FINAL")
        with self.assertRaises(ValueError):
            extract_final("Final answer: 53", "exception")

    def test_multiple_retraction_and_cutoff(self):
        for text in ("Final answer: 53 or 35", "Final answer: 53\nFinal answer: 53",
                     "Final answer: 53\nActually, the answer is 35.",
                     "Final answer: 53\n" + "x" * 201):
            with self.subTest(text=text):
                self.assertEqual(extract_final(text, "stop").status, "INVALID_MULTIPLE")
        for text in ("Final answer: 53", "no answer"):
            self.assertEqual(extract_final(text, "length").status, "TRUNCATED")

    def test_numbers_and_delimiters(self):
        self.assertTrue(compare("1,234", item(gold="1234")))
        self.assertIsNone(number("1,23"))
        self.assertIsNone(number("nan"))
        self.assertIsNone(number("1e1000000"))
        self.assertIsNone(number("1/0"))
        self.assertIsNone(compare("value 53", item()))
        for answer in ("53, 35", "53/35/99", "B/C", "53 or 35"):
            self.assertTrue(is_ambiguous(answer, item()))
        self.assertFalse(is_ambiguous("3/4", item("fraction", "3/4")))

    def test_fraction_decimal_policy(self):
        self.assertTrue(compare("3/4", item("fraction", "3/4")))
        self.assertFalse(compare("0.75", item("fraction", "3/4")))
        self.assertTrue(compare("0.75", item("fraction", "3/4", {"accept_decimal": True})))
        self.assertFalse(compare("0.75001", item("fraction", "3/4", {"accept_decimal": True})))

    def test_exact_decimal_rounding_and_tolerances(self):
        self.assertFalse(compare("0.2778", item("decimal", "0.277777")))
        self.assertTrue(compare("2.345", item("decimal", "2.34", {"round_to": 2})))
        self.assertFalse(compare("2.355", item("decimal", "2.35", {"round_to": 2})))
        self.assertTrue(compare("10.01", item("decimal", "10", {"abs_tol": "0.01"})))
        self.assertFalse(compare("10.0101", item("decimal", "10", {"abs_tol": "0.01"})))
        self.assertTrue(compare("101", item("decimal", "100", {"rel_tol": "0.01"})))
        self.assertFalse(compare("0.001", item("decimal", "0", {"rel_tol": "0.01"})))
        for accept in ({"round_to": True}, {"abs_tol": "-1"}, {"abs_tol": 0.01},
                       {"round_to": 2, "abs_tol": "0.1"}):
            self.assertIsNone(compare("53", item("decimal", "53", accept)))

    def test_units_are_not_erased(self):
        q = item("unit_number", "53 V", {"unit_aliases": ["V", "volt"], "require_unit": True})
        for answer in ("53 V", "53 volt", "53V"):
            self.assertTrue(compare(answer, q))
        for answer in ("53 A", "53 mV", "53 v", "53"):
            self.assertFalse(compare(answer, q))
        self.assertTrue(compare("53", replace(q, accept=q.accept | {"require_unit": False})))
        self.assertIsNone(compare("53 V", replace(q, gold="53 A")))

    def test_fractional_units_parse_the_whole_number(self):
        q = item("unit_number", "23/12 A", {"unit_aliases": ["A"], "require_unit": True})
        for value in ("23/12 A", "23 / 12 A", "46/24A"):
            self.assertTrue(compare(value, q))
            self.assertFalse(is_ambiguous(value, q))
        self.assertFalse(compare("23/12 V", q))
        self.assertFalse(compare("23/13 A", q))
        self.assertIsNone(compare("23/0 A", q))

    def test_mcq_letter_content_conflict_and_gold_independence(self):
        q = item("mcq", "B", choices=("42", "24", "other"))
        for answer in ("B", "(B)", "B.", "(B).", "B. 24", "(B): 24"):
            self.assertTrue(compare(answer, q))
        self.assertFalse(compare("A", q))
        self.assertEqual(mcq_choice("B. 42", q.choices), ("B", True))
        self.assertTrue(is_ambiguous("B. 42", q))
        self.assertIsNone(compare("J", q))
        self.assertEqual(mcq_choice("(B", q.choices), (None, False))
        self.assertEqual(is_ambiguous("B. 42", q), is_ambiguous("B. 42", replace(q, gold="A")))

    def test_text_is_whole_normalized_answer(self):
        q = item("short_text", "major", {"aliases": ["Major key"]})
        self.assertTrue(compare("ＭＡＪＯＲ", q))
        self.assertTrue(compare(" Major   key. ", q))
        self.assertFalse(compare("majority", q))
        self.assertFalse(compare("caterpillar", item("short_text", "cat")))

    def test_case_sensitive_symbol_policy(self):
        q = item("short_text", "m3", {"case_sensitive": True})
        self.assertTrue(compare(" m3. ", q))
        self.assertFalse(compare("M3", q))
        self.assertIsNone(compare("m3", replace(q, accept={"case_sensitive": 1})))

    def test_list_order_and_multiplicity(self):
        q = item("list", "A B A", {"ordered": True})
        self.assertTrue(compare("[A, B, A]", q))
        self.assertFalse(compare("A A B", q))
        q = replace(q, accept={"ordered": False})
        self.assertTrue(compare("A A B", q))
        self.assertFalse(compare("A B B", q))
        self.assertTrue(compare("1, 2", item("list", "1 2", {"element_type": "int"})))
        self.assertTrue(compare("1.00 2", item("list", "1 2", {"element_type": "decimal"})))
        self.assertTrue(compare("2 1.00", item("list", "1 2", {"element_type": "decimal", "ordered": False})))
        self.assertFalse(compare("0.5 1/4", item("list", "1/4 1/2", {"element_type": "fraction", "ordered": False})))
        self.assertTrue(compare("0.5 1/4", item("list", "1/4 1/2", {
            "element_type": "fraction", "ordered": False, "element_accept": {"accept_decimal": True}})))
        self.assertIsNone(compare("1,,2", item("list", "1 2", {"element_type": "int"})))


if __name__ == "__main__":
    unittest.main()
