"""Synthetic direct-edge assertions; these fixtures are never training targets."""
from copy import deepcopy
import unittest

from fine_tuning.error_note import v18_rows as v18, v18b_rows as rows
from fine_tuning.tests.test_error_note_v17_rows import TOKENIZER, output
from fine_tuning.tests.test_error_note_v18b_rows import graph_item, note


class DirectEdgeClaims(unittest.TestCase):
    def audit(self, text, *, complete=True, directed=False):
        item, private = graph_item(text + "\nFinal answer: A", complete=complete, directed=directed)
        ledger = rows.audit_t1_v18b(item, verifier=private, tokenizer=TOKENIZER)
        return item, private, [claim for claim in ledger["claims"] if claim["kind"] == "EDGE_PRESENT"]

    def test_affirmative_bold_case_and_byte_offsets(self):
        for phrase in ("node 0 is directly connected to node 2",
                       "**node 0 is directly connected to node 2**",
                       "Node **0** is directly connected to node **2**",
                       "NODE 0 IS DIRECTLY CONNECTED TO NODE 2",
                       "Node 0 and node 2 are directly connected",
                       "There is an edge between node 0 and node 2"):
            with self.subTest(phrase=phrase):
                text = "관측: " + phrase + " via an edge."
                item, _, claims = self.audit(text)
                self.assertEqual(len(claims), 1, claims)
                claim = claims[0]
                self.assertEqual((claim["assertion"], claim["verdict"]), ("ASSERTED", "CONTRADICTED"))
                self.assertEqual(claim["original_entity"], [0, 2])
                self.assertIs(claim["original_value"], True)
                start, end = claim["original_byte_spans"][0]
                self.assertEqual(item["t1"]["text"].encode()[start:end].decode(), claim["original_quote"])

    def test_existing_edge_reversed_undirected_and_explicit_denial(self):
        for phrase, expected, value in (
            ("Node 1 is directly connected to node 0.", "VERIFIED", True),
            ("Node 0 is not directly connected to node 2.", "VERIFIED", False),
            ("Node 0 is not directly connected to node 1.", "CONTRADICTED", False),
            ("There is no direct edge between node 0 and node 2.", "VERIFIED", False),
            ("There is not an edge between node 0 and node 1.", "CONTRADICTED", False),
        ):
            with self.subTest(phrase=phrase):
                _, _, claims = self.audit(phrase)
                self.assertEqual(len(claims), 1, claims)
                self.assertEqual((claims[0]["assertion"], claims[0]["verdict"]), ("ASSERTED", expected))
                self.assertIs(claims[0]["original_value"], value)

    def test_paths_questions_hypotheticals_and_withdrawals_are_not_asserted_edges(self):
        for phrase in (
            "There is a path between node 0 and node 2.",
            "Node 0 is connected to node 2 through node 1.",
            "Node 0 is indirectly connected to node 2.",
            "Is node 0 directly connected to node 2?",
            "Determine whether node 0 is directly connected to node 2.",
            "Suppose node 0 is directly connected to node 2.",
            "Suppose:\nNode 0 is directly connected to node 2.",
            "If there is an edge between node 0 and node 2, a path exists.",
            "There might be an edge between node 0 and node 2.",
            "Node 0 is directly connected to node 2. Actually, that was incorrect.",
            "The statement 'node 0 is directly connected to node 2' is false.",
            "It is not true that node 0 is directly connected to node 2.",
        ):
            with self.subTest(phrase=phrase):
                _, _, claims = self.audit(phrase)
                self.assertFalse(any(c["assertion"] == "ASSERTED" and c["verdict"] != "UNKNOWN" for c in claims), claims)

    def test_embedded_propositions_do_not_reject_a1_or_train_false_repairs(self):
        edge = "node 0 is directly connected to node 2"
        for phrase in (
            f"It is impossible that {edge}.",
            f"It is inconceivable that {edge}.",
            f"I deny that {edge}.",
            f"No, {edge}.",
            f"Without evidence, {edge}.",
            f"It is impossible that\n{edge}.",
            f"The statement is:\n{edge}.",
            f'"{edge}."',
            f"`{edge}`",
            f"> {edge}.",
            f"{edge}, which is impossible.",
            f"{edge} only in the hypothesis.",
            "Node 0 is not directly connected to node 2? ",
            'It is impossible that "node 0 is not directly connected to node 1".',
        ):
            with self.subTest(phrase=phrase):
                item, private, claims = self.audit(phrase)
                self.assertTrue(claims)
                self.assertTrue(all(c["verdict"] == "UNKNOWN" for c in claims), claims)
                admitted = rows.verify_solution_v18b(item, verifier=private, tokenizer=TOKENIZER, gold="A")
                self.assertTrue(admitted["accepted"], admitted)
                item["t1"]["status"] = "INCORRECT"
                for quote in (phrase, claims[0]["original_quote"]):
                    receipt, _ = note(item, private, quote=quote, corrected="false", unit="")
                    self.assertFalse(receipt["n_accepted"], receipt)
                    self.assertFalse(any(receipt["verified_content_token_mask"]), receipt)
                    self.assertFalse(receipt["verified_byte_spans"], receipt)
                    self.assertIsNone(rows.n_row_v18b(item, tokenizer=TOKENIZER, receipt=receipt))

    def test_actual_005147_assertive_context_is_preserved(self):
        phrase = ("- From the visual layout and edge connections, we can see that "
                  "**node 0 is directly connected to node 2** via an edge. "
                  "There is a clear line connecting them.")
        item, private, claims = self.audit(phrase)
        self.assertEqual([(c["assertion"], c["verdict"]) for c in claims],
                         [("ASSERTED", "CONTRADICTED")])
        receipt, _ = note(item, private, quote=claims[0]["original_quote"], corrected="false", unit="")
        self.assertTrue(receipt["n_accepted"], receipt)
        self.assertTrue(any(receipt["verified_content_token_mask"]), receipt)
        admitted = rows.verify_solution_v18b(item, verifier=private, tokenizer=TOKENIZER, gold="A")
        self.assertFalse(admitted["accepted"], admitted)

    def test_directed_edges_require_explicit_order_and_sealed_scope(self):
        for phrase, expected in (
            ("Node 0 is directly connected to node 1.", "UNKNOWN"),
            ("There is an edge between node 0 and node 1.", "UNKNOWN"),
            ("There is an edge from node 0 to node 1.", "VERIFIED"),
            ("THERE IS AN EDGE FROM NODE 1 TO NODE 0.", "CONTRADICTED"),
            ("There is no edge from node 1 to node 0.", "VERIFIED"),
            ("There is an edge from node 0 and node 1.", None),
        ):
            with self.subTest(phrase=phrase):
                _, _, claims = self.audit(phrase, directed=True)
                self.assertEqual([c["verdict"] for c in claims], [] if expected is None else [expected])
        _, _, claims = self.audit("Node 0 is directly connected to node 4.", complete=False)
        self.assertEqual([c["verdict"] for c in claims], ["UNKNOWN"])

    def test_later_opposite_assertion_retracts_the_earlier_edge(self):
        _, _, claims = self.audit("Node 0 is directly connected to node 2.\n"
                                  "There is no edge between node 2 and node 0.")
        self.assertEqual([(c["assertion"], c["verdict"]) for c in claims],
                         [("RETRACTED", "UNKNOWN"), ("ASSERTED", "VERIFIED")])

    def test_wrong_direct_edge_rejects_a1_and_c_but_preserves_v18(self):
        item, private, _ = self.audit("**node 0 is directly connected to node 2** via an edge.")
        untouched = deepcopy(item)
        old = v18.verify_solution_v18(item, verifier=private, tokenizer=TOKENIZER, gold="A")
        self.assertTrue(old["accepted"], old)
        receipt = rows.verify_solution_v18b(item, verifier=private, tokenizer=TOKENIZER, gold="A")
        self.assertIn("PROCESS_CONTRADICTION", receipt["reasons"])
        self.assertIsNone(rows.a1_row_v17(item, tokenizer=TOKENIZER, receipt=receipt))
        item["resolve"] = output(item["t1"]["text"], prompt=item["p0_text"])
        receipt = rows.verify_solution_v18b(item, stage="resolve", verifier=private,
                                          tokenizer=TOKENIZER, gold="A")
        self.assertIn("PROCESS_CONTRADICTION", receipt["reasons"])
        self.assertIsNone(rows.c_row_v17(item, tokenizer=TOKENIZER, receipt=receipt))
        self.assertEqual(item["t1"], untouched["t1"])


if __name__ == "__main__":
    unittest.main()
