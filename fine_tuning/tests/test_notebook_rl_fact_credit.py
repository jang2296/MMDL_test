"""Authored CPU credit-boundary fixtures, never model self-improvement evidence."""
import copy
import json
import unittest
from types import SimpleNamespace

from fine_tuning.notebook_rl.fact_credit import group_credit, token_spans, value_spans
from fine_tuning.notebook_rl.local_eval import digest
from fine_tuning.notebook_rl.notebook_objective import FACT_CREDIT_VERSION
from fine_tuning.notebook_rl.updates import group_has_signal
from fine_tuning.tests.test_notebook_rl_updates import Characters


def row(text, rewards, *, index=0, strict=0, required=2, tokens=None):
    sample = SimpleNamespace(sample_id=f"s{index}", text=text, reward=strict,
        completion_token_ids=[ord(c) for c in text] if tokens is None else tokens)
    score = {"objective_version": FACT_CREDIT_VERSION, "text_sha256": digest(text),
             "strict_reward": strict, "required_slot_count": required, "fact_rewards": rewards}
    return sample, score


class FactCreditTests(unittest.TestCase):
    def credit(self, rows, tokenizer=None):
        return group_credit([r[0] for r in rows], {r[0].sample_id: r[1] for r in rows},
                            tokenizer or Characters())

    def test_json_paths_preserve_escaped_literals_repeated_values_and_whitespace(self):
        text = ' {"lesson":"7", "corrected_facts" : {"a":"7", "b":"\\u03b1\\n7"},"other":{"a":"7"}} '
        spans = value_spans(text)
        self.assertEqual({k: text[a:b] for k, (a, b) in spans.items()},
                         {"a": '"7"', "b": '"\\u03b1\\n7"'})
        self.assertGreater(spans["a"][0], text.index('"corrected_facts"'))
        for invalid in ('{"corrected_facts":{"a":"1","a":"2"}}',
                        '{"corrected_facts":{},"corrected_facts":{}}',
                        '{"corrected_facts":{"a":7}}', '{"corrected_facts":{},"x":NaN}',
                        '{"corrected_facts":[]}'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                value_spans(invalid)

    def test_mixed_slot_credit_has_local_signs_and_leaves_prose_untouched(self):
        texts = [json.dumps({"corrected_facts": {"a": "7", "b": "wrong"}, "lesson": "made up"}),
                 json.dumps({"corrected_facts": {"a": "wrong", "b": "5"}, "lesson": "made up"})]
        rows = [row(texts[0], {"a": 1, "b": 0}), row(texts[1], {"a": 0, "b": 1}, index=1)]
        out = self.credit(rows)
        self.assertTrue(group_has_signal(SimpleNamespace(kind="DIAG", samples=[r[0] for r in rows]),
                                        {r[0].sample_id: r[1] for r in rows}))
        for sample, score in rows:
            credit = out[sample.sample_id]
            self.assertEqual(credit["strict_advantage"], 0)
            selected = set()
            for fact in credit["facts"]:
                self.assertEqual(fact["advantage"], .5 if score["fact_rewards"][fact["slot"]] else -.5)
                self.assertTrue(fact["token_indices"])
                for index in fact["token_indices"]:
                    self.assertEqual(credit["partial_token_advantages"][index], .0625 / 2 * fact["advantage"])
                selected.update(fact["token_indices"])
            self.assertTrue(all(value == 0 for i, value in enumerate(credit["partial_token_advantages"])
                                if i not in selected))

    def test_alias_keys_bind_only_original_escaped_value_literals(self):
        canonical = "cell[a][b]"
        for alias in ("CELL[A][B]", "cell[ a ][ b ]", "cell['a'][\"b\"]", "cell[ａ][ｂ]"):
            for escaped in (False, True):
                with self.subTest(alias=alias, escaped=escaped):
                    text = ('{"lesson":"7", "corrected_facts":{' + json.dumps(alias, ensure_ascii=escaped)
                            + ': "\\u0037"}, "other":{"cell[a][b]":"7"}}')
                    start = text.index('"\\u0037"')
                    end = start + len('"\\u0037"')
                    self.assertEqual(value_spans(text), {canonical: (start, end)})
                    rows = [row(text, {canonical: 1}, required=1),
                            row("invalid", {}, index=1, required=1)]
                    original = copy.deepcopy(rows)
                    credit = self.credit(rows)["s0"]
                    fact = credit["facts"][0]
                    self.assertEqual(fact["slot"], canonical)
                    self.assertEqual(fact["literal_span"], [start, end])
                    self.assertEqual(fact["token_indices"], list(range(start, end)))
                    self.assertEqual(credit["partial_token_advantages"],
                                     [0.] * start + [.03125] * (end - start) + [0.] * (len(text) - end))
                    self.assertEqual(rows, original)

    def test_equal_and_conflicting_alias_collisions_are_rejected(self):
        for value in ("7", "999"):
            text = json.dumps({"corrected_facts": {"cell[a][b]": "7", "CELL[ A ][ B ]": value}})
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "colliding normalized"):
                value_spans(text)
            with self.assertRaisesRegex(ValueError, "colliding normalized"):
                self.credit([row(text, {"cell[a][b]": 1}), row("invalid", {}, index=1)])

    def test_all_wrong_slot_stays_zero_despite_other_variance_invalid_and_missing(self):
        text = json.dumps({"corrected_facts": {"e4": "H->C", "other": "7"}})
        missing = json.dumps({"corrected_facts": {"e4": "H->C"}})
        rows = [row(text, {"e4": 0, "other": 1}), row(missing, {"e4": 0}, index=1),
                row("invalid JSON", {}, index=2)]
        out = self.credit(rows)
        e4 = out["s0"]["facts"][0]
        self.assertEqual((e4["slot"], e4["advantage"]), ("e4", 0))
        self.assertTrue(all(out["s0"]["partial_token_advantages"][i] == 0 for i in e4["token_indices"]))
        other = out["s0"]["facts"][1]
        self.assertAlmostEqual(other["group_mean"], 1 / 3)
        self.assertEqual(out["s2"]["mapping_status"], "NO_ADMISSIBLE_FACTS")
        self.assertFalse(any(out["s2"]["partial_token_advantages"]))
        same = [row(text, {"e4": 0, "other": 1}, index=i) for i in range(4)]
        self.assertTrue(all(not any(r["partial_token_advantages"]) for r in self.credit(same).values()))

    def test_crossing_and_special_tokens_are_excluded_without_changing_group_mean(self):
        parts = ['{"corrected_facts":{"a":', '"', '7', '","b":', '"5"', '}}', '']
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: "".join(parts[i] for i in ids))
        text = "".join(parts)
        rows = [row(text, {"a": 1, "b": 0}, tokens=list(range(7))),
                row("invalid", {}, index=1, tokens=[99])]
        original_decode = tokenizer.decode
        tokenizer.decode = lambda ids, **kw: "invalid" if ids == [99] else original_decode(ids, **kw)
        out = self.credit(rows, tokenizer)["s0"]
        self.assertEqual(out["facts"][0]["token_indices"], [1, 2])
        self.assertEqual(out["facts"][1]["token_indices"], [4])
        self.assertEqual(out["partial_token_advantages"][3], 0)  # closing quote crosses into another key
        self.assertEqual(out["partial_token_advantages"][6], 0)  # EOS/zero width

    def test_unstable_prefix_abstains_and_full_decode_tamper_is_error(self):
        text = '{"corrected_facts":{"a":"7"}}'
        rows = [row(text, {"a": 1}, required=1), row(text, {"a": 1}, index=1, required=1)]
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: "?" if len(ids) == 1 else "".join(chr(i) for i in ids))
        out = self.credit(rows, tokenizer)
        self.assertTrue(all(r["mapping_status"] == "UNSTABLE_TOKEN_PREFIX" for r in out.values()))
        self.assertTrue(all(not any(r["partial_token_advantages"]) for r in out.values()))
        with self.assertRaisesRegex(ValueError, "tokens and text differ"):
            token_spans(Characters(), [65], "B")

    def test_receipt_and_group_boundary_guards(self):
        text = '{"corrected_facts":{"a":"7"}}'
        rows = [row(text, {"a": 1}, required=1), row(text, {"a": 0}, index=1, required=1)]
        for key, bad in (("text_sha256", "bad"), ("required_slot_count", 2),
                         ("strict_reward", 1), ("fact_rewards", {"a": True}),
                         ("fact_rewards", {" A ": 1}),
                         ("objective_version", "verified-note-fact-token-credit-v1")):
            changed = copy.deepcopy(rows)
            changed[0][1][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.credit(changed)
        with self.assertRaises(ValueError):
            self.credit(rows[:1])
        with self.assertRaises(ValueError):
            self.credit([rows[0], rows[0]])
        changed = copy.deepcopy(rows)
        changed[0][1]["fact_rewards"] = {"missing": 1}
        with self.assertRaisesRegex(ValueError, "span missing"):
            self.credit(changed)


if __name__ == "__main__":
    unittest.main()
