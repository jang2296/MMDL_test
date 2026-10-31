"""Authored raw-credit diagnostics; no fixture is evidence of model learning."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import whole_reference_credit as credit
from fine_tuning.notebook_rl import whole_reference_notebook as reference
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures
from fine_tuning.tests.test_notebook_rl_synthesis_output import output_lesson
from fine_tuning.tests.test_notebook_rl_updates import Characters


def compact(tree, audit, item):
    """Translate authored fixture judgments; this helper never supplies model input."""
    samples = episode.solution_samples(tree)
    inventory = reference.public_inventory(samples)
    keys = {(row["attempt"], row["kind"], row["rule"], row["locus"]): ref
            for ref, row in inventory["references"].items() if row["kind"] != "ABSENCE"}

    def ref_for(endpoint):
        return keys[tuple(endpoint[key] for key in ("attempt", "kind", "rule", "locus"))]

    rows = {ref: [ref, "N", [], None] for ref in inventory["assessment_refs"]}
    for attempt in audit["attempts"]:
        for field, flag in (("issues", "I"), ("unverified", "U")):
            for claim in attempt[field]:
                ref = ref_for(claim | {"attempt": attempt["attempt"]})
                rows[ref][1] = "IU" if rows[ref][1] == "I" and flag == "U" else flag
                if flag == "U":
                    rows[ref][2] = [keys[(attempt["attempt"], "TRACE", blocked["rule"], blocked["locus"])]
                                   for blocked in claim["blocked_by"]]
    note = fixtures.note_for(tree, audit, item)
    for lesson in note["lessons"]:
        ref = ref_for(lesson["earlier"])
        later = (inventory["absence_refs"][str(lesson["earlier"]["attempt"])]
                 if lesson["relation"] == "NOT_REVISITED" else ref_for(lesson["later"]))
        rows[ref][3] = [lesson["relation"], later]
    return {"corrected_facts": copy.deepcopy(item.slots),
            "attempts": [[row["attempt"], row["outcome"], row["answer"]] for row in audit["attempts"]],
            "assessments": list(rows.values()), "success_attempt": audit["success_attempt"],
            "corrected_answer": audit["attempts"][audit["success_attempt"] - 1]["answer"]}


class WholeReferenceCreditTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        texts = [fixtures.response(self.item.slots)] * 4 + [fixtures.response(self.item.slots, answer="accept")] * 2
        self.prepare(texts)

    def prepare(self, texts):
        self.tree = fixtures.tree_from(self.actor, self.verifier, texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.note = compact(self.tree, self.audit, self.item)
        self.native = next(row for row in self.note["assessments"] if row[3] and row[3][0] == "PERSISTED")

    def sample(self, note, index=0, *, text=None, finish="stop"):
        raw = json.dumps(note, ensure_ascii=False, separators=(",", ":")) if text is None else text
        return SimpleNamespace(sample_id=f"raw-{index}", group_id="original-four", kind="DIAG",
            text=raw, finish_reason=finish, completion_token_ids=[ord(char) for char in raw], reward=0.)

    def check(self, note=None, *, sample=None, audit=None):
        return credit.check(sample or self.sample(self.note if note is None else note),
                            self.tree, self.audit if audit is None else audit, self.actor, self.verifier)

    def group(self, notes, *, tokenizer=None):
        samples = [self.sample(note, index) for index, note in enumerate(notes)]
        rows = {sample.sample_id: self.check(sample=sample) for sample in samples}
        return samples, rows, credit.group_credit(samples, rows, tokenizer or Characters())

    def partial(self, *, relation="PERSISTED", fact="q0", selected=True):
        note = copy.deepcopy(self.note)
        note["corrected_facts"] = {"start": fact}
        for row in note["assessments"]:
            row[3] = [relation, self.native[3][1]] if row[0] == self.native[0] and selected else None
        return note

    def test_full_rendering_strict_and_independent_partial_preserve_original_issues(self):
        full = self.check()
        self.assertEqual(full["strict_reward"], 1.)
        partial = self.check(self.partial())
        self.assertEqual(partial["strict_reward"], 0.)
        self.assertEqual(partial["fact_rewards"], {"start": 1})
        self.assertEqual(partial["native_verified_count"], 1)
        self.assertEqual(partial["output_verified_count"], 0)
        self.assertEqual(len(partial["original_issue_ids"]), sum(len(row["issues"]) for row in self.audit["attempts"]))
        self.assertEqual(partial["original_issue_ids"], full["original_issue_ids"])
        self.assertAlmostEqual(partial["diagnostic_atomic_reward"], .0625 / len(self.item.slots)
                               + .0625 / len(partial["original_issue_ids"]))
        self.assertTrue(partial["program_rendered_only"])
        self.assertNotIn("objective_version", partial)
        self.assertNotIn("effective_reward", partial)

    def test_relation_false_for_retained_error_and_sibling_never_becomes_repair(self):
        wrong = self.check(self.partial(relation="CORRECTED"))
        self.assertEqual(wrong["native_verified_count"], 0)
        self.assertTrue(wrong["relationship_spans"])
        sibling = copy.deepcopy(self.partial())
        source = next(row for row in sibling["assessments"] if row[0] == self.native[0])
        destination = next(ref for ref, row in wrong["inventory"]["references"].items()
            if row["attempt"] == 2 and row["kind"] == "TRACE" and row["rule"] == "DFA_TRANSITION")
        source[3] = ["PERSISTED", destination]
        self.assertEqual(self.check(sibling)["native_verified_count"], 0)

    def test_fresh_audit_and_raw_sample_bindings_are_required(self):
        audit = copy.deepcopy(self.audit)
        audit["attempts"][0]["issues"] = []
        with self.assertRaisesRegex(ValueError, "fresh verification"):
            self.check(audit=audit)
        samples, rows, _ = self.group([self.partial()] * 4)
        for field, value in (("text", samples[0].text + " "), ("completion_token_ids", [7]), ("finish_reason", "length")):
            sample = SimpleNamespace(**(vars(samples[0]) | {field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                credit.validate_score(rows[samples[0].sample_id], sample)
        for field in ("fact_spans", "relationship_spans"):
            mutated = copy.deepcopy(rows[samples[0].sample_id])
            mutated[field][next(iter(mutated[field]))] = [0, 1]
            mutated = credit._seal({key: value for key, value in mutated.items() if key != "score_sha256"})
            with self.subTest(field=field), self.assertRaises(ValueError):
                credit.validate_score(mutated, samples[0])

    def test_four_original_samples_center_each_unit_before_any_masking(self):
        good, bad = self.partial(), self.partial(relation="CORRECTED", fact="q3")
        missing = self.partial(selected=False)
        missing["corrected_facts"] = {}
        samples, rows, out = self.group([good, bad, bad, missing])
        first, second = out[samples[0].sample_id], out[samples[1].sample_id]
        self.assertEqual((first["relationships"][0]["group_mean"], first["relationships"][0]["advantage"]), (.25, .75))
        self.assertEqual(second["relationships"][0]["advantage"], -.25)
        self.assertTrue(all(value <= 0 for value in second["native_token_advantages"]))
        self.assertAlmostEqual(sum(first["native_token_advantages"]), .0625 * .75 / len(rows[samples[0].sample_id]["original_issue_ids"]))
        self.assertEqual(out[samples[3].sample_id]["relationships"], [])
        self.assertEqual(out[samples[3].sample_id]["partial_token_advantages"], [0.] * len(samples[3].text))
        self.assertEqual(first["native_reward_signal_count"], 1)
        self.assertTrue(first["pre_mis"])
        self.assertFalse(first["strict_credit_applied"])
        self.assertFalse(first["gradient_measured"])
        with self.assertRaisesRegex(ValueError, "original complete"):
            credit.group_credit(samples[:3], rows, Characters())
        with self.assertRaisesRegex(ValueError, "original complete"):
            credit.group_credit(samples, {key: value for key, value in rows.items() if key != samples[2].sample_id}, Characters())

    def test_uniform_four_have_no_token_signal_even_when_all_correct(self):
        _, _, out = self.group([self.partial()] * 4)
        for row in out.values():
            self.assertFalse(any(row["partial_token_advantages"]))
            self.assertFalse(row["native_signal"])
            self.assertEqual(row["native_reward_signal_count"], 0)
            self.assertEqual(row["optimizer_updates"], 0)

    def test_unicode_escaped_facts_and_repeated_decisions_map_only_raw_literals(self):
        note = copy.deepcopy(self.note)
        note["corrected_facts"] = {"한글": "q0", "START": "q0"}
        sources = [row for row in note["assessments"] if row[1] == "I"][:2]
        for source in sources:
            source[3] = copy.deepcopy(self.native[3])
        raw = json.dumps(note, ensure_ascii=False, separators=(",", ":")).replace('"START":"q0"', '"START":"\\u00710"')
        sample = self.sample(note, text=raw)
        row = self.check(sample=sample)
        start, end = row["fact_spans"]["start"]
        self.assertEqual(raw[start:end], '"\\u00710"')
        decisions = credit._raw_decisions(raw)
        repeats = {}
        for _, (literal, span) in decisions.items():
            repeats.setdefault(tuple(literal), []).append(span)
        self.assertTrue(any(len(spans) > 1 for spans in repeats.values()))
        self.assertTrue(all(len({tuple(span) for span in spans}) == len(spans) for spans in repeats.values()))
        for origin, span in row["relationship_spans"].items():
            self.assertEqual(span, decisions[row["selected_refs"][origin]][1])
        others = [self.sample({}, index=i, text="{}") for i in range(1, 4)]
        samples = [sample, *others]
        rows = {s.sample_id: self.check(sample=s) for s in samples}
        with patch.object(credit.existing, "group_credit", side_effect=AssertionError("synthetic action")):
            result = credit.group_credit(samples, rows, Characters())[sample.sample_id]
        allowed = {index for kind in ("facts", "relationships") for entry in result[kind] for index in entry["token_indices"]}
        self.assertTrue(all(value == 0 for index, value in enumerate(result["partial_token_advantages"]) if index not in allowed))
        self.assertNotEqual(row["expanded_text_sha256"], row["text_sha256"])

    def test_truncation_missing_assessments_and_duplicate_refs_fabricate_no_units(self):
        missing = copy.deepcopy(self.note)
        missing["assessments"].pop()
        duplicate = copy.deepcopy(self.note)
        duplicate["assessments"][1] = copy.deepcopy(duplicate["assessments"][0])
        cases = [self.sample(self.note, finish="length"), self.sample(missing), self.sample(duplicate)]
        for sample in cases:
            row = self.check(sample=sample)
            self.assertFalse(row["expansion_captured"])
            self.assertEqual(row["fact_rewards"], {})
            self.assertEqual(row["relationship_spans"], {})
            self.assertFalse(any(row["relationship_rewards"].values()))
            self.assertEqual(row["strict_reward"], 0.)
            credit.validate_score(row, sample)

    def test_output_finalization_is_separate_from_native_and_answer_correctness(self):
        wrong = fixtures.response(self.item.slots)
        missing = wrong.rsplit("\nFinal answer:", 1)[0]
        self.prepare([missing] * 2 + [wrong] * 2 + [fixtures.response(self.item.slots, answer="accept")] * 2)
        note = self.partial(selected=False)
        inventory = reference.public_inventory(episode.solution_samples(self.tree))
        refs = {row["attempt"]: ref for ref, row in inventory["references"].items() if row["kind"] == "OUTPUT"}
        for row in note["assessments"]:
            if row[0] == refs[1]:
                row[3] = ["FINALIZED", refs[3]]
        row = self.check(note)
        origin = next(key for key, value in row["relationship_rewards"].items() if value)
        self.assertEqual(row["output_verified_count"], 1)
        self.assertEqual(row["native_verified_count"], 0)
        self.assertFalse(row["relationship_later_answer_correct"][origin])
        self.assertTrue(synthesis._lesson(output_lesson(episode.solution_samples(self.tree)),
                                         episode.solution_samples(self.tree), self.audit, bounded=True)["verified"])
        other = copy.deepcopy(note)
        for assessment in other["assessments"]:
            assessment[3] = None
        _, _, out = self.group([note, other, other, other])
        first = out["raw-0"]
        self.assertFalse(first["native_signal"])
        self.assertTrue(first["output_signal"])
        self.assertEqual(first["native_reward_signal_count"], 0)
        self.assertEqual(first["output_reward_signal_count"], 1)

    def test_unstable_prefix_and_tokens_crossing_unit_boundaries_are_excluded(self):
        unstable = SimpleNamespace(decode=lambda ids, **kw: "?" if len(ids) == 1 else "".join(chr(i) for i in ids))
        _, _, out = self.group([self.partial(), self.partial(selected=False), self.partial(selected=False), self.partial(selected=False)],
                               tokenizer=unstable)
        self.assertTrue(all(row["mapping_status"] == "UNSTABLE_TOKEN_PREFIX" for row in out.values()))
        self.assertTrue(all(not any(row["partial_token_advantages"]) for row in out.values()))
        notes = [self.partial()] + [self.partial(selected=False)] * 3
        samples = [self.sample(note, i) for i, note in enumerate(notes)]
        rows = {s.sample_id: self.check(sample=s) for s in samples}
        span = next(iter(rows["raw-0"]["relationship_spans"].values()))
        text = samples[0].text
        pieces = [text[:span[0] + 1], text[span[0] + 1:span[1] - 1], text[span[1] - 1:], ""]
        samples[0].completion_token_ids = [1000000 + i for i in range(4)]
        rows["raw-0"] = self.check(sample=samples[0])
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: "".join(pieces[i - 1000000] if i >= 1000000 else chr(i) for i in ids))
        first = credit.group_credit(samples, rows, tokenizer)["raw-0"]
        self.assertEqual(first["relationships"][0]["token_indices"], [1])
        self.assertEqual(first["native_token_advantages"][0], 0.)
        self.assertEqual(first["native_token_advantages"][2:], [0., 0.])
        samples[0].completion_token_ids = [9]
        rows["raw-0"] = self.check(sample=samples[0])
        with self.assertRaisesRegex(ValueError, "tokens and text differ"):
            credit.group_credit(samples, rows, Characters())


if __name__ == "__main__":
    unittest.main()
