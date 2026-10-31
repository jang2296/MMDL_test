"""Authored public-pair claims test credit bookkeeping, not model learning."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import paired_notebook as paired, partial_notebook, referenced_notebook as referenced
from fine_tuning.notebook_rl import synthesis_credit as credit
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures
from fine_tuning.tests.test_notebook_rl_updates import Characters


class PairedCreditTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        self.texts = [fixtures.response(self.item.slots)] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2
        self.tree = fixtures.tree_from(self.actor, self.verifier, self.texts)
        self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": paired.VERSION}
        self.refresh()

    def refresh(self):
        self.samples = episode.solution_samples(self.tree)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.index = paired.public_pairs(self.samples)
        self.typed = next(row for row in fixtures.note_for(self.tree, self.audit, self.item)["lessons"]
                          if row["relation"] == "CORRECTED")
        self.claim = self.compact(self.typed)
        self.origin = credit.issue_id(self.typed["earlier"]["attempt"], self.typed["earlier"])

    def compact(self, lesson):
        fields = ("attempt", "kind", "rule", "locus")
        pair_id = next(key for key, pair in self.index.items()
                       if all(pair[side][field] == lesson[side][field]
                              for side in ("earlier", "later") for field in fields))
        return {"pair_id": pair_id, "relation": lesson["relation"]}

    def note(self, lessons=None):
        return {"coverage": "PARTIAL", "lessons": [self.claim] if lessons is None else lessons}

    def score(self, note=None, *, raw=None, finish="stop"):
        text = json.dumps(self.note() if note is None else note, separators=(",", ":")) if raw is None else raw
        return credit.score_note(text, finish, self.tree, self.audit, self.verifier, actor=self.actor)

    def group(self, texts):
        samples, scores = [], {}
        for index, text in enumerate(texts):
            row = self.score(raw=text)
            sample = SimpleNamespace(sample_id=f"paired-{index}", group_id="original-four", kind="DIAG",
                text=text, finish_reason="stop", reward=row["strict_reward"],
                completion_token_ids=[ord(char) for char in text])
            samples.append(sample)
            scores[sample.sample_id] = row
        return samples, scores, credit.group_credit(samples, scores, Characters())

    def test_correct_claim_expands_only_bookkeeping_and_keeps_missing_facts_zero(self):
        raw = json.dumps(self.note(), separators=(",", ":"))
        strict = synthesis.grade_note(raw, "stop", self.tree, self.audit, self.verifier,
                                      actor=self.actor, contract=synthesis.BOUNDED_VERSION)
        with patch.object(synthesis, "_lesson", wraps=synthesis._lesson) as checked:
            row = self.score(raw=raw)
        self.assertEqual(checked.call_args.args[0], paired.expand_lesson(self.claim, self.index))
        self.assertEqual(row["strict_report_sha256"], credit._strict_digest(strict))
        self.assertEqual(row["strict_reward"], 0.)
        self.assertEqual(row["fact_rewards"], {})
        self.assertEqual(row["fact_spans"], {})
        self.assertEqual(row["required_fact_count"], len(self.item.slots))
        self.assertEqual(row["relationship_rewards"][self.origin], 1)
        self.assertEqual(row["atomic_reward"], credit.RELATIONSHIP_WEIGHT / len(row["original_issue_ids"]))
        self.assertEqual(row["public_pairs"], self.index)
        self.assertFalse(row["lesson_semantics_verified"])
        self.assertFalse(row["model_claims_used_for_routing"])

    def test_public_pair_inventory_is_independent_of_private_truth_and_sample_outcomes(self):
        samples = copy.deepcopy(self.samples)
        for sample in samples:
            sample.result, sample.reward = None, 99.
        with patch.object(self.verifier, "grade", side_effect=AssertionError("private grade")), \
                patch.object(synthesis, "audit_episode", side_effect=AssertionError("private audit")):
            self.assertEqual(paired.public_pairs(samples), self.index)

    def test_new_dialect_requires_explicit_policy_and_legacy_receipts_stay_equal(self):
        for policy in (None, partial_notebook.VERSION, referenced.VERSION, paired.VERSION + "-unknown"):
            self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": policy}
            row = self.score()
            self.assertFalse(any(row["relationship_rewards"].values()))
            self.assertFalse(any(key in row for key in credit._PAIRED_FIELDS[1:]))
        raw = json.dumps({"lessons": [self.typed]}, separators=(",", ":"))
        self.tree.notebook["rewarded_notebook"] = {}
        legacy = self.score(raw=raw)
        self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": partial_notebook.VERSION}
        self.assertEqual(self.score(raw=raw), legacy)
        self.assertEqual(legacy["relationship_rewards"][self.origin], 1)
        self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": referenced.VERSION}
        endpoints = referenced.public_endpoints(self.samples)
        refs = {side: next(key for key, endpoint in endpoints.items()
                          if all(endpoint[field] == self.typed[side][field]
                                 for field in ("attempt", "kind", "rule", "locus")))
                for side in ("earlier", "later")}
        reference_raw = json.dumps({"coverage": "PARTIAL", "corrected_facts": {}, "lessons": [{
            "relation": "CORRECTED", "earlier_ref": refs["earlier"], "later_ref": refs["later"]}]})
        reference = self.score(raw=reference_raw)
        self.tree.notebook["rewarded_notebook"]["public_pairs"] = {"ignored": {}}
        self.assertEqual(self.score(raw=reference_raw), reference)
        self.assertEqual(reference["relationship_rewards"][self.origin], 1)

    def test_wrong_relation_unknown_membership_and_nonissue_pairs_earn_zero(self):
        unknown = self.claim | {"pair_id": "unknown"}
        nonissue = next(key for key, pair in self.index.items() if pair["earlier"]["kind"] == "OBSERVATION")
        for claim in (self.claim | {"relation": "PERSISTED"}, self.claim | {"relation": "FINALIZED"},
                      unknown, self.claim | {"pair_id": nonissue}):
            with self.subTest(claim=claim):
                row = self.score(self.note([claim]))
                self.assertFalse(any(row["relationship_rewards"].values()))
        self.assertEqual(self.score(self.note([unknown]))["selected_lesson_indices"], {})

    def test_persisted_native_error_is_verified_without_becoming_a_correction(self):
        pair_id = next(key for key, pair in self.index.items()
                       if pair["earlier"] == self.index[self.claim["pair_id"]]["earlier"]
                       and pair["later"]["attempt"] == 3)
        claim = {"pair_id": pair_id, "relation": "PERSISTED"}
        row = self.score(self.note([claim]))
        self.assertEqual(row["relationship_rewards"][self.origin], 1)
        self.assertEqual(row["relationship_credit_types"][self.origin], credit.NATIVE_ERROR_RELATIONSHIP)
        self.assertEqual(row["relationship_later_answer_correct"], {})
        self.assertFalse(any(self.score(self.note([claim | {"relation": "CORRECTED"}]))["relationship_rewards"].values()))

    def test_first_identifiable_claim_blocks_later_guesses_across_pair_ids(self):
        origin = self.index[self.claim["pair_id"]]["earlier"]
        alternate = next(key for key, pair in self.index.items()
                         if pair["earlier"] == origin and key != self.claim["pair_id"])
        bad = [self.claim | {"relation": "UNKNOWN"}, self.claim | {"relation": True},
               self.claim | {"extra": "unsupported"}, self.typed,
               self.claim | {"pair_id": alternate, "relation": "FINALIZED"}]
        for first in bad:
            with self.subTest(first=first):
                row = self.score(self.note([first, self.claim]))
                self.assertEqual(row["selected_lesson_indices"][self.origin], 0)
                self.assertEqual(row["relationship_rewards"][self.origin], 0)
        row = self.score(self.note([self.claim | {"pair_id": "unknown"}, self.claim]))
        self.assertEqual(row["selected_lesson_indices"][self.origin], 1)
        self.assertEqual(row["relationship_rewards"][self.origin], 1)
        row = self.score(self.note([self.claim, self.claim | {"relation": "PERSISTED"}]))
        self.assertEqual(row["selected_lesson_indices"][self.origin], 0)
        self.assertEqual(row["relationship_rewards"][self.origin], 1)

    def test_exact_paired_envelope_rejects_mixed_roots_and_partial_json(self):
        notes = [self.note() | {"corrected_facts": {"start": "q0"}}, self.note() | {"coverage": "COMPLETE"},
                 {"lessons": [self.claim]}, self.note() | {"lessons": "invalid"}]
        for note in notes:
            with self.subTest(note=note):
                row = self.score(note)
                self.assertEqual(row["atomic_reward"], 0.)
                self.assertEqual(row["selected_lesson_indices"], {})
        for raw in ("invalid JSON", '{"coverage":"PARTIAL","lessons":[',
                    '{"coverage":"PARTIAL","lessons":[],"lessons":[]}'):
            self.assertEqual(self.score(raw=raw)["atomic_reward"], 0.)
        self.assertEqual(self.score(finish="length")["atomic_reward"], 0.)

    def test_only_original_compact_spans_receive_full_n4_fixed_denominator_credit(self):
        claim = json.dumps(self.claim, separators=(",", ":")).replace("attempt1", "attempt\\u0031", 1)
        raw = '{"coverage":"PARTIAL","lessons":[' + claim + ',' + claim + ']}'
        samples, scores, output = self.group([raw, "{}", "invalid JSON", "{}"])
        first = scores[samples[0].sample_id]
        start, end = first["relationship_spans"][self.origin]
        self.assertEqual(raw[start:end], claim)
        self.assertNotIn("evidence_ranges", raw)
        self.assertGreater(len(first["original_issue_ids"]), 1)
        entry = output[samples[0].sample_id]["relationships"][0]
        self.assertEqual(entry["token_indices"], list(range(start, end)))
        self.assertEqual((entry["reward"], entry["group_mean"], entry["advantage"]), (1, .25, .75))
        vector = output[samples[0].sample_id]["relationship_token_advantages"]
        self.assertTrue(all(value == 0 for index, value in enumerate(vector) if not start <= index < end))
        self.assertAlmostEqual(sum(vector), credit.RELATIONSHIP_WEIGHT * .75 / len(first["original_issue_ids"]))
        self.assertTrue(all(value == 0 for value in output[samples[0].sample_id]["fact_token_advantages"]))
        self.assertTrue(credit.has_signal(samples, scores))
        restored = json.loads(json.dumps(scores, sort_keys=True))
        self.assertEqual(credit.group_credit(samples, restored, Characters()), output)
        with self.assertRaisesRegex(ValueError, "original complete"):
            credit.group_credit(samples[:2], scores, Characters())

    def test_receipt_metadata_selection_spans_and_policy_tampering_are_rejected(self):
        samples, scores, _ = self.group([json.dumps(self.note()), "{}", "{}", "{}"])
        original = scores[samples[0].sample_id]
        for name in ("index", "hash", "binding", "selection", "span", "policy", "mixed"):
            row = copy.deepcopy(original)
            if name == "index":
                row["public_pairs"][self.claim["pair_id"]]["earlier"]["locus"] = "forged"
            elif name == "hash":
                row["public_pairs_sha256"] = "0" * 64
            elif name == "binding":
                row["public_pairs_binding_sha256"] = "0" * 64
            elif name == "selection":
                row["selected_lesson_indices"][self.origin] = 1
            elif name == "span":
                row["relationship_spans"][self.origin] = [0, 1]
            elif name == "policy":
                row["notebook_generation_policy"] = referenced.VERSION
            else:
                row["public_endpoints"] = {}
            row = credit._seal({key: value for key, value in row.items() if key != "score_sha256"})
            with self.subTest(name=name), self.assertRaises(ValueError):
                credit.validate_score(row, samples[0])

    def test_rehashed_malformed_pair_endpoint_metadata_is_rejected(self):
        samples, scores, _ = self.group([json.dumps(self.note()), "{}", "{}", "{}"])
        original = scores[samples[0].sample_id]
        for change in ({"attempt": True}, {"evidence_ranges": [[True, 2]]},
                       {"evidence_ranges": [[2, 2]]}, {"kind": "OUTPUT"}, {"extra": "invented"}):
            row = copy.deepcopy(original)
            row["public_pairs"][self.claim["pair_id"]]["earlier"].update(change)
            row.update(credit._paired_metadata(row["public_pairs"], row["solution_sha256"], row["transcript_sha256"]))
            row = credit._seal({key: value for key, value in row.items() if key != "score_sha256"})
            with self.subTest(change=change), self.assertRaises(ValueError):
                credit.validate_score(row, samples[0])

    def test_fresh_score_rebuilds_pairs_instead_of_trusting_saved_inventory(self):
        before = self.score()
        self.tree.notebook["rewarded_notebook"].update(public_pairs={"forged": self.typed},
                                                      public_pairs_sha256="0" * 64)
        with patch.object(paired, "public_pairs", wraps=paired.public_pairs) as rebuilt:
            self.assertEqual(self.score(), before)
        rebuilt.assert_called_once_with(self.samples)

    def test_output_finalization_is_separate_from_native_and_answer_correctness(self):
        self.texts[0] = self.texts[0].rsplit("\nFinal answer:", 1)[0]
        self.tree = fixtures.tree_from(self.actor, self.verifier, self.texts)
        self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": paired.VERSION}
        self.refresh()
        pair_id = next(key for key, pair in self.index.items()
                       if pair["earlier"]["attempt"] == 1 and pair["later"]["attempt"] == 3
                       and pair["earlier"]["kind"] == "OUTPUT")
        claim = {"pair_id": pair_id, "relation": "FINALIZED"}
        row = self.score(self.note([claim]))
        origin = credit.issue_id(1, {"kind": "OUTPUT", "rule": "FINALIZATION", "locus": "final_answer"})
        self.assertEqual(row["relationship_rewards"][origin], 1)
        self.assertEqual(row["relationship_later_answer_correct"], {origin: False})
        self.assertEqual(row["relationship_credit_types"][origin], credit.OUTPUT_FINALIZATION)
        self.assertFalse(any(value for key, value in row["relationship_rewards"].items() if key != origin))
        self.assertFalse(any(self.score(self.note([claim | {"relation": "CORRECTED"}]))["relationship_rewards"].values()))


if __name__ == "__main__":
    unittest.main()
