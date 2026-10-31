"""CPU-authored reference claims test token credit, not pretrained model learning."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import partial_notebook, referenced_notebook as referenced, synthesis_credit as credit
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures
from fine_tuning.tests.test_notebook_rl_updates import Characters


class ReferencedCreditTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        self.texts = [fixtures.response(self.item.slots)] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2
        self.tree = fixtures.tree_from(self.actor, self.verifier, self.texts)
        self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": referenced.VERSION}
        self.refresh()

    def refresh(self):
        self.samples = episode.solution_samples(self.tree)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.index = referenced.public_endpoints(self.samples)
        self.typed = next(row for row in fixtures.note_for(self.tree, self.audit, self.item)["lessons"]
                          if row["relation"] == "CORRECTED")
        self.claim = self.compact(self.typed)
        self.origin = credit.issue_id(self.typed["earlier"]["attempt"], self.typed["earlier"])

    def ref(self, endpoint):
        return next(key for key, value in self.index.items()
                    if all(value[field] == endpoint[field] for field in ("attempt", "kind", "rule", "locus")))

    def compact(self, lesson):
        return {"relation": lesson["relation"], "earlier_ref": self.ref(lesson["earlier"]),
                "later_ref": self.ref(lesson["later"])}

    def note(self, lessons=None):
        return {"coverage": "PARTIAL", "corrected_facts": {}, "lessons": [self.claim] if lessons is None else lessons}

    def score(self, note=None, *, raw=None, finish="stop"):
        text = json.dumps(self.note() if note is None else note, separators=(",", ":")) if raw is None else raw
        return credit.score_note(text, finish, self.tree, self.audit, self.verifier, actor=self.actor)

    def group(self, texts):
        samples, scores = [], {}
        for index, text in enumerate(texts):
            row = self.score(raw=text)
            sample = SimpleNamespace(sample_id=f"reference-{index}", group_id="original-four", kind="DIAG",
                text=text, finish_reason="stop", reward=row["strict_reward"],
                completion_token_ids=[ord(char) for char in text])
            samples.append(sample)
            scores[sample.sample_id] = row
        return samples, scores, credit.group_credit(samples, scores, Characters())

    def test_compact_claim_is_independently_verified_and_strict_grades_original_text(self):
        raw = json.dumps(self.note(), separators=(",", ":"))
        expected = synthesis.grade_note(raw, "stop", self.tree, self.audit, self.verifier,
                                        actor=self.actor, contract=synthesis.BOUNDED_VERSION)
        with patch.object(synthesis, "_lesson", wraps=synthesis._lesson) as checked:
            row = self.score(raw=raw)
        self.assertEqual(row["strict_reward"], expected["reward"])
        self.assertEqual(row["strict_report_sha256"], credit._strict_digest(expected))
        self.assertEqual(row["strict_reward"], 0.)
        self.assertEqual(row["relationship_rewards"][self.origin], 1)
        self.assertEqual(row["atomic_reward"], .0625 / len(row["original_issue_ids"]))
        self.assertEqual(checked.call_args.args[0], referenced.expand_lesson(self.claim, self.index))
        self.assertEqual(row["public_endpoints"], self.index)
        self.assertFalse(row["lesson_semantics_verified"])

    def test_reference_policy_is_explicit_and_legacy_and_partial_receipts_match(self):
        for policy in (None, partial_notebook.VERSION, referenced.VERSION + "-unknown"):
            self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": policy}
            compact = self.score()
            self.assertFalse(any(compact["relationship_rewards"].values()))
            self.assertFalse(any(key in compact for key in credit._REFERENCE_FIELDS))
        raw = json.dumps({"lessons": [self.typed]}, separators=(",", ":"))
        self.tree.notebook["rewarded_notebook"] = {}
        base = self.score(raw=raw)
        self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": partial_notebook.VERSION}
        self.assertEqual(self.score(raw=raw), base)
        self.assertEqual(base["relationship_rewards"][self.origin], 1)
        self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": referenced.VERSION}
        self.assertFalse(any(self.score(raw=raw)["relationship_rewards"].values()))

    def test_false_native_claim_same_identity_and_ancestry_still_fail(self):
        sibling = next(row for row in fixtures.note_for(self.tree, self.audit, self.item)["lessons"]
                       if row["relation"] == "OTHER_BRANCH_COMPARISON")
        mismatch = next(key for key, endpoint in self.index.items()
                        if endpoint["attempt"] == self.typed["later"]["attempt"]
                        and endpoint["kind"] == "TRACE" and endpoint["locus"] != self.typed["later"]["locus"])
        claims = [self.claim | {"relation": "PERSISTED"}, self.claim | {"later_ref": mismatch},
                  self.compact(sibling) | {"relation": "CORRECTED"}]
        for claim in claims:
            with self.subTest(claim=claim):
                self.assertFalse(any(self.score(self.note([claim]))["relationship_rewards"].values()))

    def test_bad_first_identifiable_claim_blocks_later_repair_and_mixed_dialect(self):
        invalid = [self.claim | {"later_ref": "a999:e1"}, self.claim | {"relation": "UNKNOWN"},
                   self.claim | {"later_ref": True}, self.claim | {"extra": "unsupported"}, self.typed]
        for first in invalid:
            with self.subTest(first=first):
                row = self.score(self.note([first, self.claim]))
                self.assertEqual(row["selected_lesson_indices"][self.origin], 0)
                self.assertEqual(row["relationship_rewards"][self.origin], 0)
                self.assertNotIn(self.origin, row["relationship_spans"])
        unknown = self.claim | {"earlier_ref": "unknown"}
        self.assertEqual(self.score(self.note([unknown, self.claim]))["selected_lesson_indices"][self.origin], 1)
        for raw in ("invalid JSON", '{"lessons":[', '{"lessons":[],"lessons":[]}'):
            self.assertEqual(self.score(raw=raw)["atomic_reward"], 0.)
        self.assertEqual(self.score(finish="length")["atomic_reward"], 0.)

    def test_only_original_compact_object_tokens_receive_full_n4_centered_credit(self):
        claim = json.dumps(self.claim, separators=(",", ":")).replace("a1:", "a\\u0031:")
        raw = '{"coverage":"PARTIAL","corrected_facts":{},"lessons":[' + claim + ',' + claim + ']}'
        samples, scores, output = self.group([raw, "{}", "invalid JSON", "{}"])
        first = scores[samples[0].sample_id]
        start, end = first["relationship_spans"][self.origin]
        self.assertEqual(raw[start:end], claim)
        self.assertNotIn("evidence_ranges", raw)
        self.assertEqual(first["selected_lesson_indices"][self.origin], 0)
        entry = output[samples[0].sample_id]["relationships"][0]
        self.assertEqual(entry["token_indices"], list(range(start, end)))
        self.assertEqual((entry["reward"], entry["group_mean"], entry["advantage"]), (1, .25, .75))
        vector = output[samples[0].sample_id]["relationship_token_advantages"]
        self.assertTrue(all(value == 0 for index, value in enumerate(vector) if not start <= index < end))
        self.assertAlmostEqual(sum(vector), .0625 * .75 / len(first["original_issue_ids"]))
        self.assertTrue(credit.has_signal(samples, scores))
        self.assertEqual(output[samples[0].sample_id]["full_group_size"], 4)
        for sample in samples:
            credit.validate_score(scores[sample.sample_id], sample)
        with self.assertRaisesRegex(ValueError, "original complete"):
            credit.group_credit(samples[:2], scores, Characters())
        restored = json.loads(json.dumps(scores, sort_keys=True))
        self.assertEqual(credit.group_credit(samples, restored, Characters()), output)

    def test_receipt_index_hash_source_selection_and_original_spans_are_bound(self):
        samples, scores, _ = self.group([json.dumps(self.note()), "{}", "{}", "{}"])
        original = scores[samples[0].sample_id]
        for name in ("index", "hash", "binding", "selection", "span", "policy"):
            row = copy.deepcopy(original)
            if name == "index":
                row["public_endpoints"][self.claim["earlier_ref"]]["locus"] = "forged"
            elif name == "hash":
                row["public_endpoints_sha256"] = "0" * 64
            elif name == "binding":
                row["public_endpoints_binding_sha256"] = "0" * 64
            elif name == "selection":
                row["selected_lesson_indices"][self.origin] = 1
            elif name == "span":
                row["relationship_spans"][self.origin] = [0, 1]
            else:
                row["notebook_generation_policy"] = partial_notebook.VERSION
            row = credit._seal({key: value for key, value in row.items() if key != "score_sha256"})
            with self.subTest(name=name), self.assertRaises(ValueError):
                credit.validate_score(row, samples[0])

    def test_rehashed_malformed_endpoint_metadata_cannot_change_receipt_types(self):
        samples, scores, _ = self.group([json.dumps(self.note()), "{}", "{}", "{}"])
        original = scores[samples[0].sample_id]
        for change in ({"attempt": True}, {"evidence_ranges": [[True, 2]]},
                       {"evidence_ranges": [[2, 2]]}, {"kind": "OUTPUT"}):
            row = copy.deepcopy(original)
            row["public_endpoints"][self.claim["earlier_ref"]].update(change)
            row.update(credit._reference_metadata(row["public_endpoints"], row["solution_sha256"], row["transcript_sha256"]))
            row = credit._seal({key: value for key, value in row.items() if key != "score_sha256"})
            with self.subTest(change=change), self.assertRaises(ValueError):
                credit.validate_score(row, samples[0])

    def test_fresh_score_regenerates_public_index_instead_of_trusting_saved_metadata(self):
        before = self.score()
        self.tree.notebook["rewarded_notebook"].update(public_endpoints={"forged": self.typed["earlier"]},
                                                       public_endpoints_sha256="0" * 64)
        with patch.object(referenced, "public_endpoints", wraps=referenced.public_endpoints) as rebuilt:
            self.assertEqual(self.score(), before)
        rebuilt.assert_called_once_with(self.samples)

    def test_output_finalization_keeps_answer_correctness_separate(self):
        self.texts[0] = self.texts[0].rsplit("\nFinal answer:", 1)[0]
        self.tree = fixtures.tree_from(self.actor, self.verifier, self.texts)
        self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": referenced.VERSION}
        self.refresh()
        endpoint = {"kind": "OUTPUT", "rule": "FINALIZATION", "locus": "final_answer"}
        claim = {"relation": "FINALIZED", "earlier_ref": self.ref(endpoint | {"attempt": 1}),
                 "later_ref": self.ref(endpoint | {"attempt": 3})}
        row = self.score(self.note([claim]))
        origin = credit.issue_id(1, endpoint)
        self.assertEqual(row["relationship_rewards"][origin], 1)
        self.assertEqual(row["relationship_later_answer_correct"], {origin: False})
        sibling = claim | {"earlier_ref": self.ref(endpoint | {"attempt": 2})}
        self.assertFalse(any(self.score(self.note([sibling]))["relationship_rewards"].values()))


if __name__ == "__main__":
    unittest.main()
