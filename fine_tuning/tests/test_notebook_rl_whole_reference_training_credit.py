"""Production raw-reference credit on authored fixtures, never model learning evidence."""
import copy
import hashlib
import json
import unittest
from dataclasses import replace

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, losses, notebook_synthesis as synthesis
from fine_tuning.notebook_rl import synthesis_credit as credit, whole_reference_notebook as reference
from fine_tuning.notebook_rl.local_eval import digest
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures
from fine_tuning.tests.test_notebook_rl_updates import Characters
from fine_tuning.tests.test_notebook_rl_whole_reference_notebook import compact


class WholeReferenceTrainingCreditTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        wrong = fixtures.response(self.item.slots)
        self.prepare([wrong] * 4 + [fixtures.response(self.item.slots, answer="accept")] * 2)

    def prepare(self, texts):
        self.tree = fixtures.tree_from(self.actor, self.verifier, texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.typed = fixtures.note_for(self.tree, self.audit, self.item)
        self.note = compact(episode.solution_samples(self.tree), self.typed)
        self.native = next(row for row in self.note["assessments"] if row[3] and row[3][0] == "PERSISTED")
        self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": reference.VERSION}

    def partial(self, *, relation="PERSISTED", selected=True, fact="q0"):
        note = copy.deepcopy(self.note)
        note["corrected_facts"] = {"start": fact}
        for row in note["assessments"]:
            row[3] = [relation, self.native[3][1]] if row[0] == self.native[0] and selected else None
        return note

    def score(self, note=None, *, text=None, finish="stop", index=0):
        raw = json.dumps(self.note if note is None else note, ensure_ascii=False, separators=(",", ":")) if text is None else text
        sample = replace(self.tree.groups[0].samples[0], sample_id=f"raw-{index}", group_id="original-four",
            kind="DIAG", parent_sample_id=None, text=raw, finish_reason=finish, prompt_token_ids=[1, 2],
            completion_token_ids=list(map(ord, raw)), sampler_logprobs=[-.1] * len(raw), result=None, reward=0.)
        groups = [group for group in self.tree.groups if group.kind != "DIAG"]
        existing = [s for group in self.tree.groups if group.kind == "DIAG" for s in group.samples if s.sample_id != sample.sample_id]
        self.tree.groups = [*groups, replace(groups[0], group_id=sample.group_id, kind="DIAG", samples=[*existing, sample],
            usable=bool(existing), reason=None if existing else "FIXTURE_GROUP_BUILDING")]
        row = credit.score_note(raw, finish, self.tree, self.audit, self.verifier, actor=self.actor, sample_id=sample.sample_id)
        sample.reward = row["strict_reward"]
        return sample, row

    def group(self, notes):
        pairs = [self.score(note, index=index) for index, note in enumerate(notes)]
        samples, scores = [pair[0] for pair in pairs], {pair[0].sample_id: pair[1] for pair in pairs}
        return samples, scores, credit.group_credit(samples, scores, Characters())

    def test_canonical_and_raw_scores_have_same_semantics_and_separate_provenance(self):
        sample, raw = self.score()
        canonical_tree = copy.deepcopy(self.tree)
        canonical_tree.notebook = {}
        expanded = reference.expand(sample.text, "stop", episode.solution_samples(self.tree))
        canonical = credit.score_note(expanded["expanded_text"], "stop", canonical_tree,
            self.audit, self.verifier, actor=self.actor)
        for key in ("strict_reward", "atomic_reward", "effective_reward", "fact_rewards", "relationship_rewards",
                    "relationship_credit_types", "required_fact_count", "original_issue_ids", "relationship_relations"):
            self.assertEqual(raw[key], canonical[key], key)
        self.assertEqual(raw["strict_reward"], 1.)
        self.assertEqual(raw["text_sha256"], digest(sample.text))
        self.assertEqual(raw["raw_text_sha256"], hashlib.sha256(sample.text.encode()).hexdigest())
        self.assertEqual(raw["expanded_text_sha256"], digest(expanded["expanded_text"]))
        self.assertNotEqual(raw["text_sha256"], raw["expanded_text_sha256"])
        credit.validate_score(raw, sample)
        self.assertNotIn("diagnostic_only", raw)

    def test_strict_failure_still_has_independently_verified_native_units(self):
        sample, row = self.score(self.partial())
        self.assertEqual(row["strict_reward"], 0.)
        self.assertEqual(row["fact_rewards"], {"start": 1})
        self.assertEqual(sum(row["relationship_rewards"].values()), 1)
        self.assertEqual(len(row["original_issue_ids"]), sum(len(row["issues"]) for row in self.audit["attempts"]))
        self.assertEqual(row["atomic_reward"], .0625 / len(self.item.slots) + .0625 / len(row["original_issue_ids"]))
        credit.validate_score(row, sample)
        self.assertEqual(sum(self.score(self.partial(relation="CORRECTED"))[1]["relationship_rewards"].values()), 0)

    def test_decoder_maps_only_raw_fact_and_decision_literals_including_repeats(self):
        note = self.partial()
        sources = [row for row in note["assessments"] if row[1] == "I"][:2]
        for source in sources:
            source[3] = copy.deepcopy(self.native[3])
        raw = json.dumps(note, separators=(",", ":")).replace('"start":"q0"', '"START":"\\u00710"')
        sample, row = self.score(text=raw)
        start, end = row["fact_spans"]["start"]
        self.assertEqual(raw[start:end], '"\\u00710"')
        _, spans = reference._decode(raw)
        expected = {tuple(spans[("assessments", index, 3)]) for index, source in enumerate(note["assessments"]) if source[3]}
        self.assertEqual({tuple(span) for span in row["relationship_spans"].values()}, expected)
        self.assertEqual(len(expected), 2)
        others = [self.score(text="{}", index=i) for i in range(1, 4)]
        samples = [sample, *(pair[0] for pair in others)]
        scores = {s.sample_id: score for s, score in [(sample, row), *others]}
        result = credit.group_credit(samples, scores, Characters())[sample.sample_id]
        allowed = {i for name in ("facts", "relationships") for entry in result[name] for i in entry["token_indices"]}
        self.assertTrue(any(result["partial_token_advantages"]))
        self.assertTrue(all(value == 0 for i, value in enumerate(result["partial_token_advantages"]) if i not in allowed))

    def test_original_n4_means_and_loss_mask_do_not_recenter(self):
        good, bad, missing = self.partial(), self.partial(relation="CORRECTED", fact="q3"), self.partial(selected=False)
        missing["corrected_facts"] = {}
        samples, scores, result = self.group([good, bad, bad, missing])
        entry = result["raw-0"]["relationships"][0]
        self.assertEqual((entry["group_mean"], entry["advantage"]), (.25, .75))
        self.assertEqual(result["raw-1"]["relationships"][0]["advantage"], -.25)
        expected = .0625 * .75 / len(scores["raw-0"]["original_issue_ids"])
        self.assertAlmostEqual(sum(result["raw-0"]["relationship_token_advantages"]), expected)
        current = torch.zeros((1, len(samples[0].text)), dtype=torch.float64, requires_grad=True)
        vector = torch.tensor([result["raw-0"]["relationship_token_advantages"]], dtype=torch.float64)
        loss = losses.group_loss(current, current.detach(), current.detach(), torch.zeros(1, dtype=torch.float64),
            torch.ones_like(current), torch.tensor([True]), eps_low=.2, eps_high=.2, beta=0., norm_const=4., partial_adv=vector)
        self.assertAlmostEqual(loss.item(), -expected / 4.)
        loss.backward()
        self.assertAlmostEqual(current.grad.sum().item(), -expected / 4.)
        width = max(len(sample.text) for sample in samples)
        masked_current = torch.zeros((4, width), dtype=torch.float64, requires_grad=True)
        partial = torch.zeros_like(masked_current)
        actions = torch.zeros_like(masked_current)
        old = torch.zeros_like(masked_current)
        for i, sample in enumerate(samples):
            partial[i, :len(sample.text)] = torch.tensor(result[sample.sample_id]["relationship_token_advantages"])
            actions[i, :len(sample.text)] = 1.
        old[0] = .1
        keep, _ = losses.mis_mask(old, torch.zeros_like(old), actions)
        self.assertEqual(keep.tolist(), [False, True, True, True])
        masked_loss = losses.group_loss(masked_current, old, torch.zeros_like(old), torch.zeros(4, dtype=torch.float64),
            actions, keep, eps_low=.2, eps_high=.2, beta=0., norm_const=4., partial_adv=partial)
        self.assertAlmostEqual(masked_loss.item(), .0625 * .5 / len(scores["raw-0"]["original_issue_ids"]) / 4.)
        masked_loss.backward()
        self.assertFalse(any(masked_current.grad[0]))
        self.assertGreater(masked_current.grad[1].sum().item(), 0.)
        self.assertTrue(credit.has_signal(samples, scores))
        with self.assertRaisesRegex(ValueError, "original complete"):
            credit.group_credit(samples[:3], scores, Characters())
        same, same_scores, same_result = self.group([good] * 4)
        self.assertFalse(credit.has_signal(same, same_scores))
        self.assertTrue(all(not any(row["partial_token_advantages"]) for row in same_result.values()))

    def test_malformed_unknown_duplicate_and_truncated_responses_get_no_units(self):
        unknown = copy.deepcopy(self.note)
        unknown["assessments"][0][3] = ["PERSISTED", "unknown-ref"]
        duplicate = copy.deepcopy(self.note)
        duplicate["assessments"][1] = copy.deepcopy(duplicate["assessments"][0])
        for text, finish in (("{}", "stop"), ("not json", "stop"), (json.dumps(unknown), "stop"),
                             (json.dumps(duplicate), "stop"), (json.dumps(self.note), "length")):
            sample, row = self.score(text=text, finish=finish)
            self.assertEqual(row["effective_reward"], 0.)
            self.assertEqual(row["fact_spans"], {})
            self.assertEqual(row["relationship_spans"], {})
            credit.validate_score(row, sample)

    def test_raw_tokens_and_source_metadata_are_bound_and_mixed_rows_are_rejected(self):
        sample, row = self.score(self.partial())
        with self.assertRaisesRegex(ValueError, "actual raw sample"):
            credit.score_note(sample.text, "stop", self.tree, self.audit, self.verifier,
                              actor=self.actor, sample_id="missing")
        for field, value in (("text", sample.text + " "), ("completion_token_ids", [7]),
                             ("prompt_token_ids", [9]), ("sample_id", "other"), ("finish_reason", "length")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                credit.validate_score(row, replace(sample, **{field: value}))
        for key, value in (("expanded_text_sha256", "a" * 64), ("notebook_generation_policy", "legacy"),
                           ("public_endpoints", {}), ("diagnostic_only", False)):
            changed = credit._seal({**{k: v for k, v in row.items() if k != "score_sha256"}, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                credit.validate_score(changed, sample)
        changed = copy.deepcopy(row)
        changed["whole_reference_binding"]["original_samples"][0]["finish_reason"] = "length"
        changed = credit._seal({k: v for k, v in changed.items() if k != "score_sha256"})
        with self.assertRaises(ValueError):
            credit.validate_score(changed, sample)
        samples, scores, _ = self.group([self.partial()] * 4)
        changed = copy.deepcopy(scores["raw-1"])
        changed["whole_reference_binding"]["original_samples"][0]["text"] += " "
        changed = credit._seal({k: v for k, v in changed.items() if k != "score_sha256"})
        with self.assertRaises(ValueError):
            credit.group_credit(samples, scores | {"raw-1": changed}, Characters())

    def test_output_denominator_and_presence_credit_stay_separate_from_answer_correctness(self):
        wrong = fixtures.response(self.item.slots)
        missing = wrong.rsplit("\nFinal answer:", 1)[0]
        self.prepare([missing] * 2 + [wrong] * 2 + [fixtures.response(self.item.slots, answer="accept")] * 2)
        note = self.partial(selected=False)
        inventory = reference.public_inventory(episode.solution_samples(self.tree))
        outputs = {row["attempt"]: ref for ref, row in inventory["references"].items() if row["kind"] == "OUTPUT"}
        next(row for row in note["assessments"] if row[0] == outputs[1])[3] = ["FINALIZED", outputs[3]]
        _, row = self.score(note)
        output = [key for key, value in row["relationship_credit_types"].items() if value == credit.OUTPUT_FINALIZATION]
        self.assertEqual(len(output), 2)
        self.assertEqual(row["relationship_rewards"][output[0]], 1)
        self.assertFalse(row["relationship_later_answer_correct"][output[0]])
        self.assertEqual(len(row["original_issue_ids"]), sum(len(row["issues"]) for row in self.audit["attempts"]) + 2)

    def test_diagnostic_receipts_and_unversioned_reference_relationships_are_not_production(self):
        from fine_tuning.notebook_rl import whole_reference_credit as diagnostic

        sample, _ = self.score(self.partial())
        row = diagnostic.check(sample, self.tree, self.audit, self.actor, self.verifier)
        with self.assertRaises(ValueError):
            credit.validate_score(row, sample)
        self.tree.notebook = {}
        legacy = credit.score_note(sample.text, "stop", self.tree, self.audit, self.verifier, actor=self.actor)
        self.assertEqual(legacy["strict_reward"], 0.)
        self.assertFalse(any(legacy["relationship_rewards"].values()))
        self.assertNotIn("whole_reference_binding", legacy)
        self.assertNotIn("notebook_generation_policy", legacy)

    def test_explicit_observed_policy1_bridge_is_diagnostic_and_preserves_original_history(self):
        sample, _ = self.score(self.partial())
        observed = replace(sample, sample_id="observed-policy1", policy_version=1)
        before = copy.deepcopy(self.tree)
        row = credit.score_note(observed.text, observed.finish_reason, self.tree, self.audit, self.verifier,
            actor=self.actor, sample_id=observed.sample_id, observed_sample=observed)
        self.assertEqual(self.tree, before)
        self.assertTrue(row["diagnostic_only"])
        self.assertEqual(sum(row["relationship_rewards"].values()), 1)
        self.assertEqual(row["whole_reference_sample_binding"]["policy_version"], 1)
        with self.assertRaises(ValueError):
            credit.validate_score(row, observed)
        relabeled = credit._seal({key: value for key, value in row.items() if key not in ("score_sha256", "diagnostic_only")})
        with self.assertRaises(ValueError):
            credit.validate_score(relabeled, observed)
        samples = [replace(observed, sample_id=f"diagnostic-{i}") for i in range(4)]
        with self.assertRaises(ValueError):
            credit.group_credit(samples, {sample.sample_id: row for sample in samples}, Characters())
        for changes in ({"policy_version": 0}, {"policy_version": 2}, {"sample_id": sample.sample_id},
                        {"text": "{}"}, {"completion_token_ids": [7]}, {"kind": "T1"}, {"problem_id": "other"}):
            invalid = replace(observed, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                credit.score_note(observed.text, observed.finish_reason, self.tree, self.audit, self.verifier,
                    actor=self.actor, sample_id=invalid.sample_id, observed_sample=invalid)
        self.tree.notebook = {}
        with self.assertRaisesRegex(ValueError, "explicit whole reference"):
            credit.score_note(observed.text, "stop", self.tree, self.audit, self.verifier,
                actor=self.actor, sample_id=observed.sample_id, observed_sample=observed)


if __name__ == "__main__":
    unittest.main()
