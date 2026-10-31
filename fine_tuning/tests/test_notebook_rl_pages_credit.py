"""Authored page reward/token fixtures; these are not real-model learning evidence."""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

from fine_tuning.notebook_rl import synthesis_credit as credit, updates, whole_reference_pages as pages
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests import test_notebook_rl_pages_collection as collection


def page_fixture(testcase, *, partial=False, malformed=False):
    collector = collection.PageCollectionTests()
    fixture = collector.fixture(mutation="malformed" if malformed else None)
    testcase.addCleanup(collector.doCleanups)
    if partial:
        for replies in fixture.engine.replies[4:]:
            for ordinal, raw in enumerate(replies):
                body = json.loads(raw)
                if "corrected_facts" in body:
                    body["corrected_facts"] = {}
                elif ordinal in (1, 3):
                    for row in body["assessments"]:
                        row[3] = None
                replies[ordinal] = json.dumps(body, separators=(",", ":"))
    collector.append(fixture)
    return fixture


class PagesCreditTests(unittest.TestCase):
    def score(self, fixture):
        note = fixture.tree.notebook["rewarded_notebook"]
        return credit.score_page_bundles(fixture.tree, note["synthesis_audit"], fixture.verifier, actor=fixture.actor)

    def test_strict_reward_is_repeated_but_positive_atomic_units_have_one_raw_owner(self):
        fixture = page_fixture(self)
        before = copy.deepcopy(fixture.tree)
        scores = self.score(fixture)
        groups = [group for group in fixture.tree.groups if group.kind == "DIAG"]
        self.assertEqual(fixture.tree, before)
        for ordinal in range(4):
            rows = [scores[group.samples[ordinal].sample_id] for group in groups]
            common = rows[0]["page_binding"]["bundle_score"]
            self.assertTrue(all(row["strict_reward"] == 1 for row in rows))
            self.assertAlmostEqual(sum(row["atomic_reward"] for row in rows), common["atomic_reward"])
            self.assertEqual(sum(len(row["fact_rewards"]) for row in rows), len(common["fact_rewards"]))
            for issue, reward in common["relationship_rewards"].items():
                self.assertEqual(sum(row["relationship_rewards"][issue] for row in rows), reward)
                self.assertEqual(sum(issue in row["relationship_spans"] for row in rows), int(issue in common["relationship_spans"]))
        for group in groups:
            for sample in group.samples:
                row = scores[sample.sample_id]
                credit.validate_score(row, sample)
                self.assertEqual(row["required_fact_count"], len(fixture.item.slots))
                self.assertEqual(row["page_binding"]["page_plan"], fixture.tree.notebook["rewarded_notebook"]["page_plan"])
                for name in ("fact_spans", "relationship_spans"):
                    for start, end in row[name].values():
                        json.loads(sample.text[start:end])

    def test_original_four_centering_and_global_issue_denominator_apply_to_owned_tokens(self):
        fixture = page_fixture(self, partial=True)
        scores = self.score(fixture)
        self.assertEqual(updates.notebook_signal_groups(fixture.tree, scores), 1)
        owners = 0
        for group in fixture.tree.groups:
            if group.kind != "DIAG":
                continue
            result = credit.group_credit(group.samples, scores, fixture.engine.tokenizer)
            for sample in group.samples:
                for unit in result[sample.sample_id]["relationships"]:
                    if unit["reward"] == 1:
                        owners += 1
                        self.assertEqual((unit["group_mean"], unit["advantage"]), (.5, .5))
                        self.assertAlmostEqual(unit["coefficient"] * len(unit["token_indices"]),
                            credit.RELATIONSHIP_WEIGHT * .5 / len(scores[sample.sample_id]["original_issue_ids"]))
                self.assertEqual(result[sample.sample_id]["strict_advantage"], 0.)
        self.assertGreater(owners, 0)
        original = groups = [group for group in fixture.tree.groups if group.kind == "DIAG"]
        self.assertGreater(sum(updates.group_has_signal(group, scores) for group in groups), 1)
        with self.assertRaisesRegex(ValueError, "original complete"):
            credit.group_credit(original[0].samples[:3], scores, fixture.engine.tokenizer)

    def test_one_malformed_page_zeros_the_entire_ordinal_without_salvage(self):
        fixture = page_fixture(self, malformed=True)
        scores = self.score(fixture)
        for group in fixture.tree.groups:
            if group.kind != "DIAG":
                continue
            sample = group.samples[2]
            row = scores[sample.sample_id]
            self.assertEqual((row["strict_reward"], row["atomic_reward"], row["effective_reward"]), (0., 0., 0.))
            self.assertEqual((row["fact_spans"], row["relationship_spans"]), ({}, {}))
            self.assertFalse(any(row["relationship_rewards"].values()))
            credit.validate_score(row, sample)

    def test_resealed_wrong_span_raw_page_order_and_foreign_membership_are_rejected(self):
        fixture = page_fixture(self, partial=True)
        scores = self.score(fixture)
        group = next(group for group in fixture.tree.groups if group.kind == "DIAG" and any(
            scores[sample.sample_id]["relationship_spans"] for sample in group.samples))
        sample = group.samples[0]
        row = scores[sample.sample_id]
        for mutation in ("span", "raw", "ordinal", "membership", "policy", "count"):
            changed = copy.deepcopy(row)
            binding = changed["page_binding"]
            if mutation == "span":
                changed["relationship_spans"][next(iter(changed["relationship_spans"]))] = [0, 1]
            elif mutation == "raw":
                binding["raw_pages"][0]["text"] += " "
            elif mutation == "ordinal":
                binding["raw_pages"][0], binding["raw_pages"][1] = binding["raw_pages"][1], binding["raw_pages"][0]
            elif mutation == "membership":
                binding["page_calls"][0]["sample_ids"][0] = "foreign"
            elif mutation == "policy":
                binding["raw_pages"][0]["policy_version"] = 9
            else:
                changed["required_fact_count"] += 1
            changed["page_binding_sha256"] = credit.digest(binding)
            changed = credit._seal({key: value for key, value in changed.items() if key != "score_sha256"})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                credit.validate_score(changed, sample)
        for changes in ({"completion_token_ids": [99]}, {"prompt_token_ids": [3]}, {"seed": sample.seed + 1}):
            with self.assertRaises(ValueError):
                credit.validate_score(row, replace(sample, **changes))
        changed_tree = copy.deepcopy(fixture.tree)
        changed_tree.notebook["rewarded_notebook"]["page_calls"][1]["sample_ids"].reverse()
        with self.assertRaises(ValueError):
            credit.score_page_bundles(changed_tree, changed_tree.notebook["rewarded_notebook"]["synthesis_audit"],
                                     fixture.verifier, actor=fixture.actor)

    def test_explicit_policy_one_bridge_is_diagnostic_and_never_training_credit(self):
        fixture = page_fixture(self)
        observed = copy.deepcopy([group for group in fixture.tree.groups if group.kind == "DIAG"])
        for index, group in enumerate(observed):
            group.group_id = f"observed-page-{index}"
            for ordinal, sample in enumerate(group.samples):
                sample.sample_id, sample.group_id, sample.policy_version = f"{group.group_id}:{ordinal}", group.group_id, 1
        before = copy.deepcopy(fixture.tree)
        rows = credit.score_page_bundles(fixture.tree, fixture.tree.notebook["rewarded_notebook"]["synthesis_audit"],
            fixture.verifier, actor=fixture.actor, observed_groups=observed)
        self.assertEqual(fixture.tree, before)
        self.assertTrue(all(row["diagnostic_only"] and row["training_allowed"] is False for row in rows.values()))
        for group in observed:
            with self.assertRaises(ValueError):
                credit.group_credit(group.samples, rows, fixture.engine.tokenizer)
            sample = group.samples[0]
            relabeled = credit._seal({key: value for key, value in rows[sample.sample_id].items()
                                      if key not in ("diagnostic_only", "training_allowed", "score_sha256")})
            with self.assertRaises(ValueError):
                credit.validate_score(relabeled, sample)
        observed[0].samples[0].prompt_token_ids = [9]
        with self.assertRaisesRegex(ValueError, "membership"):
            credit.score_page_bundles(fixture.tree, fixture.tree.notebook["rewarded_notebook"]["synthesis_audit"],
                fixture.verifier, actor=fixture.actor, observed_groups=observed)

    def test_page_layout_uses_all_originals_and_pure_public_plan(self):
        fixture = page_fixture(self)
        originals = [sample for group in fixture.tree.groups if group.kind != "DIAG" for sample in group.samples]
        public = [SimpleNamespace(**{key: getattr(sample, key) for key in
            ("text", "sample_id", "parent_sample_id", "finish_reason")}) for sample in originals]
        self.assertEqual(pages.plan(public), pages.plan(originals))
        self.assertEqual(pages.plan(originals)["sha256"], _hash({key: value for key, value in pages.plan(originals).items() if key != "sha256"}))


if __name__ == "__main__":
    unittest.main()
