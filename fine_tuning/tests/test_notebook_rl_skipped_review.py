"""User decision 2026-11-05 opt-ins: recorded no-sample-review policy and empty held-out splits (defaults unchanged)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fine_tuning.notebook_rl import approval as a
from fine_tuning.notebook_rl.dataset import build_splits
from fine_tuning.tests.test_notebook_rl_pool_local import hf_problem


class SkippedReviewAndEmptyHeldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items = {f"src:{i:03d}": {"actor": SimpleNamespace(source="src", image_sha256=(f"img{i}",))}
                      for i in range(30)}
        self.review = {"format": a.AI_LOCAL_REVIEW_FORMAT_SKIPPED, "reviewer_id": "user-decision",
                       "human_verified": False, "labels_modified": False, "reviewed_at_utc": "2026-11-05T00:00:00+00:00",
                       "sampling_seed": 3407, "input_files_sha256": "f" * 64, "source_review": a.SOURCE_REVIEW_SKIPPED,
                       "sources": {}, "leak_warnings": []}

    def rule(self, review: dict, warnings: dict, allowlist: set[str]):
        return a._ai_local_review(review, "user-decision", self.items, warnings, "f" * 64, 3407, allowlist)

    def test_skipped_review_is_explicit_and_refuses_leaks_or_verdicts(self) -> None:
        allowed = set(self.items) - {"src:005"}
        # A leak warning on a non-allowlisted (excluded) item is fine; nothing is blocked or failed.
        self.assertEqual(self.rule(self.review, {"train": [{"problem_id": "src:005", "field": "question"}]}, allowed),
                         ({}, 0, None))
        eligibility = a.ai_local_eligibility({p: "src" for p in self.items}, allowed, {}, None, a.SOURCE_REVIEW_SKIPPED)
        self.assertEqual((eligibility["source_review"], len(eligibility["eligible_ids"])), (a.SOURCE_REVIEW_SKIPPED, 29))
        self.assertNotIn("source_review", a.ai_local_eligibility({p: "src" for p in self.items}, allowed, {}))
        with self.assertRaisesRegex(a._Blocked, "SKIPPED_REVIEW_ALLOWLISTED_LEAK_WARNING"):
            self.rule(self.review, {"train": [{"problem_id": "src:001", "field": "question"}]}, allowed)
        verdict = {"problem_id": "src:000", "image_sha256": ["img0"], "render_ok": True, "leak_free": True,
                   "gold_supported": True, "notes": ""}
        for changes in ({"sources": {"src": [verdict]}}, {"source_review": "SKIPPED"}, {"human_verified": True},
                        {"input_files_sha256": "e" * 64}, {"acceptance_rule": a.ACCEPTANCE_RULE}):
            with self.subTest(changes=changes), self.assertRaisesRegex(a._Blocked, "AI_LOCAL_SKIPPED_REVIEW_BINDING"):
                self.rule(self.review | changes, {}, allowed)
        # The skipped provenance cannot pass as an AI review, nor an AI review as skipped.
        skipped = {"provenance": {"reviewer_id": "user-decision", "kind": a.SOURCE_REVIEW_SKIPPED_KIND,
                                  "method": a.SOURCE_REVIEW_SKIPPED, "reviewed_at_utc": "2026-11-05T00:00:00+00:00",
                                  "human_verified": False, "labels_modified": False}}
        self.assertEqual(a._ai_local_provenance(skipped, True), "user-decision")
        with self.assertRaisesRegex(a._Blocked, "AI_LOCAL_REVIEW_PROVENANCE"):
            a._ai_local_provenance(skipped)

    def test_empty_heldout_needs_the_recorded_opt_in(self) -> None:
        pools = {"train": [hf_problem("hf-fixture-a", "train", 0)], "dev": [], "sealed": [], "style_sealed": []}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "all four nonempty splits required"):
                build_splits(Path(tmp) / "default", pools)
            with self.assertRaisesRegex(ValueError, "all four nonempty splits required"):
                build_splits(Path(tmp) / "no-train", pools | {"train": []}, allow_empty_heldout=True)
            manifest = build_splits(Path(tmp) / "opt-in", pools, allow_empty_heldout=True)
            self.assertIs(manifest["allow_empty_heldout"], True)
            self.assertEqual(manifest["counts"], {"train": 1, "dev": 0, "sealed": 0, "style_sealed": 0})
            on_disk = json.loads((Path(tmp) / "opt-in" / "manifest.yaml").read_text())
            self.assertIs(on_disk["allow_empty_heldout"], True)
            full = {split: [hf_problem("hf-fixture-a", split, i)] for i, split in enumerate(pools)}
            self.assertNotIn("allow_empty_heldout", build_splits(Path(tmp) / "full", full))


if __name__ == "__main__":
    unittest.main()
