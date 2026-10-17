"""Authored receipt/review fixtures; no actual similarity verdicts are authored."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl import adjudication as a
from fine_tuning.notebook_rl import contamination as c


class AdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.items = {pid: {"structure_hash": family, "image_sha256": [c._digest(pid)]}
                      for pid, family in (("a", "family-ab"), ("b", "family-ab"), ("c", "family-c"), ("d", "family-d"))}
        self.candidates = [{"problem_id": "a", "stage": "perceptual", "reason": "DHASH_CROP_TILE_NEAR",
                            "action": "QUARANTINE", "hamming": 0, "ref": {"id": "authored-fixture"}},
                           {"problem_id": "c", "stage": "siglip", "reason": "SIGLIP_IMAGE_TO_IMAGE",
                            "action": "REVIEW", "cosine": 0.8, "ref": "authored-fixture-only"}]
        self.raw = {"version": c.VERSION, "status": "EXCLUSION_CHECKED_WITH_LIMITS", "training_allowed": False,
                    "approval_granted": False, "blockers": [], "partial_name": None, "input_count": len(self.items),
                    "index_sha256": "1" * 64, "input_files_sha256": "2" * 64,
                    "steps": {step: {"status": "COMPLETE", "candidate_count": 0} for step in c._STEPS},
                    "candidates": self.candidates, "limitations": ["Authored unit fixture, not a real benchmark scan"]}
        self.evidence = self.root / "authored-review-evidence.txt"
        self.evidence.write_text("Authored unit-test evidence; no actual image judgment.")
        self.refresh_raw()

    def descriptor(self, path):
        return {"path": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return self.descriptor(path)

    def refresh_raw(self):
        family = c.family_quarantine(self.items, self.raw["candidates"])
        self.raw.update(allowlist=family["allowlist"], quarantined_ids=family["quarantined_ids"],
                        quarantined_families=family["families"])
        self.raw["receipt_sha256"] = c._digest({key: value for key, value in self.raw.items() if key != "receipt_sha256"})
        self.raw_descriptor = self.write("authored-raw.json", self.raw)

    def review(self, verdicts=None):
        verdicts = verdicts or {}
        return {"format": a.FORMAT, "raw_receipt_sha256": self.raw_descriptor["file_sha256"],
                "index_sha256": self.raw["index_sha256"], "input_files_sha256": self.raw["input_files_sha256"],
                "provenance": {"reviewer_id": "authored-unit-test-not-actual-ai-review",
                    "kind": "AI_DIRECT_SIMILARITY_REVIEW", "human_verified": False,
                    "reviewed_at_utc": "2026-10-17T00:00:00Z"},
                "decisions": [{"candidate_id": cid, "verdict": verdicts.get(candidate["problem_id"], "DISTINCT"),
                    "reason": "Authored test decision, not a real semantic finding.",
                    "evidence": [self.descriptor(self.evidence)]}
                    for cid, candidate in {a.candidate_id(candidate): candidate for candidate in self.raw["candidates"]}.items()]}

    def adjudicate(self, review=None):
        return a.apply(self.raw_descriptor, self.write("authored-review.json", review or self.review()), self.items)

    def test_normal_rederive_preserves_original_warnings_and_immutable_fields(self):
        reviewed = self.adjudicate()
        self.assertEqual(reviewed["candidates"], self.raw["candidates"])
        self.assertEqual(reviewed["steps"], self.raw["steps"])
        self.assertEqual(reviewed["limitations"], self.raw["limitations"])
        self.assertEqual(reviewed["allowlist"], sorted(self.items))
        self.assertEqual(a.validate(reviewed, self.items), [])
        self.assertEqual(a.validate(self.raw, self.items), self.raw["candidates"])
        self.assertFalse(reviewed["adjudication"]["human_verified"])
        self.assertFalse(reviewed["training_allowed"])
        files = {}
        self.assertEqual(a.validate(reviewed, self.items, files=files), [])
        self.assertEqual(set(files), {self.raw_descriptor["path"], str(self.root / "authored-review.json"),
                                     str(self.evidence), str(Path(a.__file__).absolute())})
        with self.assertRaisesRegex(ValueError, "repeated file binding"):
            a.validate(reviewed, self.items, files={str(self.evidence): "0" * 64})

    def test_missing_extra_and_repeated_decisions_reject(self):
        for mode in ("missing", "extra", "repeated"):
            review = self.review()
            if mode == "missing":
                review["decisions"].pop()
            else:
                extra = copy.deepcopy(review["decisions"][0])
                if mode == "extra":
                    extra["candidate_id"] = "0" * 64
                review["decisions"].append(extra)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.adjudicate(review)

    def test_identical_raw_warnings_share_one_decision_but_both_stay_in_receipt(self):
        self.raw["candidates"].append(copy.deepcopy(self.raw["candidates"][0]))
        self.refresh_raw()
        reviewed = self.adjudicate()
        self.assertEqual(len(reviewed["candidates"]), 3)
        self.assertEqual(reviewed["adjudication"]["unique_candidate_count"], 2)
        self.assertEqual(a.validate(reviewed, self.items), [])

    def test_duplicate_blacklists_complete_family_and_unresolved_is_separate(self):
        reviewed = self.adjudicate(self.review({"a": "DUPLICATE", "c": "UNRESOLVED"}))
        self.assertEqual(reviewed["allowlist"], ["d"])
        self.assertEqual(reviewed["adjudication"]["blacklisted_problem_ids"], ["a", "b"])
        self.assertEqual(reviewed["adjudication"]["unresolved_problem_ids"], ["c"])
        self.assertEqual(reviewed["adjudication"]["counts"], {"DISTINCT": 0, "DUPLICATE": 1, "UNRESOLVED": 1})
        self.assertEqual(a.validate(reviewed, self.items), self.raw["candidates"])

    def test_exact_image_and_source_or_split_failures_can_never_be_released(self):
        for stage, reason in (("image_exact", "EXACT_RAW_OR_PIXEL_IMAGE"),
                              ("source_rights", "SOURCE_RIGHTS_OR_PROVENANCE_REJECTED"),
                              ("family", "INPUT_FAMILY_SPLIT_LEAK")):
            self.raw["candidates"][0].update(stage=stage, reason=reason)
            self.refresh_raw()
            reviewed = self.adjudicate()
            self.assertEqual(reviewed["quarantined_ids"], ["a", "b"])
            self.assertEqual([row["problem_id"] for row in a.validate(reviewed, self.items)], ["a"])

    def test_tampered_receipt_raw_review_evidence_or_module_rejects(self):
        reviewed = self.adjudicate()
        for key, value in (("candidates", []), ("allowlist", []), ("index_sha256", "0" * 64),
                           ("limitations", []), ("steps", {})):
            changed = copy.deepcopy(reviewed)
            changed[key] = value
            changed["receipt_sha256"] = c._digest({k: v for k, v in changed.items() if k != "receipt_sha256"})
            with self.subTest(field=key), self.assertRaises(ValueError):
                a.validate(changed, self.items)
        changed = copy.deepcopy(reviewed)
        changed["adjudication"]["module_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            a.validate(changed, self.items)
        self.evidence.write_text("Changed evidence")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            a.validate(reviewed, self.items)
        self.evidence.write_text("Authored unit-test evidence; no actual image judgment.")
        Path(reviewed["adjudication"]["review"]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            a.validate(reviewed, self.items)
        reviewed = self.adjudicate()
        Path(self.raw_descriptor["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            a.validate(reviewed, self.items)

    def test_bad_evidence_provenance_reason_and_binding_reject(self):
        for change in ("evidence", "path", "human", "timestamp", "reason", "binding"):
            review = self.review()
            if change == "evidence":
                review["decisions"][0]["evidence"][0]["file_sha256"] = "0" * 64
            elif change == "path":
                review["decisions"][0]["evidence"][0]["path"] = "relative.txt"
            elif change == "human":
                review["provenance"]["human_verified"] = True
            elif change == "timestamp":
                review["provenance"]["reviewed_at_utc"] = "2026-10-17T00:00:00"
            elif change == "reason":
                review["decisions"][0]["reason"] = " "
            else:
                review["input_files_sha256"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.adjudicate(review)

    def test_incomplete_scans_cannot_be_cured_by_review(self):
        for field, value in (("status", "BLOCKED_NOT_CHECKED"), ("blockers", ["OCR_MISSING"]),
                             ("partial_name", "historical-only")):
            original = copy.deepcopy(self.raw)
            self.raw[field] = value
            self.refresh_raw()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "complete original"):
                self.adjudicate()
            self.raw = original
            self.refresh_raw()
        self.raw["steps"]["sscd"]["status"] = "BLOCKED_NOT_CHECKED"
        self.refresh_raw()
        with self.assertRaisesRegex(ValueError, "complete original"):
            self.adjudicate()

    def test_changed_item_family_or_image_inventory_rejects(self):
        reviewed = self.adjudicate()
        items = copy.deepcopy(self.items)
        items["d"]["image_sha256"] = ["0" * 64]
        with self.assertRaises(ValueError):
            a.validate(reviewed, items)
        with patch.object(a, "_RELEASABLE", set()), self.assertRaises(ValueError):
            a.validate(reviewed, self.items)


if __name__ == "__main__":
    unittest.main()
