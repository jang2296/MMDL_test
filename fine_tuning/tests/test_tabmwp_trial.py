"""Small self-authored assets and in-memory rows; no external dataset fixtures."""

import copy
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning import audit, tabmwp, tabmwp_trial as trial


def row(sid, source, split="train", **extra):
    return {"id": sid, "root_id": sid, "family_id": sid, "split": split,
            "source_id": source, "lineage": ["trace" if source == "trace" else "independent-nvw-v1"],
            "benchmark_derived": False, "training_allowed": True,
            "supervised_target_identifiable": True, "answer_identifiable": True,
            "user_text": "Question: Read the fixture image.", "target_text": '{"answer":"fixture"}',
            "image_paths": ["images/retention.png"], **extra}


def retention_rows():
    rows = [row(f"retention-{i}", "trace") for i in range(360)]
    quotas = {"NVW-K": {"qa": 10, "structure": 5, "grounded": 5},
              "NVW-I": {"qa": 5, "structure": 3, "grounded": 2},
              "NVW-S": {"qa": 5, "structure": 2, "grounded": 3}}
    for track, views in quotas.items():
        for view, count in views.items():
            rows.extend(row(f"retention-{track}-{view}-{i}", "nvw-independent-procedural-v1",
                            nvw_track=track, view=view) for i in range(count))
    return rows


def native_rows():
    variants = {v: {} for v in ("A", "B")}
    for role, count in (("train", 1600), ("dev", 200), ("sealed", 200)):
        for variant in variants:
            rows = []
            for i in range(count):
                native = row(f"native-{role}-{i}", tabmwp.SOURCE_ID,
                             split="holdout" if role == "sealed" else role,
                             lineage=[tabmwp.SOURCE_ID], image_paths=["tables/native.png"],
                             training_allowed=False, native_solution_used=False, teacher_generated=False)
                if variant == "B":
                    native["target_text"] = json.dumps({"facts": {"headers": ["Fixture"], "rows": [["cell"]]},
                                                       "answer": "fixture"})
                rows.append(native)
            variants[variant][role] = rows
    return variants


class TabMWPTrialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.prepared = self.source / "prepared"
        self.images = self.source / "images"
        self.retention = self.root / "historical"
        self.output = self.root / "trial"
        self.prepared.mkdir(parents=True)
        (self.images / "tables").mkdir(parents=True)
        (self.retention / "images").mkdir(parents=True)
        Image.new("RGB", (4, 3), "white").save(self.images / "tables/native.png")
        Image.new("RGB", (4, 3), "black").save(self.retention / "images/retention.png")
        audit.dump(self.prepared / "manifest.json", {"preparation_hash": "fixture-native-preparation"})
        audit.dump(self.source / "source_receipt.json", {"fixture": True})
        (self.source / "README.md").write_text("Own fixture license evidence")
        self.image_receipt = self.images / "receipt_01.json"
        audit.dump(self.image_receipt, {"status": "COMPLETE", "fixture": True})
        self.native = native_rows()
        self.ret = retention_rows()
        self.parent = {"sources": {k: {"source_known": True, "rights_approved": True,
                                       "benchmark_derived": False, "parents": []}
                                   for k in ("trace", "independent-nvw-v1")},
                       "contamination_audit": {"coverage": {"files": [], "record_count": 0}},
                       "files": {"images/retention.png": audit.sha(self.retention / "images/retention.png")}}

    def prepare_fixture(self):
        bound = {str(self.image_receipt): audit.sha(self.image_receipt)}
        with patch.object(trial, "native_bundle", return_value=(self.native, bound, self.image_receipt)), \
                patch.object(trial, "retention_bundle", return_value=(self.ret, self.parent, {})):
            return trial.prepare(self.prepared, self.images, self.retention, self.output)

    def test_pair_contract_split_answers_and_retention_boundaries(self):
        a = self.native["A"]["train"] + self.ret
        b = self.native["B"]["train"] + copy.deepcopy(self.ret)
        ids = [r["id"] for r in self.ret]
        trial.pair_check(a, b, ids)
        for where, field, value in ((0, "split", "dev"), (0, "user_text", "different"),
                                     (0, "teacher_generated", True),
                                     (1600, "target_text", "changed retention")):
            changed = copy.deepcopy(b)
            changed[where][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                trial.pair_check(a, changed, ids)
        changed = copy.deepcopy(b)
        changed[0]["target_text"] = '{"facts":{},"answer":"different"}'
        with self.assertRaises(ValueError):
            trial.pair_check(a, changed, ids)
        with self.assertRaises(ValueError):
            trial.pair_check(a[:-1], b[:-1], ids)

    def test_new_preparation_source_trust_candidate_and_original_isolation(self):
        originals = copy.deepcopy((self.native, self.ret, self.parent))
        manifest = self.prepare_fixture()
        variants, loaded = trial.validate_trial(self.output)
        self.assertEqual(manifest, loaded)
        self.assertEqual((self.native, self.ret, self.parent), originals)
        self.assertEqual(len(variants["A"]), 2000)
        self.assertEqual(manifest["status"], "PREPARED_PENDING_GATES")
        self.assertFalse(manifest["training_allowed"])
        self.assertEqual(manifest["training_contract"]["arm"], "FM")
        self.assertEqual(manifest["training_contract"]["learning_rate"], 1e-5)
        self.assertEqual(manifest["training_contract"]["expected_updates"], 125)
        from fine_tuning.train import order_hash
        consumed = list(manifest["record_ids"])
        random.Random(tabmwp.SEED).shuffle(consumed)
        self.assertEqual(manifest["record_order_sha256"], order_hash(manifest["record_ids"]))
        self.assertEqual(manifest["consumption_order_sha256"], order_hash(consumed))
        self.assertEqual(set(manifest["files"]), {"tables/native.png", "images/retention.png"})
        self.assertEqual([r["id"] for r in variants["A"]], [r["id"] for r in variants["B"]])
        by_id = {r["id"]: r for r in variants["A"]}
        for original in self.ret:
            self.assertEqual(by_id[original["id"]], original)
        registry = trial.read_json(self.output / "source_registry_candidate.json")["sources"]
        self.assertFalse(registry[tabmwp.SOURCE_ID]["rights_approved"])
        self.assertEqual(registry["trace"], self.parent["sources"]["trace"])
        with self.assertRaises(FileExistsError):
            trial.prepare(self.prepared, self.images, self.retention, self.output)
        self.image_receipt.write_text("changed original receipt")
        with self.assertRaises(ValueError):
            trial.validate_trial(self.output)

    def test_mutated_trial_image_and_proposal_cannot_approve(self):
        self.prepare_fixture()
        image = self.output / "tables/native.png"
        original = image.read_bytes()
        image.write_bytes(b"wrong image")
        with self.assertRaises(ValueError):
            trial.validate_trial(self.output)
        image.write_bytes(original)
        with (self.output / "B/proposed.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaises(ValueError):
            trial.validate_trial(self.output)

    def test_closed_retention_selection_and_dev_rejected(self):
        frozen = {"receipt.json": "fixture"}
        audit.dump(self.retention / "receipt.json", {"historical": True})
        frozen["receipt.json"] = audit.sha(self.retention / "receipt.json")
        with patch.object(trial.prepare_training, "closed_pilot", return_value=(
                self.ret, self.parent, {}, frozen, {})):
            selected, _, bound = trial.retention_bundle(self.retention)
            self.assertEqual(Counter(r["source_id"] for r in selected),
                             {"trace": 360, "nvw-independent-procedural-v1": 40})
            self.assertIn(str(self.retention / "images/retention.png"), bound)
            broken = copy.deepcopy(self.ret)
            broken[0]["split"] = "dev"
            with patch.object(trial.prepare_training, "closed_pilot", return_value=(
                    broken, self.parent, {}, frozen, {})), self.assertRaises(ValueError):
                trial.retention_bundle(self.retention)
            (self.retention / "images/retention.png").write_bytes(b"changed historical image")
            with self.assertRaises(ValueError):
                trial.retention_bundle(self.retention)

    def test_cpu_stage_requires_exact_main_review_before_preflight(self):
        self.prepare_fixture()
        reviewed, index, review = [self.root / name for name in ("registry.json", "index.json", "review.json")]
        for path in (reviewed, index, review):
            audit.dump(path, {})
        with patch.object(trial.prepare_training, "approve_delta") as approve:
            with self.assertRaises(ValueError):
                trial.approve_cpu(self.output, reviewed, index, "no-model", review, self.root / "approval")
            approve.assert_not_called()
        self.assertFalse((self.root / "approval").exists())


if __name__ == "__main__":
    unittest.main()
