"""Admission boundaries and deterministic grouped v16 splits, without models."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from fine_tuning.error_note import data_v16 as data
from fine_tuning.notebook_rl import contamination


class DataV16Tests(unittest.TestCase):
    LAYERS4 = ("geometry", "graph", "scene", "chart")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def item(self, i, layer="geometry", **extra):
        return {"problem_id": f"{layer}:{i}", "parent_id": f"{layer}:{i}",
                "group_id": f"{layer}-group-{i}", "layer": layer, "split": "train",
                "image_sha256": hashlib.sha256(f"{layer}-image-{i}".encode()).hexdigest(),
                "pixel_sha256": hashlib.sha256(f"{layer}-pixels-{i}".encode()).hexdigest(),
                "question": "A source question", "target": f"Synthetic fact {layer} {i}.", **extra}

    @staticmethod
    def observation(item, **kwargs):
        return {"target": item["target"]}

    def split(self, items, text="", held=None, **kwargs):
        index = {"text_cache": contamination._text_cache(
            [{"text": text, "ref": "synthetic-exclusion"}] if text else [])}
        with patch.object(data, "observation_row", self.observation):
            return data.admit_and_split(items, index, held or set(), **kwargs)

    def test_pinned_source_mutation_fails(self):
        path = self.root / "source.json"
        path.write_text('{"value": 1}')
        pinned = data.file_sha256(path)
        self.assertEqual(data.read_pinned(path, pinned), {"value": 1})
        path.write_text('{"value": 2}')
        with self.assertRaisesRegex(ValueError, "HASH_MISMATCH"):
            data.read_pinned(path, pinned)

    def test_all_four_layers_have_disjoint_sentinel_and_deterministic_order(self):
        items = [self.item(i, layer) for layer in ("geometry", "graph", "scene", "chart")
                 for i in range(44)]
        quota = dict(train_quota=dict.fromkeys(self.LAYERS4, 32),
                     sentinel_quota=dict.fromkeys(self.LAYERS4, 4), dev_per_layer=4)
        actual, rejected = self.split(items, **quota)
        reversed_actual, _ = self.split(list(reversed(items)), **quota)
        self.assertEqual(actual, reversed_actual)
        self.assertFalse(rejected)
        self.assertEqual(Counter(x["split"] for x in actual),
                         {"train": 128, "sentinel": 16, "dev": 16, "sealed": 16})
        self.assertEqual(len({x["image_sha256"] for x in actual}), len(actual))

    def test_holdout_and_fresh_target_match_quarantine_whole_group(self):
        items = [self.item(i) for i in range(12)]
        items += [self.item(100, group_id=items[0]["group_id"],
                           target="A unique benchmark source text phrase seven words.")]
        items[0]["target"] = "Some other observation remains in this sibling."
        held = {items[1]["image_sha256"]}
        actual, rejected = self.split(items,
            "A unique benchmark source text phrase seven words.", held,
            train_quota={"geometry": 2}, sentinel_quota={"geometry": 4},
            dev_per_layer=1, block_size=2)
        self.assertNotIn(items[0]["group_id"], {x["group_id"] for x in actual})
        self.assertNotIn(items[1]["group_id"], {x["group_id"] for x in actual})
        self.assertIn("NEW_TARGET_TEXT_MATCH", {x["reason"] for x in rejected})
        self.assertIn("EXISTING_HOLDOUT_FAMILY", {x["reason"] for x in rejected})

    QUOTAS = dict(train_quota={"geometry": 48, "graph": 32, "scene": 48},
                  sentinel_quota={"geometry": 6, "graph": 4, "scene": 6})

    def three_layers(self):
        return [self.item(i, layer) for layer, n in (("geometry", 60), ("graph", 40), ("scene", 60))
                for i in range(n)]

    def test_insufficient_layer_raises(self):
        items = [self.item(i) for i in range(35)] + [self.item(i, "graph") for i in range(40)]
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_ELIGIBLE_DATA:geometry"):
            self.split(items, train_quota={"geometry": 32, "graph": 32},
                       sentinel_quota={"geometry": 4, "graph": 4})

    def test_quota_not_interleavable_raises(self):
        sentinel = {"geometry": 6, "graph": 4, "scene": 6}
        for train in ({"geometry": 48, "graph": 30, "scene": 50},   # 30 % 16 != 0
                      {"geometry": 3, "graph": 2},                   # total % block_size != 0
                      {"geometry": 16, "graph": 16, "scene": 15}):
            with self.assertRaisesRegex(ValueError, "QUOTA_NOT_INTERLEAVABLE"):
                self.split([], train_quota=train, sentinel_quota=sentinel
                           if len(train) == 3 else {"geometry": 1, "graph": 1})
        with self.assertRaisesRegex(ValueError, "INVALID_SPLIT_SIZE"):
            self.split([], train_quota={"geometry": 8}, sentinel_quota={"graph": 1})

    def test_interleave_three_layers_48_32_48(self):
        actual, rejected = self.split(self.three_layers(), dev_per_layer=4, **self.QUOTAS)
        self.assertFalse(rejected)
        train = actual[:128]
        self.assertEqual({x["split"] for x in train}, {"train"})
        self.assertNotIn("train", {x["split"] for x in actual[128:]})
        for b in range(16):
            self.assertEqual(Counter(x["layer"] for x in train[b * 8:(b + 1) * 8]),
                             {"geometry": 3, "graph": 2, "scene": 3})
        for layer, count in (("geometry", 48), ("graph", 32), ("scene", 48)):
            pool = [x for x in self.three_layers() if x["layer"] == layer]
            expected = sorted(pool, key=lambda x: data.digest({"seed": 3407, "group_id": x["group_id"]}))
            self.assertEqual([x["problem_id"] for x in train if x["layer"] == layer],
                             [x["problem_id"] for x in expected[:count]])

    def test_split_counts_with_sentinel_quota(self):
        actual, _ = self.split(self.three_layers(), dev_per_layer=4, **self.QUOTAS)
        self.assertEqual(Counter(f"{x['layer']}/{x['split']}" for x in actual), {
            "geometry/train": 48, "geometry/sentinel": 6, "geometry/dev": 4, "geometry/sealed": 2,
            "graph/train": 32, "graph/sentinel": 4, "graph/dev": 4,
            "scene/train": 48, "scene/sentinel": 6, "scene/dev": 4, "scene/sealed": 2})

    def test_unselected_layer_rejected(self):
        items = ([self.item(i, "geometry") for i in range(4)] + [self.item(i, "graph") for i in range(4)]
                 + [self.item(i, "chart") for i in range(3)])
        index_path, sources = self.root / "index.json", self.root / "sources.json"
        index_path.write_text("{}")
        sources.write_text("{}")
        index = {"full_coverage": True, "text_cache": contamination._text_cache([]),
                 "data": {"record_count": 0, "image_count": 0, "source_files": []}}
        with (patch.object(data, "observation_row", self.observation),
              patch.object(data, "source_items", return_value=(items, [])),
              patch.object(data, "load_content_manifest"),
              patch.object(data, "validate_admission_binding"),
              patch.object(contamination, "load_index", return_value=index)):
            summary = data.prepare(parent_path=self.root / "p", parent_sha256="x",
                sources_path=sources, index_path=index_path, index_sha256=data.file_sha256(index_path),
                holdout_paths=[], output=self.root / "out", train_quota={"geometry": 1, "graph": 1},
                sentinel_quota={"geometry": 1, "graph": 1}, dev_per_layer=1, block_size=2)
        self.assertEqual([(x["problem_id"], x["reason"]) for x in summary["rejections"]],
                         [(f"chart:{i}", "LAYER_NOT_SELECTED") for i in range(3)])
        self.assertTrue(summary["l0_ready"])
        self.assertNotIn("chart", {key.split("/")[0] for key in summary["counts"]})
        manifest = json.loads((self.root / "out/content-manifest.json").read_text())
        self.assertEqual([x["layer"] for x in manifest["items"] if x["split"] == "train"], ["geometry", "graph"])
        self.assertEqual(len(manifest["rows"]), 2)
        admission = manifest["admission"]
        self.assertEqual((admission["train_quota"], admission["sentinel_quota"], admission["block_size"]),
                         ({"geometry": 1, "graph": 1}, {"geometry": 1, "graph": 1}, 2))
        self.assertNotIn("train_per_layer", admission)

    def test_catalog_keeps_all_same_image_candidates_and_raw_diagram_only(self):
        problem, logic = self.root / "train.zip", self.root / "logic.zip"
        with zipfile.ZipFile(problem, "w") as archive:
            for i in (7, 42):
                archive.writestr(f"train/{i}/data.json", json.dumps({"choices": ["5", "6"],
                    "answer": "A", "annotat_text": f"Problem {i}", "problem_text": f"Text {i}"}))
                archive.writestr(f"train/{i}/img_diagram.png", b"same source image")
        with zipfile.ZipFile(logic, "w") as archive:
            archive.writestr("logic_forms/diagram_logic_forms_annot.json", json.dumps({
                str(i): {"diagram_logic_forms": ["Equals(LengthOf(Line(A, B)), 5)"],
                         "text_logic_forms": ["never use this"]} for i in (7, 42)}))
        catalog = data.geometry_catalog(problem, logic)
        self.assertEqual(set(catalog), {"7", "42"})
        self.assertEqual(catalog["7"]["image_sha256"], catalog["42"]["image_sha256"])
        self.assertNotIn("text_logic_forms", catalog["7"])

    def test_target_scan_receipt_cannot_be_reused_after_payload_mutation(self):
        item = self.item(1)
        manifest = {"items": [item], "rows": [{"target": "Original observation"}]}
        receipt = {"schema": "v16-data-exclusion-check-v1",
            "status": "EXCLUSION_CHECKED_WITH_LIMITS",
            "payload_sha256": data.digest(manifest), "index_sha256": "a" * 64,
            "detector_sha256": data.file_sha256(Path(contamination.__file__)),
            "data_builder_sha256": data.file_sha256(Path(data.__file__)),
            "new_images": 0, "retained_target_candidates": 0}
        path = self.root / "scan.json"
        path.write_text(json.dumps(receipt))
        manifest["admission"] = {"schema": data.SCHEMA, "index_sha256": "a" * 64,
            "exclusion_receipt": {"path": str(path), "sha256": data.file_sha256(path)}}
        manifest["rows"][0]["target"] = "Different content"
        with self.assertRaisesRegex(ValueError, "ADMISSION_BINDING_MISMATCH"):
            data.validate_admission_binding(manifest)

    def test_partial_historical_detector_is_never_reused(self):
        path = self.root / "contamination/receipt_v2.json"
        path.parent.mkdir()
        receipt = {"index_file_sha256": "a" * 64, "blockers": [],
            "status": "EXCLUSION_CHECKED_WITH_LIMITS", "steps": {
                "sscd": {"status": "PARTIAL_CHECKED"}}, "adjudication": {"human_verified": False}}
        path.write_text(json.dumps(receipt))
        shard = {"root": str(self.root), "exclusion_receipt_file_sha256": data.file_sha256(path)}
        with self.assertRaisesRegex(ValueError, "INCOMPLETE_HISTORICAL_EXCLUSION"):
            data._receipt_checked(shard, "a" * 64)


if __name__ == "__main__":
    unittest.main()
