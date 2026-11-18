"""The public selection must remain complete, ordered, private and transportable."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from fine_tuning.error_note import v18_recipe as recipe


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = recipe.load_recipe()
        cls.metadata = recipe.load_sidecars(cls.data)

    def test_fixed_selection_and_all_sources(self):
        self.assertEqual(Counter(x["role"] for x in self.data["items"]), recipe.ROLE_COUNTS)
        self.assertEqual(len(self.data["sources"]), 15)
        self.assertEqual(self.data["roles"]["train"]["ids_sha256"], recipe.TRAIN_IDS_SHA256)
        self.assertEqual(Counter(x["source"] for x in self.data["items"]), {
            "hf-mapqa": 754, "hf-ai2d": 183, "hf-chartqa": 419, "hf-ssmr": 449,
            "hf-chebi-role": 396, "hf-gvlqa": 319, "hf-ptbxl": 222,
            "hf-scienceqa": 369, "hf-slake": 210, "hf-geometry3k": 163,
            "hf-deepweeds": 116, "hf-clevr": 32, "hf-plantvillage": 51,
            "hf-nctcrc": 129, "hf-tqa": 44})

    def test_original_map_row_and_turn_are_distinct_from_conversion_index(self):
        item = next(x for x in self.data["items"]
            if x["problem_id"] == "hf-mapqa:map:train:005306:78ca5b79835b31e1")
        self.assertEqual((item["role_index"], item["ordinal"]), (62, 5306))
        self.assertEqual(item["locator"]["row_index"], 4311)
        self.assertEqual(item["locator"]["turn_index"], 5)
        self.assertEqual((item["locator"]["row_group"], item["locator"]["row_offset"]), (43, 11))
        self.assertEqual(self.data["files"][item["locator"]["file_key"]]["sha256"],
            "11d122e0ab84ebb6e1bdf8efc119c67987cd93304200ccfed6b4891ba6fe4b0d")

    def test_slake_locator_is_the_exact_archive_member(self):
        rows = [x for x in self.data["items"] if x["source"] == "hf-slake"]
        self.assertEqual(len(rows), 210)
        self.assertTrue(all(x["locator"]["image_member"].startswith("imgs/") for x in rows))

    def test_public_shards_are_bounded_and_independently_bound(self):
        for path in recipe.DEFAULT_RECIPE.parent.glob("v18_recipe*.json"):
            self.assertLess(path.stat().st_size, 5 * 1024**2)
            recipe._document(path)
        self.assertEqual(recipe.digest(self.data["files"]), self.data["files_sha256"])
        self.assertEqual(len(self.metadata["items"]), 3856)

    def test_ancillary_metadata_preserves_limits_and_not_old_acceptance(self):
        guard = self.metadata["guard"]
        self.assertFalse(guard["historical_guard_acceptance_carried"])
        self.assertEqual([x["problem_id"] for x in guard["items"]],
            [x["problem_id"] for x in self.data["items"] if x["role"] == "train"])
        auxiliary = list(self.metadata["auxiliary"].values())
        self.assertEqual(Counter(x["kind"] for x in auxiliary),
            {"generated_ocr": 224, "graph_rendering_review": 1})
        review = next(x for x in auxiliary if x["kind"] == "graph_rendering_review")
        self.assertFalse(review["human_verified"])
        for value in auxiliary:
            self.assertEqual(recipe.digest(value["metadata"]), value["metadata_sha256"])
        ocr = next(x for x in auxiliary if x["kind"] == "generated_ocr")
        self.assertFalse(ocr["raw_payload_public"])
        self.assertNotIn("text", ocr["metadata"])
        self.assertNotIn("tokens", ocr["metadata"])

    def test_role_order_and_training_flags_cannot_change(self):
        data = copy.deepcopy(self.data)
        data["items"][0], data["items"][1] = data["items"][1], data["items"][0]
        with self.assertRaisesRegex(ValueError, "ROLE_ORDER"):
            recipe.validate_recipe(data)
        data = copy.deepcopy(self.data)
        next(x for x in data["items"] if x["role"] == "monitor")["training_allowed"] = True
        with self.assertRaisesRegex(ValueError, "TRAINING_LEAK"):
            recipe.validate_recipe(data)

    def test_actor_image_path_cannot_escape_output(self):
        data = copy.deepcopy(self.data)
        data["items"][0]["images"][0]["path"] = "../../other-project.png"
        with self.assertRaisesRegex(ValueError, "IMAGE_PATH_ESCAPE"):
            recipe.validate_recipe(data)

    def test_missing_file_and_invalid_integrity_are_rejected(self):
        data = copy.deepcopy(self.data)
        data["items"][0]["locator"]["file_key"] = "missing"
        with self.assertRaisesRegex(ValueError, "REFERENCE_MISSING"):
            recipe.validate_recipe(data)
        data = copy.deepcopy(self.data)
        next(iter(data["files"].values()))["sha256"] = "z" * 64
        data["files_sha256"] = recipe.digest(data["files"])
        with self.assertRaisesRegex(ValueError, "INTEGRITY_MISSING"):
            recipe.validate_recipe(data)

    def test_corrupt_or_escaping_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "part.json"
            payload = {"items": []}
            payload["content_sha256"] = recipe.digest(payload)
            path.write_text(json.dumps(payload))
            ref = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "content_sha256": payload["content_sha256"], "count": 0}
            self.assertEqual(recipe._shards({"item_shards": [ref]}, root), [])
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHARD_DIGEST"):
                recipe._shards({"item_shards": [ref]}, root)
            ref["path"] = "../outside.json"
            with self.assertRaisesRegex(ValueError, "PATH_ESCAPE"):
                recipe._shards({"item_shards": [ref]}, root)

    def test_raw_payload_or_personal_path_cannot_be_published(self):
        for payload in [{"question": "fixture raw text"}, {"gold": "A"},
                        {"path": "/home/private/file"}, {"text": "OCR source text"}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                recipe._public(payload)

    def test_portable_reference_digest_preserves_content_binding(self):
        first = {"source_ref": {"path": "/home/one/file", "sha256": "a" * 64,
            "role": "native", "pointer": "row:7"}}
        moved = copy.deepcopy(first)
        moved["source_ref"]["path"] = "/other-host/file"
        self.assertEqual(recipe.portable_payload(first), recipe.portable_payload(moved))
        moved["source_ref"]["pointer"] = "row:8"
        self.assertNotEqual(recipe.portable_payload(first), recipe.portable_payload(moved))


if __name__ == "__main__":
    unittest.main()
