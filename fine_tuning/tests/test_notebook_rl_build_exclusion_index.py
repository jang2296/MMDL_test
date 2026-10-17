"""Explicitly synthetic native Parquet fixtures; never full benchmark approval."""
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fine_tuning.notebook_rl import build_exclusion_index as b
from fine_tuning.notebook_rl import contamination as c


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (12, 15), color).save(stream, format="PNG")
    return stream.getvalue()


class BuildExclusionIndexTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / "hub"
        self.output = self.root / "private"
        self.raw = png("green")
        self.explanation_raw = png("blue")
        self.row = {"id": "authored-only-1", "question": "Question: Café Ω  <image 1>\n",
                    "options": "[' amber  ', 'violet Ω']", "explanation": "Use <image 2>.  Raw text\n",
                    "image_1": {"bytes": self.raw, "path": "authored-1.png"},
                    "image_2": {"bytes": self.explanation_raw, "path": "authored-explanation.png"}}

    def shard(self, rows=None, number=0, **overrides):
        entry = {"dataset_id": "MMMU/MMMU", "revision": c._REVISIONS["MMMU/MMMU"],
                 "configuration": "AuthoredFixture", "split": "dev",
                 "native_path": f"AuthoredFixture/dev-{number:05d}-of-00002.parquet"}
        entry.update(overrides)
        path = (self.cache / ("datasets--" + entry["dataset_id"].replace("/", "--"))
                / "snapshots" / entry["revision"] / entry["native_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [self.row] if rows is None else rows
        pq.write_table(pa.Table.from_pylist(rows), path)
        entry.update(sha256=digest(path.read_bytes()), bytes=path.stat().st_size, records=len(rows))
        return entry, path

    def build(self, entries, **kwargs):
        return b._build(entries, self.cache, self.output,
                        partial_name="explicit-authored-parquet-only", **kwargs)

    def test_preserves_native_text_bytes_and_explanation_image_references(self):
        entry, _ = self.shard()
        result = self.build([entry])
        self.assertEqual(result["status"], "NAMED_PARTIAL_INDEX_READY")
        loaded = c.load_index(self.output / "index.json", partial_name="explicit-authored-parquet-only")
        self.assertFalse(loaded["full_coverage"])
        record = loaded["data"]["records"][0]
        self.assertEqual(record["native_texts"], {k: self.row[k] for k in ("question", "options", "explanation")})
        self.assertEqual(record["texts"], {"question": [self.row["question"]],
                                          "choices": [" amber  ", "violet Ω"],
                                          "explanation": [self.row["explanation"]]})
        self.assertEqual(record["image_fields"], ["image_1", "image_2"])
        self.assertEqual(record["image_sha256"], [digest(self.raw), digest(self.explanation_raw)])
        for raw in (self.raw, self.explanation_raw):
            self.assertEqual((self.output / "images" / digest(raw)).read_bytes(), raw)
            with Image.open(io.BytesIO(raw)) as image:
                self.assertEqual(loaded["images"][digest(raw)], c.image_fingerprints(image))
        self.assertEqual(loaded["image_refs"][digest(self.explanation_raw)][0]["image_field"], "image_2")
        with self.assertRaisesRegex(ValueError, "incomplete index"):
            c.load_index(self.output / "index.json")
        self.assertFalse(loaded["data"]["training_approval"])
        for key in ("ocr_complete", "sscd_complete", "siglip_complete", "calibration_complete"):
            self.assertFalse(loaded["data"][key])

    def test_checkpoint_resume_accumulates_two_shards_and_reuses_fingerprints(self):
        entry1, _ = self.shard()
        entry2, _ = self.shard([{**self.row, "id": "authored-only-2"}], number=1)
        checkpoint = self.build([entry1, entry2], max_files=1)
        self.assertEqual(checkpoint["status"], "CHECKPOINTED")
        self.assertEqual(checkpoint["completed_files"], 1)
        self.assertFalse((self.output / "index.json").exists())
        with patch.object(c, "image_fingerprints", side_effect=AssertionError("already fingerprinted")):
            result = self.build([entry1, entry2], max_files=1)
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["image_count"], 4)
        self.assertEqual(result["unique_images"], 2)
        original = (self.output / "index.json").read_bytes()
        with patch.object(b, "_build_shard", side_effect=AssertionError("restart must reuse shards")):
            self.build([entry1, entry2])
        self.assertEqual((self.output / "index.json").read_bytes(), original)
        loaded = c.load_index(self.output / "index.json", partial_name="explicit-authored-parquet-only")
        self.assertEqual(len(loaded["data"]["coverage"]["files"]), 2)
        self.assertEqual({r["ref"]["file_sha256"] for r in loaded["data"]["records"]},
                         {entry1["sha256"], entry2["sha256"]})

    def test_missing_tampered_native_files_and_wrong_row_count_fail(self):
        entry, path = self.shard()
        with self.assertRaisesRegex(ValueError, "missing native cache file"):
            self.build([{**entry, "native_path": "AuthoredFixture/dev-missing.parquet"}])
        self.assertFalse((self.output / "index.json").exists())
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "bytes/SHA256"):
            self.build([entry])
        entry, _ = self.shard()
        with self.assertRaisesRegex(ValueError, "row count"):
            self.build([{**entry, "records": 2}])

    def test_native_text_and_image_schema_fail_before_final_index(self):
        for changes in ({"options": "not a list"}, {"image_1": None, "image_2": None},
                        {"explanation": 99}, {"question": None}):
            with self.subTest(changes=tuple(changes)):
                entry, _ = self.shard([{**self.row, **changes}])
                with self.assertRaises(ValueError):
                    self.build([entry])
                self.assertFalse((self.output / "index.json").exists())

    def test_raw_cache_and_shard_tampering_fail_on_resume(self):
        entry, _ = self.shard()
        self.build([entry])
        shard = self.output / "shards" / (entry["sha256"] + ".json")
        original = shard.read_bytes()
        changed = json.loads(original)
        changed["payload"]["records"][0]["texts"]["question"] = ["tampered"]
        shard.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "tampered per-file shard"):
            self.build([entry])
        shard.write_bytes(original)
        (self.output / "images" / digest(self.raw)).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "cached raw image changed"):
            self.build([entry])

    def test_vision_native_missing_text_fields_are_explicitly_empty(self):
        entry, _ = self.shard([{"id": "authored-vision-only", "image": {"bytes": self.raw, "path": None}}],
                              dataset_id="MMMU/MMMU_Pro", revision=c._REVISIONS["MMMU/MMMU_Pro"],
                              configuration="vision", split="test", native_path="vision/test-authored.parquet")
        self.build([entry])
        value = json.loads((self.output / "index.json").read_bytes())
        self.assertEqual(value["records"][0]["texts"], {"question": [""], "choices": [], "explanation": []})

    def test_plan_is_read_only_and_counts_missing_native_bytes(self):
        entry, _ = self.shard()
        missing = {**entry, "native_path": "AuthoredFixture/dev-missing.parquet"}
        with patch.object(b, "_inventory", return_value=[entry, missing]):
            result = b.plan(self.cache, self.output)
        self.assertEqual(result["status"], "PLAN_ONLY")
        self.assertEqual(result["missing_download_bytes"], entry["bytes"])
        self.assertFalse(self.output.exists())
        self.assertFalse(result["cached_presence_is_sha_verified"])
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            b.plan(self.cache, b._REPO / "private-index")


if __name__ == "__main__":
    unittest.main()
