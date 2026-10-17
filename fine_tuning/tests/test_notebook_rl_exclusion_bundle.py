import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import contamination as c
from fine_tuning.notebook_rl import exclusion_bundle as b
from fine_tuning.notebook_rl.build_exclusion_index import _write_once
from fine_tuning.notebook_rl.exclusion_bundle import cache_payloads, text_entries


class BundleTests(unittest.TestCase):
    def identity(self, **overrides):
        return {"engine_id": "test-fixture-only",
                "producer_sha256": hashlib.sha256(Path(b.__file__).with_name("exclusion_engines.py").read_bytes()).hexdigest(),
                "normalizer_sha256": hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(), **overrides}

    def test_native_field_keys_match_scanner(self):
        ref = {"dataset_id": "fixture", "configuration": "a", "split": "dev", "id": "b"}
        record = {"ref": ref, "texts": {"question": ["  Question: Hi  "], "choices": ["", " Cat "], "explanation": []}}
        actual = text_entries({"records": [record]})
        self.assertEqual(actual, {c._digest(["fixture", "a", "dev", "b", "question", 0]): "hi",
                                  c._digest(["fixture", "a", "dev", "b", "choices", 1]): "cat"})

    def test_cache_requires_exact_payload_binding_and_full_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            identity = self.identity()
            root = Path(temporary) / c._digest(identity)
            (root / "images").mkdir(parents=True)
            _write_once(root / "identity.json", c._json(identity))
            sha = "a" * 64
            payload = {"sha256": sha, "vector": [1., 0.]}
            _write_once(root / "images" / (sha + ".json"), c._json({"source_sha256": sha,
                "source_kind": "image", "engine_sha256": root.name, "payload": payload,
                "payload_sha256": c._digest(payload)}))
            self.assertEqual(cache_payloads(root, {sha}), (identity, {sha: payload}))
            with self.assertRaises(ValueError):
                cache_payloads(root, {sha, "b" * 64})

    def test_cache_rejects_stale_source_even_with_consistent_identity_digest(self):
        for field in ("producer_sha256", "normalizer_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                identity = self.identity(**{field: "0" * 64})
                root = Path(temporary) / c._digest(identity)
                root.mkdir()
                _write_once(root / "identity.json", c._json(identity))
                with self.assertRaisesRegex(ValueError, "producer/normalizer code changed"):
                    cache_payloads(root, set())

    def assemble_fixture(self, base, *, changed):
        """Only detector work is mocked; the assembly publication and receipt checks are real."""
        index = {"sha256": "i" * 64, "images": {"reference": {}}, "data": {"records": []}}
        files = {str(base / "actor_inputs.jsonl"): "a" * 64}
        items = {"fixture": {"lineage": SimpleNamespace(label_provenance="PROGRAM"),
                             "image_sha256": ["input"], "text_sha256": "text"}}
        identities = {name: self.identity(engine_id=engine, revision="fixture-v1", artifact_sha256="a" * 64)
                      for name, engine in (("ocr", "tesseract"), ("sscd", "sscd_disc_mixup"),
                                           ("siglip", "google/siglip-base-patch16-224"))}
        calibration = {"index_sha256": index["sha256"], "protocol_sha256": "p" * 64,
                       "engines": {name: identities[name] for name in ("sscd", "siglip")}}
        receipt = {"status": "EXCLUSION_CHECKED_WITH_LIMITS", "training_allowed": False,
                   "input_files_sha256": c._digest({} if changed else files),
                   "candidates": [], "quarantined_ids": [], "allowlist": ["fixture"], "blockers": []}
        receipt["receipt_sha256"] = c._digest(receipt)
        def payloads(root, keys, *, texts=False):
            return identities[root.name], {key: {"sha256": key} for key in keys}
        with patch.object(b.c, "load_index", return_value=index), \
                patch.object(b, "read_json", side_effect=[calibration, {}]), \
                patch.object(b.c, "_snapshot", return_value=(items, files, {"input": {}})), \
                patch.object(b, "cache_payloads", side_effect=payloads), \
                patch.object(b.c, "_calibration"), patch.object(b.c, "_engine"), \
                patch.object(b.c, "scan", return_value=receipt):
            return b.assemble(base / "index.json", base / "build", base / "calibration.json",
                              {name: base / name for name in identities}, base / "registry.json", base / "output")

    def test_changed_scan_snapshot_publishes_no_success_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "inputs changed between detector assembly and scan"):
                self.assemble_fixture(root, changed=True)
            self.assertFalse((root / "output").exists())

    def test_summary_uses_verified_scan_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self.assemble_fixture(root, changed=False)
            receipt = json.loads((root / "output/receipt_v2.json").read_bytes())
            saved = json.loads((root / "output/summary.json").read_bytes())
            self.assertEqual(summary, saved)
            self.assertEqual(summary["input_files_sha256"], receipt["input_files_sha256"])
            self.assertFalse(summary["training_allowed"])


if __name__ == "__main__":
    unittest.main()
