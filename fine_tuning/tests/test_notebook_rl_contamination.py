"""Authored CPU fixtures: real vault/file scanning, no benchmark/model acquisition."""
import copy
import hashlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, PngImagePlugin

from fine_tuning.notebook_rl import contamination as c
from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, load_protocol
from fine_tuning.notebook_rl.vault import write_split


QUESTION = "Locate the amber triangle beside the seven violet circles"
PRINTED = "Amber violet indigo scarlet silver copper brass nickel platinum"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def image_bytes(seed=0, metadata=False):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, (24, 27, 3), dtype=np.uint8))
    stream = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    if metadata:
        info.add_text("authored", "reencoded fixture")
    image.save(stream, format="PNG", pnginfo=info)
    return stream.getvalue()


def index_fixture(raw=None):
    raw = image_bytes(99) if raw is None else raw
    entry = {"path": "fixture/native.parquet", "sha256": "a" * 64,
             "dataset_id": "MMMU/MMMU", "revision": c._REVISIONS["MMMU/MMMU"],
             "configuration": "AuthoredFixture", "split": "validation", "records": 1}
    ref = {k: entry[k] for k in ("dataset_id", "configuration", "split")}
    ref.update(id="authored-only-reference", file_sha256=entry["sha256"])
    with Image.open(io.BytesIO(raw)) as image:
        fp = c.image_fingerprints(image)
    return {"schema": "notebook-rl-exclusion-index-v2", "purpose": "exclusion_fingerprints_only",
            "training_allowed": False, "status": "COMPLETE", "coverage_verified": True,
            "producer_sha256": sha(Path(c.__file__).read_bytes()),
            "fingerprint_version": c.FINGERPRINT_VERSION, "normalizer_version": c.VERSION,
            "coverage": {"files": [entry], "record_count": 1}, "record_count": 1, "image_count": 1,
            "source_files": {entry["path"]: entry["sha256"]},
            "per_file_record_counts": {entry["path"]: 1},
            "records": [{"ref": ref, "texts": {"question": [QUESTION], "choices": [PRINTED],
                                                "explanation": ["A distinct authored explanatory sentence here"]},
                         "image_sha256": [sha(raw)], "image_fields": ["image_1"]}],
            "images": {sha(raw): fp}}


def prepared(data):
    return {"data": data, "partial_name": "authored-fixture-only"}


class ContaminationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.images = self.base / "images"
        self.images.mkdir()
        self.roots = {}
        self.records = {}
        self.registry = {"sources": {}, "upstream_origin_blocklist": ["blocked-fixture"]}
        for i, split in enumerate(("train", "dev", "sealed", "style_sealed")):
            raw = image_bytes(i)
            relative = f"{split}.png"
            (self.images / relative).write_bytes(raw)
            actor = ActorItem(f"fixture:{split}", f"source-{split}", "fixture", "plain",
                              f"Count visible objects in authored panel {i}.", (), "int",
                              (relative,), (sha(raw),))
            verifier = VerifierItem(actor.problem_id, "int", "53", {}, (), {}, {}, (), (),
                                    (f"Authored label {i} zinc iron cobalt manganese",))
            lineage = Lineage(actor.problem_id, actor.source, "fixture-v1", "INTERNAL_FIXTURE", "PROGRAM",
                              "rendered", "authored-fixture", "fixture-v1", i, "count",
                              sha(f"root-{i}".encode()), split, ())
            self.records[split] = (actor, verifier, lineage)
            self.registry["sources"][actor.source] = {
                "rights_approved": True, "source_known": True, "benchmark_derived": False,
                "model_touched": False, "label_provenance": "PROGRAM", "source_revision": "fixture-v1",
                "license": "INTERNAL_FIXTURE", "upstream_origin": "authored-fixture", "parents": [],
                "upstream_split": "generated"}
        self.data = index_fixture()

    def vaults(self, changes=None):
        changes = changes or {}
        for split, values in self.records.items():
            root = self.base / ("vault-" + split)
            self.roots[split] = root
            write_split(root, *([v] for v in changes.get(split, values)))
        return self.roots

    def run_scan(self, index=None, evidence=None, registry=None, roots=None):
        return c.scan(self.roots if roots is None else roots, self.images,
                      prepared(self.data) if index is None else index,
                      self.registry if registry is None else registry, evidence)

    def evidence(self):
        index = c._prepare(self.data, "authored-fixture-only")
        items, _, images = c._snapshot(self.roots, self.images)
        protocol = load_protocol(Path(c.__file__).parent / "protocol_v1.yaml")["sha256"]
        calibration = {"frozen_before_scan": True, "index_sha256": index["sha256"],
                       "protocol_sha256": protocol, "positive_count": 4, "negative_count": 4,
                       "tpr": 0.75, "fpr": 0.0,
                       "per_engine": {name: {"positive_count": 4, "negative_count": 4,
                                             "tpr": 0.75, "fpr": 0.0, "examples_sha256": "d" * 64}
                                      for name in ("dhash", "tiles", "sscd", "siglip")},
                       "thresholds": {"perceptual_hamming": 3, "sscd_exclude": 0.95,
                                      "sscd_review": 0.75, "siglip_review": 0.9}}
        engines = {"ocr": "tesseract", "sscd": "sscd_disc_mixup", "siglip": "google/siglip-base-fixture"}
        evidence = {"calibration": calibration}
        self.registry["scanner_policy"] = {"status": "FROZEN_BEFORE_INPUT_SCAN", "protocol_sha256": protocol,
                                            "calibration_sha256": c._digest(calibration), "engines": {}}
        for name, engine in engines.items():
            identity = {"engine_id": engine, "revision": "authored-fixture-v1", "artifact_sha256": "e" * 64}
            self.registry["scanner_policy"]["engines"][name] = identity.copy()
            evidence[name] = {**identity, "index_sha256": index["sha256"],
                              "calibration_sha256": c._digest(calibration)}
        evidence["ocr"].update(benchmark={key: {"image_sha256": key, "status": "COMPLETE", "text": PRINTED}
                                           for key in index["images"]}, inputs={})
        def vectors(keys, bindings, vector):
            return {key: {"sha256": bindings[key], "vector": vector.copy()} for key in keys}
        for name in ("sscd", "siglip"):
            evidence[name].update(benchmark_images=vectors(index["images"], {k: k for k in index["images"]}, [1., 0.]),
                                  input_images=vectors(images, {k: k for k in images}, [0., 1.]))
        benchmark_texts = {entry["key"]: sha(c.normalize_text(entry["text"]).encode())
                           for entry in index["text_cache"]["entries"] if c.normalize_text(entry["text"])}
        evidence["siglip"].update(benchmark_texts=vectors(benchmark_texts, benchmark_texts, [1., 0.]),
                                  input_texts=vectors(items, {p: i["text_sha256"] for p, i in items.items()}, [0., 1.]))
        return evidence

    def test_normalizer_image_markers_and_nonempty_exact(self):
        self.assertEqual(c.normalize_text("Question: <image 2> Ａmber <image_7>  triangle"), "amber triangle")
        cache = c._text_cache([{"text": "<image 1>", "key": "blank"}, {"text": QUESTION, "key": "q"}])
        self.assertEqual(c._text_matches("<image>", "question", cache), [])
        self.assertTrue(any(m["reason"] == "EXACT_TEXT" for m in c._text_matches("Question: " + QUESTION, "question", cache)))
        self.assertEqual(c._text_matches("53", "verifier", c._text_cache([{"text": "53", "key": "n"}])), [])

    def test_shingles_five_seven_and_eight_review(self):
        cache = c._text_cache([{"text": PRINTED, "key": "choice"}])
        matches = c._text_matches(PRINTED + " extra", "choices", cache)
        self.assertTrue({"NEAR_TEXT_5GRAM", "NEAR_TEXT_7GRAM"} <= {m["reason"] for m in matches})
        reference = " ".join(f"reference{i}" for i in range(40))
        query = " ".join([*(f"query{i}" for i in range(12)), *(f"reference{i}" for i in range(8)),
                          *(f"tail{i}" for i in range(12))])
        matches = c._text_matches(query, "question", c._text_cache([{"text": reference, "key": "long"}]))
        self.assertTrue(any(m["reason"] == "SHARED_8GRAM_REVIEW" and m["action"] == "REVIEW" for m in matches))

    def test_partial_load_rebuilds_cache_and_rejects_missing_scope(self):
        path = self.base / "index.json"
        path.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "name a partial"):
            c.load_index(path)
        loaded = c.load_index(path, partial_name="fixture-only")
        self.assertFalse(loaded["full_coverage"])
        self.assertTrue(loaded["text_cache"]["inverted"][7])
        self.assertEqual(loaded["file_sha256"], sha(path.read_bytes()))
        self.vaults({"train": (replace(self.records["train"][0], question=QUESTION), *self.records["train"][1:])})
        receipt = self.run_scan(loaded)
        self.assertEqual(receipt["status"], "BLOCKED_NOT_CHECKED")
        self.assertEqual(receipt["candidate_scan_status"], "CANDIDATE_SCAN_COMPLETE_NOT_APPROVAL")
        self.assertIn("fixture:train", receipt["quarantined_ids"])
        self.assertEqual(receipt["allowlist"], [])
        self.assertNotIn(QUESTION, json.dumps(receipt))

    def test_raw_json_wrapper_still_builds_near_cache(self):
        actor, verifier, lineage = self.records["sealed"]
        verifier = replace(verifier, accept={"nested": [PRINTED + " ending"]})
        self.vaults({"sealed": (actor, verifier, lineage)})
        receipt = self.run_scan()
        self.assertTrue(any(m.get("field") == "verifier" and m["reason"] == "NEAR_TEXT_7GRAM"
                            for m in receipt["candidates"]))
        self.assertIn("fixture:sealed", receipt["quarantined_ids"])

    def test_all_split_files_and_private_fields_bound(self):
        self.vaults()
        receipt = self.run_scan()
        self.assertEqual(receipt["input_count"], 4)
        self.assertEqual(len(receipt["input_files"]), 20)
        self.assertEqual(receipt["steps"]["all_splits"]["status"], "COMPLETE")
        for root in self.roots.values():
            self.assertIn(str(root / "verifier_private.jsonl"), receipt["input_files"])
            self.assertIn(str(root / "lineage.jsonl"), receipt["input_files"])
        missing = self.run_scan(roots={"train": self.roots["train"]})
        self.assertIn("ALL_FOUR_SPLITS_REQUIRED", missing["blockers"])
        self.assertFalse(receipt["training_allowed"])
        self.assertFalse(receipt["approval_granted"])

    def test_exact_image_raw_and_reencoded_pixel_sha(self):
        self.data = index_fixture(image_bytes(0, metadata=True))
        self.vaults()
        receipt = self.run_scan()
        self.assertTrue(any(m["problem_id"] == "fixture:train" and m["stage"] == "image_exact"
                            for m in receipt["candidates"]))
        self.assertNotEqual(next(iter(self.data["images"])), self.records["train"][0].image_sha256[0])

    def test_perceptual_crop_tile_inventory_and_bktree_boundaries(self):
        with Image.open(io.BytesIO(image_bytes(0))) as image:
            fp = c.image_fingerprints(image)
        self.assertEqual(set(fp["variants"]), set(c._VARIANTS))
        self.assertEqual(len(fp["variants"]), 15)
        tree = c.BKTree()
        tree.add(0, {"id": "zero"})
        tree.add(0, {"id": "duplicate"})
        tree.add(15, {"id": "four-bits"})
        self.assertEqual({r["id"] for _, r in tree.search(0, 3)}, {"zero", "duplicate"})
        self.assertEqual(len(tree.nodes), 2)
        self.assertTrue(c._low_information("3333333333333333"))
        self.assertFalse(c._low_information("123456789abcdef0"))
        self.data["images"][next(iter(self.data["images"]))]["variants"]["tile3-2-2"] = fp["variants"]["center"]
        self.vaults()
        receipt = self.run_scan()
        self.assertTrue(any(m["problem_id"] == "fixture:train" and m["stage"] == "perceptual"
                            and m["input_variant"] == "center" and m["hamming"] == 0
                            and m["ref"]["variant"] == "tile3-2-2" for m in receipt["candidates"]))

    def test_low_information_hash_is_review_not_automatic_exclusion(self):
        self.vaults()
        sha_input = self.records["train"][0].image_sha256[0]
        white = Image.new("RGB", (24, 27), "white")
        fp = c.image_fingerprints(white)
        for variant in self.data["images"].values():
            variant["variants"] = fp["variants"]
        items, files, images = c._snapshot(self.roots, self.images)
        images[sha_input] = fp
        with patch.object(c, "_snapshot", return_value=(items, files, images)):
            receipt = self.run_scan()
        hits = [m for m in receipt["candidates"] if m["problem_id"] == "fixture:train" and m["stage"] == "perceptual"]
        self.assertTrue(hits)
        self.assertEqual({m["action"] for m in hits}, {"REVIEW"})

    def test_family_transitive_quarantine_not_coarse_task_family(self):
        items = {"a": {"structure_hash": "x", "image_sha256": ["one"]},
                 "b": {"structure_hash": "x", "image_sha256": ["two"]},
                 "c": {"structure_hash": "y", "image_sha256": ["two"]},
                 "d": {"structure_hash": "z", "image_sha256": ["three"]}}
        result = c.family_quarantine(items, [{"problem_id": "a"}])
        self.assertEqual(result["quarantined_ids"], ["a", "b", "c"])
        self.assertEqual(result["allowlist"], ["d"])

    def test_cross_split_root_leak_quarantines_every_member(self):
        actor, verifier, lineage = self.records["dev"]
        self.vaults({"dev": (actor, verifier, replace(lineage, structure_hash=self.records["train"][2].structure_hash))})
        receipt = self.run_scan()
        leaks = {m["problem_id"] for m in receipt["candidates"] if m["reason"] == "INPUT_FAMILY_SPLIT_LEAK"}
        self.assertEqual(leaks, {"fixture:train", "fixture:dev"})

    def test_missing_evidence_never_silently_completes(self):
        self.vaults()
        result = self.run_scan()
        for stage in ("ocr_ngrams", "sscd", "siglip", "perceptual"):
            self.assertEqual(result["steps"][stage]["status"], "BLOCKED_NOT_CHECKED")
        for prefix in ("OCR_PREREQUISITE", "SSCD_PREREQUISITE", "SIGLIP_PREREQUISITE", "CALIBRATION_PREREQUISITE"):
            self.assertTrue(any(b.startswith(prefix) for b in result["blockers"]))

    def test_complete_authored_evidence_does_not_upgrade_partial_index(self):
        self.vaults()
        evidence = self.evidence()
        result = self.run_scan(evidence=evidence)
        self.assertEqual(result["blockers"], ["FULL_BENCHMARK_INDEX_NOT_AVAILABLE"])
        for stage in ("ocr_ngrams", "sscd", "siglip", "perceptual", "frozen_manifest"):
            self.assertEqual(result["steps"][stage]["status"], "COMPLETE")
        self.assertEqual(result["status"], "BLOCKED_NOT_CHECKED")
        self.assertEqual(result["allowlist"], [])

    def test_complete_inventory_metadata_alone_cannot_approve_index(self):
        data = copy.deepcopy(self.data)
        files = []
        for split, count in (("dev", 5), ("validation", 30), ("test", 350)):
            for subject in range(30):
                files.append({"path": f"native/{split}/{subject}.parquet", "sha256": sha(f"{split}-{subject}".encode()),
                              "dataset_id": "MMMU/MMMU", "revision": c._REVISIONS["MMMU/MMMU"],
                              "configuration": str(subject), "split": split, "records": count})
        for config, shards in (("standard10", 3), ("standard4", 3), ("vision", 4)):
            for shard in range(shards):
                count = 1730 // shards + (shard < 1730 % shards)
                files.append({"path": f"pro/{config}/{shard}.parquet", "sha256": sha(f"{config}-{shard}".encode()),
                              "dataset_id": "MMMU/MMMU_Pro", "revision": c._REVISIONS["MMMU/MMMU_Pro"],
                              "configuration": config, "split": "test", "records": count})
        data.update(coverage={"files": files, "record_count": 16740}, record_count=16740, image_count=19041,
                    source_files={f["path"]: f["sha256"] for f in files},
                    per_file_record_counts={f["path"]: f["records"] for f in files})
        self.assertTrue(c._coverage(data)[0])
        with self.assertRaisesRegex(ValueError, "incomplete v2"):
            c._prepare(data, None)
        # Inventories do not substitute for native per-record and per-image fingerprints.

    def test_decision_vocabulary_with_mocked_full_index_is_not_training_approval(self):
        self.vaults()
        evidence = self.evidence()
        fixture = c._prepare(self.data, "authored-fixture-only")
        fixture.update(full_coverage=True, limitations=[])
        # Only the final decision branch is isolated here; this is not a real full scan.
        with patch.object(c, "_prepare", return_value=fixture):
            result = self.run_scan(evidence=evidence)
        self.assertEqual(result["status"], "EXCLUSION_CHECKED_WITH_LIMITS")
        self.assertEqual(len(result["allowlist"]), 4)
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["approval_granted"])
        self.assertTrue(all(step["status"] == "COMPLETE" for step in result["steps"].values()))

    def test_each_engine_sha_coverage_and_calibration_fail_closed(self):
        self.vaults()
        original = self.evidence()
        cases = []
        bad = copy.deepcopy(original)
        next(iter(bad["sscd"]["input_images"].values()))["sha256"] = "f" * 64
        cases.append(("sscd", bad))
        bad = copy.deepcopy(original)
        bad["siglip"]["benchmark_texts"] = {}
        cases.append(("siglip", bad))
        bad = copy.deepcopy(original)
        bad["ocr"]["benchmark"] = {}
        cases.append(("ocr_ngrams", bad))
        bad = copy.deepcopy(original)
        bad["sscd"]["revision"] = "latest"
        cases.append(("sscd", bad))
        bad = copy.deepcopy(original)
        bad["calibration"]["per_engine"]["tiles"]["positive_count"] = 0
        cases.append(("perceptual", bad))
        for stage, evidence in cases:
            with self.subTest(stage=stage):
                result = self.run_scan(evidence=evidence)
                self.assertEqual(result["steps"][stage]["status"], "BLOCKED_NOT_CHECKED")
                self.assertEqual(result["status"], "BLOCKED_NOT_CHECKED")
                self.assertEqual(result["allowlist"], [])

    def test_ocr_program_annotations_and_human_exact_image_binding(self):
        actor, verifier, lineage = self.records["dev"]
        lineage = replace(lineage, label_provenance="HUMAN")
        self.registry["sources"][actor.source]["label_provenance"] = "HUMAN"
        self.vaults({"train": (self.records["train"][0], replace(self.records["train"][1], printed_text=(PRINTED,)),
                                self.records["train"][2]), "dev": (actor, verifier, lineage)})
        evidence = self.evidence()
        key = actor.image_sha256[0]
        evidence["ocr"]["inputs"] = {key: {"image_sha256": key, "status": "COMPLETE", "text": PRINTED}}
        result = self.run_scan(evidence=evidence)
        self.assertEqual(result["steps"]["ocr_ngrams"]["status"], "COMPLETE")
        self.assertEqual({m["problem_id"] for m in result["candidates"] if m["stage"] == "ocr_ngrams"},
                         {"fixture:train", "fixture:dev"})
        evidence["ocr"]["inputs"][key]["image_sha256"] = "0" * 64
        self.assertEqual(self.run_scan(evidence=evidence)["steps"]["ocr_ngrams"]["status"], "BLOCKED_NOT_CHECKED")

    def test_teacher_ocr_engine_and_missing_program_annotations_rejected(self):
        actor, verifier, lineage = self.records["train"]
        self.vaults({"train": (actor, replace(verifier, printed_text=()), lineage)})
        evidence = self.evidence()
        self.assertEqual(self.run_scan(evidence=evidence)["steps"]["ocr_ngrams"]["status"], "BLOCKED_NOT_CHECKED")
        identity = {"engine_id": "upper-model", "revision": "pinned", "artifact_sha256": "e" * 64}
        self.registry["scanner_policy"]["engines"]["ocr"] = identity
        evidence["ocr"].update(identity)
        self.assertTrue(any("upper-model" in b for b in self.run_scan(evidence=evidence)["blockers"]))

    def test_sscd_thresholds_and_siglip_cross_modal_candidates(self):
        self.vaults()
        evidence = self.evidence()
        keys = sorted(evidence["sscd"]["input_images"])
        for key, cosine in zip(keys, (0.95, 0.75, 0.74, 0.0), strict=True):
            evidence["sscd"]["input_images"][key]["vector"] = [cosine, math.sqrt(1 - cosine * cosine)]
        for vector in evidence["siglip"]["benchmark_texts"].values():
            vector["vector"] = [0., 1.]
        result = self.run_scan(evidence=evidence)
        actions = {m["action"] for m in result["candidates"] if m["stage"] == "sscd"}
        self.assertEqual(actions, {"REVIEW", "QUARANTINE"})
        self.assertEqual(sum(m["stage"] == "sscd" for m in result["candidates"]), 2)
        self.assertTrue(any(m["reason"] == "SIGLIP_IMAGE_TO_TEXT" for m in result["candidates"]))
        self.assertFalse(any(m["reason"] == "SIGLIP_IMAGE_TO_IMAGE" for m in result["candidates"]))

    def test_cosine_rejects_nonfinite_zero_ragged_and_bad_dimensions(self):
        self.assertAlmostEqual(c.cosine_similarities([[3., 4.]], [[3., 4.]])[0, 0], 1.)
        for q, r in (([[0, 0]], [[1, 0]]), ([[float("nan"), 1]], [[1, 0]]),
                     ([[1, 2, 3]], [[1, 0]]), ([[1], [1, 2]], [[1, 0]])):
            with self.subTest(q=q), self.assertRaises(ValueError):
                c.cosine_similarities(q, r)

    def test_source_rights_ancestry_origin_and_unknown_prerequisites(self):
        self.vaults()
        source = self.registry["sources"]["source-train"]
        for change, reason in (({"rights_approved": False}, "SOURCE_RIGHTS_OR_PROVENANCE_REJECTED"),
                               ({"model_touched": True}, "SOURCE_RIGHTS_OR_PROVENANCE_REJECTED"),
                               ({"upstream_origin": "blocked-fixture"}, "BLOCKLISTED_UPSTREAM_ORIGIN")):
            registry = copy.deepcopy(self.registry)
            registry["sources"]["source-train"].update(change)
            self.assertTrue(any(m["reason"] == reason for m in self.run_scan(registry=registry)["candidates"]))
        source["parents"] = ["missing-parent"]
        self.assertTrue(any(b.startswith("SOURCE_PREREQUISITE") for b in self.run_scan()["blockers"]))
        source["parents"] = ["source-train"]
        self.assertTrue(any("cycle" in b for b in self.run_scan()["blockers"]))
        source["parents"] = []
        source["source_revision"] = "main"
        self.assertTrue(any("unpinned source" in b for b in self.run_scan()["blockers"]))

    def test_index_provenance_reverse_refs_and_all_variants_required(self):
        mutations = [lambda d: d.update(producer_sha256="0" * 64),
                     lambda d: d["coverage"]["files"][0].update(revision="main"),
                     lambda d: d["records"][0].update(image_fields=[]),
                     lambda d: d["records"][0]["ref"].update(id=""),
                     lambda d: d["images"][next(iter(d["images"]))]["variants"].pop("center"),
                     lambda d: d.update(image_count=2),
                     lambda d: d.update(source_files={})]
        for mutate in mutations:
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, TypeError)):
                c._prepare(data, "authored-fixture-only")

    def test_symlink_index_image_and_tampered_vault_fail_closed(self):
        self.vaults()
        path = self.base / "index.json"
        path.write_text(json.dumps(self.data))
        link = self.base / "index-link.json"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            c.load_index(link, partial_name="fixture")
        image = self.images / "train.png"
        original = self.images / "original.png"
        image.rename(original)
        image.symlink_to(original)
        self.assertEqual(self.run_scan()["candidate_scan_status"], "BLOCKED_NOT_CHECKED")
        image.unlink()
        original.rename(image)
        private = self.roots["train"] / "verifier_private.jsonl"
        private.write_bytes(private.read_bytes() + b" ")
        self.assertEqual(self.run_scan()["candidate_scan_status"], "BLOCKED_NOT_CHECKED")

    def test_loaded_index_and_input_mutation_are_detected(self):
        self.vaults()
        path = self.base / "index.json"
        path.write_text(json.dumps(self.data))
        loaded = c.load_index(path, partial_name="fixture")
        path.write_text(json.dumps(self.data) + " ")
        self.assertTrue(any("changed since" in b for b in self.run_scan(index=loaded)["blockers"]))
        evidence = self.evidence()
        original = c.embed_scan
        def mutate(*args):
            result = original(*args)
            private = self.roots["train"] / "verifier_private.jsonl"
            if not private.read_bytes().endswith(b" "):
                private.write_bytes(private.read_bytes() + b" ")
            return result
        with patch.object(c, "embed_scan", side_effect=mutate):
            result = self.run_scan(evidence=evidence)
        self.assertTrue(any("changed during" in b for b in result["blockers"]))
        self.assertEqual(result["steps"]["frozen_manifest"]["status"], "BLOCKED_NOT_CHECKED")
        self.assertEqual(result["allowlist"], [])

    def test_vault_snapshot_binds_bytes_before_and_after_loaders(self):
        self.vaults()
        original = c.vault.read_lineage
        def mutate(root):
            values = original(root)
            private = root / "verifier_private.jsonl"
            private.write_bytes(private.read_bytes() + b" ")
            return values
        with patch.object(c.vault, "read_lineage", side_effect=mutate):
            result = self.run_scan()
        self.assertTrue(any("vault changed while loading" in b for b in result["blockers"]))
        self.assertEqual(result["status"], "BLOCKED_NOT_CHECKED")
        self.assertEqual(result["allowlist"], [])

    def test_index_duplicate_json_keys_and_boolean_count_rejected(self):
        path = self.base / "duplicate.json"
        path.write_bytes(b'{"status":"COMPLETE","status":"COMPLETE"}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            c.load_index(path, partial_name="fixture")
        self.data["record_count"] = True
        with self.assertRaisesRegex(ValueError, "inconsistent coverage"):
            c._prepare(self.data, "fixture")

    def test_receipt_hash_write_once_private_and_no_gold_leak(self):
        self.vaults()
        result = self.run_scan()
        body = {k: v for k, v in result.items() if k != "receipt_sha256"}
        self.assertEqual(result["receipt_sha256"], c._digest(body))
        path = self.base / "receipt.json"
        written = c.write_receipt(path, result)
        self.assertEqual(written["sha256"], sha(path.read_bytes()))
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn('"gold"', path.read_text())
        with self.assertRaises(FileExistsError):
            c.write_receipt(path, result)
        result["allowlist"] = ["forged"]
        with self.assertRaisesRegex(ValueError, "payload hash"):
            c.write_receipt(self.base / "forged.json", result)

    def test_historical_index_named_read_only_partial_and_near_cache(self):
        data = copy.deepcopy(self.data)
        for key in ("schema", "records", "images"):
            data.pop(key)
        data.update(text_sha256=[sha(b""), sha(c.normalize_text(QUESTION).encode())],
                    pixels_sha256=["b" * 64], image_bytes_sha256=["c" * 64], dhash=["123456789abcdef0"],
                    long_text_shingles=[sorted(c._grams(PRINTED, 7))])
        path = self.base / "historical.json"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "historical/incomplete"):
            c.load_index(path)
        loaded = c.load_index(path, partial_name="historical-fixture")
        self.assertNotIn(sha(b""), loaded["text_cache"]["legacy_exact"])
        self.assertTrue(c._text_matches(PRINTED, "choices", loaded["text_cache"]))
        self.vaults()
        before = path.read_bytes()
        result = self.run_scan(index=loaded)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(result["status"], "BLOCKED_NOT_CHECKED")
        self.assertTrue(any("Historical index" in value for value in result["limitations"]))


if __name__ == "__main__":
    unittest.main()
