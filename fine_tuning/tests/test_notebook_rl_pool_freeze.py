"""pool_freeze on authored shards: hf_pool builds + authored reviewed receipts. No real rows, benchmark data or GPU.

The union-receipt tests scan authored shards against the authored full-coverage index fixture (as the
approval fixtures do): never a real exclusion result, only the binding/refusal logic.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl import adjudication, approval, hf_pool, hf_pool_exclusion, pool_freeze as pf
from fine_tuning.notebook_rl import contamination as c
from fine_tuning.notebook_rl import pool_data_evidence as producer
from fine_tuning.notebook_rl.adapters import hf_common as hf
from fine_tuning.notebook_rl.data_tests.test_hf_pool import SPECS, png, quotas, supply
from fine_tuning.notebook_rl.dataset import SPLITS
from fine_tuning.notebook_rl.schema import load_protocol
from fine_tuning.tests.test_notebook_rl_contamination import image_bytes, index_fixture

SPEC_C = hf.SourceSpec("hf-fake-c", "org/fake-c", "a" * 40, "cc-by-4.0", "HUMAN", "authored annotators", "fake_c")
CATEGORIES = {"Biology": "Life", "Physics": "Physical"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path, value) -> dict:
    path.write_text(json.dumps(value))
    return {"path": str(path.absolute()), "file_sha256": _sha(path.read_bytes())}


def _selection_json(path: Path, value: dict) -> dict:
    path.write_bytes(pf._bytes(value) + b"\n")
    return {"path": str(path.absolute()), "file_sha256": _sha(path.read_bytes())}


def rows(source: str, gold: str = "A") -> list[hf.Candidate]:
    """supply() with per-source native ids and pictures, so sources and shards never share an image."""
    return [dataclasses.replace(cand, native_id=f"{source}-{cand.native_id}", images=(png(f"{source}-{cand.native_id}"),))
            for cand in supply(source, 3, gold=gold)]


def build(base: Path, name: str, sources: tuple[str, ...], gold: str = "A") -> Path:
    specs = {**SPECS, "hf-fake-c": SPEC_C}
    supplies = {s: rows(s, gold) for s in sources}
    registry = {s: (specs[s], lambda path, s=s: iter(supplies[s])) for s in sources}
    plan = base / f"{name}-plan.json"
    plan.write_text(json.dumps({"version": hf_pool.PLAN_VERSION, "seed": 7, "quotas": quotas(3, sources)}))
    (base / "raw").mkdir(exist_ok=True)
    hf_pool.build(plan, base / "raw", base / name, registry=registry)
    return base / name


def review(base: Path, root: Path, flags: dict[str, tuple[str, str, str]]) -> dict:
    """Authored raw receipt + AI adjudication {pid: (stage, reason, verdict)}, written in place as the shard receipt."""
    items, files, _ = c._snapshot({s: root / s for s in SPLITS}, root / "images")
    candidates = [{"problem_id": pid, "stage": stage, "reason": reason, "action": "QUARANTINE", "ref": {"id": pid}}
                  for pid, (stage, reason, _) in sorted(flags.items())]
    raw = {"version": c.VERSION, "status": "EXCLUSION_CHECKED_WITH_LIMITS", "training_allowed": False,
           "approval_granted": False, "blockers": [], "partial_name": None, "input_count": len(items),
           "index_sha256": "1" * 64, "index_file_sha256": "2" * 64, "input_files": files,
           "input_files_sha256": c._digest(files), "candidates": candidates,
           "steps": {step: {"status": "COMPLETE", "candidate_count": 0} for step in c._STEPS},
           "limitations": ["Authored unit fixture, not a real benchmark scan", "AI review, not human"]}
    family = c.family_quarantine(items, candidates)
    raw.update(allowlist=family["allowlist"], quarantined_ids=family["quarantined_ids"],
               quarantined_families=family["families"])
    raw["receipt_sha256"] = c._digest({k: v for k, v in raw.items() if k != "receipt_sha256"})
    tag = root.name
    raw_desc = _json(base / f"{tag}-raw.json", raw)
    evidence = _json(base / f"{tag}-evidence.json", {"note": "authored, no actual image judgment"})
    decisions = {"format": adjudication.FORMAT, "raw_receipt_sha256": raw_desc["file_sha256"],
                 "index_sha256": raw["index_sha256"], "input_files_sha256": raw["input_files_sha256"],
                 "provenance": {"reviewer_id": "authored-fixture", "kind": "AI_DIRECT_SIMILARITY_REVIEW",
                                "human_verified": False, "reviewed_at_utc": "2026-10-30T00:00:00Z"},
                 "decisions": [{"candidate_id": adjudication.candidate_id(cand), "verdict": flags[cand["problem_id"]][2],
                                "reason": "authored", "evidence": [evidence]} for cand in candidates]}
    reviewed = adjudication.apply(raw_desc, _json(base / f"{tag}-decisions.json", decisions), items)
    receipt = root / "contamination" / "receipt_v2.json"
    receipt.unlink()
    receipt.write_text(json.dumps(reviewed))
    manifest = json.loads((root / "manifest.yaml").read_bytes())
    manifest["exclusion_receipt_sha256"] = _sha(receipt.read_bytes())
    (root / "manifest.yaml").unlink()
    (root / "manifest.yaml").write_text(json.dumps(manifest))
    return {"items": items, "receipt": reviewed}


def quality_review(path: Path, items: dict, receipt: dict, block: str | None = None) -> Path:
    groups = defaultdict(list)
    for pid in receipt["allowlist"]:
        groups[items[pid]["actor"].source].append(pid)
    sources = {s: [{"problem_id": pid, "image_sha256": list(items[pid]["actor"].image_sha256), "render_ok": s != block,
                    "leak_free": True, "gold_supported": True, "notes": "authored fixture"}
                   for pid in approval.ai_review_order(p, 3407)[:min(approval.AI_LOCAL_SAMPLE, len(p))]]
               for s, p in groups.items()}
    _json(path, {"format": approval.AI_LOCAL_REVIEW_FORMAT, "reviewer_id": "fixture-ai-reviewer",
                 "human_verified": False, "labels_modified": False, "reviewed_at_utc": "2026-10-30T00:00:00+00:00",
                 "sampling_seed": 3407, "input_files_sha256": receipt["input_files_sha256"], "sources": sources,
                 "leak_warnings": []})
    return path


def first(items: dict, source: str, split: str, k: int = 0) -> str:
    return sorted(pid for pid, i in items.items() if i["actor"].source == source and i["split"] == split)[k]


def authored_index(base: Path, seed: int = 99) -> tuple[dict, dict]:
    """Authored full-coverage index (patched in for load_index, as the approval fixtures do) + its file descriptor."""
    desc = _json(base / f"index-{seed}.json", index_fixture(image_bytes(seed)))
    index = c._prepare(index_fixture(image_bytes(seed)), "authored-test-only")
    index.update(full_coverage=True, limitations=[], partial_name=None, **desc)
    return index, desc


def scan_review(base: Path, root: Path, index: dict, flags: dict[str, tuple[str, str, str]]) -> dict:
    """Authored complete scan of a hf_pool shard + authored flags + AI adjudication, written in place as its receipt."""
    protocol = load_protocol(Path(approval.__file__).with_name("protocol_v1.yaml"))
    registry = hf_pool_exclusion.registry_for(root, {**SPECS, "hf-fake-c": SPEC_C})
    items, _, images = c._snapshot({s: root / s for s in SPLITS}, root / "images")
    calibration = {"frozen_before_scan": True, "protocol_sha256": protocol["sha256"], "index_sha256": index["sha256"],
                   "positive_count": 4, "negative_count": 4, "tpr": .75, "fpr": 0.,
                   "per_engine": {k: {"positive_count": 4, "negative_count": 4, "tpr": .75, "fpr": 0.,
                                      "examples_sha256": "d" * 64} for k in ("dhash", "tiles", "sscd", "siglip")},
                   "thresholds": {"perceptual_hamming": 3, "sscd_exclude": .95, "sscd_review": .75, "siglip_review": .9}}
    evidence: dict = {"calibration": calibration}
    registry["scanner_policy"] = {"status": "FROZEN_BEFORE_INPUT_SCAN", "protocol_sha256": protocol["sha256"],
                                  "calibration_sha256": c._digest(calibration), "engines": {}}
    for name, engine in (("ocr", "tesseract"), ("sscd", "sscd_disc_mixup"), ("siglip", "google/siglip-fixture")):
        identity = {"engine_id": engine, "revision": "authored-fixture-v1", "artifact_sha256": "e" * 64}
        registry["scanner_policy"]["engines"][name] = dict(identity)
        evidence[name] = identity | {"index_sha256": index["sha256"], "calibration_sha256": c._digest(calibration)}
    human = {sha for item in items.values() if item["lineage"].label_provenance == "HUMAN" for sha in item["image_sha256"]}
    evidence["ocr"].update(benchmark={sha: {"image_sha256": sha, "text": "authored unrelated diagram label",
                                            "status": "COMPLETE"} for sha in index["images"]},
                           inputs={sha: {"image_sha256": sha, "text": "fixture legend", "status": "COMPLETE"} for sha in human})

    def vectors(bindings: dict, vector: list[float]) -> dict:
        return {key: {"sha256": sha, "vector": vector} for key, sha in bindings.items()}
    for name in ("sscd", "siglip"):
        evidence[name].update(benchmark_images=vectors({sha: sha for sha in index["images"]}, [1., 0.]),
                              input_images=vectors({sha: sha for sha in images}, [0., 1.]))
    text_refs = {e["key"]: _sha(c.normalize_text(e["text"]).encode()) for e in index["text_cache"]["entries"]}
    evidence["siglip"].update(benchmark_texts=vectors(text_refs, [1., 0.]),
                              input_texts=vectors({pid: item["text_sha256"] for pid, item in items.items()}, [0., 1.]))
    with patch.object(c, "_prepare", return_value=index):
        raw = c.scan({s: root / s for s in SPLITS}, root / "images", index, registry, evidence)
    assert raw["status"] == "EXCLUSION_CHECKED_WITH_LIMITS", raw["blockers"]
    raw["candidates"] += [{"problem_id": pid, "stage": stage, "reason": reason, "action": "QUARANTINE", "ref": {"id": pid}}
                          for pid, (stage, reason, _) in sorted(flags.items())]
    family = c.family_quarantine(items, raw["candidates"])
    raw.update(allowlist=family["allowlist"], quarantined_ids=family["quarantined_ids"],
               quarantined_families=family["families"])
    raw["receipt_sha256"] = c._digest({k: v for k, v in raw.items() if k != "receipt_sha256"})
    tag = root.name
    raw_desc = _json(base / f"{tag}-raw.json", raw)
    note = _json(base / f"{tag}-note.json", {"note": "authored, no actual image judgment"})
    decisions = {"format": adjudication.FORMAT, "raw_receipt_sha256": raw_desc["file_sha256"],
                 "index_sha256": raw["index_sha256"], "input_files_sha256": raw["input_files_sha256"],
                 "provenance": {"reviewer_id": "authored-fixture", "kind": "AI_DIRECT_SIMILARITY_REVIEW",
                                "human_verified": False, "reviewed_at_utc": "2026-10-30T00:00:00Z"},
                 "decisions": [{"candidate_id": adjudication.candidate_id(cand), "reason": "authored", "evidence": [note],
                                "verdict": flags.get(cand["problem_id"], ("", "", "DISTINCT"))[2]}
                               for cand in raw["candidates"]]}
    decisions_path = base / f"{tag}-decisions.json"
    reviewed = adjudication.apply(raw_desc, _json(decisions_path, decisions), items)
    receipt = root / "contamination" / "receipt_v2.json"
    receipt.unlink()
    receipt.write_text(json.dumps(reviewed))
    manifest = json.loads((root / "manifest.yaml").read_bytes())
    manifest["exclusion_receipt_sha256"] = _sha(receipt.read_bytes())
    (root / "manifest.yaml").unlink()
    (root / "manifest.yaml").write_text(json.dumps(manifest))
    return {"items": items, "receipt": reviewed, "raw": raw_desc, "decisions": decisions_path,
            "registry": _json(base / f"{tag}-registry.json", registry),
            "evidence": _json(base / f"{tag}-evidence.json", evidence)}


class PoolFreezeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.cat = self.base / "categories.json"
        self.cat.write_text(json.dumps(CATEGORIES))
        self.quota = self.base / "quota.json"
        self.quota.write_text(json.dumps({"Biology": 2, "Physics": 5}))

    def shards(self, prefix: str = "", gold: str = "A") -> tuple[list[Path], list[Path], dict]:
        a = build(self.base, prefix + "shard-a", ("hf-fake-a", "hf-fake-b"), gold)
        b = build(self.base, prefix + "shard-b", ("hf-fake-c",), gold)
        snap, _, _ = c._snapshot({s: a / s for s in SPLITS}, a / "images")
        flags = {first(snap, "hf-fake-a", "train", 0): ("perceptual", "DHASH_CROP_TILE_NEAR", "DUPLICATE"),
                 first(snap, "hf-fake-a", "dev", 0): ("text_ngrams", "NEAR_TEXT_5GRAM", "UNRESOLVED"),
                 first(snap, "hf-fake-a", "train", 1): ("image_exact", "EXACT_RAW_OR_PIXEL_IMAGE", "DISTINCT"),
                 first(snap, "hf-fake-a", "train", 2): ("siglip", "SIGLIP_IMAGE_TO_IMAGE", "DISTINCT")}
        ra, rb = review(self.base, a, flags), review(self.base, b, {})
        qa = quality_review(self.base / f"{prefix}qa.json", ra["items"], ra["receipt"], block="hf-fake-b")
        qb = quality_review(self.base / f"{prefix}qb.json", rb["items"], rb["receipt"])
        return [a, b], [qa, qb], {"flags": flags, "a": ra, "b": rb}

    def freeze(self, shards, reviews, name="frozen", approvals=(), seed=11) -> tuple[dict, list[dict]]:
        pf.freeze(shards, reviews, list(approvals), self.cat, self.quota, seed, self.base / name)
        manifest = json.loads((self.base / name / "dataset_manifest.json").read_bytes())
        return manifest, manifest["items"]

    def explicit(self, shards, reviews):
        found = [pf._shard(i, root, reviews[i], None) for i, root in enumerate(shards)]
        records = [row for _, rows in found for row in rows]
        eligible = [row for row in records if row["status"] is None and row["subject"] == "Biology"][:2]
        self.assertEqual(len(eligible), 2)
        self.quota.write_text(json.dumps({"Biology": 2, "Physics": 0}))
        document = {"version": pf.SELECTION_VERSION,
                    "shards": [{key: meta[key] for key in pf._SELECTION_SHARD_FIELDS} for meta, _ in found],
                    "items": [{"problem_id": row["problem_id"], "category": category}
                              for row, category in zip(eligible, ("C3", "C8"), strict=True)]}
        path = self.base / "selection.json"
        return path, document, records

    def freeze_explicit(self, shards, reviews, path, document, name="selected", **kwargs):
        descriptor = _selection_json(path, document)
        return pf.freeze(shards, reviews, [], self.cat, self.quota, 11, self.base / name,
                         selection=path, selection_sha256=descriptor["file_sha256"], **kwargs)

    def test_explicit_ids_categories_and_quota_do_not_change_original_metadata(self):
        shards, reviews, _ = self.shards()
        path, document, original = self.explicit(shards, reviews)
        self.freeze_explicit(shards, reviews, path, document)
        manifest, _ = pf._manifest(self.base / "selected/dataset_manifest.json")
        self.assertEqual(manifest["selection"], document | {"file_sha256": _sha(path.read_bytes())})
        categories = {row["problem_id"]: row["category"] for row in document["items"]}
        self.assertEqual(pf.selection_categories(manifest), categories)
        self.assertEqual(manifest["quota_order"], pf.SELECTION_ORDER)
        by_id = {row["problem_id"]: row for row in manifest["items"]}
        self.assertEqual({pid for pid, row in by_id.items() if row["status"] == pf.INCLUDED}, set(categories))
        for row in original:
            selected = row["problem_id"] in categories
            after = by_id[row["problem_id"]]
            self.assertEqual(after["status"], (pf.INCLUDED if selected else pf.QUOTA) if row["status"] is None else row["status"])
            self.assertEqual({k: v for k, v in after.items() if k not in ("category", "status")},
                             {k: v for k, v in row.items() if k not in ("category", "status")})
        self.assertNotIn("quality_exclusions", manifest)
        self.freeze_explicit(shards, reviews, path, document, name="selected-again")
        self.assertEqual((self.base / "selected/dataset_manifest.json").read_bytes(),
                         (self.base / "selected-again/dataset_manifest.json").read_bytes())

    def test_explicit_selection_rejects_tamper_duplicates_ineligible_and_shortfall(self):
        shards, reviews, _ = self.shards()
        path, document, original = self.explicit(shards, reviews)
        mutations = [document | {"items": document["items"] * 2}]
        for status in (pf.DEV, pf.SEALED, pf.BLOCKED, pf.DUPLICATE, pf.UNRESOLVED, pf.NONRELEASABLE):
            bad = copy.deepcopy(document)
            bad["items"][0]["problem_id"] = next(row["problem_id"] for row in original if row["status"] == status)
            mutations.append(bad)
        for field, value in (("problem_id", "unknown"), ("category", "C9")):
            bad = copy.deepcopy(document)
            bad["items"][0][field] = value
            mutations.append(bad)
        bad = copy.deepcopy(document)
        bad["shards"][0]["item_index_sha256"] = "0" * 64
        mutations.append(bad)
        mutations.append(document | {"items": document["items"][:1]})
        for number, bad in enumerate(mutations):
            with self.subTest(case=number), self.assertRaises(ValueError):
                self.freeze_explicit(shards, reviews, path, bad, name=f"bad-{number}")
            self.assertFalse((self.base / f"bad-{number}").exists())
        _selection_json(path, document)
        for supplied, digest in ((path, None), (None, _sha(path.read_bytes())), (path, "0" * 64)):
            with self.assertRaises(ValueError):
                pf.freeze(shards, reviews, [], self.cat, self.quota, 11, self.base / "bad-hash",
                          selection=supplied, selection_sha256=digest)
        self.assertFalse((self.base / "bad-hash").exists())

    def test_explicit_receipt_requires_exact_canonical_bytes(self):
        shards, reviews, _ = self.shards()
        path, document, _ = self.explicit(shards, reviews)
        canonical = pf._bytes(document)
        for raw in (canonical, canonical + b"\n\n", b" " + canonical + b"\n", json.dumps(document, indent=2).encode() + b"\n"):
            path.write_bytes(raw)
            with patch.object(pf, "_shard", side_effect=AssertionError("noncanonical input reached a shard")):
                with self.assertRaisesRegex(ValueError, "canonical JSON"):
                    pf.freeze(shards, reviews, [], self.cat, self.quota, 11, self.base / "bad-canonical",
                              selection=path, selection_sha256=_sha(raw))
        self.assertFalse((self.base / "bad-canonical").exists())

    def test_explicit_selection_rejects_quality_filter_and_cross_shard_family(self):
        shards, reviews, _ = self.shards()
        path, document, _ = self.explicit(shards, reviews)
        victim = document["items"][0]["problem_id"]
        quality = self.base / "quality-selected.json"
        _json(quality, {"version": pf.QUALITY_VERSION,
                        "rules": [{"id": "authored", "description": "authored quality failure", "code_sha256": "a" * 64}],
                        "items": [{"problem_id": victim, "rule_id": "authored"}]})
        with self.assertRaisesRegex(ValueError, "eligible train"):
            self.freeze_explicit(shards, reviews, path, document, name="bad-quality", quality_exclusions=quality)
        cross = pf.cross_shard_allowlist
        with patch.object(pf, "cross_shard_allowlist", side_effect=lambda rows: cross(rows) - {victim}):
            with self.assertRaisesRegex(ValueError, "quarantined family"):
                self.freeze_explicit(shards, reviews, path, document, name="bad-family")
        self.assertFalse((self.base / "bad-quality").exists())
        self.assertFalse((self.base / "bad-family").exists())

    def test_embedded_selection_validation_rejects_manifest_tampering(self):
        shards, reviews, _ = self.shards()
        path, document, _ = self.explicit(shards, reviews)
        self.freeze_explicit(shards, reviews, path, document)
        manifest = json.loads((self.base / "selected/dataset_manifest.json").read_bytes())
        victim = document["items"][0]["problem_id"]
        mutations = []
        for field, value in (("category", "C8"), ("split", "dev"), ("status", pf.QUOTA)):
            bad = copy.deepcopy(manifest)
            next(row for row in bad["items"] if row["problem_id"] == victim)[field] = value
            mutations.append(bad)
        bad = copy.deepcopy(manifest)
        bad["selection"]["items"].append(bad["selection"]["items"][0])
        mutations.append(bad)
        bad = copy.deepcopy(manifest)
        bad["quota"]["Biology"] = 3
        mutations.append(bad)
        for number, bad in enumerate(mutations):
            changed = self.base / f"manifest-tamper-{number}.json"
            _json(changed, bad)
            with self.subTest(case=number), self.assertRaises(ValueError):
                pf._manifest(changed)

    def test_original_selection_pin_rejects_coordinated_edits_before_materialization(self):
        shards, reviews, _ = self.shards()
        path, document, _ = self.explicit(shards, reviews)
        self.freeze_explicit(shards, reviews, path, document)
        manifest = json.loads((self.base / "selected/dataset_manifest.json").read_bytes())
        victim = document["items"][0]["problem_id"]
        replacement = next(row["problem_id"] for row in manifest["items"]
                           if row["status"] == pf.QUOTA and row["subject"] == "Biology")
        mutations = []
        category = copy.deepcopy(manifest)
        category["selection"]["items"][0]["category"] = "C8"
        next(row for row in category["items"] if row["problem_id"] == victim)["category"] = "C8"
        mutations.append(category)
        swap = copy.deepcopy(manifest)
        swap["selection"]["items"][0]["problem_id"] = replacement
        next(row for row in swap["items"] if row["problem_id"] == victim)["status"] = pf.QUOTA
        next(row for row in swap["items"] if row["problem_id"] == replacement).update(status=pf.INCLUDED, category="C3")
        mutations.append(swap)
        for key, value in (("selection", None), ("quota_order", None), ("quota_order", "legacy-order")):
            missing = copy.deepcopy(manifest)
            if value is None:
                del missing[key]
            else:
                missing[key] = value
            mutations.append(missing)
        for number, changed in enumerate(mutations):
            changed_path = self.base / f"coordinated-{number}.json"
            _json(changed_path, changed)
            with patch.object(pf, "_union_entry", side_effect=AssertionError("tampered selection reached a shard")) as union:
                with self.subTest(case=number), self.assertRaisesRegex(ValueError, "selection.*SHA256|selection.*binding|embedded receipt"):
                    pf.materialize(changed_path, shards, self.base / f"bad-materialize-{number}",
                                   [Path("missing-registry")] * len(shards), [Path("missing-evidence")] * len(shards))
                union.assert_not_called()
            self.assertFalse((self.base / f"bad-materialize-{number}").exists())

    def test_explicit_opt_out_is_byte_identical_to_legacy_default(self):
        shards, reviews, _ = self.shards()
        self.freeze(shards, reviews, name="legacy-default")
        pf.freeze(shards, reviews, [], self.cat, self.quota, 11, self.base / "legacy-explicit-none",
                  selection=None, selection_sha256=None)
        for name in ("dataset_manifest.json", "summary.json", "summary.md"):
            self.assertEqual((self.base / "legacy-default" / name).read_bytes(),
                             (self.base / "legacy-explicit-none" / name).read_bytes())
        self.assertNotIn("selection", json.loads((self.base / "legacy-default/dataset_manifest.json").read_bytes()))

    def test_every_status_path_and_label_blind_quota(self):
        shards, reviews, ctx = self.shards()
        manifest, items = self.freeze(shards, reviews)
        status = {r["problem_id"]: r["status"] for r in items}
        every = set(ctx["a"]["items"]) | set(ctx["b"]["items"])
        self.assertEqual(set(status), every)                      # every shard item, once
        flags = list(ctx["flags"])
        self.assertEqual([status[p] for p in flags],
                         [pf.DUPLICATE, pf.UNRESOLVED, pf.NONRELEASABLE, status[flags[3]]])
        self.assertIn(status[flags[3]], (pf.INCLUDED, pf.QUOTA))  # DISTINCT releasable: back in the pool
        by_pid = {r["problem_id"]: r for r in items}
        for pid in every - set(flags):
            r = by_pid[pid]
            expected = (pf.BLOCKED if r["source"] == "hf-fake-b" else pf.DEV if r["split"] == "dev"
                        else pf.SEALED if r["split"] in ("sealed", "style_sealed") else None)
            if expected:
                self.assertEqual(r["status"], expected, pid)
        quota = {"Biology": 2, "Physics": 5}
        for subject, q in quota.items():
            pool = sorted((r for r in items if r["subject"] == subject and r["status"] in (pf.INCLUDED, pf.QUOTA)),
                          key=lambda r: (_sha(f"11:{r['source']}:{r['native_id']}".encode()), r["problem_id"]))
            self.assertEqual([r["status"] for r in pool], [pf.INCLUDED] * min(q, len(pool)) + [pf.QUOTA] * (len(pool) - q))
        self.assertEqual(set(status.values()), set(pf.STATUSES))
        blocked = manifest["shards"][0]["blocked_sources"]
        self.assertEqual(list(blocked), ["hf-fake-b"])
        self.assertTrue(all(r.startswith("FAILED_VERDICT render_ok ") for r in blocked["hf-fake-b"]["quality_review"]))
        # ids/hashes/metadata only: no question text, choices or gold anywhere in the outputs
        text = "".join((self.base / "frozen" / n).read_text() for n in ("dataset_manifest.json", "summary.json", "summary.md"))
        for cand in rows("hf-fake-a") + rows("hf-fake-c"):
            self.assertNotIn(cand.question, text)
        self.assertNotIn('"gold"', text)
        self.assertEqual(by_pid[flags[0]]["category"], CATEGORIES[by_pid[flags[0]]["subject"]])
        self.assertEqual(set(by_pid[flags[0]]["record_sha256"]), {"actor", "verifier", "lineage"})
        # determinism (byte-identical) and label-blindness (gold flipped -> same selection)
        again, _ = self.freeze(shards, reviews, name="frozen-again")
        self.assertEqual((self.base / "frozen" / "dataset_manifest.json").read_bytes(),
                         (self.base / "frozen-again" / "dataset_manifest.json").read_bytes())
        flipped, freviews, _ = self.shards(prefix="flip-", gold="B")
        _, flipped_items = self.freeze(flipped, freviews, name="frozen-flip")
        self.assertEqual({r["problem_id"]: r["status"] for r in flipped_items}, status)
        self.assertNotEqual({r["problem_id"]: r["record_sha256"]["verifier"] for r in flipped_items},
                            {r["problem_id"]: r["record_sha256"]["verifier"] for r in items})

    def test_summary_counts_and_contamination_statement(self):
        shards, reviews, ctx = self.shards()
        self.freeze(shards, reviews)
        summary = json.loads((self.base / "frozen" / "summary.json").read_bytes())
        manifest = json.loads((self.base / "frozen" / "dataset_manifest.json").read_bytes())
        self.assertEqual(summary["dataset_manifest_sha256"], _sha((self.base / "frozen" / "dataset_manifest.json").read_bytes()))
        self.assertEqual(sum(summary["totals"].values()), len(manifest["items"]))
        self.assertEqual(set(summary["by_category"]), {"Life", "Physical"})
        self.assertEqual(sum(sum(v.values()) for v in summary["by_subject"].values()), len(manifest["items"]))
        self.assertEqual(set(summary["by_source"]), {"hf-fake-a", "hf-fake-b", "hf-fake-c"})
        contamination = summary["contamination"]
        self.assertIs(contamination["human_verified"], False)
        self.assertIn("1" * 64, contamination["statement"])
        self.assertIn("MMMU/MMMU-Pro", contamination["statement"])
        self.assertEqual(contamination["shards"][0]["adjudication_counts"], {"DISTINCT": 2, "DUPLICATE": 1, "UNRESOLVED": 1})
        self.assertEqual(contamination["shards"][0]["detectors"], sorted(c._STEPS))
        self.assertIn("Physics", (self.base / "frozen" / "summary.md").read_text())

    def test_approval_blocks_sources_and_must_bind_the_shard(self):
        shards, reviews, ctx = self.shards()
        a = shards[0]
        receipt = ctx["a"]["receipt"]
        sources = {pid: i["actor"].source for pid, i in ctx["a"]["items"].items()}
        body = {"status": "VALIDATED_FOR_MEASUREMENT", "stage": approval.AI_LOCAL_STAGE, "human_verified": False,
                "input_binding": {"build_manifest_sha256": _sha((a / "manifest.yaml").read_bytes()),
                                  "exclusion_receipt_sha256": _sha((a / "contamination/receipt_v2.json").read_bytes())},
                "eligibility": {"blocked_sources": {"hf-fake-a": ["authored"]},
                                "eligible_ids": sorted(p for p in receipt["allowlist"] if sources[p] != "hf-fake-a")}}
        good = self.base / "approval.json"
        _json(good, body)
        _, items = self.freeze(shards, [None, None], approvals=[good, None])
        by_pid = {r["problem_id"]: r for r in items}
        for pid in receipt["allowlist"]:
            if sources[pid] == "hf-fake-a":
                self.assertEqual(by_pid[pid]["status"], pf.BLOCKED)
        self.assertIsNone(by_pid[pid]["evidence"]["quality_review_sha256"])
        bad = self.base / "approval-bad.json"
        _json(bad, {**body, "input_binding": {**body["input_binding"], "build_manifest_sha256": "0" * 64}})
        with self.assertRaisesRegex(ValueError, "approval"):
            self.freeze(shards, [None, None], name="bad", approvals=[bad, None])
        self.assertFalse((self.base / "bad").exists())

    def test_quality_exclusions_status_records_and_refusals(self):
        shards, reviews, ctx = self.shards()
        plain, _ = self.freeze(shards, reviews, name="plain")
        self.assertEqual((plain["statuses"], "quality_exclusions" in plain), (list(pf.STATUSES), False))
        train, dev = first(ctx["b"]["items"], "hf-fake-c", "train"), first(ctx["b"]["items"], "hf-fake-c", "dev")
        duplicate = list(ctx["flags"])[0]
        rules = [{"id": "low-res", "description": "authored rule", "code_sha256": "a" * 64},
                 {"id": "blank", "description": "authored rule", "script_sha256": "b" * 64}]
        spec = {"version": pf.QUALITY_VERSION, "rules": rules,
                "items": [{"problem_id": train, "rule_id": "low-res"}, {"problem_id": train, "rule_id": "blank"},
                          {"problem_id": dev, "rule_id": "low-res"}, {"problem_id": duplicate, "rule_id": "low-res"}]}
        path = self.base / "quality.json"
        desc = _json(path, spec)
        args = ["freeze", "--category-map", str(self.cat), "--quota", str(self.quota), "--seed", "11",
                "--quality-exclusions", str(path), "--output", str(self.base / "filtered")]
        for shard, quality in zip(shards, reviews, strict=True):
            args += ["--shard", str(shard), "--quality-review", str(quality)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pf.main(args), 0)
        manifest = json.loads((self.base / "filtered" / "dataset_manifest.json").read_bytes())
        status = {r["problem_id"]: r["status"] for r in manifest["items"]}
        # After the contamination statuses (the duplicate stays DUPLICATE), before held-out splits and quota.
        self.assertEqual([status[train], status[dev], status[duplicate]], [pf.QUALITY_FILTER, pf.QUALITY_FILTER, pf.DUPLICATE])
        self.assertEqual(manifest["statuses"], [*pf.STATUSES, pf.QUALITY_FILTER])
        self.assertEqual(manifest["quality_exclusions"], {"version": pf.QUALITY_VERSION, "file_sha256": desc["file_sha256"],
            "rules": rules, "items": {train: ["blank", "low-res"], dev: ["low-res"], duplicate: ["low-res"]}})
        for r in plain["items"]:
            if r["problem_id"] not in (train, dev) and r["status"] not in (pf.INCLUDED, pf.QUOTA):
                self.assertEqual(status[r["problem_id"]], r["status"])
        for subject, q in {"Biology": 2, "Physics": 5}.items():
            pool = sorted((r for r in manifest["items"] if r["subject"] == subject and r["status"] in (pf.INCLUDED, pf.QUOTA)),
                          key=lambda r: (_sha(f"11:{r['source']}:{r['native_id']}".encode()), r["problem_id"]))
            self.assertEqual([r["status"] for r in pool], [pf.INCLUDED] * min(q, len(pool)) + [pf.QUOTA] * (len(pool) - q))
        summary = json.loads((self.base / "filtered" / "summary.json").read_bytes())
        self.assertEqual(summary["quality_exclusions"], {"version": pf.QUALITY_VERSION, "file_sha256": desc["file_sha256"],
            "excluded_items": 2, "rules": [rules[0] | {"listed": 3, "excluded": 2}, rules[1] | {"listed": 1, "excluded": 1}]})
        self.assertEqual(summary["totals"][pf.QUALITY_FILTER], 2)
        self.assertIn(f"| low-res | `{'a' * 64}` | 3 | 2 |", (self.base / "filtered" / "summary.md").read_text())
        refusals = {"unknown rule ids \\['gone'\\]": [{"problem_id": train, "rule_id": "gone"}],
                    "problem ids \\['nope'\\]": [{"problem_id": "nope", "rule_id": "low-res"}],
                    "repeats": [{"problem_id": dev, "rule_id": "low-res"}] * 2}
        for number, (message, rows) in enumerate(refusals.items()):
            _json(path, spec | {"items": rows})
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                pf.freeze(shards, reviews, [], self.cat, self.quota, 11, self.base / f"refused-{number}", path)
            self.assertFalse((self.base / f"refused-{number}").exists())
        _json(path, spec | {"rules": [rules[0] | {"code_sha256": "not-a-sha"}]})
        with self.assertRaisesRegex(ValueError, pf.QUALITY_VERSION):
            pf.freeze(shards, reviews, [], self.cat, self.quota, 11, self.base / "refused-x", path)

    def test_acceptance_review_failures_are_refused_per_shard(self):
        shards, reviews, ctx = self.shards()
        _, v1 = self.freeze(shards, reviews, name="v1")
        review = json.loads(reviews[1].read_bytes())
        review.update(format=approval.AI_LOCAL_REVIEW_FORMAT_V2, acceptance_rule=approval.ACCEPTANCE_RULE)
        _json(reviews[1], review)
        _, clean = self.freeze(shards, reviews, name="v2-clean")
        self.assertEqual([r["status"] for r in clean], [r["status"] for r in v1])
        # 24 hf-fake-c items, 20 reviewed: one failure blocks the source until all 24 are reviewed.
        review["sources"]["hf-fake-c"][0]["gold_supported"] = False
        _json(reviews[1], review)
        manifest, items = self.freeze(shards, reviews, name="v2-short")
        self.assertIn("ACCEPTANCE_SAMPLE_TOO_SMALL failed=1 reviewed=20 required=24",
                      manifest["shards"][1]["blocked_sources"]["hf-fake-c"]["quality_review"])
        self.assertEqual({r["status"] for r in items if r["source"] == "hf-fake-c"}, {pf.BLOCKED})
        allowed = sorted(ctx["b"]["receipt"]["allowlist"])
        order = approval.ai_review_order(allowed, 3407)
        review["sources"]["hf-fake-c"] += [dict(review["sources"]["hf-fake-c"][1], problem_id=pid,
                                                image_sha256=list(ctx["b"]["items"][pid]["actor"].image_sha256))
                                           for pid in order[20:]]
        _json(reviews[1], review)
        with self.assertRaisesRegex(ValueError, "ineligible; review the materialized vault"):
            self.freeze(shards, reviews, name="v2-failed")
        self.assertFalse((self.base / "v2-failed").exists())

    def test_refuses_unreviewed_shard_and_overlapping_shards(self):
        shards, reviews, _ = self.shards()
        raw = build(self.base, "unreviewed", ("hf-fake-c",))
        with self.assertRaisesRegex(ValueError, "reviewed"):
            self.freeze([shards[0], raw], [reviews[0], None], name="x")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.freeze([shards[0], shards[0]], [reviews[0], reviews[0]], name="y")
        self.assertFalse((self.base / "x").exists() or (self.base / "y").exists())

    def test_item_index_found_beside_the_assemble_receipts_source_build(self):
        shards, _, _ = self.shards()
        asm = self.base / "asm"
        (asm / "build-reviewed").mkdir(parents=True)
        _json(asm / "receipt.json", {"stage": "assemble", "inputs": {"build_manifest": {
            "path": str(shards[0] / "manifest.yaml"), "file_sha256": "0" * 64}}})
        self.assertEqual(pf._item_index(asm / "build-reviewed")[1], _sha((shards[0] / "item_index.json").read_bytes()))
        with self.assertRaisesRegex(ValueError, "item_index"):
            pf._item_index(self.base / "nowhere" / "x")


def flip(target: Path, offset: int = -3) -> bytes:
    raw = target.read_bytes()
    target.write_bytes(raw[:offset] + bytes([raw[offset] ^ 1]) + raw[offset + 1:])
    return raw


class UnionReceiptTests(unittest.TestCase):
    """2 scanned + adjudicated shards -> freeze -> materialize (union receipt) -> producer -> ai_reviewed_local."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.cat, self.quota = self.base / "categories.json", self.base / "quota.json"
        self.cat.write_text(json.dumps(CATEGORIES))
        self.quota.write_text(json.dumps({"Biology": 2, "Physics": 5}))
        self.index, self.index_desc = authored_index(self.base)
        self.shards = [build(self.base, "shard-a", ("hf-fake-a", "hf-fake-b")), build(self.base, "shard-b", ("hf-fake-c",))]
        snap, _, _ = c._snapshot({s: self.shards[0] / s for s in SPLITS}, self.shards[0] / "images")
        self.duplicate, released = first(snap, "hf-fake-a", "train", 0), first(snap, "hf-fake-a", "train", 1)
        flags = {self.duplicate: ("perceptual", "DHASH_CROP_TILE_NEAR", "DUPLICATE"),
                 released: ("siglip", "SIGLIP_IMAGE_TO_IMAGE", "DISTINCT")}
        self.scans = [scan_review(self.base, self.shards[0], self.index, flags),
                      scan_review(self.base, self.shards[1], self.index, {})]
        self.manifest = self.freeze()
        self.vault = self.base / "vault"
        args = ["materialize", "--manifest", str(self.manifest), "--output", str(self.vault)]
        for shard, scan in zip(self.shards, self.scans, strict=True):
            args += ["--shard", str(shard), "--source-registry", scan["registry"]["path"],
                     "--evidence", scan["evidence"]["path"]]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pf.main(args), 0)

    def freeze(self, name: str = "frozen", quality: Path | None = None) -> Path:
        pf.freeze(self.shards, [None, None], [], self.cat, self.quota, 11, self.base / name, quality)
        return self.base / name / "dataset_manifest.json"

    def materialize(self, manifest: Path, name: str) -> dict:
        return pf.materialize(manifest, self.shards, self.base / name, [Path(s["registry"]["path"]) for s in self.scans],
                              [Path(s["evidence"]["path"]) for s in self.scans])

    def kept(self, manifest: Path | None = None) -> dict:
        return {r["problem_id"]: r for r in json.loads((manifest or self.manifest).read_bytes())["items"]
                if r["status"] in pf.KEPT}

    def request(self, vault: Path | None = None, failed: str | None = None) -> dict:
        """D1-D8 from the real producer CLI with an authored AI review over the producer's own preregistered sample;
        with `failed`, a v2 (acceptance-sampling) review of 60-item prefixes whose `failed` item fails a verdict."""
        vault = vault or self.vault
        sample = producer.review_sample(vault, count=approval.ACCEPTANCE_SAMPLE if failed else approval.AI_LOCAL_SAMPLE)
        items, files, _ = c._snapshot({s: vault / s for s in SPLITS}, vault / "images")
        review = {"format": approval.AI_LOCAL_REVIEW_FORMAT, "reviewer_id": "fixture-ai-reviewer", "human_verified": False,
                  "labels_modified": False, "reviewed_at_utc": "2026-10-30T00:00:00+00:00", "sampling_seed": 3407,
                  "input_files_sha256": c._digest(files),
                  "sources": {source: [{"problem_id": pid, "image_sha256": list(items[pid]["actor"].image_sha256),
                                        "render_ok": True, "leak_free": True, "gold_supported": pid != failed,
                                        "notes": "authored fixture"} for pid in pids]
                              for source, pids in sample["sources"].items()},
                  "leak_warnings": [{"problem_id": row["problem_id"], "field": row["field"], "leak_free": True,
                                     "notes": "authored fixture"} for row in sample["leak_warnings"]]}
        if failed:
            review.update(format=approval.AI_LOCAL_REVIEW_FORMAT_V2, acceptance_rule=approval.ACCEPTANCE_RULE)
        review_desc = _json(self.base / f"review-{vault.name}.json", review)
        probe = self.base / "benchmark-probe.txt"
        probe.write_text("authored stand-in benchmark file for the denial probe")
        out_dir = self.base / ("evidence-out" if vault == self.vault else f"evidence-{vault.name}")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(producer.main([
                "produce", "--build", str(vault), "--review", review_desc["path"], "--benchmark-probe", str(probe),
                "--index", self.index_desc["path"], "--out", str(out_dir),
                "--source-registry", *(s["registry"]["path"] for s in self.scans),
                "--evidence", *(s["evidence"]["path"] for s in self.scans)]), 0)
        written = json.loads(out.getvalue())
        self.assertEqual((written["status"], written["human_verified"]), ("AI_LOCAL_DATA_EVIDENCE_WRITTEN_NOT_APPROVAL", False))
        return json.loads(Path(written["data_request"]["path"]).read_text())

    def validate(self, request: dict, **changes) -> dict:
        with patch.object(c, "load_index", return_value=self.index):
            return approval._validate(request | {"stage": approval.AI_LOCAL_STAGE} | changes)

    def assert_blocked(self, result: dict, code: str) -> None:
        self.assertEqual((result["status"], result["blockers"], result["selected_ids"]), ("BLOCKED_NOT_CHECKED", [code], []))

    def test_explicit_selection_round_trip_then_hash_only_runtime_without_shards(self):
        from fine_tuning.error_note import frozen_data as fd, order

        before = {path: path.read_bytes() for root in self.shards for path in root.rglob("*") if path.is_file()}
        found = [pf._shard(i, root, None, None) for i, root in enumerate(self.shards)]
        eligible = [row for _, rows in found for row in rows if row["status"] is None and row["subject"] == "Biology"][:2]
        self.assertEqual(len(eligible), 2)
        self.quota.write_text(json.dumps({"Biology": 2, "Physics": 0}))
        receipt = {"version": pf.SELECTION_VERSION,
                   "shards": [{key: meta[key] for key in pf._SELECTION_SHARD_FIELDS} for meta, _ in found],
                   "items": [{"problem_id": row["problem_id"], "category": category}
                             for row, category in zip(eligible, ("C3", "C8"), strict=True)]}
        selection_path = self.base / "explicit-selection.json"
        desc = _selection_json(selection_path, receipt)
        pf.freeze(self.shards, [None, None], [], self.cat, self.quota, 11, self.base / "explicit-freeze",
                  selection=selection_path, selection_sha256=desc["file_sha256"])
        manifest = self.base / "explicit-freeze/dataset_manifest.json"
        self.materialize(manifest, "explicit-vault")
        root = self.base / "explicit-vault"
        kept = self.kept(manifest)
        self.assertEqual(pf.verify(manifest, root), [])
        for split in SPLITS:
            for _, name in pf._RECORDS:
                copied = pf._lines(root / split / name)
                originals = {pid: raw for shard in self.shards for pid, raw in pf._lines(shard / split / name).items()}
                self.assertTrue(all(raw == originals[pid] for pid, raw in copied.items()))
        self.assertTrue(all(path.read_bytes() == raw for path, raw in before.items()))
        result = self.validate(self.request(root))
        self.assertEqual((result["status"], result["blockers"]), ("VALIDATED_FOR_MEASUREMENT", []))
        approved = _json(self.base / "explicit-approved.json", result)
        ids = [row["problem_id"] for row in receipt["items"]]
        sources = Counter(kept[pid]["source"] for pid in ids)
        order_path = self.base / "explicit-order.json"
        ordered = {"version": order.VERSION, "count": 2, "ids_sha256": order._ids_sha256(ids),
                   "items": [{"problem_id": pid, "source": kept[pid]["source"], "cluster": kept[pid]["category"]}
                             for pid in ids],
                   "selection": {"pilot_ids_sha256": order._ids_sha256(ids[:1]),
                                 "cluster_train_counts": {"C3": 1, "C8": 1}, "cluster_pilot_counts": {"C3": 1},
                                 "source_train_counts": dict(sources), "source_caps": dict(sources)}}
        pinned = _json(order_path, ordered)
        for shard in self.shards:
            shutil.rmtree(shard)
        selection_path.unlink()
        for scan in self.scans:
            for name in ("registry", "evidence"):
                Path(scan[name]["path"]).unlink()
        Path(self.index_desc["path"]).unlink()
        with patch.object(fd, "TRAIN_COUNT", 2), patch.object(fd, "PILOT_COUNT", 1), \
                patch.object(approval, "_validate", side_effect=AssertionError("approval rerun")), \
                patch.object(c, "scan", side_effect=AssertionError("benchmark rescan")), \
                patch.object(c, "load_index", side_effect=AssertionError("benchmark index access")):
            checked = fd.verify_frozen_data(root, approved, order_path, order_sha256=pinned["file_sha256"])
        self.assertEqual(checked["cluster_train_counts"], {"C3": 1, "C8": 1})
        self.assertEqual((checked["train_count"], checked["pilot_count"], checked["main_count"]), (2, 1, 1))

    def test_union_vault_is_approved_and_binds_every_shard(self):
        union = json.loads((self.vault / "contamination" / pf.UNION_FILE).read_bytes())
        kept = self.kept()
        self.assertEqual((union["version"], union["status"], union["human_verified"], union["training_allowed"]),
                         (pf.UNION_VERSION, pf.UNION_STATUS, False, False))
        self.assertEqual(union["items"], {pid: {"shard": r["shard"], "shard_allowlisted": True} for pid, r in kept.items()})
        self.assertEqual({r["shard"] for r in kept.values()}, {0, 1})
        self.assertNotIn(self.duplicate, union["items"])
        for shard, scan, entry in zip(self.shards, self.scans, union["shards"], strict=True):
            receipt = shard / "contamination" / "receipt_v2.json"
            self.assertEqual(entry["receipt"], {"path": str(receipt), "file_sha256": _sha(receipt.read_bytes())})
            self.assertEqual((entry["source_registry"], entry["evidence"]), (scan["registry"], scan["evidence"]))
            self.assertEqual((entry["index_sha256"], entry["evidence_sha256"], entry["input_files_sha256"]),
                             (self.index["sha256"], scan["receipt"]["evidence_sha256"], scan["receipt"]["input_files_sha256"]))
            self.assertEqual(entry["adjudication_counts"], scan["receipt"]["adjudication"]["counts"])
        self.assertEqual(union["shards"][0]["adjudication_counts"], {"DISTINCT": 1, "DUPLICATE": 1, "UNRESOLVED": 0})
        text = (self.vault / "contamination" / pf.UNION_FILE).read_text()
        self.assertNotIn('"gold"', text)
        for cand in rows("hf-fake-a"):
            self.assertNotIn(cand.question, text)
        request = self.request()
        self.assertEqual((request["source_registry"], request["evidence"]),
                         ({"shards": [s["registry"] for s in self.scans]}, {"shards": [s["evidence"] for s in self.scans]}))
        result = self.validate(request)
        self.assertEqual((result["status"], result["blockers"]), ("VALIDATED_FOR_MEASUREMENT", []))
        self.assertEqual((result["stage"], result["human_verified"], result["training_allowed"]),
                         (approval.AI_LOCAL_STAGE, False, False))
        self.assertEqual(result["input_binding"]["exclusion_receipt_sha256"],
                         _sha((self.vault / "contamination" / pf.UNION_FILE).read_bytes()))
        self.assertEqual((result["eligibility"]["eligible_ids"], result["eligibility"]["quarantined_count"],
                          result["eligibility"]["blocked_sources"]), (sorted(kept), 0, {}))
        self.assertEqual(set(result["selected_ids"]), {pid for pid, r in kept.items() if r["split"] == "train"})
        self.assertTrue(any(text.startswith("Union of 2 shard exclusion receipts") for text in result["limitations"]))
        # Zero generator items: D1/D2/D4 cover nothing, and the producer did not need D4 templates.
        gates = json.loads((self.base / "evidence-out" / "data_report.json").read_bytes())["gates"]
        d4 = json.loads(Path(gates["D4"]["artifact"]["path"]).read_bytes())["measurements"]
        d1 = json.loads(Path(gates["D1"]["artifact"]["path"]).read_bytes())["measurements"]
        self.assertEqual((d4, d1["items_checked"]), ({"templates": {}}, 0))
        # Union is an ai_reviewed_local-only binding; the request must name the union's own registries/evidence.
        self.assert_blocked(self.validate(request, stage="measurement"), "UNION_RECEIPT_AI_LOCAL_ONLY")
        swapped = {"shards": request["evidence"]["shards"][::-1]}
        self.assert_blocked(self.validate(request, evidence=swapped), "UNION_EVIDENCE_REQUEST")

    def test_quality_filtered_items_are_never_materialized_sampled_or_eligible(self):
        kept = self.kept()
        train = sorted(pid for pid, r in kept.items() if r["split"] == "train")[0]
        dev = sorted(pid for pid, r in kept.items() if r["split"] == "dev")[0]
        quality = self.base / "quality.json"
        _json(quality, {"version": pf.QUALITY_VERSION, "rules": [{"id": "low-res", "description": "authored",
                                                                   "code_sha256": "c" * 64}],
                        "items": [{"problem_id": train, "rule_id": "low-res"}, {"problem_id": dev, "rule_id": "low-res"}]})
        manifest = self.freeze("frozen-quality", quality)
        vault = self.base / "vault-quality"
        self.materialize(manifest, "vault-quality")
        filtered = self.kept(manifest)
        self.assertFalse({train, dev} & set(filtered))
        self.assertEqual(set(json.loads((vault / "contamination" / pf.UNION_FILE).read_bytes())["items"]), set(filtered))
        sample = producer.review_sample(vault, count=approval.ACCEPTANCE_SAMPLE)
        self.assertEqual({pid for pids in sample["sources"].values() for pid in pids}, set(filtered))
        # A v2 review over the materialized items: the failed item alone becomes ineligible (all n < 60 reviewed).
        failed = sorted(pid for pid, r in filtered.items() if r["split"] == "train")[0]
        request = self.request(vault, failed=failed)
        result = self.validate(request)
        self.assertEqual((result["status"], result["blockers"]), ("VALIDATED_FOR_MEASUREMENT", []))
        quality_section = result["eligibility"]["quality"]
        source = filtered[failed]["source"]
        self.assertEqual((quality_section["rule"], quality_section["sources"][source]["failed_ids"],
                          quality_section["sources"][source]["status"]), (approval.ACCEPTANCE_RULE, [failed], "ELIGIBLE"))
        self.assertEqual(result["eligibility"]["eligible_ids"], sorted(set(filtered) - {failed}))
        self.assertNotIn(failed, result["selected_ids"])
        gates = json.loads((self.base / "evidence-vault-quality" / "data_report.json").read_bytes())["gates"]
        d3 = json.loads(Path(gates["D3"]["artifact"]["path"]).read_bytes())
        self.assertTrue(any(text.startswith(approval.ACCEPTANCE_RULE) for text in d3["limitations"]))

    def test_tampered_shard_receipt_or_adjudication_is_refused(self):
        request = self.request()
        receipt = self.shards[1] / "contamination" / "receipt_v2.json"
        original = receipt.read_bytes()
        receipt.write_bytes(original + b" ")
        self.assert_blocked(self.validate(request), "ARTIFACT_SHA_MISMATCH")
        with self.assertRaises(ValueError):
            producer.review_sample(self.vault)
        with self.assertRaisesRegex(ValueError, "shard-b manifest is not bound to its exclusion receipt"):
            self.materialize(self.manifest, "vault-tampered")
        receipt.write_bytes(original)
        self.assertEqual(self.validate(request)["blockers"], [])
        # The shard receipt is re-derived (adjudication.validate): changed review decisions are refused.
        decisions = self.scans[0]["decisions"]
        decisions.write_bytes(decisions.read_bytes() + b" ")
        self.assert_blocked(self.validate(request), "MISSING_MALFORMED_OR_CHANGED_PREREQUISITE")
        with self.assertRaises(ValueError):
            producer.review_sample(self.vault)

    def test_item_not_allowlisted_in_its_shard_is_refused(self):
        request = self.request()
        manifest = json.loads(self.manifest.read_bytes())
        for row in manifest["items"]:
            if row["problem_id"] == self.duplicate:
                self.assertEqual(row["status"], pf.DUPLICATE)
                row["status"] = pf.INCLUDED
        edited = self.base / "edited-manifest.json"
        edited.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, f"{self.duplicate} is not allowlisted in shard 0"):
            self.materialize(edited, "vault-bad")
        self.assertFalse((self.base / "vault-bad").exists())
        # Even a union whose materialize guard was bypassed is refused by re-deriving the shard receipt.
        with patch.object(pf, "_allowed", lambda receipt: set(receipt["allowlist"]) | set(receipt["quarantined_ids"])):
            self.materialize(edited, "vault-bypassed")
        bypassed = self.base / "vault-bypassed"
        self.assertIn(self.duplicate, json.loads((bypassed / "contamination" / pf.UNION_FILE).read_bytes())["items"])
        self.assert_blocked(self.validate(request, build_root=str(bypassed)), "UNION_ITEM_NOT_ALLOWLISTED")
        with self.assertRaisesRegex(ValueError, "not an allowlisted record"):
            producer.review_sample(bypassed)

    def test_mixed_index_identity_is_refused(self):
        other, _ = authored_index(self.base, seed=98)
        shard = build(self.base, "shard-c", ("hf-fake-c",))
        mixed = scan_review(self.base, shard, other, {})
        self.shards[1], self.scans[1] = shard, mixed
        manifest = self.freeze("frozen-mixed")
        with self.assertRaisesRegex(ValueError, "shards mix index_sha256, index_file_sha256, calibration_sha256"):
            self.materialize(manifest, "vault-mixed")
        self.assertFalse((self.base / "vault-mixed").exists())

    def test_shard_receipt_without_adjudication_is_refused(self):
        manifest = json.loads(self.manifest.read_bytes())
        receipt, root_manifest = self.shards[1] / "contamination" / "receipt_v2.json", self.shards[1] / "manifest.yaml"
        receipt.unlink()
        receipt.write_bytes(Path(self.scans[1]["raw"]["path"]).read_bytes())  # the unreviewed raw scan
        build_manifest = json.loads(root_manifest.read_bytes())
        build_manifest["exclusion_receipt_sha256"] = _sha(receipt.read_bytes())
        root_manifest.unlink()
        root_manifest.write_text(json.dumps(build_manifest))
        manifest["shards"][1].update(build_manifest_sha256=_sha(root_manifest.read_bytes()),
                                     exclusion_receipt_file_sha256=_sha(receipt.read_bytes()))
        edited = self.base / "edited-manifest.json"
        edited.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "no reviewed \\(adjudicated\\) exclusion receipt"):
            self.materialize(edited, "vault-raw")
        self.assertFalse((self.base / "vault-raw").exists())

    def test_family_spanning_shards_stays_quarantined(self):
        def item(family: str, image: str) -> dict:
            return {"structure_hash": family, "image_sha256": [image * 64], "pixel_sha256": [image * 64]}
        a = {"a1": item("F", "1"), "a2": item("G", "2")}
        b = {"b1": item("F", "3"), "b2": item("H", "4"), "b3": item("K", "2")}
        # a1 quarantined in shard a: its family member b1 in shard b is excluded too; nothing is ever released.
        self.assertEqual(pf.cross_shard_allowlist([(a, {"a2"}), (b, {"b1", "b2", "b3"})]), {"a2", "b2", "b3"})
        self.assertEqual(pf.cross_shard_allowlist([(a, {"a1", "a2"}), (b, {"b3"})]), {"a2", "b3"})
        with self.assertRaisesRegex(ValueError, "share a problem id"):
            pf.cross_shard_allowlist([(a, set(a)), ({"a1": item("Z", "9")}, {"a1"})])
        # Every consumer applies it: materialize, the producer and the approval stage.
        request, victim, real = self.request(), sorted(self.kept())[0], pf.cross_shard_allowlist
        with patch.object(pf, "cross_shard_allowlist", lambda shards: real(shards) - {victim}):
            self.assert_blocked(self.validate(request), "UNION_CROSS_SHARD_FAMILY_QUARANTINED")
            with self.assertRaisesRegex(ValueError, "not an allowlisted record"):
                producer.review_sample(self.vault)
            with self.assertRaisesRegex(ValueError, f"{victim} shares a family"):
                self.materialize(self.manifest, "vault-family")
        self.assertFalse((self.base / "vault-family").exists())

    def test_materialize_verify_round_trip_and_tamper_detection(self):
        path, out, kept = self.manifest, self.vault, self.kept()
        self.assertEqual(pf.verify(path, out), [])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pf.main(["verify", "--manifest", str(path), "--vault", str(out)]), 0)
        self.assertEqual(json.loads((out / "freeze_binding.json").read_bytes())["counts"],
                         {split: sum(r["split"] == split for r in kept.values()) for split in SPLITS})
        # records are byte-identical lines of the shard vaults
        pid = sorted(kept)[0]
        split = kept[pid]["split"]
        line = next(x for x in (out / split / "verifier_private.jsonl").read_bytes().split(b"\n") if pid.encode() in x)
        shard_line = next(x for x in (self.shards[kept[pid]["shard"]] / split / "verifier_private.jsonl").read_bytes()
                          .split(b"\n") if pid.encode() in x)
        self.assertEqual(line, shard_line)
        request = self.request()
        image = out / "images" / kept[pid]["image_paths"][0]
        original = flip(image, offset=len(image.read_bytes()) // 2)
        self.assertEqual(pf.verify(path, out), [f"images: hash mismatch {kept[pid]['image_paths'][0]}"])
        self.assert_blocked(self.validate(request), "UNION_VAULT_NOT_FROZEN_MANIFEST")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pf.main(["verify", "--manifest", str(path), "--vault", str(out)]), 1)
        image.write_bytes(original)
        self.assertEqual(self.validate(request)["blockers"], [])
        record = out / split / "verifier_private.jsonl"
        flip(record, offset=20)
        self.assertTrue(any("verifier_private.jsonl: hash mismatch" in e or "unreadable" in e for e in pf.verify(path, out)))
        (out / "images" / "extra.png").write_bytes(b"x")
        self.assertIn("images: extra file extra.png", pf.verify(path, out))
        # materialize refuses a shard whose bytes no longer match the manifest
        source_image = next(r for r in kept.values() if r["shard"] == 0)
        flip(self.shards[0] / "images" / source_image["image_paths"][0], offset=40)
        with self.assertRaisesRegex(ValueError, "image"):
            self.materialize(path, "vault2")
        self.assertFalse((self.base / "vault2").exists())

if __name__ == "__main__":
    unittest.main()
