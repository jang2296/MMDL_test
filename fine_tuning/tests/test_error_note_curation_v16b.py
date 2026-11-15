from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from fine_tuning.error_note import curation_policy_v16b as policy
from fine_tuning.error_note import curation_v16b as curation


def candidate(index: int, source: str = "hf-chartqa") -> dict:
    pid = f"{source}:{index}"
    actor = {"problem_id": pid, "question": "Which value is shown in the chart?",
             "choices": ["one", "two"]}
    verifier = {"problem_id": pid, "answer_type": "mcq", "choices": ["one", "two"],
                "gold": "A", "accept": {}}
    lineage = {"problem_id": pid, "source": source, "source_revision": "native-revision",
               "split": "train", "structure_hash": f"structure-{index}",
               "benchmark_derived": False, "model_touched": False,
               "label_provenance": "HUMAN", "upstream_origin": "native annotators",
               "license": "CC-BY", "image_provenance": "native-image"}
    return {"problem_id": pid, "source": source, "source_revision": "native-revision",
            "native_id": str(index), "structure_hash": f"structure-{index}",
            "family_id": f"family-{index}", "actor": actor, "verifier": verifier,
            "lineage": lineage, "manifest_item": {"split": "train"},
            "record_sha256": {key: curation._digest(value) for key, value in
                              (("actor", actor), ("verifier", verifier), ("lineage", lineage))},
            "record_paths": {"actor": "/native/actor.jsonl", "verifier": "/native/verifier.jsonl",
                             "lineage": "/native/lineage.jsonl"},
            "image_paths": [f"/native/{index}.png"],
            "image_sha256": [f"image-{index}"], "pixel_sha256": [f"pixel-{index}"],
            "historical_bindings_status": "PASS_RECORDS_AND_IMAGES_RECHECKED",
            "historical_allowlisted": True, "new_fields_checked": True,
            "human_verified": False, "exclusion_reasons": [], "eligible": True}


def review(item: dict, **kwargs) -> dict:
    return {"problem_id": item["problem_id"], "source": item["source"],
            "label_status": "unknown", "legible": True, "image_needed": True,
            "reasoning_steps": 2, "level": "unknown", "reviewed_by": "codex-ai",
            "human_verified": False} | kwargs


def calibration(source: str, count: int, errors: int = 0) -> dict:
    return {"sources": {source: {"sample_count": count, "label_error_count": errors,
            "reviewed_rule_precision": {"image_needed": {"correct": count, "total": count},
                                       "reasoning_depth": {"correct": count, "total": count}}}}}


class GradabilityTests(unittest.TestCase):
    def test_native_mcq_key_and_finite_numeric_only(self):
        item = candidate(0)
        self.assertTrue(curation.gradable(item)[0])
        item["verifier"]["gold"] = "Z"
        self.assertFalse(curation.gradable(item)[0])
        item["verifier"].update(answer_type="decimal", gold="NaN")
        self.assertFalse(curation.gradable(item)[0])
        item["verifier"]["gold"] = "1,200.5"
        self.assertTrue(curation.gradable(item)[0])
        item["verifier"]["answer_type"] = "int"
        self.assertFalse(curation.gradable(item)[0])

    def test_short_text_source_guard_preserves_original_target(self):
        item = candidate(0)
        item["verifier"].update(answer_type="short_text", gold="North America")
        before = copy.deepcopy(item["verifier"])
        self.assertTrue(curation.gradable(item)[0])
        self.assertEqual(before, item["verifier"])
        item["source"] = "unknown"
        self.assertFalse(curation.gradable(item)[0])
        item["source"] = "hf-chartqa"
        item["verifier"]["gold"] = "a " * 20
        self.assertFalse(curation.gradable(item)[0])

    def test_slake_unbounded_and_anatomical_category_targets_are_rejected(self):
        item = candidate(0, "hf-slake")
        item["verifier"].update(answer_type="short_text", gold="Keep healthy")
        item["actor"]["question"] = "How to prevent the disease in this image?"
        self.assertEqual((False, "slake_unbounded_prevention_advice"), curation.gradable(item))
        item["actor"]["question"] = "How much damage will be caused to the body taking such an image?"
        item["verifier"]["gold"] = "None"
        self.assertFalse(curation.gradable(item)[0])
        item["actor"]["question"] = "What organ system is shown?"
        item["verifier"]["gold"] = "Pelvic Cavity"
        self.assertFalse(curation.gradable(item)[0])
        item["actor"]["question"] = "What organ system do the organs belong to?"
        item["verifier"]["gold"] = "Digestive System"
        self.assertTrue(curation.gradable(item)[0])

    def test_unknown_or_teacher_provenance_cannot_be_clean(self):
        item = candidate(0)
        self.assertEqual([], curation._hard_reasons(item))
        item["lineage"]["model_touched"] = True
        self.assertIn("TEACHER_OR_BENCHMARK_FIELD", curation._hard_reasons(item))
        self.assertIn("RECORD_HASH_MISMATCH", curation._hard_reasons(item))


class NewTextTests(unittest.TestCase):
    def test_every_new_leaf_scanned_and_match_quarantines_family(self):
        first, sibling, clean = candidate(0), candidate(1), candidate(2)
        sibling["family_id"] = first["family_id"]
        first["human_solution"] = {"lecture": "native lecture", "solution": "matching rationale"}
        first["raw_annotation"] = {"facts": ["one raw fact", {"nested": "matching annotation"}]}
        first["native_metadata"] = {"skill": "native skill"}
        seen = []

        def matcher(text, field, cache):
            seen.append((text, field))
            return [{"action": "REVIEW"}] if text.startswith("matching") else []

        result = curation.scan_new_texts([first, sibling, clean], {}, matcher)
        self.assertEqual({"native lecture", "matching rationale", "one raw fact",
                          "matching annotation", "native skill"}, {text for text, _ in seen})
        self.assertEqual([first["family_id"]], result["quarantined_families"])
        self.assertEqual(2, len(result["matches"]))
        self.assertFalse(result["thresholds_changed"])


class ReviewTests(unittest.TestCase):
    def test_error_and_illegibility_excluded_unknown_label_allowed(self):
        rows = [candidate(i) for i in range(3)]
        reviews = [review(rows[0], label_status="incorrect"), review(rows[1], legible=False),
                   review(rows[2])]
        stats = calibration("hf-chartqa", 3, 1)
        stats["sources"]["hf-chartqa"]["mitigation"] = "native error type excluded"
        result = curation.apply_reviews(rows, reviews, stats)
        self.assertFalse(result[0]["eligible"])
        self.assertIn("LABEL_ERROR", result[0]["exclusion_reasons"])
        self.assertFalse(result[1]["eligible"])
        self.assertTrue(result[2]["eligible"])

    def test_unreviewed_heuristic_disabled_below_80_percent(self):
        rows = [candidate(i) for i in range(21)]
        reviews = [review(item) for item in rows[:20]]
        stats = calibration("hf-chartqa", 20)
        stats["sources"]["hf-chartqa"]["reviewed_rule_precision"]["image_needed"]["correct"] = 15
        result = curation.apply_reviews(rows, reviews, stats)
        self.assertIsNone(result[-1]["image_needed"])
        self.assertTrue(result[0]["image_needed"])
        self.assertIn("image_needed", result[-1]["disabled_rules"])

    def test_review_can_exclude_invalid_question_and_ungradable_answer(self):
        rows = [candidate(0), candidate(1)]
        reviews = [review(rows[0], question_error=True),
                   review(rows[1], exclusion_reasons=["UNGRADABLE"])]
        result = curation.apply_reviews(rows, reviews, calibration("hf-chartqa", 2))
        self.assertIn("INVALID_QUESTION_PREMISE", result[0]["exclusion_reasons"])
        self.assertIn("UNGRADABLE", result[1]["exclusion_reasons"])
        self.assertFalse(any(item["eligible"] for item in result))

    def test_missing_sample_or_unmitigated_error_rate_blocks(self):
        rows = [candidate(i) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "incomplete source review"):
            curation.apply_reviews(rows, [review(rows[0])], calibration("hf-chartqa", 1))
        reviews = [review(item, label_status="incorrect") for item in rows]
        with self.assertRaisesRegex(ValueError, "needs source mitigation"):
            curation.apply_reviews(rows, reviews, calibration("hf-chartqa", 3, 3))


class SelectionTests(unittest.TestCase):
    def test_proportional_monitor_reproducible_and_image_disjoint(self):
        train = [candidate(0)]
        pool = [candidate(i, "hf-chartqa" if i < 5 else "hf-mapqa") for i in range(1, 9)]
        duplicate = candidate(20)
        duplicate["pixel_sha256"] = train[0]["pixel_sha256"]
        first = curation.monitor_items(pool + [duplicate], train, 4)
        second = curation.monitor_items(list(reversed(pool)) + [duplicate], train, 4)
        self.assertEqual([item["problem_id"] for item in first], [item["problem_id"] for item in second])
        self.assertEqual(4, len(first))
        self.assertNotIn(duplicate["problem_id"], {item["problem_id"] for item in first})
        self.assertEqual({"hf-chartqa": 2, "hf-mapqa": 2}, dict(Counter(item["source"] for item in first)))

    def test_holdout_pixels_and_mandatory_ai2d_cannot_be_omitted(self):
        train, monitor = [candidate(i) for i in range(3)], [candidate(3)]
        order = policy.interleave(train)
        checks = curation.validate_selection(train, monitor, train + monitor, [], order,
                                             total=3, monitor_count=1, source_cap=3)
        self.assertEqual(3, checks["last_batch_size"])
        held = [{"problem_id": "old", "pixel_sha256": train[0]["pixel_sha256"]}]
        with self.assertRaisesRegex(ValueError, "historical holdout overlap"):
            curation.validate_selection(train, monitor, train + monitor, held, order,
                                         total=3, monitor_count=1, source_cap=3)
        mandatory = candidate(50, "hf-ai2d")
        with self.assertRaisesRegex(ValueError, "mandatory AI2D omitted"):
            curation.validate_selection(train, monitor, train + monitor + [mandatory], [], order,
                                         total=3, monitor_count=1, source_cap=3)

    def test_prefix_balance_is_recounted_instead_of_trusting_pass_metadata(self):
        rows = [candidate(i, "hf-chartqa" if i < 16 else "hf-mapqa") for i in range(32)]
        order = policy.interleave(rows)
        self.assertEqual("PASS", curation.validate_prefix_balance(rows, order)["status"])
        invalid = copy.deepcopy(order)
        invalid["items"].sort(key=lambda item: item["source"])
        for index, batch in enumerate(invalid["batches"]):
            batch["problem_ids"] = [item["problem_id"] for item in invalid["items"][index * 16:(index + 1) * 16]]
        with self.assertRaisesRegex(ValueError, "prefix balance exceeds 2"):
            curation.validate_prefix_balance(rows, invalid)

    def test_file_references_rechecked_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "one.json"
            curation._write(path, {"value": 1})
            reference = curation._ref(path)
            self.assertEqual(path, curation._check_ref(reference, {}))
            with self.assertRaises(FileExistsError):
                curation._write(path, {"value": 2})
            path.write_text(json.dumps({"value": 3}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                curation._check_ref(reference, {})


class FamilySupplementTests(unittest.TestCase):
    @staticmethod
    def supplement(links: list[tuple[dict, dict]]) -> dict:
        return {
            "schema": "v16b-scienceqa-visual-family-supplement-v1",
            "status": "COMPLETE", "source_scope": ["hf-scienceqa"],
            "links": [{"left_problem_id": left["problem_id"],
                       "right_problem_id": right["problem_id"],
                       "left_family_id": left.get("family_id", left.get("connected_family_sha256")),
                       "right_family_id": right.get("family_id", right.get("connected_family_sha256")),
                       "decision": "SAME_VISUAL_FAMILY"} for left, right in links],
        }

    def test_cross_holdout_union_closes_old_families_and_preserves_quarantine(self):
        first, second = candidate(0, "hf-scienceqa"), candidate(1, "hf-scienceqa")
        sibling = candidate(2, "hf-chartqa")
        sibling["family_id"] = second["family_id"]
        first["exclusion_reasons"] = ["NEW_TEXT_MATCH_FAMILY", "LABEL_ERROR"]
        first["eligible"] = False
        held = {"problem_id": "hf-scienceqa:old-dev", "source": "hf-scienceqa",
                "connected_family_sha256": "historical-dev-family"}
        supplement = self.supplement([(held, first), (first, second)])
        rows, remapped_held, summary = curation.apply_family_supplement(
            [first, second, sibling], [held], supplement)
        self.assertEqual(1, len({item["family_id"] for item in rows}))
        self.assertEqual(rows[0]["family_id"], remapped_held[0]["connected_family_sha256"])
        self.assertEqual(second["family_id"], rows[1]["historical_family_id"])
        for item in rows:
            self.assertFalse(item["eligible"])
            self.assertEqual(["HOLDOUT_FAMILY", "LABEL_ERROR", "NEW_TEXT_MATCH_FAMILY"],
                             item["exclusion_reasons"])
        self.assertEqual(3, summary["newly_holdout_excluded"])
        self.assertEqual(2, summary["newly_text_excluded"])
        self.assertEqual(2, summary["newly_label_excluded"])

    def test_v16_reserved_holdout_binds_exact_candidate_id_and_images(self):
        item = candidate(0, "hf-scienceqa")
        held = {"problem_id": item["problem_id"], "source_repo": "native-repo",
                "group_id": "v16-only-group", "split": "sentinel",
                "holdout_reason": "v16_existing_reserved_split",
                "image_sha256": item["image_sha256"], "pixel_sha256": item["pixel_sha256"]}
        rows, holds, summary = curation.apply_family_supplement([item], [held])
        self.assertFalse(rows[0]["eligible"])
        self.assertIn("HOLDOUT_FAMILY", rows[0]["exclusion_reasons"])
        self.assertEqual(item["family_id"], holds[0]["connected_family_sha256"])
        self.assertEqual(1, summary["derived_reserved_holdout_bindings"])
        wrong_image = held | {"image_sha256": ["different"]}
        with self.assertRaisesRegex(ValueError, "image binding mismatch"):
            curation.apply_family_supplement([item], [wrong_image])
        with self.assertRaisesRegex(ValueError, "no candidate binding"):
            curation.apply_family_supplement([item], [held | {"problem_id": "unknown"}])

    def test_family_union_refuses_unknown_or_other_source_links(self):
        first, second = candidate(0, "hf-scienceqa"), candidate(1, "hf-chartqa")
        with self.assertRaisesRegex(ValueError, "changed source"):
            curation.apply_family_supplement([first, second], [],
                                             self.supplement([(first, second)]))
        unknown = candidate(2, "hf-scienceqa")
        with self.assertRaisesRegex(ValueError, "unknown problem"):
            curation.apply_family_supplement([first], [],
                                             self.supplement([(first, unknown)]))

    def test_family_merge_can_make_exact_3000_impossible_without_relaxation(self):
        sources = ["hf-scienceqa", "hf-chartqa", "hf-mapqa", "hf-nctcrc",
                   "hf-ssmr", "hf-gvlqa", "hf-geometry3k"]
        rows = [candidate(i, sources[i // 450]) for i in range(3000)]
        self.assertEqual(3000, len(policy.choose(rows)[0]))
        merged, _, _ = curation.apply_family_supplement(
            rows, [], self.supplement([(rows[0], rows[1])]))
        with self.assertRaisesRegex(ValueError, "insufficient eligible distinct families"):
            policy.choose(merged)


class FreezeIntegrationTests(unittest.TestCase):
    def test_exact_3000_freeze_keeps_native_references_and_partial_batch(self):
        from fine_tuning.notebook_rl import contamination

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "native"
            native.mkdir()
            historical = root / "historical.json"
            curation._write(historical, {"original": True})
            holdouts = root / "holdouts.jsonl"
            curation._write(holdouts, [], jsonl=True)
            index = root / "index.json"
            curation._write(index, {"fixture": True})
            historical_ref = curation._ref(historical)
            source_counts = {"hf-ai2d": 183, "hf-scienceqa": 450, "hf-chartqa": 450,
                             "hf-mapqa": 450, "hf-nctcrc": 450, "hf-ssmr": 450,
                             "hf-gvlqa": 450, "hf-geometry3k": 317}
            rows, enriched = [], []
            for source, count in source_counts.items():
                for _ in range(count):
                    item = candidate(len(rows), source)
                    image = native / f"{len(rows)}.png"
                    image.write_bytes(f"CPU-native-image-{len(rows)}".encode())
                    item["image_paths"] = [str(image)]
                    item["image_sha256"] = [curation._sha(image)]
                    item["historical_manifest"] = historical_ref
                    item["historical_receipt"] = historical_ref
                    item["record_paths"] = {key: str(native / f"{key}.jsonl")
                                            for key in ("actor", "verifier", "lineage")}
                    rows.append(item)
                    enriched.append({"problem_id": item["problem_id"], "source": source,
                                     "native_metadata": {}, "human_solution": None,
                                     "native_source_refs": [], "annotation_refs": [],
                                     "raw_annotation": None, "human_verified": False})
            for key in ("actor", "verifier", "lineage"):
                curation._write(native / f"{key}.jsonl", [item[key] for item in rows], jsonl=True)
            audit_path = root / "audit.json"
            curation._write(audit_path, {
                "status": "PASS_HISTORICAL_BINDINGS_RECHECKED",
                "benchmark_used_only_for_exclusion": True,
                "artifacts": {"holdouts": curation._ref(holdouts)}})
            for item in rows:
                item["audit_binding"] = curation._ref(audit_path)
            input_path = root / "audited.jsonl"
            curation._write(input_path, rows, jsonl=True)
            enrichment = root / "enrichment.jsonl"
            curation._write(enrichment, enriched, jsonl=True)
            summary_path = root / "summary.json"
            curation._write(summary_path, curation._ref(input_path) | {
                "status": "PASS_HISTORICAL_BINDINGS_RECHECKED", "count": len(rows),
                "audit_path": str(audit_path), "audit_sha256": curation._sha(audit_path)})
            reviews, stats = [], {"sources": {}, "policy_version": policy.VERSION,
                                  "policy_sha256": curation._sha(Path(policy.__file__))}
            for source, count in source_counts.items():
                sample_count = 40 if count > 400 else 20
                sampled = [item for item in rows if item["source"] == source][:sample_count]
                reviews.extend(review(item) for item in sampled)
                stats["sources"].update(calibration(source, sample_count)["sources"])
            reviews_path, calibration_path = root / "reviews.jsonl", root / "calibration.json"
            curation._write(reviews_path, reviews, jsonl=True)
            curation._write(calibration_path, stats)
            draft, frozen = root / "draft", root / "frozen"
            with patch.object(curation, "INDEX_PATH", index), patch.object(
                curation, "INDEX_SHA256", curation._sha(index)
            ), patch.object(contamination, "load_index", return_value={
                "full_coverage": True, "text_cache": contamination._text_cache([])
            }):
                curation.prepare(audit_summary_path=summary_path, enrichment_path=enrichment,
                                 output=draft)
                stale = root / "stale-calibration.json"
                curation._write(stale, stats | {"policy_sha256": "0" * 64})
                with self.assertRaisesRegex(ValueError, "calibration policy mismatch"):
                    curation.freeze(prepared_path=draft / "prepared.json", reviews_path=reviews_path,
                                    calibration_path=stale, output=frozen)
                result = curation.freeze(prepared_path=draft / "prepared.json",
                                         reviews_path=reviews_path,
                                         calibration_path=calibration_path, output=frozen)
            self.assertEqual("PASS_CURATED_NATIVE_3000_WITH_LIMITS", result["status"])
            self.assertEqual(183, result["checks"]["mandatory_ai2d"])
            self.assertEqual(188, result["checks"]["batch_count"])
            self.assertEqual(8, result["checks"]["last_batch_size"])
            manifest = json.loads((frozen / "train-3000-manifest.json").read_text())
            self.assertEqual(3000, len(manifest["items"]))
            self.assertEqual(str(native / "actor.jsonl"),
                             manifest["items"][0]["record_paths"]["actor"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                curation.freeze(prepared_path=draft / "prepared.json",
                                reviews_path=reviews_path,
                                calibration_path=calibration_path, output=frozen)


if __name__ == "__main__":
    unittest.main()
