"""CPU synthetic preparation checks; no HF payloads, processor weights or GPU calls."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fine_tuning import audit, nvw, prepare_training as preparation


def record(identity, view="qa", answerable=True, split="train"):
    return {"id": identity+"-"+view, "root_id": identity, "family_id": "family-"+identity,
            "view": view, "answer_identifiable": answerable, "split": split,
            "user_text": "<image>\nQuestion", "target_text": "A", "image_paths": ["images/"+identity+".png"]}


def joint(k=(0, 0, 0), i=(0, 0, 0), s=(0, 0, 0)):
    return {track: dict(zip(("qa", "structure", "grounded"), values))
            for track, values in zip(("NVW-K", "NVW-I", "NVW-S"), (k, i, s))}


class PreparationTests(unittest.TestCase):
    def test_approved_subset_is_score_blind_deterministic_and_exact_joint_ratios(self):
        rows = [dict(record(f"trace-{index}"), source_id="trace") for index in range(9000)]
        for track, by_view in preparation.policy()["nvw_joint_views"].items():
            for view, quota in by_view.items():
                rows.extend(dict(record(f"{track}-{view}-{index}", view), nvw_track=track,
                                 source_id="nvw-independent-procedural-v1") for index in range(quota))
        source = {row["id"]: row for row in rows}
        for count in (400, 440, 1600, 2000):
            selected = preparation.select_approved_subset(rows, count)
            self.assertEqual(selected, preparation.select_approved_subset(list(reversed(rows)), count))
            self.assertEqual(len(selected), count)
            self.assertEqual(sum(row["source_id"] == "trace" for row in selected), count*9//10)
            matrix = preparation.joint_counts(selected)
            preparation.validate_joint_quotas(matrix,
                {"qa": count//20, "structure": count//40, "grounded": count//40},
                {"NVW-K": count//20, "NVW-I": count//40, "NVW-S": count//40})
            self.assertTrue(all(row == source[row["id"]] for row in selected))
        for count in (0, 360, 401, 10000, True):
            with self.assertRaises(ValueError):
                preparation.select_approved_subset(rows, count)
        with self.assertRaises(ValueError):
            preparation.select_approved_subset(rows+rows[:1], 400)
        with self.assertRaises(ValueError):
            preparation.select_approved_subset([dict(rows[0], source_id="MMMU")], 400)
        with self.assertRaises(ValueError):
            preparation.select_approved_subset(rows[:9000], 400)

    def test_subset_manifest_inherits_approval_without_changing_parent_or_inputs(self):
        from fine_tuning.modules import BASE_REVISION
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root/"parent"
            (parent/"images").mkdir(parents=True)
            Image.new("RGB", (12, 12), "red").save(parent/"images/query.png")
            rows = [dict(record(f"trace-{index}"), source_id="trace",
                         image_paths=["images/query.png"]) for index in range(9000)]
            for track, by_view in preparation.policy()["nvw_joint_views"].items():
                for view, quota in by_view.items():
                    rows.extend(dict(record(f"{track}-{view}-{index}", view), nvw_track=track,
                                     source_id="nvw-independent-procedural-v1",
                                     image_paths=["images/query.png"]) for index in range(quota))
            rows = preparation.order(rows)
            delta_rows = [row for row in rows if row["source_id"] == "trace"][:2]
            reused = [row for row in rows if row not in delta_rows]
            audit.write_rows(parent/"parent_reapproved/records.jsonl", reused)
            delta_root = parent/"delta_000"
            audit.write_rows(delta_root/"proposed_records.jsonl", delta_rows)
            audit.write_rows(delta_root/"approved/records.jsonl", delta_rows)
            audit.dump(delta_root/"processor_preflight.json", {"fixture": "closed delta processor"})
            audit.dump(delta_root/"approved/manifest.json", {"fixture": "closed delta approval"})
            delta = {"accepted": len(delta_rows), "quarantined": 0,
                "proposed_sha256": audit.sha(delta_root/"proposed_records.jsonl"),
                "approved_records_sha256": audit.sha(delta_root/"approved/records.jsonl"),
                "processor_receipt_sha256": audit.sha(delta_root/"processor_preflight.json"),
                "approved_manifest_sha256": audit.sha(delta_root/"approved/manifest.json")}
            audit.write_rows(parent/"approved/records.jsonl", rows)
            audit.write_rows(parent/"canonical_private.jsonl", [{"fixture": True}])
            completed = {"status": "READY", "stage": "pilot", "approved_records": 10000,
                "parent_approved_reused": len(reused), "new_approved_trace": len(delta_rows), "deltas": [delta],
                "canonical_private_sha256": audit.sha(parent/"canonical_private.jsonl"),
                "source_binding": {"current": {"fixture": True}},
                "parent_files_sha256": {"processor_preflight.json": "0"*64,
                    "approved/records.jsonl": audit.sha(parent/"parent_reapproved/records.jsonl")},
                "parent_quarantine_excluded": []}
            audit.dump(parent/"completion_receipt.json", completed)
            receipt_hash = audit.sha(parent/"completion_receipt.json")
            manifest = {"stage": "pilot", "record_count": 10000,
                "same_records_order_and_images_for_all_arms": True,
                "module_arms": preparation.policy()["module_arms"],
                "files": {"images/query.png": audit.sha(parent/"images/query.png")},
                "canonical_private_sha256": audit.sha(parent/"canonical_private.jsonl"),
                "source_sha256": {"fixture": True}, "preparation_receipt_sha256": receipt_hash,
                "contamination_audit": {"composite_approval_receipt_sha256": receipt_hash},
                "model_input_audit": {"status": "PASS", "kind": "COMPOSITE_PROCESSOR_RECEIPTS",
                    "record_count": 10000, "processor_revision": BASE_REVISION,
                    "sequence_budget": 4096, "episode_pixels": 1048576,
                    "parent_approved_records": len(reused), "delta_approved_records": len(delta_rows),
                    "parent_receipt_sha256": "0"*64,
                    "delta_receipts_sha256": [delta["processor_receipt_sha256"]]}}
            audit.dump(parent/"common_manifest.json", manifest)
            names = ("approved/records.jsonl", "canonical_private.jsonl", "common_manifest.json", "completion_receipt.json")
            before = {name: audit.sha(parent/name) for name in names}
            with patch("fine_tuning.train.approved_records", return_value=(rows, manifest)):
                scale = preparation.subset_pilot(parent, root/"subset", 400, "Operating-budget test fixture")
            result = json.loads((root/"subset/common_manifest.json").read_text())
            ledger = root/"subset/approved/records.jsonl"
            self.assertEqual((scale["original_records"], scale["selected_records"]), (10000, 400))
            self.assertEqual(result["records_path"], str(ledger.resolve()))
            self.assertEqual(result["records_sha256"], audit.sha(ledger))
            self.assertEqual(result["model_input_audit"]["kind"], "INHERITED_APPROVED_SUBSET")
            self.assertFalse(result["model_input_audit"]["processor_reexecuted"])
            self.assertFalse(result["pilot_scale"]["original_10K_execution_completed"])
            source = {row["id"]: row for row in rows}
            self.assertTrue(all(row == source[row["id"]] for row in audit.read_rows(ledger)))
            self.assertEqual(before, {name: audit.sha(parent/name) for name in names})
            self.assertEqual(result["files"], manifest["files"])
            subset_receipt = json.loads((root/"subset/subset_receipt.json").read_text())
            for name, digest in subset_receipt["component_files_sha256"].items():
                copied = root/"subset/component_evidence"/name
                self.assertEqual(copied.read_bytes(), (parent/name).read_bytes())
                self.assertEqual(audit.sha(copied), digest)
            smoke_rows = [dict(row, source_kind="INDEPENDENT_SYNTHETIC", parents=[],
                               lineage=["independent-nvw-v1"], training_allowed=True,
                               benchmark_derived=False) for row in audit.read_rows(ledger)
                          if row["source_id"] == "nvw-independent-procedural-v1"]
            smoke_path = root/"smoke_records.jsonl"
            audit.write_rows(smoke_path, smoke_rows)
            smoke_bytes = smoke_path.read_bytes()
            smoke_roots = {row["root_id"] for row in smoke_rows}
            with patch("fine_tuning.train.approved_records", return_value=(rows, manifest)):
                reduced = preparation.subset_pilot(parent, root/"disjoint", 400, "Disjoint smoke roots", smoke_path)
            disjoint = audit.read_rows(root/"disjoint/approved/records.jsonl")
            self.assertFalse(smoke_roots & {row["root_id"] for row in disjoint})
            preparation.validate_joint_quotas(preparation.joint_counts(disjoint),
                {"qa": 20, "structure": 10, "grounded": 10}, {"NVW-K": 20, "NVW-I": 10, "NVW-S": 10})
            exclusion = reduced["smoke_exclusion"]
            self.assertEqual(exclusion["records_sha256"], audit.sha(smoke_path))
            self.assertEqual(exclusion["excluded_parent_records"], len(smoke_rows))
            self.assertEqual((root/"disjoint"/exclusion["evidence_path"]).read_bytes(), smoke_bytes)
            self.assertEqual(before, {name: audit.sha(parent/name) for name in names})
            unavailable = [dict(row, source_kind="INDEPENDENT_SYNTHETIC", parents=[],
                                lineage=["independent-nvw-v1"], training_allowed=True,
                                benchmark_derived=False) for row in rows
                           if row.get("nvw_track") == "NVW-I" and row["view"] == "structure"]
            smoke_path.write_text("".join(json.dumps(row)+"\n" for row in unavailable))
            with patch("fine_tuning.train.approved_records", return_value=(rows, manifest)):
                with self.assertRaisesRegex(ValueError, "track/view shortfall"):
                    preparation.subset_pilot(parent, root/"no-borrow", 400, "Must reject", smoke_path)
            self.assertFalse((root/"no-borrow").exists())
            for key, value in (("split", "development"), ("source_id", "trace"),
                               ("benchmark_derived", True), ("root_id", ""), ("training_allowed", False)):
                smoke_path.write_text(json.dumps(dict(smoke_rows[0], **{key: value}))+"\n")
                with patch("fine_tuning.train.approved_records", return_value=(rows, manifest)):
                    with self.assertRaisesRegex(ValueError, "Smoke exclusion"):
                        preparation.subset_pilot(parent, root/f"bad-smoke-{key}", 400, "Must reject", smoke_path)
            smoke_path.write_bytes(smoke_bytes)
            original_copy = preparation.copy_image
            def mutate_exclusion(source, destination, name):
                if Path(destination).name == "smoke_exclusion":
                    smoke_path.write_bytes(smoke_bytes+b"\n")
                return original_copy(source, destination, name)
            with patch("fine_tuning.train.approved_records", return_value=(rows, manifest)), \
                    patch.object(preparation, "copy_image", side_effect=mutate_exclusion):
                with self.assertRaisesRegex(ValueError, "Smoke exclusion ledger changed"):
                    preparation.subset_pilot(parent, root/"mutated-smoke", 400, "Must reject", smoke_path)
            self.assertFalse((root/"mutated-smoke/common_manifest.json").exists())
            smoke_path.write_bytes(smoke_bytes)
            with self.assertRaises(ValueError):
                preparation.subset_pilot(parent, root/"subset", 400, "Cannot overwrite")
            for key, value in (("status", "FAIL"), ("record_count", 9975), ("processor_revision", "wrong")):
                bad = dict(manifest, model_input_audit=dict(manifest["model_input_audit"], **{key: value}))
                with patch("fine_tuning.train.approved_records", return_value=(rows, bad)):
                    with self.assertRaisesRegex(ValueError, "processor/completion"):
                        preparation.subset_pilot(parent, root/f"bad-{key}", 400, "Must reject")
            bad = dict(manifest, preparation_receipt_sha256="f"*64)
            with patch("fine_tuning.train.approved_records", return_value=(rows, bad)):
                with self.assertRaisesRegex(ValueError, "processor/completion"):
                    preparation.subset_pilot(parent, root/"bad-completion", 400, "Must reject")
            for component in (parent/"parent_reapproved/records.jsonl", delta_root/"approved/records.jsonl"):
                original = component.read_bytes()
                component.write_bytes(original+b"\n")
                with patch("fine_tuning.train.approved_records", return_value=(rows, manifest)):
                    with self.assertRaisesRegex(ValueError, "bytes changed"):
                        preparation.subset_pilot(parent, root/"bad-component", 400, "Must reject")
                component.write_bytes(original)
            completion_path = parent/"completion_receipt.json"
            original_completion = completion_path.read_bytes()
            wrong_count = json.loads(original_completion)
            wrong_count["deltas"][0]["accepted"] += 1
            completion_path.write_text(json.dumps(wrong_count))
            wrong_hash = audit.sha(completion_path)
            bad = dict(manifest, preparation_receipt_sha256=wrong_hash,
                       contamination_audit={"composite_approval_receipt_sha256": wrong_hash})
            with patch("fine_tuning.train.approved_records", return_value=(rows, bad)):
                with self.assertRaisesRegex(ValueError, "components count"):
                    preparation.subset_pilot(parent, root/"bad-count", 400, "Must reject")
            completion_path.write_bytes(original_completion)
            # Recomputing the final ledger hash must not rescue an altered target.
            tampered = [dict(row) for row in rows]
            tampered[0]["target_text"] = "UNAPPROVED TARGET"
            final_ledger = parent/"approved/records.jsonl"
            final_ledger.write_text("".join(json.dumps(row)+"\n" for row in tampered))
            bad = dict(manifest, records_sha256=audit.sha(final_ledger))
            (parent/"common_manifest.json").write_text(json.dumps(bad))
            with patch("fine_tuning.train.approved_records", return_value=(tampered, bad)):
                with self.assertRaisesRegex(ValueError, "approved components"):
                    preparation.subset_pilot(parent, root/"bad-target", 400, "Must reject")
            self.assertFalse((root/"bad-target").exists())

    def test_nvw_quota_uses_distinct_unchanged_views_and_excludes_ambiguous_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [record(name, view, name != "ambiguous")
                    for name in ("a", "b", "ambiguous") for view in ("qa", "structure", "grounded")]
            audit.write_rows(root/"prototype_train.jsonl", rows)
            audit.write_rows(root/"canonical_private.jsonl", [
                {"root_id": "a", "track": "NVW-K"}, {"root_id": "b", "track": "NVW-S"},
                {"root_id": "ambiguous", "track": "NVW-I"}])
            audit.dump(root/"manifest.json", {"fixture": True})
            with patch.object(nvw, "verify_dataset", return_value={"status": "FIXTURE"}) as verification:
                selected, receipt = preparation.select_nvw(root, {"qa": 2, "structure": 1, "grounded": 1},
                    joint_quotas=joint(k=(1, 0, 0), i=(0, 1, 1), s=(1, 0, 0)))
                verification.assert_called_once_with(root)
            self.assertEqual(receipt["selected_views"], {"qa": 2, "structure": 1, "grounded": 1})
            self.assertEqual(receipt["available_views"]["qa"], 2)
            source = {row["id"]: row for row in rows}
            for row in selected:
                for key in ("user_text", "target_text", "image_paths"):
                    self.assertEqual(row[key], source[row["id"]][key])
                self.assertNotIn(row["nvw_track"], row["user_text"])
                self.assertEqual(row["nvw_track_source_sha256"], audit.sha(root/"canonical_private.jsonl"))
            self.assertTrue(all(row["answer_identifiable"] for row in selected if row["view"] == "qa"))
            with patch.object(nvw, "verify_dataset", return_value={}):
                with self.assertRaisesRegex(ValueError, "quota shortfall"):
                    preparation.select_nvw(root, {"qa": 3, "structure": 0, "grounded": 0},
                        joint_quotas=joint(k=(1, 0, 0), i=(1, 0, 0), s=(1, 0, 0)))

    def test_nvw_entire_split_universe_checked_before_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit.write_rows(root/"prototype_train.jsonl", [record("a")])
            audit.write_rows(root/"prototype_development.jsonl", [dict(record("a"), id="dev-a", split="development")])
            with patch.object(nvw, "verify_dataset", return_value={}):
                with self.assertRaisesRegex(ValueError, "leaks"):
                    preparation.select_nvw(root, {"qa": 1, "structure": 0, "grounded": 0},
                        joint_quotas=joint(k=(1, 0, 0)))

    def test_exclusive_regular_copy_reuses_same_bytes_and_rejects_collision_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root/"source", root/"destination"
            (source/"images").mkdir(parents=True)
            destination.mkdir()
            Image.new("RGB", (12, 12), "red").save(source/"images/query.png")
            checksum = preparation.copy_image(source, destination, "images/query.png")
            self.assertEqual(preparation.copy_image(source, destination, "images/query.png"), checksum)
            self.assertFalse((destination/"images/query.png").is_symlink())
            Image.new("RGB", (12, 12), "blue").save(source/"images/query.png")
            with self.assertRaisesRegex(ValueError, "collision"):
                preparation.copy_image(source, destination, "images/query.png")
            (destination/"images/alias.png").symlink_to(destination/"images/query.png")
            Image.new("RGB", (12, 12), "blue").save(source/"images/alias.png")
            with self.assertRaises(ValueError):
                preparation.copy_image(source, destination, "images/alias.png")

    def test_trace_cache_receipt_exact_coverage_no_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root/"cache"
            source = cache/"datasets--maveryn--trace/snapshots"/("a"*40)/"data/train.parquet"
            source.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([{"fixture": 1}]), source)
            receipt = {"repo_id": "maveryn/trace", "revision": "a"*40, "files": [{
                "path": "data/train.parquet", "bytes": source.stat().st_size, "sha256": audit.sha(source)}]}
            audit.dump(root/"downloads.json", receipt)
            self.assertEqual(preparation.trace_paths(root/"downloads.json", cache), [source])
            source.write_bytes(source.read_bytes()+b"corrupt")
            with self.assertRaisesRegex(ValueError, "differs"):
                preparation.trace_paths(root/"downloads.json", cache)
            receipt["files"][0]["path"] = "../escape.parquet"
            audit.dump(root/"traversal.json", receipt)
            with self.assertRaisesRegex(ValueError, "traversing"):
                preparation.trace_paths(root/"traversal.json", cache)

    def test_index_reuses_exclusion_builder_with_exact_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coverage = {"files": [{"path": str(root/"one.parquet"), "records": 1}], "record_count": 1}
            audit.dump(root/"coverage.json", coverage)
            with patch.object(preparation.sources, "benchmark_index", return_value={"status": "FIXTURE"}) as build:
                preparation.build_exclusion_index(root/"coverage.json", root/"index.json")
                build.assert_called_once_with([root/"one.parquet"], coverage, root/"index.json")

    def test_real_released_trace_train_filename_and_validation_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root/"cache"
            name = "data/train/trace_rlvr_train_64000_all1000_seed42-00000-of-00016.parquet"
            source = cache/"datasets--maveryn--trace/snapshots"/("a"*40)/name
            source.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([{"fixture": 1}]), source)
            receipt = {"repo_id": "maveryn/trace", "revision": "a"*40, "files": [{
                "path": name, "bytes": source.stat().st_size, "sha256": audit.sha(source)}]}
            audit.dump(root/"downloads.json", receipt)
            self.assertEqual(preparation.trace_paths(root/"downloads.json", cache), [source])
            receipt["files"][0]["path"] = name.replace("data/train/", "data/validation/")
            audit.dump(root/"validation.json", receipt)
            with self.assertRaisesRegex(ValueError, "receipt path"):
                preparation.trace_paths(root/"validation.json", cache)

    def test_freeze_does_not_force_count_from_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(record("a"), nvw_track="NVW-K"), dict(record("b"), nvw_track="NVW-K")]
            audit.write_rows(root/"proposed.jsonl", rows)
            audit.dump(root/"registry.json", {"sources": {"test": {"source_known": True}}})
            def preflight(*args, **kwargs):
                audit.dump(root/"processor_preflight.json", {"processor_revision": "fixture"})
                return rows, {"processor_revision": "fixture"}
            def approved(*args, **kwargs):
                folder = root/"approved"
                folder.mkdir()
                audit.write_rows(folder/"records.jsonl", rows[:1])
                return {"record_count": 1, "contamination_audit": {"quarantined": 1}}
            with patch.object(audit, "check_lineage"), patch.object(audit, "check_target_identifiability"), \
                    patch.object(audit, "preflight_records", side_effect=preflight), \
                    patch.object(audit, "approve", side_effect=approved):
                with self.assertRaisesRegex(ValueError, "quota shortfall"):
                    preparation.freeze_approved(root/"proposed.jsonl", root, root/"registry.json",
                        root/"index.json", root/"model", root, expected_records=2,
                        selection={"required_joint_views": joint(k=(2, 0, 0))})
            receipt = json.loads((root/"preparation_receipt.json").read_text())
            self.assertEqual(receipt["status"], "INCOMPLETE_QUOTA")
            self.assertFalse(receipt["training_allowed"])
            self.assertFalse((root/"common_manifest.json").exists())

    def test_one_frozen_order_manifest_for_all_four_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = preparation.order([dict(record("b"), nvw_track="NVW-S"),
                                      dict(record("a"), nvw_track="NVW-K")])
            audit.write_rows(root/"proposed.jsonl", rows)
            audit.dump(root/"registry.json", {"test": {"source_known": True}})
            def preflight(*args, **kwargs):
                audit.dump(root/"processor_preflight.json", {"processor_revision": "fixture"})
                return rows, {"processor_revision": "fixture"}
            def approved(*args, **kwargs):
                folder = root/"approved"
                folder.mkdir()
                audit.write_rows(folder/"records.jsonl", rows)
                return {"record_count": 2, "contamination_audit": {"quarantined": 0},
                        "records_sha256": audit.sha(folder/"records.jsonl")}
            with patch.object(audit, "check_lineage"), patch.object(audit, "check_target_identifiability"), \
                    patch.object(audit, "preflight_records", side_effect=preflight), \
                    patch.object(audit, "approve", side_effect=approved):
                preparation.freeze_approved(root/"proposed.jsonl", root, root/"registry.json",
                    root/"index.json", root/"model", root, expected_records=2,
                    selection={"required_joint_views": joint(k=(1, 0, 0), s=(1, 0, 0))})
            manifest = json.loads((root/"common_manifest.json").read_text())
            self.assertEqual(manifest["module_arms"], ["FL", "FM", "FV", "FML"])
            self.assertTrue(manifest["same_records_order_and_images_for_all_arms"])
            self.assertEqual(manifest["record_order_sha256"], hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest())
            self.assertIn("not the original professional mixture", manifest["mixture_policy"]["substitution"])
            self.assertEqual(manifest["nvw_tracks"], {"NVW-K": 1, "NVW-I": 0, "NVW-S": 1})

    def test_joint_matrix_validates_track_and_view_axes(self):
        configuration = preparation.policy()
        preparation.validate_joint_quotas(configuration["nvw_joint_views"], configuration["nvw_views"],
                                           configuration["nvw_tracks"])
        bad = joint(k=(500, 250, 250))
        with self.assertRaisesRegex(ValueError, "track quota marginals"):
            preparation.validate_joint_quotas(bad, configuration["nvw_views"], configuration["nvw_tracks"])
        with self.assertRaisesRegex(ValueError, "view quota marginals"):
            preparation.validate_joint_quotas(joint(k=(1, 0, 0)), {"qa": 0, "structure": 1, "grounded": 0})

    def test_track_shortfall_does_not_borrow_available_qa_from_other_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit.write_rows(root/"prototype_train.jsonl", [record("a"), record("b")])
            audit.write_rows(root/"canonical_private.jsonl", [
                {"root_id": "a", "track": "NVW-K"}, {"root_id": "b", "track": "NVW-S"}])
            with patch.object(nvw, "verify_dataset", return_value={}):
                with self.assertRaisesRegex(ValueError, "NVW-I/qa.*available 0.*no cross-track borrowing"):
                    preparation.select_nvw(root, {"qa": 1, "structure": 0, "grounded": 0},
                                           joint_quotas=joint(i=(1, 0, 0)))

    def test_equal_total_but_wrong_track_counts_blocks_common_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(record("a"), nvw_track="NVW-K"), dict(record("b"), nvw_track="NVW-K")]
            audit.write_rows(root/"proposed.jsonl", rows)
            audit.dump(root/"registry.json", {"test": {"source_known": True}})
            def preflight(*args, **kwargs):
                audit.dump(root/"processor_preflight.json", {"processor_revision": "fixture"})
                return rows, {"processor_revision": "fixture"}
            def approved(*args, **kwargs):
                (root/"approved").mkdir()
                audit.write_rows(root/"approved/records.jsonl", rows)
                return {"record_count": 2, "contamination_audit": {"quarantined": 0}}
            with patch.object(audit, "check_lineage"), patch.object(audit, "check_target_identifiability"), \
                    patch.object(audit, "preflight_records", side_effect=preflight), \
                    patch.object(audit, "approve", side_effect=approved):
                with self.assertRaisesRegex(ValueError, "quota shortfall"):
                    preparation.freeze_approved(root/"proposed.jsonl", root, root/"registry.json",
                        root/"index.json", root/"model", root, expected_records=2,
                        selection={"required_joint_views": joint(k=(1, 0, 0), i=(1, 0, 0))})
            receipt = json.loads((root/"preparation_receipt.json").read_text())
            self.assertEqual(receipt["approved_records"], 2)
            self.assertEqual(receipt["approved_nvw_tracks"]["NVW-I"], 0)
            self.assertFalse((root/"common_manifest.json").exists())

    def disjoint_fixture(self, root):
        """Closed-shaped CPU fixture: real manifest/lineage checks, mocked processor only."""
        from fine_tuning.modules import BASE_REVISION
        parent, pool, smoke = root/"full_parent", root/"pool", root/"smoke"
        for folder in (parent, pool, smoke):
            (folder/"images").mkdir(parents=True)
            Image.new("RGB", (12, 12), "blue").save(folder/"images/shared.png")
        registered = {name: {"source_known": True, "rights_approved": True,
            "benchmark_derived": False, "parents": []} for name in ("independent-nvw-v1", "trace")}
        registered["independent-nvw-v1"]["generator_sha256"] = "generator-fixture"
        audit.dump(root/"registry.json", registered)
        coverage = {"files": [{"path": str(root/"excluded"), "records": 1}], "record_count": 1}
        audit.dump(root/"index.json", {"status": "COMPLETE", "coverage_verified": True,
            "coverage": coverage, "record_count": 1, "image_count": 1,
            "source_files": {str(root/"excluded"): "fixture"}, "per_file_record_counts": {str(root/"excluded"): 1}})
        episodes = {}
        def own(identity, view="qa", track="NVW-K", split="train"):
            family, world = "family-"+identity, {"fixture": identity}
            root_id = nvw.digest([family, world])
            episodes[root_id] = {"root_id": root_id, "family_id": family, "world": world, "track": track,
                "image": "images/shared.png", "image_sha256": audit.sha(parent/"images/shared.png")}
            return dict(record(identity, view, split=split), root_id=root_id, source_id="nvw-independent-procedural-v1",
                source_kind="INDEPENDENT_SYNTHETIC", parents=[], lineage=["independent-nvw-v1"],
                image_paths=["images/shared.png"], benchmark_derived=False, training_allowed=False,
                supervised_target_identifiable=True, quality_status="SOURCE_GOLD", nvw_track=track)
        raw = [dict(record(f"trace-{i}"), source_id="trace", lineage=["trace"],
               image_paths=["images/shared.png"], benchmark_derived=False, training_allowed=False,
               supervised_target_identifiable=True, quality_status="SOURCE_GOLD") for i in range(9000)]
        own_rows = [own(f"own-{track}-{view}-{i}", view, track)
                    for track, values in preparation.policy()["nvw_joint_views"].items()
                    for view, count in values.items() for i in range(count)]
        rows = preparation.order([dict(audit.model_input_record(row), training_allowed=True,
                                      processor_receipt_sha256="a"*64) for row in raw+own_rows])
        audit.write_rows(parent/"approved/records.jsonl", rows)
        audit.write_rows(parent/"parent_reapproved/records.jsonl", rows)
        canon = [episodes[r["root_id"]] for r in own_rows]
        audit.write_rows(parent/"canonical_private.jsonl", canon)
        delta = parent/"delta_000"
        audit.write_rows(delta/"proposed_records.jsonl", [])
        audit.write_rows(delta/"approved/records.jsonl", [])
        audit.dump(delta/"approved/manifest.json", {"fixture": True})
        audit.dump(delta/"processor_preflight.json", {"status": "PASS", "kind": "ACTUAL_PROCESSOR_PREFLIGHT",
            "processor_revision": BASE_REVISION, "processor_class": "Qwen3VLProcessor", "truncation": False,
            "chat_template_sha256": "fixture-template", "sequence_budget": 4096, "episode_pixels": 1048576,
            "producer_sha256": audit.sha(audit.__file__), "records_sha256": audit.sha(delta/"proposed_records.jsonl")})
        inherited_delta = {"proposed_sha256": audit.sha(delta/"proposed_records.jsonl"),
            "processor_receipt_sha256": audit.sha(delta/"processor_preflight.json"),
            "approved_manifest_sha256": audit.sha(delta/"approved/manifest.json"),
            "approved_records_sha256": audit.sha(delta/"approved/records.jsonl"), "accepted": 0, "quarantined": 0}
        current = {n: audit.sha(Path(preparation.__file__).with_name(n)) for n in ("train.py", "modules.py")}
        completion = {"status": "READY", "stage": "pilot", "approved_records": 10000,
            "parent_approved_reused": 10000, "new_approved_trace": 0, "deltas": [inherited_delta],
            "canonical_private_sha256": audit.sha(parent/"canonical_private.jsonl"),
            "source_binding": {"current": current}, "parent_quarantine_excluded": [],
            "parent_files_sha256": {"processor_preflight.json": "a"*64,
                "approved/records.jsonl": audit.sha(parent/"parent_reapproved/records.jsonl")}}
        audit.dump(parent/"completion_receipt.json", completion)
        receipt_hash = audit.sha(parent/"completion_receipt.json")
        manifest = {"status": "APPROVED", "stage": "pilot", "record_count": 10000,
            "training_allowed": True, "benchmark_derived": False, "known_parent_ids": list(registered),
            "sources": registered, "source_sha256": current, "records_sha256": audit.sha(parent/"approved/records.jsonl"),
            "files": {"images/shared.png": audit.sha(parent/"images/shared.png")},
            "same_records_order_and_images_for_all_arms": True, "module_arms": preparation.policy()["module_arms"],
            "nvw_joint_views": preparation.policy()["nvw_joint_views"], "nvw_tracks": preparation.policy()["nvw_tracks"],
            "canonical_private_sha256": audit.sha(parent/"canonical_private.jsonl"),
            "preparation_receipt_sha256": receipt_hash,
            "contamination_audit": {"status": "PASS", "index_sha256": audit.sha(root/"index.json"),
                                    "composite_approval_receipt_sha256": receipt_hash, "quarantined": 0},
            "model_input_audit": {"status": "PASS", "kind": "COMPOSITE_PROCESSOR_RECEIPTS", "record_count": 10000,
                "processor_revision": BASE_REVISION, "sequence_budget": 4096, "episode_pixels": 1048576,
                "parent_approved_records": 10000, "delta_approved_records": 0, "parent_receipt_sha256": "a"*64,
                "delta_receipts_sha256": [inherited_delta["processor_receipt_sha256"]]}}
        audit.dump(parent/"common_manifest.json", manifest)
        selected = [next(r for r in own_rows if r["nvw_track"] == track and r["view"] == view)
                    for track, view in (("NVW-K", "qa"), ("NVW-I", "structure"), ("NVW-S", "grounded"))]
        smoke_rows = [dict(audit.model_input_record(own(identity, view)), training_allowed=True)
                      for identity in [r["id"][:-len(r["view"])-1] for r in selected]+[f"smoke-{i}" for i in range(117)]
                      for view in ("qa", "structure", "grounded")]
        audit.write_rows(smoke/"records.jsonl", smoke_rows)
        audit.write_rows(smoke/"canonical_private.jsonl", [episodes[root_id]
            for root_id in sorted({r["root_id"] for r in smoke_rows})])
        audit.dump(smoke/"smoke_manifest.json", {"status": "APPROVED", "stage": "smoke",
            "image_root": str(smoke),
            "training_allowed": True, "benchmark_derived": False, "record_count": 360,
            "records_sha256": audit.sha(smoke/"records.jsonl"), "source_sha256": current,
            "contamination_audit": {"status": "PASS"}, "model_input_audit": {"status": "PASS", "record_count": 360,
                "receipt_sha256": "b"*64, "processor_revision": BASE_REVISION,
                "sequence_budget": 4096, "episode_pixels": 1048576}})
        candidates = [own(f"replacement-{t}-{v}-{i}", v, t) for t,v in
                      (("NVW-K", "qa"), ("NVW-I", "structure"), ("NVW-S", "grounded")) for i in range(2)]
        audit.write_rows(pool/"prototype_train.jsonl", candidates)
        development, holdout = own("dev", split="development"), own("holdout", split="sealed_holdout")
        audit.write_rows(pool/"prototype_development.jsonl", [development])
        audit.write_rows(pool/"prototype_sealed_holdout.jsonl", [holdout])
        audit.write_rows(pool/"canonical_private.jsonl", [episodes[r["root_id"]] for r in candidates+[development, holdout]])
        audit.dump(pool/"manifest.json", {"generator_sha256": "generator-fixture"})
        model = root/BASE_REVISION
        model.mkdir()
        self.enterContext(patch.object(nvw, "verify_dataset", return_value={"status": "CPU_FIXTURE"}))
        def processor(records_path, image_root, model_path, output, **kwargs):
            proposed = audit.read_rows(records_path)
            receipt = {"status": "PASS", "kind": "ACTUAL_PROCESSOR_PREFLIGHT",
                "processor_revision": BASE_REVISION, "processor_class": "Qwen3VLProcessor",
                "chat_template_sha256": "fixture-template", "truncation": False,
                "sequence_budget": 4096, "episode_pixels": 1048576, "producer_sha256": audit.sha(audit.__file__),
                "records_sha256": audit.sha(records_path), "records": {r["id"]: {
                    "status": "PASS", "input_sha256": audit.input_fingerprint(r, image_root), "assistant_tokens": 1,
                    "sequence_tokens": 3, "image_grid_thw": [[1, 1, 1]], "tensor_sha256": {k: "CPU-fixture" for k in
                    ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")}} for r in proposed}}
            audit.dump(output, receipt)
            return proposed, receipt
        preflight = self.enterContext(patch.object(audit, "preflight_records", side_effect=processor))
        return parent, pool, smoke, model, preflight

    def full_disjoint(self, root, parent, pool, smoke, model, output):
        return preparation.disjoint_full_pilot(parent, pool, smoke/"records.jsonl", smoke/"smoke_manifest.json",
            root/"registry.json", root/"index.json", model, output)

    def test_full_disjoint_replaces_nvw_only_preserves_exact_10k_and_supports_future_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, pool, smoke, model, preflight = self.disjoint_fixture(root)
            before = {str(p.relative_to(parent)): audit.sha(p) for p in parent.rglob("*") if p.is_file()}
            original = audit.read_rows(parent/"approved/records.jsonl")
            output = root/"repaired"
            receipt = self.full_disjoint(root, parent, pool, smoke, model, output)
            final = audit.read_rows(output/"approved/records.jsonl")
            self.assertEqual((len(final), receipt["parent_approved_reused"], receipt["new_approved_nvw"]), (10000, 9997, 3))
            self.assertEqual(preparation.joint_counts(final), preparation.policy()["nvw_joint_views"])
            self.assertEqual([r for r in original if r["source_id"] == "trace"], [r for r in final if r["source_id"] == "trace"])
            originals = {r["id"]: r for r in original}
            self.assertTrue(all(r == originals[r["id"]] for r in final if r["id"] in originals))
            self.assertFalse({r["root_id"] for r in final} & {r["root_id"] for r in audit.read_rows(smoke/"records.jsonl")})
            self.assertEqual(before, {str(p.relative_to(parent)): audit.sha(p) for p in parent.rglob("*") if p.is_file()})
            self.assertEqual(preflight.call_count, 1)
            self.assertEqual(len(audit.read_rows(preflight.call_args.args[0])), 3)
            from fine_tuning.train import approved_records
            approved_records(output/"approved/records.jsonl", output/"common_manifest.json", output)
            preparation.subset_pilot(output, root/"future-subset", 400, "Separate smaller pilot")
            self.assertEqual(len(audit.read_rows(root/"future-subset/approved/records.jsonl")), 400)

    def test_full_disjoint_never_waives_replacement_quarantine_or_missing_processor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, pool, smoke, model, _preflight = self.disjoint_fixture(root)
            with patch.object(audit, "contamination_candidates", return_value=["NEAR_IMAGE_REVIEW_REQUIRED"]):
                with self.assertRaisesRegex(ValueError, "quarantine is never waived"):
                    self.full_disjoint(root, parent, pool, smoke, model, root/"rejected")
            self.assertFalse((root/"rejected/common_manifest.json").exists())
            rejected = json.loads((root/"rejected/completion_receipt.json").read_text())
            self.assertFalse(rejected["training_allowed"])
            with self.assertRaisesRegex(ValueError, "snapshot is unavailable"):
                self.full_disjoint(root, parent, pool, smoke, root/"missing-model", root/"pending")
            draft = json.loads((root/"pending/candidate_preparation.json").read_text())
            self.assertFalse(draft["training_allowed"])
            self.assertFalse((root/"pending/common_manifest.json").exists())

    def test_full_disjoint_rejects_changed_approval_pool_source_and_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, pool, smoke, model, preflight = self.disjoint_fixture(root)
            target = parent/"parent_reapproved/records.jsonl"
            data = target.read_bytes()
            target.write_bytes(data+b"\n")
            with self.assertRaisesRegex(ValueError, "component bytes changed"):
                self.full_disjoint(root, parent, pool, smoke, model, root/"bad-parent")
            target.write_bytes(data)
            ledger = smoke/"records.jsonl"
            original = ledger.read_bytes()
            ledger.write_bytes(original+b"\n")
            with self.assertRaisesRegex(ValueError, "smoke ledger/manifest"):
                self.full_disjoint(root, parent, pool, smoke, model, root/"bad-smoke")
            ledger.write_bytes(original)
            (pool/"manifest.json").write_text(json.dumps({"generator_sha256": "unapproved"}))
            with self.assertRaisesRegex(ValueError, "rights-approved source"):
                self.full_disjoint(root, parent, pool, smoke, model, root/"bad-generator")
            self.assertFalse(preflight.called)
            (pool/"manifest.json").write_text(json.dumps({"generator_sha256": "generator-fixture"}))
            processor = preflight.side_effect
            def mutate_smoke(*args, **kwargs):
                result = processor(*args, **kwargs)
                ledger.write_bytes(original+b"\n")
                return result
            preflight.side_effect = mutate_smoke
            with self.assertRaisesRegex(ValueError, "Smoke exclusion ledger/manifest changed"):
                self.full_disjoint(root, parent, pool, smoke, model, root/"mutable-smoke")
            self.assertFalse((root/"mutable-smoke/common_manifest.json").exists())

    def test_full_disjoint_rejects_coherently_renamed_canonical_roots(self):
        for boundary in ("parent", "pool", "smoke"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                parent, pool, smoke, model, preflight = self.disjoint_fixture(root)
                folder = {"parent": parent, "pool": pool, "smoke": smoke}[boundary]
                canonical_path = folder/"canonical_private.jsonl"
                canonical = audit.read_rows(canonical_path)
                original_root = canonical[0]["root_id"]
                canonical[0]["root_id"] = "0"*64
                canonical_path.write_text("".join(json.dumps(episode)+"\n" for episode in canonical))
                ledgers = {"parent": [parent/"approved/records.jsonl", parent/"parent_reapproved/records.jsonl"],
                           "pool": [pool/"prototype_train.jsonl"], "smoke": [smoke/"records.jsonl"]}[boundary]
                # Rename both payload and its canonical owner, then recompute the ledger/hash chain.
                for path in ledgers:
                    rows = audit.read_rows(path)
                    path.write_text("".join(json.dumps(dict(row, root_id="0"*64) if row["root_id"] == original_root else row)+"\n"
                                            for row in rows))
                if boundary == "parent":
                    completion = json.loads((parent/"completion_receipt.json").read_text())
                    completion["canonical_private_sha256"] = audit.sha(canonical_path)
                    completion["parent_files_sha256"]["approved/records.jsonl"] = audit.sha(ledgers[1])
                    (parent/"completion_receipt.json").write_text(json.dumps(completion))
                    manifest = json.loads((parent/"common_manifest.json").read_text())
                    receipt_sha = audit.sha(parent/"completion_receipt.json")
                    manifest.update(records_sha256=audit.sha(ledgers[0]), canonical_private_sha256=audit.sha(canonical_path),
                                    preparation_receipt_sha256=receipt_sha)
                    manifest["contamination_audit"]["composite_approval_receipt_sha256"] = receipt_sha
                    (parent/"common_manifest.json").write_text(json.dumps(manifest))
                elif boundary == "smoke":
                    manifest = json.loads((smoke/"smoke_manifest.json").read_text())
                    manifest["records_sha256"] = audit.sha(ledgers[0])
                    (smoke/"smoke_manifest.json").write_text(json.dumps(manifest))
                output = root/"renamed"
                with self.assertRaisesRegex(ValueError, "Canonical root fingerprint"):
                    self.full_disjoint(root, parent, pool, smoke, model, output)
                self.assertFalse(preflight.called)
                self.assertFalse((output/"common_manifest.json").exists())
                self.assertFalse((output/"completion_receipt.json").exists())

    def test_full_disjoint_rejects_different_actual_processor_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, pool, smoke, model, preflight = self.disjoint_fixture(root)
            processor = preflight.side_effect
            def different_template(*args, **kwargs):
                proposed, receipt = processor(*args, **kwargs)
                receipt["chat_template_sha256"] = "different-actual-template"
                Path(args[3]).write_text(json.dumps(receipt))
                return proposed, receipt
            preflight.side_effect = different_template
            output = root/"wrong-template"
            with self.assertRaisesRegex(ValueError, "Replacement processor contract differs"):
                self.full_disjoint(root, parent, pool, smoke, model, output)
            self.assertFalse((output/"common_manifest.json").exists())
            self.assertFalse((output/"completion_receipt.json").exists())
            self.assertFalse(json.loads((output/"candidate_preparation.json").read_text())["training_allowed"])

    def completion_fixture(self, root):
        parent, original_source = root/"parent", root/"original_source"
        (parent/"images").mkdir(parents=True)
        original_source.mkdir()
        Image.new("RGB", (12, 12), "green").save(parent/"images/shared.png")
        for name in ("train.py", "modules.py", "audit.py", "sources.py"):
            shutil.copyfile(Path(preparation.__file__).with_name(name), original_source/name)
        # A training-only source change must be bound, not mislabeled as a new processor run.
        with (original_source/"train.py").open("a") as stream:
            stream.write("\n# previous frozen training version\n")
        expected = preparation.policy()
        registered = {name: {"source_known": True, "rights_approved": True,
                            "benchmark_derived": False, "parents": []} for name in ("nvw", "trace")}
        audit.dump(root/"registry.json", registered)
        coverage = {"files": [{"path": str(root/"excluded-source"), "records": 1}], "record_count": 1}
        audit.dump(root/"index.json", {"status": "COMPLETE", "coverage_verified": True,
            "coverage": coverage, "record_count": 1, "image_count": 1,
            "source_files": {str(root/"excluded-source"): "fixture"},
            "per_file_record_counts": {str(root/"excluded-source"): 1}})
        audit.dump(root/"downloads.json", {"fixture": "already cached source receipt"})
        def make(identity, source_id, view="qa", **extra):
            return dict(record(identity, view), source_id=source_id, lineage=[source_id],
                image_paths=["images/shared.png"], benchmark_derived=False, training_allowed=False,
                quality_status="SOURCE_GOLD", supervised_target_identifiable=True,
                original_split="train", **extra)
        nvw_rows = []
        for track, values in expected["nvw_joint_views"].items():
            for view, count in values.items():
                nvw_rows.extend(make(f"nvw_{track}_{view}_{i}", "nvw", view, nvw_track=track)
                                for i in range(count))
        trace_rows = [make(f"trace_{i}", "trace") for i in range(expected["trace_records"])]
        deltas = [make(f"delta_{i}", "trace") for i in range(25)]
        development = make("development", "trace")
        development["split"] = "dev"
        raw = preparation.order(nvw_rows+trace_rows)
        audit.write_rows(parent/"proposed_records.jsonl", raw)
        audit.write_rows(parent/"records.jsonl", trace_rows+[development])
        audit.dump(parent/"conversion.json", {"status": "CONVERTED_NOT_APPROVED"})
        audit.write_rows(parent/"canonical_private.jsonl", [{"root_id": row["root_id"],
            "image": "images/shared.png", "image_sha256": audit.sha(parent/"images/shared.png")} for row in nvw_rows])
        rejected = {row["id"] for row in trace_rows[:25]}
        def contamination(row, image_root, index):
            return ["NEAR_IMAGE_REVIEW_REQUIRED"] if row["id"] in rejected else []
        self.enterContext(patch.object(audit, "contamination_candidates", side_effect=contamination))
        def processor(records_path, image_root, model_path, output, **kwargs):
            rows = audit.read_rows(records_path)
            receipt = {"status": "PASS", "kind": "ACTUAL_PROCESSOR_PREFLIGHT",
                "processor_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17", "processor_class": "Qwen3VLProcessor",
                "chat_template_sha256": "template", "truncation": False, "sequence_budget": 4096,
                "episode_pixels": 1048576, "producer_sha256": audit.sha(audit.__file__),
                "records_sha256": audit.sha(records_path), "records": {row["id"]: {
                    "status": "PASS", "input_sha256": audit.input_fingerprint(row, image_root),
                    "assistant_tokens": 1, "sequence_tokens": 3, "image_grid_thw": [[1, 1, 1]],
                    "tensor_sha256": {key: "fixture" for key in
                                      ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")}}
                    for row in rows}}
            audit.dump(output, receipt)
            return rows, receipt
        processor(parent/"proposed_records.jsonl", parent, root/"model", parent/"processor_preflight.json")
        manifest = audit.approve(parent/"proposed_records.jsonl", parent, registered, root/"index.json",
                                 parent/"approved", parent/"processor_preflight.json")
        manifest["source_sha256"] = {name: audit.sha(original_source/name) for name in ("train.py", "modules.py")}
        (parent/"approved/manifest.json").write_text(json.dumps(manifest))
        audit.dump(parent/"preparation_receipt.json", {"status": "INCOMPLETE_QUOTA", "stage": "pilot",
            "expected_records": 10000, "approved_records": 9975, "quarantined_records": 25,
            "mixture_policy_sha256": audit.sha(preparation.POLICY_PATH),
            "model_input_receipt_sha256": audit.sha(parent/"processor_preflight.json"),
            "selection": {"trace_conversion_sha256": audit.sha(parent/"conversion.json"),
                          "trace_downloads_sha256": audit.sha(root/"downloads.json")}})
        preflight = self.enterContext(patch.object(audit, "preflight_records", side_effect=processor))
        self.enterContext(patch.object(preparation, "trace_paths", return_value=[root/"cached-source"]))
        def conversion(paths, output, seed, train_limit):
            (output/"images").mkdir(parents=True)
            shutil.copyfile(parent/"images/shared.png", output/"images/shared.png")
            return trace_rows+deltas+[development]
        self.enterContext(patch.object(preparation.sources, "trace_records", side_effect=conversion))
        return parent, original_source, preflight, rejected, deltas

    def complete(self, root, parent, original_source, output):
        return preparation.complete_pilot(parent, original_source, root/"downloads.json", root,
            root/"registry.json", root/"index.json", root/"model", output)

    def test_completion_preserves_1000_joint_views_10k_order_and_parent_delta_proofs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, original_source, preflight, rejected, _deltas = self.completion_fixture(root)
            hashes = {str(path.relative_to(parent)): audit.sha(path) for path in parent.rglob("*") if path.is_file()}
            output = root/"completed"
            receipt = self.complete(root, parent, original_source, output)
            rows = audit.read_rows(output/"approved/records.jsonl")
            manifest = json.loads((output/"common_manifest.json").read_text())
            self.assertEqual(manifest["records_path"], str((output/"approved/records.jsonl").resolve()))
            self.assertEqual(manifest["image_root"], str(output.resolve()))
            self.assertEqual(manifest["records_sha256"], audit.sha(output/"approved/records.jsonl"))
            self.assertFalse((output/"records.jsonl").exists())
            self.assertEqual((len(rows), receipt["parent_approved_reused"], receipt["new_approved_trace"]), (10000, 9975, 25))
            self.assertEqual(preparation.joint_counts(rows), preparation.policy()["nvw_joint_views"])
            self.assertFalse(rejected & {row["id"] for row in rows})
            self.assertEqual(rows, preparation.order(rows))
            self.assertEqual(manifest["model_input_audit"]["kind"], "COMPOSITE_PROCESSOR_RECEIPTS")
            self.assertFalse(manifest["model_input_audit"]["full_preflight_reexecuted"])
            self.assertEqual(len(audit.read_rows(preflight.call_args.args[0])), 25)
            self.assertEqual(preflight.call_count, 1)
            self.assertNotEqual(receipt["source_binding"]["parent"]["train.py"], receipt["source_binding"]["current"]["train.py"])
            self.assertEqual(hashes, {str(path.relative_to(parent)): audit.sha(path) for path in parent.rglob("*") if path.is_file()})
            from fine_tuning.train import approved_records
            approved_records(output/"approved/records.jsonl", output/"common_manifest.json", output)

    def test_completion_rejects_corrupted_parent_wrong_source_and_changed_collator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, original_source, preflight, _rejected, _deltas = self.completion_fixture(root)
            raw_path, module_path = parent/"proposed_records.jsonl", original_source/"modules.py"
            for target in (raw_path, module_path):
                original = target.read_bytes()
                target.write_bytes(original+b"\ncorruption")
                with self.assertRaises(ValueError):
                    self.complete(root, parent, original_source, root/f"bad_{target.name}")
                target.write_bytes(original)
            train = original_source/"train.py"
            train.write_text(train.read_text().replace("class AssistantCollator:", "class ChangedCollator:"))
            manifest_path = parent/"approved/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["source_sha256"]["train.py"] = audit.sha(train)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                self.complete(root, parent, original_source, root/"bad_collator")
            self.assertFalse(preflight.called)

    def test_completion_does_not_rescue_quarantined_replacements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, original_source, _preflight, rejected, deltas = self.completion_fixture(root)
            reject_all = rejected | {row["id"] for row in deltas}
            with patch.object(audit, "contamination_candidates", side_effect=lambda row, *_:
                              ["NEAR_IMAGE_REVIEW_REQUIRED"] if row["id"] in reject_all else []):
                with self.assertRaisesRegex(ValueError, "replacement quota"):
                    self.complete(root, parent, original_source, root/"rejected")
            self.assertFalse((root/"rejected/common_manifest.json").exists())
            receipt = json.loads((root/"rejected/completion_receipt.json").read_text())
            self.assertFalse(receipt["training_allowed"])


if __name__ == "__main__":
    unittest.main()
