"""Synthetic CPU fixtures only; not actual approval or GPU experiment evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning import audit, prepare_controls, prepare_training


def fixture_rows():
    def row(identity, source, view="qa", track=None):
        value = {"id": identity, "root_id": identity, "family_id": "family-"+identity,
                 "split": "train", "view": view, "source_id": source, "lineage": [source],
                 "user_text": "Fixture question", "target_text": "A", "image_paths": ["images/query.png"],
                 "training_allowed": True, "benchmark_derived": False}
        if track:
            value["nvw_track"] = track
        return value
    rows = [row(f"trace-{i}", "trace") for i in range(9000)]
    for track, views in prepare_training.policy()["nvw_joint_views"].items():
        for view, count in views.items():
            rows.extend(row(f"{track}-{view}-{i}", "nvw-independent-procedural-v1", view, track)
                        for i in range(count))
    return prepare_training.order(rows)


class ControlTests(unittest.TestCase):
    def test_selection_is_frozen_score_blind_and_preserves_common_records(self):
        rows = fixture_rows()
        reference = prepare_training.select_approved_subset(rows, 1600)
        control = prepare_controls.select_trace_control(rows, reference)
        self.assertEqual(control, prepare_controls.select_trace_control(list(reversed(rows)), reference))
        self.assertEqual(len(control), 1600)
        self.assertTrue(all(row["source_id"] == "trace" for row in control))
        common = {row["id"]: row for row in reference if row["source_id"] == "trace"}
        self.assertEqual(len(common), 1440)
        self.assertEqual(common, {row["id"]: row for row in control if row["id"] in common})
        source = {row["id"]: row for row in rows}
        self.assertTrue(all(row == source[row["id"]] for row in control))
        altered = [dict(row) for row in reference]
        altered[0]["target_text"] = "Unapproved alteration"
        with self.assertRaisesRegex(ValueError, "frozen 10%"):
            prepare_controls.select_trace_control(rows, altered)
        with self.assertRaises(ValueError):
            prepare_controls.select_trace_control(rows+rows[:1], reference)
        restricted = [dict(row, family_id="shared") if row["source_id"] == "trace" else row for row in rows]
        restricted_reference = prepare_training.select_approved_subset(restricted, 1600)
        shared_control = prepare_controls.select_trace_control(restricted, restricted_reference)
        self.assertEqual(len(shared_control), 1600)
        shared_reference_roots = {row["root_id"] for row in restricted_reference}
        self.assertEqual(sum(row["root_id"] not in shared_reference_roots for row in shared_control), 160)
        held_out = [dict(row, split="dev") if row["id"] == restricted[0]["id"] else row for row in restricted]
        with self.assertRaisesRegex(ValueError, "train-split"):
            prepare_controls.select_trace_control(held_out, restricted_reference)
        collapsed = [dict(row, root_id="shared-root") if row["source_id"] == "trace" else row for row in rows]
        collapsed_reference = prepare_training.select_approved_subset(collapsed, 1600)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            prepare_controls.select_trace_control(collapsed, collapsed_reference)

    def test_prepare_inherits_verified_rows_and_refuses_overwrite_or_wrong_parent(self):
        from fine_tuning.modules import BASE_REVISION
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, reference, output = root/"parent", root/"reference", root/"control"
            (parent/"images").mkdir(parents=True)
            Image.new("RGB", (12, 12), "red").save(parent/"images/query.png")
            rows = fixture_rows()
            audit.write_rows(parent/"approved/records.jsonl", rows)
            audit.write_rows(parent/"canonical_private.jsonl", [{"fixture": True}])
            completed = {"kind": "ROOT_DISJOINT_FULL_PILOT", "fixture": "mocked closed parent"}
            audit.dump(parent/"completion_receipt.json", completed)
            registered = {source: {"benchmark_derived": False, "rights_approved": True,
                                  "source_known": True, "parents": []}
                          for source in ("trace", "nvw-independent-procedural-v1")}
            manifest = {"status": "APPROVED", "training_allowed": True, "benchmark_derived": False,
                "record_count": len(rows), "records_sha256": audit.sha(parent/"approved/records.jsonl"),
                "known_parent_ids": sorted(registered), "sources": registered,
                "source_sha256": {name: audit.sha(Path(prepare_controls.__file__).with_name(name))
                                  for name in ("train.py", "modules.py")},
                "files": {"images/query.png": audit.sha(parent/"images/query.png")},
                "contamination_audit": {"status": "PASS", "fixture": True},
                "model_input_audit": {"status": "PASS", "record_count": len(rows),
                    "processor_revision": BASE_REVISION, "sequence_budget": 4096, "episode_pixels": 1048576}}
            audit.dump(parent/"common_manifest.json", manifest)
            frozen = {name: audit.sha(parent/name) for name in
                      ("approved/records.jsonl", "common_manifest.json", "canonical_private.jsonl", "completion_receipt.json")}
            evidence = (rows, manifest, completed, frozen, {})
            with patch.object(prepare_training, "closed_pilot", return_value=evidence):
                prepare_training.subset_pilot(parent, reference, 1600, "Synthetic fixture")
                result = prepare_controls.prepare(parent, reference, output)
                self.assertEqual(result["common_unchanged_TRACE_records"], 1440)
                self.assertEqual(result["replacement_TRACE_records"], 160)
                self.assertFalse(result["GPU_training_executed"])
                self.assertFalse(result["professional_bridge_tested"])
                saved = json.loads((output/"common_manifest.json").read_text())
                self.assertEqual(saved["record_count"], 1600)
                self.assertEqual(saved["records_sha256"], audit.sha(output/"approved/records.jsonl"))
                self.assertEqual(saved["model_input_audit"]["kind"], "INHERITED_APPROVED_TRACE_CONTROL")
                self.assertEqual(frozen, {name: audit.sha(parent/name) for name in frozen})
                for forbidden in (output, parent/"new", reference/"new"):
                    with self.assertRaises(ValueError):
                        prepare_controls.prepare(parent, reference, forbidden)
            with patch.object(prepare_training, "closed_pilot", return_value=(rows, manifest, {}, frozen, {})):
                with self.assertRaisesRegex(ValueError, "root-disjoint"):
                    prepare_controls.prepare(parent, reference, root/"wrong-parent")
            # The live reference originated from an earlier closed parent with
            # an explicit smoke exclusion, not the subsequently repaired 10K.
            smoke_row = next(row for row in rows if row["source_id"] != "trace")
            smoke_row = dict(smoke_row, source_kind="INDEPENDENT_SYNTHETIC", parents=[],
                             lineage=["independent-nvw-v1"])
            manifest["sources"]["independent-nvw-v1"] = {
                "benchmark_derived": False, "rights_approved": True,
                "source_known": True, "parents": []}
            smoke_path = root/"smoke_records.jsonl"
            audit.write_rows(smoke_path, [smoke_row])
            earlier = (rows, manifest, {"kind": "EARLIER_CLOSED_PARENT"}, frozen, {})
            excluded_reference = root/"excluded-reference"
            with patch.object(prepare_training, "closed_pilot", return_value=earlier):
                prepare_training.subset_pilot(parent, excluded_reference, 1600, "Fixture exclusion", smoke_path)
                excluded = prepare_controls.prepare(parent, excluded_reference, root/"excluded-control")
                self.assertEqual(excluded["reference_smoke_exclusion"]["excluded_parent_records"], 1)
                copied = excluded_reference/excluded["reference_smoke_exclusion"]["evidence_path"]
                original = copied.read_bytes()
                copied.write_bytes(original+b"\n")
                with self.assertRaisesRegex(ValueError, "smoke exclusion differs"):
                    prepare_controls.prepare(parent, excluded_reference, root/"bad-smoke")
                copied.write_bytes(original)
            receipt_path = reference/"subset_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["records_sha256"] = "f"*64
            receipt_path.write_text(json.dumps(receipt))
            with patch.object(prepare_training, "closed_pilot", return_value=evidence):
                with self.assertRaisesRegex(ValueError, "Reference approval"):
                    prepare_controls.prepare(parent, reference, root/"bad-reference")


if __name__ == "__main__":
    unittest.main()
