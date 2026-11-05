"""Authored CPU hash/count fixtures only, never approval of an actual 3,500-item dataset."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from fine_tuning.error_note import frozen_data as fd, order
from fine_tuning.notebook_rl import approval, contamination, pool_freeze as pf
from fine_tuning.notebook_rl.dataset import SPLITS
from fine_tuning.tests.test_notebook_rl_frozen_approval import unionize
from fine_tuning.tests.test_notebook_rl_pool_local import _sha, _write_json


class FrozenDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "vault"
        (self.root / "contamination").mkdir(parents=True)
        (self.root / "images").mkdir()
        (self.root / "images/shared.png").write_bytes(b"authored CPU image bytes, no real input")
        _write_json(self.root / "manifest.yaml", {"format": "authored CPU fixture"})
        self.ids = [f"authored-{i:04d}" for i in range(3500)]
        for split in SPLITS:
            directory = self.root / split
            directory.mkdir()
            ids = self.ids if split == "train" else [f"authored-{split}"]
            for kind, name in pf._RECORDS:
                rows = [{"problem_id": pid, "image_paths": ["shared.png"], "kind": kind} for pid in ids]
                (directory / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
            _write_json(directory / "manifest.yaml", {})
        union = unionize(self.base, self.root)
        self.manifest_path = union["dataset"]
        self.manifest = json.loads(self.manifest_path.read_bytes())
        self.manifest["category_map"] = {"subject": "C1"}
        for row in self.manifest["items"]:
            row.update(source="authored-source", subject="subject", category="C1")
        self.manifest["items"].append({"problem_id": "authored-excluded", "split": "train", "status": pf.QUOTA})
        self.approval_path = self.base / "approval.json"
        self._bind_manifest()
        self.order_path = self.base / "order.json"
        self.ordered = {"version": order.VERSION, "count": 3500, "ids_sha256": order._ids_sha256(self.ids),
                        "items": [{"problem_id": pid, "source": "authored-source", "cluster": "C1"}
                                  for pid in self.ids],
                        "selection": {"pilot_ids_sha256": order._ids_sha256(self.ids[:300]),
                                      "cluster_train_counts": {"C1": 3500}, "cluster_pilot_counts": {"C1": 300},
                                      "source_train_counts": {"authored-source": 3500},
                                      "source_caps": {"authored-source": 3500}}}
        self._write_order()
        # Simulate runtime: the original shard, registries and scanner evidence are unavailable.
        shutil.rmtree(union["shard"])
        for path in union["inputs"]:
            path.unlink()

    def _bind_manifest(self):
        manifest_sha = _write_json(self.manifest_path, self.manifest)
        binding = json.loads((self.root / "freeze_binding.json").read_bytes())
        binding.update(dataset_manifest_sha256=manifest_sha,
                       counts=dict(Counter(r["split"] for r in self.manifest["items"] if r["status"] in pf.KEPT)))
        binding_sha = _write_json(self.root / "freeze_binding.json", binding)
        union_path = self.root / "contamination" / pf.UNION_FILE
        union = json.loads(union_path.read_bytes())
        union["dataset_manifest"]["file_sha256"] = manifest_sha
        union["freeze_binding_sha256"] = binding_sha
        union["items"] = {r["problem_id"]: {"shard": 0, "shard_allowlisted": True}
                          for r in self.manifest["items"] if r["status"] in pf.KEPT}
        union["union_receipt_sha256"] = approval._hash({k: v for k, v in union.items() if k != "union_receipt_sha256"})
        union_sha = _write_json(union_path, union)
        selected = [r["problem_id"] for r in self.manifest["items"] if r["status"] == pf.INCLUDED]
        checked = {"status": "VALIDATED_FOR_MEASUREMENT", "stage": approval.AI_LOCAL_STAGE,
                   "code_sha256": approval.code_hashes(), "blockers": [], "measurement_allowed": True,
                   "training_allowed": False, "human_verified": False, "selected_ids": selected,
                   "eligibility": {"eligible_ids": selected},
                   "input_binding": {"build_manifest_sha256": _sha((self.root / "manifest.yaml").read_bytes()),
                                     "exclusion_receipt_sha256": union_sha}}
        self.frozen = {"path": str(self.approval_path), "file_sha256": _write_json(self.approval_path, checked)}

    def _write_order(self):
        ids = [item["problem_id"] for item in self.ordered["items"]]
        self.ordered.update(count=len(ids), ids_sha256=order._ids_sha256(ids))
        self.order_sha = _write_json(self.order_path, self.ordered)

    def _explicit_categories(self):
        self.manifest["shards"][0]["item_index_sha256"] = "7" * 64
        self.manifest.update(category_map={"subject": "C3"}, quota={"subject": 3500}, quota_order=pf.SELECTION_ORDER)
        categories = {pid: "C8" if i >= 3150 else "C3" for i, pid in enumerate(self.ids)}
        receipt = {"version": pf.SELECTION_VERSION,
                   "shards": [{key: meta[key] for key in pf._SELECTION_SHARD_FIELDS} for meta in self.manifest["shards"]],
                   "items": [{"problem_id": pid, "category": category} for pid, category in categories.items()]}
        self.manifest["selection"] = receipt | {"file_sha256": _sha(pf._bytes(receipt) + b"\n")}
        for row in self.manifest["items"]:
            if row["status"] == pf.INCLUDED:
                row["category"] = categories[row["problem_id"]]
        for item in self.ordered["items"]:
            item["cluster"] = categories[item["problem_id"]]
        self.ordered["selection"].update(cluster_train_counts={"C3": 3150, "C8": 350}, cluster_pilot_counts={"C3": 300})
        self._bind_manifest()
        self._write_order()

    def verify(self, **kwargs):
        return fd.verify_frozen_data(self.root, self.frozen, self.order_path,
                                     order_sha256=self.order_sha, **kwargs)

    def test_exact_train_and_pilot_without_index_or_scan(self):
        with patch.object(approval, "_validate", side_effect=AssertionError("full approval re-run")), \
                patch.object(contamination, "scan", side_effect=AssertionError("scan re-run")), \
                patch.object(contamination, "load_index", side_effect=AssertionError("benchmark index opened")), \
                patch.object(pf, "verify", wraps=pf.verify) as verify:
            result = self.verify()
        verify.assert_called_once()
        self.assertEqual((result["mode"], result["train_count"], result["pilot_count"], result["main_count"]),
                         ("frozen-approval-verify-v1", 3500, 300, 3200))
        self.assertEqual(result["vault"]["kept_items"], 3503)
        self.assertEqual(result["ordered_ids_sha256"], order._ids_sha256(self.ids))
        self.assertEqual(result["manifest_sha256"], _sha(self.manifest_path.read_bytes()))

    def test_bound_per_id_categories_allow_same_subject_c3_and_c8_without_receipt_files(self):
        self._explicit_categories()
        with patch.object(approval, "_validate", side_effect=AssertionError("approval rerun")), \
                patch.object(contamination, "load_index", side_effect=AssertionError("benchmark index opened")):
            result = self.verify()
        self.assertEqual(result["cluster_train_counts"], {"C3": 3150, "C8": 350})
        self.assertEqual(result["cluster_pilot_counts"], {"C3": 300})
        # Counts still come from the bound receipt when order counters are absent.
        del self.ordered["selection"]["cluster_train_counts"]
        del self.ordered["selection"]["cluster_pilot_counts"]
        self._write_order()
        self.assertEqual(self.verify()["cluster_train_counts"], result["cluster_train_counts"])

    def test_explicit_order_category_or_missing_cluster_cannot_fall_back_to_subject(self):
        self._explicit_categories()
        original = copy.deepcopy(self.ordered)
        for missing in (False, True):
            self.ordered = copy.deepcopy(original)
            if missing:
                del self.ordered["items"][-1]["cluster"]
            else:
                self.ordered["items"][-1]["cluster"] = "C3"
            self._write_order()
            with self.assertRaisesRegex(ValueError, "bound per-ID selection"):
                self.verify()

    def test_explicit_manifest_category_membership_and_subject_quota_are_checked(self):
        self._explicit_categories()
        original = copy.deepcopy(self.manifest)
        for label in ("category", "duplicate", "unknown", "heldout", "shortfall", "shard"):
            self.manifest = copy.deepcopy(original)
            if label == "category":
                next(row for row in self.manifest["items"] if row["problem_id"] == self.ids[-1])["category"] = "C3"
            elif label == "duplicate":
                self.manifest["selection"]["items"].append(self.manifest["selection"]["items"][0])
            elif label in ("unknown", "heldout"):
                self.manifest["selection"]["items"][0]["problem_id"] = "missing" if label == "unknown" else "authored-dev"
            elif label == "shortfall":
                self.manifest["quota"]["subject"] = 3501
            else:
                self.manifest["selection"]["shards"][0]["item_index_sha256"] = "0" * 64
            self._bind_manifest()
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.verify()

    def test_coordinated_receipt_row_and_order_edits_cannot_reuse_original_selection_pin(self):
        self._explicit_categories()
        victim = self.ids[-1]
        next(row for row in self.manifest["selection"]["items"] if row["problem_id"] == victim)["category"] = "C3"
        next(row for row in self.manifest["items"] if row["problem_id"] == victim)["category"] = "C3"
        self.ordered["items"][-1]["cluster"] = "C3"
        self.ordered["selection"]["cluster_train_counts"] = {"C3": 3151, "C8": 349}
        self._bind_manifest()
        self._write_order()
        with self.assertRaisesRegex(ValueError, "selection SHA256"):
            self.verify()

    def test_short_order_and_duplicate_ids_are_refused(self):
        original = copy.deepcopy(self.ordered)
        for label in ("3499", "duplicate", "held-out", "excluded"):
            with self.subTest(label=label):
                self.ordered = copy.deepcopy(original)
                if label == "3499":
                    self.ordered["items"].pop()
                else:
                    self.ordered["items"][-1]["problem_id"] = {
                        "duplicate": self.ids[0], "held-out": "authored-dev", "excluded": "authored-excluded"}[label]
                self._write_order()
                with self.assertRaisesRegex(ValueError, "duplicate|3500"):
                    self.verify()

    def test_3499_included_is_refused_even_when_all_bytes_are_rebound(self):
        self.manifest["items"][0]["status"] = pf.DEV
        self._bind_manifest()
        with self.assertRaisesRegex(ValueError, "3500"):
            self.verify()

    def test_duplicate_manifest_id_and_wrong_split_are_refused(self):
        original = copy.deepcopy(self.manifest)
        self.manifest["items"].append(copy.deepcopy(self.manifest["items"][0]))
        self._bind_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.verify()
        self.manifest = original
        held_out = next(row for row in self.manifest["items"] if row["split"] == "dev")
        held_out["status"] = pf.INCLUDED
        self.manifest["items"][0]["status"] = pf.DEV
        self._bind_manifest()
        with self.assertRaisesRegex(ValueError, "3500"):
            self.verify()

    def test_wrong_missing_or_conflicting_pilot_pin_is_refused(self):
        with self.assertRaisesRegex(ValueError, "pilot300"):
            self.verify(expected_pilot_sha256="0" * 64)
        self.ordered["selection"]["pilot_ids_sha256"] = order._ids_sha256(self.ids[1:301])
        self._write_order()
        with self.assertRaisesRegex(ValueError, "pilot300"):
            self.verify()
        del self.ordered["selection"]["pilot_ids_sha256"]
        self._write_order()
        with self.assertRaisesRegex(ValueError, "pilot300"):
            self.verify()
        self.assertEqual(self.verify(expected_pilot_sha256=order._ids_sha256(self.ids[:300]))["pilot_count"], 300)

    def test_quota_and_metadata_are_compared_to_manifest(self):
        original = copy.deepcopy(self.ordered)
        mutations = (("cluster_train_counts", {"C1": 3499}), ("cluster_pilot_counts", {"C1": 299}),
                     ("source_train_counts", {"different-source": 3500}),
                     ("source_caps", {"authored-source": 3499}), ("source_caps", {"authored-source": True}))
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                self.ordered = copy.deepcopy(original)
                self.ordered["selection"][key] = value
                self._write_order()
                with self.assertRaisesRegex(ValueError, key):
                    self.verify()
        self.ordered = original
        self.ordered["items"][0]["cluster"] = "C2"
        self._write_order()
        with self.assertRaisesRegex(ValueError, "category_map"):
            self.verify()

    def test_external_digests_and_approval_source_bindings_cannot_be_replaced(self):
        for path in (self.order_path, self.approval_path, self.root / "images/shared.png",
                     self.root / "train/actor_inputs.jsonl"):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    self.verify()
                path.write_bytes(original)
        checked = json.loads(self.approval_path.read_bytes())
        checked["code_sha256"]["changed.py"] = "0" * 64
        self.frozen["file_sha256"] = _write_json(self.approval_path, checked)
        with self.assertRaisesRegex(ValueError, "stale source"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
