"""Own synthetic pool/offline fixtures only; no pretrained weights/HF/GPU calls."""
import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning import audit, bridge_evaluation as bridge
from fine_tuning.evaluate import generation, score_external, verify_cohort_images, verify_transfer_checkpoint
from fine_tuning.modules import BASE_ID, BASE_REVISION


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT/"generated/graph_cs_bridge_split_pool32_v2"
APPROVALS = ROOT/"runs/recovered_graph_pool32_child_approval_v2/proof"


def structure():
    return {"image_size": [1200, 1000], "coordinate_system": "pixel_xy_top_left",
            "objects": [{"id": "B", "bbox": [300,400,340,440], "initially_active": False},
                        {"id": "A", "bbox": [100,400,140,440], "initially_active": True}],
            "relations": [{"type": "DIRECTED_TO", "from": "A", "to": "B"}]}


class BridgeEvaluationTests(unittest.TestCase):
    def test_circuit_metrics_use_the_declared_contract_not_graph_or_v8_fallback(self):
        from fine_tuning.circuit_scoring import VERSION
        gold = {"metric_kind": "circuit_qa", "target": "B", "circuit_contract_version": VERSION}
        self.assertTrue(score_external("B", gold, "stop")["correct"])
        self.assertEqual("NO_PARSE", score_external("Final answer: B", gold, "stop")["parse_status"])
        self.assertFalse(score_external("B", gold, "length")["correct"])
        self.assertFalse(score_external("B", dict(gold, target="C"), "stop")["correct"])
        with self.assertRaises(ValueError):
            score_external("B", dict(gold, circuit_contract_version="unknown"), "stop")
        with self.assertRaises(KeyError):
            score_external("B", {"metric_kind": "circuit_qa", "target": "B"}, "stop")

    def test_letter_ids_work_only_in_new_metric_and_array_order_is_not_semantics(self):
        target = structure()
        original = {"metric_kind": "structure", "target": json.dumps(target)}
        self.assertEqual("SCHEMA_INVALID", score_external(json.dumps(target), original, "stop")["parse_status"])
        gold = dict(original, metric_kind="graph_structure")
        shuffled = dict(target, objects=list(reversed(target["objects"])))
        result = score_external(json.dumps(shuffled), gold, "stop")
        self.assertTrue(result["correct"])
        self.assertEqual(1.0, result["structure_f1"])
        self.assertEqual(2, result["bbox_hits"])
        predicted = dict(target, relations=[{"type":"DIRECTED_TO", "from":"B", "to":"A"}])
        self.assertFalse(score_external(json.dumps(predicted), gold, "stop")["correct"])

    def test_geometry_is_not_topology_and_no_wrong_answer_gold_rescue(self):
        target, prediction = structure(), structure()
        prediction["objects"][0]["bbox"] = [900,700,940,740]
        result = score_external(json.dumps(prediction), {"metric_kind":"graph_structure", "target":json.dumps(target)}, "stop")
        self.assertTrue(result["topology_exact"])
        self.assertEqual(1, result["bbox_hits"])
        target["objects"][0]["initially_active"] = True
        self.assertFalse(score_external(json.dumps(prediction), {"metric_kind":"graph_structure", "target":json.dumps(target)}, "stop")["correct"])

    def test_invalid_duplicate_nodes_links_unknown_endpoint_boxes_and_extra_answers_fail(self):
        variants = []
        for field, value in (("id",0), ("id","Z"), ("id","AB"), ("initially_active",1),
                             ("bbox",[0,0,1201,100]), ("bbox",[20,20,10,10]), ("bbox",[0,0,10**400,100])):
            invalid = structure()
            invalid["objects"][0][field] = value
            variants.append(invalid)
        invalid = structure()
        invalid["objects"].append(dict(invalid["objects"][0]))
        variants.append(invalid)
        invalid = structure()
        invalid["relations"].append(dict(invalid["relations"][0]))
        variants.append(invalid)
        for end in ("Z", [], "A"):
            invalid = structure()
            invalid["relations"][0]["to"] = end
            variants.append(invalid)
        variants.append(dict(structure(), answer="D"))
        gold = {"metric_kind":"graph_structure", "target":json.dumps(structure())}
        for invalid in variants:
            with self.subTest(invalid=invalid):
                self.assertEqual("SCHEMA_INVALID", score_external(json.dumps(invalid), gold, "stop")["parse_status"])
        self.assertEqual("NO_PARSE", score_external(json.dumps(structure())+" revise it", gold, "length")["parse_status"])

    def test_root_selection_is_gold_blind_order_invariant_and_does_not_borrow_sealed(self):
        class GoldGuard(dict):
            def __getitem__(self, key):
                if key in {"target_text","answer","semantic_sha256","conditions"}:
                    raise AssertionError("Selected by gold")
                return super().__getitem__(key)
        rows = [GoldGuard(row) for row in audit.read_rows(POOL/"pairing.jsonl")]
        self.assertEqual(bridge.select_roots(rows), bridge.select_roots(list(reversed(rows))))
        self.assertEqual(2, len(bridge.select_roots(rows)))
        with self.assertRaisesRegex(ValueError, "Insufficient development"):
            bridge.select_roots(rows, 3)
        conflicting = [dict(id="x",root_id="r",family_id="same",split="train"),
                       dict(id="y",root_id="s",family_id="same",split="development")]
        with self.assertRaisesRegex(ValueError, "leaks"):
            bridge.select_roots(conflicting, 1)

    def test_actual_own_cohort_has_all_conditions_disjoint_roots_and_no_gold_in_inputs(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary)/"cohort"
            result = bridge.prepare(POOL, APPROVALS, out)
            inputs, gold, ledger = (audit.read_rows(out/name) for name in ("inputs.jsonl","gold.jsonl","pairing.jsonl"))
            self.assertEqual(24, result["count"])
            self.assertEqual(24, len(set(row["id"] for row in inputs)))
            self.assertEqual({"A0","A1","A2","A3"}, {row["condition"] for row in ledger})
            self.assertEqual(2, len(set(row["root_id"] for row in ledger)))
            self.assertEqual({"development"}, {row["split"] for row in ledger})
            self.assertEqual([r["id"] for r in inputs], [r["id"] for r in gold])
            self.assertFalse(result["GPU_evaluation_completed"])
            self.assertEqual(4, len(result["training_evidence"]))
            train = audit.read_rows(APPROVALS/"A0_approved/records.jsonl")
            self.assertFalse({r["root_id"] for r in ledger} & {r["root_id"] for r in train})
            self.assertFalse({r["family_id"] for r in ledger} & {r["family_id"] for r in train})
            for item, entry in zip(inputs, ledger, strict=True):
                self.assertFalse(set(item) & {"target_text","answer","states","condition","root_id","family_id","view"})
                self.assertFalse(item["training_allowed"])
                if entry["view"] == "structure":
                    self.assertIn("image_size=[1200,1000]", item["user_text"])
                    self.assertIn("visible uppercase letter A-H", item["user_text"])
                    self.assertNotIn("960x800", item["user_text"])
                if entry["view"] == "grounded":
                    self.assertEqual(3, len(item["image_paths"]))
            verify_cohort_images(out, result)
            self.assertEqual(result["inputs_sha256"], audit.sha(out/"inputs.jsonl"))
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                bridge.prepare(POOL, APPROVALS, out)
            sample = next(row for row in gold if row["metric_kind"] == "graph_structure")
            self.assertTrue(score_external(sample["target"], sample, "stop")["correct"])
            capture = next(item for item in inputs if item["id"] == sample["id"])
            protocol = {"evaluation": {"min_pixels":262144,"max_pixels":1048576}}
            with patch("fine_tuning.evaluate.prepare_protocol_inputs", side_effect=RuntimeError("before GPU")) as call:
                with self.assertRaisesRegex(RuntimeError, "before GPU"):
                    generation(SimpleNamespace(config=None), None, capture, out, protocol)
                self.assertEqual(capture["user_text"], call.call_args.args[2][0]["content"][-1]["text"])

    def test_changed_development_gold_changes_only_gold_not_model_inputs(self):
        read = audit.read_rows
        def changed(path):
            rows = copy.deepcopy(read(path))
            if Path(path).parent == POOL and Path(path).name in {"A0.jsonl","A1.jsonl","A2.jsonl","A3.jsonl"}:
                for row in rows:
                    if row["split"] == "development":
                        if row["view"] == "qa":
                            row["target_text"] = "A" if row["target_text"] != "A" else "B"
                        elif row["view"] == "structure":
                            target = json.loads(row["target_text"])
                            target["objects"][0]["initially_active"] = not target["objects"][0]["initially_active"]
                            row["target_text"] = json.dumps(target)
            return rows
        with TemporaryDirectory() as temporary:
            a, b = Path(temporary)/"a", Path(temporary)/"b"
            first = bridge.prepare(POOL, APPROVALS, a)
            with patch("fine_tuning.audit.read_rows", side_effect=changed):
                second = bridge.prepare(POOL, APPROVALS, b)
            self.assertEqual(first["inputs_sha256"], second["inputs_sha256"])
            self.assertEqual(first["pairing_sha256"], second["pairing_sha256"])
            self.assertNotEqual(first["gold_sha256"], second["gold_sha256"])

    def test_corrupt_source_and_unapproved_training_fail_before_creating_output(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary)/"out"
            with patch("fine_tuning.audit.sha", return_value="0"*64):
                with self.assertRaisesRegex(ValueError, "pool bytes"):
                    bridge.prepare(POOL, APPROVALS, out)
            self.assertFalse(out.exists())
            read = audit.read_rows
            def invalid(path):
                rows = copy.deepcopy(read(path))
                if Path(path).name == "records.jsonl":
                    rows[0]["split"] = "development"
                return rows
            with patch("fine_tuning.audit.read_rows", side_effect=invalid):
                with self.assertRaisesRegex(ValueError, "row differs"):
                    bridge.prepare(POOL, APPROVALS, out)
            self.assertFalse(out.exists())

    def test_directory_symlink_cannot_supply_pool_inputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            # An internal symlink must never bypass the per-component guard.
            fake = root/"pool"
            fake.mkdir()
            (fake/"manifest.json").symlink_to(POOL/"manifest.json")
            with self.assertRaisesRegex(ValueError, "Symlink"):
                bridge.prepare(fake, APPROVALS, root/"out")

    def test_checkpoint_must_match_one_approved_train_condition_before_GPU(self):
        manifest = {"evaluation_domain":bridge.DOMAIN,
                    "training_evidence":[{"manifest_sha256":"m","records_sha256":"r","records":1}]}
        verify_transfer_checkpoint(manifest, None)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = {"base_id":BASE_ID,"base_revision":BASE_REVISION,
                "epoch_exposure_complete":True,"manifest_sha256":"m","records_sha256":"r",
                "planned_records":1,"exposure":{"records":1}}
            (root/"metadata.json").write_text(json.dumps(metadata))
            verify_transfer_checkpoint(manifest, root)
            for key, value in (("records_sha256","not excluded"),("epoch_exposure_complete",False),
                               ("base_revision","wrong"),("exposure",{"records":0}),("planned_records",0)):
                (root/"metadata.json").write_text(json.dumps(dict(metadata, **{key:value})))
                with self.assertRaisesRegex(ValueError, "frozen train-only"):
                    verify_transfer_checkpoint(manifest, root)


if __name__ == "__main__":
    unittest.main()
