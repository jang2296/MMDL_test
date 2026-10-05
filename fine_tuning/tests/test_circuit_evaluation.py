"""CPU cohort contracts with constructed proof fixtures, not actual data approval."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from fine_tuning import audit, bridge_evaluation as cohort, circuit_bridge, run_bridge_smoke
from fine_tuning.evaluate import score_external, verify_cohort_images, verify_transfer_checkpoint
from fine_tuning.modules import BASE_ID, BASE_REVISION


class CircuitEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.pool, cls.approvals = cls.root/"pool", cls.root/"approvals"
        circuit_bridge.generate(cls.pool, count=12)
        cls.approvals.mkdir()
        cls.contract = audit.training_input_contract(Path(circuit_bridge.__file__).with_name("bridge_training_protocol_v1.json"))
        # Constructed receipts exercise byte/identity guards only. No processor
        # execution, rights approval or contamination audit occurs in this test.
        for arm in circuit_bridge.ARMS:
            rows = [audit.model_input_record(r) for r in audit.read_rows(cls.pool/(arm+".jsonl")) if r["split"] == "train"]
            proof = {"status": "PASS", "kind": "ACTUAL_PROCESSOR_PREFLIGHT",
                "fixture_kind": "CONSTRUCTED_CPU_CONTRACT_NOT_ACTUAL_APPROVAL",
                "producer_sha256": audit.sha(Path(audit.__file__)),
                **{k: cls.contract[k] for k in ("processor_revision", "min_pixels", "max_pixels", "episode_pixels", "sequence_budget")},
                "processor_class": "Qwen3VLProcessor", "truncation": False,
                "training_input_contract": cls.contract, "chat_template_sha256": "a"*64,
                "records_sha256": audit.sha(cls.pool/(arm+".jsonl")),
                "records": {r["id"]: {"status": "PASS", "input_sha256": audit.input_fingerprint(r, cls.pool),
                    "sequence_tokens": 100, "assistant_tokens": 10, "image_grid_thw": [[1, 2, 2]]*len(r["image_paths"]),
                    "tensor_sha256": {k: "b"*64 for k in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")}}
                    for r in rows}}
            processor = cls.approvals/(arm+"_processor.json")
            audit.dump(processor, proof)
            approved = [dict(r, training_allowed=True, processor_receipt_sha256=audit.sha(processor)) for r in rows]
            directory = cls.approvals/(arm+"_approved")
            directory.mkdir()
            audit.write_rows(directory/"records.jsonl", approved)
            manifest = {"status": "APPROVED", "training_allowed": True, "benchmark_derived": False,
                "fixture_kind": "CONSTRUCTED_CPU_CONTRACT_NOT_ACTUAL_APPROVAL",
                "contamination_audit": {"status": "PASS"}, "record_count": len(rows),
                "records_sha256": audit.sha(directory/"records.jsonl"),
                "model_input_audit": {"status": "PASS", "record_count": len(rows), "receipt_sha256": audit.sha(processor),
                    **{k: cls.contract[k] for k in ("processor_revision", "min_pixels", "max_pixels", "episode_pixels", "sequence_budget")},
                    "training_input_contract": cls.contract},
                "source_sha256": {name: audit.sha(Path(audit.__file__).with_name(name)) for name in ("train.py", "modules.py")},
                "known_parent_ids": [circuit_bridge.SOURCE_ID],
                "sources": {circuit_bridge.SOURCE_ID: {"parents": [], "benchmark_derived": False,
                    "rights_approved": True, "source_known": True}},
                "files": {name: audit.sha(cls.pool/name) for r in rows for name in r["image_paths"]}}
            audit.dump(directory/"manifest.json", manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_conditions_views_typed_gold_and_gold_blind_input_projection(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)/"cohort"
            manifest = cohort.prepare(self.pool, self.approvals, output, count=1)
            inputs, gold, ledger = [audit.read_rows(output/name) for name in ("inputs.jsonl", "gold.jsonl", "pairing.jsonl")]
            self.assertEqual((cohort.CIRCUIT_DOMAIN, 12), (manifest["evaluation_domain"], manifest["count"]))
            self.assertEqual(12, len({r["id"] for r in inputs}))
            self.assertEqual([r["id"] for r in inputs], [r["id"] for r in gold])
            self.assertEqual({"development"}, {r["split"] for r in ledger})
            train = audit.read_rows(self.approvals/"A0_approved/records.jsonl")
            for field in ("root_id", "family_id"):
                self.assertFalse({r[field] for r in train} & {r[field] for r in ledger})
            for item, target, entry in zip(inputs, gold, ledger, strict=True):
                self.assertFalse(set(item) & {"target_text", "conversations", "answer", "world", "solution", "family_id", "condition", "view"})
                self.assertEqual("circuit_"+entry["view"], target["metric_kind"])
                self.assertEqual(circuit_bridge.VERSION, target["circuit_contract_version"])
                self.assertTrue(score_external(target["target"], target, "stop")["correct"])
                self.assertFalse(score_external(target["target"], target, "length")["correct"])
                self.assertNotIn("<image>", item["user_text"])
            self.assertFalse(manifest["GPU_evaluation_completed"])
            verify_cohort_images(output, manifest)
            with self.assertRaises(FileExistsError):
                cohort.prepare(self.pool, self.approvals, output, count=1)

    def test_existing_four_root_candidate_cannot_borrow_sealed_or_train_roots(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            circuit_bridge.generate(root/"pool", count=4)
            with self.assertRaisesRegex(ValueError, "Insufficient development"):
                cohort.prepare(root/"pool", root/"missing-approvals", root/"out", count=1)
            self.assertFalse((root/"out").exists())

    def test_output_cannot_mutate_pool_or_approval_inventory(self):
        for parent in (self.pool, self.approvals):
            with self.assertRaisesRegex(ValueError, "outside immutable"):
                cohort.prepare(self.pool, self.approvals, parent/"new-output", count=1)
            self.assertFalse((parent/"new-output").exists())

    def test_missing_or_wrong_processor_fields_fail_even_with_rebound_file_hashes(self):
        paths = [self.approvals/"A0_processor.json", self.approvals/"A0_approved/records.jsonl",
                 self.approvals/"A0_approved/manifest.json"]
        original = [p.read_bytes() for p in paths]
        try:
            for field in ("processor_revision", "training_input_contract", "truncation", "row_grid", "row_tensor"):
                proof = json.loads(original[0])
                if field == "row_grid":
                    next(iter(proof["records"].values())).pop("image_grid_thw")
                elif field == "row_tensor":
                    next(iter(proof["records"].values()))["tensor_sha256"].pop("labels")
                else:
                    proof.pop(field)
                paths[0].write_text(json.dumps(proof))
                records = [dict(json.loads(line), processor_receipt_sha256=audit.sha(paths[0])) for line in original[1].splitlines()]
                paths[1].write_text("".join(json.dumps(r)+"\n" for r in records))
                manifest = json.loads(original[2])
                manifest["model_input_audit"]["receipt_sha256"] = audit.sha(paths[0])
                manifest["records_sha256"] = audit.sha(paths[1])
                paths[2].write_text(json.dumps(manifest))
                with TemporaryDirectory() as temporary:
                    output = Path(temporary)/"out"
                    with self.assertRaisesRegex(ValueError, "Circuit .*processor"):
                        cohort.prepare(self.pool, self.approvals, output, count=1)
                    self.assertFalse(output.exists())
        finally:
            for path, data in zip(paths, original, strict=True):
                path.write_bytes(data)

    def test_source_mutation_and_stale_processor_are_rejected_before_output(self):
        processor = self.approvals/"A0_processor.json"
        original = processor.read_bytes()
        try:
            value = json.loads(original)
            value["producer_sha256"] = "0"*64
            processor.write_text(json.dumps(value))
            with TemporaryDirectory() as temporary:
                out = Path(temporary)/"out"
                with self.assertRaisesRegex(ValueError, "processor receipt"):
                    cohort.prepare(self.pool, self.approvals, out, count=1)
                self.assertFalse(out.exists())
        finally:
            processor.write_bytes(original)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            circuit_bridge.generate(root/"pool", count=12)
            (root/"pool"/"A3.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "pool bytes"):
                cohort.prepare(root/"pool", self.approvals, root/"out", count=1)
            self.assertFalse((root/"out").exists())

    def test_checkpoint_requires_matching_closed_training_exposure(self):
        manifest = {"evaluation_domain": cohort.CIRCUIT_DOMAIN,
            "training_evidence": [{"manifest_sha256": "m", "records_sha256": "r", "records": 30}]}
        metadata = {"base_id": BASE_ID, "base_revision": BASE_REVISION,
            "epoch_exposure_complete": True, "manifest_sha256": "m", "records_sha256": "r",
            "planned_records": 30, "exposure": {"records": 30}}
        verify_transfer_checkpoint(manifest, None)
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            audit.dump(checkpoint/"metadata.json", metadata)
            verify_transfer_checkpoint(manifest, checkpoint)
            for key, value in (("records_sha256", "other"), ("epoch_exposure_complete", False),
                               ("base_revision", "other"), ("planned_records", 29), ("exposure", {"records": 29})):
                altered = copy.deepcopy(metadata)
                altered[key] = value
                (checkpoint/"metadata.json").write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "frozen train-only"):
                    verify_transfer_checkpoint(manifest, checkpoint)

    def test_shared_consumer_binds_all_circuit_train_rows_and_original_graph_smoke_rejects(self):
        child=Path(circuit_bridge.__file__).with_name("bridge_training_protocol_v1.json")
        for arm in circuit_bridge.ARMS:
            rows, _manifest, _records, contract, frozen=run_bridge_smoke.inputs(self.approvals,self.pool,child,arm)
            expected=[r for r in audit.read_rows(self.pool/(arm+".jsonl")) if r["split"]=="train"]
            self.assertEqual({r["id"] for r in expected},{r["id"] for r in rows})
            self.assertEqual(self.contract,contract)
            self.assertEqual({"train"},{r["split"] for r in rows})
            self.assertTrue(all(audit.sha(Path(name))==digest for name,digest in frozen.items()))
        args=SimpleNamespace(previous_pid=1,previous_birth="1",previous_controller=self.root,
            approval=self.approvals,image_root=self.pool,input_protocol=child)
        controller=Mock()
        with patch.object(run_bridge_smoke,"process_gone",return_value=True), \
                patch.object(run_bridge_smoke.runner,"closed_controller",return_value=({},[],{})):
            with self.assertRaisesRegex(ValueError,"Graph-only smoke"):
                run_bridge_smoke.run(args,controller)
        controller.stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
