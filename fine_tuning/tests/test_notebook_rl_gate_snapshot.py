"""Actual tiny CPU exports and corrupted snapshots; never native GPU evidence."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl import checkpoint, export, gate_snapshot as gate
from fine_tuning.notebook_rl.policy import merged_named_tensors
from fine_tuning.tests.test_notebook_rl_checkpoint import action_step, fixture, fresh_tiny


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="notebook-snapshot-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.model, self.base, self.protocol, self.binding = fixture(self.root)
        self.bundle = fresh_tiny(self.model, "MOD-RV", self.protocol)
        self.optimizer, self.scheduler = checkpoint.make_optimizer(self.bundle, self.protocol)
        action_step(self.bundle, self.optimizer, self.scheduler)
        self.ckpt = self.root/"step1"
        checkpoint.save(self.bundle, self.optimizer, self.scheduler, 1,
            {"epoch": 0, "position": 0, "discarded_problem_ids": []}, checkpoint.capture_rng(),
            self.ckpt, protocol=self.protocol, binding=self.binding)
        self.expected = dict(merged_named_tensors(self.bundle))
        # The capture's live model really advances again; the helper must reload step1.
        action_step(self.bundle, self.optimizer, self.scheduler)
        self.captured = {"status": "CAPTURE_COMPLETED", "saved_step": 1, "live_final_step": 2,
                         "binding": self.binding, "shared_checkpoint": str(self.ckpt)}
        self.out = self.root/"export"
        actual_export = export.export_merged
        def tiny_export(*args, **kwargs):
            return actual_export(*args, **kwargs, max_shard_size=4096,
                _test_policy_factory=fresh_tiny, _test_base_manifest=self.base)
        with patch.object(gate.export, "export_merged", side_effect=tiny_export):
            self.descriptor = gate.export_step1(self.captured, self.model, self.out, "MOD-RV", self.binding)
        self.manifest = gate.validate_snapshot(self.descriptor, self.binding, self.ckpt)

    def rewrite(self, mutate):
        value = copy.deepcopy(self.manifest)
        mutate(value)
        path = Path(self.descriptor["path"])
        path.write_text(json.dumps(value))
        return {"path": str(path), "file_sha256": checkpoint.file_hash(path)}

    def rewrite_shard(self, mutate):
        path = next(self.out.glob("*.safetensors"))
        values = load_file(str(path))
        mutate(values)
        save_file(values, str(path), metadata={"format": "pt"})
        return self.rewrite(lambda m: m["files"].update({path.name: checkpoint.file_hash(path)}))

    def test_actual_step1_export_streams_complete_tied_inventory_not_live2(self):
        values = dict(gate.iter_tensors(self.descriptor, self.binding, self.ckpt))
        self.assertEqual(set(values), set(self.expected))
        self.assertNotIn("lm_head.weight", values)  # Tied weight is canonical only once.
        self.assertGreater(len(list(self.out.glob("*.safetensors"))), 1)
        for name, value in values.items():
            self.assertTrue(torch.equal(value, self.expected[name]), name)
            self.assertEqual(value.dtype, torch.bfloat16)
            self.assertEqual(value.device.type, "cpu")
        self.assertEqual(self.manifest["checkpoint_files"],
                         {p.name: checkpoint.file_hash(p) for p in self.ckpt.iterdir()})
        self.assertFalse(torch.cuda.is_initialized())

    def test_unbound_capture_wrong_step_arm_or_cuda_refused_before_export(self):
        changes = [{"saved_step": 2}, {"live_final_step": 1}, {"status": "INCOMPLETE"},
                   {"binding": {}}, {"shared_checkpoint": "relative"}]
        with patch.object(gate.export, "export_merged") as exporter:
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    gate.export_step1(self.captured | change, self.model, self.root/"unused", "MOD-RV", self.binding)
            with self.assertRaisesRegex(ValueError, "arm"):
                gate.export_step1(self.captured, self.model, self.root/"unused", "MOD-FML", self.binding)
            with patch.object(torch.cuda, "is_initialized", return_value=True), self.assertRaisesRegex(ValueError, "CPU"):
                gate.export_step1(self.captured, self.model, self.root/"unused", "MOD-RV", self.binding)
            exporter.assert_not_called()

    def test_descriptor_checkpoint_binding_and_file_corruption_refused(self):
        with self.assertRaises(ValueError):
            gate.validate_snapshot(self.descriptor | {"file_sha256": "0"*64}, self.binding, self.ckpt)
        for field, value in (("binding", {}), ("arm", "MOD-FML"), ("checkpoint_files", {}),
                             ("checkpoint_state_sha256", {})):
            descriptor = self.rewrite(lambda m: m.update({field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "checkpoint"):
                gate.validate_snapshot(descriptor, self.binding, self.ckpt)
        # Restore original manifest before testing a changed payload.
        descriptor = self.rewrite(lambda m: None)
        (self.out/"tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            gate.validate_snapshot(descriptor, self.binding, self.ckpt)

    def test_actual_step2_checkpoint_is_rejected_despite_claimed_step1(self):
        step2 = self.root/"step2"
        checkpoint.save(self.bundle, self.optimizer, self.scheduler, 2,
            {"epoch": 0, "position": 0, "discarded_problem_ids": []}, checkpoint.capture_rng(),
            step2, protocol=self.protocol, binding=self.binding)
        with patch.object(gate.export, "export_merged") as exporter, self.assertRaisesRegex(ValueError, "step1"):
            gate.export_step1(self.captured | {"shared_checkpoint": str(step2)},
                self.model, self.root/"unused", "MOD-RV", self.binding)
        exporter.assert_not_called()
        with self.assertRaisesRegex(ValueError, "step1"):
            gate.validate_snapshot(self.descriptor, self.binding, step2)

    def test_safe_paths_symlinks_and_unlisted_files_refused(self):
        for name in ("../escape", "/absolute", "nested/file", "nested\\file", "export_manifest.json"):
            descriptor = self.rewrite(lambda m: m["files"].update({name: "a"*64}))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "safe flat"):
                gate.validate_snapshot(descriptor, self.binding, self.ckpt)
        descriptor = self.rewrite(lambda m: None)
        (self.out/"unlisted").write_text("extra")
        with self.assertRaisesRegex(ValueError, "inventory"):
            gate.validate_snapshot(descriptor, self.binding, self.ckpt)
        (self.out/"unlisted").unlink()
        path = self.out/"tokenizer.json"
        target = self.root/"tokenizer-copy.json"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "hash"):
            gate.validate_snapshot(descriptor, self.binding, self.ckpt)

    def test_tensor_manifest_names_duplicates_and_dtype_refused(self):
        mutations = [lambda m: m["tensors"].pop(),
                     lambda m: m["tensors"].append(copy.deepcopy(m["tensors"][0])),
                     lambda m: m["tensors"][0].update(name="unexpected.weight"),
                     lambda m: m["tensors"][0].update(dtype="torch.float32")]
        for mutate in mutations:
            descriptor = self.rewrite(mutate)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                gate.validate_snapshot(descriptor, self.binding, self.ckpt)

    def test_extra_shard_name_fails_even_with_updated_file_hash(self):
        descriptor = self.rewrite_shard(lambda values: values.update({"extra.weight": torch.zeros(1, dtype=torch.bfloat16)}))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            list(gate.iter_tensors(descriptor, self.binding, self.ckpt))

    def test_missing_shard_name_fails_complete_coverage(self):
        descriptor = self.rewrite_shard(lambda values: values.pop(next(iter(values))))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            list(gate.iter_tensors(descriptor, self.binding, self.ckpt))

    def test_wrong_actual_dtype_or_nonfinite_values_refused(self):
        for nonfinite in (False, True):
            # Each iteration mutates a newly selected tensor; the preceding mismatch remains a failure too.
            def mutate(values):
                name = next(iter(values))
                values[name] = values[name].bfloat16().fill_(float("nan")) if nonfinite else values[name].float()
            descriptor = self.rewrite_shard(mutate)
            with self.subTest(nonfinite=nonfinite), self.assertRaisesRegex(ValueError, "tensor data"):
                list(gate.iter_tensors(descriptor, self.binding, self.ckpt))

    def test_actual_tensor_hash_and_post_stream_mutation_refused(self):
        descriptor = self.rewrite_shard(lambda values: values[next(iter(values))].add_(1))
        with self.assertRaisesRegex(ValueError, "tensor data"):
            list(gate.iter_tensors(descriptor, self.binding, self.ckpt))

    def test_final_revalidation_detects_mutation_after_first_yield(self):
        stream = gate.iter_tensors(self.descriptor, self.binding, self.ckpt)
        name, value = next(stream)
        self.assertEqual(tensor_sha256(value), tensor_sha256(self.expected[name]))
        (self.out/"tokenizer.json").write_text("changed after first yield")
        with self.assertRaisesRegex(ValueError, "hash"):
            list(stream)


if __name__ == "__main__":
    unittest.main()
