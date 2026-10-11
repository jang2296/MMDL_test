"""Tiny real Torch/PEFT export checks; no model download or GPU inference."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from peft.tuners.lora import LoraModel
from transformers import Qwen3VLForConditionalGeneration

from experiments.input_recovery import checkpoints
from fine_tuning.modules import (BASE_ID, BASE_REVISION, configure_arm,
                                 load_checkpoint, save_checkpoint, tensor_sha256)
from fine_tuning.tests.test_training import tiny_model


class CheckpointExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = tiny_model().to(dtype=torch.bfloat16)
        self.model_path = self.root / BASE_REVISION
        self.base.save_pretrained(self.model_path, safe_serialization=True)
        (self.model_path / "preprocessor_config.json").write_text('{"fixture": "unchanged"}\n')
        (self.model_path / "chat_template.json").write_text('{"fixture": "unchanged"}\n')
        self.manifest_path = self.root / "base_manifest.json"
        self.manifest_path.write_text(json.dumps({"repo_id": BASE_ID, "revision": BASE_REVISION,
            "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": checkpoints._hash(p)}
                      for p in sorted(self.model_path.iterdir())]}))
        self.patch_manifest = patch.object(checkpoints, "BASE_MANIFEST", self.manifest_path)
        self.patch_manifest.start()
        self.addCleanup(self.patch_manifest.stop)

    def checkpoint(self, arm="FML"):
        model, inventory, _ = configure_arm(copy.deepcopy(self.base), arm)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.025)
                elif parameter.requires_grad and "lora_" not in name:
                    parameter.add_(0.021)
        path = self.root / f"checkpoint_{arm}"
        save_checkpoint(model, path, {"arm": arm}, inventory)
        return path

    def export(self, checkpoint, output=None, side_effect=None):
        with patch.object(Qwen3VLForConditionalGeneration, "from_pretrained",
                          side_effect=side_effect or (lambda *a, **kw: copy.deepcopy(self.base))) as loader:
            result = checkpoints.export_checkpoint(self.model_path, checkpoint, output or self.root / "export")
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertIsNone(loader.call_args.kwargs["device_map"])
        return result

    def test_all_arms_preserve_complete_weights_mergers_and_vision_lora(self):
        for arm in ("FL", "FM", "FV", "FML"):
            with self.subTest(arm=arm):
                checkpoint = self.checkpoint(arm)
                original_files = {p.name: checkpoints._hash(p) for p in self.model_path.iterdir()}
                restored = load_checkpoint(copy.deepcopy(self.base), checkpoint)
                expected = {name: value.detach().clone() for name, value in self.base.state_dict().items()}
                lora_targets = []
                for name, module in restored.named_modules():
                    if hasattr(module, "lora_A"):
                        target = name.removeprefix("base_model.model.")
                        expected[target + ".weight"] = (
                            module.base_layer.weight.float() + module.get_delta_weight("default").float()
                        ).to(dtype=torch.bfloat16)
                        lora_targets.append(target)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                for row in metadata["inventory"]["trainable"]:
                    if row["group"] == "M":
                        expected[row["name"].removeprefix("base_model.model.")] = (
                            dict(restored.named_parameters())[row["name"]].detach().to(dtype=torch.bfloat16)
                        )
                destination = self.root / f"export_{arm}"
                events = []
                original_merge = LoraModel.merge_and_unload

                def restore(base, source):
                    result = load_checkpoint(base, source)
                    for name, digest in metadata["state_sha256"].items():
                        self.assertEqual(tensor_sha256(dict(result.named_parameters())[name]), digest)
                    events.append("verified_restore")
                    return result

                def merge(model, *args, **kwargs):
                    self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
                    self.assertTrue(kwargs["safe_merge"])
                    events.append("FP32_merge")
                    return original_merge(model, *args, **kwargs)

                with patch.object(checkpoints, "load_checkpoint", side_effect=restore), \
                        patch.object(LoraModel, "merge_and_unload", autospec=True, side_effect=merge):
                    manifest = self.export(checkpoint, destination)
                self.assertEqual(events, ["verified_restore", "FP32_merge"] if lora_targets else ["verified_restore"])
                exported = Qwen3VLForConditionalGeneration.from_pretrained(
                    destination, local_files_only=True, dtype=torch.bfloat16)
                self.assertEqual(set(expected), set(exported.state_dict()))
                for name, value in exported.state_dict().items():
                    self.assertTrue(torch.equal(value, expected[name]), name)
                self.assertEqual(set(manifest["merged_lora_modules"]), set(lora_targets))
                self.assertEqual(len(manifest["mergers"]), 4)
                self.assertEqual(len(manifest["merger_tensor_names"]), 24)
                if arm == "FV":
                    self.assertTrue(all("visual" in name for name in lora_targets))
                    self.assertTrue(any(not torch.equal(value, self.base.state_dict()[name])
                                        for name, value in expected.items() if name.endswith("qkv.weight")))
                for row in manifest["tensors"]:
                    self.assertEqual(row["sha256"], tensor_sha256(exported.state_dict()[row["name"]]))
                for name, digest in manifest["files"].items():
                    self.assertEqual(checkpoints._hash(destination / name), digest)
                self.assertFalse(manifest["precision_policy"]["unmerged_inference_bit_equivalent"])
                self.assertEqual(original_files, {p.name: checkpoints._hash(p) for p in self.model_path.iterdir()})
                for name in ("config.json", "preprocessor_config.json", "chat_template.json"):
                    self.assertEqual((destination / name).read_bytes(), (self.model_path / name).read_bytes())

    def test_no_overwrite_symlink_or_source_child(self):
        checkpoint = self.checkpoint("FM")
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "preserved"
        marker.write_text("owned by caller")
        with self.assertRaises(FileExistsError):
            self.export(checkpoint, existing)
        self.assertEqual(marker.read_text(), "owned by caller")
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            self.export(checkpoint, dangling)
        for root in (checkpoint, self.model_path):
            with self.assertRaisesRegex(ValueError, "outside source"):
                self.export(checkpoint, root / "new")

    def test_wrong_identity_path_and_source_hash_rejected_before_load(self):
        checkpoint = self.checkpoint("FM")
        metadata_path = checkpoint / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata_path.write_text(json.dumps({**metadata, "base_revision": "wrong"}))
        with self.assertRaisesRegex(ValueError, "base identity"):
            self.export(checkpoint)
        metadata_path.write_text(json.dumps(metadata))
        (checkpoint / "mergers.pt").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "path/hash"):
            self.export(checkpoint)
        (checkpoint / "mergers.pt").unlink()
        escaped = copy.deepcopy(metadata)
        escaped["files"]["../escape"] = "0" * 64
        metadata_path.write_text(json.dumps(escaped))
        with self.assertRaisesRegex(ValueError, "unbound files"):
            self.export(checkpoint)
        metadata_path.write_text(json.dumps(metadata))
        (self.model_path / "config.json").write_text("wrong bytes")
        with self.assertRaisesRegex(ValueError, "Pinned base file"):
            self.export(checkpoint)

    def test_missing_master_or_wrong_shape_rejected(self):
        checkpoint = self.checkpoint("FM")
        path = checkpoint / "metadata.json"
        metadata = json.loads(path.read_text())
        missing = copy.deepcopy(metadata)
        missing["state_sha256"].pop(next(iter(missing["state_sha256"])))
        path.write_text(json.dumps(missing))
        with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
            self.export(checkpoint)
        wrong = copy.deepcopy(metadata)
        wrong["inventory"]["trainable"][0]["shape"] = [999]
        path.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self.export(checkpoint)
        path.write_text(json.dumps(metadata))
        state = torch.load(checkpoint / "mergers.pt", weights_only=True)
        removed = next(iter(state))
        state.pop(removed)
        metadata["state_sha256"].pop(removed)
        metadata["inventory"]["trainable"] = [row for row in metadata["inventory"]["trainable"]
                                                if row["name"] != removed]
        torch.save(state, checkpoint / "mergers.pt")
        metadata["files"]["mergers.pt"] = checkpoints._hash(checkpoint / "mergers.pt")
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "every tensor"):
            self.export(checkpoint)

    def test_source_change_and_failed_save_never_commit_manifest(self):
        checkpoint = self.checkpoint("FM")

        def change_source(*args, **kwargs):
            (self.model_path / "chat_template.json").write_text("changed during loading")
            return copy.deepcopy(self.base)

        with self.assertRaisesRegex(ValueError, "source changed"):
            self.export(checkpoint, side_effect=change_source)
        self.assertFalse((self.root / "export").exists())
        (self.model_path / "chat_template.json").write_text('{"fixture": "unchanged"}\n')
        with patch.object(Qwen3VLForConditionalGeneration, "save_pretrained", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.export(checkpoint)
        self.assertFalse((self.root / "export/export_manifest.json").exists())
        with self.assertRaises(FileExistsError):
            self.export(checkpoint)


if __name__ == "__main__":
    unittest.main()
