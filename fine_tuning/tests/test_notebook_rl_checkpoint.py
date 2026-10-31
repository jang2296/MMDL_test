"""Tiny real Qwen/PEFT resume and HF exports, never pretrained 4B inference."""
from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3VLForConditionalGeneration

from fine_tuning.modules import BASE_ID, BASE_REVISION, tensor_sha256
from fine_tuning.notebook_rl.checkpoint import (capture_rng, digest, file_hash, load_eval,
    load_resume, make_optimizer, read_metadata, restore_rng, save)
from fine_tuning.notebook_rl.export import export_merged, validate_export_identity
from fine_tuning.notebook_rl.policy import (assert_base_unchanged, completion_logprobs,
    configure_policy, merged_named_tensors, reference_logprobs)
from fine_tuning.tests.test_notebook_rl_policy import config, inputs
from fine_tuning.tests.test_training import tiny_model


def fresh_tiny(model_path, arm, protocol):
    base = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path), dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa")
    # CPU policy has no CUDA autocast; use FP32 arithmetic on the exact BF16 source values.
    base.float()
    return configure_policy(base, None, arm, protocol, enforce_counts=False)


def fixture(root):
    model_path = root / BASE_REVISION
    model = tiny_model().bfloat16()
    model.config.tie_word_embeddings = True
    model.config.text_config.tie_word_embeddings = True
    model.tie_weights()
    model.save_pretrained(model_path, safe_serialization=True)
    # Authored minimal metadata exercises byte-preserving copies, not processor semantics.
    auxiliary = {"chat_template.json": {"chat_template": "{{ messages }}"}, "vocab.json": {},
        "tokenizer.json": {}, "tokenizer_config.json": {}, "preprocessor_config.json": {},
        "video_preprocessor_config.json": {}}
    for name, value in auxiliary.items():
        (model_path / name).write_text(json.dumps(value), encoding="utf-8")
    (model_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    manifest = {"repo_id": BASE_ID, "revision": BASE_REVISION, "files": [
        {"path": p.name, "bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in sorted(model_path.iterdir())]}
    manifest_path = root / "tiny_base_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    protocol = config()
    binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
               "data_sha256": digest({"authored": "tiny-data"}), "code_sha256": digest({"authored": "test-code"})}
    return model_path, manifest_path, protocol, binding


def action_step(bundle, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    logprobs = completion_logprobs(bundle, inputs(), 3, chunk=1)
    # Random multiplier simultaneously verifies Python, NumPy, and Torch RNG restoration.
    multiplier = random.random() + float(np.random.random()) + float(torch.rand(()))
    loss = (logprobs[0, 0] - logprobs[0, 1]) * multiplier
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def snapshot(bundle):
    return {n: tensor_sha256(p) for n, p in bundle.model.named_parameters()}


def worker(root):
    torch.set_num_threads(2)
    arguments = json.loads((root / "worker.json").read_text())
    bundle, optimizer, scheduler, step, cursor = load_resume(root / BASE_REVISION, root / "checkpoint",
        config(), binding=arguments["binding"], _test_policy_factory=fresh_tiny)
    actual = completion_logprobs(bundle, inputs(), 3).detach()
    loss = action_step(bundle, optimizer, scheduler)
    torch.save({"logprobs": actual, "loss": loss, "state": snapshot(bundle), "step": step,
                "cursor": cursor, "lrs": scheduler.get_last_lr()}, root / "worker_result.pt")


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="notebook-checkpoint-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model_path, self.base_manifest, self.protocol, self.binding = fixture(self.root)
        self.bundle = fresh_tiny(self.model_path, "MOD-RV", self.protocol)
        self.optimizer, self.scheduler = make_optimizer(self.bundle, self.protocol)
        self.cursor = {"epoch": 0, "position": 17, "discarded_problem_ids": ["zero-adv-3"],
                       "order_sha256": "e" * 64}
        random.seed(3407)
        np.random.seed(3407)
        torch.manual_seed(3407)

    def checkpoint(self, steps=2):
        for _ in range(steps):
            action_step(self.bundle, self.optimizer, self.scheduler)
        self.bundle.route.apply(False)
        metadata = save(self.bundle, self.optimizer, self.scheduler, steps, self.cursor,
            capture_rng(), self.root / "checkpoint", protocol=self.protocol, binding=self.binding)
        return self.root / "checkpoint", metadata

    def rewrite(self, ckpt, mutate):
        metadata = json.loads((ckpt / "metadata.json").read_text())
        mutate(metadata)
        metadata["files"] = {n: file_hash(ckpt / n) for n in metadata["files"]}
        (ckpt / "metadata.json").write_text(json.dumps(metadata))
        (ckpt / "metadata.sha256").write_text(file_hash(ckpt / "metadata.json") + "\n")

    def test_fresh_subprocess_exact_logits_next_random_loss_and_next_parameters(self):
        ckpt, metadata = self.checkpoint()
        reference = reference_logprobs(self.bundle, inputs(), 3)
        logprobs = completion_logprobs(self.bundle, inputs(), 3).detach()
        loss = action_step(self.bundle, self.optimizer, self.scheduler)
        expected_state = snapshot(self.bundle)
        (self.root / "worker.json").write_text(json.dumps({"binding": self.binding}))
        result = subprocess.run([sys.executable, "-m", "fine_tuning.tests.test_notebook_rl_checkpoint",
                                "--worker", str(self.root)],
            cwd=Path(__file__).resolve().parents[2], env=os.environ.copy(), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        actual = torch.load(self.root / "worker_result.pt", weights_only=True)
        self.assertTrue(torch.equal(actual["logprobs"], logprobs))
        self.assertTrue(torch.equal(actual["loss"], loss))
        self.assertEqual(actual["state"], expected_state)
        self.assertEqual((actual["step"], actual["cursor"]), (2, self.cursor))
        self.assertEqual(actual["lrs"], self.scheduler.get_last_lr())
        restored = load_eval(self.model_path, ckpt, binding=self.binding, _test_policy_factory=fresh_tiny)
        self.assertTrue(torch.equal(reference_logprobs(restored, inputs(), 3), reference))
        self.assertFalse(restored.route.vision_on)
        self.assertFalse(restored.model.training)
        self.assertEqual(metadata["binding"], self.binding)
        self.assertEqual(metadata["binding"]["protocol_sha256"], self.protocol["sha256"])

    def test_all_mergers_fp32_and_route_off_vision_are_saved(self):
        ckpt, metadata = self.checkpoint()
        masters = load_file(str(ckpt / "mergers.safetensors"))
        adapters = load_file(str(ckpt / "adapter.safetensors"))
        self.assertEqual(len(masters), 24)
        self.assertTrue(all(t.dtype == torch.float32 for t in masters.values()))
        vision = {r["name"] for r in self.bundle.inventory["trainable"] if r["group"] == "V"}
        self.assertTrue(vision <= adapters.keys())
        self.assertFalse(any(p.requires_grad for p in self.bundle.params["V"]))
        for name, value in (masters | adapters).items():
            self.assertEqual(tensor_sha256(value), metadata["state_sha256"][name])
        with self.assertRaises(FileExistsError):
            save(self.bundle, self.optimizer, self.scheduler, 2, self.cursor, capture_rng(), ckpt,
                protocol=self.protocol, binding=self.binding)

    def test_corruption_metadata_extra_files_and_symlinks_refused(self):
        ckpt, _ = self.checkpoint(0)
        path = ckpt / "runtime.pt"
        original = path.read_bytes()
        path.write_bytes(original + b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_metadata(ckpt, binding=self.binding)
        path.write_bytes(original)
        (ckpt / "extra").write_text("unbound")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            read_metadata(ckpt, binding=self.binding)
        (ckpt / "extra").unlink()
        (ckpt / "runtime.pt").rename(ckpt / "payload")
        (ckpt / "runtime.pt").symlink_to(ckpt / "payload")
        with self.assertRaises(ValueError):
            read_metadata(ckpt, binding=self.binding)

    def test_wrong_data_code_prompts_and_protocol_refused_before_factory(self):
        ckpt, _ = self.checkpoint(0)
        def forbidden(*args):
            self.fail("factory called before binding verification")
        for key in self.binding:
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_eval(self.model_path, ckpt, binding=self.binding | {key: "0" * 64},
                          _test_policy_factory=forbidden)
        changed = copy.deepcopy(self.protocol)
        changed["optimizer"]["lr"]["L"] *= 2
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            load_resume(self.model_path, ckpt, changed, binding=self.binding, _test_policy_factory=forbidden)
        with self.assertRaisesRegex(ValueError, "binding"):
            save(self.bundle, self.optimizer, self.scheduler, 0, self.cursor, capture_rng(), self.root / "bad",
                 protocol=self.protocol, binding={})

    def test_unexpected_adapter_keys_and_missing_merger_even_with_recomputed_checksums(self):
        ckpt, _ = self.checkpoint(0)
        path = ckpt / "adapter.safetensors"
        tensors = load_file(str(path))
        save_file(tensors | {"unexpected.weight": torch.ones(1)}, str(path))
        self.rewrite(ckpt, lambda m: None)
        with self.assertRaisesRegex(ValueError, "tensor keys"):
            load_eval(self.model_path, ckpt, binding=self.binding, _test_policy_factory=fresh_tiny)
        save_file(tensors, str(path))
        path = ckpt / "mergers.safetensors"
        tensors = load_file(str(path))
        tensors.pop(next(iter(tensors)))
        save_file(tensors, str(path))
        self.rewrite(ckpt, lambda m: None)
        with self.assertRaisesRegex(ValueError, "tensor keys"):
            load_eval(self.model_path, ckpt, binding=self.binding, _test_policy_factory=fresh_tiny)

    def test_frozen_base_and_missing_optimizer_partition_refused(self):
        self.optimizer.param_groups.pop()
        with self.assertRaisesRegex(ValueError, "every installed"):
            save(self.bundle, self.optimizer, self.scheduler, 0, self.cursor, capture_rng(), self.root / "bad",
                 protocol=self.protocol, binding=self.binding)
        with torch.no_grad():
            self.bundle.model.get_base_model().model.visual.patch_embed.proj.weight.add_(1)
        with self.assertRaisesRegex(ValueError, "frozen"):
            save(self.bundle, self.optimizer, self.scheduler, 0, self.cursor, capture_rng(), self.root / "bad2",
                 protocol=self.protocol, binding=self.binding)

    def test_rng_roundtrip_and_linear_warmup(self):
        saved = capture_rng()
        expected = (random.random(), np.random.random(), torch.rand(()))
        restore_rng(saved)
        actual = (random.random(), np.random.random(), torch.rand(()))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertEqual(self.optimizer.param_groups[0]["lr"], self.protocol["optimizer"]["lr"]["L"] / 10)
        for _ in range(12):
            self.optimizer.step()
            self.scheduler.step()
        self.assertEqual(self.optimizer.param_groups[0]["lr"], self.protocol["optimizer"]["lr"]["L"])

    def test_unexpected_runtime_and_nonfinite_adamw_moments_refused(self):
        ckpt, _ = self.checkpoint(1)
        path = ckpt / "runtime.pt"
        runtime = torch.load(path, weights_only=True)
        torch.save(runtime | {"unexpected": True}, path)
        self.rewrite(ckpt, lambda m: None)
        with self.assertRaisesRegex(ValueError, "unexpected runtime"):
            load_resume(self.model_path, ckpt, self.protocol, binding=self.binding, _test_policy_factory=fresh_tiny)
        next(iter(runtime["optimizer"]["state"].values()))["exp_avg"].flatten()[0] = float("nan")
        torch.save(runtime, path)
        self.rewrite(ckpt, lambda m: None)
        with self.assertRaisesRegex(ValueError, "AdamW"):
            load_resume(self.model_path, ckpt, self.protocol, binding=self.binding, _test_policy_factory=fresh_tiny)

    def test_scheduler_step_or_arbitrary_lambda_cannot_silently_change_resume(self):
        with self.assertRaisesRegex(ValueError, "scheduler state"):
            save(self.bundle, self.optimizer, self.scheduler, 1, self.cursor, capture_rng(), self.root / "bad-step",
                 protocol=self.protocol, binding=self.binding)
        self.scheduler.lr_lambdas = [lambda step: .1] * len(self.optimizer.param_groups)
        with self.assertRaisesRegex(ValueError, "make_optimizer"):
            save(self.bundle, self.optimizer, self.scheduler, 0, self.cursor, capture_rng(), self.root / "bad-lambda",
                 protocol=self.protocol, binding=self.binding)
        self.assertFalse((self.root / "bad-lambda").exists())

    def test_fixed_arm_no_vision_adapter_and_fresh_reference_mismatch_refused(self):
        self.bundle = fresh_tiny(self.model_path, "MOD-FML", self.protocol)
        self.optimizer, self.scheduler = make_optimizer(self.bundle, self.protocol)
        ckpt, _ = self.checkpoint(1)
        restored, optimizer, scheduler, step, cursor = load_resume(self.model_path, ckpt, self.protocol,
            binding=self.binding, _test_policy_factory=fresh_tiny)
        self.assertEqual(restored.params["V"], [])
        self.assertEqual([g["name"] for g in optimizer.param_groups], ["L", "M"])
        self.assertEqual((step, cursor), (1, self.cursor))
        self.assertEqual(scheduler.get_last_lr(), self.scheduler.get_last_lr())
        def wrong_reference(path, arm, protocol):
            b = fresh_tiny(path, arm, protocol)
            b.base_sha256[next(iter(b.base_sha256))] = "0" * 64
            return b
        with self.assertRaisesRegex(ValueError, "fresh F0"):
            load_eval(self.model_path, ckpt, binding=self.binding, _test_policy_factory=wrong_reference)

    def test_sharded_bf16_export_with_tied_weights_and_immutable_sources(self):
        ckpt, _ = self.checkpoint()
        before = snapshot(self.bundle)
        sources = {p: file_hash(p) for p in self.model_path.iterdir()}
        source_checkpoint = {p: file_hash(p) for p in ckpt.iterdir()}
        expected = dict(merged_named_tensors(self.bundle))
        out = self.root / "export"
        manifest = export_merged(ckpt, self.model_path, out, "MOD-RV", binding=self.binding,
            max_shard_size=4096, _test_policy_factory=fresh_tiny, _test_base_manifest=self.base_manifest)
        self.assertEqual(manifest["format"], "derived-hf-bf16-v1")
        self.assertEqual(manifest["status"], "EXPORTED")
        self.assertEqual(len(manifest["merger_tensor_names"]), 24)
        self.assertGreater(len(list(out.glob("*.safetensors"))), 1)
        reloaded = Qwen3VLForConditionalGeneration.from_pretrained(str(out), dtype=torch.bfloat16,
            local_files_only=True, attn_implementation="sdpa")
        self.assertEqual(set(dict(reloaded.named_parameters())), set(expected))
        for name, p in reloaded.named_parameters():
            self.assertTrue(torch.equal(p, expected[name]), name)
        self.assertIs(reloaded.lm_head.weight, reloaded.model.language_model.embed_tokens.weight)
        self.assertNotIn("lm_head.weight", expected)
        self.assertEqual(before, snapshot(self.bundle))
        self.assertEqual(sources, {p: file_hash(p) for p in self.model_path.iterdir()})
        self.assertEqual(source_checkpoint, {p: file_hash(p) for p in ckpt.iterdir()})
        self.assertTrue(all(file_hash(out / name) == value for name, value in manifest["files"].items()))
        for name in ("config.json", "preprocessor_config.json", "tokenizer.json", "chat_template.json"):
            self.assertEqual((out / name).read_bytes(), (self.model_path / name).read_bytes())
        assert_base_unchanged(self.bundle)
        identity = validate_export_identity(out, "MOD-RV")
        self.assertEqual(identity["arm"], "MOD-RV")
        self.assertEqual(identity["manifest_sha256"], digest(manifest))
        self.assertEqual(identity["files"], manifest["files"])
        with self.assertRaises(FileExistsError):
            export_merged(ckpt, self.model_path, out, "MOD-RV", binding=self.binding,
                _test_policy_factory=fresh_tiny, _test_base_manifest=self.base_manifest)

    def test_export_identity_rejects_wrong_precision_hash_and_path(self):
        ckpt, _ = self.checkpoint(0)
        out = self.root / "identity-export"
        manifest = export_merged(ckpt, self.model_path, out, "MOD-RV", binding=self.binding,
            _test_policy_factory=fresh_tiny, _test_base_manifest=self.base_manifest)
        marker = out / "export_manifest.json"
        for field, value in (("status", "PENDING"), ("format", "unknown"), ("arm", "MOD-FML"),
                             ("base_id", "other/model"), ("base_revision", "0"*40),
                             ("precision_policy", manifest["precision_policy"] | {"export": "FP32"}),
                             ("precision_policy", manifest["precision_policy"] | {"unmerged_inference_bit_equivalent": 0}),
                             ("files", {"../escape": "a"*64}),
                             ("files", {"/etc/passwd": "a"*64}),
                             ("files", {"C:\\escape": "a"*64}),
                             ("files", {"config.json": "0"*64}), ("files", {})):
            marker.write_text(json.dumps(manifest | {field: value}))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_export_identity(out, "MOD-RV")
        marker.write_text(json.dumps(manifest))
        name = "config.json"
        (out / name).unlink()
        (out / name).symlink_to(self.model_path / name)
        with self.assertRaises(ValueError):
            validate_export_identity(out, "MOD-RV")

    def test_wrong_arm_pinned_base_corruption_and_source_destination_refused(self):
        ckpt, _ = self.checkpoint(0)
        kwargs = {"binding": self.binding, "_test_policy_factory": fresh_tiny,
                  "_test_base_manifest": self.base_manifest}
        with self.assertRaisesRegex(ValueError, "arm"):
            export_merged(ckpt, self.model_path, self.root / "bad", "MOD-FML", **kwargs)
        with self.assertRaisesRegex(ValueError, "outside"):
            export_merged(ckpt, self.model_path, self.model_path / "bad", "MOD-RV", **kwargs)
        (self.model_path / "tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "base file"):
            export_merged(ckpt, self.model_path, self.root / "bad2", "MOD-RV", **kwargs)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
    else:
        unittest.main()
