"""Real tiny-Qwen G12 subprocess roundtrip with the production reward update."""
from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from fine_tuning.modules import BASE_REVISION
from fine_tuning.notebook_rl import gate_checkpoint as gate
from fine_tuning.notebook_rl import gate_inputs as handoff
from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.engines import RolloutRequest
from fine_tuning.notebook_rl.schema import ActorItem, ProblemTree, from_json, to_json
from fine_tuning.notebook_rl.trainer import SyncProbe
from fine_tuning.notebook_rl.updates import optimizer_step
from fine_tuning.tests.test_notebook_rl_checkpoint import fixture, fresh_tiny
from fine_tuning.tests.test_notebook_rl_policy import config
from fine_tuning.tests.test_notebook_rl_updates import Preparation, tree


class ImagePreparation(Preparation):
    """Authored CPU pixels, real multimodal forwards; never native evidence."""
    def prepare(self, request):
        return SimpleNamespace(token_ids=request.prompt_token_ids)

    def model_inputs(self, request, completion, device):
        result = super().model_inputs(request, completion, device)
        if len(request.images) == 2:
            for key in ("pixel_values", "image_grid_thw"):
                result[key] = result[key].repeat(2, 1)
        return result


def prepare(bundle, step):
    value = ImagePreparation()
    value.policy_version = step
    return value


def probes():
    image = Image.new("RGB", (8, 8), (31, 63, 127))
    return [SyncProbe(RolloutRequest("one", [4, 3, 3, 3, 3, 5, 9], None,
                    [image], 1, 3, [3407]), [10, 11, 2]),
            SyncProbe(RolloutRequest("two", [4, 3, 3, 3, 3, 5, 4, 3, 3, 3, 3, 5, 9], None,
                    [image, image], 1, 3, [3408]), [12, 13, 2])]


def worker(root, stage):
    torch.set_num_threads(2)
    args = json.loads((root / "arguments.json").read_text())
    protocol, model_path = config(), root / BASE_REVISION
    images = {"p": [Image.new("RGB", (8, 8), (31, 63, 127))]}
    with ExitStack() as stack, \
            patch("torch.nn.functional.cross_entropy", side_effect=AssertionError("CE forbidden")), \
            patch("fine_tuning.train.AssistantCollator", side_effect=AssertionError("SFT forbidden")):
        if stage == "capture":
            b = fresh_tiny(model_path, "MOD-RV", protocol)
            opt, scheduler = make_optimizer(b, protocol)
            random.seed(3407)
            np.random.seed(3407)
            torch.manual_seed(3407)
            optimizer_step(b, opt, scheduler, [tree(b)], images, prepare(b, 0), protocol,
                           "ALG-T1", step=0, seed=3407)
            row = tree(b, version=1)
            (root / "tree.json").write_text(to_json(row))
            digest = gate.checkpoint.file_hash(root / "images/one.png")
            actors = {pid: ActorItem(pid, "AUTHORED_CPU", "fixture", "plain", "Authored question", (), "int",
                ("one.png",) * count, (digest,) * count) for pid, count in (("p", 1), ("q", 2))}
            descriptor = handoff.save_inputs(root / "replay_inputs.json", trees=[row], images=images,
                actors=actors, probes=probes(), probe_actors={"one": actors["p"], "two": actors["q"]},
                preparation=prepare(b, 1), image_root=root / "images", protocol=protocol, binding=args["binding"],
                step=1, algorithm="ALG-T1", seed=3407, cursor=args["cursor"])
            (root / "input_descriptor.json").write_text(json.dumps(descriptor))
            persisted = handoff.load_inputs(descriptor, image_root=root / "images", protocol=protocol,
                                           binding=args["binding"], stack=stack)
            b.route.apply(False)  # Reload must preserve OFF despite the next step's ON route.
            gate._capture_checkpoint(b, opt, scheduler, persisted.trees, persisted.images, prepare(b, 1), persisted.probes,
                protocol, "ALG-T1", step=1, seed=3407, cursor=args["cursor"], binding=args["binding"],
                model_path=model_path, output=root / "capture", _test_manifest=root / "tiny_base_manifest.json")
        else:
            binding = dict(args["binding"])
            if stage == "wrong-binding":
                binding["data_sha256"] = "0" * 64
            descriptor = json.loads((root / "input_descriptor.json").read_text())
            persisted = handoff.load_inputs(descriptor, image_root=root / "images", protocol=protocol,
                                           binding=args["binding"], stack=stack)
            gate._replay_checkpoint(model_path, root / "capture", root / stage,
                persisted.trees, persisted.images, persisted.probes, prepare,
                protocol=protocol, binding=binding, device="cpu", _test_manifest=root / "tiny_base_manifest.json",
                _test_policy_factory=fresh_tiny)


class CheckpointGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.addClassCleanup(torch.set_num_threads, cls.threads)
        cls.tmp = tempfile.TemporaryDirectory(prefix="notebook-g12-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        _, _, _, cls.binding = fixture(cls.root)
        (cls.root / "images").mkdir()
        Image.new("RGB", (8, 8), (31, 63, 127)).save(cls.root / "images/one.png")
        cls.cursor = {"epoch": 0, "position": 2, "discarded_problem_ids": ["no-signal"],
                      "history": [{"step": 1}], "sync_pending": False, "order_sha256": "d" * 64}
        (cls.root / "arguments.json").write_text(json.dumps({"binding": cls.binding, "cursor": cls.cursor}))
        cls.capture = cls.child("capture")
        cls.replay = cls.child("replay")

    @classmethod
    def child(cls, stage, expected=0):
        process = subprocess.Popen([sys.executable, "-m", "fine_tuning.tests.test_notebook_rl_gate_checkpoint",
                                    "--worker", str(cls.root), stage],
            cwd=Path(__file__).resolve().parents[2], env=os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=60)
        if process.returncode != expected:
            raise AssertionError(f"{stage} exit {process.returncode}: {stdout[-1000:]} {stderr[-4000:]}")
        return process

    def compare(self, label):
        return gate.compare_checkpoint(self.root / "capture", self.root / "replay", self.capture,
                                       self.replay, self.root / label)

    def test_real_sequential_process_full_logits_next_rl_loss_and_state(self):
        result = self.compare("equal")
        self.assertTrue(result["passed"])
        self.assertTrue(result["fresh_process"])
        self.assertTrue(result["logits_equal"])
        self.assertTrue(result["next_loss_equal"])
        self.assertEqual(len(result["logit_errors"]), 8)
        self.assertTrue(all(e["max_abs"] == 0 and e["bit_equal"] for e in result["logit_errors"].values()))
        self.assertFalse(result["native_execution"])
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["gpu_gates_certified"])
        self.assertNotEqual(self.capture.pid, self.replay.pid)
        a = json.loads((self.root / "capture/result.json").read_text())
        b = json.loads((self.root / "replay/result.json").read_text())
        self.assertEqual(a["cursor"], self.cursor)
        for name in ("before", "after", "batch", "input_calls", "update", "probes"):
            self.assertEqual(a["measured"][name], b["measured"][name])
        self.assertFalse(a["measured"]["before"]["vision_on"])
        self.assertTrue(a["measured"]["after"]["vision_on"])
        self.assertEqual(a["measured"]["update"]["step_after"], 2)
        self.assertGreater(len(a["measured"]["input_calls"]), 2)
        self.assertTrue(a["measured"]["before"]["optimizer"]["state"])

    def test_changed_next_parameters_preserve_failure_details(self):
        path = self.root / "replay/result.json"
        original = path.read_bytes()
        altered = json.loads(original)
        altered["measured"]["after"]["parameters"][next(iter(altered["measured"]["after"]["parameters"]))]["sha256"] = "0" * 64
        try:
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "roundtrip differs"):
                self.compare("changed-parameter")
            report = json.loads((self.root / "changed-parameter/comparison.json").read_text())
            self.assertFalse(report["passed"])
            self.assertIn("after", report["different_fields"])
            failure = json.loads((self.root / "changed-parameter/error.json").read_text())
            self.assertIn("ValueError", failure["traceback"])
        finally:
            path.write_bytes(original)

    def test_raw_logits_corruption_refused(self):
        path = self.root / "replay/logits.safetensors"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "raw logits changed"):
                self.compare("changed-logits")
        finally:
            path.write_bytes(original)

    def test_measured_logit_difference_reports_actual_error(self):
        path, record = self.root / "replay/logits.safetensors", self.root / "replay/result.json"
        original, original_record = path.read_bytes(), record.read_bytes()
        try:
            tensors = load_file(str(path))
            tensors["0.policy.logits"][0, 0, 0] += .125
            save_file(tensors, str(path))
            changed = json.loads(original_record)
            changed["measured"]["logits_sha256"] = gate.checkpoint.file_hash(path)
            record.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "roundtrip differs"):
                self.compare("numeric-error")
            report = json.loads((self.root / "numeric-error/comparison.json").read_text())
            error = report["logit_errors"]["0.policy.logits"]
            self.assertFalse(error["bit_equal"])
            self.assertAlmostEqual(error["max_abs"], .125, places=6)
        finally:
            path.write_bytes(original)
            record.write_bytes(original_record)

    def test_no_matched_actions_cannot_pass_or_erase_saved_checkpoint(self):
        b, opt, scheduler, step, cursor = gate.checkpoint.load_resume(self.root / BASE_REVISION,
            self.root / "capture/checkpoint", config(), binding=self.binding, _test_policy_factory=fresh_tiny)
        row = from_json(ProblemTree, (self.root / "tree.json").read_text())
        for sample in row.groups[0].samples:
            sample.sampler_logprobs = [value - .1 for value in sample.sampler_logprobs]
        output = self.root / "no-matched-actions"
        with self.assertRaisesRegex(ValueError, "actual next production RL optimizer step"):
            gate._capture_checkpoint(b, opt, scheduler, [row], {"p": probes()[0].request.images},
                prepare(b, step), probes(), config(), "ALG-T1", step=step, seed=3407, cursor=cursor,
                binding=self.binding, model_path=self.root / BASE_REVISION, output=output,
                _test_manifest=self.root / "tiny_base_manifest.json")
        self.assertEqual(scheduler.last_epoch, step)
        self.assertTrue((output / "checkpoint/runtime.pt").exists())
        self.assertTrue((output / "error.json").exists())
        self.assertFalse((output / "result.json").exists())

    def test_completed_process_boolean_or_wrong_handle_cannot_claim_fresh_process(self):
        with self.assertRaisesRegex(ValueError, "actual owned child"):
            gate.compare_checkpoint(self.root / "capture", self.root / "replay", True,
                                    self.replay, self.root / "fake-process")
        with self.assertRaisesRegex(ValueError, "actual owned child"):
            gate.compare_checkpoint(self.root / "capture", self.root / "replay", self.capture,
                                    self.capture, self.root / "same-process")

    def test_same_live_capture_process_refused_before_model_load(self):
        path = self.root / "capture/result.json"
        original = path.read_bytes()
        changed = json.loads(original)
        changed["process"] = gate._process()
        try:
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "capture process must exit"), \
                    patch.object(gate.checkpoint, "load_resume", side_effect=AssertionError("must not load")):
                gate._replay_checkpoint(self.root / BASE_REVISION, self.root / "capture", self.root / "live",
                    [], {}, [], prepare, protocol=config(), binding=self.binding, device="cpu",
                    _test_manifest=self.root / "tiny_base_manifest.json", _test_policy_factory=fresh_tiny)
        finally:
            path.write_bytes(original)

    def test_wrong_binding_refused_in_real_child_and_raw_error_kept(self):
        self.child("wrong-binding", expected=1)
        failure = json.loads((self.root / "wrong-binding/error.json").read_text())
        self.assertIn("source or input binding differs", failure["message"])
        self.assertFalse((self.root / "wrong-binding/result.json").exists())

    def test_public_cpu_and_private_promotion_refused(self):
        b = fresh_tiny(self.root / BASE_REVISION, "MOD-RV", config())
        opt, scheduler = make_optimizer(b, config())
        with self.assertRaisesRegex(ValueError, "CUDA"):
            gate.capture_checkpoint(b, opt, scheduler, [], {}, prepare(b, 0), probes(), config(),
                "ALG-T1", step=0, seed=3407, cursor=self.cursor, binding=self.binding,
                model_path=self.root / BASE_REVISION, output=self.root / "public-cpu")
        with self.assertRaisesRegex(ValueError, "promote"):
            gate.replay_checkpoint(self.root / BASE_REVISION, self.root / "capture", self.root / "promotion",
                                   [], {}, [], prepare, protocol=config(), binding=self.binding, device="cpu")
        self.assertEqual(scheduler.last_epoch, 0)

    def test_source_file_mutation_pinned_identity_and_write_once(self):
        manifest = self.root / "tiny_base_manifest.json"
        original = manifest.read_bytes()
        altered = copy.deepcopy(json.loads(original))
        altered["revision"] = "0" * 40
        try:
            manifest.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "pinned source"):
                gate._model_files(self.root / BASE_REVISION, config(), manifest)
        finally:
            manifest.write_bytes(original)
        with self.assertRaises(FileExistsError):
            gate.compare_checkpoint(self.root / "capture", self.root / "replay", self.capture,
                                    self.replay, self.root / "capture")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), sys.argv[3])
    else:
        unittest.main()
