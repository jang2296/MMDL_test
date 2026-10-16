"""Authored private CPU handoff fixtures; never a real-data/GPU approval."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl import gate_inputs as handoff
from fine_tuning.notebook_rl.engines import RolloutRequest
from fine_tuning.notebook_rl.schema import ActorItem, Group, ProblemTree, Sample, to_json
from fine_tuning.notebook_rl.trainer import SyncProbe
from fine_tuning.tests.test_notebook_rl_policy import config


class GateInputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="notebook-g12-inputs-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image_root = self.root / "images"
        self.image_root.mkdir()
        for name, rgb in (("one.png", (31, 63, 127)), ("two.png", (127, 63, 31))):
            Image.new("RGB", (8, 8), rgb).save(self.image_root / name)
        self.protocol = config()
        self.binding = {"protocol_sha256": self.protocol["sha256"],
            "prompts_sha256": self.protocol["prompts_sha256"], "code_sha256": "c" * 64, "data_sha256": "d" * 64}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.actors = {"p": self.actor("p", ("one.png",)), "q": self.actor("q", ("one.png", "two.png"))}
        self.images = {"p": handoff._load_images(self.actors["p"], self.image_root, self.stack)}
        sample = Sample("p:v1:T1:0", "p", "p:v1:T1", "T1", 1, None, [1, 2], [3, 4],
                        [-.1, -.2], "stop", "Final answer: 4", 3407, 1)
        self.trees = [ProblemTree("p", 1, [Group("p:v1:T1", "T1", "p", "e" * 64,
                              [sample, replace(sample, sample_id="p:v1:T1:1", seed=3408)],
                              True, None)], "VISION_OFF", {}, {})]
        self.probes = [SyncProbe(RolloutRequest("single", None, [{"role": "user", "content": "authored"}],
                            self.images["p"], 1, 3, [3407]), [10, 2]),
                       SyncProbe(RolloutRequest("multi", [1, 5, 6], None,
                            handoff._load_images(self.actors["q"], self.image_root, self.stack),
                            1, 4, [3408]), [11, 12, 2])]
        self.preparation = SimpleNamespace(policy_version=1,
            prepare=lambda r: SimpleNamespace(token_ids=r.prompt_token_ids or [1, 2, 3]))
        self.kwargs = dict(trees=self.trees, images=self.images, actors=self.actors, probes=self.probes,
            probe_actors={"single": self.actors["p"], "multi": self.actors["q"]}, preparation=self.preparation,
            image_root=self.image_root, protocol=self.protocol, binding=self.binding, step=1,
            algorithm="ALG-NB3", seed=3407, cursor={"epoch": 0, "position": 1, "discarded_problem_ids": []})

    def actor(self, pid, names):
        return ActorItem(pid, "AUTHORED_CPU_FIXTURE", "fixture", "plain", "Authored question", (), "int",
                         names, tuple(hashlib.sha256((self.image_root / n).read_bytes()).hexdigest() for n in names))

    def save(self, **kwargs):
        return handoff.save_inputs(self.root / "inputs.json", **(self.kwargs | kwargs))

    def load(self, desc):
        return handoff.load_inputs(desc, image_root=self.image_root, protocol=self.protocol,
                                   binding=self.binding, stack=self.stack)

    def test_roundtrip_literalizes_messages_and_preserves_every_request_field(self):
        desc = self.save()
        result = self.load(desc)
        self.assertEqual([to_json(t) for t in result.trees], [to_json(t) for t in self.trees])
        self.assertEqual(handoff._pixels(result.images["p"]), handoff._pixels(self.images["p"]))
        self.assertEqual([p.request.prompt_token_ids for p in result.probes], [[1, 2, 3], [1, 5, 6]])
        self.assertIsNone(result.probes[0].request.messages)
        self.assertIsNone(self.probes[0].request.prompt_token_ids)  # Caller is not mutated.
        for actual, expected in zip(result.probes, self.probes, strict=True):
            for key in ("request_id", "n", "max_new_tokens", "seeds"):
                self.assertEqual(getattr(actual.request, key), getattr(expected.request, key))
            self.assertEqual(actual.completion_ids, expected.completion_ids)
        packet = json.loads(Path(desc["path"]).read_text())
        self.assertFalse(packet["training_allowed"])
        self.assertFalse(packet["gpu_gates_certified"])
        self.assertEqual((result.step, result.algorithm, result.seed), (1, "ALG-NB3", 3407))
        with self.assertRaises(FileExistsError):
            self.save()

    def test_changed_packet_binding_pixels_and_source_bytes_rejected(self):
        desc = self.save()
        path = Path(desc["path"])
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "packet bytes"):
            self.load(desc)
        path.write_bytes(original)
        with patch.dict(self.binding, data_sha256="0" * 64), self.assertRaisesRegex(ValueError, "identity"):
            self.load(desc)
        Image.new("RGB", (8, 8), "red").save(self.image_root / "two.png")
        with self.assertRaisesRegex(ValueError, "image byte hash"):
            self.load(desc)

    def test_stale_actions_and_cross_problem_groups_cannot_be_persisted(self):
        mutations = [lambda t: setattr(t, "policy_version", 0),
                     lambda t: setattr(t.groups[0].samples[0], "policy_version", 0),
                     lambda t: setattr(t.groups[0], "problem_id", "q"),
                     lambda t: setattr(t.groups[0].samples[0], "group_id", "another")]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                trees = copy.deepcopy(self.trees)
                mutate(trees[0])
                with self.assertRaises(ValueError):
                    self.save(trees=trees)
                self.assertFalse((self.root / "inputs.json").exists())

    def test_changed_actual_images_or_wrong_source_order_rejected(self):
        changed = Image.new("RGB", (8, 8), "white")
        self.addCleanup(changed.close)
        with self.assertRaisesRegex(ValueError, "batch pixels"):
            self.save(images={"p": [changed]})
        wrong = replace(self.probes[1], request=replace(self.probes[1].request,
                                                       images=list(reversed(self.probes[1].request.images))))
        with self.assertRaisesRegex(ValueError, "probe pixels"):
            self.save(probes=[self.probes[0], wrong])

    def test_duplicate_probe_ids_and_unbounded_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.save(probes=[self.probes[0], self.probes[0]])
        wrong = replace(self.probes[1], completion_ids=[1] * 33)
        with self.assertRaisesRegex(ValueError, "bound"):
            self.save(probes=[self.probes[0], wrong])
        req = replace(self.probes[0].request, prompt_token_ids=[1] * (self.protocol["context_limit"] - 2), messages=None)
        with self.assertRaisesRegex(ValueError, "bound"):
            self.save(probes=[replace(self.probes[0], request=req), self.probes[1]])

    def test_symlink_input_output_and_write_into_image_root_rejected(self):
        self.save()
        alias = self.root / "alias.json"
        alias.symlink_to(self.root / "inputs.json")
        desc = {"path": str(alias), "file_sha256": hashlib.sha256(alias.read_bytes()).hexdigest()}
        with self.assertRaises(ValueError):
            self.load(desc)
        with self.assertRaisesRegex(ValueError, "private output"):
            handoff.save_inputs(self.image_root / "bad.json", **self.kwargs)

    def test_missing_capture_binding_cannot_start_replay(self):
        desc = self.save()
        capture = self.root / "capture"
        capture.mkdir()
        (capture / "inputs.json").write_text(json.dumps({"input_packet": desc, "capture_result_sha256": "0" * 64}))
        (capture / "result.json").write_text("{}")
        with patch("fine_tuning.notebook_rl.gate_checkpoint.replay_checkpoint",
                   side_effect=AssertionError("must not allocate")), self.assertRaisesRegex(ValueError, "bound"):
            handoff.replay_from_inputs(self.root / "model", capture, self.root / "replay", desc,
                image_root=self.image_root, protocol=self.protocol, binding=self.binding, device="cuda:0")

    def test_wrappers_bind_completed_capture_and_set_replayed_preparation_version(self):
        from fine_tuning.notebook_rl import gate_checkpoint as gate

        desc = self.save()
        capture = self.root / "capture"
        bundle = SimpleNamespace(processor=SimpleNamespace(tokenizer=object()), model=SimpleNamespace(config=object()))
        def observed_capture(b, opt, sched, trees, images, prep, probes, protocol, algorithm, **kw):
            self.assertIs(b, bundle)
            self.assertEqual([to_json(t) for t in trees], [to_json(t) for t in self.trees])
            self.assertEqual(probes[0].request.prompt_token_ids, [1, 2, 3])
            self.assertEqual(kw["step"], 1)
            self.assertEqual(kw["cursor"], self.kwargs["cursor"])
            kw["output"].mkdir()
            result = {"step": kw["step"], "cursor": kw["cursor"], "binding": kw["binding"],
                "measured": {"batch": gate._batch(trees, images, algorithm, kw["seed"], False)}}
            (kw["output"] / "result.json").write_text(json.dumps(result))
            return result
        with patch.object(gate, "capture_checkpoint", side_effect=observed_capture) as native:
            handoff.capture_from_inputs(bundle, object(), object(), self.preparation, desc,
                image_root=self.image_root, protocol=self.protocol, binding=self.binding,
                model_path=self.root / "model", output=capture)
        native.assert_called_once()
        receipt = json.loads((capture / "inputs.json").read_text())
        self.assertEqual(receipt["input_packet"], desc)
        def observed_replay(model_path, previous, output, trees, images, probes, factory, **kw):
            self.assertEqual(previous, capture)
            self.assertEqual(kw["device"], "cuda:0")
            self.assertEqual(probes[1].request.seeds, [3408])
            prep = factory(bundle, 1)
            self.assertEqual(prep.policy_version, 1)
            self.assertEqual(prep.image_cfg, self.protocol["image"])
            with self.assertRaisesRegex(ValueError, "version differs"):
                factory(bundle, 0)
            return {"authored_wrapper_test": True, "native_execution": False}
        with patch.object(gate, "replay_checkpoint", side_effect=observed_replay) as native:
            result = handoff.replay_from_inputs(self.root / "model", capture, self.root / "replay", desc,
                image_root=self.image_root, protocol=self.protocol, binding=self.binding, device="cuda:0")
        native.assert_called_once()
        self.assertFalse(result["native_execution"])
        changed = json.loads((capture / "result.json").read_text())
        changed["measured"]["batch"]["seed"] += 1
        (capture / "result.json").write_text(json.dumps(changed))
        receipt["capture_result_sha256"] = gate.checkpoint.file_hash(capture / "result.json")
        (capture / "inputs.json").write_text(json.dumps(receipt))
        with patch.object(gate, "replay_checkpoint", side_effect=AssertionError("must not allocate")), \
                self.assertRaisesRegex(ValueError, "captured RL batch"):
            handoff.replay_from_inputs(self.root / "model", capture, self.root / "replay", desc,
                image_root=self.image_root, protocol=self.protocol, binding=self.binding, device="cuda:0")


if __name__ == "__main__":
    unittest.main()
