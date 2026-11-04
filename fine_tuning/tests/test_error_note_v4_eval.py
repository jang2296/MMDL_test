"""Version-4 evaluation wiring: final MMMU preregistration v6 (one declared FT-HYB-V4 export), a tiny real v4
export, and the external-dev registration of the hybrid / rl-only exports. CPU fixtures only."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fine_tuning.error_note import evaluate as evaluation
from fine_tuning.error_note.protocol import ARM, plain_protocol_v4
from fine_tuning.notebook_rl import proof_eval as pe
from fine_tuning.tests.test_error_note_v3_eval import model_ref as v3_model_ref
from fine_tuning.tests import test_error_note_v3_external_eval as v3_external  # module, so its tests run once
from fine_tuning.tests.test_notebook_rl_proof_eval import IDS, REST, variant_cfg, write_run, write_subset
from mmdl.runtime.artifacts import digest, sha256_file
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION

HASHES = {"rows": "a" * 64, "order": "b" * 64, "init": "c" * 64}


def v4_protocol(arm="hybrid", hashes=HASHES, steps=1, round_index=2):
    return plain_protocol_v4("bf16", steps, arm=arm, round_index=round_index, start=500, limit=3000, contracts=hashes)


def v4_model_ref(root: Path, *, arm: str = "hybrid", name: str = "model_ref.json") -> Path:
    protocol = v4_protocol(arm)
    binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
               "data_sha256": HASHES["rows"], "code_sha256": "c" * 64}
    value = {"kind": "merged", "dtype": "bfloat16", "base": {"id": MODEL_ID, "revision": MODEL_REVISION},
             "processor_revision": MODEL_REVISION, "train_config_sha256": digest(protocol),
             "data_manifest_sha256": HASHES["rows"], "files": [{"path": "fixture.safetensors", "bytes": 1, "sha256": "e" * 64}],
             "export": {"arm": ARM, "step": 1, "binding": binding, "training_protocol": protocol,
                        "checkpoint_files": {"metadata.json": "d" * 64}}}
    path = root / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class V6FinalEvaluation(unittest.TestCase):
    template: Path

    @classmethod
    def setUpClass(cls) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="note-v4-eval-template-")
        cls.addClassCleanup(tmp.cleanup)
        root = cls.template = Path(tmp.name)
        b2 = root / "b2.jsonl"
        b2.write_text("".join(json.dumps({"id": sid, "correct": i >= 342}) + "\n" for i, sid in enumerate(IDS)))
        wrong, rest = write_subset(root / "wrong.json", b2), write_subset(root / "rest.json", b2, kind="rest")
        ref = v4_model_ref(root)
        pe.preregister(root / "prereg.json", wrong, rest, b2, "v6", ft_v4_model_ref=ref)
        loaded = json.loads((root / "prereg.json").read_text())
        cfg = variant_cfg(pe.GATE_62_VARIANT)
        with mock.patch.object(pe, "gpu_info", return_value=("NVIDIA A40", 46068)), mock.patch.dict(os.environ):
            os.environ.pop(pe.SLOTS_ENV, None)
            os.environ[pe.VLLM_INPROCESS_ENV] = "0"
            write_run(root, "ft-wrong", cfg, loaded["arms"][pe.ARM_V4]["model_sha256"], lambda i: i < 49)
            write_run(root, "ft-rest", cfg, loaded["arms"][pe.ARM_V4]["model_sha256"], lambda i: True, ids=REST)
            write_run(root, "f0-wrong", cfg, loaded["arms"]["F0"]["model_sha256"], lambda i: i % 2 == 0)
            write_run(root, "f0-rest", cfg, loaded["arms"]["F0"]["model_sha256"], lambda i: i % 3 != 0, ids=REST)

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="note-v4-eval-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "volume"
        shutil.copytree(self.template, self.root)
        self.prereg, self.ref, self.ledger = self.root / "prereg.json", self.root / "model_ref.json", self.root / "ledger.jsonl"

    def analyze(self, include_f0: bool = False) -> dict:
        runs = [self.root / "ft-wrong", self.root / "ft-rest"] + ([self.root / "f0-wrong", self.root / "f0-rest"]
                                                                if include_f0 else [])
        for run in runs:
            arm = "F0" if run.name.startswith("f0") else pe.ARM_V4
            pe.ledger_start(self.prereg, arm, pe.GATE_62_VARIANT, run.name, self.ledger, run.name.split("-")[-1])
        return pe.analyze(self.prereg, runs, {pe.ARM_V4: self.ref}, self.ledger)

    def test_v6_declares_one_hybrid_model(self) -> None:
        prereg, _ = pe.load_preregistration(self.prereg)
        self.assertEqual((prereg["format"], set(prereg["arms"])), (pe.FORMAT_V6, {"F0", pe.ARM_V4}))
        self.assertEqual(prereg["arms"][pe.ARM_V4]["model_ref_sha256"], sha256_file(self.ref))
        self.assertEqual(prereg["arms"][pe.ARM_V4]["round"], 2)
        self.assertEqual((prereg["gate_62"]["minimum_correct"], prereg["gate_67.4"]["minimum_correct"]), (559, 607))
        with self.assertRaisesRegex(ValueError, "requires --ft-v4-model-ref"):
            pe.preregister(self.root / "x.json", self.root / "wrong.json", self.root / "rest.json", self.root / "b2.jsonl", "v6")
        with self.assertRaisesRegex(ValueError, "error-note-protocol-v4 export"):
            pe.preregister(self.root / "y.json", self.root / "wrong.json", self.root / "rest.json", self.root / "b2.jsonl",
                           "v6", ft_v4_model_ref=v3_model_ref(self.root / "ft-wrong"))
        with self.assertRaisesRegex(ValueError, "one final BF16 hybrid export"):
            pe.preregister(self.root / "z.json", self.root / "wrong.json", self.root / "rest.json", self.root / "b2.jsonl",
                           "v6", ft_v4_model_ref=v4_model_ref(self.root, arm="rl-only", name="control.json"))
        with self.assertRaises(ValueError):  # a v4 export cannot pass as a v3 NOTE export
            pe.preregister(self.root / "w.json", self.root / "wrong.json", self.root / "rest.json", self.root / "b2.jsonl",
                           "v5", self.ref)

    def test_single_attempt_ledger_and_final_gates(self) -> None:
        pe.ledger_start(self.prereg, pe.ARM_V4, pe.GATE_62_VARIANT, "one", self.ledger, "wrong")
        with self.assertRaises(ValueError):
            pe.ledger_start(self.prereg, pe.ARM_V4, pe.GATE_62_VARIANT, "two", self.ledger, "wrong")
        with self.assertRaisesRegex(ValueError, "not a preregistered run"):
            pe.ledger_start(self.prereg, pe.ARM_V3, pe.GATE_62_VARIANT, "three", self.ledger, "rest")
        self.ledger.unlink()
        report = self.analyze(include_f0=True)
        self.assertEqual(report["preregistration_format"], pe.FORMAT_V6)
        self.assertEqual(report["full_900"][pe.ARM_V4]["seeds"]["3407"]["correct"], 607)
        gate = pe.gate_v3(report, 67.4)
        self.assertEqual((gate["format"], gate["arm"], gate["status"]), ("error-note-final-gate-v4", pe.ARM_V4, "PASS"))
        self.assertEqual(gate["ft_hyb_v4"]["correct"], 607)
        self.assertIsNotNone(gate["paired_f0"])
        self.assertEqual(pe.gate_62(report)["status"], "PASS")
        changed = copy.deepcopy(report)
        changed["full_900"][pe.ARM_V4]["seeds"]["3407"]["correct"] = 558
        self.assertEqual(pe.gate_62(changed)["status"], "FAIL")
        with self.assertRaisesRegex(ValueError, "no 342-item pilot gate"):
            pe.gate_300(report)
        output = self.root / "gate.json"
        with mock.patch("builtins.print") as printed:
            self.assertEqual(pe.main(["gate-67.4", "--prereg", str(self.prereg), "--ledger", str(self.ledger),
                                      "--runs", *(str(self.root / n) for n in ("ft-wrong", "ft-rest", "f0-wrong", "f0-rest")),
                                      "--output", str(output)]), 0)
        self.assertIn("ft_hyb_v4=607", printed.call_args[0][0])


    def test_eval_stage_verifies_the_v4_export_before_any_start(self) -> None:
        from fine_tuning.notebook_rl.pod import chain, eval_stage
        from fine_tuning.tests.test_notebook_rl_proof_eval import SUBSET

        proof = self.root / "eval-data/proof"
        proof.mkdir(parents=True)
        for tag in ("wrong", "rest"):
            subset = proof / f"{tag}.json"
            shutil.copyfile(self.root / f"{tag}.json", subset)
            p = {"version": eval_stage.PARAMS_VERSION, "commit": "a" * 40, "repo_root": str(pe.ROOT),
                 "python": f"{self.root}/venv-0123456789abcdef/bin/python", "volume_root": str(self.root),
                 "artifact_root": str(proof / "artifacts"), "prereg": str(proof / "prereg.json"),
                 "prereg_sha256": sha256_file(self.prereg), "ledger": str(proof / "ledger.jsonl"), "run_prefix": "final-v4",
                 "subset": str(subset), "subset_sha256": sha256_file(subset), "subset_tag": tag,
                 "arm": pe.ARM_V4, "seeds": [3407], "vllm_inprocess": True, "eval_slots": "auto",
                 "checkpoint": str(self.root / "checkpoint-final"), "checkpoint_metadata_sha256": "d" * 64,
                 "binding": str(self.root / "binding.json"), "binding_sha256": "e" * 64, "training_commit": "f" * 40,
                 "ft_export": str(self.root), "reuse_export": True, "ft_model_ref_sha256": sha256_file(self.ref)}
            spec = eval_stage.plan("eval-ft", p)
            self.assertIs(chain.check_spec(spec), spec)
            names = [step["name"] for step in spec["steps"]]
            self.assertLess(names.index("verify-v4-export"), next(i for i, name in enumerate(names) if name.startswith("ledger-")))
            evaluate = next(step["argv"] for step in spec["steps"] if step["name"].startswith("eval-"))
            self.assertEqual([evaluate[i + 1] for i, arg in enumerate(evaluate) if arg == "--sample-id"],
                             SUBSET if tag == "wrong" else REST)
            for change in ({"seeds": [3408]}, {"vllm_inprocess": False}, {"reuse_export": False}):
                with self.assertRaises(ValueError):
                    eval_stage.plan("eval-ft", p | change)


class V4TinyExport(unittest.TestCase):
    def test_final_v4_checkpoint_exports_and_verifies(self) -> None:
        import torch

        from fine_tuning.notebook_rl import checkpoint as ckpt
        from fine_tuning.tests.test_notebook_rl_checkpoint import action_step, fixture, fresh_tiny

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="note-v4-tiny-export-") as tmp:
            root = Path(tmp)
            base, base_manifest, _, _ = fixture(root)
            protocol = v4_protocol()
            binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
                       "data_sha256": HASHES["rows"], "code_sha256": "c" * 64}
            bundle = fresh_tiny(base, ARM, protocol)
            optimizer, scheduler = ckpt.make_optimizer(bundle, protocol)
            cursor = {"epoch": 0, "position": 0, "discarded_problem_ids": []}
            ckpt.save(bundle, optimizer, scheduler, 0, cursor, ckpt.capture_rng(), root / "intermediate",
                      protocol=protocol, binding=binding)
            binding_file = root / "binding.json"
            binding_file.write_text(json.dumps(binding))
            with self.assertRaisesRegex(ValueError, "v4 export requires the final BF16 checkpoint"):
                pe.export_for_eval(root / "intermediate", binding_file, base, root / "invalid-export", "f" * 40)
            action_step(bundle, optimizer, scheduler)
            cursor["position"] = 1
            ckpt.save(bundle, optimizer, scheduler, 1, cursor, ckpt.capture_rng(), root / "checkpoint-final",
                      protocol=protocol, binding=binding)
            ref = pe.export_for_eval(root / "checkpoint-final", binding_file, base, root / "export", "f" * 40,
                                     _test_policy_factory=fresh_tiny, _test_base_manifest=base_manifest)
            self.assertEqual(ref["export"]["training_protocol"], protocol)
            model_path = root / "export/model_ref.json"
            pe._v4_model_ref(model_path)
            b2 = root / "b2.jsonl"
            b2.write_text("".join(json.dumps({"id": sid, "correct": i >= 342}) + "\n" for i, sid in enumerate(IDS)))
            wrong, rest = write_subset(root / "wrong.json", b2), write_subset(root / "rest.json", b2, kind="rest")
            prereg = root / "prereg.json"
            pe.preregister(prereg, wrong, rest, b2, "v6", ft_v4_model_ref=model_path)
            argv = ["verify-v4-export", "--model-ref", str(model_path), "--checkpoint", str(root / "checkpoint-final"),
                    "--binding", str(binding_file), "--prereg", str(prereg)]
            with mock.patch("builtins.print"):
                self.assertEqual(pe.main(argv), 0)


class V4ExternalDev(unittest.TestCase):
    """The v3 external-dev fixture with its NOTE / NO-NOTE exports replaced by v4 hybrid / rl-only exports."""

    def setUp(self) -> None:
        v3_external.ExternalEvaluationTests.setUp(self)  # type: ignore[arg-type]
        self.arms = {arm: json.loads(Path(item["model_ref"]).read_text())
                     for arm, item in json.loads(self.models.read_text())["arms"].items()}

    def convert(self, arm: str, *, train_arm: str, round_index: int = 2, init: str = "c" * 64) -> None:
        item = json.loads(self.models.read_text())["arms"][arm]
        model, ref_path = Path(item["model_path"]), Path(item["model_ref"])
        ref = json.loads(ref_path.read_text())
        hashes = {"rows": evaluation.file_hash(self.rows), "order": evaluation.file_hash(self.order), "init": init}
        protocol = v4_protocol(train_arm, hashes, round_index=round_index)
        binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
                   "data_sha256": hashes["rows"], "code_sha256": "b" * 64}
        exported = json.loads((model / "export_manifest.json").read_text())
        exported |= {"binding": binding, "resolved_protocol_sha256": evaluation.digest(protocol)}
        (model / "export_manifest.json").write_text(json.dumps(exported))
        ref["train_config_sha256"] = evaluation.digest(protocol)
        ref["artifact_revision"] = evaluation.file_hash(model / "export_manifest.json")
        ref["export"] |= {"binding": binding, "training_protocol": protocol,
                          "export_manifest_sha256": evaluation.digest(exported)}
        ref["files"] = [{"path": path.name, "bytes": path.stat().st_size, "sha256": evaluation.file_hash(path)}
                        for path in sorted(model.iterdir())]
        ref_path.write_text(json.dumps(ref))

    def test_hybrid_and_rl_only_exports_register(self) -> None:
        self.convert("NOTE", train_arm="hybrid", init="c" * 64)
        self.convert("NO-NOTE", train_arm="rl-only", init="d" * 64)  # each arm chains its own earlier round
        registered = evaluation.preregister(self.args)
        self.assertEqual(set(registered["models"]), set(evaluation.ARMS))

    def test_mismatched_v4_exports_are_refused(self) -> None:
        for note_arm, control_arm, control_round in (("rl-only", "rl-only", 2), ("hybrid", "hybrid", 2),
                                                     ("hybrid", "rl-only", 3)):
            with self.subTest(note=note_arm, control=control_arm, round=control_round):
                v3_external.ExternalEvaluationTests.setUp(self)  # type: ignore[arg-type]
                self.convert("NOTE", train_arm=note_arm)
                self.convert("NO-NOTE", train_arm=control_arm, round_index=control_round, init="d" * 64)
                with self.assertRaises(ValueError):
                    evaluation.preregister(self.args)


if __name__ == "__main__":
    unittest.main()
