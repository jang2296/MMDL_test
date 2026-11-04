"""Final v3 evaluation registration, synthetic 900-item ledgers and a tiny real BF16 export."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fine_tuning.error_note.protocol import ARM, plain_protocol_v3
from fine_tuning.notebook_rl import proof_eval as pe
from fine_tuning.notebook_rl.pod import chain, eval_stage
from fine_tuning.tests.test_notebook_rl_proof_eval import IDS, REST, SUBSET, variant_cfg, write_run, write_subset
from mmdl.runtime.artifacts import digest, sha256_file
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION


def model_ref(root: Path, *, arm: str = "note") -> Path:
    hashes = {key: "a" * 64 for key in ("rows", "order", "audit", "sidecar", "solution_audit", "admission")}
    protocol = plain_protocol_v3("bf16", 1, arm=arm, contracts=hashes)
    binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
               "data_sha256": hashes["rows"], "code_sha256": "c" * 64}
    value = {"kind": "merged", "dtype": "bfloat16", "base": {"id": MODEL_ID, "revision": MODEL_REVISION},
             "processor_revision": MODEL_REVISION, "train_config_sha256": digest(protocol),
             "data_manifest_sha256": hashes["rows"], "files": [{"path": "fixture.safetensors", "bytes": 1, "sha256": "b" * 64}],
             "export": {"arm": ARM, "step": 1, "binding": binding, "training_protocol": protocol,
                        "checkpoint_files": {"metadata.json": "d" * 64}}}
    path = root / "model_ref.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class V3FinalEvaluationTests(unittest.TestCase):
    template: Path

    @classmethod
    def setUpClass(cls) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="note-v3-eval-template-")
        cls.addClassCleanup(tmp.cleanup)
        root = cls.template = Path(tmp.name)
        b2 = root / "b2.jsonl"
        b2.write_text("".join(json.dumps({"id": sid, "correct": i >= 342}) + "\n" for i, sid in enumerate(IDS)))
        wrong, rest = write_subset(root / "wrong.json", b2), write_subset(root / "rest.json", b2, kind="rest")
        ref = model_ref(root)
        prereg = root / "prereg.json"
        pe.preregister(prereg, wrong, rest, b2, "v5", ref)
        loaded = json.loads(prereg.read_text())
        cfg = variant_cfg(pe.GATE_62_VARIANT)
        with mock.patch.object(pe, "gpu_info", return_value=("NVIDIA A40", 46068)), mock.patch.dict(os.environ):
            os.environ.pop(pe.SLOTS_ENV, None)
            os.environ[pe.VLLM_INPROCESS_ENV] = "0"
            write_run(root, "ft-wrong", cfg, loaded["arms"][pe.ARM_V3]["model_sha256"], lambda i: i < 49)
            write_run(root, "ft-rest", cfg, loaded["arms"][pe.ARM_V3]["model_sha256"], lambda i: True, ids=REST)
            write_run(root, "f0-wrong", cfg, loaded["arms"]["F0"]["model_sha256"], lambda i: i % 2 == 0)
            write_run(root, "f0-rest", cfg, loaded["arms"]["F0"]["model_sha256"], lambda i: i % 3 != 0, ids=REST)

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="note-v3-eval-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "volume"
        shutil.copytree(self.template, self.root)
        self.prereg, self.ref, self.ledger = self.root / "prereg.json", self.root / "model_ref.json", self.root / "ledger.jsonl"
        self.ft = [self.root / "ft-wrong", self.root / "ft-rest"]
        self.f0 = [self.root / "f0-wrong", self.root / "f0-rest"]

    def analyze(self, include_f0: bool = False) -> dict:
        runs = self.ft + (self.f0 if include_f0 else [])
        for run in runs:
            arm = "F0" if run.name.startswith("f0") else pe.ARM_V3
            pe.ledger_start(self.prereg, arm, pe.GATE_62_VARIANT, run.name, self.ledger, run.name.split("-")[-1])
        return pe.analyze(self.prereg, runs, {pe.ARM_V3: self.ref}, self.ledger)

    def test_v5_declares_one_note_model_one_seed_and_all_original_ids(self) -> None:
        prereg, _ = pe.load_preregistration(self.prereg)
        self.assertEqual(prereg["format"], pe.FORMAT_V5)
        self.assertEqual(set(prereg["arms"]), {"F0", "FT-NOTE-V3"})
        self.assertEqual(prereg["arms"][pe.ARM_V3]["model_ref_sha256"], sha256_file(self.ref))
        self.assertEqual(prereg["runs"], [{"arm": arm, "variant": pe.GATE_62_VARIANT, "subset": tag, "role": "statistical"}
                                          for arm in ("F0", pe.ARM_V3) for tag in ("wrong", "rest")])
        self.assertFalse(set(prereg["evaluation_subsets"]["wrong"]["ids"]) & set(prereg["evaluation_subsets"]["rest"]["ids"]))
        self.assertEqual(set(SUBSET) | set(REST), set(IDS))
        self.assertNotIn("gate_300", prereg)
        with self.assertRaisesRegex(ValueError, "requires --ft-v3-model-ref"):
            pe.preregister(self.root / "missing.json", self.root / "wrong.json", self.root / "rest.json", self.root / "b2.jsonl", "v5")
        target = self.root / "cli-prereg.json"
        with mock.patch("builtins.print"):
            self.assertEqual(pe.main(["preregister", "--format", "v5", "--output", str(target),
                                      "--subset", str(self.root / "wrong.json"), "--rest-subset", str(self.root / "rest.json"),
                                      "--b2-predictions", str(self.root / "b2.jsonl"), "--ft-v3-model-ref", str(self.ref)]), 0)
        self.assertEqual(pe.load_preregistration(target)[0]["arms"], prereg["arms"] | {
            pe.ARM_V3: prereg["arms"][pe.ARM_V3] | {"model_ref": str(self.ref.resolve())}})

    def test_ledger_refuses_wrong_arm_seed_and_repeat(self) -> None:
        for arm, variant in ((pe.ARM_V2, pe.GATE_62_VARIANT), ("FT-NOTE", pe.GATE_62_VARIANT),
                             (pe.ARM_V3, "mmmu-val-v8-8k-seed3408-v2")):
            with self.assertRaisesRegex(ValueError, "not a preregistered run"):
                pe.ledger_start(self.prereg, arm, variant, "wrong", self.ledger, "wrong")
        pe.ledger_start(self.prereg, pe.ARM_V3, pe.GATE_62_VARIANT, "one", self.ledger, "wrong")
        with self.assertRaises(ValueError):
            pe.ledger_start(self.prereg, pe.ARM_V3, pe.GATE_62_VARIANT, "two", self.ledger, "wrong")

    def test_final_gates_and_reference_only_f0(self) -> None:
        report = self.analyze()
        self.assertEqual(report["full_900"][pe.ARM_V3]["seeds"]["3407"]["correct"], 607)
        self.assertEqual(pe.gate_62(report)["status"], "PASS")
        gate = pe.gate_v3(report, 67.4, f0_reference=541)
        self.assertEqual((gate["status"], gate["minimum_correct"]), ("PASS", 607))
        self.assertIsNone(gate["paired_f0"])
        self.assertIn("different conditions/registration", gate["f0_reference"]["source"])
        for count, target, status in ((558, 62, "FAIL"), (559, 62, "PASS"), (606, 67.4, "FAIL"), (607, 67.4, "PASS")):
            changed = copy.deepcopy(report)
            changed["full_900"][pe.ARM_V3]["seeds"]["3407"]["correct"] = count
            self.assertEqual(pe.gate_v3(changed, target)["status"], status)
        with self.assertRaisesRegex(ValueError, "no 342-item pilot gate"):
            pe.gate_300(report)
        for command in ("gate-62", "gate-67.4"):
            output = self.root / f"{command}.json"
            with mock.patch("builtins.print"):
                self.assertEqual(pe.main([command, "--prereg", str(self.prereg), "--ledger", str(self.ledger),
                                          "--runs", *map(str, self.ft), "--output", str(output), "--f0-reference", "541"]), 0)
            self.assertEqual(json.loads(output.read_text())["arm"], pe.ARM_V3)

    def test_failed_v5_run_cannot_be_replaced_or_hidden_in_analysis(self) -> None:
        for arm, tag in ((pe.ARM_V3, "wrong"), ("F0", "rest")):
            with self.subTest(arm=arm):
                ledger = self.root / f"{arm}-failed.jsonl"
                pe.ledger_start(self.prereg, arm, pe.GATE_62_VARIANT, "first", ledger, tag)
                pe.ledger_fail(ledger, "first", "interrupted after a saved answer", self.root)
                with self.assertRaisesRegex(ValueError, "including a failed run"):
                    pe.ledger_start(self.prereg, arm, pe.GATE_62_VARIANT, "replacement", ledger, tag)
                # Legacy writer may permit retries; v5 analysis still refuses their selection.
                pe._append(ledger, "START", arm=arm, variant=pe.GATE_62_VARIANT, subset=tag,
                           run_id="replacement", preregistration_sha256=sha256_file(self.prereg))
                with self.assertRaisesRegex(ValueError, "repeated attempts"):
                    pe.analyze(self.prereg, [], {pe.ARM_V3: self.ref}, ledger)

    def test_current_f0_is_paired_and_mismatched_registration_is_refused(self) -> None:
        report = self.analyze(include_f0=True)
        paired = pe.gate_v3(report, 67.4)["paired_f0"]
        self.assertEqual((paired["correct"], paired["ft_minus_f0"]), (543, 64))
        self.assertIn("p_value", paired["mcnemar"])
        path = self.f0[0] / pe.REGISTRATION_FILE
        registration = json.loads(path.read_text()) | {"proof_eval_sha256": "old helper bytes"}
        path.write_text(json.dumps(registration))
        with self.assertRaisesRegex(ValueError, "share one registration"):
            pe.analyze(self.prereg, self.ft + self.f0, {pe.ARM_V3: self.ref}, self.ledger)

    def test_model_and_partial_or_forged_registration_refused(self) -> None:
        manifest = self.ft[0] / "run_manifest.json"
        value = json.loads(manifest.read_text())
        value["identity"]["model_sha256"] = "9" * 64
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "one bound export|checkpoint declared"):
            self.analyze()
        prereg = json.loads(self.prereg.read_text())
        prereg["runs"].append({"arm": pe.ARM_V3, "variant": "mmmu-val-v8-8k-seed3408-v2", "subset": "wrong", "role": "statistical"})
        self.prereg.write_text(json.dumps(prereg))
        self.prereg.with_name("prereg.json.sha256").write_text(sha256_file(self.prereg) + "\n")
        with self.assertRaisesRegex(ValueError, "one declared NOTE checkpoint"):
            pe.load_preregistration(self.prereg)

    def test_final_ref_rejects_no_note_and_changed_protocol_binding(self) -> None:
        model_ref(self.root, arm="no-note")
        with self.assertRaisesRegex(ValueError, "final BF16 NOTE export"):
            pe._v3_model_ref(self.ref)
        model_ref(self.root)
        ref = json.loads(self.ref.read_text())
        ref["export"]["binding"]["data_sha256"] = "9" * 64
        self.ref.write_text(json.dumps(ref))
        with self.assertRaisesRegex(ValueError, "matching protocol binding"):
            pe._v3_model_ref(self.ref)

    def test_eval_stage_pins_and_validates_export_before_any_start(self) -> None:
        proof = self.root / "eval-data/proof"
        proof.mkdir(parents=True)
        for tag in ("wrong", "rest"):
            subset = proof / f"{tag}.json"
            shutil.copyfile(self.root / f"{tag}.json", subset)
            p = {"version": eval_stage.PARAMS_VERSION, "commit": "a" * 40, "repo_root": str(pe.ROOT),
                 "python": f"{self.root}/venv-0123456789abcdef/bin/python", "volume_root": str(self.root),
                 "artifact_root": str(proof / "artifacts"), "prereg": str(proof / "prereg.json"),
                 "prereg_sha256": sha256_file(self.prereg), "ledger": str(proof / "ledger.jsonl"), "run_prefix": "final-v3",
                 "subset": str(subset), "subset_sha256": sha256_file(subset), "subset_tag": tag,
                 "arm": pe.ARM_V3, "seeds": [3407], "vllm_inprocess": True, "eval_slots": "auto",
                 "checkpoint": str(self.root / "checkpoint-final"), "checkpoint_metadata_sha256": "d" * 64,
                 "binding": str(self.root / "binding.json"), "binding_sha256": "e" * 64, "training_commit": "f" * 40,
                 "ft_export": str(self.root), "reuse_export": True, "ft_model_ref_sha256": sha256_file(self.ref)}
            spec = eval_stage.plan("eval-ft", p)
            self.assertIs(chain.check_spec(spec), spec)
            names = [step["name"] for step in spec["steps"]]
            self.assertLess(names.index("verify-v3-export"), next(i for i, name in enumerate(names) if name.startswith("ledger-")))
            self.assertNotIn("export-ft", names)
            evaluate = next(step["argv"] for step in spec["steps"] if step["name"].startswith("eval-"))
            self.assertEqual([evaluate[i + 1] for i, arg in enumerate(evaluate) if arg == "--sample-id"], SUBSET if tag == "wrong" else REST)
            self.assertIn("VLLM_ENABLE_V1_MULTIPROCESSING=0", evaluate)
            for change in ({"seeds": [3408]}, {"vllm_inprocess": False}, {"reuse_export": False}):
                with self.assertRaises(ValueError):
                    eval_stage.plan("eval-ft", p | change)


class V3TinyExportTests(unittest.TestCase):
    def test_final_v3_checkpoint_exports_and_verifies_while_intermediate_is_refused(self) -> None:
        import torch

        from fine_tuning.notebook_rl import checkpoint as ckpt
        from fine_tuning.tests.test_notebook_rl_checkpoint import action_step, fixture, fresh_tiny

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="note-v3-tiny-export-") as tmp:
            root = Path(tmp)
            base, base_manifest, _, _ = fixture(root)
            hashes = {key: "a" * 64 for key in ("rows", "order", "audit", "sidecar", "solution_audit", "admission")}
            protocol = plain_protocol_v3("bf16", 1, arm="note", contracts=hashes)
            binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
                       "data_sha256": hashes["rows"], "code_sha256": "c" * 64}
            bundle = fresh_tiny(base, ARM, protocol)
            optimizer, scheduler = ckpt.make_optimizer(bundle, protocol)
            cursor = {"epoch": 0, "position": 0, "discarded_problem_ids": []}
            ckpt.save(bundle, optimizer, scheduler, 0, cursor, ckpt.capture_rng(), root / "intermediate", protocol=protocol, binding=binding)
            binding_file = root / "binding.json"
            binding_file.write_text(json.dumps(binding))
            with self.assertRaisesRegex(ValueError, "final BF16 checkpoint"):
                pe.export_for_eval(root / "intermediate", binding_file, base, root / "invalid-export", "f" * 40)
            action_step(bundle, optimizer, scheduler)
            cursor["position"] = 1
            ckpt.save(bundle, optimizer, scheduler, 1, cursor, ckpt.capture_rng(), root / "checkpoint-final", protocol=protocol, binding=binding)
            ref = pe.export_for_eval(root / "checkpoint-final", binding_file, base, root / "export", "f" * 40,
                                     _test_policy_factory=fresh_tiny, _test_base_manifest=base_manifest)
            self.assertEqual(ref["export"]["training_protocol"], protocol)
            model_path = root / "export/model_ref.json"
            pe._v3_model_ref(model_path)
            b2 = root / "b2.jsonl"
            b2.write_text("".join(json.dumps({"id": sid, "correct": i >= 342}) + "\n" for i, sid in enumerate(IDS)))
            wrong, rest = write_subset(root / "wrong.json", b2), write_subset(root / "rest.json", b2, kind="rest")
            prereg = root / "prereg.json"
            pe.preregister(prereg, wrong, rest, b2, "v5", model_path)
            argv = ["verify-v3-export", "--model-ref", str(model_path), "--checkpoint", str(root / "checkpoint-final"),
                    "--binding", str(binding_file), "--prereg", str(prereg)]
            with mock.patch("builtins.print"):
                self.assertEqual(pe.main(argv), 0)
            # A second valid export cannot borrow the first checkpoint's metadata.
            with torch.no_grad():
                next(p for p in bundle.model.parameters() if p.requires_grad).add_(0.125)
            ckpt.save(bundle, optimizer, scheduler, 1, cursor, ckpt.capture_rng(), root / "other-checkpoint",
                      protocol=protocol, binding=binding)
            other = pe.export_for_eval(root / "other-checkpoint", binding_file, base, root / "other-export", "f" * 40,
                                       _test_policy_factory=fresh_tiny, _test_base_manifest=base_manifest)
            mixed = copy.deepcopy(ref)
            mixed["files"] = other["files"]
            mixed["artifact_revision"] = other["artifact_revision"]
            mixed["export"]["export_manifest_sha256"] = other["export"]["export_manifest_sha256"]
            mixed_path = root / "other-export/mixed-ref.json"
            mixed_path.write_text(json.dumps(mixed))
            mixed_prereg = root / "mixed-prereg.json"
            pe.preregister(mixed_prereg, wrong, rest, b2, "v5", mixed_path)
            with self.assertRaisesRegex(ValueError, "declared checkpoint reference differ"):
                pe.verify_v3_export(mixed_path, root / "checkpoint-final", binding_file, mixed_prereg)
            binding_file.write_text(json.dumps(binding | {"data_sha256": "9" * 64}))
            with self.assertRaises(ValueError):
                pe.main(argv)


if __name__ == "__main__":
    unittest.main()
