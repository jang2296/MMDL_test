"""Single-response campaign evaluation wiring on authored CPU fixtures: tiny real pilot/final exports bound by the
campaign plan, the external-dev F0/HYB mode, the pilot-gate JSON single_campaign._gate validates, and the one final
MMMU preregistration v7. No pretrained weights, benchmark rows, GPU or network."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from fine_tuning.error_note import audit, rl, single, single_campaign as campaign
from fine_tuning.error_note import evaluate as evaluation
from fine_tuning.error_note.protocol import (ARM, VERSION_SINGLE, VERSION_SINGLE_V12, VERSION_SINGLE_V13,
                                             plain_protocol_single)
from fine_tuning.notebook_rl import checkpoint as ckpt, proof_eval as pe
from fine_tuning.notebook_rl.pool_freeze import _RECORDS, _lines
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.tests.test_error_note_collect import FakeStream, default_output, no_gpu
from fine_tuning.tests.test_error_note_single_campaign import plan_for
from fine_tuning.tests.test_notebook_rl_checkpoint import fixture, fresh_tiny
from fine_tuning.tests.test_notebook_rl_proof_eval import IDS, REST, variant_cfg, write_run, write_subset
from fine_tuning.tests.test_notebook_rl_schema import records

DEV_SOURCES = ("hf-scienceqa", "hf-scienceqa", "hf-plantvillage", "fixture", "fixture", "fixture")
# F0: d0 wrong, d1/d2 right, d3 wrong, d4 no final answer, d5 token cap. HYB: all right.
F0_OUTPUTS = [("Final answer: A", "stop"), ("Final answer: B", "stop"), ("Final answer: B", "stop"),
              ("Final answer: A", "stop"), ("I am not sure.", "stop"), ("Step " * 20, "length")]


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def stream_for(outputs):
    def output(request):
        text, finish = outputs[int(request["request_id"].split("-")[1])]
        return default_output(request) | {"text": text, "finish_reason": finish,
                                          "completion_tokens": 8192 if finish == "length" else 5}
    return FakeStream(output)


class SingleCampaignEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        tmp = tempfile.TemporaryDirectory(prefix="single-eval-")
        cls.addClassCleanup(tmp.cleanup)
        root = cls.root = Path(tmp.name)
        for target, value in ((evaluation, "DEV_COUNT"),):
            patcher = mock.patch.object(target, value, len(DEV_SOURCES))
            patcher.start()
            cls.addClassCleanup(patcher.stop)
        cls.base, cls.base_manifest, _, _ = fixture(root)
        cls._dev_pool(root)
        cls._campaign(root)
        cls._dev_evaluation(root)

    # ------------------------------------------------------------------ fixtures
    @classmethod
    def _dev_pool(cls, root: Path) -> None:
        """Six held-out dev MCQ items (three in the retention sources) and the F0 baseline reference."""
        vault, (actor0, item0, lineage0, *_) = root / "dev-pool", records()
        (vault / "images").mkdir(parents=True)
        actors, items, lineages, rows = [], [], [], []
        for index, source in enumerate(DEV_SOURCES):
            pid, image = f"dev-{index}", vault / "images" / f"dev-{index}.png"
            Image.new("RGB", (2, 2), (index, 11, 23)).save(image)
            sha = evaluation.file_hash(image)
            actors.append(replace(actor0, problem_id=pid, source=source, family="fixture-family", mode="plain",
                                  question=f"Question {pid}?", choices=("alpha", "beta"), answer_format="mcq",
                                  image_paths=(image.name,), image_sha256=(sha,)))
            items.append(replace(item0, problem_id=pid, answer_type="mcq", gold="B", accept={}, choices=("alpha", "beta"),
                                 slots={}, slot_types={}, required_slots=(), oracle_fact_lines=(), printed_text=()))
            lineages.append(replace(lineage0, problem_id=pid, source=source, split="dev",
                                    structure_hash=hashlib.sha256(pid.encode()).hexdigest()))
            rows.append({"problem_id": pid, "source": source, "source_revision": lineage0.source_revision, "native_id": pid,
                         "split": "dev", "status": "HELD_OUT_DEV", "family": "fixture-family", "subject": "Authored",
                         "image_paths": [image.name], "image_sha256": [sha], "pixel_sha256": [sha]})
        write_split(vault / "dev", actors, items, lineages)
        raw = {kind: _lines(vault / "dev" / name) for kind, name in _RECORDS}
        for row in rows:
            row["record_sha256"] = {kind: hashlib.sha256(lines[row["problem_id"]]).hexdigest() for kind, lines in raw.items()}
        cls.vault, cls.manifest = vault, write(root / "dev-manifest.json", {"items": rows})
        cls.rules = write(root / "dev-rules.json", {"version": "external-dev-answer-rules-v1",
                          "vault_manifest_sha256": evaluation.file_hash(vault / "dev/manifest.yaml"), "items": {}})
        model = root / "model-F0"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"qwen3_vl"}')
        (model / "model.safetensors").write_bytes(b"F0")
        ref = {"kind": "base", "dtype": "bfloat16", "processor_revision": evaluation.BASE_REVISION,
               "repo_id": evaluation.BASE_ID, "revision": evaluation.BASE_REVISION,
               "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": evaluation.file_hash(p)}
                         for p in sorted(model.iterdir())]}
        cls.f0_model, cls.f0_ref = model, write(root / "baseline.json", ref)
        patcher = mock.patch.object(evaluation, "BASE_MODEL_REF", cls.f0_ref)
        patcher.start()
        cls.addClassCleanup(patcher.stop)

    @classmethod
    def _campaign(cls, root: Path) -> None:
        """A sealed 316-item campaign (19 pilot blocks + 1) with committed checkpoints 18/19 and a final checkpoint 20."""
        actor0, item0, lineage0, *_ = records()
        ids = [f"p{i}" for i in range(316)]
        write_split(root / "build/train",
                    [replace(actor0, problem_id=pid, source="authored-test", image_sha256=(f"{i:064x}",))
                     for i, pid in enumerate(ids)],
                    [replace(item0, problem_id=pid) for pid in ids],
                    [replace(lineage0, problem_id=pid, source="authored-test", split="train",
                             structure_hash=hashlib.sha256(f"train-{pid}".encode()).hexdigest()) for pid in ids])
        dev = evaluation.dev_ids(cls.manifest)
        plan = plan_for(316, "train")
        plan.update(build_root=str(root / "build"), data_admission={"train_count": 316, "pilot_count": 300},
                    dev_ids_sha256=dev["ids_sha256"], retention_ids_sha256=dev["retention_ids_sha256"])
        plan["sha256"] = audit.digest({k: v for k, v in plan.items() if k != "sha256"})
        out = cls.out = root / "campaign"
        out.mkdir()
        single._sealed(out / "campaign-plan.json", plan)
        cls.plan, cls.plan_file = plan, out / "campaign-plan.json"
        protocol, binding, blocks = plan["protocol"], campaign._binding(plan), plan["blocks"]
        torch.manual_seed(3407)
        bundle = fresh_tiny(cls.base, ARM, protocol)
        optimizer, scheduler = ckpt.make_optimizer(bundle, protocol)
        policy = rl.policy_sha256(bundle)

        def prefix(slot: int) -> list[str]:
            return [pid for block in blocks[:slot] for pid in block]

        def advance(steps: int) -> None:
            with warnings.catch_warnings():  # authored zero-signal history: scheduler steps without optimizer steps
                warnings.simplefilter("ignore")
                for _ in range(steps):
                    scheduler.step()

        def save(cursor: dict) -> dict:
            path = out / f"checkpoint-{cursor['slot']}"
            single.save_single_continuation(bundle, optimizer, scheduler, cursor=cursor, path=path, protocol=protocol,
                                           binding=binding, allow_save=True)
            return campaign._descriptor(path / "metadata.json")

        advance(18)
        parent = save({"slot": 18, "updates": 18, "consumed_problem_ids": prefix(18), "policy_sha256": policy,
                       "last_transaction_sha256": None, "parent_checkpoint": None})
        for slot in range(19):
            rows = [{"problem_id": pid, "kind": "RL", "status": "INCORRECT" if pid == "p1" else "CORRECT"}
                    for pid in blocks[slot]]
            if slot == 0:
                rows += [{"problem_id": "p0", "kind": "A1"}, {"problem_id": "p1", "kind": "A2"}, {"problem_id": "p1", "kind": "B"}]
            closed = write(out / f"block-{slot}/rows.json", {"status": "ROWS_READY", "rows": rows, "rows_sha256": audit.digest(rows),
                                                             "contract": {"expected_problem_ids": blocks[slot]}})
            transaction = {"version": campaign.VERSION, "plan_sha256": plan["sha256"], "slot": slot, "ids": blocks[slot],
                           "policy_sha256": policy, "parent_checkpoint": parent, "rows": campaign._descriptor(closed),
                           "contract_sha256": audit.digest({"expected_problem_ids": blocks[slot]})}
            single._sealed(out / f"transaction-{slot}.json", transaction)
        advance(1)
        cursor = {"slot": 19, "updates": 19, "consumed_problem_ids": prefix(19), "policy_sha256": policy,
                  "last_transaction_sha256": audit.digest(transaction), "parent_checkpoint": parent}
        pilot = save(cursor)
        single._state(out / "campaign-state.json", {"version": campaign.VERSION, "plan_sha256": plan["sha256"],
                      "cursor": cursor, "checkpoint": pilot, "pending": None, "rng": None, "proofs": [],
                      "status": "PILOT_COMPLETE", "pilot_checkpoint": pilot})
        cls.receipt = root / "reload-19.json"
        cls.receipt.write_text('noise\n{"status": "RELOADED", "slot": 19, "updates": 19}\n')
        advance(1)
        save({"slot": 20, "updates": 20, "consumed_problem_ids": prefix(20), "policy_sha256": policy,
              "last_transaction_sha256": "e" * 64, "parent_checkpoint": pilot})
        cls.exports = {slot: root / f"export-{slot}" for slot in (19, 20)}
        for slot, target in cls.exports.items():
            pe.export_for_eval(out / f"checkpoint-{slot}", cls.plan_file, cls.base, target, "f" * 40,
                               _test_policy_factory=fresh_tiny, _test_base_manifest=cls.base_manifest)

    @classmethod
    def _dev_evaluation(cls, root: Path) -> None:
        cls.runs, cls.prereg = root / "dev-runs", root / "dev-prereg.json"
        cls.registered = evaluation.preregister_hybrid(cls.hybrid_args(cls.plan_file, cls.exports[19], cls.prereg))
        for arm, outputs in (("F0", F0_OUTPUTS), ("HYB", [("Final answer: B", "stop")] * len(DEV_SOURCES))):
            args = argparse.Namespace(prereg=cls.prereg, arm=arm, output=cls.runs / arm)
            assert evaluation.run(args, stream=stream_for(outputs), load=lambda actor: [object()], guarded=False,
                                  gpu_probe=no_gpu) == 0
            with mock.patch.object(evaluation, "process_startticks", return_value=None):
                evaluation.grade(argparse.Namespace(prereg=cls.prereg, run=cls.runs / arm))
        cls.analysis = root / "dev-analysis.json"
        cls.report = evaluation.analyze_hybrid(argparse.Namespace(prereg=cls.prereg, f0=cls.runs / "F0",
                                                                  hyb=cls.runs / "HYB", output=cls.analysis))

    @classmethod
    def hybrid_args(cls, plan_file: Path, export: Path, output: Path, run_root: Path | None = None) -> argparse.Namespace:
        return argparse.Namespace(manifest=cls.manifest, vault_root=cls.vault, campaign_plan=plan_file,
                                  f0_model_path=cls.f0_model, hyb_export=export, dev_rules=cls.rules,
                                  run_root=run_root or cls.runs, code_commit="a" * 40, output=output)

    def gate_args(self, name: str, **changes) -> argparse.Namespace:
        values = {"analysis": self.analysis, "campaign_output": self.out, "reload_receipt": self.receipt,
                  "remaining_cost": 1.0, "remaining_budget": 2.0, "output": self.root / f"gate-{name}.json"}
        return argparse.Namespace(**(values | changes))

    # ------------------------------------------------------------------ export
    def test_campaign_exports_bind_plan_slot_and_refuse_other_checkpoints(self) -> None:
        from fine_tuning.notebook_rl.export import validate_export_reference

        refs = {slot: json.loads((target / "model_ref.json").read_text()) for slot, target in self.exports.items()}
        for slot, stage in ((19, "pilot"), (20, "final")):
            campaign_ref = refs[slot]["export"]["campaign"]
            self.assertEqual((campaign_ref["stage"], campaign_ref["slot"], campaign_ref["plan_sha256"]),
                             (stage, slot, self.plan["sha256"]))
            self.assertEqual(refs[slot]["export"]["binding"], campaign._binding(self.plan))
            self.assertEqual(refs[slot]["export"]["training_protocol"], self.plan["protocol"])
            self.assertEqual(refs[slot]["data_manifest_sha256"], self.plan["sha256"])
            validate_export_reference(self.exports[slot] / "model", refs[slot])
        self.assertEqual(refs[19]["export"]["checkpoint_files"]["metadata.json"],
                         evaluation.file_hash(self.out / "checkpoint-19/metadata.json"))
        with self.assertRaisesRegex(ValueError, "pilot \\(slot 19, 300 items\\) or the final"):
            pe.export_for_eval(self.out / "checkpoint-18", self.plan_file, self.base, self.root / "x18", "f" * 40,
                               _test_policy_factory=fresh_tiny, _test_base_manifest=self.base_manifest)
        plain = write(self.root / "plain-binding.json", campaign._binding(self.plan))
        with self.assertRaisesRegex(ValueError, "campaign-plan.json"):
            pe.export_for_eval(self.out / "checkpoint-19", plain, self.base, self.root / "x19", "f" * 40,
                               _test_policy_factory=fresh_tiny, _test_base_manifest=self.base_manifest)
        self.assertFalse((self.root / "x18").exists() or (self.root / "x19").exists())

    def test_single_stage_allows_a_stopped_main_checkpoint_only_after_the_pilot(self) -> None:
        self.assertEqual([pe._single_stage(*case) for case in ((19, 300, 219), (219, 3500, 219), (75, 1200, 219))],
                         ["pilot", "final", "stopped"])
        for case in ((18, 288, 219), (19, 304, 219), (0, 0, 219), (220, 3500, 219)):
            with self.assertRaisesRegex(ValueError, "or the final"):
                pe._single_stage(*case)

    # ------------------------------------------------------------------ external dev F0 / HYB
    def test_hybrid_registration_pins_dev_retention_and_pilot_export(self) -> None:
        dev = self.registered["dev"]
        self.assertEqual((dev["count"], dev["retention_count"]), (6, 3))
        self.assertEqual(dev["retention_ids"], ["dev-0", "dev-1", "dev-2"])
        self.assertEqual(dev["ids_sha256"], hashlib.sha256("\n".join(f"dev-{i}" for i in range(6)).encode()).hexdigest())
        self.assertEqual(set(self.registered["models"]), {"F0", "HYB"})
        self.assertEqual(self.registered["protocol"]["max_new_tokens"], 8192)
        self.assertEqual(self.registered["protocol"]["image"]["max_pixels"], 1310720)
        changed = copy.deepcopy(self.plan) | {"dev_ids_sha256": "0" * 64}
        changed["sha256"] = audit.digest({k: v for k, v in changed.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "preregistered SHA256"):
            evaluation.preregister_hybrid(self.hybrid_args(write(self.root / "changed-plan.json", changed),
                                                           self.exports[19], self.root / "unused.json", self.root / "unused-runs"))
        with self.assertRaisesRegex(ValueError, "pilot checkpoint-19 export"):
            evaluation._models_hybrid(self.f0_model, self.exports[20], self.plan)
        with self.assertRaisesRegex(ValueError, "preregistered run path"):
            evaluation.run(argparse.Namespace(prereg=self.prereg, arm="NOTE", output=self.runs / "NOTE"), guarded=False)

    def test_hybrid_analysis_counts_and_paired_statistics(self) -> None:
        arms, paired = self.report["arms"], self.report["paired"]["HYB-F0"]
        self.assertEqual({name: arms["F0"][name] for name in evaluation.GATE_METRICS},
                         {"correct": 2, "retention_correct": 2, "no_final": 2, "token_cap": 1})
        self.assertEqual({name: arms["HYB"][name] for name in evaluation.GATE_METRICS},
                         {"correct": 6, "retention_correct": 3, "no_final": 0, "token_cap": 0})
        self.assertEqual(arms["F0"]["statuses"], {"INCORRECT": 2, "CORRECT": 2, "INVALID_NO_FINAL": 1, "TRUNCATED": 1})
        self.assertEqual((paired["gains"], paired["losses"], paired["net"]), (4, 0, 4))
        self.assertEqual(paired["mcnemar_exact_descriptive"]["p_value"], 0.125)  # 2 * C(4, 0) / 2 ** 4
        low, high = paired["item_accuracy_ci95"]
        self.assertLessEqual(low, 4 / 6)
        self.assertGreaterEqual(high, 4 / 6)
        retention = self.report["paired"]["HYB-F0-retention"]
        self.assertEqual((retention["gains"], retention["losses"]), (1, 0))
        self.assertTrue(self.report["point_rule_holds"])
        self.assertEqual((self.report["system_failures"], self.report["missing"]), (0, 0))
        self.assertEqual(json.loads(self.analysis.read_text()), self.report)

    # ------------------------------------------------------------------ pilot gate
    def test_pilot_gate_is_accepted_by_campaign_gate(self) -> None:
        gate, status = evaluation.pilot_gate(self.gate_args("accepted"))
        self.assertEqual(status, "ACCEPTED", gate["check_failures"])
        self.assertEqual(gate["functional_checks"], dict.fromkeys(("data", "probability", "gradients", "function"), True))
        self.assertEqual(gate["pilot_checkpoint"], campaign._descriptor(self.out / "checkpoint-19/metadata.json"))
        state = single._state(self.out / "campaign-state.json")
        descriptor = campaign._descriptor(self.root / "gate-accepted.json")
        campaign._gate(self.plan, state, descriptor | {"raw": (self.root / "gate-accepted.json").read_bytes()}, True)
        self.assertEqual(state["main_gate"]["file_sha256"], descriptor["file_sha256"])
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            argv = ["pilot-gate", "--analysis", str(self.analysis), "--campaign-output", str(self.out),
                    "--reload-receipt", str(self.receipt), "--remaining-cost", "1", "--remaining-budget", "2",
                    "--output", str(self.root / "gate-cli.json")]
            self.assertEqual(evaluation.main(argv), 0)
        self.assertEqual(json.loads(printed.getvalue())["status"], "ACCEPTED")

    def test_pilot_gate_holds_on_failed_function_budget_or_changed_analysis(self) -> None:
        receipt = self.root / "reload-wrong.json"
        receipt.write_text('{"status": "RELOADED", "slot": 18, "updates": 18}\n')
        gate, status = evaluation.pilot_gate(self.gate_args("function", reload_receipt=receipt))
        self.assertTrue(status.startswith("HOLD"))
        self.assertFalse(gate["functional_checks"]["function"])
        gate, status = evaluation.pilot_gate(self.gate_args("budget", remaining_cost=3.0))
        self.assertTrue(status.startswith("HOLD"))
        self.assertTrue(all(gate["functional_checks"].values()))
        with self.assertRaisesRegex(ValueError, "operator"):
            evaluation.pilot_gate(self.gate_args("negative", remaining_cost=-1.0))
        tampered = json.loads(self.analysis.read_text())
        tampered["gate_metrics"]["f0"]["correct"] = 0
        with self.assertRaisesRegex(ValueError, "regrading"):
            evaluation.pilot_gate(self.gate_args("tampered", analysis=write(self.root / "tampered.json", tampered)))
        self.assertFalse((self.root / "gate-negative.json").exists() or (self.root / "gate-tampered.json").exists())

    # ------------------------------------------------------------------ final MMMU v7
    def test_final_v7_declares_one_export_verifies_it_and_passes_above_558(self) -> None:
        from fine_tuning.notebook_rl.pod import chain, eval_stage

        with tempfile.TemporaryDirectory(prefix="single-v7-") as tmp:
            root = Path(tmp)
            b2 = root / "b2.jsonl"
            b2.write_text("".join(json.dumps({"id": sid, "correct": i >= 342}) + "\n" for i, sid in enumerate(IDS)))
            wrong, rest = write_subset(root / "wrong.json", b2), write_subset(root / "rest.json", b2, kind="rest")
            ref, prereg, ledger = self.exports[20] / "model_ref.json", root / "prereg.json", root / "ledger.jsonl"
            with self.assertRaisesRegex(ValueError, "final BF16 single-campaign export"):
                pe.preregister(root / "pilot.json", wrong, rest, b2, "v7", ft_single_model_ref=self.exports[19] / "model_ref.json")
            with self.assertRaisesRegex(ValueError, "requires --ft-single-model-ref"):
                pe.preregister(root / "none.json", wrong, rest, b2, "v7")
            pe.preregister(prereg, wrong, rest, b2, "v7", ft_single_model_ref=ref)
            loaded, _ = pe.load_preregistration(prereg)
            self.assertEqual((loaded["format"], set(loaded["arms"])), (pe.FORMAT_V7, {"F0", pe.ARM_SINGLE}))
            self.assertEqual(loaded["arms"][pe.ARM_SINGLE]["campaign_plan_sha256"], self.plan["sha256"])
            self.assertEqual(loaded["gate_62"]["minimum_correct"], 559)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pe.main(["verify-single-export", "--model-ref", str(ref), "--checkpoint",
                                          str(self.out / "checkpoint-20"), "--binding", str(self.plan_file),
                                          "--prereg", str(prereg)]), 0)
            with self.assertRaisesRegex(ValueError, "checkpoint metadata or binding differs"):
                pe.verify_v3_export(ref, self.out / "checkpoint-19", self.plan_file, prereg)
            cfg = variant_cfg(pe.GATE_62_VARIANT)
            with mock.patch.object(pe, "gpu_info", return_value=("NVIDIA A40", 46068)), mock.patch.dict(os.environ):
                os.environ.pop(pe.SLOTS_ENV, None)
                os.environ[pe.VLLM_INPROCESS_ENV] = "0"
                model_sha = loaded["arms"][pe.ARM_SINGLE]["model_sha256"]
                write_run(root, "ft-wrong", cfg, model_sha, lambda i: i < 1)
                write_run(root, "ft-rest", cfg, model_sha, lambda i: True, ids=REST)
            for name in ("ft-wrong", "ft-rest"):
                pe.ledger_start(prereg, pe.ARM_SINGLE, pe.GATE_62_VARIANT, name, ledger, name.split("-")[-1])
            report = pe.analyze(prereg, [root / "ft-wrong", root / "ft-rest"], {pe.ARM_SINGLE: ref}, ledger)
            gate = pe.gate_62(report, 541)
            self.assertEqual((gate["format"], gate["status"], gate["ft_single"]["correct"]),
                             ("error-note-final-gate-single-v1", "PASS", 559))
            self.assertEqual(gate["f0_reference"]["correct"], 541)
            changed = copy.deepcopy(report)
            changed["full_900"][pe.ARM_SINGLE]["seeds"]["3407"]["correct"] = 558
            self.assertEqual(pe.gate_62(changed)["status"], "FAIL")
            with self.assertRaises(ValueError):  # one attempt per (arm, variant, subset), even after a failure
                pe.ledger_start(prereg, pe.ARM_SINGLE, pe.GATE_62_VARIANT, "again", ledger, "wrong")

            volume = root / "volume"
            proof = volume / "eval-data/proof"
            proof.mkdir(parents=True)
            subset = Path(shutil.copyfile(wrong, proof / "wrong.json"))
            pinned = Path(shutil.copyfile(prereg, proof / "prereg.json"))
            params = {"version": eval_stage.PARAMS_VERSION, "commit": "a" * 40, "repo_root": str(pe.ROOT),
                      "python": f"{volume}/venv-0123456789abcdef/bin/python", "volume_root": str(volume),
                      "artifact_root": str(proof / "artifacts"), "prereg": str(pinned),
                      "prereg_sha256": evaluation.file_hash(pinned), "ledger": str(proof / "ledger.jsonl"), "run_prefix": "single",
                      "subset": str(subset), "subset_sha256": evaluation.file_hash(subset), "subset_tag": "wrong",
                      "arm": pe.ARM_SINGLE, "seeds": [3407], "vllm_inprocess": True, "eval_slots": "auto",
                      "checkpoint": str(self.out / "checkpoint-20"), "checkpoint_metadata_sha256": "2" * 64,
                      "binding": str(self.plan_file), "binding_sha256": "3" * 64, "training_commit": "f" * 40,
                      "ft_export": str(self.exports[20]), "reuse_export": True, "ft_model_ref_sha256": "4" * 64}
            spec = eval_stage.plan("eval-ft", params)
            self.assertIs(chain.check_spec(spec), spec)
            names = [step["name"] for step in spec["steps"]]
            verify = names.index("verify-single-export")
            self.assertLess(verify, next(i for i, name in enumerate(names) if name.startswith("ledger-")))
            argv = spec["steps"][verify]["argv"]
            self.assertEqual(argv[argv.index("--binding") + 1], str(self.plan_file))
            ledgers = [step["argv"] for step in spec["steps"] if step["name"].startswith("ledger-")]
            self.assertEqual([a[a.index("--variant") + 1] for a in ledgers], [pe.GATE_62_VARIANT])  # v7: the 8k one
            with self.assertRaises(ValueError):
                eval_stage.plan("eval-ft", {k: v for k, v in params.items() if k not in ("reuse_export", "ft_model_ref_sha256")})

    def test_final_v8_evaluates_all_900_ids_in_one_run_with_the_same_gate(self) -> None:
        from fine_tuning.notebook_rl.pod import chain, eval_stage

        with tempfile.TemporaryDirectory(prefix="single-v8-") as tmp:
            root = Path(tmp)
            b2 = root / "b2.jsonl"
            b2.write_text("".join(json.dumps({"id": sid, "correct": i >= 342}) + "\n" for i, sid in enumerate(IDS)))
            wrong, rest = write_subset(root / "wrong.json", b2), write_subset(root / "rest.json", b2, kind="rest")
            everything = root / "all.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pe.main(["make-subset", "--kind", "all", "--output", str(everything),
                                          "--b2-predictions", str(b2)]), 0)
            ref, prereg, ledger = self.exports[20] / "model_ref.json", root / "prereg.json", root / "ledger.jsonl"
            with self.assertRaisesRegex(ValueError, "requires --all-subset"):
                pe.preregister(root / "x.json", wrong, rest, b2, "v8", ft_single_model_ref=ref)
            with self.assertRaisesRegex(ValueError, "v8-only"):
                pe.preregister(root / "y.json", wrong, rest, b2, "v7", ft_single_model_ref=ref, all_subset=everything)
            with self.assertRaises(ValueError):  # the 342 wrong IDs are not the one-run subset
                pe.preregister(root / "z.json", wrong, rest, b2, "v8", ft_single_model_ref=ref, all_subset=wrong)
            with contextlib.redirect_stdout(io.StringIO()):  # the CLI path passes --all-subset through
                self.assertEqual(pe.main(["preregister", "--output", str(prereg), "--subset", str(wrong), "--rest-subset",
                                          str(rest), "--b2-predictions", str(b2), "--format", "v8",
                                          "--ft-single-model-ref", str(ref), "--all-subset", str(everything)]), 0)
            loaded, _ = pe.load_preregistration(prereg)
            self.assertEqual(loaded["mmmu_val_budget"]["runs"], 2)
            self.assertEqual((loaded["format"], set(loaded["evaluation_subsets"])), (pe.FORMAT_V8, {"all"}))
            self.assertEqual([(run["arm"], run["subset"]) for run in loaded["runs"]], [("F0", "all"), (pe.ARM_SINGLE, "all")])
            self.assertEqual(loaded["evaluation_subsets"]["all"]["ids"], sorted(IDS))
            self.assertEqual(loaded["gate_62"]["minimum_correct"], 559)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pe.main(["verify-single-export", "--model-ref", str(ref), "--checkpoint",
                                          str(self.out / "checkpoint-20"), "--binding", str(self.plan_file),
                                          "--prereg", str(prereg)]), 0)
            self.assertEqual(set(loaded["protocol_variants"]), {pe.FINAL_VARIANT_V8})
            self.assertEqual({run["variant"] for run in loaded["runs"]}, {pe.FINAL_VARIANT_V8})
            self.assertEqual((loaded["gate_62"]["variant"], loaded["gate_67.4"]["variant"]), (pe.FINAL_VARIANT_V8,) * 2)
            for variant, tag in ((pe.FINAL_VARIANT_V8, "wrong"), (pe.GATE_62_VARIANT, "all")):
                with self.assertRaisesRegex(ValueError, "not a preregistered run"):
                    pe.ledger_start(prereg, pe.ARM_SINGLE, variant, "ft-wrong", ledger, tag)
            eight_k = root / "eight-k.json"  # a v8 preregistration on the 8k variant is refused
            eight_k.write_text(prereg.read_text().replace(pe.FINAL_VARIANT_V8, pe.GATE_62_VARIANT))
            Path(f"{eight_k}.sha256").write_text(evaluation.file_hash(eight_k) + "\n")
            with self.assertRaisesRegex(ValueError, "v7 requires one declared"):
                pe.load_preregistration(eight_k)
            cfg = variant_cfg(pe.FINAL_VARIANT_V8)
            with mock.patch.object(pe, "gpu_info", return_value=("NVIDIA A40", 46068)), mock.patch.dict(os.environ):
                os.environ.pop(pe.SLOTS_ENV, None)
                os.environ[pe.VLLM_INPROCESS_ENV] = "0"
                model_sha = loaded["arms"][pe.ARM_SINGLE]["model_sha256"]
                write_run(root, "ft-all", cfg, model_sha, lambda i: i < 559, ids=sorted(IDS))
            pe.ledger_start(prereg, pe.ARM_SINGLE, pe.FINAL_VARIANT_V8, "ft-all", ledger, "all")
            report = pe.analyze(prereg, [root / "ft-all"], {pe.ARM_SINGLE: ref}, ledger)
            self.assertEqual(report["full_900"][pe.ARM_SINGLE]["items"], 900)
            gate = pe.gate_62(report, 541)
            self.assertEqual((gate["status"], gate["ft_single"]["correct"], gate["items"], gate["variant"]),
                             ("PASS", 559, 900, pe.FINAL_VARIANT_V8))
            self.assertIn("B2 conditions", gate["note"])
            changed = copy.deepcopy(report)
            changed["full_900"][pe.ARM_SINGLE]["seeds"]["3407"]["correct"] = 558
            self.assertEqual(pe.gate_62(changed)["status"], "FAIL")
            with self.assertRaises(ValueError):  # one attempt per (arm, variant, subset)
                pe.ledger_start(prereg, pe.ARM_SINGLE, pe.FINAL_VARIANT_V8, "again", ledger, "all")

            volume = root / "volume"
            proof = volume / "eval-data/proof"
            proof.mkdir(parents=True)
            subset = Path(shutil.copyfile(everything, proof / "all.json"))
            pinned = Path(shutil.copyfile(prereg, proof / "prereg.json"))
            params = {"version": eval_stage.PARAMS_VERSION, "commit": "a" * 40, "repo_root": str(pe.ROOT),
                      "python": f"{volume}/venv-0123456789abcdef/bin/python", "volume_root": str(volume),
                      "artifact_root": str(proof / "artifacts"), "prereg": str(pinned),
                      "prereg_sha256": evaluation.file_hash(pinned), "ledger": str(proof / "ledger.jsonl"), "run_prefix": "single",
                      "subset": str(subset), "subset_sha256": evaluation.file_hash(subset), "subset_tag": "all",
                      "arm": pe.ARM_SINGLE, "seeds": [3407], "vllm_inprocess": True, "eval_slots": "auto",
                      "checkpoint": str(self.out / "checkpoint-20"), "checkpoint_metadata_sha256": "2" * 64,
                      "binding": str(self.plan_file), "binding_sha256": "3" * 64, "training_commit": "f" * 40,
                      "ft_export": str(self.exports[20]), "reuse_export": True, "ft_model_ref_sha256": "4" * 64}
            spec = eval_stage.plan("eval-ft", params)
            self.assertIs(chain.check_spec(spec), spec)
            runs = [step for step in spec["steps"] if step["name"].startswith("eval-")]
            self.assertEqual([step["name"] for step in runs], ["eval-single-ft-single-b2-s3407-all-evaluation"])
            argv = runs[0]["argv"]
            self.assertEqual(argv[argv.index("--protocol") + 1], f"{pe.ROOT}/fine_tuning/notebook_rl/eval_configs/mmmu_val_v8_b2_seed3407.yaml")
            self.assertEqual([argv[i + 1] for i, token in enumerate(argv) if token == "--sample-id"], sorted(IDS))
            ledgers = [step["argv"] for step in spec["steps"] if step["name"].startswith("ledger-")]
            self.assertEqual([a[a.index("--variant") + 1] for a in ledgers], [pe.FINAL_VARIANT_V8])


class SingleCampaignEvaluationV12(unittest.TestCase):
    """The v11 export / v7 preregistration / gate-62 wiring above, on a sealed v12 campaign of 21 slots: checkpoint 19 is
    the pilot, 20 a stopped main run (campaign-state.json names it the newest) and 21 the final. Authored fixtures only."""
    STAGES = {19: "pilot", 20: "stopped", 21: "final"}
    SINGLE_VERSION = VERSION_SINGLE_V12  # SingleCampaignEvaluationV13 reruns every check on a v13 plan

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        tmp = tempfile.TemporaryDirectory(prefix="single-eval-v12-")
        cls.addClassCleanup(tmp.cleanup)
        root = cls.root = Path(tmp.name)
        cls.base, cls.base_manifest, _, _ = fixture(root)
        plan = plan_for(332, "train")  # 300 pilot items in 19 blocks, then 32 main items in 2 blocks
        plan["protocol"] = plain_protocol_single("bf16", len(plan["blocks"]), version=cls.SINGLE_VERSION, token_scale=1024.,
                                                 contracts={key: "a" * 64 for key in ("source", "dataset", "order", "data_admission")})
        plan["sha256"] = audit.digest({k: v for k, v in plan.items() if k != "sha256"})
        out = cls.out = root / "campaign"
        out.mkdir()
        single._sealed(out / "campaign-plan.json", plan)
        cls.plan, cls.plan_file = plan, out / "campaign-plan.json"
        protocol, binding, blocks = plan["protocol"], campaign._binding(plan), plan["blocks"]
        torch.manual_seed(3407)
        bundle = fresh_tiny(cls.base, ARM, protocol)
        optimizer, scheduler = ckpt.make_optimizer(bundle, protocol)
        policy, done, parent = rl.policy_sha256(bundle), 0, None

        def save(slot: int) -> dict:
            nonlocal done, parent
            with warnings.catch_warnings():  # authored zero-signal history: scheduler steps without optimizer steps
                warnings.simplefilter("ignore")
                for _ in range(slot - done):
                    scheduler.step()
            done = slot
            cursor = {"slot": slot, "updates": slot, "consumed_problem_ids": [pid for block in blocks[:slot] for pid in block],
                      "policy_sha256": policy, "last_transaction_sha256": None, "parent_checkpoint": parent}
            path = out / f"checkpoint-{slot}"
            single.save_single_continuation(bundle, optimizer, scheduler, cursor=cursor, path=path, protocol=protocol,
                                           binding=binding, allow_save=True)
            parent = campaign._descriptor(path / "metadata.json")
            return cursor

        cursors = {slot: save(slot) for slot in (18, 19, 20, 21)}
        single._state(out / "campaign-state.json", {"version": campaign.VERSION, "plan_sha256": plan["sha256"],
                      "cursor": cursors[20], "checkpoint": campaign._descriptor(out / "checkpoint-20/metadata.json"),
                      "pending": None, "rng": None, "proofs": [], "status": "BOUNDED_STOP"})
        cls.exports = {slot: root / f"export-{slot}" for slot in cls.STAGES}
        for slot, target in cls.exports.items():
            pe.export_for_eval(out / f"checkpoint-{slot}", cls.plan_file, cls.base, target, "f" * 40,
                               _test_policy_factory=fresh_tiny, _test_base_manifest=cls.base_manifest)

    def test_v12_exports_bind_plan_slot_and_refuse_other_checkpoints(self) -> None:
        from fine_tuning.notebook_rl.export import validate_export_reference

        for slot, stage in self.STAGES.items():
            ref = json.loads((self.exports[slot] / "model_ref.json").read_text())
            campaign_ref = ref["export"]["campaign"]
            self.assertEqual((campaign_ref["stage"], campaign_ref["slot"], campaign_ref["plan_sha256"]),
                             (stage, slot, self.plan["sha256"]))
            self.assertEqual(ref["export"]["binding"], campaign._binding(self.plan))
            self.assertEqual(ref["export"]["training_protocol"], self.plan["protocol"])
            self.assertEqual(ref["export"]["training_protocol"]["version"], self.SINGLE_VERSION)
            self.assertEqual(ref["data_manifest_sha256"], self.plan["sha256"])
            validate_export_reference(self.exports[slot] / "model", ref)
        with self.assertRaisesRegex(ValueError, "pilot \\(slot 19, 300 items\\) or the final"):
            pe.export_for_eval(self.out / "checkpoint-18", self.plan_file, self.base, self.root / "x18", "f" * 40,
                               _test_policy_factory=fresh_tiny, _test_base_manifest=self.base_manifest)
        plain = write(self.root / "plain-binding.json", campaign._binding(self.plan))
        with self.assertRaisesRegex(ValueError, "campaign-plan.json"):
            pe.export_for_eval(self.out / "checkpoint-21", plain, self.base, self.root / "x21", "f" * 40,
                               _test_policy_factory=fresh_tiny, _test_base_manifest=self.base_manifest)
        # a stopped export must be the campaign's newest checkpoint: here the state names the pilot instead
        shutil.copytree(self.out / "checkpoint-20", self.root / "alt/checkpoint-20")
        single._state(self.root / "alt/campaign-state.json", single._read(self.out / "campaign-state.json")
                      | {"checkpoint": campaign._descriptor(self.out / "checkpoint-19/metadata.json")})
        with self.assertRaisesRegex(ValueError, "newest checkpoint"):
            pe.export_for_eval(self.root / "alt/checkpoint-20", self.plan_file, self.base, self.root / "x20", "f" * 40,
                               _test_policy_factory=fresh_tiny, _test_base_manifest=self.base_manifest)
        self.assertFalse(any((self.root / name).exists() for name in ("x18", "x20", "x21")))

    def test_v12_single_stage_follows_the_sealed_plan_length(self) -> None:
        last = len(self.plan["blocks"])
        self.assertEqual(last, 21)
        self.assertEqual([pe._single_stage(*case) for case in ((19, 300, last), (20, 316, last), (last, 332, last))],
                         ["pilot", "stopped", "final"])
        for case in ((18, 288, last), (19, 304, last), (0, 0, last), (last + 1, 336, last)):
            with self.assertRaisesRegex(ValueError, "or the final"):
                pe._single_stage(*case)
        self.assertEqual(pe._single_versions(), (VERSION_SINGLE, VERSION_SINGLE_V12, VERSION_SINGLE_V13))

    def test_v12_unknown_protocol_versions_are_still_refused(self) -> None:
        unknown = "error-note-single-response-v14"
        ref = json.loads((self.exports[21] / "model_ref.json").read_text())
        ref["export"]["training_protocol"]["version"] = unknown
        with self.assertRaisesRegex(ValueError, "requires an error-note-single-response export"):
            pe._single_model_ref(write(self.root / "unknown-ref/model_ref.json", ref))
        changed = copy.deepcopy(self.plan)
        changed["protocol"]["version"] = unknown
        changed["sha256"] = audit.digest({k: v for k, v in changed.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "invalid single-response protocol"):
            pe.export_for_eval(self.out / "checkpoint-21", write(self.root / "unknown-plan.json", changed), self.base,
                               self.root / "x-unknown", "f" * 40, _test_policy_factory=fresh_tiny,
                               _test_base_manifest=self.base_manifest)
        self.assertFalse((self.root / "x-unknown").exists())

    def test_v12_final_v7_declares_one_export_verifies_it_and_passes_above_558(self) -> None:
        self._v7_gate_62(21)

    def test_v12_stopped_main_checkpoint_is_accepted_like_a_final_one_by_v7(self) -> None:
        self._v7_gate_62(20)

    def _v7_gate_62(self, slot: int) -> None:
        from fine_tuning.notebook_rl.pod import chain, eval_stage

        with tempfile.TemporaryDirectory(prefix="single-v12-v7-") as tmp:
            root = Path(tmp)
            b2 = root / "b2.jsonl"
            b2.write_text("".join(json.dumps({"id": sid, "correct": i >= 342}) + "\n" for i, sid in enumerate(IDS)))
            wrong, rest = write_subset(root / "wrong.json", b2), write_subset(root / "rest.json", b2, kind="rest")
            ref, prereg, ledger = self.exports[slot] / "model_ref.json", root / "prereg.json", root / "ledger.jsonl"
            checkpoint = self.out / f"checkpoint-{slot}"
            with self.assertRaisesRegex(ValueError, "final BF16 single-campaign export"):
                pe.preregister(root / "pilot.json", wrong, rest, b2, "v7", ft_single_model_ref=self.exports[19] / "model_ref.json")
            pe.preregister(prereg, wrong, rest, b2, "v7", ft_single_model_ref=ref)
            loaded, _ = pe.load_preregistration(prereg)
            self.assertEqual((loaded["format"], set(loaded["arms"])), (pe.FORMAT_V7, {"F0", pe.ARM_SINGLE}))
            self.assertEqual(loaded["arms"][pe.ARM_SINGLE]["campaign_plan_sha256"], self.plan["sha256"])
            self.assertEqual(loaded["gate_62"]["minimum_correct"], 559)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pe.main(["verify-single-export", "--model-ref", str(ref), "--checkpoint", str(checkpoint),
                                          "--binding", str(self.plan_file), "--prereg", str(prereg)]), 0)
            with self.assertRaisesRegex(ValueError, "checkpoint metadata or binding differs"):
                pe.verify_v3_export(ref, self.out / "checkpoint-19", self.plan_file, prereg)
            cfg = variant_cfg(pe.GATE_62_VARIANT)
            with mock.patch.object(pe, "gpu_info", return_value=("NVIDIA A40", 46068)), mock.patch.dict(os.environ):
                os.environ.pop(pe.SLOTS_ENV, None)
                os.environ[pe.VLLM_INPROCESS_ENV] = "0"
                model_sha = loaded["arms"][pe.ARM_SINGLE]["model_sha256"]
                write_run(root, "ft-wrong", cfg, model_sha, lambda i: i < 1)
                write_run(root, "ft-rest", cfg, model_sha, lambda i: True, ids=REST)
            for name in ("ft-wrong", "ft-rest"):
                pe.ledger_start(prereg, pe.ARM_SINGLE, pe.GATE_62_VARIANT, name, ledger, name.split("-")[-1])
            report = pe.analyze(prereg, [root / "ft-wrong", root / "ft-rest"], {pe.ARM_SINGLE: ref}, ledger)
            gate = pe.gate_62(report, 541)
            self.assertEqual((gate["format"], gate["status"], gate["ft_single"]["correct"]),
                             ("error-note-final-gate-single-v1", "PASS", 559))
            self.assertEqual(gate["f0_reference"]["correct"], 541)
            changed = copy.deepcopy(report)
            changed["full_900"][pe.ARM_SINGLE]["seeds"]["3407"]["correct"] = 558
            self.assertEqual(pe.gate_62(changed)["status"], "FAIL")
            with self.assertRaises(ValueError):  # one attempt per (arm, variant, subset), even after a failure
                pe.ledger_start(prereg, pe.ARM_SINGLE, pe.GATE_62_VARIANT, "again", ledger, "wrong")

            volume = root / "volume"
            proof = volume / "eval-data/proof"
            proof.mkdir(parents=True)
            subset = Path(shutil.copyfile(wrong, proof / "wrong.json"))
            pinned = Path(shutil.copyfile(prereg, proof / "prereg.json"))
            params = {"version": eval_stage.PARAMS_VERSION, "commit": "a" * 40, "repo_root": str(pe.ROOT),
                      "python": f"{volume}/venv-0123456789abcdef/bin/python", "volume_root": str(volume),
                      "artifact_root": str(proof / "artifacts"), "prereg": str(pinned),
                      "prereg_sha256": evaluation.file_hash(pinned), "ledger": str(proof / "ledger.jsonl"), "run_prefix": "single",
                      "subset": str(subset), "subset_sha256": evaluation.file_hash(subset), "subset_tag": "wrong",
                      "arm": pe.ARM_SINGLE, "seeds": [3407], "vllm_inprocess": True, "eval_slots": "auto",
                      "checkpoint": str(checkpoint), "checkpoint_metadata_sha256": "2" * 64,
                      "binding": str(self.plan_file), "binding_sha256": "3" * 64, "training_commit": "f" * 40,
                      "ft_export": str(self.exports[slot]), "reuse_export": True, "ft_model_ref_sha256": "4" * 64}
            spec = eval_stage.plan("eval-ft", params)
            self.assertIs(chain.check_spec(spec), spec)
            names = [step["name"] for step in spec["steps"]]
            verify = names.index("verify-single-export")
            self.assertLess(verify, next(i for i, name in enumerate(names) if name.startswith("ledger-")))
            argv = spec["steps"][verify]["argv"]
            self.assertEqual(argv[argv.index("--binding") + 1], str(self.plan_file))
            ledgers = [step["argv"] for step in spec["steps"] if step["name"].startswith("ledger-")]
            self.assertEqual([a[a.index("--variant") + 1] for a in ledgers], [pe.GATE_62_VARIANT])
            with self.assertRaises(ValueError):
                eval_stage.plan("eval-ft", {k: v for k, v in params.items() if k not in ("reuse_export", "ft_model_ref_sha256")})


class SingleCampaignEvaluationV13(SingleCampaignEvaluationV12):
    """The same export / v7 preregistration / gate-62 checks on a sealed v13 campaign (gate v8, no H rows)."""
    SINGLE_VERSION = VERSION_SINGLE_V13

if __name__ == "__main__":
    unittest.main()
