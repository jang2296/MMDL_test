"""CPU-only v3 loss, shared admission and resume checks; fixtures are never training data."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fine_tuning.error_note import train as trainer
from fine_tuning.error_note import plan as planner
from fine_tuning.error_note.audit import FILE_VERSION, text_sha256
from fine_tuning.error_note.contract import (CONTEXT_LIMIT, KIND_NOTE, ROUTE_FULL, build_rows_v3,
                                             parse_note_v2, resolve_prompt)
from fine_tuning.error_note.protocol import _digest, plain_protocol, plain_protocol_v2, plain_protocol_v3
from fine_tuning.notebook_rl import checkpoint as ckpt
from fine_tuning.tests.test_error_note_train import COMMIT, RunFilesV2, encoded, tiny_factory
from fine_tuning.tests.test_error_note_v3_audit import audited, example, tokenize
from fine_tuning.tests.test_error_note_v3_plan import params as plan_params


def v3_encode(row):
    return encoded(row["row_id"], prompt=(7, 8) if row["kind"] == "A" else (7, 8, 8, 8, 11),
                   target=(9, 10) if row["kind"] == "A" else (12, 13, 14), route=row["route"], kind=row["kind"],
                   seed=int(row["problem_id"].split("-")[1]) % 5)


class V3Files(RunFilesV2):
    def __init__(self, root, groups=3):
        super().__init__(root, groups)
        self.audit, self.sidecar, self.admission = (self.root / name for name in ("audit.json", "sidecar.json", "admission.json"))
        ids = list(dict.fromkeys(row["problem_id"] for row in trainer._read_jsonl(self.rows)))
        audits, rows = {}, []
        for pid in ids:
            index = int(pid.split("-")[1])
            record = example(correct=index % 2 == 0, misread=index % 2 == 1)
            record["problem_id"] = pid
            if index % 2:
                lesson = parse_note_v2(record["note"]["text"]).lesson
                record["resolve"] = record["t1"] | {"status": "CORRECT", "prompt": resolve_prompt(record["p0_text"], lesson),
                                                       "lesson_sha256": text_sha256(lesson)}
            record = audited(record)
            accepted, drops = build_rows_v3(record, tokenize)
            if drops:
                raise AssertionError(drops)
            rows.extend(accepted)
            audits[pid] = record["audit"]
        self.rows.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.audit.write_text(json.dumps({"version": FILE_VERSION, "items": audits}) + "\n")
        self.sidecar.write_text('{"version":"source-rules-fixture"}\n')

    def admit(self, encode=v3_encode, *extra):
        args = trainer.build_parser().parse_args([
            "admit", "--rows", str(self.rows), "--build-root", str(self.root), "--order", str(self.order),
            "--limit", str(self.count), "--precision", "bf16", "--model-path", str(self.root),
            "--output", str(self.admission), "--audit", str(self.audit), "--sidecar", str(self.sidecar),
            "--code-commit", COMMIT, *extra])
        return trainer.run_admit(args, encode=encode)

    def args(self, output="run", arm="note", *extra):
        return super().args(output, arm, "--protocol-version", "v3", "--admission", str(self.admission),
                            "--audit", str(self.audit), "--sidecar", str(self.sidecar), *extra)

    def train(self, *args):
        return trainer.run_train(self.args(*args), policy_factory=tiny_factory, device="cpu",
                                  make_encoder=lambda bundle, root: v3_encode)


class V3TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def updates(self, name):
        return trainer._read_jsonl(self.root / name / "updates.jsonl")

    def test_legacy_hashes_and_v3_protocol_binding(self):
        self.assertEqual(plain_protocol("bf16")["sha256"], "8c4548feeff1ff3acff79c7319751dc99a7a34f25d09effb3311c626b1645d77")
        self.assertEqual(plain_protocol_v2("bf16", 214)["sha256"], "fce086a2268d1c61d40d86b9ec3c6ab34d3923e5ca0c12a02546e44fbe5a0de9")
        hashes = dict.fromkeys(("rows", "order", "audit", "sidecar", "admission"), "a" * 64)
        note = plain_protocol_v3("bf16", 3, arm="note", contracts=hashes)
        control = plain_protocol_v3("bf16", 3, arm="no-note", contracts=hashes)
        self.assertEqual(note["training_contract"]["loss"], {
            "row": "mean_target_and_eos_nll", "A": .9, "B": .1, "renormalize_missing_kind": False})
        self.assertEqual(note["optimizer"], control["optimizer"])
        self.assertEqual(note["max_new_tokens"], 8192)
        self.assertEqual(note["image"], plain_protocol("bf16")["image"])
        self.assertNotEqual(note["sha256"], control["sha256"])
        for key in hashes:
            changed = plain_protocol_v3("bf16", 3, arm="note", contracts=hashes | {key: "b" * 64})
            self.assertNotEqual(note["sha256"], changed["sha256"])
        with self.assertRaises(ValueError):
            plain_protocol_v3("bf16", 3, arm="solution", contracts=hashes)
        with self.assertRaises(ValueError):
            plain_protocol_v3("bf16", 3, arm="note", contracts={})

    def gradient(self, rows, weights, budget=0):
        bundle = tiny_factory()
        optimizer, scheduler = ckpt.make_optimizer(bundle, plain_protocol("bf16"))
        gradients = {}

        def capture(current):
            gradients.update({name: p.grad.detach().clone() for name, p in current.model.named_parameters() if p.grad is not None})
            return {}

        with patch.object(trainer, "_clip_groups", capture):
            stats = trainer.run_update(bundle, optimizer, scheduler, rows, budget, kind_weights=weights)
        return stats, gradients

    def test_weighted_row_means_and_preclip_gradient_match_independent_sum(self):
        rows = [encoded("a1", target=(9, 10), route=ROUTE_FULL),
                encoded("a2", target=(10,), seed=1),
                encoded("b", target=(11, 12, 13, 14, 15), kind=KIND_NOTE, seed=2)]
        expected = tiny_factory()
        expected_loss = 0
        for row, coefficient in zip(rows, (.45, .45, .1), strict=True):
            expected.route.apply(row["route"] == ROUTE_FULL)
            loss = -trainer.batch_logprobs(expected, [row])[0].mean()
            expected_loss += coefficient * float(loss.detach())
            (coefficient * loss).backward()
        expected_grads = {name: p.grad for name, p in expected.model.named_parameters() if p.grad is not None}
        for budget in (0, 1000):
            stats, gradients = self.gradient(rows, {"A": .9, "B": .1}, budget)
            self.assertAlmostEqual(stats["loss"], expected_loss, places=6)
            self.assertEqual(stats["denominators"], {"A": 2, "B": 1})
            self.assertEqual(stats["row_coefficients"], {"a1": .45, "a2": .45, "b": .1})
            self.assertEqual(gradients.keys(), expected_grads.keys())
            for name in gradients:
                torch.testing.assert_close(gradients[name], expected_grads[name], rtol=1e-4, atol=1e-6)

    def test_missing_B_keeps_point_nine_and_zero_B_matches_common_A(self):
        a = encoded("a", route=ROUTE_FULL)
        b = encoded("b", kind=KIND_NOTE, seed=2)
        stats, only_a = self.gradient([a], {"A": .9, "B": .1})
        zero_stats, zero_b = self.gradient([a, b], {"A": .9, "B": 0})
        self.assertEqual(stats["row_coefficients"], {"a": .9})
        self.assertEqual(stats["weighted_loss_by_kind"]["B"], 0)
        self.assertAlmostEqual(stats["loss"], zero_stats["loss"], places=7)
        for name in only_a:
            torch.testing.assert_close(zero_b[name], only_a[name], rtol=1e-5, atol=1e-7)
        with self.assertRaisesRegex(ValueError, "needs A"):
            self.gradient([b], {"A": .9, "B": .1})

    def test_common_admission_disables_note_only_groups_and_preserves_A_routes(self):
        files = V3Files(self.root)
        rows = trainer._read_jsonl(files.rows)
        rows = [row for row in rows if not (16 <= int(row["problem_id"].split("-")[1]) < 32 and row["kind"] == "A")]
        files.rows.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertEqual(files.admit(), 0)
        admission = self.read("admission.json")
        self.assertEqual(admission["total_steps"], 2)
        self.assertEqual(admission["groups"][1]["inactive_reason"], "NO_ENCODED_A")
        for arm in ("note", "no-note"):
            self.assertEqual(files.train(arm, arm), 0)
        note, control = self.updates("note"), self.updates("no-note")
        self.assertEqual([row["lr"] for row in note if not row["skipped"]], [row["lr"] for row in control if not row["skipped"]])
        self.assertEqual([row["skipped"] for row in note], [False, True, False])
        self.assertEqual([row["skipped"] for row in control], [False, True, False])
        self.assertEqual(control[0]["routes"][ROUTE_FULL], 1)
        self.assertEqual(note[0]["routes_by_kind"]["A"], control[0]["routes_by_kind"]["A"])
        for rid, coefficient in control[0]["row_coefficients"].items():
            self.assertEqual(note[0]["row_coefficients"][rid], coefficient)
        self.assertEqual(self.read("note/initial_state.json")["sha256"], self.read("no-note/initial_state.json")["sha256"])

    def test_preflight_records_drops_before_schedule_and_later_drop_or_mutation_stops(self):
        files = V3Files(self.root, groups=1)

        def too_long(row):
            value = v3_encode(row)
            if row["kind"] == "B":
                value["input_ids"] = torch.ones(CONTEXT_LIMIT + 1, dtype=torch.long)
            return value

        self.assertEqual(files.admit(too_long), 0)
        manifest = self.read("admission.json")
        self.assertEqual(manifest["total_steps"], 1)
        self.assertEqual(len(manifest["groups"][0]["drops"]), 2)
        self.assertTrue(all(drop["reason"] == "OVER_MAX_ROW_TOKENS" for drop in manifest["groups"][0]["drops"]))
        self.assertEqual(trainer.run_train(files.args(), policy_factory=tiny_factory, device="cpu",
                         make_encoder=lambda bundle, root: too_long), 0)
        self.assertEqual(self.read("run/result.json")["rows_by_kind"], {"A": 2})

        def changed(row):
            value = too_long(row)
            value["pixel_values"][0, 0] += 1
            return value

        self.assertEqual(trainer.run_train(files.args("changed"), policy_factory=tiny_factory, device="cpu",
                         make_encoder=lambda bundle, root: changed), 1)
        self.assertEqual(self.read("changed/result.json")["status"], "ADMISSION_CHANGED")
        self.assertFalse(self.updates("changed"))

        def later_drop(row):
            if row["kind"] == "A":
                raise trainer.DropRow("CONTROL_TOKEN")
            return too_long(row)

        self.assertEqual(trainer.run_train(files.args("drop"), policy_factory=tiny_factory, device="cpu",
                         make_encoder=lambda bundle, root: later_drop), 1)
        self.assertEqual(self.read("drop/result.json")["step"], 0)

    def test_changed_receipts_or_admission_fail_before_initialization(self):
        files = V3Files(self.root, groups=1)
        self.assertEqual(files.admit(), 0)
        original_audit = files.audit.read_bytes()
        files.audit.write_text('{"version":"changed-audit"}\n')
        with patch.object(trainer, "build_policy", side_effect=AssertionError("must not initialize")):
            with self.assertRaises(ValueError):
                trainer.run_train(files.args(), device="cpu")
        files.audit.write_bytes(original_audit)
        manifest = self.read("admission.json")
        manifest["total_steps"] += 1
        manifest["sha256"] = _digest({k: v for k, v in manifest.items() if k != "sha256"})
        trainer._write_json(files.admission, manifest)
        with self.assertRaisesRegex(ValueError, "total_steps"):
            files.train()

    def test_no_A_groups_rejected_and_BF16_limit_forbidden(self):
        files = V3Files(self.root, groups=1)
        rows = [row for row in trainer._read_jsonl(files.rows) if row["kind"] == "B"]
        files.rows.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertEqual(files.admit(), 1)
        with self.assertRaisesRegex(ValueError, "no A-present"):
            files.train()
        with self.assertRaisesRegex(ValueError, "NF4"):
            trainer.run_train(files.args("cap", "note", "--max-row-tokens", "2600"), device="cpu")

    def test_NF4_functional_cap_is_frozen_before_shared_admission(self):
        files = V3Files(self.root, groups=1)
        self.assertEqual(files.admit(v3_encode, "--precision", "nf4", "--max-row-tokens", "12"), 0)
        admission = self.read("admission.json")
        self.assertEqual(admission["inputs"]["encoding"]["max_row_tokens"], 12)
        self.assertEqual(admission["inputs"]["encoding"]["max_new_tokens"], 8192)
        self.assertEqual(admission["total_steps"], 1)
        self.assertEqual(len(admission["groups"][0]["drops"]), 2)
        groups = trainer._read_groups(files.rows, files.order, files.count, "note", 16, v3=True, audit_path=files.audit)
        args = files.args("local", "note", "--precision", "nf4", "--max-row-tokens", "12")
        self.assertEqual(trainer._read_admission(args, groups), admission)
        args.max_row_tokens = 13
        with self.assertRaisesRegex(ValueError, "identity differs"):
            trainer._read_admission(args, groups)

    def test_continuous_and_resume_match_and_fresh_reload_checks_v3_identity(self):
        files = V3Files(self.root)
        self.assertEqual(files.admit(), 0)
        self.assertEqual(files.train("whole", "note", "--save-every", "1"), 0)
        whole, cut = self.root / "whole", self.root / "cut"
        cut.mkdir()
        for name in ("binding.json", "initial_state.json", "admission.json", "updates.jsonl", "drops_local.jsonl"):
            shutil.copyfile(whole / name, cut / name)
        shutil.copytree(whole / "checkpoint-1", cut / "checkpoint-1")
        with (cut / "updates.jsonl").open("a") as stream:
            stream.write('{"group":')
        self.assertEqual(files.train("cut", "note", "--save-every", "1", "--resume"), 0)
        binding = self.read("whole/binding.json")
        original = ckpt.read_metadata(whole / "checkpoint-final", binding=binding)
        resumed = ckpt.read_metadata(cut / "checkpoint-final", binding=binding)
        for key in ("state_sha256", "step", "cursor", "protocol"):
            self.assertEqual(original[key], resumed[key])
        self.assertEqual([row["loss"] for row in self.updates("whole")], [row["loss"] for row in self.updates("cut")])
        args = trainer.build_parser().parse_args(["reload", "--run", str(cut), "--model-path", str(self.root),
                                                  "--precision", "bf16", "--protocol-version", "v3"])
        self.assertEqual(trainer.run_reload(args, policy_factory=tiny_factory, device="cpu"), 0)
        args.protocol_version = "v2"
        self.assertEqual(trainer.run_reload(args, policy_factory=tiny_factory, device="cpu"), 1)
        args.protocol_version = "v3"
        (cut / "admission.json").write_text("{}")
        self.assertEqual(trainer.run_reload(args, policy_factory=tiny_factory, device="cpu"), 1)

    def test_planner_resume_command_starts_fresh_and_copies_admission(self):
        files = V3Files(self.root, groups=1)
        self.assertEqual(files.admit(), 0)
        spec = planner.plan("note-v3-train", plan_params("note-v3-train"))
        command = next(step["argv"] for step in spec["steps"] if step["name"] == "train-note")
        args = trainer.build_parser().parse_args(command[3:])
        self.assertTrue(args.resume)
        fixture = files.args("planner-fresh")
        for key in ("rows", "build_root", "order", "limit", "model_path", "output", "audit", "sidecar", "admission"):
            setattr(args, key, getattr(fixture, key))
        self.assertFalse(args.output.exists())
        self.assertEqual(trainer.run_train(args, policy_factory=tiny_factory, device="cpu",
                         make_encoder=lambda bundle, root: v3_encode), 0)
        self.assertEqual(self.read("planner-fresh/result.json")["status"], "COMPLETE")
        self.assertEqual((args.output / "admission.json").read_bytes(), files.admission.read_bytes())

    def test_resume_of_bound_run_rejects_missing_saved_admission_before_initialization(self):
        files = V3Files(self.root, groups=1)
        self.assertEqual(files.admit(), 0)

        def dropped(row):
            raise trainer.DropRow("CONTROL_TOKEN")

        args = files.args("bound")
        self.assertEqual(trainer.run_train(args, policy_factory=tiny_factory, device="cpu",
                         make_encoder=lambda bundle, root: dropped), 1)
        self.assertTrue((args.output / "binding.json").exists())
        (args.output / "admission.json").unlink()
        args.resume = True
        with patch.object(trainer, "build_policy", side_effect=AssertionError("must not initialize")):
            with self.assertRaisesRegex(ValueError, "saved admission is missing"):
                trainer.run_train(args, device="cpu")


if __name__ == "__main__":
    unittest.main()
