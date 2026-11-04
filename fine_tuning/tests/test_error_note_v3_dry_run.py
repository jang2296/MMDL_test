"""Tiny CPU forward/backward diagnostics; no pretrained weights, optimizer or checkpoint writes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fine_tuning.error_note import train as trainer
from fine_tuning.error_note.contract import ROUTE_LM
from fine_tuning.tests.test_error_note_train import tiny_factory
from fine_tuning.tests.test_error_note_v3_train import V3Files, v3_encode


class DryRunTests(unittest.TestCase):
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
        self.files = V3Files(self.root, groups=1)

    def args(self, output="dry"):
        args = trainer.build_parser().parse_args([
            "dry-run", "--rows", str(self.files.rows), "--build-root", str(self.root), "--order", str(self.files.order),
            "--limit", "1", "--model-path", str(self.root), "--output", str(self.root / output), "--precision", "nf4",
            "--code-commit", "a" * 40, "--audit", str(self.files.audit), "--sidecar", str(self.files.sidecar),
            "--admission", str(self.files.admission)])
        args.precision = "bf16"  # Injected tiny CPU policy only; the public CLI remains NF4.
        return args

    def run_diagnostic(self, args, bundle=None, *, policy_factory=None, encode=v3_encode):
        bundle = bundle or tiny_factory()
        with patch.object(trainer.ckpt, "make_optimizer", side_effect=AssertionError("no optimizer")), \
             patch.object(torch.optim.AdamW, "__init__", side_effect=AssertionError("no optimizer constructor")), \
             patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("no step")), \
             patch.object(torch.optim.lr_scheduler.LRScheduler, "step", side_effect=AssertionError("no scheduler")), \
             patch.object(torch, "save", side_effect=AssertionError("no tensor save")), \
             patch.object(trainer.ckpt, "save", side_effect=AssertionError("no checkpoint save")), \
             patch.object(trainer, "_save", side_effect=AssertionError("no checkpoint")), \
             patch.object(trainer, "save_file", side_effect=AssertionError("no tensor save")), \
             patch.object(trainer, "_write_probe", side_effect=AssertionError("no saved probe")):
            code = trainer.run_dry_run(args, policy_factory=policy_factory or (lambda *a: bundle), device="cpu",
                                       make_encoder=lambda *a: encode)
        return code, json.loads((args.output / "dry_run.json").read_text()), bundle

    def test_actual_A_and_B_backward_without_optimizer_steps_or_parameter_changes(self):
        self.assertEqual(self.files.admit(v3_encode, "--limit", "1"), 0)
        bundle = tiny_factory()
        before = trainer._all_parameter_hashes(bundle)
        code, report, bundle = self.run_diagnostic(self.args(), bundle)
        self.assertEqual((code, report["status"]), (0, "PASS"))
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["scheduler_steps"], 0)
        self.assertFalse(report["weights_saved"])
        self.assertTrue(report["weights_unchanged"])
        self.assertTrue(report["frozen_base_verified"])
        self.assertTrue(report["gradients_cleared"])
        self.assertEqual(report["parameter_hashes_before"], before)
        self.assertEqual(report["parameter_hashes_after"], trainer._all_parameter_hashes(bundle))
        self.assertEqual(report["encoded_rows_by_kind"], {"A": 1, "B": 1})
        self.assertEqual(set(self.root.joinpath("dry").iterdir()), {self.root / "dry/dry_run.json"})
        for kind, coefficient in (("A", .9), ("B", .1)):
            component = report["components"][kind]
            self.assertTrue(component["nonzero_gradient"])
            self.assertEqual(component["coefficient"], coefficient)
            self.assertEqual(component["routes"], {ROUTE_LM: 1})
            self.assertIsNone(component["grad_norms"]["V"])
            self.assertGreater(component["grad_norms"]["L"], 0)
            self.assertGreater(component["grad_norms"]["M"], 0)
            self.assertEqual(component["expected_gradient_groups"], {
                "L": "finite_positive", "M": "finite_positive", "V": "none"})
            self.assertTrue(component["route_check_passed"])
            self.assertIsNone(component["route_check_reason"])
            self.assertAlmostEqual(component["weighted_loss"], coefficient * component["row_nll_mean"])
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))

    def test_B_gradient_is_independent_of_A_and_mask_uses_target_plus_EOS_only(self):
        self.files.admit(v3_encode, "--limit", "1")
        _, report, bundle = self.run_diagnostic(self.args())
        row = v3_encode(next(row for row in trainer._read_jsonl(self.files.rows) if row["problem_id"] == "item-0" and row["kind"] == "B"))
        direct = tiny_factory()
        direct.route.apply(False)
        values = trainer.batch_logprobs(direct, [row])[0]
        (.1 * -values.mean()).backward()
        expected = trainer._gradient_norms(direct)
        for group in ("L", "M"):
            self.assertAlmostEqual(report["components"]["B"]["grad_norms"][group], expected[group], places=6)
        mask = report["loss_mask"]["item-0:B"]
        self.assertEqual(mask["target_and_eos_tokens"], 4)
        self.assertEqual(mask["prompt_tokens_excluded"], row["input_ids"].numel() - 4)
        base = bundle.model.get_base_model()
        ids = row["input_ids"].unsqueeze(0)
        labels = ids.clone()
        labels[:, :-row["target_len"]] = -100
        with torch.no_grad():
            expected_loss = base(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                pixel_values=row["pixel_values"], image_grid_thw=row["image_grid_thw"], use_cache=False).loss
        self.assertAlmostEqual(report["components"]["B"]["row_nll_mean"], float(expected_loss), places=5)

    def test_nonfinite_failure_clears_gradients_and_never_updates_weights(self):
        self.files.admit(v3_encode, "--limit", "1")
        bundle = tiny_factory()
        original = trainer.batch_logprobs

        def nonfinite(policy, rows):
            values = original(policy, rows)
            if rows[0]["kind"] == "B":
                return [value * float("nan") for value in values]
            return values

        with patch.object(trainer, "batch_logprobs", side_effect=nonfinite):
            code, report, bundle = self.run_diagnostic(self.args(), bundle)
        self.assertEqual((code, report["status"]), (1, "FAIL"))
        self.assertIn("nonfinite", report["error"])
        self.assertTrue(report["weights_unchanged"])
        self.assertTrue(report["gradients_cleared"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))

    def test_small_fixed_prefix_preserves_FULL_route_gradient_without_updates(self):
        self.assertEqual(self.files.admit(v3_encode, "--limit", "10"), 0)
        args = self.args()
        args.limit = 10
        code, report, _ = self.run_diagnostic(args)
        self.assertEqual((code, report["status"]), (0, "PASS"))
        for component in report["components"].values():
            self.assertEqual(component["routes"], {"FULL": 1, "LM": 1})
            self.assertGreater(component["grad_norms"]["V"], 0)
            self.assertEqual(component["expected_gradient_groups"]["V"], "finite_positive")
            self.assertTrue(component["route_check_passed"])
        self.assertTrue(report["weights_unchanged"])

    def test_missing_required_or_unexpected_vision_gradient_cannot_pass(self):
        original = trainer._backward_rows
        for group, unexpected, limit in (("L", False, 1), ("M", False, 1),
                                         ("V", False, 10), ("V", True, 1)):
            with self.subTest(group=group, unexpected=unexpected):
                self.files.admission.unlink(missing_ok=True)
                self.assertEqual(self.files.admit(v3_encode, "--limit", str(limit)), 0)

                def broken_backward(bundle, rows, plan, coefficients):
                    losses = original(bundle, rows, plan, coefficients)
                    if rows[0]["kind"] == "B":
                        for parameter in bundle.params[group]:
                            parameter.grad = torch.ones_like(parameter) if unexpected else None
                    return losses

                args = self.args(f"bad-{group}-{unexpected}")
                args.limit = limit
                with patch.object(trainer, "_backward_rows", side_effect=broken_backward):
                    code, report, _ = self.run_diagnostic(args)
                self.assertEqual((code, report["status"]), (1, "GRADIENT_ROUTE_MISMATCH"))
                self.assertTrue(report["components"]["A"]["route_check_passed"])
                self.assertFalse(report["components"]["B"]["route_check_passed"])
                reason = "UNEXPECTED_GRADIENT" if unexpected else "MISSING_OR_NONPOSITIVE_GRADIENT"
                self.assertEqual(report["components"]["B"]["route_check_reason"], f"{reason}:{group}")
                self.assertTrue(report["weights_unchanged"])
                self.assertTrue(report["gradients_cleared"])

    def test_initialization_failure_writes_unverified_fail_receipt(self):
        self.files.admit(v3_encode, "--limit", "1")

        def failed_factory(*args):
            raise RuntimeError("NF4 initialization failed")

        code, report, _ = self.run_diagnostic(self.args(), policy_factory=failed_factory)
        self.assertEqual((code, report["status"]), (1, "FAIL"))
        self.assertEqual(report["error"], "RuntimeError: NF4 initialization failed")
        self.assertEqual(report["errors"][0]["phase"], "policy_initialization")
        self.assertEqual(report["components"], {})
        for key in ("parameter_hashes_before", "parameter_hashes_after", "weights_unchanged",
                    "frozen_base_verified", "gradients_cleared"):
            self.assertIsNone(report[key])
        self.assertEqual(set(self.root.joinpath("dry").iterdir()), {self.root / "dry/dry_run.json"})

    def test_initial_and_final_hash_failure_do_not_claim_unchanged_weights(self):
        self.files.admit(v3_encode, "--limit", "1")
        original = trainer._all_parameter_hashes
        for failure_at, phase in ((1, "initial_parameter_hashes"), (2, "final_parameter_hashes")):
            with self.subTest(phase=phase):
                calls = 0

                def failed_hash(bundle):
                    nonlocal calls
                    calls += 1
                    if calls == failure_at:
                        raise MemoryError(f"{phase} unavailable")
                    return original(bundle)

                with patch.object(trainer, "_all_parameter_hashes", side_effect=failed_hash):
                    code, report, _ = self.run_diagnostic(self.args(phase))
                self.assertEqual((code, report["status"]), (1, "FAIL"))
                self.assertEqual(report["errors"][0]["phase"], phase)
                self.assertIsNone(report["weights_unchanged"])
                self.assertIsNone(report["parameter_hashes_before" if failure_at == 1 else "parameter_hashes_after"])
                self.assertTrue(report["gradients_cleared"])
                if failure_at == 1:
                    self.assertEqual(report["components"], {})
                else:
                    self.assertIsNone(report["frozen_base_verified"])

    def test_final_hash_failure_preserves_original_backward_error(self):
        self.files.admit(v3_encode, "--limit", "1")
        original = trainer._all_parameter_hashes
        calls = 0

        def failed_final_hash(bundle):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("final hash unavailable")
            return original(bundle)

        with patch.object(trainer, "_all_parameter_hashes", side_effect=failed_final_hash), \
             patch.object(trainer, "_backward_rows", side_effect=RuntimeError("original backward failure")):
            code, report, _ = self.run_diagnostic(self.args())
        self.assertEqual((code, report["status"]), (1, "FAIL"))
        self.assertEqual(report["error"], "RuntimeError: original backward failure")
        self.assertEqual([error["phase"] for error in report["errors"]], ["backward_A", "final_parameter_hashes"])
        self.assertTrue(report["gradients_cleared"])
        self.assertIsNone(report["weights_unchanged"])

    def test_resolve_note_pairs_count_only_actually_encoded_admitted_rows(self):
        def drop_note(row):
            if row["row_id"] == "item-9:B":
                raise trainer.DropRow("CONTROL_TOKEN")
            return v3_encode(row)

        for name, limit, encode, sources, paired in (
            ("t1-only", 1, v3_encode, {"T1": 1, "NOTE": 1}, []),
            ("resolved", 10, v3_encode, {"T1": 1, "RESOLVE": 1, "NOTE": 2}, ["item-9"]),
            ("dropped-note", 10, drop_note, {"T1": 1, "RESOLVE": 1, "NOTE": 1}, [])):
            with self.subTest(name=name):
                self.files.admission.unlink(missing_ok=True)
                self.assertEqual(self.files.admit(encode, "--limit", str(limit)), 0)
                args = self.args(name)
                args.limit = limit
                code, report, _ = self.run_diagnostic(args, encode=encode)
                self.assertEqual((code, report["status"]), (0, "PASS"))
                self.assertEqual(report["rows_by_target_source"], sources)
                self.assertEqual(report["paired_resolve_note_problem_ids"], paired)

    def test_provenance_summarizes_existing_inventory_without_tensor_payloads(self):
        self.files.admit(v3_encode, "--limit", "1")
        bundle = tiny_factory()
        quantization = {"contract": {"method": "fixture"}, "versions": {"torch": "fixture"},
                        "reference": "fixture", "modules": ["fixture.layer"],
                        "state_sha256": {"fixture.layer": {"scale": {"sha256": "a" * 64}}}}
        bundle.inventory["quantization"] = quantization
        code, report, _ = self.run_diagnostic(self.args(), bundle)
        self.assertEqual((code, report["status"]), (0, "PASS"))
        provenance = report["model_provenance"]
        self.assertEqual(provenance["base_revision"], bundle.inventory["base_revision"])
        self.assertEqual(provenance["trainable_counts"], bundle.inventory["trainable_counts"])
        self.assertEqual(provenance["quantization"], {
            "contract": quantization["contract"], "versions": quantization["versions"],
            "reference": "fixture", "module_count": 1,
            "modules_sha256": trainer._digest(quantization["modules"]),
            "state_sha256": trainer._digest(quantization["state_sha256"])})

    def test_missing_A_or_reviewed_B_is_nonpass_without_model_initialization(self):
        rows = trainer._read_jsonl(self.files.rows)
        for missing, reason in (("A", "NO_A_ROWS"), ("B", "NO_ADMITTED_B")):
            self.files.rows.write_text("".join(json.dumps(row) + "\n" for row in rows if row["kind"] != missing))
            self.files.admission.unlink(missing_ok=True)
            self.files.admit(v3_encode, "--limit", "1")
            with patch.object(trainer, "build_policy", side_effect=AssertionError("must not load model")):
                code = trainer.run_dry_run(self.args("missing-" + missing), device="cpu")
            report = json.loads((self.root / ("missing-" + missing) / "dry_run.json").read_text())
            self.assertEqual((code, report["status"], report["reason"]), (1, "NO_USABLE_SUPERVISION", reason))
            self.assertIsNone(report["weights_unchanged"])
            self.assertEqual(report["optimizer_steps"], 0)


if __name__ == "__main__":
    unittest.main()
