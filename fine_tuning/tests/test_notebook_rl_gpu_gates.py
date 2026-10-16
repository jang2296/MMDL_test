"""Live tiny-policy inventory and report boundary checks, not 4B GPU approvals."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl import gpu_gates as gates
from fine_tuning.notebook_rl.policy import configure_policy
from fine_tuning.notebook_rl.schema import load_protocol
from fine_tuning.tests.test_training import tiny_model


def fixture(arm="MOD-RV"):
    protocol = load_protocol(gates.HERE / "protocol_v1.yaml")
    bundle = configure_policy(tiny_model().bfloat16(), None, arm, protocol, enforce_counts=False)
    protocol["expected_trainable"] = dict(bundle.inventory["trainable_counts"])
    return bundle, protocol


def inspect(bundle, protocol):
    return gates._inventory(bundle, protocol, expected_targets={"L": 21, "V": 6})


class InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_live_parameter_counts_and_all_four_masters_without_mutation(self):
        bundle, protocol = fixture()
        before = {n: tensor_sha256(p) for n, p in bundle.model.named_parameters()}
        result = inspect(bundle, protocol)
        self.assertEqual(result["trainable_counts"], protocol["expected_trainable"])
        self.assertEqual(result["mergers"], 4)
        self.assertEqual(sum(len(row["master"]) for row in result["merger_tensors"]), 24)
        self.assertEqual(len(result["installed_targets"]["L"]), 21)
        self.assertEqual(len(result["installed_targets"]["V"]), 6)
        self.assertFalse(result["pinned_architecture_counts"])
        self.assertEqual(before, {n: tensor_sha256(p) for n, p in bundle.model.named_parameters()})

    def test_closed_vision_route_keeps_installed_counts_but_reports_disabled_gradients(self):
        bundle, protocol = fixture()
        bundle.route.apply(False)
        result = inspect(bundle, protocol)
        self.assertGreater(result["trainable_counts"]["V"], 0)
        self.assertEqual(result["currently_enabled_counts"]["V"], 0)
        self.assertFalse(bundle.route.vision_on)
        self.assertTrue(all(not p.requires_grad for p in bundle.params["V"]))
        bundle, protocol = fixture("MOD-FML")
        result = inspect(bundle, protocol)
        self.assertEqual(result["installed_targets"]["V"], [])
        self.assertEqual(result["trainable_counts"]["V"], 0)
        self.assertEqual(len(result["targets"]["V"]), 6)

    def test_public_gate_rejects_tiny_model_even_with_forged_cached_counts(self):
        bundle, protocol = fixture()
        with self.assertRaisesRegex(ValueError, "pinned parameter"):
            gates.measure_inventory(bundle, protocol)
        protocol["expected_trainable"] = dict(gates._COUNTS)
        bundle.inventory["trainable_counts"] = dict(gates._COUNTS)
        with self.assertRaisesRegex(ValueError, "actual target count"):
            gates.measure_inventory(bundle, protocol)

    def test_optimizer_route_and_cached_metadata_tampering_rejected(self):
        for change in ("params", "route", "count", "target", "requires_grad"):
            with self.subTest(change=change):
                bundle, protocol = fixture()
                if change == "params":
                    bundle.params["M"].append(bundle.params["L"][0])
                elif change == "route":
                    bundle.route.params = []
                elif change == "count":
                    bundle.inventory["trainable_counts"]["L"] += 1
                elif change == "target":
                    bundle.inventory["targets"]["L"].pop()
                else:
                    bundle.params["M"][0].requires_grad_(False)
                with self.assertRaises(ValueError):
                    inspect(bundle, protocol)

    def test_wrong_master_dtype_alias_nonfinite_or_mutated_base_rejected(self):
        for change in ("dtype", "alias", "nonfinite", "base"):
            with self.subTest(change=change):
                bundle, protocol = fixture()
                base = bundle.model.get_base_model()
                wrapper = base.get_submodule(bundle.inventory["mergers"][0])
                if change == "dtype":
                    wrapper.modules_to_save["default"].bfloat16()
                elif change == "alias":
                    base.register_parameter("unexpected_alias", bundle.params["L"][0])
                elif change == "nonfinite":
                    with torch.no_grad():
                        bundle.params["M"][0].flatten()[0] = float("nan")
                else:
                    with torch.no_grad():
                        base.model.visual.patch_embed.proj.weight.add_(1)
                with self.assertRaises(ValueError):
                    inspect(bundle, protocol)

    def test_adapter_scaling_dropout_and_disabled_wrapper_rejected(self):
        for change in ("scaling", "dropout", "disabled"):
            with self.subTest(change=change):
                bundle, protocol = fixture()
                base = bundle.model.get_base_model()
                target = base.get_submodule(bundle.inventory["targets"]["L"][0])
                if change == "scaling":
                    target.scaling["default"] *= 2
                elif change == "dropout":
                    target.lora_dropout["default"] = torch.nn.Dropout(.1)
                else:
                    base.get_submodule(bundle.inventory["mergers"][0]).enable_adapters(False)
                with self.assertRaises(ValueError):
                    inspect(bundle, protocol)


class NumericalReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_actual_cpu_numerics_report_is_not_gpu_or_training_approval(self):
        output = self.root / "probe.json"
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(gates.main(["--numeric-check", "--device", "cpu", "--out", str(output)]), 0)
        report = json.loads(output.read_bytes())
        self.assertEqual(report["status"], "CHECK_COMPLETED_NOT_GPU_APPROVAL")
        self.assertEqual(report["format"], "notebook-rl-numerical-probe-v1")
        self.assertTrue(report["measurements"]["numeric_tests_match"])
        self.assertTrue(report["measurements"]["trl_formula_match"])
        self.assertFalse(report["gpu_gates_certified"])
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["model_loaded"])
        self.assertFalse(json.loads(out.getvalue())["training_allowed"])

    def test_failed_or_interrupted_numerics_preserve_incomplete_record(self):
        for error in (ValueError("fixture"), KeyboardInterrupt()):
            output = self.root / (type(error).__name__ + ".json")
            with patch("fine_tuning.notebook_rl.gate_numerics.measure_numerics", side_effect=error), \
                    self.assertRaises(type(error)):
                gates.run_numerical_probe("cpu", output)
            report = json.loads(output.read_bytes())
            self.assertEqual(report["status"], "INCOMPLETE")
            self.assertEqual(report["error_type"], type(error).__name__)
            self.assertNotIn("measurements", report)
            self.assertFalse(report["training_allowed"])

    def test_input_source_drift_fails_and_outputs_never_overwrite_or_enter_checkout(self):
        output = self.root / "changed.json"
        with patch.object(gates, "_sources", side_effect=[{"x": "before"}, {"x": "after"}]), \
                self.assertRaisesRegex(ValueError, "source changed"):
            gates.run_numerical_probe("cpu", output)
        report = json.loads(output.read_bytes())
        self.assertEqual(report["status"], "INCOMPLETE")
        before = output.read_bytes()
        with self.assertRaises(ValueError):
            gates.run_numerical_probe("cpu", output)
        self.assertEqual(output.read_bytes(), before)
        with self.assertRaises(ValueError):
            gates.run_numerical_probe("cpu", gates.ROOT / "MMDL" / "not-written.json")
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            gates.run_numerical_probe("cpu", alias / "not-written.json")


if __name__ == "__main__":
    unittest.main()
