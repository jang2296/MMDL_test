"""Explicit BF16 rehearsal profiles cannot become production learning or scores."""
import unittest
from copy import deepcopy

from fine_tuning.error_note import campaign_v16b as campaign
from fine_tuning.error_note.audit import digest
from fine_tuning.error_note.protocol import (
    V16B_CONTRACT_HASHES, plain_protocol_v18b, validate_protocol_v18b,
)
from fine_tuning.error_note.v18b_rehearsal import profile_settings, parser, arguments, policy_factory
from fine_tuning.error_note.v18b_rehearsal import tiny_rehearsal_transport


class RehearsalProtocolTests(unittest.TestCase):
    def protocol(self, profile="tiny-bf16"):
        return plain_protocol_v18b("bf16", rl_blocks=profile_settings(profile)["blocks"], prep_updates=1,
            contracts={key: "a" * 64 for key in V16B_CONTRACT_HASHES}, rehearsal_profile=profile)

    def test_rehearsal_is_exact_bf16_vllm_schedule_and_nonreportable(self):
        for profile, parents, slots in (("tiny-bf16", 8, 12), ("a100-bf16", 4, 6)):
            with self.subTest(profile=profile):
                p = self.protocol(profile)
                validate_protocol_v18b(p)
                self.assertFalse(p["reportable"])
                self.assertFalse(p["training_contract"]["cloud_allowed"])
                self.assertEqual(p["training_contract"]["rl_items"], parents)
                self.assertEqual(p["training_contract"]["total_slots"], slots)
                self.assertEqual(p["optimizer"]["total_steps"], slots)
        for field, value in (("reportable", True), ("rl_items", 3000), ("finish_updates", 0)):
            p = self.protocol()
            p["training_contract"][field] = value
            p["sha256"] = digest({key: val for key, val in p.items() if key != "sha256"})
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_protocol_v18b(p)

    def test_production_acceptance_cannot_be_relaxed_by_resealing_profile_shape(self):
        p = self.protocol()
        p["training_contract"].pop("rehearsal_profile")
        p["sha256"] = digest({key: val for key, val in p.items() if key != "sha256"})
        with self.assertRaises(ValueError):
            validate_protocol_v18b(p)
        for profile in ("wrong", ""):
            with self.assertRaises(ValueError):
                self.protocol(profile)

    def test_formal_export_rejects_even_completed_rehearsal(self):
        p = self.protocol()
        from unittest.mock import patch
        with patch.object(campaign, "_plan_check"), self.assertRaisesRegex(ValueError, "non-reportable"):
            campaign.export_metadata({}, {"protocol": p}, None)

    def test_deadlines_are_positive_and_bounded_per_device(self):
        for profile, limit in (("tiny-bf16", 1800), ("a100-bf16", 1500)):
            argv = ["prepare", "--profile", profile, "--data-root", "/data", "--model-path", "/model", "--output", "/out"]
            self.assertEqual(arguments(parser().parse_args(argv + ["--deadline-seconds", str(limit)])).deadline_seconds, limit)
            for bad in (0, -1, limit + 1):
                with self.subTest(profile=profile, seconds=bad), self.assertRaises(ValueError):
                    arguments(parser().parse_args(argv + ["--deadline-seconds", str(bad)]))

    def test_real_model_cannot_use_the_tiny_numeric_exemption(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=36, hidden_size=2560),
                                 vision_config=SimpleNamespace(depth=27, hidden_size=1152))
        with patch("transformers.AutoConfig.from_pretrained", return_value=config), patch(
                "transformers.Qwen3VLForConditionalGeneration.from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "fixed two-layer"):
                policy_factory(None, "MOD-RV", self.protocol())
            load.assert_not_called()

    def test_printable_builder_copies_both_pinned_terminal_ids_and_exact_generation_bytes(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from transformers import GenerationConfig
        from fine_tuning.tests.test_error_note_single_vllm_lora_gpu import build_tiny, SNAPSHOT, EOS

        with TemporaryDirectory() as directory:
            path = build_tiny(Path(directory) / "model", printable_output=True)
            expected = (SNAPSHOT / "generation_config.json").read_bytes()
            self.assertEqual((path / "generation_config.json").read_bytes(), expected)
            config = GenerationConfig.from_pretrained(str(path), local_files_only=True)
            self.assertIsInstance(config.eos_token_id, list)
            self.assertEqual(set(config.eos_token_id), set(EOS))
            self.assertEqual(config.pad_token_id, 151643)

    def test_fixture_keeps_numeric_failure_and_rejects_structural_or_nonfinite_results(self):
        numeric = {"pass": False, "power": True, "cap": .04, "mean_abs_on": .1,
            "mean_abs_off_vs_on": .05, "amplified_mean_abs_on": .2,
            "real": {"total": {"rms_effect": 1.}}, "amplified": {},
            "stage_contract": {"require_power": True}}
        raw = {name: {"pass": True} for name in ("pf1", "pf2", "pf3", "pf4", "pf2_restored")}
        raw["pf5"] = numeric
        result = deepcopy(raw)
        tiny_rehearsal_transport(result)
        self.assertTrue(result["pass"])
        self.assertIs(result["pf5"]["pass"], False)
        self.assertFalse(result["production_admitted"])
        self.assertEqual(result["numerical_status"], "UNCONFIRMED")
        self.assertEqual(result["failed_numeric_checks"][0]["actual"], .1)
        for name in ("pf1", "pf2", "pf3", "pf4", "pf2_restored"):
            failed = deepcopy(raw)
            failed[name]["pass"] = False
            tiny_rehearsal_transport(failed)
            self.assertFalse(failed["pass"])
        failed = deepcopy(raw)
        failed["pf5"]["mean_abs_on"] = float("nan")
        tiny_rehearsal_transport(failed)
        self.assertFalse(failed["pass"])


if __name__ == "__main__":
    unittest.main()
