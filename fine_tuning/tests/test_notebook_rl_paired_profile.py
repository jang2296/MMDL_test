"""Explicit paired-generation boundaries; fixtures do not certify model learning."""
import copy
import json
from pathlib import Path
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import local_profile as profile, local_run, local_train, paired_notebook
from fine_tuning.notebook_rl.checkpoint import digest


def paired_protocol():
    return profile.build_protocol(verified_note_synthesis_credit=True, paired_notebook=True, context_limit=24576)


class PairedProfileTests(unittest.TestCase):
    def test_existing_protocol_hashes_and_defaults_remain_exact(self):
        cases: list[tuple[dict[str, Any], str]] = [({}, "38e348b367a4f6c511c6f163a35240c5ece157ca50bb2f7d4f48f98e57077a71"),
            ({"verified_note_atoms": True}, "5b0f72d297f22a8ab42047315c5af48fd0806a27dc0eb88e88191bcec68ee56f"),
            ({"verified_note_components": True}, "85bb33be8eb08306f733888004092645f0919dd8f2e25c2ea7162575fdeb3cf7"),
            ({"verified_note_fact_credit": True}, "dda72bcd5312e7477a4e897930c72759fea0b2856c094e633b058920ab762ac5"),
            ({"verified_note_fact_credit": True, "context_limit": 24576},
             "99876e56fd2543302e3e1d74922270b3c3a45c480fd963f161fd124a7cdeeaf2"),
            ({"verified_note_synthesis": True}, "a1bdcaf25775bb43961f12d8d1afa9d46fff7af13d8fc3ef72b00fdaa9157ebb"),
            ({"verified_note_synthesis": True, "context_limit": 24576},
             "06f39a074c324aa62e6530cdf744e42eb936f685d78c0037a5d60c6fae0a49db"),
            ({"verified_note_synthesis_credit": True, "context_limit": 24576},
             "1b40a7e4169e0c94da8a88b78d98bcd67092fd1404b3004066382d779d233c91"),
            ({"verified_note_synthesis_credit": True, "context_limit": 24576, "partial_notebook": True},
             "bb5fe2247b6c59c539a2c266a1cba7f9f501b85d50163d2177ddeb1d377e4267"),
            ({"verified_note_synthesis_credit": True, "context_limit": 24576, "referenced_notebook": True},
             "40a5e7707ea8b2ff142bf58e418b508bfc93396f72e85a246b1ebd12a8247e47")]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(profile.build_protocol(**kwargs)["sha256"], expected)

    def test_paired_protocol_is_distinct_and_keeps_relationship_admission(self):
        current = paired_protocol()
        reference = profile.build_protocol(verified_note_synthesis_credit=True, referenced_notebook=True, context_limit=24576)
        self.assertEqual(current["protocol_id"], "local-nf4-paired-notebook-synthesis-credit-24k-v1")
        self.assertEqual(profile.restore_protocol(json.loads(json.dumps(current))), current)
        self.assertEqual(current["notebook_generation_policy"], paired_notebook.VERSION)
        self.assertEqual(current["notebook_prompt_sha256"], digest(paired_notebook.prompt_contract()))
        changed = {"protocol_id", "sha256", "notebook_generation_policy", "notebook_prompt_sha256"}
        self.assertEqual({k: v for k, v in current.items() if k not in changed},
                         {k: v for k, v in reference.items() if k not in changed})
        prompts = profile.local_prompts(current)
        self.assertEqual(prompts["rewarded_notebook"], paired_notebook.prompt_contract())
        self.assertEqual(prompts["notebook_generation_policy"], paired_notebook.VERSION)
        self.assertEqual((prompts["notebook_max_rounds"], current["rollout"]["diag"]["n"],
                          current["rollout"]["diag"]["max_new_tokens"]), (1, 4, 8192))
        self.assertIs(prompts["notebook_strict_feedback_repairs"], False)
        self.assertIsNone(local_run.minimum_execution_contract(None, current))
        self.assertEqual(local_train.execution_admission({}, current), local_train.STRICT_ADMISSION)
        self.assertEqual(local_train.execution_admission(
            {"admission_policy": local_train.SYNTHESIS_CREDIT_ADMISSION}, current), local_train.SYNTHESIS_CREDIT_ADMISSION)
        for check in (lambda: local_run.minimum_execution_contract(local_run.VERIFIED_UPDATE_EXPLORATION, current),
                      lambda: local_train.execution_admission({"admission_policy": local_run.VERIFIED_UPDATE_EXPLORATION}, current)):
            with self.assertRaises(ValueError):
                check()

    def test_mode_requires_exclusive_boolean_synthesis_credit_and_24k(self):
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576,
                                       paired_notebook=cast(Any, value))
        for kwargs in ({}, {"verified_note_synthesis": True}, {"verified_note_synthesis_credit": True},
                       {"verified_note_synthesis_credit": True, "context_limit": 24576, "partial_notebook": True},
                       {"verified_note_synthesis_credit": True, "context_limit": 24576, "referenced_notebook": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                profile.build_protocol(paired_notebook=True, **kwargs)

    def test_rehashed_mode_prompt_budget_and_truth_tampering_are_rejected(self):
        reference = profile.build_protocol(verified_note_synthesis_credit=True, referenced_notebook=True, context_limit=24576)
        replacements = {"notebook_generation_policy": reference["notebook_generation_policy"],
            "notebook_prompt_sha256": reference["notebook_prompt_sha256"], "notebook_max_rounds": True,
            "notebook_strict_feedback_repairs": 0, "notebook_reward": "paired-success", "notebook_objective": {}}
        for key, value in [*replacements.items(), ("notebook_generation_policy", None)]:
            changed = copy.deepcopy(paired_protocol())
            if value is None:
                del changed[key]
            else:
                changed[key] = value
            changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
            for check in (profile.restore_protocol, profile.local_prompts):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    check(changed)

    def test_cli_prepare_capture_and_cache_forward_paired_mode(self):
        import torch
        from fine_tuning.notebook_rl import quantized_cache

        flags = ["--verified-note-synthesis-credit", "--context-limit", "24576", "--paired-notebook"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as prepared, patch("builtins.print"):
            self.assertEqual(local_run.main(["prepare", "--previous-request", "/unused/request",
                "--selection", "/unused/selection", "--output", "/unused/data", *flags]), 0)
            self.assertEqual(prepared.call_args.args[2], paired_protocol())
        with patch.object(local_run, "capture") as capture:
            self.assertEqual(local_run.main(["minimal", "--preparation", "/unused/preparation",
                "--model-path", "/unused/base", "--output", "/unused/capture", *flags]), 0)
            self.assertIs(capture.call_args.kwargs["paired_notebook"], True)
            self.assertIs(capture.call_args.kwargs["referenced_notebook"], False)
            self.assertIs(capture.call_args.kwargs["partial_notebook"], False)
        with tempfile.TemporaryDirectory() as directory:
            for stage in ("save", "check"):
                with self.subTest(stage=stage), patch.object(quantized_cache, "_sources", return_value={}), \
                        patch.object(quantized_cache, "build_protocol", side_effect=ValueError("CPU boundary")) as build, \
                        patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution"), \
                        patch.multiple(torch.cuda, reset_peak_memory_stats=lambda: None,
                            max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0), \
                        patch.object(quantized_cache, "build_quantized_policy") as save, \
                        patch.object(quantized_cache, "load_quantized_policy") as load, \
                        patch("traceback.print_exc"), patch("builtins.print"):
                    self.assertEqual(quantized_cache.main([stage, "--model-path", "/unused/base", "--cache", "/unused/cache",
                        "--parent", "/unused/parent", "--output", str(Path(directory) / stage), *flags]), 1)
                    self.assertIs(build.call_args.kwargs["paired_notebook"], True)
                    save.assert_not_called()
                    load.assert_not_called()

    def test_invalid_or_conflicting_cli_modes_fail_before_io(self):
        from fine_tuning.notebook_rl import quantized_cache

        commands = [(local_run.main, ["prepare", "--previous-request", "/unused/request",
            "--selection", "/unused/selection", "--output", "/unused/data"]),
            (local_run.main, ["minimal", "--preparation", "/unused/preparation",
             "--model-path", "/unused/base", "--output", "/unused/capture"]),
            (quantized_cache.main, ["save", "--model-path", "/unused/base", "--cache", "/unused/cache",
             "--output", "/unused/output"])]
        for command, args in commands:
            for extra in ([], ["--verified-note-synthesis-credit"],
                          ["--verified-note-synthesis-credit", "--context-limit", "24576", "--partial-notebook"],
                          ["--verified-note-synthesis-credit", "--context-limit", "24576", "--referenced-notebook"]):
                with self.subTest(command=command, extra=extra), patch("sys.stderr"), \
                        patch.object(Path, "mkdir") as mkdir, self.assertRaises(SystemExit) as stopped:
                    command([*args, "--paired-notebook", *extra])
                self.assertEqual(stopped.exception.code, 2)
                mkdir.assert_not_called()

    def test_capture_rejects_weaker_policy_before_input_or_model_access(self):
        with patch.object(local_run, "read_preparation") as read, \
                patch("fine_tuning.notebook_rl.quantized_policy.build_quantized_policy") as loader:
            with self.assertRaisesRegex(ValueError, "explicit verified-update policy"):
                local_run.capture(Path("/unused/preparation"), Path("/unused/model"), Path("/unused/capture"),
                    standalone=False, verified_note_synthesis_credit=True, paired_notebook=True, context_limit=24576,
                    execution_policy=local_run.VERIFIED_UPDATE_EXPLORATION)
            read.assert_not_called()
            loader.assert_not_called()

    def test_data_preparation_is_bound_to_new_protocol_before_descriptor_access(self):
        current = paired_protocol()
        prepared = {"format": "notebook-rl-local-data-v1", "status": "LOCAL_DATA_REVALIDATED",
            "profile": profile.PROFILE, "local_protocol_sha256": current["sha256"], "cloud_allowed": False,
            "validated_data": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preparation.json"
            path.write_text(json.dumps(prepared))
            with patch.object(local_run, "read_descriptor", side_effect=ValueError("descriptor boundary")) as read:
                with self.assertRaisesRegex(ValueError, "descriptor boundary"):
                    local_run.read_preparation(path, current)
                read.assert_called_once()
            prepared["local_protocol_sha256"] = profile.build_protocol(
                verified_note_synthesis_credit=True, referenced_notebook=True, context_limit=24576)["sha256"]
            path.write_text(json.dumps(prepared))
            with patch.object(local_run, "read_descriptor") as read:
                with self.assertRaisesRegex(ValueError, "current explicitly authorized"):
                    local_run.read_preparation(path, current)
                read.assert_not_called()

    def test_reload_restores_paired_profile_before_loading_weights(self):
        protocol = paired_protocol()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = {"pid": 900001, "protocol": protocol, "standalone_base": False,
                "preparation": str(root / "preparation.json"), "binding": {"code_sha256": digest({})}}
            for name, value in (("launch.json", launch), ("reload_expected.json", {}),
                ("result.json", {"status": "UPDATED_SAVED_PENDING_FRESH_RELOAD", "optimizer_updates": 1})):
                (root / name).write_text(json.dumps(value))
            with patch("psutil.pid_exists", return_value=False), \
                    patch.object(local_run.approval, "code_hashes", return_value={}), \
                    patch.object(local_run, "read_preparation", side_effect=ValueError("CPU reload boundary")) as read, \
                    patch("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy") as loader:
                with self.assertRaisesRegex(ValueError, "CPU reload boundary"):
                    local_run.reload_capture(root, root / "restored")
                self.assertEqual(read.call_args.args[1], protocol)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
