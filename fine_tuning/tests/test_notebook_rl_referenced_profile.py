"""Explicit reference generation profile boundaries; no real model execution."""
import copy
import json
from pathlib import Path
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import local_profile as profile, local_run, local_train, referenced_notebook
from fine_tuning.notebook_rl.checkpoint import digest


def referenced_protocol():
    return profile.build_protocol(verified_note_synthesis_credit=True, referenced_notebook=True, context_limit=24576)


class ReferencedProfileTests(unittest.TestCase):
    def test_exact_reference_contract_preserves_partial_and_strict_protocols(self):
        partial = profile.build_protocol(verified_note_synthesis_credit=True, partial_notebook=True, context_limit=24576)
        self.assertEqual(partial["sha256"], "bb5fe2247b6c59c539a2c266a1cba7f9f501b85d50163d2177ddeb1d377e4267")
        current = referenced_protocol()
        self.assertEqual(current["protocol_id"], profile.REFERENCED_NOTEBOOK_PROTOCOL_ID)
        self.assertEqual(profile.restore_protocol(json.loads(json.dumps(current))), current)
        self.assertEqual(current["notebook_generation_policy"], referenced_notebook.VERSION)
        self.assertEqual(current["notebook_prompt_sha256"], digest(referenced_notebook.prompt_contract()))
        changed = {"protocol_id", "sha256", "notebook_generation_policy", "notebook_prompt_sha256"}
        self.assertEqual({k: v for k, v in current.items() if k not in changed},
                         {k: v for k, v in partial.items() if k not in changed})
        prompts = profile.local_prompts(current)
        self.assertEqual(prompts["rewarded_notebook"], referenced_notebook.prompt_contract())
        self.assertEqual(prompts["notebook_generation_policy"], referenced_notebook.VERSION)
        self.assertEqual(prompts["notebook_max_rounds"], 1)
        self.assertIs(prompts["notebook_strict_feedback_repairs"], False)
        for admission in (local_run.VERIFIED_UPDATE_EXPLORATION, local_train.SYNTHESIS_CREDIT_ADMISSION):
            self.assertEqual(local_train.execution_admission({"admission_policy": admission}, current), admission)
        self.assertEqual(local_run.minimum_execution_contract(local_run.VERIFIED_UPDATE_EXPLORATION, current),
                         local_run.minimum_execution_contract(local_run.VERIFIED_UPDATE_EXPLORATION, partial))

    def test_reference_mode_requires_exclusive_boolean_credit_and_24k(self):
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576,
                                       referenced_notebook=cast(Any, value))
        for kwargs in ({}, {"verified_note_synthesis": True}, {"verified_note_synthesis_credit": True},
                       {"verified_note_synthesis_credit": True, "context_limit": 24576, "partial_notebook": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                profile.build_protocol(referenced_notebook=True, **kwargs)

    def test_rehashed_mode_prompt_budget_and_truth_tampering_are_rejected(self):
        partial = profile.build_protocol(verified_note_synthesis_credit=True, partial_notebook=True, context_limit=24576)
        replacements = {"notebook_generation_policy": partial["notebook_generation_policy"],
            "notebook_prompt_sha256": partial["notebook_prompt_sha256"], "notebook_max_rounds": True,
            "notebook_strict_feedback_repairs": 0, "notebook_reward": "reference-success", "notebook_objective": {}}
        for key, value in [*replacements.items(), ("notebook_generation_policy", None)]:
            changed = copy.deepcopy(referenced_protocol())
            if value is None:
                del changed[key]
            else:
                changed[key] = value
            changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
            for check in (profile.restore_protocol, profile.local_prompts):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    check(changed)
        with self.assertRaises(ValueError):
            profile.local_prompts(profile.build_protocol() | {"notebook_generation_policy": referenced_notebook.VERSION})

    def test_cli_prepare_capture_and_cache_forward_reference_mode(self):
        import torch
        from fine_tuning.notebook_rl import quantized_cache

        flags = ["--verified-note-synthesis-credit", "--context-limit", "24576", "--referenced-notebook"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as prepared, patch("builtins.print"):
            self.assertEqual(local_run.main(["prepare", "--previous-request", "/unused/request",
                "--selection", "/unused/selection", "--output", "/unused/data", *flags]), 0)
            self.assertEqual(prepared.call_args.args[2], referenced_protocol())
        with patch.object(local_run, "capture") as capture:
            self.assertEqual(local_run.main(["minimal", "--preparation", "/unused/preparation",
                "--model-path", "/unused/base", "--output", "/unused/capture", *flags]), 0)
            self.assertIs(capture.call_args.kwargs["referenced_notebook"], True)
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
                    self.assertIs(build.call_args.kwargs["referenced_notebook"], True)
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
                          ["--verified-note-synthesis-credit", "--context-limit", "24576", "--partial-notebook"]):
                with self.subTest(command=command, extra=extra), patch("sys.stderr"), \
                        patch.object(Path, "mkdir") as mkdir, self.assertRaises(SystemExit) as stopped:
                    command([*args, "--referenced-notebook", *extra])
                self.assertEqual(stopped.exception.code, 2)
                mkdir.assert_not_called()

    def test_reload_restores_referenced_profile_before_loading_weights(self):
        protocol = referenced_protocol()
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
