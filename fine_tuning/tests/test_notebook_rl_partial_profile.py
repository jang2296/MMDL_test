"""Partial generation profile boundaries; no model generation or learning claims."""
import copy
import json
from pathlib import Path
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import local_profile as profile, local_run, local_train, partial_notebook
from fine_tuning.notebook_rl.checkpoint import digest


def partial_protocol():
    return profile.build_protocol(verified_note_synthesis_credit=True, partial_notebook=True, context_limit=24576)


class PartialProfileTests(unittest.TestCase):
    def test_partial_generation_changes_only_explicit_generation_contract(self):
        old = profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        current = partial_protocol()
        self.assertEqual(current["protocol_id"], profile.PARTIAL_NOTEBOOK_PROTOCOL_ID)
        self.assertEqual(profile.restore_protocol(json.loads(json.dumps(current))), current)
        self.assertEqual(current["notebook_generation_policy"], partial_notebook.VERSION)
        self.assertEqual(current["notebook_prompt_sha256"], digest(partial_notebook.prompt_contract()))
        self.assertEqual(current["notebook_max_rounds"], 1)
        self.assertIs(current["notebook_strict_feedback_repairs"], False)
        changed = {"protocol_id", "sha256", "notebook_prompt_sha256", "notebook_generation_policy",
                   "notebook_max_rounds", "notebook_strict_feedback_repairs"}
        self.assertEqual({k: v for k, v in current.items() if k not in changed},
                         {k: v for k, v in old.items() if k not in changed})
        self.assertEqual(current["rollout"]["diag"]["n"], 4)
        self.assertEqual(current["rollout"]["diag"]["max_new_tokens"], 8192)
        self.assertEqual(current["context_limit"], 24576)
        prompts, old_prompts = profile.local_prompts(current), profile.local_prompts(old)
        self.assertEqual(prompts["rewarded_notebook"], partial_notebook.prompt_contract())
        self.assertEqual(prompts["notebook_generation_policy"], partial_notebook.VERSION)
        self.assertEqual(prompts["notebook_max_rounds"], 1)
        self.assertIs(prompts["notebook_strict_feedback_repairs"], False)
        self.assertEqual({k: v for k, v in prompts.items() if k not in changed | {"rewarded_notebook"}},
                         {k: v for k, v in old_prompts.items() if k not in changed | {"rewarded_notebook"}})

    def test_legacy_protocol_digests_and_restore_are_unchanged(self):
        variants: list[tuple[dict[str, Any], str]] = [({}, "38e348b367a4f6c511c6f163a35240c5ece157ca50bb2f7d4f48f98e57077a71"),
            ({"verified_note_atoms": True}, "5b0f72d297f22a8ab42047315c5af48fd0806a27dc0eb88e88191bcec68ee56f"),
            ({"verified_note_components": True}, "85bb33be8eb08306f733888004092645f0919dd8f2e25c2ea7162575fdeb3cf7"),
            ({"verified_note_fact_credit": True}, "dda72bcd5312e7477a4e897930c72759fea0b2856c094e633b058920ab762ac5"),
            ({"verified_note_fact_credit": True, "context_limit": 24576},
             "99876e56fd2543302e3e1d74922270b3c3a45c480fd963f161fd124a7cdeeaf2"),
            ({"verified_note_synthesis": True}, "a1bdcaf25775bb43961f12d8d1afa9d46fff7af13d8fc3ef72b00fdaa9157ebb"),
            ({"verified_note_synthesis": True, "context_limit": 24576},
             "06f39a074c324aa62e6530cdf744e42eb936f685d78c0037a5d60c6fae0a49db"),
            ({"verified_note_synthesis_credit": True, "context_limit": 24576},
             "1b40a7e4169e0c94da8a88b78d98bcd67092fd1404b3004066382d779d233c91")]
        for kwargs, expected in variants:
            with self.subTest(kwargs=kwargs):
                protocol = profile.build_protocol(**kwargs)
                self.assertEqual(protocol["sha256"], expected)
                self.assertEqual(profile.restore_protocol(json.loads(json.dumps(protocol))), protocol)
                self.assertNotIn("notebook_generation_policy", protocol)

    def test_partial_flag_requires_true_boolean_explicit_credit_and_24k(self):
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576,
                                       partial_notebook=cast(Any, value))
        for kwargs in ({}, {"verified_note_synthesis": True}, {"verified_note_fact_credit": True},
                       {"verified_note_synthesis_credit": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                profile.build_protocol(partial_notebook=True, **kwargs)

    def test_rehashed_generation_reward_budget_and_boolean_tampering_are_rejected(self):
        replacements = {"notebook_generation_policy": "unknown", "notebook_prompt_sha256": "0" * 64,
            "notebook_max_rounds": True, "notebook_strict_feedback_repairs": 0,
            "notebook_reward": "partial-pass", "context_limit": 16384, "notebook_objective": {}}
        for key, value in [*replacements.items(), ("notebook_generation_policy", None)]:
            changed = copy.deepcopy(partial_protocol())
            if value is None:
                del changed[key]
            else:
                changed[key] = value
            changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
            for check in (profile.restore_protocol, profile.local_prompts):
                with self.subTest(key=key, check=check), self.assertRaises(ValueError):
                    check(changed)

    def test_existing_admission_retains_exact_execution_and_signal_requirements(self):
        old = profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        new = partial_protocol()
        policy = local_run.VERIFIED_UPDATE_EXPLORATION
        self.assertEqual(local_run.minimum_execution_contract(policy, new),
                         local_run.minimum_execution_contract(policy, old))
        for admission in (policy, local_train.SYNTHESIS_CREDIT_ADMISSION):
            self.assertEqual(local_train.execution_admission({"admission_policy": admission}, new), admission)
        contract = local_run.minimum_execution_contract(policy, new)
        assert contract is not None
        self.assertIs(contract["require_retained_learning_signal"], True)
        with self.assertRaises(ValueError):
            local_run.minimum_binding({}, new, contract | {"require_retained_learning_signal": False})
        with self.assertRaises(ValueError):
            local_train.notebook_learning_gate(Path("/unused"), {}, {}, new)

    def test_prepare_and_minimal_cli_propagate_only_explicit_partial_flag(self):
        flags = ["--verified-note-synthesis-credit", "--context-limit", "24576", "--partial-notebook"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as prepared, patch("builtins.print"):
            self.assertEqual(local_run.main(["prepare", "--previous-request", "/unused/request",
                "--selection", "/unused/selection", "--output", "/unused/data", *flags]), 0)
            self.assertEqual(prepared.call_args.args[2], partial_protocol())
        with patch.object(local_run, "capture") as capture:
            self.assertEqual(local_run.main(["minimal", "--preparation", "/unused/preparation",
                "--model-path", "/unused/base", "--output", "/unused/capture", *flags]), 0)
            self.assertIs(capture.call_args.kwargs["partial_notebook"], True)
        with patch.object(local_run, "read_preparation", side_effect=ValueError("CPU preparation boundary")) as read, \
                patch("fine_tuning.notebook_rl.quantized_policy.build_quantized_policy") as loader:
            with self.assertRaisesRegex(ValueError, "CPU preparation boundary"):
                local_run.capture(Path("/unused/preparation"), Path("/unused/base"), Path("/unused/capture"),
                    standalone=False, verified_note_synthesis_credit=True, partial_notebook=True, context_limit=24576)
            self.assertEqual(read.call_args.args[1], partial_protocol())
            loader.assert_not_called()

    def test_invalid_partial_cli_fails_before_io_or_capture(self):
        from fine_tuning.notebook_rl import quantized_cache

        commands = [(local_run.main, ["prepare", "--previous-request", "/unused/request",
            "--selection", "/unused/selection", "--output", "/unused/data"]),
            (local_run.main, ["minimal", "--preparation", "/unused/preparation",
             "--model-path", "/unused/base", "--output", "/unused/capture"]),
            (quantized_cache.main, ["save", "--model-path", "/unused/base", "--cache", "/unused/cache",
             "--output", "/unused/output"])]
        for command, args in commands:
            for extra in ([], ["--verified-note-synthesis", "--context-limit", "24576"],
                          ["--verified-note-synthesis-credit"]):
                with self.subTest(command=command, extra=extra), patch("sys.stderr"), \
                        patch.object(Path, "mkdir") as mkdir, self.assertRaises(SystemExit) as stopped:
                    command([*args, "--partial-notebook", *extra])
                self.assertEqual(stopped.exception.code, 2)
                mkdir.assert_not_called()

    def test_cache_cli_forwards_partial_policy_without_loading_weights(self):
        import torch
        from fine_tuning.notebook_rl import quantized_cache

        with tempfile.TemporaryDirectory() as directory:
            for stage in ("save", "check"):
                with self.subTest(stage=stage), patch.object(quantized_cache, "_sources", return_value={}), \
                        patch.object(quantized_cache, "build_protocol", side_effect=ValueError("CPU protocol boundary")) as build, \
                        patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution"), \
                        patch.multiple(torch.cuda, reset_peak_memory_stats=lambda: None,
                            max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0), \
                        patch.object(quantized_cache, "build_quantized_policy") as save, \
                        patch.object(quantized_cache, "load_quantized_policy") as load, \
                        patch("traceback.print_exc"), patch("builtins.print"):
                    self.assertEqual(quantized_cache.main([stage, "--model-path", "/unused/base", "--cache", "/unused/cache",
                        "--parent", "/unused/parent", "--output", str(Path(directory) / stage),
                        "--verified-note-synthesis-credit", "--context-limit", "24576", "--partial-notebook"]), 1)
                    self.assertIs(build.call_args.kwargs["partial_notebook"], True)
                    self.assertIs(build.call_args.kwargs["verified_note_synthesis_credit"], True)
                    save.assert_not_called()
                    load.assert_not_called()

    def test_reload_restores_saved_partial_or_legacy_profile_before_weights(self):
        old = profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        for protocol in (old, partial_protocol()):
            with self.subTest(protocol=protocol["protocol_id"]), tempfile.TemporaryDirectory() as directory:
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
