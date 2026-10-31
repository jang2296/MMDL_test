"""Explicit synthesis protocol boundaries; fixtures are not model-learning evidence."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from fine_tuning.notebook_rl import notebook_synthesis, reasoning_trace
from fine_tuning.notebook_rl.checkpoint import digest
from fine_tuning.notebook_rl.generators import graph, graph_pair
from fine_tuning.notebook_rl.local_profile import (
    SYNTHESIS_PROTOCOL_ID, SYNTHESIS_24K_PROTOCOL_ID, build_protocol, local_prompts, restore_protocol,
)
from fine_tuning.notebook_rl.prompts import (
    for_actor, native_solution_prompt_contract, retry_text, turn1_text,
)


class SynthesisProfileTests(unittest.TestCase):
    def test_new_reward_has_its_own_complete_roundtrip_contract(self):
        old = build_protocol()
        for context, identity in ((16384, SYNTHESIS_PROTOCOL_ID), (24576, SYNTHESIS_24K_PROTOCOL_ID)):
            with self.subTest(context=context):
                current = build_protocol(verified_note_synthesis=True, context_limit=context)
                self.assertEqual(current["protocol_id"], identity)
                self.assertTrue(identity.endswith("-v31"))
                self.assertEqual(restore_protocol(json.loads(json.dumps(current))), current)
                self.assertEqual(current["notebook_reward"], notebook_synthesis.BOUNDED_VERSION)
                self.assertEqual(current["solution_trace_contract"], reasoning_trace.VERSION)
                self.assertEqual(current["solution_prompt_sha256"], digest(native_solution_prompt_contract()))
                self.assertNotIn("notebook_objective", current)
                self.assertEqual(current["notebook_format_feedback_policy"], notebook_synthesis.FORMAT_DIAGNOSTICS_VERSION)
                self.assertEqual(current["notebook_format_feedback_sha256"],
                                 digest(notebook_synthesis.format_feedback_contract()))
                self.assertNotIn("solution_trace_contract", old)
                for key in ("model", "quantization", "image", "batch", "local_budget", "notebook_route_rule"):
                    self.assertEqual(current[key], old[key])
                expected_rollout = copy.deepcopy(old["rollout"])
                expected_rollout["diag"]["max_new_tokens"] = 8192
                self.assertEqual(current["rollout"], expected_rollout)
                prompts = local_prompts(current)
                self.assertEqual(prompts["rewarded_notebook"],
                                 notebook_synthesis.prompt_contract(notebook_synthesis.BOUNDED_VERSION))

    def test_notebook_budget_is_explicit_and_cannot_be_rehashed_to_a_smaller_limit(self):
        current = build_protocol(verified_note_synthesis=True, context_limit=24576)
        self.assertEqual(current["rollout"]["diag"]["max_new_tokens"], 8192)
        self.assertEqual(build_protocol()["rollout"]["diag"]["max_new_tokens"], 2048)
        current["rollout"]["diag"]["max_new_tokens"] = 2048
        current["sha256"] = digest({k: v for k, v in current.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "saved protocol changed"):
            restore_protocol(current)

    def test_rehashed_prompt_reward_or_scope_tampering_is_rejected(self):
        current = build_protocol(verified_note_synthesis=True)
        replacements = {
            "notebook_reward": "legacy", "notebook_synthesis_contract": notebook_synthesis.VERSION,
            "notebook_prompt_sha256": "0" * 64, "solution_trace_contract": "native-reasoning-trace-v1",
            "solution_prompt_sha256": "0" * 64, "notebook_feedback_policy": "give-correct-facts",
            "notebook_retry_instruction_sha256": "0" * 64, "context_limit": 24576,
            "synthesis_scope": "full_diagnosis", "notebook_objective": {},
            "notebook_format_feedback_policy": "private-expected-values",
            "notebook_format_feedback_sha256": "0" * 64,
        }
        for key, value in replacements.items():
            altered = copy.deepcopy(current)
            altered[key] = value
            altered["sha256"] = digest({k: v for k, v in altered.items() if k != "sha256"})
            with self.subTest(key=key):
                for check in (restore_protocol, local_prompts):
                    with self.assertRaisesRegex(ValueError, "saved protocol changed"):
                        check(altered)

    def test_new_contract_cannot_be_spliced_into_legacy_or_partial_rewards(self):
        for flag in ("verified_note_atoms", "verified_note_components", "verified_note_fact_credit"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                build_protocol(verified_note_synthesis=True, **{flag: True})
        with self.assertRaises(ValueError):
            build_protocol(verified_note_synthesis=cast(Any, "true"))
        with self.assertRaisesRegex(ValueError, "own protocol"):
            local_prompts(build_protocol() | {"notebook_synthesis_contract": notebook_synthesis.BOUNDED_VERSION})

    def test_each_public_task_gets_an_executable_trace_shape_and_no_prose_template(self):
        prompts = local_prompts(build_protocol(verified_note_synthesis=True))
        problems = [graph.generate(i, 3407, "train", "obs") for i in (0, 1, 2, 3, 7, 11)]
        problems.append(graph_pair.generate(0, 3407, "train", "obs"))
        for problem in problems:
            actor = problem.actor
            with self.subTest(id=actor.problem_id):
                text = turn1_text(actor, prompts)
                self.assertIn(actor.question, text)
                self.assertIn(reasoning_trace.instruction_for_question(actor.question, actor.source), text)
                self.assertNotIn("<your reasoning>", text)
                self.assertNotIn("<your new reasoning>", text)
                self.assertNotIn("{TRACE_INSTRUCTION}", text)
                bound = for_actor(actor, prompts)
                retry = retry_text("INCORRECT", "wrong-submitted-value", "obs", actor.family, bound)
                self.assertIn('"wrong-submitted-value": INCORRECT', retry)
                self.assertIn("correct answer is not provided", retry)
                self.assertIn("Trace:", retry)
                self.assertNotIn("Error check:\n", retry)
                self.assertEqual(prompts, local_prompts(build_protocol(verified_note_synthesis=True)))

    def test_invalid_and_truncated_feedback_never_claims_incorrectness(self):
        actor = graph.generate(0, 3407, "train", "obs").actor
        prompts = for_actor(actor, local_prompts(build_protocol(verified_note_synthesis=True)))
        for status in ("INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"):
            with self.subTest(status=status):
                text = retry_text(status, None, "obs", actor.family, prompts)
                self.assertIn("correctness is unknown", text)
                self.assertNotIn("INCORRECT", text)
        for answer in (None, "", "A\nB", "A\rB"):
            with self.assertRaises(ValueError):
                retry_text("INCORRECT", answer, "obs", actor.family, prompts)

    def test_unsupported_or_unbound_actor_is_rejected_before_generation(self):
        actor = graph.generate(0, 3407, "train", "obs").actor
        prompts = local_prompts(build_protocol(verified_note_synthesis=True))
        for altered in (replace(actor, source="unsupported"), replace(actor, question="different question"),
                        replace(actor, mode="plain")):
            with self.subTest(actor=altered.problem_id), self.assertRaises(ValueError):
                turn1_text(altered, prompts)
        with self.assertRaisesRegex(ValueError, "actor-bound"):
            retry_text("INCORRECT", "A", "obs", actor.family, prompts)
        with self.assertRaises(ValueError):
            for_actor(actor, prompts | {"solution_trace_contract": "unknown"})

    def test_prepare_and_minimal_cli_bind_new_flag_without_model_execution(self):
        from fine_tuning.notebook_rl import local_run

        for context in (16384, 24576):
            flags = ["--verified-note-synthesis", "--context-limit", str(context)]
            with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                       return_value={"status": "LOCAL_DATA_REVALIDATED"}) as prepare, patch("builtins.print"):
                self.assertEqual(local_run.main(["prepare", "--previous-request", "/private/request.json",
                    "--selection", "/private/selection.json", "--output", "/private/unused", *flags]), 0)
                self.assertEqual(prepare.call_args.args[2],
                                 build_protocol(verified_note_synthesis=True, context_limit=context))
            with patch("fine_tuning.notebook_rl.local_run.capture") as capture:
                self.assertEqual(local_run.main(["minimal", "--preparation", "/private/preparation.json",
                    "--model-path", "/private/base", "--output", "/private/unused", *flags]), 0)
                self.assertIs(capture.call_args.kwargs["verified_note_synthesis"], True)
                self.assertEqual(capture.call_args.kwargs["context_limit"], context)

    def test_minimal_capture_binds_new_protocol_before_loading_weights(self):
        from fine_tuning.notebook_rl.local_run import capture

        with patch("fine_tuning.notebook_rl.local_run.read_preparation",
                   side_effect=ValueError("CPU preparation boundary")) as prepared, \
                patch("fine_tuning.notebook_rl.quantized_policy.build_quantized_policy") as loader:
            with self.assertRaisesRegex(ValueError, "CPU preparation boundary"):
                capture(Path("/private/preparation.json"), Path("/private/base"), Path("/private/unused"),
                        standalone=False, verified_note_synthesis=True, context_limit=24576)
            self.assertEqual(prepared.call_args.args[1], build_protocol(verified_note_synthesis=True, context_limit=24576))
            loader.assert_not_called()

    def test_cache_cli_binds_new_protocol_before_loading_weights(self):
        import torch
        from fine_tuning.notebook_rl import quantized_cache

        with tempfile.TemporaryDirectory() as directory:
            for stage in ("save", "check"):
                with patch("fine_tuning.notebook_rl.quantized_cache._sources", return_value={}), \
                        patch("fine_tuning.notebook_rl.quantized_cache.build_protocol",
                              side_effect=ValueError("CPU protocol boundary")) as build, \
                        patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution"), \
                        patch.multiple(torch.cuda, reset_peak_memory_stats=lambda: None,
                                       max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0), \
                        patch("fine_tuning.notebook_rl.quantized_cache.build_quantized_policy") as save, \
                        patch("fine_tuning.notebook_rl.quantized_cache.load_quantized_policy") as load, \
                        patch("traceback.print_exc"), patch("builtins.print"):
                    self.assertEqual(quantized_cache.main([stage, "--model-path", "/private/original",
                        "--cache", "/private/cache", "--parent", "/private/parent", "--output",
                        str(Path(directory) / stage), "--verified-note-synthesis", "--context-limit", "24576"]), 1)
                    self.assertIs(build.call_args.kwargs["verified_note_synthesis"], True)
                    self.assertEqual(build.call_args.kwargs["context_limit"], 24576)
                    save.assert_not_called()
                    load.assert_not_called()

    def test_conflicting_cli_objectives_fail_before_work(self):
        from fine_tuning.notebook_rl import local_run, quantized_cache

        commands = ((local_run.main, ["prepare", "--previous-request", "/private/request.json",
            "--selection", "/private/selection.json", "--output", "/private/unused"]),
            (local_run.main, ["minimal", "--preparation", "/private/preparation.json",
             "--model-path", "/private/base", "--output", "/private/unused"]),
            (quantized_cache.main, ["save", "--model-path", "/private/original", "--cache", "/private/cache",
             "--output", "/private/unused"]))
        for command, args in commands:
            for other in ("--verified-note-atoms", "--verified-note-components", "--verified-note-fact-credit"):
                with self.subTest(stage=args[0], other=other), patch("sys.stderr"), \
                        patch.object(Path, "mkdir") as mkdir, self.assertRaises(SystemExit) as stopped:
                    command([*args, "--verified-note-synthesis", other])
                self.assertEqual(stopped.exception.code, 2)
                mkdir.assert_not_called()
