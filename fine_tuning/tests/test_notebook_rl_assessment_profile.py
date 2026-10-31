"""CPU lifecycle contracts and authored notes; no pretrained-model learning claim."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode
from fine_tuning.notebook_rl import local_profile, local_run, local_train, quantized_cache, updates
from fine_tuning.notebook_rl.checkpoint import digest
from fine_tuning.notebook_rl.rollout import RolloutConfig
from fine_tuning.tests.test_notebook_rl_synthesis_integration import ChunkTokenizer
from fine_tuning.tests import test_notebook_rl_whole_reference_training_episode as episode_fixtures


OPTIONS: dict[str, Any] = {"verified_note_synthesis_credit": True, "pair_draft_synthesis": True,
                          "whole_reference_notebook": True, "verified_note_assessment_credit": True, "context_limit": 24576}
FLAGS = ["--verified-note-assessment-credit", "--verified-note-synthesis-credit", "--pair-draft-synthesis",
         "--whole-reference-notebook", "--context-limit", "24576"]


class AssessmentTokenizer(ChunkTokenizer):
    def encode(self, text, **kwargs):
        if not text.startswith("{"):
            return super().encode(text, **kwargs)
        for character in text:
            if character not in self.chunks:
                self.chunks[character] = len(self.chunks) + 100
        return [self.chunks[character] for character in text]


class AssessmentProfileTests(unittest.TestCase):
    def test_new_objective_preserves_all_generation_prompts_and_old_profile(self):
        current = local_profile.build_protocol(**OPTIONS)
        previous = local_profile.build_protocol(**(OPTIONS | {"verified_note_assessment_credit": False}))
        self.assertEqual({key for key in current if current[key] != previous.get(key)},
                         {"protocol_id", "notebook_objective", "notebook_update_requirement", "sha256"})
        self.assertEqual(current["protocol_id"], local_profile.ASSESSMENT_PROTOCOL_ID)
        self.assertEqual(current["notebook_update_requirement"], updates.ASSESSMENT_REQUIREMENT)
        self.assertEqual(previous["notebook_update_requirement"], updates.NATIVE_RELATIONSHIP_REQUIREMENT)
        self.assertEqual(local_profile.local_prompts(current), local_profile.local_prompts(previous))
        for protocol in (current, previous):
            self.assertEqual(local_profile.restore_protocol(json.loads(json.dumps(protocol))), protocol)

    def test_options_and_resealed_objective_requirement_or_prompt_changes_fail_closed(self):
        for changes in ({"verified_note_synthesis_credit": False}, {"pair_draft_synthesis": False},
                        {"whole_reference_notebook": False}, {"whole_reference_pages": True},
                        {"pair_action_notebook": True}, {"partial_notebook": True},
                        {"referenced_notebook": True}, {"paired_notebook": True},
                        {"context_limit": 16384}, {"verified_note_assessment_credit": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                local_profile.build_protocol(**(OPTIONS | changes))
        for key in ("notebook_objective", "notebook_update_requirement", "notebook_prompt_sha256",
                    "notebook_generation_policy", "protocol_id"):
            current = local_profile.build_protocol(**OPTIONS)
            current[key] = "changed"
            current["sha256"] = digest({k: v for k, v in current.items() if k != "sha256"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                local_profile.restore_protocol(current)

    def test_cli_prepare_minimal_and_cache_propagate_explicit_assessment_flag(self):
        prepare = ["prepare", "--previous-request", "/private/request", "--selection", "/private/selection",
                   "--output", "/private/unused"]
        minimal = ["minimal", "--preparation", "/private/preparation", "--model-path", "/private/model",
                   "--output", "/private/unused"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as called, patch("builtins.print"):
            self.assertEqual(local_run.main(prepare + FLAGS), 0)
            self.assertEqual(called.call_args.args[2], local_profile.build_protocol(**OPTIONS))
        with patch.object(local_run, "capture") as called:
            self.assertEqual(local_run.main(minimal + FLAGS), 0)
            self.assertIs(called.call_args.kwargs["verified_note_assessment_credit"], True)
        for command in (prepare, minimal):
            invalid: tuple[list[str], ...] = (["--verified-note-assessment-credit"],
                                              [flag for flag in FLAGS if flag != "--whole-reference-notebook"])
            if command is minimal:
                invalid += (FLAGS + ["--execution-policy", local_run.VERIFIED_UPDATE_EXPLORATION],)
            for flags in invalid:
                with self.subTest(command=command[0], flags=flags), patch("sys.stderr"), \
                        patch.object(Path, "mkdir") as mkdir, self.assertRaises(SystemExit):
                    local_run.main(command + flags)
                mkdir.assert_not_called()
        for stage in ("save", "check"):
            with tempfile.TemporaryDirectory() as directory, patch.object(quantized_cache, "build_protocol",
                    side_effect=ValueError("CPU protocol boundary")) as build, \
                    patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution"), \
                    patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=0), \
                    patch("torch.cuda.max_memory_reserved", return_value=0), patch("traceback.print_exc"), patch("builtins.print"):
                command = [stage, "--cache", "/private/cache", "--parent", "/private/parent",
                           "--model-path", "/private/model", "--output", str(Path(directory) / "out")]
                with patch("sys.stderr"), self.assertRaises(SystemExit):
                    quantized_cache.main(command + ["--verified-note-assessment-credit"])
                self.assertFalse((Path(directory) / "out").exists())
                self.assertEqual(quantized_cache.main(command + FLAGS), 1)
                self.assertIs(build.call_args.kwargs["verified_note_assessment_credit"], True)

    def test_assessment_requires_its_explicit_admission_and_never_implies_strict_notebook_success(self):
        protocol = local_profile.build_protocol(**OPTIONS)
        self.assertEqual(local_train.execution_admission({"admission_policy": local_train.ASSESSMENT_CREDIT_ADMISSION},
                                                        protocol), local_train.ASSESSMENT_CREDIT_ADMISSION)
        with self.assertRaises(ValueError):
            local_run.minimum_execution_contract(local_run.VERIFIED_UPDATE_EXPLORATION, protocol)
        for admission in (local_train.STRICT_ADMISSION, local_train.EXPLORATORY_ADMISSION,
                          local_train.SYNTHESIS_CREDIT_ADMISSION, local_train.VERIFIED_UPDATE_EXPLORATION,
                          local_train.FIXED_TEN_NATIVE_EXPLORATION):
            with self.subTest(admission=admission), self.assertRaisesRegex(ValueError, "assessment-credit.*unsupported"):
                local_train.execution_admission({"admission_policy": admission}, protocol)
        with self.assertRaisesRegex(ValueError, "assessment-credit.*unsupported"):
            local_train.notebook_learning_gate(Path("/private/unused"), {}, {}, protocol)
        with self.assertRaisesRegex(ValueError, "assessment-credit.*unsupported"):
            local_train.fact_credit_learning_gate(Path("/private/unused"), {}, {}, protocol,
                                                 actors={}, verifier=None, tokenizer=None)


class AssessmentLifecycleTests(unittest.TestCase):
    def setUp(self):
        case = episode_fixtures.WholeReferenceTrainingEpisodeTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.case, self.f = case, case.f
        f = self.f
        f.protocol = local_profile.build_protocol(**OPTIONS)
        f.prompts = local_profile.local_prompts(f.protocol)
        f.cfg = RolloutConfig.from_protocol(f.protocol, "ALG-NB3", master_seed=3407)
        f.engine.tokenizer = AssessmentTokenizer()
        for index, sample in enumerate(episode.solution_samples(f.tree)):
            parent = episode.solution_samples(f.tree)[0] if index in (2, 3) else (
                episode.solution_samples(f.tree)[2] if index in (4, 5) else None)
            sample.prompt_token_ids = ([1, 2] if parent is None else
                parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2])
            sample.completion_token_ids = f.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
        note = json.loads(case.good)
        note["corrected_facts"] = {}
        for row in note["assessments"]:
            row[3] = None
        self.case.append([json.dumps(note), "{}", json.dumps(note), "{}"])
        updates.bind_notebook_objective(f.tree, f.actor, f.verifier, f.protocol["notebook_objective"])

    def test_original_assessment_signal_and_real_cpu_update_evidence(self):
        f = self.f
        self.assertTrue(local_run.assessment_signal([f.tree], f.protocol["notebook_objective"], f.engine.tokenizer))
        update = f.step(f.optimizer_fixture())
        self.assertEqual(update["status"], "UPDATED_UNSYNCHRONIZED")
        summary = local_run.assessment_update_evidence(update, f.protocol)
        self.assertTrue(summary["assessment_learning_verified"])
        self.assertFalse(summary["strict_notebook_correctness_verified"])
        self.assertTrue(all(sample.reward == 0 for sample in f.tree.groups[-1].samples))
        for mutation in ("mis", "hash", "flag_count", "overall_count", "nonfinite", "no_vjp", "not_committed", "vision"):
            forged = copy.deepcopy(update)
            sid = forged["retained_assessment_signal"]["signal_sample_ids"][0]
            gradient = next(row["assessment_gradient"] for row in forged["notebook_backward"]
                            if row.get("assessment_gradient", {}).get("measured"))
            if mutation == "mis":
                forged["mean_log_ratio_by_sample"][sid] = 999.
            elif mutation == "hash":
                gradient["token_advantages_sha256"] = "wrong"
            elif mutation == "flag_count":
                forged["retained_assessment_signal"]["by_expected_flags"]["I"]["positive_assessment_tokens"] = 10**9
            elif mutation == "overall_count":
                forged["retained_assessment_signal"]["positive_assessment_tokens"] += 1
            elif mutation == "nonfinite":
                gradient["group_norms"]["L"] = float("nan")
            elif mutation == "no_vjp":
                gradient["measured"] = False
            elif mutation == "vision":
                gradient["group_norms"]["V"] = 1.
                gradient["nonzero_groups"] = ["L", "V"]
            else:
                forged["assessment_learning_verified"] = False
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "assessment-only"):
                local_run.assessment_update_evidence(forged, f.protocol)

    def test_minimum_uses_assessment_preflight_and_never_native_relationship_gate(self):
        f = self.f
        f.tree.notebook.pop("learning_objective")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            build.mkdir()
            (build / "manifest.yaml").write_text(json.dumps({"split_manifests": {"train": "fixture"}}))
            f.verifier.metadata = {"vault_manifest_sha256": "fixture"}
            returns = {"local_run.read_preparation": ({"build_root": str(build)}, {}, [f.actor]),
                "local_run.minimal_actors": [f.actor], "local_run._binding": {}, "local_run._install_actor_guards": [],
                "local_run.load_images": f.images[f.actor.problem_id], "local_run.make_probes": [],
                "local_calibrate.verify_original": {}, "quantized_policy.build_quantized_policy": SimpleNamespace(inventory={}),
                "shared_policy_engine.SharedPolicyRolloutEngine": f.engine,
                "checkpoint.make_optimizer": (object(), object()), "verifier_service.VerifierService": nullcontext(f.verifier),
                "trainer.synchronize": {}, "rollout.build_tree": f.tree, "notebook_learning.append_rewarded_notebooks": None,
                "updates.partition_learning_trees": ([f.tree], [], []), "local_run.parameter_hashes": {}}
            with ExitStack() as stack:
                for target, value in returns.items():
                    stack.enter_context(patch("fine_tuning.notebook_rl." + target, return_value=value))
                for target in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
                    stack.enter_context(patch("torch.cuda." + target, return_value=0))
                stack.enter_context(patch.object(local_run, "error_relationship_signal", side_effect=AssertionError("old gate")))
                stack.enter_context(patch("builtins.print"))
                optimizer = stack.enter_context(patch.object(updates, "optimizer_step",
                    return_value={"status": "NO_RETAINED_ASSESSMENT_SIGNAL"}))
                result = local_run.capture(root / "preparation", root / "model", root / "capture", standalone=False, **OPTIONS)
            self.assertEqual(result["status"], "NO_RETAINED_ASSESSMENT_SIGNAL")
            self.assertFalse(result["assessment_learning_verified"])
            optimizer.assert_called_once()

    def test_reload_rejects_changed_source_protocol_or_assessment_evidence_before_loading(self):
        protocol = local_profile.build_protocol(**OPTIONS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch: dict[str, Any] = {"pid": 123, "protocol": protocol, "binding": {"code_sha256": "wrong"}}
            result = {"status": "UPDATED_SAVED_PENDING_FRESH_RELOAD", "optimizer_updates": 1}
            for name, value in (("launch.json", launch), ("result.json", result), ("reload_expected.json", {})):
                (root / name).write_text(json.dumps(value))
            with patch("psutil.pid_exists", return_value=False), \
                    patch("fine_tuning.notebook_rl.approval.code_hashes", return_value={}), \
                    patch("fine_tuning.notebook_rl.checkpoint.load_resume") as load:
                with self.assertRaisesRegex(ValueError, "same code/protocol"):
                    local_run.reload_capture(root, root / "reload")
                launch["binding"]["code_sha256"] = digest({})
                (root / "launch.json").write_text(json.dumps(launch))
                (root / "update.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "assessment-only"):
                    local_run.reload_capture(root, root / "reload")
                launch["protocol"]["notebook_update_requirement"] = updates.NATIVE_RELATIONSHIP_REQUIREMENT
                (root / "launch.json").write_text(json.dumps(launch))
                with self.assertRaisesRegex(ValueError, "saved protocol changed"):
                    local_run.reload_capture(root, root / "reload")
                load.assert_not_called()
            self.assertFalse((root / "reload").exists())
