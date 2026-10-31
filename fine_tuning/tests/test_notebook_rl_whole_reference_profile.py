"""CPU contract and call-boundary checks; no pretrained model or improvement claim."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import local_profile, local_run, local_train, quantized_cache, updates
from fine_tuning.notebook_rl import whole_reference_notebook as reference
from fine_tuning.notebook_rl.checkpoint import digest
from fine_tuning.tests import test_notebook_rl_native_commit_gate as native_fixtures
from fine_tuning.tests.test_notebook_rl_synthesis_credit_admission import output_fixture
from fine_tuning.tests import test_notebook_rl_whole_reference_training_episode as episode_fixtures
from fine_tuning.notebook_rl.schema import to_json


def protocol(**changes):
    return local_profile.build_protocol(**({"verified_note_synthesis_credit": True,
        "pair_draft_synthesis": True, "whole_reference_notebook": True,
        "context_limit": 24576} | changes))


class WholeReferenceProfileTests(unittest.TestCase):
    def test_canonical_profile_keeps_generation_budget_and_requires_native_commit(self):
        actual = protocol()
        legacy = protocol(whole_reference_notebook=False)
        changed = {key for key in actual if actual.get(key) != legacy.get(key)}
        self.assertEqual(changed, {"protocol_id", "sha256", "notebook_generation_policy",
            "notebook_reference_contract", "notebook_reference_contract_sha256",
            "notebook_prompt_sha256", "notebook_update_requirement"})
        self.assertEqual(actual["notebook_update_requirement"], updates.NATIVE_RELATIONSHIP_REQUIREMENT)
        self.assertEqual(local_profile.restore_protocol(json.loads(json.dumps(actual))), actual)
        prompts = local_profile.local_prompts(actual)
        self.assertEqual(prompts["rewarded_notebook"], reference.prompt_contract())
        self.assertEqual(prompts["notebook_reference_contract_sha256"], digest(reference.contract()))
        self.assertEqual(prompts["notebook_max_rounds"], 1)
        self.assertEqual(local_train.execution_admission({"admission_policy": local_train.SYNTHESIS_CREDIT_ADMISSION},
                                                        actual), local_train.SYNTHESIS_CREDIT_ADMISSION)
        with self.assertRaises(ValueError):
            local_run.minimum_execution_contract(local_run.VERIFIED_UPDATE_EXPLORATION, actual)

    def test_incomplete_or_resealed_contracts_are_rejected(self):
        for changes in ({"whole_reference_notebook": 1}, {"pair_draft_synthesis": False},
                        {"verified_note_synthesis_credit": False}, {"context_limit": 16384},
                        {"pair_action_notebook": True}, {"partial_notebook": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                protocol(**changes)
        for field in ("notebook_update_requirement", "notebook_reference_contract", "notebook_prompt_sha256"):
            forged = protocol()
            forged[field] = "changed"
            forged["sha256"] = digest({k: v for k, v in forged.items() if k != "sha256"})
            with self.subTest(field=field), self.assertRaises(ValueError):
                local_profile.restore_protocol(forged)

    def test_cli_prepare_minimal_and_cache_forward_explicit_reference_mode(self):
        flags = ["--whole-reference-notebook", "--pair-draft-synthesis",
                 "--verified-note-synthesis-credit", "--context-limit", "24576"]
        prepare = ["prepare", "--previous-request", "/private/request", "--selection", "/private/selection",
                   "--output", "/private/new-data"]
        minimal = ["minimal", "--preparation", "/private/preparation", "--model-path", "/private/model",
                   "--output", "/private/new-minimal"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as called:
            self.assertEqual(local_run.main(prepare + flags), 0)
            self.assertEqual(called.call_args.args[2], protocol())
        with patch.object(local_run, "capture") as called:
            self.assertEqual(local_run.main(minimal + flags), 0)
            self.assertIs(called.call_args.kwargs["whole_reference_notebook"], True)
        for args in (prepare, minimal):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                local_run.main(args + ["--whole-reference-notebook"])
        for stage in ("save", "check"):
            with tempfile.TemporaryDirectory() as directory, patch.object(quantized_cache, "build_protocol",
                    side_effect=ValueError("CPU protocol boundary")) as called, \
                    patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution"), \
                    patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=0), \
                    patch("torch.cuda.max_memory_reserved", return_value=0), patch("traceback.print_exc"):
                output = Path(directory) / "out"
                args = [stage, "--cache", "/private/cache", "--parent", "/private/parent",
                        "--model-path", "/private/model", "--output", str(output)]
                with patch("sys.stderr"), self.assertRaises(SystemExit):
                    quantized_cache.main(args + ["--whole-reference-notebook"])
                self.assertFalse(output.exists())
                self.assertEqual(quantized_cache.main(args + flags), 1)
                self.assertIs(called.call_args.kwargs["whole_reference_notebook"], True)

    def test_output_finalization_does_not_satisfy_native_signal(self):
        fixture = output_fixture()
        self.addCleanup(fixture.doCleanups)
        args = ([fixture.tree], fixture.protocol["notebook_objective"], fixture.engine.tokenizer)
        self.assertTrue(local_run.error_relationship_signal(*args))
        self.assertFalse(local_run.error_relationship_signal(*args, native_only=True))

    def test_native_evidence_is_bound_to_actual_retained_credit(self):
        fixture = native_fixtures.NativeCommitGateTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        optimizer = fixture.prepare()
        update = fixture.run_step(optimizer)
        self.assertEqual(update["status"], "UPDATED_UNSYNCHRONIZED")
        summary = local_run.native_relationship_update_evidence(update, fixture.f.protocol)
        self.assertTrue(summary["native_relationship_only_gradient_verified"])
        self.assertFalse(summary["strict_notebook_correctness_verified"])
        for mutation in ("unretained", "output_only", "wrong_hash", "missing_requirement", "token_count", "no_vjp"):
            forged = copy.deepcopy(update)
            signal = forged["retained_native_relationship_signal"]
            sid = signal["signal_sample_ids"][0]
            if mutation == "unretained":
                forged["mean_log_ratio_by_sample"][sid] = 999.
            elif mutation == "output_only":
                for entry in forged["synthesis_token_credit"][sid]["relationships"]:
                    entry["credit_type"] = "OUTPUT_FINALIZATION"
            elif mutation == "missing_requirement":
                forged["native_relationship_commit_required"] = False
            elif mutation == "token_count":
                signal["positive_native_relationship_tokens"] = 10**9
            else:
                for row in forged["notebook_backward"]:
                    native = row.get("native_relationship_gradient", {})
                    if mutation == "no_vjp":
                        native["measured"] = False
                    else:
                        native["credit_sha256"] = "wrong"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                local_run.native_relationship_update_evidence(forged, fixture.f.protocol)

    def test_complete_reference_admission_regrades_raw_tree_and_native_update(self):
        case = episode_fixtures.WholeReferenceTrainingEpisodeTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.append()
        f = case.f
        updates.bind_notebook_objective(f.tree, f.actor, f.verifier, f.protocol["notebook_objective"])
        optimizer = f.optimizer_fixture()
        first = f.tree.groups[-1].samples[0].sample_id

        def first_zero(bundle, inputs, completion_len, *, chunk):
            if inputs["sample"].sample_id == first:
                return torch.tensor([inputs["sample"].sampler_logprobs], dtype=torch.float64) + bundle.model["L"].sum() * 0
            return optimizer[4](bundle, inputs, completion_len, chunk=chunk)

        update = f.step((*optimizer[:4], first_zero, optimizer[5]))
        self.assertEqual(update["status"], "UPDATED_UNSYNCHRONIZED")
        measured = [row for row in update["notebook_backward"] if row["relationship_gradient"]["measured"]]
        self.assertEqual(measured[0]["relationship_gradient"]["nonzero_groups"], [])
        self.assertNotIn(first, local_run.relationship_learning_summary(update)["native_relationship_gradient_sample_ids"])
        result = {"notebook_signal_groups": 1, "notebook_gradient_contribution_verified": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw-0000.json"
            path.write_text(json.dumps({"kind": "TREE", "raw": json.loads(to_json(f.tree))}))
            kwargs = {"actors": {f.actor.problem_id: f.actor}, "verifier": f.verifier,
                      "tokenizer": f.engine.tokenizer}
            self.assertEqual(local_train.fact_credit_learning_gate(root, result, update, f.protocol, **kwargs), [path])
            forged = copy.deepcopy(update)
            for row in forged["notebook_backward"]:
                row["native_relationship_gradient"] = {"measured": False}
            with self.assertRaises(ValueError):
                local_train.fact_credit_learning_gate(root, result, forged, f.protocol, **kwargs)
            with self.assertRaises(ValueError):
                local_train.notebook_learning_gate(root, result, forged, f.protocol)


if __name__ == "__main__":
    unittest.main()
