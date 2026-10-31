"""CPU fixtures for post-update plumbing; scripted replies are not Qwen improvement."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import checkpoint, fact_context_probe as capture
from fine_tuning.notebook_rl import notebook_whole_trained_probe as probe
from fine_tuning.notebook_rl.engines import Completion
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.rollout import RolloutConfig
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_whole_synthesis_probe as whole_tests
from fine_tuning.tests import test_notebook_rl_whole_reference_trained_probe as reference_tests


class WholeTrainedProbeTests(unittest.TestCase):
    def fixture(self, *, assessment=False):
        f, actors, trees, good = (reference_tests.WholeReferenceTrainedProbeTests.fixture(self) if assessment
                                 else whole_tests.WholeSynthesisProbeTests.fixture(self))
        if assessment:
            f.protocol = build_protocol(verified_note_synthesis_credit=True,
                pair_draft_synthesis=True, whole_reference_notebook=True,
                verified_note_assessment_credit=True, context_limit=24576)
            f.prompts = local_prompts(f.protocol)
            f.cfg = RolloutConfig.from_protocol(f.protocol, "ALG-NB3", master_seed=3407)
        f.engine.policy_version = 1
        f.engine.context_limit = 24576
        return f, actors, trees, good

    def collect(self, f, actors, trees, replies, *, record=None, deadline=None):
        f.engine.replies = [replies]
        events = []
        rows, skipped = probe.collect(actors, trees, f.images, f.prompts, f.engine, f.verifier,
            record or (lambda *row: events.append(row)),
            time.monotonic() + 60 if deadline is None else deadline, protocol=f.protocol)
        return rows, skipped, events

    def test_only_canonical_complete_whole_credit_protocols_are_supported(self):
        protocol = build_protocol(verified_note_synthesis_credit=True,
            pair_draft_synthesis=True, context_limit=24576)
        contract = probe.contract(protocol)
        self.assertEqual((contract["n"], contract["max_new_tokens"], contract["context_limit"]), (4, 8192, 24576))
        for key in ("heldout", "training_allowed", "notebook_reward_effect_isolated", "draft_regeneration"):
            self.assertIs(contract[key], False)
        for protocol in (build_protocol(), build_protocol(verified_note_synthesis=True),
                build_protocol(verified_note_synthesis_credit=True, context_limit=24576),
                build_protocol(verified_note_synthesis_credit=True, paired_notebook=True, context_limit=24576),
                {"protocol_id": "unsupported"}):
            with self.assertRaises(ValueError):
                probe.contract(protocol)
        altered = build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        altered["context_limit"] = 16384
        with self.assertRaises(ValueError):
            probe.contract(altered)

    def test_assessment_contract_is_explicit_and_rejects_resealed_objective_mismatch(self):
        protocol = build_protocol(verified_note_synthesis_credit=True,
            pair_draft_synthesis=True, whole_reference_notebook=True,
            verified_note_assessment_credit=True, context_limit=24576)
        policy = probe.contract(protocol)
        self.assertEqual(policy["protocol_id"], probe.ASSESSMENT_VERSION)
        self.assertEqual(policy["objective"], probe.credit.contract(probe.credit.ASSESSMENT_VERSION))
        self.assertEqual((policy["n"], policy["max_new_tokens"], policy["context_limit"], policy["optimizer_updates"]),
                         (4, 8192, 24576, 0))
        for key in ("heldout", "training_allowed", "draft_regeneration", "self_improvement_verified",
                    "notebook_reward_effect_isolated", "all_reasoning_verified"):
            self.assertIs(policy[key], False)
        for assessment in (False, True):
            changed = copy.deepcopy(protocol)
            if assessment:
                changed["notebook_objective"] = probe.credit.contract()
            else:
                changed["protocol_id"] = probe.WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID
            changed["sha256"] = probe.credit.digest({key: value for key, value in changed.items() if key != "sha256"})
            with self.subTest(assessment=assessment), self.assertRaises(ValueError):
                probe.contract(changed)

    def test_assessment_scores_original_and_observed_raw_samples_with_unchanged_history_and_strict_checks(self):
        f, actors, trees, good = self.fixture(assessment=True)
        originals = [to_json(tree) for tree in trees]
        bad = json.loads(good)
        normal = next(row for row in bad["assessments"] if row[1] == "N")
        normal[1] = "I"
        with patch.object(probe.credit, "score_note", wraps=probe.credit.score_note) as scorer:
            rows, skipped, _ = self.collect(f, actors, trees, [good, json.dumps(bad), "not JSON", (good, "length")])
        self.assertEqual([to_json(tree) for tree in trees], originals)
        self.assertEqual(skipped, [{"problem_id": actors[0].problem_id, "reason": "NO_VERIFIED_RECOVERY"}])
        request, row = f.engine.requests[0], rows[0]
        baseline = f.tree.groups[-1]
        self.assertEqual(request.request_id, f"{f.actor.problem_id}:{probe.ASSESSMENT_VERSION}")
        self.assertEqual(request.messages, f.tree.notebook["rewarded_notebook"]["notebook_messages"])
        self.assertIs(request.images, f.images[f.actor.problem_id])
        self.assertEqual(request.seeds, [sample.seed for sample in baseline.samples])
        self.assertEqual((request.n, request.max_new_tokens, f.engine.context_limit), (4, 8192, 24576))
        self.assertEqual((row["before_totals"]["strict_verified_notes"], row["after_totals"]["strict_verified_notes"]), (0, 1))
        self.assertEqual((row["history_policy_version"], row["generation_policy_version"], row["optimizer_updates"]), (0, 1, 0))
        self.assertFalse(row["heldout"])
        self.assertFalse(row["self_improvement_verified"])
        self.assertEqual(scorer.call_count, 8)
        for call, checked in zip(scorer.call_args_list[:4], row["baseline"], strict=True):
            self.assertNotIn("observed_sample", call.kwargs)
            sample = next(sample for sample in baseline.samples if sample.sample_id == call.kwargs["sample_id"])
            self.assertEqual(call.kwargs["objective_version"], probe.credit.ASSESSMENT_VERSION)
            self.assertEqual(checked["score"]["objective_version"], probe.credit.ASSESSMENT_VERSION)
            probe.credit.validate_score(checked["score"], sample)
        for call, checked in zip(scorer.call_args_list[4:], row["samples"], strict=True):
            sample = call.kwargs["observed_sample"]
            self.assertEqual((call.kwargs["sample_id"], sample.policy_version), (sample.sample_id, 1))
            self.assertEqual(call.kwargs["objective_version"], probe.credit.ASSESSMENT_VERSION)
            self.assertEqual(checked["score"]["objective_version"], probe.credit.ASSESSMENT_VERSION)
            self.assertEqual(sample.prompt_token_ids, baseline.samples[0].prompt_token_ids)
            self.assertEqual(f.engine.tokenizer.decode(sample.completion_token_ids), checked["text"])
            self.assertTrue(checked["score"]["diagnostic_only"])
            self.assertFalse(checked["eligible_for_rl_loss"])
            self.assertFalse(checked["used_for_routing"])
            with self.assertRaises(ValueError):
                probe.credit.validate_score(checked["score"], sample)
        score = row["samples"][0]["score"]
        self.assertEqual(sum(score["assessment_rewards"].values()), score["native_assessment_count"])
        self.assertEqual(row["samples"][1]["score"]["assessment_rewards"][normal[0]], 0)
        self.assertTrue(all(not any(value["score"]["assessment_rewards"].values()) for value in row["samples"][2:]))
        actual = scorer.call_args_list[4].kwargs["observed_sample"]
        audit = probe.synthesis.audit_episode(f.tree, f.actor, f.verifier, contract=probe.synthesis.BOUNDED_VERSION)
        legacy = probe._checked(actual, f.tree, audit, f.actor, f.verifier)
        self.assertEqual(legacy["strict_check"], row["samples"][0]["strict_check"])
        self.assertEqual(legacy["score"]["fact_rewards"], score["fact_rewards"])
        self.assertEqual(legacy["score"]["relationship_rewards"], score["relationship_rewards"])
        self.assertAlmostEqual(score["atomic_reward"] - legacy["score"]["atomic_reward"], probe.credit.ASSESSMENT_WEIGHT)

    def test_undrafted_or_empty_draft_history_cannot_certify_original_pixel_tensors(self):
        for drafted in (False, True):
            with self.subTest(drafted=drafted):
                f, actors, trees, good = self.fixture()
                f.tree.groups = [group for group in f.tree.groups if group.kind != "DIAG"]
                f.tree.notebook = {}
                f.protocol = build_protocol(verified_note_synthesis_credit=True,
                    pair_draft_synthesis=drafted, context_limit=24576)
                f.prompts = local_prompts(f.protocol)
                f.cfg = RolloutConfig.from_protocol(f.protocol, "ALG-NB3", master_seed=3407)
                f.engine.policy_version = 0
                with patch("fine_tuning.notebook_rl.notebook_pair_drafts._plan", return_value=([], None)):
                    f.append([[good] * 4])
                    if drafted:
                        saved = f.tree.notebook["rewarded_notebook"]["pair_drafts"]
                        self.assertEqual((saved["status"], saved["calls"]), ("NO_PUBLIC_TARGETS", []))
                    f.engine.requests.clear()
                    f.engine.policy_version = 1
                    prepare = f.engine.prepare

                    def changed_pixels(request):
                        prepared = prepare(request)
                        prepared.pixels["pixel_values"] += 1
                        return prepared

                    f.engine.prepare = changed_pixels
                    expected = "historical draft pixel identities required" if drafted else "draft-assisted"
                    with self.assertRaisesRegex(ValueError, expected):
                        self.collect(f, actors, trees, [good] * 4)
                    self.assertEqual(f.engine.requests, [])

    def test_same_original_prompt_drafts_images_seeds_and_fresh_strict_units_are_compared(self):
        f, actors, trees, good = self.fixture()
        originals = [to_json(tree) for tree in trees]
        note = f.tree.notebook["rewarded_notebook"]
        baseline_text = f.tree.groups[-1].samples[0].text
        rows, skipped, events = self.collect(f, actors, trees, [good, baseline_text, "not JSON", (good, "length")])
        self.assertEqual(skipped, [{"problem_id": actors[0].problem_id, "reason": "NO_VERIFIED_RECOVERY"}])
        self.assertEqual([to_json(tree) for tree in trees], originals)
        self.assertEqual(len(f.engine.requests), 1)
        request = f.engine.requests[0]
        self.assertEqual(request.messages, note["notebook_messages"])
        self.assertIs(request.images, f.images[f.actor.problem_id])
        self.assertEqual(request.seeds, [sample.seed for sample in f.tree.groups[-1].samples])
        self.assertEqual((request.n, request.max_new_tokens), (4, 8192))
        self.assertEqual([event[0] for event in events], ["CALL_STARTED", "CALL_GENERATED", "CALL_COMPLETED"])
        started = events[0][1]
        self.assertEqual((started["history_policy_version"], started["generation_policy_version"]), (0, 1))
        self.assertEqual(started["prompt_token_ids"], f.tree.groups[-1].samples[0].prompt_token_ids)
        self.assertEqual(len(started["baseline_replay"]["samples"]), 4)
        row = rows[0]
        self.assertEqual(row["before_totals"]["strict_verified_notes"], 0)
        self.assertEqual(row["after_totals"]["strict_verified_notes"], 1)
        self.assertGreater(row["after_totals"]["verified_relationship_units"]["NATIVE_ERROR_RELATIONSHIP"], 0)
        self.assertEqual(row["optimizer_updates"], 0)
        self.assertTrue(all(not sample["eligible_for_rl_loss"] and not sample["used_for_routing"]
                            for sample in row["samples"]))
        self.assertEqual(capture.trained_note_counts(rows)["recovered"], 1)
        for sample in row["samples"]:
            self.assertEqual(sample["strict_check"]["note_text_sha256"], probe.synthesis._sha(sample["text"]))
            self.assertEqual(sample["score"]["raw_text_sha256"], probe.synthesis._sha(sample["text"]))

    def test_historical_seed_prompt_text_and_draft_tampering_fail_before_generation(self):
        for mutation in ("seed", "messages", "text", "draft", "pixels"):
            with self.subTest(mutation=mutation):
                f, actors, trees, good = self.fixture()
                if mutation == "seed":
                    f.tree.groups[-1].samples[0].seed += 1
                elif mutation == "messages":
                    f.tree.notebook["rewarded_notebook"]["notebook_messages"][0]["content"][-1]["text"] += "changed"
                elif mutation == "text":
                    f.tree.groups[0].samples[0].text += " changed"
                elif mutation == "draft":
                    f.tree.notebook["rewarded_notebook"]["pair_drafts"]["calls"][0]["completions"][0]["text"] += "changed"
                else:
                    prepare = f.engine.prepare

                    def changed_pixels(request):
                        prepared = prepare(request)
                        prepared.pixels["pixel_values"] += 1
                        return prepared

                    f.engine.prepare = changed_pixels
                with self.assertRaises(ValueError):
                    self.collect(f, actors, trees, [good] * 4)
                self.assertEqual(f.engine.requests, [])

    def test_generated_token_binding_context_policy_and_deadline_fail_closed(self):
        for mutation in ("decode", "prompt", "n", "context", "policy", "history", "deadline"):
            with self.subTest(mutation=mutation):
                f, actors, trees, good = self.fixture()
                events = []
                sample = f.engine.sample

                def changed(requests):
                    output = sample(requests)
                    if mutation == "decode":
                        value = output[0].completions[0]
                        output[0].completions[0] = Completion(value.token_ids, value.logprobs,
                            value.finish_reason, value.text + " changed")
                    elif mutation == "prompt":
                        output[0] = replace(output[0], prompt_token_ids=[123])
                    elif mutation == "n":
                        output[0].completions.pop()
                    elif mutation == "policy":
                        f.engine.policy_version = 2
                    elif mutation == "history":
                        f.tree.notebook["changed"] = True
                    elif mutation == "deadline":
                        clock.return_value = 100
                    return output

                policy = probe.contract(f.protocol)
                if mutation == "context":
                    f.engine.context_limit = len(f.tree.groups[-1].samples[0].prompt_token_ids) + 8192 - 1
                    policy["context_limit"] = f.engine.context_limit
                with patch.object(probe, "contract", return_value=policy), \
                        patch.object(f.engine, "sample", side_effect=changed), \
                        patch.object(probe.time, "monotonic", return_value=0) as clock:
                    with self.assertRaises(TimeoutError if mutation == "deadline" else ValueError):
                        self.collect(f, actors, trees, [good] * 4, deadline=60,
                                     record=lambda *row: events.append(row))
                if mutation != "context":
                    self.assertEqual(events[1][0], "CALL_GENERATED")
                self.assertFalse(any(event[0] == "CALL_COMPLETED" for event in events))

    def test_capture_dispatch_loads_verified_policy_and_keeps_existing_lifecycle_guards(self):
        f, actors, trees, good = self.fixture()
        expected_hashes = {"L": {"l": "l"}, "M": {"m": "m"}, "V": {"v": "v"}}
        for assessment, mutation in ((False, None), (False, "weights"), (False, "source"), (False, "parent"), (True, None)):
            if assessment:
                f, actors, trees, good = self.fixture(assessment=True)
            with self.subTest(assessment=assessment, mutation=mutation), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                root = Path(temporary)
                data, parent = root / "data", root / "parent"
                model, cache, reload, output = (root / name for name in ("model", "cache", "reload", "output"))
                capture.save(data / "manifest.yaml", {"split_manifests": {"train": "vault"}})
                capture.save(parent / "reload_expected.json", {"trainable_sha256": expected_hashes,
                    "step": 1, "frozen_base_sha256": "base", "quantization": {"kind": "fixture-nf4"}})
                parent_file = parent / "bound.json"
                capture.save(parent_file, {"unchanged": True})
                f.verifier.metadata = {"vault_manifest_sha256": "vault"}
                bundle = SimpleNamespace(base_sha256="base", inventory={"quantization": {"kind": "fixture-nf4"}})

                def mocked(target, **kwargs):
                    return stack.enter_context(patch(target, **kwargs))

                stack.enter_context(patch.dict(os.environ, MMDL_FACT_PROBE_SUPERVISOR=str(os.getppid())))
                mocked("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={})
                mocked("fine_tuning.notebook_rl.fact_context_probe.trained_parent", return_value={
                    "launch": {"protocol": f.protocol, "binding": {"history": "original"}}, "files": []})
                mocked("fine_tuning.notebook_rl.local_run.cache_readiness",
                    return_value={"original_model_root": str(root / "original")})
                mocked("fine_tuning.notebook_rl.local_run.read_preparation",
                    return_value=({"build_root": str(data)}, {}, actors))
                mocked("fine_tuning.notebook_rl.local_run.minimal_actors", return_value=actors)
                mocked("fine_tuning.notebook_rl.fact_context_probe.closed_parent", return_value=(trees,
                    {"files": {str(parent_file): checkpoint.file_hash(parent_file)}}))
                mocked("fine_tuning.notebook_rl.approval.code_hashes", side_effect=[{"source": "before"},
                    {"source": "after" if mutation == "source" else "before"}])
                mocked("fine_tuning.notebook_rl.local_run._install_actor_guards", return_value=[])
                mocked("fine_tuning.notebook_rl.quantized_cache.block_original_access",
                    return_value={"denied_original_access_attempts": 0})
                mocked("fine_tuning.notebook_rl.verifier_service.VerifierService", return_value=nullcontext(f.verifier))
                load = mocked("fine_tuning.notebook_rl.checkpoint.load_eval", return_value=bundle)
                mocked("fine_tuning.notebook_rl.local_run.parameter_hashes",
                    return_value={} if mutation == "weights" else expected_hashes)
                engine = mocked("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine", return_value=f.engine)
                mocked("fine_tuning.notebook_rl.dataset.load_images", return_value=f.images[f.actor.problem_id])
                mocked("fine_tuning.notebook_rl.local_run.make_probes", return_value=[])
                mocked("fine_tuning.notebook_rl.fact_context_probe.bind_history_tokens")
                mocked("fine_tuning.notebook_rl.trainer.synchronize", return_value={})
                mocked("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged")
                original_sample = f.engine.sample

                def generate(requests):
                    if mutation == "parent":
                        parent_file.write_text("changed")
                    return original_sample(requests)

                mocked_legacy = mocked("fine_tuning.notebook_rl.notebook_layout_probe.collect")
                f.engine.replies = [[good] * 4]
                f.engine.requests.clear()
                with patch.object(f.engine, "sample", side_effect=generate):
                    if mutation:
                        with self.assertRaises(ValueError):
                            capture.capture(root / "prep", model, cache, parent, output, trained_notebook_reload=reload)
                        self.assertEqual(capture.read(output / "result.json")["status"], "FAILED")
                    else:
                        result = capture.capture(root / "prep", model, cache, parent, output, trained_notebook_reload=reload)
                        self.assertEqual(result["same_history_note_comparison"]["after_strict_valid"], 4)
                        self.assertEqual((result["history_policy_version"], result["generation_policy_version"]), (0, 1))
                        self.assertEqual((result["calls"], result["samples"], result["optimizer_updates"]), (1, 4, 0))
                        self.assertFalse(result["heldout"])
                        self.assertEqual(engine.call_args.args[2], 24576)
                        self.assertEqual(engine.call_args.kwargs["policy_version"], 1)
                        self.assertEqual(capture.read(output / "launch.json")["diagnostic"], probe.contract(f.protocol))
                load.assert_called_once()
                mocked_legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
