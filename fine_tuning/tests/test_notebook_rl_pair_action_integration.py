"""Authored raw-label CPU plumbing; no real-model learning claim."""
import json
from pathlib import Path
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, local_profile as profile, local_run, local_train
from fine_tuning.notebook_rl import notebook_pair_probe as probe, pair_action_notebook as action, synthesis_credit, updates
from fine_tuning.notebook_rl.checkpoint import digest
from fine_tuning.notebook_rl.engines import RolloutRequest
from fine_tuning.notebook_rl.rollout import RolloutConfig
from fine_tuning.notebook_rl.schema import ProblemTree, to_json
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration
from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleTokenizer


def pair_action_protocol():
    return profile.build_protocol(verified_note_synthesis_credit=True, pair_action_notebook=True, context_limit=24576)


def pair_action_fixture(testcase, replies=None, tokenizer=None, *, with_notebooks=True, all_targets=False):
    fixture = integration.SynthesisIntegrationTests()
    fixture.setUp()
    testcase.addCleanup(fixture.doCleanups)
    fixture.protocol = pair_action_protocol()
    fixture.prompts = profile.local_prompts(fixture.protocol)
    fixture.cfg = RolloutConfig.from_protocol(fixture.protocol, "ALG-NB3", master_seed=3407)
    if tokenizer is not None:
        fixture.engine.tokenizer = tokenizer
    samples = episode.solution_samples(fixture.tree)
    for index, sample in enumerate(samples):
        if index < 4:
            sample.text = sample.text.replace("start = q0", "start = q999")
        if index == 1:
            sample.text = sample.text.replace("accepting = q3", "accepting = q999")
        if index == 0:
            sample.text = sample.text.rsplit("\nFinal answer:", 1)[0]
        sample.completion_token_ids = fixture.engine.tokenizer.encode(sample.text) + [2]
        sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
        sample.result = fixture.verifier.grade(sample.problem_id, sample.text, sample.finish_reason, fixture.actor.mode)
        sample.reward = float(sample.result.status == "CORRECT")
        if sample.parent_sample_id is not None:
            parent = next(value for value in samples if value.sample_id == sample.parent_sample_id)
            sample.prompt_token_ids = parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2]
    original_prepare = fixture.engine.prepare
    input_tokenizer = integration.ChunkTokenizer()

    def prepare(request):
        prepared = original_prepare(request)
        if tokenizer is not None and request.prompt_token_ids is None:
            prepared.token_ids = input_tokenizer.encode(json.dumps(request.messages, ensure_ascii=False, sort_keys=True))
        prepared.pixels = {"pixel_values": torch.zeros((1, 2))}
        prepared.images, prepared.image_spans = request.images, ()
        return prepared

    fixture.engine.prepare = prepare
    if not with_notebooks:
        return fixture
    observed = {"completion": {"text": "Observed facts:\n" + "\n".join(
        f"{key} = {value}" for key, value in fixture.item.slots.items()) + "\nSolution:"}}
    targets = probe.select_targets(samples, observed)
    replies = replies if replies is not None else [
        ["UNRESOLVED"] * 4, ["PERSISTED", "UNRESOLVED"] * 2, ["FINALIZED", "CORRECTED"] * 2]
    scripted = dict(zip((target["pair_id"] for target in targets), replies, strict=True))
    plan, _ = action._plan(samples, observed)
    count = len(plan) if all_targets else max(index for index, target in enumerate(plan, 1)
                                           if target["pair_id"] in scripted)
    fixture.engine.replies = [[observed["completion"]["text"]],
        *[scripted.get(target["pair_id"], ["UNRESOLVED"] * 4) for target in plan]]
    action.append([fixture.tree], fixture.images, fixture.engine, fixture.verifier,
        fixture.prompts, fixture.cfg, actors={fixture.actor.problem_id: fixture.actor},
        should_stop=lambda: sum(group.kind == "DIAG" for group in fixture.tree.groups) >= count)
    return fixture


def scripted_groups(fixture):
    record = fixture.tree.notebook["rewarded_notebook"]
    return [group for group in fixture.tree.groups if group.kind == "DIAG"
            and (target := record["target_per_group"][group.group_id]["target"])["earlier"]["attempt"] == 1
            and target["later"]["attempt"] == 3 and target["earlier"]["locus"] in ("accepting", "start", "final_answer")]


class PairActionIntegrationTests(unittest.TestCase):
    def test_character_label_fixture_inputs_reconstruct_and_distinguish_prompt_content(self):
        fixture = pair_action_fixture(self, tokenizer=ReversibleTokenizer(), with_notebooks=False)

        def prepare(text):
            request = RolloutRequest("fixture:input", None, [{"role": "user", "content": text}],
                fixture.images[fixture.actor.problem_id], 4, 128, [1, 2, 3, 4])
            return fixture.engine.prepare(request).token_ids

        first = prepare("all original history; target one")
        self.assertNotEqual(first, prepare("all original history; target two"))
        self.assertEqual(first, prepare("all original history; target one"))
        self.assertEqual(fixture.engine.tokenizer.encode(" PERSISTED "), [ord(char) + 100 for char in " PERSISTED "])
        self.assertEqual(fixture.engine.requests, [])

    def test_explicit_profile_roundtrip_preserves_solve_budget_with_versioned_verifier(self):
        previous = profile.build_protocol(verified_note_synthesis_credit=True, paired_notebook=True, context_limit=24576)
        self.assertEqual(previous["verifier_version"], profile.VERIFIER_VERSION)
        legacy_shape = {k: v for k, v in previous.items() if k not in ("sha256", "verifier_version")}
        self.assertEqual(digest(legacy_shape), "d78bcbb5985414058ba2cc646191606849d5fc3baa947d734b67b2d6b4a0599b")
        current = pair_action_protocol()
        self.assertEqual(current["protocol_id"], profile.PAIR_ACTION_NOTEBOOK_PROTOCOL_ID)
        self.assertEqual(profile.restore_protocol(json.loads(json.dumps(current))), current)
        self.assertEqual(current["notebook_generation_policy"], action.VERSION)
        self.assertEqual(current["notebook_generation_contract"], action.contract())
        self.assertEqual(current["notebook_prompt_sha256"], digest(action.prompt_contract()))
        self.assertEqual(current["notebook_objective"], previous["notebook_objective"])
        self.assertEqual(current["rollout"]["n"], previous["rollout"]["n"])
        self.assertEqual(current["rollout"]["max_new_tokens"], previous["rollout"]["max_new_tokens"])
        self.assertEqual((current["rollout"]["diag"]["n"], current["rollout"]["diag"]["max_new_tokens"]), (4, 128))
        prompts = profile.local_prompts(current)
        self.assertEqual(prompts["notebook_generation_contract"], action.contract())
        self.assertEqual(prompts["rewarded_notebook"], action.prompt_contract())
        self.assertEqual(local_train.execution_admission({}, current), local_train.STRICT_ADMISSION)
        self.assertEqual(local_train.execution_admission(
            {"admission_policy": local_train.SYNTHESIS_CREDIT_ADMISSION}, current), local_train.SYNTHESIS_CREDIT_ADMISSION)
        with self.assertRaises(ValueError):
            local_run.minimum_execution_contract(local_run.VERIFIED_UPDATE_EXPLORATION, current)
        for key, value in (("notebook_generation_contract", {}), ("notebook_max_rounds", True)):
            changed = current | {key: value}
            changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
            with self.assertRaises(ValueError):
                profile.restore_protocol(changed)

    def test_flag_requires_exclusive_boolean_synthesis_credit_and_24k(self):
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576,
                                       pair_action_notebook=cast(Any, value))
        for kwargs in ({}, {"verified_note_synthesis": True}, {"verified_note_synthesis_credit": True},
                {"verified_note_synthesis_credit": True, "context_limit": 24576, "partial_notebook": True},
                {"verified_note_synthesis_credit": True, "context_limit": 24576, "referenced_notebook": True},
                {"verified_note_synthesis_credit": True, "context_limit": 24576, "paired_notebook": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                profile.build_protocol(pair_action_notebook=True, **kwargs)

    def test_prepare_minimal_and_both_cache_clis_forward_explicit_policy(self):
        from fine_tuning.notebook_rl import quantized_cache

        flags = ["--pair-action-notebook", "--verified-note-synthesis-credit", "--context-limit", "24576"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as prepared, patch("builtins.print"):
            self.assertEqual(local_run.main(["prepare", "--previous-request", "/request", "--selection", "/selection",
                                           "--output", "/output", *flags]), 0)
            self.assertEqual(prepared.call_args.args[2], pair_action_protocol())
        with patch.object(local_run, "capture") as capture:
            self.assertEqual(local_run.main(["minimal", "--preparation", "/prep", "--model-path", "/model",
                                            "--output", "/output", *flags]), 0)
            self.assertIs(capture.call_args.kwargs["pair_action_notebook"], True)
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
                    self.assertEqual(quantized_cache.main([stage, "--model-path", "/model", "--cache", "/cache",
                        "--parent", "/parent", "--output", str(Path(directory) / stage), *flags]), 1)
                    self.assertIs(build.call_args.kwargs["pair_action_notebook"], True)
                    save.assert_not_called()
                    load.assert_not_called()
        for old in ("--partial-notebook", "--referenced-notebook", "--paired-notebook"):
            with self.subTest(old=old), patch("sys.stderr"), self.assertRaises(SystemExit):
                local_run.main(["minimal", "--preparation", "/prep", "--model-path", "/model",
                                "--output", "/output", *flags, old])

    def test_early_dispatch_bypasses_legacy_round_rules_and_propagates_errors(self):
        tree = ProblemTree("dispatch", 0, [], "VISION_OFF", {},
                           {"rewarded_notebook": {"notebook_generation_policy": action.VERSION}})
        with patch.object(action, "validate_update_binding", side_effect=ValueError("action binding")) as validate:
            with self.assertRaisesRegex(ValueError, "action binding"):
                episode.validate_update_binding(tree)
            validate.assert_called_once()
        with patch.object(action, "append", side_effect=ValueError("action generation")) as append:
            with self.assertRaisesRegex(ValueError, "action generation"):
                episode.append_episode_notebooks([], {}, None, None,
                    {"notebook_generation_policy": action.VERSION}, None, actors={})
            append.assert_called_once()

    def test_raw_labels_regrade_strict_zero_and_pass_sample_id_to_credit(self):
        fixture = pair_action_fixture(self)
        groups = [group for group in fixture.tree.groups if group.kind == "DIAG"]
        self.assertLess(len(groups), len(action._plan(episode.solution_samples(fixture.tree), None)[0]))
        self.assertEqual(fixture.tree.notebook["rewarded_notebook"]["stop_reason"], "BUDGET_STOP")
        self.assertTrue(all(len(group.samples) == 4 for group in groups))
        self.assertEqual([sample.text for group in scripted_groups(fixture) for sample in group.samples],
                         ["UNRESOLVED"] * 4 + ["PERSISTED", "UNRESOLVED"] * 2 + ["FINALIZED", "CORRECTED"] * 2)
        self.assertTrue(all(sample.reward == 0 and sample.result is None for group in groups for sample in group.samples))
        receipt = episode.regrade_synthesis(fixture.tree, fixture.actor, fixture.verifier, fixture.engine.tokenizer)
        self.assertEqual(len(receipt["samples"]), 4 * len(groups))
        bound = updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier,
                                                fixture.protocol["notebook_objective"])
        self.assertEqual(set(bound["samples"]), {sample.sample_id for group in groups for sample in group.samples})
        audit = fixture.tree.notebook["rewarded_notebook"]["synthesis_audit"]
        for group in scripted_groups(fixture):
            sample = group.samples[0]
            individual = synthesis_credit.score_note(sample.text, sample.finish_reason, fixture.tree,
                audit, fixture.verifier, actor=fixture.actor, sample_id=sample.sample_id)
            self.assertEqual(bound["samples"][sample.sample_id], individual)
        self.assertTrue(local_run.error_relationship_signal([fixture.tree], fixture.protocol["notebook_objective"], fixture.engine.tokenizer))
        old = integration.SynthesisIntegrationTests()
        old.setUp()
        self.addCleanup(old.doCleanups)
        old.protocol = profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        old.prompts = profile.local_prompts(old.protocol)
        old.append([["{}"] * 4] * 3)
        with patch.object(synthesis_credit, "score_note", wraps=synthesis_credit.score_note) as score:
            updates.bind_notebook_objective(old.tree, old.actor, old.verifier, old.protocol["notebook_objective"])
        self.assertTrue(all("sample_id" not in call.kwargs for call in score.call_args_list))

    def test_real_tiny_label_gradient_passes_fixed_ten_gate_without_strict_success(self):
        fixture = pair_action_fixture(self, [["UNRESOLVED"] * 4, ["PERSISTED"] * 3 + ["UNRESOLVED"],
                                             ["FINALIZED"] * 4])
        objective = fixture.protocol["notebook_objective"]
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier, objective)
        result = fixture.step(fixture.optimizer_fixture())
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertTrue(result["retained_learning_signal"]["policy_gradient_present"])
        summary_types = local_run.relationship_learning_summary(result)
        self.assertTrue(summary_types["native_relationship_only_gradient_verified"])
        self.assertIn("NATIVE_ERROR_RELATIONSHIP", summary_types["positive_verified_relationship_credit_types"])
        scores = updates.notebook_objective_scores(fixture.tree, objective)
        summary = {"notebook_signal_groups": sum(group.kind == "DIAG" and updates.group_has_signal(group, scores)
                                                 for group in fixture.tree.groups)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw-0000.json"
            path.write_text(json.dumps({"kind": "TREE", "raw": json.loads(to_json(fixture.tree))}))
            self.assertEqual(local_train.fact_credit_learning_gate(root, summary, result, fixture.protocol,
                actors={fixture.actor.problem_id: fixture.actor}, verifier=fixture.verifier,
                tokenizer=fixture.engine.tokenizer), [path])

    def test_uniform_verified_labels_stay_signal_free_without_forced_update(self):
        fixture = pair_action_fixture(self, [["UNRESOLVED"] * 4, ["PERSISTED"] * 4, ["FINALIZED"] * 4])
        objective = fixture.protocol["notebook_objective"]
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier, objective)
        self.assertFalse(local_run.error_relationship_signal([fixture.tree], objective, fixture.engine.tokenizer))
        selected, unresolved, signal_free = updates.partition_learning_trees(
            [fixture.tree], include_notebooks=True, notebook_objective=objective)
        self.assertEqual((selected, unresolved, signal_free), ([], [], [fixture.tree.problem_id]))

    def test_reload_and_native_eval_restore_explicit_profile_before_weight_access(self):
        from fine_tuning.notebook_rl import local_eval

        protocol = pair_action_protocol()
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
            execution = {"admission_policy": local_train.SYNTHESIS_CREDIT_ADMISSION}
            launch.update(execution_contract=execution, execution_contract_sha256=digest(execution))
            (root / "launch.json").write_text(json.dumps(launch))
            with patch.object(local_train, "training_parent_gate", side_effect=ValueError("CPU evaluation boundary")) as gate, \
                    patch("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy") as loader:
                with self.assertRaisesRegex(ValueError, "CPU evaluation boundary"):
                    local_eval.evaluate_ladder(root, root / "cohort", root / "gate", root / "eval",
                        condition="F0", arm="notebook", reload_result=root / "reload.json",
                        minimal_capture=root / "minimal", minimal_reload=root / "minimal-reload")
                self.assertEqual(gate.call_args.kwargs["admission"], local_train.SYNTHESIS_CREDIT_ADMISSION)
                loader.assert_not_called()

    def test_masked_label_signal_cannot_commit_a_zero_policy_gradient_update(self):
        fixture = pair_action_fixture(self)
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier, fixture.protocol["notebook_objective"])
        bundle, optimizer, scheduler, preparation, logprobs, reference = fixture.optimizer_fixture()

        def mismatched(bundle, inputs, completion_len, *, chunk):
            values = logprobs(bundle, inputs, completion_len, chunk=chunk)
            return values + (.1 if inputs["sample"].kind == "DIAG" else 0)

        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = fixture.step((bundle, optimizer, scheduler, preparation, mismatched, reference))
        self.assertEqual(result["status"], "NO_RETAINED_NATIVE_RELATIONSHIP_SIGNAL")
        step.assert_not_called()
        self.assertEqual((scheduler.last_epoch, optimizer.state), (0, {}))

    def test_output_only_variance_cannot_commit_without_native_relationship_credit(self):
        fixture = pair_action_fixture(self, [["UNRESOLVED"] * 4, ["PERSISTED"] * 4,
                                             ["FINALIZED"] * 3 + ["CORRECTED"]])
        objective = fixture.protocol["notebook_objective"]
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier, objective)
        self.assertTrue(local_run.error_relationship_signal([fixture.tree], objective, fixture.engine.tokenizer))
        self.assertFalse(local_run.error_relationship_signal(
            [fixture.tree], objective, fixture.engine.tokenizer, native_only=True))
        values = fixture.optimizer_fixture()
        before = {key: value.detach().clone() for key, value in values[0].model.items()}
        with patch.object(values[1], "step", wraps=values[1].step) as step:
            result = fixture.step(values)
        self.assertEqual(result["status"], "NO_RETAINED_NATIVE_RELATIONSHIP_SIGNAL")
        self.assertTrue(result["retained_learning_signal"]["policy_gradient_present"])
        self.assertFalse(result["retained_native_relationship_signal"]["policy_gradient_present"])
        step.assert_not_called()
        self.assertEqual((values[2].last_epoch, values[1].state), (0, {}))
        self.assertTrue(all(torch.equal(before[key], value.detach()) for key, value in values[0].model.items()))

    def test_target_tampering_during_backward_fails_fresh_regrade_before_commit(self):
        fixture = pair_action_fixture(self)
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier, fixture.protocol["notebook_objective"])
        values = fixture.optimizer_fixture()
        before = {key: value.detach().clone() for key, value in values[0].model.items()}
        tampered = False

        def progress(event):
            nonlocal tampered
            if event["stage"] == "backward" and event["state"] == "RETURN" and not tampered:
                fixture.tree.notebook["rewarded_notebook"]["target_plan"][0]["earlier"]["attempt"] = 99
                tampered = True

        with patch.object(values[1], "step", wraps=values[1].step) as step, self.assertRaises(ValueError):
            fixture.step(values, on_progress=progress)
        self.assertTrue(tampered)
        step.assert_not_called()
        self.assertEqual((values[2].last_epoch, values[1].state), (0, {}))
        self.assertTrue(all(torch.equal(before[key], value.detach()) for key, value in values[0].model.items()))
        self.assertTrue(all(value.grad is None for value in values[0].model.values()))


if __name__ == "__main__":
    unittest.main()
