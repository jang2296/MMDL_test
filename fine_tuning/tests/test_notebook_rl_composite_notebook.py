"""Authored compositional CPU fixtures; drafts are never the rewarded notebook."""
import copy
import json
from pathlib import Path
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, local_profile as profile, local_run, local_train
from fine_tuning.notebook_rl import notebook_pair_drafts as drafts, notebook_synthesis as synthesis, updates
from fine_tuning.notebook_rl.checkpoint import digest
from fine_tuning.notebook_rl.notebook_learning import append_rewarded_notebooks
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash
from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration


def composite_protocol():
    return profile.build_protocol(verified_note_synthesis_credit=True, pair_draft_synthesis=True, context_limit=24576)


class CompositeNotebookTests(unittest.TestCase):
    def fixture(self, *, final=None, blocked=None, stop_after_drafts=False, empty_plan=False):
        fixture = integration.SynthesisIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol = composite_protocol()
        fixture.prompts = profile.local_prompts(fixture.protocol)
        fixture.cfg = RolloutConfig.from_protocol(fixture.protocol, "ALG-NB3", master_seed=3407)
        fixture.engine.blocked = blocked
        original_prepare = fixture.engine.prepare

        def prepare(request):
            prepared = original_prepare(request)
            prepared.pixels = {"pixel_values": torch.zeros((1, 2))}
            prepared.images, prepared.image_spans = request.images, ()
            return prepared

        fixture.engine.prepare = prepare
        observation = "Observed facts:\n" + "\n".join(
            f"{key} = {value}" for key, value in fixture.item.slots.items()) + "\nSolution:"
        fixture.raw_drafts = ["CORRECTED", "{broken", ("PERSISTED", "length"), "UNRESOLVED"]
        fixture.engine.replies = [[observation], *([] if empty_plan else [fixture.raw_drafts]),
                                  [fixture.good, "{}", fixture.good, "{}"] if final is None else final]
        fixture.draft_events, fixture.observation_events, fixture.generated_groups = [], [], []
        stopped = False

        def on_draft(event):
            nonlocal stopped
            fixture.draft_events.append(event)
            if event["phase"] == "DRAFTS_COMPLETED" and stop_after_drafts:
                stopped = True

        before = [to_json(sample) for sample in episode.solution_samples(fixture.tree)]
        fixture.report = append_rewarded_notebooks([fixture.tree], fixture.images, fixture.engine, fixture.verifier,
            fixture.prompts, fixture.cfg, actors={fixture.actor.problem_id: fixture.actor},
            on_group=fixture.generated_groups.append, on_observation=fixture.observation_events.append,
            on_draft=on_draft, should_stop=lambda: stopped)
        self.assertEqual(before, [to_json(sample) for sample in episode.solution_samples(fixture.tree)])
        return fixture

    def test_explicit_profile_restores_with_versioned_verifier_and_same_solve_budget(self):
        base = profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        self.assertEqual(base["verifier_version"], profile.VERIFIER_VERSION)
        legacy_shape = {k: v for k, v in base.items() if k not in ("sha256", "verifier_version")}
        self.assertEqual(digest(legacy_shape), "1b40a7e4169e0c94da8a88b78d98bcd67092fd1404b3004066382d779d233c91")
        current = composite_protocol()
        self.assertEqual(profile.restore_protocol(json.loads(json.dumps(current))), current)
        self.assertEqual(current["protocol_id"], profile.PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID)
        self.assertNotIn("notebook_generation_policy", current)
        self.assertEqual(current["notebook_draft_policy"], drafts.VERSION)
        self.assertEqual(current["notebook_draft_contract"], drafts.contract())
        self.assertEqual(current["notebook_draft_contract_sha256"], digest(drafts.contract()))
        self.assertEqual(current["notebook_prompt_sha256"], base["notebook_prompt_sha256"])
        self.assertEqual(current["rollout"], base["rollout"])
        self.assertEqual((current["notebook_max_rounds"], current["rollout"]["diag"]["n"],
                          current["rollout"]["diag"]["max_new_tokens"]), (1, 4, 8192))
        self.assertEqual(current["notebook_objective"], base["notebook_objective"])
        self.assertEqual(local_train.execution_admission(
            {"admission_policy": local_train.SYNTHESIS_CREDIT_ADMISSION}, current), local_train.SYNTHESIS_CREDIT_ADMISSION)
        with self.assertRaises(ValueError):
            local_run.minimum_execution_contract(local_run.VERIFIED_UPDATE_EXPLORATION, current)
        for key, value in (("notebook_draft_contract", {}), ("notebook_draft_contract_sha256", "changed"),
                           ("notebook_max_rounds", True), ("notebook_strict_feedback_repairs", True)):
            changed = current | {key: value}
            changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                profile.restore_protocol(changed)

    def test_flag_requires_exclusive_boolean_synthesis_credit_and_24k(self):
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576,
                                       pair_draft_synthesis=cast(Any, value))
        choices: tuple[dict[str, Any], ...] = ({}, {"verified_note_synthesis": True}, {"verified_note_synthesis_credit": True},
                *({"verified_note_synthesis_credit": True, "context_limit": 24576, key: True} for key in
                  ("partial_notebook", "referenced_notebook", "paired_notebook", "pair_action_notebook")))
        for kwargs in choices:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                profile.build_protocol(pair_draft_synthesis=True, **kwargs)

    def test_prepare_minimal_and_cache_clis_forward_distinct_profile(self):
        from fine_tuning.notebook_rl import quantized_cache

        flags = ["--pair-draft-synthesis", "--verified-note-synthesis-credit", "--context-limit", "24576"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as prepared, patch("builtins.print"):
            self.assertEqual(local_run.main(["prepare", "--previous-request", "/request", "--selection", "/selection",
                                           "--output", "/output", *flags]), 0)
            self.assertEqual(prepared.call_args.args[2], composite_protocol())
        with patch.object(local_run, "capture") as capture:
            self.assertEqual(local_run.main(["minimal", "--preparation", "/prep", "--model-path", "/model",
                                            "--output", "/output", *flags]), 0)
            self.assertIs(capture.call_args.kwargs["pair_draft_synthesis"], True)
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
                    self.assertIs(build.call_args.kwargs["pair_draft_synthesis"], True)
                    save.assert_not_called()
                    load.assert_not_called()
        for old in ("--partial-notebook", "--referenced-notebook", "--paired-notebook", "--pair-action-notebook"):
            with self.subTest(old=old), patch("sys.stderr"), self.assertRaises(SystemExit):
                local_run.main(["minimal", "--preparation", "/prep", "--model-path", "/model",
                                "--output", "/output", *flags, old])

    def test_all_history_success_and_every_raw_draft_are_preserved_for_one_whole_note(self):
        fixture = self.fixture()
        record = fixture.tree.notebook["rewarded_notebook"]
        proposal = record["pair_drafts"]
        self.assertEqual(proposal["status"], "COMPLETE")
        self.assertEqual([row["text"] for row in proposal["calls"][0]["completions"]],
                         ["CORRECTED", "{broken", "PERSISTED", "UNRESOLVED"])
        self.assertEqual([row["finish_reason"] for row in proposal["calls"][0]["completions"]],
                         ["stop", "stop", "length", "stop"])
        self.assertFalse(proposal["eligible_for_rl_loss"])
        self.assertEqual(fixture.draft_events[-1]["phase"], "DRAFTS_COMPLETED")
        self.assertEqual(fixture.draft_events[-1]["record"], proposal)
        self.assertTrue(all("DRAFT" not in event["phase"] for event in fixture.observation_events))
        self.assertEqual((fixture.report["generated_groups"], len(fixture.generated_groups), len(record["rounds"])), (1, 1, 1))
        self.assertEqual(len(fixture.engine.requests), 3)
        final = fixture.engine.requests[-1]
        self.assertEqual((final.n, final.max_new_tokens), (4, 8192))
        self.assertIs(final.images, fixture.images[fixture.actor.problem_id])
        body = final.messages[0]["content"][-1]["text"]
        self.assertIn(drafts.quoted_context(proposal), body)
        self.assertIn(synthesis.instruction(episode.solution_samples(fixture.tree), record["synthesis_audit"], record["notebook_schema"]), body)
        transcript = json.loads(body.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
        self.assertEqual(len(transcript), 6)
        for row, sample in zip(transcript, episode.solution_samples(fixture.tree), strict=True):
            self.assertEqual("".join(line for _, line in row["source_lines"]), sample.text)
        group = fixture.tree.groups[-1]
        self.assertEqual([sample.text for sample in group.samples], [fixture.good, "{}", fixture.good, "{}"])
        self.assertEqual([sample.reward for sample in group.samples], [1., 0., 1., 0.])
        self.assertEqual(len(fixture.tree.notebook["model_notes"]), 4)
        restored = from_json(ProblemTree, to_json(fixture.tree))
        episode.regrade_synthesis(restored, fixture.actor, fixture.verifier, fixture.engine.tokenizer)

    def test_only_final_sample_tokens_enter_existing_credit_and_optimizer(self):
        fixture = self.fixture()
        final_ids = {sample.sample_id for sample in fixture.tree.groups[-1].samples}
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier, fixture.protocol["notebook_objective"])
        self.assertEqual(set(updates.notebook_objective_scores(fixture.tree, fixture.protocol["notebook_objective"])), final_ids)
        result = fixture.step(fixture.optimizer_fixture())
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual({row["sample_id"] for row in result["notebook_backward"]}, final_ids)
        self.assertEqual(set(result["synthesis_token_credit"]), final_ids)
        self.assertEqual(result["n_groups_by_kind"]["DIAG"], 1)

    def test_failed_final_group_has_no_repair_or_draft_reward(self):
        fixture = self.fixture(final=["{}"] * 4)
        record = fixture.tree.notebook["rewarded_notebook"]
        self.assertEqual(len(record["rounds"]), 1)
        self.assertEqual(record["stop_reason"], "NOTEBOOK_ROUND_LIMIT")
        self.assertEqual([sample.reward for sample in fixture.tree.groups[-1].samples], [0.] * 4)
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier, fixture.protocol["notebook_objective"])
        self.assertFalse(local_run.error_relationship_signal([fixture.tree], fixture.protocol["notebook_objective"], fixture.engine.tokenizer))

    def test_final_context_and_budget_block_keep_drafts_without_whole_note_actions(self):
        for kwargs, reason in (({"blocked": "EPISODE_NOTEBOOK"}, "CONTEXT_BLOCKED"),
                               ({"stop_after_drafts": True}, "BUDGET_STOP")):
            with self.subTest(reason=reason):
                fixture = self.fixture(**kwargs)
                record = fixture.tree.notebook["rewarded_notebook"]
                self.assertEqual((record["status"], record["stop_reason"]), ("FINAL_NOTE_BLOCKED", reason))
                self.assertEqual(record["pair_drafts"]["status"], "COMPLETE")
                self.assertEqual(len(record["pair_drafts"]["calls"]), 1)
                self.assertEqual(record["group_ids"], [])
                self.assertFalse(any(group.kind == "DIAG" for group in fixture.tree.groups))
                self.assertEqual(fixture.tree.notebook.get("model_notes", []), [])
                episode.validate_update_binding(fixture.tree, tokenizer=fixture.engine.tokenizer,
                    draft_policy=drafts.VERSION, preparation=fixture.engine, images=fixture.images[fixture.actor.problem_id])

    def test_draft_context_block_is_not_a_whole_note_or_training_group(self):
        fixture = self.fixture(blocked="PAIR_DRAFT")
        record = fixture.tree.notebook["rewarded_notebook"]
        self.assertEqual((record["status"], record["stop_reason"]), ("DRAFT_BLOCKED", "CONTEXT_BLOCKED"))
        self.assertEqual(record["pair_drafts"]["status"], "CONTEXT_BLOCKED")
        self.assertEqual(record["group_ids"], [])
        self.assertFalse(any(group.kind == "DIAG" for group in fixture.tree.groups))
        self.assertEqual(fixture.draft_events[-1]["phase"], "DRAFTS_COMPLETED")

    def test_unavailable_public_pairs_preserve_stage_availability_and_generate_whole_note(self):
        from fine_tuning.notebook_rl import pair_action_notebook

        for blocked, status in ((None, "NO_PUBLIC_TARGETS"), ("PUBLIC_ENDPOINTS_UNAVAILABLE", "PUBLIC_TARGET_BLOCKED")):
            with self.subTest(status=status), patch.object(pair_action_notebook, "_plan", return_value=([], blocked)):
                fixture = self.fixture(empty_plan=True)
                record = fixture.tree.notebook["rewarded_notebook"]
                self.assertEqual(record["status"], "GENERATED")
                self.assertEqual(record["pair_drafts"]["status"], status)
                self.assertEqual(record["pair_drafts"]["calls"], [])
                self.assertEqual((len(record["rounds"]), len(fixture.generated_groups)), (1, 1))
                self.assertEqual((fixture.engine.requests[-1].n, fixture.engine.requests[-1].max_new_tokens), (4, 8192))
                self.assertIn(drafts.quoted_context(record["pair_drafts"]),
                              fixture.engine.requests[-1].messages[0]["content"][-1]["text"])
                self.assertEqual([sample.text for sample in fixture.tree.groups[-1].samples],
                                 [fixture.good, "{}", fixture.good, "{}"])

    def test_draft_text_plan_and_missing_policy_tampering_fail_even_after_outer_reseal(self):
        fixture = self.fixture()
        for mutation in ("text", "plan", "policy", "missing", "rounds"):
            changed = copy.deepcopy(fixture.tree)
            record = changed.notebook["rewarded_notebook"]
            if mutation == "text":
                record["pair_drafts"]["calls"][0]["completions"][0]["text"] = "changed"
                inner = record["pair_drafts"]
                inner["sha256"] = _hash({key: value for key, value in inner.items() if key != "sha256"})
                record["pair_drafts_sha256"] = _hash(inner)
            elif mutation == "plan":
                record["pair_drafts"]["target_plan"][0]["earlier"]["attempt"] = 99
                inner = record["pair_drafts"]
                inner["sha256"] = _hash({key: value for key, value in inner.items() if key != "sha256"})
                record["pair_drafts_sha256"] = _hash(inner)
            elif mutation == "policy":
                record["notebook_draft_policy"] = "changed"
            elif mutation == "rounds":
                record["notebook_max_rounds"] = 3
            else:
                for key in ("pair_drafts", "pair_drafts_sha256", "notebook_draft_policy", "notebook_draft_contract", "notebook_draft_contract_sha256"):
                    del record[key]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                episode.validate_update_binding(changed, tokenizer=fixture.engine.tokenizer, draft_policy=drafts.VERSION)


if __name__ == "__main__":
    unittest.main()
