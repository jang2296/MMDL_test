"""Actual CPU optimizer updates from authored repair fixtures; no Qwen quality claim."""
import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import episode_notebook, local_profile, repair_training, updates
from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.policy import VisionRoute
from fine_tuning.notebook_rl.repair_contract import ORIGIN_VERSION, public_origin_inventory
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_repair_training as prepared
from fine_tuning.tests.test_notebook_rl_repair_verifier import joint


class RepairUpdateTests(unittest.TestCase):
    def setUp(self):
        self.f = prepared.RepairTrainingTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.protocol = local_profile.build_protocol(repair_notebook=True, context_limit=24576)
        self.f.fixture.cfg = replace(self.f.fixture.cfg, master_seed=3407)
        self.f.fixture.prompts = local_profile.local_prompts(self.protocol)
        self.model = torch.nn.ParameterDict({"L": torch.nn.Parameter(torch.zeros(16000, dtype=torch.float64)),
            "M": torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)),
            "V": torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)),
            "F": torch.nn.Parameter(torch.tensor([.25], dtype=torch.float64), requires_grad=False)})
        self.params = {key: [self.model[key]] for key in ("L", "M", "V")}
        self.bundle = SimpleNamespace(model=self.model, params=self.params, route=VisionRoute(self.params["V"]),
                                      inventory={"arm": "MOD-RV"})
        self.optimizer, self.scheduler = make_optimizer(self.bundle, self.protocol)
        self.shifted: set[str] = set()
        self.reference_shift = 0.

    def attach_inputs(self, tree, engine):
        samples = {sample.sample_id: sample for group in tree.groups for sample in group.samples}

        def model_inputs(request, completion, device):
            sample = samples[request.request_id]
            self.assertEqual(completion, sample.completion_token_ids)
            self.assertEqual(request.prompt_token_ids, sample.prompt_token_ids)
            return {"sample": sample}

        engine.model_inputs = model_inputs

        def logprobs(bundle, inputs, completion_len, *, chunk):
            sample = inputs["sample"]
            ids = torch.tensor(sample.completion_token_ids)
            # The fixture end token has its own installed tiny parameter entry.
            ids = ids.remainder(bundle.model["L"].numel())
            values = bundle.model["L"][ids].unsqueeze(0) + bundle.model["M"]
            if bundle.route.vision_on:
                values = values + bundle.model["V"]
            return torch.tensor([sample.sampler_logprobs], dtype=torch.float64) + values + (
                .1 if sample.sample_id in self.shifted else 0.)

        def reference(bundle, inputs, completion_len, *, chunk):
            return torch.tensor([inputs["sample"].sampler_logprobs], dtype=torch.float64) + self.reference_shift

        return logprobs, reference

    def step(self, tree, engine, **options):
        logprobs, reference = self.attach_inputs(tree, engine)
        with patch.object(updates, "completion_logprobs", side_effect=logprobs) as forward, patch.object(
                updates, "reference_logprobs", side_effect=reference) as ref:
            result = updates.optimizer_step(self.bundle, self.optimizer, self.scheduler, [tree],
                {tree.problem_id: self.f.images}, engine, options.pop("protocol", self.protocol),
                options.pop("algorithm", "ALG-NB3"), step=0, seed=options.pop("seed", 3407),
                observation_actors={tree.problem_id: self.f.actor}, objective_verifier=self.f.verifier, **options)
        return result, forward.call_count, ref.call_count

    def online(self, **options):
        tree, engine = self.f.online(**options)
        self.f.attach(tree, engine)
        return tree, engine

    def answers(self):
        f = self.f.fixture
        tree, engine = self.f.collect([[f.right, f.wrong, f.wrong, f.wrong],
            [joint(f.right), joint(f.wrong)], [joint(f.right)] * 2])
        self.f.attach(tree, engine, [f.final(tree)] * 4)
        return tree, engine

    def unchanged_parameters(self, before):
        for key, value in self.model.items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        self.assertFalse(self.optimizer.state)
        self.assertEqual(self.scheduler.last_epoch, 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.values()))

    def test_online_only_wrong_answers_commit_once_without_a_final_note(self):
        tree, engine = self.online()
        before = to_json(tree)
        weights = {key: value.detach().clone() for key, value in self.model.items()}
        with patch.object(self.optimizer, "step", wraps=self.optimizer.step) as commit, patch.object(
                self.scheduler, "step", wraps=self.scheduler.step) as schedule, patch.object(
                repair_training, "prepare_tree_credit", wraps=repair_training.prepare_tree_credit) as prepare:
            result, forwards, references = self.step(tree, engine, require_retained_learning_signal=True)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual((commit.call_count, schedule.call_count, self.scheduler.last_epoch, prepare.call_count), (1, 1, 1, 2))
        self.assertEqual((forwards, references), (16, 8))
        self.assertEqual(result["n_groups_by_kind"], {"T1": 1, "T2": 1, "T3": 1, "DIAG": 0})
        self.assertTrue(all(sample.reward == 0 for group in tree.groups for sample in group.samples))
        self.assertTrue(result["repair_aggregate_gradient"]["S"])
        self.assertTrue(result["retained_repair_signal"]["policy_gradient_present"])
        self.assertFalse(result["retained_repair_signal"]["parameter_gradient_cancellation_verified"])
        self.assertFalse(result["repair_aggregate_gradient"]["benefit_verified"])
        self.assertFalse(torch.equal(self.model["L"], weights["L"]))
        torch.testing.assert_close(self.model["V"], weights["V"], atol=0, rtol=0)
        torch.testing.assert_close(self.model["F"], weights["F"], atol=0, rtol=0)
        self.assertNotIn(self.model["V"], self.optimizer.state)
        self.assertEqual(to_json(tree), before)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.values()))

    def test_final_channel_uses_original_claim_tokens_with_same_forward(self):
        tree, engine = self.f.online(wrong_answer=False)
        history = episode_notebook.solution_samples(tree)
        inventory = public_origin_inventory(history, contract_version=ORIGIN_VERSION,
            response_contracts=tree.notebook["repair_collector"]["response_contracts"])
        origin = next(row for row in inventory.origins if row.sample_id == history[4].parent_sample_id and row.rule == "COST_SUM")
        later = next(row for row in inventory.origins if row.sample_id == history[4].sample_id and row.rule == "COST_SUM")
        claim = {"origin_ref": origin.ref_id, "later_ref": later.ref_id, "relation": "CORRECTED", "rule": "COST_SUM",
                 "repaired_value": {"field": "cost", "value": self.f.trace["cost"]}}
        good = self.f.fixture.final(tree, claims=[claim], lesson_text="UNVERIFIED_LESSON")
        bad = self.f.fixture.final(tree, claims=[claim | {"repaired_value": {"field": "cost", "value": 999}}])
        self.f.attach(tree, engine, [good, bad, self.f.fixture.final(tree), self.f.fixture.final(tree)])
        result, forwards, references = self.step(tree, engine)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertGreater(result["repair_aggregate_gradient"]["channels"]["final"]["signed"]["L"], 0)
        self.assertGreater(result["repair_aggregate_gradient"]["channels"]["final"]["positive"]["L"], 0)
        self.assertEqual((forwards, references), (20, 10))
        self.assertTrue(all(sample.reward == 0 for sample in tree.groups[-1].samples))

    def test_r0_answer_update_preserves_collection_kl_calls_and_denominators(self):
        tree, engine = self.answers()
        self.reference_shift = .2
        from fine_tuning.notebook_rl import losses
        with patch.object(losses, "group_loss", wraps=losses.group_loss) as loss:
            result, forwards, references = self.step(tree, engine, notebook_reward_enabled=False)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertTrue(result["retained_repair_signal"]["policy_gradient_present"])
        self.assertFalse(result["repair_aggregate_gradient"]["S"])
        self.assertEqual((forwards, references), (24, 12))
        self.assertEqual(sum(call.kwargs["beta"] == .01 for call in loss.call_args_list), 12)
        receipt = result["repair_preparation"][tree.problem_id]
        for group in receipt["groups"]:
            self.assertEqual(group["original_group_size"], 4 if group["kind"] in ("T1", "DIAG") else 2)
            for row in group["samples"]:
                self.assertEqual(row["answer_coefficients"], row["applied_answer_coefficients"])
                self.assertFalse(any(row["applied_online_coefficients"]))
                self.assertFalse(any(row["applied_final_coefficients"]))
        self.assertEqual(self.scheduler.last_epoch, 1)

    def test_all_zero_coefficients_never_allow_kl_only_update(self):
        f = self.f.fixture
        tree, engine = self.f.collect([[f.wrong] * 4, [joint(f.wrong)] * 2, [joint(f.wrong)] * 2])
        self.f.attach(tree, engine)
        self.reference_shift = .2
        before = {key: value.detach().clone() for key, value in self.model.items()}
        result, forwards, references = self.step(tree, engine)
        self.assertEqual(result["status"], "NO_LEARNING_SIGNAL")
        self.assertEqual((forwards, references), (0, 0))
        self.unchanged_parameters(before)

    def test_full_mis_exclusion_keeps_original_n_and_never_recenters(self):
        tree, engine = self.online()
        self.shifted = {sample.sample_id for group in tree.groups for sample in group.samples}
        before = {key: value.detach().clone() for key, value in self.model.items()}
        result, forwards, references = self.step(tree, engine)
        self.assertEqual(result["status"], "NO_MATCHED_ACTIONS")
        self.assertEqual((forwards, references), (8, 0))
        self.assertEqual(result["masked_fraction"], 1.)
        self.assertFalse(result["retained_repair_signal"]["policy_gradient_present"])
        self.assertEqual([row["disposition"] for row in result["repair_aggregate_gradient"]["members"]], ["MIS_EXCLUDED"] * 4)
        positive = result["repair_preparation"][tree.problem_id]["groups"][1]["samples"][0]
        unit = next(row for row in positive["component_credit"]["unit_rows"] if row["semantic_score"] == 1)
        self.assertEqual(unit["mean"], .5)
        self.unchanged_parameters(before)

    def test_mis_can_remove_positive_repair_without_blocking_other_policy_credit(self):
        tree, engine = self.online()
        self.shifted = {tree.groups[1].samples[0].sample_id}
        result, forwards, references = self.step(tree, engine)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual((forwards, references), (15, 7))
        self.assertTrue(result["retained_repair_signal"]["policy_gradient_present"])
        self.assertFalse(result["repair_aggregate_gradient"]["S"])
        rows = result["repair_preparation"][tree.problem_id]["groups"][1]["samples"]
        unit = next(row for row in rows[1]["component_credit"]["unit_rows"] if row["token_indices"])
        self.assertEqual((unit["mean"], unit["advantage"]), (.5, -.5))

    def test_unresolved_or_incomplete_tree_never_reaches_forward(self):
        unknown = "Solution:\nNo authored observations.\nFinal answer: 999"
        tree, engine = self.f.collect([[unknown] * 4, [joint(unknown)] * 2, [joint(unknown)] * 2])
        self.f.attach(tree, engine)
        before = {key: value.detach().clone() for key, value in self.model.items()}
        result, forwards, references = self.step(tree, engine)
        self.assertEqual(result["status"], "NO_ADMITTED_REPAIR_TREES")
        self.assertIn("UNRESOLVED_ROUTING", result["repair_preparation"][tree.problem_id]["admission_reasons"])
        self.assertEqual((forwards, references), (0, 0))
        self.unchanged_parameters(before)
        tree, engine = self.f.online(wrong_answer=False)
        engine.block_final = True
        self.f.attach(tree, engine)
        result, forwards, references = self.step(tree, engine)
        self.assertEqual(result["status"], "NO_ADMITTED_REPAIR_TREES")
        self.assertIn("INCOMPLETE_FINAL_COLLECTION", result["repair_preparation"][tree.problem_id]["admission_reasons"])
        self.assertEqual((forwards, references), (0, 0))
        self.unchanged_parameters(before)

    def test_malformed_sibling_keeps_full_n2_and_visual_on_route(self):
        tree, engine = self.online(malformed=True, visual=True)
        result, _, _ = self.step(tree, engine)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertTrue(result["routes"][tree.problem_id])
        self.assertEqual(result["n_groups_by_kind"]["T2"], 1)
        group = result["repair_preparation"][tree.problem_id]["groups"][1]
        self.assertEqual(group["original_group_size"], 2)
        self.assertEqual(len(group["samples"]), 2)
        self.assertFalse(any(group["samples"][1]["applied_online_coefficients"]))

    def test_late_source_protocol_credit_and_pixel_mutations_abort_and_clear_gradients(self):
        for mode in ("source", "protocol", "credit", "pixel"):
            with self.subTest(mode=mode):
                tree, engine = self.online()
                before = {key: value.detach().clone() for key, value in self.model.items()}
                captured = []
                original = repair_training.prepare_tree_credit

                def prepare(*args, **kwargs):
                    row = original(*args, **kwargs)
                    captured.append(row)
                    return row

                def observe(event):
                    if event["stage"] != "optimizer":
                        return
                    self.assertTrue(any(parameter.grad is not None for parameter in self.model.values()))
                    if mode == "source":
                        tree.groups[0].samples[0].text += "changed"
                    elif mode == "protocol":
                        self.protocol["loss"]["kl"]["beta"] = 0.
                    elif mode == "credit":
                        captured[0]["groups"][1]["samples"][0]["applied_online_coefficients"][0] = 999.
                    else:
                        self.f.images[0].putpixel((0, 0), (0, 0, 0))

                with patch.object(repair_training, "prepare_tree_credit", side_effect=prepare), self.assertRaisesRegex(
                        ValueError, "mutated"):
                    self.step(tree, engine, on_progress=observe)
                self.unchanged_parameters(before)
                self.protocol = local_profile.build_protocol(repair_notebook=True, context_limit=24576)
                self.f.images[0].putpixel((0, 0), (255, 255, 255))

    def test_resealed_protocol_and_implicit_algorithms_are_rejected_before_forward(self):
        tree, engine = self.online()
        for options in ({"algorithm": "ALG-T1"}, {"seed": 1}, {"vision_disabled": True},
                        {"protocol": local_profile.build_protocol()}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.step(tree, engine, **options)
        altered = copy.deepcopy(self.protocol)
        altered["loss"]["kl"]["beta"] = 0.
        from fine_tuning.notebook_rl.checkpoint import digest
        altered["sha256"] = digest({key: value for key, value in altered.items() if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "saved protocol changed"):
            self.step(tree, engine, protocol=altered)


if __name__ == "__main__":
    unittest.main()
