"""Authored native history and differentiable CPU loss plumbing, never learned improvement."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_synthesis as synthesis, updates
from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.local_train import notebook_learning_gate
from fine_tuning.notebook_rl.notebook_learning import append_rewarded_notebooks
from fine_tuning.notebook_rl.policy import VisionRoute
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash
from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleEngine
from fine_tuning.tests.test_notebook_rl_notebook_synthesis import native, note_for, response, tree_from


class ChunkTokenizer:
    """A lossless CPU fixture codec; chunk size carries no real model capacity claim."""
    unk_token_id = -1
    end = 2

    def __init__(self):
        self.chunks: dict[str, int] = {}

    def convert_tokens_to_ids(self, token):
        if token != "<|im_end|>":
            raise ValueError("unexpected conversation token")
        return self.end

    def encode(self, text, **_):
        chunks = [text[index:index + 12] for index in range(0, len(text), 12)]
        for chunk in chunks:
            if chunk not in self.chunks:
                self.chunks[chunk] = len(self.chunks) + 100
        return [self.chunks[chunk] for chunk in chunks]

    def decode(self, ids, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        if clean_up_tokenization_spaces:
            raise ValueError("actual output must remain lossless")
        reverse = {value: key for key, value in self.chunks.items()}
        return "".join(reverse[token] for token in ids if not (skip_special_tokens and token == self.end))


class SynthesisIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = native()
        self.protocol = build_protocol(verified_note_synthesis=True)
        self.prompts = local_prompts(self.protocol)
        texts = [response(self.item.slots)] * 4 + [response(self.item.slots, wrong=False)] * 2
        self.tree = tree_from(self.actor, self.verifier, texts)
        self.engine = ReversibleEngine([])
        self.engine.tokenizer = ChunkTokenizer()
        samples = episode.solution_samples(self.tree)
        for index, sample in enumerate(samples):
            parent = samples[0] if index in (2, 3) else samples[2] if index in (4, 5) else None
            sample.prompt_token_ids = ([1, 2] if parent is None else
                parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2])
            sample.completion_token_ids = self.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.reward = float(sample.result.status == "CORRECT")
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier,
                                            contract=synthesis.BOUNDED_VERSION)
        self.note = note_for(self.tree, self.audit, self.item)
        for attempt in self.note["attempts"]:
            attempt["unverified"] = []
        self.good = json.dumps(self.note, separators=(",", ":"))
        self.images = {self.actor.problem_id: [Image.new("RGB", (2, 2), "red")]}
        self.addCleanup(self.images[self.actor.problem_id][0].close)
        self.cfg = RolloutConfig.from_protocol(self.protocol, "ALG-NB3", master_seed=3407)

    def append(self, rounds=None):
        observation = "Observed facts:\n" + "\n".join(f"{key} = {value}" for key, value in self.item.slots.items()) + "\nSolution:"
        self.engine.replies = [[observation], *(rounds or [[self.good, "{}", self.good, "{}"]])]
        return append_rewarded_notebooks([self.tree], self.images, self.engine, self.verifier,
            self.prompts, self.cfg, actors={self.actor.problem_id: self.actor})

    def optimizer_fixture(self):
        model = torch.nn.ParameterDict({"L": torch.nn.Parameter(torch.zeros(4096, dtype=torch.float64)),
            "M": torch.nn.Parameter(torch.zeros((), dtype=torch.float64)),
            "V": torch.nn.Parameter(torch.zeros((), dtype=torch.float64))})
        bundle = SimpleNamespace(model=model, inventory={"arm": "MOD-RV"},
            params={key: [model[key]] for key in ("L", "M", "V")}, route=VisionRoute([model["V"]]))
        optimizer, scheduler = make_optimizer(bundle, self.protocol)
        samples = {sample.sample_id: sample for group in self.tree.groups for sample in group.samples}
        preparation = SimpleNamespace(policy_version=0, tokenizer=self.engine.tokenizer,
            prepare=self.engine.prepare, model_inputs=lambda request, completion, device: {"sample": samples[request.request_id]})

        def logprobs(bundle, inputs, completion_len, *, chunk):
            sample = inputs["sample"]
            baseline = torch.tensor([sample.sampler_logprobs], dtype=torch.float64)
            derivative = (bundle.model["L"][torch.tensor(sample.completion_token_ids)].unsqueeze(0)
                          if sample.kind == "DIAG" else bundle.model["L"].sum() * 0)
            return baseline + derivative

        def reference(bundle, inputs, completion_len, *, chunk):
            return torch.tensor([inputs["sample"].sampler_logprobs], dtype=torch.float64)

        return bundle, optimizer, scheduler, preparation, logprobs, reference

    def step(self, fixture, **kwargs):
        bundle, optimizer, scheduler, preparation, logprobs, reference = fixture
        with patch.object(updates, "completion_logprobs", side_effect=logprobs), \
                patch.object(updates, "reference_logprobs", side_effect=reference):
            return updates.optimizer_step(bundle, optimizer, scheduler, [self.tree], self.images,
                preparation, self.protocol, "ALG-NB3", step=0, seed=3407,
                observation_actors={self.actor.problem_id: self.actor}, objective_verifier=self.verifier, **kwargs)

    def test_full_history_generation_uses_original_images_and_independent_route(self):
        before = copy.deepcopy(self.tree.groups)
        self.assertEqual(self.append()["reward_version"], synthesis.BOUNDED_VERSION)
        self.assertEqual(self.tree.groups[:-1], before)
        self.assertEqual([sample.reward for sample in self.tree.groups[-1].samples], [1., 0., 1., 0.])
        self.assertEqual(self.tree.route, "VISION_OFF")
        record = self.tree.notebook["rewarded_notebook"]
        self.assertFalse(record["used_for_routing"])
        self.assertEqual(record["episode_audit"]["visual_attempts"], [])
        self.assertTrue(record["synthesis_audit"]["attempts"][0]["issues"])
        self.assertTrue(all(not row["lesson_semantics_verified"] for row in self.tree.notebook["model_notes"]))
        request = self.engine.requests[-1]
        self.assertIs(request.images, self.images[self.actor.problem_id])
        body = request.messages[0]["content"][-1]["text"]
        transcript = json.loads(body.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
        self.assertEqual(len(transcript), 6)
        for row, sample in zip(transcript, episode.solution_samples(self.tree), strict=True):
            self.assertEqual("".join(line for _, line in row["source_lines"]), sample.text)
        self.assertNotIn('"checks":', body)
        receipt = episode.regrade_synthesis(self.tree, self.actor, self.verifier, self.engine.tokenizer)
        self.assertEqual(set(receipt["samples"]), {sample.sample_id for sample in self.tree.groups[-1].samples})

    def test_failed_feedback_preserves_literal_note_parent_without_expected_values(self):
        self.append([["{}"] * 4, [self.good, "{}", self.good, "{}"]])
        first, second = self.tree.groups[-2:]
        request = self.engine.requests[-1]
        parent = first.samples[0]
        self.assertEqual(request.prompt_token_ids[:len(parent.prompt_token_ids) + len(parent.completion_token_ids)],
                         parent.prompt_token_ids + parent.completion_token_ids)
        feedback = self.tree.notebook["rewarded_notebook"]["rounds"][1]["feedback"]
        payload = json.loads(feedback.removeprefix(synthesis.RETRY_INSTRUCTION))
        self.assertEqual(set(payload), {"contract", "feedback_policy", "failure_code", "checks", "failed_attempts",
                                        "failed_lessons", "format_diagnostics", "reinspection_slots"})
        self.assertEqual(payload["format_diagnostics"]["version"], synthesis.FORMAT_DIAGNOSTICS_VERSION)
        self.assertTrue(synthesis._valid_format_diagnostics(payload["format_diagnostics"]))
        self.assertNotIn('"corrected_facts":', feedback)
        self.assertNotIn(json.dumps(self.item.slots, separators=(",", ":")), feedback)
        self.assertEqual(second.samples[0].parent_sample_id, parent.sample_id)
        episode.validate_update_binding(self.tree, tokenizer=self.engine.tokenizer)

    def test_invalid_fact_sibling_reinspection_replays_and_forged_verifier_result_is_rejected(self):
        note = copy.deepcopy(self.note)
        note["corrected_facts"] = {"trans": {"q0": {"0": "q3"}}, "accepting": "q0,q1,q2,q3"}
        invalid = json.dumps(note)
        self.append([[invalid] * 4, [self.good] * 4])
        feedback = self.tree.notebook["rewarded_notebook"]["rounds"][1]["feedback"]
        payload = json.loads(feedback.removeprefix(synthesis.RETRY_INSTRUCTION))
        self.assertEqual(payload["reinspection_slots"], ["accepting"])
        self.assertEqual(self.tree.notebook["model_notes"][0]["failure_code"], "INVALID_FIELDS")
        self.assertTrue(all(sample.result is None for group in self.tree.groups if group.kind == "DIAG"
                            for sample in group.samples))
        restored = json.loads(to_json(self.tree))
        tree = from_json(ProblemTree, json.dumps(restored))
        episode.regrade_synthesis(tree, self.actor, self.verifier, self.engine.tokenizer)
        # A coordinated, resealed result forgery remains invalid against the fresh verifier.
        row = tree.notebook["model_notes"][1]
        receipt = row["fact_reinspection"]
        result = receipt["result"]["slots"]
        result.update(wrong_required=[], required_correct=1, all_correct=1)
        receipt["sha256"] = _hash({key: value for key, value in receipt.items() if key != "sha256"})
        episode.validate_update_binding(tree, tokenizer=self.engine.tokenizer)
        with self.assertRaisesRegex(ValueError, "fresh synthesis grade or sample reward differs"):
            episode.regrade_synthesis(tree, self.actor, self.verifier, self.engine.tokenizer)

    def test_initial_prepare_overflow_preserves_and_finalizes_the_complete_episode(self):
        before = copy.deepcopy(self.tree.groups)
        self.engine.blocked = "EPISODE_NOTEBOOK"
        result = self.append()
        record = self.tree.notebook["rewarded_notebook"]
        self.assertEqual(record["status"], "CONTEXT_BLOCKED")
        self.assertEqual(record["stop_reason"], "CONTEXT_BLOCKED")
        self.assertTrue(record["routing_finalized"])
        self.assertEqual(record["group_ids"], [])
        self.assertEqual(result["generated_groups"], 0)
        self.assertEqual(self.tree.groups, before)
        self.assertEqual(len(self.engine.requests), 1)  # Only the preserved fresh observation.
        self.assertEqual(record["synthesis_audit"]["attempt_count"], 6)
        episode.validate_update_binding(self.tree, tokenizer=self.engine.tokenizer)
        self.assertEqual(episode.regrade_synthesis(self.tree, self.actor, self.verifier,
                                                 self.engine.tokenizer)["samples"], {})
        self.assertTrue(to_json(self.tree))

    def test_non_context_preparation_errors_are_not_relabelled_as_capacity_limits(self):
        original = self.engine.prepare

        def broken(request):
            if "EPISODE_NOTEBOOK" in request.request_id:
                raise ValueError("image integrity failure")
            return original(request)

        with patch.object(self.engine, "prepare", side_effect=broken):
            with self.assertRaisesRegex(ValueError, "image integrity failure"):
                self.append()
        self.assertEqual(len(self.engine.requests), 1)
        self.assertFalse(any(group.kind == "DIAG" for group in self.tree.groups))

    def test_positive_note_advantage_enters_loss_and_strict_expansion_gate(self):
        self.append()
        fixture = self.optimizer_fixture()
        with patch.object(self.verifier, "trace", wraps=self.verifier.trace) as trace:
            update = self.step(fixture)
        self.assertGreater(trace.call_count, 0)
        self.assertEqual(update["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertTrue(any(row["advantage"] > 0 and row["reward_gradient_isolated"] and row["changed_groups"]
                            for row in update["notebook_backward"]))
        self.assertEqual(fixture[2].last_epoch, 1)
        self.assertTrue(torch.count_nonzero(fixture[0].model["L"]))
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].model.parameters()))
        result = {"notebook_signal_groups": 1, "notebook_gradient_contribution_verified": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw-0000.json"
            path.write_text(json.dumps({"kind": "TREE", "raw": json.loads(to_json(self.tree))}))
            self.assertEqual(notebook_learning_gate(Path(directory), result, update, self.protocol), [path])
            forged = copy.deepcopy(update)
            forged["synthesis_regrade_receipts"][self.actor.problem_id]["samples"].clear()
            with self.assertRaisesRegex(ValueError, "regraded synthesis evidence differs"):
                notebook_learning_gate(Path(directory), result, forged, self.protocol)

    def test_control_drops_note_pg_preserves_note_kl_and_original_scores(self):
        # Add genuine solution reward variance so the control has an eligible tree.
        sample = self.tree.groups[0].samples[1]
        text = response(self.item.slots, wrong=False)
        sample.text, sample.completion_token_ids = text, self.engine.tokenizer.encode(text) + [2]
        sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
        sample.result = self.verifier.grade(self.actor.problem_id, text, "stop", "obs")
        sample.reward = 1.
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.note = note_for(self.tree, self.audit, self.item)
        for attempt in self.note["attempts"]:
            attempt["unverified"] = []
        self.good = json.dumps(self.note, separators=(",", ":"))
        self.append()
        before = to_json(self.tree)
        bundle, optimizer, scheduler, preparation, logprobs, reference = self.optimizer_fixture()

        def shifted_reference(bundle, inputs, completion_len, *, chunk):
            value = reference(bundle, inputs, completion_len, chunk=chunk)
            return value + (.005 if inputs["sample"].kind == "DIAG" else 0.)

        with patch.object(updates, "group_loss", wraps=updates.group_loss) as loss:
            update = self.step((bundle, optimizer, scheduler, preparation, logprobs, shifted_reference),
                               notebook_reward_enabled=False)
        self.assertEqual(update["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual(update["effective_rewards"][self.tree.groups[-1].group_id], [0.] * 4)
        self.assertEqual(update["notebook_backward"], [])
        self.assertEqual(update["n_groups_by_kind"]["DIAG"], 1)
        for call in loss.call_args_list[-4:]:
            self.assertEqual(call.args[3].tolist(), [0.])
            self.assertEqual(call.kwargs["beta"], self.protocol["loss"]["kl"]["beta"])
        self.assertGreater(torch.count_nonzero(bundle.model["L"]).item(), 0)
        self.assertEqual(to_json(self.tree), before)

    def test_all_failed_notes_have_no_signal_and_never_change_parameters(self):
        self.append([["{}"] * 4] * 3)
        self.assertFalse(updates.has_learning_signal(self.tree, include_notebooks=True))
        fixture = self.optimizer_fixture()
        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
            with self.assertRaisesRegex(ValueError, "exclude signal-free solution trees"):
                self.step(fixture)
        step.assert_not_called()
        self.assertEqual(fixture[1].state, {})

    def test_forged_note_report_reward_tokens_or_audit_stop_before_forward(self):
        self.append()
        baseline = copy.deepcopy(self.tree)
        for mutation in ("text", "report", "reward", "tokens", "audit"):
            with self.subTest(mutation=mutation):
                self.tree = copy.deepcopy(baseline)
                note = self.tree.notebook["model_notes"][0]
                sample = self.tree.groups[-1].samples[0]
                record = self.tree.notebook["rewarded_notebook"]
                if mutation == "text":
                    sample.text = note["text"] = "{}"
                elif mutation == "report":
                    note["checks"]["all_issues_synthesized"] = False
                elif mutation == "reward":
                    sample.reward = sample.diag_reward = note["reward"] = 0.
                elif mutation == "tokens":
                    sample.completion_token_ids = self.engine.tokenizer.encode("{}") + [2]
                    sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
                else:
                    record["synthesis_audit"]["attempts"][0]["issues"] = []
                    record["synthesis_audit_sha256"] = _hash(record["synthesis_audit"])
                record["notebook_groups_sha256"] = _hash([json.loads(to_json(group)) for group in self.tree.groups if group.kind == "DIAG"])
                fixture = self.optimizer_fixture()
                with patch.object(updates, "completion_logprobs") as forward, \
                        patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
                    with self.assertRaises(ValueError):
                        updates.optimizer_step(fixture[0], fixture[1], fixture[2], [self.tree], self.images,
                            fixture[3], self.protocol, "ALG-NB3", step=0, seed=3407,
                            observation_actors={self.actor.problem_id: self.actor}, objective_verifier=self.verifier)
                forward.assert_not_called()
                step.assert_not_called()
                self.assertEqual(fixture[1].state, {})

    def test_late_report_mutation_is_regraded_immediately_before_optimizer(self):
        self.append()
        fixture = self.optimizer_fixture()

        def mutate(event):
            if event["stage"] == "optimizer":
                self.tree.notebook["model_notes"][0]["bounded_claims_verified"] = False

        with patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
            with self.assertRaisesRegex(ValueError, "fresh synthesis grade"):
                self.step(fixture, on_progress=mutate)
        step.assert_not_called()
        self.assertEqual(fixture[1].state, {})
        self.assertTrue(all(torch.count_nonzero(parameter) == 0 and parameter.grad is None
                            for parameter in fixture[0].model.parameters()))

    def test_missing_fresh_verifier_and_changed_actor_images_are_rejected(self):
        self.append()
        fixture = self.optimizer_fixture()
        for verifier, actor, images in ((None, self.actor, self.images),
                (self.verifier, replace(self.actor, question="A changed question"), self.images),
                (self.verifier, self.actor, {self.actor.problem_id: []})):
            with self.subTest(verifier=verifier, actor=actor, images=images), self.assertRaises(ValueError):
                updates.optimizer_step(fixture[0], fixture[1], fixture[2], [self.tree], images, fixture[3],
                    self.protocol, "ALG-NB3", step=0, seed=3407,
                    observation_actors={self.actor.problem_id: actor}, objective_verifier=verifier)
        self.assertEqual(fixture[1].state, {})

    def test_legacy_reward_one_remains_legacy_and_cannot_admit_synthesis(self):
        from fine_tuning.tests.test_notebook_rl_local_train import LocalTrainTests

        legacy = LocalTrainTests()
        root, path, _, result, update = legacy.notebook_evidence()
        self.addCleanup(legacy.doCleanups)
        self.assertEqual(notebook_learning_gate(root, result, update, build_protocol()), [path])
        with self.assertRaisesRegex(ValueError, "fresh regraded synthesis evidence"):
            notebook_learning_gate(root, result, update, self.protocol)
        with self.assertRaises(ValueError):
            notebook_learning_gate(root, result, update | {"synthesis_regrade_receipts": {}}, self.protocol)


if __name__ == "__main__":
    unittest.main()
