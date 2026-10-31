"""Paired-note CPU plumbing preserves original evidence and strict success metrics."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, local_run, local_train, paired_notebook, referenced_notebook
from fine_tuning.notebook_rl import updates
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.notebook_learning import append_rewarded_notebooks
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash, seeds
from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration
from fine_tuning.tests.test_notebook_rl_paired_profile import paired_protocol


class PairedNotebookIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.f = integration.SynthesisIntegrationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.protocol = paired_protocol()
        self.f.protocol = self.protocol
        self.cfg = RolloutConfig.from_protocol(self.protocol, "ALG-NB3", master_seed=3407)

    @staticmethod
    def note(lessons=None):
        return json.dumps({"coverage": "PARTIAL", "lessons": lessons or []})

    def compact_lesson(self):
        full = next(lesson for lesson in self.f.note["lessons"] if lesson["relation"] == "CORRECTED")
        index = paired_notebook.public_pairs(episode.solution_samples(self.f.tree))
        pair_id = next(key for key, pair in index.items()
            if all(pair[side][field] == full[side][field]
                   for side in ("earlier", "later") for field in ("attempt", "kind", "rule", "locus")))
        return {"pair_id": pair_id, "relation": full["relation"]}

    def append(self, texts):
        observation = "Observed facts:\n" + "\n".join(
            f"{key} = {value}" for key, value in self.f.item.slots.items()) + "\nSolution:"
        self.f.engine.replies = [[observation], texts]
        before = [to_json(sample) for sample in episode.solution_samples(self.f.tree)]
        report = append_rewarded_notebooks([self.f.tree], self.f.images, self.f.engine, self.f.verifier,
            local_prompts(self.protocol), self.cfg, actors={self.f.actor.problem_id: self.f.actor})
        self.assertEqual(before, [to_json(sample) for sample in episode.solution_samples(self.f.tree)])
        self.assertEqual(len(self.f.engine.requests), 2)
        self.assertFalse(self.f.engine.replies)
        self.record = self.f.tree.notebook["rewarded_notebook"]
        self.group = self.f.tree.groups[-1]
        self.assertEqual(report["generated_groups"], 1)
        self.assertEqual(self.record["notebook_generation_policy"], paired_notebook.VERSION)
        self.assertEqual([sample.text for sample in self.group.samples], texts)
        self.assertEqual([sample.reward for sample in self.group.samples], [0.] * 4)
        self.assertTrue(all(sample.result is None for sample in self.group.samples))
        self.assertIsNone(self.record["rounds"][0]["feedback"])
        self.assertIsNone(self.record["rounds"][0]["repair_parent"])
        self.assertIsNone(self.record["rounds"][0]["continuation_suffix"])

    def test_pair_actions_preserve_all_six_attempts_images_reread_and_seeds(self):
        texts = [self.note([self.compact_lesson()]), self.note(), "{broken", self.note()]
        self.append(texts)
        self.assertEqual(self.record["stop_reason"], "PARTIAL_NOTEBOOK_CAPTURED")
        request = self.f.engine.requests[-1]
        self.assertIs(request.images, self.f.images[self.f.actor.problem_id])
        parent = next(sample for sample in episode.solution_samples(self.f.tree)
                      if sample.sample_id == self.group.samples[0].parent_sample_id)
        self.assertEqual(request.seeds, seeds(self.f.actor.problem_id, 0, parent.turn,
                         "EPISODE_NOTEBOOK", 4, master_seed=3407))
        self.assertEqual((request.n, request.max_new_tokens), (4, 8192))
        self.assertIn("fresh_observation", self.record)
        body = request.messages[0]["content"][-1]["text"]
        transcript = json.loads(body.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
        self.assertEqual(len(transcript), 6)
        for row, sample in zip(transcript, episode.solution_samples(self.f.tree), strict=True):
            self.assertEqual("".join(line for _, line in row["source_lines"]), sample.text)
        self.assertIn("<public_pairs>", body)
        self.assertNotIn('"checks":', body)
        restored = from_json(ProblemTree, to_json(self.f.tree))
        receipt = episode.regrade_synthesis(restored, self.f.actor, self.f.verifier, self.f.engine.tokenizer)
        self.assertEqual(len(receipt["samples"]), 4)
        self.assertTrue(all(row["reward"] == 0 for row in receipt["samples"].values()))

    def test_empty_and_malformed_groups_close_without_repairs_or_learning_signal(self):
        for texts, stop in (([self.note()] * 4, "NO_NOTEBOOK_CLAIMS"),
                            (["{broken"] * 4, "PARTIAL_NOTEBOOK_CAPTURE_FAILED")):
            with self.subTest(stop=stop):
                self.setUp()
                self.append(texts)
                self.assertEqual(self.record["stop_reason"], stop)
                episode.regrade_synthesis(self.f.tree, self.f.actor, self.f.verifier, self.f.engine.tokenizer)
                updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier,
                                                self.protocol["notebook_objective"])
                self.assertFalse(local_run.error_relationship_signal([self.f.tree],
                    self.protocol["notebook_objective"], self.f.engine.tokenizer))

    def test_mode_metadata_and_extra_repair_group_are_rejected(self):
        self.append([self.note([self.compact_lesson()])] * 4)
        original = copy.deepcopy(self.f.tree)
        for policy in (None, referenced_notebook.VERSION, "unknown"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                episode.validate_update_binding(original, generation_policy=policy)
        for field, value in (("notebook_generation_policy", referenced_notebook.VERSION),
                             ("partial_artifacts", []), ("stop_reason", "VALID_NOTE")):
            changed = copy.deepcopy(original)
            changed.notebook["rewarded_notebook"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                episode.validate_update_binding(changed, tokenizer=self.f.engine.tokenizer)
        changed = copy.deepcopy(original)
        changed.groups.append(copy.deepcopy(changed.groups[-1]))
        record = changed.notebook["rewarded_notebook"]
        record["group_ids"].append(record["group_ids"][-1])
        record["rounds"].append(copy.deepcopy(record["rounds"][-1]))
        record["notebook_groups_sha256"] = _hash([json.loads(to_json(g)) for g in changed.groups if g.kind == "DIAG"])
        record["rounds_sha256"] = _hash(record["rounds"])
        with self.assertRaisesRegex(ValueError, "one original n4"):
            episode.validate_update_binding(changed, tokenizer=self.f.engine.tokenizer)

    def test_missing_mode_wrong_prompt_or_wrong_n4_fails_before_generation(self):
        for mutation in ("missing", "prompt", "n"):
            prompts = local_prompts(self.protocol)
            cfg = self.cfg
            if mutation == "missing":
                del prompts["notebook_generation_policy"]
            elif mutation == "prompt":
                prompts["rewarded_notebook"] = referenced_notebook.prompt_contract()
            else:
                cfg = replace(cfg, n_diag=3)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                append_rewarded_notebooks([copy.deepcopy(self.f.tree)], self.f.images, self.f.engine, self.f.verifier,
                    prompts, cfg, actors={self.f.actor.problem_id: self.f.actor})
            self.assertEqual(self.f.engine.requests, [])

    def test_optimizer_rejects_other_profile_before_commit(self):
        self.append([self.note([self.compact_lesson()]), self.note()] * 2)
        updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier, self.protocol["notebook_objective"])
        fixture = self.f.optimizer_fixture()
        for kwargs in ({}, {"partial_notebook": True}, {"referenced_notebook": True}):
            self.f.protocol = build_protocol(verified_note_synthesis_credit=True, context_limit=24576, **kwargs)
            with self.subTest(kwargs=kwargs), patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
                with self.assertRaisesRegex(ValueError, "execution protocol"):
                    self.f.step(fixture)
                step.assert_not_called()
                self.assertEqual(fixture[1].state, {})

    def assert_relationship_learning_gate(self, *, expected_type):
        objective = self.protocol["notebook_objective"]
        updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier, objective)
        self.assertTrue(local_run.error_relationship_signal([self.f.tree], objective, self.f.engine.tokenizer))
        result = self.f.step(self.f.optimizer_fixture())
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        measured = [row["relationship_gradient"] for row in result["notebook_backward"]
                    if row["relationship_gradient"]["measured"]]
        self.assertTrue(any(row["finite"] and row["nonzero_groups"]
                            and expected_type in row["positive_credit_types"] for row in measured))
        scores = updates.notebook_objective_scores(self.f.tree, objective)
        summary = {"notebook_signal_groups": sum(g.kind == "DIAG" and g.usable and updates.group_has_signal(g, scores)
                                                 for g in self.f.tree.groups)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw-0000.json"
            path.write_text(json.dumps({"kind": "TREE", "raw": json.loads(to_json(self.f.tree))}))
            kwargs = {"actors": {self.f.actor.problem_id: self.f.actor}, "verifier": self.f.verifier,
                      "tokenizer": self.f.engine.tokenizer}
            self.assertEqual(local_train.fact_credit_learning_gate(root, summary, result, self.protocol, **kwargs), [path])
            forged = copy.deepcopy(result)
            for row in forged["notebook_backward"]:
                row["relationship_gradient"] = {"scope": "relationship_policy_loss_only", "measured": False}
            with self.assertRaisesRegex(ValueError, "retained positive verified relationship-token gradient"):
                local_train.fact_credit_learning_gate(root, summary, forged, self.protocol, **kwargs)
        self.assertTrue(all(sample.reward == 0 for sample in self.group.samples))

    def test_native_relationship_requires_actual_isolated_gradient_for_fixed_ten(self):
        corrected = self.compact_lesson()
        self.append([self.note([corrected]), self.note([corrected | {"relation": "PERSISTED"}])] * 2)
        self.assert_relationship_learning_gate(expected_type="NATIVE_ERROR_RELATIONSHIP")

    def test_finalization_relationship_keeps_separate_learning_scope(self):
        samples = episode.solution_samples(self.f.tree)
        samples[0].text = samples[0].text.rsplit("\nFinal answer:", 1)[0]
        for index, sample in enumerate(samples):
            parent = samples[0] if index in (2, 3) else samples[2] if index in (4, 5) else None
            sample.prompt_token_ids = ([1, 2] if parent is None else
                parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2])
            sample.completion_token_ids = self.f.engine.tokenizer.encode(sample.text) + [2]
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
            sample.result = self.f.verifier.grade(self.f.actor.problem_id, sample.text, sample.finish_reason, self.f.actor.mode)
            sample.reward = sample.result.reward_answer
        pair_id = next(key for key, pair in paired_notebook.public_pairs(samples).items()
            if pair["earlier"]["attempt"] == 1 and pair["later"]["attempt"] == 3 and pair["earlier"]["kind"] == "OUTPUT")
        self.append([self.note([{"pair_id": pair_id, "relation": "FINALIZED"}]), self.note()] * 2)
        self.assert_relationship_learning_gate(expected_type="OUTPUT_FINALIZATION")


if __name__ == "__main__":
    unittest.main()
