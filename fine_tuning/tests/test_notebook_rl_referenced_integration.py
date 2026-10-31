"""Authored compact-reference wiring; no real model or learned improvement."""
import copy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, partial_notebook, referenced_notebook
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.notebook_learning import append_rewarded_notebooks
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash, seeds
from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration
from fine_tuning.tests.test_notebook_rl_referenced_profile import referenced_protocol


class ReferencedNotebookIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.f = integration.SynthesisIntegrationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.protocol = referenced_protocol()
        self.f.protocol = self.protocol
        self.cfg = RolloutConfig.from_protocol(self.protocol, "ALG-NB3", master_seed=3407)

    @staticmethod
    def note(facts=None, lessons=None):
        return json.dumps({"coverage": "PARTIAL", "corrected_facts": facts or {}, "lessons": lessons or []})

    def compact_lesson(self):
        full = next(lesson for lesson in self.f.note["lessons"] if lesson["relation"] == "CORRECTED")
        index = referenced_notebook.public_endpoints(episode.solution_samples(self.f.tree))

        def reference(endpoint):
            return next(key for key, value in index.items()
                if all(value[field] == endpoint[field] for field in ("attempt", "kind", "rule", "locus")))

        return {"relation": full["relation"], "earlier_ref": reference(full["earlier"]),
                "later_ref": reference(full["later"])}

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
        self.assertEqual(self.record["notebook_generation_policy"], referenced_notebook.VERSION)
        self.assertEqual([sample.text for sample in self.group.samples], texts)
        self.assertEqual([sample.reward for sample in self.group.samples], [0.] * 4)
        self.assertTrue(all(sample.result is None for sample in self.group.samples))
        self.assertIsNone(self.record["rounds"][0]["feedback"])
        self.assertIsNone(self.record["rounds"][0]["repair_parent"])
        self.assertIsNone(self.record["rounds"][0]["continuation_suffix"])

    def test_compact_actions_preserve_full_history_images_seeds_and_strict_metric(self):
        texts = [self.note({"start": "q0"}, [self.compact_lesson()]), self.note(), "{broken", self.note({"start": "q1"})]
        self.append(texts)
        self.assertEqual(self.record["stop_reason"], "PARTIAL_NOTEBOOK_CAPTURED")
        request = self.f.engine.requests[-1]
        self.assertIs(request.images, self.f.images[self.f.actor.problem_id])
        parent = next(sample for sample in episode.solution_samples(self.f.tree)
                      if sample.sample_id == self.group.samples[0].parent_sample_id)
        self.assertEqual(request.seeds, seeds(self.f.actor.problem_id, 0, parent.turn,
                         "EPISODE_NOTEBOOK", 4, master_seed=3407))
        self.assertEqual((request.n, request.max_new_tokens), (4, 8192))
        body = request.messages[0]["content"][-1]["text"]
        transcript = json.loads(body.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
        for row, sample in zip(transcript, episode.solution_samples(self.f.tree), strict=True):
            self.assertEqual("".join(line for _, line in row["source_lines"]), sample.text)
        restored = from_json(ProblemTree, to_json(self.f.tree))
        receipt = episode.regrade_synthesis(restored, self.f.actor, self.f.verifier, self.f.engine.tokenizer)
        self.assertEqual(len(receipt["samples"]), 4)
        self.assertTrue(all(row["reward"] == 0 for row in receipt["samples"].values()))

    def test_all_empty_and_malformed_groups_close_without_repairs_or_success(self):
        for texts, stop in (([self.note()] * 4, "NO_NOTEBOOK_CLAIMS"),
                            (["{broken"] * 4, "PARTIAL_NOTEBOOK_CAPTURE_FAILED")):
            with self.subTest(stop=stop):
                self.setUp()
                self.append(texts)
                self.assertEqual(self.record["stop_reason"], stop)
                episode.regrade_synthesis(self.f.tree, self.f.actor, self.f.verifier, self.f.engine.tokenizer)

    def test_mode_metadata_prompt_and_extra_repair_group_are_rejected(self):
        self.append([self.note({"start": "q0"})] * 4)
        original = copy.deepcopy(self.f.tree)
        for policy in (None, partial_notebook.VERSION, "unknown"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                episode.validate_update_binding(original, generation_policy=policy)
        for field, value in (("notebook_generation_policy", partial_notebook.VERSION),
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

    def test_missing_mode_or_wrong_n4_fails_before_generation(self):
        for mutation in ("missing", "prompt", "n"):
            prompts = local_prompts(self.protocol)
            cfg = self.cfg
            if mutation == "missing":
                del prompts["notebook_generation_policy"]
            elif mutation == "prompt":
                prompts["rewarded_notebook"] = partial_notebook.prompt_contract()
            else:
                cfg = replace(cfg, n_diag=3)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                append_rewarded_notebooks([copy.deepcopy(self.f.tree)], self.f.images, self.f.engine, self.f.verifier,
                    prompts, cfg, actors={self.f.actor.problem_id: self.f.actor})
            self.assertEqual(self.f.engine.requests, [])

    def test_optimizer_rejects_legacy_and_partial_profiles_before_commit(self):
        from fine_tuning.notebook_rl import updates

        self.append([self.note({"start": "q0"}), self.note({"start": "q1"})] * 2)
        updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier, self.protocol["notebook_objective"])
        fixture = self.f.optimizer_fixture()
        for partial in (False, True):
            self.f.protocol = build_protocol(verified_note_synthesis_credit=True, partial_notebook=partial, context_limit=24576)
            with self.subTest(partial=partial), patch.object(fixture[1], "step", wraps=fixture[1].step) as step:
                with self.assertRaisesRegex(ValueError, "execution protocol"):
                    self.f.step(fixture)
                step.assert_not_called()
                self.assertEqual(fixture[1].state, {})

    def test_verified_compact_relationship_updates_original_tokens_without_strict_success(self):
        import torch
        from fine_tuning.notebook_rl import synthesis_credit, updates

        corrected = self.compact_lesson()
        persisted = corrected | {"relation": "PERSISTED"}
        texts = [self.note(lessons=[corrected]), self.note(lessons=[persisted])] * 2
        self.append(texts)
        objective = self.protocol["notebook_objective"]
        updates.bind_notebook_objective(self.f.tree, self.f.actor, self.f.verifier, objective)
        scores = updates.notebook_objective_scores(self.f.tree, objective)
        self.assertTrue(updates.group_has_signal(self.group, scores))
        self.assertEqual([sum(scores[s.sample_id]["relationship_rewards"].values()) for s in self.group.samples],
                         [1, 0, 1, 0])
        self.assertTrue(all(not score["fact_rewards"] and score["strict_reward"] == 0 for score in scores.values()))
        fixture = self.f.optimizer_fixture()
        bundle, optimizer, scheduler = fixture[:3]
        before_weight = bundle.model["L"].detach().clone()
        before_tree = to_json(self.f.tree)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = self.f.step(fixture)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual((step.call_count, scheduler.last_epoch), (1, 1))
        self.assertFalse(torch.equal(bundle.model["L"].detach(), before_weight))
        measured = [row for row in result["notebook_backward"] if row["relationship_gradient"]["measured"]]
        self.assertEqual(len(measured), 1)
        gradient = measured[0]["relationship_gradient"]
        self.assertEqual(gradient["scope"], "relationship_policy_loss_only")
        self.assertTrue(gradient["finite"])
        self.assertGreater(gradient["group_norms"]["L"], 0)
        self.assertIn("L", gradient["nonzero_groups"])
        self.assertGreater(gradient["positive_tokens"], 0)
        credit = result["synthesis_token_credit"][measured[0]["sample_id"]]
        self.assertEqual(gradient["credit_sha256"], _hash(credit))
        self.assertEqual(gradient["positive_tokens"], sum(value > 0 for value in credit["relationship_token_advantages"]))
        self.assertTrue(any(entry["reward"] == 1 and entry["advantage"] > 0 and entry["token_indices"]
                            and entry["credit_type"] == synthesis_credit.NATIVE_ERROR_RELATIONSHIP
                            for entry in credit["relationships"]))
        self.assertTrue(any("L" in row["changed_groups"] for row in result["notebook_backward"]))
        for sample in self.group.samples:
            self.assertEqual(sample.reward, 0.)
            self.assertEqual(result["synthesis_token_credit"][sample.sample_id]["token_ids_sha256"],
                             _hash(sample.completion_token_ids))
            self.assertEqual(set(json.loads(sample.text)["lessons"][0]), {"relation", "earlier_ref", "later_ref"})
            self.assertNotIn("evidence_ranges", sample.text)
        self.assertEqual(to_json(self.f.tree), before_tree)
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))


if __name__ == "__main__":
    unittest.main()
