"""CPU optimizer plumbing with authored notes and differentiable fake logprobs, not 4B results."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import fact_credit, updates
from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.notebook_learning import append_rewarded_notebooks
from fine_tuning.notebook_rl.policy import VisionRoute
from fine_tuning.notebook_rl.schema import Group, to_json
from fine_tuning.tests import test_notebook_rl_notebook_objective as fixtures
from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleEngine


class FactCreditUpdateTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.NotebookObjectiveTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.tree, self.actor, self.verifier = fixture.tree, fixture.actor, fixture.verifier
        self.protocol = build_protocol(verified_note_fact_credit=True)
        first, second = copy.deepcopy(fixture.note), copy.deepcopy(fixture.note)
        first["corrected_facts"] = {"CELL[ A ][ B ]": "7", fixtures.B: "999"}
        second["corrected_facts"] = {"cell['a'][\"b\"]": "999", fixtures.B: "5"}
        self.engine = ReversibleEngine([[fixtures.RIGHT], [json.dumps(first), json.dumps(second)] * 2])
        self.images = {self.tree.problem_id: [object()]}
        groups: list[Group] = []
        append_rewarded_notebooks([self.tree], self.images, self.engine, self.verifier,
            local_prompts(self.protocol), fixture.fixture.cfg, actors={self.tree.problem_id: self.actor},
            on_group=groups.append, should_stop=lambda: bool(groups))
        self.group = self.tree.groups[-1]
        self.objective = self.protocol["notebook_objective"]
        updates.bind_notebook_objective(self.tree, self.actor, self.verifier, self.objective)
        self.samples = {s.sample_id: s for g in self.tree.groups for s in g.samples}
        self.model = torch.nn.ParameterDict({
            "L": torch.nn.Parameter(torch.zeros(256, dtype=torch.float64)),
            "M": torch.nn.Parameter(torch.zeros((), dtype=torch.float64)),
            "V": torch.nn.Parameter(torch.zeros((), dtype=torch.float64)),
        })
        self.bundle = SimpleNamespace(model=self.model, inventory={"arm": "MOD-RV"},
            params={key: [self.model[key]] for key in ("L", "M", "V")},
            route=VisionRoute([self.model["V"]]))
        self.optimizer, self.scheduler = make_optimizer(self.bundle, self.protocol)
        self.preparation = SimpleNamespace(policy_version=0, tokenizer=self.engine.tokenizer,
            prepare=self.engine.prepare,
            model_inputs=lambda request, completion, device: {"sample": self.samples[request.request_id]})

    def logprobs(self, bundle, inputs, completion_len, *, chunk):
        sample = inputs["sample"]
        baseline = torch.tensor([sample.sampler_logprobs], dtype=torch.float64)
        self.assertEqual(baseline.shape, (1, completion_len))
        if sample.kind == "DIAG":
            # Character-dependent derivatives prevent centered slot rewards from cancelling.
            ids = torch.tensor(sample.completion_token_ids)
            return baseline + bundle.model["L"][ids].unsqueeze(0)
        return baseline + bundle.model["L"].sum() * 0

    def reference_logprobs(self, bundle, inputs, completion_len, *, chunk):
        return torch.tensor([inputs["sample"].sampler_logprobs], dtype=torch.float64)

    def step(self, **kwargs):
        return updates.optimizer_step(self.bundle, self.optimizer, self.scheduler,
            [self.tree], self.images, self.preparation, self.protocol, "ALG-NB3", step=0,
            seed=3407, observation_actors={self.tree.problem_id: self.actor},
            objective_verifier=self.verifier, **kwargs)

    def test_equal_scalar_rewards_still_update_from_freshly_verified_fact_tokens(self):
        scores = updates.notebook_objective_scores(self.tree, self.objective)
        self.assertEqual([s.reward for s in self.group.samples], [0.] * 4)
        self.assertEqual([scores[s.sample_id]["effective_reward"] for s in self.group.samples],
                         [.03125] * 4)
        self.assertTrue(updates.group_has_signal(self.group, scores))
        before = to_json(self.tree)
        with patch.object(self.verifier, "grade", wraps=self.verifier.grade) as grade, \
                patch.object(fact_credit, "group_credit", wraps=fact_credit.group_credit) as credit, \
                patch.object(updates, "group_loss", wraps=updates.group_loss) as loss, \
                patch.object(updates, "completion_logprobs", side_effect=self.logprobs), \
                patch.object(updates, "reference_logprobs", side_effect=self.reference_logprobs):
            result = self.step()
        self.assertGreater(grade.call_count, 0)
        credit.assert_called_once()
        partial_calls = [call for call in loss.call_args_list if call.kwargs["partial_adv"] is not None]
        self.assertEqual(len(partial_calls), 4)
        for call in partial_calls:
            self.assertEqual(call.args[3].tolist(), [0.])
            self.assertTrue(bool((call.kwargs["partial_adv"] > 0).any()))
            self.assertTrue(bool((call.kwargs["partial_adv"] < 0).any()))
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual((result["step_after"], self.scheduler.last_epoch), (1, 1))
        self.assertEqual(result["masked_fraction"], 0)
        self.assertGreater(result["grad_norm"]["L"], 0)
        for character in ("7", "5"):
            self.assertGreater(self.model["L"][ord(character) + 100].item(), 0)
        self.assertLess(self.model["L"][ord("9") + 100].item(), 0)
        credited_ids = {self.samples[sample_id].completion_token_ids[i]
            for sample_id, row in result["fact_token_credit"].items()
            for fact in row["facts"] for i in fact["token_indices"]}
        untouched = [i for i in range(256) if i not in credited_ids]
        self.assertTrue(torch.equal(self.model["L"][untouched], torch.zeros(len(untouched), dtype=torch.float64)))
        self.assertTrue(all(row["nonzero_partial_tokens"] > 0
                            and row["reward_gradient_isolated"] and row["changed_groups"] == ["L"]
                            for row in result["notebook_backward"]))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertEqual(to_json(self.tree), before)

    def test_answer_only_control_removes_the_only_signal_before_any_optimizer_step(self):
        before = to_json(self.tree)
        with patch.object(self.verifier, "grade", wraps=self.verifier.grade) as grade, \
                patch.object(fact_credit, "group_credit", wraps=fact_credit.group_credit) as credit, \
                patch.object(updates, "completion_logprobs") as forward, \
                patch.object(self.optimizer, "step", wraps=self.optimizer.step) as step:
            with self.assertRaisesRegex(ValueError, "exclude signal-free solution trees"):
                self.step(notebook_reward_enabled=False)
        self.assertGreater(grade.call_count, 0)
        credit.assert_not_called()
        forward.assert_not_called()
        step.assert_not_called()
        self.assertEqual(self.scheduler.last_epoch, 0)
        self.assertEqual(self.optimizer.state, {})
        self.assertTrue(all(bool((parameter == 0).all()) and parameter.grad is None
                            for parameter in self.model.parameters()))
        self.assertEqual(to_json(self.tree), before)


if __name__ == "__main__":
    unittest.main()
