"""Real tiny-Qwen RL gradients on controlled action/reward fixtures, not 4B results."""
import copy
import hashlib
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.notebook_learning import NOTEBOOK_REWARD_VERSION
from fine_tuning.notebook_rl.policy import assert_base_unchanged, completion_logprobs, reference_logprobs
from fine_tuning.notebook_rl.schema import Group, ProblemTree, Sample, VerifierResult
from fine_tuning.notebook_rl.updates import (action_mask, effective_rewards, has_learning_signal,
                                           optimizer_step, route_flags)
from fine_tuning.tests.test_notebook_rl_policy import bundle, config, inputs


class Characters:
    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids if i != 2)


class Preparation:
    """Same synthetic image tensors at every forward; no runtime/GPU attestation."""
    policy_version = 0
    tokenizer = Characters()

    def model_inputs(self, request, completion, device):
        value = inputs()
        ids = torch.tensor([request.prompt_token_ids+completion], device=device)
        return value | {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def tree(b, pid="p", version=0, route="VISION_ON"):
    samples = []
    for i, completion in enumerate(([10, 11, 2], [12, 13, 2])):
        request = SimpleNamespace(prompt_token_ids=[4, 3, 3, 3, 3, 5, 9])
        with torch.no_grad():
            lp = completion_logprobs(b, Preparation().model_inputs(request, completion, torch.device("cpu")), 3)
        result = VerifierResult("CORRECT" if i == 0 else "INCORRECT", str(i), float(i == 0), None, "fixture")
        samples.append(Sample(f"{pid}:{i}", pid, pid+":T1", "T1", 1, None,
            request.prompt_token_ids, list(completion), lp[0].tolist(), "stop", f"Final answer: {i}",
            3407+i, version, result, None, float(i == 0)))
    group = Group(pid+":T1", "T1", pid, hashlib.sha256(b"fixture").hexdigest(), samples, True, None)
    return ProblemTree(pid, version, [group], route, {}, {})


class UpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_reward_step_without_sft_reference_unchanged(self):
        b, protocol, prep = bundle(), config(), Preparation()
        opt, scheduler = make_optimizer(b, protocol)
        record = tree(b, route="VISION_OFF")
        before = {g: [tensor_sha256(p) for p in ps] for g, ps in b.params.items()}
        reference = reference_logprobs(b, inputs(), 3).clone()
        with patch("torch.nn.functional.cross_entropy", side_effect=AssertionError("SFT forbidden")), \
                patch("fine_tuning.train.AssistantCollator", side_effect=AssertionError("SFT forbidden")):
            result = optimizer_step(b, opt, scheduler, [record], {"p": [object()]}, prep, protocol,
                                    "ALG-T1", step=0, seed=3407)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertTrue(result["engine_sync_required"])
        self.assertEqual(result["step_after"], 1)
        self.assertEqual(prep.policy_version, 0)
        self.assertEqual(scheduler.last_epoch, 1)
        for g in ("L", "M"):
            self.assertGreater(result["grad_norm"][g], 0)
            self.assertNotEqual(before[g], [tensor_sha256(p) for p in b.params[g]])
        self.assertIsNone(result["grad_norm"]["V"])
        self.assertEqual(before["V"], [tensor_sha256(p) for p in b.params["V"]])
        self.assertTrue(torch.equal(reference, reference_logprobs(b, inputs(), 3)))
        self.assertTrue(all(p.grad is None for p in b.model.parameters()))
        assert_base_unchanged(b)

    def test_vision_off_after_on_skips_adam_momentum(self):
        b, protocol, prep = bundle(), config(), Preparation()
        opt, scheduler = make_optimizer(b, protocol)
        for step in range(2):
            prep.policy_version = step  # CPU fixture only, not a vLLM sync certificate.
            row = tree(b, version=step, route="VISION_ON" if step == 0 else "VISION_OFF")
            before = [p.detach().clone() for p in b.params["V"]]
            result = optimizer_step(b, opt, scheduler, [row], {"p": [object()]}, prep, protocol,
                                    "ALG-T1", step=step, seed=3407)
            if step == 0:
                self.assertGreater(result["grad_norm"]["V"], 0)
            else:
                self.assertIsNone(result["grad_norm"]["V"])
                self.assertTrue(all(torch.equal(old, p) for old, p in zip(before, b.params["V"], strict=True)))
        assert_base_unchanged(b)

    def test_mixed_routes_vision_gradient_only_from_on_tree(self):
        source = bundle()
        rows = [tree(source, "a", route="VISION_ON"), tree(source, "b", route="VISION_OFF")]
        observed = []
        for selected in ([rows[0]], rows):
            b = copy.deepcopy(source)
            opt, scheduler = make_optimizer(b, config())
            clip = torch.nn.utils.clip_grad_norm_
            captured = []
            def capture(parameters, *args, **kwargs):
                ps = list(parameters)
                if ps and id(ps[0]) in {id(p) for p in b.params["V"]}:
                    captured.extend(p.grad.detach().clone() for p in ps)
                return clip(ps, *args, **kwargs)
            with patch("torch.nn.utils.clip_grad_norm_", side_effect=capture):
                optimizer_step(b, opt, scheduler, selected, {t.problem_id: [object()] for t in selected},
                               Preparation(), config(), "ALG-T1", step=0, seed=3407)
            observed.append(captured)
        self.assertTrue(observed[0])
        for full, half in zip(*observed, strict=True):
            torch.testing.assert_close(half, full*.5, rtol=1e-6, atol=1e-9)

    def test_mismatch_all_masked_never_fake_updates(self):
        b = bundle()
        opt, scheduler = make_optimizer(b, config())
        row = tree(b)
        for sample in row.groups[0].samples:
            sample.sampler_logprobs = [p-.1 for p in sample.sampler_logprobs]
        before = {n: tensor_sha256(p) for n, p in b.model.named_parameters()}
        events = []
        result = optimizer_step(b, opt, scheduler, [row], {"p": [object()]}, Preparation(), config(),
                                "ALG-T1", step=0, seed=3407, on_progress=events.append)
        self.assertEqual(result["status"], "NO_MATCHED_ACTIONS")
        self.assertEqual(result["masked_fraction"], 1)
        self.assertFalse(result["engine_sync_required"])
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertFalse(opt.state)
        self.assertEqual(before, {n: tensor_sha256(p) for n, p in b.model.named_parameters()})
        self.assertEqual([(e["stage"], e["state"]) for e in events],
                         [("old", state) for _ in range(2) for state in ("START", "RETURN")])

    def test_progress_preserves_exact_update_and_adds_no_stage_synchronization(self):
        from fine_tuning.notebook_rl.updates import _stage_time
        source = bundle()
        row, events, results = tree(source), [], []
        for observer in (None, events.append):
            b = copy.deepcopy(source)
            opt, scheduler = make_optimizer(b, config())
            with patch("fine_tuning.notebook_rl.updates._stage_time", wraps=_stage_time) as timer:
                result = optimizer_step(b, opt, scheduler, [row], {"p": [object()]}, Preparation(),
                                        config(), "ALG-T1", step=0, seed=3407, on_progress=observer)
            result.pop("seconds")
            results.append((result, {n: tensor_sha256(p) for n, p in b.model.named_parameters()},
                            scheduler.state_dict(), timer.call_count, torch.get_rng_state().clone()))
        self.assertEqual(results[0][:-1], results[1][:-1])
        self.assertTrue(torch.equal(results[0][-1], results[1][-1]))
        expected = [(stage, state, index) for stage in ("old", "reference", "backward")
                    for index in (1, 2) for state in ("START", "RETURN")]
        self.assertEqual([(e["stage"], e["state"], e["action_index"]) for e in events],
                         expected + [("optimizer", "START", 0)])
        self.assertEqual(len({id(e) for e in events}), len(events))
        self.assertEqual(events, json.loads(json.dumps(events, allow_nan=False)))
        for event in events:
            self.assertEqual(event["event"], "UPDATE_PROGRESS")
            self.assertEqual(event["step"], 0)
            self.assertEqual(event["action_count"], 2)
            self.assertGreaterEqual(event["host_elapsed_seconds"], 0)
            self.assertTrue(all(type(value) in (str, int, float, bool) for value in event.values()))
            if event["stage"] != "optimizer":
                self.assertEqual(event["sample_id"], f"p:{event['action_index']-1}")
                self.assertEqual((event["kind"], event["prompt_tokens"], event["completion_tokens"]),
                                 ("T1", 7, 3))
                self.assertTrue(event["vision_on"])

    def test_observer_failure_never_commits_and_clears_backward_gradients(self):
        for stage, state in (("old", "START"), ("reference", "RETURN"),
                             ("backward", "RETURN"), ("optimizer", "START")):
            with self.subTest(stage=stage):
                b = bundle()
                opt, scheduler = make_optimizer(b, config())
                row = tree(b)
                before = {n: tensor_sha256(p) for n, p in b.model.named_parameters()}
                def observe(event):
                    if (event["stage"], event["state"]) == (stage, state):
                        if stage in ("backward", "optimizer"):
                            self.assertTrue(any(p.grad is not None for p in b.model.parameters()))
                        raise RuntimeError("observer failed")
                with self.assertRaisesRegex(RuntimeError, "observer failed"):
                    optimizer_step(b, opt, scheduler, [row], {"p": [object()]}, Preparation(),
                                   config(), "ALG-T1", step=0, seed=3407, on_progress=observe)
                self.assertEqual(scheduler.last_epoch, 0)
                self.assertFalse(opt.state)
                self.assertTrue(all(p.grad is None for p in b.model.parameters()))
                self.assertEqual(before, {n: tensor_sha256(p) for n, p in b.model.named_parameters()})

    def test_noncallable_progress_is_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError, "observer must be callable"):
            optimizer_step(None, None, None, [], {}, None, {}, "ALG-T1", step=0, seed=3407,
                           on_progress=False)

    def test_preupdate_difference_aborts_and_clears_partial_gradients(self):
        b, protocol, prep = bundle(), config(), Preparation()
        opt, scheduler = make_optimizer(b, protocol)
        row = tree(b)
        original = completion_logprobs
        def altered(*args, **kwargs):
            lp = original(*args, **kwargs)
            return lp+.01 if torch.is_grad_enabled() else lp
        before = [tensor_sha256(p) for ps in b.params.values() for p in ps]
        with patch("fine_tuning.notebook_rl.updates.completion_logprobs", side_effect=altered), \
                self.assertRaisesRegex(RuntimeError, "current/old"):
            optimizer_step(b, opt, scheduler, [row], {"p": [object()]}, prep, protocol,
                           "ALG-T1", step=0, seed=3407)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertTrue(all(p.grad is None for p in b.model.parameters()))
        self.assertEqual(before, [tensor_sha256(p) for ps in b.params.values() for p in ps])

    def test_uniform_groups_retained_if_another_solution_group_has_signal(self):
        b = bundle()
        row = tree(b)
        uniform = copy.deepcopy(row.groups[0])
        uniform.group_id += ":uniform"
        for sample in uniform.samples:
            sample.group_id = uniform.group_id
            sample.sample_id += ":uniform"
            sample.reward = 0
        row.groups.append(uniform)
        self.assertTrue(has_learning_signal(row))
        opt, scheduler = make_optimizer(b, config())
        result = optimizer_step(b, opt, scheduler, [row], {"p": [object()]}, Preparation(), config(),
                                "ALG-T1", step=0, seed=3407)
        self.assertEqual(result["n_groups_by_kind"]["T1"], 2)
        row.groups = [uniform]
        self.assertFalse(has_learning_signal(row))

    def test_noteonly_mask_uses_token_spans_and_missing_note_has_no_action(self):
        sample = tree(bundle()).groups[0].samples[0]
        text = "Error check:\nRecheck 한글.\nObserved facts:\ncell = 3\nSolution:\nFinal answer: 3"
        ids = [ord(c) for c in text]+[2]
        note = replace(sample, turn=2, text=text, completion_token_ids=ids)
        mask = action_mask(note, "ALG-NB-noteonly", Characters())[0].tolist()
        selected = "".join(chr(i) for i, yes in zip(ids, mask, strict=True) if yes)
        self.assertEqual(selected, text[:text.index("Observed facts:")])
        self.assertFalse(mask[-1])
        self.assertTrue(action_mask(replace(note, turn=1), "ALG-NB-noteonly", Characters()).all())
        no_note = replace(note, text="Final answer: 3", completion_token_ids=list(map(ord, "Final answer: 3")))
        self.assertFalse(action_mask(no_note, "ALG-NB-noteonly", Characters()).any())
        with self.assertRaisesRegex(ValueError, "text differ"):
            action_mask(replace(note, text="wrong text"), "ALG-NB-noteonly", Characters())

    def test_only_verified_local_notebook_variance_can_admit_an_update(self):
        b, protocol = bundle(), config()
        row = tree(b, route="VISION_OFF")
        note = copy.deepcopy(row.groups[0])
        note.kind, note.group_id = "DIAG", "p:NOTEBOOK"
        for sample in note.samples:
            sample.kind, sample.group_id = "DIAG", note.group_id
            sample.sample_id += ":note"
            sample.diag_reward, sample.result = sample.reward, None
        for sample in row.groups[0].samples:
            sample.reward = 0
        row.groups.append(note)
        self.assertFalse(has_learning_signal(row, include_notebooks=True))
        row.notebook["rewarded_notebook"] = {
            "group_id": note.group_id, "status": "GENERATED",
            "reward_version": NOTEBOOK_REWARD_VERSION,
        }
        self.assertFalse(has_learning_signal(row))
        self.assertTrue(has_learning_signal(row, include_notebooks=True))
        opt, sched = make_optimizer(b, protocol)
        with self.assertRaisesRegex(ValueError, "signal-free"):
            optimizer_step(b, opt, sched, [row], {"p": [object()]}, Preparation(),
                           protocol, "ALG-T1", step=0, seed=3407)
        protocol.update(profile="local-nf4-notebook-rl-v1",
                        notebook_reward=NOTEBOOK_REWARD_VERSION)
        before = [tensor_sha256(p) for p in b.params["L"]]
        result = optimizer_step(b, opt, sched, [row], {"p": [object()]}, Preparation(),
                                protocol, "ALG-T1", step=0, seed=3407)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertEqual(result["n_groups_by_kind"]["DIAG"], 1)
        self.assertNotEqual(before, [tensor_sha256(p) for p in b.params["L"]])
        self.assertEqual(len(result["notebook_backward"]), 2)
        self.assertTrue(any(row["advantage"] != 0 and "L" in row["changed_groups"]
                            for row in result["notebook_backward"]))
        self.assertTrue(all("V" not in row["changed_groups"] for row in result["notebook_backward"]))
        self.assertTrue(all(row["reward_gradient_isolated"] for row in result["notebook_backward"]))
        changed_reference = bundle()
        opt, sched = make_optimizer(changed_reference, protocol)
        with patch("fine_tuning.notebook_rl.updates.reference_logprobs",
                   side_effect=lambda *args, **kwargs: reference_logprobs(*args, **kwargs) - .1):
            uncertain = optimizer_step(changed_reference, opt, sched, [row], {"p": [object()]},
                Preparation(), protocol, "ALG-T1", step=0, seed=3407)
        self.assertTrue(any(r["changed_groups"] for r in uncertain["notebook_backward"]))
        self.assertFalse(any(r["reward_gradient_isolated"] for r in uncertain["notebook_backward"]))

    def test_kl_only_gradient_cannot_certify_notebook_reward_contribution(self):
        from fine_tuning.notebook_rl.losses import group_loss
        x = torch.tensor(0., requires_grad=True)
        current = torch.stack((-1 + x, -1 - x)).unsqueeze(0)
        old, reference = current.detach().clone(), torch.tensor([[-.8, -1.]])
        args = (current, old, reference, torch.tensor([.5]), torch.ones_like(current, dtype=torch.bool),
                torch.tensor([True]))
        plain = group_loss(*args, eps_low=.2, eps_high=.2, beta=0., norm_const=2)
        combined = group_loss(*args, eps_low=.2, eps_high=.2, beta=.01, norm_const=2)
        self.assertEqual(float(torch.autograd.grad(plain, x, retain_graph=True)[0]), 0.)
        self.assertNotEqual(float(torch.autograd.grad(combined, x)[0]), 0.)
        self.assertFalse(torch.equal(current.detach(), reference))

    def test_zero_gradient_allocation_is_not_a_notebook_contribution(self):
        from fine_tuning.notebook_rl.updates import _gradient_identity
        p = torch.nn.Parameter(torch.ones(2))
        before = _gradient_identity({"L": [p]})
        p.grad = torch.zeros_like(p)
        self.assertEqual(_gradient_identity({"L": [p]}), before)
        p.grad[0] = 1
        self.assertNotEqual(_gradient_identity({"L": [p]}), before)

    def test_control_rewards_and_exact_random_route_count_reproducible(self):
        b = bundle()
        rows = [tree(b, str(i), route="VISION_ON" if i < 3 else "VISION_OFF") for i in range(8)]
        random_routes = route_flags(rows, "MOD-RR", 3407, 2)
        self.assertEqual(sum(random_routes.values()), 3)
        self.assertEqual(random_routes, route_flags(list(reversed(rows)), "MOD-RR", 3407, 2))
        groups = [r.groups[0] for r in rows]
        before = copy.deepcopy(groups)
        random_rewards = effective_rewards(groups, "CTL-RAND", 3407, 2)
        self.assertEqual(random_rewards, effective_rewards(groups, "CTL-RAND", 3407, 2))
        self.assertTrue(all(set(v) <= {0., 1.} for v in random_rewards.values()))
        self.assertTrue(all(v == [1., 1.] for v in effective_rewards(groups, "CTL-FORMAT", 3407, 2).values()))
        self.assertEqual(before, groups)

    def test_stale_version_partition_pending_grads_and_blind_fail_closed(self):
        for problem in ("stale", "partition", "grad", "blind"):
            b, prep = bundle(), Preparation()
            opt, scheduler = make_optimizer(b, config())
            row = tree(b)
            if problem == "stale":
                prep.policy_version = 1
            if problem == "partition":
                opt.param_groups.reverse()
            if problem == "grad":
                b.params["L"][0].grad = torch.zeros_like(b.params["L"][0])
            with self.assertRaises(ValueError):
                optimizer_step(b, opt, scheduler, [row], {"p": [object()]}, prep, config(),
                    "CTL-BLIND" if problem == "blind" else "ALG-T1", step=0, seed=3407)


if __name__ == "__main__":
    unittest.main()
