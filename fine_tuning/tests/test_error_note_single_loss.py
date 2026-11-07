"""Single-response math and tiny CPU Qwen checks; these are not pretrained Qwen learning evidence."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from transformers.modeling_layers import GradientCheckpointingLayer

from fine_tuning.error_note import rl
from fine_tuning.error_note import single
from fine_tuning.error_note import train_v4 as train
from fine_tuning.error_note.contract import ROUTE_FULL, ROUTE_LM
from fine_tuning.error_note.protocol import (SINGLE_FORMAT_ANCHOR_BETA, SINGLE_FORMAT_ANCHOR_POSITIONS,
                                             SINGLE_REPLAY_MAX_EXCLUDED_ROWS, SINGLE_REPLAY_ROW_FAILURE,
                                             SINGLE_REWARD_GATE, SINGLE_SFT_WEIGHTING_V10, SINGLE_SFT_WEIGHTS_V10,
                                             SINGLE_TOKEN_SCALE_V10, VERSION_SINGLE, VERSION_SINGLE_V8,
                                             VERSION_SINGLE_V9, single_baseline_contract, single_source_baseline)
from fine_tuning.error_note import protocol as protocol_module
from fine_tuning.notebook_rl.input_preparation import digest
from fine_tuning.tests.test_error_note_train import encoded, tiny_factory
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE


_SHORT = None


def setUpModule():
    """Tiny-model fixtures use short targets; the 32-token solution rule is tested in single_runtime/loss."""
    global _SHORT
    from fine_tuning.error_note import single as _single

    _SHORT = patch.object(_single, "SOLUTION_MIN_TOKENS", 1)
    _SHORT.start()


def tearDownModule():
    _SHORT.stop()

SAMPLING = {"temperature": .7, "top_p": 1., "top_k": 0, "presence_penalty": 1.5}


def gradients(bundle):
    return {name: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for name, p in bundle.model.named_parameters() if id(p) in {id(q) for ps in bundle.params.values() for q in ps}}


def bind_reward_fixture(row, *, finish_reason="stop", baseline=None):
    for key, value in {"prompt": "Question: authored CPU fixture", "target": "Authored final answer: B", "target_source": "T1",
                       "image_paths": ["fixture.png"], "image_sha256": ["a" * 64], "subject": "Biology", "source": "authored-test"}.items():
        row.setdefault(key, value)
    row["generation"].setdefault("text", row["target"])
    t1 = {"text": row["target"], "generation": row["generation"], "prompt": row["prompt"], "status": row["status"],
          "finish_reason": finish_reason}
    return single.bind_single_reward(row, t1, PROVENANCE, baseline)


def full_support(bundle, actions):
    """top_k=0/top_p=1 fixtures sample from the whole vocabulary."""
    ids = list(range(bundle.model.get_base_model().lm_head.out_features))
    return [ids] * len(actions), [digest(ids)] * len(actions)


def pin_reward_fixture(rows, contract):
    contract.update(version=VERSION_SINGLE_V9,
        reward_provenance_sha256={row["problem_id"]: digest(row["reward_evidence"]["provenance"]) for row in rows if row["kind"] == "RL"},
        reward_evidence_sha256={row["problem_id"]: digest(row["reward_evidence"]) for row in rows if row["kind"] == "RL"})
    contract["rl"]["reward"] = SINGLE_REWARD_GATE
    return contract


def block(bundle, count=2, reward=1):
    rows = []
    for i in range(count):
        row = encoded(f"item-{i}:RL", (7, 8), tuple(range(9, 11 + i)), ROUTE_LM, "RL", seed=i)
        actions = row["input_ids"][-row["target_len"]:].tolist()
        prompt = row["input_ids"][:-row["target_len"]].tolist()
        row.update(problem_id=f"item-{i}", reward=reward, complete=True,
                   status="CORRECT" if reward else "INCORRECT", input_sha256=train._single_encoded_input_sha256(row))
        support, hashes = full_support(bundle, actions)
        row["generation"] = {"prompt_token_ids": prompt, "completion_token_ids": actions,
                             "input_sha256": row["input_sha256"], "policy_sha256": rl.policy_sha256(bundle),
                             "sampling_sha256": rl.sampling_sha256(SAMPLING), "support_token_ids": support,
                             "support_sha256": hashes}
        rows.append(row)
    bundle.route.apply(False)
    with torch.no_grad():
        values = train.single_batch_statistics(bundle, rows, SAMPLING)
    for row, value in zip(rows, values, strict=True):
        row["generation"].update(sampling_logprobs=value["logprobs"].tolist())
        bind_reward_fixture(row)
    contract = {"expected_problem_ids": [r["problem_id"] for r in rows], "policy_sha256": rl.policy_sha256(bundle),
                "sampling": SAMPLING, "rl": {"token_scale": 1024., "beta": .02, "format_anchor": {
                    "positions": SINGLE_FORMAT_ANCHOR_POSITIONS, "beta": SINGLE_FORMAT_ANCHOR_BETA}},
                "replay_support": {"row_failure": SINGLE_REPLAY_ROW_FAILURE,
                                   "max_excluded_rows": SINGLE_REPLAY_MAX_EXCLUDED_ROWS},
                "sft_weights": {"A1": .05, "A2": .5, "B": .05}}
    return rows, pin_reward_fixture(rows, contract)


def as_v10(rows, contract):
    """The v10 block contract: shipped source baseline, token scale 256, A2 0.35 per row; RL rows rebound with b(source)."""
    contract = copy.deepcopy(contract)
    contract["rl"].update(token_scale=SINGLE_TOKEN_SCALE_V10, baseline=single_baseline_contract())
    contract.update(sft_weights=dict(SINGLE_SFT_WEIGHTS_V10), sft_weighting=dict(SINGLE_SFT_WEIGHTING_V10))
    for row in rows:
        if row["kind"] == "RL":
            bind_reward_fixture(row, baseline=contract["rl"]["baseline"])
    pin_reward_fixture(rows, contract)
    contract["version"] = VERSION_SINGLE
    return contract


def as_v8(contract):
    """The sealed v8 contract shape: no format anchor and no replay row exclusion."""
    old = copy.deepcopy(contract) | {"version": VERSION_SINGLE_V8}
    del old["rl"]["format_anchor"], old["replay_support"]
    return old


def perturbed_factory(*args):
    """theta != F0: deterministic LoRA noise, so the raw F0 KL and the format anchor are nonzero."""
    bundle = tiny_factory(*args)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in bundle.params["L"]:
            parameter.add_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * .05)
    return bundle


def spiked(rows, contract, indices, nats):
    """Authored per-row replay mismatch: every stored sampling log probability of the row moves by `nats`."""
    rows = copy.deepcopy(rows)
    for index in indices:
        rows[index]["generation"]["sampling_logprobs"] = [v - nats for v in rows[index]["generation"]["sampling_logprobs"]]
        bind_reward_fixture(rows[index])
    return rows, pin_reward_fixture(rows, copy.deepcopy(contract))


def manual_rl_gradients(bundle, rows, count):
    """Fixed-denominator PG + raw KL + first-position anchor over `rows` only (B = count)."""
    bundle.model.zero_grad(set_to_none=True)
    bundle.model.train()
    bundle.route.apply(False)
    values = train.single_batch_statistics(bundle, rows, SAMPLING, reference=True)
    loss = sum(-row["reward"] * value["logprobs"].sum() / (count * 1024.) + .02 * value["kl"].sum() / (count * 1024.)
               + SINGLE_FORMAT_ANCHOR_BETA * value["kl"][:SINGLE_FORMAT_ANCHOR_POSITIONS].sum() / count
               for row, value in zip(rows, values, strict=True))
    loss.backward()
    result = gradients(bundle)
    bundle.model.zero_grad(set_to_none=True)
    return result


class SingleMath(unittest.TestCase):
    def test_processed_scores_match_pinned_transformers_warpers_and_generated_only_presence(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 29, requires_grad=True)
        generated = torch.tensor([[1, 1, 4], [5, 8, 5]])
        expected = logits - torch.zeros_like(logits).scatter(1, generated, 1.) * 1.5
        for warper in (TemperatureLogitsWarper(.7), TopKLogitsWarper(20), TopPLogitsWarper(.8)):
            expected = warper(generated, expected)
        actual = rl.processed_scores(logits, generated)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(rl.support_sha256(actual), rl.support_sha256(expected))
        actual[torch.isfinite(actual)].sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        raw = torch.tensor([[2., 2., 2.]])
        values = rl.processed_scores(raw, torch.tensor([[1, 1]]), temperature=1., top_p=1., top_k=0)
        torch.testing.assert_close(values, torch.tensor([[2., .5, 2.]]))

    def test_stored_support_keeps_sampled_boundary_action_and_matches_processed_inside(self):
        scores = torch.tensor([[3., 2.9, 2.85, 0.]], requires_grad=True)
        prefix = torch.tensor([[3]])
        processed = rl.processed_scores(scores, prefix, top_k=2, top_p=1.)
        self.assertFalse(bool(torch.isfinite(processed[0, 2])))  # replay-derived support would give -inf
        value = rl.stored_support_logprob(scores, prefix, 2, [0, 1, 2], top_k=2, top_p=1.)
        expected = scores[0, 2] / .7 - (scores[0, :3] / .7).logsumexp(0)
        torch.testing.assert_close(value, expected)
        value.backward()
        self.assertTrue(bool(torch.isfinite(scores.grad).all()) and float(scores.grad.abs().sum()) > 0)
        same = rl.stored_support_logprob(scores.detach(), prefix, 1, rl.support_ids(processed.detach())[0], top_k=2, top_p=1.)
        torch.testing.assert_close(same, processed.detach().log_softmax(-1)[0, 1])
        with self.assertRaisesRegex(ValueError, "outside"):
            rl.stored_support_logprob(scores, prefix, 3, [0, 1, 2])

    def test_ties_top_k_zero_and_empty_generated_prefix(self):
        scores = torch.tensor([[1., 1., 1., 0.]])
        empty = torch.empty(1, 0, dtype=torch.long)
        for top_k, top_p in ((0, 1.), (2, 1.), (0, .8)):
            expected = scores / .7
            if top_k:
                expected = TopKLogitsWarper(top_k)(empty, expected)
            if top_p < 1:
                expected = TopPLogitsWarper(top_p)(empty, expected)
            actual = rl.processed_scores(scores, empty, top_k=top_k, top_p=top_p)
            torch.testing.assert_close(actual, expected)

    def test_n1_has_true_pg_gradient_wrong_reward_is_zero_and_denominator_is_fixed(self):
        scores = torch.tensor([1., 2., 3.], requires_grad=True)
        lp = scores.log_softmax(0)[[1, 2]]
        loss = rl.single_pg(lp, 1, problem_count=2, token_scale=8.)
        expected = -lp.sum() / 16.
        torch.testing.assert_close(loss, expected)
        gradient = torch.autograd.grad(loss, scores, retain_graph=True)[0]
        self.assertGreater(float(gradient.abs().sum()), 0)
        wrong = rl.single_pg(lp, 0, problem_count=2, token_scale=8.)
        self.assertEqual(wrong.item(), 0)
        self.assertEqual(float(torch.autograd.grad(wrong, scores)[0].abs().sum()), 0)
        singleton = rl.single_pg(scores.log_softmax(0)[[1]], 1, problem_count=1, token_scale=8.)
        self.assertGreater(float(torch.autograd.grad(singleton, scores)[0].abs().sum()), 0)

    def test_raw_conditional_kl_matches_full_vocabulary_value_and_gradient(self):
        torch.manual_seed(5)
        head = torch.nn.Linear(5, 17, bias=True).requires_grad_(False)
        original = torch.randn(4, 5)
        current = (original + .4 * torch.randn(4, 5)).requires_grad_()
        actual = train._single_raw_kl(head, current, original, 3)
        logp0, logp = head(original).log_softmax(-1), head(current).log_softmax(-1)
        expected = (logp0.exp() * (logp0 - logp)).sum(-1)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-5)
        g_actual = torch.autograd.grad(actual.sum(), current, retain_graph=True)[0]
        g_expected = torch.autograd.grad(expected.sum(), current)[0]
        torch.testing.assert_close(g_actual, g_expected, atol=1e-7, rtol=1e-5)
        self.assertTrue(bool((actual > 0).all()))


class SingleEncoding(unittest.TestCase):
    def prepared(self):
        ids = torch.tensor([[4, 3, 5, 7]])
        item = {"inputs": {"input_ids": ids, "pixel_values": torch.ones(1, 4),
                           "image_grid_thw": torch.tensor([[1, 2, 2]])}, "prompt_token_ids": ids[0].tolist()}
        encoded_row = {"input_ids": torch.cat([ids[0], torch.tensor([2])]), "target_len": 1,
                       "pixel_values": item["inputs"]["pixel_values"], "image_grid_thw": item["inputs"]["image_grid_thw"]}
        item["input_sha256"] = train._single_encoded_input_sha256(encoded_row)
        return item

    def row(self, kind="RL", actions=(49, 38, 2)):
        row = {"row_id": "p:" + kind, "problem_id": "p", "kind": kind, "route": ROUTE_LM,
                "prompt": "Question: original", "target": "hello",
                "generation": {"text": "hello", "prompt_token_ids": [4, 3, 5, 7],
                               "completion_token_ids": list(actions), "input_sha256": self.prepared()["input_sha256"]}}
        if kind == "RL":
            row.update(status="CORRECT" if 2 in actions else "TRUNCATED", complete=True)
            bind_reward_fixture(row, finish_reason="stop" if 2 in actions else "length")
        return row

    @patch.object(train, "prepare_protocol_inputs")
    def test_original_ids_and_actual_eos_survive_decode_encode_counterexample(self, prepare):
        prepare.return_value = self.prepared(), []
        tokenizer = SimpleNamespace(all_special_tokens=[], encode=lambda *args, **kwargs: self.fail("RL retokenized"))
        result = train.encode_single_row(SimpleNamespace(tokenizer=tokenizer), None, self.row(), [object()])
        self.assertEqual(result["input_ids"].tolist(), [4, 3, 5, 7, 49, 38, 2])
        self.assertEqual(result["target_len"], 3)
        self.assertEqual(prepare.call_args.args[2][0]["content"][-1]["text"], "Question: original")
        # Length/loop termination never invents EOS, even if the visible text looks identical.
        truncated = train.encode_single_row(SimpleNamespace(tokenizer=tokenizer), None, self.row(actions=(49, 38)), [])
        self.assertEqual(truncated["input_ids"].tolist()[-2:], [49, 38])
        self.assertEqual(truncated["target_len"], 2)

    @patch.object(train, "prepare_protocol_inputs")
    def test_prompt_and_image_binding_changes_are_refused(self, prepare):
        prepare.return_value = self.prepared(), []
        processor = SimpleNamespace(tokenizer=SimpleNamespace(all_special_tokens=[]))
        for change in ({"prompt_token_ids": [4, 3, 5, 8]}, {"input_sha256": "b" * 64}):
            row = self.row()
            row["generation"].update(change)
            with self.assertRaises(ValueError):
                train.encode_single_row(processor, None, row, [])
        with self.assertRaisesRegex(ValueError, "placeholder"):
            train.encode_single_row(processor, SimpleNamespace(image_token_id=49), self.row(), [])

    @patch.object(train, "prepare_protocol_inputs")
    def test_sft_uses_actual_targets_with_p0_a2_and_note_b_prompt_masks(self, prepare):
        prepare.return_value = self.prepared(), []
        processor = SimpleNamespace(tokenizer=SimpleNamespace(all_special_tokens=[]))
        for kind in ("A1", "A2", "B"):
            row = self.row(kind)
            row["prompt"] = "gold and wrong solution are input only" if kind == "B" else "Question: P0"
            row["generation"]["prompt_token_ids"] = [999]  # A2 was generated with a lesson; teacher input is P0.
            result = train.encode_single_row(processor, None, row, [])
            target_start = len(result["input_ids"]) - result["target_len"]
            self.assertEqual(result["input_ids"][:target_start].tolist(), [4, 3, 5, 7])
            self.assertEqual(result["input_ids"][target_start:].tolist(), [49, 38, 2])
            changed = copy.deepcopy(row)
            changed["target"] = "reviewer's rewritten text"
            with self.assertRaises(ValueError):
                train.encode_single_row(processor, None, changed, [])


class SingleRewardGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_program_gate_rewards_only_admissible_correct_and_keeps_originals(self):
        owner = tiny_factory()
        rows, _ = block(owner, count=1)
        for status, reward in (("CORRECT", 1), ("INCORRECT", 0), ("TRUNCATED", 0), ("INVALID_NO_FINAL", -1)):
            row = copy.deepcopy(rows[0])
            row["status"] = status
            original = copy.deepcopy(row["generation"])
            bind_reward_fixture(row)
            self.assertEqual(row["reward"], reward)
            self.assertEqual(single.validate_single_reward(row)["accepted"], reward == 1)
            self.assertEqual(row["reward_evidence"]["decision"]["method"], "program")
            self.assertEqual(row["generation"], original)
            self.assertNotIn("text", row["reward_evidence"]["t1_metadata"])
            self.assertNotIn("generation", row["reward_evidence"]["t1_metadata"])
        for actions, finish, reason in ((list(range(4097)), "stop", "TOO_LONG"),
                                        ([9] * 512, "stop", "REPETITION"),
                                        ([9, 10], "length", "NOT_STOPPED")):
            row = copy.deepcopy(rows[0])
            row["generation"]["completion_token_ids"] = actions
            bind_reward_fixture(row, finish_reason=finish)
            self.assertEqual(row["reward"], 0)
            self.assertIn(reason, row["reward_evidence"]["decision"]["reason"])
            self.assertEqual(row["generation"]["completion_token_ids"], actions)
            row["reward_evidence"]["review"] = {"content": {"result": "PASS"}}
            with self.assertRaisesRegex(ValueError, "malformed"):
                single.validate_single_reward(row)

    def test_v7_reward_table_penalizes_only_wrong_bare_uncommitted_or_looping_answers(self):
        owner = tiny_factory()
        rows, _ = block(owner, count=1)
        long_ids = list(range(100, 140))
        loop_text = "Wait, let me count again. " * 200
        cases = (  # status, finish, completion ids, text, reward, penalty
            ("CORRECT", "stop", None, None, 0, None),  # correct bare answer: 0 (v5 gave -1)
            ("INCORRECT", "stop", None, None, -1, "BARE_WRONG"),
            ("INVALID_NO_FINAL", "stop", None, None, -1, "BARE_WRONG"),
            ("INCORRECT", "stop", long_ids, None, 0, None),
            ("INVALID_NO_FINAL", "stop", long_ids, None, -1, "NO_COMMITTED_ANSWER"),
            ("INVALID_MULTIPLE", "stop", long_ids, None, -1, "NO_COMMITTED_ANSWER"),
            ("TRUNCATED", "length", long_ids, None, 0, None),  # cut off without a loop
            ("TRUNCATED", "length", long_ids, loop_text, -1, "LOOP"),
            ("INCORRECT", "stop", long_ids, loop_text, -1, "LOOP"),
            ("CORRECT", "stop", long_ids, loop_text, 0, None),  # a correct answer never gets a penalty
            ("UNVERIFIABLE", "stop", None, None, 0, None),  # may be correct: never penalized
        )
        with patch.object(single, "SOLUTION_MIN_TOKENS", 32):
            for status, finish, ids, text, reward, penalty in cases:
                with self.subTest(status=status, finish=finish, long=ids is not None, loop=text is not None):
                    row = copy.deepcopy(rows[0])
                    row["status"] = status
                    if ids is not None:
                        row["generation"]["completion_token_ids"] = ids
                    if text is not None:
                        row["target"] = row["generation"]["text"] = text
                    bind_reward_fixture(row, finish_reason=finish)
                    decision = row["reward_evidence"]["decision"]
                    self.assertEqual((row["reward"], decision["penalty"]), (reward, penalty))
                    self.assertNotIn("bare", decision)
                    self.assertEqual(single.validate_single_reward(row), decision)
            forged = copy.deepcopy(rows[0])
            forged["status"] = "INCORRECT"
            bind_reward_fixture(forged)
            forged["reward"] = 0
            with self.assertRaisesRegex(ValueError, "program gate"):
                single.validate_single_reward(forged)
            forged["reward"] = -1
            forged["reward_evidence"]["decision"]["penalty"] = None
            with self.assertRaisesRegex(ValueError, "program gate"):
                single.validate_single_reward(forged)

    def test_v5_gate_committed_answer_is_never_uncommitted_and_denial_earns_zero(self):
        owner = tiny_factory()
        rows, _ = block(owner, count=1)
        long_ids = list(range(100, 140))
        denial = "The diagram is mislabeled, so I use the angle sum. Final answer: 32"
        cases = (  # status, extracted, text, reward, penalty, reason
            ("INVALID_MULTIPLE", "new york, connecticut, and massachusetts", None, 0, None, "T1_NOT_CORRECT"),
            ("INVALID_NO_FINAL", "a committed answer too long to compare", None, 0, None, "T1_NOT_CORRECT"),
            ("INVALID_NO_FINAL", None, None, -1, "NO_COMMITTED_ANSWER", "T1_NOT_CORRECT"),
            ("CORRECT", "32", denial, 0, None, "T1_ANSWER_DENIAL:mislabel"),
            ("CORRECT", "32", "The angle sum gives 32 degrees here. Final answer: 32", 1, None, None))
        for status, extracted, text, reward, penalty, reason in cases:
            with self.subTest(status=status, extracted=extracted, text=text):
                row = copy.deepcopy(rows[0])
                row.update(status=status, prompt="Question: authored CPU fixture", target_source="T1",
                           image_paths=["fixture.png"], image_sha256=["a" * 64], subject="Biology", source="authored-test")
                row["generation"]["completion_token_ids"] = long_ids
                row["target"] = row["generation"]["text"] = text or "Authored final answer: B"
                t1 = {"text": row["target"], "generation": row["generation"], "prompt": row["prompt"],
                      "status": status, "finish_reason": "stop", "extracted": extracted}
                single.bind_single_reward(row, t1, PROVENANCE)
                decision = row["reward_evidence"]["decision"]
                self.assertEqual((row["reward"], decision["penalty"], decision["reason"]), (reward, penalty, reason))
                self.assertEqual(single.validate_single_reward(row), decision)
                if reason is None:  # the same denial-free target is also an admissible A1
                    self.assertIsNone(single.answer_denial(row["target"]))

    def test_negative_reward_pushes_the_bare_answer_down(self):
        owner = tiny_factory()
        rows, contract = block(owner, count=1)
        with patch.object(single, "SOLUTION_MIN_TOKENS", 32):
            row = copy.deepcopy(rows[0])
            row["status"] = "INCORRECT"
            bind_reward_fixture(row)
        self.assertEqual(row["reward"], -1)
        logprobs = torch.tensor([-0.5, -1.0], requires_grad=True)
        term = rl.single_pg(logprobs, row["reward"], problem_count=1, token_scale=4.0)
        term.backward()
        self.assertGreater(float(logprobs.grad.min()), 0)  # minimizing the loss lowers these log probabilities
        self.assertAlmostEqual(float(term), -1.5 / 4.0)
        with self.assertRaises(ValueError):
            rl.single_pg(logprobs.detach(), 2, problem_count=1, token_scale=4.0)

    def test_forged_reward_and_decision_stale_or_malformed_receipts_fail(self):
        owner = tiny_factory()
        rows, contract = block(owner, count=1, reward=0)
        forged = copy.deepcopy(rows)
        forged[0]["reward"] = 1
        forged[0]["reward_evidence"]["decision"].update(accepted=True, reason=None)
        with self.assertRaisesRegex(ValueError, "program gate"):
            train.backward_single_block(owner, forged, contract, components=("PG",))
        positive, _ = block(owner, count=1)
        for mutate in (lambda e: e["t1_metadata"].update(seconds=123),
                       lambda e: e["decision"].update(method="ai_review"),
                       lambda e: e["decision"]["bindings"].update(source_sha256="f" * 64)):
            bad = copy.deepcopy(positive[0])
            mutate(bad["reward_evidence"])
            with self.assertRaises(ValueError):
                single.validate_single_reward(bad)
        self.assertTrue(all(p.grad is None for ps in owner.params.values() for p in ps))

    def test_sealed_source_and_evidence_pins_reject_joint_receipt_rewrites(self):
        owner = tiny_factory()
        rows, contract = block(owner, count=1)
        bad = copy.deepcopy(rows)
        evidence = bad[0]["reward_evidence"]
        evidence["provenance"]["revision"] = "different-source"
        evidence["decision"] = single._single_reward_decision(bad[0], evidence)
        single.validate_single_reward(bad[0])
        with self.assertRaisesRegex(ValueError, "sealed contract"):
            train.backward_single_block(owner, bad, contract, components=("PG",))

    def test_accepted_pg_nonzero_rejected_pg_zero_fixed_membership_and_full_kl(self):
        owner = tiny_factory()
        rows, contract = block(owner)
        originals = [copy.deepcopy(row["generation"]) for row in rows]
        rows[1]["status"] = "INCORRECT"
        bind_reward_fixture(rows[1])
        pin_reward_fixture(rows, contract)
        with patch.object(train, "_single_raw_kl", wraps=train._single_raw_kl) as divergence:
            stats = train.backward_single_block(owner, rows, contract, token_budget=10000, force_full_kl=True)
        self.assertTrue(stats["has_gradient"] and stats["replay_verified"])
        self.assertEqual(stats["rl"]["rows"], 2)
        self.assertEqual(stats["rl"]["action_tokens"], sum(row["target_len"] for row in rows))
        expected = -sum(rows[0]["generation"]["sampling_logprobs"]) / (2 * contract["rl"]["token_scale"])
        self.assertAlmostEqual(stats["rl"]["pg"], expected, places=7)
        self.assertEqual(divergence.call_count, 2)
        self.assertTrue(all(call.kwargs["force_full"] for call in divergence.call_args_list))
        self.assertEqual([row["generation"] for row in rows], originals)
        owner.model.zero_grad(set_to_none=True)
        rejected = [rows[1]]
        reject_contract = copy.deepcopy(contract) | {"expected_problem_ids": [rows[1]["problem_id"]]}
        pin_reward_fixture(rejected, reject_contract)
        zero = train.backward_single_block(owner, rejected, reject_contract, components=("PG",))
        self.assertEqual(zero["rl"]["pg"], 0)
        self.assertFalse(zero["has_gradient"])
        self.assertTrue(all(p.grad is None for ps in owner.params.values() for p in ps))

    def test_wrong_t1_never_uses_a2_approval_and_v1_contracts_cannot_resume(self):
        owner = tiny_factory()
        rows, contract = block(owner, count=1, reward=0)
        self.assertEqual(rows[0]["reward"], 0)
        self.assertEqual(rows[0]["reward_evidence"]["decision"]["reason"], "T1_NOT_CORRECT")
        bad = copy.deepcopy(rows[0])
        bad["reward_evidence"]["review"] = {"target_source": "RESOLVE", "content": {"result": "PASS"}}
        with self.assertRaisesRegex(ValueError, "malformed"):
            single.validate_single_reward(bad)
        for version in (None, "error-note-single-response-v1", "error-note-single-response-v2",
                        "error-note-single-response-v3"):
            old = copy.deepcopy(contract)
            old["version"] = version
            with self.assertRaisesRegex(ValueError, "cannot resume"):
                train.backward_single_block(owner, rows, old)


class SingleTinyPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_eval_entry_activates_actual_qwen_checkpointing_and_restores_mode(self):
        bundle = tiny_factory()
        bundle.model.eval()
        rows, contract = block(bundle, count=1)
        layer = next(module for module in bundle.model.modules() if isinstance(module, GradientCheckpointingLayer))
        calls = []
        original = layer._gradient_checkpointing_func

        def checkpoint_call(*args, **kwargs):
            calls.append((layer.training, torch.is_grad_enabled()))
            return original(*args, **kwargs)

        self.assertFalse(bundle.model.training)
        with patch.object(layer, "_gradient_checkpointing_func", side_effect=checkpoint_call):
            stats = train.backward_single_block(bundle, rows, contract, components=("PG",))
        self.assertTrue(stats["replay_verified"] and stats["has_gradient"])
        self.assertTrue(calls and all(training for training, _ in calls))
        self.assertTrue(any(enabled for _, enabled in calls))
        self.assertFalse(bundle.model.training)
        self.assertTrue(all(not module.training for module in bundle.model.modules()))

    def test_replay_failure_restores_incoming_train_or_eval_mode(self):
        for incoming in (False, True):
            bundle = tiny_factory()
            bundle.model.train(incoming)
            rows, contract = block(bundle, count=1)
            rows[0]["generation"]["sampling_logprobs"][0] += 1.
            # An authored receipt can bind changed metadata; numerical q replay must still reject it independently.
            bind_reward_fixture(rows[0])
            pin_reward_fixture(rows, contract)
            with self.assertRaisesRegex(ValueError, "replay changed"):
                train.backward_single_block(bundle, rows, contract, components=("PG",))
            self.assertEqual(bundle.model.training, incoming)
            self.assertTrue(all(module.training == incoming for module in bundle.model.modules()))
            self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))

    def test_short_row_engine_noise_passes_but_block_mean_and_token_max_bounds_hold(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=2)
        # One short row off by 0.3 nats per token (cross-engine first-token noise) stays admissible ...
        noisy = copy.deepcopy(rows)
        noisy[0]["generation"]["sampling_logprobs"] = [v - .3 for v in noisy[0]["generation"]["sampling_logprobs"]]
        bind_reward_fixture(noisy[0])
        noisy_contract = pin_reward_fixture(noisy, copy.deepcopy(contract))
        with patch.object(train, "SINGLE_REPLAY_BLOCK_MEAN_ATOL", .5):
            self.assertTrue(train.backward_single_block(bundle, noisy, noisy_contract, components=("PG",))["replay_verified"])
        bundle.model.zero_grad(set_to_none=True)
        # ... but not when the whole block drifts, nor any single token beyond the max bound.
        with self.assertRaisesRegex(ValueError, "block mean"):
            train.backward_single_block(bundle, noisy, noisy_contract, components=("PG",))
        spike = copy.deepcopy(rows)
        spike[1]["generation"]["sampling_logprobs"][0] -= 3.5
        bind_reward_fixture(spike[1])
        spike_contract = pin_reward_fixture(spike, copy.deepcopy(contract))
        with patch.object(train, "SINGLE_REPLAY_BLOCK_MEAN_ATOL", 10.):
            with self.assertRaisesRegex(ValueError, "row mean"):
                train.backward_single_block(bundle, spike, as_v8(spike_contract), components=("PG",))
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))

    def test_v9_drops_one_failing_long_row_from_pg_kl_and_anchor_with_fixed_denominators(self):
        bundle = perturbed_factory()
        rows, contract = block(bundle, count=4)  # 3..6 action tokens; the authored rows count as long below
        bad, bad_contract = spiked(rows, contract, [0], .15)  # row mean .15 > .1; block mean .45/18 < .05
        with patch.object(train, "SINGLE_REPLAY_ROW_MEAN_MIN_TOKENS", 1):
            stats = train.backward_single_block(bundle, bad, bad_contract, components=("PG", "KL"))
        excluded = stats["replay"]["excluded"]
        self.assertEqual([(e["problem_id"], e["tokens"]) for e in excluded], [("item-0", 3)])
        self.assertAlmostEqual(excluded[0]["row_mean"], .15, places=5)
        self.assertTrue(stats["has_gradient"] and stats["replay_verified"])
        expected = -sum(sum(row["generation"]["sampling_logprobs"]) for row in rows[1:]) / (4 * 1024.)
        self.assertAlmostEqual(stats["rl"]["pg"], expected, places=6)
        self.assertGreater(stats["rl"]["anchor"], 0)
        self.assertAlmostEqual(stats["loss"], stats["rl"]["pg"] + stats["rl"]["kl"] + stats["rl"]["anchor"], places=9)
        actual = gradients(bundle)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in actual.values()))
        for name, value in manual_rl_gradients(bundle, rows[1:], 4).items():
            torch.testing.assert_close(actual[name], value, atol=1e-7, rtol=1e-4)
        bundle.model.zero_grad(set_to_none=True)
        # The same mismatch on a short row (< 64 tokens, max <= 3) is engine noise, not a failure.
        self.assertEqual(train.backward_single_block(bundle, bad, bad_contract, components=("PG",))["replay"]["excluded"], [])
        bundle.model.zero_grad(set_to_none=True)
        # Excluded rows still count in the unchanged block mean.
        drift, drift_contract = spiked(rows, contract, [0], .5)
        with patch.object(train, "SINGLE_REPLAY_ROW_MEAN_MIN_TOKENS", 1), self.assertRaisesRegex(ValueError, "block mean"):
            train.backward_single_block(bundle, drift, drift_contract, components=("PG",))
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))

    def test_v9_holds_on_a_third_failing_row_and_v8_on_the_first(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=4)
        with patch.object(train, "SINGLE_REPLAY_BLOCK_MEAN_ATOL", 10.):
            two, two_contract = spiked(rows, contract, [0, 2], 3.5)  # row max 3.5 > 3 at any length
            stats = train.backward_single_block(bundle, two, two_contract, components=("PG",))
            self.assertEqual(sorted(e["problem_id"] for e in stats["replay"]["excluded"]), ["item-0", "item-2"])
            bundle.model.zero_grad(set_to_none=True)
            three, three_contract = spiked(rows, contract, [0, 1, 2], 3.5)
            with self.assertRaisesRegex(ValueError, "row mean .*RL HOLD"):
                train.backward_single_block(bundle, three, three_contract, components=("PG",))
            one, one_contract = spiked(rows, contract, [1], 3.5)
            with self.assertRaisesRegex(ValueError, "row mean .*RL HOLD"):
                train.backward_single_block(bundle, one, as_v8(one_contract), components=("PG",))
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))
        malformed = copy.deepcopy(contract)
        malformed["rl"]["format_anchor"]["beta"] = 0.
        with self.assertRaisesRegex(ValueError, "v9 format anchor"):
            train.backward_single_block(bundle, rows, malformed, components=("PG",))

    def test_v9_format_anchor_is_beta_first_positions_raw_kl_over_b_and_zero_at_f0(self):
        bundle = perturbed_factory()
        rows, contract = block(bundle, count=4)  # 5 and 6 action tokens exceed the 4 anchored positions
        with torch.no_grad():
            values = train.single_batch_statistics(bundle, rows, SAMPLING, reference=True)
        stats = train.backward_single_block(bundle, rows, contract, components=("KL",))
        anchor = sum(SINGLE_FORMAT_ANCHOR_BETA * float(v["kl"][:SINGLE_FORMAT_ANCHOR_POSITIONS].sum()) for v in values) / 4
        kl = sum(.02 * float(v["kl"].sum()) for v in values) / (4 * 1024.)
        self.assertGreater(anchor, 0)
        self.assertAlmostEqual(stats["rl"]["anchor"] / anchor, 1., places=5)
        self.assertAlmostEqual(stats["rl"]["kl"] / kl, 1., places=5)
        self.assertEqual((stats["rl"]["pg"], stats["replay"]["excluded"]), (0., []))
        self.assertTrue(stats["has_gradient"])
        old = train.backward_single_block(bundle, rows, as_v8(contract), components=("PG", "KL"))
        self.assertNotIn("anchor", old["rl"])
        self.assertNotIn("excluded", old["replay"])
        self.assertAlmostEqual(old["loss"], old["rl"]["pg"] + old["rl"]["kl"], places=12)
        fresh = tiny_factory()
        f0_rows, f0_contract = block(fresh, count=4)
        zero = train.backward_single_block(fresh, f0_rows, f0_contract, components=("KL",))
        self.assertEqual((zero["rl"]["anchor"], zero["rl"]["kl"], zero["loss"]), (0., 0., 0.))
        self.assertFalse(zero["has_gradient"])

    def test_nonzero_actual_or_configured_attention_and_lora_dropout_is_rejected(self):
        bundle = tiny_factory()
        bundle.model.eval()
        rows, contract = block(bundle, count=1)
        attention = next(module for module in bundle.model.modules() if hasattr(module, "attention_dropout"))
        changes = ((bundle.model.get_base_model().config.text_config, "attention_dropout", .1),
                   (bundle.model.peft_config["default"], "lora_dropout", .1),
                   (attention, "attention_dropout", .1))
        for owner, field, value in changes:
            with patch.object(owner, field, value):
                with self.assertRaisesRegex(ValueError, "dropout"):
                    train.backward_single_block(bundle, rows, contract, components=("PG",))
            self.assertFalse(bundle.model.training)
        # The installed LoRA config may claim zero while the actual module was changed.
        with patch.object(attention, "injected_dropout", torch.nn.Dropout(.1), create=True):
            with self.assertRaisesRegex(ValueError, "actual dropout"):
                train.backward_single_block(bundle, rows, contract, components=("PG",))
        self.assertFalse(bundle.model.training)
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))

    def test_partitioned_microbatches_have_same_pg_and_kl_gradients(self):
        bundle = tiny_factory()
        rows, contract = block(bundle)
        first = train.backward_single_block(bundle, rows, contract, components=("PG", "KL"))
        grad = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        second = train.backward_single_block(bundle, rows, contract, token_budget=1000, components=("PG", "KL"))
        self.assertAlmostEqual(first["loss"], second["loss"], places=7)
        for name, value in gradients(bundle).items():
            torch.testing.assert_close(value, grad[name], atol=1e-7, rtol=1e-4)
        self.assertTrue(second["replay_verified"] and second["has_gradient"])
        self.assertTrue(all(p.grad is None for p in bundle.params["V"]))

    def test_wrong_only_pg_creates_no_gradient_and_no_sft_weight_redistribution(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=1, reward=0)
        report = train.backward_single_block(bundle, rows, contract, components=("PG",))
        self.assertEqual(report["loss"], 0.)
        self.assertFalse(report["has_gradient"])
        self.assertEqual(report["sft"], {})
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))
        report = train.backward_single_block(bundle, rows, contract, components=("PG", "KL"))
        self.assertEqual(report["loss"], 0.)
        self.assertFalse(report["has_gradient"])

    def test_missing_partial_duplicate_or_ungraded_items_fail_before_backward(self):
        bundle = tiny_factory()
        rows, contract = block(bundle)
        cases = [rows[:1], rows + [rows[0] | {"row_id": "duplicate"}],
                 [rows[0] | {"complete": False}, rows[1]],
                 [rows[0] | {"status": "UNVERIFIABLE", "reward": 0}, rows[1]],
                 [rows[0] | {"status": "INCORRECT"}, rows[1]]]
        for bad in cases:
            with self.assertRaises(ValueError):
                train.backward_single_block(bundle, bad, contract)
            self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))

    def test_stale_policy_support_and_logprob_are_refused_before_backward(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=1)
        actions = rows[0]["generation"]["completion_token_ids"]
        for key, value in (("sampling_logprobs", [0.] * rows[0]["target_len"]),
                           ("support_sha256", [digest([])] * rows[0]["target_len"]),
                           ("support_token_ids", [[a + 1] for a in actions]),
                           ("support_token_ids", rows[0]["generation"]["support_token_ids"][:-1]),
                           ("policy_sha256", "b" * 64)):
            bad = copy.deepcopy(rows)
            bad[0]["generation"][key] = value
            with self.assertRaises(ValueError):
                train.backward_single_block(bundle, bad, contract)
        with torch.no_grad():
            bundle.params["L"][0].add_(.1)
        with self.assertRaises(ValueError):
            train.backward_single_block(bundle, rows, contract)
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))

    def test_f0_kl_zero_changed_policy_positive_and_original_merger_is_immutable(self):
        bundle = tiny_factory()
        rows, _ = block(bundle, count=1)
        bundle.route.apply(False)
        initial = train.single_batch_statistics(bundle, rows, SAMPLING, reference=True, chunk=2, vocab_chunk=7)[0]
        torch.testing.assert_close(initial["kl"], torch.zeros_like(initial["kl"]), atol=3e-7, rtol=0)
        with torch.no_grad():
            for parameter in bundle.params["M"]:
                parameter.add_(.05 * torch.randn_like(parameter))
        changed = train.single_batch_statistics(bundle, rows, SAMPLING, reference=True, chunk=2, vocab_chunk=7)[0]
        self.assertGreater(float(changed["kl"].sum().detach()), 0)
        changed["kl"].sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in bundle.params["M"] if p.grad is not None), 0)
        self.assertFalse(bundle.route.vision_on)
        root = bundle.inventory["mergers"][0]
        with torch.no_grad():
            next(bundle.model.get_base_model().get_submodule(root).original_module.parameters()).add_(1)
        with self.assertRaises(ValueError):
            train.single_batch_statistics(bundle, rows, SAMPLING, reference=True)

    def test_sft_kind_means_and_full_route_have_vision_gradients(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=2, reward=0)
        for i, kind in enumerate(("A1", "A2", "B")):
            row = encoded(f"sft-{kind}", (7, 8, 11) if kind == "B" else (7, 8), (12, 13),
                          ROUTE_LM if kind == "A1" else ROUTE_FULL, kind)
            row["problem_id"] = f"item-{i % 2}"
            rows.append(row)
        report = train.backward_single_block(bundle, rows, contract, components=("A2", "B"))
        self.assertEqual(set(report["sft"]), {"A2", "B"})
        self.assertEqual(report["sft"]["A2"]["weight"], .5)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in bundle.params["V"] if p.grad is not None), 0)
        for group in ("M", "L"):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in bundle.params[group] if p.grad is not None), 0)

    def test_all_components_keep_per_kind_row_means_across_partitions(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=2)
        expected = {}
        for kind in ("A1", "A2", "B"):
            nlls = []
            for i, target in enumerate(((12,), (12, 13, 14, 15))):
                route = ROUTE_FULL if i and kind != "A1" else ROUTE_LM
                row = encoded(f"sft-{kind}-{i}", (7, 8, 11) if kind == "B" else (7, 8), target, route, kind)
                row["problem_id"] = f"item-{i}"
                rows.append(row)
                bundle.route.apply(route == ROUTE_FULL)
                with torch.no_grad():
                    nlls.append(float(-train.batch_logprobs(bundle, [row])[0].mean()))
            expected[kind] = sum(nlls) / len(nlls)
        bundle.route.apply(False)
        first = train.backward_single_block(bundle, rows, contract)
        original_gradients = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        second = train.backward_single_block(bundle, rows, contract, token_budget=1000)
        for kind, nll in expected.items():
            self.assertAlmostEqual(first["sft"][kind]["row_nll_mean"], nll, places=6)
            self.assertAlmostEqual(second["sft"][kind]["weighted_loss"], nll * contract["sft_weights"][kind], places=6)
        self.assertAlmostEqual(first["loss"], second["loss"], places=6)
        for name, value in gradients(bundle).items():
            torch.testing.assert_close(value, original_gradients[name], atol=2e-6, rtol=2e-4)



class SingleV10Tests(unittest.TestCase):
    """v10: fixed per-source baseline advantage on every RL row, token scale 256, A2 weight per row."""

    def test_source_baseline_values_row_binding_and_tampered_table(self):
        import json
        import tempfile
        from pathlib import Path

        pin = single_baseline_contract()
        self.assertEqual(pin["table_sha256"], "74a6cc57589f0887fd01c0f8ce0dcc91babc312df8d7d4b272c94e3af8df2d45")
        for pid, value in (("hf-scienceqa:7", .7551), ("hf-clevr:3", -.026), ("hf-slake:a:b", -.2778),
                           ("hf-unlisted:1", .3002), ("authored-test-1", .3002)):
            self.assertEqual(single_source_baseline(pid, pin)["value"], value)
        rows, _ = block(tiny_factory(), count=1, reward=0)
        row = rows[0]
        row["problem_id"] = "hf-scienceqa:7"
        bind_reward_fixture(row, baseline=pin)
        self.assertEqual((row["reward"], row["baseline"], row["advantage"]), (0, .7551, -.7551))
        self.assertEqual(row["reward_evidence"]["baseline"]["source"], "hf-scienceqa")
        single.validate_single_reward(row)
        for field, value in (("advantage", .5), ("baseline", .3002)):
            forged = copy.deepcopy(row)
            forged[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "baseline/advantage"):
                single.validate_single_reward(forged)
        stripped = copy.deepcopy(row)
        del stripped["reward_evidence"]["baseline"]
        with self.assertRaisesRegex(ValueError, "baseline"):
            single.validate_single_reward(stripped)
        with tempfile.TemporaryDirectory() as directory:
            table = json.loads(protocol_module.SINGLE_BASELINE_TABLE.read_text())
            table["sources"]["hf-scienceqa"]["mean"] = 1.0
            tampered = Path(directory) / "source_baseline_v10.json"
            tampered.write_text(json.dumps(table))
            with patch.object(protocol_module, "SINGLE_BASELINE_TABLE", tampered), \
                    self.assertRaisesRegex(ValueError, "differs from the sealed baseline pin"):
                single.validate_single_reward(row)

    def test_zero_reward_rows_get_negative_advantage_pg_with_finite_gradients(self):
        owner = tiny_factory()
        rows, contract = block(owner, count=2, reward=0)
        old = train.backward_single_block(owner, rows, contract, components=("PG",))  # v9: b=0, scale 1024
        self.assertEqual((old["rl"]["pg"], old["rl"]["token_scale"], old["has_gradient"]), (0, 1024., False))
        self.assertNotIn("pg_rows", old["rl"])
        owner.model.zero_grad(set_to_none=True)
        contract = as_v10(rows, contract)
        b = .3002  # item-N is not a table source: global_mean
        self.assertEqual([(row["reward"], row["advantage"]) for row in rows], [(0, -b), (0, -b)])
        stats = train.backward_single_block(owner, rows, contract, components=("PG",))
        expected = sum(b * sum(row["generation"]["sampling_logprobs"]) for row in rows) / (2 * 256.)
        self.assertLess(expected, 0)
        self.assertAlmostEqual(stats["rl"]["pg"], expected, places=6)
        self.assertEqual((stats["rl"]["pg_rows"], stats["rl"]["token_scale"]), (2, 256.))
        self.assertAlmostEqual(stats["rl"]["advantage_mean"], -b, places=12)
        self.assertAlmostEqual(stats["rl"]["advantage_abs_mean"], b, places=12)
        self.assertTrue(stats["has_gradient"])
        grads = [p.grad for ps in owner.params.values() for p in ps if p.grad is not None]
        self.assertTrue(grads and all(bool(torch.isfinite(g).all()) for g in grads))
        for change in ({"token_scale": 1024.}, {"baseline": 0.0}):
            bad = copy.deepcopy(contract)
            bad["rl"].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "v10 source-baseline"):
                train.backward_single_block(owner, rows, bad, components=("PG",))
        # A changed shipped table no longer matches the sealed table SHA256: training refuses before any backward.
        with patch.object(train, "single_baseline_contract", return_value={"method": "fixed-source-mean-v1",
                          "table_sha256": "0" * 64, "unknown_source": "global_mean"}), \
                self.assertRaisesRegex(ValueError, "v10 source-baseline"):
            train.backward_single_block(owner, rows, contract, components=("PG",))

    def test_a2_rows_each_carry_the_full_per_row_weight_and_v9_keeps_per_kind(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=2)
        nlls = []
        bundle.route.apply(False)
        for i, target in enumerate(((12,), (12, 13, 14))):
            row = encoded(f"sft-A2-{i}", (7, 8), target, ROUTE_LM, "A2")
            row["problem_id"] = f"item-{i}"
            rows.append(row)
            with torch.no_grad():
                nlls.append(float(-train.batch_logprobs(bundle, [row])[0].mean()))
        for version, weight, coefficient in (("v9", .5, .25), ("v10", .35, .35)):
            if version == "v10":
                contract = as_v10(rows, contract)
            bundle.model.zero_grad(set_to_none=True)
            with patch.object(train, "_backward_rows", wraps=train._backward_rows) as backward:
                stats = train.backward_single_block(bundle, rows, contract, components=("A2",))
            with self.subTest(version=version):
                self.assertTrue(backward.call_args_list)
                for call in backward.call_args_list:
                    self.assertEqual(call.args[3], [coefficient, coefficient])
                self.assertEqual(stats["sft"]["A2"]["weight"], weight)
                total = sum(nlls) if version == "v10" else sum(nlls) / 2
                self.assertAlmostEqual(stats["sft"]["A2"]["weighted_loss"], weight * total, places=6)
        self.assertAlmostEqual(sum(call.args[3][0] for call in backward.call_args_list[:1]) * 2, .7)
        self.assertEqual(stats["sft"]["A2"]["weighting"], "per_row")


if __name__ == "__main__":
    unittest.main()
