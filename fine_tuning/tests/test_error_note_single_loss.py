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
from fine_tuning.error_note.protocol import SINGLE_REWARD_GATE, VERSION_SINGLE
from fine_tuning.notebook_rl.input_preparation import digest
from fine_tuning.tests.test_error_note_train import encoded, tiny_factory
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE

SAMPLING = {"temperature": .7, "top_p": 1., "top_k": 0, "presence_penalty": 1.5}


def gradients(bundle):
    return {name: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for name, p in bundle.model.named_parameters() if id(p) in {id(q) for ps in bundle.params.values() for q in ps}}


def bind_reward_fixture(row, *, finish_reason="stop"):
    for key, value in {"prompt": "Question: authored CPU fixture", "target": "Authored final answer: B", "target_source": "T1",
                       "image_paths": ["fixture.png"], "image_sha256": ["a" * 64], "subject": "Biology", "source": "authored-test"}.items():
        row.setdefault(key, value)
    row["generation"].setdefault("text", row["target"])
    t1 = {"text": row["target"], "generation": row["generation"], "prompt": row["prompt"], "status": row["status"],
          "finish_reason": finish_reason}
    return single.bind_single_reward(row, t1, PROVENANCE)


def full_support(bundle, actions):
    """top_k=0/top_p=1 fixtures sample from the whole vocabulary."""
    ids = list(range(bundle.model.get_base_model().lm_head.out_features))
    return [ids] * len(actions), [digest(ids)] * len(actions)


def pin_reward_fixture(rows, contract):
    contract.update(version=VERSION_SINGLE,
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
                "sampling": SAMPLING, "rl": {"token_scale": 1024., "beta": .02},
                "sft_weights": {"A1": .05, "A2": .5, "B": .05}}
    return rows, pin_reward_fixture(rows, contract)


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
        for status, reward in (("CORRECT", 1), ("INCORRECT", 0), ("TRUNCATED", 0), ("INVALID_NO_FINAL", 0)):
            row = copy.deepcopy(rows[0])
            row["status"] = status
            original = copy.deepcopy(row["generation"])
            bind_reward_fixture(row)
            self.assertEqual(row["reward"], reward)
            self.assertEqual(single.validate_single_reward(row)["accepted"], bool(reward))
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
        for version in (None, "error-note-single-response-v1", "error-note-single-response-v2"):
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


if __name__ == "__main__":
    unittest.main()
