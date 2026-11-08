"""SINGLE v11 CPU checks: penalty-table advantage, raw-distribution PG on negative rows, per-row token cap and the
dataset reference in the note prompt. Tiny random models only; these are not pretrained Qwen learning evidence."""
import copy
import unittest
from unittest.mock import patch

import torch

from fine_tuning.error_note import protocol as protocol_module
from fine_tuning.error_note import single
from fine_tuning.error_note import train_v4 as train
from fine_tuning.error_note.contract import (NOTE_INSTRUCTION_SINGLE, REFERENCE_HEADER, SINGLE_VERDICT_LABELS,
                                            note_prompt_single)
from fine_tuning.error_note.protocol import (SINGLE_ADVANTAGE_V11, SINGLE_REWARD_GATE, SINGLE_REWARD_GATE_V5,
                                             VERSION_SINGLE, plain_protocol_single, single_prompts_sha256)
from fine_tuning.notebook_rl.input_preparation import digest
from fine_tuning.tests import test_error_note_single_runtime as runtime_tests
from fine_tuning.tests.test_error_note_single_loss import (SAMPLING, bind_reward_fixture, block, gradients,
                                                           perturbed_factory)
from fine_tuning.tests.test_error_note_train import encoded, tiny_factory
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE

PINS = {key: "a" * 64 for key in ("source", "dataset", "order", "data_admission")}
_SHORT = None


def setUpModule():
    """Tiny-model rows are a few tokens long; the 32-token rule is exercised with patched lengths below."""
    global _SHORT
    _SHORT = patch.object(single, "SOLUTION_MIN_TOKENS", 1)
    _SHORT.start()


def tearDownModule():
    _SHORT.stop()


def as_v11(rows, contract, finish=None):
    """Rebind every RL row under gate v6 and turn the tiny v9 contract into the v11 block contract."""
    contract = copy.deepcopy(contract)
    v11 = plain_protocol_single("bf16", 1, contracts=PINS)["training_contract"]
    contract["rl"].update({key: v11["rl"][key] for key in
                           ("baseline", "advantage", "token_scale", "row_token_cap", "row_token_cap_exempt",
                            "negative_logprob", "reward")})
    contract.update(version=VERSION_SINGLE, sft_weights=v11["sft_weights"], sft_weighting=v11["sft_weighting"])
    for row in rows:
        if row["kind"] == "RL":
            bind_reward_fixture(row, finish_reason=(finish or {}).get(row["problem_id"], "stop"), gate=SINGLE_REWARD_GATE)
    contract.update(
        reward_provenance_sha256={r["problem_id"]: digest(r["reward_evidence"]["provenance"]) for r in rows if r["kind"] == "RL"},
        reward_evidence_sha256={r["problem_id"]: digest(r["reward_evidence"]) for r in rows if r["kind"] == "RL"})
    return contract


def trainable(bundle):
    return [p for ps in bundle.params.values() for p in ps if p.requires_grad]


def long_rows(bundle, lengths):
    """Negative RL rows of the given target lengths (end token included) with stored full-vocabulary sampling."""
    rows, contract = block(bundle, count=len(lengths), reward=0)
    vocab = bundle.model.get_base_model().lm_head.out_features
    for index, length in enumerate(lengths):
        tokens = tuple(9 + (i * 7 + index) % (vocab - 12) for i in range(length - 1))
        row = encoded(f"item-{index}:RL", (7, 8), tokens, "LM", "RL", seed=index)
        old = rows[index]
        row.update({key: old[key] for key in old if key not in row})
        actions = row["input_ids"][-row["target_len"]:].tolist()
        row["input_sha256"] = train._single_encoded_input_sha256(row)
        row["generation"] = dict(old["generation"], prompt_token_ids=row["input_ids"][:-row["target_len"]].tolist(),
                                 completion_token_ids=actions, input_sha256=row["input_sha256"],
                                 support_token_ids=[list(range(vocab))] * len(actions),
                                 support_sha256=[digest(list(range(vocab)))] * len(actions))
        rows[index] = row
    bundle.route.apply(False)
    with torch.no_grad():
        for row, value in zip(rows, train.single_batch_statistics(bundle, rows, SAMPLING, chunk=512), strict=True):
            row["generation"]["sampling_logprobs"] = value["logprobs"].tolist()
    return rows, contract


def pg_reference(bundle, rows, terms):
    """Manual v11 PG gradient: terms[i] = (key, coefficient) multiplies rows[i]'s summed log probabilities."""
    bundle.model.zero_grad(set_to_none=True)
    bundle.model.train()
    bundle.route.apply(False)
    values = train.single_batch_statistics(bundle, rows, SAMPLING)
    sum(coefficient * value[key].sum() for value, (key, coefficient) in zip(values, terms, strict=True)).backward()
    result = gradients(bundle)
    bundle.model.zero_grad(set_to_none=True)
    return result, values


class PenaltyTable(unittest.TestCase):
    def test_every_gate_v6_case_round_trips_and_correct_or_unverifiable_never_go_negative(self):
        rows, _ = block(tiny_factory(), count=1)
        long_ids, loop = list(range(100, 140)), "Wait, let me count again. " * 200
        reasoned = "The angle sum gives 32 degrees here. Final answer: 32"
        cases = (  # status, finish, ids, text, extracted, reward, penalty
            ("CORRECT", "stop", long_ids, reasoned, "32", 1, None),
            ("CORRECT", "stop", None, None, "B", 0, None),  # bare correct answer: 0, never negative
            ("CORRECT", "stop", long_ids, loop, "32", 0, None),
            ("CORRECT", "length", long_ids, reasoned, "32", 0, None),
            ("UNVERIFIABLE", "stop", None, None, None, 0, None),
            ("UNVERIFIABLE", "length", long_ids, loop, None, 0, None),
            ("INCORRECT", "stop", None, None, "A", -1, "BARE_WRONG"),
            ("INVALID_NO_FINAL", "stop", long_ids, reasoned, None, -1, "NO_COMMITTED_ANSWER"),
            ("INCORRECT", "stop", long_ids, loop, "A", -1, "LOOP"),
            ("TRUNCATED", "length", long_ids, loop, None, -1, "LOOP"),
            ("TRUNCATED", "length", long_ids, reasoned, None, -1, "TRUNCATED_NO_LOOP"),
            ("INCORRECT", "stop", long_ids, reasoned, "31", -1, "REASONED_WRONG"),
            ("INVALID_MULTIPLE", "stop", long_ids, reasoned, "31 or 33", -1, "REASONED_WRONG"),
            ("INVALID_NO_FINAL", "stop", long_ids, reasoned, "31", -1, "REASONED_WRONG"))
        seen = set()
        with patch.object(single, "SOLUTION_MIN_TOKENS", 32):
            for status, finish, ids, text, extracted, reward, penalty in cases:
                with self.subTest(status=status, finish=finish, penalty=penalty, text=text and text[:9]):
                    row = copy.deepcopy(rows[0])
                    row["status"] = status
                    if ids is not None:
                        row["generation"]["completion_token_ids"] = ids
                    if text is not None:
                        row["target"] = row["generation"]["text"] = text
                    t1 = {"text": row["target"], "generation": row["generation"], "prompt": row["prompt"],
                          "status": status, "finish_reason": finish, "extracted": extracted}
                    single.bind_single_reward(row, t1, PROVENANCE)
                    decision = row["reward_evidence"]["decision"]
                    value = SINGLE_ADVANTAGE_V11["accepted" if reward == 1 else penalty or "other"]
                    self.assertEqual((row["reward"], decision["penalty"], row["advantage"]), (reward, penalty, value))
                    self.assertEqual(row["reward_evidence"]["advantage"], {"version": "penalty-table-v1", "value": value})
                    self.assertEqual(single.validate_single_reward(row), decision)
                    self.assertEqual(single.single_advantage(decision), value)
                    if status in ("CORRECT", "UNVERIFIABLE"):
                        self.assertGreaterEqual(row["advantage"], 0)
                    seen.add("accepted" if reward == 1 else penalty or "other")
                    forged = copy.deepcopy(row)
                    forged["advantage"] = value - .25
                    with self.assertRaisesRegex(ValueError, "penalty table"):
                        single.validate_single_reward(forged)
        self.assertEqual(seen, set(SINGLE_ADVANTAGE_V11))
        # v5 evidence keeps v5 behaviour and may not carry a table advantage; the v10 baseline is gate v5 only.
        old = copy.deepcopy(rows[0])
        old["status"] = "TRUNCATED"
        bind_reward_fixture(old, finish_reason="length")
        self.assertEqual((old["reward"], old["reward_evidence"]["decision"]["penalty"]), (0, None))
        self.assertNotIn("advantage", old)
        old["reward_evidence"]["advantage"] = {"version": "penalty-table-v1", "value": 0.0}
        with self.assertRaisesRegex(ValueError, "gate v5"):
            single.validate_single_reward(old)
        with self.assertRaisesRegex(ValueError, "gate v5 only"):
            bind_reward_fixture(copy.deepcopy(rows[0]), baseline={}, gate=SINGLE_REWARD_GATE)


class RawNegativeGradient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_singleton_support_gives_zero_q_gradient_and_negative_rows_use_raw_p(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=1, reward=0)
        row = rows[0]
        actions = row["generation"]["completion_token_ids"]
        # The sampler kept one candidate at every position: log q = 0 there, so q-only PG had no gradient at all.
        row["generation"].update(support_token_ids=[[a] for a in actions], support_sha256=[digest([a]) for a in actions])
        bundle.route.apply(False)
        bundle.model.train()
        value = train.single_batch_statistics(bundle, [row], SAMPLING)[0]
        self.assertEqual(value["logprobs"].tolist(), [0.] * len(actions))
        q = torch.autograd.grad(value["logprobs"].sum(), trainable(bundle), allow_unused=True, retain_graph=True)
        self.assertTrue(all(g is None or float(g.abs().max()) == 0 for g in q))
        p = torch.autograd.grad(value["raw_logprobs"].sum(), trainable(bundle), allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in p if g is not None), 0)
        self.assertTrue(bool((value["raw_logprobs"] < 0).all()))  # the full vocabulary keeps other candidates
        row["generation"]["sampling_logprobs"] = value["logprobs"].detach().tolist()
        contract = as_v11(rows, contract)
        self.assertEqual((row["reward"], row["advantage"]), (-1, -.5))  # stopped, reasoned, wrong
        bundle.model.zero_grad(set_to_none=True)
        stats = train.backward_single_block(bundle, rows, contract, components=("PG",))
        self.assertTrue(stats["has_gradient"])
        self.assertEqual((stats["rl"]["pg_rows_negative"], stats["rl"]["pg_rows_positive"], stats["rl"]["capped_rows"]),
                         (1, 0, 0))
        self.assertEqual(stats["rl"]["pg_positive"], 0)
        actual = gradients(bundle)
        expected, values = pg_reference(bundle, rows, [("raw_logprobs", .5 / 1024.)])
        self.assertAlmostEqual(stats["rl"]["pg"], float(.5 * values[0]["raw_logprobs"].detach().sum() / 1024.), places=6)
        self.assertAlmostEqual(stats["rl"]["pg_negative"], stats["rl"]["pg"], places=12)
        for name, grad in expected.items():
            torch.testing.assert_close(actual[name], grad, atol=1e-8, rtol=1e-5)

    def test_row_cap_halves_a_4096_token_row_and_keeps_a_1000_token_row(self):
        bundle = tiny_factory()
        rows, contract = long_rows(bundle, (4096, 1000))  # target_len counts the end token
        contract = as_v11(rows, contract)
        self.assertEqual([(r["target_len"], r["advantage"]) for r in rows], [(4096, -.5), (1000, -.5)])
        bundle.model.zero_grad(set_to_none=True)
        stats = train.backward_single_block(bundle, rows, contract, components=("PG",))
        self.assertEqual((stats["rl"]["capped_rows"], stats["rl"]["cap_exempt_rows"], stats["rl"]["pg_rows_negative"]),
                         (1, 0, 2))
        actual = gradients(bundle)
        weights = [.5, 1.]  # min(1, 2048/4096), min(1, 2048/1000)
        expected, values = pg_reference(bundle, rows, [("raw_logprobs", .5 * w / (2 * 1024.)) for w in weights])
        self.assertAlmostEqual(stats["rl"]["pg"], sum(float(.5 * w * v["raw_logprobs"].detach().sum()) / 2048.
                                                      for w, v in zip(weights, values, strict=True)), places=5)
        for name, grad in expected.items():
            torch.testing.assert_close(actual[name], grad, atol=1e-8, rtol=1e-4)

    def test_loop_rows_are_exempt_from_the_row_cap(self):
        bundle = tiny_factory()
        rows, contract = long_rows(bundle, (8192, 8192))
        loop = "Wait, let me count again. " * 200
        rows[0]["status"], rows[0]["target"] = "TRUNCATED", loop
        rows[0]["generation"]["text"] = loop
        contract = as_v11(rows, contract, finish={rows[0]["problem_id"]: "length"})
        self.assertEqual([(r["target_len"], r["reward_evidence"]["decision"]["penalty"], r["advantage"]) for r in rows],
                         [(8192, "LOOP", -1.), (8192, "REASONED_WRONG", -.5)])
        self.assertEqual(contract["rl"]["row_token_cap_exempt"], ["LOOP"])
        bundle.model.zero_grad(set_to_none=True)
        stats = train.backward_single_block(bundle, rows, contract, components=("PG",))
        self.assertEqual((stats["rl"]["capped_rows"], stats["rl"]["cap_exempt_rows"], stats["rl"]["pg_rows_negative"]),
                         (1, 1, 2))
        actual = gradients(bundle)
        weights = [1., .25]  # LOOP keeps the full row; REASONED_WRONG gets min(1, 2048/8192)
        coefficients = [-a * w / (2 * 1024.) for a, w in zip((-1., -.5), weights, strict=True)]
        expected, values = pg_reference(bundle, rows, [("raw_logprobs", c) for c in coefficients])
        self.assertAlmostEqual(stats["rl"]["pg"], sum(float(c * v["raw_logprobs"].detach().sum())
                                                      for c, v in zip(coefficients, values, strict=True)), places=4)
        for name, grad in expected.items():
            torch.testing.assert_close(actual[name], grad, atol=1e-8, rtol=1e-4)
        for bad_exempt in ([], ["LOOP", "TRUNCATED_NO_LOOP"], None):
            bad = copy.deepcopy(contract)
            bad["rl"]["row_token_cap_exempt"] = bad_exempt
            with self.subTest(exempt=bad_exempt), self.assertRaisesRegex(ValueError, "LOOP exemption"):
                train.backward_single_block(bundle, rows, bad, components=("PG",))

    def test_zero_advantage_rows_add_no_pg_but_keep_kl_and_anchor_and_positive_rows_use_stored_q(self):
        bundle = perturbed_factory()
        rows, contract = block(bundle, count=2, reward=1)
        rows[1]["status"] = "UNVERIFIABLE"
        contract = as_v11(rows, contract)
        self.assertEqual([(r["reward"], r["advantage"]) for r in rows], [(1, 1.), (0, 0.)])
        zero_rows = [rows[1]]
        zero_contract = copy.deepcopy(contract) | {"expected_problem_ids": [rows[1]["problem_id"]]}
        for key in ("reward_provenance_sha256", "reward_evidence_sha256"):
            zero_contract[key] = {rows[1]["problem_id"]: contract[key][rows[1]["problem_id"]]}
        stats = train.backward_single_block(bundle, zero_rows, zero_contract, components=("PG",))
        self.assertEqual((stats["rl"]["pg"], stats["rl"]["pg_rows"], stats["has_gradient"]), (0, 1, False))
        self.assertEqual((stats["rl"]["pg_rows_positive"], stats["rl"]["pg_rows_negative"]), (0, 0))
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))
        kept = train.backward_single_block(bundle, zero_rows, zero_contract, components=("PG", "KL"))
        self.assertEqual(kept["rl"]["pg"], 0)
        self.assertGreater(kept["rl"]["anchor"], 0)
        self.assertTrue(kept["has_gradient"])
        bundle.model.zero_grad(set_to_none=True)
        mixed = train.backward_single_block(bundle, rows, contract, components=("PG",))
        expected = -sum(rows[0]["generation"]["sampling_logprobs"]) / (2 * 1024.)
        self.assertAlmostEqual(mixed["rl"]["pg"], expected, places=6)
        self.assertAlmostEqual(mixed["rl"]["pg_positive"], expected, places=6)
        self.assertEqual((mixed["rl"]["pg_rows_positive"], mixed["rl"]["pg_rows_negative"]), (1, 0))
        actual = gradients(bundle)
        reference, _ = pg_reference(bundle, rows, [("logprobs", -1 / (2 * 1024.)), ("logprobs", 0.)])
        for name, grad in reference.items():
            torch.testing.assert_close(actual[name], grad, atol=1e-8, rtol=1e-5)
        bad = copy.deepcopy(contract)
        bad["rl"]["row_token_cap"] = 1024
        with self.assertRaisesRegex(ValueError, "v11 penalty table"):
            train.backward_single_block(bundle, rows, bad, components=("PG",))

    def test_contract_version_and_row_gate_must_match(self):
        bundle = tiny_factory()
        rows, v9 = block(bundle, count=1, reward=0)  # gate v5 rows under a v9 contract
        v6_rows = copy.deepcopy(rows)
        v11 = as_v11(v6_rows, v9)
        with self.assertRaisesRegex(ValueError, "baseline/advantage binding"):  # v5 rows under the v11 contract
            train.backward_single_block(bundle, rows, v11 | {"reward_evidence_sha256": v9["reward_evidence_sha256"]},
                                        components=("PG",))
        with self.assertRaisesRegex(ValueError, "baseline/advantage binding"):  # v6 rows under the v9 contract
            train.backward_single_block(bundle, v6_rows, v9 | {"reward_evidence_sha256": v11["reward_evidence_sha256"]},
                                        components=("PG",))
        with self.assertRaisesRegex(ValueError, "reward contract required"):  # v9 cannot claim gate v6
            train.backward_single_block(bundle, v6_rows, v9 | {"rl": v9["rl"] | {"reward": SINGLE_REWARD_GATE}},
                                        components=("PG",))
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))


class ProtocolAndReference(unittest.TestCase):
    def test_protocol_v11_fields_and_prompt_hash_cover_the_reference_header(self):
        protocol = plain_protocol_single("bf16", 2, contracts=PINS)
        rl = protocol["training_contract"]["rl"]
        self.assertEqual((protocol["version"], protocol["training_contract"]["version"], rl["reward"]),
                         ("error-note-single-response-v11", VERSION_SINGLE, "single-t1-program-gate-v6"))
        self.assertEqual((rl["baseline"], rl["token_scale"], rl["row_token_cap"], rl["negative_logprob"], rl["beta"]),
                         ("none", 1024., 2048, "raw_full_vocab_log_softmax_T1_no_presence", .02))
        self.assertEqual(rl["row_token_cap_exempt"], ["LOOP"])
        self.assertEqual(rl["advantage"], {"version": "penalty-table-v1", "values": SINGLE_ADVANTAGE_V11})
        self.assertEqual(len(rl["plan_exceptions"]), 2)
        self.assertEqual(rl["format_anchor"], {"positions": 4, "beta": .5})
        self.assertEqual(protocol["training_contract"]["generation_counts"], {"t1": 1, "note": 1, "resolve": 1})
        self.assertEqual(protocol["training_contract"]["note_reference"], {"version": "none"})
        single._protocol_check(protocol)
        bound = {"version": "dataset-reference-v1", "fields_sha256": "b" * 64, "supplement_sha256": "c" * 64,
                 "max_chars": 4000}
        self.assertEqual(plain_protocol_single("bf16", 2, contracts=PINS, note_reference=bound)
                         ["training_contract"]["note_reference"], bound)
        for bad in ({}, {"version": 1}, "dataset-reference-v1"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "note_reference"):
                plain_protocol_single("bf16", 2, contracts=PINS, note_reference=bad)
        with self.assertRaisesRegex(ValueError, "1024"):
            plain_protocol_single("bf16", 2, contracts=PINS, token_scale=256.)
        before = single_prompts_sha256()
        with patch.object(protocol_module, "REFERENCE_HEADER", REFERENCE_HEADER + " "):
            self.assertNotEqual(single_prompts_sha256(), before)
        self.assertEqual(single_prompts_sha256(), before)

    def test_a_changed_reference_supplement_changes_the_protocol_identity(self):
        pin = {"version": "note-dataset-reference-v1", "fields_sha256": "b" * 64, "supplement_sha256": "c" * 64,
               "max_chars": 4000, "items_with_reference": 7}
        first = plain_protocol_single("bf16", 2, contracts=PINS, note_reference=pin)
        single._protocol_check(first)
        for change in ({"supplement_sha256": "d" * 64}, {"items_with_reference": 8}, {"fields_sha256": "e" * 64}):
            other = plain_protocol_single("bf16", 2, contracts=PINS, note_reference=pin | change)
            self.assertNotEqual(other["sha256"], first["sha256"], change)
        self.assertNotEqual(first["sha256"], plain_protocol_single("bf16", 2, contracts=PINS)["sha256"])

    def test_note_prompt_inserts_reference_after_the_correct_answer_only_when_nonblank(self):
        p0, raw, gold = "Question: exact P0", "Original prose\nFinal answer: 7", "8"
        plain = note_prompt_single(p0, raw, "INCORRECT", gold, read="7")
        layout = (f"{p0}\n\nYour earlier solution:\n{raw}\n\nGrading result: {SINGLE_VERDICT_LABELS['INCORRECT']}\n"
                  f"The grader read your final answer as: 7\nCorrect answer: {gold}\n\n")
        self.assertEqual(plain, layout + NOTE_INSTRUCTION_SINGLE)
        for blank in (None, "", "  \n "):
            self.assertEqual(note_prompt_single(p0, raw, "INCORRECT", gold, read="7", reference=blank), plain)
        self.assertNotIn(REFERENCE_HEADER, plain)
        text = "Lecture: area = length x width.\nSolution: 2 x 4 = 8."
        self.assertEqual(note_prompt_single(p0, raw, "INCORRECT", gold, read="7", reference=text),
                         layout + f"{REFERENCE_HEADER}\n{text}\n\n" + NOTE_INSTRUCTION_SINGLE)
        for sentence in ("If a dataset reference is shown, compare your earlier solution with it step by step and name "
                         "the first fact or step where they differ.", "Do not copy sentences from the reference."):
            self.assertEqual(NOTE_INSTRUCTION_SINGLE.count(sentence), 1)

    def test_run_single_block_reference_callable_reaches_only_the_stored_note_prompt(self):
        case = runtime_tests.RuntimeTests("test_program_gate_runs_without_pauses_keeps_originals_and_gold_only_in_note_request")
        case.setUp()
        try:
            asked = []

            def reference(pid):
                asked.append(pid)
                return f"Solution: the dataset explains item {pid}."

            result = case.run_block(reference=reference)
            self.assertEqual(result["status"], "ROWS_READY")
            self.assertEqual(asked, ["p1"])  # notes exist only for the wrong first solve
            shown = f"{REFERENCE_HEADER}\nSolution: the dataset explains item p1.\n\n"
            for request in case.seen:
                text = request["messages"][0]["content"][-1]["text"]
                self.assertEqual(shown in text, request["stage"] == "note", request["stage"])
            state = single._state(case.out / "phase.json")
            self.assertIn(shown, state["records"]["p1"]["note"]["prompt"])
            self.assertTrue(state["records"]["p1"]["note"]["prompt"].endswith(shown + NOTE_INSTRUCTION_SINGLE))
            closed = single._read(case.out / result["rows_path"])
            self.assertEqual([shown in r["prompt"] for r in closed["rows"]], [False, False, False, False, True])
        finally:
            case.doCleanups()


if __name__ == "__main__":
    unittest.main()
