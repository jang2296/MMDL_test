"""SINGLE v13 CPU checks: spiral window, reward gate v8 (praise + window), advantage table v3, windowed PG, the v13
train contract and the runtime A1 praise rule. Tiny random models only; these are not pretrained Qwen learning evidence."""
import copy
import json
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.error_note import single
from fine_tuning.error_note import train_v4 as train
from fine_tuning.error_note.contract import HUMAN_TARGET_SOURCE, KIND_HUMAN, LESSON_LEAK_VERSION_V2
from fine_tuning.error_note.protocol import (SINGLE_ADVANTAGE_V12, SINGLE_ADVANTAGE_V13, SINGLE_ADVANTAGE_VERSION_V13,
                                             SINGLE_PRAISE_V13, SINGLE_REWARD_GATE_V7, SINGLE_REWARD_GATE_V8,
                                             SINGLE_SFT_WEIGHTING_V10, SINGLE_SPIRAL_V13, VERSION_SINGLE,
                                             VERSION_SINGLE_V12, VERSION_SINGLE_V13, plain_protocol_single, spiral_window)
from fine_tuning.notebook_rl.input_preparation import digest
from fine_tuning.tests import test_error_note_single_runtime as runtime_tests
from fine_tuning.tests import test_error_note_single_v12 as v12
from fine_tuning.tests.test_error_note_single_loss import SAMPLING, block, gradients
from fine_tuning.tests.test_error_note_single_v11 import long_rows
from fine_tuning.tests.test_error_note_train import encoded, tiny_factory, tiny_processor
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE

PINS = v12.PINS
MARKERS = SINGLE_SPIRAL_V13["marker_token_ids"]  # the sealed Qwen3-VL IDs of "Wait" and " Wait"
LOOP = "Wait, let me count again. " * 200
REASONED = "The angle sum gives 32 degrees here. Final answer: 32"
INSIDE = "the angle is 32 degrees"
_SHORT = None


def setUpModule():
    """Tiny-model rows are a few tokens long; the 32-token rule is exercised with patched lengths below."""
    global _SHORT
    _SHORT = patch.object(single, "SOLUTION_MIN_TOKENS", 1)
    _SHORT.start()


def tearDownModule():
    _SHORT.stop()


def ramp(count):
    """`count` distinct token IDs (no marker, no repetition)."""
    return list(range(100, 100 + count))


def protocol_v13():
    return plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V13)


def bind_v13(row, **kwargs):
    return v12.bind_v12(row, gate=SINGLE_REWARD_GATE_V8, **kwargs)


def case_row(base, *, status, finish="stop", ids=None, text=None, extracted=None, provenance=v12.GOLD_32):
    """A copy of one fixture RL row graded as `status`, bound under gate v8 (open item)."""
    row = copy.deepcopy(base)
    row["status"] = status
    if ids is not None:
        row["generation"]["completion_token_ids"] = ids
    if text is not None:
        row["target"] = row["generation"]["text"] = text
    return bind_v13(row, finish_reason=finish, extracted=extracted, provenance=provenance)


def as_v13(rows, contract, finish=None):
    """Rebind every RL row under gate v8 and turn the tiny v9 contract into the v13 block contract (no human_solution)."""
    contract = copy.deepcopy(contract)
    v13 = protocol_v13()["training_contract"]
    contract["rl"].update({key: v13["rl"][key] for key in
                           ("baseline", "advantage", "token_scale", "row_token_cap", "row_token_cap_exempt",
                            "negative_logprob", "reward", "spiral_window", "praise")})
    contract.update(version=VERSION_SINGLE_V13, sft_weights=v13["sft_weights"], sft_weighting=v13["sft_weighting"])
    for row in rows:
        if row["kind"] == "RL":
            bind_v13(row, finish_reason=(finish or {}).get(row["problem_id"], "stop"))
    contract.update(
        reward_provenance_sha256={r["problem_id"]: digest(r["reward_evidence"]["provenance"]) for r in rows if r["kind"] == "RL"},
        reward_evidence_sha256={r["problem_id"]: digest(r["reward_evidence"]) for r in rows if r["kind"] == "RL"})
    return contract


def raw_logprobs(bundle, rows):
    """Raw full-vocabulary log p of every sampled token, one tensor per row (no gradient)."""
    bundle.model.train()
    bundle.route.apply(False)
    with torch.no_grad():
        return [value["raw_logprobs"] for value in train.single_batch_statistics(bundle, rows, SAMPLING)]


def random_rows(bundle, lengths):
    """Like test_error_note_single_v11.long_rows but CORRECT rows whose tokens are a seeded random draw, so they pass
    the repetition rules and can be accepted (target_len counts the end token)."""
    rows, contract = block(bundle, count=len(lengths), reward=1)
    vocab = bundle.model.get_base_model().lm_head.out_features
    for index, length in enumerate(lengths):
        tokens = tuple(random.Random(index).choices(range(9, vocab - 3), k=length - 1))
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


class SpiralWindow(unittest.TestCase):
    def ids(self, count, markers):
        """`count` ramp IDs with a marker token at each (index, marker id) of `markers`."""
        ids = ramp(count)
        for index, token in markers:
            ids[index] = token
        return ids

    def test_start_is_the_third_marker_counting_both_marker_token_ids(self):
        ids = self.ids(3000, [(10, MARKERS[0]), (50, MARKERS[1]), (100, MARKERS[0]), (200, MARKERS[1])])
        self.assertEqual(spiral_window(ids), [100, 2148])
        ids = self.ids(3000, [(10, MARKERS[1]), (50, MARKERS[1]), (60, MARKERS[1])])
        self.assertEqual(spiral_window(ids), [60, 2108])

    def test_end_is_capped_at_the_row_length(self):
        self.assertEqual(spiral_window(self.ids(500, [(1, MARKERS[0]), (2, MARKERS[0]), (300, MARKERS[0])])), [300, 500])
        self.assertEqual(spiral_window(self.ids(300, [(0, MARKERS[0]), (1, MARKERS[0]), (299, MARKERS[0])])), [299, 300])

    def test_window_is_at_most_2048_tokens(self):
        ids = self.ids(9000, [(5, MARKERS[0]), (6, MARKERS[0]), (7, MARKERS[0])])
        self.assertEqual(spiral_window(ids), [7, 7 + 2048])

    def test_fewer_than_three_markers_start_at_1024_or_half_the_row(self):
        self.assertEqual(spiral_window(ramp(5000)), [1024, 3072])
        self.assertEqual(spiral_window(ramp(3000)), [1024, 3000])
        self.assertEqual(spiral_window(ramp(2048)), [1024, 2048])
        self.assertEqual(spiral_window(ramp(500)), [250, 500])
        self.assertEqual(spiral_window(ramp(1)), [0, 1])
        self.assertEqual(spiral_window(self.ids(500, [(3, MARKERS[0]), (9, MARKERS[1])])), [250, 500])  # two markers only

    def test_a_non_marker_id_never_counts(self):
        self.assertEqual(spiral_window(self.ids(500, [(3, MARKERS[0]), (4, MARKERS[0]), (5, 13825)])), [250, 500])

    def test_empty_or_non_integer_input_is_refused(self):
        for label, bad in (("empty", []), ("none", None), ("string", "abc"), ("tuple", (1, 2, 3)), ("float", [1, 2.0]),
                           ("bool", [1, True]), ("text id", [1, "2"])):
            with self.subTest(label), self.assertRaisesRegex(ValueError, "spiral window needs the sampled token IDs"):
                spiral_window(bad)

    def test_the_sealed_rule_is_the_designed_one(self):
        self.assertEqual(SINGLE_SPIRAL_V13, {"version": "spiral-window-v1", "marker": "Wait", "marker_token_ids": [13824, 14190],
                                             "marker_count": 3, "fallback_start": 1024, "window_tokens": 2048,
                                             "penalties": ["LOOP", "TRUNCATED_NO_LOOP"], "logprob": "raw", "row_weight": 1.0})
        self.assertEqual(SINGLE_PRAISE_V13, {"version": "praise-v1", "min_tokens": 256, "applies_to": ["PG", "A1"]})


class AdvantageTableV13(v12.TinyThreads):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.base = block(tiny_factory(), count=1)[0][0]

    def test_the_table_values(self):
        self.assertEqual(SINGLE_ADVANTAGE_V13, {"accepted_praised": 1.0, "accepted": 0.0, "BARE_WRONG": -1.0,
                                                "NO_COMMITTED_ANSWER": -1.0, "LOOP": -1.0, "TRUNCATED_NO_LOOP": -1.0,
                                                "REASONED_WRONG": 0.0, "CONTAINED_OPEN": 0.0, "other": 0.0})
        self.assertEqual(SINGLE_ADVANTAGE_VERSION_V13, "penalty-table-v3")
        self.assertEqual(SINGLE_REWARD_GATE_V8, "single-t1-program-gate-v8")

    def test_every_gate_v8_case_round_trips_with_praise_window_and_the_forgery_checks(self):
        cases = (  # status, finish, ids, text, extracted, reward, penalty, table key, advantage, praise, window
            ("CORRECT", "stop", ramp(256), REASONED, "32", 1, None, "accepted_praised", 1.0, True, None),
            ("CORRECT", "stop", ramp(255), REASONED, "32", 1, None, "accepted", 0.0, False, None),
            ("CORRECT", "stop", ramp(40), REASONED, "32", 1, None, "accepted", 0.0, False, None),  # reward stays 1
            ("CORRECT", "stop", None, None, "32", 0, None, "other", 0.0, False, None),  # bare correct: not accepted
            ("CORRECT", "stop", ramp(256), LOOP, "32", 0, None, "other", 0.0, False, None),
            ("CORRECT", "length", ramp(256), REASONED, "32", 0, None, "other", 0.0, False, None),
            ("UNVERIFIABLE", "stop", None, None, None, 0, None, "other", 0.0, False, None),
            ("UNVERIFIABLE", "length", ramp(40), LOOP, None, 0, None, "other", 0.0, False, None),
            ("INCORRECT", "stop", None, None, "A", -1, "BARE_WRONG", "BARE_WRONG", -1.0, False, None),
            ("INVALID_NO_FINAL", "stop", ramp(40), REASONED, None, -1, "NO_COMMITTED_ANSWER", "NO_COMMITTED_ANSWER", -1.0,
             False, None),
            ("INCORRECT", "stop", ramp(40), LOOP, "A", -1, "LOOP", "LOOP", -1.0, False, [20, 40]),
            ("TRUNCATED", "length", ramp(40), LOOP, None, -1, "LOOP", "LOOP", -1.0, False, [20, 40]),
            ("TRUNCATED", "length", ramp(40), REASONED, None, -1, "TRUNCATED_NO_LOOP", "TRUNCATED_NO_LOOP", -1.0, False,
             [20, 40]),  # v12: -0.5
            ("TRUNCATED", "length", ramp(3000), REASONED, None, -1, "TRUNCATED_NO_LOOP", "TRUNCATED_NO_LOOP", -1.0, False,
             [1024, 3000]),
            ("TRUNCATED", "length", ramp(5000), LOOP, None, -1, "LOOP", "LOOP", -1.0, False, [1024, 3072]),
            ("INCORRECT", "stop", ramp(40), REASONED, "31", -1, "REASONED_WRONG", "REASONED_WRONG", 0.0, False, None),
            ("INVALID_MULTIPLE", "stop", ramp(40), REASONED, "31 or 33", -1, "REASONED_WRONG", "REASONED_WRONG", 0.0, False,
             None),
            ("INVALID_NO_FINAL", "stop", ramp(40), REASONED, "31", -1, "REASONED_WRONG", "REASONED_WRONG", 0.0, False, None),
            ("INCORRECT", "stop", ramp(40), REASONED, INSIDE, 0, None, "CONTAINED_OPEN", 0.0, False, None),
            ("INCORRECT", "stop", None, None, INSIDE, 0, None, "CONTAINED_OPEN", 0.0, False, None),
            ("INCORRECT", "length", ramp(40), LOOP, INSIDE, -1, "LOOP", "LOOP", -1.0, False, [20, 40]),
            ("INCORRECT", "stop", ramp(40), LOOP, INSIDE, -1, "LOOP", "LOOP", -1.0, False, [20, 40]),
            ("INCORRECT", "length", ramp(40), REASONED, INSIDE, -1, "TRUNCATED_NO_LOOP", "TRUNCATED_NO_LOOP", -1.0, False,
             [20, 40]))
        seen = set()
        with patch.object(single, "SOLUTION_MIN_TOKENS", 32):
            for status, finish, ids, text, extracted, reward, penalty, key, advantage, praise, window in cases:
                with self.subTest(status=status, finish=finish, key=key, n=ids and len(ids), text=text and text[:9]):
                    row = case_row(self.base, status=status, finish=finish, ids=ids, text=text, extracted=extracted)
                    decision = row["reward_evidence"]["decision"]
                    self.assertEqual((row["reward"], decision["penalty"], row["advantage"]), (reward, penalty, advantage))
                    self.assertIs(type(row["reward"]), int)
                    self.assertIs(type(row["advantage"]), float)
                    self.assertEqual(SINGLE_ADVANTAGE_V13[key], advantage)
                    self.assertEqual((decision["praise"], decision["window"]), (praise, window))
                    self.assertEqual(decision["reason"] == single.CONTAINED_OPEN, key == "CONTAINED_OPEN")
                    self.assertEqual(row["reward_evidence"]["version"], SINGLE_REWARD_GATE_V8)
                    self.assertEqual(row["reward_evidence"]["advantage"], {"version": "penalty-table-v3", "value": advantage})
                    self.assertEqual(row["reward_evidence"]["item"], {"option_count": 0})
                    self.assertEqual(single.validate_single_reward(row), decision)
                    self.assertEqual(single.single_advantage(decision, SINGLE_REWARD_GATE_V8), advantage)
                    seen.add(key)
                    # Gate v8 is gate v7 plus the two fields: nothing else about the decision moved.
                    v7 = v12.case_row(self.base, status=status, finish=finish, ids=ids, text=text, extracted=extracted)
                    self.assertEqual({k: v for k, v in decision.items() if k not in ("praise", "window")},
                                     v7["reward_evidence"]["decision"])
                    self.assertEqual(row["reward"], v7["reward"])
                    forged = copy.deepcopy(row)
                    forged["advantage"] = advantage - .25
                    with self.assertRaisesRegex(ValueError, "v13 penalty table"):
                        single.validate_single_reward(forged)
                    forged = copy.deepcopy(row)
                    forged["reward_evidence"]["advantage"]["value"] = advantage - .25
                    with self.assertRaisesRegex(ValueError, "v13 penalty table"):
                        single.validate_single_reward(forged)
                    edits = ({"reason": "T1_FORGED"}, {"penalty": "LOOP" if penalty != "LOOP" else None},
                             {"accepted": not decision["accepted"]}, {"praise": not decision["praise"]},
                             {"window": [0, 1]} if window is None else {"window": None},
                             {"window": [1, 2]} if window is None else {"window": [window[0] + 1, window[1]]})
                    for edit in edits:
                        forged = copy.deepcopy(row)
                        forged["reward_evidence"]["decision"].update(edit)
                        with self.subTest(edit=edit), self.assertRaisesRegex(ValueError, "differs from the program gate"):
                            single.validate_single_reward(forged)
                    for dropped in ("praise", "window"):
                        forged = copy.deepcopy(row)
                        del forged["reward_evidence"]["decision"][dropped]
                        with self.subTest(dropped=dropped), self.assertRaisesRegex(ValueError, "differs from the program gate"):
                            single.validate_single_reward(forged)
                    forged = copy.deepcopy(row)
                    forged["reward_evidence"]["version"] = SINGLE_REWARD_GATE_V7  # a v8 decision under the v7 gate name
                    with self.assertRaisesRegex(ValueError, "differs from the program gate"):
                        single.validate_single_reward(forged)
                    forged = copy.deepcopy(row)
                    forged["reward_evidence"]["item"]["option_count"] = 3  # an MCQ count against an open P0 text
                    with self.assertRaisesRegex(ValueError, "option count"):
                        single.validate_single_reward(forged)
        self.assertEqual(seen, set(SINGLE_ADVANTAGE_V13))

    def test_gate_v7_decisions_carry_neither_praise_nor_window(self):
        for status, finish, ids, text in (("CORRECT", "stop", ramp(256), REASONED), ("TRUNCATED", "length", ramp(40), REASONED),
                                          ("TRUNCATED", "length", ramp(40), LOOP)):
            with self.subTest(status=status, text=text[:9]):
                row = v12.case_row(self.base, status=status, finish=finish, ids=ids, text=text, extracted="32")
                self.assertEqual(row["reward_evidence"]["version"], SINGLE_REWARD_GATE_V7)
                self.assertTrue({"praise", "window"}.isdisjoint(row["reward_evidence"]["decision"]))
                self.assertEqual(row["reward_evidence"]["advantage"]["version"], "penalty-table-v2")
        # The v12 table is unchanged: an accepted row is +1 at any length, a cut-off without a loop -0.5.
        accepted = v12.case_row(self.base, status="CORRECT", ids=ramp(40), text=REASONED, extracted="32")
        self.assertEqual(accepted["advantage"], SINGLE_ADVANTAGE_V12["accepted"])
        self.assertEqual(SINGLE_ADVANTAGE_V12["TRUNCATED_NO_LOOP"], -.5)

    def test_praise_starts_at_exactly_256_tokens_and_needs_acceptance(self):
        for count, praise in ((255, False), (256, True), (257, True), (1, False)):
            with self.subTest(count=count):
                row = case_row(self.base, status="CORRECT", ids=ramp(count), text=REASONED, extracted="32")
                decision = row["reward_evidence"]["decision"]
                self.assertEqual((decision["accepted"], decision["praise"], row["reward"]), (True, praise, 1))
                self.assertEqual(row["advantage"], 1.0 if praise else 0.0)
        with patch.dict(SINGLE_PRAISE_V13, min_tokens=40):  # the threshold is read from the sealed constant
            row = case_row(self.base, status="CORRECT", ids=ramp(40), text=REASONED, extracted="32")
            self.assertEqual((row["reward_evidence"]["decision"]["praise"], row["advantage"]), (True, 1.0))
        # A long answer that is not accepted (a loop, or cut off) is never praised.
        for finish, text in (("stop", LOOP), ("length", REASONED)):
            row = case_row(self.base, status="CORRECT", finish=finish, ids=ramp(400), text=text, extracted="32")
            self.assertEqual((row["reward_evidence"]["decision"]["accepted"], row["reward_evidence"]["decision"]["praise"]),
                             (False, False))

    def test_window_follows_the_markers_inside_the_sampled_ids(self):
        ids = ramp(3000)
        for index in (10, 50, 100):
            ids[index] = MARKERS[index % 2]
        row = case_row(self.base, status="TRUNCATED", finish="length", ids=ids, text=REASONED)
        decision = row["reward_evidence"]["decision"]
        self.assertEqual((decision["penalty"], decision["window"]), ("TRUNCATED_NO_LOOP", [100, 2148]))
        self.assertEqual(single.validate_single_reward(row), decision)
        row["generation"]["completion_token_ids"] = ramp(3000)  # other IDs: the sealed window no longer matches
        with self.assertRaisesRegex(ValueError, "differs from the program gate"):
            single.validate_single_reward(row)


class WindowedPg(v12.TinyThreads):
    def test_window_pg_value_and_gradient_follow_the_sealed_window_only(self):
        bundle = tiny_factory()
        rows, contract = long_rows(bundle, (10, 10))
        rows[0]["status"] = "TRUNCATED"
        contract = as_v13(rows, contract, finish={rows[0]["problem_id"]: "length"})
        self.assertEqual([(r["advantage"], r["reward_evidence"]["decision"]["penalty"], r["reward_evidence"]["decision"]["window"])
                          for r in rows], [(-1.0, "TRUNCATED_NO_LOOP", [5, 10]), (0.0, "REASONED_WRONG", None)])
        bundle.model.zero_grad(set_to_none=True)
        stats = train.backward_single_block(bundle, rows, contract, components=("PG",))
        actual = gradients(bundle)
        rl = stats["rl"]
        self.assertEqual((rl["window_rows"], rl["window_tokens"]), (1, 5))
        self.assertEqual((rl["pg_rows"], rl["pg_rows_negative"], rl["pg_rows_positive"], rl["capped_rows"]), (2, 1, 0, 0))
        raw = raw_logprobs(bundle, rows)
        window_value = float(raw[0][5:10].sum()) / (2 * 1024.)  # -A * 1.0 * sum / (B * token_scale), A = -1
        self.assertAlmostEqual(rl["pg_negative"], window_value, delta=1e-5)
        self.assertAlmostEqual(rl["pg"], window_value, delta=1e-5)
        self.assertLess(window_value, 0)
        self.assertGreater(abs(float(raw[0].sum()) / (2 * 1024.) - window_value), 1e-4)  # not the whole-row value
        references = {}
        for label, taken in (("window", slice(5, 10)), ("whole", slice(0, 10))):
            bundle.model.zero_grad(set_to_none=True)
            bundle.model.train()
            bundle.route.apply(False)
            (train.single_batch_statistics(bundle, rows, SAMPLING)[0]["raw_logprobs"][taken].sum() / (2 * 1024.)).backward()
            references[label] = gradients(bundle)
        for name, grad in references["window"].items():
            torch.testing.assert_close(actual[name], grad, atol=1e-8, rtol=1e-4)
        self.assertTrue(any(not torch.allclose(actual[name], grad, atol=1e-8, rtol=1e-4)
                            for name, grad in references["whole"].items()))

    def test_loop_and_cut_off_rows_both_read_their_own_window_and_nothing_else(self):
        bundle = tiny_factory()
        rows, contract = long_rows(bundle, (10, 10))
        rows[0]["status"], rows[0]["target"], rows[0]["generation"]["text"] = "TRUNCATED", LOOP, LOOP
        rows[1]["status"] = "TRUNCATED"
        contract = as_v13(rows, contract, finish={row["problem_id"]: "length" for row in rows})
        self.assertEqual([(r["reward_evidence"]["decision"]["penalty"], r["advantage"]) for r in rows],
                         [("LOOP", -1.0), ("TRUNCATED_NO_LOOP", -1.0)])
        bundle.model.zero_grad(set_to_none=True)
        stats = train.backward_single_block(bundle, rows, contract, components=("PG",))
        rl = stats["rl"]
        self.assertEqual((rl["window_rows"], rl["window_tokens"], rl["pg_rows_negative"]), (2, 10, 2))
        raw = raw_logprobs(bundle, rows)
        self.assertAlmostEqual(rl["pg_negative"], float(raw[0][5:10].sum() + raw[1][5:10].sum()) / (2 * 1024.), delta=1e-5)

    def test_marker_window_from_a_patched_rule_is_a_strict_subrange_of_the_row(self):
        bundle = tiny_factory()
        rows, contract = long_rows(bundle, (10, 10))
        ids = rows[0]["generation"]["completion_token_ids"]
        rows[0]["status"] = "TRUNCATED"
        # Third marker at index 6 (the first ten tokens are distinct), window cut to 3 tokens.
        with patch.dict(SINGLE_SPIRAL_V13, marker_token_ids=[ids[2], ids[4], ids[6]], window_tokens=3):
            contract = as_v13(rows, contract, finish={rows[0]["problem_id"]: "length"})
            self.assertEqual(rows[0]["reward_evidence"]["decision"]["window"], [6, 9])
            self.assertEqual(contract["rl"]["spiral_window"]["window_tokens"], 3)
            bundle.model.zero_grad(set_to_none=True)
            stats = train.backward_single_block(bundle, rows, contract, components=("PG",))
        self.assertEqual((stats["rl"]["window_rows"], stats["rl"]["window_tokens"]), (1, 3))
        raw = raw_logprobs(bundle, rows)
        self.assertAlmostEqual(stats["rl"]["pg_negative"], float(raw[0][6:9].sum()) / (2 * 1024.), delta=1e-5)
        self.assertGreater(abs(float(raw[0].sum()) / (2 * 1024.) - stats["rl"]["pg_negative"]), 1e-4)
        # The contract built under the patch is refused once the constant is back (the sealed rule is the constant).
        with self.assertRaisesRegex(ValueError, "single train: v13"):
            train.backward_single_block(bundle, rows, contract, components=("PG",))

    def test_a_windowed_row_longer_than_the_row_cap_keeps_full_mass(self):
        bundle = tiny_factory()
        rows, contract = long_rows(bundle, (4096,))
        rows[0]["status"] = "TRUNCATED"
        contract = as_v13(rows, contract, finish={rows[0]["problem_id"]: "length"})
        self.assertEqual(rows[0]["reward_evidence"]["decision"]["window"], [1024, 3072])
        bundle.model.zero_grad(set_to_none=True)
        stats = train.backward_single_block(bundle, rows, contract, components=("PG",))
        rl = stats["rl"]
        self.assertEqual((rl["window_rows"], rl["window_tokens"], rl["capped_rows"], rl["pg_rows_negative"]), (1, 2048, 0, 1))
        raw = raw_logprobs(bundle, rows)[0]
        self.assertAlmostEqual(rl["pg_negative"], float(raw[1024:3072].sum()) / 1024., delta=1e-3)
        self.assertGreater(abs(rl["pg_negative"] - float(raw.sum()) / 1024.), 1.)  # not the whole row either

    def test_praised_row_gets_the_stored_q_term_and_an_unpraised_accepted_row_none(self):
        bundle = tiny_factory()
        rows, contract = random_rows(bundle, (256, 255))
        contract = as_v13(rows, contract)
        self.assertEqual([(r["reward"], r["advantage"], r["reward_evidence"]["decision"]["praise"],
                           r["reward_evidence"]["decision"]["window"]) for r in rows],
                         [(1, 1.0, True, None), (1, 0.0, False, None)])
        bundle.model.zero_grad(set_to_none=True)
        stats = train.backward_single_block(bundle, rows, contract, components=("PG",))
        rl = stats["rl"]
        self.assertEqual((rl["window_rows"], rl["window_tokens"]), (0, 0))
        self.assertEqual((rl["pg_rows"], rl["pg_rows_positive"], rl["pg_rows_negative"], rl["capped_rows"]), (2, 1, 0, 0))
        expected = -sum(rows[0]["generation"]["sampling_logprobs"]) / (2 * 1024.)
        self.assertAlmostEqual(rl["pg_positive"], expected, delta=1e-4)
        self.assertTrue(stats["has_gradient"])


class TrainContractV13(v12.TinyThreads):
    def setUp(self):
        self.bundle = tiny_factory()
        self.rows, base = block(self.bundle, count=1, reward=0)
        self.contract = as_v13(self.rows, base)

    def refused(self, contract, pattern="single train: v13", rows=None, components=("PG",)):
        with self.assertRaisesRegex(ValueError, pattern):
            train.backward_single_block(self.bundle, self.rows if rows is None else rows, contract, components=components)
        self.assertTrue(all(p.grad is None for ps in self.bundle.params.values() for p in ps))

    def test_a_valid_v13_contract_runs(self):
        stats = train.backward_single_block(self.bundle, self.rows, self.contract, components=("PG", "KL"))
        self.assertEqual((stats["rl"]["window_rows"], stats["rl"]["window_tokens"]), (0, 0))

    def test_a_contract_with_a_human_solution_pin_is_refused(self):
        v12_contract = plain_protocol_single("bf16", 1, contracts=PINS, version=VERSION_SINGLE_V12)["training_contract"]
        self.refused(self.contract | {"human_solution": v12_contract["human_solution"]})
        self.refused(self.contract | {"human_solution": {}})

    def test_a_contract_without_the_spiral_window_or_praise_rule_is_refused(self):
        for key in ("spiral_window", "praise"):
            with self.subTest(missing=key):
                bad = copy.deepcopy(self.contract)
                del bad["rl"][key]
                self.refused(bad)

    def test_a_contract_that_changes_the_window_or_praise_rule_or_keeps_the_v12_table_is_refused(self):
        rl = self.contract["rl"]
        for label, change in (
                ("window tokens", {"spiral_window": rl["spiral_window"] | {"window_tokens": 1024}}),
                ("window fallback", {"spiral_window": rl["spiral_window"] | {"fallback_start": 512}}),
                ("window markers", {"spiral_window": rl["spiral_window"] | {"marker_token_ids": [13824]}}),
                ("praise threshold", {"praise": rl["praise"] | {"min_tokens": 128}}),
                ("v12 table", {"advantage": {"version": "penalty-table-v2", "values": SINGLE_ADVANTAGE_V12}}),
                ("v13 values under the v12 name", {"advantage": {"version": "penalty-table-v2",
                                                                 "values": SINGLE_ADVANTAGE_V13}}),
                ("changed value", {"advantage": {"version": SINGLE_ADVANTAGE_VERSION_V13,
                                                 "values": SINGLE_ADVANTAGE_V13 | {"TRUNCATED_NO_LOOP": -.5}}})):
            with self.subTest(label):
                self.refused(self.contract | {"rl": rl | change})

    def test_a_contract_with_the_v12_weights_or_the_v7_gate_is_refused(self):
        v12_weights = plain_protocol_single("bf16", 1, contracts=PINS, version=VERSION_SINGLE_V12)["training_contract"]
        self.refused(self.contract | {"sft_weights": v12_weights["sft_weights"]}, pattern="single train")
        self.refused(self.contract | {"rl": self.contract["rl"] | {"reward": SINGLE_REWARD_GATE_V7}},
                     pattern="reward contract required")

    def test_gate_v7_rows_under_a_v13_contract_are_refused(self):
        v7_rows = copy.deepcopy(self.rows)
        for row in v7_rows:
            v12.bind_v12(row)
        contract = self.contract | {"reward_evidence_sha256": {r["problem_id"]: digest(r["reward_evidence"]) for r in v7_rows}}
        self.refused(contract, pattern="binding differs from the contract version", rows=v7_rows)

    def test_an_h_kind_row_in_a_v13_block_is_refused(self):
        h = {"row_id": "item-0:H", "kind": KIND_HUMAN, "route": "LM", "problem_id": "item-0", "complete": True,
             "target_source": HUMAN_TARGET_SOURCE, "prompt": "question a a", "target": " ".join(["a", "b"] * 20),
             "image_paths": ["x.png"], "image_sha256": ["a" * 64], "subject": "s", "source": "authored-test"}
        encoded_h = train.encode_single_row(tiny_processor(), SimpleNamespace(image_token_id=3), h, [Image.new("RGB", (16, 16))])
        self.refused(self.contract, pattern="invalid kind", rows=self.rows + [encoded_h], components=("H",))

    def test_a_tampered_window_praise_or_advantage_on_a_stored_row_is_refused_at_train_time(self):
        rows, contract = long_rows(self.bundle, (10,))
        rows[0]["status"] = "TRUNCATED"
        contract = as_v13(rows, contract, finish={rows[0]["problem_id"]: "length"})
        for label, edit in (("window", lambda r: r["reward_evidence"]["decision"].update(window=[0, 10])),
                            ("praise", lambda r: r["reward_evidence"]["decision"].update(praise=True)),
                            ("advantage", lambda r: r.update(advantage=-.5))):
            with self.subTest(label):
                forged = copy.deepcopy(rows)
                edit(forged[0])
                with self.assertRaisesRegex(ValueError, "differs from the program gate|penalty table"):
                    train.backward_single_block(self.bundle, forged, contract, components=("PG",))
        self.assertTrue(all(p.grad is None for ps in self.bundle.params.values() for p in ps))


class ProtocolV13(unittest.TestCase):
    def test_v13_protocol_fields(self):
        protocol = protocol_v13()
        contract = protocol["training_contract"]
        self.assertEqual((protocol["version"], contract["version"]),
                         ("error-note-single-response-v13", "error-note-single-response-v13"))
        self.assertEqual(VERSION_SINGLE_V13, "error-note-single-response-v13")
        rl = contract["rl"]
        self.assertEqual(rl["reward"], "single-t1-program-gate-v8")
        self.assertEqual(rl["advantage"], {"version": SINGLE_ADVANTAGE_VERSION_V13, "values": SINGLE_ADVANTAGE_V13})
        self.assertEqual(rl["spiral_window"], SINGLE_SPIRAL_V13)
        self.assertEqual(rl["praise"], SINGLE_PRAISE_V13)
        self.assertEqual((rl["baseline"], rl["token_scale"], rl["row_token_cap"], rl["row_token_cap_exempt"], rl["beta"]),
                         ("none", 1024., 2048, ["LOOP"], .02))
        self.assertEqual(rl["reward_values"]["CONTAINED_OPEN"], 0)
        self.assertEqual(contract["sft_weights"], {"A1": .05, "A2": .35, "B": .05})
        self.assertEqual(contract["sft_weighting"], SINGLE_SFT_WEIGHTING_V10)
        self.assertEqual(contract["sft_weighting"], {"A1": "per_kind_total", "A2": "per_row", "B": "per_kind_total"})
        self.assertNotIn("human_solution", contract)
        self.assertNotIn("H", contract["routes"])
        self.assertEqual(contract["routes"], {"RL": "LM", "A1": "LM", "A2_B": "accepted_note_self_report"})
        self.assertEqual(contract["b_rows"], "every_program_accepted_note_whatever_the_resolve_result")
        self.assertEqual(contract["program_filters"]["lesson_leak"], LESSON_LEAK_VERSION_V2)
        self.assertEqual(contract["generation_counts"], {"t1": 1, "note": 1, "resolve": 1})
        single._protocol_check(protocol)
        self.assertEqual(single.PROTOCOL_GATES[VERSION_SINGLE_V13], SINGLE_REWARD_GATE_V8)

    def test_v13_shares_the_v12_note_filters_and_b_rule_and_differs_from_v11_and_v12(self):
        v11 = plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE)
        v12_protocol = plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V12)
        v13 = protocol_v13()
        self.assertEqual((v11["version"], v12_protocol["version"]),
                         ("error-note-single-response-v11", "error-note-single-response-v12"))
        self.assertEqual(len({v11["sha256"], v12_protocol["sha256"], v13["sha256"]}), 3)
        single._protocol_check(v11)
        single._protocol_check(v12_protocol)
        for key in ("program_filters", "b_rows", "generation_counts", "length_policy"):
            self.assertEqual(v13["training_contract"][key], v12_protocol["training_contract"][key], key)
        self.assertEqual(v12_protocol["training_contract"]["rl"]["reward"], SINGLE_REWARD_GATE_V7)
        self.assertNotIn("spiral_window", v12_protocol["training_contract"]["rl"])
        self.assertNotIn("praise", v11["training_contract"]["rl"])
        self.assertEqual(v12_protocol["training_contract"]["sft_weights"]["H"], .02)

    def test_the_returned_protocol_is_a_deep_copy_of_the_module_constants(self):
        spiral, praise, table = copy.deepcopy(SINGLE_SPIRAL_V13), copy.deepcopy(SINGLE_PRAISE_V13), dict(SINGLE_ADVANTAGE_V13)
        before = protocol_v13()
        mutated = protocol_v13()
        rl = mutated["training_contract"]["rl"]
        rl["spiral_window"]["marker_token_ids"].append(1)
        rl["spiral_window"]["window_tokens"] = 1
        rl["spiral_window"]["penalties"].clear()
        rl["praise"]["min_tokens"] = 1
        rl["praise"]["applies_to"].append("B")
        rl["advantage"]["values"]["LOOP"] = 5.0
        self.assertEqual((SINGLE_SPIRAL_V13, SINGLE_PRAISE_V13, SINGLE_ADVANTAGE_V13), (spiral, praise, table))
        self.assertEqual(protocol_v13(), before)

    def test_a_tampered_v13_protocol_or_weights_are_refused(self):
        tampered = protocol_v13()
        tampered["training_contract"]["rl"]["reward"] = SINGLE_REWARD_GATE_V7  # gate v7 under a v13 protocol
        with self.assertRaisesRegex(ValueError, "invalid single-response protocol"):
            single._protocol_check(tampered)
        with self.assertRaisesRegex(ValueError, "nonnegative weights"):  # H belongs to v12 only
            plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V13,
                                  sft_weights={"A1": .05, "A2": .35, "B": .05, "H": .02})
        with self.assertRaisesRegex(ValueError, "version must be v11, v12, v13 or v14"):
            plain_protocol_single("bf16", 2, contracts=PINS, version="error-note-single-response-v15")

    def test_cli_accepts_v13_and_keeps_v12_as_the_default(self):
        parser = single.build_parser()
        required = ["prepare", "--order-sha256", "a" * 64] + [
            item for name in ("build-root", "order", "frozen-approval", "source-provenance", "training-rules",
                              "model-path", "output") for item in ("--" + name, "/x")]
        self.assertEqual(parser.parse_args(required).single_version, "v12")
        self.assertEqual(parser.parse_args(required + ["--single-version", "v13"]).single_version, "v13")
        self.assertEqual(single.SINGLE_CLI_VERSIONS["v13"], VERSION_SINGLE_V13)


class RuntimeV13(unittest.TestCase):
    NAME = "test_program_gate_runs_without_pauses_keeps_originals_and_gold_only_in_note_request"

    def case(self, *, marker_ids=(8,), marker="a"):
        """The runtime fixture under a v13 protocol. The tiny tokenizer decodes ID 8 as the word "a", so the sealed
        marker rule is patched to it; the protocol is built inside the patch so it equals the patched constant."""
        patcher = patch.dict(SINGLE_SPIRAL_V13, marker_token_ids=list(marker_ids), marker=marker)
        patcher.start()
        self.addCleanup(patcher.stop)
        case = runtime_tests.RuntimeTests(self.NAME)
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.protocol = plain_protocol_single("bf16", 219, contracts=PINS, token_scale=1024., version=VERSION_SINGLE_V13)
        case.owner.processor = tiny_processor()
        return case

    def closed(self, case, provenance=None):
        def grade(pid, text, finish, mode):
            return ("CORRECT" if text.endswith("B") else "INCORRECT", text[-1])

        provenance = {pid: PROVENANCE for pid in case.actors} if provenance is None else provenance
        result = single.run_single_block(case.owner, actors=case.actors, items=case.items, provenance=provenance,
                                         gold=lambda pid: "B", grade=grade,
                                         load_images=lambda actor: [Image.new("RGB", (8, 8))], protocol=case.protocol,
                                         output=case.out, limit=2, stream=case.stream)
        self.assertEqual(result["status"], "ROWS_READY")
        return json.loads((case.out / result["rows_path"]).read_text())

    def test_a_correct_t1_below_the_praise_minimum_gets_no_a1_row_and_advantage_zero(self):
        closed = self.closed(self.case())
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "RL", "A2", "B"])
        self.assertEqual(closed["row_decisions"]["p0:A1"], {"accepted": False, "method": "program", "reason": "A1_NOT_PRAISED"})
        p0, p1 = (next(r for r in closed["rows"] if r["kind"] == "RL" and r["problem_id"] == pid) for pid in ("p0", "p1"))
        self.assertEqual(len(p0["generation"]["completion_token_ids"]), 41)  # the fixture's T1 is 41 tokens < 256
        self.assertEqual((p0["reward"], p0["advantage"], p0["reward_evidence"]["decision"]["praise"]), (1, 0.0, False))
        self.assertEqual((p1["reward"], p1["advantage"], p1["reward_evidence"]["decision"]["penalty"]),
                         (-1, 0.0, "REASONED_WRONG"))
        self.assertIsNone(p1["reward_evidence"]["decision"]["window"])
        for row in (p0, p1):
            self.assertEqual(row["reward_evidence"]["version"], SINGLE_REWARD_GATE_V8)
            self.assertEqual(row["reward_evidence"]["item"], {"option_count": 2})
        self.assertEqual(closed["contract"]["version"], VERSION_SINGLE_V13)
        self.assertEqual(closed["contract"]["rl"]["praise"], SINGLE_PRAISE_V13)

    def test_with_the_threshold_lowered_the_a1_row_is_admitted_and_the_rl_row_is_praised(self):
        with patch.dict(SINGLE_PRAISE_V13, min_tokens=1):
            closed = self.closed(self.case())
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "A1", "RL", "A2", "B"])
        self.assertEqual(closed["row_decisions"]["p0:A1"], {"accepted": True, "method": "program", "reason": None})
        p0 = next(r for r in closed["rows"] if r["kind"] == "RL" and r["problem_id"] == "p0")
        self.assertEqual((p0["reward"], p0["advantage"], p0["reward_evidence"]["decision"]["praise"]), (1, 1.0, True))
        self.assertEqual(closed["contract"]["rl"]["praise"], SINGLE_PRAISE_V13 | {"min_tokens": 1})  # sealed with the rule

    def test_no_h_rows_even_when_the_provenance_has_a_scienceqa_solution(self):
        lecture, solution = v12.human_fields()
        provenance = {pid: PROVENANCE for pid in (f"p{i}" for i in range(10))} | {"p0": v12.scienceqa(lecture, solution)}
        case = self.case()
        closed = self.closed(case, provenance)
        self.assertNotIn("H", [r["kind"] for r in closed["rows"]])
        self.assertNotIn("p0:H", closed["row_decisions"])
        self.assertNotIn("human_solution", closed["contract"])
        self.assertNotIn("H", closed["contract"]["sft_weights"])
        self.assertEqual([(r["problem_id"], r["stage"]) for r in case.seen],
                         [("p0", "t1"), ("p1", "t1"), ("p1", "note"), ("p1", "resolve")])

    def test_the_wrong_answer_still_gets_its_b_row_after_a_failed_resolve(self):
        case = self.case()
        original = case.stream

        def wrong(requests, stats, stop_after):
            for request, output in original(requests, stats, stop_after):
                if request["stage"] == "resolve":
                    output["text"] = output["generation"]["text"] = " ".join(f"step{j}" for j in range(38)) + " Final answer: A"
                yield request, output

        case.stream = wrong
        closed = self.closed(case)
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "RL", "B"])  # v12's B rule, no A1 below the minimum
        self.assertFalse(closed["row_decisions"]["p1:A2"]["accepted"])
        self.assertTrue(closed["row_decisions"]["p1:B"]["accepted"])

    def test_the_runtime_refuses_a_tokenizer_whose_marker_ids_are_not_the_marker_word(self):
        for label, kwargs in (("sealed ids are not Wait here", {"marker_ids": MARKERS, "marker": "Wait"}),
                              ("one id differs", {"marker_ids": (8, 9), "marker": "a"}),
                              ("wrong word", {"marker_ids": (8,), "marker": "Wait"})):
            with self.subTest(label):
                case = self.case(**kwargs)
                with self.assertRaisesRegex(ValueError, "spiral marker tokens"):
                    self.closed(case)
                self.assertEqual(case.seen, [])
                self.assertFalse((case.out / "binding.json").exists())  # refused before anything was sealed

    def test_v12_runtime_does_not_check_the_marker_ids(self):
        case = runtime_tests.RuntimeTests(self.NAME)
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.protocol = plain_protocol_single("bf16", 219, contracts=PINS, token_scale=1024., version=VERSION_SINGLE_V12)
        case.owner.processor = tiny_processor()  # the tiny tokenizer decodes the sealed v13 IDs to ""
        closed = self.closed(case)
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "A1", "RL", "A2", "B"])
        self.assertEqual(closed["rows"][0]["reward_evidence"]["version"], SINGLE_REWARD_GATE_V7)


if __name__ == "__main__":
    unittest.main()
