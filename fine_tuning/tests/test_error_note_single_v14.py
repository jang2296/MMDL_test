"""SINGLE v14 CPU checks: tail window, reward gate v9 (tail), advantage table v4, the tail PG term, the v14 train
contract, the A2 minimum / "B follows its A2" rule in the runtime, scorer v2 wiring and the v14 protocol. Tiny random
models only; these are not pretrained Qwen learning evidence."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.error_note import single, single_grading
from fine_tuning.error_note import train_v4 as train
from fine_tuning.error_note.audit import text_sha256
from fine_tuning.error_note.contract import DENIAL_PHRASES, IMAGE, LESSON_LEAK_VERSION_V2, has_repetition
from fine_tuning.error_note.protocol import (SINGLE_A2_MIN_TOKENS_V14, SINGLE_ADVANTAGE_V13, SINGLE_ADVANTAGE_V14,
                                             SINGLE_ADVANTAGE_VERSION_V14, SINGLE_B_ROWS_V14, SINGLE_CONTRACT_HASHES,
                                             SINGLE_IMAGE_V14, SINGLE_OPEN_SCORER, SINGLE_PRAISE_V13, SINGLE_REWARD_GATE_V8,
                                             SINGLE_REWARD_GATE_V9, SINGLE_SFT_WEIGHTING_V10, SINGLE_SPIRAL_V13,
                                             SINGLE_TAIL_PRAISE_V14, VERSION_SINGLE, VERSION_SINGLE_V12, VERSION_SINGLE_V13,
                                             VERSION_SINGLE_V14, plain_protocol_single, tail_window)
from fine_tuning.error_note.single_grading import NORMALIZATION_RULES_V2, SCORER_V2, UNIT_ABBREVIATIONS
from fine_tuning.error_note.v4 import has_block_repetition, target_problem
from fine_tuning.notebook_rl.answers import TURN1_EVAL_CONTRACT
from fine_tuning.notebook_rl.input_preparation import digest
from fine_tuning.tests import test_error_note_single_loss as loss_tests
from fine_tuning.tests import test_error_note_single_runtime as runtime_tests
from fine_tuning.tests import test_error_note_single_v12 as v12
from fine_tuning.tests import test_error_note_single_v13 as v13
from fine_tuning.tests.test_error_note_single_loss import SAMPLING, block, gradients
from fine_tuning.tests.test_error_note_single_v11 import long_rows
from fine_tuning.tests.test_error_note_train import encoded, tiny_factory, tiny_processor
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE

PINS = v12.PINS
MARKERS = SINGLE_SPIRAL_V13["marker_token_ids"]  # the sealed Qwen3-VL IDs of "Wait" and " Wait"
TINY_MARKERS = [26, 27]  # inside the 32-token tiny vocabulary and outside the filler range 9..24
REASONED, LOOP = v13.REASONED, v13.LOOP
FLOOR = 256  # SINGLE_TAIL_PRAISE_V14["min_tokens"] and SINGLE_A2_MIN_TOKENS_V14
_SHORT = None


def setUpModule():
    """Tiny-model rows are a few tokens long; the 32-token rule is exercised with patched lengths below."""
    global _SHORT
    _SHORT = patch.object(single, "SOLUTION_MIN_TOKENS", 1)
    _SHORT.start()


def tearDownModule():
    _SHORT.stop()


def rep(count):
    """`count` IDs of a period-6 loop: has_repetition is True from 24 IDs on."""
    return [100 + i % 6 for i in range(count)]


def varied(count, start=1000):
    """`count` distinct IDs: no repetition and no marker token."""
    return list(range(start, start + count))


def tail_ids(total, tail=100, marker=MARKERS[0]):
    """`total` IDs: a repeating prefix, one marker token and a clean varied tail of `tail` IDs."""
    return rep(total - tail - 1) + [marker] + varied(tail)


def tiny_clean(count, seed=0):
    """`count` seeded IDs from the tiny vocabulary's filler range (no repetition, no tiny marker)."""
    return random.Random(seed).choices(range(9, 25), k=count)


def tiny_tail_tokens(prefix, tail, *, marker=TINY_MARKERS[0], seed=0):
    """Target tokens (the end token is added by `encoded`): a looping prefix, one marker and a clean tail."""
    return [9 + i % 6 for i in range(prefix)] + [marker] + tiny_clean(tail, seed)


def protocol_v14(total_slots=2, schedule_steps=2):
    return plain_protocol_single("bf16", total_slots, contracts=PINS, version=VERSION_SINGLE_V14, schedule_steps=schedule_steps)


def bind_v14(row, **kwargs):
    return v12.bind_v12(row, gate=SINGLE_REWARD_GATE_V9, **kwargs)


def case_row(base, *, status, finish="stop", ids=None, text=None, extracted="32", provenance=v12.GOLD_32,
             gate=SINGLE_REWARD_GATE_V9):
    """A copy of one fixture RL row graded as `status`, bound under `gate` (an open item)."""
    row = copy.deepcopy(base)
    row["status"] = status
    if ids is not None:
        row["generation"]["completion_token_ids"] = ids
    if text is not None:
        row["target"] = row["generation"]["text"] = text
    return v12.bind_v12(row, gate=gate, finish_reason=finish, extracted=extracted, provenance=provenance)


def as_v14(rows, contract, finish=None):
    """Rebind every RL row under gate v9 and turn the tiny v9 contract into the v14 block contract."""
    contract = copy.deepcopy(contract)
    v14 = protocol_v14()["training_contract"]
    contract["rl"].update({key: v14["rl"][key] for key in
                           ("baseline", "advantage", "token_scale", "row_token_cap", "row_token_cap_exempt",
                            "negative_logprob", "reward", "spiral_window", "praise", "tail_praise")})
    contract.update({key: v14[key] for key in ("version", "sft_weights", "sft_weighting", "a2_min_tokens", "b_rows",
                                                "schedule_steps", "total_slots")})
    for row in rows:
        if row["kind"] == "RL":
            bind_v14(row, finish_reason=(finish or {}).get(row["problem_id"], "stop"))
    contract.update(
        reward_provenance_sha256={r["problem_id"]: digest(r["reward_evidence"]["provenance"]) for r in rows if r["kind"] == "RL"},
        reward_evidence_sha256={r["problem_id"]: digest(r["reward_evidence"]) for r in rows if r["kind"] == "RL"})
    return contract


def token_rows(bundle, sequences):
    """CORRECT RL rows whose sampled target tokens are `sequences` (the end token 2 is appended by `encoded`), with the
    stored full-vocabulary sampling log probabilities of the tiny policy (like v13.random_rows, with chosen tokens)."""
    rows, contract = block(bundle, count=len(sequences), reward=1)
    vocab = bundle.model.get_base_model().lm_head.out_features
    for index, tokens in enumerate(sequences):
        row = encoded(f"item-{index}:RL", (7, 8), tuple(tokens), "LM", "RL", seed=index)
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


def reference_gradient(bundle, rows, row_index, start, end, row_count):
    """Gradient of -sum(stored-support log q over [start, end)) / (row_count * 1024) of one row, nothing else."""
    bundle.model.zero_grad(set_to_none=True)
    bundle.model.train()
    bundle.route.apply(False)
    value = train.single_batch_statistics(bundle, rows, SAMPLING)[row_index]["logprobs"]
    (-value[start:end].sum() / (row_count * 1024.)).backward()
    result = gradients(bundle)
    bundle.model.zero_grad(set_to_none=True)
    return result


class FixtureChecks(unittest.TestCase):
    def test_the_id_fixtures_repeat_or_stay_clean_as_intended(self):
        self.assertTrue(has_repetition(rep(24)))
        self.assertFalse(has_repetition(varied(2000)))
        self.assertFalse(has_block_repetition(varied(2000)))
        ids = tail_ids(256)
        self.assertEqual((len(ids), ids.count(MARKERS[0])), (256, 1))
        self.assertTrue(has_repetition(ids))
        self.assertFalse(has_repetition(ids[ids.index(MARKERS[0]) + 1:]))
        self.assertEqual(target_problem(REASONED, ids, "stop", "CORRECT", min_tokens=1), "REPETITION")
        self.assertEqual(target_problem(REASONED, varied(300), "stop", "CORRECT", min_tokens=1), None)
        tiny = tiny_tail_tokens(160, 100)
        self.assertTrue(has_repetition(tiny))
        self.assertFalse(has_repetition(tiny_clean(2900)) or has_block_repetition(tiny_clean(2900)))


class TailWindow(unittest.TestCase):
    def ids(self, count, markers):
        """`count` varied IDs with a marker token at each (index, marker id) of `markers`."""
        ids = varied(count)
        for index, token in markers:
            ids[index] = token
        return ids

    def test_no_marker_gives_no_window(self):
        self.assertIsNone(tail_window(varied(300)))
        self.assertIsNone(tail_window(self.ids(300, [(100, 13825)])))  # a neighbouring ID never counts

    def test_one_marker_with_a_clean_tail_starts_right_after_it(self):
        self.assertEqual(tail_window(self.ids(300, [(100, MARKERS[0])])), [101, 300])
        self.assertEqual(tail_window(self.ids(300, [(100, MARKERS[1])])), [101, 300])  # both marker IDs count

    def test_several_markers_use_the_last_one_of_either_id(self):
        self.assertEqual(tail_window(self.ids(300, [(50, MARKERS[0]), (100, MARKERS[0]), (200, MARKERS[0])])), [201, 300])
        self.assertEqual(tail_window(self.ids(300, [(10, MARKERS[0]), (50, MARKERS[1])])), [51, 300])
        self.assertEqual(tail_window(self.ids(300, [(10, MARKERS[1]), (50, MARKERS[0])])), [51, 300])

    def test_a_tail_longer_than_1024_keeps_only_its_last_1024(self):
        self.assertEqual(tail_window(self.ids(3000, [(100, MARKERS[0])])), [1976, 3000])
        self.assertEqual(tail_window(self.ids(1025, [(0, MARKERS[0])])), [1, 1025])  # exactly 1024 after the marker
        self.assertEqual(tail_window(self.ids(1026, [(0, MARKERS[0])])), [2, 1026])
        self.assertEqual(tail_window(self.ids(1027, [(0, MARKERS[0])])), [3, 1027])

    def test_the_window_needs_at_least_32_tokens(self):
        self.assertIsNone(tail_window(self.ids(10 + 1 + 31, [(10, MARKERS[0])])))
        self.assertEqual(tail_window(self.ids(10 + 1 + 32, [(10, MARKERS[0])])), [11, 43])
        self.assertIsNone(tail_window(self.ids(300, [(299, MARKERS[0])])))  # marker is the last token: empty tail

    def test_a_repeating_tail_gives_no_window_but_the_same_row_with_a_clean_tail_does(self):
        head = varied(50)
        repeating, clean = head + [MARKERS[0]] + rep(60), head + [MARKERS[0]] + varied(60, 2000)
        self.assertTrue(has_repetition(rep(60)))
        self.assertIsNone(tail_window(repeating))
        self.assertEqual(tail_window(clean), [51, 111])
        # A repeating prefix before the marker does not matter, only the window does.
        self.assertTrue(has_repetition(rep(80) + [MARKERS[0]] + varied(60)))
        self.assertEqual(tail_window(rep(80) + [MARKERS[0]] + varied(60)), [81, 141])

    def test_empty_or_non_integer_input_is_refused(self):
        for label, bad in (("empty", []), ("none", None), ("string", "abc"), ("tuple", (1, 2, 3)), ("float", [1, 2.0]),
                           ("bool", [1, True]), ("text id", [1, "2"])):
            with self.subTest(label), self.assertRaisesRegex(ValueError, "tail window needs the sampled token IDs"):
                tail_window(bad)

    def test_the_sealed_rule_is_the_designed_one(self):
        rule = SINGLE_TAIL_PRAISE_V14
        self.assertEqual(rule["marker_token_ids"], MARKERS)
        self.assertEqual(rule["marker_token_ids"], SINGLE_SPIRAL_V13["marker_token_ids"])
        self.assertEqual((rule["version"], rule["min_tokens"], rule["window_tokens"], rule["min_window_tokens"],
                          rule["logprob"], rule["row_weight"], rule["applies_to"]),
                         ("tail-praise-v1", 256, 1024, 32, "stored_support", 1.0, ["PG"]))
        self.assertEqual(rule["reasons"], ["T1_INELIGIBLE_REPETITION", "T1_INELIGIBLE_BLOCK_REPETITION",
                                           "T1_INELIGIBLE_LOOP", "T1_INELIGIBLE_TOO_LONG"])
        self.assertEqual(rule["window_filter"], ["has_repetition", "has_block_repetition"])
        self.assertEqual(rule["answer_denial"], "no_tail_when_the_whole_answer_denies_the_question")
        self.assertEqual((SINGLE_A2_MIN_TOKENS_V14, FLOOR), (256, rule["min_tokens"]))


class RewardGateV9(v12.TinyThreads):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.base = block(tiny_factory(), count=1)[0][0]

    def test_constants(self):
        self.assertEqual(SINGLE_REWARD_GATE_V9, "single-t1-program-gate-v9")
        self.assertEqual(SINGLE_ADVANTAGE_VERSION_V14, "penalty-table-v4")
        self.assertEqual(SINGLE_ADVANTAGE_V14, SINGLE_ADVANTAGE_V13 | {"tail_praised": 1.0})
        self.assertEqual(SINGLE_ADVANTAGE_V14["tail_praised"], 1.0)
        self.assertEqual(single.PROTOCOL_GATES[VERSION_SINGLE_V14], SINGLE_REWARD_GATE_V9)
        self.assertIn(SINGLE_REWARD_GATE_V9, single.GATES_V8)
        self.assertIn(SINGLE_REWARD_GATE_V9, single.TABLE_GATES)

    def test_every_tail_case_round_trips_and_every_neighbour_has_no_tail(self):
        loop_ids = varied(150) + [MARKERS[0]] + varied(120, 2000)  # clean IDs: only the text loops
        denial = "The figure has a mislabel but the work is clear. Final answer: 32"
        cases = (  # label, status, finish, ids, text, extracted, reason, tail window expected, reward, advantage, penalty
            ("repetition", "CORRECT", "stop", tail_ids(256), REASONED, "32", "T1_INELIGIBLE_REPETITION", True, 0, 1.0, None),
            ("repetition, other marker", "CORRECT", "stop", tail_ids(300, marker=MARKERS[1]), REASONED, "32",
             "T1_INELIGIBLE_REPETITION", True, 0, 1.0, None),
            ("loop text", "CORRECT", "stop", loop_ids, LOOP, "32", "T1_INELIGIBLE_LOOP", True, 0, 1.0, None),
            ("too long", "CORRECT", "stop", tail_ids(4200), REASONED, "32", "T1_INELIGIBLE_TOO_LONG", True, 0, 1.0, None),
            ("255 tokens", "CORRECT", "stop", tail_ids(255), REASONED, "32", "T1_INELIGIBLE_REPETITION", False, 0, 0.0, None),
            ("no marker", "CORRECT", "stop", rep(300), REASONED, "32", "T1_INELIGIBLE_REPETITION", False, 0, 0.0, None),
            ("31-token window", "CORRECT", "stop", rep(250) + [MARKERS[0]] + varied(31), REASONED, "32",
             "T1_INELIGIBLE_REPETITION", False, 0, 0.0, None),
            ("repeating window", "CORRECT", "stop", rep(200) + [MARKERS[0]] + rep(60), REASONED, "32",
             "T1_INELIGIBLE_REPETITION", False, 0, 0.0, None),
            ("not stopped", "CORRECT", "length", tail_ids(300), REASONED, "32", "T1_INELIGIBLE_NOT_STOPPED", False, 0, 0.0,
             None),
            ("answer denial", "CORRECT", "stop", loop_ids, denial, "32", "T1_ANSWER_DENIAL:mislabel", False, 0, 0.0, None),
            ("not correct", "INCORRECT", "stop", tail_ids(300), REASONED, "31", "T1_NOT_CORRECT", False, -1, 0.0,
             "REASONED_WRONG"),
            ("accepted praised", "CORRECT", "stop", loop_ids, REASONED, "32", None, False, 1, 1.0, None),
            ("accepted unpraised", "CORRECT", "stop", varied(100), REASONED, "32", None, False, 1, 0.0, None))
        for label, status, finish, ids, text, extracted, reason, tail, reward, advantage, penalty in cases:
            with self.subTest(label):
                row = case_row(self.base, status=status, finish=finish, ids=ids, text=text, extracted=extracted)
                decision = row["reward_evidence"]["decision"]
                self.assertEqual((decision["reason"], decision["penalty"], row["reward"], row["advantage"]),
                                 (reason, penalty, reward, advantage))
                self.assertEqual(decision["tail"], tail_window(ids) if tail else None)
                self.assertEqual(decision["tail"] is not None, tail)
                self.assertEqual(decision["accepted"], reason is None)
                self.assertEqual((decision["praise"], decision["window"]), (reason is None and len(ids) >= FLOOR, None))
                self.assertIs(type(row["reward"]), int)
                self.assertIs(type(row["advantage"]), float)
                self.assertEqual(row["reward_evidence"]["version"], SINGLE_REWARD_GATE_V9)
                self.assertEqual(row["reward_evidence"]["advantage"], {"version": "penalty-table-v4", "value": advantage})
                self.assertEqual(single.validate_single_reward(row), decision)
                self.assertEqual(single.single_advantage(decision, SINGLE_REWARD_GATE_V9), advantage)

    def test_a_row_with_a_tail_is_not_accepted_not_penalized_and_not_an_a1_target(self):
        row = case_row(self.base, status="CORRECT", ids=tail_ids(256), text=REASONED)
        decision = row["reward_evidence"]["decision"]
        self.assertEqual((decision["accepted"], decision["penalty"], row["reward"], row["advantage"]), (False, None, 0, 1.0))
        self.assertEqual(decision["tail"], tail_window(tail_ids(256)))
        self.assertEqual(decision["tail"], [156, 256])
        self.assertIsNotNone(target_problem(REASONED, tail_ids(256), "stop", "CORRECT", min_tokens=1))  # no A1 target

    def test_a_repeating_row_whose_text_denies_the_question_gets_no_tail_and_the_same_row_without_denial_does(self):
        for ids, reason in ((tail_ids(300), "T1_INELIGIBLE_REPETITION"), (tail_ids(4200), "T1_INELIGIBLE_TOO_LONG")):
            window = tail_window(ids)
            self.assertIsNotNone(window)
            clean = case_row(self.base, status="CORRECT", ids=ids, text=REASONED)
            self.assertEqual((clean["reward_evidence"]["decision"]["reason"], clean["reward_evidence"]["decision"]["tail"],
                              clean["reward"], clean["advantage"]), (reason, window, 0, 1.0))
            for phrase in DENIAL_PHRASES:
                with self.subTest(reason=reason, phrase=phrase):
                    text = f"The {phrase}, so I assume the figure shows the usual case. Final answer: 32"
                    row = case_row(self.base, status="CORRECT", ids=ids, text=text)
                    decision = row["reward_evidence"]["decision"]
                    self.assertEqual((decision["reason"], decision["accepted"], decision["penalty"], decision["tail"]),
                                     (reason, False, None, None))  # still the repetition/length reason, no tail
                    self.assertEqual((row["reward"], row["advantage"]), (0, 0.0))
                    self.assertEqual(row["reward_evidence"]["advantage"], {"version": "penalty-table-v4", "value": 0.0})
                    self.assertEqual(single.validate_single_reward(row), decision)
                    self.assertEqual(single.single_advantage(decision, SINGLE_REWARD_GATE_V9), 0.0)
                    forged = copy.deepcopy(row)
                    forged["reward_evidence"]["decision"]["tail"] = window  # the tail the same IDs would earn without denial
                    with self.assertRaisesRegex(ValueError, "differs from the program gate"):
                        single.validate_single_reward(forged)
                    forged = copy.deepcopy(row)
                    forged["advantage"] = 1.0
                    with self.assertRaisesRegex(ValueError, "v14 penalty table"):
                        single.validate_single_reward(forged)
        # Denial markup (emphasis, curly apostrophes, case) is read like the A1/A2 rule reads it.
        row = case_row(self.base, status="CORRECT", ids=tail_ids(300), text="The **Question is Incomplete**. Final answer: 32")
        self.assertEqual((row["reward_evidence"]["decision"]["tail"], row["advantage"]), (None, 0.0))

    def test_the_tail_threshold_is_exactly_256_tokens_and_is_read_from_the_sealed_rule(self):
        for count, tail in ((255, None), (256, [156, 256]), (257, [157, 257]), (1000, [900, 1000])):
            with self.subTest(count=count):
                row = case_row(self.base, status="CORRECT", ids=tail_ids(count), text=REASONED)
                self.assertEqual(row["reward_evidence"]["decision"]["tail"], tail)
                self.assertEqual(row["advantage"], 0.0 if tail is None else 1.0)
        with patch.dict(SINGLE_TAIL_PRAISE_V14, min_tokens=100):
            row = case_row(self.base, status="CORRECT", ids=tail_ids(180), text=REASONED)
            self.assertEqual((row["reward_evidence"]["decision"]["tail"], row["advantage"]), ([80, 180], 1.0))

    def test_gate_v9_is_gate_v8_plus_the_tail_for_every_row_without_a_tail(self):
        loop_ids = varied(150) + [MARKERS[0]] + varied(120, 2000)
        cases = (("CORRECT", "stop", varied(256), REASONED, "32"), ("CORRECT", "stop", varied(40), REASONED, "32"),
                 ("CORRECT", "stop", varied(256), LOOP, "32"), ("CORRECT", "length", varied(256), REASONED, "32"),
                 ("INCORRECT", "stop", varied(40), LOOP, "A"), ("INCORRECT", "stop", varied(40), REASONED, "31"),
                 ("INCORRECT", "stop", None, None, "A"), ("INVALID_NO_FINAL", "stop", varied(40), REASONED, None),
                 ("TRUNCATED", "length", varied(40), LOOP, None), ("TRUNCATED", "length", varied(3000), REASONED, None),
                 ("TRUNCATED", "length", loop_ids, REASONED, None), ("INCORRECT", "stop", varied(40), REASONED, v12.CONTAINED),
                 ("UNVERIFIABLE", "stop", None, None, None))
        for status, finish, ids, text, extracted in cases:
            with self.subTest(status=status, finish=finish, n=ids and len(ids), text=text and text[:9]), \
                    patch.object(single, "SOLUTION_MIN_TOKENS", 32):
                new = case_row(self.base, status=status, finish=finish, ids=ids, text=text, extracted=extracted)
                old = v13.case_row(self.base, status=status, finish=finish, ids=ids, text=text, extracted=extracted)
                decision = new["reward_evidence"]["decision"]
                self.assertIsNone(decision["tail"])
                self.assertEqual({k: v for k, v in decision.items() if k != "tail"}, old["reward_evidence"]["decision"])
                self.assertEqual((new["reward"], new["advantage"]), (old["reward"], old["advantage"]))

    def test_gate_v8_decisions_carry_no_tail_key_even_for_a_row_that_gate_v9_would_praise(self):
        row = case_row(self.base, status="CORRECT", ids=tail_ids(256), text=REASONED, gate=SINGLE_REWARD_GATE_V8)
        decision = row["reward_evidence"]["decision"]
        self.assertNotIn("tail", decision)
        self.assertEqual((row["reward"], row["advantage"], decision["praise"]), (0, 0.0, False))
        self.assertEqual(row["reward_evidence"]["advantage"], {"version": "penalty-table-v3", "value": 0.0})
        self.assertEqual(single.validate_single_reward(row), decision)

    def test_single_advantage_ranks_penalty_then_contained_open_then_tail(self):
        base = {"accepted": False, "reason": "T1_INELIGIBLE_REPETITION", "penalty": None, "praise": False, "window": None}
        for label, decision, expected in (
                ("tail", base | {"tail": [10, 300]}, 1.0), ("no tail", base | {"tail": None}, 0.0),
                ("accepted praised", base | {"accepted": True, "reason": None, "praise": True, "tail": None}, 1.0),
                ("accepted unpraised", base | {"accepted": True, "reason": None, "tail": None}, 0.0),
                ("penalty first", base | {"penalty": "LOOP", "tail": [10, 300]}, -1.0),
                ("contained open first", base | {"reason": single.CONTAINED_OPEN, "tail": [10, 300]}, 0.0)):
            with self.subTest(label):
                self.assertEqual(single.single_advantage(decision, SINGLE_REWARD_GATE_V9), expected)
        # Under gate v8 the same decision never reads a tail.
        self.assertEqual(single.single_advantage(base | {"tail": [10, 300]}, SINGLE_REWARD_GATE_V8), 0.0)

    def test_forged_tail_praise_window_advantage_or_gate_label_is_refused(self):
        ids = tail_ids(256)
        row = case_row(self.base, status="CORRECT", ids=ids, text=REASONED)
        plain = case_row(self.base, status="CORRECT", ids=tail_ids(255), text=REASONED)
        forged = copy.deepcopy(row)
        forged["advantage"] = 0.0
        with self.assertRaisesRegex(ValueError, "v14 penalty table"):
            single.validate_single_reward(forged)
        forged = copy.deepcopy(row)
        forged["reward_evidence"]["advantage"]["value"] = 0.0
        with self.assertRaisesRegex(ValueError, "v14 penalty table"):
            single.validate_single_reward(forged)
        for source, label, edit in (
                (row, "tail dropped", {"tail": None}), (row, "tail whole row", {"tail": [0, 256]}),
                (row, "tail shifted", {"tail": [157, 256]}), (row, "tail shortened", {"tail": [156, 255]}),
                (row, "praise", {"praise": True}), (row, "window", {"window": [0, 1]}), (row, "accepted", {"accepted": True}),
                (row, "penalty", {"penalty": "LOOP"}), (row, "reason", {"reason": "T1_FORGED"}),
                (plain, "tail added", {"tail": [10, 255]}), (plain, "tail added at the real start", {"tail": [155, 255]})):
            with self.subTest(label):
                forged = copy.deepcopy(source)
                forged["reward_evidence"]["decision"].update(edit)
                with self.assertRaisesRegex(ValueError, "differs from the program gate"):
                    single.validate_single_reward(forged)
        forged = copy.deepcopy(row)
        del forged["reward_evidence"]["decision"]["tail"]
        with self.assertRaisesRegex(ValueError, "differs from the program gate"):
            single.validate_single_reward(forged)
        forged = copy.deepcopy(row)
        forged["reward_evidence"]["version"] = SINGLE_REWARD_GATE_V8  # a v9 decision under the v8 gate name
        with self.assertRaisesRegex(ValueError, "differs from the program gate"):
            single.validate_single_reward(forged)
        v8 = case_row(self.base, status="CORRECT", ids=ids, text=REASONED, gate=SINGLE_REWARD_GATE_V8)
        v8["reward_evidence"]["version"] = SINGLE_REWARD_GATE_V9  # a v8 decision under the v9 gate name has no tail
        with self.assertRaisesRegex(ValueError, "differs from the program gate"):
            single.validate_single_reward(v8)
        forged = copy.deepcopy(row)
        forged["generation"]["completion_token_ids"] = rep(255) + [MARKERS[0]]  # other IDs: the sealed tail no longer matches
        with self.assertRaisesRegex(ValueError, "differs from the program gate"):
            single.validate_single_reward(forged)


class TailPg(v12.TinyThreads):
    """The tail row's PG term, with the tail rule's marker IDs patched into the 32-token tiny vocabulary."""

    def setUp(self):
        self.enterContext(patch.dict(SINGLE_TAIL_PRAISE_V14, marker_token_ids=list(TINY_MARKERS)))
        self.bundle = tiny_factory()

    def backward(self, rows, contract):
        self.bundle.model.zero_grad(set_to_none=True)
        return train.backward_single_block(self.bundle, rows, contract, components=("PG",))

    def test_value_and_gradient_follow_the_sealed_tail_only(self):
        rows, contract = token_rows(self.bundle, [tiny_tail_tokens(160, 100), tiny_clean(20, seed=3)])
        contract = as_v14(rows, contract)
        decisions = [r["reward_evidence"]["decision"] for r in rows]
        self.assertEqual([(r["reward"], r["advantage"], d["tail"], d["penalty"]) for r, d in zip(rows, decisions, strict=True)],
                         [(0, 1.0, [161, 262], None), (1, 0.0, None, None)])
        self.assertEqual(len(rows[0]["generation"]["completion_token_ids"]), 262)
        stats = self.backward(rows, contract)
        actual, rl = gradients(self.bundle), stats["rl"]
        self.assertEqual((rl["tail_rows"], rl["tail_tokens"], rl["window_rows"], rl["window_tokens"]), (1, 101, 0, 0))
        self.assertEqual((rl["pg_rows_positive"], rl["pg_rows_negative"], rl["capped_rows"]), (1, 0, 0))
        stored = rows[0]["generation"]["sampling_logprobs"]
        expected = -sum(stored[161:262]) / (2 * 1024.)  # -A * 1.0 * sum(stored log q over the tail) / (B * token_scale)
        self.assertAlmostEqual(rl["pg_positive"], expected, delta=1e-4)
        self.assertAlmostEqual(rl["pg"], expected, delta=1e-4)
        self.assertGreater(expected, 0)
        self.assertGreater(abs(-sum(stored) / (2 * 1024.) - expected), 1e-3)  # not the whole-row value
        self.assertGreater(abs(-sum(stored[200:262]) / (2 * 1024.) - expected), 1e-3)  # not another range
        for name, grad in reference_gradient(self.bundle, rows, 0, 161, 262, 2).items():
            torch.testing.assert_close(actual[name], grad, atol=1e-8, rtol=1e-4)
        for label, (start, end) in (("whole row", (0, 262)), ("late half", (200, 262)), ("prefix", (0, 161))):
            other = reference_gradient(self.bundle, rows, 0, start, end, 2)
            self.assertTrue(any(not torch.allclose(actual[name], grad, atol=1e-8, rtol=1e-4) for name, grad in other.items()),
                            label)

    def test_a_different_tail_range_gives_a_different_value_and_gradient(self):
        tokens = [9 + i % 6 for i in range(160)] + [TINY_MARKERS[0]] + tiny_clean(40) + [TINY_MARKERS[1]] + tiny_clean(60, 1)
        seen = {}
        for markers, start in (([26, 27], 202), ([26], 161)):
            with patch.dict(SINGLE_TAIL_PRAISE_V14, marker_token_ids=markers):
                rows, contract = token_rows(self.bundle, [tokens])
                contract = as_v14(rows, contract)
                self.assertEqual(rows[0]["reward_evidence"]["decision"]["tail"], [start, 263])
                stats = self.backward(rows, contract)
                grads = gradients(self.bundle)
            self.assertEqual((stats["rl"]["tail_rows"], stats["rl"]["tail_tokens"]), (1, 263 - start))
            self.assertAlmostEqual(stats["rl"]["pg_positive"],
                                   -sum(rows[0]["generation"]["sampling_logprobs"][start:]) / 1024., delta=1e-4)
            for name, grad in reference_gradient(self.bundle, rows, 0, start, 263, 1).items():
                torch.testing.assert_close(grads[name], grad, atol=1e-8, rtol=1e-4)
            seen[start] = (stats["rl"]["pg_positive"], grads)
        self.assertGreater(abs(seen[202][0] - seen[161][0]), 1e-3)
        self.assertTrue(any(not torch.allclose(seen[202][1][name], seen[161][1][name], atol=1e-8, rtol=1e-4)
                            for name in seen[202][1]))

    def test_a_tail_row_longer_than_the_row_cap_keeps_full_mass(self):
        rows, contract = token_rows(self.bundle, [tiny_tail_tokens(2400, 500)])
        contract = as_v14(rows, contract)
        self.assertEqual(rows[0]["reward_evidence"]["decision"]["tail"], [2401, 2902])
        stats = self.backward(rows, contract)
        rl = stats["rl"]
        self.assertEqual((rl["tail_rows"], rl["tail_tokens"], rl["capped_rows"], rl["pg_rows_positive"]), (1, 501, 0, 1))
        stored = rows[0]["generation"]["sampling_logprobs"]
        self.assertAlmostEqual(rl["pg_positive"], -sum(stored[2401:]) / 1024., delta=1e-3)
        capped = -(2048 / 2902) * sum(stored) / 1024.
        self.assertGreater(abs(rl["pg_positive"] - capped), .5)  # neither the capped whole row nor the whole row

    def test_a_row_without_a_tail_gets_no_tail_term_and_the_stats_count_none(self):
        rows, contract = token_rows(self.bundle, [tiny_clean(260, 5), tiny_clean(30, 6)])  # praised (whole row) and unpraised
        contract = as_v14(rows, contract)
        self.assertEqual([(r["advantage"], r["reward_evidence"]["decision"]["tail"]) for r in rows], [(1.0, None), (0.0, None)])
        stats = self.backward(rows, contract)
        rl = stats["rl"]
        self.assertEqual((rl["tail_rows"], rl["tail_tokens"], rl["pg_rows_positive"]), (0, 0, 1))
        self.assertAlmostEqual(rl["pg_positive"], -sum(rows[0]["generation"]["sampling_logprobs"]) / (2 * 1024.), delta=1e-3)

    def test_a_spiral_window_row_behaves_exactly_as_under_v13_and_a_v13_contract_has_no_tail_keys(self):
        rows, base = long_rows(self.bundle, (10, 10))
        rows[0]["status"] = "TRUNCATED"
        finish = {rows[0]["problem_id"]: "length"}
        rows14 = copy.deepcopy(rows)
        contract13 = v13.as_v13(rows, base, finish=finish)
        contract14 = as_v14(rows14, base, finish=finish)
        self.assertEqual([(r["advantage"], r["reward_evidence"]["decision"]["window"]) for r in rows14],
                         [(-1.0, [5, 10]), (0.0, None)])
        self.assertEqual(rows14[0]["reward_evidence"]["decision"]["tail"], None)
        stats13 = self.backward(rows, contract13)
        grads13 = gradients(self.bundle)
        stats14 = self.backward(rows14, contract14)
        grads14 = gradients(self.bundle)
        self.assertEqual(set(stats14["rl"]) - set(stats13["rl"]), {"tail_rows", "tail_tokens"})
        self.assertFalse({"tail_rows", "tail_tokens"} & set(stats13["rl"]))
        for key, value in stats13["rl"].items():
            self.assertAlmostEqual(stats14["rl"][key], value, delta=1e-9, msg=key)
        self.assertEqual((stats14["rl"]["tail_rows"], stats14["rl"]["tail_tokens"]), (0, 0))
        self.assertEqual((stats14["rl"]["window_rows"], stats14["rl"]["window_tokens"]), (1, 5))
        for name, grad in grads13.items():
            torch.testing.assert_close(grads14[name], grad, atol=1e-9, rtol=1e-6)

    def test_a_tampered_tail_praise_window_or_advantage_on_a_stored_row_is_refused_at_train_time(self):
        rows, contract = token_rows(self.bundle, [tiny_tail_tokens(160, 100)])
        contract = as_v14(rows, contract)
        for label, edit in (("tail dropped", lambda r: r["reward_evidence"]["decision"].update(tail=None)),
                            ("tail whole row", lambda r: r["reward_evidence"]["decision"].update(tail=[0, 262])),
                            ("tail shifted", lambda r: r["reward_evidence"]["decision"].update(tail=[162, 262])),
                            ("praise", lambda r: r["reward_evidence"]["decision"].update(praise=True)),
                            ("window", lambda r: r["reward_evidence"]["decision"].update(window=[161, 262])),
                            ("advantage", lambda r: r.update(advantage=0.0)),
                            ("evidence advantage", lambda r: r["reward_evidence"]["advantage"].update(value=0.0))):
            with self.subTest(label):
                forged = copy.deepcopy(rows)
                edit(forged[0])
                with self.assertRaisesRegex(ValueError, "differs from the program gate|penalty table"):
                    train.backward_single_block(self.bundle, forged, contract, components=("PG",))
        self.assertTrue(all(p.grad is None for ps in self.bundle.params.values() for p in ps))


class TrainContractV14(v12.TinyThreads):
    def setUp(self):
        self.bundle = tiny_factory()
        self.rows, base = block(self.bundle, count=1, reward=0)
        self.contract = as_v14(self.rows, base)

    def refused(self, contract, pattern="single train: v14", rows=None, components=("PG",)):
        with self.assertRaisesRegex(ValueError, pattern):
            train.backward_single_block(self.bundle, self.rows if rows is None else rows, contract, components=components)
        self.assertTrue(all(p.grad is None for ps in self.bundle.params.values() for p in ps))

    def a2_row(self, tokens):
        row = encoded("sft-A2-0", (7, 8), tuple(tokens), "LM", "A2")
        row["problem_id"] = "item-0"
        return row

    def test_a_valid_v14_contract_runs_and_counts_no_tail_rows(self):
        stats = train.backward_single_block(self.bundle, self.rows, self.contract, components=("PG", "KL"))
        self.assertEqual((stats["rl"]["tail_rows"], stats["rl"]["tail_tokens"]), (0, 0))
        self.assertEqual((stats["rl"]["window_rows"], stats["rl"]["window_tokens"]), (0, 0))

    def test_a_contract_missing_or_changing_a_v14_rule_is_refused(self):
        for key in ("tail_praise",):
            with self.subTest(missing=key):
                bad = copy.deepcopy(self.contract)
                del bad["rl"][key]
                self.refused(bad)
        for key in ("a2_min_tokens", "b_rows", "schedule_steps"):
            with self.subTest(missing=key):
                bad = copy.deepcopy(self.contract)
                del bad[key]
                self.refused(bad)
        rule = self.contract["rl"]["tail_praise"]
        for label, change in (("min tokens", {"min_tokens": 128}), ("window tokens", {"window_tokens": 512}),
                              ("min window", {"min_window_tokens": 16}), ("markers", {"marker_token_ids": [13824]}),
                              ("reasons", {"reasons": rule["reasons"][:2]}), ("logprob", {"logprob": "raw"}),
                              ("row weight", {"row_weight": .5}), ("applies to", {"applies_to": ["PG", "A1"]}),
                              ("answer denial", {"answer_denial": "tail_even_when_the_answer_denies"}),
                              ("answer denial dropped", {"answer_denial": None})):
            with self.subTest(tail_praise=label):
                self.refused(self.contract | {"rl": self.contract["rl"] | {"tail_praise": rule | change}})
        for label, change in (("a2 minimum 128", {"a2_min_tokens": 128}), ("a2 minimum 0", {"a2_min_tokens": 0}),
                              ("v13 B rule", {"b_rows": "every_program_accepted_note_whatever_the_resolve_result"}),
                              ("schedule 0", {"schedule_steps": 0}), ("schedule above the slots", {"schedule_steps": 3}),
                              ("schedule bool", {"schedule_steps": True}), ("schedule float", {"schedule_steps": 2.0}),
                              ("schedule text", {"schedule_steps": "2"})):
            with self.subTest(label):
                self.refused(self.contract | change)
        self.assertEqual(self.contract["a2_min_tokens"], 256)
        self.assertEqual(self.contract["b_rows"], SINGLE_B_ROWS_V14)

    def test_a_contract_with_the_v13_table_gate_v8_or_a_human_solution_pin_is_refused(self):
        rl = self.contract["rl"]
        v12_contract = plain_protocol_single("bf16", 1, contracts=PINS, version=VERSION_SINGLE_V12)["training_contract"]
        self.refused(self.contract | {"rl": rl | {"advantage": {"version": "penalty-table-v3", "values": SINGLE_ADVANTAGE_V13}}})
        self.refused(self.contract | {"rl": rl | {"advantage": {"version": SINGLE_ADVANTAGE_VERSION_V14,
                                                                  "values": SINGLE_ADVANTAGE_V13}}})
        self.refused(self.contract | {"rl": rl | {"advantage": {"version": SINGLE_ADVANTAGE_VERSION_V14,
                                                                  "values": SINGLE_ADVANTAGE_V14 | {"tail_praised": .5}}}})
        self.refused(self.contract | {"rl": rl | {"reward": SINGLE_REWARD_GATE_V8}}, pattern="reward contract required")
        self.refused(self.contract | {"human_solution": v12_contract["human_solution"]})
        self.refused(self.contract | {"human_solution": {}})
        self.refused(self.contract | {"version": VERSION_SINGLE_V13}, pattern="reward contract required")

    def test_gate_v8_rows_under_a_v14_contract_are_refused(self):
        v8_rows = copy.deepcopy(self.rows)
        for row in v8_rows:
            v13.bind_v13(row)
        contract = self.contract | {"reward_evidence_sha256": {r["problem_id"]: digest(r["reward_evidence"]) for r in v8_rows}}
        self.refused(contract, pattern="binding differs from the contract version", rows=v8_rows)

    def test_an_a2_row_needs_at_least_256_target_tokens(self):
        for tokens, ok in ((254, False), (255, True), (256, True), (10, False)):  # target_len counts the end token
            with self.subTest(sampled_tokens=tokens + 1):
                rows = self.rows + [self.a2_row(tiny_clean(tokens, 2))]
                if ok:
                    stats = train.backward_single_block(self.bundle, rows, self.contract, components=("PG",))
                    self.assertTrue(stats["replay_verified"])
                else:
                    self.refused(self.contract, pattern="v14 A2 row below the sealed token minimum", rows=rows)

    def test_schedule_steps_up_to_the_slots_are_accepted(self):
        for steps, slots in ((1, 1), (2, 2), (1, 2)):
            with self.subTest(steps=steps, slots=slots):
                contract = self.contract | {"schedule_steps": steps, "total_slots": slots}
                self.assertTrue(train.backward_single_block(self.bundle, self.rows, contract,
                                                            components=("PG",))["replay_verified"])


class ProtocolV14(unittest.TestCase):
    def test_v14_protocol_fields(self):
        protocol = plain_protocol_single("bf16", 219, contracts=PINS, version=VERSION_SINGLE_V14, schedule_steps=64)
        v13_protocol = plain_protocol_single("bf16", 219, contracts=PINS, version=VERSION_SINGLE_V13)
        contract, rl = protocol["training_contract"], protocol["training_contract"]["rl"]
        self.assertEqual(VERSION_SINGLE_V14, "error-note-single-response-v14")
        self.assertEqual((protocol["version"], contract["version"]), (VERSION_SINGLE_V14, VERSION_SINGLE_V14))
        self.assertEqual((protocol["optimizer"]["total_steps"], contract["total_slots"], contract["schedule_steps"]),
                         (64, 219, 64))
        self.assertEqual(v13_protocol["optimizer"]["total_steps"], 219)
        self.assertEqual({k: v for k, v in protocol["optimizer"].items() if k != "total_steps"},
                         {k: v for k, v in v13_protocol["optimizer"].items() if k != "total_steps"})
        self.assertEqual(protocol["image"], SINGLE_IMAGE_V14)
        self.assertEqual((protocol["image"]["min_pixels"], protocol["image"]["max_pixels"]), (1003520, 4014080))
        self.assertEqual(protocol["image"]["preprocessing"], IMAGE["preprocessing"])
        self.assertEqual(v13_protocol["image"], IMAGE)  # the earlier versions keep the module constant
        self.assertEqual(rl["reward"], SINGLE_REWARD_GATE_V9)
        self.assertEqual(rl["advantage"], {"version": "penalty-table-v4", "values": SINGLE_ADVANTAGE_V13 | {"tail_praised": 1.0}})
        self.assertEqual(rl["tail_praise"], SINGLE_TAIL_PRAISE_V14)
        self.assertEqual(rl["spiral_window"], SINGLE_SPIRAL_V13)
        self.assertEqual(rl["praise"], SINGLE_PRAISE_V13)
        self.assertEqual({k: v for k, v in rl.items() if k not in ("reward", "advantage", "tail_praise")},
                         {k: v for k, v in v13_protocol["training_contract"]["rl"].items() if k not in ("reward", "advantage")})
        self.assertEqual(contract["b_rows"], "program_accepted_note_whose_resolve_passes_the_v11_a2_checks")
        self.assertEqual(contract["b_rows"], SINGLE_B_ROWS_V14)
        self.assertEqual(contract["a2_min_tokens"], 256)
        self.assertEqual(contract["answer_contracts"], {"mcq": TURN1_EVAL_CONTRACT, "open": SCORER_V2})
        self.assertEqual(contract["answer_contracts"]["open"], SCORER_V2)
        scorer = contract["answer_scorer"]
        self.assertEqual((scorer["id"], scorer["rules"], scorer["unit_abbreviations"]),
                         (SCORER_V2, list(NORMALIZATION_RULES_V2), list(UNIT_ABBREVIATIONS)))
        self.assertNotEqual(SCORER_V2, SINGLE_OPEN_SCORER)
        self.assertEqual(v13_protocol["training_contract"]["answer_contracts"]["open"], SINGLE_OPEN_SCORER)
        self.assertEqual(v13_protocol["training_contract"]["answer_scorer"]["id"], SINGLE_OPEN_SCORER)
        self.assertNotIn("human_solution", contract)
        self.assertNotIn("H", contract["routes"])
        self.assertEqual(contract["routes"], {"RL": "LM", "A1": "LM", "A2_B": "accepted_note_self_report"})
        self.assertEqual(contract["sft_weights"], {"A1": .05, "A2": .35, "B": .05})
        self.assertEqual(contract["sft_weighting"], SINGLE_SFT_WEIGHTING_V10)
        self.assertEqual(contract["program_filters"]["lesson_leak"], LESSON_LEAK_VERSION_V2)
        self.assertEqual(contract["generation_counts"], {"t1": 1, "note": 1, "resolve": 1})
        for key in ("program_filters", "generation_counts", "length_policy", "sft_weights", "sft_weighting", "routes",
                    "replay_support", "sampling", "max_tokens", "note_max_tokens", "per_update_items"):
            self.assertEqual(contract[key], v13_protocol["training_contract"][key], key)
        self.assertEqual(set(contract) ^ set(v13_protocol["training_contract"]), {"a2_min_tokens", "schedule_steps"})
        single._protocol_check(protocol)
        self.assertEqual(single.PROTOCOL_GATES[VERSION_SINGLE_V14], SINGLE_REWARD_GATE_V9)

    def test_schedule_steps_is_required_and_validated_for_v14_only(self):
        for label, steps in (("missing", None), ("zero", 0), ("negative", -1), ("above the slots", 220), ("bool", True),
                             ("float", 64.0), ("text", "64")):
            with self.subTest(label), self.assertRaisesRegex(ValueError, "schedule_steps belongs to v14"):
                plain_protocol_single("bf16", 219, contracts=PINS, version=VERSION_SINGLE_V14, schedule_steps=steps)
        for steps in (1, 64, 219):
            with self.subTest(steps=steps):
                protocol = plain_protocol_single("bf16", 219, contracts=PINS, version=VERSION_SINGLE_V14, schedule_steps=steps)
                self.assertEqual((protocol["optimizer"]["total_steps"], protocol["training_contract"]["schedule_steps"]),
                                 (steps, steps))
        for version in (VERSION_SINGLE, VERSION_SINGLE_V12, VERSION_SINGLE_V13):
            for steps in (64, 1, 0, 219):
                with self.subTest(version=version, steps=steps), \
                        self.assertRaisesRegex(ValueError, "schedule_steps belongs to v14"):
                    plain_protocol_single("bf16", 219, contracts=PINS, version=version, schedule_steps=steps)

    def test_v11_v12_and_v13_protocols_are_unchanged_by_v14(self):
        pins = {key: (str(index) * 64)[:64] for index, key in enumerate(sorted(SINGLE_CONTRACT_HASHES), 1)}
        for version, sha in ((VERSION_SINGLE, "cec12b01fdaf030fb8ae3071046615f2ef3ab581bcacdb13fb6bc17140c1d6a1"),
                             (VERSION_SINGLE_V12, "37efa42a25e5723b7406ca115c53dbbc65aefce61c88b3bb404f6a6ade982055"),
                             (VERSION_SINGLE_V13, "401859866f1befde2316430eee0bc345b9bfab63aee8508f414e8d485820bb1d")):
            with self.subTest(version=version):
                protocol = plain_protocol_single("bf16", 219, contracts=pins, version=version)
                self.assertEqual(protocol["sha256"], sha)
                self.assertNotIn("schedule_steps", protocol["training_contract"])
                self.assertEqual(protocol["optimizer"]["total_steps"], 219)
                single._protocol_check(protocol)
        self.assertEqual(plain_protocol_single("bf16", 219, contracts=pins)["sha256"],
                         plain_protocol_single("bf16", 219, contracts=pins, version=VERSION_SINGLE)["sha256"])

    def test_the_returned_protocol_is_a_deep_copy_of_the_module_constants(self):
        tail, table, image = copy.deepcopy(SINGLE_TAIL_PRAISE_V14), dict(SINGLE_ADVANTAGE_V14), dict(SINGLE_IMAGE_V14)
        spiral, praise = copy.deepcopy(SINGLE_SPIRAL_V13), copy.deepcopy(SINGLE_PRAISE_V13)
        before = protocol_v14(219, 64)
        mutated = protocol_v14(219, 64)
        rl = mutated["training_contract"]["rl"]
        rl["tail_praise"]["marker_token_ids"].append(1)
        rl["tail_praise"]["reasons"].clear()
        rl["tail_praise"]["min_tokens"] = 1
        rl["tail_praise"]["applies_to"].append("A1")
        rl["spiral_window"]["marker_token_ids"].append(1)
        rl["praise"]["min_tokens"] = 1
        rl["advantage"]["values"]["tail_praised"] = 5.0
        mutated["image"]["min_pixels"] = 1
        mutated["training_contract"]["answer_contracts"]["open"] = "x"
        mutated["training_contract"]["answer_scorer"]["rules"].clear()
        self.assertEqual((SINGLE_TAIL_PRAISE_V14, SINGLE_ADVANTAGE_V14, SINGLE_IMAGE_V14), (tail, table, image))
        self.assertEqual((SINGLE_SPIRAL_V13, SINGLE_PRAISE_V13), (spiral, praise))
        self.assertEqual(protocol_v14(219, 64), before)

    def test_a_tampered_v14_protocol_is_refused(self):
        tampered = protocol_v14()
        tampered["training_contract"]["rl"]["reward"] = SINGLE_REWARD_GATE_V8  # gate v8 under a v14 protocol
        with self.assertRaisesRegex(ValueError, "invalid single-response protocol"):
            single._protocol_check(tampered)
        with self.assertRaisesRegex(ValueError, "nonnegative weights"):  # H belongs to v12 only
            plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V14, schedule_steps=2,
                                  sft_weights={"A1": .05, "A2": .35, "B": .05, "H": .02})


class GradingWiring(unittest.TestCase):
    def test_scorer_v2_grades_with_units_and_the_v8_scorer_without(self):
        for contract, units in ((SCORER_V2, True), (SINGLE_OPEN_SCORER, False)):
            with self.subTest(contract=contract), patch.object(single_grading, "grade_single",
                                                               return_value=("CORRECT", "855")) as grade:
                self.assertEqual(single.grade_answer("verifier", contract, "p0", "text", "stop", "plain", actor="actor"),
                                 ("CORRECT", "855"))
                grade.assert_called_once_with("verifier", "actor", "p0", "text", "stop", "plain", units=units)

    def test_both_open_scorers_need_the_item_actor(self):
        for contract in (SCORER_V2, SINGLE_OPEN_SCORER):
            with self.subTest(contract=contract), self.assertRaisesRegex(ValueError, "needs the item actor"):
                single.grade_answer("verifier", contract, "p0", "text", "stop", "plain")

    def test_the_mcq_contract_still_uses_the_verifier(self):
        verifier = SimpleNamespace(grade=lambda *args, **kwargs: SimpleNamespace(status="CORRECT", extracted="B"))
        with patch.object(single_grading, "grade_single") as grade:
            self.assertEqual(single.grade_answer(verifier, TURN1_EVAL_CONTRACT, "p0", "B", "stop", "plain"), ("CORRECT", "B"))
            grade.assert_not_called()


class RuntimeV14(unittest.TestCase):
    NAME = v13.RuntimeV13.NAME

    def case(self, *, version=VERSION_SINGLE_V14, marker_ids=(8,), marker="a"):
        """The runtime fixture under a v13 or v14 protocol. The tiny tokenizer decodes ID 8 as the word "a", so both
        marker rules are patched to it; the protocol is built inside the patch so it equals the patched constants."""
        for rule, extra in ((SINGLE_SPIRAL_V13, {"marker": marker}), (SINGLE_TAIL_PRAISE_V14, {})):
            patcher = patch.dict(rule, marker_token_ids=list(marker_ids), **extra)
            patcher.start()
            self.addCleanup(patcher.stop)
        case = runtime_tests.RuntimeTests(self.NAME)
        case.setUp()
        self.addCleanup(case.doCleanups)
        extra = {"schedule_steps": 64} if version == VERSION_SINGLE_V14 else {}
        case.protocol = plain_protocol_single("bf16", 219, contracts=PINS, token_scale=1024., version=version, **extra)
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

    @staticmethod
    def steps(count, final="B"):
        return " ".join(f"step{j}" for j in range(count)) + f" Final answer: {final}"

    @staticmethod
    def rewrite(case, stage, pid, text, actions):
        """Make the fake engine return `text` with these sampled IDs for one stage (and one question)."""
        original = case.stream

        def stream(requests, stats, stop_after):
            for request, output in original(requests, stats, stop_after):
                if request["stage"] == stage and request["problem_id"] == pid:
                    generation = output["generation"]
                    output["text"] = generation["text"] = text
                    generation["completion_token_ids"] = list(actions)
                    generation["sampling_logprobs"] = [-.5] * len(actions)
                    generation["support_sha256"] = ["c" * 64] * len(actions)
                    output["completion_tokens"] = len(actions)
                yield request, output

        case.stream = stream

    def long_resolve(self, case, count, text=None):
        self.rewrite(case, "resolve", "p1", text or self.steps(count - 1), list(range(100, 100 + count - 1)) + [2])

    def test_a_short_resolve_is_no_a2_row_but_its_note_is_still_a_b_row(self):
        closed = self.closed(self.case())  # the fixture's re-solve is 41 tokens
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "RL", "B"])
        decisions = closed["row_decisions"]
        self.assertEqual(decisions["p1:A2"], {"accepted": False, "method": "program", "reason": "A2_UNDER_MIN_TOKENS"})
        self.assertEqual(decisions["p1:B"], {"accepted": True, "method": "program", "reason": None})
        self.assertEqual(decisions["p0:A1"], {"accepted": False, "method": "program", "reason": "A1_NOT_PRAISED"})
        self.assertEqual(closed["rows"][2]["row_id"], "p1:B")
        contract = closed["contract"]
        self.assertEqual((contract["version"], contract["a2_min_tokens"], contract["b_rows"], contract["schedule_steps"]),
                         (VERSION_SINGLE_V14, 256, SINGLE_B_ROWS_V14, 64))
        self.assertEqual(contract["rl"]["reward"], SINGLE_REWARD_GATE_V9)
        for row in closed["rows"][:2]:
            self.assertEqual(row["reward_evidence"]["version"], SINGLE_REWARD_GATE_V9)
            self.assertIsNone(row["reward_evidence"]["decision"]["tail"])
            self.assertEqual(single.validate_single_reward(row), row["reward_evidence"]["decision"])
        p0, p1 = closed["rows"][:2]
        self.assertEqual((p0["reward"], p0["advantage"], p1["reward"], p1["advantage"]), (1, 0.0, -1, 0.0))

    def test_a_correct_resolve_of_at_least_256_tokens_is_an_a2_row_followed_by_its_b_row(self):
        for count, kinds, reason in ((256, ["RL", "RL", "A2", "B"], None), (300, ["RL", "RL", "A2", "B"], None),
                                     (255, ["RL", "RL", "B"], "A2_UNDER_MIN_TOKENS")):
            with self.subTest(sampled_tokens=count):
                case = self.case()
                self.long_resolve(case, count)
                closed = self.closed(case)
                self.assertEqual([r["kind"] for r in closed["rows"]], kinds)
                self.assertEqual(closed["row_decisions"]["p1:A2"], {"accepted": reason is None, "method": "program",
                                                                     "reason": reason})
                self.assertTrue(closed["row_decisions"]["p1:B"]["accepted"])
                if reason is None:
                    a2 = next(r for r in closed["rows"] if r["kind"] == "A2")
                    self.assertEqual(len(a2["generation"]["completion_token_ids"]), count)

    def test_a_failed_resolve_gives_no_b_row_in_v14_and_still_one_in_v13(self):
        wrong = self.steps(38, final="A")
        for version, kinds, b_row in ((VERSION_SINGLE_V14, ["RL", "RL"], False), (VERSION_SINGLE_V13, ["RL", "RL", "B"], True)):
            with self.subTest(version=version):
                case = self.case(version=version)
                self.long_resolve(case, 300, text=wrong)
                closed = self.closed(case)
                self.assertEqual([r["kind"] for r in closed["rows"]], kinds)
                self.assertEqual(closed["row_decisions"]["p1:A2"], {"accepted": False, "method": "program",
                                                                     "reason": "A2_NOT_ADMISSIBLE"})
                self.assertEqual("p1:B" in closed["row_decisions"], b_row)
                if b_row:
                    self.assertTrue(closed["row_decisions"]["p1:B"]["accepted"])

    def test_a_resolve_that_fails_a_v11_a2_check_gives_no_b_row_in_v14_even_when_it_is_long(self):
        echo = "As per the lesson, read each label. " + self.steps(300)
        for version, kinds, b_row in ((VERSION_SINGLE_V14, ["RL", "RL"], False), (VERSION_SINGLE_V13, ["RL", "RL", "B"], True)):
            with self.subTest(version=version):
                case = self.case(version=version)
                self.long_resolve(case, 300, text=echo)
                closed = self.closed(case)
                self.assertEqual([r["kind"] for r in closed["rows"]], kinds)
                self.assertEqual(closed["row_decisions"]["p1:A2"], {"accepted": False, "method": "program",
                                                                     "reason": "A2_RESOLVE_LEAK:lesson"})
                self.assertEqual("p1:B" in closed["row_decisions"], b_row)

    def test_a_correct_t1_rejected_for_repetition_with_a_clean_tail_is_praised_on_its_tail_and_gets_no_a1_row(self):
        case = self.case()
        ids = rep(240) + [8] + varied(60) + [2]
        self.rewrite(case, "t1", "p0", self.steps(250), ids)
        closed = self.closed(case)
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "RL", "B"])
        self.assertEqual(closed["row_decisions"]["p0:A1"],
                         {"accepted": False, "method": "program", "reason": "A1_NOT_ADMISSIBLE"})
        p0 = closed["rows"][0]
        decision = p0["reward_evidence"]["decision"]
        self.assertEqual((p0["problem_id"], len(p0["generation"]["completion_token_ids"])), ("p0", 302))
        self.assertEqual(decision["tail"], tail_window(ids))
        self.assertEqual(decision["tail"], [241, 302])
        self.assertEqual((decision["accepted"], decision["penalty"], decision["reason"], decision["praise"]),
                         (False, None, "T1_INELIGIBLE_REPETITION", False))
        self.assertEqual((p0["reward"], p0["advantage"]), (0, 1.0))
        self.assertEqual(p0["reward_evidence"]["advantage"], {"version": "penalty-table-v4", "value": 1.0})
        self.assertEqual(single.validate_single_reward(p0), decision)
        self.assertEqual(closed["contract"]["reward_evidence_sha256"]["p0"], digest(p0["reward_evidence"]))
        self.assertEqual(closed["contract"]["rl"]["tail_praise"], SINGLE_TAIL_PRAISE_V14)

    def test_a_clean_long_correct_t1_keeps_its_a1_row_and_is_praised_on_the_whole_row_without_a_tail(self):
        case = self.case()
        self.rewrite(case, "t1", "p0", self.steps(250), varied(300) + [2])
        closed = self.closed(case)
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "A1", "RL", "B"])
        self.assertEqual(closed["row_decisions"]["p0:A1"], {"accepted": True, "method": "program", "reason": None})
        p0 = closed["rows"][0]
        self.assertEqual((p0["reward"], p0["advantage"], p0["reward_evidence"]["decision"]["praise"],
                          p0["reward_evidence"]["decision"]["tail"]), (1, 1.0, True, None))

    def test_a_t1_rejected_for_repetition_without_a_marker_or_under_256_tokens_gets_no_tail(self):
        for label, ids in (("no marker", rep(300) + [2]), ("under 256", rep(150) + [8] + varied(60) + [2])):
            with self.subTest(label):
                case = self.case()
                self.rewrite(case, "t1", "p0", self.steps(250), ids)
                closed = self.closed(case)
                p0 = closed["rows"][0]
                self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "RL", "B"])
                self.assertEqual((p0["reward"], p0["advantage"], p0["reward_evidence"]["decision"]["tail"]), (0, 0.0, None))

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


class EncodeImageConfig(unittest.TestCase):
    """`encode_single_row` prepares a row's images with the plan's image config (v14: the B2 pixel budget) and with the
    module constant IMAGE when none is given (v11-v13). `prepare_protocol_inputs` is patched to record its arguments."""
    FIXTURE = "test_original_ids_and_actual_eos_survive_decode_encode_counterexample"

    def encode(self, kind="RL", **kwargs):
        fixture = loss_tests.SingleEncoding(self.FIXTURE)  # the row / prepared-input fixtures of the loss module
        processor = SimpleNamespace(tokenizer=SimpleNamespace(all_special_tokens=[]))
        with patch.object(train, "prepare_protocol_inputs", return_value=(fixture.prepared(), [])) as prepare:
            result = train.encode_single_row(processor, None, fixture.row(kind), [object()], **kwargs)
        self.assertEqual(result["target_len"], 3)
        prepare.assert_called_once()
        self.assertEqual((len(prepare.call_args.args), prepare.call_args.kwargs), (5, {}))  # image config is the last argument
        return prepare.call_args.args[-1]

    def test_the_two_image_configs_are_distinguishable(self):
        self.assertNotEqual(SINGLE_IMAGE_V14, IMAGE)
        self.assertEqual((SINGLE_IMAGE_V14["min_pixels"], SINGLE_IMAGE_V14["max_pixels"]), (1003520, 4014080))
        self.assertEqual((IMAGE["min_pixels"], IMAGE["max_pixels"]), (262144, 1310720))

    def test_the_image_argument_is_the_config_the_row_is_prepared_with(self):
        for kind in ("RL", "A2"):
            with self.subTest(kind=kind):
                config = self.encode(kind, image=SINGLE_IMAGE_V14)
                self.assertEqual(config, SINGLE_IMAGE_V14)
                self.assertNotEqual(config, IMAGE)
        custom = {"min_pixels": 1, "max_pixels": 2, "preprocessing": "x"}  # whatever is given is what is used
        self.assertEqual(self.encode(image=custom), custom)

    def test_without_an_image_argument_the_module_constant_is_used(self):
        for kind in ("RL", "A2"):
            with self.subTest(kind=kind):
                config = self.encode(kind)
                self.assertEqual(config, IMAGE)
                self.assertNotEqual(config, SINGLE_IMAGE_V14)
        self.assertEqual(self.encode(image=None), IMAGE)

    def test_the_v14_protocol_image_is_the_config_a_v14_plan_passes(self):
        self.assertEqual(self.encode(image=protocol_v14()["image"]), SINGLE_IMAGE_V14)
        old = plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V13)
        self.assertEqual(self.encode(image=old["image"]), IMAGE)  # reading protocol["image"] changes nothing before v14


class CliV14(unittest.TestCase):
    REQUIRED = ["prepare", "--order-sha256", "a" * 64] + [
        item for name in ("build-root", "order", "frozen-approval", "source-provenance", "training-rules",
                          "model-path", "output") for item in ("--" + name, "/x")]

    def test_cli_accepts_v14_and_keeps_v12_as_the_default(self):
        parser = single.build_parser()
        self.assertEqual(parser.parse_args(self.REQUIRED).single_version, "v12")
        self.assertEqual(parser.parse_args(self.REQUIRED + ["--single-version", "v14"]).single_version, "v14")
        self.assertEqual(single.SINGLE_CLI_VERSIONS["v14"], VERSION_SINGLE_V14)
        self.assertEqual(single.SINGLE_CLI_VERSIONS["v13"], VERSION_SINGLE_V13)

    def test_prepare_seals_a_v14_protocol_whose_schedule_steps_are_the_planned_blocks(self):
        from fine_tuning.error_note.answer_contract import VERSION as RULES_VERSION
        from fine_tuning.error_note.source_evidence import VERSION as SOURCE_VERSION
        from fine_tuning.notebook_rl.schema import ActorItem, to_json

        out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        actors = [ActorItem(f"p{i}", "authored-test", "fixture", "plain", f"Question {i}", ("red", "blue"), "mcq",
                            ("fixture.png",), ("a" * 64,)) for i in range(20)]
        items = [{"problem_id": a.problem_id, "subject": "Biology", "source": a.source} for a in actors]
        root = out / "vault"
        (root / "train").mkdir(parents=True)
        (root / "train" / "manifest.yaml").write_text("authored fixture train")
        (root / "manifest.yaml").write_text("authored fixture union")
        train_sha = single.file_sha(root / "train" / "manifest.yaml")
        sources = {a.problem_id: PROVENANCE | {"actor_sha256": text_sha256(to_json(a)), "image_sha256": list(a.image_sha256),
                   "vault_manifest_sha256": train_sha} for a in actors}
        (out / "source.json").write_text(json.dumps({"version": SOURCE_VERSION, "items": sources}))
        (out / "rules.json").write_text(json.dumps({"version": RULES_VERSION, "vault_manifest_sha256": train_sha, "items": {}}))
        (out / "model").mkdir()
        (out / "model" / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "torch_dtype": "bfloat16"}))

        def args(name, **extra):
            return SimpleNamespace(build_root=root, source_provenance=out / "source.json", training_rules=out / "rules.json",
                                   order_sha256="a" * 64, model_path=out / "model", output=out / name, mode="local-smoke",
                                   token_scale=1024., max_num_seqs=1, max_num_batched_tokens=24576, token_budget=0,
                                   stop_after_seconds=None, limit=10, **extra)

        with patch.object(single, "_validate_data", return_value=({"items": items}, {"authored_fixture": True})), \
                patch("fine_tuning.notebook_rl.vault.read_actors", return_value=actors), \
                patch.object(single, "source_manifest", return_value={"base_commit": "a" * 40, "actual_files": {}}), \
                patch.object(single, "generation_identity", return_value={"model_files_sha256": "b" * 64}):
            plan = single.prepare(args("v14", single_version="v14"))
            old = single.prepare(args("v13", single_version="v13"))
        protocol = plan["protocol"]
        self.assertEqual((protocol["version"], len(plan["blocks"])), (VERSION_SINGLE_V14, 2))  # 20 items: blocks of 16 and 4
        self.assertEqual(protocol["training_contract"]["schedule_steps"], len(plan["blocks"]))
        self.assertEqual(protocol["optimizer"]["total_steps"], len(plan["blocks"]))
        self.assertEqual(protocol["training_contract"]["total_slots"], len(plan["blocks"]))
        self.assertEqual(protocol["image"], SINGLE_IMAGE_V14)
        self.assertEqual(protocol["training_contract"]["rl"]["reward"], SINGLE_REWARD_GATE_V9)
        self.assertNotIn("schedule_steps", old["protocol"]["training_contract"])
        self.assertEqual(old["protocol"]["image"], IMAGE)


if __name__ == "__main__":
    unittest.main()
