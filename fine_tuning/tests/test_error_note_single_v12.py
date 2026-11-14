"""SINGLE v12 CPU checks: reward gate v7 (CONTAINED_OPEN, advantage table v2), lesson leak rule v2, B rows after a failed
re-solve and H rows (human-written solutions). Tiny random models only; these are not pretrained Qwen learning evidence."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.error_note import references, single
from fine_tuning.error_note import train as trainer
from fine_tuning.error_note import train_v4 as train
from fine_tuning.error_note.audit import text_sha256
from fine_tuning.error_note.contract import (HUMAN_TARGET_SOURCE, KIND_HUMAN, answer_tokens, b_target_problem,
                                             contained_answer, hint_leak_single, hint_leak_single_v2)
from fine_tuning.error_note.protocol import (SINGLE_ADVANTAGE_V12, SINGLE_REWARD_GATE, SINGLE_REWARD_GATE_V7,
                                             SINGLE_SFT_WEIGHTS_V12, VERSION_SINGLE, VERSION_SINGLE_V12,
                                             plain_protocol_single)
from fine_tuning.notebook_rl.input_preparation import digest
from fine_tuning.tests import test_error_note_single_runtime as runtime_tests
from fine_tuning.tests.test_error_note_single_loss import (bind_reward_fixture, block, gradients,
                                                           perturbed_factory)
from fine_tuning.tests.test_error_note_single_v11 import as_v11
from fine_tuning.tests.test_error_note_train import tiny_factory, tiny_processor
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE

PINS = {key: "a" * 64 for key in ("source", "dataset", "order", "data_admission")}
GOLD_32 = PROVENANCE | {"label": {"value": "32", "sha256": text_sha256("32")}}
GOLD_CHILE = PROVENANCE | {"label": {"value": "Chile", "sha256": text_sha256("Chile")}}
MCQ_PROMPT = "Question: authored CPU fixture\nOptions:\nA. 5\nB. 6" + single._MCQ_P0_END
CONTAINED = "the country with the longest coastline is chile"
_SHORT = None


def setUpModule():
    """Tiny-model rows are a few tokens long; the 32-token rule is exercised with patched lengths below."""
    global _SHORT
    _SHORT = patch.object(single, "SOLUTION_MIN_TOKENS", 1)
    _SHORT.start()


def tearDownModule():
    _SHORT.stop()


def bind_v12(row, *, finish_reason="stop", extracted=None, option_count=0, provenance=PROVENANCE,
             gate=SINGLE_REWARD_GATE_V7):
    """bind_reward_fixture for gate v7: the same row defaults, the item's option count and the grader's extracted answer."""
    for key, value in {"prompt": "Question: authored CPU fixture", "target": "Authored final answer: B", "target_source": "T1",
                       "image_paths": ["fixture.png"], "image_sha256": ["a" * 64], "subject": "Biology",
                       "source": "authored-test"}.items():
        row.setdefault(key, value)
    row["generation"].setdefault("text", row["target"])
    t1 = {"text": row["target"], "generation": row["generation"], "prompt": row["prompt"], "status": row["status"],
          "finish_reason": finish_reason, "extracted": extracted}
    return single.bind_single_reward(row, t1, provenance, gate=gate, option_count=option_count)


def as_v12(rows, contract, finish=None):
    """Rebind every RL row under gate v7 (open items) and turn the tiny v9 contract into the v12 block contract."""
    contract = copy.deepcopy(contract)
    v12 = plain_protocol_single("bf16", 1, contracts=PINS, version=VERSION_SINGLE_V12)["training_contract"]
    contract["rl"].update({key: v12["rl"][key] for key in
                           ("baseline", "advantage", "token_scale", "row_token_cap", "row_token_cap_exempt",
                            "negative_logprob", "reward")})
    contract.update(version=VERSION_SINGLE_V12, sft_weights=v12["sft_weights"], sft_weighting=v12["sft_weighting"],
                    human_solution=v12["human_solution"])
    for row in rows:
        if row["kind"] == "RL":
            bind_v12(row, finish_reason=(finish or {}).get(row["problem_id"], "stop"), option_count=0)
    contract.update(
        reward_provenance_sha256={r["problem_id"]: digest(r["reward_evidence"]["provenance"]) for r in rows if r["kind"] == "RL"},
        reward_evidence_sha256={r["problem_id"]: digest(r["reward_evidence"]) for r in rows if r["kind"] == "RL"})
    return contract


def case_row(base, *, status, finish="stop", ids=None, text=None, extracted=None, prompt=None, provenance=GOLD_32,
             option_count=0):
    """A copy of one fixture RL row graded as `status`, bound under gate v7."""
    row = copy.deepcopy(base)
    row["status"] = status
    if ids is not None:
        row["generation"]["completion_token_ids"] = ids
    if text is not None:
        row["target"] = row["generation"]["text"] = text
    if prompt is not None:
        row["prompt"] = prompt
    return bind_v12(row, finish_reason=finish, extracted=extracted, option_count=option_count, provenance=provenance)


def human_fields(lecture_words=16, solution_words=16):
    """Lecture and solution built from the tiny tokenizer's vocabulary: one word is one token."""
    return " ".join(["a", "b"] * (lecture_words // 2)), " ".join(["question", "a"] * (solution_words // 2))


def scienceqa(lecture, solution):
    return PROVENANCE | {"source": "hf-scienceqa", "annotations": [{"field": "lecture", "value": lecture},
                                                                    {"field": "solution", "value": solution}]}


class TinyThreads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)


class AdvantageTableV12(TinyThreads):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.base = block(tiny_factory(), count=1)[0][0]

    def test_every_gate_v7_case_round_trips_and_forged_advantage_decision_or_item_is_refused(self):
        long_ids, loop = list(range(100, 140)), "Wait, let me count again. " * 200
        reasoned = "The angle sum gives 32 degrees here. Final answer: 32"
        inside = "the angle is 32 degrees"
        cases = (  # status, finish, ids, text, extracted, reward, penalty, table key
            ("CORRECT", "stop", long_ids, reasoned, "32", 1, None, "accepted"),
            ("CORRECT", "stop", None, None, "32", 0, None, "other"),  # bare correct answer: 0, never negative
            ("CORRECT", "stop", long_ids, loop, "32", 0, None, "other"),
            ("CORRECT", "length", long_ids, reasoned, "32", 0, None, "other"),
            ("UNVERIFIABLE", "stop", None, None, None, 0, None, "other"),
            ("UNVERIFIABLE", "length", long_ids, loop, None, 0, None, "other"),
            ("INCORRECT", "stop", None, None, "A", -1, "BARE_WRONG", "BARE_WRONG"),
            ("INVALID_NO_FINAL", "stop", long_ids, reasoned, None, -1, "NO_COMMITTED_ANSWER", "NO_COMMITTED_ANSWER"),
            ("INCORRECT", "stop", long_ids, loop, "A", -1, "LOOP", "LOOP"),
            ("TRUNCATED", "length", long_ids, loop, None, -1, "LOOP", "LOOP"),
            ("TRUNCATED", "length", long_ids, reasoned, None, -1, "TRUNCATED_NO_LOOP", "TRUNCATED_NO_LOOP"),
            ("INCORRECT", "stop", long_ids, reasoned, "31", -1, "REASONED_WRONG", "REASONED_WRONG"),  # advantage 0.0
            ("INVALID_MULTIPLE", "stop", long_ids, reasoned, "31 or 33", -1, "REASONED_WRONG", "REASONED_WRONG"),
            ("INVALID_NO_FINAL", "stop", long_ids, reasoned, "31", -1, "REASONED_WRONG", "REASONED_WRONG"),
            ("INCORRECT", "stop", long_ids, reasoned, inside, 0, None, "CONTAINED_OPEN"),
            ("INCORRECT", "stop", None, None, inside, 0, None, "CONTAINED_OPEN"),  # a short contained answer is no BARE_WRONG
            # Loop, cut-off and uncommitted penalties outrank CONTAINED_OPEN (coordinator correction 2026-11-11).
            ("INCORRECT", "length", long_ids, loop, inside, -1, "LOOP", "LOOP"),
            ("INCORRECT", "stop", long_ids, loop, inside, -1, "LOOP", "LOOP"),
            ("INCORRECT", "length", long_ids, reasoned, inside, -1, "TRUNCATED_NO_LOOP", "TRUNCATED_NO_LOOP"))
        seen = set()
        with patch.object(single, "SOLUTION_MIN_TOKENS", 32):
            for status, finish, ids, text, extracted, reward, penalty, key in cases:
                with self.subTest(status=status, finish=finish, key=key, text=text and text[:9]):
                    row = case_row(self.base, status=status, finish=finish, ids=ids, text=text, extracted=extracted)
                    decision = row["reward_evidence"]["decision"]
                    value = SINGLE_ADVANTAGE_V12[key]
                    self.assertEqual((row["reward"], decision["penalty"], row["advantage"]), (reward, penalty, value))
                    self.assertIs(type(row["advantage"]), float)
                    self.assertEqual(decision["reason"] == single.CONTAINED_OPEN, key == "CONTAINED_OPEN")
                    self.assertEqual(row["reward_evidence"]["version"], SINGLE_REWARD_GATE_V7)
                    self.assertEqual(row["reward_evidence"]["advantage"], {"version": "penalty-table-v2", "value": value})
                    self.assertEqual(row["reward_evidence"]["item"], {"option_count": 0})
                    self.assertEqual(single.validate_single_reward(row), decision)
                    self.assertEqual(single.single_advantage(decision, SINGLE_REWARD_GATE_V7), value)
                    seen.add(key)
                    forged = copy.deepcopy(row)
                    forged["advantage"] = value - .25
                    with self.assertRaisesRegex(ValueError, "v12 penalty table"):
                        single.validate_single_reward(forged)
                    forged = copy.deepcopy(row)
                    forged["reward_evidence"]["advantage"]["value"] = value - .25
                    with self.assertRaisesRegex(ValueError, "v12 penalty table"):
                        single.validate_single_reward(forged)
                    for edit in ({"reason": "T1_FORGED"}, {"penalty": "LOOP" if penalty != "LOOP" else None},
                                 {"accepted": not decision["accepted"]}):
                        forged = copy.deepcopy(row)
                        forged["reward_evidence"]["decision"].update(edit)
                        with self.subTest(edit=edit), self.assertRaisesRegex(ValueError, "differs from the program gate"):
                            single.validate_single_reward(forged)
                    forged = copy.deepcopy(row)
                    forged["reward_evidence"]["item"]["option_count"] = 3  # an MCQ count against an open P0 text
                    with self.assertRaisesRegex(ValueError, "option count"):
                        single.validate_single_reward(forged)
                    forged = copy.deepcopy(row)
                    del forged["reward_evidence"]["item"]
                    with self.assertRaisesRegex(ValueError, "malformed or unsupported"):
                        single.validate_single_reward(forged)
        self.assertEqual(seen, set(SINGLE_ADVANTAGE_V12))

    def test_reasoned_wrong_keeps_reward_minus_one_with_a_zero_advantage(self):
        row = case_row(self.base, status="INCORRECT", ids=list(range(100, 140)), extracted="31",
                       text="The angle sum gives 32 degrees here. Final answer: 31")
        self.assertEqual((row["reward"], row["advantage"], row["reward_evidence"]["decision"]["penalty"]),
                         (-1, 0.0, "REASONED_WRONG"))
        v11 = copy.deepcopy(row)
        bind_reward_fixture(v11, gate=SINGLE_REWARD_GATE)
        self.assertEqual((v11["reward"], v11["advantage"]), (-1, -.5))  # v11 keeps its -0.5

    def test_option_count_belongs_to_gate_v7_only(self):
        row = copy.deepcopy(self.base)
        with self.assertRaisesRegex(ValueError, "option count belongs to gates v7 and v8"):
            bind_v12(copy.deepcopy(row), option_count=None)
        for gate in (SINGLE_REWARD_GATE, "single-t1-program-gate-v5"):
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, "option count belongs to gates v7 and v8|v10 source"):
                bind_v12(copy.deepcopy(row), option_count=0, gate=gate)
        with self.assertRaisesRegex(ValueError, "option count"):
            bind_v12(copy.deepcopy(row), option_count=True)  # not an int
        with self.assertRaisesRegex(ValueError, "option count"):
            bind_v12(copy.deepcopy(row), option_count=-1)


class ContainedOpen(TinyThreads):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.base = block(tiny_factory(), count=1)[0][0]

    def test_answer_tokens_and_containment_rule(self):
        for gold, extracted, expected in (("Chile", CONTAINED, True), ("5", "25", False), ("Chile", "chile", False),
                                          ("8.6", "8.6 kg", True), ("8.6", "18.6", False), ("", "anything", False),
                                          ("two words", "the two words here", True), ("two words", "words two", False)):
            with self.subTest(gold=gold, extracted=extracted):
                self.assertIs(contained_answer(gold, extracted), expected)
        self.assertEqual(answer_tokens("8.6 kg"), ["8.6", "kg"])
        self.assertEqual(answer_tokens("The country is Chile."), ["the", "country", "is", "chile"])

    def test_open_incorrect_answer_containing_the_gold_is_contained_open_even_when_short(self):
        row = case_row(self.base, status="INCORRECT", extracted=CONTAINED, provenance=GOLD_CHILE)
        decision = row["reward_evidence"]["decision"]
        self.assertEqual((decision["reason"], decision["penalty"], decision["accepted"]),
                         (single.CONTAINED_OPEN, None, False))
        self.assertEqual((row["reward"], row["advantage"]), (0, 0.0))
        with patch.object(single, "SOLUTION_MIN_TOKENS", 32):
            short = case_row(self.base, status="INCORRECT", extracted=CONTAINED, provenance=GOLD_CHILE)
            self.assertEqual(short["reward_evidence"]["decision"]["reason"], single.CONTAINED_OPEN)
            self.assertIsNone(short["reward_evidence"]["decision"]["penalty"])
            self.assertEqual((short["reward"], short["advantage"]), (0, 0.0))
            # The same short row with a non-containing answer is a bare wrong answer: the category took the penalty away.
            wrong = case_row(self.base, status="INCORRECT", extracted="25",
                             provenance=PROVENANCE | {"label": {"value": "5", "sha256": text_sha256("5")}})
            self.assertEqual(wrong["reward_evidence"]["decision"]["penalty"], "BARE_WRONG")
            self.assertEqual((wrong["reward"], wrong["advantage"]), (-1, -1.0))

    def test_numeric_containment_is_token_based(self):
        for gold, extracted, category in (("5", "25", False), ("5", "x = 5", True), ("8.6", "18.6", False),
                                          ("8.6", "8.6 kg", True)):
            with self.subTest(gold=gold, extracted=extracted):
                row = case_row(self.base, status="INCORRECT", extracted=extracted,
                               provenance=PROVENANCE | {"label": {"value": gold, "sha256": text_sha256(gold)}})
                decision = row["reward_evidence"]["decision"]
                self.assertEqual(decision["reason"] == single.CONTAINED_OPEN, category)
                self.assertEqual(decision["penalty"] is None, category)

    def test_mcq_items_are_never_contained_open(self):
        open_row = case_row(self.base, status="INCORRECT", extracted=CONTAINED, provenance=GOLD_CHILE)
        mcq = case_row(self.base, status="INCORRECT", extracted=CONTAINED, provenance=GOLD_CHILE,
                       prompt=MCQ_PROMPT, option_count=2)
        self.assertEqual(open_row["reward_evidence"]["decision"]["reason"], single.CONTAINED_OPEN)
        self.assertNotEqual(mcq["reward_evidence"]["decision"]["reason"], single.CONTAINED_OPEN)
        self.assertIsNotNone(mcq["reward_evidence"]["decision"]["penalty"])
        self.assertEqual(mcq["reward_evidence"]["item"], {"option_count": 2})
        self.assertEqual(single.validate_single_reward(mcq), mcq["reward_evidence"]["decision"])
        # An MCQ item needs no gold label: containment is never evaluated.
        case_row(self.base, status="INCORRECT", extracted="A", prompt=MCQ_PROMPT, option_count=2, provenance=PROVENANCE)

    def test_option_count_must_agree_with_the_p0_text(self):
        with self.assertRaisesRegex(ValueError, "option count matching its P0 text"):
            case_row(self.base, status="INCORRECT", extracted="A", prompt=MCQ_PROMPT, option_count=0)
        with self.assertRaisesRegex(ValueError, "option count matching its P0 text"):
            case_row(self.base, status="INCORRECT", extracted="A", option_count=2)
        with self.assertRaisesRegex(ValueError, "option count matching its P0 text"):  # CORRECT rows are checked too
            case_row(self.base, status="CORRECT", extracted="A", prompt=MCQ_PROMPT, option_count=0)

    def test_open_incorrect_with_an_answer_needs_a_source_bound_label(self):
        with self.assertRaisesRegex(ValueError, "source-bound gold label"):
            case_row(self.base, status="INCORRECT", extracted=CONTAINED, provenance=PROVENANCE)
        wrong_sha = PROVENANCE | {"label": {"value": "Chile", "sha256": "0" * 64}}
        with self.assertRaisesRegex(ValueError, "source-bound gold label"):
            case_row(self.base, status="INCORRECT", extracted=CONTAINED, provenance=wrong_sha)
        for label in ("Chile", None, {"value": 3, "sha256": "0" * 64}):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "source-bound gold label"):
                case_row(self.base, status="INCORRECT", extracted=CONTAINED, provenance=PROVENANCE | {"label": label})
        # No extracted answer or a graded-correct row never reads the label.
        for status, extracted in (("INCORRECT", None), ("INCORRECT", "  "), ("CORRECT", "32"), ("INVALID_NO_FINAL", None)):
            with self.subTest(status=status, extracted=extracted):
                case_row(self.base, status=status, extracted=extracted, provenance=PROVENANCE)
        # A tampered label on a stored row is caught by validate_single_reward.
        row = case_row(self.base, status="INCORRECT", extracted=CONTAINED, provenance=GOLD_CHILE)
        row["reward_evidence"]["provenance"] = row["reward_evidence"]["provenance"] | {
            "label": {"value": "Peru", "sha256": text_sha256("Chile")}}
        with self.assertRaisesRegex(ValueError, "source-bound gold label"):
            single.validate_single_reward(row)


class ReasonedWrongRows(TinyThreads):
    def test_zero_advantage_reasoned_wrong_row_adds_no_pg_but_keeps_kl_and_anchor(self):
        bundle = perturbed_factory()
        rows, contract = block(bundle, count=1, reward=0)
        contract = as_v12(rows, contract)
        decision = rows[0]["reward_evidence"]["decision"]
        self.assertEqual((rows[0]["reward"], rows[0]["advantage"], decision["penalty"]), (-1, 0.0, "REASONED_WRONG"))
        bundle.model.zero_grad(set_to_none=True)
        stats = train.backward_single_block(bundle, rows, contract, components=("PG",))
        self.assertEqual((stats["rl"]["pg"], stats["rl"]["pg_rows"], stats["has_gradient"]), (0, 1, False))
        self.assertEqual((stats["rl"]["pg_rows_positive"], stats["rl"]["pg_rows_negative"]), (0, 0))
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))
        kept = train.backward_single_block(bundle, rows, contract, components=("PG", "KL"))
        self.assertEqual(kept["rl"]["pg"], 0)
        self.assertGreater(kept["rl"]["anchor"], 0)
        self.assertTrue(kept["has_gradient"])


class LeakRuleV2(unittest.TestCase):
    CHOICES = ("photosynthesis", "respiration", "fermentation")

    def test_naming_a_wrong_option_is_allowed_but_the_gold_a_letter_or_an_elimination_is_not(self):
        wrong_only = "Cells release energy through respiration in the mitochondria."
        self.assertIsNone(hint_leak_single_v2(wrong_only, "A", self.CHOICES))
        self.assertEqual(hint_leak_single(wrong_only, "A", self.CHOICES), "OPTION_TEXT:B")  # v1 still refuses it
        self.assertTrue(hint_leak_single_v2("Plants are photosynthesizing in the leaf.", "A", self.CHOICES)
                        .startswith("STEM:"))
        self.assertEqual(hint_leak_single_v2("Compare respiration with fermentation carefully.", "A", self.CHOICES),
                         "ELIMINATION")
        for lesson, prefix in (("Pick option B.", "OPTION_LETTER:"), ("Think about (b) carefully.", "PAREN_LETTER:"),
                               ("Choose B.", "DIRECTED_OPTION:")):
            with self.subTest(lesson=lesson):
                self.assertTrue(hint_leak_single_v2(lesson, "A", self.CHOICES).startswith(prefix))

    def test_two_option_item_naming_its_wrong_option_is_elimination_and_open_gold_stems_are_refused(self):
        self.assertEqual(hint_leak_single_v2("Check the red curve first.", "B", ("red", "blue")), "ELIMINATION")
        self.assertEqual(hint_leak_single_v2("A reddish tint is not the answer.", "B", ("reddish", "bluish")),
                         "ELIMINATION")  # named through its stem
        self.assertIsNone(hint_leak_single_v2("Check the legend first.", "B", ("red", "blue")))
        self.assertIsNone(hint_leak_single_v2("Only one option: compare.", "A", ("photosynthesis",)))
        self.assertTrue(hint_leak_single_v2("The chilean coast is very long.", "Chile", ()).startswith("STEM:"))
        self.assertIsNone(hint_leak_single_v2("Read the map legend carefully.", "Chile", ()))
        with self.assertRaisesRegex(ValueError, "aliases"):
            hint_leak_single_v2("Read the map legend.", "Chile", (), aliases=(" ",))

    def test_b_answer_denial_v12_catches_reference_and_grader_blame_only_in_v12(self):
        for note, phrase in (("There is an error in the reference.", "error in the reference"),
                             ("the grader misread my answer", "grader misread"),
                             ("The given answer seems wrong.", "given answer seems wrong")):
            with self.subTest(note=note):
                self.assertEqual(b_target_problem(note, v12=True), "ANSWER_DENIAL:" + phrase)
                self.assertIsNone(b_target_problem(note, v12=False))
                self.assertIsNone(b_target_problem(note))
        self.assertEqual(b_target_problem("The figure was mislabeled.", v12=True), "ANSWER_DENIAL:mislabeled")
        self.assertEqual(b_target_problem("The figure was mislabeled."), "ANSWER_DENIAL:mislabeled")
        self.assertIsNone(b_target_problem("I confused the two labels.", v12=True))


class BRowsAfterFailedResolve(unittest.TestCase):
    V12_KEYS = dict(contracts=PINS, token_scale=1024., version=VERSION_SINGLE_V12)

    def case(self, *, v12, lesson=None, wrong_resolve=True):
        case = runtime_tests.RuntimeTests("test_program_gate_runs_without_pauses_keeps_originals_and_gold_only_in_note_request")
        case.setUp()
        self.addCleanup(case.doCleanups)
        if v12:
            case.protocol = plain_protocol_single("bf16", 219, **self.V12_KEYS)
            case.owner.processor = tiny_processor()
        original = case.stream

        def wrong(requests, stats, stop_after):
            for request, output in original(requests, stats, stop_after):
                if wrong_resolve and request["stage"] == "resolve":
                    output["text"] = output["generation"]["text"] = " ".join(f"step{j}" for j in range(38)) + " Final answer: A"
                yield request, output

        case.stream = wrong
        if lesson is not None:
            case.stream = case.leaking(lesson)
        return case

    def closed(self, case):
        result = case.run_block()
        self.assertEqual(result["status"], "ROWS_READY")
        return json.loads((case.out / result["rows_path"]).read_text())

    def test_v12_keeps_the_note_row_after_a_wrong_resolve_and_v11_drops_it(self):
        v11 = self.closed(self.case(v12=False))
        self.assertEqual([r["kind"] for r in v11["rows"]], ["RL", "A1", "RL"])
        self.assertFalse(v11["row_decisions"]["p1:A2"]["accepted"])
        self.assertNotIn("p1:B", v11["row_decisions"])
        v12 = self.closed(self.case(v12=True))
        self.assertEqual([r["kind"] for r in v12["rows"]], ["RL", "A1", "RL", "B"])
        self.assertFalse(v12["row_decisions"]["p1:A2"]["accepted"])
        self.assertTrue(v12["row_decisions"]["p1:B"]["accepted"])
        self.assertEqual(v12["row_decisions"]["p1:A2"]["reason"], "A2_NOT_ADMISSIBLE")
        note = v12["rows"][3]
        self.assertEqual((note["row_id"], note["route"], note["target_source"]), ("p1:B", "FULL", "NOTE"))
        for row in (r for r in v12["rows"] if r["kind"] == "RL"):
            evidence = row["reward_evidence"]
            self.assertEqual((evidence["version"], evidence["item"]), ("single-t1-program-gate-v7", {"option_count": 2}))
        wrong = next(r for r in v12["rows"] if r["kind"] == "RL" and r["problem_id"] == "p1")
        self.assertEqual((wrong["reward"], wrong["advantage"], wrong["reward_evidence"]["decision"]["penalty"]),
                         (-1, 0.0, "REASONED_WRONG"))
        self.assertEqual(v12["contract"]["version"], VERSION_SINGLE_V12)
        self.assertEqual(v12["contract"]["sft_weights"], SINGLE_SFT_WEIGHTS_V12)
        self.assertEqual(v12["contract"]["b_rows"], "every_program_accepted_note_whatever_the_resolve_result")

    def test_v12_with_a_correct_resolve_still_builds_a2_then_b(self):
        closed = self.closed(self.case(v12=True, wrong_resolve=False))
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "A1", "RL", "A2", "B"])
        self.assertTrue(closed["row_decisions"]["p1:A2"]["accepted"])

    def test_v12_refuses_a_note_that_names_the_gold_or_the_only_wrong_option_of_two(self):
        refused = self.closed(self.case(v12=True, lesson="Look at the blue curve first."))
        self.assertFalse(refused["note_decisions"]["p1"]["accepted"])
        self.assertIn("LEAK", refused["note_decisions"]["p1"]["reason"])
        self.assertEqual([r["kind"] for r in refused["rows"]], ["RL", "A1", "RL"])  # no note, no re-solve, no B
        eliminated = self.closed(self.case(v12=True, lesson="Compare the red curve with the legend before answering."))
        self.assertEqual(eliminated["note_decisions"]["p1"]["reason"], "LESSON_LEAK_ELIMINATION")
        self.assertEqual([r["kind"] for r in eliminated["rows"]], ["RL", "A1", "RL"])
        accepted = self.closed(self.case(v12=True, lesson="Compare each curve with the legend before answering."))
        self.assertTrue(accepted["note_decisions"]["p1"]["accepted"])
        self.assertEqual([r["kind"] for r in accepted["rows"]], ["RL", "A1", "RL", "B"])

    def test_v12_refuses_a_note_that_blames_the_reference_even_with_a_wrong_resolve(self):
        original = self.case(v12=True)
        stream = original.stream

        def blaming(requests, stats, stop_after):
            for request, output in stream(requests, stats, stop_after):
                if request["stage"] == "note":
                    output["text"] = output["generation"]["text"] = output["text"].replace(
                        "confused two labels.", "confused two labels; the given answer seems wrong.")
                yield request, output

        original.stream = blaming
        closed = self.closed(original)
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "A1", "RL"])
        self.assertEqual(closed["row_decisions"]["p1:B"], {"accepted": False, "method": "program",
                                                           "reason": "B_ANSWER_DENIAL:given answer seems wrong"})


class HumanRows(TinyThreads):
    TOKENIZER = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.TOKENIZER = tiny_processor().tokenizer

    def record(self):
        return {"problem_id": "p0", "image_paths": ["fixture.png"], "image_sha256": ["a" * 64], "subject": "Biology",
                "source": "authored-test", "p0_text": "Question: authored question 0\nOptions:\nA. red\nB. blue"}

    def test_human_solution_reads_the_mapped_annotation_fields_only(self):
        item = {"source": "hf-scienceqa", "annotations": [{"field": "lecture", "value": "L"},
                                                          {"field": "solution", "value": "S"},
                                                          {"field": "lecture", "value": "ignored second value"}]}
        self.assertEqual(references.human_solution(item), {"lecture": "L", "solution": "S"})
        self.assertEqual(references.human_solution({"source": "hf-scienceqa", "annotations": [
            {"field": "solution", "value": ["x", "y"]}]}), {"lecture": "", "solution": "x, y"})
        self.assertEqual(references.human_solution({"source": "hf-scienceqa"}), {"lecture": "", "solution": ""})
        for other in ({"source": "hf-clevr", "annotations": item["annotations"]}, PROVENANCE, {}):
            self.assertIsNone(references.human_solution(other))
        self.assertEqual(references.HUMAN_SOLUTION_FIELDS, {"hf-scienceqa": ("lecture", "solution")})

    def test_human_row_builds_lecture_solution_and_final_answer_under_the_p0_prompt(self):
        lecture, solution = human_fields()
        record = self.record()
        row, reason = single.human_row(record, {"lecture": lecture, "solution": solution}, "B", ("red", "blue"), self.TOKENIZER)
        self.assertIsNone(reason)
        self.assertEqual(row["target"], lecture + "\n\n" + solution + "\n\nFinal Answer: B. blue")
        self.assertEqual((row["kind"], row["route"], row["target_source"], row["row_id"]),
                         (KIND_HUMAN, "LM", HUMAN_TARGET_SOURCE, "p0:H"))
        self.assertEqual((row["prompt"], row["complete"]), (record["p0_text"], True))
        self.assertNotIn("generation", row)
        self.assertEqual({k: row[k] for k in ("problem_id", "image_paths", "image_sha256", "subject", "source")},
                         {k: record[k] for k in ("problem_id", "image_paths", "image_sha256", "subject", "source")})
        # A blank lecture leaves no leading block; the 32 token minimum counts the final line (16 + 4 words).
        row, reason = single.human_row(record, {"lecture": " ", "solution": solution + " " + solution}, "A", ("red", "blue"),
                                       self.TOKENIZER)
        self.assertIsNone(reason)
        self.assertEqual(row["target"], solution + " " + solution + "\n\nFinal Answer: A. red")
        # An open item's answer is the gold text.
        row, _ = single.human_row(record, {"lecture": lecture, "solution": solution}, "Chile", (), self.TOKENIZER)
        self.assertTrue(row["target"].endswith("\n\nFinal Answer: Chile"))

    def test_human_row_refusals(self):
        lecture, solution = human_fields()
        record, choices = self.record(), ("red", "blue")
        cases = (({"lecture": "a b", "solution": "a b"}, "B", "H_TOO_SHORT"),
                 ({"lecture": lecture, "solution": "  "}, "B", "H_SOLUTION_BLANK"),
                 ({"lecture": lecture, "solution": solution + " <|im_end|>"}, "B", "H_SPECIAL_TOKEN"),
                 ({"lecture": lecture, "solution": solution}, "Z", "H_GOLD_NOT_AN_OPTION"),
                 ({"lecture": lecture, "solution": solution}, "C", "H_GOLD_NOT_AN_OPTION"))
        for fields, gold, reason in cases:
            with self.subTest(reason=reason, gold=gold):
                self.assertEqual(single.human_row(record, fields, gold, choices, self.TOKENIZER), (None, reason))
        # 31 words in total (16 + 11 + the four-word final line) is one short of the minimum; one more word admits it.
        short = {"lecture": lecture, "solution": " ".join(["a"] * 11)}
        self.assertEqual(single.human_row(record, short, "B", choices, self.TOKENIZER), (None, "H_TOO_SHORT"))
        enough = {"lecture": lecture, "solution": " ".join(["a"] * 12)}
        self.assertIsNone(single.human_row(record, enough, "B", choices, self.TOKENIZER)[1])

    def run_block(self, case, provenance):
        def grade(pid, text, finish, mode):
            return ("CORRECT" if text.endswith("B") else "INCORRECT", text[-1])

        return single.run_single_block(case.owner, actors=case.actors, items=case.items, provenance=provenance,
                                       gold=lambda pid: "B", grade=grade,
                                       load_images=lambda actor: [Image.new("RGB", (8, 8))], protocol=case.protocol,
                                       output=case.out, limit=2, stream=case.stream)

    def test_runtime_v12_appends_an_h_row_for_a_scienceqa_item_and_v11_does_not(self):
        lecture, solution = human_fields()
        provenance = {pid: PROVENANCE for pid in (f"p{i}" for i in range(10))} | {"p0": scienceqa(lecture, solution)}
        v11 = runtime_tests.RuntimeTests("test_program_gate_runs_without_pauses_keeps_originals_and_gold_only_in_note_request")
        v11.setUp()
        self.addCleanup(v11.doCleanups)
        result = self.run_block(v11, provenance)
        closed = json.loads((v11.out / result["rows_path"]).read_text())
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "A1", "RL", "A2", "B"])
        self.assertNotIn("p0:H", closed["row_decisions"])

        v12 = runtime_tests.RuntimeTests("test_program_gate_runs_without_pauses_keeps_originals_and_gold_only_in_note_request")
        v12.setUp()
        self.addCleanup(v12.doCleanups)
        v12.protocol = plain_protocol_single("bf16", 219, contracts=PINS, token_scale=1024., version=VERSION_SINGLE_V12)
        v12.owner.processor = tiny_processor()
        result = self.run_block(v12, provenance)
        closed = json.loads((v12.out / result["rows_path"]).read_text())
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "A1", "RL", "A2", "B", "H"])
        self.assertEqual(closed["row_decisions"]["p0:H"], {"accepted": True, "method": "program", "reason": None})
        self.assertNotIn("p1:H", closed["row_decisions"])  # an unmapped source has no human solution
        h = closed["rows"][-1]
        self.assertEqual((h["row_id"], h["route"], h["target_source"]), ("p0:H", "LM", "HUMAN_SOLUTION"))
        self.assertEqual(h["prompt"], closed["rows"][0]["prompt"])  # the unchanged hint-free P0 text
        self.assertTrue(h["prompt"].endswith(single._MCQ_P0_END))
        self.assertEqual(h["target"], lecture + "\n\n" + solution + "\n\nFinal Answer: B. blue")
        self.assertEqual(closed["contract"]["human_solution"]["fields_sha256"], references.human_solution_sha256())
        # H rows read no gold before the note stage ended and add no generation request.
        self.assertEqual([(r["problem_id"], r["stage"]) for r in v12.seen],
                         [("p0", "t1"), ("p1", "t1"), ("p1", "note"), ("p1", "resolve")])

    def test_runtime_v12_skips_h_when_the_solution_is_short_or_blank(self):
        for fields, reason in (human_fields(2, 2), "H_TOO_SHORT"), (("a b", ""), "H_SOLUTION_BLANK"):
            with self.subTest(reason=reason):
                case = runtime_tests.RuntimeTests("test_program_gate_runs_without_pauses_keeps_originals_and_gold_only_in_note_request")
                case.setUp()
                self.addCleanup(case.doCleanups)
                case.protocol = plain_protocol_single("bf16", 219, contracts=PINS, token_scale=1024., version=VERSION_SINGLE_V12)
                case.owner.processor = tiny_processor()
                provenance = {pid: PROVENANCE for pid in case.actors} | {"p0": scienceqa(*fields)}
                result = self.run_block(case, provenance)
                closed = json.loads((case.out / result["rows_path"]).read_text())
                self.assertEqual(closed["row_decisions"]["p0:H"], {"accepted": False, "method": "program", "reason": reason})
                self.assertNotIn("H", [r["kind"] for r in closed["rows"]])

    def h_row(self, words=40, **changes):
        row = {"row_id": "item-0:H", "kind": KIND_HUMAN, "route": "LM", "problem_id": "item-0", "complete": True,
               "target_source": HUMAN_TARGET_SOURCE, "prompt": "question a a", "target": " ".join(["a", "b"] * (words // 2)),
               "image_paths": ["x.png"], "image_sha256": ["a" * 64], "subject": "s", "source": "authored-test"}
        return row | changes

    def encode(self, row):
        return train.encode_single_row(tiny_processor(), SimpleNamespace(image_token_id=3), row, [Image.new("RGB", (16, 16))])

    def test_encoding_an_h_row_appends_the_end_token_and_refuses_unsafe_rows(self):
        encoded = self.encode(self.h_row())
        self.assertEqual(encoded["target_len"], 41)
        self.assertEqual(encoded["input_ids"][-41:].tolist(), [8, 9] * 20 + [7])  # 40 target words, then <|im_end|>
        self.assertEqual(int((encoded["input_ids"][-41:] == 3).sum()), 0)
        self.assertNotIn("generation", encoded)
        for label, bad in (("short", self.h_row(words=30)), ("generation", self.h_row(generation={"completion_token_ids": [8]})),
                           ("full route", self.h_row(route="FULL")), ("wrong source", self.h_row(target_source="NOTE")),
                           ("blank", self.h_row(target=" ")), ("special in target", self.h_row(target="a <|im_end|> " + "b " * 40)),
                           ("special in prompt", self.h_row(prompt="question <|image_pad|>"))):
            with self.subTest(label), self.assertRaises(ValueError):
                self.encode(bad)

    def test_h_backward_is_a_weighted_row_mean_and_never_touches_pg_or_kl(self):
        bundle = perturbed_factory()
        rows, contract = block(bundle, count=1, reward=1)
        contract = as_v12(rows, contract)
        h = self.encode(self.h_row())
        stats = train.backward_single_block(bundle, rows + [h], contract, components=("H",))
        self.assertEqual(stats["sft"]["H"]["weight"], .02)
        self.assertEqual((stats["sft"]["H"]["rows"], stats["sft"]["H"]["weighting"]), (1, "per_kind_total"))
        self.assertEqual(set(stats["sft"]), {"H"})
        self.assertEqual((stats["rl"]["pg"], stats["rl"]["kl"], stats["rl"]["anchor"]), (0., 0., 0.))
        self.assertTrue(stats["has_gradient"])
        actual = gradients(bundle)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in actual.values()))
        self.assertGreater(sum(float(grad.abs().sum()) for grad in actual.values()), 0)
        bundle.model.zero_grad(set_to_none=True)
        bundle.model.train()
        bundle.route.apply(False)
        loss = .02 * (-trainer.batch_logprobs(bundle, [h])[0].mean())
        loss.backward()
        expected = gradients(bundle)
        self.assertAlmostEqual(stats["sft"]["H"]["weighted_loss"], float(loss.detach()), places=6)
        for name, grad in expected.items():
            torch.testing.assert_close(actual[name], grad, atol=1e-8, rtol=1e-5)

    def test_h_rows_are_excluded_from_pg_kl_and_anchor(self):
        bundle = perturbed_factory()
        rows, contract = block(bundle, count=1, reward=1)
        contract = as_v12(rows, contract)
        h = self.encode(self.h_row())
        bundle.model.zero_grad(set_to_none=True)
        alone = train.backward_single_block(bundle, rows, contract, components=("PG", "KL"))
        reference = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        both = train.backward_single_block(bundle, rows + [h], contract, components=("PG", "KL"))
        self.assertEqual(both["sft"], {})
        self.assertEqual(both["rl"], alone["rl"])
        self.assertEqual((both["rl"]["pg_rows_positive"], both["rl"]["pg_rows_negative"]), (1, 0))
        self.assertGreater(both["rl"]["anchor"], 0)
        for name, grad in gradients(bundle).items():
            torch.testing.assert_close(grad, reference[name], atol=1e-8, rtol=1e-5)

    def test_an_h_row_under_the_v11_contract_is_refused(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=1, reward=1)
        contract = as_v11(rows, contract)
        with self.assertRaisesRegex(ValueError, "invalid kind"):
            train.backward_single_block(bundle, rows + [self.encode(self.h_row())], contract, components=("H",))
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))

    def test_h_rows_need_unique_ids_and_the_lm_route(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=1, reward=1)
        contract = as_v12(rows, contract)
        h = self.encode(self.h_row())
        for label, rows_ in (("duplicate", rows + [h, copy.deepcopy(h)]),
                             ("generation", rows + [h | {"generation": {"completion_token_ids": [8]}}]),
                             ("full route", rows + [h | {"route": "FULL"}]),
                             ("wrong source", rows + [h | {"target_source": "NOTE"}]),
                             ("unknown problem", rows + [h | {"problem_id": "item-9", "row_id": "item-9:H"}])):
            with self.subTest(label), self.assertRaises(ValueError):
                train.backward_single_block(bundle, rows_, contract, components=("H",))


class ProtocolV12(unittest.TestCase):
    def test_v12_protocol_fields(self):
        protocol = plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V12)
        contract = protocol["training_contract"]
        self.assertEqual((protocol["version"], contract["version"]),
                         ("error-note-single-response-v12", "error-note-single-response-v12"))
        rl = contract["rl"]
        self.assertEqual(rl["reward"], "single-t1-program-gate-v7")
        self.assertEqual(rl["advantage"], {"version": "penalty-table-v2", "values": SINGLE_ADVANTAGE_V12})
        self.assertEqual(rl["reward_values"]["CONTAINED_OPEN"], 0)
        self.assertEqual((rl["baseline"], rl["token_scale"], rl["row_token_cap"], rl["beta"]), ("none", 1024., 2048, .02))
        self.assertEqual(contract["sft_weights"], {"A1": .05, "A2": .35, "B": .05, "H": .02})
        self.assertEqual(contract["sft_weighting"], {"A1": "per_kind_total", "A2": "per_row", "B": "per_kind_total",
                                                     "H": "per_kind_total"})
        self.assertEqual(contract["human_solution"]["fields"], {"hf-scienceqa": ["lecture", "solution"]})
        self.assertEqual(contract["human_solution"]["fields_sha256"], references.human_solution_sha256())
        self.assertEqual(contract["routes"]["H"], "LM")
        self.assertEqual(contract["generation_counts"], {"t1": 1, "note": 1, "resolve": 1})
        single._protocol_check(protocol)
        single._protocol_check(plain_protocol_single("bf16", 2, contracts=PINS))

    def test_default_protocol_stays_v11_and_the_versions_differ(self):
        default = plain_protocol_single("bf16", 2, contracts=PINS)
        v11 = plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE)
        v12 = plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V12)
        self.assertEqual((default["version"], default["training_contract"]["version"]),
                         ("error-note-single-response-v11", "error-note-single-response-v11"))
        self.assertEqual(default, v11)
        self.assertNotIn("human_solution", default["training_contract"])
        self.assertEqual(default["training_contract"]["rl"]["reward"], SINGLE_REWARD_GATE)
        self.assertNotEqual(default["sha256"], v12["sha256"])
        self.assertEqual(default["training_contract"]["sft_weights"], {"A1": .05, "A2": .35, "B": .05})

    def test_bad_version_or_weights_are_refused(self):
        with self.assertRaisesRegex(ValueError, "version must be v11, v12, v13, v14 or v15"):
            plain_protocol_single("bf16", 2, contracts=PINS, version="error-note-single-response-v16")
        with self.assertRaisesRegex(ValueError, "nonnegative weights"):
            plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V12,
                                  sft_weights={"A1": .05, "A2": .35, "B": .05})
        with self.assertRaisesRegex(ValueError, "nonnegative weights"):
            plain_protocol_single("bf16", 2, contracts=PINS, sft_weights={"A1": .05, "A2": .35, "B": .05, "H": .1})
        tampered = plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V12)
        tampered["training_contract"]["rl"]["reward"] = SINGLE_REWARD_GATE  # gate v6 under a v12 protocol
        with self.assertRaisesRegex(ValueError, "invalid single-response protocol"):
            single._protocol_check(tampered)


class ContractGateBinding(TinyThreads):
    def test_a_v12_contract_with_gate_v6_rows_and_a_v11_contract_with_gate_v7_rows_are_refused(self):
        bundle = tiny_factory()
        rows, contract = block(bundle, count=1, reward=0)
        v12 = as_v12(rows, contract)
        v6_rows = copy.deepcopy(rows)
        for row in v6_rows:
            bind_reward_fixture(row, gate=SINGLE_REWARD_GATE)
        with self.assertRaisesRegex(ValueError, "binding|reward contract"):
            train.backward_single_block(bundle, v6_rows, v12 | {"reward_evidence_sha256": {
                r["problem_id"]: digest(r["reward_evidence"]) for r in v6_rows}}, components=("PG",))
        v11_rows, v11_base = block(bundle, count=1, reward=0)
        v11 = as_v11(v11_rows, v11_base)
        v7_rows = copy.deepcopy(v11_rows)
        for row in v7_rows:
            bind_v12(row)
        with self.assertRaisesRegex(ValueError, "binding|reward contract"):
            train.backward_single_block(bundle, v7_rows, v11 | {"reward_evidence_sha256": {
                r["problem_id"]: digest(r["reward_evidence"]) for r in v7_rows}}, components=("PG",))
        # The v12 contract needs the v12 table, weights and human-solution pin, not just the gate name.
        for label, bad in (("table", v12 | {"rl": v12["rl"] | {"advantage": {"version": "penalty-table-v2", "values": {
                               **SINGLE_ADVANTAGE_V12, "REASONED_WRONG": -.5}}}}),
                           ("weights", v12 | {"sft_weights": {"A1": .05, "A2": .35, "B": .05, "H": .2}}),
                           ("pin", v12 | {"human_solution": {"fields_sha256": "0" * 64}})):
            with self.subTest(label), self.assertRaisesRegex(ValueError, "v12 penalty table"):
                train.backward_single_block(bundle, rows, bad, components=("PG",))
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))


class NoLessonControl(unittest.TestCase):
    """Diagnostic lesson-free re-solve (user 2026-11-11, local CLI only): after the sealed rows, never a row."""

    def case(self):
        case = runtime_tests.RuntimeTests("test_program_gate_runs_without_pauses_keeps_originals_and_gold_only_in_note_request")
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.protocol = plain_protocol_single("bf16", 219, contracts=PINS, token_scale=1024., version=VERSION_SINGLE_V12)
        case.owner.processor = tiny_processor()
        return case

    @staticmethod
    def grade(pid, text, finish, mode):
        return ("CORRECT" if text.endswith("B") else "INCORRECT", text[-1])

    def test_control_runs_after_the_sealed_rows_and_leaves_them_byte_identical(self):
        from fine_tuning.error_note.contract import RESOLVE_HEADER, RESOLVE_INSTRUCTION_SINGLE, sample_seed

        plain = self.case()
        rows_plain = (plain.out / plain.run_block()["rows_path"]).read_bytes()
        case = self.case()
        case.owner.model.load_state_dict(plain.owner.model.state_dict())  # the same policy, so the same binding
        result = case.run_block()
        rows_path = case.out / result["rows_path"]
        before, phase = rows_path.read_bytes(), (case.out / "phase.json").read_bytes()
        self.assertEqual(before, rows_plain)
        seen = len(case.seen)
        diagnostic = case.out.parent / (case.out.name + "-diagnostic")
        body = single.run_no_lesson_control(case.owner, block=case.out, actors=case.actors, grade=self.grade,
                                            load_images=lambda actor: [Image.new("RGB", (8, 8))], protocol=case.protocol,
                                            output=diagnostic, stream=case.stream)
        self.assertEqual((rows_path.read_bytes(), (case.out / "phase.json").read_bytes()), (before, phase))
        self.assertEqual(json.loads((diagnostic / single.NO_LESSON_CONTROL_FILE).read_text()), body)
        self.assertEqual((body["status"], body["diagnostic_only"], body["training_rows"]), ("COMPLETE", True, 0))
        self.assertEqual(body["items"], [{"problem_id": "p1", "source": "authored-test", "t1_status": "INCORRECT",
                                          "lesson_resolve_status": "CORRECT", "control_status": "CORRECT",
                                          "tokens": 41, "control_finish_reason": "stop"}])
        control = case.seen[seen:]
        self.assertEqual([(r["problem_id"], r["stage"]) for r in control], [("p1", "control")])
        self.assertEqual(control[0]["seed"], sample_seed("p1", "control"))
        self.assertNotIn(control[0]["seed"], {r["seed"] for r in case.seen[:seen] if r["problem_id"] == "p1"})
        text = control[0]["messages"][0]["content"][-1]["text"]
        self.assertTrue(text.startswith(RESOLVE_INSTRUCTION_SINGLE + "\n\n"))
        self.assertNotIn(RESOLVE_HEADER, text)
        self.assertNotIn("Read each label", text)  # the lesson
        self.assertFalse((case.out / "no-lesson-control").exists())  # its own ledger, outside the block directory
        self.assertTrue((diagnostic / "no-lesson-control").is_dir())

    def test_control_refuses_an_unsealed_block_and_only_the_local_cli_has_the_flag(self):
        from fine_tuning.error_note import single_campaign as campaign

        case = self.case()
        case.run_block()
        state = single._state(case.out / "phase.json")
        single._state(case.out / "phase.json", state | {"status": "COLLECTING_T1"})
        with self.assertRaisesRegex(ValueError, "sealed rows"):
            single.run_no_lesson_control(case.owner, block=case.out, actors=case.actors, grade=self.grade,
                                         load_images=lambda actor: [], protocol=case.protocol,
                                         output=case.out.parent / "x", stream=case.stream)
        self.assertIn("no_lesson_control", {action.dest for action in single.build_parser()._actions})
        self.assertNotIn("no_lesson_control", {action.dest for action in campaign.build_parser()._actions})


class LocalCliV12(unittest.TestCase):
    def test_parser_defaults_to_v12_and_accepts_limit_16(self):
        parser = single.build_parser()
        required = ["prepare", "--order-sha256", "a" * 64] + [
            item for name in ("build-root", "order", "frozen-approval", "source-provenance", "training-rules",
                              "model-path", "output") for item in ("--" + name, "/x")]
        args = parser.parse_args(required)
        self.assertEqual((args.single_version, args.limit, args.no_lesson_control), ("v12", 2, False))
        self.assertEqual(parser.parse_args(required + ["--limit", "16", "--single-version", "v11"]).limit, 16)
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            parser.parse_args(required + ["--limit", "12"])

    def test_prepare_seals_the_version_fixed16_and_the_note_reference_pin(self):
        from fine_tuning.error_note.answer_contract import VERSION as RULES_VERSION
        from fine_tuning.error_note.source_evidence import VERSION as SOURCE_VERSION
        from fine_tuning.notebook_rl.schema import ActorItem, to_json
        import tempfile
        from pathlib import Path

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
        supplement = out / "supplement.json"
        supplement.write_text(json.dumps({"version": references.SUPPLEMENT_VERSION, "items": {"p3": {"annotations": []}}}))
        (out / "model").mkdir()
        (out / "model" / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "torch_dtype": "bfloat16"}))

        def args(name, **extra):
            return SimpleNamespace(build_root=root, source_provenance=out / "source.json", training_rules=out / "rules.json",
                                   order_sha256="a" * 64, model_path=out / "model", output=out / name, mode="local-smoke",
                                   token_scale=1024., max_num_seqs=1, max_num_batched_tokens=24576, token_budget=0,
                                   stop_after_seconds=None, **extra)

        with patch.object(single, "_validate_data", return_value=({"items": items}, {"authored_fixture": True})), \
                patch("fine_tuning.notebook_rl.vault.read_actors", return_value=actors), \
                patch.object(single, "source_manifest", return_value={"base_commit": "a" * 40, "actual_files": {}}), \
                patch.object(single, "generation_identity", return_value={"model_files_sha256": "b" * 64}):
            v12 = single.prepare(args("v12", limit=16))
            self.assertEqual((v12["protocol"]["version"], len(v12["fixed10"]), len(v12["fixed16"])),
                             (VERSION_SINGLE_V12, 10, 16))
            self.assertEqual(v12["fixed16"], items[:16])
            v11 = single.prepare(args("v11", limit=10, single_version="v11"))
            self.assertEqual(v11["protocol"]["version"], VERSION_SINGLE)
            self.assertNotIn("fixed16", v11)
            self.assertEqual(v11["protocol"]["training_contract"]["note_reference"], {"version": "none"})
            with self.assertRaisesRegex(ValueError, "binding changed"):  # the sealed plan does not move to 16 items
                single.prepare(args("v11", limit=16, single_version="v11"))
            sha = single.file_sha(supplement)
            pinned = single.prepare(args("ref", note_reference_supplement=supplement, note_reference_supplement_sha256=sha))
            self.assertEqual(pinned["protocol"]["training_contract"]["note_reference"],
                             references.note_reference_pin(sources, {"p3": {"annotations": []}}, sha,
                                                           [item["problem_id"] for item in items]))
            with self.assertRaisesRegex(ValueError, "together"):
                single.prepare(args("bad", note_reference_supplement=supplement))


if __name__ == "__main__":
    unittest.main()
