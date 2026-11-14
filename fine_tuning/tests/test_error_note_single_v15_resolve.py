"""SINGLE v15 wave 3 CPU checks: up to three lesson re-solves per wrong item; the first attempt that passes the v14 A2
checks is the only A2 row. Fake engine streams and a tiny tokenizer only; these are not pretrained Qwen evidence."""
import hashlib
import json
import unittest
from unittest.mock import patch

from PIL import Image

from fine_tuning.error_note import single
from fine_tuning.error_note.collect_v4 import SINGLE_LEDGER
from fine_tuning.error_note.contract import sample_seed
from fine_tuning.error_note.protocol import (SINGLE_SPIRAL_V13, SINGLE_TAIL_PRAISE_V14, VERSION_SINGLE_V14,
                                             VERSION_SINGLE_V15, plain_protocol_single)
from fine_tuning.tests import test_error_note_single_runtime as runtime_tests
from fine_tuning.tests import test_error_note_single_v14 as v14
from fine_tuning.tests.test_error_note_train import tiny_processor
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE

STEPS = v14.RuntimeV14.steps
WRONG = (STEPS(38, final="A"), list(range(100, 140)) + [2])  # 41 tokens, graded INCORRECT
SHORT = (STEPS(38), list(range(100, 140)) + [2])  # 41 tokens, CORRECT, under the 256-token A2 minimum
LONG = (STEPS(299), list(range(100, 399)) + [2])  # 300 tokens, CORRECT
ECHO = ("As per the lesson, read each label. " + STEPS(300), list(range(100, 399)) + [2])  # lesson echo


class SeedTests(unittest.TestCase):
    def test_resolve_seed_is_unchanged_and_attempt_seeds_are_distinct(self):
        self.assertEqual(sample_seed("p1", "resolve"), 2010512595)  # pinned before v15
        for stage, key in (("resolve2", "3407:p1:resolve:2"), ("resolve3", "3407:p1:resolve:3")):
            self.assertEqual(sample_seed("p1", stage),
                             int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**31))
        stages = ("t1", "note", "control", "resolve", "resolve2", "resolve3")
        seeds = [sample_seed(f"item-{i}", stage) for i in range(50) for stage in stages]
        self.assertEqual(len(set(seeds)), len(seeds))
        for stage in ("resolve4", "resolve1", "Resolve2"):
            with self.assertRaisesRegex(ValueError, "unknown seed stage"):
                sample_seed("p1", stage)


class ResolveChain(unittest.TestCase):
    NAME = v14.RuntimeV14.NAME

    def case(self, version=VERSION_SINGLE_V15, outputs=None):
        """The v14 runtime fixture (p1's T1 is wrong; every other output is a 41-token CORRECT text) under a v14 or
        v15 protocol; ``outputs`` rewrites p1's re-solve stages."""
        for rule, extra in ((SINGLE_SPIRAL_V13, {"marker": "a"}), (SINGLE_TAIL_PRAISE_V14, {})):
            patcher = patch.dict(rule, marker_token_ids=[8], **extra)
            patcher.start()
            self.addCleanup(patcher.stop)
        case = runtime_tests.RuntimeTests(self.NAME)
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.protocol = plain_protocol_single("bf16", 219, contracts=v14.PINS, token_scale=1024., version=version,
                                              schedule_steps=64)
        case.owner.processor = tiny_processor()
        for stage, (text, actions) in (outputs or {}).items():
            v14.RuntimeV14.rewrite(case, stage, "p1", text, actions)
        return case

    @staticmethod
    def block(case, continuous=False, events=None):
        stream = case.stream
        if continuous:  # counts engine calls: a working follow() chain leaves the staged pass nothing to generate
            inner = case.continuous([] if events is None else events)

            def stream(*args):
                case.calls += 1
                return inner(*args)
            stream.continuous, case.calls = True, 0
        return single.run_single_block(case.owner, actors=case.actors, items=case.items,
                                       provenance={pid: PROVENANCE for pid in case.actors}, gold=lambda pid: "B",
                                       grade=lambda pid, text, finish, mode: ("CORRECT" if text.endswith("B")
                                                                              else "INCORRECT", text[-1]),
                                       load_images=lambda actor: [Image.new("RGB", (8, 8))], protocol=case.protocol,
                                       output=case.out, limit=2, stream=stream)

    def closed(self, case, continuous=False):
        result = self.block(case, continuous)
        self.assertEqual(result["status"], "ROWS_READY")
        return json.loads((case.out / result["rows_path"]).read_text())

    def test_attempt_chain_on_staged_and_continuous_engines(self):
        table = (  # p1 re-solve outputs -> row kinds, A2 decision, re-solve stages generated
            ("first passes", {"resolve": LONG}, ["RL", "RL", "A2", "B"], (1, 1, None), ["resolve"]),
            ("wrong then long", {"resolve": WRONG, "resolve2": LONG}, ["RL", "RL", "A2", "B"], (2, 2, None),
             ["resolve", "resolve2"]),
            ("short then long", {"resolve2": LONG}, ["RL", "RL", "A2", "B"], (2, 2, None), ["resolve", "resolve2"]),
            ("all fail", {"resolve": WRONG, "resolve3": ECHO}, ["RL", "RL"], (None, 3, "A2_RESOLVE_LEAK:lesson"),
             ["resolve", "resolve2", "resolve3"]))
        for name, outputs, kinds, (attempt, attempts, reason), stages in table:
            staged, case = None, self.case(outputs=outputs)
            for continuous in (False, True):
                with self.subTest(name, continuous=continuous):
                    if continuous:  # the same policy and fake engine, a fresh block directory
                        case.out, case.seen = case.out / "pipelined", []
                    closed = self.closed(case, continuous)
                    self.assertEqual([r["kind"] for r in closed["rows"]], kinds)
                    if continuous:
                        self.assertEqual(case.calls, 1)
                    self.assertEqual(closed["row_decisions"]["p1:A2"],
                                     {"accepted": attempt is not None, "method": "program", "reason": reason,
                                      "attempt": attempt, "attempts": attempts})
                    self.assertEqual("p1:B" in closed["row_decisions"], attempt is not None)
                    resolves = [r for r in case.seen if r["stage"].startswith("resolve")]
                    self.assertEqual([r["stage"] for r in resolves], stages)
                    self.assertTrue(all(r["problem_id"] == "p1" for r in resolves))
                    self.assertEqual(len({json.dumps(r["messages"]) for r in resolves}), 1)  # one prompt, every attempt
                    self.assertEqual([r["seed"] for r in resolves], [sample_seed("p1", s) for s in stages])
                    records = json.loads((case.out / "phase.json").read_text())["records"]["p1"]
                    self.assertEqual(len({records[s]["lesson_sha256"] for s in stages}), 1)
                    self.assertNotIn({"resolve": "resolve2", "resolve2": "resolve3"}.get(stages[-1]), records)
                    if attempt is not None:
                        a2, b = closed["rows"][2:]
                        self.assertEqual((a2["target_source"], a2["generation"]["stage"], a2["route"]),
                                         ("RESOLVE", stages[attempt - 1], "FULL"))
                        self.assertEqual(a2["target"], outputs[stages[attempt - 1]][0])
                        self.assertEqual((b["row_id"], b["route"]), ("p1:B", "FULL"))
                        self.assertTrue(closed["row_decisions"]["p1:B"]["accepted"])
                    if staged is None:
                        staged = closed["rows"]
                    else:
                        self.assertEqual(closed["rows"], staged)

    def test_v14_generates_one_resolve_whose_request_equals_v15_attempt_one(self):
        cases = {}
        for version in (VERSION_SINGLE_V14, VERSION_SINGLE_V15):
            cases[version] = case = self.case(version)
            closed = self.closed(case)
            if version == VERSION_SINGLE_V14:
                self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "RL", "B"])
                self.assertEqual(closed["row_decisions"]["p1:A2"],
                                 {"accepted": False, "method": "program", "reason": "A2_UNDER_MIN_TOKENS"})
        old, new = cases[VERSION_SINGLE_V14].seen, cases[VERSION_SINGLE_V15].seen
        self.assertEqual([(r["problem_id"], r["stage"]) for r in old],
                         [("p0", "t1"), ("p1", "t1"), ("p1", "note"), ("p1", "resolve")])
        self.assertEqual(old[3]["seed"], 2010512595)
        self.assertEqual(new[:4], old)
        self.assertEqual([r["stage"] for r in new[4:]], ["resolve2", "resolve3"])
        records = json.loads((cases[VERSION_SINGLE_V14].out / "phase.json").read_text())["records"]["p1"]
        self.assertFalse({"resolve2", "resolve3"} & set(records))

    def test_a_crash_after_resolve2_started_is_never_resampled_and_a_completed_block_reuses_its_ledger(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                case = self.case(outputs={"resolve": WRONG, "resolve2": LONG})
                original = case.stream

                def crashing(requests, stats, stop_after, original=original):
                    for request, output in original(requests, stats, stop_after):
                        if request["stage"] == "resolve2":
                            raise RuntimeError("authored engine crash")
                        yield request, output

                case.stream = crashing
                with self.assertRaisesRegex(RuntimeError, "authored engine crash"):
                    self.block(case, continuous)
                seen = len(case.seen)
                ledger = json.loads((case.out / SINGLE_LEDGER).read_text())
                self.assertEqual(ledger["requests"]["p1:resolve2"]["state"], "STARTED")
                case.stream = original
                with self.assertRaisesRegex(ValueError, "stale|resampling is forbidden"):
                    self.block(case, continuous)
                self.assertEqual(len(case.seen), seen)
                self.assertNotEqual(json.loads((case.out / "phase.json").read_text())["status"], "ROWS_READY")

                done = self.case(outputs={"resolve": WRONG, "resolve2": LONG})
                rows = self.closed(done, continuous)["rows"]
                seen = len(done.seen)
                self.assertEqual(self.closed(done, continuous)["rows"], rows)
                self.assertEqual(len(done.seen), seen)  # every attempt, follow-ups included, comes from the ledger


if __name__ == "__main__":
    unittest.main()
