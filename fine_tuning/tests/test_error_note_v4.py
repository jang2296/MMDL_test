"""Authored cases for error-note version 4: seeds, groups, the loop detector, the program gate, self-check parsing,
row construction, the RL term, protocol v4 and the continuous stream's WAIT/observe hooks. No model or GPU."""
import unittest
from types import SimpleNamespace

import torch

from fine_tuning.error_note import rl, v4
from fine_tuning.error_note.collect import WAIT, StreamStats, continuous_stream
from fine_tuning.error_note.contract import NOTE_INSTRUCTION_V3, ROUTE_FULL, ROUTE_LM, sample_seed
from fine_tuning.error_note.protocol import plain_protocol_v4
from fine_tuning.notebook_rl import losses

CHOICES = ("Tomato mosaic virus", "Powdery mildew", "Healthy")
GOLD = "B"


def note(image="CORRECT", cause="KNOWLEDGE", lesson="Check every leaf surface for white patches before deciding.",
         shows="The leaf has 3 white spots near the top edge. The veins look normal."):
    return (f"What the image shows: {shows}\nImage reading: {image}\nCause: {cause}\n"
            f"What went wrong: I said healthy.\nLesson: {lesson}")


def tok(text):
    return list(range(len(text.split())))


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def sample(j, status, text=None, finish="stop"):
    return {"j": j, "seed": 100 + j, "text": words(40) if text is None else text, "finish_reason": finish, "status": status}


def record(rollouts, **extra):
    return {"problem_id": "p", "subject": "Biology", "source": "src", "p0_text": "Question: which?", "image_paths": ["p.png"],
            "rollouts": rollouts, **extra}


class SeedsAndGroups(unittest.TestCase):
    def test_seeds_are_deterministic_and_disjoint(self):
        seeds = {v4.v4_seed("p", r, purpose, j) for r in (1, 2) for purpose in v4.PURPOSES for j in range(4)}
        self.assertEqual(len(seeds), 2 * len(v4.PURPOSES) * 4)
        self.assertEqual(v4.v4_seed("p", 1, "roll", 0), v4.v4_seed("p", 1, "roll", 0))
        self.assertTrue(seeds.isdisjoint({sample_seed("p", s) for s in ("t1", "note", "resolve")}))
        for bad in (("", 1, "roll", 0), ("p", 0, "roll", 0), ("p", 1, "other", 0), ("p", 1, "roll", -1)):
            with self.assertRaises(ValueError):
                v4.v4_seed(*bad)

    def test_reward_classify_and_adaptive_rule(self):
        self.assertEqual([v4.reward(s) for s in ("CORRECT", "INCORRECT", "TRUNCATED", v4.LOOP_STOPPED)], [1, 0, 0, 0])
        self.assertIsNone(v4.reward("UNVERIFIABLE"))
        self.assertEqual(v4.classify_group(["CORRECT", "INCORRECT", "CORRECT", "TRUNCATED"])[0], v4.MIXED)
        self.assertEqual(v4.classify_group(["CORRECT", "CORRECT"])[0], v4.ALL_CORRECT)
        self.assertEqual(v4.classify_group(["INCORRECT"] * 4)[0], v4.ALL_WRONG)
        self.assertEqual(v4.classify_group(["CORRECT", "UNVERIFIABLE", "CONTEXT_BLOCKED"])[0], v4.UNUSABLE)
        self.assertEqual(v4.classify_group(["CORRECT", "UNVERIFIABLE", "INCORRECT"])[1], [(0, 1), (2, 0)])
        self.assertFalse(v4.needs_more(["CORRECT", "CORRECT"]))
        self.assertTrue(v4.needs_more(["CORRECT", "INCORRECT"]))
        with self.assertRaises(ValueError):
            v4.needs_more(["CORRECT"])
        self.assertEqual(v4.advantages([1, 0, 0, 1]), [.5, -.5, -.5, .5])
        with self.assertRaises(ValueError):
            v4.advantages([1])


class LoopDetector(unittest.TestCase):
    def test_non_overlapping_block_repeats(self):
        block = list(range(64))
        self.assertTrue(v4.has_block_repetition(block * 8))
        self.assertFalse(v4.has_block_repetition(block * 7))
        self.assertFalse(v4.has_block_repetition(list(range(64 * 9))))
        self.assertFalse(v4.has_block_repetition([5] * 64 * 7 + [6]))  # one token repeated: overlapping windows only
        self.assertTrue(v4.has_block_repetition([5] * 64 * 8))

    def test_incremental_matches_batch(self):
        ids = list(range(10)) + list(range(64)) * 8 + [1, 2, 3]
        detector = v4.BlockRepetition()
        found = [detector.feed(ids[i:i + 37]) for i in range(0, len(ids), 37)]
        self.assertTrue(found[-1])
        self.assertEqual(found.index(True), next(i for i in range(0, len(ids), 37)
                                                 if v4.has_block_repetition(ids[:i + 37])) // 37)


class GateAndSelfcheck(unittest.TestCase):
    def row(self, text, finish="stop"):
        return {"problem_id": "p", "text": text, "finish_reason": finish}

    def test_program_gate(self):
        passed = v4.program_gate(self.row(note()), GOLD, CHOICES)
        self.assertEqual(passed["result"], "PASS")
        self.assertTrue(passed["lesson"].startswith("Check every leaf"))
        cases = {"NO_NOTE": {"problem_id": "p", "status": "SKIPPED_X"},
                 "NOTE_NOT_STOPPED": self.row(note(), "length"), "NOTE_UNPARSED": self.row("nothing here"),
                 "MISREAD_WITH_CAUSE_NONE": self.row(note(image="MISREAD", cause="NONE")),
                 "HINT_LEAK:GOLD_OPTION_TEXT": self.row(note(lesson="Look for powdery mildew first."))}
        for reason, value in cases.items():
            gate = v4.program_gate(value, GOLD, CHOICES)
            self.assertEqual((gate["result"], gate["reason"], gate["lesson"]), ("FAIL", reason, None))
        self.assertEqual(v4.program_gate(self.row(note(lesson="Answer B when unsure.")), GOLD, CHOICES)["reason"],
                         "HINT_LEAK:OPTION_LETTER")
        self.assertEqual(v4.program_gate(self.row(note(lesson="Pick B when unsure.")), GOLD, CHOICES)["reason"],
                         "HINT_LEAK:GOLD_LETTER")
        lesson = "Compare against the TYLCV leaf pattern."
        self.assertEqual(v4.program_gate(self.row(note(lesson=lesson)), GOLD, CHOICES, acronyms=["TYLCV"])["reason"],
                         "ACRONYM_CANDIDATE")
        self.assertEqual(v4.program_gate(self.row(note(lesson="White fungus means one disease.")), GOLD, CHOICES,
                                         aliases=["white fungus"])["reason"], "ALIAS_LEAK")
        with self.assertRaises(ValueError):
            v4.program_gate(self.row(note()), " ", CHOICES)

    def test_claims_are_checkable_and_gold_blind(self):
        shows = ("The leaf is green overall. The leaf has 3 white spots near the top edge. "
                 "It cannot be read whether spores exist. Powdery mildew is visible on the leaf surface.")
        claims = v4.extract_claims(note(shows=shows), GOLD, CHOICES)
        self.assertEqual(claims, ["The leaf has 3 white spots near the top edge.", "The leaf is green overall."])
        self.assertEqual(v4.extract_claims("no headings", GOLD, CHOICES), [])
        self.assertIn('Statement: "A claim here."', v4.probe_prompt("A claim here."))
        with self.assertRaises(ValueError):
            v4.probe_prompt('say "quoted"')

    def test_probe_parsing_and_decision(self):
        self.assertEqual([v4.parse_probe(t) for t in ("TRUE. It is.", "false - no", "Cannot tell.", "CANNOT_TELL", "maybe")],
                         ["TRUE", "FALSE", "CANNOT_TELL", "CANNOT_TELL", "INVALID"])
        self.assertEqual(v4.decide_selfcheck([["TRUE", "TRUE"], ["TRUE", "TRUE"]]), v4.SELF_CONSISTENT)
        self.assertEqual(v4.decide_selfcheck([["TRUE", "FALSE"]]), v4.SELF_CONTRADICTED)
        self.assertEqual(v4.decide_selfcheck([["TRUE", "CANNOT_TELL"]]), v4.UNCHECKED)
        self.assertEqual(v4.decide_selfcheck([]), v4.UNCHECKED)
        self.assertEqual(v4.decide_selfcheck([["TRUE"], []]), v4.UNCHECKED)


class Rows(unittest.TestCase):
    def wrong(self, gate_text=None, check=v4.SELF_CONSISTENT, resolves=(), image="CORRECT"):
        text = gate_text or note(image=image)
        gate = v4.program_gate({"text": text, "finish_reason": "stop"}, GOLD, CHOICES)
        return record([sample(j, "INCORRECT") for j in range(4)],
                      note={"prompt": "note prompt with gold", "text": text, "finish_reason": "stop"}, gate=gate,
                      selfcheck={"result": check}, resolves=list(resolves))

    def test_mixed_group_gives_rl_rows_only(self):
        rows, drops = v4.build_rows_v4(record([sample(0, "CORRECT"), sample(1, "INCORRECT"),
                                               sample(2, "TRUNCATED", finish="length"), sample(3, "UNVERIFIABLE")]), tok)
        self.assertEqual([row["kind"] for row in rows], ["RL"] * 3)
        for value, expected in zip([row["advantage"] for row in rows], [2 / 3, -1 / 3, -1 / 3], strict=True):
            self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(sum(row["advantage"] for row in rows), 0.0)
        self.assertEqual([row["append_end"] for row in rows], [True, True, False])
        self.assertEqual({row["group_size"] for row in rows}, {3})
        self.assertEqual({row["route"] for row in rows}, {ROUTE_LM})
        self.assertEqual(drops, [])

    def test_all_correct_gives_one_retention_row(self):
        rows, _ = v4.build_rows_v4(record([sample(0, "CORRECT", text="short"), sample(1, "CORRECT")]), tok)
        self.assertEqual([(row["kind"], row["row_id"]) for row in rows], [("A1", "p:A1:1")])
        rows, drops = v4.build_rows_v4(record([sample(0, "CORRECT", text="x"), sample(1, "CORRECT", text="y")]), tok)
        self.assertEqual((rows, drops[0]["reason"]), ([], "NO_USABLE_CORRECT_ROLLOUT"))

    def test_all_wrong_note_path(self):
        good = sample(0, "CORRECT")
        rows, drops = v4.build_rows_v4(self.wrong(resolves=[sample(0, "INCORRECT"), good]), tok)
        self.assertEqual([row["kind"] for row in rows], ["A2", "B"])
        self.assertEqual(rows[0]["prompt"], "Question: which?")  # the lesson never enters a training prompt
        self.assertEqual(rows[0]["row_id"], "p:A2:0")
        self.assertEqual(rows[1]["target_source"], "NOTE")
        self.assertIn("RESOLVE_0:NOT_CORRECT", [d["reason"] for d in drops])
        misread = v4.build_rows_v4(self.wrong(image="MISREAD", resolves=[good]), tok)[0]
        self.assertEqual({row["route"] for row in misread}, {ROUTE_FULL})
        unchecked = v4.build_rows_v4(self.wrong(image="MISREAD", check=v4.UNCHECKED, resolves=[good]), tok)[0]
        self.assertEqual({row["route"] for row in unchecked}, {ROUTE_LM})

    def test_all_wrong_without_authorized_lesson_or_a2(self):
        self.assertEqual(v4.build_rows_v4(self.wrong(check=v4.SELF_CONTRADICTED, resolves=[sample(0, "CORRECT")]), tok)[0], [])
        rows, drops = v4.build_rows_v4(self.wrong(check=v4.UNCHECKED), tok)
        self.assertEqual(rows, [])
        self.assertIn("NOTE_UNCHECKED_WITHOUT_A2", [d["reason"] for d in drops])
        consistent = v4.build_rows_v4(self.wrong(), tok)[0]
        self.assertEqual([row["kind"] for row in consistent], ["B"])
        leak = v4.build_rows_v4(self.wrong(resolves=[sample(0, "CORRECT", text=words(40) + " as the lesson says")]), tok)
        self.assertIn("RESOLVE_0:LEAK:lesson", [d["reason"] for d in leak[1]])
        failed = self.wrong(gate_text=note(image="MISREAD", cause="NONE"))
        self.assertEqual(v4.build_rows_v4(failed, tok)[0], [])
        self.assertEqual(v4.build_rows_v4(self.wrong(resolves=[sample(0, "CORRECT")]), tok, arm="rl-only")[0], [])

    def test_summary_counts_lesson_and_control(self):
        wrong = self.wrong(resolves=[sample(0, "CORRECT"), sample(1, "INCORRECT")])
        wrong["controls"] = [sample(0, "INCORRECT"), sample(1, "INCORRECT")]
        records = [wrong, record([sample(0, "CORRECT"), sample(1, "INCORRECT")])]
        rows, drops = [], []
        for item in records:
            built, dropped = v4.build_rows_v4(item, tok)
            rows += built
            drops += dropped
        summary = v4.summarize_v4(records, rows, drops)
        self.assertEqual(summary["groups"], {v4.ALL_WRONG: 1, v4.MIXED: 1})
        self.assertEqual(summary["lesson_resolve"], {"samples": 2, "correct": 1, "rate": .5})
        self.assertEqual(summary["no_lesson_control"], {"samples": 2, "correct": 0, "rate": 0.0})
        self.assertNotIn("note prompt with gold", str(summary))


class RLTerm(unittest.TestCase):
    def test_gradient_is_advantage_times_score(self):
        current = [torch.tensor([-1.0, -2.0], requires_grad=True), torch.tensor([-0.5], requires_grad=True)]
        term = rl.group_term(current, None, [0.5, -0.5], group_tokens=3, eps_low=.2, eps_high=.28, beta=0.0)
        term.backward()
        self.assertTrue(torch.allclose(current[0].grad, torch.tensor([-0.5 / 3] * 2)))
        self.assertTrue(torch.allclose(current[1].grad, torch.tensor([0.5 / 3])))

    def test_micro_batches_sum_to_the_group(self):
        values = [torch.tensor([-1.0, -2.0, -0.3]), torch.tensor([-0.5]), torch.tensor([-0.7, -0.1])]
        reference = [v - 0.2 for v in values]
        whole = rl.group_term(values, reference, [.5, -.25, -.25], group_tokens=6, eps_low=.2, eps_high=.28, beta=.02)
        parts = (rl.group_term(values[:1], reference[:1], [.5], group_tokens=6, eps_low=.2, eps_high=.28, beta=.02)
                 + rl.group_term(values[1:], reference[1:], [-.25, -.25], group_tokens=6, eps_low=.2, eps_high=.28, beta=.02))
        self.assertTrue(torch.allclose(whole, parts))
        kl = sum(float(losses.k3(r, v).sum()) for r, v in zip(reference, values, strict=True)) * .02 / 6
        self.assertAlmostEqual(float(rl.group_term(values, reference, [0, 0, 0], group_tokens=6, eps_low=.2,
                                                   eps_high=.28, beta=.02)), kl, places=6)

    def test_invalid_inputs(self):
        value = [torch.tensor([-1.0])]
        for kwargs in ({"group_tokens": 0}, {"group_tokens": 1, "beta": .1}):
            with self.assertRaises(ValueError):
                rl.group_term(value, None, [1.0], **({"eps_low": .2, "eps_high": .28, "beta": 0.0} | kwargs))
        with self.assertRaises(ValueError):
            rl.group_term(value, None, [1.0, 2.0], group_tokens=1, eps_low=.2, eps_high=.28, beta=0.0)
        with self.assertRaises(ValueError):
            rl.pad([])


class ProtocolV4(unittest.TestCase):
    CONTRACTS = {"rows": "a" * 64, "order": "b" * 64, "init": None}

    def test_values_and_rules(self):
        protocol = plain_protocol_v4("bf16", 7, arm="hybrid", round_index=1, start=0, limit=500, contracts=self.CONTRACTS)
        self.assertEqual(protocol["version"], "error-note-protocol-v4")
        self.assertEqual(protocol["optimizer"]["lr"], v4.LR_V4)
        self.assertEqual(protocol["optimizer"]["total_steps"], 7)
        self.assertEqual(protocol["training_contract"]["sft_weights"], v4.SFT_WEIGHTS)
        control = plain_protocol_v4("bf16", 7, arm="rl-only", round_index=1, start=0, limit=500, contracts=self.CONTRACTS)
        self.assertEqual(control["training_contract"]["sft_weights"], {"A2": 0.0, "B": 0.0, "A1": v4.SFT_WEIGHTS["A1"]})
        self.assertNotEqual(protocol["sha256"], control["sha256"])
        self.assertEqual(protocol, plain_protocol_v4("bf16", 7, arm="hybrid", round_index=1, start=0, limit=500,
                                                     contracts=dict(self.CONTRACTS)))
        later = plain_protocol_v4("bf16", 7, arm="hybrid", round_index=2, start=500, limit=10,
                                  contracts=self.CONTRACTS | {"init": "c" * 64})
        self.assertEqual(later["training_contract"]["hashes"]["init"], "c" * 64)
        self.assertIn("prompts_sha256", protocol)
        self.assertTrue(NOTE_INSTRUCTION_V3)

    def test_refusals(self):
        bad = [dict(arm="solution"), dict(round_index=2), dict(round_index=0), dict(start=-1),
               dict(contracts={"rows": "a" * 64, "order": None, "init": None}),
               dict(contracts={"rows": "a" * 64, "order": "b" * 64})]
        for change in bad:
            kwargs = dict(arm="hybrid", round_index=1, start=0, limit=5, contracts=self.CONTRACTS) | change
            with self.assertRaises(ValueError, msg=str(change)):
                plain_protocol_v4("bf16", 3, **kwargs)


class FakeEngine:
    """add_request/step/abort_request: each request finishes after `steps[request_id]` steps (one token per step)."""

    def __init__(self, steps, loop_ids=()):
        self.steps, self.loop_ids, self.active, self.aborted = dict(steps), set(loop_ids), {}, []

    def add_request(self, request_id, prompt, params):
        self.active[request_id] = 0

    def abort_request(self, ids):
        self.aborted += list(ids)
        for request_id in ids:
            self.active.pop(request_id, None)

    def step(self):
        outputs = []
        for request_id in list(self.active):
            self.active[request_id] += 1
            finished = self.active[request_id] >= self.steps[request_id]
            outputs.append(SimpleNamespace(request_id=request_id, finished=finished))
            if finished:
                del self.active[request_id]
        return outputs


class StreamHooks(unittest.TestCase):
    def stream(self, engine, requests, **kwargs):
        return continuous_stream(engine, requests, 2, lambda r: ("p", None, {"calls": 0}),
                                 lambda o, r, i: {"finished": r["request_id"]}, StreamStats(), **kwargs)

    def run_stream(self, engine, requests, **kwargs):
        return list(self.stream(engine, requests, **kwargs))

    def test_wait_resumes_after_results(self):
        done = []

        def requests():
            yield {"request_id": "a"}
            while "a" not in done:
                yield WAIT
            yield {"request_id": "b"}

        engine = FakeEngine({"a": 3, "b": 1})
        results = []
        for request, row in self.stream(engine, requests()):  # consumed lazily, as run_generation does
            results.append(row["finished"])
            done.append(request["request_id"])
        self.assertEqual(results, ["a", "b"])

    def test_wait_with_nothing_active_is_an_error(self):
        def requests():
            yield WAIT

        with self.assertRaises(RuntimeError):
            self.run_stream(FakeEngine({}), requests())

    def test_observe_aborts_and_yields_partial(self):
        def observe(output, info):
            info["calls"] += 1
            return output.request_id == "loop" and info["calls"] >= 2

        engine = FakeEngine({"loop": 10, "ok": 3})
        rows = self.run_stream(engine, iter([{"request_id": "loop"}, {"request_id": "ok"}]), observe=observe,
                               partial=lambda o, r, i: {"partial": r["request_id"]})
        self.assertEqual([row for _, row in rows], [{"partial": "loop", "seconds": rows[0][1]["seconds"]},
                                                    {"finished": "ok", "seconds": rows[1][1]["seconds"]}])
        self.assertEqual(engine.aborted, ["loop"])
        with self.assertRaises(ValueError):
            self.run_stream(engine, iter([]), observe=observe)


if __name__ == "__main__":
    unittest.main()


class VllmDeltaOutputs(unittest.TestCase):
    """The loop stop against real vllm 0.11 RequestOutput/CompletionOutput objects in DELTA form (skipped without vllm)."""

    def setUp(self):
        try:
            from vllm.outputs import CompletionOutput, RequestOutput
        except ImportError:  # pragma: no cover - vllm is pinned in env/requirements-vllm.lock
            self.skipTest("vllm is not installed")
        self.RequestOutput, self.CompletionOutput = RequestOutput, CompletionOutput

    def engine(self, pieces):
        """step() returns one DELTA RequestOutput per active request, its last piece finished like vLLM's processor."""
        test = self

        class Engine:
            def __init__(self):
                self.active, self.aborted = {}, []

            def add_request(self, request_id, prompt, params):
                self.active[request_id] = 0

            def abort_request(self, ids):
                self.aborted += ids
                for request_id in ids:
                    self.active.pop(request_id)

            def step(self):
                outputs = []
                for request_id in list(self.active):
                    index = self.active[request_id]
                    finished = index == len(pieces[request_id]) - 1
                    completion = test.CompletionOutput(0, "", pieces[request_id][index], None, None,
                                                       finish_reason="stop" if finished else None)
                    outputs.append(test.RequestOutput(request_id, None, [1, 2], None, [completion], finished))
                    self.active[request_id] = index + 1
                    if finished:
                        del self.active[request_id]
                return outputs
        return Engine()

    def test_pieces_are_joined_and_a_loop_is_aborted(self):
        from fine_tuning.error_note.collect import delta_ids, loop_stop_info, observe_delta

        block = list(range(64))
        pieces = {"ok": [[5, 6], [7], [8, 9]], "loop": [block] * 20}
        engine = self.engine(pieces)
        rows = list(continuous_stream(engine, iter([{"request_id": "ok"}, {"request_id": "loop"}]), 2,
                                      lambda r: ("p", None, loop_stop_info(3)),
                                      lambda o, r, i: {"ids": delta_ids(o, i), "finish": "stop"}, StreamStats(),
                                      observe=observe_delta,
                                      partial=lambda o, r, i: {"ids": list(i["ids"]), "finish": "loop_stop"}))
        by_id = {request["request_id"]: row for request, row in rows}
        self.assertEqual(by_id["ok"]["ids"], [5, 6, 7, 8, 9])
        self.assertEqual(by_id["loop"]["finish"], "loop_stop")
        self.assertEqual(len(by_id["loop"]["ids"]), 64 * 8)  # stopped at the eighth non-overlapping repeat
        self.assertEqual(engine.aborted, ["loop"])
