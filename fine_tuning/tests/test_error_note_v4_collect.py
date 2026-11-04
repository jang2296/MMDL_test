"""Authored cases for the version-4 collect stages with fake streams, graders and tokenizers (no GPU or weights)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fine_tuning.error_note import collect_v4 as c4
from fine_tuning.error_note import v4
from fine_tuning.error_note.collect import LOOP_STOP_FINISH, StreamStats, read_jsonl_done
from fine_tuning.error_note.contract import resolve_prompt
from fine_tuning.error_note.order import VERSION, _ids_sha256
from fine_tuning.notebook_rl.prompts import p0_eval_text

CHOICES = ("alpha", "beta", "gamma")
GOLD = {"p0": "B", "p1": "B", "p2": "C", "p3": "A"}
KW = dict(S=2, B=16384, engine="vllm", gpu_probe=lambda: (None, None, None))


def actors(n=4):
    return {f"p{i}": SimpleNamespace(problem_id=f"p{i}", question=f"Question {i}?", choices=CHOICES,
                                     image_paths=(f"p{i}.png",), mode="plain") for i in range(n)}


def items(n=4):
    return [{"problem_id": f"p{i}", "subject": "Math", "source": "src"} for i in range(n)]


def load(actor):
    return [object()]


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def tok(text):
    return list(range(len(text.split())))


class FakeStream:
    """Yields one output per request; `answers(request)` decides the text. Records the requests it saw."""

    def __init__(self, answers, fail_after=None):
        self.answers, self.requests, self.fail_after = answers, [], fail_after

    def __call__(self, requests, stats, stop_after):
        for request in requests:
            if self.fail_after is not None and len(self.requests) >= self.fail_after:
                raise RuntimeError("killed")
            self.requests.append(request)
            stats.steps += 1
            stats.step_seconds += .1
            text, finish = self.answers(request)
            yield request, {"text": text, "finish_reason": finish, "prompt_tokens": 9,
                            "completion_tokens": len(text.split()), "seconds": .1}


def grader(calls):
    def grade(pid, text, finish_reason, mode):
        calls.append((pid, text))
        if finish_reason == "length":
            return "TRUNCATED", None
        return ("CORRECT", "B") if "RIGHT" in text else ("INCORRECT", "A")
    return grade


# p0: always right (adaptive stop after 2), p1: mixed, p2: always wrong, p3: first right then loop/wrong
def rollout_answers(request):
    pid, j = request["item"], request["j"]
    if pid == "p0" or (pid == "p1" and j in (0, 3)) or (pid == "p3" and j == 0):
        return words(40) + " RIGHT", "stop"
    if pid == "p3" and j == 2:
        return words(30), LOOP_STOP_FINISH
    return words(40) + " wrong", "stop"


class Rollout(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def run_rollout(self, stream=None, calls=None, **extra):
        stream = stream or FakeStream(rollout_answers)
        return c4.run_rollout(items(), actors(), load, stream, StreamStats(), grader([] if calls is None else calls),
                              self.out, round_index=1, **KW, **extra), stream

    def test_adaptive_groups_and_seeds(self):
        calls = []
        receipt, stream = self.run_rollout(calls=calls)
        groups = c4.rollouts_of(self.out)
        self.assertEqual({pid: len(rows) for pid, rows in groups.items()}, {"p0": 2, "p1": 4, "p2": 4, "p3": 4})
        self.assertEqual(receipt["groups"], {v4.ALL_CORRECT: 1, v4.MIXED: 2, v4.ALL_WRONG: 1})
        self.assertEqual(groups["p3"][2]["status"], v4.LOOP_STOPPED)
        self.assertNotIn(words(30), [text for _, text in calls])  # a loop stop is never graded
        for request in stream.requests:
            self.assertEqual(request["seed"], v4.v4_seed(request["item"], 1, "roll", request["j"]))
            self.assertEqual(request["prompt"], p0_eval_text(f"Question {request['item'][1]}?", CHOICES))
        order = [(r["item"], r["j"]) for r in stream.requests]
        self.assertLess(order.index(("p1", 0)), order.index(("p1", 2)))  # j=2,3 only after j=0,1 are graded

    def test_resume_generates_nothing_twice(self):
        self.run_rollout()
        before = (self.out / "rollouts.jsonl").read_bytes()
        _, again = self.run_rollout(stream=FakeStream(lambda r: self.fail("regenerated")))
        self.assertEqual(again.requests, [])
        self.assertEqual((self.out / "rollouts.jsonl").read_bytes(), before)

    def test_reused_f0_t1_is_rollout_zero_and_regraded(self):
        reuse = {"p2": {"problem_id": "p2", "stage": "t1", "seed": 77, "text": words(40) + " RIGHT",
                        "finish_reason": "stop", "prompt_tokens": 9, "completion_tokens": 41, "status": "INCORRECT"}}
        _, stream = self.run_rollout(reuse_t1=reuse)
        row = read_jsonl_done(self.out / "rollouts.jsonl")["p2#r0"]
        self.assertEqual((row["seed"], row["status"], row["reused_t1"]), (77, "CORRECT", True))
        self.assertNotIn(("p2", 0), [(r["item"], r["j"]) for r in stream.requests])
        with self.assertRaises(ValueError):
            c4.run_rollout(items(), actors(), load, FakeStream(rollout_answers), StreamStats(), grader([]), self.out,
                           round_index=2, reuse_t1=reuse, **KW)


class NotePath(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        c4.run_rollout(items(), actors(), load, FakeStream(rollout_answers), StreamStats(), grader([]), self.out,
                       round_index=1, **KW)

    def note_text(self, lesson, shows="The figure has 3 bars on the left side. The title is short."):
        return (f"What the image shows: {shows}\nImage reading: CORRECT\nCause: REASONING\n"
                f"What went wrong: I misread a bar.\nLesson: {lesson}")

    def write_notes(self, lesson="Compare each bar with the axis before choosing."):
        chosen = c4.note_inputs(c4.rollouts_of(self.out), list(GOLD))
        self.assertEqual(set(chosen), {"p2"})
        self.assertEqual(chosen["p2"]["note_j"], 0)
        from fine_tuning.error_note.collect import run_note

        stream = FakeStream(lambda r: (self.note_text(lesson), "stop"))
        run_note([items()[2]], actors(), chosen, GOLD, load, stream, StreamStats(), grader([]), self.out, **KW,
                 version="v2")
        return c4.write_gates(self.out, ["p2"], GOLD, actors(), 1)

    def test_loop_stopped_note_input_is_cut_off(self):
        groups = {"x": [{"j": 0, "status": v4.LOOP_STOPPED, "text": "t"}, {"j": 1, "status": "INCORRECT", "text": "u"}]}
        self.assertEqual(c4.note_inputs(groups, ["x"])["x"], {"text": "t", "status": "TRUNCATED", "note_j": 0})

    def test_gate_file_has_lesson_and_claims_but_no_gold(self):
        gates = self.write_notes()
        self.assertEqual(gates["items"]["p2"]["result"], "PASS")
        self.assertEqual(gates["items"]["p2"]["claims"], ["The figure has 3 bars on the left side.",
                                                          "The title is short."])
        self.assertEqual(c4.read_gates(self.out), gates["items"])
        text = (self.out / c4.GATES_FILE).read_text()
        self.assertNotIn("gamma", text)  # p2's gold option text

    def test_leaking_lesson_is_withheld(self):
        gates = self.write_notes(lesson="Choose gamma whenever bars tie.")
        self.assertEqual((gates["items"]["p2"]["result"], gates["items"]["p2"]["lesson"]), ("FAIL", None))
        self.assertNotIn("gamma", (self.out / c4.GATES_FILE).read_text())

    def test_selfcheck_resolve_control_and_rows(self):
        self.write_notes()
        gates = c4.read_gates(self.out)
        probes = FakeStream(lambda r: ("TRUE. It matches.", "stop"))
        c4.run_selfcheck(items(), actors(), gates, load, probes, StreamStats(), self.out, round_index=1, **KW)
        self.assertEqual(len(probes.requests), 2 * v4.PROBE_SEEDS)
        self.assertTrue(all(r["max_tokens"] == v4.PROBE_MAX_TOKENS for r in probes.requests))
        self.assertTrue(all("gamma" not in r["prompt"] for r in probes.requests))
        checks = c4.read_selfcheck(self.out)
        self.assertEqual(checks["p2"]["result"], v4.SELF_CONSISTENT)

        def answers(request):
            return (words(40) + (" RIGHT" if request["lesson"] and request["j"] == 1 else " wrong"), "stop")

        stream = FakeStream(answers)
        c4.run_resolve_v4(items(), actors(), gates, checks, load, stream, StreamStats(), grader([]), self.out,
                          round_index=1, control=True, **KW)
        pairs = {}
        for request in stream.requests:
            pairs.setdefault(request["j"], {})[request["lesson"]] = request
        self.assertEqual(sorted(pairs), [0, 1])
        p0 = p0_eval_text("Question 2?", CHOICES)
        for j, pair in pairs.items():
            self.assertEqual(pair[True]["seed"], pair[False]["seed"])
            self.assertEqual(pair[True]["prompt"], resolve_prompt(p0, gates["p2"]["lesson"]))
            self.assertEqual(pair[False]["prompt"], p0)
        records = c4.make_records_v4(items(), actors(), self.out)
        rows, drops, summary = c4.collect_rows_v4(records, tok)
        kinds = {(row["problem_id"], row["kind"]) for row in rows}
        self.assertEqual(kinds, {("p0", "A1"), ("p1", "RL"), ("p3", "RL"), ("p2", "A2"), ("p2", "B")})
        a2 = next(row for row in rows if row["kind"] == "A2")
        self.assertEqual((a2["prompt"], a2["row_id"]), (p0, "p2:A2:1"))
        self.assertEqual(summary["lesson_resolve"]["correct"], 1)
        self.assertEqual(summary["no_lesson_control"]["correct"], 0)
        p3 = [row for row in rows if row["problem_id"] == "p3"]
        self.assertEqual([row["append_end"] for row in p3], [True, True, False, True])

    def test_contradicted_note_gets_no_resolve(self):
        self.write_notes()
        gates = c4.read_gates(self.out)
        probes = FakeStream(lambda r: ("FALSE, there are 4 bars.", "stop"))
        c4.run_selfcheck(items(), actors(), gates, load, probes, StreamStats(), self.out, round_index=1, **KW)
        stream = FakeStream(lambda r: self.fail("resolved a contradicted note"))
        receipt = c4.run_resolve_v4(items(), actors(), gates, c4.read_selfcheck(self.out), load, stream, StreamStats(),
                                    grader([]), self.out, round_index=1, **KW)
        self.assertEqual(receipt["eligible"], 0)


class EdgeCases(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def test_killed_rollout_resumes_to_the_same_groups(self):
        with self.assertRaises(RuntimeError):
            c4.run_rollout(items(), actors(), load, FakeStream(rollout_answers, fail_after=5), StreamStats(), grader([]),
                           self.out, round_index=1, **KW)
        partial = read_jsonl_done(self.out / "rollouts.jsonl")
        self.assertEqual(len(partial), 5)
        resumed = FakeStream(rollout_answers)
        c4.run_rollout(items(), actors(), load, resumed, StreamStats(), grader([]), self.out, round_index=1, **KW)
        self.assertTrue(set(partial).isdisjoint(r["request_id"] for r in resumed.requests))
        self.assertEqual({pid: len(rows) for pid, rows in c4.rollouts_of(self.out).items()},
                         {"p0": 2, "p1": 4, "p2": 4, "p3": 4})

    def test_round_without_all_wrong_groups_has_an_empty_note_path(self):
        right = FakeStream(lambda r: (words(40) + " RIGHT", "stop"))
        c4.run_rollout(items(), actors(), load, right, StreamStats(), grader([]), self.out, round_index=1, **KW)
        receipt = c4.run_note_v4(items(), actors(), GOLD, load, FakeStream(lambda r: self.fail("note generated")),
                                 StreamStats(), grader([]), self.out, round_index=1, answer_contract="p0-eval-aligned-v1",
                                 **KW)
        self.assertEqual((receipt["note"], receipt["all_wrong_groups"], receipt["passed"]), (None, 0, 0))
        self.assertEqual(c4.read_gates(self.out), {})
        c4.run_selfcheck(items(), actors(), {}, load, FakeStream(lambda r: self.fail("probed")), StreamStats(), self.out,
                         round_index=1, **KW)
        c4.run_resolve_v4(items(), actors(), {}, c4.read_selfcheck(self.out), load, FakeStream(lambda r: self.fail("resolved")),
                          StreamStats(), grader([]), self.out, round_index=1, control=True, **KW)
        rows, _, summary = c4.collect_rows_v4(c4.make_records_v4(items(), actors(), self.out), tok)
        self.assertEqual(summary["rows_by_kind"], {"A1": 4})

    def test_selfcheck_and_resolve_resume_without_regeneration(self):
        c4.run_rollout(items(), actors(), load, FakeStream(rollout_answers), StreamStats(), grader([]), self.out,
                       round_index=1, **KW)
        note = ("What the image shows: The figure has 3 bars on the left side. The title is short.\n"
                "Image reading: CORRECT\nCause: REASONING\nWhat went wrong: I misread a bar.\n"
                "Lesson: Compare each bar with the axis before choosing.")
        c4.run_note_v4(items(), actors(), GOLD, load, FakeStream(lambda r: (note, "stop")), StreamStats(), grader([]),
                       self.out, round_index=1, answer_contract="p0-eval-aligned-v1", **KW)
        gates = c4.read_gates(self.out)
        with self.assertRaises(RuntimeError):
            c4.run_selfcheck(items(), actors(), gates, load, FakeStream(lambda r: ("TRUE.", "stop"), fail_after=3),
                             StreamStats(), self.out, round_index=1, **KW)
        rest = FakeStream(lambda r: ("TRUE.", "stop"))
        c4.run_selfcheck(items(), actors(), gates, load, rest, StreamStats(), self.out, round_index=1, **KW)
        self.assertEqual(len(rest.requests), 2 * v4.PROBE_SEEDS - 3)
        self.assertEqual(c4.read_selfcheck(self.out)["p2"]["result"], v4.SELF_CONSISTENT)
        first = FakeStream(lambda r: (words(40) + " wrong", "stop"))
        c4.run_resolve_v4(items(), actors(), gates, c4.read_selfcheck(self.out), load, first, StreamStats(), grader([]),
                          self.out, round_index=1, control=True, **KW)
        again = FakeStream(lambda r: self.fail("regenerated"))
        c4.run_resolve_v4(items(), actors(), gates, c4.read_selfcheck(self.out), load, again, StreamStats(), grader([]),
                          self.out, round_index=1, control=True, **KW)
        self.assertEqual(len(first.requests), 2 * v4.RESOLVE_M)
        (self.out / c4.GATES_FILE).write_text((self.out / c4.GATES_FILE).read_text().replace("short", "long"))
        with self.assertRaises(ValueError):  # an edited gates file breaks its digest
            c4.read_gates(self.out)


class CLI(unittest.TestCase):
    def test_parser_and_collection_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            order = Path(tmp) / "order.json"
            ids = [item["problem_id"] for item in items()]
            order.write_text(json.dumps({"version": VERSION, "items": items(), "count": len(ids),
                                         "ids_sha256": _ids_sha256(ids)}))
            base = ["--round", "1", "--build-root", tmp, "--order", str(order), "--start", "0", "--limit", "4",
                    "--output", str(Path(tmp) / "out"), "--answer-contract", "p0-eval-aligned-v1"]
            args = c4.build_parser().parse_args(["rows", *base, "--tokenizer-path", tmp])
            first = c4.bind_collection_v4(args)
            self.assertEqual(c4.bind_collection_v4(args), first)
            self.assertEqual(c4._items(args), items())
            args.limit = 3
            with self.assertRaises(ValueError):
                c4.bind_collection_v4(args)
            args.start, args.limit = 3, 2
            with self.assertRaises(ValueError):
                c4._items(args)
            gen = c4.build_parser().parse_args(["rollout", *base, "--model-path", tmp, "--engine", "hf", "--loop-stop",
                                                "--prefix-caching"])
            self.assertTrue(gen.loop_stop and gen.prefix_caching)
            self.assertEqual((gen.kv_cache_gib, gen.max_num_seqs), ("auto", "auto"))


if __name__ == "__main__":
    unittest.main()
