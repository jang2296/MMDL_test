"""Authored cases for the error-note collect stage; fake engines, graders and tokenizers only, no GPU, weights or network."""
import contextlib
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fine_tuning.error_note import collect
from fine_tuning.error_note.collect import (
    InputTooLong, ResourceLog, StreamStats, append_jsonl, auto_kv_gib, auto_seqs, bounded_map, collect_rows,
    continuous_stream, gpu_total_gib, make_records, read_jsonl_done, resolve_kv, run_note, run_solve, sweep)
from fine_tuning.error_note.contract import MAX_NEW_TOKENS, sample_seed
from fine_tuning.notebook_rl.prompts import p0_eval_text

ROOT = Path(__file__).resolve().parents[2]


def no_gpu():
    return None, None, None


def make_items(n):
    return [{"problem_id": f"p{i}", "subject": "Biology" if i % 2 else "Math", "source": "src"} for i in range(n)]


def make_actors(n):
    return {f"p{i}": SimpleNamespace(problem_id=f"p{i}", question=f"Question {i}?", choices=("alpha", "beta"),
                                     image_paths=(f"p{i}.png",), mode="plain") for i in range(n)}


def load(actor):
    return [object()]


def default_output(request):
    return {"text": "Step one. Final answer: A", "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 5,
            "seconds": 0.1}


class FakeStream:
    """Stands in for the engine stream: yields one output per request and records what it was asked to generate."""

    def __init__(self, make_output=default_output, fail_after=None):
        self.make_output, self.fail_after, self.requests = make_output, fail_after, []

    def __call__(self, requests, stats, stop_after):
        for request in requests:
            if self.fail_after is not None and len(self.requests) >= self.fail_after:
                raise RuntimeError("killed")
            self.requests.append(request)
            stats.steps += 1
            stats.step_seconds += 0.5
            stats.active_sum += 2
            yield request, self.make_output(request)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def note_text(image="CORRECT", corrected="Step. Final answer: B"):
    return "\n".join(["What the image shows: " + words(40), f"Image reading: {image}", "Cause: NONE",
                      "What went wrong: nothing", "How to solve it: read the labels", "Corrected solution:\n" + corrected])


def tok(text):
    return list(range(len(text.split())))


def write_order(path, n):
    ids = [f"p{i}" for i in range(n)]
    path.write_text(json.dumps({"version": "error-note-order-v1", "master_seed": 3407, "count": n,
                                "items": [{"problem_id": i, "subject": "S", "source": "s"} for i in ids],
                                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}))


class SolveTests(unittest.TestCase):
    def test_solve_writes_and_resumes(self):
        grade = lambda pid, text, finish_reason, mode: ("CORRECT", "A")  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = FakeStream(fail_after=2)
            with self.assertRaises(RuntimeError):
                run_solve(make_items(4), make_actors(4), load, first, StreamStats(), grade, out, S=2, B=16384,
                          engine="vllm", gpu_probe=no_gpu)
            self.assertEqual(sorted(read_jsonl_done(out / "t1.jsonl")), ["p0", "p1"])
            failed = json.loads((out / "receipt-t1.json").read_text())
            self.assertEqual((failed["items_done"], failed["completed"]), (2, False))
            self.assertIn("killed", failed["error"])
            with open(out / "t1.jsonl", "ab") as stream:  # a kill in the middle of a write
                stream.write(b'{"problem_id": "p2", "te')
            second = FakeStream()
            with self.assertWarns(RuntimeWarning):
                receipt = run_solve(make_items(4), make_actors(4), load, second, StreamStats(), grade, out, S=2,
                                    B=16384, engine="vllm", gpu_probe=no_gpu)
            self.assertEqual([r["request_id"] for r in second.requests], ["p2", "p3"])
            self.assertEqual((receipt["items_done"], receipt["completed"], receipt["finished_tokens"]), (2, True, 10))
            self.assertEqual(receipt["token_source"], "active_times_steps")
            rows = read_jsonl_done(out / "t1.jsonl")
            self.assertEqual(sorted(rows), ["p0", "p1", "p2", "p3"])
            self.assertEqual(set(rows["p3"]), {"problem_id", "stage", "seed", "prompt_tokens", "completion_tokens",
                                               "finish_reason", "text", "status", "extracted", "seconds", "engine"})
            self.assertEqual((rows["p3"]["stage"], rows["p3"]["status"], rows["p3"]["extracted"], rows["p3"]["engine"]),
                             ("t1", "CORRECT", "A", "vllm"))
            self.assertEqual(rows["p3"]["seed"], sample_seed("p3", "t1"))
            self.assertTrue((out / "throughput.jsonl").read_text().strip())
            third = FakeStream()
            run_solve(make_items(4), make_actors(4), load, third, StreamStats(), grade, out, S=2, B=16384,
                      engine="vllm", gpu_probe=no_gpu)
            self.assertEqual(third.requests, [])  # nothing left: no engine is ever asked

    def test_solve_prompt_has_no_gold(self):
        gold = "SECRETGOLD"
        seen = []

        def grade(pid, text, finish_reason, mode):
            seen.append(gold)  # only the grader side may know the gold
            return "INCORRECT", "A"

        with tempfile.TemporaryDirectory() as tmp:
            stream = FakeStream()
            run_solve(make_items(2), make_actors(2), load, stream, StreamStats(), grade, Path(tmp), S=2, B=16384,
                      engine="hf", gpu_probe=no_gpu)
        self.assertTrue(seen)
        self.assertNotIn("gold", run_solve.__code__.co_varnames[:run_solve.__code__.co_argcount])
        for request in stream.requests:
            self.assertNotIn(gold, json.dumps(request["messages"]))
            content = request["messages"][0]["content"]
            self.assertEqual(content[0], {"type": "image"})
            self.assertEqual(content[-1]["text"], p0_eval_text(f"Question {request['request_id'][1:]}?", ("alpha", "beta")))
            self.assertEqual((request["max_tokens"], request["seed"]), (MAX_NEW_TOKENS, sample_seed(request["request_id"], "t1")))
            self.assertEqual(len(request["images"]), 1)

    def test_solve_records_context_blocked(self):
        blocked = lambda request: {"blocked": True, "error": "too long"}  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            run_solve(make_items(1), make_actors(1), load, FakeStream(blocked), StreamStats(),
                      lambda *a: self.fail("a blocked item is never graded"), Path(tmp), S=1, B=16384, engine="vllm",
                      gpu_probe=no_gpu)
            row = read_jsonl_done(Path(tmp) / "t1.jsonl")["p0"]
        self.assertEqual((row["status"], row["completion_tokens"]), ("CONTEXT_BLOCKED", 0))
        self.assertNotIn("text", row)


class NoteTests(unittest.TestCase):
    def t1_rows(self, statuses):
        return {f"p{i}": {"problem_id": f"p{i}", "status": s, "text": f"my first solution {i}", "finish_reason": "stop"}
                for i, s in enumerate(statuses)}

    def test_note_prompt_contains_gold_and_grades_corrected(self):
        graded = []

        def grade(pid, text, finish_reason, mode):
            graded.append((pid, text, finish_reason))
            return "CORRECT", "B"

        def output(request):
            text = note_text("MISREAD") if request["request_id"] == "p0" else "no headings at all"
            return default_output(request) | {"text": text}

        with tempfile.TemporaryDirectory() as tmp:
            stream = FakeStream(output)
            run_note(make_items(2), make_actors(2), self.t1_rows(["INCORRECT", "INCORRECT"]), {"p0": "B", "p1": "B"},
                     load, stream, StreamStats(), grade, Path(tmp), S=2, B=16384, engine="vllm", gpu_probe=no_gpu)
            rows = read_jsonl_done(Path(tmp) / "note.jsonl")
        request = stream.requests[0]
        for expected in ("Correct answer: B", "Grading result: WRONG", "my first solution 0", "Question 0?"):
            self.assertIn(expected, request["prompt"])
        self.assertEqual(request["messages"][0]["content"][-1]["text"], request["prompt"])
        self.assertEqual(request["seed"], sample_seed("p0", "note"))
        self.assertEqual(graded, [("p0", "Step. Final answer: B", "stop")])  # the unparsed note is not graded
        self.assertEqual(set(rows["p0"]), {"problem_id", "stage", "seed", "prompt", "prompt_tokens", "completion_tokens",
                                           "finish_reason", "text", "image_reading", "cause", "corrected_status",
                                           "corrected_extracted", "seconds", "engine"})
        self.assertEqual((rows["p0"]["image_reading"], rows["p0"]["cause"], rows["p0"]["corrected_status"],
                          rows["p0"]["corrected_extracted"]), ("MISREAD", "NONE", "CORRECT", "B"))
        self.assertEqual((rows["p1"]["image_reading"], rows["p1"]["corrected_status"], rows["p1"]["corrected_extracted"]),
                         (None, None, None))

    def test_note_skips_unverifiable(self):
        statuses = ["UNVERIFIABLE", "CONTEXT_BLOCKED", "CORRECT", "INCORRECT"]
        gold = {f"p{i}": "B" for i in range(4)}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stream = FakeStream()
            args = (make_items(4), make_actors(4), self.t1_rows(statuses), gold, load)
            kwargs = dict(S=2, B=16384, engine="vllm", gpu_probe=no_gpu)
            run_note(*args, stream, StreamStats(), lambda *a: ("CORRECT", "B"), out, **kwargs)
            self.assertEqual([r["request_id"] for r in stream.requests], ["p2", "p3"])
            rows = read_jsonl_done(out / "note.jsonl")
            self.assertEqual(rows["p0"], {"problem_id": "p0", "stage": "note", "status": "SKIPPED_UNVERIFIABLE"})
            self.assertEqual(rows["p1"]["status"], "SKIPPED_CONTEXT_BLOCKED")
            again = FakeStream()
            run_note(*args, again, StreamStats(), lambda *a: ("CORRECT", "B"), out, **kwargs)  # resume: no duplicates
            self.assertEqual(again.requests, [])
            self.assertEqual(len(read_jsonl_done(out / "note.jsonl")), 4)
            with self.assertRaises(ValueError):
                run_note(make_items(5), make_actors(5), self.t1_rows(statuses), gold, load, FakeStream(), StreamStats(),
                         lambda *a: ("CORRECT", "B"), out, **kwargs)


class FakeClock:
    now = 0.0

    def __call__(self):
        return self.now


class FakeEngine:
    def __init__(self, clock, length=3):
        self.clock, self.length = clock, length
        self.remaining, self.aborted, self.max_running, self.added = {}, [], 0, []

    def add_request(self, request_id, prompt, params):
        self.remaining[request_id] = self.length
        self.added.append(request_id)
        self.max_running = max(self.max_running, len(self.remaining))

    def step(self):
        self.clock.now += 1
        finished = []
        for request_id in list(self.remaining):
            self.remaining[request_id] -= 1
            if not self.remaining[request_id]:
                del self.remaining[request_id]
                finished.append(SimpleNamespace(request_id=request_id, finished=True))
        return finished

    def abort_request(self, request_ids):
        self.aborted += request_ids
        for request_id in request_ids:
            self.remaining.pop(request_id, None)


class StreamTests(unittest.TestCase):
    def run_stream(self, ids, limit, stop_after=None):
        clock, stats = FakeClock(), StreamStats()
        engine = FakeEngine(clock)

        def prepare(request):
            if request["request_id"] == "big":
                raise InputTooLong("too long")
            return f"prompt-{request['request_id']}", None, None

        out = list(continuous_stream(engine, ({"request_id": i} for i in ids), limit, prepare,
                                     lambda output, request, info: {"completion_tokens": 3}, stats,
                                     stop_after=stop_after, clock=clock))
        return engine, stats, out

    def test_continuous_stream_refills_to_limit_and_aborts_on_deadline(self):
        engine, stats, out = self.run_stream([f"r{i}" for i in range(5)], 2)
        self.assertEqual([request["request_id"] for request, _ in out], ["r0", "r1", "r2", "r3", "r4"])
        self.assertEqual((engine.max_running, engine.aborted, stats.stopped_early), (2, [], False))
        self.assertEqual((stats.steps, stats.active_sum), (9, 15))  # 2 active for 6 steps, then 1 for 3
        self.assertEqual([row["seconds"] for _, row in out], [1.5, 1.5, 1.5, 1.5, 3.0])  # step wall shared by the active
        engine, stats, out = self.run_stream([f"r{i}" for i in range(10)], 2, stop_after=4)
        self.assertEqual([request["request_id"] for request, _ in out], ["r0", "r1"])
        self.assertEqual((sorted(engine.aborted), engine.remaining, engine.max_running), (["r2", "r3"], {}, 2))
        self.assertTrue(stats.stopped_early)
        engine, stats, out = self.run_stream(["a", "big", "b"], 2)  # recorded without generating, never added
        self.assertEqual([(r["request_id"], o.get("blocked", False)) for r, o in out],
                         [("big", True), ("a", False), ("b", False)])
        self.assertEqual(engine.added, ["a", "b"])


class SweepTests(unittest.TestCase):
    class Runner:
        def __init__(self, out, total, tps, per_child=2, fail_with=0):
            self.out, self.total, self.tps, self.per_child, self.fail_with, self.calls = out, total, tps, per_child, fail_with, []
            self.commands = []

        def __call__(self, command):
            self.commands.append(command)
            seqs = int(command[command.index("--max-num-seqs") + 1])
            stopping = "--stop-after-seconds" in command
            self.calls.append((seqs, stopping))
            if self.fail_with:
                return self.fail_with
            done = read_jsonl_done(self.out / "t1.jsonl")
            fresh = [f"p{i}" for i in range(self.total) if f"p{i}" not in done][:self.per_child if stopping else self.total]
            for pid in fresh:
                append_jsonl(self.out / "t1.jsonl", {"problem_id": pid})
            (self.out / "receipt-t1.json").write_text(json.dumps({"S": seqs, "tokens_per_second": self.tps[seqs],
                                                                  "items_done": len(fresh)}))
            return 0

    def args(self, tmp, total, seqs="32,48,64", kv=28.0):
        write_order(tmp / "order.json", total)
        return SimpleNamespace(build_root=tmp / "V", order=tmp / "order.json", limit=total, output=tmp / "out",
                               model_path=tmp / "M", engine="vllm", kv_cache_gib=kv, max_num_batched_tokens=16384,
                               gpu_memory_utilization=0.9, seqs=seqs, seconds=300.0)

    def test_sweep_picks_best_and_runs_final(self):
        tps = {32: 100.0, 48: 150.0, 64: 150.0}
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            args = self.args(tmp, 10)
            runner = self.Runner(tmp / "out", 10, tps)
            self.assertEqual(sweep(args, runner), 0)
            self.assertEqual(runner.calls, [(32, True), (48, True), (64, True), (48, False)])  # tie goes to the smaller S
            result = json.loads((tmp / "out" / "sweep.json").read_text())
            self.assertEqual((result["best_seqs"], [r["S"] for r in result["runs"]], result["final"]["S"]), (48, [32, 48, 64], 48))
            for seqs in (32, 48, 64):
                self.assertTrue((tmp / "out" / f"receipt-t1-s{seqs}.json").exists())
            self.assertEqual(len(read_jsonl_done(tmp / "out" / "t1.jsonl")), 10)  # no generated item was discarded
        with tempfile.TemporaryDirectory() as raw:  # everything finished in the first child: the rest are skipped
            tmp = Path(raw)
            runner = self.Runner(tmp / "out", 2, tps)
            self.assertEqual(sweep(self.args(tmp, 2), runner), 0)
            self.assertEqual(runner.calls, [(32, True)])
            self.assertEqual(json.loads((tmp / "out" / "sweep.json").read_text())["best_seqs"], 32)
        with tempfile.TemporaryDirectory() as raw:  # a failed child stops the sweep with its exit code
            tmp = Path(raw)
            runner = self.Runner(tmp / "out", 10, tps, fail_with=3)
            self.assertEqual(sweep(self.args(tmp, 10), runner), 3)
            self.assertEqual(len(runner.calls), 1)

    def test_sweep_fixed_records_provenance_and_never_reads_the_gpu(self):
        def no_probe():
            raise AssertionError("a fixed KV size and an explicit --seqs never read the GPU")

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            runner = self.Runner(tmp / "out", 2, {32: 100.0, 48: 150.0, 64: 150.0})
            self.assertEqual(sweep(self.args(tmp, 2), runner, no_probe), 0)
            result = json.loads((tmp / "out" / "sweep.json").read_text())
        self.assertEqual({k: result[k] for k in ("gpu_total_gib", "kv_cache_gib_resolved", "seqs_resolved", "kv_source")},
                         {"gpu_total_gib": None, "kv_cache_gib_resolved": 28.0, "seqs_resolved": [32, 48, 64],
                          "kv_source": "fixed"})

    def test_sweep_auto_uses_the_probe_and_resolved_list(self):
        probed = []

        def probe():
            probed.append(1)
            return 44.42

        expected = auto_seqs(28)
        tps = {s: 100.0 + i for i, s in enumerate(expected)}
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            runner = self.Runner(tmp / "out", 10, tps)
            self.assertEqual(sweep(self.args(tmp, 10, seqs="auto", kv="auto"), runner, probe), 0)
            result = json.loads((tmp / "out" / "sweep.json").read_text())
        self.assertEqual(probed, [1])  # read once
        self.assertEqual(runner.calls, [(s, True) for s in expected] + [(expected[-1], False)])
        for command in runner.commands:  # the children resolve "auto" themselves
            self.assertEqual(command[command.index("--kv-cache-gib") + 1], "auto")
        self.assertEqual((result["gpu_total_gib"], result["kv_cache_gib_resolved"], result["seqs_resolved"],
                          result["kv_source"], result["best_seqs"]), (44.42, 28, expected, "auto", expected[-1]))
        with tempfile.TemporaryDirectory() as raw:  # auto candidates from a fixed KV size need no GPU read
            tmp = Path(raw)
            runner = self.Runner(tmp / "out", 2, {s: 1.0 for s in auto_seqs(8)})
            self.assertEqual(sweep(self.args(tmp, 2, seqs="auto", kv=8.9), runner, lambda: self.fail("no GPU read")), 0)
            self.assertEqual(json.loads((tmp / "out" / "sweep.json").read_text())["seqs_resolved"], auto_seqs(8))


class AutoSizingTests(unittest.TestCase):
    def test_auto_kv_gib(self):
        for total, expected in ((44.42, 28), (79.1, 63), (94.5, 78)):  # A40, H100, 96 GB
            self.assertEqual(auto_kv_gib(total), expected)
        self.assertEqual(auto_kv_gib(24.0), 8)
        for total in (20, 23.99):
            with self.assertRaises(ValueError):
                auto_kv_gib(total)

    def test_auto_seqs_follow_the_formula(self):
        def expected(kv_gib):
            estimate = kv_gib * 7280 / 4800
            return sorted({max(8, round(f * estimate) // 8 * 8) for f in (0.75, 1.0, 1.25)})

        for kv_gib in (28, 63, 78):
            got = auto_seqs(kv_gib)
            self.assertEqual(got, expected(kv_gib))
            self.assertEqual(len(got), 3)
            self.assertTrue(all(s % 8 == 0 and s >= 8 for s in got))
        self.assertEqual(auto_seqs(28), [32, 40, 48])
        self.assertEqual(auto_seqs(8), [8])  # all three candidates collapse to the floor

    def test_reserve_covers_the_measured_memory(self):
        self.assertGreaterEqual(collect.AUTO_RESERVE_GIB, collect.WEIGHTS_GIB + collect.OTHER_VLLM_GIB)

    def test_resolve_kv_reads_the_gpu_only_for_auto(self):
        def no_probe():
            raise AssertionError("fixed must not read the GPU")

        self.assertEqual(resolve_kv(12.5, no_probe),
                         {"gpu_total_gib": None, "kv_cache_gib_resolved": 12.5, "kv_source": "fixed"})
        self.assertEqual(resolve_kv("auto", lambda: 79.1),
                         {"gpu_total_gib": 79.1, "kv_cache_gib_resolved": 63, "kv_source": "auto"})
        with self.assertRaises(ValueError):
            resolve_kv("auto", lambda: 20.0)

    def test_gpu_total_gib_from_nvidia_smi_sample(self):
        with mock.patch.object(collect, "_gpu_sample", return_value=(7, 100, 46068)):
            self.assertAlmostEqual(gpu_total_gib(), 46068 / 1024)
        with mock.patch.object(collect, "_gpu_sample", return_value=(None, None, None)):
            with self.assertRaises(RuntimeError):
                gpu_total_gib()

    def test_cli_parses_auto_and_numbers(self):
        base = ["sweep", "--build-root", "b", "--order", "o", "--limit", "1", "--output", "out", "--model-path", "m",
                "--engine", "vllm"]
        parser = collect._parser()
        args = parser.parse_args(base + ["--seqs", "32,48"])
        self.assertEqual((args.kv_cache_gib, args.seqs), (28, "32,48"))  # defaults unchanged
        args = parser.parse_args(base + ["--seqs", "auto", "--kv-cache-gib", "auto"])
        self.assertEqual((args.kv_cache_gib, args.seqs), ("auto", "auto"))
        self.assertEqual(parser.parse_args(base + ["--seqs", "8", "--kv-cache-gib", "12.5"]).kv_cache_gib, 12.5)
        solve = parser.parse_args(["solve", *base[1:], "--kv-cache-gib", "auto"])
        self.assertEqual(solve.kv_cache_gib, "auto")
        for bad in ("fast", "0", "-1", "nan", "inf"):
            with self.subTest(value=bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(base + ["--seqs", "8", "--kv-cache-gib", bad])

    def test_receipt_carries_the_resolved_kv(self):
        extra = {"gpu_total_gib": 44.42, "kv_cache_gib_resolved": 28, "kv_source": "auto"}
        with tempfile.TemporaryDirectory() as tmp:
            receipt = run_solve(make_items(1), make_actors(1), load, FakeStream(), StreamStats(receipt_extra=dict(extra)),
                                lambda *a: ("CORRECT", "A"), Path(tmp), S=1, B=16384, engine="vllm", gpu_probe=no_gpu)
            written = json.loads((Path(tmp) / "receipt-t1.json").read_text())
        for key, value in extra.items():
            self.assertEqual((receipt[key], written[key]), (value, value))


class UtilityTests(unittest.TestCase):
    def test_bounded_map_order_and_bound(self):
        pulled = [0]

        def source():
            for i in range(30):
                pulled[0] += 1
                yield i

        def double(i):
            time.sleep((30 - i) * 0.0005)  # later items finish first
            return i * 2

        got, worst = [], 0
        for value in bounded_map(double, source(), workers=4, buffer=6):
            worst = max(worst, pulled[0] - len(got))
            got.append(value)
        self.assertEqual(got, [i * 2 for i in range(30)])
        self.assertLessEqual(worst, 6)
        with self.assertRaises(ValueError):
            list(bounded_map(double, [1], workers=1, buffer=0))

    def test_read_jsonl_done_truncated_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            self.assertEqual(read_jsonl_done(path), {})
            append_jsonl(path, {"problem_id": "a", "text": "café"})
            with open(path, "ab") as stream:
                stream.write(b'{"problem_id": "b", "te')
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(sorted(read_jsonl_done(path)), ["a"])
            self.assertTrue(path.read_bytes().endswith(b"\n"))  # the cut line is gone, so appends stay well formed
            append_jsonl(path, {"problem_id": "b"})
            self.assertEqual(sorted(read_jsonl_done(path)), ["a", "b"])
            self.assertEqual(read_jsonl_done(path)["a"]["text"], "café")
            append_jsonl(path, {"problem_id": "a"})
            with self.assertRaises(ValueError):
                read_jsonl_done(path)

    def test_import_is_light(self):
        code = ("import sys, fine_tuning.error_note.collect; "
                "bad = [m for m in ('torch', 'vllm', 'transformers', 'peft', 'bitsandbytes') if m in sys.modules]; "
                "assert not bad, bad")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr[-500:])

    def test_resource_log_writes_lines(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(collect, "_gpu_sample", return_value=(7, 100, 200)):
            path = Path(tmp) / "resource.jsonl"
            with ResourceLog(path, every=0.05):
                time.sleep(0.2)
            lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertEqual({k: lines[0][k] for k in ("gpu_util_percent", "gpu_mem_used_mib", "gpu_mem_total_mib")},
                         {"gpu_util_percent": 7, "gpu_mem_used_mib": 100, "gpu_mem_total_mib": 200})
        self.assertTrue({"utc", "cpu_percent", "loadavg", "mem_available_kib"} <= set(lines[0]))


class RowsTests(unittest.TestCase):
    def test_rows_join_and_summary(self):
        items, actors = make_items(4), make_actors(4)
        t1 = {
            "p0": {"problem_id": "p0", "status": "CORRECT", "text": words(40, "t"), "finish_reason": "stop"},
            "p1": {"problem_id": "p1", "status": "INCORRECT", "text": words(40, "t"), "finish_reason": "stop"},
            "p2": {"problem_id": "p2", "status": "UNVERIFIABLE", "text": words(40, "t"), "finish_reason": "stop"},
            "p3": {"problem_id": "p3", "status": "INCORRECT", "text": words(40, "t"), "finish_reason": "stop"}}

        def note(text, status="CORRECT"):
            return {"problem_id": "x", "prompt": "NOTE PROMPT", "text": text, "finish_reason": "stop", "corrected_status": status}

        notes = {"p0": note(note_text("CORRECT", words(40, "c") + " Final answer: B")),
                 "p1": note(note_text("MISREAD", words(40, "c") + " Final answer: B")),
                 "p2": {"problem_id": "p2", "stage": "note", "status": "SKIPPED_UNVERIFIABLE"},
                 "p3": note("garbage without headings", None)}
        records = make_records(items, actors, t1, notes)
        self.assertEqual(records[0]["p0_text"], p0_eval_text("Question 0?", ("alpha", "beta")))
        self.assertEqual((records[0]["image_paths"], records[0]["subject"], records[1]["subject"]), (["p0.png"], "Math", "Biology"))
        self.assertNotIn("note", records[2])
        rows, drops, summary = collect_rows(records, tok)
        self.assertEqual([r["row_id"] for r in rows], ["p0:A", "p0:B", "p1:A", "p1:B"])
        self.assertEqual((rows[0]["target_source"], rows[2]["target_source"], rows[2]["route"]), ("T1", "CORRECTED", "FULL"))
        self.assertEqual(sorted((d["problem_id"], d["kind"], d["reason"]) for d in drops),
                         [("p2", "A", "NO_NOTE"), ("p2", "B", "NO_NOTE"), ("p3", "A", "NOTE_UNPARSED"), ("p3", "B", "NOTE_UNPARSED")])
        self.assertEqual((summary["items"], summary["rows"], summary["drops"]), (4, 4, 4))
        self.assertEqual(summary["rows_by_kind"], {"A": 2, "B": 2})
        self.assertEqual(summary["rows_by_target_source"], {"T1": 1, "CORRECTED": 1, "NOTE": 2})
        self.assertEqual(summary["rows_by_route"], {"LM": 2, "FULL": 2})
        self.assertEqual(summary["rows_by_subject"], {"Math": 2, "Biology": 2})
        self.assertEqual(summary["drops_by_reason"], {"NO_NOTE": 2, "NOTE_UNPARSED": 2})
        self.assertEqual(summary["image_reading"], {"CORRECT": 1, "MISREAD": 1, "NO_NOTE": 1, "UNPARSED": 1})
        self.assertEqual(summary["cause"], {"NONE": 2, "NO_NOTE": 1, "UNPARSED": 1})
        self.assertEqual(summary["t1_status"], {"CORRECT": 1, "INCORRECT": 2, "UNVERIFIABLE": 1})
        self.assertEqual(summary["mean_target_tokens"]["A"], 41.5)
        self.assertNotIn("w0", json.dumps(summary))  # counts only
        with self.assertRaises(ValueError):
            make_records(items, actors, {}, {})


if __name__ == "__main__":
    unittest.main()
