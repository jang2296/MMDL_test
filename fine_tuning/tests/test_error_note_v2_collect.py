"""Authored cases for the version-2 collect stages (note v2, lessons.json, gold-blind resolve, rows v2, CLI).

Fake engines, graders, services and tokenizers only: no GPU, weights or network.
"""
import contextlib
import hashlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fine_tuning.error_note import collect, contract
from fine_tuning.error_note.collect import (
    StreamStats, append_jsonl, auto_seqs, collect_rows_v2, length_report, make_records_v2, read_jsonl_done,
    read_lessons, run_note, run_resolve)
from fine_tuning.error_note.contract import MAX_NEW_TOKENS, NOTE_INSTRUCTION_V2, resolve_prompt, sample_seed
from fine_tuning.notebook_rl.prompts import p0_eval_text

CHOICES = ("alpha", "beta")
LESSON = "Read each label before computing."
LEAK_LESSON = "The correct answer is on the label."
LEAK_SKIP = "HINT_LEAK:PHRASE:correct answer"


def no_gpu():
    return None, None, None


KWARGS = dict(S=2, B=16384, engine="vllm", gpu_probe=no_gpu)


def make_items(n):
    return [{"problem_id": f"p{i}", "subject": "Biology" if i % 2 else "Math", "source": "src"} for i in range(n)]


def make_actors(n):
    return {f"p{i}": SimpleNamespace(problem_id=f"p{i}", question=f"Question {i}?", choices=CHOICES,
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


def tok(text):
    return list(range(len(text.split())))


def write_order(path, n):
    ids = [f"p{i}" for i in range(n)]
    path.write_text(json.dumps({"version": "error-note-order-v1", "master_seed": 3407, "count": n,
                                "items": [{"problem_id": i, "subject": "S", "source": "s"} for i in ids],
                                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}))


def v2_note_text(lesson=LESSON, image="CORRECT"):
    return "\n".join(["What the image shows: " + words(40), f"Image reading: {image}", "Cause: NONE",
                      "What went wrong: nothing", "Lesson: " + lesson])


def t1_rows(statuses):
    return {f"p{i}": {"problem_id": f"p{i}", "status": s, "text": f"my first solution {i}", "finish_reason": "stop"}
            for i, s in enumerate(statuses)}


def pid_number(request):
    return request["request_id"][1:]


class NoteV2Tests(unittest.TestCase):
    def test_note_v2_prompt_rows_and_lessons(self):
        statuses = ["INCORRECT", "INCORRECT", "INCORRECT", "UNVERIFIABLE", "INCORRECT", "INCORRECT"]
        gold = {f"p{i}": "B" for i in range(6)}
        texts = {"p0": (v2_note_text(), "stop"), "p1": (v2_note_text("Option B fits."), "stop"),
                 "p2": (v2_note_text(), "length"), "p4": ("no headings at all", "stop")}

        def output(request):
            if request["request_id"] == "p5":
                return {"blocked": True, "error": "too long"}
            text, finish_reason = texts[request["request_id"]]
            return default_output(request) | {"text": text, "finish_reason": finish_reason}

        def grade(*args):
            self.fail("the v2 note has no corrected solution, so nothing is graded")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stream = FakeStream(output)
            args = (make_items(6), make_actors(6), t1_rows(statuses), gold, load)
            run_note(*args, stream, StreamStats(), grade, out, **KWARGS, version="v2")
            request = next(r for r in stream.requests if r["request_id"] == "p0")
            self.assertEqual(sorted(r["request_id"] for r in stream.requests), ["p0", "p1", "p2", "p4", "p5"])
            for expected in ("Correct answer: B", "Grading result: WRONG", "my first solution 0", "Question 0?",
                             NOTE_INSTRUCTION_V2):
                self.assertIn(expected, request["prompt"])
            self.assertNotIn("Corrected solution", request["prompt"])
            self.assertEqual(request["messages"][0]["content"][-1]["text"], request["prompt"])
            self.assertEqual(request["seed"], sample_seed("p0", "note"))
            rows = read_jsonl_done(out / "note.jsonl")
            self.assertEqual(set(rows["p0"]), {"problem_id", "stage", "seed", "note_version", "prompt", "prompt_tokens",
                                               "completion_tokens", "finish_reason", "text", "image_reading", "cause",
                                               "lesson", "hint_leak", "seconds", "engine"})
            self.assertEqual((rows["p0"]["note_version"], rows["p0"]["image_reading"], rows["p0"]["cause"],
                              rows["p0"]["lesson"], rows["p0"]["hint_leak"]), ("v2", "CORRECT", "NONE", LESSON, None))
            self.assertEqual((rows["p1"]["lesson"], rows["p1"]["hint_leak"]), ("Option B fits.", "OPTION_LETTER:B"))
            self.assertEqual((rows["p4"]["image_reading"], rows["p4"]["lesson"], rows["p4"]["hint_leak"]),
                             (None, None, None))
            self.assertEqual(rows["p2"]["finish_reason"], "length")
            self.assertEqual(rows["p3"], {"problem_id": "p3", "stage": "note", "status": "SKIPPED_UNVERIFIABLE"})
            self.assertEqual((rows["p5"]["status"], rows["p5"]["note_version"]), ("CONTEXT_BLOCKED", "v2"))

            raw = (out / "lessons.json").read_text(encoding="utf-8")
            value = json.loads(raw)
            self.assertEqual((value["version"], value["note_bytes"]),
                             ("error-note-lessons-v1", (out / "note.jsonl").stat().st_size))
            self.assertEqual(value["items"], {
                "p0": {"lesson": LESSON, "skip": None},
                "p1": {"lesson": None, "skip": "HINT_LEAK:OPTION_LETTER:B"},
                "p2": {"lesson": None, "skip": "NOTE_NOT_STOPPED"},
                "p3": {"lesson": None, "skip": "NO_NOTE"},
                "p4": {"lesson": None, "skip": "NOTE_UNPARSED"},
                "p5": {"lesson": None, "skip": "NO_NOTE"}})
            for forbidden in ("Option B fits.", "Correct answer", "my first solution", "Grading result"):
                self.assertNotIn(forbidden, raw)  # lessons.json is what the gold-blind process may open

            again = FakeStream(output)  # resume: nothing is generated and lessons.json is rewritten unchanged
            run_note(*args, again, StreamStats(), grade, out, **KWARGS, version="v2")
            self.assertEqual(again.requests, [])
            self.assertEqual((out / "lessons.json").read_text(encoding="utf-8"), raw)

    def test_note_versions_do_not_mix(self):
        grade = lambda *a: ("CORRECT", "B")  # noqa: E731
        args = (make_items(2), make_actors(2), t1_rows(["INCORRECT", "INCORRECT"]), {"p0": "B", "p1": "B"}, load)
        with tempfile.TemporaryDirectory() as tmp:  # v1 notes, then v2
            out = Path(tmp)
            run_note(*args, FakeStream(), StreamStats(), grade, out, **KWARGS)
            stream = FakeStream()
            with self.assertRaises(ValueError):
                run_note(*args, stream, StreamStats(), grade, out, **KWARGS, version="v2")
            self.assertEqual(stream.requests, [])
        with tempfile.TemporaryDirectory() as tmp:  # v2 notes, then v1
            out = Path(tmp)
            run_note(*args, FakeStream(), StreamStats(), grade, out, **KWARGS, version="v2")
            stream = FakeStream()
            with self.assertRaises(ValueError):
                run_note(*args, stream, StreamStats(), grade, out, **KWARGS)
            self.assertEqual(stream.requests, [])
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            run_note(*args, FakeStream(), StreamStats(), grade, Path(tmp), **KWARGS, version="v3")

    def test_read_lessons_refuses_a_stale_file(self):
        args = (make_items(2), make_actors(2), t1_rows(["INCORRECT", "INCORRECT"]), {"p0": "B", "p1": "B"}, load)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_note(*args, FakeStream(), StreamStats(), lambda *a: self.fail("not graded"), out, **KWARGS, version="v2")
            self.assertEqual(sorted(read_lessons(out)), ["p0", "p1"])
            value = json.loads((out / "lessons.json").read_text())
            collect.write_json(out / "lessons.json", value | {"version": "other"})
            with self.assertRaisesRegex(ValueError, "unexpected version"):
                read_lessons(out)
            collect.write_json(out / "lessons.json", value)
            self.assertEqual(sorted(read_lessons(out)), ["p0", "p1"])
            append_jsonl(out / "note.jsonl", {"problem_id": "late"})  # note.jsonl grew after lessons.json was written
            with self.assertRaisesRegex(ValueError, "older"):
                read_lessons(out)


class ResolveTests(unittest.TestCase):
    STATUSES = ["CORRECT", "INCORRECT", "INVALID_NO_FINAL", "INCORRECT"]

    def lessons(self):
        clean = {f"p{i}": {"lesson": f"Check labels carefully, case {i}.", "skip": None} for i in range(4)}
        return clean | {"p2": {"lesson": None, "skip": LEAK_SKIP}}

    def grader(self, graded):
        results = {"p1": ("CORRECT", "A"), "p3": ("INCORRECT", "B")}

        def grade(pid, text, finish_reason, mode):
            graded.append((pid, text, finish_reason, mode))
            return results[pid]

        return grade

    def resolve_once(self, out, stream, graded=None, *, lessons=None, t1=None):
        graded = [] if graded is None else graded
        return run_resolve(make_items(4), make_actors(4), t1 or t1_rows(self.STATUSES), lessons or self.lessons(), load,
                           stream, StreamStats(), self.grader(graded), out, **KWARGS)

    def test_resolve_signature_has_no_gold(self):
        names = set(inspect.signature(run_resolve).parameters)
        self.assertFalse({"gold", "golds", "verifier"} & names)

    def test_resolve_generates_only_failed_items_gold_blind(self):
        graded = []
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stream = FakeStream()
            self.resolve_once(out, stream, graded)
            self.assertEqual(sorted(r["request_id"] for r in stream.requests), ["p1", "p3"])  # p0 correct, p2 skipped
            for request in stream.requests:
                pid = request["request_id"]
                expected = resolve_prompt(p0_eval_text(f"Question {pid_number(request)}?", CHOICES),
                                          f"Check labels carefully, case {pid_number(request)}.")
                self.assertEqual(request["prompt"], expected)
                self.assertEqual(request["messages"][0]["content"][-1]["text"], expected)
                for forbidden in ("Correct answer", "Grading result", "my first solution"):
                    self.assertNotIn(forbidden, request["prompt"])
                    self.assertNotIn(forbidden, json.dumps(request["messages"]))
                self.assertEqual((request["seed"], request["max_tokens"]), (sample_seed(pid, "resolve"), MAX_NEW_TOKENS))
                self.assertEqual(len(request["images"]), 1)
            self.assertEqual(sorted(graded), [("p1", "Step one. Final answer: A", "stop", "plain"),
                                              ("p3", "Step one. Final answer: A", "stop", "plain")])
            rows = read_jsonl_done(out / "resolve.jsonl")
            self.assertEqual(set(rows), {"p1", "p2", "p3"})  # a correct T1 has no row at all
            self.assertEqual(rows["p2"], {"problem_id": "p2", "stage": "resolve", "skipped": LEAK_SKIP})
            self.assertEqual(set(rows["p1"]), {"problem_id", "stage", "seed", "prompt", "prompt_tokens", "completion_tokens",
                                               "finish_reason", "text", "status", "extracted", "seconds", "engine"})
            self.assertEqual((rows["p1"]["status"], rows["p1"]["extracted"], rows["p1"]["stage"]), ("CORRECT", "A", "resolve"))
            self.assertEqual((rows["p3"]["status"], rows["p3"]["extracted"]), ("INCORRECT", "B"))
            self.assertEqual(rows["p3"]["seed"], sample_seed("p3", "resolve"))
            receipt = json.loads((out / "receipt-resolve.json").read_text())
            self.assertEqual((receipt["candidates"], receipt["skipped"], receipt["items_done"], receipt["completed"]),
                             (3, {LEAK_SKIP: 1}, 2, True))

    def test_resolve_resume_generates_nothing_more(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = FakeStream()
            self.resolve_once(out, first)
            self.assertEqual(len(first.requests), 2)  # exactly one generation per item
            second, graded = FakeStream(), []
            self.resolve_once(out, second, graded)  # p3 is still INCORRECT and is not retried
            self.assertEqual((second.requests, graded), ([], []))
            self.assertEqual(sorted(read_jsonl_done(out / "resolve.jsonl")), ["p1", "p2", "p3"])

    def test_resolve_kill_then_resume_generates_only_the_missing_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaises(RuntimeError):
                self.resolve_once(out, FakeStream(fail_after=1))
            self.assertEqual(sorted(read_jsonl_done(out / "resolve.jsonl")), ["p1", "p2"])
            failed = json.loads((out / "receipt-resolve.json").read_text())
            self.assertEqual((failed["completed"], failed["items_done"]), (False, 1))
            self.assertIn("killed", failed["error"])
            second = FakeStream()
            self.resolve_once(out, second)
            self.assertEqual([r["request_id"] for r in second.requests], ["p3"])
            self.assertEqual(sorted(read_jsonl_done(out / "resolve.jsonl")), ["p1", "p2", "p3"])

    def test_resolve_refuses_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stream = FakeStream()
            lessons = {pid: entry for pid, entry in self.lessons().items() if pid != "p3"}
            with self.assertRaises(ValueError):
                self.resolve_once(out, stream, lessons=lessons)
            t1 = {pid: row for pid, row in t1_rows(self.STATUSES).items() if pid != "p2"}
            with self.assertRaises(ValueError):
                self.resolve_once(out, stream, t1=t1)
            self.assertEqual(stream.requests, [])
            self.assertFalse((out / "resolve.jsonl").exists())


class GenerateIsolationTests(unittest.TestCase):
    class Service:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def grade(self, problem_id, text, finish_reason, mode, answer_contract=None):
            return SimpleNamespace(status="CORRECT", extracted="A")

    class NoLog:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def test_resolve_stage_never_opens_the_gold(self):
        chosen_from = auto_seqs(20)
        chosen = chosen_from[len(chosen_from) // 2]
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            out = tmp / "out"
            out.mkdir()
            write_order(tmp / "order.json", 3)
            for pid in ("p0", "p1"):
                append_jsonl(out / "t1.jsonl", {"problem_id": pid, "stage": "t1", "status": "INCORRECT",
                                                "text": f"my first solution {pid}", "finish_reason": "stop"})
            (out / "note.jsonl").write_bytes(b'{"problem_id": "p0"}\n{"problem_id": "p1"}\n')
            collect.write_json(out / "lessons.json", {
                "version": collect.LESSONS_VERSION, "note_bytes": (out / "note.jsonl").stat().st_size,
                "items": {pid: {"lesson": f"Check labels carefully, case {pid[1:]}.", "skip": None} for pid in ("p0", "p1")}})
            args = collect._parser().parse_args([
                "resolve", "--build-root", str(tmp / "V"), "--order", str(tmp / "order.json"), "--limit", "2",
                "--output", str(out), "--model-path", str(tmp), "--engine", "hf", "--max-num-seqs", "auto",
                "--kv-cache-gib", "20"])
            stream, stdout = FakeStream(), io.StringIO()
            with contextlib.ExitStack() as stack:
                verifier = stack.enter_context(mock.patch("fine_tuning.notebook_rl.vault.read_verifier",
                                                          side_effect=AssertionError("resolve must not read the gold")))
                stack.enter_context(mock.patch("fine_tuning.notebook_rl.vault.read_actors",
                                               return_value=list(make_actors(3).values())))
                stack.enter_context(mock.patch("fine_tuning.notebook_rl.dataset.load_images",
                                               side_effect=lambda actor, root: [object()]))
                stack.enter_context(mock.patch("fine_tuning.notebook_rl.verifier_service.VerifierService", self.Service))
                guards = stack.enter_context(mock.patch.object(collect, "install_training_guards", return_value=[]))
                stack.enter_context(mock.patch.object(collect, "_hf_stream_builder", return_value=stream))
                stack.enter_context(mock.patch.object(collect, "ResourceLog", self.NoLog))
                stack.enter_context(contextlib.redirect_stdout(stdout))
                code = collect._generate(args, "resolve")
            self.assertEqual(code, 0)
            verifier.assert_not_called()
            guards.assert_called_once_with(tmp / "V", vllm_schema=None)
            self.assertEqual(args.max_num_seqs, chosen)
            rows = read_jsonl_done(out / "resolve.jsonl")
            self.assertEqual({pid: (row["status"], row["extracted"]) for pid, row in rows.items()},
                             {"p0": ("CORRECT", "A"), "p1": ("CORRECT", "A")})  # --limit 2: p2 is outside the prefix
            self.assertEqual(sorted(r["request_id"] for r in stream.requests), ["p0", "p1"])
            receipt = json.loads((out / "receipt-resolve.json").read_text())
            self.assertEqual(receipt["max_num_seqs_auto"], {"candidates": chosen_from, "chosen": chosen})
            self.assertEqual((receipt["S"], receipt["completed"]), (chosen, True))
            self.assertEqual(json.loads(stdout.getvalue())["stage"], "resolve")


def rows_inputs():
    """Five items: T1 correct; T1 wrong + re-solve correct; wrong + hint-leak skip; wrong + re-solve wrong (T1 cut by a
    loop); blocked T1 with no note and no re-solve."""
    def note_row(pid, lesson=LESSON, image="CORRECT"):
        return {"problem_id": pid, "stage": "note", "note_version": "v2", "prompt": "NOTE PROMPT",
                "text": v2_note_text(lesson, image), "finish_reason": "stop", "lesson": lesson,
                "hint_leak": contract.hint_leak(lesson, "B", CHOICES)}

    def solve_row(pid, status, text=None, finish_reason="stop"):
        return {"problem_id": pid, "stage": "t1", "status": status, "text": text or words(40, "t"),
                "finish_reason": finish_reason}

    t1 = {"p0": solve_row("p0", "CORRECT"), "p1": solve_row("p1", "INCORRECT"), "p2": solve_row("p2", "INCORRECT"),
          "p3": solve_row("p3", "TRUNCATED", "abc " * 2000, "length"),
          "p4": {"problem_id": "p4", "stage": "t1", "status": "CONTEXT_BLOCKED", "error": "too long"}}
    notes = {"p0": note_row("p0"), "p1": note_row("p1", image="MISREAD"), "p2": note_row("p2", LEAK_LESSON),
             "p3": note_row("p3"), "p4": {"problem_id": "p4", "stage": "note", "status": "SKIPPED_CONTEXT_BLOCKED"}}
    resolve = {"p1": solve_row("p1", "CORRECT", words(40, "r")) | {"stage": "resolve"},
               "p2": {"problem_id": "p2", "stage": "resolve", "skipped": LEAK_SKIP},
               "p3": solve_row("p3", "INCORRECT", words(40, "r")) | {"stage": "resolve"}}
    return t1, notes, resolve


class RowsV2Tests(unittest.TestCase):
    def test_records_rows_and_summary(self):
        t1, notes, resolve = rows_inputs()
        items, actors = make_items(5), make_actors(5)
        records = make_records_v2(items, actors, t1, notes, resolve)
        self.assertEqual(records[0]["p0_text"], p0_eval_text("Question 0?", CHOICES))
        self.assertEqual(set(records[0]["note"]), {"prompt", "text", "finish_reason"})
        self.assertNotIn("resolve", records[0])
        self.assertEqual(records[1]["resolve"], {"text": words(40, "r"), "finish_reason": "stop", "status": "CORRECT"})
        self.assertEqual(records[2]["resolve"], {"skipped": LEAK_SKIP})
        self.assertEqual(records[4]["t1"], {"text": "", "finish_reason": "", "status": "CONTEXT_BLOCKED"})
        self.assertNotIn("note", records[4])  # a SKIPPED note row counts as no note
        self.assertNotIn("resolve", records[4])

        rows, drops, summary = collect_rows_v2(records, tok, {"t1": t1, "note": notes, "resolve": resolve})
        self.assertEqual([(r["row_id"], r["target_source"], r["route"]) for r in rows], [
            ("p0:A", "T1", "LM"), ("p0:B", "NOTE", "LM"), ("p1:A", "RESOLVE", "FULL"), ("p1:B", "NOTE", "FULL"),
            ("p2:B", "NOTE", "LM"), ("p3:B", "NOTE", "LM")])
        for row in rows:  # training prompts: the plain P0 text (never the lesson) or the note prompt
            if row["kind"] == "A":
                self.assertEqual(row["prompt"], p0_eval_text(f"Question {row['problem_id'][1:]}?", CHOICES))
                self.assertNotIn("Lesson", row["prompt"])
            else:
                self.assertEqual(row["prompt"], "NOTE PROMPT")
        self.assertEqual(sorted((d["problem_id"], d["kind"], d["reason"]) for d in drops), [
            ("p2", "A", "RESOLVE_SKIPPED:" + LEAK_SKIP), ("p3", "A", "NOT_CORRECT"), ("p4", "A", "NO_RESOLVE"),
            ("p4", "B", "NO_NOTE")])
        self.assertEqual((summary["rows_version"], summary["items"], summary["rows"], summary["drops"]), ("v2", 5, 6, 4))
        self.assertEqual(summary["rows_by_target_source"], {"T1": 1, "RESOLVE": 1, "NOTE": 4})
        self.assertEqual(summary["rows_by_kind"], {"A": 2, "B": 4})
        self.assertEqual(summary["rows_by_route"], {"LM": 4, "FULL": 2})
        self.assertEqual(summary["drops_by_reason"], {"RESOLVE_SKIPPED:" + LEAK_SKIP: 1, "NOT_CORRECT": 1,
                                                      "NO_RESOLVE": 1, "NO_NOTE": 1})
        self.assertEqual(summary["image_reading"], {"CORRECT": 3, "MISREAD": 1, "NO_NOTE": 1})
        self.assertEqual(summary["t1_status"], {"CORRECT": 1, "INCORRECT": 2, "TRUNCATED": 1, "CONTEXT_BLOCKED": 1})
        self.assertEqual(summary["resolve"], {"candidates": 4, "no_row": 1, "attempted": 2, "correct": 1,
                                              "success_rate": 0.5, "skipped_by_reason": {LEAK_SKIP: 1},
                                              "hint_leak_skips": 1})
        self.assertEqual(summary["note_hint_leak_by_reason"], {"NONE": 3, "PHRASE": 1})
        cut_loop = {"generated": 4, "length": 1, "length_loop": 1, "length_not_loop": 0, "not_loop_share": 0.0}
        none_cut = {"length": 0, "length_loop": 0, "length_not_loop": 0, "not_loop_share": None}
        self.assertEqual(summary["length_finishes"], {"t1": cut_loop, "note": {"generated": 4} | none_cut,
                                                      "resolve": {"generated": 2} | none_cut})
        self.assertNotIn("w0", json.dumps(summary))  # counts only

    def test_length_finishes_split_loops_from_the_rest(self):
        loop = {"text": "abc " * 2000, "finish_reason": "length"}
        varied = {"text": words(800), "finish_reason": "length"}
        done = {"text": words(50), "finish_reason": "stop"}
        self.assertEqual(length_report({"a": loop, "b": varied, "c": done, "d": {"status": "CONTEXT_BLOCKED"}}),
                         {"generated": 3, "length": 2, "length_loop": 1, "length_not_loop": 1, "not_loop_share": 0.5})
        self.assertEqual(length_report({}), {"generated": 0, "length": 0, "length_loop": 0, "length_not_loop": 0,
                                            "not_loop_share": None})
        t1, notes, resolve = rows_inputs()
        notes["extra"] = {"problem_id": "extra", "stage": "note", "note_version": "v2", "lesson": None, "hint_leak": None,
                          **varied}  # a cut note that is not a loop
        records = make_records_v2(make_items(5), make_actors(5), t1, {k: v for k, v in notes.items() if k != "extra"}, resolve)
        _, _, summary = collect_rows_v2(records, tok, {"t1": t1, "note": notes, "resolve": resolve})
        self.assertEqual(summary["length_finishes"]["t1"],
                         {"generated": 4, "length": 1, "length_loop": 1, "length_not_loop": 0, "not_loop_share": 0.0})
        self.assertEqual(summary["length_finishes"]["note"],
                         {"generated": 5, "length": 1, "length_loop": 0, "length_not_loop": 1, "not_loop_share": 1.0})

    def test_records_refuse_v1_notes_and_missing_t1(self):
        t1, notes, resolve = rows_inputs()
        v1_note = {"problem_id": "p0", "stage": "note", "prompt": "x", "text": v2_note_text(), "finish_reason": "stop",
                   "corrected_status": "CORRECT"}
        with self.assertRaises(ValueError):
            make_records_v2(make_items(5), make_actors(5), t1, notes | {"p0": v1_note}, resolve)
        with self.assertRaises(ValueError):
            make_records_v2(make_items(5), make_actors(5), {}, {}, {})


class CliV2Tests(unittest.TestCase):
    BASE = ["--build-root", "b", "--order", "o", "--limit", "1", "--output", "out"]
    ENGINE = ["--model-path", "m", "--engine", "hf"]

    def test_resolve_options(self):
        parser = collect._parser()
        args = parser.parse_args(["resolve", *self.BASE, *self.ENGINE, "--max-num-seqs", "auto"])
        self.assertEqual((args.command, args.max_num_seqs, args.max_num_seqs_from), ("resolve", "auto", None))
        args = parser.parse_args(["resolve", *self.BASE, *self.ENGINE, "--max-num-seqs-from", "sweep.json"])
        self.assertEqual((args.max_num_seqs, args.max_num_seqs_from), (32, Path("sweep.json")))
        self.assertEqual(parser.parse_args(["resolve", *self.BASE, *self.ENGINE, "--max-num-seqs", "48"]).max_num_seqs, 48)

    def test_note_and_rows_versions(self):
        parser = collect._parser()
        self.assertEqual(parser.parse_args(["note", *self.BASE, *self.ENGINE]).note_version, "v1")
        args = parser.parse_args(["note", *self.BASE, *self.ENGINE, "--note-version", "v2", "--max-num-seqs", "auto",
                                  "--max-num-seqs-from", "sweep.json"])
        self.assertEqual((args.note_version, args.max_num_seqs, args.max_num_seqs_from), ("v2", "auto", Path("sweep.json")))
        rows = ["rows", *self.BASE, "--tokenizer-path", "t"]
        self.assertEqual(parser.parse_args(rows).rows_version, "v1")
        self.assertEqual(parser.parse_args([*rows, "--rows-version", "v2"]).rows_version, "v2")

    def test_bad_values_exit(self):
        parser = collect._parser()
        for argv in (["resolve", *self.BASE, *self.ENGINE, "--max-num-seqs", "0"],
                     ["resolve", *self.BASE, *self.ENGINE, "--max-num-seqs", "fast"],
                     ["note", *self.BASE, *self.ENGINE, "--note-version", "v3"],
                     ["rows", *self.BASE, "--tokenizer-path", "t", "--rows-version", "v3"]):
            with self.subTest(argv=argv[-2:]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
