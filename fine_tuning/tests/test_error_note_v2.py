"""Authored CPU cases for the error-note version-2 contract: note parsing, hint-leak check, re-solve prompt, seeds,
loop check, training rows, spread order and the committed v2 order.

Fake tokenizer and temporary files only: no model, GPU, network or benchmark data."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from fine_tuning.error_note import order
from fine_tuning.error_note.contract import (
    RESOLVE_HEADER, ROUTE_FULL, ROUTE_LM, build_rows_v2, hint_leak, is_loop, parse_note, parse_note_v2, resolve_prompt,
    sample_seed, spread_order)

ROOT = Path(__file__).resolve().parents[2]
ORDER_DIR = ROOT / "fine_tuning" / "error_note"

# A real version-1 note from the HF training items (kept verbatim).
V1_NOTE = """What the image shows: The graph contains node 4 and node 0, and there is a direct edge (line) connecting them. This is visible in the leftmost diagram.
Image reading: CORRECT
Cause: NONE
What went wrong: None — the observation was accurate and directly supported by the visual structure of the graph.
How to solve it: In an undirected graph, if there is a direct edge between two nodes, then there is a path (in fact, the shortest path) between them. Simply check for the existence of an edge between the two specified nodes.
Corrected solution: Since there is a direct edge between node 4 and node 0, there is a path between them. Final answer: A"""


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def tok(text):
    """Whitespace tokenizer with distinct ids, so the repetition check never fires on ordinary word text."""
    return list(range(len(text.split())))


def solution(prefix="s"):
    """40 tokens ending with the answer B."""
    return words(37, prefix) + " Final answer: B"


def attempt(text=None, finish="stop", status="CORRECT"):
    return {"text": solution() if text is None else text, "finish_reason": finish, "status": status}


def note_v2(image="CORRECT", cause="NONE", lesson="Recheck the labels before counting.", shows=None):
    shows = shows if shows is not None else "What the image shows: " + words(40)
    lines = [shows, f"Image reading: {image}", f"Cause: {cause}", "What went wrong: nothing", f"Lesson: {lesson}"]
    return "\n".join(lines)


def note_entry(text=None, finish="stop", image="CORRECT"):
    return {"prompt": "NOTE PROMPT", "text": note_v2(image=image) if text is None else text, "finish_reason": finish}


def record(**changes):
    """A collected item as build_rows_v2 reads it: T1 correct and a correct v2 note; no resolve key unless given."""
    rec = {"problem_id": "p1", "subject": "Biology", "source": "src", "p0_text": "P0 TEXT", "image_paths": ["a.png"],
           "t1": attempt(), "note": note_entry()}
    rec.update(changes)
    return rec


class ParseNoteV2Tests(unittest.TestCase):
    def test_good_note_fields(self):
        text = note_v2(lesson="Count each bar first.")
        parsed = parse_note_v2(text)
        self.assertEqual((parsed.image_reading, parsed.cause, parsed.lesson), ("CORRECT", "NONE", "Count each bar first."))
        self.assertEqual(parsed.note_text, text.strip())

    def test_markdown_headings_parse(self):
        text = "\n".join(["**What the image shows:** " + words(40), "## Image reading: MISREAD", "Cause: **REASONING**",
                         "**What went wrong**: a label", "### **Lesson:** Read the axis first."])
        parsed = parse_note_v2(text)
        self.assertEqual((parsed.image_reading, parsed.cause, parsed.lesson),
                         ("MISREAD", "REASONING", "Read the axis first."))

    def test_rejects_missing_heading(self):
        lines = ("Image reading: CORRECT", "Cause: NONE", "What went wrong: nothing", "Lesson: Recheck the labels before counting.")
        for line in lines:
            with self.subTest(line=line):
                self.assertIsNone(parse_note_v2(note_v2().replace(line, "")))

    def test_rejects_heading_twice_and_out_of_order(self):
        self.assertIsNone(parse_note_v2(note_v2() + "\nCause: NONE"))
        swapped = note_v2().replace("Image reading: CORRECT\nCause: NONE", "Cause: NONE\nImage reading: CORRECT")
        self.assertIsNone(parse_note_v2(swapped))

    def test_rejects_empty_body_and_unknown_enum(self):
        self.assertIsNone(parse_note_v2(note_v2(lesson="")))
        self.assertIsNone(parse_note_v2(note_v2().replace("What went wrong: nothing", "What went wrong:")))
        self.assertIsNone(parse_note_v2(note_v2(image="MAYBE")))
        self.assertIsNone(parse_note_v2(note_v2(cause="OTHER")))


class RealV1NoteRegressionTests(unittest.TestCase):
    def test_v1_note_parses_as_v1_and_not_as_v2(self):
        parsed = parse_note(V1_NOTE)
        self.assertEqual((parsed.image_reading, parsed.cause), ("CORRECT", "NONE"))
        self.assertIsNone(parse_note_v2(V1_NOTE))

    def test_v2_form_of_v1_note_parses_and_passes_hint_check(self):
        lines = [line.replace("How to solve it:", "Lesson:") for line in V1_NOTE.splitlines()
                 if not line.startswith("Corrected solution:")]
        parsed = parse_note_v2("\n".join(lines))
        self.assertEqual((parsed.image_reading, parsed.cause), ("CORRECT", "NONE"))
        self.assertTrue(parsed.lesson.startswith("In an undirected graph"))
        self.assertIsNone(hint_leak(parsed.lesson, "A", ("Yes", "No")))


class HintLeakTests(unittest.TestCase):
    def test_true_cases_return_the_reason(self):
        cases = [
            ("Option B is right when the slope is negative.", "A", ("Yes", "No"), "OPTION_LETTER:B"),
            ("So the answer is C.", "A", ("Yes", "No"), "OPTION_LETTER:C"),
            ("The answer is the largest value.", "A", ("Yes", "No"), "PHRASE:the answer is"),
            ("Find the correct answer by elimination.", "A", ("Yes", "No"), "PHRASE:correct answer"),
            ("Compare (c) with the legend.", "A", ("Yes", "No"), "PAREN_LETTER:C"),
            ("Look for MARYLAND   on the map.", "**D. Maryland**", ("Ohio", "Texas", "Utah", "Maryland"), "GOLD_OPTION_TEXT"),
            ("Check the state named texas on the map.", "B", ("Ohio", "Texas"), "GOLD_OPTION_TEXT"),
            ("Divide the total by 12.5 first.", "12.5", (), "GOLD_TEXT"),
            ("Count the 2 items carefully.", "2", (), "GOLD_TEXT"),
            # Known false-positive bias: "a" is an article here, but the rule flags it on purpose.
            ("Express the answer as a fraction.", "A", ("Yes", "No"), "OPTION_LETTER:A"),
            # Short gold options are checked as words like a short open gold (audit 2026-11-02): before, an option
            # under 4 characters (Yes/No, time signatures, keys: 29% of the HF MC items) was never checked.
            ("If no shared node links them, then no path exists.", "B", ("Yes", "No"), "GOLD_OPTION_TEXT"),
            ("Yes means the edge exists, so check edges.", "A", ("Yes", "No"), "GOLD_OPTION_TEXT"),
            ("Check whether 2 lines cross.", "A", ("2", "3")),
            ("Four quarter notes per bar is 4/4 time.", "C", ("2/4", "3/4", "4/4")),
            # Case, spacing and punctuation of the option are ignored.
            ("Confirm whether it makes its own food (photosynthesis).", "A", ("It makes its own food.", "It eats")),
            # The gold letter on its own (audit 2026-11-02).
            ("Pick C when the curve rises.", "C", ("x", "y", "z"), "GOLD_LETTER:C"),
            ("Check the slope first. Therefore, D.", "D", ("w", "x", "y", "z"), "GOLD_LETTER:D"),
            ("Check the slope first, so the result is A.", "A", ("w", "x"), "GOLD_LETTER:A"),
        ]
        cases = [case if len(case) == 4 else (*case, "GOLD_OPTION_TEXT") for case in cases]
        for lesson, gold, choices, reason in cases:
            with self.subTest(lesson=lesson):
                self.assertEqual(hint_leak(lesson, gold, choices), reason)

    def test_false_cases_return_none(self):
        cases = [
            ("Count the beats of each note in the measure.", "B", ("Yes", "No")),
            ("Count 4 beats in each 4-beat bar.", "C", ("2/4", "3/4", "4/4")),
            ("Count 22 items carefully.", "2", ()),
            ("Count 22 items carefully.", "A", ("2", "3")),
            ("Read each bar against the gridlines before computing differences.", "A", ("10", "20", "30", "40")),
            ("In F# major the key signature has six sharps.", "F", ("q", "r", "s", "t", "u", "v")),
            ("A careful reading of the axis labels helps. A: start with the units.", "A", ("w", "x")),
            ("Let b be the base length and Fe the iron content.", "B", ("w", "x")),
        ]
        for lesson, gold, choices in cases:
            with self.subTest(lesson=lesson):
                self.assertIsNone(hint_leak(lesson, gold, choices))


class ResolvePromptTests(unittest.TestCase):
    def test_lesson_block_then_plain_p0_text(self):
        prompt = resolve_prompt("P0 TEXT", "Count each bar first.")
        self.assertEqual(prompt, RESOLVE_HEADER + "Count each bar first." + "\n\n" + "P0 TEXT")
        for leaked in ("SECRETGOLD", "Grading result", "Correct answer"):
            self.assertNotIn(leaked, prompt)

    def test_blank_inputs_raise(self):
        with self.assertRaises(ValueError):
            resolve_prompt("  ", "lesson")
        with self.assertRaises(ValueError):
            resolve_prompt("P0", "  \n")


class SampleSeedStageTests(unittest.TestCase):
    def test_stage_keys(self):
        cases = (("resolve", "3407:p-9:resolve"), ("t1", "3407:p-9"), ("note", "3407:p-9:note"))
        for stage, key in cases:
            with self.subTest(stage=stage):
                expected = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 2**31
                self.assertEqual(sample_seed("p-9", stage), expected)

    def test_unknown_stage_raises(self):
        with self.assertRaises(ValueError):
            sample_seed("p-9", "solve")


class IsLoopTests(unittest.TestCase):
    def test_repeated_tail_is_a_loop(self):
        self.assertTrue(is_loop("abc " * 2000))
        varied = " ".join(str(i * 7919 % 10007) for i in range(1500))
        self.assertTrue(is_loop(varied + " " + "This sentence repeats. " * 200))

    def test_short_and_varied_text_is_not_a_loop(self):
        self.assertFalse(is_loop("short"))
        self.assertFalse(is_loop(" ".join(str(i * 7919 % 10007) for i in range(1500))))


class BuildRowsV2Tests(unittest.TestCase):
    def test_correct_t1_and_correct_note_give_t1_row_and_note_row(self):
        rec = record()
        rows, drops = build_rows_v2(rec, tok)
        self.assertEqual(drops, [])
        a, b = rows
        self.assertEqual((a["kind"], a["target_source"], a["route"], a["prompt"], a["target_tokens"]),
                         ("A", "T1", ROUTE_LM, "P0 TEXT", 40))
        self.assertEqual(a["target"], rec["t1"]["text"])
        self.assertEqual((b["kind"], b["target_source"], b["route"], b["prompt"]), ("B", "NOTE", ROUTE_LM, "NOTE PROMPT"))
        self.assertEqual(b["target"], rec["note"]["text"].strip())

    def test_misread_note_drops_t1_row_and_routes_note_full(self):
        rows, drops = build_rows_v2(record(note=note_entry(image="MISREAD")), tok)
        self.assertEqual(drops, [{"problem_id": "p1", "kind": "A", "reason": "T1_SELF_MISREAD"}])
        self.assertEqual([(row["kind"], row["route"]) for row in rows], [("B", ROUTE_FULL)])

    def test_missing_or_unfinished_or_unparsed_note_drops_note_row_only(self):
        cases = [
            (record(note=None), "NO_NOTE"),
            (record(note=note_entry(finish="length")), "NOTE_NOT_STOPPED"),
            (record(note=note_entry(text="no headings here")), "NOTE_UNPARSED"),
        ]
        for rec, reason in cases:
            with self.subTest(reason=reason):
                rows, drops = build_rows_v2(rec, tok)
                self.assertEqual([(row["kind"], row["route"]) for row in rows], [("A", ROUTE_LM)])
                self.assertEqual(drops, [{"problem_id": "p1", "kind": "B", "reason": reason}])

    def test_incorrect_t1_uses_resolve_with_plain_p0_prompt(self):
        rec = record(t1=attempt(status="INCORRECT"), resolve=attempt(solution("r"), status="CORRECT"))
        rows, drops = build_rows_v2(rec, tok)
        self.assertEqual(drops, [])
        a = rows[0]
        self.assertEqual((a["kind"], a["target_source"]), ("A", "RESOLVE"))
        self.assertEqual(a["prompt"], "P0 TEXT")
        self.assertEqual(a["target"], solution("r"))

    def test_resolve_leak_phrases_drop_solution_row(self):
        cases = [
            ("As the lesson from before says, " + solution("r"), "LEAK:lesson"),
            ("Using this lesson, " + solution("r"), "LEAK:lesson"),  # the plan's "lesson" class, not only "the lesson"
            ("Recheck the previous attempt. " + solution("r"), "LEAK:previous attempt"),
        ]
        for text, reason in cases:
            with self.subTest(reason=reason):
                rec = record(t1=attempt(status="INCORRECT"), resolve=attempt(text))
                rows, drops = build_rows_v2(rec, tok)
                self.assertEqual([row["kind"] for row in rows], ["B"])
                self.assertEqual(drops, [{"problem_id": "p1", "kind": "A", "reason": reason}])

    def test_missing_skipped_or_failed_resolve_drops_solution_row(self):
        cases = [
            ({}, "NO_RESOLVE"),
            ({"resolve": {"skipped": "HINT_LEAK:PHRASE:correct answer"}},
             "RESOLVE_SKIPPED:HINT_LEAK:PHRASE:correct answer"),
            ({"resolve": attempt(status="INCORRECT")}, "NOT_CORRECT"),
            ({"resolve": attempt(finish="length")}, "NOT_STOPPED"),
        ]
        for extra, reason in cases:
            with self.subTest(reason=reason):
                rows, drops = build_rows_v2(record(t1=attempt(status="INCORRECT"), **extra), tok)
                self.assertEqual([row["kind"] for row in rows], ["B"])
                self.assertEqual(drops, [{"problem_id": "p1", "kind": "A", "reason": reason}])

    def test_looping_solution_drops_solution_row(self):
        rows, drops = build_rows_v2(record(t1=attempt("loop text " * 400)), tok)
        self.assertEqual(drops, [{"problem_id": "p1", "kind": "A", "reason": "LOOP"}])
        self.assertEqual([row["kind"] for row in rows], ["B"])

    def test_short_note_and_looping_note_drop_note_row(self):
        short = note_entry(text=note_v2(shows="What the image shows: few words"))
        looping = note_entry(text=note_v2(lesson="Always recheck the labels. " * 200))
        for note, reason in ((short, "NOTE_EMPTY"), (looping, "NOTE_LOOP")):
            with self.subTest(reason=reason):
                rows, drops = build_rows_v2(record(note=note), tok)
                self.assertEqual([row["kind"] for row in rows], ["A"])
                self.assertEqual(drops, [{"problem_id": "p1", "kind": "B", "reason": reason}])

    def test_no_row_is_ever_corrected(self):
        records = [
            record(),
            record(t1=attempt(status="INCORRECT"), resolve=attempt(solution("r"), status="CORRECT")),
            record(note=note_entry(image="MISREAD")),
            record(t1=attempt(status="INCORRECT")),
        ]
        for rec in records:
            rows, _ = build_rows_v2(rec, tok)
            for row in rows:
                self.assertIn(row["target_source"], ("T1", "RESOLVE", "NOTE"))


class SpreadOrderTests(unittest.TestCase):
    def items(self):
        sources = ["x", "x", "x", "x", "y", "y"]
        return [{"problem_id": pid, "source": src} for pid, src in zip("abcdef", sources)]

    def test_spread_matches_hand_computed_order(self):
        self.assertEqual(spread_order(self.items(), list("abcdef")), list("aebcfd"))

    def test_repeatable_and_base_order_kept_within_each_source(self):
        first = spread_order(self.items(), list("abcdef"))
        self.assertEqual(first, spread_order(self.items(), list("abcdef")))
        self.assertEqual([p for p in first if p in {"a", "b", "c", "d"}], list("abcd"))
        self.assertEqual([p for p in first if p in {"e", "f"}], list("ef"))

    def test_duplicate_or_mismatched_ids_raise(self):
        for base in ("abcdea", "abcde", "abcdeg"):
            with self.subTest(base=base):
                with self.assertRaises(ValueError):
                    spread_order(self.items(), list(base))


class OrderSpreadCliTests(unittest.TestCase):
    def test_spread_writes_v2_order_and_refuses_overwrite(self):
        ids = [f"p{i}" for i in range(6)]
        sources = ["x", "x", "x", "x", "y", "y"]
        ids_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        base_items = [{"problem_id": pid, "subject": "S", "source": src} for pid, src in zip(ids, sources)]
        with tempfile.TemporaryDirectory() as tmp:
            base, out = Path(tmp) / "base.json", Path(tmp) / "out.json"
            base.write_text(json.dumps({"version": "error-note-order-v1", "master_seed": 3407, "count": 6,
                                        "items": base_items, "ids_sha256": ids_sha}))
            argv = ["spread", "--base", str(base), "--output", str(out)]
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(order.main(argv), 0)
            self.assertIn("count=6 ids_sha256=", stdout.getvalue())
            result = order.read_order(out)
            new_ids = [item["problem_id"] for item in result["items"]]
            self.assertEqual(result["version"], "error-note-order-v2")
            self.assertEqual((result["count"], sorted(new_ids)), (6, ids))
            self.assertEqual(new_ids, ["p0", "p4", "p1", "p2", "p5", "p3"])
            self.assertEqual(result["ids_sha256"], hashlib.sha256("\n".join(new_ids).encode()).hexdigest())
            self.assertEqual(result["base_order"]["sha256"], hashlib.sha256(base.read_bytes()).hexdigest())
            self.assertEqual(result["base_order"]["ids_sha256"], ids_sha)
            with self.assertRaises(SystemExit):
                order.main(argv)


class CommittedOrderV2Tests(unittest.TestCase):
    def test_committed_v2_is_the_spread_of_the_v1_order(self):
        v1_path = ORDER_DIR / "item_order.json"
        v1 = order.read_order(v1_path)
        v2 = order.read_order(ORDER_DIR / "item_order_v2.json")
        self.assertEqual((v2["version"], v2["count"]), ("error-note-order-v2", 3417))
        self.assertEqual({item["problem_id"] for item in v2["items"]}, {item["problem_id"] for item in v1["items"]})
        expected = order.build_v2(v1, hashlib.sha256(v1_path.read_bytes()).hexdigest())
        self.assertEqual(v2["items"], expected["items"])


if __name__ == "__main__":
    unittest.main()
