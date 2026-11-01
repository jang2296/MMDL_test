"""Authored cases for the error-note contract and the pinned item order; no benchmark data, weights, or network."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from fine_tuning.error_note import contract, order
from fine_tuning.error_note.contract import (
    NOTE_INSTRUCTION, ParsedNote, build_rows, has_repetition, leaked_phrase, note_prompt, order_key, parse_note,
    round_robin_order, sample_seed, select_rows, update_groups, verdict_label)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def tok(text):
    """Whitespace tokenizer with unique ids, so repetition never triggers by accident; REPEAT-prefixed text always repeats."""
    count = len(text.split())
    return [7] * count if text.startswith("REPEAT") else list(range(count))


def note_text(image="Image reading: CORRECT", cause="Cause: NONE", corrected="Corrected solution:\nStep 1. Final answer: B",
              shows=None, wrong="What went wrong: nothing", how="How to solve it: read the labels"):
    shows = shows if shows is not None else "What the image shows: " + words(40)
    return "\n".join([shows, image, cause, wrong, how, corrected])


def record(*, t1_text=None, t1_status="CORRECT", t1_finish="stop", note="default", corrected=None, corrected_status="CORRECT",
           note_finish="stop", image="Image reading: CORRECT"):
    corrected = corrected if corrected is not None else words(40, "c")
    if note == "default":
        note = {"prompt": "NOTE PROMPT", "text": note_text(image=image, corrected="Corrected solution:\n" + corrected),
                "finish_reason": note_finish, "corrected_status": corrected_status}
    rec = {"problem_id": "p1", "subject": "Biology", "source": "src", "p0_text": "P0 TEXT", "image_paths": ["a.png"],
           "t1": {"text": t1_text if t1_text is not None else words(40, "t"), "finish_reason": t1_finish, "status": t1_status}}
    if note is not None:
        rec["note"] = note
    return rec


class SampleSeedTests(unittest.TestCase):
    def test_t1_matches_mmmu_rule(self):
        for problem_id in ("a-1", "hf-x:y:train:000001:abc"):
            value = hashlib.sha256(f"3407:{problem_id}".encode()).digest()
            self.assertEqual(sample_seed(problem_id, "t1"), int.from_bytes(value[:8], "big") % 2**31)

    def test_note_stage_differs_and_unknown_raises(self):
        value = hashlib.sha256(b"3407:a-1:note").digest()
        self.assertEqual(sample_seed("a-1", "note"), int.from_bytes(value[:8], "big") % 2**31)
        self.assertNotEqual(sample_seed("a-1", "note"), sample_seed("a-1", "t1"))
        self.assertEqual(sample_seed("a-1", "t1", 5), int.from_bytes(hashlib.sha256(b"5:a-1").digest()[:8], "big") % 2**31)
        with self.assertRaises(ValueError):
            sample_seed("a-1", "other")


class PromptTests(unittest.TestCase):
    def test_verdict_labels(self):
        self.assertEqual([verdict_label(s) for s in ("CORRECT", "INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED")],
                         ["CORRECT", "WRONG", "NO CLEAR FINAL ANSWER", "NO CLEAR FINAL ANSWER", "CUT OFF"])
        for status in ("UNVERIFIABLE", "bogus"):
            with self.assertRaises(ValueError):
                verdict_label(status)

    def test_exact_bytes_and_whitespace_kept(self):
        self.assertEqual(note_prompt("Q?", " step 1 \n", "INCORRECT", "B"),
                         "Q?\n\nYour earlier solution:\n step 1 \n\n\nGrading result: WRONG\nCorrect answer: B\n\n" + NOTE_INSTRUCTION)
        self.assertIn("\n\nYour earlier solution:\n\n\nGrading result: CUT OFF\n", note_prompt("Q?", "", "TRUNCATED", "B"))

    def test_blank_inputs_raise(self):
        for args in (("  ", "t", "CORRECT", "B"), ("Q", "t", "CORRECT", " "), ("Q", "t", "UNVERIFIABLE", "B")):
            with self.assertRaises(ValueError):
                note_prompt(*args)


class ParseNoteTests(unittest.TestCase):
    def test_plain_note(self):
        parsed = parse_note(note_text())
        self.assertEqual((parsed.image_reading, parsed.cause, parsed.corrected_text), ("CORRECT", "NONE", "Step 1. Final answer: B"))
        self.assertTrue(parsed.note_text.startswith("What the image shows:"))
        self.assertTrue(parsed.note_text.endswith("How to solve it: read the labels"))
        self.assertNotIn("Corrected solution", parsed.note_text)

    def test_markdown_and_punctuation(self):
        parsed = parse_note(note_text(image="**Image reading:** MISREAD", cause="## Cause: REASONING",
                                      wrong="**What went wrong**: a label", how="### **How to solve it:** read",
                                      corrected="**Corrected solution:** Final answer: B"))
        self.assertEqual((parsed.image_reading, parsed.cause, parsed.corrected_text), ("MISREAD", "REASONING", "Final answer: B"))
        parsed = parse_note(note_text(image="Image reading: CORRECT.", cause="Cause: **NO_FINAL_ANSWER**"))
        self.assertEqual((parsed.image_reading, parsed.cause), ("CORRECT", "NO_FINAL_ANSWER"))
        self.assertEqual(parse_note(note_text(image="Image reading: UNSURE, because")).image_reading, "UNSURE")

    def test_same_line_body_keeps_leading_bold(self):
        self.assertEqual(parse_note(note_text(corrected="Corrected solution: **Step 1** Final answer: B")).corrected_text,
                         "**Step 1** Final answer: B")

    def test_rejects_missing_heading(self):
        for drop in ("How to solve it: read the labels", "Cause: NONE", "What went wrong: nothing"):
            self.assertIsNone(parse_note(note_text().replace(drop, "")))

    def test_rejects_out_of_order_and_duplicate(self):
        self.assertIsNone(parse_note(note_text(image="Cause: NONE", cause="Image reading: CORRECT")))
        self.assertIsNone(parse_note(note_text() + "\nCause: NONE"))
        self.assertIsNone(parse_note(note_text(corrected="Corrected solution: a\nCorrected solution: b")))

    def test_rejects_unknown_enum_and_case(self):
        for kwargs in ({"image": "Image reading: MAYBE"}, {"cause": "Cause: OTHER"}, {"image": "Image reading:"},
                       {"image": "Image reading: misread"}, {"image": "image reading: CORRECT"}):
            with self.subTest(kwargs=kwargs):
                self.assertIsNone(parse_note(note_text(**kwargs)))

    def test_rejects_empty_sections(self):
        for corrected in ("Corrected solution:", "Corrected solution:   \n  ", "**Corrected solution:**"):
            self.assertIsNone(parse_note(note_text(corrected=corrected)))
        self.assertIsNone(parse_note(note_text(wrong="What went wrong:")))
        self.assertIsNone(parse_note(note_text(shows="What the image shows:  ")))

    def test_heading_must_start_a_line(self):
        self.assertIsNone(parse_note(note_text(cause="I think the Cause: NONE here")))
        text = note_text(wrong="What went wrong: nothing, see Cause: above and Corrected solution: below")
        self.assertEqual(parse_note(text).cause, "NONE")
        self.assertIsNone(parse_note("not a note"))


class RepetitionAndLeakTests(unittest.TestCase):
    def test_window_and_count_boundaries(self):
        self.assertTrue(has_repetition([1, 2, 3, 1, 2, 3], window=3, count=2))
        self.assertFalse(has_repetition([1, 2, 3, 1, 2, 4], window=3, count=2))
        self.assertFalse(has_repetition([1, 2], window=3, count=1))
        self.assertTrue(has_repetition([1, 2, 3], window=3, count=1))
        run = list(range(100, 112))
        self.assertFalse(has_repetition(run + [1] + run + [2]))
        self.assertTrue(has_repetition(run + [1] + run + [2] + run))

    def test_overlapping_run(self):
        self.assertFalse(has_repetition([5] * 13))
        self.assertTrue(has_repetition([5] * 14))
        self.assertFalse(has_repetition([5] * 11))

    def test_leaked_phrase(self):
        self.assertEqual(leaked_phrase("As in the Earlier Attempt, x"), "earlier attempt")
        self.assertEqual(leaked_phrase("I was told so"), "was told")
        for text in ("the correct answer is B", "note that x", "grade 5", ""):
            self.assertIsNone(leaked_phrase(text))


class BuildRowsTests(unittest.TestCase):
    def kinds(self, rows):
        return {row["kind"]: row for row in rows}

    def test_t1_correct_and_image_correct_uses_t1(self):
        rec = record(t1_text=" " + words(40, "t") + " ")
        rows, drops = build_rows(rec, tok)
        by = self.kinds(rows)
        self.assertEqual(drops, [])
        self.assertEqual(by["A"]["target"], rec["t1"]["text"])
        self.assertEqual((by["A"]["target_source"], by["A"]["route"], by["A"]["prompt"], by["A"]["row_id"]),
                         ("T1", "LM", "P0 TEXT", "p1:A"))
        self.assertEqual((by["B"]["target_source"], by["B"]["prompt"], by["B"]["route"], by["B"]["row_id"]), ("NOTE", "NOTE PROMPT", "LM", "p1:B"))
        self.assertEqual(by["B"]["target"], parse_note(rec["note"]["text"]).note_text)
        self.assertEqual((by["A"]["target_tokens"], by["B"]["target_tokens"]), (40, len(tok(by["B"]["target"]))))
        for row in rows:
            self.assertEqual((row["problem_id"], row["subject"], row["source"], row["image_paths"]), ("p1", "Biology", "src", ["a.png"]))

    def test_misread_uses_corrected_and_full_route(self):
        rows, drops = build_rows(record(image="Image reading: MISREAD"), tok)
        by = self.kinds(rows)
        self.assertEqual(drops, [])
        self.assertEqual((by["A"]["target_source"], by["A"]["route"], by["B"]["route"]), ("CORRECTED", "FULL", "FULL"))
        self.assertEqual(by["A"]["target"], words(40, "c"))

    def test_t1_wrong_uses_corrected(self):
        by = self.kinds(build_rows(record(t1_status="INCORRECT"), tok)[0])
        self.assertEqual((by["A"]["target_source"], by["A"]["route"]), ("CORRECTED", "LM"))

    def test_short_or_unstopped_t1_falls_to_corrected(self):
        for kwargs in ({"t1_text": words(31)}, {"t1_finish": "length"}):
            with self.subTest(kwargs=kwargs):
                by = self.kinds(build_rows(record(**kwargs), tok)[0])
                self.assertEqual(by["A"]["target_source"], "CORRECTED")

    def test_corrected_problems_drop_only_solution_row(self):
        cases = [({"corrected_status": "INCORRECT"}, "NOT_CORRECT"), ({"corrected": words(5, "c")}, "TOO_SHORT"),
                 ({"corrected": words(4097, "c")}, "TOO_LONG"), ({"corrected": "REPEAT " + words(40, "c")}, "REPETITION"),
                 ({"corrected": words(40, "c") + " the Earlier Attempt said"}, "LEAK:earlier attempt")]
        for kwargs, reason in cases:
            with self.subTest(reason=reason):
                rows, drops = build_rows(record(t1_status="INCORRECT", **kwargs), tok)
                self.assertEqual([r["kind"] for r in rows], ["B"])
                self.assertEqual(drops, [{"problem_id": "p1", "kind": "A", "reason": reason}])

    def test_corrected_target_is_not_replaced_by_t1_when_it_fails(self):
        rows, drops = build_rows(record(corrected_status="INCORRECT"), tok)  # T1 correct + image CORRECT: T1 passes first
        self.assertEqual(self.kinds(rows)["A"]["target_source"], "T1")
        self.assertEqual(drops, [])

    def test_note_absent_unstopped_or_unparsed_drops_both(self):
        bad_text = {"prompt": "x", "text": "no headings", "finish_reason": "stop", "corrected_status": "CORRECT"}
        for rec, reason in ((record(note=None), "NO_NOTE"), (record(note_finish="length"), "NOTE_NOT_STOPPED"),
                            (record(note=bad_text), "NOTE_UNPARSED")):
            with self.subTest(reason=reason):
                rows, drops = build_rows(rec, tok)
                self.assertEqual(rows, [])
                self.assertEqual(drops, [{"problem_id": "p1", "kind": k, "reason": reason} for k in ("A", "B")])

    def test_short_note_drops_note_row_only(self):
        rec = record()
        rec["note"]["text"] = note_text(shows="What the image shows: few words", corrected="Corrected solution:\n" + words(40, "c"))
        rows, drops = build_rows(rec, tok)
        self.assertEqual([r["kind"] for r in rows], ["A"])
        self.assertEqual(drops, [{"problem_id": "p1", "kind": "B", "reason": "NOTE_EMPTY"}])


class GroupAndArmTests(unittest.TestCase):
    def test_update_groups(self):
        ids = [f"i{n}" for n in range(300)]
        groups = update_groups(ids)
        self.assertEqual((len(groups), len(groups[-1])), (38, 4))
        self.assertEqual([x for g in groups for x in g], ids)
        self.assertEqual(update_groups([], 8), [])
        with self.assertRaises(ValueError):
            update_groups(ids, 0)

    def test_select_rows(self):
        rows = [{"kind": "A", "route": "FULL", "row_id": "1:A"}, {"kind": "B", "route": "FULL", "row_id": "1:B"}]
        before = [dict(r) for r in rows]
        solution = select_rows(rows, "solution")
        self.assertEqual(solution, [{"kind": "A", "route": "LM", "row_id": "1:A"}])
        self.assertEqual(rows, before)
        self.assertEqual(select_rows(rows, "note"), rows)
        with self.assertRaises(ValueError):
            select_rows(rows, "other")


def item(n, subject, structure=None):
    return {"problem_id": f"id{n}", "subject": subject, "structure_hash": structure or f"s{n}"}


class RoundRobinTests(unittest.TestCase):
    def setUp(self):
        self.items = [item(n, "A") for n in range(10)] + [item(n, "B") for n in range(10, 20)] + [item(n, "C") for n in range(20, 22)]

    def test_deterministic_and_complete(self):
        first = round_robin_order(self.items)
        self.assertEqual(first, round_robin_order(list(reversed(self.items))))
        self.assertEqual(sorted(first), sorted(i["problem_id"] for i in self.items))

    def test_subjects_interleave_evenly(self):
        out = round_robin_order(self.items)
        subject = {i["problem_id"]: i["subject"] for i in self.items}
        self.assertEqual([subject[i] for i in out[:6]], ["A", "B", "C", "A", "B", "C"])
        for n in range(1, 7):
            counts = [sum(subject[i] == s for i in out[:n]) for s in "ABC"]
            self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual({subject[i] for i in out[6:]}, {"A", "B"})
        in_subject = [i for i in out if subject[i] == "A"]
        self.assertEqual(in_subject, sorted(in_subject, key=order_key))

    def test_later_duplicate_structure_is_dropped(self):
        items = [item(1, "A", "same"), item(2, "B", "same"), item(3, "A")]
        out = round_robin_order(items)
        keep = min(("id1", "id2"), key=order_key)
        self.assertEqual(sorted(out), sorted([keep, "id3"]))

    def test_duplicate_problem_id_raises(self):
        with self.assertRaises(ValueError):
            round_robin_order([item(1, "A"), {"problem_id": "id1", "subject": "B", "structure_hash": "other"}])

    def test_order_key(self):
        self.assertEqual(order_key("x"), hashlib.sha256(b"3407:x").hexdigest())


def fixtures():
    ids = ["a1", "a2", "b1", "b2", "b3"]
    approval = {"selected_ids": ids}
    subjects = {"a1": "Art", "a2": "Art", "b1": "Bio", "b2": "Bio", "b3": "Bio"}
    manifest = {"items": [{"problem_id": i, "subject": subjects[i], "source": f"src-{i[0]}", "status": "INCLUDED"} for i in ids]
                + [{"problem_id": "x9", "subject": "Art", "source": "src-x", "status": "HELD_OUT_DEV"}]}
    lineage = [{"problem_id": i, "structure_hash": f"h-{i}"} for i in ids + ["x9"]]
    lineage[1]["structure_hash"] = lineage[0]["structure_hash"]  # a2 duplicates a1
    return approval, manifest, lineage


class OrderTests(unittest.TestCase):
    def test_build(self):
        result = order.build(*fixtures())
        ids = [i["problem_id"] for i in result["items"]]
        dropped = result["dropped_duplicate_structure_ids"]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(sorted(ids + dropped), ["a1", "a2", "b1", "b2", "b3"])
        self.assertEqual((result["version"], result["master_seed"], result["count"]), ("error-note-order-v1", 3407, 4))
        self.assertEqual(result["ids_sha256"], hashlib.sha256("\n".join(ids).encode()).hexdigest())
        self.assertEqual(set(result["items"][0]), {"problem_id", "subject", "source"})
        self.assertEqual([i["subject"] for i in result["items"]][:2], ["Art", "Bio"])
        self.assertIsInstance(result["rule"], str)

    def test_build_rejects_bad_inputs(self):
        approval, manifest, lineage = fixtures()
        with self.assertRaises(ValueError):
            order.build({"selected_ids": ["a1", "nope"]}, manifest, lineage)
        with self.assertRaises(ValueError):
            order.build(approval, manifest, [r for r in lineage if r["problem_id"] != "b1"])
        with self.assertRaises(ValueError):
            order.build({"selected_ids": ["a1", "x9"]}, manifest, lineage)

    def test_read_order_round_trip_and_tamper(self):
        result = order.build(*fixtures())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "order.json"
            path.write_text(json.dumps(result))
            self.assertEqual(order.read_order(path), result)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(order.read_order(path, expected_sha256=digest), result)
            with self.assertRaises(ValueError):
                order.read_order(path, expected_sha256="0" * 64)
            for label, mutate in (("id", lambda d: d["items"][0].update(problem_id="zzz")), ("count", lambda d: d.update(count=99)),
                                  ("version", lambda d: d.update(version="old")),
                                  ("duplicate", lambda d: d["items"].__setitem__(1, dict(d["items"][0])))):
                with self.subTest(label=label):
                    bad = json.loads(json.dumps(result))
                    mutate(bad)
                    path.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        order.read_order(path)

    def test_main_builds_and_refuses_overwrite(self):
        approval, manifest, lineage = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "approval.json").write_text(json.dumps(approval))
            (tmp / "manifest.json").write_text(json.dumps(manifest))
            (tmp / "lineage.jsonl").write_text("\n".join(json.dumps(r) for r in lineage) + "\n")
            argv = ["build", "--approval", str(tmp / "approval.json"), "--manifest", str(tmp / "manifest.json"),
                    "--lineage", str(tmp / "lineage.jsonl"), "--output", str(tmp / "out.json")]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(order.main(argv), 0)
            self.assertIn("count=4 ids_sha256=", out.getvalue())
            self.assertEqual(order.read_order(tmp / "out.json")["count"], 4)
            with self.assertRaises(SystemExit):
                order.main(argv)


class ContractSurfaceTests(unittest.TestCase):
    def test_parsed_note_is_frozen_and_prompt_instruction_unchanged(self):
        with self.assertRaises(Exception):
            ParsedNote("CORRECT", "NONE", "a", "b").cause = "IMAGE"
        self.assertTrue(contract.NOTE_INSTRUCTION.endswith('End with "Final answer: ...".'))


if __name__ == "__main__":
    unittest.main()
