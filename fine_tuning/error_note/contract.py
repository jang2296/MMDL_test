"""Fixed values, the note prompt, note parsing and training-row construction. CPU only, no model imports.

Plan: claudedocs/research_self_note_finetuning_plan_20261101.md sections 1.2, 3.1-3.3, 3.6.
Version 2 (claudedocs/plan_self_note_v2_20261102.md): a note without a corrected solution, one gold-blind re-solve
with the note's Lesson as the only hint, rows from T1 or that re-solve plus the note. The v1 names stay unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import re
import string
from typing import Callable
import zlib

MASTER_SEED = 3407
MAX_NEW_TOKENS = 8192
CONTEXT_LIMIT = 24576
IMAGE = {"min_pixels": 262144, "max_pixels": 1310720, "preprocessing": "qwen_vl_utils_0_0_14"}
TURN1_ANSWER_CONTRACT = "p0-eval-aligned-v1"
PER_UPDATE_ITEMS = 8
MIN_TARGET_TOKENS = 32
MAX_TARGET_TOKENS = 4096
REPEAT_WINDOW = 12
REPEAT_COUNT = 3
LEAK_PHRASES = ("earlier solution", "earlier attempt", "previous attempt", "previous solution", "grading result",
                "was told", "this note", "error note", "given answer")
IMAGE_READINGS = ("CORRECT", "MISREAD", "UNSURE")
CAUSES = ("IMAGE", "REASONING", "KNOWLEDGE", "CALCULATION", "NO_FINAL_ANSWER", "NONE")
ROUTE_FULL, ROUTE_LM = "FULL", "LM"
KIND_SOLUTION, KIND_NOTE = "A", "B"
VERDICT_LABELS = {"CORRECT": "CORRECT", "INCORRECT": "WRONG", "INVALID_NO_FINAL": "NO CLEAR FINAL ANSWER",
                  "INVALID_MULTIPLE": "NO CLEAR FINAL ANSWER", "TRUNCATED": "CUT OFF"}
HEADINGS = ("What the image shows", "Image reading", "Cause", "What went wrong", "How to solve it", "Corrected solution")
NOTE_INSTRUCTION = """Review your own work like a student writing an error note.
Use these headings in this order, each starting on a new line.

What the image shows: list the facts this problem needs, checking each one against the image.
Image reading: CORRECT, MISREAD or UNSURE. Write MISREAD if your earlier solution used a fact the image does not show, even if its final answer was right.
Cause: IMAGE, REASONING, KNOWLEDGE, CALCULATION, NO_FINAL_ANSWER or NONE.
What went wrong: the exact step and why. If nothing went wrong, say what was done properly.
How to solve it: the method to use on similar problems.
Corrected solution: one complete solution from the beginning. Keep the parts of your earlier solution that were right and fix it from the first mistake on. Do not mention this note, an earlier attempt, a grade or being told the answer. End with "Final answer: ..."."""


@dataclass(frozen=True)
class ParsedNote:
    image_reading: str  # one of IMAGE_READINGS
    cause: str  # one of CAUSES
    note_text: str  # the completion before the "Corrected solution" heading, stripped
    corrected_text: str  # the body after the "Corrected solution" heading, stripped


def sample_seed(problem_id: str, stage: str, master_seed: int = MASTER_SEED) -> int:
    # stage "t1" is the MMMU rule of CLAUDE.md §7.2; stage "note" hashes the same string with ":note" appended.
    if stage == "t1":
        key = f"{master_seed}:{problem_id}"
    elif stage == "note":
        key = f"{master_seed}:{problem_id}:note"
    elif stage == "resolve":
        key = f"{master_seed}:{problem_id}:resolve"
    else:
        raise ValueError(f"unknown seed stage: {stage!r}")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**31)


def verdict_label(status: str) -> str:
    if status not in VERDICT_LABELS:
        raise ValueError(f"no verdict label for status {status!r}")
    return VERDICT_LABELS[status]


def note_prompt(p0_text: str, t1_text: str, status: str, gold: str) -> str:
    if not p0_text.strip():
        raise ValueError("p0_text is blank")
    if not gold.strip():
        raise ValueError("gold is blank")
    return (f"{p0_text}\n\nYour earlier solution:\n{t1_text}\n\nGrading result: {verdict_label(status)}\n"
            f"Correct answer: {gold}\n\n{NOTE_INSTRUCTION}")


def _heading_pattern(name: str) -> re.Pattern:
    # line start, optional "#", optional "*"/"**", the name, optional "**", ":", optional "**" glued to the colon
    return re.compile(rf"^[ \t]*(?:#+[ \t]*)?(?:\*+[ \t]*)?{re.escape(name)}[ \t]*(?:\*+[ \t]*)?:\*{{0,2}}[ \t]*", re.MULTILINE)


_HEADING_PATTERNS = tuple(_heading_pattern(name) for name in HEADINGS)


def _enum_word(value_line: str, allowed: tuple[str, ...]) -> str | None:
    words = value_line.split()
    word = words[0].strip("*" + string.punctuation) if words else ""
    return word if word in allowed else None


def parse_note(text: str) -> ParsedNote | None:
    matches = []
    for pattern in _HEADING_PATTERNS:
        found = list(pattern.finditer(text))
        if len(found) != 1:
            return None
        matches.append(found[0])
    if any(a.start() >= b.start() for a, b in zip(matches, matches[1:])):
        return None
    bodies = [text[m.end():(matches[i + 1].start() if i + 1 < len(matches) else len(text))].strip()
              for i, m in enumerate(matches)]
    if not all(bodies):
        return None
    image_reading = _enum_word(bodies[1].split("\n", 1)[0], IMAGE_READINGS)
    cause = _enum_word(bodies[2].split("\n", 1)[0], CAUSES)
    if image_reading is None or cause is None:
        return None
    return ParsedNote(image_reading, cause, text[:matches[-1].start()].strip(), bodies[-1])


def has_repetition(token_ids: list[int], window: int = REPEAT_WINDOW, count: int = REPEAT_COUNT) -> bool:
    seen: dict[tuple[int, ...], int] = {}
    for i in range(len(token_ids) - window + 1):
        run = tuple(token_ids[i:i + window])
        seen[run] = seen.get(run, 0) + 1
        if seen[run] >= count:
            return True
    return False


def leaked_phrase(text: str) -> str | None:
    lowered = text.lower()
    return next((phrase for phrase in LEAK_PHRASES if phrase in lowered), None)


def _target_problem(token_ids: list[int], finish_reason: str, status: str) -> str | None:
    if finish_reason != "stop":
        return "NOT_STOPPED"
    if status != "CORRECT":
        return "NOT_CORRECT"
    if len(token_ids) < MIN_TARGET_TOKENS:
        return "TOO_SHORT"
    if len(token_ids) > MAX_TARGET_TOKENS:
        return "TOO_LONG"
    if has_repetition(token_ids):
        return "REPETITION"
    return None


def build_rows(record: dict, tokenize: Callable[[str], list[int]]) -> tuple[list[dict], list[dict]]:
    """One collected item -> (training rows, drop records). Every dropped row is recorded; nothing is replaced.

    record keys: problem_id, subject, source, p0_text, image_paths, t1 {text, finish_reason, status},
    note (absent when no note was written) {prompt, text, finish_reason, corrected_status}.
    Row keys: row_id "<problem_id>:<kind>", problem_id, kind, prompt, target, route, target_source
    ("T1" | "CORRECTED" | "NOTE"), image_paths, target_tokens, subject, source.
    Drop keys: problem_id, kind, reason.
    """
    problem_id = record["problem_id"]
    drops: list[dict] = []

    def drop(kind: str, reason: str) -> None:
        drops.append({"problem_id": problem_id, "kind": kind, "reason": reason})

    note = record.get("note")
    parsed = None
    if not note:
        reason = "NO_NOTE"
    elif note["finish_reason"] != "stop":
        reason = "NOTE_NOT_STOPPED"
    else:
        parsed = parse_note(note["text"])
        reason = "NOTE_UNPARSED"
    if parsed is None:
        drop(KIND_SOLUTION, reason)
        drop(KIND_NOTE, reason)
        return [], drops
    assert note is not None

    route = ROUTE_FULL if parsed.image_reading == "MISREAD" else ROUTE_LM

    def make_row(kind: str, prompt: str, target: str, target_source: str, target_tokens: int) -> dict:
        return {"row_id": f"{problem_id}:{kind}", "problem_id": problem_id, "kind": kind, "prompt": prompt, "target": target,
                "route": route, "target_source": target_source, "image_paths": list(record["image_paths"]),
                "target_tokens": target_tokens, "subject": record["subject"], "source": record["source"]}

    rows: list[dict] = []
    t1 = record["t1"]
    chosen = None
    if t1["status"] == "CORRECT" and parsed.image_reading == "CORRECT":
        ids = tokenize(t1["text"])
        if _target_problem(ids, t1["finish_reason"], t1["status"]) is None:
            chosen = (t1["text"], "T1", len(ids))
    if chosen is None:
        ids = tokenize(parsed.corrected_text)
        problem = _target_problem(ids, note["finish_reason"], note["corrected_status"])
        if problem is None:
            phrase = leaked_phrase(parsed.corrected_text)
            problem = f"LEAK:{phrase}" if phrase else None
        if problem is None:
            chosen = (parsed.corrected_text, "CORRECTED", len(ids))
        else:
            drop(KIND_SOLUTION, problem)
    if chosen is not None:
        rows.append(make_row(KIND_SOLUTION, record["p0_text"], *chosen))

    note_ids = tokenize(parsed.note_text)
    if len(note_ids) < MIN_TARGET_TOKENS:
        drop(KIND_NOTE, "NOTE_EMPTY")
    else:
        rows.append(make_row(KIND_NOTE, note["prompt"], parsed.note_text, "NOTE", len(note_ids)))
    return rows, drops


def update_groups(item_ids: list[str], per_update: int = PER_UPDATE_ITEMS) -> list[list[str]]:
    if per_update < 1:
        raise ValueError("per_update must be at least 1")
    return [item_ids[i:i + per_update] for i in range(0, len(item_ids), per_update)]


def select_rows(rows: list[dict], arm: str) -> list[dict]:
    if arm == "note":
        return list(rows)
    if arm == "solution":
        return [{**row, "route": ROUTE_LM} for row in rows if row["kind"] == KIND_SOLUTION]
    raise ValueError(f"unknown arm: {arm!r}")


def order_key(problem_id: str, master_seed: int = MASTER_SEED) -> str:
    return hashlib.sha256(f"{master_seed}:{problem_id}".encode()).hexdigest()


def round_robin_order(items: list[dict], master_seed: int = MASTER_SEED) -> list[str]:
    """items: {"problem_id", "subject", "structure_hash"}. Results and gold are never inputs."""
    ids = [item["problem_id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate problem_id")
    seen_structures: set = set()
    by_subject: dict[str, list[str]] = {}
    for item in sorted(items, key=lambda item: order_key(item["problem_id"], master_seed)):
        if item["structure_hash"] in seen_structures:
            continue
        seen_structures.add(item["structure_hash"])
        by_subject.setdefault(item["subject"], []).append(item["problem_id"])
    queues = [by_subject[subject] for subject in sorted(by_subject)]
    return [queue[turn] for turn in range(max(map(len, queues), default=0)) for queue in queues if turn < len(queue)]


# ------------------------------------------------------------------ version 2

HEADINGS_V2 = ("What the image shows", "Image reading", "Cause", "What went wrong", "Lesson")
NOTE_INSTRUCTION_V2 = """Review your own work like a student writing an error note.
Use these headings in this order, each starting on a new line. Keep the whole note short, about 200 words at most, and do not write a solution.

What the image shows: list the facts this problem needs, checking each one against the image.
Image reading: CORRECT, MISREAD or UNSURE. Write MISREAD if your earlier solution used a fact the image does not show, even if its final answer was right.
Cause: IMAGE, REASONING, KNOWLEDGE, CALCULATION, NO_FINAL_ANSWER or NONE.
What went wrong: the exact step and why. If nothing went wrong, say what was done properly.
Lesson: 1 to 3 general sentences on how to solve this kind of problem. They must be usable without knowing the answer: do not name or restate the correct answer, any option letter or the content of any option."""
NOTE_INSTRUCTION_V3 = """Review your own work against the question, image and given answer. A wrong final answer does not by itself establish why you were wrong.
Use these headings in this order, each starting on a new line. Keep the whole note short, about 200 words at most, and do not write a solution.

What the image shows: record only relevant directly visible details. Distinguish these from outside knowledge; say when a needed fact cannot be read from the image.
Image reading: CORRECT, MISREAD or UNSURE. Use MISREAD only for a concrete contradiction between an earlier visual claim and the image. An unsupported claim about outside knowledge is not an image misreading. Use UNSURE when the relevant visible details cannot be checked.
Cause: IMAGE, REASONING, KNOWLEDGE, CALCULATION, NO_FINAL_ANSWER or NONE. Choose only a supported cause. Use NONE if no specific error is established, and explain any uncertainty below.
What went wrong: quote an exact false claim from your earlier solution and explicitly correct that claim. State whether the correction is supported by the question/image or outside knowledge. If the claim or its correction cannot be established, say what remains uncertain. Do not invent motives, assumptions or causes that the earlier solution does not support. If no error is established, say so without fabricating one.
Lesson: 1 to 3 general sentences on how to solve this kind of problem. They must be usable without knowing the answer: do not name or restate the correct answer, any option letter or the content of any option."""
# The 3,417 run (2026-11-03) with "Apply it silently: do not mention, quote or refer to this lesson" got bare answers
# (median 8 tokens, 1,071 of 1,255 under 32), so the plain header is back. A re-solve that echoes the lesson is still
# dropped by RESOLVE_LEAK_PHRASES, never edited.
RESOLVE_HEADER = "Lesson from your own review of a similar problem:\n"
# Phrases a re-solve must not contain: at evaluation there is no hint to refer to. Bare "note" is left out on
# purpose: music items use it as a word of the subject.
RESOLVE_LEAK_PHRASES = ("lesson", "my review", "the hint", "as noted in", "earlier attempt", "previous attempt",
                        "review of")
LOOP_WINDOW = 3000
LOOP_RATIO = 0.12
OPTION_LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class ParsedNoteV2:
    image_reading: str  # one of IMAGE_READINGS
    cause: str  # one of CAUSES
    note_text: str  # the whole completion, stripped
    lesson: str  # the body after the "Lesson" heading, stripped


def note_prompt_v2(p0_text: str, t1_text: str, status: str, gold: str) -> str:
    if not p0_text.strip():
        raise ValueError("p0_text is blank")
    if not gold.strip():
        raise ValueError("gold is blank")
    return (f"{p0_text}\n\nYour earlier solution:\n{t1_text}\n\nGrading result: {verdict_label(status)}\n"
            f"Correct answer: {gold}\n\n{NOTE_INSTRUCTION_V2}")


def note_prompt_v3(p0_text: str, t1_text: str, status: str, gold: str) -> str:
    """The v2 note schema with the source-aware instruction, used only by training-answer-v1."""
    return note_prompt_v2(p0_text, t1_text, status, gold).removesuffix(NOTE_INSTRUCTION_V2) + NOTE_INSTRUCTION_V3


_HEADING_PATTERNS_V2 = tuple(_heading_pattern(name) for name in HEADINGS_V2)


def parse_note_v2(text: str) -> ParsedNoteV2 | None:
    """The parse_note rules: every heading exactly once and in order, non-empty bodies, enum first words."""
    matches = []
    for pattern in _HEADING_PATTERNS_V2:
        found = list(pattern.finditer(text))
        if len(found) != 1:
            return None
        matches.append(found[0])
    if any(a.start() >= b.start() for a, b in zip(matches, matches[1:])):
        return None
    bodies = [text[m.end():(matches[i + 1].start() if i + 1 < len(matches) else len(text))].strip()
              for i, m in enumerate(matches)]
    if not all(bodies):
        return None
    image_reading = _enum_word(bodies[1].split("\n", 1)[0], IMAGE_READINGS)
    cause = _enum_word(bodies[2].split("\n", 1)[0], CAUSES)
    if image_reading is None or cause is None:
        return None
    return ParsedNoteV2(image_reading, cause, text.strip(), bodies[-1])


def _normal(text: str) -> str:
    return " ".join(text.lower().split())


def _has_words(text: str, needle: str) -> bool:
    """The word tokens of `needle` occur consecutively in `text`; case, spacing and punctuation are ignored."""
    words, run = re.findall(r"\w+", text.lower()), re.findall(r"\w+", needle.lower())
    return bool(run) and any(words[i:i + len(run)] == run for i in range(len(words) - len(run) + 1))


_LETTER_REFERENCE = re.compile(r"\b(?:answers?|options?|choices?)\b\W*(?:\w+\W+)?\(?([a-j])\)?(?!\w)", re.IGNORECASE)
_PAREN_LETTER = re.compile(r"\(([a-j])\)", re.IGNORECASE)
_GOLD_LETTER = re.compile(r"\(?([A-J])\)?(?:[.:)]\s*.*)?", re.DOTALL)
_ANSWER_PHRASES = ("correct answer", "the answer is")


def _has_gold_letter(lesson: str, letter: str) -> bool:
    """The gold letter as its own capital token ("Pick C.", "C is correct"); "F#" or "Fe" does not count. For "A",
    usually the article, only "A)", "A." or "A:" that does not start a sentence counts (audit 2026-11-02)."""
    if letter != "A":
        return re.search(rf"(?<![\w'#]){letter}(?![\w'#])", lesson) is not None
    return any(not re.search(r"(?:^|[.!?:]\s+)\s*$", lesson[:m.start()])
               for m in re.finditer(r"(?<![\w'])A(?=[).:])", lesson))


def hint_leak(lesson: str, gold: str, choices: tuple[str, ...] | list[str]) -> str | None:
    """The first reason the lesson may give the answer away, or None.

    Biased toward false positives on purpose: a flagged lesson only costs one re-solve, a leaked one would teach an
    answer the model never found. "option a ..." or "(a)" is flagged although "a" is usually the article, an open
    question's short gold (e.g. "2") is flagged wherever it appears as a word, and so are the words of the gold option
    of any length ("Yes", "4/4", "It makes its own food.")."""
    match = _LETTER_REFERENCE.search(lesson)
    if match:
        return f"OPTION_LETTER:{match[1].upper()}"
    match = _PAREN_LETTER.search(lesson)
    if match:
        return f"PAREN_LETTER:{match[1].upper()}"
    normal = _normal(lesson)
    phrase = next((p for p in _ANSWER_PHRASES if p in normal), None)
    if phrase:
        return f"PHRASE:{phrase}"
    bare = gold.strip().strip("*").strip()
    letter = _GOLD_LETTER.fullmatch(bare) if choices else None
    if letter is not None:
        if _has_gold_letter(lesson, letter[1]):
            return f"GOLD_LETTER:{letter[1]}"
        index = OPTION_LETTERS.index(letter[1])
        return "GOLD_OPTION_TEXT" if index < len(choices) and _has_words(lesson, choices[index]) else None
    if bare and re.search(rf"(?<!\w){re.escape(_normal(bare))}(?!\w)", normal):
        return "GOLD_TEXT"
    return None


def resolve_prompt(p0_text: str, lesson: str) -> str:
    """The re-solve input: the lesson block, then the unchanged P0 text. No gold, T1 text or grading result."""
    if not p0_text.strip():
        raise ValueError("p0_text is blank")
    if not lesson.strip():
        raise ValueError("lesson is blank")
    return f"{RESOLVE_HEADER}{lesson.strip()}\n\n{p0_text}"


def is_loop(text: str, window: int = LOOP_WINDOW, ratio: float = LOOP_RATIO) -> bool:
    """The last `window` characters (at least window // 2 of them) compress below `ratio` of their size."""
    tail = text[-window:]
    if len(tail) < window // 2:
        return False
    data = tail.encode()
    return len(zlib.compress(data)) < ratio * len(data)


def resolve_leaked_phrase(text: str) -> str | None:
    lowered = text.lower()
    return next((phrase for phrase in RESOLVE_LEAK_PHRASES + LEAK_PHRASES if phrase in lowered), None)


def build_rows_v2(record: dict, tokenize: Callable[[str], list[int]]) -> tuple[list[dict], list[dict]]:
    """One collected item -> (training rows, drop records). Texts are used as generated; every drop is recorded.

    record keys: the build_rows keys with note {prompt, text, finish_reason} (a v2 note) and optional resolve
    {text, finish_reason, status} or {skipped: reason}. Row A is T1 when T1 is correct, else the re-solve, whose
    prompt is the plain P0 text (the evaluation input; the lesson is never in a training prompt). Row B is the note.
    There is no gold-aware corrected solution in version 2.
    """
    problem_id = record["problem_id"]
    drops: list[dict] = []

    def drop(kind: str, reason: str) -> None:
        drops.append({"problem_id": problem_id, "kind": kind, "reason": reason})

    note = record.get("note")
    parsed = None
    if not note:
        note_reason = "NO_NOTE"
    elif note["finish_reason"] != "stop":
        note_reason = "NOTE_NOT_STOPPED"
    else:
        parsed = parse_note_v2(note["text"])
        note_reason = "NOTE_UNPARSED"
    route = ROUTE_FULL if parsed is not None and parsed.image_reading == "MISREAD" else ROUTE_LM

    def make_row(kind: str, prompt: str, target: str, target_source: str, target_tokens: int) -> dict:
        return {"row_id": f"{problem_id}:{kind}", "problem_id": problem_id, "kind": kind, "prompt": prompt, "target": target,
                "route": route, "target_source": target_source, "image_paths": list(record["image_paths"]),
                "target_tokens": target_tokens, "subject": record["subject"], "source": record["source"]}

    rows: list[dict] = []
    t1, resolve = record["t1"], record.get("resolve")
    if t1["status"] == "CORRECT":
        source, solution = "T1", t1
    elif resolve is None:
        source, solution = "RESOLVE", None
        drop(KIND_SOLUTION, "NO_RESOLVE")
    elif "skipped" in resolve:
        source, solution = "RESOLVE", None
        drop(KIND_SOLUTION, f"RESOLVE_SKIPPED:{resolve['skipped']}")
    else:
        source, solution = "RESOLVE", resolve
    if solution is not None:
        ids = tokenize(solution["text"])
        problem = _target_problem(ids, solution["finish_reason"], solution["status"])
        if problem is None and is_loop(solution["text"]):
            problem = "LOOP"
        if problem is None and source == "T1" and parsed is not None and parsed.image_reading == "MISREAD":
            problem = "T1_SELF_MISREAD"
        if problem is None and source == "RESOLVE":
            phrase = resolve_leaked_phrase(solution["text"])
            problem = f"LEAK:{phrase}" if phrase else None
        if problem is None:
            rows.append(make_row(KIND_SOLUTION, record["p0_text"], solution["text"], source, len(ids)))
        else:
            drop(KIND_SOLUTION, problem)

    if parsed is None:
        drop(KIND_NOTE, note_reason)
        return rows, drops
    assert note is not None
    note_ids = tokenize(parsed.note_text)
    if len(note_ids) < MIN_TARGET_TOKENS:
        drop(KIND_NOTE, "NOTE_EMPTY")
    elif is_loop(parsed.note_text):
        drop(KIND_NOTE, "NOTE_LOOP")
    else:
        rows.append(make_row(KIND_NOTE, note["prompt"], parsed.note_text, "NOTE", len(note_ids)))
    return rows, drops


def build_rows_v3(record: dict, tokenize: Callable[[str], list[int]]) -> tuple[list[dict], list[dict]]:
    """The unchanged v2 targets/routes with source-bound whole-note and lesson review gates.

    Extra record keys: audit (note-audit-v1), source_provenance (the private source dict).
    A re-solve additionally carries its exact prompt and lesson_sha256. Missing review is a drop;
    a present but stale/forged binding is an error. No review text ever becomes a target.
    """
    from .audit import text_sha256, validate_audit

    audit = record.get("audit")
    if audit is not None:
        if "source_provenance" not in record:
            raise ValueError("rows v3: source provenance required for audit validation")
        validate_audit(audit, record, record["source_provenance"])
    rows, drops = build_rows_v2(record, tokenize)
    admitted = []
    for row in rows:
        reason = None
        content_ok = (audit is not None and audit["content"]["result"] == "PASS" and
                      audit["content"]["scope"] == "entire_note")
        if row["kind"] == KIND_NOTE and not content_ok:
            reason = "NOTE_CONTENT_UNKNOWN" if audit is None else f"NOTE_CONTENT_{audit['content']['result']}"
        if row["target_source"] == "RESOLVE":
            clear = (audit is not None and audit["leak"]["result"] == "CLEAR" and
                     audit["leak"]["scope"] == "entire_lesson")
            if not clear or not content_ok:
                reason = "RESOLVE_LESSON_NOT_AUTHORIZED"
            else:
                parsed = parse_note_v2(record["note"]["text"])
                if parsed is None:
                    raise ValueError("rows v3: authorized audit requires a parsed note")
                resolve = record["resolve"]
                if (resolve.get("prompt") != resolve_prompt(record["p0_text"], parsed.lesson) or
                        resolve.get("lesson_sha256") != text_sha256(parsed.lesson)):
                    raise ValueError("rows v3: resolve prompt/lesson binding mismatch")
        if reason is None:
            row |= {"prompt_sha256": text_sha256(row["prompt"]), "target_sha256": text_sha256(row["target"]),
                    "audit_sha256": audit["sha256"] if audit is not None else None,
                    "source_sha256": audit["bindings"]["source_sha256"] if audit is not None else None}
            if row["target_source"] == "RESOLVE":
                row["lesson_sha256"] = record["resolve"]["lesson_sha256"]
            admitted.append(row)
        else:
            drops.append({"problem_id": record["problem_id"], "kind": row["kind"], "reason": reason})
    return admitted, drops


def select_rows_v3(rows: list[dict], arm: str) -> list[dict]:
    """Both arms retain identical A targets, order and FULL/LM routes; no-note removes only B."""
    if arm == "note":
        return list(rows)
    if arm == "no-note":
        return [row for row in rows if row["kind"] == KIND_SOLUTION]
    raise ValueError(f"unknown v3 arm: {arm!r}")


def spread_order(items: list[dict], base_ids: list[str]) -> list[str]:
    """Every source spread over the whole run: within a source the base order is kept, and all items are sorted by
    ((rank in source + 0.5) / source size, base position). items: {"problem_id", "source"}; no results or gold."""
    source = {item["problem_id"]: item["source"] for item in items}
    if len(base_ids) != len(set(base_ids)) or len(items) != len(base_ids) or set(source) != set(base_ids):
        raise ValueError("spread_order: items and base_ids must be the same unique ids")
    by_source: dict[str, list[str]] = {}
    for problem_id in base_ids:
        by_source.setdefault(source[problem_id], []).append(problem_id)
    position = {problem_id: index for index, problem_id in enumerate(base_ids)}
    key = {problem_id: (Fraction(2 * rank + 1, 2 * len(ids)), position[problem_id])
           for ids in by_source.values() for rank, problem_id in enumerate(ids)}
    return sorted(base_ids, key=key.__getitem__)
