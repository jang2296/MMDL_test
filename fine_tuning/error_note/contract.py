"""Fixed values, the note prompt, note parsing and training-row construction. CPU only, no model imports.

Plan: claudedocs/research_self_note_finetuning_plan_20261101.md sections 1.2, 3.1-3.3, 3.6.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import string
from typing import Callable

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
