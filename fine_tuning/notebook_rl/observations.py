"""Parse declared observations only; absent observations are not visual errors."""
from __future__ import annotations

import re
import unicodedata

from .answers import number
from .schema import SlotReport, VerifierItem

_LINE = re.compile(r"^[ \t]*-?[ \t]*([a-z_]+(?:\[[^\]\n]+\]){0,2})[ \t]*=[ \t]*(.+?)[ \t]*$", re.I)
_BEGIN = re.compile(r"^[ \t]*Observed facts:[ \t]*$", re.I | re.M)
_END = re.compile(r"^[ \t]*Solution:[ \t]*$", re.I | re.M)


def slot_id(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"\s+", "", value)
    return re.sub(r"\[([^\]]+)\]", lambda m: "[" + m[1].replace("'", "").replace('"', "") + "]", value)


def parse_observations(text: str) -> dict[str, list[str]]:
    starts = list(_BEGIN.finditer(text))
    # Repeated blocks are not silently resolved by choosing the most convenient one.
    if not starts:
        return {}
    result: dict[str, list[str]] = {}
    for i, start in enumerate(starts):
        end = _END.search(text, start.end())
        if end is None or (i + 1 < len(starts) and end.start() > starts[i + 1].start()):
            continue
        for line in text[start.end():end.start()].splitlines():
            match = _LINE.fullmatch(line)
            if match:
                result.setdefault(slot_id(match[1]), []).append(match[2])
    return result


def normalize_slot(value: str, slot_type: str) -> str | None:
    value = unicodedata.normalize("NFKC", value).strip()
    if slot_type == "number":
        # Units may be omitted only in observation slots, never in final-answer scoring.
        parsed = number(value)
        if parsed is None:
            unit = r"[A-Za-zµΩ°%]+(?:[0-9]|\^-?[0-9]+)?"
            match = re.fullmatch(rf"(.+?)[ \t]*({unit}(?:[ /·*]{unit})*)", value)
            if match:
                parsed = number(match[1])
        return str(parsed) if parsed is not None else None
    if slot_type == "pitch":
        value = value.replace("♯", "#").replace("♭", "b").replace("♮", "")
        value = re.sub("sharp", "#", value, flags=re.I)
        value = re.sub("flat", "b", value, flags=re.I)
        value = re.sub("natural", "", value, flags=re.I)
        value = re.sub(r"\s+", "", value)
        match = re.fullmatch(r"([A-Ga-g])([#b]{0,2})(-?[0-9])", value)
        return match[1].upper() + match[2] + match[3] if match else None
    if slot_type == "text":
        return " ".join(value.casefold().split()) or None
    if slot_type == "relation":
        compact = re.sub(r"\s+", "", value.casefold())
        match = re.fullmatch(r"([a-z0-9_]+)(?:<->|↔|,|-)([a-z0-9_]+)", compact)
        if match:
            return "-".join(sorted((match[1], match[2])))
        return compact if re.fullmatch(r"[a-z0-9_]+", compact) else None
    raise ValueError("observation: unsupported slot type")


def _normalize(value: str, kind: str, key: str) -> str | None:
    if kind == "relation" and key.startswith(("trans[", "dir[")):
        compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())
        compact = compact.replace("→", "->")
        # Directed endpoints retain order; a DFA transition's value may be one state.
        return compact if re.fullmatch(r"[a-z0-9_]+(?:(?:->|-|,)[a-z0-9_]+)?", compact) else None
    return normalize_slot(value, kind)


def compare_slots(parsed: dict[str, list[str]], item: VerifierItem) -> SlotReport:
    if set(item.slots) != set(item.slot_types):
        raise ValueError("observation: mismatched reference slot types")
    keys = {slot_id(k): k for k in item.slots}
    if len(keys) != len(item.slots):
        raise ValueError("observation: colliding normalized reference slot IDs")
    required = {slot_id(k) for k in item.required_slots}
    if not required <= keys.keys() or len(required) != len(item.required_slots):
        raise ValueError("observation: invalid required slots")
    observed: dict[str, list[str]] = {}
    for key, values in parsed.items():
        if not isinstance(key, str) or not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("observation: expected raw value lists")
        key = slot_id(key)
        if key in keys and values:
            observed.setdefault(key, []).extend(values)
    correct, wrong, conflicts = set(), set(), set()
    for key, original in keys.items():
        kind = item.slot_types[original]
        expected = _normalize(item.slots[original], kind, key)
        if expected is None:
            raise ValueError("observation: unverifiable reference slot")
        if key not in observed:
            continue
        normalized = [_normalize(v, kind, key) for v in observed[key]]
        if len(set(normalized)) > 1:
            conflicts.add(key)
        if all(v == expected for v in normalized):
            correct.add(key)
        else:
            wrong.add(key)
    # Missing slots are absent evidence. Only present wrong slots justify a visual route.
    return SlotReport(len(observed), len(required), len(correct & required),
                      tuple(sorted(wrong & required)), len(correct), len(keys),
                      tuple(sorted(conflicts)))
