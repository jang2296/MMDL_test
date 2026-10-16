"""Private deterministic rewards. This object must stay in the verifier process."""
from __future__ import annotations

import re
from pathlib import Path

from .answers import VERSION, compare, extract_final, is_ambiguous
from .observations import compare_slots, parse_observations, slot_id
from .schema import DiagTarget, Mode, VerifierResult, to_json
from .vault import read_verifier


def diagnosis_fields(text: str) -> tuple[str, str] | None:
    """A single three-line note, not a search for whichever label earns reward."""
    match = re.fullmatch(
        r"\s*Cause:[ \t]*(VISUAL_MISREAD|VISUAL_RELATION|NON_VISUAL|FINALIZATION)[ \t]*\n"
        r"[ \t]*Wrong item:[ \t]*([^\n]+)\n[ \t]*Lesson:[ \t]*([^\n]+)[ \t]*", text)
    if match is None or not match[2].strip() or not match[3].strip():
        return None
    return ("VISUAL" if match[1].startswith("VISUAL_") else match[1], slot_id(match[2]))


class Verifier:
    def __init__(self, vault_root: Path):
        self.items = read_verifier(vault_root)

    def cause(self, problem_id: str, result: VerifierResult):
        from .attribution import attempt_cause
        return attempt_cause(result, self.items[problem_id])

    def diagnosis_target(self, problem_id: str, result: VerifierResult):
        from .attribution import diagnosis_target
        return diagnosis_target(result, self.items[problem_id])

    def route(self, problem_id: str, results: list[VerifierResult], rule: str):
        from .attribution import problem_route
        return problem_route(results, self.items[problem_id], rule)

    def grade(self, problem_id: str, text: str, finish_reason: str, mode: Mode) -> VerifierResult:
        if mode not in ("obs", "plain"):
            raise ValueError("verifier: unsupported observation mode")
        item = self.items[problem_id]
        extracted = extract_final(text, finish_reason)
        slots = compare_slots(parse_observations(text), item) if mode == "obs" and item.slots else None
        if extracted.status != "OK":
            result = VerifierResult(extracted.status, extracted.answer, 0.0, slots, VERSION)
        elif extracted.answer is None:
            raise ValueError("verifier: extractor returned OK without answer")
        elif is_ambiguous(extracted.answer, item):
            result = VerifierResult("INVALID_MULTIPLE", extracted.answer, 0.0, slots, VERSION)
        else:
            matched = compare(extracted.answer, item)
            result = VerifierResult("UNVERIFIABLE" if matched is None else "CORRECT" if matched else "INCORRECT",
                                    extracted.answer, float(matched is True), slots, VERSION)
        to_json(result)
        return result

    def grade_diagnosis(self, problem_id: str, text: str, target: DiagTarget) -> float:
        item = self.items[problem_id]
        to_json(target)
        if not {slot_id(k) for k in target.wrong_slots} <= {slot_id(k) for k in item.required_slots}:
            raise ValueError("verifier: diagnosis target contains unregistered required slots")
        fields = diagnosis_fields(text)
        if fields is None or fields[0] != target.cause3:
            return 0.0
        if target.cause3 != "VISUAL" and fields[1] != "none":
            return 0.0
        return 1.0 + 0.25 * (target.cause3 == "VISUAL" and fields[1] in {
            slot_id(k) for k in target.wrong_slots})


def answer_reward(result: VerifierResult, *, visual_lambda: float = 0.0) -> float:
    """Optional S4 bonus is bounded and cannot reward facts-only or invalid answers."""
    to_json(result)
    if type(visual_lambda) not in (float, int) or not 0 <= visual_lambda <= 0.1:
        raise ValueError("verifier: visual coefficient must be in [0,0.1]")
    if result.status in ("UNVERIFIABLE", "CONTEXT_BLOCKED", "SYSTEM_ERROR"):
        raise ValueError("verifier: this result must be excluded, not assigned a training reward")
    if result.status not in ("CORRECT", "INCORRECT") or result.slots is None or not result.slots.required:
        return result.reward_answer
    return result.reward_answer + visual_lambda * result.slots.required_correct / result.slots.required
