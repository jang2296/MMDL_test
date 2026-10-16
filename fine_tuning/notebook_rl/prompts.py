"""Gold-free prompt builders; retry histories are joined by token ID elsewhere."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from .schema import ActorItem, Mode, Status, validate_actor

_TOKENS = {
    "turn1_plain": {"P0_BODY"}, "turn1_obs": {"P0_BODY", "OBSERVATION_INSTRUCTION"},
    "retry_incorrect": {"PREV_ANSWER", "OBS_SECTION"},
    "retry_invalid": {"REASON", "OBS_SECTION"}, "diagnose": {"K", "K_MINUS_1"},
    "oracle_facts": {"FACT_LINES", "P0_BODY"},
    "self_perception_probe": {"OBSERVED_TEXT", "P0_BODY"},
}
_PLACEHOLDER = re.compile(r"\{([A-Z_][A-Z_0-9]*)\}")


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError("prompts: duplicate or non-string key")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_prompts(path: Path) -> dict:
    raw = path.read_bytes()
    data = yaml.load(raw, Loader=_UniqueLoader)
    if type(data) is not dict or set(data) != {*_TOKENS, "observation_instruction"}:
        raise ValueError("prompts: invalid template keys")
    for key, tokens in _TOKENS.items():
        if (not isinstance(data[key], str) or not data[key].strip()
                or set(_PLACEHOLDER.findall(data[key])) != tokens):
            raise ValueError("prompts: invalid template placeholders")
    obs = data["observation_instruction"]
    if (type(obs) is not dict or not obs or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in obs.items()
    )):
        raise ValueError("prompts: invalid observation instructions")
    return data | {"sha256": hashlib.sha256(raw).hexdigest()}


def _render(template: str, **values: str) -> str:
    # Single template pass: literal placeholders inside user text cannot be substituted.
    return _PLACEHOLDER.sub(lambda m: values[m[1]], template)


def p0_body(question: str, choices: tuple[str, ...]) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("prompts: nonempty question required")
    if len(choices) > 10 or any(not isinstance(v, str) or not v.strip() for v in choices):
        raise ValueError("prompts: invalid choices")
    body = "Question: " + question + "\n"
    if choices:
        body += "Options:\n" + "".join(f"{chr(65 + i)}. {v}\n" for i, v in enumerate(choices))
        body += "Please select the correct answer from the options above.\n"
    return body


def _observation(mode: Mode, family: str, prompts: dict) -> str:
    if mode not in ("obs", "plain"):
        raise ValueError("prompts: unsupported mode")
    if mode == "plain":
        return ""
    if family not in prompts["observation_instruction"]:
        raise ValueError("prompts: missing family observation contract")
    return prompts["observation_instruction"][family]


def turn1_text(actor: ActorItem, prompts: dict) -> str:
    validate_actor(actor)
    return _render(prompts["turn1_" + actor.mode],
                   P0_BODY=p0_body(actor.question, actor.choices),
                   OBSERVATION_INSTRUCTION=_observation(actor.mode, actor.family, prompts))


def retry_text(status: Status, prev_answer: str | None, mode: Mode, family: str, prompts: dict) -> str:
    if status not in ("INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"):
        raise ValueError("prompts: this status cannot receive correctness retry feedback")
    obs = _observation(mode, family, prompts)
    section = "Observed facts:\n" + obs if obs else ""
    if status == "INCORRECT":
        if not isinstance(prev_answer, str) or not prev_answer.strip() or "\n" in prev_answer or "\r" in prev_answer:
            raise ValueError("prompts: incorrect status requires one submitted answer")
        return _render(prompts["retry_incorrect"], PREV_ANSWER=prev_answer, OBS_SECTION=section)
    reasons = {"INVALID_NO_FINAL": "no Final answer line was given",
               "INVALID_MULTIPLE": "more than one or conflicting final answers were given",
               "TRUNCATED": "the response was cut off at its output limit"}
    return _render(prompts["retry_invalid"], REASON=reasons[status], OBS_SECTION=section)


def diagnose_text(k: int, prompts: dict) -> str:
    if type(k) is not int or k not in (2, 3):
        raise ValueError("prompts: diagnosis requires a successful retry at turn 2 or 3")
    return _render(prompts["diagnose"], K=str(k), K_MINUS_1=str(k - 1))


def oracle_text(actor: ActorItem, fact_lines: tuple[str, ...], prompts: dict) -> str:
    validate_actor(actor)
    if not fact_lines or any(not isinstance(line, str) or not line.strip() for line in fact_lines):
        raise ValueError("prompts: oracle requires verified fact lines")
    return _render(prompts["oracle_facts"], FACT_LINES="\n".join(fact_lines),
                   P0_BODY=p0_body(actor.question, actor.choices))


def probe_text(observed_text: str, actor: ActorItem, prompts: dict) -> str:
    validate_actor(actor)
    if actor.choices or not isinstance(observed_text, str) or not observed_text.strip():
        raise ValueError("prompts: probe requires open question and nonempty observed text")
    return _render(prompts["self_perception_probe"], OBSERVED_TEXT=observed_text,
                   P0_BODY=p0_body(actor.question, actor.choices))


def user_message(text: str, n_images: int) -> dict:
    if not isinstance(text, str) or type(n_images) is not int or not 0 <= n_images <= 2:
        raise ValueError("prompts: text and 0..2 image placeholders required")
    return {"role": "user", "content": [*({"type": "image"} for _ in range(n_images)),
                                        {"type": "text", "text": text}]}
