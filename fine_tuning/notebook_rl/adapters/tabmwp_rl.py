"""Native train annotations as verifier gold, never native solution targets."""
from __future__ import annotations

from decimal import Decimal

from fine_tuning import tabmwp

from ..answers import number
from ..observations import slot_id
from ..schema import ActorItem, AnswerType, Lineage, Mode, Split, VerifierItem, to_json


def convert(native_problem: dict, pid: str, split: Split, image_rel: str, image_sha: str,
            mode: Mode) -> tuple[ActorItem, VerifierItem, Lineage]:
    if native_problem.get("split") != "train":
        raise ValueError("TabMWP RL: upstream train only, including internal dev/sealed")
    if not isinstance(pid, str) or not pid.isdecimal():
        raise ValueError("TabMWP RL: native decimal ID required")
    question, answer = native_problem.get("question"), native_problem.get("answer")
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError("TabMWP RL: missing question or native answer")
    question_type = native_problem.get("ques_type")
    native_choices = native_problem.get("choices")
    choices: tuple[str, ...] = ()
    kind: AnswerType
    accept: dict = {}
    if question_type == "multi_choice":
        if (not isinstance(native_choices, list) or not 2 <= len(native_choices) <= 10
                or any(not isinstance(c, str) or not c.strip() for c in native_choices)
                or len(set(native_choices)) != len(native_choices) or native_choices.count(answer) != 1):
            raise ValueError("TabMWP RL: invalid original choices/answer")
        choices = tuple(native_choices)
        answer = chr(65 + choices.index(answer))
        kind = "mcq"
    elif question_type == "free_text" and native_choices is None:
        native_type = native_problem.get("ans_type")
        if native_type == "integer_number":
            kind = "int"
            value = number(answer)
            if value is None or value.denominator != 1 or any(c in answer for c in ".eE/"):
                raise ValueError("TabMWP RL: malformed integer annotation")
        elif native_type in ("decimal_number", "fraction_number"):
            value = number(answer)
            if value is None:
                raise ValueError("TabMWP RL: malformed numeric annotation")
            if "/" in answer or native_type == "fraction_number":
                kind, accept = "fraction", {"accept_decimal": False}
            else:
                kind = "decimal"
                exponent = Decimal(answer.replace(",", "")).as_tuple().exponent
                assert isinstance(exponent, int)
                accept = {"round_to": max(0, -exponent)}
        elif native_type in ("extractive_text", "other_text", "boolean_text"):
            kind = "short_text"
            if native_type == "boolean_text":
                canonical = answer.strip().casefold()
                if canonical not in ("yes", "no", "true", "false"):
                    raise ValueError("TabMWP RL: unsupported boolean annotation")
                accept = {"aliases": ["true" if canonical == "yes" else "yes"]} if canonical in ("yes", "true") else {"aliases": ["false" if canonical == "no" else "no"]}
        else:
            raise ValueError("TabMWP RL: unsupported native answer type")
    else:
        raise ValueError("TabMWP RL: unsupported question/choice contract")

    facts = tabmwp.table_facts(native_problem)
    headers, rows = facts["headers"], facts["rows"]
    # Use visible labels only. Missing/duplicate labels need a separately specified
    # coordinate contract, not invented labels that the model cannot see.
    if (headers is None or len(rows) < 1 or len(headers) < 2
            or len({slot_id(x) for x in headers}) != len(headers)
            or len({slot_id(row[0]) for row in rows}) != len(rows)
            or any(not x.strip() or any(c in x for c in "[]\n\r") for x in [*headers, *(row[0] for row in rows)])):
        raise ValueError("TabMWP RL: table needs unique visible row/column labels")
    slots = {f"cell[{row[0]}][{header}]": cell
             for row in rows for header, cell in zip(headers, row, strict=True)}
    kinds = {key: "number" if number(value) is not None else "text" for key, value in slots.items()}
    problem_id = f"tabmwp-rl:{pid}"
    actor = ActorItem(problem_id, "tabmwp", "table", mode, question, choices, kind,
                      (image_rel,), (image_sha,))
    # The native data do not identify all causally necessary cells. An empty required
    # set prevents claiming a verified visual/nonvisual cause from guessed relevance.
    verifier = VerifierItem(problem_id, kind, answer, accept, choices, slots, kinds, (),
                            tuple(f"{key} = {value}" for key, value in slots.items()),
                            tuple([*([] if not facts["title"] else [facts["title"]]),
                                   *headers, *(cell for row in rows for cell in row)]))
    lineage = Lineage(problem_id, "tabmwp", tabmwp.REVISION, tabmwp.DATASET_LICENSE, "HUMAN",
                      f"{tabmwp.RAW_URL}data/tabmwp/tables/{pid}.png", "IXL", None, None,
                      "native-table-question", tabmwp.family_id(facts), split,
                      (f"{tabmwp.SOURCE_ID}:train:{pid}",))
    for record in (actor, verifier, lineage):
        to_json(record)
    return actor, verifier, lineage
