"""Authored financial charts with exact dual solvers; records are not training-approved."""
from __future__ import annotations

import random
from fractions import Fraction

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..schema import AnswerType, Mode, Split
from .common import GeneratedProblem, build, matches_split, png_bytes, structure_hash

GENERATOR_VERSION = "gen-chart-v1"
FAMILY = "chart_bar"
OBSERVATION_INSTRUCTION = (
    "List every bar as one line: bar[<series>][<label>] = <value>\n"
)
_KINDS = ("chart_bar", "chart_line", "table")
_TASKS = ("growth", "margin", "largest_difference", "largest_item")
_SERIES = ("Revenue", "Cost", "Forecast", "Budget")
_TITLE = "Quarterly results (thousands)"


def _select(index: int, seed: int, split: Split, style: str) -> dict:
    if type(index) is not int or index < 0 or type(seed) is not int or seed < 0:
        raise ValueError("chart: nonnegative integer index and seed required")
    # Validate the split/style pair before the bounded rejection loop.
    matches_split("0000000000000000", split, style)
    rng = random.Random(seed * 1_000_003 + index)
    for _ in range(1000):
        n_series, n_items, scale = rng.randint(2, 4), rng.randint(4, 8), rng.choice((1, 10))
        values = [[rng.randint(80 * scale, 200 * scale) for _ in range(n_items)]]
        values.append([rng.randint(10 * scale, value - 5 * scale) for value in values[0]])
        values.extend([[rng.randint(20 * scale, 220 * scale) for _ in range(n_items)]
                       for _ in range(n_series - 2)])
        structure = {"series": list(_SERIES[:n_series]),
                     "items": [f"P{i + 1}" for i in range(n_items)],
                     "values": values, "scale": scale}
        if matches_split(structure_hash(structure), split, style):
            return structure
    raise RuntimeError("chart: no structure in requested split after 1000 candidates")


def _array_answer(structure: dict, task: str) -> Fraction | str:
    revenue, cost = structure["values"][:2]
    if task == "growth":
        return Fraction(100 * (revenue[1] - revenue[0]), revenue[0])
    if task == "margin":
        return Fraction(100 * (revenue[0] - cost[0]), revenue[0])
    gaps = [abs(r - c) for r, c in zip(revenue, cost, strict=True)]
    if task == "largest_difference":
        return Fraction(max(gaps), structure["scale"])
    if task == "largest_item":
        return structure["items"][gaps.index(max(gaps))]
    raise ValueError("chart: unknown task")


def _pandas_answer(structure: dict, task: str) -> Fraction | str:
    frame = pd.DataFrame(
        {name: [Fraction(value, structure["scale"]) for value in column]
         for name, column in zip(structure["series"], structure["values"], strict=True)},
        index=structure["items"],
    )
    if task == "growth":
        return frame["Revenue"].pct_change(fill_method=None).iloc[1] * 100
    if task == "margin":
        return (1 - frame["Cost"].div(frame["Revenue"])).iloc[0] * 100
    gaps = frame["Revenue"].sub(frame["Cost"]).abs()
    if task == "largest_difference":
        return gaps.max()
    if task == "largest_item":
        return str(gaps.idxmax())  # pandas, like the explicit rule, picks the first tied row.
    raise ValueError("chart: unknown task")


def _decimal(value: Fraction) -> str:
    """Nearest tenth, with exact rational halfway cases rounded away from zero."""
    quotient, remainder = divmod(abs(value.numerator) * 10, value.denominator)
    quotient += 2 * remainder >= value.denominator
    sign = "-" if value < 0 and quotient else ""
    return f"{sign}{quotient // 10}.{quotient % 10}"


def _value(ticks: int, scale: int) -> str:
    return str(ticks) if scale == 1 else f"{ticks // 10}.{ticks % 10}"


def _question(task: str) -> str:
    if task == "growth":
        question = "For Revenue, what is the percentage change from P1 to P2?"
    elif task == "margin":
        question = ("For P1, what is the gross profit margin (%)? "
                    "Use (Revenue - Cost) / Revenue * 100.")
    elif task == "largest_difference":
        question = ("What is the largest absolute difference between Revenue and Cost "
                    "over all displayed categories, in thousands?")
    else:
        return ("Which category has the largest absolute difference between Revenue and Cost? "
                "If tied, choose the earliest category in the displayed order. "
                "Answer with its category label.")
    return question + " Round to one decimal place; round exact halfway cases away from zero."


def _choices(answer: Fraction, structure: dict, seed: int) -> tuple[tuple[str, ...], str]:
    revenue, cost = structure["values"][:2]
    scale = structure["scale"]
    # Sign reversal, missing percent conversion, wrong denominator, adjacent cells, scale errors.
    mistakes = [-answer, answer / 100, answer * 100, answer / 2, answer * 2,
                Fraction(revenue[1] - revenue[0], scale),
                Fraction(100 * (revenue[1] - revenue[0]), revenue[1]),
                Fraction(revenue[0], scale), Fraction(cost[0], scale)]
    correct = _decimal(answer)
    options = [correct]
    for candidate in mistakes:
        text = _decimal(candidate)
        if text not in options:
            options.append(text)
    rng = random.Random(seed)
    while len(options) < 10:
        text = _decimal(answer + Fraction(rng.choice((-1, 1)) * rng.randint(1, 1000), 10))
        if text not in options:
            options.append(text)
    options = options[:10]
    rng.shuffle(options)
    return tuple(options), chr(65 + options.index(correct))


def _slot(family: str, series: str, item: str) -> str:
    if family == "chart_bar":
        return f"bar[{series}][{item}]"
    if family == "chart_line":
        return f"point[{series}][{item}]"
    return f"cell[{item}][{series}]"


def _render(structure: dict, family: str, style: str) -> tuple[bytes, tuple[str, ...]]:
    shifted = style == "shift"
    font = "DejaVu Serif" if shifted else "DejaVu Sans"
    background = "#fff9ed" if shifted else "white"
    colors = ("#932648", "#327b45", "#bc7c14", "#5d5198") if shifted else (
        "#377eb8", "#ff7f00", "#4daf4a", "#984ea3")
    figure = Figure(figsize=(10, 7) if shifted else (12, 6), facecolor=background)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    ax.set_facecolor(background)
    ax.set_title(_TITLE, fontfamily=font, fontsize=15, pad=24)
    series, items, values, scale = (structure[k] for k in ("series", "items", "values", "scale"))
    labels = [[_value(v, scale) for v in column] for column in values]
    printed = [_TITLE, *series, *items, *(value for column in labels for value in column)]
    if family == "table":
        ax.set_axis_off()
        cells = [[labels[j][i] for j in range(len(series))] for i in range(len(items))]
        table = ax.table(cellText=cells, rowLabels=items, colLabels=series,
                         loc="center", cellLoc="center", colLoc="center",
                         bbox=[0.12, 0.06, 0.86, 0.85])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        for (row, column), cell in table.get_celld().items():
            cell.get_text().set_fontfamily(font)
            cell.set_linewidth(1.7 if shifted else 0.8)
            cell.set_facecolor("#eee1c8" if shifted else "#e8f0f8"
                               if row == 0 or column == -1 else background)
    else:
        n_series = len(series)
        annotations: list[list[tuple[float, float, str, str]]] = [[] for _ in items]
        for j, name in enumerate(series):
            ys = [v / scale for v in values[j]]
            if family == "chart_bar":
                width = 0.8 / n_series
                xs = [i + (j - (n_series - 1) / 2) * width for i in range(len(items))]
                ax.bar(xs, ys, width=width * 0.9, label=name, color=colors[j],
                       edgecolor="black", linewidth=1.2 if shifted else 0.4)
            else:
                xs = list(range(len(items)))
                ax.plot(xs, ys, marker=("o", "s", "^", "D")[j], label=name,
                        color=colors[j], linewidth=2.8 if shifted else 1.5,
                        linestyle="--" if shifted else "-")
            for i, (x, y, label) in enumerate(zip(xs, ys, labels[j], strict=True)):
                annotations[i].append((x, y, label, colors[j]))
        maximum = max(max(column) for column in values) / scale
        top = ((int(maximum) // 50) + 2) * 50
        for category in annotations:
            previous_y = -float("inf")
            for x, y, label, color in sorted(category, key=lambda item: item[1]):
                label_y = max(y + top * 0.025, previous_y + top * 0.05)
                ax.annotate(label, (x, y), xytext=(x, label_y), textcoords="data",
                            ha="center", va="bottom", fontsize=9, fontfamily=font,
                            bbox={"facecolor": background, "edgecolor": "none", "pad": 0.4},
                            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.6})
                previous_y = label_y
        ticks = list(range(0, top + 1, 50))
        ax.set_ylim(0, top)
        ax.set_yticks(ticks, [str(tick) for tick in ticks], fontfamily=font)
        ax.set_xticks(range(len(items)), items, fontfamily=font)
        ax.set_ylabel("Value (thousands)", fontfamily=font)
        ax.set_xlabel("Category", fontfamily=font)
        ax.legend(loc="upper left", ncols=2 if shifted else n_series,
                  prop={"family": font, "size": 10}, framealpha=1)
        printed.extend(["Value (thousands)", "Category", *map(str, ticks)])
    figure.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.12)
    return png_bytes(figure, dpi=110 if shifted else 100), tuple(printed)


def generate(index: int, seed: int, split: Split, mode: Mode,
             style: str = "default") -> GeneratedProblem:
    if mode not in ("obs", "plain"):
        raise ValueError("chart: unsupported mode")
    structure = _select(index, seed, split, style)
    family, task = _KINDS[index % 3], _TASKS[(index // 3) % 4]
    answer = _array_answer(structure, task)
    if answer != _pandas_answer(structure, task):
        raise ValueError("chart: independent solvers disagree")
    answer_type: AnswerType = "short_text" if isinstance(answer, str) else "decimal"
    choices: tuple[str, ...] = ()
    gold = answer if isinstance(answer, str) else _decimal(answer)
    if not isinstance(answer, str) and index % 5 < 2:
        choices, gold = _choices(answer, structure, int(structure_hash(structure), 16))
        answer_type = "mcq"
    slots, required = {}, []
    for j, name in enumerate(structure["series"]):
        for i, item in enumerate(structure["items"]):
            key = _slot(family, name, item)
            slots[key] = _value(structure["values"][j][i], structure["scale"])
            needed = ((task == "growth" and j == 0 and i < 2)
                      or (task == "margin" and j < 2 and i == 0)
                      or (task in ("largest_difference", "largest_item") and j < 2))
            if needed:
                required.append(key)
    image, printed = _render(structure, family, style)
    return build(source=GENERATOR_VERSION, family=family, index=index, split=split, mode=mode,
                 question=_question(task), choices=choices, answer_type=answer_type, gold=gold,
                 accept={},
                 slots=slots, slot_types={key: "number" for key in slots},
                 required_slots=required, oracle_fact_lines=[f"{k} = {v}" for k, v in slots.items()],
                 printed_text=printed, structure=structure, template_id=task,
                 image_bytes=(image,), generator_version=GENERATOR_VERSION, seed=seed, style=style)


def self_check(p: GeneratedProblem) -> None:
    """Recompute both solvers and rerender, then check the complete immutable record."""
    if not isinstance(p, GeneratedProblem):
        raise ValueError("chart: expected GeneratedProblem")
    if (p.lineage.generator_version != GENERATOR_VERSION or p.lineage.generator_seed is None
            or p.actor.source != GENERATOR_VERSION):
        raise ValueError("chart: wrong generator identity")
    try:
        index = int(p.actor.problem_id.split(":")[-2])
        style = p.lineage.image_provenance.rsplit(":", 1)[-1]
        expected = generate(index, p.lineage.generator_seed, p.lineage.split, p.actor.mode, style)
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError("chart: invalid generation identity") from exc
    if p != expected:
        raise ValueError("chart: render, answer, slots or lineage mismatch")
