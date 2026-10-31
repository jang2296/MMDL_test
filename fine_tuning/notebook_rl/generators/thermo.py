"""Authored thermodynamics/energy problems with exact solvers; not training-approved."""
from __future__ import annotations

import random
from fractions import Fraction

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrow, Rectangle

from ..schema import Mode, Split
from .common import GeneratedProblem, build, matches_split, png_bytes, structure_hash

GENERATOR_VERSION = "gen-thermo-v1"
FAMILY = "thermo"
OBSERVATION_INSTRUCTION = (
    "For a P-V diagram list each state as state[<label>] = <volume in m^3>,<pressure in kPa> "
    "(numbers only, volume with one decimal, comma without spaces); for an energy-flow diagram "
    "list each arrow as flow[<label as printed>] = <power with unit>; for a demand chart list "
    "each hourly step as power[<start hour>-<end hour>] = <power with unit>.\n"
)
_TASKS = ("cycle_work", "sankey_efficiency", "load_energy")
_QUESTIONS = {
    "cycle_work": ("The diagram shows a cycle of an ideal gas. What is the net work done BY the gas "
                   "during one cycle, in kJ? (Positive if the gas does net work.)"),
    "sankey_efficiency": "What is the efficiency of the device, in percent?",
    "load_energy": "How much electrical energy is consumed over the whole period shown, in kWh?",
}
_ROUNDING = " Round to one decimal place."
_GRID = tuple((v, p) for v in range(1, 11) for p in range(100, 901, 50))  # (V in 0.1 m^3, P in kPa)
_LOSS_NAMES = ("Heat loss", "Friction", "Exhaust gas", "Radiation", "Electrical loss",
               "Noise and vibration")
_CYCLE_TITLE = "Ideal-gas cycle on a P-V diagram"
_SANKEY_TITLE = "Energy flow through a device"
_LOAD_TITLE = "Electric power demand"


# --- exact solvers ---------------------------------------------------------------------------

def _points(structure: dict) -> list[tuple[Fraction, Fraction]]:
    return [(Fraction(v, 10), Fraction(p)) for _, v, p in structure["states"]]


def _work_shoelace(structure: dict) -> Fraction:
    """Work by the gas = -(signed counter-clockwise shoelace area in the V-P plane), kPa*m^3 = kJ."""
    pts = _points(structure)
    twice_area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))
    return -twice_area / 2


def _work_legs(structure: dict) -> Fraction:
    """Independent check: closed integral of P dV as trapezoids under each leg."""
    pts = _points(structure)
    return sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))


def _efficiency(structure: dict) -> Fraction:
    return Fraction(structure["useful"] * 100, structure["input"])


def _efficiency_from_losses(structure: dict) -> Fraction:
    return 100 - Fraction(sum(kw for _, kw in structure["losses"]) * 100, structure["input"])


def _energy(structure: dict) -> int:
    return sum(structure["values"])  # one-hour intervals: kW * 1 h = kWh


def _rounded(value: Fraction) -> str:
    q, r = divmod(abs(value.numerator) * 10, value.denominator)
    q += r * 2 >= value.denominator
    return f"{'-' if value < 0 and q else ''}{q // 10}.{q % 10}"


def _tie(value: Fraction) -> bool:
    return (abs(value) * 10) % 1 == Fraction(1, 2)


# --- candidate structures --------------------------------------------------------------------

def _cross(o, a, b) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_meet(p1, p2, p3, p4) -> bool:
    """Closed-segment intersection, so touching and overlapping count as meeting."""
    def within(a, b, c) -> bool:  # c collinear with a-b
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    d1, d2, d3, d4 = _cross(p3, p4, p1), _cross(p3, p4, p2), _cross(p1, p2, p3), _cross(p1, p2, p4)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    return ((d1 == 0 and within(p3, p4, p1)) or (d2 == 0 and within(p3, p4, p2))
            or (d3 == 0 and within(p1, p2, p3)) or (d4 == 0 and within(p1, p2, p4)))


def _simple_polygon(points: list[tuple[int, int]]) -> bool:
    """Distinct vertices, no collinear consecutive triple, no touching/crossing of opposite edges."""
    n = len(points)
    if len(set(points)) != n or any(_cross(points[i], points[(i + 1) % n], points[(i + 2) % n]) == 0
                                    for i in range(n)):
        return False
    edges = [(points[i], points[(i + 1) % n]) for i in range(n)]
    return not any(_segments_meet(*edges[i], *edges[j])
                   for i in range(n) for j in range(i + 2, n) if (j + 1) % n != i)


def _draw_cycle(rng: random.Random) -> dict | None:
    points = rng.sample(_GRID, rng.randint(3, 4))
    if not _simple_polygon(points):
        return None
    return {"task": "cycle_work", "states": [[label, v, p] for label, (v, p) in zip("ABCD", points)]}


def _draw_sankey(rng: random.Random) -> dict | None:
    total, count = rng.randint(100, 2000), rng.randint(1, 3)
    cuts = sorted(rng.sample(range(1, total), count))
    parts = [b - a for a, b in zip([0, *cuts], [*cuts, total])]
    if min(parts) * 20 < total:  # every arrow keeps at least 5 percent of the input
        return None
    rng.shuffle(parts)
    names = rng.sample(_LOSS_NAMES, count)
    structure = {"task": "sankey_efficiency", "input": total, "useful": parts[0],
                 "losses": [[name, kw] for name, kw in zip(names, parts[1:])]}
    return None if _tie(_efficiency(structure)) else structure


def _draw_load(rng: random.Random) -> dict | None:
    values = [rng.randint(5, 80) for _ in range(rng.randint(6, 12))]
    return None if len(set(values)) == 1 else {"task": "load_energy", "values": values}


_DRAW = {"cycle_work": _draw_cycle, "sankey_efficiency": _draw_sankey, "load_energy": _draw_load}


def _select(index: int, seed: int, split: Split, style: str) -> dict:
    if type(index) is not int or index < 0 or type(seed) is not int or seed < 0:
        raise ValueError("thermo: nonnegative integer index and seed required")
    matches_split("0000000000000000", split, style)
    rng = random.Random(seed * 1_000_003 + index)
    draw = _DRAW[_TASKS[index % len(_TASKS)]]
    for _ in range(1000):
        structure = draw(rng)
        if structure is not None and matches_split(structure_hash(structure), split, style):
            return structure
    raise RuntimeError("thermo: no structure in requested split after 1000 candidates")


# --- answer and observation slots ------------------------------------------------------------

def _answer(structure: dict) -> tuple[str, str, dict]:
    task = structure["task"]
    if task == "cycle_work":
        work = _work_shoelace(structure)
        if work != _work_legs(structure):
            raise ValueError("thermo: independent solvers disagree")
        if work == 0 or _tie(work):
            raise ValueError("thermo: degenerate cycle")
        return _rounded(work), "decimal", {"round_to": 1}
    if task == "sankey_efficiency":
        efficiency = _efficiency(structure)
        if efficiency != _efficiency_from_losses(structure):
            raise ValueError("thermo: independent solvers disagree")
        if not 0 < efficiency < 100 or _tie(efficiency):
            raise ValueError("thermo: degenerate energy balance")
        return _rounded(efficiency), "decimal", {"round_to": 1}
    return str(_energy(structure)), "int", {}


def _slots(structure: dict) -> dict[str, str]:
    task = structure["task"]
    if task == "cycle_work":
        return {f"state[{label}]": f"{v / 10:.1f},{p}" for label, v, p in structure["states"]}
    if task == "sankey_efficiency":
        return {"flow[Input]": f"{structure['input']} kW",
                "flow[Useful output]": f"{structure['useful']} kW",
                **{f"flow[{name}]": f"{kw} kW" for name, kw in structure["losses"]}}
    return {f"power[{i}-{i + 1}]": f"{kw} kW" for i, kw in enumerate(structure["values"])}


# --- rendering -------------------------------------------------------------------------------

def _look(style: str) -> tuple[bool, str, str]:
    shifted = style == "shift"
    return shifted, "DejaVu Serif" if shifted else "DejaVu Sans", "#fff9ed" if shifted else "white"


def _render_cycle(structure: dict, style: str) -> tuple[bytes, tuple[str, ...]]:
    shifted, font, background = _look(style)
    line, arrow = ("#8a2244", "#157b64") if shifted else ("#1f3f8f", "#b22222")
    states = structure["states"]
    n = len(states)
    xs, ys = [v / 10 for _, v, _ in states], [p for _, _, p in states]
    figure = Figure(figsize=(12, 6) if shifted else (13, 6), facecolor=background)
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0.07, 0.14, 0.58, 0.76), facecolor=background)
    ax.grid(True, color="#b5b5b5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.fill(xs, ys, color=line, alpha=0.1)
    ax.plot(xs + xs[:1], ys + ys[:1], color=line, linewidth=3 if shifted else 2.2)
    for i in range(n):
        x1, y1, x2, y2 = xs[i], ys[i], xs[(i + 1) % n], ys[(i + 1) % n]
        ax.annotate("", xy=(x1 + 0.65 * (x2 - x1), y1 + 0.65 * (y2 - y1)),
                    xytext=(x1 + 0.35 * (x2 - x1), y1 + 0.35 * (y2 - y1)),
                    arrowprops={"arrowstyle": "-|>", "color": arrow, "linewidth": 2.4,
                                "mutation_scale": 26, "shrinkA": 0, "shrinkB": 0})
    ax.plot(xs, ys, "o", color="black", markersize=7)
    cx, cy = sum(xs) / n, sum(ys) / n
    for (label, _, _), x, y in zip(states, xs, ys):
        ax.annotate(label, (x, y), xytext=(15 if x >= cx else -15, 15 if y >= cy else -15),
                    textcoords="offset points", ha="center", va="center", fontsize=16,
                    fontweight="bold", fontfamily=font)
    x_ticks, y_ticks = [f"{i / 10:.1f}" for i in range(11)], [str(i) for i in range(0, 1001, 100)]
    ax.set_xticks([i / 10 for i in range(11)], x_ticks, fontfamily=font)
    ax.set_yticks(range(0, 1001, 100), y_ticks, fontfamily=font)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1000)
    ax.set_xlabel("Volume V (m³)", fontfamily=font, fontsize=13)
    ax.set_ylabel("Pressure P (kPa)", fontfamily=font, fontsize=13)
    ax.set_title(_CYCLE_TITLE, fontfamily=font, fontsize=16)
    header = f"{'State':<8}{'V (m³)':<9}{'P (kPa)'}"
    rows = [f"{label:<8}{v / 10:<9.1f}{p}" for label, v, p in states]
    note = "Arrows show the direction of travel."
    for i, text in enumerate([header, *rows]):
        figure.text(0.68, 0.78 - 0.08 * i, text, family="DejaVu Sans Mono", fontsize=14,
                    fontweight="bold" if i == 0 else "normal")
    figure.text(0.68, 0.78 - 0.08 * (n + 1), note, fontfamily=font, fontsize=12)
    printed = (_CYCLE_TITLE, "Volume V (m³)", "Pressure P (kPa)", header, *rows, note,
               *(label for label, _, _ in states), *x_ticks, *y_ticks)
    return png_bytes(figure, dpi=110 if shifted else 100), printed


def _render_sankey(structure: dict, style: str) -> tuple[bytes, tuple[str, ...]]:
    shifted, font, background = _look(style)
    block, useful_color, loss_color = (("#e7d9b8", "#157b64", "#8a2244") if shifted
                                       else ("#dde6f3", "#2a8a3e", "#b22222"))
    total = structure["input"]
    figure = Figure(figsize=(12, 7.7), facecolor=background)
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0.02, 0.02, 0.96, 0.88), facecolor=background)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((4.2, 2.5), 5, 4, facecolor=block, edgecolor="black", linewidth=2))
    ax.text(6.7, 4.5, "Device", ha="center", va="center", fontsize=18, fontfamily=font)
    printed = [_SANKEY_TITLE, "Device"]

    def flow(x, y, dx, dy, kw, color, label, label_x, label_y=None):
        width = max(0.12, 1.6 * kw / total)
        head = 1.5 * width + 0.25
        ax.add_patch(FancyArrow(x, y, dx, dy, width=width, head_width=head, head_length=0.7,
                                length_includes_head=True, color=color, linewidth=0))
        text = f"{label}: {kw} kW"
        if label_y is None:  # horizontal arrows carry their label just above the head
            ax.text(label_x, y + head / 2 + 0.15, text, ha="center", va="bottom", fontsize=12,
                    fontfamily=font)
        else:
            ax.text(label_x, label_y, text, ha="center", va="bottom" if dy > 0 else "top",
                    fontsize=12, fontfamily=font)
        printed.append(text)

    flow(0.3, 4.5, 3.9, 0, total, "#555555", "Input", 2.25)
    flow(9.2, 4.5, 4.5, 0, structure["useful"], useful_color, "Useful output", 11.45)
    layout = {1: [(6.7, 1)], 2: [(5.2, 1), (8.2, -1)], 3: [(5.2, 1), (8.2, 1), (6.7, -1)]}
    for (name, kw), (x, side) in zip(structure["losses"], layout[len(structure["losses"])]):
        if side > 0:
            flow(x, 6.5, 0, 1.9, kw, loss_color, name, x, 8.55)
        else:
            flow(x, 2.5, 0, -1.9, kw, loss_color, name, x, 0.45)
    figure.text(0.5, 0.95, _SANKEY_TITLE, ha="center", fontfamily=font, fontsize=16)
    return png_bytes(figure, dpi=100 if shifted else 90), tuple(printed)


def _render_load(structure: dict, style: str) -> tuple[bytes, tuple[str, ...]]:
    shifted, font, background = _look(style)
    fill, edge = ("#157b64", "#0d4d3e") if shifted else ("#6aa9e0", "#1f3f8f")
    values = structure["values"]
    n = len(values)
    figure = Figure(figsize=(10, 6.6) if shifted else (11, 6), facecolor=background)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111, facecolor=background)
    ax.grid(True, color="#b5b5b5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.stairs(values, list(range(n + 1)), fill=True, facecolor=fill, edgecolor=edge, linewidth=2)
    labels = [f"{kw} kW" for kw in values]
    for i, (kw, text) in enumerate(zip(values, labels)):
        ax.text(i + 0.5, kw + 1.5, text, ha="center", va="bottom", fontsize=11, fontfamily=font)
    x_ticks, y_ticks = [str(i) for i in range(n + 1)], [str(i) for i in range(0, 91, 10)]
    ax.set_xticks(range(n + 1), x_ticks, fontfamily=font)
    ax.set_yticks(range(0, 91, 10), y_ticks, fontfamily=font)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 95)
    ax.set_xlabel("Time (h)", fontfamily=font, fontsize=13)
    ax.set_ylabel("Power demand (kW)", fontfamily=font, fontsize=13)
    ax.set_title(_LOAD_TITLE, fontfamily=font, fontsize=16)
    printed = (_LOAD_TITLE, "Time (h)", "Power demand (kW)", *labels, *x_ticks, *y_ticks)
    return png_bytes(figure, dpi=110 if shifted else 100), printed


_RENDER = {"cycle_work": _render_cycle, "sankey_efficiency": _render_sankey,
           "load_energy": _render_load}


# --- public contract -------------------------------------------------------------------------

def generate(index: int, seed: int, split: Split, mode: Mode,
             style: str = "default") -> GeneratedProblem:
    if mode not in ("obs", "plain"):
        raise ValueError("thermo: unsupported mode")
    structure = _select(index, seed, split, style)
    task = structure["task"]
    gold, answer_type, accept = _answer(structure)
    question = _QUESTIONS[task] + (_ROUNDING if answer_type == "decimal" else "")
    slots = _slots(structure)
    image, printed = _RENDER[task](structure, style)
    return build(source=GENERATOR_VERSION, family=FAMILY, index=index, split=split, mode=mode,
                 question=question, choices=(), answer_type=answer_type, gold=gold,
                 accept=accept, slots=slots, slot_types={key: "text" if task == "cycle_work" else "number"
                                                         for key in slots},
                 required_slots=tuple(slots),
                 oracle_fact_lines=tuple(f"{k} = {v}" for k, v in slots.items()),
                 printed_text=printed, structure=structure, template_id=task,
                 image_bytes=(image,), generator_version=GENERATOR_VERSION, seed=seed, style=style)


def self_check(p: GeneratedProblem) -> None:
    """Rebuild solutions, records, and PNG bytes from the generation identity."""
    if not isinstance(p, GeneratedProblem):
        raise ValueError("thermo: expected GeneratedProblem")
    if (p.lineage.generator_version != GENERATOR_VERSION or p.lineage.generator_seed is None
            or p.actor.source != GENERATOR_VERSION):
        raise ValueError("thermo: wrong generator identity")
    try:
        index = int(p.actor.problem_id.split(":")[-2])
        style = p.lineage.image_provenance.rsplit(":", 1)[-1]
        expected = generate(index, p.lineage.generator_seed, p.lineage.split, p.actor.mode, style)
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError("thermo: invalid generation identity") from exc
    if p != expected:
        raise ValueError("thermo: render, answer, slots or lineage mismatch")
