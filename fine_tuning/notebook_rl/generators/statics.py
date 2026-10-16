"""Authored beam statics with independent exact solvers; not training-approved."""
from __future__ import annotations

import random
from fractions import Fraction

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from sympy import Rational, SingularityFunction, diff, expand, solve
from sympy.physics.continuum_mechanics.beam import Beam

from ..schema import Mode, Split
from .common import GeneratedProblem, build, matches_split, png_bytes, structure_hash

GENERATOR_VERSION = "gen-statics-v1"
FAMILY = "statics"
OBSERVATION_INSTRUCTION = (
    "List the span as span = <length>; each load as load[<index>] = <magnitude>@<position>; "
    "each support as support[<label>] = <type>@<position>. "
    "Include units and distributed load start/end positions.\n"
)
_TASKS = ("reaction", "max_moment", "max_location")
_TITLE = "Beam loading diagram"
_SIGN = "Signed loads: + downward; - upward."


def _select(index: int, seed: int, split: Split, style: str) -> dict:
    if type(index) is not int or index < 0 or type(seed) is not int or seed < 0:
        raise ValueError("statics: nonnegative integer index and seed required")
    matches_split("0000000000000000", split, style)
    rng = random.Random(seed * 1_000_003 + index)
    for _ in range(1000):
        length = rng.randint(6, 16)
        positions = sorted(rng.sample(range(1, length), rng.randint(1, 3)))
        loads = [[position, rng.choice((-1, 1)) * rng.randint(2, 30)] for position in positions]
        uniform = None
        if rng.getrandbits(1):
            start, end = sorted(rng.sample(range(length + 1), 2))
            uniform = [start, end, rng.choice((-1, 1)) * rng.randint(1, 8)]
        structure = {"kind": rng.choice(("simple", "cantilever")), "length": length,
                     "loads": loads, "uniform": uniform}
        if structure["kind"] == "simple":
            reactions = _equilibrium(structure)
            if reactions["reaction"] < 0 or reactions["reaction_b"] < 0:
                continue  # Ordinary drawn supports must retain contact under the signed loads.
        if matches_split(structure_hash(structure), split, style):
            return structure
    raise RuntimeError("statics: no structure in requested split after 1000 candidates")


def _breaks(structure: dict) -> list[Fraction]:
    points = {Fraction(0), Fraction(structure["length"]),
              *(Fraction(position) for position, _ in structure["loads"])}
    if structure["uniform"] is not None:
        points.update(map(Fraction, structure["uniform"][:2]))
    return sorted(points)


def _equilibrium(structure: dict) -> dict[str, Fraction]:
    length, loads, uniform = structure["length"], structure["loads"], structure["uniform"]
    total_force = Fraction(sum(force for _, force in loads))
    total_moment = Fraction(sum(position * force for position, force in loads))
    if uniform is not None:
        start, end, intensity = uniform
        total_force += intensity * (end - start)
        total_moment += Fraction(intensity * (end - start) * (start + end), 2)
    right = total_moment / length if structure["kind"] == "simple" else Fraction(0)
    left = total_force - right
    fixed_moment = total_moment if structure["kind"] == "cantilever" else Fraction(0)

    def moment(x: Fraction) -> Fraction:
        value = left * x - fixed_moment
        value -= sum(force * max(x - position, 0) for position, force in loads)
        if uniform is not None:
            start, end, intensity = uniform
            value -= Fraction(intensity, 2) * (max(x - start, 0)**2 - max(x - end, 0)**2)
        return value

    candidates = set(_breaks(structure))
    if uniform is not None:
        start, end, intensity = uniform
        for a, b in zip(_breaks(structure), _breaks(structure)[1:]):
            if start <= a < end:
                shear = left - sum(force for position, force in loads if position <= a)
                shear -= intensity * max(0, min(a - start, end - start))
                root = a + shear / intensity
                if a < root < b:
                    candidates.add(root)
    magnitudes = {x: abs(moment(x)) for x in candidates}
    maximum = max(magnitudes.values())
    return {"reaction": left, "reaction_b": right, "max_moment": maximum,
            "max_location": min(x for x, value in magnitudes.items() if value == maximum)}


def _fraction(value) -> Fraction:
    if value.is_Rational is not True:
        raise ValueError("statics: expected exact rational SymPy result")
    numerator, denominator = value.as_numer_denom()
    return Fraction(int(numerator), int(denominator))


def _sympy(structure: dict) -> dict[str, Fraction]:
    length = structure["length"]
    beam = Beam(length, 1, 1)
    fixed = structure["kind"] == "cantilever"
    at_a = beam.apply_support(0, "fixed" if fixed else "pin")
    reactions = list(at_a) if fixed else [at_a]
    at_b = None
    if not fixed:
        at_b = beam.apply_support(length, "roller")
        reactions.append(at_b)
    for position, force in structure["loads"]:
        beam.apply_load(-force, position, -1)
    if structure["uniform"] is not None:
        start, end, intensity = structure["uniform"]
        beam.apply_load(-intensity, start, 0, end=end)
    beam.solve_for_reaction_loads(*reactions)
    expression, x = beam.bending_moment(), beam.variable
    cuts = _breaks(structure)
    magnitudes: dict[Fraction, Fraction] = {}
    # Solve each polynomial interval explicitly; Beam.max_bmoment's generic limit path is slow.
    for left, right in zip(cuts, cuts[1:]):
        a, b = Rational(left.numerator, left.denominator), Rational(right.numerator, right.denominator)
        mid = (a + b) / 2
        replacements = {sf: (x - sf.args[1])**sf.args[2] if sf.args[1] < mid else 0
                        for sf in expression.atoms(SingularityFunction)}
        polynomial = expand(expression.xreplace(replacements))
        positions = [a, b, *(root for root in solve(diff(polynomial, x), x) if a < root < b)]
        for position in positions:
            magnitudes[_fraction(position)] = abs(_fraction(polynomial.subs(x, position)))
    maximum = max(magnitudes.values())
    return {"reaction": _fraction(beam.reaction_loads[reactions[0]]),
            "reaction_b": Fraction(0) if fixed else _fraction(beam.reaction_loads[at_b]),
            "max_moment": maximum,
            "max_location": min(p for p, value in magnitudes.items() if value == maximum)}


def _rounded(value: Fraction) -> str:
    q, r = divmod(abs(value.numerator) * 10, value.denominator)
    q += r * 2 >= value.denominator
    return f"{'-' if value < 0 and q else ''}{q // 10}.{q % 10}"


def _question(task: str) -> tuple[str, str, list[str]]:
    if task == "reaction":
        question = "What is the vertical support reaction at A? Take upward as positive."
        unit, aliases = "kN", ["kN"]
    elif task == "max_moment":
        question = "What is the maximum absolute bending moment anywhere along the beam?"
        unit, aliases = "kN*m", ["kN*m", "kN m", "kNm"]
    else:
        question = ("At what distance x from A does the absolute bending moment attain its maximum? "
                    "If several positions tie, give the smallest x.")
        unit, aliases = "m", ["m"]
    question += (f" Ignore beam self-weight. Answer in {unit}, rounded to one decimal place; "
                 "round exact halfway cases away from zero. Include the unit.")
    return question, unit, aliases


def _slots(structure: dict) -> dict[str, str]:
    length = structure["length"]
    slots = {"span": f"{length} m",
             "support[A]": f"{'fixed' if structure['kind'] == 'cantilever' else 'pin'}@0 m"}
    if structure["kind"] == "simple":
        slots["support[B]"] = f"roller@{length} m"
    for i, (position, force) in enumerate(structure["loads"], 1):
        slots[f"load[{i}]"] = f"{force} kN@{position} m"
    if structure["uniform"] is not None:
        start, end, intensity = structure["uniform"]
        slots["load[U]"] = f"{intensity} kN/m@{start} m..{end} m"
    return slots


def _render(structure: dict, style: str) -> tuple[bytes, tuple[str, ...]]:
    shifted = style == "shift"
    font = "DejaVu Serif" if shifted else "DejaVu Sans"
    background = "#fff9ed" if shifted else "white"
    point_color, uniform_color = (("#8a2244", "#157b64") if shifted else ("#b22222", "#2266aa"))
    figure = Figure(figsize=(10, 6) if shifted else (12, 5), facecolor=background)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    ax.set_facecolor(background)
    length = structure["length"]
    ax.plot([0, length], [0, 0], color="#404040", linewidth=6 if shifted else 4)
    printed = [_TITLE, _SIGN, "x (m)"]
    for position, label, kind in ((0, "A", "fixed" if structure["kind"] == "cantilever" else "pin"),
                                   *(([(length, "B", "roller")]) if structure["kind"] == "simple" else [])):
        if kind == "fixed":
            ax.plot([position, position], [-0.6, 0.6], color="black", linewidth=5)
            for y in (-0.5, -0.25, 0, 0.25, 0.5):
                ax.plot([-0.3, 0], [y - 0.2, y], color="black", linewidth=1)
        else:
            ax.add_patch(Polygon([(position, -0.04), (position - 0.3, -0.5),
                                  (position + 0.3, -0.5)], closed=True,
                                 facecolor="#d0d0d0", edgecolor="black"))
            if kind == "roller":
                ax.plot([position - 0.15, position + 0.15], [-0.6, -0.6],
                        "o", color="black", markersize=5)
        text = f"{label} ({kind})"
        ax.text(position, -0.9, text, ha="center", fontsize=11, fontfamily=font)
        printed.append(text)
    for i, (position, force) in enumerate(structure["loads"], 1):
        side = 1 if force > 0 else -1
        tail = side * (1.15 + 0.45 * (i - 1))
        ax.annotate("", (position, side * 0.03), xytext=(position, tail),
                    arrowprops={"arrowstyle": "-|>", "color": point_color,
                                "linewidth": 2.4 if shifted else 1.6})
        label = f"F{i}: {force} kN"
        ax.text(position, tail + side * 0.14, label, ha="center",
                va="bottom" if side > 0 else "top", color=point_color,
                fontsize=11, fontfamily=font)
        printed.append(label)
    if structure["uniform"] is not None:
        start, end, intensity = structure["uniform"]
        side = 1 if intensity > 0 else -1
        for position in [start + (end - start) * i / 8 for i in range(9)]:
            ax.annotate("", (position, side * 0.04), xytext=(position, side * 0.75),
                        arrowprops={"arrowstyle": "->", "color": uniform_color,
                                    "linewidth": 1.5 if shifted else 1})
        ax.plot([start, end], [side * 0.75] * 2, color=uniform_color, linewidth=1.5)
        label = f"UDL: {intensity} kN/m, x={start}..{end} m"
        ax.text((start + end) / 2, side * 3.1, label, ha="center", va="center",
                color=uniform_color, fontsize=12, fontfamily=font)
        printed.append(label)
    ax.set_title(_TITLE, fontfamily=font, fontsize=16)
    ax.set_xlim(-0.8, length + 0.8)
    ax.set_ylim(-3.7, 3.7)
    ax.set_xticks(range(length + 1), [str(i) for i in range(length + 1)], fontfamily=font)
    ax.set_xlabel("x (m)", fontfamily=font, fontsize=12)
    ax.set_yticks([])
    for name in ("left", "right", "top"):
        ax.spines[name].set_visible(False)
    printed.extend(str(i) for i in range(length + 1))
    figure.text(0.5, 0.04, _SIGN, ha="center", fontfamily=font, fontsize=11)
    figure.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.2)
    return png_bytes(figure, dpi=110 if shifted else 100), tuple(printed)


def generate(index: int, seed: int, split: Split, mode: Mode,
             style: str = "default") -> GeneratedProblem:
    if mode not in ("obs", "plain"):
        raise ValueError("statics: unsupported mode")
    structure = _select(index, seed, split, style)
    answers = _equilibrium(structure)
    if answers != _sympy(structure):
        raise ValueError("statics: independent solvers disagree")
    task = _TASKS[index % len(_TASKS)]
    question, unit, aliases = _question(task)
    gold = f"{_rounded(answers[task])} {unit}"
    slots = _slots(structure)
    image, printed = _render(structure, style)
    return build(source=GENERATOR_VERSION, family=FAMILY, index=index, split=split, mode=mode,
                 question=question, choices=(), answer_type="unit_number", gold=gold,
                 accept={"unit_aliases": aliases, "require_unit": True}, slots=slots,
                 slot_types={key: "number" if key == "span" else "text" for key in slots},
                 required_slots=tuple(slots),
                 oracle_fact_lines=tuple(f"{k} = {v}" for k, v in slots.items()),
                 printed_text=printed, structure=structure, template_id=task,
                 image_bytes=(image,), generator_version=GENERATOR_VERSION, seed=seed, style=style)


def self_check(p: GeneratedProblem) -> None:
    """Rebuild independent solutions, complete records, and PNG bytes from the identity."""
    if not isinstance(p, GeneratedProblem):
        raise ValueError("statics: expected GeneratedProblem")
    if (p.lineage.generator_version != GENERATOR_VERSION or p.lineage.generator_seed is None
            or p.actor.source != GENERATOR_VERSION):
        raise ValueError("statics: wrong generator identity")
    try:
        index = int(p.actor.problem_id.split(":")[-2])
        style = p.lineage.image_provenance.rsplit(":", 1)[-1]
        expected = generate(index, p.lineage.generator_seed, p.lineage.split, p.actor.mode, style)
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError("statics: invalid generation identity") from exc
    if p != expected:
        raise ValueError("statics: render, answer, slots or lineage mismatch")
