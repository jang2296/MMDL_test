"""Authored metal stress-strain curves with independent exact solvers; not training-approved."""
from __future__ import annotations

import random
from fractions import Fraction

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..schema import Mode, Split
from .common import GeneratedProblem, build, matches_split, png_bytes, structure_hash

GENERATOR_VERSION = "gen-materials-v1"
FAMILY = "materials"
OBSERVATION_INSTRUCTION = (
    "List each labelled point as point[<label>] = <strain in percent>,<stress in MPa> "
    "with no spaces and no units.\n"
)
_TASKS = ("youngs_modulus", "ultimate_strength", "fracture_strain")
_TITLE = "Engineering stress-strain curve"
_X_LABEL = "Strain (%)"
_Y_LABEL = "Stress (MPa)"
# Strains are stored in tenths of a percent and stresses in MPa, so every value is an exact integer.
# A strain of t tenths of a percent is t / 1000, so E = S / (t / 1000) MPa = S / t GPa.


class _Unplaceable(ValueError):
    """A point label has no collision-free placement on the rendered curve."""


def _tenths(value: int) -> str:
    return f"{value // 10}.{value % 10}"


def _decimal(value: Fraction) -> str:
    """Exact one-decimal text of a value that is a whole number of tenths."""
    if (value * 10).denominator != 1 or value < 0:
        raise ValueError("materials: value is not a nonnegative whole number of tenths")
    return _tenths(int(value * 10))


def _structure(rng: random.Random) -> dict:
    modulus = rng.randint(50, 250)
    t_p = rng.randint(1, min(4, 700 // modulus))
    hardening = rng.randint(2, 3)
    strain, stress = t_p, modulus * t_p
    points = [["P", strain, stress]]
    for label in [f"H{i}" for i in range(1, hardening + 1)] + ["U"]:
        strain += rng.randint(10, 40)
        stress += rng.randint(40, 150)
        points.append([label, strain, stress])
    fracture_stress = stress - rng.randint(50, stress // 2)
    points.append(["F", strain + rng.randint(10, 60), fracture_stress])
    return {"points": points}


def _select(index: int, seed: int, split: Split, style: str) -> dict:
    if type(index) is not int or index < 0 or type(seed) is not int or seed < 0:
        raise ValueError("materials: nonnegative integer index and seed required")
    matches_split("0000000000000000", split, style)
    rng = random.Random(seed * 1_000_003 + index)
    for _ in range(1000):
        structure = _structure(rng)
        if matches_split(structure_hash(structure), split, style):
            try:
                _render(structure, style)
            except _Unplaceable:
                continue  # Every point label must be legible; skip curves with no clear placement.
            return structure
    raise RuntimeError("materials: no structure in requested split after 1000 candidates")


def _validate(structure: dict) -> None:
    """Reject curves that are not a clean elastic, hardening, necking, fracture sequence."""
    points = structure["points"]
    labels = [p[0] for p in points]
    hardening = len(points) - 3
    if hardening not in (2, 3) or labels != ["P", *(f"H{i}" for i in range(1, hardening + 1)), "U", "F"]:
        raise ValueError("materials: unexpected point labels")
    if any(len(p) != 3 or type(p[1]) is not int or type(p[2]) is not int for p in points):
        raise ValueError("materials: integer strain (tenths of percent) and stress (MPa) required")
    strains, stresses = [p[1] for p in points], [p[2] for p in points]
    if strains[0] <= 0 or stresses[0] <= 0 or any(b <= a for a, b in zip(strains, strains[1:])):
        raise ValueError("materials: strain must be positive and strictly increasing")
    rising = stresses[:-1]
    if any(b <= a for a, b in zip(rising, rising[1:])):
        raise ValueError("materials: stress must rise strictly up to the ultimate point")
    ultimate, fracture = stresses[-2], stresses[-1]
    if ultimate != max(stresses) or stresses.count(ultimate) != 1:
        raise ValueError("materials: U must be the unique maximum stress")
    if fracture >= ultimate or fracture <= 0:
        raise ValueError("materials: fracture stress must lie below the ultimate stress")
    if stresses[0] % strains[0] or not 50 <= stresses[0] // strains[0] <= 250:
        raise ValueError("materials: proportional limit must give an integer modulus of 50-250 GPa")


def _polyline(structure: dict) -> list[tuple[Fraction, Fraction]]:
    return [(Fraction(0), Fraction(0)),
            *((Fraction(strain, 10), Fraction(stress)) for _, strain, stress in structure["points"])]


def _solve(structure: dict) -> dict[str, Fraction]:
    """Read the answers from the drawn geometry: slope, curve maximum and last vertex."""
    curve = _polyline(structure)
    (x0, y0), (x1, y1) = curve[0], curve[1]
    slope = (y1 - y0) / (x1 - x0)  # MPa per percent strain
    return {"youngs_modulus": slope / 10,  # slope * 100 / 1000 GPa
            "ultimate_strength": max(y for _, y in curve), "fracture_strain": curve[-1][0]}


def _fields(structure: dict) -> dict[str, Fraction]:
    """Read the same answers from the labelled P, U and F records."""
    by_label = {label: (strain, stress) for label, strain, stress in structure["points"]}
    return {"youngs_modulus": Fraction(by_label["P"][1], by_label["P"][0]),
            "ultimate_strength": Fraction(by_label["U"][1]),
            "fracture_strain": Fraction(by_label["F"][0], 10)}


def _question(task: str) -> tuple[str, str, dict]:
    if task == "youngs_modulus":
        return ("From the linear elastic region of the curve, what is the Young's modulus "
                "of the material, in GPa? Answer as an integer.", "int", {})
    if task == "ultimate_strength":
        return ("What is the ultimate tensile strength of the material, in MPa? "
                "Answer as an integer.", "int", {})
    return ("What is the engineering strain at fracture, in percent? "
            "Round to one decimal place.", "decimal", {"round_to": 1})


def _slots(structure: dict) -> dict[str, str]:
    return {f"point[{label}]": f"{_tenths(strain)},{stress}"
            for label, strain, stress in structure["points"]}


_FALLBACKS = (((10, -18), "left", "top"), ((-8, 10), "right", "bottom"),
              ((0, 12), "center", "bottom"), ((0, -14), "center", "top"),
              ((14, -32), "left", "top"), ((8, 8), "left", "bottom"),
              ((-10, -14), "right", "top"), ((14, -40), "left", "top"),
              ((14, -52), "left", "top"))


def _candidates(label: str) -> tuple[tuple[tuple[int, int], str, str], ...]:
    """Placements (offset in points, horizontal and vertical alignment) in priority order."""
    if label == "P":
        first = ((10, -20), "left", "center")
    elif label == "U":
        first = ((8, 8), "left", "bottom")
    elif label == "F":
        first = ((8, -2), "left", "center")
    elif int(label[1:]) % 2:
        first = ((-8, 10), "right", "bottom")
    else:
        first = ((10, -18), "left", "top")
    return (first, *(c for c in _FALLBACKS if c != first))


def _render(structure: dict, style: str) -> tuple[bytes, tuple[str, ...]]:
    shifted = style == "shift"
    dpi = 110 if shifted else 100
    font = "DejaVu Serif" if shifted else "DejaVu Sans"
    background = "#fff9ed" if shifted else "white"
    line_color, point_color = (("#8a2244", "#157b64") if shifted else ("#1f4e9c", "#b22222"))
    figure = Figure(figsize=(10, 6) if shifted else (12, 6), facecolor=background, dpi=dpi)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    ax.set_facecolor(background)
    curve = [(float(x), float(y)) for x, y in _polyline(structure)]
    line, = ax.plot(*zip(*curve), color=line_color, linewidth=2.6 if shifted else 2)
    ax.plot(*zip(*curve[1:]), "o", color=point_color, markersize=6)
    last_strain, ultimate = structure["points"][-1][1], structure["points"][-2][2]
    x_max = (-(-last_strain // 10) + 5 + 4) // 5 * 5  # multiple of 5, at least 5 percent of headroom
    y_step = 100 if ultimate <= 600 else 200
    y_max = (ultimate // y_step + 2) * y_step
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    x_ticks, y_ticks = list(range(0, x_max + 1, 5)), list(range(0, y_max + 1, y_step))
    ax.set_xticks(x_ticks, [str(t) for t in x_ticks], fontfamily=font)
    ax.set_yticks(y_ticks, [str(t) for t in y_ticks], fontfamily=font)
    ax.grid(True, color="#c8c8c8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(_TITLE, fontfamily=font, fontsize=16)
    ax.set_xlabel(_X_LABEL, fontfamily=font, fontsize=12)
    ax.set_ylabel(_Y_LABEL, fontfamily=font, fontsize=12)
    figure.subplots_adjust(left=0.09, right=0.97, top=0.9, bottom=0.12)
    # The layout is final here, so label boxes measured now equal the saved pixels.
    renderer = figure.canvas.get_renderer()
    figure.draw_without_rendering()
    blocked = [t.get_window_extent(renderer)
               for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text()]
    axes_box = ax.get_window_extent(renderer)
    curve_path = line.get_transform().transform_path(line.get_path())
    printed = [_TITLE, _X_LABEL, _Y_LABEL]
    for label, strain, stress in structure["points"]:
        text = f"{label} ({_tenths(strain)}, {stress})"
        for offset, ha, va in _candidates(label):
            annotation = ax.annotate(text, (strain / 10, stress), xytext=offset,
                                     textcoords="offset points", ha=ha, va=va, color=point_color,
                                     fontsize=10, fontfamily=font)
            box = annotation.get_window_extent(renderer).padded(2)
            if (box.x0 >= axes_box.x0 and box.y0 >= axes_box.y0 and box.x1 <= axes_box.x1
                    and box.y1 <= axes_box.y1 and not any(box.overlaps(b) for b in blocked)
                    and not curve_path.intersects_bbox(box, filled=False)):
                blocked.append(box)
                break
            annotation.remove()
        else:
            raise _Unplaceable(f"materials: no collision-free label placement for {label}")
        printed.append(text)
    printed.extend(str(t) for t in x_ticks + y_ticks)
    return png_bytes(figure, dpi=dpi), tuple(printed)


def generate(index: int, seed: int, split: Split, mode: Mode,
             style: str = "default") -> GeneratedProblem:
    if mode not in ("obs", "plain"):
        raise ValueError("materials: unsupported mode")
    structure = _select(index, seed, split, style)
    _validate(structure)
    answers = _solve(structure)
    if answers != _fields(structure):
        raise ValueError("materials: independent solvers disagree")
    task = _TASKS[index % len(_TASKS)]
    question, answer_type, accept = _question(task)
    gold = str(int(answers[task])) if answer_type == "int" else _decimal(answers[task])
    slots = _slots(structure)
    image, printed = _render(structure, style)
    return build(source=GENERATOR_VERSION, family=FAMILY, index=index, split=split, mode=mode,
                 question=question, choices=(), answer_type=answer_type, gold=gold,
                 accept=accept, slots=slots, slot_types={key: "text" for key in slots},
                 required_slots=tuple(slots),
                 oracle_fact_lines=tuple(f"{k} = {v}" for k, v in slots.items()),
                 printed_text=printed, structure=structure, template_id=task,
                 image_bytes=(image,), generator_version=GENERATOR_VERSION, seed=seed, style=style)


def self_check(p: GeneratedProblem) -> None:
    """Rebuild independent solutions, complete records, and PNG bytes from the identity."""
    if not isinstance(p, GeneratedProblem):
        raise ValueError("materials: expected GeneratedProblem")
    if (p.lineage.generator_version != GENERATOR_VERSION or p.lineage.generator_seed is None
            or p.actor.source != GENERATOR_VERSION):
        raise ValueError("materials: wrong generator identity")
    try:
        index = int(p.actor.problem_id.split(":")[-2])
        style = p.lineage.image_provenance.rsplit(":", 1)[-1]
        expected = generate(index, p.lineage.generator_seed, p.lineage.split, p.actor.mode, style)
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError("materials: invalid generation identity") from exc
    if p != expected:
        raise ValueError("materials: render, answer, slots or lineage mismatch")
