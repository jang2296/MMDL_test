"""Program-authored notated music; independent theory checks, no teacher solutions."""
from __future__ import annotations

import io
import random
import xml.etree.ElementTree as ET

import cairosvg
import verovio
from music21 import chord, interval, key, note
from PIL import Image

from ..schema import Mode, Split
from .common import GeneratedProblem, build, matches_split, png_bytes, structure_hash

GENERATOR_VERSION = "gen-music-v1"
FAMILY = "music"
OBSERVATION_INSTRUCTION = (
    "List each note in order as note[<measure>.<index>] = <pitch with octave, e.g. C#4>; "
    "also clef = <treble or bass>, key_sig = <signed number of sharps; flats negative>, "
    "time_sig = <numerator/denominator>.\n"
)
_TASKS = ("interval", "chord", "major_key", "time_signature", "pitch")
_STEPS = "CDEFGAB"
_SEMITONES = (0, 2, 4, 5, 7, 9, 11)
_MAJOR_KEYS = {-4: "Ab", -3: "Eb", -2: "Bb", -1: "F", 0: "C", 1: "G", 2: "D", 3: "A", 4: "E"}


def _pitch(diatonic: int, alteration: int) -> str:
    return _STEPS[diatonic % 7] + ("#" * alteration if alteration >= 0 else "b" * -alteration) + str(diatonic // 7)


def _midi(pitch: tuple[int, int] | list[int]) -> int:
    diatonic, alter = pitch
    return 12 * (diatonic // 7 + 1) + _SEMITONES[diatonic % 7] + alter


def _signature(sharps: int) -> dict[str, int]:
    letters = "FCGDAEB" if sharps >= 0 else "BEADGCF"
    return {letter: 1 if sharps >= 0 else -1 for letter in letters[:abs(sharps)]}


def _select(index: int, seed: int, split: Split, style: str) -> dict:
    matches_split("0000000000000000", split, style)
    rng = random.Random(seed * 1_000_003 + index)
    for _ in range(1000):
        beats, beat_type = rng.choice(((2, 4), (3, 4), (4, 4), (6, 8)))
        measures = rng.choice([n for n in range(1, 5) if 4 <= beats * n <= 12])
        clef, sharps = rng.choice(("treble", "bass")), rng.randint(-4, 4)
        accidental = _signature(sharps)
        lower = 28 if clef == "treble" else 16
        pitches = []
        for _ in range(beats * measures):
            diatonic = rng.randint(lower, lower + 10)
            pitches.append([diatonic, accidental.get(_STEPS[diatonic % 7], 0)])
        if _TASKS[index % len(_TASKS)] == "chord":
            root = rng.randint(lower, lower + 4)
            quality = rng.choice(((4, 7), (3, 7), (3, 6), (4, 8)))
            root_pitch = [root, accidental.get(_STEPS[root % 7], 0)]
            pitches[0] = root_pitch
            for i, (step, semitones) in enumerate(zip((2, 4), quality, strict=True), 1):
                natural = _midi([root + step, 0])
                pitches[i] = [root + step, _midi(root_pitch) + semitones - natural]
        world = {"clef": clef, "sharps": sharps, "beats": beats, "beat_type": beat_type,
                 "measures": measures, "pitches": pitches}
        if matches_split(structure_hash(world), split, style):
            return world
    raise RuntimeError("music: no structure in requested split after 1000 candidates")


def _manual_interval(first, second) -> str:
    degree = abs(second[0] - first[0]) + 1
    semitones = abs(_midi(second) - _midi(first))
    simple = (degree - 1) % 7
    normal = _SEMITONES[simple] + 12 * ((degree - 1) // 7)
    difference = semitones - normal
    if simple in (0, 3, 4):
        quality = "P" if difference == 0 else "A" * difference if difference > 0 else "d" * -difference
    else:
        quality = "M" if difference == 0 else "m" if difference == -1 else "A" * difference if difference > 0 else "d" * (-difference - 1)
    return quality + str(degree)


def _answers(world: dict, task: str, target: int) -> tuple[str, str]:
    pitches = world["pitches"]
    names = [_pitch(*p) for p in pitches]
    if task == "interval":
        return interval.Interval(note.Note(names[0]), note.Note(names[1])).name, _manual_interval(*pitches[:2])
    if task == "chord":
        distances = tuple(_midi(p) - _midi(pitches[0]) for p in pitches[1:3])
        qualities: dict[tuple[int, ...], str] = {(4, 7): "major", (3, 7): "minor", (3, 6): "diminished", (4, 8): "augmented"}
        manual = qualities.get(distances)
        if manual is None:
            raise ValueError("music: invalid authored triad")
        return chord.Chord(names[:3]).quality, manual
    if task == "major_key":
        return key.KeySignature(world["sharps"]).asKey("major").tonic.name.replace("-", "b") + " major", _MAJOR_KEYS[world["sharps"]] + " major"
    if task == "time_signature":
        from music21 import meter
        written = f"{world['beats']}/{world['beat_type']}"
        return meter.TimeSignature(written).ratioString, written
    if task == "pitch":
        # Library spelling and pitch height both checked against independent arithmetic.
        parsed = note.Note(names[target])
        if parsed.pitch.midi != _midi(pitches[target]):
            raise ValueError("music: pitch height disagrees with independent arithmetic")
        return parsed.nameWithOctave.replace("-", "b"), names[target]
    raise ValueError("music: unsupported task")


def _xml(world: dict) -> bytes:
    root = ET.Element("score-partwise", version="4.0")
    parts = ET.SubElement(root, "part-list")
    part_name = ET.SubElement(parts, "score-part", id="P1")
    ET.SubElement(part_name, "part-name").text = " "
    part = ET.SubElement(root, "part", id="P1")
    beats, beat_type = world["beats"], world["beat_type"]
    signature = _signature(world["sharps"])
    for m in range(world["measures"]):
        measure = ET.SubElement(part, "measure", number=str(m + 1))
        if m == 0:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = "2" if beat_type == 8 else "1"
            key_xml = ET.SubElement(attrs, "key")
            ET.SubElement(key_xml, "fifths").text = str(world["sharps"])
            time = ET.SubElement(attrs, "time")
            ET.SubElement(time, "beats").text = str(beats)
            ET.SubElement(time, "beat-type").text = str(beat_type)
            clef = ET.SubElement(attrs, "clef")
            ET.SubElement(clef, "sign").text = "G" if world["clef"] == "treble" else "F"
            ET.SubElement(clef, "line").text = "2" if world["clef"] == "treble" else "4"
        active_accidentals: dict[int, int] = {}
        for n, (diatonic, alteration) in enumerate(world["pitches"][m * beats:(m + 1) * beats]):
            note_xml = ET.SubElement(measure, "note")
            pitch_xml = ET.SubElement(note_xml, "pitch")
            step = _STEPS[diatonic % 7]
            ET.SubElement(pitch_xml, "step").text = step
            ET.SubElement(pitch_xml, "alter").text = str(alteration)
            ET.SubElement(pitch_xml, "octave").text = str(diatonic // 7)
            ET.SubElement(note_xml, "duration").text = "1"
            ET.SubElement(note_xml, "type").text = "eighth" if beat_type == 8 else "quarter"
            if alteration != active_accidentals.get(diatonic, signature.get(step, 0)):
                ET.SubElement(note_xml, "accidental").text = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}[alteration]
            active_accidentals[diatonic] = alteration
            lyric = ET.SubElement(note_xml, "lyric", number="1")
            ET.SubElement(lyric, "syllabic").text = "single"
            ET.SubElement(lyric, "text").text = f"{m + 1}.{n + 1}"
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _render(world: dict, style: str) -> bytes:
    toolkit = verovio.toolkit()
    toolkit.resetXmlIdSeed(3407)
    if not toolkit.setOptions({"pageWidth": 2400 if style == "default" else 2800, "pageHeight": 4000,
                               "scale": 100, "adjustPageHeight": True, "breaks": "none",
                               "lyricWordSpace": 2.0, "lyricSize": 3.5, "spacingLinear": 0.6,
                               "header": "none", "footer": "none", "svgViewBox": True,
                               "font": "Bravura" if style == "default" else "Leipzig"}):
        raise ValueError("music: renderer options rejected")
    if not toolkit.loadData(_xml(world).decode()) or toolkit.getPageCount() != 1:
        raise ValueError("music: score did not render to exactly one complete page")
    raw = cairosvg.svg2png(bytestring=toolkit.renderToSVG(1).encode(), background_color="white")
    with Image.open(io.BytesIO(raw)) as image:
        return png_bytes(image, dpi=100)


def generate(index: int, seed: int, split: Split, mode: Mode, style: str = "default") -> GeneratedProblem:
    if type(index) is not int or index < 0 or type(seed) is not int or seed < 0 or mode not in ("obs", "plain"):
        raise ValueError("music: invalid index, seed or mode")
    world = _select(index, seed, split, style)
    task = _TASKS[index % len(_TASKS)]
    target = index % len(world["pitches"])
    answer, independent = _answers(world, task, target)
    if answer != independent:
        raise ValueError("music: independent solvers disagree")
    beats = world["beats"]
    labels = [f"{i // beats + 1}.{i % beats + 1}" for i in range(len(world["pitches"]))]
    questions = {
        "interval": f"What is the named interval between notes {labels[0]} and {labels[1]}? Ignore direction; retain compound size. Use quality and number, such as M3 or P5.",
        "chord": f"What triad quality is formed by notes {', '.join(labels[:3])}? Answer major, minor, diminished, or augmented.",
        "major_key": "Which major key is indicated by the written key signature? Use an English note name and 'major', for example C major or Bb major.",
        "time_signature": "What time signature is written at the start? Give numerator/denominator exactly as written; do not reduce the fraction.",
        "pitch": f"What is the pitch of note {labels[target]}? Include the written accidental and octave, for example C#4 or Bb3; do not respell enharmonically.",
    }
    slots = {f"note[{label}]": _pitch(*p) for label, p in zip(labels, world["pitches"], strict=True)}
    slots |= {"clef": world["clef"], "key_sig": str(world["sharps"]), "time_sig": f"{world['beats']}/{world['beat_type']}"}
    kinds = {k: "pitch" if k.startswith("note[") else "number" if k == "key_sig" else "text" for k in slots}
    required = {"interval": tuple(f"note[{v}]" for v in labels[:2]),
                "chord": tuple(f"note[{v}]" for v in labels[:3]), "major_key": ("key_sig",),
                "time_signature": ("time_sig",), "pitch": (f"note[{labels[target]}]",)}[task]
    return build(GENERATOR_VERSION, FAMILY, index, split, mode, questions[task], (), "short_text", answer,
                 {"case_sensitive": True} if task == "interval" else {},
                 slots, kinds, required, tuple(f"{k} = {v}" for k, v in slots.items()),
                 (*labels, str(world["beats"]), str(world["beat_type"])), world, task,
                 (_render(world, style),), GENERATOR_VERSION, seed, style)


def self_check(p: GeneratedProblem) -> None:
    if p.actor.source != GENERATOR_VERSION or p.lineage.generator_seed is None:
        raise ValueError("music: wrong generator identity")
    index = int(p.actor.problem_id.split(":")[-2])
    style = p.lineage.image_provenance.rsplit(":", 1)[-1]
    expected = generate(index, p.lineage.generator_seed, p.lineage.split, p.actor.mode, style)
    if p != expected:
        raise ValueError("music: rendered score, answer, slots or lineage mismatch")
