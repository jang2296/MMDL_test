"""ChEBI-SMILES rendering and program rewards; no download or training approval."""
from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import re
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import cairosvg
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from ..schema import Mode, Split
from .common import GeneratedProblem, build, matches_split, png_bytes, structure_hash

GENERATOR_VERSION = "gen-chem-v1"
FAMILY = "molecule"
OBSERVATION_INSTRUCTION = "List counts as lines: atoms[<element>] = <count>; rings = <count>\n"
_TASKS = ("formula", "rings", "donors", "carboxyl")


@lru_cache(maxsize=2)
def _source(raw: bytes) -> tuple[tuple[str, str], ...]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t")
    if reader.fieldnames != ["chebi_id", "smiles"]:
        raise ValueError("chem: TSV columns must be exactly chebi_id, smiles")
    eligible: dict[str, str] = {}
    ids: set[str] = set()
    for row in reader:
        ident, smiles = row["chebi_id"], row["smiles"]
        if (not isinstance(ident, str) or not re.fullmatch(r"CHEBI:[0-9]+", ident)
                or ident in ids or not isinstance(smiles, str)):
            raise ValueError("chem: invalid or duplicate source identity")
        ids.add(ident)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("chem: invalid source SMILES")
        # Bounded neutral, connected organic subset: no salt, isotope or radical ambiguity.
        if (not 6 <= mol.GetNumHeavyAtoms() <= 30 or len(Chem.GetMolFrags(mol)) != 1
                or any(a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                       or a.GetAtomicNum() not in (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
                       for a in mol.GetAtoms())):
            continue
        # Atom-map IDs are annotation, not molecular identity or a new split family.
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        # Repeated source IDs for the same molecule are one structure family.
        eligible.setdefault(canonical, ident)
    if not eligible:
        raise ValueError("chem: no eligible 6..30-heavy-atom source molecules")
    return tuple(sorted((ident, smiles) for smiles, ident in eligible.items()))


def _load_source() -> tuple[bytes, tuple[tuple[str, str], ...]]:
    path = os.environ.get("NOTEBOOK_RL_CHEBI_TSV")
    if not path:
        raise ValueError("chem: NOTEBOOK_RL_CHEBI_TSV is required; no automatic download")
    raw = Path(path).read_bytes()
    return raw, _source(raw)


def _counts(mol) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
        if atom.GetAtomicNum() != 1:
            counts["H"] += atom.GetTotalNumHs()
    return {k: v for k, v in counts.items() if v}


def _manual_formula(mol) -> str:
    counts = _counts(mol)
    order = (["C", "H"] if "C" in counts else []) + sorted(set(counts) - {"C", "H"})
    if "C" not in counts and "H" in counts:
        order = sorted([*order, "H"])
    return "".join(element + (str(counts[element]) if counts[element] != 1 else "")
                   for element in order if element in counts)


def _cycle_rank(mol) -> int:
    parent = list(range(mol.GetNumAtoms()))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    cycles = 0
    for bond in mol.GetBonds():
        left, right = root(bond.GetBeginAtomIdx()), root(bond.GetEndAtomIdx())
        if left == right:
            cycles += 1
        else:
            parent[left] = right
    return cycles


def _manual_donors(mol) -> int:
    # Explicit atom/valence cases matching RDKit's documented Lipinski donor definition;
    # this traversal does not call its SMARTS matcher or descriptor.
    count = 0
    for atom in mol.GetAtoms():
        z, h = atom.GetAtomicNum(), atom.GetTotalNumHs()
        valence = atom.GetTotalValence()
        if ((z == 7 and not atom.GetIsAromatic() and h > 0 and valence == 3)
                or (z in (8, 16) and h == 1)
                or (z == 7 and atom.GetIsAromatic() and h == 1)):
            count += 1
    return count


def _manual_carboxyl(mol) -> int:
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
            continue
        double_o = single_oh = 0
        for bond in atom.GetBonds():
            neighbor = bond.GetOtherAtom(atom)
            if neighbor.GetAtomicNum() == 8:
                double_o += bond.GetBondType() == Chem.BondType.DOUBLE
                single_oh += bond.GetBondType() == Chem.BondType.SINGLE and neighbor.GetTotalNumHs() == 1
        count += bool(double_o and single_oh)
    return count


def _answers(mol, task: str) -> tuple[str, str]:
    if task == "formula":
        return rdMolDescriptors.CalcMolFormula(mol), _manual_formula(mol)
    if task == "rings":
        return str(len(Chem.GetSSSR(mol))), str(_cycle_rank(mol))
    if task == "donors":
        return str(rdMolDescriptors.CalcNumHBD(mol)), str(_manual_donors(mol))
    if task == "carboxyl":
        query = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
        return str(len(mol.GetSubstructMatches(query))), str(_manual_carboxyl(mol))
    raise ValueError("chem: unsupported task")


def _render(mol, style: str) -> bytes:
    drawn = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(drawn, canonOrient=True)
    drawer = rdMolDraw2D.MolDraw2DSVG(900, 650)
    # Boost.Python's distributed stubs mis-type mutable drawing properties.
    options: Any = drawer.drawOptions()
    options.clearBackground = True
    options.bondLineWidth = 3.0 if style == "shift" else 2.0
    options.rotate = 35.0 if style == "shift" else 0.0
    options.baseFontSize = 0.8 if style == "shift" else 0.6
    if style == "shift":
        options.useBWAtomPalette()
        options.setBackgroundColour((1.0, 0.97, 0.90))
    drawer.DrawMolecule(drawn)
    drawer.FinishDrawing()
    raw = cairosvg.svg2png(bytestring=drawer.GetDrawingText().encode())
    with Image.open(io.BytesIO(raw)) as image:
        return png_bytes(image, dpi=100)


def generate(index: int, seed: int, split: Split, mode: Mode, style: str = "default") -> GeneratedProblem:
    if type(index) is not int or index < 0 or type(seed) is not int or seed < 0 or mode not in ("obs", "plain"):
        raise ValueError("chem: invalid index, seed or mode")
    matches_split("0000000000000000", split, style)
    raw, candidates = _load_source()
    rng = random.Random(seed * 1_000_003 + index)
    for _ in range(1000):
        ident, smiles = candidates[rng.randrange(len(candidates))]
        structure = {"canonical_smiles": smiles}
        if matches_split(structure_hash(structure), split, style):
            break
    else:
        raise RuntimeError("chem: requested structure split absent after 1000 candidates")
    mol = Chem.MolFromSmiles(smiles)
    task = _TASKS[index % len(_TASKS)]
    answer, independent = _answers(mol, task)
    if answer != independent:
        raise ValueError("chem: independent solvers disagree")
    questions = {
        "formula": "What is the molecular formula of the neutral molecule shown? Include implicit hydrogens; use Hill order.",
        "rings": "How many independent rings (cycle rank) does the molecule shown contain? Answer as an integer.",
        "donors": "How many hydrogen-bond donor atoms does the neutral molecule shown have under the Lipinski definition?",
        "carboxyl": "How many neutral carboxylic acid groups (-C(=O)OH) are present in the molecule shown?",
    }
    slots = {f"atoms[{element}]": str(count) for element, count in sorted(_counts(mol).items())}
    slots["rings"] = str(_cycle_rank(mol))
    # Atom totals cannot certify the positions of donor/carboxyl groups.
    required = tuple(k for k in slots if k != "rings") if task == "formula" else ("rings",) if task == "rings" else ()
    p = build(GENERATOR_VERSION, FAMILY, index, split, mode, questions[task], (),
              "short_text" if task == "formula" else "int", answer,
              {"case_sensitive": True} if task == "formula" else {}, slots,
              {k: "number" for k in slots}, required, tuple(f"{k} = {v}" for k, v in slots.items()),
              (), structure, task, (_render(mol, style),), GENERATOR_VERSION, seed, style)
    return replace(p, lineage=replace(p.lineage, license="CC-BY-4.0 (ChEBI source; review required)",
                   upstream_origin="ChEBI", parents=(ident, "chebi_tsv_sha256:" + hashlib.sha256(raw).hexdigest())))


def self_check(p: GeneratedProblem) -> None:
    if p.actor.source != GENERATOR_VERSION or p.lineage.generator_seed is None:
        raise ValueError("chem: wrong generator identity")
    index = int(p.actor.problem_id.split(":")[-2])
    style = p.lineage.image_provenance.rsplit(":", 1)[-1]
    expected = generate(index, p.lineage.generator_seed, p.lineage.split, p.actor.mode, style)
    if p != expected:
        raise ValueError("chem: source, rendered pixels, solver, slots or lineage mismatch")
