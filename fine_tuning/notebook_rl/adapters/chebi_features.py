"""Standalone RDKit feature extraction for ChEBI-20-MM (run with the notebook-data env; stdlib + rdkit only).

Reads only the curated SMILES column and writes counts computed by RDKit. The free-text
description column is never read into the output. Deliberately imports nothing from the package.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

from rdkit import Chem, RDLogger, __version__ as RDKIT_VERSION
from rdkit.Chem import rdMolDescriptors

VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _features(cid: int, smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "cid": cid,
        "smiles_sha256": hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "n_O": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
        "n_N": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
    }


def _replace_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run(csv_path: Path, output: Path) -> dict:
    RDLogger.DisableLog("rdApp.*")
    csv.field_size_limit(1 << 24)
    rows_in = 0
    seen: set[int] = set()
    lines: list[str] = []
    with open(csv_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_in += 1
            try:
                cid = int(row["CID"])
            except (KeyError, ValueError):
                continue
            smiles = (row.get("SMILES") or "").strip()
            if cid in seen or not smiles:
                continue
            record = _features(cid, smiles)
            if record is None:
                continue
            seen.add(cid)
            lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_sha256": _sha256_file(csv_path),
        "rdkit_version": RDKIT_VERSION,
        "rows_in": rows_in,
        "rows_out": len(lines),
        "script_sha256": _sha256_file(Path(__file__)),
        "version": VERSION,
    }
    _replace_atomic(output, "".join(line + "\n" for line in lines))
    _replace_atomic(output.with_name(output.name + ".manifest.json"), json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run(args.csv, args.output)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
