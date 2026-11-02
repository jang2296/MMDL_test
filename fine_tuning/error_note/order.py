"""Write the pinned training order (item_order.json). Run once locally; the file is committed and pods only read it.

CLI: python -m fine_tuning.error_note.order build --approval APPROVAL_EXPORT.json --manifest DATASET_MANIFEST.json
         --lineage VAULT/train/lineage.jsonl --output fine_tuning/error_note/item_order.json
     python -m fine_tuning.error_note.order spread --base fine_tuning/error_note/item_order.json
         --output fine_tuning/error_note/item_order_v2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .contract import MASTER_SEED, round_robin_order, spread_order

VERSION = "error-note-order-v1"
VERSION_V2 = "error-note-order-v2"


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def build(approval: dict, manifest: dict, lineage_rows: list[dict]) -> dict:
    """Training ids from approval["selected_ids"], subject and source from manifest["items"], structure_hash from lineage."""
    manifest_rows = {row["problem_id"]: row for row in manifest["items"]}
    structure_hashes = {row["problem_id"]: row["structure_hash"] for row in lineage_rows}
    items = []
    for problem_id in approval["selected_ids"]:
        if problem_id not in manifest_rows:
            raise ValueError(f"{problem_id} is missing from the manifest")
        if problem_id not in structure_hashes:
            raise ValueError(f"{problem_id} is missing from the lineage")
        row = manifest_rows[problem_id]
        if row["status"] != "INCLUDED":
            raise ValueError(f"{problem_id} has manifest status {row['status']}, not INCLUDED")
        items.append({"problem_id": problem_id, "subject": row["subject"], "structure_hash": structure_hashes[problem_id]})
    order = round_robin_order(items)
    kept = set(order)
    return {
        "version": VERSION,
        "master_seed": MASTER_SEED,
        "rule": "Selected ids sorted by sha256(master_seed:id), one item kept per structure_hash, then taken round-robin across "
                "subjects in sorted subject order.",
        "dropped_duplicate_structure_ids": [item["problem_id"] for item in items if item["problem_id"] not in kept],
        "items": [{"problem_id": i, "subject": manifest_rows[i]["subject"], "source": manifest_rows[i]["source"]} for i in order],
        "count": len(order),
        "ids_sha256": _ids_sha256(order),
    }


def build_v2(base: dict, base_sha256: str) -> dict:
    """The same ids as the v1 order, sources spread evenly over the run (contract.spread_order). Ids and sources only."""
    by_id = {item["problem_id"]: item for item in base["items"]}
    order = spread_order(base["items"], [item["problem_id"] for item in base["items"]])
    return {
        "version": VERSION_V2,
        "master_seed": MASTER_SEED,
        "base_order": {"version": base["version"], "sha256": base_sha256, "ids_sha256": base["ids_sha256"]},
        "rule": "The v1 order's items; within each source the v1 order is kept, and all items are sorted by "
                "((rank in source + 0.5) / source size, v1 position), so every source is spread over the whole run.",
        "items": [by_id[problem_id] for problem_id in order],
        "count": len(order),
        "ids_sha256": _ids_sha256(order),
    }


def read_order(path: Path, *, expected_sha256: str | None = None) -> dict:
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("item order file sha256 does not match the expected value")
    data = json.loads(raw)
    ids = [item["problem_id"] for item in data["items"]]
    if data.get("version") not in (VERSION, VERSION_V2):
        raise ValueError(f"unexpected order version {data.get('version')!r}")
    if data["count"] != len(ids):
        raise ValueError("count does not match the number of items")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate problem_id in the order")
    if data["ids_sha256"] != _ids_sha256(ids):
        raise ValueError("ids_sha256 does not match the items")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    for name in ("approval", "manifest", "lineage", "output"):
        build_parser.add_argument(f"--{name}", type=Path, required=True)
    spread_parser = sub.add_parser("spread")
    for name in ("base", "output"):
        spread_parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.command == "spread":
        base = read_order(args.base)
        if base["version"] != VERSION:
            raise SystemExit("spread: the base must be a v1 order")
        order = build_v2(base, hashlib.sha256(args.base.read_bytes()).hexdigest())
        args.output.write_text(json.dumps(order, indent=1) + "\n")
        print(f"count={order['count']} ids_sha256={order['ids_sha256']}")
        return 0
    lineage_rows = [json.loads(line) for line in args.lineage.read_text().splitlines() if line.strip()]
    order = build(json.loads(args.approval.read_text()), json.loads(args.manifest.read_text()), lineage_rows)
    args.output.write_text(json.dumps(order, indent=1) + "\n")
    print(f"count={order['count']} ids_sha256={order['ids_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
