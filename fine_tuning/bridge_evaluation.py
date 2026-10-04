"""Private common development probes for the bounded synthetic Graph/CS bridge.

No source acquisition, training or GPU call. This is not full professional G7.
All checkpoints receive the same root/condition/view inputs; gold stays separate.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image

from . import audit
from .ablation import ARMS, FLAGS, VIEWS
from .evaluate import digest
from .graph_bridge import SCOPE, VERSION


DOMAIN = "BOUNDED_SYNTHETIC_GRAPH_BRIDGE_DEVELOPMENT"
STRUCTURE_PROMPT = (
    '\nReturn one JSON object with exactly image_size, coordinate_system, objects and relations. '
    'Use image_size=[{width},{height}], coordinate_system="pixel_xy_top_left", and original image pixels. '
    'Each object has exactly id (its visible uppercase letter A-H), bbox=[left,top,right,bottom], '
    'and initially_active (boolean; double-ring means active). Each relation has exactly '
    'type="DIRECTED_TO", from and to (visible node letters). Interpret direction using the visible '
    'SOURCE/TAIL and TARGET/HEAD legend. Include every node/link once, no inferred final states, '
    'renderer glyph indices, answer or explanation. Array order does not matter.'
)
GROUNDED_PROMPT = (
    '\nReturn one JSON object with states (step/state), possible_answers, answer_identifiable, '
    'answer and option. Use visible node letters in each state, sorted alphabetically. '
    'Do not include reasoning or alternative submissions.'
)


def graph_structure(value):
    """Gold-independent strict visible-node/topology schema; geometry is separate."""
    if (not isinstance(value, dict) or set(value) != {"image_size", "coordinate_system", "objects", "relations"}
            or value["coordinate_system"] != "pixel_xy_top_left"
            or not isinstance(value["image_size"], list) or len(value["image_size"]) != 2
            or any(type(x) is not int or x <= 0 for x in value["image_size"])
            or not isinstance(value["objects"], list) or not value["objects"]
            or not isinstance(value["relations"], list)):
        raise ValueError("Invalid graph structure schema")
    width, height = value["image_size"]
    ids = set()
    for item in value["objects"]:
        if (not isinstance(item, dict) or set(item) != {"id", "bbox", "initially_active"}
                or not isinstance(item["id"], str) or item["id"] not in "ABCDEFGH" or len(item["id"]) != 1
                or item["id"] in ids or type(item["initially_active"]) is not bool):
            raise ValueError("Invalid or repeated visible graph node")
        box = item["bbox"]
        if (not isinstance(box, list) or len(box) != 4
                or any(type(x) not in {int, float} or (type(x) is float and not math.isfinite(x)) for x in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError("Invalid original-frame graph box")
        ids.add(item["id"])
    links = set()
    for item in value["relations"]:
        if (not isinstance(item, dict) or set(item) != {"type", "from", "to"}
                or item["type"] != "DIRECTED_TO" or type(item["from"]) is not str or type(item["to"]) is not str
                or item["from"] not in ids or item["to"] not in ids or item["from"] == item["to"]
                or (item["from"], item["to"]) in links):
            raise ValueError("Invalid or repeated graph link")
        links.add((item["from"], item["to"]))
    return dict(value, objects=sorted(value["objects"], key=lambda r: r["id"]),
                relations=sorted(value["relations"], key=lambda r: (r["from"], r["to"])))


def target_structure(scene):
    """Project observed render geometry, not solver states or final answers."""
    return graph_structure({"image_size": scene["image_size"], "coordinate_system": scene["coordinate_system"],
        "objects": [{key: item[key] for key in ("id", "bbox", "initially_active")} for item in scene["objects"]],
        "relations": [{key: item[key] for key in ("type", "from", "to")} for item in scene["relations"]]})


def select_roots(pairings, count=2, seed=3407):
    """Never read targets/outcomes or consume sealed roots to fill a quota."""
    if type(count) is not int or count <= 0 or type(seed) is not int:
        raise ValueError("Positive root count and integer seed required")
    audit.check_splits([dict(id=row["root_id"], root_id=row["root_id"],
                            family_id=row["family_id"], split=row["split"]) for row in pairings])
    chosen = sorted((row for row in pairings if row["split"] == "development"),
                    key=lambda row: digest([seed, "graph-bridge-development-v1", row["root_id"]]))[:count]
    if len(chosen) != count:
        raise ValueError("Insufficient development roots; no sealed/train substitution")
    return [row["root_id"] for row in chosen]


def prepare(pool, approvals, output, count=2, seed=3407):
    root, approvals, out = Path(pool), Path(approvals), Path(output)
    if out.exists() or out.is_symlink():
        raise FileExistsError("Refuse to overwrite a frozen cohort")
    manifest = json.loads(audit.image_path(root, "manifest.json").read_text())
    if manifest.get("version") != VERSION or manifest.get("scope") != SCOPE or any(manifest.get(k) is not v for k,v in FLAGS.items()):
        raise ValueError("Unchanged independent graph candidate required")
    for name, expected in manifest["files"].items():
        path = audit.image_path(root, name)
        if audit.sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Graph source pool bytes changed")
    pairings = audit.read_rows(root/"pairing.jsonl")
    chosen = select_roots(pairings, count, seed)
    rows = [row for arm in ARMS for row in audit.read_rows(root/(arm+".jsonl"))]
    audit.check_splits(rows)
    keyed = {(row["root_id"], row["condition"], row["view"]): row for row in rows}
    if len(keyed) != len(rows) or any(any(row.get(k) is not v for k,v in FLAGS.items()) for row in rows):
        raise ValueError("Duplicate/tainted graph condition identities")
    canonical_rows = audit.read_rows(root/"canonical_private.jsonl")
    canonical = {(row["root_id"], row["condition"]): row for row in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise ValueError("Duplicate canonical condition identities")
    training_evidence, training_rows = [], []
    for arm in ARMS:
        directory = approvals/(arm+"_approved")
        path = audit.image_path(directory, "manifest.json")
        approval = json.loads(path.read_text())
        records = audit.image_path(directory, "records.jsonl")
        approved = audit.read_rows(records)
        if (approval.get("status") != "APPROVED" or approval.get("training_allowed") is not True
                or approval.get("benchmark_derived") is not False or approval.get("contamination_audit", {}).get("status") != "PASS"
                or approval.get("model_input_audit", {}).get("status") != "PASS"
                or audit.sha(records) != approval.get("records_sha256") or len(approved) != approval.get("record_count")
                or len(approved) != approval["model_input_audit"].get("record_count")
                or not approved):
            raise ValueError("Closed train-only approval required for every condition")
        expected = {r["id"]: r for r in rows if r["condition"] == arm and r["split"] == "train"}
        if {r["id"] for r in approved} != set(expected) or len(approved) != len(expected):
            raise ValueError("Approved graph train coverage differs from the source pool")
        for row in approved:
            restored = dict(row, training_allowed=False)
            receipt = restored.pop("processor_receipt_sha256", None)
            if (restored != audit.model_input_record(expected[row["id"]])
                    or not isinstance(receipt, str) or len(receipt) != 64
                    or receipt != approval["model_input_audit"].get("receipt_sha256") or row["training_allowed"] is not True):
                raise ValueError("Approved graph row differs beyond permission/proof")
            audit.check_lineage(row, approval["sources"])
            for name in row["image_paths"]:
                if approval["files"].get(name) != audit.sha(audit.image_path(root, name)):
                    raise ValueError("Approved graph train image changed")
        training_rows.extend(approved)
        training_evidence.append({"condition": arm, "manifest_sha256": audit.sha(path),
            "records_sha256": audit.sha(records), "records": len(approved)})
    selected = [keyed[(rid, arm, view)] for rid in chosen for arm in ARMS for view in VIEWS]
    audit.check_splits(training_rows+selected)
    if ({r["root_id"] for r in selected} & {r["root_id"] for r in training_rows}
            or {r["family_id"] for r in selected} & {r["family_id"] for r in training_rows}):
        raise ValueError("Graph development overlaps an approved training root/family")
    inputs, gold, ledger, images = [], [], [], {}
    out.mkdir(parents=True, exist_ok=False)
    (out/"images").mkdir()
    for original in selected:
        # Explicit projection before placeholder normalization: target fields are
        # inaccessible to the input builder, including its placeholder check.
        row = audit.model_input_record({k: original[k] for k in ("id", "image_paths", "user_text")})
        paths, sizes = [], []
        for i, name in enumerate(row["image_paths"]):
            source = audit.image_path(root, name)
            destination = "images/"+audit.sha(source)+f"_{i}.png"
            if destination not in images:
                shutil.copyfile(source, out/destination)
            paths.append(destination)
            images[destination] = audit.sha(out/destination)
            with Image.open(source) as image:
                sizes.append(list(image.size))
        user = row["user_text"]
        if original["view"] == "structure":
            user += STRUCTURE_PROMPT.format(width=sizes[0][0], height=sizes[0][1])
        elif original["view"] == "grounded":
            user += GROUNDED_PROMPT
        item = {"id": row["id"], "subject": "signal_graph", "question_type": "open", "question": "",
                "options": [], "user_text": user, "image_paths": paths, **FLAGS}
        item["input_record_sha256"] = digest(item)
        inputs.append(item)
        ledger.append({"id": row["id"], "root_id": original["root_id"], "family_id": original["family_id"],
            "condition": original["condition"], "view": original["view"], "split": "development", **FLAGS})
    # Freeze all input records before reading selected assistant targets/options.
    for original, entry in zip(selected, ledger, strict=True):
        target, kind = original["target_text"], original["view"]
        if kind == "structure":
            target, kind = json.dumps(target_structure(json.loads(target)), sort_keys=True), "graph_structure"
        gold.append({"target": target, "metric_kind": kind,
            "options": [str(value) for value in canonical[(original["root_id"], original["condition"])]["options"]] if kind == "qa" else [],
            **entry})
    audit.write_rows(out/"inputs.jsonl", inputs)
    audit.write_rows(out/"gold.jsonl", gold)
    audit.write_rows(out/"pairing.jsonl", ledger)
    result = {"status": "COHORT_FROZEN", "evaluation_domain": DOMAIN, "count": len(inputs),
        "root_count": count, "root_ids": chosen, "split": "development", "conditions": list(ARMS), "views": list(VIEWS),
        "inputs_sha256": audit.sha(out/"inputs.jsonl"), "gold_sha256": audit.sha(out/"gold.jsonl"),
        "pairing_sha256": audit.sha(out/"pairing.jsonl"), "images": images,
        "pool_manifest_sha256": audit.sha(root/"manifest.json"), "training_evidence": training_evidence,
        "builder_source_sha256": audit.sha(__file__), "evaluator_source_sha256": audit.sha(Path(__file__).with_name("evaluate.py")),
        "prompt_sha256": digest([STRUCTURE_PROMPT, GROUNDED_PROMPT]), "seed": seed,
        "selection_policy": "development root hash, no targets/outcomes; all A0-A3 and three views, same cohort for every checkpoint",
        "metric_policy": "QA V8; graph typed topology exact and leaf F1 ignoring bbox, bbox IoU>=0.5 separately; grounded answer/states separate",
        "limits": "Bounded synthetic registered-family development, not fullG7/non-synthetic transfer/sealed holdout; shared roots are not independent items",
        "GPU_evaluation_completed": False, **FLAGS}
    audit.dump(out/"manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("pool", "approvals", "output"):
        parser.add_argument("--"+field, type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    print(json.dumps(prepare(args.pool, args.approvals, args.output, args.count, args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
