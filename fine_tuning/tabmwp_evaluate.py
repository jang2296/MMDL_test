"""Frozen teacher-free TabMWP A/B external evaluation, separate from MMMU/V8 runs.

Prepare is CPU-only; infer never opens gold; score runs only after local recovery.
The 8K engine recipe is reused, not its MMMU population or diagnostic prompts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

from experiments.bottleneck_diagnostic.input_builder import stage_seed
from experiments.bottleneck_diagnostic.runner import _loaded_request, _verify_images
from experiments.input_recovery import runner as recovery
from mmdl.evaluation.parsers import score_with_parser, scoring_source_files
from mmdl.runtime.artifacts import atomic_bytes, atomic_text, digest, read_json, sha256_file, write_json

from . import tabmwp, tabmwp_images
from .audit import image_path, read_rows

PROTOCOL_ID = "tabmwp-ab-external8k-v1"
CANDIDATES = ("F0", "A", "B")
COUNTS = {"QA": 48, "FACTS": 24, "RETENTION": 60}
RETENTION_MANIFEST_SHA256 = "b75afba0a4f70e785282ee3ba664e770a0c90c0b43ef5654b331f40005b1755f"
SAMPLE_FIELDS = {"sample_id", "native_id", "mode", "subject", "question_type", "question",
                 "options", "user_text", "images", "family_id", "root_id"}
FACTS_PROMPT = (
    "Transcribe only the visible table. Return exactly one JSON object with keys title, headers, "
    "rows. title is its visible title string, or null when absent. headers is the ordered list "
    "of visible column-header strings, or null for a headerless table. rows is an ordered list "
    "of rows, each an ordered list of cell strings. Preserve spelling, units, and cell order; "
    "do not invent pandas headers, calculate an answer, or include commentary."
)


def _indexed(values, key="sample_id"):
    return recovery.unique(values, key)


def _new_destination(path, sources=()):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError("Immutable output must be new")
    resolved = path.resolve()
    if resolved.is_relative_to(recovery.ROOT / "MMDL"):
        raise ValueError("Professor submission checkout is protected")
    if any(resolved.is_relative_to(Path(source).resolve())
           or Path(source).resolve().is_relative_to(resolved) for source in sources):
        raise ValueError("Output must be separate from immutable inputs and private gold")
    return path


def _input_key(text, hashes):
    return digest({"user_text": text, "ordered_image_sha256": hashes})


def _training_inputs(records_path, image_root, native_ids):
    records = read_rows(records_path)
    _indexed(records, "id")
    native = [row for row in records if row.get("source_id") == tabmwp.SOURCE_ID]
    if (len(records) != 2000 or {row["id"] for row in native} != set(native_ids)
            or len(native) != 1600 or any(row.get("split") != "train"
                or row.get("training_allowed") is not True or row.get("benchmark_derived") is not False
                for row in records)):
        raise ValueError("The fixed native1600 plus retention400 training mixture is required")
    families, roots, inputs, sources, ordered = set(), set(), set(), {str(records_path): sha256_file(records_path)}, []
    for row in records:
        families.add(row["family_id"])
        roots.add(row["root_id"])
        hashes = []
        for name in row["image_paths"]:
            path = image_path(image_root, name)
            sources[str(path)] = sha256_file(path)
            hashes.append(sources[str(path)])
        inputs.add(_input_key(row["user_text"], hashes))
        ordered.append({"id": row["id"], "family_id": row["family_id"], "root_id": row["root_id"],
                        "user_text": row["user_text"], "ordered_image_sha256": hashes})
    evidence = {"path": str(records_path), "sha256": sources[str(records_path)], "record_count": 2000,
                "input_order_sha256": digest(ordered),
                "source_counts": dict(Counter(row["source_id"] for row in records)),
                "retention_records_sha256": digest([row for row in records if row["source_id"] != tabmwp.SOURCE_ID])}
    return families, roots, inputs, sources, evidence


def prepare(prepared_root, images_root, retention_root, training_records, training_image_root,
            output, gold_output):
    """Freeze fixed development requests, with sealed data used only for exclusion/validation."""
    prepared_root, images_root, retention_root = map(Path, (prepared_root, images_root, retention_root))
    training_records, training_image_root = map(Path, (training_records, training_image_root))
    sources = (prepared_root.parent, images_root, retention_root, training_image_root, training_records)
    output = _new_destination(output, (*sources, gold_output))
    gold_output = _new_destination(gold_output, (*sources, output))
    paths, bound, preparation_hash = tabmwp_images.selection(prepared_root)
    image_receipt_path = images_root / "receipt_01.json"
    # The approved acquisition may need bounded continuation; take its last immutable receipt.
    receipts = sorted(images_root.glob("receipt_*.json"))
    if receipts:
        image_receipt_path = receipts[-1]
    receipt = read_json(image_receipt_path)
    entries = tabmwp_images.verify_tree(read_json(images_root / "commit.json"),
                                       read_json(images_root / "tree.json"))
    if (receipt.get("status") != "COMPLETE" or receipt.get("preparation_hash") != preparation_hash
            or receipt.get("revision") != tabmwp.REVISION
            or receipt.get("tree_sha1") != tabmwp_images.TREE_SHA
            or receipt.get("selection_sha256") != digest(paths)
            or set(receipt.get("files", {})) != set(paths)
            or receipt.get("expected_count") != 2000 or receipt.get("verified_count") != 2000
            or receipt.get("errors") or receipt.get("changed_sources")
            or receipt.get("tree_metadata_sha256") != sha256_file(images_root / "tree.json")
            or receipt.get("commit_metadata_sha256") != sha256_file(images_root / "commit.json")):
        raise ValueError("Complete pinned source/selection-bound image acquisition required")
    for path in (image_receipt_path, images_root / "commit.json", images_root / "tree.json"):
        bound[str(path)] = sha256_file(path)
    native_train = read_rows(prepared_root / "A/train.jsonl")
    dev = _indexed(read_rows(prepared_root / "A/dev.jsonl"), "id")
    sealed = read_rows(prepared_root / "A/sealed.jsonl")
    families, roots, train_inputs, training_sources, training_evidence = _training_inputs(
        training_records, training_image_root, [row["id"] for row in native_train])
    bound.update(training_sources)
    families.update(row["family_id"] for row in native_train + sealed)
    roots.update(row["root_id"] for row in native_train + sealed)
    for row in native_train + sealed:
        train_inputs.add(_input_key(row["user_text"],
                                    [receipt["files"][name]["sha256"] for name in row["image_paths"]]))
    eligible = [row for row in dev.values() if row["family_id"] not in families
                and row["root_id"] not in roots
                and _input_key(row["user_text"], [receipt["files"][name]["sha256"]
                                                  for name in row["image_paths"]]) not in train_inputs]
    chosen = sorted(eligible, key=lambda row: (digest([3407, PROTOCOL_ID, row["id"]]), row["id"]))[:48]
    if len(chosen) != 48:
        raise ValueError("Insufficient input/family-disjoint native development items")
    retention_manifest = read_json(retention_root / "manifest.json")
    if (sha256_file(retention_root / "manifest.json") != RETENTION_MANIFEST_SHA256
            or retention_manifest.get("status") != "COHORT_FROZEN" or retention_manifest.get("count") != 60
            or retention_manifest.get("inputs_sha256") != sha256_file(retention_root / "inputs.jsonl")
            or retention_manifest.get("gold_sha256") != sha256_file(retention_root / "gold.jsonl")):
        raise ValueError("The immutable existing external60 cohort is required")
    retention = read_rows(retention_root / "inputs.jsonl")
    retention_gold = _indexed(read_rows(retention_root / "gold.jsonl"), "id")
    if (len(retention) != 60 or set(_indexed(retention, "id")) != set(retention_gold)
            or Counter(row["metric_kind"] for row in retention_gold.values())
            != {"qa": 12, "trace_qa": 24, "structure": 12, "grounded": 12}):
        raise ValueError("Exact retained60 IDs and metric coverage required")
    for name in ("manifest.json", "inputs.jsonl", "gold.jsonl"):
        bound[str(retention_root / name)] = sha256_file(retention_root / name)
    samples, gold, payloads = [], [], {}

    def image(source_root, name, expected):
        path = image_path(source_root, name)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Source image bytes differ from frozen acquisition")
        with Image.open(path) as opened:
            size = list(opened.size)
            opened.verify()
        bound[str(path)] = expected
        relative = f"images/{expected}.png"
        payloads[relative] = data
        return {"path": relative, "sha256": expected, "original_size": size}

    # Read answers only after the outcome-independent development selection is fixed.
    problems, _ = tabmwp.load_source(prepared_root.parent)
    for index, row in enumerate(chosen):
        pid = row["source_native_id"]
        native = problems["dev"][pid]
        name = row["image_paths"][0]
        data = image_path(images_root, name).read_bytes()
        proof = tabmwp_images.verify_png(data, entries["data/tabmwp/" + name])
        if any(proof[key] != receipt["files"][name][key] for key in proof):
            raise ValueError("Selected PNG receipt differs from actual pinned pixels")
        images = [image(images_root, name, proof["sha256"])]
        options = native["choices"] or []
        kind = "multiple-choice" if native["ques_type"] == "multi_choice" else "open"
        answer = native["answer"]
        if kind == "multiple-choice":
            if options.count(answer) != 1:
                raise ValueError("Native MCQ answer must identify exactly one original option")
            answer = chr(65 + options.index(answer))
        sample = {"sample_id": row["id"] + "__QA", "native_id": row["id"], "mode": "QA",
                  "subject": "TabMWP", "question_type": kind, "question": native["question"],
                  "options": options, "user_text": row["user_text"], "images": images,
                  "family_id": row["family_id"], "root_id": row["root_id"]}
        samples.append(sample)
        gold.append({"sample_id": sample["sample_id"], "metric_kind": "tabmwp_qa",
                     "answer": answer, "native_answer": native["answer"]})
        if index < 24:
            facts = dict(sample, sample_id=row["id"] + "__FACTS", mode="FACTS",
                         question_type="open", question="", options=[], user_text=FACTS_PROMPT)
            samples.append(facts)
            gold.append({"sample_id": facts["sample_id"], "metric_kind": "table_facts",
                         "target": tabmwp.table_facts(native)})
    for row in retention:
        target = retention_gold[row["id"]]
        images = [image(retention_root, name, retention_manifest["images"][name])
                  for name in row["image_paths"]]
        if (target["family_id"] in families or target["root_id"] in roots
                or _input_key(row["user_text"], [item["sha256"] for item in images]) in train_inputs):
            raise ValueError("Existing retention60 overlaps actual training/sealed inputs or families")
        sample = {"sample_id": row["id"] + "__RETENTION", "native_id": row["id"],
                  "mode": "RETENTION", "subject": row["subject"], "question_type": row["question_type"],
                  "question": row["question"], "options": row["options"], "user_text": row["user_text"],
                  "images": images, "family_id": target["family_id"], "root_id": target["root_id"]}
        samples.append(sample)
        gold.append(dict(target, sample_id=sample["sample_id"]))
    _indexed(samples)
    if any(sha256_file(path) != value for path, value in bound.items()):
        raise ValueError("Preparation source changed; no frozen outputs emitted")
    output.mkdir(parents=True, exist_ok=False)
    gold_output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        atomic_bytes(output / name, data)
    atomic_text(output / "samples.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples))
    atomic_text(gold_output / "gold.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold))
    manifest = {"status": "COHORT_FROZEN", "protocol_id": PROTOCOL_ID, "count": 132,
                "counts": COUNTS, "tabmwp_preparation_hash": preparation_hash,
                "qa_ids": [row["id"] for row in chosen], "facts_ids": [row["id"] for row in chosen[:24]],
                "retention_ids": [row["id"] for row in retention],
                "selection_policy": "SHA256 canonical [3407, protocol_id, native_id]; first48 QA/first24 facts; existing retention60 unchanged",
                "samples_sha256": sha256_file(output / "samples.jsonl"),
                "approved_training_records": training_evidence,
                "sample_hashes": {row["sample_id"]: digest(row) for row in samples},
                "gold_sha256": sha256_file(gold_output / "gold.jsonl"),
                "source_sha256": bound, "facts_prompt_sha256": digest(FACTS_PROMPT),
                "training_allowed": False, "public_upload_allowed": False,
                "trust": "SOURCE_GOLD; teacher-free native table JSON/text, not human pixel/solver certification",
                "expansion_policy": "QA paired gains>losses AND cells recall/exact improve AND equal-weight retention QA/TRACE primary>=freshF0; no automatic50K"}
    manifest["cohort_hash"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    write_json(gold_output / "binding.json", {"cohort_hash": manifest["cohort_hash"],
               "gold_sha256": manifest["gold_sha256"], "training_allowed": False,
               "public_upload_allowed": False})
    return manifest


def load_cohort(data_root):
    """Gold is intentionally not an argument, opened file, or inference sample field."""
    root = Path(data_root)
    manifest = read_json(root / "manifest.json")
    if (manifest.get("cohort_hash") != digest({k: v for k, v in manifest.items() if k != "cohort_hash"})
            or manifest.get("status") != "COHORT_FROZEN" or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("counts") != COUNTS or manifest.get("count") != sum(COUNTS.values())
            or manifest.get("training_allowed") is not False or manifest.get("public_upload_allowed") is not False
            or manifest.get("samples_sha256") != sha256_file(root / "samples.jsonl")
            or manifest.get("facts_prompt_sha256") != digest(FACTS_PROMPT)):
        raise ValueError("Frozen external cohort/hash/count differs")
    samples = _indexed(read_rows(root / "samples.jsonl"))
    if (Counter(row["mode"] for row in samples.values()) != COUNTS
            or manifest["sample_hashes"] != {sid: digest(row) for sid, row in samples.items()}):
        raise ValueError("External request coverage or sample identity differs")
    qa = [row["native_id"] for row in samples.values() if row["mode"] == "QA"]
    facts = [row["native_id"] for row in samples.values() if row["mode"] == "FACTS"]
    retention = [row["native_id"] for row in samples.values() if row["mode"] == "RETENTION"]
    if qa != manifest["qa_ids"] or facts != qa[:24] or facts != manifest["facts_ids"] or retention != manifest["retention_ids"]:
        raise ValueError("Frozen QA/facts/retention order differs")
    for sample in samples.values():
        if (set(sample) != SAMPLE_FIELDS or not isinstance(sample["user_text"], str)
                or not sample["user_text"].strip() or not sample["images"]
                or sample["question_type"] not in {"multiple-choice", "open"}
                or not isinstance(sample["options"], list)
                or any(not isinstance(value, str) for value in sample["options"])
                or sample["mode"] == "FACTS" and sample["user_text"] != FACTS_PROMPT):
            raise ValueError("Unknown/gold-bearing or malformed inference sample")
    return manifest, samples


def make_requests(samples, candidate):
    if candidate not in CANDIDATES:
        raise ValueError("Unknown A/B candidate")
    requests = []
    for sid, sample in samples.items():
        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": item["path"]} for item in sample["images"]],
            {"type": "text", "text": sample["user_text"]}]}]
        request = {"sample_id": f"{candidate}__{sid}", "original_sample_id": sid,
                   "mode": sample["mode"], "seed": stage_seed(3407, sid, sample["mode"]),
                   "max_new_tokens": 8192, "messages": messages, "images": sample["images"],
                   "sample_hash": digest(sample), "prompt_hash": digest(messages),
                   "training_allowed": False, "public_upload_allowed": False}
        request["request_hash"] = digest(request)
        requests.append(request)
    return requests


def candidate_identity(model_path, candidate, cohort, training_manifest=None,
                       training_manifest_sha256=None, checkpoint_metadata=None):
    export_arm = "F0" if candidate == "F0" else "FM"
    identity = recovery.model_identity(Path(model_path), export_arm)
    if candidate == "F0":
        if any(value is not None for value in (training_manifest, training_manifest_sha256, checkpoint_metadata)):
            raise ValueError("Fresh F0 cannot carry training evidence")
        return identity
    if (training_manifest is None or checkpoint_metadata is None
            or not isinstance(training_manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", training_manifest_sha256)):
        raise ValueError("New A/B requires an explicitly frozen training manifest and checkpoint metadata")
    training_manifest, checkpoint_metadata = Path(training_manifest), Path(checkpoint_metadata)
    if sha256_file(training_manifest) != training_manifest_sha256:
        raise ValueError("Training manifest changed from its accepted frozen hash")
    manifest, metadata = read_json(training_manifest), read_json(checkpoint_metadata)
    export = read_json(Path(model_path) / "export_manifest.json")
    source = manifest.get("sources", {}).get(tabmwp.SOURCE_ID, {})
    current_sources = {name: sha256_file(Path(__file__).with_name(name)) for name in ("train.py", "modules.py")}
    if (manifest.get("status") != "APPROVED" or manifest.get("training_allowed") is not True
            or manifest.get("benchmark_derived") is not False or manifest.get("record_count") != 2000
            or manifest.get("tabmwp_condition") != candidate
            or manifest.get("tabmwp_preparation_hash") != cohort["tabmwp_preparation_hash"]
            or tabmwp.SOURCE_ID not in manifest.get("sources", {})
            or source.get("rights_approved") is not True or source.get("source_known") is not True
            or source.get("benchmark_derived") is not False
            or manifest.get("source_sha256") != current_sources
            or metadata.get("source_sha256") != current_sources
            or metadata.get("manifest_sha256") != training_manifest_sha256
            or metadata.get("records_sha256") != manifest.get("records_sha256")
            or metadata.get("base_id") != recovery.MODEL_ID or metadata.get("base_revision") != recovery.MODEL_REVISION
            or metadata.get("arm") != "FM" or metadata.get("seed") != 3407
            or metadata.get("termination") != "EPOCH_COMPLETE" or metadata.get("epoch_exposure_complete") is not True
            or metadata.get("epochs") != 1 or metadata.get("planned_records") != 2000
            or metadata.get("exposure", {}).get("records") != 2000
            or metadata.get("exposure", {}).get("updates") != 125
            or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("consumption_order_sha256", "")))
            or metadata.get("consumed_order_sha256") != manifest.get("consumption_order_sha256")
            or metadata.get("record_order_sha256") != metadata.get("consumed_order_sha256")
            or metadata.get("budgets", {}).get("accumulation") != 16
            or metadata.get("budgets", {}).get("max_steps") != 0
            or metadata.get("base_precision") != "BF16" or metadata.get("trainable_master_precision") != "FP32"
            or export.get("checkpoint_files", {}).get("metadata.json") != sha256_file(checkpoint_metadata)
            or export.get("source_sha256", {}).get("modules.py") != current_sources["modules.py"]
            or export.get("source_sha256", {}).get("checkpoints.py") != sha256_file(recovery.ROOT / "experiments/input_recovery/checkpoints.py")):
        raise ValueError("Legacy/incomplete/wrong-condition A/B training or exported checkpoint identity")
    contract = metadata.get("training_input_contract")
    # The supplied approved manifest fixes actual train budgets; never infer them from old G6.
    proof = manifest.get("model_input_audit", {})
    if (not isinstance(contract, dict) or proof.get("status") != "PASS"
            or proof.get("record_count") != 2000
            or any(metadata.get("budgets", {}).get(key) != contract.get(key)
                   for key in ("min_pixels", "max_pixels", "episode_pixels", "sequence_budget"))
            or any(proof.get(key) != contract.get(key) for key in ("sequence_budget", "episode_pixels"))
            or contract.get("processor_revision", recovery.MODEL_REVISION) != recovery.MODEL_REVISION):
        raise ValueError("New A/B training budgets differ from their fixed approved processor proof")
    return identity | {"candidate": candidate, "training_manifest_sha256": training_manifest_sha256,
                       "checkpoint_metadata_sha256": sha256_file(checkpoint_metadata),
                       "training_contract": contract, "accepted_training_manifest": manifest}


def infer(data_root, model_path, base_path, output, candidate, deadline_utc, commit,
          training_manifest=None, training_manifest_sha256=None, checkpoint_metadata=None,
          backend_factory=None, slots=4):
    data_root, model_path, base_path, output = map(Path, (data_root, model_path, base_path, output))
    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    if type(slots) is not int or slots != 4:
        raise ValueError("External A/B evaluation requires exactly four continuous slots")
    if deadline.utcoffset() is None or datetime.now(timezone.utc) >= deadline:
        raise ValueError("An unexpired absolute timezone-aware deadline is required")
    recovery.checkout(recovery.ROOT, commit)
    if output.is_symlink() or output.resolve().is_relative_to(recovery.ROOT / "MMDL") or output.resolve().is_relative_to(data_root.resolve()):
        raise ValueError("Inference output must be outside the protected/input directories")
    cohort, samples = load_cohort(data_root)
    requests = make_requests(samples, candidate)
    requested = _indexed(requests)
    _verify_images(data_root.resolve(), list(samples.values()))
    config = yaml.safe_load(recovery.PROTOCOL.read_text())
    if config["execution"]["batch_size"] != 4 or config["generation"]["max_new_tokens"] != 8192:
        raise ValueError("Frozen input-recovery-v1 8K/four continuous slots required")
    model = candidate_identity(model_path, candidate, cohort, training_manifest,
                               training_manifest_sha256, checkpoint_metadata)
    identity = {"protocol_id": PROTOCOL_ID, "engine_protocol": config,
                "population": "48 TabMWP development QA +24 same-table facts +existing external60; not MMMU",
                "prompt_policy": "native QA P0; fixed table-facts transcription; unchanged retention prompts",
                "candidate": candidate, "model": model, "processor_base": recovery.model_identity(base_path, "F0"),
                "code_commit": commit, "deadline_utc": deadline_utc,
                "cohort_hash": cohort["cohort_hash"],
                "input_files": {name: sha256_file(data_root / name) for name in ("manifest.json", "samples.jsonl")},
                "run_kind": "ACTUAL_INFERENCE" if backend_factory is None else "TEST_ONLY",
                "training_allowed": False, "public_upload_allowed": False}
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).is_symlink() for name in (".lock", "manifest.json", "status.json", "responses")):
        raise ValueError("Symlink inference control/output artifacts forbidden")
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != identity:
                raise ValueError("Resume changed candidate/model/inputs/protocol/code/original deadline")
        elif any(path.name != ".lock" for path in output.iterdir()):
            raise ValueError("Nonempty unbound inference directory")
        else:
            write_json(manifest_path, identity)
        response_root = output / "responses"
        response_root.mkdir(exist_ok=True)
        done = _read_completions(response_root, requested, candidate, identity["run_kind"])
        pending = [request for request in requests if request["sample_id"] not in done]
        started = time.monotonic()

        def status(state):
            write_json(output / "status.json", {"status": state, "candidate": candidate,
                       "expected": 132, "completed": len(done), "missing": 132 - len(done),
                       "max_active_requests": 4, "gpu_deadline_utc": deadline_utc,
                       "wall_seconds_this_process": time.monotonic() - started})

        status("RUNNING")
        if not pending:
            status("COMPLETED")
            return identity
        if backend_factory is None:
            from experiments.input_recovery.backend import create_backend
            backend_factory = create_backend
        stream = None
        try:
            backend = backend_factory(model_path, base_path, "F0" if candidate == "F0" else "FM", config, deadline_utc)
            backend_path = output / f"backend_{len(list(output.glob('backend_*.json'))):03d}.json"
            if backend_path.exists() or backend_path.is_symlink():
                raise ValueError("Prior backend attempt evidence is immutable")
            write_json(backend_path, backend.describe())
            stream = backend.generate_stream(_loaded_request(request, data_root.resolve()) for request in pending)
            for raw in stream:
                if datetime.now(timezone.utc) >= deadline:
                    raise TimeoutError("Original absolute deadline reached")
                sid = raw.get("sample_id")
                if sid not in requested or sid in done or not isinstance(raw.get("raw_response"), str):
                    raise ValueError("Unknown/duplicate/invalid backend completion")
                request = requested[sid]
                row = raw | {"original_sample_id": request["original_sample_id"], "candidate": candidate,
                             "mode": request["mode"], "repetition": 3407, "request": request,
                             "request_hash": request["request_hash"], "runtime_status": "COMPLETED",
                             "run_kind": identity["run_kind"], "training_allowed": False, "public_upload_allowed": False}
                row["record_hash"] = digest(row)
                _validate_response(row, request, candidate, identity["run_kind"])
                write_json(response_root / (request["request_hash"] + ".json"), row)
                done[sid] = row
                status("RUNNING")
        except BaseException:
            status("INCOMPLETE")
            raise
        finally:
            if stream is not None:
                stream.close()
        status("COMPLETED" if len(done) == 132 else "INCOMPLETE")
    return identity


def _validate_response(row, request, candidate, run_kind):
    if (row.get("candidate") != candidate or row.get("run_kind") != run_kind
            or not isinstance(row.get("raw_response"), str)
            or row.get("finish_reason") not in {"stop", "eos", "length"}
            or any(type(row.get(key)) is not int or row[key] < 0 for key in ("input_tokens", "generated_tokens"))
            or row["generated_tokens"] > 8192 or row["input_tokens"] <= 0):
        raise ValueError("Invalid or non-completed inference response")
    recovery.validate_completion(row, request)


def _read_completions(response_root, requested, candidate, run_kind):
    if Path(response_root).is_symlink():
        raise ValueError("Symlink response directory forbidden")
    done = {}
    for path in sorted(Path(response_root).glob("*.json")):
        row = read_json(path)
        sid = row.get("sample_id")
        if (path.is_symlink() or sid not in requested or sid in done or path.stem != row.get("request_hash")):
            raise ValueError("Unknown/duplicate/unbound or non-completed inference response")
        _validate_response(row, requested[sid], candidate, run_kind)
        done[sid] = row
    return done


def score_qa(raw, sample, target, finish_reason):
    """A/B JSON answer extraction uses only the complete submission and original choices."""
    from .evaluate import terminal_json
    text, submitted = raw, str(raw).strip()
    if submitted.startswith(("{", "```")):
        try:
            value = terminal_json(raw)
            answer = value.get("answer")
            if (set(value) not in ({"answer"}, {"facts", "answer"})
                    or type(answer) not in {str, int, float} or not str(answer).strip()
                    or type(answer) is float and not math.isfinite(answer)):
                raise ValueError("A/B JSON must submit one scalar answer")
            text = str(answer)
            # Native MCQ targets are option text, whereas V8 compares option labels.
            matches = [index for index, option in enumerate(sample["options"]) if option == text]
            if sample["question_type"] == "multiple-choice" and len(matches) == 1:
                text = chr(65 + matches[0])
            text = "Final answer: " + text
        except (ValueError, TypeError):
            return {"correct": False, "parse_status": "NO_PARSE", "parsed_answer": None,
                    "answer_source": "INVALID_WHOLE_JSON"}
        source = "WHOLE_JSON_ANSWER"
    else:
        source = "V8_FINAL_ANSWER"
    result = score_with_parser(text, sample["question_type"], sample["options"], target["answer"],
                               "team-final-answer-v8", finish_reason, question=sample["question"])
    return result | {"answer_source": source}


def score_facts(raw, target):
    """Ordered string facts, no coercion/tolerance; format and semantic field scores are separate."""
    from .evaluate import terminal_json
    try:
        submitted = terminal_json(raw)
    except (ValueError, TypeError):
        submitted = None
    observed = submitted
    if isinstance(submitted, dict) and set(submitted) == {"facts", "answer"}:
        observed = submitted["facts"]
    observed = observed if isinstance(observed, dict) else {}
    rows = observed.get("rows")
    rows = rows if isinstance(rows, list) else []
    headers = observed.get("headers")
    valid_rows = bool(rows) and all(isinstance(row, list) and bool(row)
                      and all(type(cell) is str for cell in row) for row in rows)
    valid_rows = valid_rows and len({len(row) for row in rows}) == 1
    schema = (isinstance(submitted, dict) and set(submitted) == {"title", "headers", "rows"}
              and (observed.get("title") is None or type(observed.get("title")) is str)
              and (headers is None or isinstance(headers, list) and bool(headers)
                   and all(type(cell) is str for cell in headers)) and valid_rows)
    def equal(key):
        return key in observed and digest(observed[key]) == digest(target[key])
    hits = sum(index < len(rows) and isinstance(rows[index], list) and column < len(rows[index])
               and type(rows[index][column]) is str and rows[index][column] == cell
               for index, row in enumerate(target["rows"]) for column, cell in enumerate(row))
    denominator = sum(len(row) for row in target["rows"])
    observed_cells = sum(len(row) for row in rows if isinstance(row, list))
    expected_headers = target["headers"] or []
    header_hits = sum(isinstance(headers, list) and index < len(headers)
                      and type(headers[index]) is str and headers[index] == cell
                      for index, cell in enumerate(expected_headers))
    exact = all(equal(key) for key in ("title", "headers", "rows"))
    return {"correct": exact, "facts_exact": exact, "json_valid": submitted is not None,
            "schema_valid": bool(schema), "title_exact": equal("title"), "headers_exact": equal("headers"),
            "rows_exact": equal("rows"), "cell_hits": hits, "cell_denominator": denominator,
            "observed_cells": observed_cells, "cell_recall": hits / denominator,
            "cell_precision": hits / observed_cells if observed_cells else 0.0,
            "header_hits": header_hits, "header_denominator": len(expected_headers),
            "parse_status": "NO_PARSE" if submitted is None else "PARSED" if schema else "SCHEMA_INVALID"}


def _group(cells, expected):
    return {"planned": expected, "completed": len(cells), "correct": sum(row["correct"] for row in cells),
            "accuracy": sum(row["correct"] for row in cells) / expected if len(cells) == expected else None,
            "partial_completed_accuracy": sum(row["correct"] for row in cells) / len(cells) if cells else None,
            "status": "COMPLETED" if len(cells) == expected else "INCOMPLETE"}


def score(data_root, gold_root, run_roots, output):
    """Explicit local CPU replay; incomplete denominators and expansion gates stay null."""
    from .evaluate import score_external
    data_root, gold_root, output = map(Path, (data_root, gold_root, output))
    output = _new_destination(output, (data_root, gold_root, *run_roots))
    cohort, samples = load_cohort(data_root)
    binding = read_json(gold_root / "binding.json")
    if (binding.get("cohort_hash") != cohort["cohort_hash"]
            or binding.get("gold_sha256") != cohort["gold_sha256"]
            or sha256_file(gold_root / "gold.jsonl") != cohort["gold_sha256"]):
        raise ValueError("Gold differs from its independent frozen cohort binding")
    gold = _indexed(read_rows(gold_root / "gold.jsonl"))
    if set(gold) != set(samples):
        raise ValueError("Gold/request coverage differs")
    cells, run_sources, manifests = [], {}, {}
    for run_root in map(Path, run_roots):
        manifest = read_json(run_root / "manifest.json")
        candidate = manifest["candidate"]
        if candidate not in CANDIDATES or candidate in manifests:
            raise ValueError("Unknown or repeated candidate run root")
        config = yaml.safe_load(recovery.PROTOCOL.read_text())
        if (manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("engine_protocol") != config
                or manifest.get("cohort_hash") != cohort["cohort_hash"]
                or manifest.get("input_files") != {name: sha256_file(data_root / name)
                                                   for name in ("manifest.json", "samples.jsonl")}
                or manifest.get("run_kind") not in {"TEST_ONLY", "ACTUAL_INFERENCE"}
                or manifest.get("model", {}).get("arm") != ("F0" if candidate == "F0" else "FM")):
            raise ValueError("Run population/engine/cohort/model identity differs")
        manifests[candidate] = manifest
        run_sources[str(run_root)] = sha256_file(run_root / "manifest.json")
        requests = _indexed(make_requests(samples, candidate))
        done = _read_completions(run_root / "responses", requests, candidate, manifest["run_kind"])
        for row in done.values():
            sid = row["original_sample_id"]
            sample, target = samples[sid], gold[sid]
            if sample["mode"] == "QA":
                result = score_qa(row["raw_response"], sample, target, row["finish_reason"])
            elif sample["mode"] == "FACTS":
                result = score_facts(row["raw_response"], target["target"])
            else:
                result = score_external(row["raw_response"], target, row["finish_reason"])
            path = run_root / "responses" / (row["request_hash"] + ".json")
            cells.append({"sample_id": sid, "candidate": candidate, "mode": sample["mode"],
                          "metric_kind": target["metric_kind"], **result,
                          "finish_reason": row["finish_reason"], "generated_tokens": row["generated_tokens"],
                          "input_tokens": row["input_tokens"], "run_kind": manifest["run_kind"],
                          "response_sha256": sha256_file(path), "response_path": str(path),
                          "training_allowed": False, "public_upload_allowed": False})
    if (len({m["code_commit"] for m in manifests.values()}) > 1
            or len({digest(m["processor_base"]) for m in manifests.values()}) > 1):
        raise ValueError("Paired candidates must use the same fixed code and processor base")
    by_cell = {(row["candidate"], row["sample_id"]): row for row in cells}
    missing = [f"{candidate}__{sid}" for candidate in CANDIDATES for sid in samples
               if (candidate, sid) not in by_cell]
    groups = {}
    for candidate in CANDIDATES:
        selected = [row for row in cells if row["candidate"] == candidate]
        groups[candidate] = {mode: _group([row for row in selected if row["mode"] == mode], count)
                             for mode, count in COUNTS.items()}
        facts = [row for row in selected if row["mode"] == "FACTS"]
        denominator = sum(sum(len(row) for row in gold[sid]["target"]["rows"])
                          for sid, sample in samples.items() if sample["mode"] == "FACTS")
        hits = sum(row["cell_hits"] for row in facts)
        groups[candidate]["FACTS"].update(cell_hits=hits, cell_denominator=denominator,
            cell_recall=hits / denominator if len(facts) == 24 else None,
            json_valid=sum(row["json_valid"] for row in facts), schema_valid=sum(row["schema_valid"] for row in facts),
            headers_exact=sum(row["headers_exact"] for row in facts), title_exact=sum(row["title_exact"] for row in facts))
        retention = {}
        for kind, expected in (("qa", 12), ("trace_qa", 24), ("structure", 12), ("grounded", 12)):
            retention[kind] = _group([row for row in selected if row["mode"] == "RETENTION" and row["metric_kind"] == kind], expected)
        groups[candidate]["retention_metrics"] = retention
        primary = [retention[kind]["accuracy"] for kind in ("qa", "trace_qa")]
        groups[candidate]["retention_primary"] = sum(primary) / 2 if None not in primary else None
    complete = not missing
    actual = complete and all(m["run_kind"] == "ACTUAL_INFERENCE" for m in manifests.values())
    gates = {}
    for candidate in ("A", "B"):
        pairs = [(by_cell.get(("F0", sid)), by_cell.get((candidate, sid)))
                 for sid, sample in samples.items() if sample["mode"] == "QA"]
        observed = [(left, right) for left, right in pairs if left is not None and right is not None]
        gains = sum(not left["correct"] and right["correct"] for left, right in observed)
        losses = sum(left["correct"] and not right["correct"] for left, right in observed)
        facts, base = groups[candidate]["FACTS"], groups["F0"]["FACTS"]
        passes = (gains > losses and facts["correct"] > base["correct"]
                  and facts["cell_recall"] > base["cell_recall"]
                  and groups[candidate]["retention_primary"] >= groups["F0"]["retention_primary"]) if complete else None
        gates[candidate] = {"paired_qa_completed": len(observed), "qa_gains": gains, "qa_losses": losses,
                            "criteria_pass": passes, "expansion_eligible": bool(passes) if actual else None,
                            "status": "EVIDENCE_ONLY_NO_AUTOMATIC_TRAINING" if actual else "INCOMPLETE_OR_TEST_ONLY",
                            "automatic_50k_training_allowed": False}
    summary = {"protocol_id": PROTOCOL_ID, "status": "COMPLETED" if complete else "INCOMPLETE",
               "planned_per_candidate": 132, "planned_total": 396, "completed": len(cells),
               "missing_request_ids": missing, "groups": groups, "expansion_gate": gates,
               "source_manifests": run_sources, "gold_sha256": cohort["gold_sha256"],
               "scoring_sources": {str(path): sha256_file(path) for path in
                                   [Path(__file__), recovery.ROOT / "fine_tuning/evaluate.py",
                                    *scoring_source_files("team-final-answer-v8")]},
               "limits": "Selected native-source development, not MMMU accuracy, human semantic certification, pure perception causation, or matched target-token exposure",
               "training_allowed": False, "public_upload_allowed": False}
    output.mkdir(parents=True, exist_ok=False)
    atomic_text(output / "scores.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cells))
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparer = commands.add_parser("prepare")
    for name in ("prepared-root", "images-root", "retention-root", "training-records",
                 "training-image-root", "output", "gold-output"):
        preparer.add_argument("--" + name, type=Path, required=True)
    inference = commands.add_parser("infer")
    for name in ("data-root", "model-path", "base-path", "output"):
        inference.add_argument("--" + name, type=Path, required=True)
    inference.add_argument("--candidate", choices=CANDIDATES, required=True)
    inference.add_argument("--slots", type=int, choices=(4,), required=True)
    for name in ("deadline-utc", "commit"):
        inference.add_argument("--" + name, required=True)
    for name in ("training-manifest", "checkpoint-metadata"):
        inference.add_argument("--" + name, type=Path)
    inference.add_argument("--training-manifest-sha256")
    scorer = commands.add_parser("score")
    for name in ("data-root", "gold-root", "output"):
        scorer.add_argument("--" + name, type=Path, required=True)
    scorer.add_argument("--runs", nargs="+", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        result = prepare(**args)
    elif command == "infer":
        result = infer(**args)
    else:
        args["run_roots"] = args.pop("runs")
        result = score(**args)
    print(json.dumps({key: result[key] for key in ("status", "protocol_id", "candidate") if key in result}))


if __name__ == "__main__":
    main()
