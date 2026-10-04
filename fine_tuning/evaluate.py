"""Private paired evaluation; immutable cohorts, fresh F0, unchanged public V8.

Benchmark gold is stored separately and is never accepted by generation().
This 4096-token diagnostic protocol does not replace the historical 32768 run.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from PIL import Image

from mmdl.evaluation.backends.input_preparation import prepare_protocol_inputs
from mmdl.evaluation.parsers import score_with_parser
from mmdl.evaluation import prompt as mmmu_prompt
from mmdl.runtime.contracts import DATA_ID, DATA_REVISION, SUBJECTS
from mmdl.runtime.generation import GeneratedOnlyPresencePenalty

from .audit import dump, image_fingerprint, model_input_record, read_rows, sha, write_rows
from .modules import BASE_ID, BASE_REVISION, load_checkpoint
from .train import load_base


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sample_seed(sample_id, master=3407):
    return int.from_bytes(hashlib.sha256(f"{master}:{sample_id}".encode()).digest()[:8], "big") % 2**31


def selected_ids(rows, per_subject=2, seed=3407):
    groups = {subject: [] for subject in SUBJECTS}
    for row in rows:
        if row["subject"] not in groups:
            raise ValueError("Unknown benchmark subject")
        groups[row["subject"]].append(row["id"])
    if not 1 <= per_subject <= 30 or any(len(v) != 30 or len(set(v)) != 30 for v in groups.values()):
        raise ValueError("Selection requires complete 30x30 validation identities")
    return [sample_id for subject in SUBJECTS for sample_id in sorted(groups[subject], key=lambda sid:
        hashlib.sha256(f"{seed}:nvw-eval-v1:{sid}".encode()).hexdigest())[:per_subject]]


def inference_fields(row, subject):
    options = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]
    if not isinstance(options, list) or any(not isinstance(x, str) for x in options):
        raise ValueError("Unchanged ordered options required")
    kind = row["question_type"]
    if kind not in {"multiple-choice", "open"}:
        raise ValueError("Unknown question type")
    value = {"id": row["id"], "subject": subject, "question_type": kind,
             "question": row["question"], "options": options}
    if isinstance(row.get("hint"), str) and row["hint"].strip():
        value["hint"] = row["hint"]
    return value


def prepare_mmmu(data_root, output, per_subject=2):
    """Consume pinned validation only; select before consulting any gold/outcome."""
    import pyarrow.parquet as pq

    root, out = Path(data_root), Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["repo_id"] != DATA_ID or manifest["revision"] != DATA_REVISION:
        raise ValueError("MMMU identity mismatch")
    files = {r["path"]: r for r in manifest["files"]}
    original, candidates = {}, []
    for subject in SUBJECTS:
        name = f"{subject}/validation-00000-of-00001.parquet"
        path = root / name
        if sha(path) != files[name]["sha256"] or path.stat().st_size != files[name]["bytes"]:
            raise ValueError("MMMU file mismatch")
        for row in pq.read_table(path).to_pylist():
            sample = inference_fields(row, subject)
            if sample["id"] in original or not sample["id"].startswith(f"validation_{subject}_"):
                raise ValueError("Repeated/wrong validation identity")
            candidates.append(sample)
            original[sample["id"]] = row
    chosen = selected_ids(candidates, per_subject)
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    inputs, gold, image_audit = [], [], []
    by_id = {x["id"]: x for x in candidates}
    for sid in chosen:
        row, sample = original[sid], by_id[sid]
        text = sample["question"] + "\n" + "\n".join(sample["options"])
        referenced = sorted({int(x) for x in re.findall(r"<image\s+(\d+)>", text)})
        # Original MMMU questions without an explicit marker use their first
        # question image. Extra unreferenced slots are not silently sent.
        slots = referenced or [1]
        paths = []
        for number in slots:
            entry = row.get(f"image_{number}")
            if not isinstance(entry, dict) or entry.get("bytes") is None:
                raise ValueError("Referenced inference image bytes missing")
            raw = entry["bytes"]
            name = f"images/{sid}_{number}.png"
            with Image.open(io.BytesIO(raw)) as image:
                rgb = image.convert("RGB")
                fp = image_fingerprint(rgb)
                rgb.save(out / name)
            paths.append(name)
            image_audit.append({"id": sid, "original_field": f"image_{number}",
                "image_path": name, "reference_position": len(paths)-1,
                "role": "QUESTION_OR_CHOICE_REFERENCE" if referenced else "FIRST_QUESTION_IMAGE_DEFAULT",
                "source_bytes_sha256": hashlib.sha256(raw).hexdigest(), **fp})
        sample.update(image_paths=paths, original_image_fields=[f"image_{n}" for n in slots],
                      benchmark_derived=True, training_allowed=False, public_upload_allowed=False)
        sample["input_record_sha256"] = digest(sample)
        inputs.append(sample)
        gold.append({"id": sid, "answer": row["answer"], "benchmark_derived": True,
                     "training_allowed": False})
    write_rows(out / "inputs.jsonl", inputs)
    write_rows(out / "gold.jsonl", gold)
    write_rows(out / "image_roles.jsonl", image_audit)
    dump(out / "manifest.json", {"status": "COHORT_FROZEN", "dataset_id": DATA_ID,
        "dataset_revision": DATA_REVISION, "split": "validation", "count": len(inputs),
        "selection_policy": "SHA256(3407:nvw-eval-v1:sample_id), no gold or past outcome",
        "inputs_sha256": sha(out / "inputs.jsonl"), "gold_sha256": sha(out / "gold.jsonl"),
        "image_roles_sha256": sha(out / "image_roles.jsonl"), "ids": chosen,
        "source_manifest_sha256": sha(root / "manifest.json"), "training_allowed": False,
        "image_policy": "question/choice references only, default image_1 if no explicit reference",
        "historical_32768_comparable": False})


def visible_structure(value):
    """Drop renderer-internal glyph codes and duplicate line coordinates, not visible semantics."""
    if isinstance(value, dict):
        return {k: visible_structure(v) for k, v in value.items() if k not in {"glyph_code", "segment"}}
    if isinstance(value, list):
        return [visible_structure(v) for v in value]
    return value


def prepare_external(nvw_root, trace_root, output, roots_per_family=4, trace_count=24):
    """Freeze development-only QA/structure/state tests before observing predictions."""
    sources = [(Path(nvw_root), r) for r in read_rows(Path(nvw_root)/"prototype_development.jsonl")]
    canonical = {r["root_id"]: r for r in read_rows(Path(nvw_root)/"canonical_private.jsonl")}
    trace = read_rows(Path(trace_root)/"records.jsonl")
    groups = defaultdict(dict)
    for root, row in sources:
        if row["split"] != "development" or row["benchmark_derived"] is not False:
            raise ValueError("Independent development records required")
        groups[row["root_id"]][row["view"]] = (root, row)
    chosen = []
    for family in ("signal_graph", "spatial_composition", "finite_state"):
        candidates = [values for values in groups.values() if
            values["qa"][1]["family_id"].startswith(family+"-")]
        if len(candidates) < roots_per_family:
            raise ValueError("Insufficient independent development roots")
        for values in sorted(candidates, key=lambda x: digest(x["qa"][1]["root_id"]))[:roots_per_family]:
            if set(values) != {"qa", "structure", "grounded"}:
                raise ValueError("All three independent views required")
            chosen.extend(values[view] for view in ("qa", "structure", "grounded"))
    dev = [r for r in trace if r["split"] == "dev" and r["benchmark_derived"] is False]
    # Domain-round-robin hash order, never a target- or outcome-based selection.
    domains = defaultdict(list)
    for row in sorted(dev, key=lambda r: digest(r["id"])):
        domains[row["domain"]].append(row)
    selected_trace = []
    while len(selected_trace) < trace_count:
        added = False
        for domain in sorted(domains):
            if domains[domain] and len(selected_trace) < trace_count:
                selected_trace.append(domains[domain].pop(0))
                added = True
        if not added:
            raise ValueError("Insufficient family-disjoint TRACE development records")
    chosen.extend((Path(trace_root), row) for row in selected_trace)
    train_families = {r["family_id"] for r in trace if r["split"] == "train"}
    if train_families & {r["family_id"] for r in selected_trace}:
        raise ValueError("TRACE family leakage")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out/"images").mkdir()
    inputs, gold, images = [], [], {}
    for root, original in chosen:
        row = model_input_record(original)
        paths = []
        for i, name in enumerate(row["image_paths"]):
            source = (root/name).resolve()
            if not source.is_relative_to(root.resolve()) or (root/name).is_symlink():
                raise ValueError("External image escapes source")
            name = "images/"+sha(source)+f"_{i}.png"
            if not (out/name).exists():
                shutil.copyfile(source, out/name)
            paths.append(name)
            images[name] = sha(out/name)
        view = row.get("view", "trace_qa")
        target = row["target_text"]
        user = row["user_text"]
        if view == "structure":
            target = json.dumps(visible_structure(json.loads(target)), sort_keys=True)
            user += ("\nReturn one JSON object with image_size, coordinate_system, objects, relations and legend. "
                     "Objects use visible integer id (sequence index from zero when unlabeled), bbox in the original "
                     "960x800 pixel frame, and visible attributes. Relations use type/from/to. Legend uses "
                     "type or operator and observed support. Do not infer renderer glyph indices.")
        elif view == "grounded":
            user += ("\nReturn one JSON object with states (step/state), possible_answers, answer_identifiable, "
                     "answer and option. Include rules only when uniquely identifiable; do not guess hidden rules.")
        item = {"id": row["id"], "subject": row.get("domain", row["family_id"].split("-")[0]),
            "question_type": "open", "question": "", "options": [], "user_text": user,
            "image_paths": paths, "benchmark_derived": False, "training_allowed": False,
            "public_upload_allowed": False}
        inputs.append(item)
        gold.append({"id": row["id"], "target": target, "metric_kind": view,
            "family_id": row["family_id"], "root_id": row["root_id"],
            "options": [str(value) for value in canonical[row["root_id"]]["options"]] if view == "qa" else [],
            "answer_type": row.get("typed_answer", {}).get("type"), "training_allowed": False})
    write_rows(out/"inputs.jsonl", inputs)
    write_rows(out/"gold.jsonl", gold)
    dump(out/"manifest.json", {"status": "COHORT_FROZEN", "count": len(inputs),
        "inputs_sha256": sha(out/"inputs.jsonl"), "gold_sha256": sha(out/"gold.jsonl"),
        "images": images, "selection_policy": "development family/root hash, TRACE domain round robin; no gold/outcomes",
        "training_allowed": False, "benchmark_derived": False,
        "metric_policy": "single terminal JSON; typed exact answer; visible structure leaf F1 and bbox IoU>=0.5; exact supported states",
        "limits": "Registered-family development, not unseen DSL/renderer or sealed holdout; formatting and semantic errors reported separately"})


def terminal_json(raw, *, parse_float=float):
    """No searching intermediate candidates or rescuing an answer with gold."""
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.I)
        text = re.sub(r"\n?```$", "", text).strip()
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Conflicting duplicate JSON keys")
            result[key] = value
        return result
    def finite_constant(_value):
        raise ValueError("Nonfinite JSON constants are not answers")
    decoder = json.JSONDecoder(object_pairs_hook=unique_pairs, parse_constant=finite_constant,
                               parse_float=parse_float)
    value, end = decoder.raw_decode(text)
    if text[end:].strip() or not isinstance(value, dict):
        raise ValueError("Exactly one complete submitted JSON object required")
    return value


def structure_metrics(prediction, target):
    def facts(value, prefix=""):
        if isinstance(value, dict):
            return [fact for k,v in value.items() if k != "bbox"
                    for fact in facts(v, prefix+"/"+k)]
        if isinstance(value, list):
            return [fact for i,v in enumerate(value) for fact in facts(v, prefix+f"/{i}")]
        return [(prefix, json.dumps(value, sort_keys=True))]
    expected, observed = Counter(facts(target)), Counter(facts(visible_structure(prediction)))
    true = sum((expected & observed).values())
    precision = true/sum(observed.values()) if observed else 0.0
    recall = true/sum(expected.values()) if expected else 0.0
    hits, boxes = 0, 0
    by_id = {item.get("id"): item for item in prediction.get("objects", []) if isinstance(item, dict)}
    for item in target.get("objects", []):
        if "bbox" not in item:
            continue
        boxes += 1
        box = by_id.get(item["id"], {}).get("bbox")
        reference = item["bbox"]
        if not isinstance(box, list) or len(box) != 4 or any(type(x) not in {int,float} for x in box):
            continue
        intersection = max(0,min(box[2],reference[2])-max(box[0],reference[0])) * max(0,min(box[3],reference[3])-max(box[1],reference[1]))
        area = max(0,box[2]-box[0])*max(0,box[3]-box[1]) + (reference[2]-reference[0])*(reference[3]-reference[1])-intersection
        hits += area > 0 and intersection/area >= 0.5
    return {"structure_precision": precision, "structure_recall": recall,
            "structure_f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "bbox_hits": hits, "bbox_denominator": boxes}


def score_external(raw, gold, finish_reason):
    kind = gold["metric_kind"]
    if kind in {"circuit_qa", "circuit_structure", "circuit_grounded"}:
        from .circuit_scoring import score_circuit
        return score_circuit(raw, gold["target"], kind.removeprefix("circuit_"),
                             finish_reason, contract_version=gold["circuit_contract_version"])
    if kind == "qa":
        from mmdl.evaluation.final_answer_parser_v8 import extract_mcq
        parsed = extract_mcq(raw, gold["options"], finish_reason)
        answer = parsed["answer"]
        return {"correct": answer == gold["target"], "extracted_answer": answer,
                "parse_status": parsed["status"], "metric_kind": kind}
    if kind == "chartqa_human_exact":
        try:
            submitted = terminal_json(raw, parse_float=Decimal)
            answer = submitted.get("answer")
            if set(submitted) != {"answer"} or type(answer) not in {str, int, Decimal} or not str(answer).strip():
                raise ValueError("One submitted nonempty scalar answer required")
            if type(answer) in {int, Decimal} and not Decimal(str(answer)).is_finite():
                raise ValueError("Finite numeric answer required")
        except (ValueError, TypeError, InvalidOperation):
            return {"correct": False, "parse_status": "NO_PARSE", "metric_kind": kind}
        # Extraction is complete before gold is read. No hidden-candidate rescue.
        answers = json.loads(gold["target"])["answers"]
        def equal(reference):
            if str(answer).strip().casefold() == reference.strip().casefold():
                return True
            try:
                left, right = Decimal(str(answer).strip()), Decimal(reference.strip())
                return left.is_finite() and right.is_finite() and left == right
            except InvalidOperation:
                return False
        return {"correct": any(equal(reference) for reference in answers),
                "extracted_answer": str(answer) if type(answer) is Decimal else answer,
                "parse_status": "PARSED", "metric_kind": kind}
    try:
        prediction, target = terminal_json(raw), json.loads(gold["target"])
    except (ValueError, TypeError):
        return {"correct": False, "parse_status": "NO_PARSE", "metric_kind": kind}
    result = {"parse_status": "PARSED", "metric_kind": kind}
    if kind == "structure":
        if (not isinstance(prediction.get("objects"), list)
                or any(not isinstance(item, dict) or type(item.get("id")) is not int
                       for item in prediction["objects"])):
            return {"correct": False, "parse_status": "SCHEMA_INVALID", "metric_kind": kind}
        result.update(structure_metrics(prediction, target), correct=digest(visible_structure(prediction)) == digest(target))
    elif kind == "graph_structure":
        from .bridge_evaluation import graph_structure
        try:
            observed, expected = graph_structure(prediction), graph_structure(target)
        except (ValueError, TypeError, KeyError):
            return {"correct": False, "parse_status": "SCHEMA_INVALID", "metric_kind": kind}
        def topology(value):
            return dict(value, objects=[{k:v for k,v in item.items() if k != "bbox"} for item in value["objects"]])
        exact = digest(topology(observed)) == digest(topology(expected))
        result.update(structure_metrics(observed, expected), correct=exact, topology_exact=exact)
    elif kind == "grounded":
        result.update(correct=digest(prediction.get("option")) == digest(target["option"]) and
                      digest(prediction.get("answer")) == digest(target["answer"]) and
                      prediction.get("answer_identifiable") is target["answer_identifiable"],
                      supported_states_exact=digest(prediction.get("states")) == digest(target["states"]),
                      rules_exact=digest(prediction.get("rules")) == digest(target.get("rules")))
    elif kind == "trace_qa":
        answer, reference = prediction.get("answer"), target["answer"]
        equal = type(answer) is type(reference) and answer == reference
        if gold["answer_type"] in {"number", "integer"} and type(answer) in {int,float,str}:
            try:
                equal = Decimal(str(answer)).is_finite() and Decimal(str(answer)) == Decimal(str(reference))
            except InvalidOperation:
                equal = False
        result.update(correct=equal, extracted_answer=answer)
    else:
        raise ValueError("Unknown frozen external metric")
    return result


def verify_cohort_images(cohort, manifest):
    root = Path(cohort).resolve()
    names = {name for row in read_rows(root/"inputs.jsonl") for name in row["image_paths"]}
    if "images" in manifest:
        if set(manifest["images"]) != names:
            raise ValueError("Frozen image inventory differs from model inputs")
        for name in names:
            path = root/name
            if not path.resolve().is_relative_to(root) or path.is_symlink() or sha(path) != manifest["images"][name]:
                raise ValueError("Frozen image bytes changed")
    else:
        roles = root/"image_roles.jsonl"
        if sha(roles) != manifest["image_roles_sha256"]:
            raise ValueError("Frozen MMMU image roles changed")
        rows = read_rows(roles)
        if {r["image_path"] for r in rows} != names or len(rows) != len(names):
            raise ValueError("MMMU image role coverage changed")
        for row in rows:
            path = root/row["image_path"]
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                raise ValueError("Image escapes cohort")
            with Image.open(path) as image:
                actual = image_fingerprint(image)
            if actual["pixels_sha256"] != row["pixels_sha256"] or actual["size"] != row["size"]:
                raise ValueError("Frozen MMMU image pixels changed")


def generation(model, processor, record, image_root, protocol):
    """Strictly gold-blind model call; all outputs/costs are observed, not guessed."""
    from transformers import LogitsProcessorList, set_seed

    allowed = {"id", "subject", "question_type", "question", "options", "hint", "user_text",
               "image_paths", "original_image_fields", "input_record_sha256", "benchmark_derived",
               "training_allowed", "public_upload_allowed"}
    if set(record) - allowed:
        raise ValueError("Generation record contains non-inference fields")
    cfg = protocol["evaluation"]
    images, processed = [], []
    try:
        for name in record["image_paths"]:
            path = (Path(image_root) / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(Path(image_root).resolve()):
                raise ValueError("Image path escapes cohort")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        user = record.get("user_text") or mmmu_prompt.build_prompt(
            record, Path(mmmu_prompt.__file__).resolve().parents[3])
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": user})
        messages = [{"role": "user", "content": content}]
        prepared, processed = prepare_protocol_inputs(processor, model.config, messages, images,
            {"preprocessing": "qwen_vl_utils_0_0_14", "min_pixels": cfg["min_pixels"],
             "max_pixels": cfg["max_pixels"]})
        n = prepared["input_tokens"]
        if n + cfg["max_new_tokens"] > cfg["context_tokens"]:
            raise ValueError("Evaluation context overflow; no truncation or budget fallback")
        seed = sample_seed(record["id"], protocol["master_seed"])
        set_seed(seed)
        batch = {k: v.to("cuda") for k, v in prepared["inputs"].items()}
        parameters = {k: cfg[k] for k in ("do_sample", "temperature", "top_p", "top_k",
                                        "repetition_penalty", "max_new_tokens")}
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.generate(**batch, **parameters, use_cache=True,
                logits_processor=LogitsProcessorList([GeneratedOnlyPresencePenalty(n, cfg["presence_penalty"])]))
        output = result[0, n:].tolist()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finish = "stop" if output and output[-1] in eos else "length" if len(output) == cfg["max_new_tokens"] else "other"
        return {"id": record["id"], "raw_response": processor.tokenizer.decode(output, skip_special_tokens=True),
            "output_token_ids": output, "finish_reason": finish, "seed": seed,
            "seconds": time.monotonic()-started, "input_tokens": n, "output_tokens": len(output),
            "image_tokens": int(prepared["inputs"]["image_grid_thw"].prod(-1).sum()) // processor.image_processor.merge_size**2,
            "image_grid_thw": prepared["image_grid_thw"], "processed_image_sizes": prepared["processed_image_sizes"],
            "input_sha256": prepared["input_sha256"], "input_tensors": prepared["input_tensors"],
            "chat_prompt": prepared["chat_prompt"], "prompt_token_ids": prepared["prompt_token_ids"]}
    finally:
        closed = set()
        for image in images + processed:
            if id(image) not in closed:
                image.close()
                closed.add(id(image))


def resume_results(parent, cohort, protocol_path, checkpoint, external):
    """Reuse only a closed budget-stopped prefix; never alter its original ledger."""
    parent, cohort = Path(parent), Path(cohort)
    summary = json.loads((parent/"summary.json").read_text())
    metadata_hash = sha(Path(checkpoint)/"metadata.json") if checkpoint else None
    manifest = json.loads((cohort/"manifest.json").read_text())
    if (summary.get("status") != "BUDGET_STOP"
            or summary.get("model_id") != BASE_ID or summary.get("model_revision") != BASE_REVISION
            or summary.get("cohort_sha256") != sha(cohort/"manifest.json")
            or summary.get("protocol_sha256") != sha(protocol_path)
            or summary.get("checkpoint_metadata_sha256") != metadata_hash
            or summary.get("checkpoint") != (str(checkpoint) if checkpoint else "F0-frozen")
            or summary.get("benchmark_derived") is not (not external)
            or summary.get("predictions_sha256") != sha(parent/"predictions.jsonl")):
        raise ValueError("Resume requires a closed matching budget-stopped evaluation")
    rows = read_rows(parent/"predictions.jsonl")
    expected = [row["id"] for row in read_rows(cohort/"inputs.jsonl")]
    if (len(rows) != manifest["count"] or [row["id"] for row in rows] != expected
            or len(set(expected)) != len(expected)):
        raise ValueError("Resume evaluation identities/order/coverage changed")
    reused, stopped = {}, False
    for row in rows:
        if row["status"] == "BUDGET_STOP":
            stopped = True
        elif row["status"] == "COMPLETED" and not stopped:
            reused[row["id"]] = row
        else:
            raise ValueError("Only a contiguous completed prefix can be reused")
    if not stopped or summary.get("completed") != len(reused):
        raise ValueError("Resume completion count is inconsistent")
    return reused, summary


def completed_histogram(rows, field):
    return dict(Counter(row[field] for row in rows if row["status"] == "COMPLETED"))


def verify_transfer_checkpoint(cohort_manifest, checkpoint):
    """A source-transfer cohort must exclude the evaluated checkpoint's training."""
    if cohort_manifest.get("evaluation_domain") == "BOUNDED_SYNTHETIC_GRAPH_BRIDGE_DEVELOPMENT" and checkpoint is not None:
        metadata = json.loads((Path(checkpoint)/"metadata.json").read_text())
        pairs = {(entry["manifest_sha256"], entry["records_sha256"]): entry["records"] for entry in cohort_manifest["training_evidence"]}
        count = pairs.get((metadata.get("manifest_sha256"), metadata.get("records_sha256")))
        if (metadata.get("epoch_exposure_complete") is not True or metadata.get("base_id") != BASE_ID
                or metadata.get("base_revision") != BASE_REVISION
                or count is None or metadata.get("planned_records") != count or metadata.get("exposure", {}).get("records") != count):
            raise ValueError("Graph checkpoint is not bound to the frozen train-only exclusions")
    if cohort_manifest.get("evaluation_domain") != "PUBLISHED_CHARTQA_HUMAN_SOURCE_TRANSFER" or checkpoint is None:
        return
    expected = {entry["checkpoint_metadata_sha256"] for entry in cohort_manifest["training_evidence"]}
    if sha(Path(checkpoint) / "metadata.json") not in expected:
        raise ValueError("Source-transfer checkpoint is not bound to the frozen training exclusion evidence")


def evaluate(model_path, cohort, output, protocol_path, checkpoint=None, budget_seconds=14400, external=False,
             resume_from=None):
    from transformers import AutoProcessor

    evaluator_source_sha256 = sha(__file__)
    cohort, out = Path(cohort), Path(output)
    cfg = json.loads(Path(protocol_path).read_text())
    cohort_manifest = json.loads((cohort / "manifest.json").read_text())
    if sha(cohort / "inputs.jsonl") != cohort_manifest["inputs_sha256"] or sha(cohort / "gold.jsonl") != cohort_manifest["gold_sha256"]:
        raise ValueError("Frozen cohort changed")
    verify_cohort_images(cohort, cohort_manifest)
    verify_transfer_checkpoint(cohort_manifest, checkpoint)
    reused, parent_summary = resume_results(resume_from, cohort, protocol_path, checkpoint, external) if resume_from else ({}, {})
    if out.exists():
        raise ValueError("Evaluation output already exists; do not overwrite")
    inputs, gold = read_rows(cohort / "inputs.jsonl"), {r["id"]: r for r in read_rows(cohort / "gold.jsonl")}
    if not inputs or len(inputs) != cohort_manifest["count"] or len(gold) != len(inputs):
        raise ValueError("Cohort coverage mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 GPU required")
    out.mkdir(parents=True)
    started = time.monotonic()
    model = load_base(Path(model_path))
    if checkpoint:
        model = load_checkpoint(model, checkpoint)
    model.to("cuda").eval()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=cfg["evaluation"]["min_pixels"], max_pixels=cfg["evaluation"]["max_pixels"])
    torch.cuda.reset_peak_memory_stats()
    completed = []
    with (out / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for record in inputs:
            if record["id"] in reused:
                result = reused[record["id"]]
            elif time.monotonic()-started >= budget_seconds:
                result = {"id": record["id"], "status": "BUDGET_STOP", "correct": None}
            else:
                result = generation(model, processor, record, cohort, cfg)
                reference = gold[record["id"]]
                scored = score_external(result["raw_response"], reference, result["finish_reason"]) if external else score_with_parser(
                    result["raw_response"], record["question_type"], record["options"], reference["answer"],
                    "team-final-answer-v8", result["finish_reason"], question=record["question"])
                result.update(scored, status="COMPLETED", subject=record["subject"],
                              answer=reference.get("target", reference.get("answer")))
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            completed.append(result)
            print(json.dumps({"id": result["id"], "status": result["status"],
                              "completed": len(completed), "total": len(inputs)}), flush=True)
    full = all(r["status"] == "COMPLETED" for r in completed)
    summary = {"status": "EVALUATED" if full else "BUDGET_STOP", "denominator": len(inputs),
        "model_id": BASE_ID, "model_revision": BASE_REVISION,
        "completed": sum(r["status"] == "COMPLETED" for r in completed),
        "correct": sum(r.get("correct") is True for r in completed),
        "accuracy": sum(r.get("correct") is True for r in completed)/len(inputs) if full else None,
        "finish_reasons": completed_histogram(completed, "finish_reason"),
        "budget_stopped": sum(r["status"] == "BUDGET_STOP" for r in completed),
        "seconds": time.monotonic()-started + parent_summary.get("seconds", 0), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "protocol_sha256": sha(protocol_path),
        "cohort_sha256": sha(cohort / "manifest.json"), "checkpoint": str(checkpoint) if checkpoint else "F0-frozen",
        "checkpoint_metadata_sha256": sha(Path(checkpoint)/"metadata.json") if checkpoint else None,
        "parser": "team-final-answer-v8", "historical_62_percent_comparable": False,
        "evaluator_source_sha256": evaluator_source_sha256,
        "predictions_sha256": sha(out / "predictions.jsonl"), "benchmark_derived": not external,
        "training_allowed": False, "public_upload_allowed": False}
    if resume_from:
        summary.update(resume_from=str(resume_from), parent_summary_sha256=sha(Path(resume_from)/"summary.json"),
                       reused_completed=len(reused), fresh_completed=summary["completed"]-len(reused),
                       parent_evaluator_source_sha256=parent_summary["evaluator_source_sha256"],
                       parent_seconds_kind=parent_summary.get("seconds_kind", "EVALUATOR_WALL_TIME"),
                       parent_peak_allocated_bytes=parent_summary.get("peak_allocated_bytes"),
                       peak_scope="NEW_CONTINUATION_PROCESS_ONLY")
    if external:
        summary["metric_kind_counts"] = completed_histogram(completed, "metric_kind")
        summary["mixed_metric_accuracy_is_not_MMMU"] = True
        if cohort_manifest.get("evaluation_domain") == "BOUNDED_SYNTHETIC_GRAPH_BRIDGE_DEVELOPMENT":
            summary.update(parser="graph-development-qa-v8-structure-topology-v1-grounded-v1",
                metric_policy=cohort_manifest["metric_policy"], evaluation_domain=cohort_manifest["evaluation_domain"],
                full_professional_G7_completed=False, professional_transfer_improvement_verified=False)
        if cohort_manifest.get("evaluation_domain") == "PUBLISHED_CHARTQA_HUMAN_SOURCE_TRANSFER":
            summary.update(parser="single-terminal-json-scalar-exact-v1",
                metric_policy=cohort_manifest["metric_policy"],
                evaluation_domain=cohort_manifest["evaluation_domain"],
                source_transfer_evaluation_GPU_completed=full,
                professional_transfer_improvement_verified=False,
                full_professional_G7_completed=False)
    dump(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-mmmu")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--per-subject", type=int, default=2)
    prepare_ext = sub.add_parser("prepare-external")
    for field in ("nvw-root", "trace-root", "output"):
        prepare_ext.add_argument("--"+field, type=Path, required=True)
    prepare_ext.add_argument("--roots-per-family", type=int, default=4)
    prepare_ext.add_argument("--trace-count", type=int, default=24)
    for command in ("run", "run-external"):
        run = sub.add_parser(command)
        for field in ("model-path", "cohort", "output", "protocol"):
            run.add_argument("--"+field, type=Path, required=True)
        run.add_argument("--checkpoint", type=Path)
        run.add_argument("--budget-seconds", type=int, default=14400)
        run.add_argument("--resume-from", type=Path, help="Closed matching budget-stopped result; new output required")
    args = parser.parse_args()
    if args.command == "prepare-mmmu":
        prepare_mmmu(args.data_root, args.output, args.per_subject)
    elif args.command == "prepare-external":
        prepare_external(args.nvw_root, args.trace_root, args.output, args.roots_per_family, args.trace_count)
    else:
        print(json.dumps(evaluate(args.model_path, args.cohort, args.output, args.protocol,
                                  args.checkpoint, args.budget_seconds, args.command == "run-external", args.resume_from)))


if __name__ == "__main__":
    main()
