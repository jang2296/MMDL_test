"""Preregister and score one gold-blind external-dev answer per F0/NOTE/NO-NOTE item.

CLI: preregister --manifest M --vault-root V --train-order O --train-limit N --train-rows R
     --models MODELS.json --dev-rules RULES.json --run-root RUNS --code-commit 40HEX --output P.json
     run --prereg P.json --arm {F0,NOTE,NO-NOTE} --output RUN
     grade --prereg P.json --run RUN  # separate process, after the generator has exited
     analyze --prereg P.json --f0 RUN --note RUN --no-note RUN --output REPORT.json
Registration and analysis are CPU-only. Only run loads an explicitly pinned BF16 model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fine_tuning.modules import BASE_ID, BASE_REVISION
from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT, compare_training
from fine_tuning.notebook_rl.dataset import load_images
from fine_tuning.notebook_rl.pool_freeze import _RECORDS, _lines
from fine_tuning.notebook_rl.proof_eval import mcnemar_exact
from fine_tuning.notebook_rl.prompts import p0_eval_text
from fine_tuning.notebook_rl.local_run import host_identity
from fine_tuning.notebook_rl.pod.common import process_startticks
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.vault import read_actors, read_lineage
from fine_tuning.notebook_rl.verifier import Verifier

from . import collect
from .contract import CONTEXT_LIMIT, IMAGE, MASTER_SEED, MAX_NEW_TOKENS, sample_seed
from .answer_contract import _ACCEPT, _FIELDS, _PROVENANCE
from .collect_v3 import source_identity
from .order import read_order
from .protocol import plain_protocol_v3

VERSION = "error-note-external-dev-v1"
ARMS = ("F0", "NOTE", "NO-NOTE")
DEV_COUNT = 402
ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL_REF = ROOT / "manifests/models/baseline.json"
GATE: dict[str, Any] = {"version": "external-family-paired-v1", "confidence_per_comparison": .975,
        "nominal_familywise_confidence": .95, "bootstrap_resamples": 20000, "bootstrap_seed": 3407,
        "unit": "item_accuracy_cluster_resampling", "expand": "both_lower_bounds_strictly_positive",
        "automatic_expansion": False}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("external eval: duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("external eval: JSON object required")
    return value


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


def _code_hashes() -> dict:
    return {"fine_tuning_production": source_identity()}


def _file_inventory(ref: dict) -> dict[str, dict]:
    files = ref.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("external eval: a model file inventory is required")
    result = {}
    for row in files:
        name = row.get("path")
        if (not isinstance(name, str) or Path(name).name != name or name in ("", ".", "..") or name in result
                or type(row.get("bytes")) is not int or row["bytes"] < 1
                or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64
                or set(row["sha256"]) - set("0123456789abcdef")):
            raise ValueError("external eval: unsafe model file inventory")
        result[name] = row
    return result


def _models(path: Path, rows_sha256: str) -> dict:
    source = read_json(path)
    if set(source) != {"version", "arms"} or source["version"] != "error-note-external-models-v1" or set(source["arms"]) != set(ARMS):
        raise ValueError("external eval: exactly one F0, NOTE and NO-NOTE model must be declared")
    result = {}
    for arm in ARMS:
        item = source["arms"][arm]
        if set(item) != {"model_path", "model_ref"}:
            raise ValueError("external eval: model path and model-ref required")
        model_path, model_ref = Path(item["model_path"]).absolute(), Path(item["model_ref"]).absolute()
        ref = read_json(model_ref)
        _file_inventory(ref)
        if ref.get("dtype") != "bfloat16" or ref.get("processor_revision") != BASE_REVISION:
            raise ValueError("external eval: pinned BF16 processor/model required")
        if arm == "F0":
            if (file_hash(model_ref) != file_hash(BASE_MODEL_REF) or ref.get("kind") != "base"
                    or ref.get("repo_id") != BASE_ID or ref.get("revision") != BASE_REVISION):
                raise ValueError("external eval: F0 must be the original pinned baseline")
        else:
            export = ref.get("export", {})
            protocol = export.get("training_protocol", {})
            contract = protocol.get("training_contract", {})
            expected = plain_protocol_v3("bf16", protocol.get("optimizer", {}).get("total_steps"),
                arm="note" if arm == "NOTE" else "no-note", contracts=contract.get("hashes", {}))
            binding = export.get("binding", {})
            if (ref.get("kind") != "merged" or ref.get("base") != {"id": BASE_ID, "revision": BASE_REVISION}
                    or protocol != expected or export.get("step") != expected["optimizer"]["total_steps"]
                    or binding.get("protocol_sha256") != protocol["sha256"]
                    or binding.get("prompts_sha256") != protocol["prompts_sha256"]
                    or binding.get("data_sha256") != rows_sha256
                    or contract["hashes"]["rows"] != rows_sha256
                    or ref.get("data_manifest_sha256") != rows_sha256
                    or ref.get("train_config_sha256") != digest(protocol)
                    or not export.get("checkpoint_files", {}).get("metadata.json")):
                raise ValueError("external eval: a single final v3 BF16 checkpoint with these rows is required")
        result[arm] = {"model_path": str(model_path), "model_ref": str(model_ref),
                       "model_ref_sha256": file_hash(model_ref), "files_sha256": digest(ref["files"])}
    note, control = (read_json(Path(result[arm]["model_ref"]))["export"]["training_protocol"] for arm in ARMS[1:])
    if (note["training_contract"]["hashes"] != control["training_contract"]["hashes"]
            or note["optimizer"] != control["optimizer"]):
        raise ValueError("external eval: NOTE and NO-NOTE must share rows/admission/order/optimizer")
    return result


def _keys(row: dict, lineage: Any) -> set[str]:
    return {"source:" + digest([row["source"], row["source_revision"], row["native_id"]]),
            "structure:" + lineage.structure_hash,
            "parent:" + digest([row["source"], row["problem_id"]]),
            *("image:" + value for value in row["image_sha256"]),
            *("pixel:" + value for value in row["pixel_sha256"]),
            *("parent:" + digest([row["source"], value]) for value in lineage.parents)}


def _data(args: argparse.Namespace) -> tuple[list[dict], dict]:
    manifest = read_json(args.manifest)
    rows = manifest["items"]
    by_id = {row["problem_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("external eval: duplicate frozen manifest ID")
    train_ids = [row["problem_id"] for row in read_order(args.train_order)["items"]][:args.train_limit]
    if len(train_ids) != args.train_limit or args.train_limit < 1:
        raise ValueError("external eval: invalid fixed training prefix")
    if any(pid not in by_id or by_id[pid]["status"] != "INCLUDED" or by_id[pid]["split"] != "train" for pid in train_ids):
        raise ValueError("external eval: training IDs are not included train records")
    training_rows = [json.loads(line) for line in args.train_rows.read_text(encoding="utf-8").split("\n") if line.strip()]
    if (not training_rows or len({row["row_id"] for row in training_rows}) != len(training_rows)
            or not {row["problem_id"] for row in training_rows} <= set(train_ids)):
        raise ValueError("external eval: training rows lie outside the pinned training prefix")
    dev = sorted((row for row in rows if row["status"] == "HELD_OUT_DEV"), key=lambda row: row["problem_id"])
    if len(dev) != DEV_COUNT or any(row["split"] != "dev" for row in dev):
        raise ValueError(f"external eval: all {DEV_COUNT} frozen dev items are required")
    actors = {actor.problem_id: actor for actor in read_actors(args.vault_root / "dev")}
    lineages = read_lineage(args.vault_root / "dev")
    train_lineage = read_lineage(args.vault_root / "train")
    train_raw = _lines(args.vault_root / "train/lineage.jsonl")
    if any(pid not in train_lineage or hashlib.sha256(train_raw[pid]).hexdigest() != by_id[pid]["record_sha256"]["lineage"]
           for pid in train_ids):
        raise ValueError("external eval: training lineage differs from frozen manifest")
    ids = {row["problem_id"] for row in dev}
    if set(actors) != ids or set(lineages) != ids or ids & set(train_ids):
        raise ValueError("external eval: dev vault differs from the frozen IDs or overlaps training")
    exact = {kind: _lines(args.vault_root / "dev" / name) for kind, name in _RECORDS}
    used_keys = set().union(*(_keys(by_id[pid], train_lineage[pid]) for pid in train_ids))
    groups: dict[str, list[str]] = {}
    registered = []
    for row in dev:
        pid, actor, lineage = row["problem_id"], actors[row["problem_id"]], lineages[row["problem_id"]]
        if (lineage.split != "dev" or actor.source != row["source"] or lineage.source_revision != row["source_revision"]
                or list(actor.image_sha256) != row["image_sha256"] or list(actor.image_paths) != row["image_paths"]
                or any(hashlib.sha256(exact[kind][pid]).hexdigest() != row["record_sha256"][kind] for kind, _ in _RECORDS)):
            raise ValueError("external eval: frozen dev record identity mismatch")
        keys = _keys(row, lineage)
        if keys & used_keys:
            raise ValueError(f"external eval: dev/train source, parent, structure or image overlap: {pid}")
        linked = [key for key, members in groups.items() if keys & set(members)]
        family = min([pid, *linked])
        union = keys | set().union(*(set(groups.pop(key)) for key in linked))
        groups[family] = sorted(union)
        registered.append({"problem_id": pid, "source": row["source"], "taxonomy_family": row["family"],
                           "structure_hash": lineage.structure_hash, "keys": sorted(keys),
                           "actor_sha256": hashlib.sha256(to_json(actor).encode()).hexdigest(),
                           "prompt_sha256": hashlib.sha256(p0_eval_text(actor.question, actor.choices).encode()).hexdigest()})
    for row in registered:
        keys = set(row.pop("keys"))
        row["independent_family"] = next(key for key, members in groups.items() if keys & set(members))
    return registered, actors


def paired_report(items: list[dict], reference: dict[str, bool], note: dict[str, bool]) -> dict:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in items:
        pid = row["problem_id"]
        clusters[row["independent_family"]].append(float(note[pid]) - float(reference[pid]))
    sums = np.asarray([sum(clusters[key]) for key in sorted(clusters)], dtype=np.float64)
    sizes = np.asarray([len(clusters[key]) for key in sorted(clusters)], dtype=np.int64)
    interval = None
    if len(sums) >= 2:
        rng = np.random.default_rng(GATE["bootstrap_seed"])
        samples = []
        for offset in range(0, GATE["bootstrap_resamples"], 1000):
            indices = rng.integers(0, len(sums), (min(1000, GATE["bootstrap_resamples"] - offset), len(sums)))
            samples.append(sums[indices].sum(1) / sizes[indices].sum(1))
        means = np.concatenate(samples)
        tail = (1 - GATE["confidence_per_comparison"]) / 2
        interval = [float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))]
    paired = mcnemar_exact(reference, note)
    gains, losses = paired["f0_wrong_ft_right"], paired["f0_right_ft_wrong"]
    return {"gains": gains, "losses": losses, "net": gains - losses, "item_mean_difference": (gains - losses) / len(items),
            "independent_families": len(sums), "equal_family_mean_difference_secondary": float((sums / sizes).mean()),
            "item_accuracy_ci97_5": interval, "mcnemar_exact_descriptive": paired}


def _dev_verifier(args: argparse.Namespace, actors: dict) -> Verifier:
    """Trusted grading only: private rules and gold are never included in generation requests."""
    verifier = Verifier(args.vault_root / "dev")
    lineages = read_lineage(args.vault_root / "dev")
    manifest = {row["problem_id"]: row for row in read_json(args.manifest)["items"]}
    data = read_json(args.dev_rules)
    open_ids = {pid for pid, item in verifier.items.items() if item.answer_type != "mcq"}
    if (set(data) != {"version", "vault_manifest_sha256", "items"}
            or data["version"] != "external-dev-answer-rules-v1"
            or data["vault_manifest_sha256"] != file_hash(args.vault_root / "dev/manifest.yaml")
            or not isinstance(data["items"], dict) or set(data["items"]) != open_ids):
        raise ValueError("external eval: exact dev rules coverage and manifest binding required")
    for pid, rule in data["items"].items():
        actor, item, lineage, native = actors[pid], verifier.items[pid], lineages[pid], manifest[pid]
        if not isinstance(rule, dict) or set(rule) != _FIELDS:
            raise ValueError("external eval: invalid dev answer rule")
        provenance = rule["provenance"]
        if (rule["actor_sha256"] != hashlib.sha256(to_json(actor).encode()).hexdigest()
                or rule["image_sha256"] != list(actor.image_sha256) or rule["source"] != actor.source
                or rule["source_revision"] != lineage.source_revision or rule["native_id"] != native["native_id"]
                or rule["split"] != "dev" or lineage.split != "dev"
                or not isinstance(provenance, dict) or set(provenance) != _PROVENANCE
                or provenance["label_provenance"] != lineage.label_provenance
                or type(provenance["human_verified"]) is not bool
                or any(not isinstance(provenance[key], str) or not provenance[key].strip()
                       for key in ("field", "location", "sha256", "review_scope"))
                or len(provenance["sha256"]) != 64 or set(provenance["sha256"]) - set("0123456789abcdef")):
            raise ValueError("external eval: dev source/actor/provenance binding mismatch")
        same = rule["answer_type"] == item.answer_type and rule["accept"] == item.accept
        native_range = item.answer_type == "short_text" and rule["answer_type"] == "range" and rule["accept"] == {}
        if (rule["answer_type"] not in _ACCEPT or not (same or native_range)
                or compare_training(item.gold, item, rule) is not True):
            raise ValueError("external eval: unsupported rule or frozen acceptance policy changed")
    verifier.training_rules = data["items"]
    if any(compare_training(item.gold, item, verifier.training_rules.get(pid)) is not True for pid, item in verifier.items.items()):
        raise ValueError("external eval: an original dev gold is ungradable")
    return verifier


def _protocol() -> dict:
    return {"answer_contract": TRAINING_ANSWER_CONTRACT, "prompt": "P0", "master_seed": MASTER_SEED,
            "max_new_tokens": MAX_NEW_TOKENS, "image": dict(IMAGE), "context_limit": CONTEXT_LIMIT,
            "sampling": {"n": 1, "temperature": .7, "top_p": .8, "top_k": 20,
                         "repetition_penalty": 1., "presence_penalty": 1.5},
            "engine": "vllm-0.11.0-bf16", "max_num_seqs": 16, "max_num_batched_tokens": 16384,
            "kv_cache_gib": "auto", "gpu_memory_utilization": .9}


def preregister(args: argparse.Namespace) -> dict:
    if len(args.code_commit) != 40 or set(args.code_commit) - set("0123456789abcdef"):
        raise ValueError("external eval: a full code commit is required")
    if any((args.run_root / arm).exists() for arm in ARMS):
        raise ValueError("external eval: registered run paths must be unused before registration")
    items, actors = _data(args)
    _dev_verifier(args, actors)
    files = {key: {"path": str(getattr(args, key).absolute()), "sha256": file_hash(getattr(args, key))}
             for key in ("manifest", "train_order", "train_rows", "models", "dev_rules")}
    models = _models(args.models, files["train_rows"]["sha256"])
    note = read_json(Path(models["NOTE"]["model_ref"]))["export"]["training_protocol"]
    if note["training_contract"]["hashes"]["order"] != files["train_order"]["sha256"]:
        raise ValueError("external eval: model and declared training order differ")
    body = {"version": VERSION, "code_commit": args.code_commit, "source_hashes": _code_hashes(),
            "files": files, "vault_root": str(args.vault_root.absolute()), "train_limit": args.train_limit,
            "vault_manifests": {split: file_hash(args.vault_root / split / "manifest.yaml") for split in ("train", "dev")},
            "models": models, "items": items, "protocol": _protocol(), "gate": dict(GATE),
            "runs": {arm: str((args.run_root / arm).absolute()) for arm in ARMS},
            "counts": {"items": len(items), "sources": dict(Counter(row["source"] for row in items)),
                       "taxonomy_families": dict(Counter(row["taxonomy_family"] for row in items)),
                       "independent_families": len({row["independent_family"] for row in items})}}
    result = body | {"sha256": digest(body)}
    _write_once(args.output, result)
    return result


def _registration(path: Path) -> dict:
    prereg = read_json(path)
    if (prereg.get("version") != VERSION or prereg.get("sha256") != digest({k: v for k, v in prereg.items() if k != "sha256"})
            or prereg.get("source_hashes") != _code_hashes() or prereg.get("protocol") != _protocol()
            or prereg.get("gate") != GATE):
        raise ValueError("external eval: preregistration/protocol/source identity changed")
    return prereg


def _context(path: Path) -> tuple[dict, argparse.Namespace, dict, Verifier]:
    prereg = _registration(path)
    for value in prereg["files"].values():
        if file_hash(Path(value["path"])) != value["sha256"]:
            raise ValueError("external eval: a registered input file changed")
    args = argparse.Namespace(**{key: Path(value["path"]) for key, value in prereg["files"].items()},
                              vault_root=Path(prereg["vault_root"]), train_limit=prereg["train_limit"])
    if any(file_hash(args.vault_root / split / "manifest.yaml") != sha for split, sha in prereg["vault_manifests"].items()):
        raise ValueError("external eval: vault manifest changed")
    items, actors = _data(args)
    if items != prereg["items"] or _models(args.models, prereg["files"]["train_rows"]["sha256"]) != prereg["models"]:
        raise ValueError("external eval: registered items/model references changed")
    return prereg, args, actors, _dev_verifier(args, actors)


def _generation_context(path: Path) -> tuple[dict, argparse.Namespace, dict]:
    """Public inputs only: no private rules, verifier records, notes or training targets are opened."""
    prereg = _registration(path)
    for name in ("manifest", "train_order", "models"):
        value = prereg["files"][name]
        if file_hash(Path(value["path"])) != value["sha256"]:
            raise ValueError("external eval: registered public inputs changed")
    inputs = argparse.Namespace(vault_root=Path(prereg["vault_root"]),
        dev_rules=Path(prereg["files"]["dev_rules"]["path"]), train_rows=Path(prereg["files"]["train_rows"]["path"]))
    if file_hash(inputs.vault_root / "dev/manifest.yaml") != prereg["vault_manifests"]["dev"]:
        raise ValueError("external eval: dev manifest changed")
    actors = {actor.problem_id: actor for actor in read_actors(inputs.vault_root / "dev")}
    if set(actors) != {row["problem_id"] for row in prereg["items"]}:
        raise ValueError("external eval: actor IDs differ from preregistration")
    for row in prereg["items"]:
        if hashlib.sha256(to_json(actors[row["problem_id"]]).encode()).hexdigest() != row["actor_sha256"]:
            raise ValueError("external eval: public actor changed")
    if _models(Path(prereg["files"]["models"]["path"]), prereg["files"]["train_rows"]["sha256"]) != prereg["models"]:
        raise ValueError("external eval: model references changed")
    return prereg, inputs, actors


def _verify_model(model: dict) -> None:
    if file_hash(Path(model["model_ref"])) != model["model_ref_sha256"]:
        raise ValueError("external eval: model reference changed")
    ref = read_json(Path(model["model_ref"]))
    root = Path(model["model_path"])
    if {path.name for path in root.iterdir()} != set(_file_inventory(ref)):
        raise ValueError("external eval: model directory differs from its complete pinned inventory")
    for name, item in _file_inventory(ref).items():
        path = root / name
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError("external eval: pinned model file mismatch")
    if ref["kind"] == "merged":
        from fine_tuning.notebook_rl.export import validate_export_reference

        validate_export_reference(root, ref)


def _raw_rows(prereg: dict, path: Path) -> dict:
    rows = collect.read_jsonl_done(path, repair=False)
    ids = {row["problem_id"] for row in prereg["items"]}
    if set(rows) != ids:
        raise ValueError("external eval: raw coverage differs from preregistration")
    for pid, row in rows.items():
        if row["seed"] != sample_seed(pid, "t1") or row.get("stage") != "t1" or row.get("engine") != "vllm":
            raise ValueError("external eval: seed/stage/engine changed")
        if row["status"] == "CONTEXT_BLOCKED":
            continue
        if (row["status"] != "NOT_GRADED" or row.get("extracted") is not None
                or row.get("finish_reason") not in ("stop", "length") or not isinstance(row.get("text"), str)
                or type(row.get("completion_tokens")) is not int or not 0 <= row["completion_tokens"] <= MAX_NEW_TOKENS):
            raise ValueError("external eval: invalid single-answer raw output")
    return rows


def _grades(prereg: dict, actors: dict, verifier: Verifier, rows: dict) -> list[dict]:
    grades = []
    for item in prereg["items"]:
        pid = item["problem_id"]
        row = rows[pid]
        if row["status"] == "CONTEXT_BLOCKED":
            status, extracted = "CONTEXT_BLOCKED", None
        else:
            result = verifier.grade(pid, row["text"], row["finish_reason"], actors[pid].mode,
                                    answer_contract=TRAINING_ANSWER_CONTRACT)
            status, extracted = result.status, result.extracted
        grades.append({"problem_id": pid, "status": status, "extracted": extracted, "raw_record_sha256": digest(row)})
    return grades


def run(args: argparse.Namespace, *, stream: Any = None, load: Callable | None = None,
        guarded: bool = True, gpu_probe: Callable | None = None) -> int:
    prereg, inputs, actors = _generation_context(args.prereg)
    out = args.output
    if str(out.absolute()) != prereg["runs"][args.arm]:
        raise ValueError("external eval: arm output must be its one preregistered run path")
    model = prereg["models"][args.arm]
    binding = {"prereg_sha256": prereg["sha256"], "arm": args.arm, "model": model}
    # Atomic claim: even concurrent invocations cannot generate a second answer for this arm.
    out.mkdir(parents=True, exist_ok=False)
    _verify_model(model)
    _write_once(out / "binding.json", binding)
    owner = {"pid": os.getpid(), "host": host_identity(), "startticks": process_startticks(os.getpid())}
    if owner["startticks"] is None:
        raise ValueError("external eval: Linux process identity is required")
    _write_once(out / "owner.json", owner)
    stats = collect.StreamStats()
    if guarded:
        collect.install_training_guards(inputs.vault_root, vllm_schema={}, private_files=[inputs.dev_rules, inputs.train_rows])
    options = argparse.Namespace(model_path=Path(model["model_path"]), **{
        key: prereg["protocol"][key] for key in ("max_num_seqs", "max_num_batched_tokens", "kv_cache_gib", "gpu_memory_utilization")})
    engine = stream if stream is not None else collect._vllm_stream_builder(options, stats)
    loader = load if load is not None else lambda actor: load_images(actor, inputs.vault_root / "images")
    receipt = collect.run_solve(prereg["items"], actors, loader, engine, stats,
        lambda pid, text, finish, mode: ("NOT_GRADED", None),
        out, S=options.max_num_seqs, B=options.max_num_batched_tokens, engine="vllm", gpu_probe=gpu_probe)
    rows = _raw_rows(prereg, out / "t1.jsonl")
    _write_once(out / "generation.json", {"binding": binding, "raw_sha256": file_hash(out / "t1.jsonl"),
        "items": len(rows), "completed": receipt["completed"], "owner": owner})
    return 0 if receipt["completed"] else 1


def grade(args: argparse.Namespace) -> int:
    """Trusted CPU stage, called in a new process after the generation/model process exits."""
    prereg = _registration(args.prereg)
    matches = [arm for arm, path in prereg["runs"].items() if path == str(args.run.absolute())]
    if len(matches) != 1:
        raise ValueError("external eval: only the preregistered arm run may be graded")
    generation = read_json(args.run / "generation.json")
    owner = read_json(args.run / "owner.json")
    if owner != generation["owner"] or owner["host"] != host_identity():
        raise ValueError("external eval: grading requires the recorded generation host")
    if process_startticks(owner["pid"]) == owner["startticks"]:
        raise ValueError("external eval: generation process must exit before trusted grading")
    prereg, _, actors, verifier = _context(args.prereg)
    arm = matches[0]
    binding = {"prereg_sha256": prereg["sha256"], "arm": arm, "model": prereg["models"][arm]}
    if (read_json(args.run / "binding.json") != binding or generation["binding"] != binding
            or generation["raw_sha256"] != file_hash(args.run / "t1.jsonl") or not generation["completed"]):
        raise ValueError("external eval: incomplete or mismatched generation binding")
    if (args.run / "grading.jsonl").exists() or (args.run / "summary.json").exists():
        raise ValueError("external eval: refusing a second grading artifact")
    rows = _raw_rows(prereg, args.run / "t1.jsonl")
    grades = _grades(prereg, actors, verifier, rows)
    collect.write_atomic(args.run / "grading.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in grades))
    statuses = Counter(row["status"] for row in grades)
    valid = not statuses["CONTEXT_BLOCKED"] and not statuses["UNVERIFIABLE"]
    _write_once(args.run / "summary.json", {"binding": binding, "raw_sha256": generation["raw_sha256"],
        "grading_sha256": file_hash(args.run / "grading.jsonl"), "items": len(rows), "correct": statuses["CORRECT"],
        "statuses": dict(statuses), "fully_graded": valid, "accuracy": statuses["CORRECT"] / len(rows) if valid else None})
    return 0 if valid else 1


def analyze(args: argparse.Namespace) -> dict:
    prereg, _, actors, verifier = _context(args.prereg)
    arms, outcomes, fully_graded = {}, {}, True
    for arm, name in zip(ARMS, ("f0", "note", "no_note"), strict=True):
        root = getattr(args, name)
        if str(root.absolute()) != prereg["runs"][arm]:
            raise ValueError("external eval: analysis must use the one preregistered run for every arm")
        summary = read_json(root / "summary.json")
        binding = {"prereg_sha256": prereg["sha256"], "arm": arm, "model": prereg["models"][arm]}
        if (read_json(root / "binding.json") != binding or summary["binding"] != binding
                or summary["raw_sha256"] != file_hash(root / "t1.jsonl")
                or summary["grading_sha256"] != file_hash(root / "grading.jsonl")):
            raise ValueError("external eval: run/model/raw binding mismatch")
        raw = _raw_rows(prereg, root / "t1.jsonl")
        grades = _grades(prereg, actors, verifier, raw)
        rows = collect.read_jsonl_done(root / "grading.jsonl", repair=False)
        if list(rows.values()) != grades:
            raise ValueError("external eval: grading artifact differs from frozen raw regrading")
        statuses = Counter(row["status"] for row in rows.values())
        valid = not statuses["CONTEXT_BLOCKED"] and not statuses["UNVERIFIABLE"]
        if summary["statuses"] != dict(statuses) or summary["correct"] != statuses["CORRECT"] or summary["fully_graded"] != valid:
            raise ValueError("external eval: summary differs from raw regrading")
        fully_graded = fully_graded and valid
        outcomes[arm] = {pid: row["status"] == "CORRECT" for pid, row in rows.items()}
        arms[arm] = {"correct": statuses["CORRECT"], "items": len(rows), "statuses": dict(statuses),
                     "accuracy": statuses["CORRECT"] / len(rows) if valid else None,
                     "raw_sha256": summary["raw_sha256"]}
    paired = {name: paired_report(prereg["items"], outcomes[reference], outcomes["NOTE"])
              for name, reference in (("NOTE-F0", "F0"), ("NOTE-NO-NOTE", "NO-NOTE"))}
    expand = fully_graded and all(value["item_accuracy_ci97_5"] is not None and value["item_accuracy_ci97_5"][0] > 0
                                 for value in paired.values())
    report = {"version": VERSION, "prereg_sha256": prereg["sha256"], "arms": arms, "paired": paired,
              "decision": "EXPAND_ELIGIBLE" if expand else "HOLD", "automatic_expansion": False,
              "gate": prereg["gate"], "fully_graded": fully_graded,
              "scope": "external dev pilot only; no MMMU or general improvement claim"}
    _write_once(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("preregister")
    for name in ("manifest", "vault-root", "train-order", "train-rows", "models", "dev-rules", "run-root", "output"):
        register.add_argument(f"--{name}", type=Path, required=True)
    register.add_argument("--train-limit", type=int, required=True)
    register.add_argument("--code-commit", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--prereg", type=Path, required=True)
    execute.add_argument("--arm", choices=ARMS, required=True)
    execute.add_argument("--output", type=Path, required=True)
    grading = sub.add_parser("grade")
    grading.add_argument("--prereg", type=Path, required=True)
    grading.add_argument("--run", type=Path, required=True)
    analysis = sub.add_parser("analyze")
    for name in ("prereg", "f0", "note", "no-note", "output"):
        analysis.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preregister":
        preregister(args)
        return 0
    if args.command == "run":
        return run(args)
    if args.command == "grade":
        return grade(args)
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
