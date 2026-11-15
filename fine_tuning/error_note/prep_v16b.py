"""Source-bound F0 preparation and monitor collection for the v16b campaign."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .audit import digest
from .collect import append_jsonl
from .contract import MASTER_SEED, sample_seed
from . import single

VERSION = "v16b-preparation-v1"


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seal(path: Path, body: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = body | {"sha256": digest(body)}
    single._sealed(path, result)
    return result


def read_sealed(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"v16b: sealed artifact changed: {path.name}")
    return value


def preflight_items(items: list[dict]) -> dict:
    """Check every native note contract and optional observation shape without loading a model."""
    from .v16b_rows import build_obs_prompt_v16b, observation_eligibility_v16b, problem_contract_v16b

    records = []
    for item in items:
        contract = problem_contract_v16b(item)
        eligibility = observation_eligibility_v16b(item)
        if eligibility["eligible"]:
            build_obs_prompt_v16b(item)
        records.append({"problem_id": item["problem_id"], "note_contract_sha256": digest(contract),
                        "observation_eligibility": eligibility})
    return {"version": "v16b-data-contract-preflight-v1", "count": len(records), "items": records,
            "observation_eligible": sum(row["observation_eligibility"]["eligible"] for row in records),
            "first_solution_count": len(records), "model_loaded": False}


def select(data_root: Path, output: Path, *, local_sequential: bool = False) -> dict:
    from .data_v16b import _identities, load_items, validate

    validation = validate(data_root)
    splits = {name: load_items(data_root, name) for name in ("train", "prep", "monitor")}
    previous: set[tuple[str, str]] = set()
    for name, items in splits.items():
        identities = set().union(*(_identities(item) for item in items))
        if previous & identities or len({item["problem_id"] for item in items}) != len(items):
            raise ValueError("v16b preparation: overlapping split/family/image identities")
        previous.update(identities)
        if any(item["training_allowed"] is not (name != "monitor") for item in items):
            raise ValueError("v16b preparation: invalid split training permission")
    if len(splits["prep"]) != 512 or len(splits["train"]) != 3000:
        raise ValueError("v16b preparation: the frozen 512/3000 payload is required")
    preflight = seal(output / "data-contract-preflight.json", preflight_items(
        [item for items in splits.values() for item in items]))
    if local_sequential:
        # A native human target guarantees that the short functional run can exercise preparation SFT.
        splits["prep"] = sorted(splits["prep"], key=lambda x: (
            not bool(x.get("references", {}).get("human_solution")), digest(x["problem_id"])))[:2]
        splits["monitor"] = splits["monitor"][:2]
    return seal(output / "selection.json", {"version": VERSION, "local_sequential": local_sequential,
        "data_root": str(data_root.absolute()), "payload_sha256": file_sha(data_root / "payload-receipt.json"),
        "prep_manifest_sha256": file_sha(data_root / "prep-manifest.json"),
        "validation": validation, "data_contract_preflight_sha256": preflight["sha256"], "prep_ids": [x["problem_id"] for x in splits["prep"]],
        "monitor_ids": [x["problem_id"] for x in splits["monitor"]], "human_verified": False})


def p0(item: dict) -> str:
    from .run_v16 import _p0

    return _p0(item)


def images(item: dict) -> list:
    from PIL import Image

    result = []
    for name, expected in zip(item["image_paths"], item["image_sha256"], strict=True):
        path = Path(name)
        if file_sha(path) != expected:
            raise ValueError("v16b: image identity changed")
        with Image.open(path) as image:
            result.append(image.convert("RGB"))
    return result


def grader(data_root: Path, split: str):
    from fine_tuning.notebook_rl.verifier import Verifier
    from .single_grading import grade_single

    verifier = Verifier(data_root / "vault" / split)

    def grade(item: dict, output: dict) -> tuple[str, Any]:
        actor = verifier.actors[item["problem_id"]]
        return grade_single(verifier, actor, actor.problem_id, output["text"], output["finish_reason"],
                            actor.mode, units=True)
    return grade


def make_request(item: dict, prompt: str, stage: str, *, cap: int, sampling: dict,
                 version: int = 0, protocol_sha256: str, policy_sha256: str, engine_step: int = 0) -> dict:
    from fine_tuning.notebook_rl.prompts import user_message
    from .collect import single_request_sha256

    request = {"request_id": digest({"problem_id": item["problem_id"], "stage": stage}),
        "problem_id": item["problem_id"], "block_id": stage, "stage": stage,
        "seed": sample_seed(item["problem_id"], stage, MASTER_SEED), "max_tokens": cap,
        "messages": [user_message(prompt, len(item["image_paths"]))], "images": images(item),
        "kind": stage, "sampling": dict(sampling), "policy_version": version,
        "policy_sha256": policy_sha256, "protocol_sha256": protocol_sha256, "admit_step": engine_step}
    request["request_sha256"] = single_request_sha256(request)
    return request


def collect(bundle: Any, sampler: Any, items: list[dict], *, data_root: Path, split: str, out: Path,
            stage: str, protocol_sha256: str, sampling: dict, cap: int, version: int = 0,
            observation: bool = False) -> list[dict]:
    """One durable request per parent/stage; ambiguous interrupted requests require recovery."""
    from .rl import policy_sha256
    from .v16b_rows import build_obs_prompt_v16b, observation_eligibility_v16b

    policy = policy_sha256(bundle)
    if hasattr(sampler, "protocol_sha256"):
        sampler.protocol_sha256 = protocol_sha256
    grade = grader(data_root, split)
    registry = sampler.live_registry()
    if str(version) not in registry:
        sampler.add_version(bundle, version)
    elif registry[str(version)]["policy_sha256"] != policy:
        raise ValueError("v16b: monitor sampler has a stale policy")
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "requests.jsonl"
    logged = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    records: dict[str, dict] = {}
    pending: dict[str, tuple[dict, str]] = {}
    eligibility = {item["problem_id"]: observation_eligibility_v16b(item) for item in items} if observation else {}
    if observation:
        seal(out / "observation-skips.json", {"version": VERSION,
            "items": [receipt for receipt in eligibility.values() if not receipt["eligible"]]})
    def finish_pending(keep: int = 0) -> None:
        # One sampler.step() per pass, so a freed slot is refilled at once instead of idling for a full drain.
        while len(pending) > keep:
            if not sampler.has_unfinished_requests():
                raise RuntimeError("v16b: collector lost an admitted request")
            for request, output in sampler.step():
                rid = request["request_id"]
                if rid not in pending:
                    raise ValueError("v16b: unknown/duplicate preparation completion")
                item, prompt = pending.pop(rid)
                output = dict(output, prompt=prompt)
                if sampling["top_p"] < 1 or sampling["top_k"]:
                    output["generation"]["probability_space"] = "processed-truncated-b2"
                if not observation:
                    status, extracted = grade(item, output)
                    output.update(status=status, extracted=extracted)
                body = {"problem_id": item["problem_id"], "stage": stage, "prompt": prompt,
                    "request_sha256": request["request_sha256"], "policy_sha256": policy,
                    "protocol_sha256": protocol_sha256, "output": output}
                stored = seal(out / "generations" / (rid + ".json"), body)
                records[item["problem_id"]] = stored
                append_jsonl(ledger, {"request_id": rid, "status": "COMPLETED", "sha256": stored["sha256"]})

    for item in items:
        if observation and not eligibility[item["problem_id"]]["eligible"]:
            continue
        prompt = build_obs_prompt_v16b(item) if observation else p0(item)
        request = make_request(item, prompt, stage, cap=cap, sampling=sampling, version=version,
            protocol_sha256=protocol_sha256, policy_sha256=policy, engine_step=sampler.engine_steps)
        rid = request["request_id"]
        path = out / "generations" / (rid + ".json")
        previous = [row for row in logged if row["request_id"] == rid]
        if previous:
            if not path.is_file() or len([row for row in previous if row["status"] == "STARTED"]) != 1:
                raise ValueError("v16b: STARTED request lacks a durable single result; recovery required")
            stored = read_sealed(path)
            if (stored["request_sha256"] != request["request_sha256"] or stored["policy_sha256"] != policy
                    or stored["protocol_sha256"] != protocol_sha256
                    or any(row["sha256"] != stored["sha256"] for row in previous if row["status"] == "COMPLETED")):
                raise ValueError("v16b: stored preparation request identity changed")
            records[item["problem_id"]] = stored
            continue
        if path.exists():
            raise ValueError("v16b: result has no request ownership ledger")
        append_jsonl(ledger, {"request_id": rid, "status": "STARTED",
                              "request_sha256": request["request_sha256"]})
        sampler.submit(request)
        pending[rid] = (item, prompt)
        # Bound host image memory as well as the engine's active request count: keep one slot free.
        slots = getattr(sampler, "max_num_seqs", 1)
        if len(pending) >= slots:
            finish_pending(keep=slots - 1)
    finish_pending()
    return [records[item["problem_id"]] for item in items if item["problem_id"] in records]




def summary(records: list[dict]) -> dict:
    lengths = [len(row["output"]["generation"]["completion_token_ids"]) for row in records]
    return {"count": len(records), "correct": sum(row["output"].get("status") == "CORRECT" for row in records),
        "missing": sum(row["output"].get("extracted") is None for row in records),
        "length": sum(row["output"]["finish_reason"] == "length" for row in records),
        "median_tokens": statistics.median(lengths) if lengths else None}


def generate(data_root: Path, output: Path, bundle: Any, sampler: Any, *, image: dict,
             local_sequential: bool = False) -> dict:
    from .data_v16b import load_items
    from .protocol import EVAL_SAMPLING_V16B, TRAIN_SAMPLING_V16B

    selection = select(data_root, output, local_sequential=local_sequential)
    context = {"version": VERSION, "selection_sha256": selection["sha256"],
               "image": image, "cap": 256 if local_sequential else 8192, "initial_policy": "fresh_F0"}
    contract = digest(context)
    splits = {name: {item["problem_id"]: item for item in load_items(data_root, name)} for name in ("prep", "monitor")}
    preps = [splits["prep"][pid] for pid in selection["prep_ids"]]
    monitors = [splits["monitor"][pid] for pid in selection["monitor_ids"]]
    kwargs = dict(data_root=data_root, protocol_sha256=contract, cap=context["cap"])
    prep = collect(bundle, sampler, preps, split="prep", out=output / "prep", stage="v16b-prep",
                   sampling=EVAL_SAMPLING_V16B, **kwargs)
    monitor = collect(bundle, sampler, monitors, split="monitor", out=output / "monitor-f0", stage="v16b-monitor",
                      sampling=EVAL_SAMPLING_V16B, **kwargs)
    obs = collect(bundle, sampler, preps, split="prep", out=output / "prep-obs", stage="v16b-prep-obs",
                  sampling=TRAIN_SAMPLING_V16B, observation=True,
                  **(kwargs | {"cap": 256 if local_sequential else 1024}))
    return seal(output / "f0-generations.json", context | {"prep": prep, "monitor": monitor, "obs": obs,
                "monitor_summary": summary(monitor)})


def rows(data_root: Path, output: Path, tokenizer: Any, *, local_sequential: bool = False) -> dict:
    from .data_v16b import load_items
    from .v16b_rows import obs_score, prep_prior_v16b, prep_rows

    selection = select(data_root, output, local_sequential=local_sequential)
    generated = read_sealed(output / "f0-generations.json")
    if generated["selection_sha256"] != selection["sha256"]:
        raise ValueError("v16b: F0 preparation selection changed")
    items = {item["problem_id"]: item for item in load_items(data_root, "prep")}
    records = [dict(items[row["problem_id"]], p0_text=row["prompt"], t1=row["output"])
               for row in generated["prep"]]
    if [row["problem_id"] for row in records] != selection["prep_ids"]:
        raise ValueError("v16b: incomplete F0 preparation")
    selected = prep_rows(records, tokenizer=tokenizer)
    if not selected:
        raise ValueError("v16b: preparation has no admissible SFT rows")
    batches = [selected[index:index + 16] for index in range(0, len(selected), 16)]
    sealed = seal(output / "prep-rows.json", {"version": VERSION, "selection_sha256": selection["sha256"],
        "f0_generations_sha256": generated["sha256"], "rows": selected, "batches": batches,
        "prep_updates": len(batches), "local_sequential": local_sequential})
    prior = prep_prior_v16b(records)
    scores: dict[str, list[float]] = defaultdict(list)
    for row in generated["obs"]:
        item, out = items[row["problem_id"]], row["output"]
        score = obs_score(item, item["layer"], out["text"], finish_reason=out["finish_reason"])
        if score.get("score") is not None:
            scores[item["layer"]].append(score["score"])
    seal(output / "baseline-init.json", {"version": VERSION, "prep_rows_sha256": sealed["sha256"],
        "source_rates": prior["source_rates"], "overall": prior["overall"],
        "layer_rates": {layer: sum(values) / len(values) for layer, values in scores.items()},
        "missing_obs_policy": "first-admission-batch-zero-advantage", "prep_prior": prior})
    return sealed


def eta_report(block_seconds: list[float], *, done: int, total: int) -> dict:
    if (type(done) is not int or type(total) is not int or not 0 <= done <= total or not block_seconds
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
                   for x in block_seconds)):
        raise ValueError("v16b: valid observed block times and fixed progress are required")
    mean = sum(block_seconds) / len(block_seconds)
    return {"version": "v16b-eta-v1", "done": done, "total": total, "mean_block_seconds": mean,
            "remaining_seconds": mean * (total - done), "changes_plan": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "generate", "rows"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--precision", choices=("nf4", "bf16"), default="bf16")
    parser.add_argument("--engine", choices=("hf", "vllm"), default="vllm")
    parser.add_argument("--nf4-cache", type=Path)
    parser.add_argument("--local-sequential", action="store_true")
    parser.add_argument("--allow-model-run", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--kv-cache-gib", type=float, default=16.)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.65)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        result = select(args.data_root, args.output, local_sequential=args.local_sequential)
    elif args.command == "rows":
        from transformers import AutoProcessor
        if args.model_path is None:
            raise ValueError("v16b preparation: --model-path is required")
        processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
        result = rows(args.data_root, args.output, processor.tokenizer, local_sequential=args.local_sequential)
    else:
        if not args.allow_model_run or args.model_path is None:
            raise ValueError("v16b preparation: explicit --allow-model-run and --model-path are required")
        select(args.data_root, args.output, local_sequential=args.local_sequential)
        from .campaign_v16b import load_f0, make_sampler
        from .protocol import SINGLE_IMAGE_V14, SMOKE_IMAGE_V16B, TRAIN_SAMPLING_V16B
        from .run_v16 import _setup_execution
        _setup_execution(MASTER_SEED)
        image = dict(SMOKE_IMAGE_V16B if args.local_sequential else SINGLE_IMAGE_V14)
        bundle, protocol = load_f0(args, image=image)
        sampler = make_sampler(bundle, args, protocol, image=image, sampling=TRAIN_SAMPLING_V16B)
        if not args.local_sequential:  # fail fast: the two-version gate on identical weights, minutes into the pod
            sampler.add_version(bundle, 0)
            sampler.add_version(bundle, 1)
        result = generate(args.data_root, args.output, bundle, sampler, image=image,
                          local_sequential=args.local_sequential)
        sampler.release()
    print(json.dumps({"status": "PREPARED", "sha256": result["sha256"], "command": args.command}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
