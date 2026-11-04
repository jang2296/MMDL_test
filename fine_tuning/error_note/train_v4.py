"""Version-4 training of one round: an RL term on group rollouts plus weighted SFT on A2/B/A1 rows, then save/reload.

Plan: claudedocs/plan_rl_sft_hybrid_v4_20261104.md sections 3.5-3.6 and 4.2(5). One optimizer step per order group of
PER_UPDATE_ITEMS_V2 items. The loss of a step is

    L = coefficient / n_groups * sum_g rl.group_term(g) + sum_k w_k * mean(NLL of the kind-k rows)

with the values fixed in protocol.plain_protocol_v4. A missing kind adds 0 (no renormalization). Round 1 starts from
F0; a later round loads the previous round's checkpoint-final (--init-run) and starts a fresh optimizer.

CLI: python -m fine_tuning.error_note.train_v4 train --rows R --build-root V --order O --round K --start S --limit N
         --arm {hybrid,rl-only} --precision {bf16,nf4} --model-path M --output DIR --code-commit 40HEX
         [--init-run PREVIOUS_RUN] [--token-budget T] [--gradient-checkpointing {on,off}] [--save-every 50]
         [--max-row-tokens K] [--resume]
     python -m fine_tuning.error_note.train_v4 dry-run (the train arguments, at most 10 items, no optimizer or save)
     python -m fine_tuning.error_note.train_v4 reload --run DIR --model-path M --precision {bf16,nf4}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from fine_tuning.notebook_rl import checkpoint as ckpt
from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs
from fine_tuning.notebook_rl.policy import PolicyBundle, assert_base_unchanged, build_policy
from fine_tuning.notebook_rl.prompts import user_message
from fine_tuning.notebook_rl.quantized_policy import assert_quantized_base_unchanged

from . import rl
from .contract import CONTEXT_LIMIT, IMAGE, MASTER_SEED, ROUTE_FULL, ROUTE_LM, update_groups
from .order import read_order
from .protocol import ARM, PER_UPDATE_ITEMS_V2, VERSION_V4, plain_protocol_v4
from .train import (END_TOKEN, ENCODE_BUFFER, ENCODE_WORKERS, PROBE_ATOL, PROBE_ROWS, DropRow, NonFiniteStop,
                    _all_parameter_hashes, _append_jsonl, _backward_rows, _changed_counts, _check_output, _clip_groups,
                    _commit, _encode_or_drop, _factory, _gradient_norms, _keep_before, _newest_checkpoint, _peak_gib,
                    _read_jsonl, _read_probe_rows, _save, _summary, _tensor_counts, _trainable_hashes, _write_json,
                    _write_probe, batch_logprobs, compute_probe, encoded_groups, plan_microbatches)
from .v4 import ARMS, KIND_A2, KIND_B, KIND_RL, KINDS, SFT_KINDS

STATUS_COMPLETE, STATUS_NONFINITE, STATUS_NO_UPDATES = "COMPLETE", "NONFINITE", "NO_UPDATES"


def encode_row_v4(processor: Any, model_config: Any, row: dict, images: list[Any]) -> dict:
    """train.encode_row, except that a truncated or loop-stopped RL rollout gets no end token (append_end false)."""
    tokenizer = processor.tokenizer
    if any(token in text for token in tokenizer.all_special_tokens for text in (row["prompt"], row["target"])):
        raise DropRow("CONTROL_TOKEN")
    prepared, _ = prepare_protocol_inputs(processor, model_config, [user_message(row["prompt"], len(images))], images, IMAGE)
    inputs = prepared["inputs"]
    target = tokenizer.encode(row["target"], add_special_tokens=False)
    if row.get("append_end", True):
        target.append(tokenizer.convert_tokens_to_ids(END_TOKEN))
    if not target:
        raise DropRow("EMPTY_TARGET")
    return _with_row_fields({"row_id": row["row_id"], "kind": row["kind"], "route": row["route"],
                             "input_ids": torch.cat([inputs["input_ids"][0], torch.tensor(target, dtype=torch.long)]),
                             "target_len": len(target), "pixel_values": inputs["pixel_values"],
                             "image_grid_thw": inputs["image_grid_thw"]}, row)


def _with_row_fields(item: dict, row: dict) -> dict:
    return item | {"problem_id": row["problem_id"], **({"advantage": float(row["advantage"])} if row["kind"] == KIND_RL else {})}


def _actor_encoder_v4(processor: Any, config: Any, build_root: Path) -> Callable[[dict], dict]:
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.vault import read_actors

    actors = {actor.problem_id: actor for actor in read_actors(Path(build_root) / "train")}

    def encode(row: dict) -> dict:
        actor = actors[row["problem_id"]]
        if list(actor.image_paths) != row["image_paths"]:
            raise ValueError(f"train v4: row {row['row_id']} images differ from the vault")
        images = load_images(actor, Path(build_root) / "images")
        try:
            return encode_row_v4(processor, config, row, images)
        finally:
            for image in images:
                image.close()
    return encode


def _vault_encoder_v4(bundle: PolicyBundle, build_root: Path) -> Callable[[dict], dict]:
    return _actor_encoder_v4(bundle.processor, bundle.model.get_base_model().config, build_root)


def read_groups_v4(rows_path: Path, order_path: Path, start: int, limit: int, arm: str) -> list[list[dict]]:
    """Order groups of the round's slice; the rl-only arm keeps the RL and A1 rows only (the same RL rows as hybrid)."""
    if arm not in ARMS:
        raise ValueError(f"train v4: arm must be one of {ARMS}")
    items = read_order(order_path)["items"]
    if start < 0 or limit < 1 or start + limit > len(items):
        raise ValueError(f"train v4: start/limit outside the order of {len(items)} items")
    ids = [item["problem_id"] for item in items[start:start + limit]]
    rows = _read_jsonl(rows_path)
    known = set(ids)
    for row in rows:
        if (row.get("kind") not in KINDS or row.get("route") not in (ROUTE_FULL, ROUTE_LM)
                or row.get("problem_id") not in known or not isinstance(row.get("target"), str)):
            raise ValueError(f"train v4: invalid row {row.get('row_id')!r} (kind, route or problem_id)")
        if row["kind"] == KIND_RL and (not isinstance(row.get("advantage"), (int, float)) or not math.isfinite(row["advantage"])
                                       or type(row.get("group_size")) is not int or row["group_size"] < 2):
            raise ValueError(f"train v4: RL row {row['row_id']} needs a finite advantage and group_size >= 2")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("train v4: duplicate row_id")
    if arm == "rl-only":
        rows = [row for row in rows if row["kind"] not in (KIND_A2, KIND_B)]
    by_id: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda row: (KINDS.index(row["kind"]), row["row_id"])):
        by_id.setdefault(row["problem_id"], []).append(row)
    return [[row for pid in group for row in by_id.get(pid, [])] for group in update_groups(ids, PER_UPDATE_ITEMS_V2)]


def schedule_v4(rows_path: Path, order_path: Path, start: int, limit: int) -> tuple[int, list[bool]]:
    """The update schedule of both arms: one scheduler step per order group that has a hybrid row. The rl-only arm
    advances the scheduler without a weight change where only note-derived rows were, so LR and step count match."""
    scheduled = [bool(group) for group in read_groups_v4(rows_path, order_path, start, limit, "hybrid")]
    return sum(scheduled), scheduled


def _reference(bundle: PolicyBundle, rows: list[dict]) -> list[torch.Tensor]:
    try:
        with torch.no_grad(), bundle.model.disable_adapter():
            return [value.detach() for value in batch_logprobs(bundle, rows)]
    finally:
        bundle.route.apply(bundle.route.vision_on)  # PEFT re-enables every adapter on exit (policy.reference_logprobs)


def backward_v4(bundle: PolicyBundle, encoded: list[dict], contract: dict, token_budget: int = 0,
                kinds: Iterable[str] = KINDS) -> dict:
    """Backward of the selected loss terms of one update group; never builds an optimizer or changes weights."""
    settings, weights = contract["rl"], contract["sft_weights"]
    selected = set(kinds)
    stats: dict[str, Any] = {"rl": {"groups": 0, "rows": 0, "action_tokens": 0, "surrogate": 0.0,
                                    "mean_abs_advantage": None}, "sft": {}}
    rl_groups: dict[str, list[dict]] = defaultdict(list)
    for row in encoded:
        if row["kind"] == KIND_RL:
            rl_groups[row["problem_id"]].append(row)
    if KIND_RL in selected and rl_groups:
        scale = settings["coefficient"] / len(rl_groups)
        advantages_seen = []
        for rows in rl_groups.values():
            group_tokens = sum(row["target_len"] for row in rows)
            plan = plan_microbatches([row["input_ids"].numel() for row in rows], [row["route"] for row in rows], token_budget)
            for indices in plan:
                batch = [rows[i] for i in indices]
                bundle.route.apply(batch[0]["route"] == ROUTE_FULL)
                current = batch_logprobs(bundle, batch)
                reference = _reference(bundle, batch) if settings["beta"] else None
                term = rl.group_term(current, reference, [row["advantage"] for row in batch], group_tokens=group_tokens,
                                     eps_low=settings["eps_low"], eps_high=settings["eps_high"], beta=settings["beta"])
                if not bool(torch.isfinite(term)):
                    raise NonFiniteStop("nonfinite RL term")
                (term * scale).backward()
                stats["rl"]["surrogate"] += float(term.detach()) * scale
            advantages_seen += [abs(row["advantage"]) for row in rows]
            stats["rl"]["rows"] += len(rows)
            stats["rl"]["action_tokens"] += group_tokens
        stats["rl"]["groups"] = len(rl_groups)
        stats["rl"]["mean_abs_advantage"] = sum(advantages_seen) / len(advantages_seen)
    for kind in SFT_KINDS:
        rows = [row for row in encoded if row["kind"] == kind]
        if kind not in selected or not rows or not weights[kind]:
            continue
        plan = plan_microbatches([row["input_ids"].numel() for row in rows], [row["route"] for row in rows], token_budget)
        losses = _backward_rows(bundle, rows, plan, [weights[kind] / len(rows)] * len(rows))
        mean = losses[kind] / len(rows)
        if not math.isfinite(mean):
            raise NonFiniteStop(f"nonfinite {kind} loss")
        stats["sft"][kind] = {"rows": len(rows), "row_nll_mean": mean, "weight": weights[kind],
                              "weighted_loss": weights[kind] * mean}
    bundle.route.apply(True)
    return stats


def run_update_v4(bundle: PolicyBundle, optimizer: torch.optim.AdamW, scheduler: Any, encoded: list[dict],
                  contract: dict, token_budget: int = 0) -> dict:
    """One optimizer step over every row of one order group. NonFiniteStop is raised before any parameter changes."""
    import time

    started = time.perf_counter()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    stats = backward_v4(bundle, encoded, contract, token_budget)
    grad_norms = _clip_groups(bundle)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    bundle.route.apply(True)
    kinds = Counter(row["kind"] for row in encoded)
    loss = stats["rl"]["surrogate"] + sum(value["weighted_loss"] for value in stats["sft"].values())
    return {"rows": len(encoded), "rows_by_kind": dict(kinds), "routes": dict(Counter(row["route"] for row in encoded)),
            "tokens": sum(row["input_ids"].numel() for row in encoded),
            "target_tokens": sum(row["target_len"] for row in encoded), "loss": loss, **stats,
            "grad_norms": grad_norms, "seconds": time.perf_counter() - started, "peak_allocated_gib": _peak_gib(),
            "skipped": False}


def _contracts(args: argparse.Namespace) -> tuple[dict, dict | None]:
    init = None
    if args.round == 1:
        if args.init_run is not None:
            raise ValueError("train v4: round 1 starts from F0 (no --init-run)")
    else:
        if args.init_run is None:
            raise ValueError("train v4: a later round needs --init-run (the previous round's run directory)")
        final = Path(args.init_run) / "checkpoint-final"
        init = {"checkpoint": final, "binding": json.loads((Path(args.init_run) / "binding.json").read_text(encoding="utf-8"))}
        ckpt.read_metadata(final, binding=init["binding"])
    return {"rows": ckpt.file_hash(args.rows), "order": ckpt.file_hash(args.order),
            "init": ckpt.file_hash(init["checkpoint"] / "metadata.json") if init else None}, init


def _protocol(args: argparse.Namespace, total_steps: int, contracts: dict) -> dict:
    return plain_protocol_v4(args.precision, max(1, total_steps), arm=args.arm, round_index=args.round,
                             start=args.start, limit=args.limit, contracts=contracts)


def _initial_bundle(args: argparse.Namespace, protocol: dict, init: dict | None, policy_factory: Callable | None,
                    device: str) -> PolicyBundle:
    if init is not None:
        bundle = ckpt.load_eval(args.model_path, init["checkpoint"], binding=init["binding"], policy_factory=policy_factory)
        bundle.model.train()
    else:
        bundle = (policy_factory or build_policy)(args.model_path, ARM, protocol)
    if args.precision == "bf16":
        bundle.model.to(device)
    return bundle


def run_train_v4(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0",
                 make_encoder: Callable[[PolicyBundle, Path], Callable[[dict], dict]] = _vault_encoder_v4,
                 map_fn: Callable[..., Iterable] = map) -> int:
    if args.gradient_checkpointing == "off" and args.precision == "nf4":
        raise SystemExit("train v4: --gradient-checkpointing off is bf16 only")
    out: Path = args.output
    groups = read_groups_v4(args.rows, args.order, args.start, args.limit, args.arm)
    total_steps, scheduled = schedule_v4(args.rows, args.order, args.start, args.limit)
    contracts, init = _contracts(args)
    protocol = _protocol(args, total_steps, contracts)
    contract = protocol["training_contract"]
    binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
               "data_sha256": contracts["rows"], "code_sha256": hashlib.sha256(args.code_commit.encode()).hexdigest()}
    run = {"arm": args.arm, "precision": args.precision, "round": args.round, "start": args.start, "limit": args.limit,
           "max_row_tokens": args.max_row_tokens, "order_sha256": contracts["order"], "init": contracts["init"]}
    _check_output(args, binding, run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").unlink(missing_ok=True)
    assert_base = assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged
    torch.manual_seed(MASTER_SEED)
    newest = _newest_checkpoint(out) if args.resume else None
    step, position = 0, 0
    if newest is None:
        bundle = _initial_bundle(args, protocol, init, policy_factory, device)
        optimizer, scheduler = ckpt.make_optimizer(bundle, protocol)
        _write_json(out / "binding.json", binding)
        _write_json(out / "initial_state.json", {"run": run, "sha256": _trainable_hashes(bundle)})
    else:
        bundle, optimizer, scheduler, step, cursor = ckpt.load_resume(
            args.model_path, newest, protocol, binding=binding, device=device, policy_factory=policy_factory)
        position = cursor["position"]
    if args.gradient_checkpointing == "off":
        bundle.model.gradient_checkpointing_disable()
    updates_path, drops_path = out / "updates.jsonl", out / "drops_local.jsonl"
    for path in (updates_path, drops_path):
        _keep_before(path, position)
        path.touch()
    encode = make_encoder(bundle, args.build_root)
    for index, (encoded, drops) in enumerate(encoded_groups(groups[position:], encode, args.max_row_tokens or CONTEXT_LIMIT,
                                                            map_fn), start=position):
        for drop in drops:
            _append_jsonl(drops_path, {"group": index, **drop})
        if not encoded:
            if scheduled[index]:  # the hybrid arm steps here: keep both arms on one LR schedule (no weight change)
                optimizer.zero_grad(set_to_none=True)
                optimizer.step()  # every gradient is None: AdamW leaves every parameter and moment unchanged
                scheduler.step()
                step += 1
            _append_jsonl(updates_path, {"step": step, "group": index, "rows": 0, "skipped": True,
                                         "scheduler_only": scheduled[index]})
            continue
        lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
        try:
            stats = run_update_v4(bundle, optimizer, scheduler, encoded, contract, args.token_budget)
        except NonFiniteStop as exc:
            _write_json(out / "result.json", {"status": STATUS_NONFINITE, "group": index, "step": step, "detail": str(exc)})
            return 1
        step += 1
        _append_jsonl(updates_path, {"step": step, "group": index, **stats, "lr": lrs})
        if args.save_every and step % args.save_every == 0:
            _save(bundle, optimizer, scheduler, step, index + 1, out / f"checkpoint-{step}", protocol, binding)
    summary = _summary(_read_jsonl(updates_path))
    if not summary["updates"]:
        _write_json(out / "result.json", {"status": STATUS_NO_UPDATES, "arm": args.arm, **summary})
        return 1
    if step != total_steps:
        raise ValueError(f"train v4: {step} scheduler steps differ from the scheduled {total_steps}")
    assert_base(bundle)
    initial = json.loads((out / "initial_state.json").read_text(encoding="utf-8"))["sha256"]
    changed = _changed_counts(bundle, initial)
    _save(bundle, optimizer, scheduler, step, len(groups), out / "checkpoint-final", protocol, binding)
    probe_rows: list[dict] = []
    for row in (row for group in groups for row in group):
        item = _encode_or_drop(row, encode, args.max_row_tokens or CONTEXT_LIMIT)
        probe_rows += [item] if "input_ids" in item else []
        if len(probe_rows) == PROBE_ROWS:
            break
    _write_probe(bundle, probe_rows, out)
    _write_json(out / "result.json", {
        "status": STATUS_COMPLETE, "arm": args.arm, "precision": args.precision, "round": args.round, **summary,
        "changed_tensors": changed, "total_tensors": _tensor_counts(bundle), "protocol_version": VERSION_V4,
        "total_steps": protocol["optimizer"]["total_steps"], "init": contracts["init"],
        "gradient_checkpointing": args.gradient_checkpointing, "token_budget": args.token_budget,
        "checkpoint": str(out / "checkpoint-final")})
    return 0


def run_dry_run_v4(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0",
                   make_encoder: Callable[[PolicyBundle, Path], Callable[[dict], dict]] = _vault_encoder_v4,
                   map_fn: Callable[..., Iterable] = map) -> int:
    """At most 10 predeclared items: each loss kind backpropagated alone, gradients checked and cleared, weights unchanged."""
    if not 1 <= args.limit <= 10:
        raise ValueError("dry-run v4: limit must be 1..10 items")
    out = args.output
    if (out / "dry_run.json").exists():
        raise ValueError("dry-run v4: refusing to overwrite an existing diagnostic")
    out.mkdir(parents=True, exist_ok=True)
    groups = read_groups_v4(args.rows, args.order, args.start, args.limit, args.arm)
    contracts, init = _contracts(args)
    protocol = _protocol(args, schedule_v4(args.rows, args.order, args.start, args.limit)[0], contracts)
    report: dict[str, Any] = {"status": "FAIL", "optimizer_steps": 0, "scheduler_steps": 0, "weights_saved": False,
                              "scope": "diagnostic only: finite per-kind backward with unchanged weights; no learning claim",
                              "protocol_sha256": protocol["sha256"], "components": {}, "weights_unchanged": None,
                              "gradients_cleared": None, "rows_by_kind": dict(Counter(r["kind"] for g in groups for r in g))}
    assert_base = assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged
    bundle = None
    before = None
    try:
        torch.manual_seed(MASTER_SEED)
        bundle = _initial_bundle(args, protocol, init, policy_factory, device)
        assert_base(bundle)
        before = _all_parameter_hashes(bundle)
        encode = make_encoder(bundle, args.build_root)
        encoded = [row for kept, _ in encoded_groups(groups, encode, args.max_row_tokens or CONTEXT_LIMIT, map_fn)
                   for row in kept]
        report["encoded_rows_by_kind"] = dict(Counter(row["kind"] for row in encoded))
        report["paired_a2_b_problem_ids"] = sorted({r["problem_id"] for r in encoded if r["kind"] == KIND_A2}
                                                   & {r["problem_id"] for r in encoded if r["kind"] == KIND_B})
        for kind in KINDS:
            rows = [row for row in encoded if row["kind"] == kind]
            if not rows:
                continue
            bundle.model.zero_grad(set_to_none=True)
            stats = backward_v4(bundle, rows, protocol["training_contract"], args.token_budget, kinds=[kind])
            norms = _gradient_norms(bundle)
            full = any(row["route"] == ROUTE_FULL for row in rows)
            weight_zero = kind != KIND_RL and not protocol["training_contract"]["sft_weights"][kind]
            expected = ({"L": "none", "M": "none", "V": "none"} if weight_zero else
                        {"L": "finite_positive", "M": "finite_positive", "V": "finite_positive" if full else "none"})
            def absent(group: str) -> bool:
                value = norms.get(group)
                return value is None or not value > 0

            errors = [f"{'UNEXPECTED' if need == 'none' else 'MISSING'}_GRADIENT:{group}"
                      for group, need in expected.items() if (need == "none") != absent(group)]
            report["components"][kind] = {"rows": len(rows), "stats": stats, "grad_norms": norms,
                                          "expected": expected, "route_check_passed": not errors,
                                          "route_check_reason": ";".join(errors) or None}
            bundle.model.zero_grad(set_to_none=True)
        report["status"] = ("PASS" if report["components"] and all(c["route_check_passed"] for c in report["components"].values())
                            else "NO_ROWS" if not report["components"] else "GRADIENT_ROUTE_MISMATCH")
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if bundle is not None:
            bundle.model.zero_grad(set_to_none=True)
            report["gradients_cleared"] = all(p.grad is None for p in bundle.model.parameters())
            bundle.route.apply(True)
            report["weights_unchanged"] = before == _all_parameter_hashes(bundle) if before is not None else None
            if report["weights_unchanged"] is False:
                report["status"] = "FAIL"
        _write_json(out / "dry_run.json", report)
    return 0 if report["status"] == "PASS" else 1


def run_reload_v4(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0") -> int:
    """Fresh process: rebuild F0, load checkpoint-final, recompute the probe. Any disagreement is a FAIL."""
    report: dict[str, Any] = {"status": "FAIL", "step": None, "tensors": None, "max_abs_delta": None, "greedy_equal": None}
    try:
        final = args.run / "checkpoint-final"
        binding = json.loads((args.run / "binding.json").read_text(encoding="utf-8"))
        metadata = ckpt.read_metadata(final, binding=binding)
        recorded = metadata["protocol"]
        contract = recorded.get("training_contract", {})
        protocol = plain_protocol_v4(args.precision, recorded.get("optimizer", {}).get("total_steps"),
                                     arm=contract.get("arm"), round_index=contract.get("round"),
                                     start=contract.get("start"), limit=contract.get("limit"),
                                     contracts=contract.get("hashes", {}))
        if (recorded != json.loads(json.dumps(protocol)) or binding["data_sha256"] != contract["hashes"]["rows"]
                or metadata["step"] != protocol["optimizer"]["total_steps"]):
            raise ValueError("reload v4: the checkpoint protocol, rows binding or final step differs")
        bundle, _, _, step, _ = ckpt.load_resume(args.model_path, final, protocol, binding=binding, device=device,
                                                 policy_factory=policy_factory)
        (assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged)(bundle)
        report.update(step=step, tensors=_tensor_counts(bundle))
        probe = json.loads((args.run / "probe.json").read_text(encoding="utf-8"))
        actual = compute_probe(bundle, _read_probe_rows(args.run, probe))
        if [len(v) for v in actual["logprobs"]] != [len(v) for v in probe["logprobs"]]:
            raise ValueError("reload v4: probe token counts differ")
        delta = max((abs(a - b) for new, old in zip(actual["logprobs"], probe["logprobs"], strict=True)
                     for a, b in zip(new, old, strict=True)), default=0.0)
        report.update(max_abs_delta=delta, greedy_equal=actual["greedy_ids"] == probe["greedy_ids"])
        if delta <= PROBE_ATOL and report["greedy_equal"]:
            report["status"] = "PASS"
    except (ValueError, OSError, KeyError, TypeError, NonFiniteStop) as exc:
        report["error"] = str(exc)
    _write_json(args.run / "reload.json", report)
    return 0 if report["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "dry-run"):
        p = sub.add_parser(name)
        for key in ("rows", "build-root", "order", "model-path", "output"):
            p.add_argument(f"--{key}", type=Path, required=True)
        for key in ("round", "start", "limit"):
            p.add_argument(f"--{key}", type=int, required=True)
        p.add_argument("--arm", choices=ARMS, required=True)
        p.add_argument("--precision", choices=("bf16", "nf4"), required=True)
        p.add_argument("--code-commit", type=_commit, required=True)
        p.add_argument("--init-run", type=Path, default=None)
        p.add_argument("--token-budget", type=int, default=0)
        p.add_argument("--max-row-tokens", type=int, default=None, help="local 8GB only; never set on the rented GPU")
        if name == "train":
            p.add_argument("--gradient-checkpointing", choices=("on", "off"), default="on")
            p.add_argument("--save-every", type=int, default=50)
            p.add_argument("--resume", action="store_true")
    reload = sub.add_parser("reload")
    reload.add_argument("--run", type=Path, required=True)
    reload.add_argument("--model-path", type=Path, required=True)
    reload.add_argument("--precision", choices=("bf16", "nf4"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reload":
        return run_reload_v4(args, policy_factory=_factory(args.precision))
    from .collect import ResourceLog, bounded_map, install_training_guards

    install_training_guards(args.build_root)
    args.output.mkdir(parents=True, exist_ok=True)
    map_fn = partial(bounded_map, workers=ENCODE_WORKERS, buffer=ENCODE_BUFFER)
    with ResourceLog(args.output / "resource.jsonl"):
        if args.command == "dry-run":
            return run_dry_run_v4(args, policy_factory=_factory(args.precision), map_fn=map_fn)
        return run_train_v4(args, policy_factory=_factory(args.precision), map_fn=map_fn)


if __name__ == "__main__":
    raise SystemExit(main())
