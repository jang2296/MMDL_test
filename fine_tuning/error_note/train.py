"""Next-token training on the model's own solutions and notes, with save and fresh-process reload.

Plan: claudedocs/research_self_note_finetuning_plan_20261101.md sections 3.4, 5.5. Rows come from the collect stage.
Loss covers target tokens and the end token only: a wrong solution inside a note prompt and all image tokens are input.

CLI: python -m fine_tuning.error_note.train train --rows ROWS.jsonl --build-root V --order ORDER.json --limit N
         --arm {note,solution} --precision {bf16,nf4} --model-path M --output DIR --code-commit 40HEX
         [--token-budget T] [--gradient-checkpointing {on,off}] [--save-every 50] [--max-row-tokens K] [--resume]
         [--protocol-version {v1,v2}]
     python -m fine_tuning.error_note.train reload --run DIR --model-path M --precision {bf16,nf4} [--protocol-version {v1,v2}]
Protocol v2 (claudedocs/plan_self_note_v2_20261102.md section 4): 16 items per update and a linear LR decay whose end is
the number of non-empty update groups of the rows file, so the protocol (and its binding) follows the rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint as recompute

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl import checkpoint as ckpt
from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs
from fine_tuning.notebook_rl.policy import PolicyBundle, assert_base_unchanged, build_policy
from fine_tuning.notebook_rl.prompts import user_message
from fine_tuning.notebook_rl.quantized_policy import assert_quantized_base_unchanged, build_quantized_policy

from .contract import IMAGE, MASTER_SEED, PER_UPDATE_ITEMS, ROUTE_FULL, ROUTE_LM, select_rows, update_groups
from .order import read_order
from .protocol import ARM, GRAD_CLIP, PER_UPDATE_ITEMS_V2, VERSION_V2, plain_protocol, plain_protocol_v2

END_TOKEN = "<|im_end|>"
PROJECTION_CHUNK = 1024
PROBE_ROWS = 2
PROBE_NEW_TOKENS = 16
PROBE_ATOL = 1e-3
PROBE_TENSORS = ("input_ids", "pixel_values", "image_grid_thw")
ENCODE_WORKERS = 8
ENCODE_BUFFER = 32  # rows ready ahead of the GPU; each holds its pixels in RAM
STATUS_COMPLETE, STATUS_NONFINITE, STATUS_NO_UPDATES = "COMPLETE", "NONFINITE", "NO_UPDATES"


class NonFiniteStop(RuntimeError):
    """A loss or gradient is not finite. The run stops; nothing is skipped."""


class DropRow(Exception):
    """A row that must not be trained on; the argument is the recorded reason."""


def plan_microbatches(lengths: list[int], routes: list[str], token_budget: int = 0) -> list[list[int]]:
    """Index lists: same route, longest first, max(length) * count <= budget. An over-budget row stays alone; 0 = one row each."""
    if len(lengths) != len(routes):
        raise ValueError("lengths and routes differ in size")
    batches: list[list[int]] = []
    for route in sorted(set(routes)):
        current: list[int] = []
        for index in sorted((i for i, r in enumerate(routes) if r == route), key=lambda i: (-lengths[i], i)):
            if current and token_budget and lengths[current[0]] * (len(current) + 1) <= token_budget:
                current.append(index)
            else:
                if current:
                    batches.append(current)
                current = [index]
        batches.append(current)
    return batches


def encode_row(processor: Any, model_config: Any, row: dict, images: list[Any]) -> dict:
    """CPU only. input_ids = the evaluation-style prompt + target ids + end token; target_len counts the last two."""
    tokenizer = processor.tokenizer
    if any(token in text for token in tokenizer.all_special_tokens for text in (row["prompt"], row["target"])):
        raise DropRow("CONTROL_TOKEN")  # model text would otherwise become image placeholders or end tokens
    prepared, _ = prepare_protocol_inputs(processor, model_config, [user_message(row["prompt"], len(images))], images, IMAGE)
    inputs = prepared["inputs"]
    target = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.convert_tokens_to_ids(END_TOKEN)]
    return {"row_id": row["row_id"], "kind": row["kind"], "route": row["route"],
            "input_ids": torch.cat([inputs["input_ids"][0], torch.tensor(target, dtype=torch.long)]), "target_len": len(target),
            "pixel_values": inputs["pixel_values"], "image_grid_thw": inputs["image_grid_thw"]}


def _pad_id(bundle: PolicyBundle) -> int:
    tokenizer = getattr(bundle.processor, "tokenizer", None)
    pad = tokenizer.pad_token_id if tokenizer is not None else bundle.model.get_base_model().config.pad_token_id
    if type(pad) is not int:
        raise ValueError("train: an explicit pad token id is required")
    return pad


def _image_inputs(rows: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    with_images = [row for row in rows if row.get("pixel_values") is not None]
    if not with_images:
        return {}
    return {"pixel_values": torch.cat([row["pixel_values"] for row in with_images]).to(device),
            "image_grid_thw": torch.cat([row["image_grid_thw"] for row in with_images]).to(device)}


def _autocast(device: torch.device) -> Any:
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def batch_logprobs(bundle: PolicyBundle, encoded_rows: list[dict], *, chunk: int = PROJECTION_CHUNK) -> list[torch.Tensor]:
    """Per-row 1-D log probabilities of the target tokens from one right-padded forward (policy._logprobs for many rows)."""
    if not encoded_rows:
        raise ValueError("train: an empty micro-batch")
    base = bundle.model.get_base_model()
    device = next(base.parameters()).device
    lengths = [row["input_ids"].numel() for row in encoded_rows]
    if any(not 0 < row["target_len"] < length for row, length in zip(encoded_rows, lengths, strict=True)):
        raise ValueError("train: a nonempty prompt and target are required")
    ids = torch.full((len(encoded_rows), max(lengths)), _pad_id(bundle), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, (row, length) in enumerate(zip(encoded_rows, lengths, strict=True)):
        ids[index, :length] = row["input_ids"]
        mask[index, :length] = 1
    ids, mask = ids.to(device), mask.to(device)

    def project(hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = base.lm_head(hidden).float()
        return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    results = []
    with _autocast(device):
        # The body directly, so no full-vocabulary logits are allocated at prompt positions.
        hidden = base.model(input_ids=ids, attention_mask=mask, **_image_inputs(encoded_rows, device),
                            use_cache=False, return_dict=True).last_hidden_state
        for index, (row, length) in enumerate(zip(encoded_rows, lengths, strict=True)):
            start = length - row["target_len"]
            states, actions = hidden[index, start - 1:length - 1], ids[index, start:length]
            pieces = []
            for offset in range(0, row["target_len"], chunk):
                h, tokens = states[offset:offset + chunk], actions[offset:offset + chunk]
                # Without the recompute, backward would retain all vocabulary logits of every chunk.
                pieces.append(recompute(project, h, tokens, use_reentrant=False) if torch.is_grad_enabled()
                              else project(h, tokens))
            values = torch.cat(pieces)
            if not bool(torch.isfinite(values).all()):
                raise NonFiniteStop(f"nonfinite log probability in row {row['row_id']}")
            results.append(values)
    return results


def _clip_groups(bundle: PolicyBundle) -> dict[str, float | None]:
    """Norm before clipping per group; None for a group with no gradient, which AdamW then skips."""
    norms: dict[str, float | None] = {}
    for group in ("L", "M", "V"):
        active = [p for p in bundle.params[group] if p.grad is not None]
        norm = float(torch.nn.utils.clip_grad_norm_(active, GRAD_CLIP)) if active else None
        if norm is not None and not math.isfinite(norm):
            raise NonFiniteStop(f"nonfinite gradient in group {group}")
        norms[group] = norm
    return norms


def _peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_initialized() else 0.0


def run_update(bundle: PolicyBundle, optimizer: torch.optim.AdamW, scheduler: Any, encoded: list[dict],
               token_budget: int = 0) -> dict:
    """One optimizer step over every row of one order group. Raises NonFiniteStop before any parameter changes."""
    started = time.perf_counter()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    plan = plan_microbatches([row["input_ids"].numel() for row in encoded], [row["route"] for row in encoded], token_budget)
    loss_by_kind: dict[str, float] = defaultdict(float)
    for indices in plan:
        rows = [encoded[index] for index in indices]
        bundle.route.apply(rows[0]["route"] == ROUTE_FULL)
        row_losses = [-values.mean() for values in batch_logprobs(bundle, rows)]
        (torch.stack(row_losses).sum() / len(encoded)).backward()
        for row, value in zip(rows, row_losses, strict=True):
            loss_by_kind[row["kind"]] += float(value.detach())
    grad_norms = _clip_groups(bundle)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    bundle.route.apply(True)
    kinds = Counter(row["kind"] for row in encoded)
    return {"rows": len(encoded), "rows_by_kind": dict(kinds), "routes": dict(Counter(row["route"] for row in encoded)),
            "tokens": sum(row["input_ids"].numel() for row in encoded), "target_tokens": sum(row["target_len"] for row in encoded),
            "loss": sum(loss_by_kind.values()) / len(encoded), "loss_by_kind": {k: loss_by_kind[k] / n for k, n in kinds.items()},
            "grad_norms": grad_norms, "seconds": time.perf_counter() - started, "peak_allocated_gib": _peak_gib(),
            "microbatches": len(plan), "skipped": False}


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    # Line iteration, not str.splitlines: raw U+2028 inside JSON text is valid and must not split a row.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _keep_before(path: Path, position: int) -> None:
    """Drop this run's records past the resume point, including a line cut by a crash."""
    if not path.exists():
        return
    kept = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["group"] < position:
                kept.append(line)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(kept), encoding="utf-8")
    os.replace(tmp, path)


def _read_groups(rows_path: Path, order_path: Path, limit: int, arm: str,
                 per_update: int = PER_UPDATE_ITEMS) -> list[list[dict]]:
    """Groups come from the order, so both arms get the same updates; a group may be empty."""
    ids = [item["problem_id"] for item in read_order(order_path)["items"]]
    if not 1 <= limit <= len(ids):
        raise ValueError(f"train: limit {limit} is outside the order of {len(ids)} items")
    rows = select_rows(_read_jsonl(rows_path), arm)
    if len({row["row_id"] for row in rows}) != len(rows) or any(row["route"] not in (ROUTE_FULL, ROUTE_LM) for row in rows):
        raise ValueError("train: duplicate row_id or unknown route in the rows file")
    by_id: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda row: row["kind"]):
        by_id.setdefault(row["problem_id"], []).append(row)
    return [[row for problem_id in group for row in by_id.get(problem_id, [])]
            for group in update_groups(ids[:limit], per_update)]


def _encode_or_drop(row: dict, encode: Callable[[dict], dict], max_row_tokens: int | None) -> dict:
    try:
        item = encode(row)
    except DropRow as exc:
        return {"row_id": row["row_id"], "reason": str(exc)}
    tokens = item["input_ids"].numel()
    if max_row_tokens and tokens > max_row_tokens:
        return {"row_id": row["row_id"], "reason": "OVER_MAX_ROW_TOKENS", "tokens": tokens}
    return item


def encoded_groups(groups: list[list[dict]], encode: Callable[[dict], dict], max_row_tokens: int | None,
                   map_fn: Callable[..., Iterable]) -> Iterator[tuple[list[dict], list[dict]]]:
    """(kept encoded rows, drop records) per group. map_fn keeps input order and may run ahead of the GPU."""
    fn = partial(_encode_or_drop, encode=encode, max_row_tokens=max_row_tokens)
    results = iter(map_fn(fn, (row for group in groups for row in group)))
    for group in groups:
        items = [next(results) for _ in group]
        yield [item for item in items if "input_ids" in item], [item for item in items if "input_ids" not in item]


def _vault_encoder(bundle: PolicyBundle, build_root: Path) -> Callable[[dict], dict]:
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.vault import read_actors

    actors = {actor.problem_id: actor for actor in read_actors(Path(build_root) / "train")}
    config = bundle.model.get_base_model().config

    def encode(row: dict) -> dict:
        actor = actors[row["problem_id"]]
        if list(actor.image_paths) != row["image_paths"]:
            raise ValueError(f"train: row {row['row_id']} images differ from the vault")
        images = load_images(actor, Path(build_root) / "images")
        try:
            return encode_row(bundle.processor, config, row, images)
        finally:
            for image in images:
                image.close()
    return encode


def _trainable_hashes(bundle: PolicyBundle) -> dict[str, str]:
    parameters = dict(bundle.model.named_parameters())
    return {row["name"]: tensor_sha256(parameters[row["name"]]) for row in bundle.inventory["trainable"]}


def _tensor_counts(bundle: PolicyBundle) -> dict[str, int]:
    counts = Counter(row["group"] for row in bundle.inventory["trainable"])
    return {group: counts[group] for group in ("L", "M", "V")}


def _changed_counts(bundle: PolicyBundle, initial: dict[str, str]) -> dict[str, int]:
    current = _trainable_hashes(bundle)
    changed: Counter[str] = Counter()
    for row in bundle.inventory["trainable"]:
        changed[row["group"]] += current[row["name"]] != initial[row["name"]]
    return {group: changed[group] for group in ("L", "M", "V")}


def _save(bundle: PolicyBundle, optimizer: torch.optim.AdamW, scheduler: Any, step: int, position: int, out: Path,
          protocol: dict, binding: dict) -> None:
    """Whole directory or nothing: checkpoint.save refuses an existing path, and a crash must not leave a half one to resume from."""
    bundle.route.apply(True)
    tmp = out.with_name(out.name + ".tmp")
    for stale in (tmp, out):  # a retried resume recomputes checkpoint-final; numbered ones never repeat
        if stale.exists():
            shutil.rmtree(stale)
    ckpt.save(bundle, optimizer, scheduler, step, {"epoch": 0, "position": position, "discarded_problem_ids": []},
              ckpt.capture_rng(), tmp, protocol=protocol, binding=binding)
    os.replace(tmp, out)


def _newest_checkpoint(out: Path) -> Path | None:
    numbered = [(int(match[1]), path) for path in out.iterdir() if (match := re.fullmatch(r"checkpoint-(\d+)", path.name))]
    return max(numbered)[1] if numbered else None


def _greedy_ids(bundle: PolicyBundle, row: dict) -> list[int]:
    from transformers import GenerationConfig

    base = bundle.model.get_base_model()
    device = next(base.parameters()).device
    prompt = row["input_ids"][:-row["target_len"]].unsqueeze(0).to(device)
    defaults = base.generation_config
    config = GenerationConfig(do_sample=False, max_new_tokens=PROBE_NEW_TOKENS, use_cache=True,
                              eos_token_id=defaults.eos_token_id, pad_token_id=defaults.pad_token_id)
    with _autocast(device):
        output = base.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), **_image_inputs([row], device),
                               generation_config=config, use_model_defaults=False)
    return output[0, prompt.shape[1]:].tolist()


def compute_probe(bundle: PolicyBundle, rows: list[dict]) -> dict:
    """Per-token log probabilities one row at a time in eval mode, plus greedy ids of the first row."""
    bundle.route.apply(True)
    bundle.model.eval()
    with torch.no_grad():
        return {"logprobs": [batch_logprobs(bundle, [row])[0].cpu().tolist() for row in rows],
                "greedy_ids": _greedy_ids(bundle, rows[0])}


def _write_probe(bundle: PolicyBundle, rows: list[dict], out: Path) -> None:
    # The reload command takes no rows or images, so the exact probe inputs stay beside the probe.
    save_file({f"{key}.{index}": row[key].contiguous().clone() for index, row in enumerate(rows)
               for key in PROBE_TENSORS if row.get(key) is not None}, str(out / "probe_inputs.safetensors"))
    _write_json(out / "probe.json", {"row_ids": [row["row_id"] for row in rows],
                                     "target_lens": [row["target_len"] for row in rows], **compute_probe(bundle, rows)})


def _read_probe_rows(run: Path, probe: dict) -> list[dict]:
    tensors = load_file(str(run / "probe_inputs.safetensors"))
    return [{"row_id": row_id, "target_len": target_len, **{key: tensors.get(f"{key}.{index}") for key in PROBE_TENSORS}}
            for index, (row_id, target_len) in enumerate(zip(probe["row_ids"], probe["target_lens"], strict=True))]


def _summary(updates: list[dict]) -> dict:
    real = [row for row in updates if not row["skipped"]]
    seconds, tokens = sum(row["seconds"] for row in real), sum(row["tokens"] for row in real)
    totals: dict[str, Counter] = {"rows_by_kind": Counter(), "route_counts": Counter()}
    for row in real:
        totals["rows_by_kind"].update(row["rows_by_kind"])
        totals["route_counts"].update(row["routes"])
    return {"updates": len(real), "skipped_updates": len(updates) - len(real), "rows_used": sum(row["rows"] for row in real),
            "rows_by_kind": dict(totals["rows_by_kind"]), "route_counts": dict(totals["route_counts"]),
            "mean_loss_first_update": real[0]["loss"] if real else None, "mean_loss_last_update": real[-1]["loss"] if real else None,
            "seconds": seconds, "tokens_per_second": tokens / seconds if seconds else 0.0,
            "peak_allocated_gib": max((row["peak_allocated_gib"] for row in real), default=0.0)}


def _check_output(args: argparse.Namespace, binding: dict, run: dict) -> None:
    out = args.output
    if ((out / "result.json").exists() or (out / "binding.json").exists()) and not args.resume:
        raise SystemExit(f"train: {out} already holds a run; pass --resume or choose a new directory")
    if not args.resume:
        return
    if (out / "result.json").exists() and json.loads((out / "result.json").read_text())["status"] == STATUS_COMPLETE:
        raise SystemExit(f"train: {out} is already complete")
    if (out / "binding.json").exists() and (json.loads((out / "binding.json").read_text()) != binding
                                            or json.loads((out / "initial_state.json").read_text())["run"] != run):
        raise SystemExit(f"train: --resume with different rows, commit, arm, precision, limit or order than {out}")


def run_train(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0",
              make_encoder: Callable[[PolicyBundle, Path], Callable[[dict], dict]] = _vault_encoder,
              map_fn: Callable[..., Iterable] = map) -> int:
    if args.gradient_checkpointing == "off" and args.precision == "nf4":
        raise SystemExit("train: --gradient-checkpointing off is bf16 only")
    out: Path = args.output
    v2 = getattr(args, "protocol_version", "v1") == "v2"
    groups = _read_groups(args.rows, args.order, args.limit, args.arm, PER_UPDATE_ITEMS_V2 if v2 else PER_UPDATE_ITEMS)
    protocol = plain_protocol_v2(args.precision, max(1, sum(1 for group in groups if group))) if v2 \
        else plain_protocol(args.precision)
    binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
               "data_sha256": ckpt.file_hash(args.rows), "code_sha256": hashlib.sha256(args.code_commit.encode()).hexdigest()}
    # token_budget and checkpointing are left out on purpose: they change memory use, not the gradient, so an OOM can resume smaller.
    run = {"arm": args.arm, "precision": args.precision, "limit": args.limit, "max_row_tokens": args.max_row_tokens,
           "order_sha256": ckpt.file_hash(args.order)}
    _check_output(args, binding, run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").unlink(missing_ok=True)
    assert_base = assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged
    torch.manual_seed(MASTER_SEED)  # before building, so both arms start from the same adapter init
    newest = _newest_checkpoint(out) if args.resume else None
    step, position = 0, 0
    if newest is None:
        bundle = (policy_factory or build_policy)(args.model_path, ARM, protocol)
        if args.precision == "bf16":
            bundle.model.to(device)
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
    stream = encoded_groups(groups[position:], encode, args.max_row_tokens, map_fn)
    for index, (encoded, drops) in enumerate(stream, start=position):
        for drop in drops:
            _append_jsonl(drops_path, {"group": index, **drop})
        if not encoded:
            _append_jsonl(updates_path, {"step": step, "group": index, "rows": 0, "skipped": True})
            continue
        lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
        try:
            stats = run_update(bundle, optimizer, scheduler, encoded, args.token_budget)
        except NonFiniteStop as exc:
            _write_json(out / "result.json", {"status": STATUS_NONFINITE, "arm": args.arm, "precision": args.precision,
                                              "group": index, "step": step, "detail": str(exc)})
            return 1
        step += 1
        _append_jsonl(updates_path, {"step": step, "group": index, **stats, **({"lr": lrs} if v2 else {})})
        if args.save_every and step % args.save_every == 0:
            _save(bundle, optimizer, scheduler, step, index + 1, out / f"checkpoint-{step}", protocol, binding)
    summary = _summary(_read_jsonl(updates_path))
    if not step:
        _write_json(out / "result.json", {"status": STATUS_NO_UPDATES, "arm": args.arm, "precision": args.precision, **summary})
        return 1
    assert_base(bundle)
    initial = json.loads((out / "initial_state.json").read_text(encoding="utf-8"))["sha256"]
    changed = _changed_counts(bundle, initial)
    _save(bundle, optimizer, scheduler, step, len(groups), out / "checkpoint-final", protocol, binding)
    probe_rows: list[dict] = []
    for row in (row for group in groups for row in group):
        item = _encode_or_drop(row, encode, args.max_row_tokens)
        probe_rows += [item] if "input_ids" in item else []
        if len(probe_rows) == PROBE_ROWS:
            break
    _write_probe(bundle, probe_rows, out)
    _write_json(out / "result.json", {
        "status": STATUS_COMPLETE, "arm": args.arm, "precision": args.precision, **summary, "changed_tensors": changed,
        "total_tensors": _tensor_counts(bundle), "gradient_checkpointing": args.gradient_checkpointing,
        "token_budget": args.token_budget, "checkpoint": str(out / "checkpoint-final"),
        **({"protocol_version": VERSION_V2, "total_steps": protocol["optimizer"]["total_steps"]} if v2 else {})})
    return 0


def run_reload(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0") -> int:
    """Fresh process: rebuild F0, load checkpoint-final and recompute the probe. Any disagreement is a FAIL with a nonzero exit."""
    report: dict[str, Any] = {"status": "FAIL", "step": None, "tensors": None, "max_abs_delta": None, "greedy_equal": None}
    try:
        final = args.run / "checkpoint-final"
        binding = json.loads((args.run / "binding.json").read_text(encoding="utf-8"))
        recorded = ckpt.read_metadata(final, binding=binding)["protocol"]
        if getattr(args, "protocol_version", "v1") == "v2":  # only the step count comes from the run; the rest is code
            protocol = plain_protocol_v2(args.precision, recorded.get("optimizer", {}).get("total_steps"))
        else:
            protocol = plain_protocol(args.precision)
        if recorded != json.loads(json.dumps(protocol)):
            raise ValueError(f"reload: the checkpoint protocol differs from the {args.precision} protocol")
        bundle, _, _, step, _ = ckpt.load_resume(args.model_path, final, protocol, binding=binding, device=device,
                                                 policy_factory=policy_factory)
        (assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged)(bundle)
        report.update(step=step, tensors=_tensor_counts(bundle))
        probe = json.loads((args.run / "probe.json").read_text(encoding="utf-8"))
        actual = compute_probe(bundle, _read_probe_rows(args.run, probe))
        if [len(values) for values in actual["logprobs"]] != [len(values) for values in probe["logprobs"]]:
            raise ValueError("reload: probe token counts differ")
        delta = max((abs(a - b) for new, old in zip(actual["logprobs"], probe["logprobs"], strict=True)
                     for a, b in zip(new, old, strict=True)), default=0.0)
        report.update(max_abs_delta=delta, greedy_equal=actual["greedy_ids"] == probe["greedy_ids"])
        if delta <= PROBE_ATOL and report["greedy_equal"]:
            report["status"] = "PASS"
    except (ValueError, OSError, KeyError, NonFiniteStop) as exc:
        report["error"] = str(exc)
    _write_json(args.run / "reload.json", report)
    return 0 if report["status"] == "PASS" else 1


def _commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("a full 40-hex commit is required")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    for name in ("rows", "build-root", "order", "model-path", "output"):
        train.add_argument(f"--{name}", type=Path, required=True)
    train.add_argument("--limit", type=int, required=True)
    train.add_argument("--arm", choices=("note", "solution"), required=True)
    train.add_argument("--precision", choices=("bf16", "nf4"), required=True)
    train.add_argument("--code-commit", type=_commit, required=True)
    train.add_argument("--token-budget", type=int, default=0)
    train.add_argument("--gradient-checkpointing", choices=("on", "off"), default="on")
    train.add_argument("--save-every", type=int, default=50)
    train.add_argument("--max-row-tokens", type=int, default=None, help="local 8GB only; never set on the rented GPU")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--protocol-version", choices=("v1", "v2"), default="v1")
    reload = sub.add_parser("reload")
    reload.add_argument("--run", type=Path, required=True)
    reload.add_argument("--model-path", type=Path, required=True)
    reload.add_argument("--precision", choices=("bf16", "nf4"), required=True)
    reload.add_argument("--protocol-version", choices=("v1", "v2"), default="v1")
    return parser


def _factory(precision: str) -> Callable | None:
    return partial(build_quantized_policy, device_index=0) if precision == "nf4" else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reload":
        return run_reload(args, policy_factory=_factory(args.precision))
    from .collect import ResourceLog, bounded_map, install_training_guards

    install_training_guards(args.build_root)
    args.output.mkdir(parents=True, exist_ok=True)
    with ResourceLog(args.output / "resource.jsonl"):
        return run_train(args, policy_factory=_factory(args.precision),
                         map_fn=partial(bounded_map, workers=ENCODE_WORKERS, buffer=ENCODE_BUFFER))


if __name__ == "__main__":
    raise SystemExit(main())
