"""One on-policy optimizer update, not a deployment or data-approval entry point.

Inputs are actual sampled actions and bounded verifier results. No answer key,
teacher completion, labels, or cross-entropy target enters this module. The caller
must synchronize the rollout engine after UPDATED before collecting another batch.
"""
from __future__ import annotations

import hashlib
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch

from .answers import extract_final
from .engines import RolloutRequest
from .losses import group_advantages, group_loss, k3, mis_mask
from .policy import PolicyBundle, completion_logprobs, reference_logprobs
from .schema import Group, ProblemTree, Sample, to_json


def has_learning_signal(tree: ProblemTree) -> bool:
    to_json(tree)
    return any(g.usable and g.kind != "DIAG" and len({s.reward for s in g.samples}) > 1
               for g in tree.groups)


def action_mask(sample: Sample, algorithm: str, tokenizer: Any) -> torch.Tensor:
    """Only this completion's tokens; notebook-only spans use actual token decoding."""
    mask = torch.ones((1, len(sample.completion_token_ids)), dtype=torch.bool)
    if algorithm != "ALG-NB-noteonly" or sample.turn == 1 or sample.kind == "DIAG":
        return mask
    mask.zero_()
    kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    decoded = tokenizer.decode(sample.completion_token_ids, **kwargs)
    if decoded != sample.text:
        raise ValueError("update: tokenizer and recorded completion text differ")
    starts = list(re.finditer(r"^Error check:[ \t]*\r?$", decoded, re.M))
    if len(starts) != 1:
        return mask
    start = starts[0].start()
    end = re.search(r"^(?:Observed facts|Solution):", decoded[starts[0].end():], re.M)
    if end is None:
        return mask  # An unclosed note is not a verified note/solution boundary.
    stop = starts[0].end() + end.start()
    previous = 0
    for index in range(len(sample.completion_token_ids)):
        prefix = tokenizer.decode(sample.completion_token_ids[:index+1], **kwargs)
        current = len(prefix)
        if current < previous:
            raise ValueError("update: nonmonotone decoding cannot identify note token spans")
        mask[0, index] = current > start and previous < stop
        previous = current
    return mask


def route_flags(trees: list[ProblemTree], arm: str, seed: int, step: int) -> dict[str, bool]:
    ids = [t.problem_id for t in trees]
    if len(ids) != len(set(ids)):
        raise ValueError("update: duplicate problems")
    if arm == "MOD-FML":
        return dict.fromkeys(ids, False)
    if arm == "MOD-FVML":
        return dict.fromkeys(ids, True)
    if arm == "MOD-RV":
        return {t.problem_id: t.route == "VISION_ON" for t in trees}
    if arm == "MOD-RR":
        count = sum(t.route == "VISION_ON" for t in trees)
        ranked = sorted(ids, key=lambda x: hashlib.sha256(f"route:{seed}:{step}:{x}".encode()).digest())
        selected = set(ranked[:count])
        return {x: x in selected for x in ids}
    raise ValueError("update: MOD-RS needs a separately frozen self-diagnosis selection policy")


def effective_rewards(groups: list[Group], algorithm: str, seed: int, step: int) -> dict[str, list[float]]:
    """Control rewards do not overwrite observed outcomes or raw trajectories."""
    solution = [s for g in groups if g.kind != "DIAG" for s in g.samples]
    if not solution or any(s.result is None for s in solution):
        raise ValueError("update: solution samples require verifier results")
    actual_rate = sum(s.result.status == "CORRECT" for s in solution if s.result) / len(solution)
    result = {}
    for group in groups:
        values = []
        for sample in group.samples:
            if algorithm == "CTL-RAND" and group.kind != "DIAG":
                key = hashlib.sha256(f"reward:{seed}:{step}:{sample.sample_id}".encode()).digest()
                value = float(random.Random(int.from_bytes(key, "big")).random() < actual_rate)
            elif algorithm == "CTL-FORMAT" and group.kind != "DIAG":
                value = float(extract_final(sample.text, sample.finish_reason).status == "OK")
            else:
                value = sample.reward
            values.append(value)
        result[group.group_id] = values
    return result


@dataclass
class _PreparedAction:
    sample: Sample
    request: RolloutRequest
    mask: torch.Tensor
    advantage: float
    weight: float
    norm: int
    vision_on: bool
    old: torch.Tensor | None = None
    reference: torch.Tensor | None = None
    keep: bool = False


def optimizer_step(bundle: PolicyBundle, optimizer: Any, scheduler: Any,
                   trees: list[ProblemTree], images: dict[str, list], preparation: Any,
                   protocol: dict, algorithm: str, *, step: int, seed: int,
                   vision_disabled: bool = False) -> dict:
    """Compute all old/reference LPs first, then microbatch backward and ONE update.

    Returns unsynchronized evidence. It is deliberately not a metrics.step_record:
    only the controller can attest a subsequent weight push and probe comparison.
    A forward/backward failure clears partial gradients and never calls optimizer.step.
    """
    if (algorithm not in protocol["algorithms"] or protocol["loss"]["updates_per_batch"] != 1
            or not trees or type(step) is not int or step < 0 or scheduler.last_epoch != step
            or preparation.policy_version != step or any(t.policy_version != step for t in trees)):
        raise ValueError("update: fresh same-policy batch and one-update contract required")
    if set(images) != {t.problem_id for t in trees}:
        raise ValueError("update: exact image coverage required")
    if any(not has_learning_signal(t) for t in trees):
        raise ValueError("update: dynamic sampling must exclude signal-free solution trees")
    parameters = [p for ps in bundle.params.values() for p in ps]
    expected = [p for g in ("L", "M", "V") for p in bundle.params[g]]
    if ([id(p) for g in optimizer.param_groups for p in g["params"]] != [id(p) for p in expected]
            or any(p.grad is not None for p in bundle.model.parameters())):
        raise ValueError("update: optimizer partitions or cleared gradient boundary differ")
    groups = [g for t in trees for g in t.groups if g.usable]
    flags = route_flags(trees, bundle.inventory["arm"], seed, step)
    if type(vision_disabled) is not bool:
        raise ValueError("update: explicit grounding-health route flag required")
    if vision_disabled:
        flags = dict.fromkeys(flags, False)
    if algorithm == "CTL-BLIND":
        # Runtime must replace before sampling, not merely before the gradient pass.
        if any(t.notebook.get("input_condition") != "BLIND_SAMPLED" for t in trees):
            raise ValueError("update: blind rollout input attestation missing")
        for values in images.values():
            for image in values:
                if image.mode != "RGB" or image.getextrema() != ((128, 128),)*3:
                    raise ValueError("update: CTL-BLIND requires the sampled same-size gray image")
    rewards = effective_rewards(groups, algorithm, seed, step)
    lambdas = protocol["loss"]["group_weights"]
    denominator = sum(lambdas[g.kind] for g in groups)
    if denominator <= 0:
        raise ValueError("update: positive fixed group-weight denominator required")
    actions = []
    missing_actions = 0
    for tree in trees:
        for group in tree.groups:
            if not group.usable:
                continue
            cap = (protocol["rollout"]["diag"]["max_new_tokens"] if group.kind == "DIAG"
                   else protocol["rollout"]["max_new_tokens"][group.samples[0].turn])
            advantages = group_advantages(torch.tensor(rewards[group.group_id], dtype=torch.float32))
            for index, sample in enumerate(group.samples):
                if not 0 < len(sample.completion_token_ids) <= cap:
                    raise ValueError("update: empty or over-cap actual completion")
                mask = action_mask(sample, algorithm, preparation.tokenizer)
                if not mask.any():
                    missing_actions += 1
                    continue
                request = RolloutRequest(sample.sample_id, sample.prompt_token_ids, None,
                    images[tree.problem_id], 1, cap, [sample.seed])
                actions.append(_PreparedAction(sample, request, mask, float(advantages[index]),
                    lambdas[group.kind]/denominator, len(group.samples)*cap, flags[tree.problem_id]))
    base = {"step_before": step, "step_after": step, "engine_sync_required": False,
            "n_problems": len(trees), "n_groups_by_kind": {k: sum(g.kind == k for g in groups)
            for k in ("T1", "T2", "T3", "DIAG")}, "no_action_samples": missing_actions,
            "routes": flags, "vision_health_disabled": vision_disabled, "effective_rewards": rewards}
    if not actions:
        return base | {"status": "NO_ACTIONS"}
    device = next(bundle.model.parameters()).device
    bundle.model.train()
    timings, abs_diff, delta_values = {}, [], {}
    began = time.monotonic()
    with torch.no_grad():
        for action in actions:
            bundle.route.apply(action.vision_on)
            inputs = preparation.model_inputs(action.request, action.sample.completion_token_ids, device)
            old = completion_logprobs(bundle, inputs, len(action.sample.completion_token_ids),
                                     chunk=protocol["model"]["logprob_chunk"])
            sampler = torch.tensor([action.sample.sampler_logprobs], device=device, dtype=old.dtype)
            mask = action.mask.to(device)
            keep, delta = mis_mask(old, sampler, mask, protocol["loss"]["mismatch"]["delta_seq"])
            action.old, action.keep = old.detach().cpu(), bool(keep.item())
            delta_values[action.sample.sample_id] = float(delta.item())
            abs_diff.extend((old-sampler).abs()[mask].cpu().tolist())
            del inputs, old, sampler
    timings["old"] = time.monotonic()-began
    masked = sum(not a.keep for a in actions)/len(actions)
    base.update(masked_fraction=masked, mean_abs_lp_diff=statistics.mean(abs_diff),
                mean_log_ratio_by_sample=delta_values)
    if all(not a.keep for a in actions):
        return base | {"status": "NO_MATCHED_ACTIONS", "seconds": timings}
    began = time.monotonic()
    for action in actions:
        if action.keep:
            bundle.route.apply(action.vision_on)
            inputs = preparation.model_inputs(action.request, action.sample.completion_token_ids, device)
            action.reference = reference_logprobs(bundle, inputs, len(action.sample.completion_token_ids)).cpu()
            del inputs
    timings["ref"] = time.monotonic()-began
    began = time.monotonic()
    total_loss, kl_sum, token_count, max_old_diff = 0.0, 0.0, 0, 0.0
    try:
        for action in actions:
            if not action.keep:
                continue
            assert action.old is not None and action.reference is not None
            bundle.route.apply(action.vision_on)
            inputs = preparation.model_inputs(action.request, action.sample.completion_token_ids, device)
            current = completion_logprobs(bundle, inputs, len(action.sample.completion_token_ids),
                                         chunk=protocol["model"]["logprob_chunk"])
            old, reference, mask = action.old.to(device), action.reference.to(device), action.mask.to(device)
            difference = float((current.detach()-old).abs().max())
            max_old_diff = max(max_old_diff, difference)
            if difference >= 1e-3:
                raise RuntimeError("update: pre-update current/old log probabilities differ")
            loss = action.weight * group_loss(current, old, reference,
                torch.tensor([action.advantage], device=device), mask,
                torch.tensor([True], device=device), eps_low=protocol["loss"]["clip"]["eps_low"],
                eps_high=protocol["loss"]["clip"]["eps_high"], beta=protocol["loss"]["kl"]["beta"],
                norm_const=action.norm)
            kl_sum += float(k3(reference, current.detach())[mask].sum())
            token_count += int(mask.sum())
            total_loss += float(loss.detach())
            loss.backward()
            del loss, current, inputs
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise RuntimeError("HEALTH_STOP:NONFINITE_GRADIENT")
        trainable_ids = {id(p) for p in parameters}
        if any(p.grad is not None for p in bundle.model.parameters() if id(p) not in trainable_ids):
            raise RuntimeError("update: frozen base acquired a gradient")
        grad_norm = {}
        for partition, ps in bundle.params.items():
            active = [p for p in ps if p.grad is not None]
            grad_norm[partition] = float(torch.nn.utils.clip_grad_norm_(active,
                protocol["optimizer"]["grad_clip_per_group"], error_if_nonfinite=True)) if active else None
        optimizer.step()
        scheduler.step()
    finally:
        optimizer.zero_grad(set_to_none=True)
    timings["train"] = time.monotonic()-began
    return base | {"status": "UPDATED_UNSYNCHRONIZED", "step_after": step+1,
        "engine_sync_required": True, "loss": total_loss, "kl_mean": kl_sum/token_count,
        "grad_norm": grad_norm, "max_current_old_difference": max_old_diff, "seconds": timings}
