"""Version-4 RL term on stored rollouts: notebook_rl.losses.group_loss with the current policy as the old policy.

Each rollout is used in exactly one optimizer step (one pass over the round), so old = current.detach(): the ratio is 1
and the gradient is advantage * grad log pi (the clip range never binds). The k3 KL term is against the adapter-disabled
F0 reference (policy.reference_logprobs' context). The normalization is the group's total action tokens, so a long
sample is not weighted more per token than a short one, and the loss decomposes over micro-batches of one group.
"""
from __future__ import annotations

import torch

from fine_tuning.notebook_rl import losses


def pad(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D per-row log probabilities -> [B, T] right-padded with zeros and the binary action mask."""
    if not values or any(v.ndim != 1 or v.numel() == 0 for v in values):
        raise ValueError("rl: nonempty 1-D log probabilities required")
    width = max(v.numel() for v in values)
    mask = torch.zeros((len(values), width), dtype=torch.long, device=values[0].device)
    for index, value in enumerate(values):
        mask[index, :value.numel()] = 1
    return torch.stack([torch.nn.functional.pad(v, (0, width - v.numel())) for v in values]), mask


def group_term(current: list[torch.Tensor], reference: list[torch.Tensor] | None, advantages: list[float],
               *, group_tokens: int, eps_low: float, eps_high: float, beta: float) -> torch.Tensor:
    """The loss of one micro-batch of one rollout group, already divided by the whole group's action tokens."""
    if len(current) != len(advantages) or (reference is not None and len(reference) != len(current)):
        raise ValueError("rl: one advantage (and reference) per row required")
    if type(group_tokens) is not int or group_tokens < sum(v.numel() for v in current):
        raise ValueError("rl: group_tokens must cover every row of the micro-batch")
    if beta and reference is None:
        raise ValueError("rl: a reference is required when beta > 0")
    cur, mask = pad(current)
    ref = pad(reference)[0].to(cur.device) if reference is not None else cur.detach()
    adv = torch.tensor(advantages, dtype=torch.float32, device=cur.device)
    keep = torch.ones(len(current), dtype=torch.bool, device=cur.device)
    return losses.group_loss(cur, cur.detach(), ref, adv, mask, keep, eps_low=eps_low, eps_high=eps_high, beta=beta,
                             norm_const=float(group_tokens))
