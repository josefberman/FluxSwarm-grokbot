"""PCGrad: project conflicting gradients (actor parameters only)."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def pcgrad_merge_actor(
    actor: nn.Module,
    losses: Sequence[torch.Tensor],
    *,
    protect_primary: bool = False,
    primary_index: int = 0,
) -> None:
    """Backward each loss, project conflicting grads on actor params, write mean.

    Critic parameters are untouched. Callers should then backward critic/entropy
    losses separately.

    ``protect_primary``: never project the primary objective (progress) away;
    only project the other objectives off it. The primary gradient is added at
    full strength so energy/smoothness cannot cancel upstream learning.
    """
    params = [p for p in actor.parameters() if p.requires_grad]
    if not params:
        return

    grads = []
    for loss in losses:
        actor.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        g = torch.cat([
            (p.grad.detach().flatten() if p.grad is not None else torch.zeros(p.numel(), device=p.device))
            for p in params
        ])
        grads.append(g)

    if protect_primary and len(grads) > 1:
        g_pri = grads[primary_index].clone()
        others = []
        for i, gi in enumerate(grads):
            if i == primary_index:
                continue
            g = gi.clone()
            denom = g_pri.norm() ** 2 + 1e-8
            dot = torch.dot(g, g_pri)
            if dot < 0:
                g = g - (dot / denom) * g_pri
            others.append(g)
        extra = torch.stack(others).mean(dim=0) if others else torch.zeros_like(g_pri)
        merged = g_pri + extra
    else:
        projected = []
        for i, gi in enumerate(grads):
            g = gi.clone()
            for j, gj in enumerate(grads):
                if i == j:
                    continue
                dot = torch.dot(g, gj)
                if dot < 0:
                    g = g - dot / (gj.norm() ** 2 + 1e-8) * gj
            projected.append(g)
        merged = torch.stack(projected).mean(dim=0)

    offset = 0
    for p in params:
        n = p.numel()
        p.grad = merged[offset : offset + n].view_as(p).clone()
        offset += n
