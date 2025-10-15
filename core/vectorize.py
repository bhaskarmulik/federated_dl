from __future__ import annotations
from typing import Iterable, Tuple
import torch

def flatten_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

def load_flat_params(model: torch.nn.Module, vec: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        num = p.numel()
        p.data.copy_(vec[offset:offset+num].view_as(p))
        offset += num

def model_dim(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def weighted_avg(deltas: Iterable[Tuple[torch.Tensor, int]]) -> torch.Tensor:
    deltas = list(deltas)
    total = sum(n for _, n in deltas) or 1
    out = None
    for d, n in deltas:
        w = n / total
        out = d*w if out is None else out + d*w
    return out
