from __future__ import annotations
from typing import Dict, Iterable, Tuple, List
import itertools
import torch
from flkit.security.prg import prg_like

def two_phase_mask(delta: torch.Tensor, client_id: str, commit_set: List[str], seeds: Dict[Tuple[str,str], bytes]) -> torch.Tensor:
    m_sum = torch.zeros_like(delta)
    for j in commit_set:
        if j == client_id: 
            continue
        key = tuple(sorted((client_id, j)))
        seed = seeds[key]
        m = prg_like(delta, seed)  # produce tensor-shaped mask
        sign = +1 if client_id < j else -1
        m_sum += sign * m
    return delta + m_sum
