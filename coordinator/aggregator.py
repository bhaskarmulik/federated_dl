from __future__ import annotations
from typing import List, Tuple
import torch
from flkit.core.vectorize import weighted_avg

def fedavg(deltas: List[Tuple[torch.Tensor, int]]) -> torch.Tensor:
    return weighted_avg(deltas)
