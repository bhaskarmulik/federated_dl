"""
server/secure_agg.py
=====================
Masking-based Secure Aggregation for federated learning.

CORRECT protocol (Bonawitz et al. 2017 simplified):
  - Client i has sample_count n_i, total N = Σ n_j
  - Client sends: masked_i = (n_i / N) * update_i  +  r_i
    where Σ r_i = 0  (pairwise-cancelling masks)
  - Server computes: Σ masked_i = Σ (n_i/N)*update_i + Σ r_i = weighted_avg_update

This ensures masks cancel in the unweighted sum so the server
recovers the FedAvg result without seeing individual updates.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

StateDict = Dict[str, np.ndarray]


class SecureAggregator:
    """Server-side secure aggregation orchestrator."""

    def __init__(self, n_clients: int):
        self.n_clients = n_clients
        self._updates: Dict[str, Tuple[StateDict, int]] = {}

    def receive(self, client_id: str, masked_update: StateDict, n_samples: int) -> None:
        self._updates[client_id] = (masked_update, n_samples)

    @property
    def ready(self) -> bool:
        return len(self._updates) >= self.n_clients

    def aggregate(self) -> StateDict:
        """
        Sum pre-weighted masked updates.
        Each client already applied weight (n_i / N) before masking,
        so simple sum gives the FedAvg result (masks cancel).
        """
        if not self._updates:
            raise RuntimeError("No updates received")

        result: StateDict = {}
        for cid, (sd, n) in self._updates.items():
            for k, arr in sd.items():
                if k not in result:
                    result[k] = np.zeros_like(arr, dtype=np.float64)
                result[k] += arr.astype(np.float64)   # sum (weights already applied)

        return {k: v.astype(np.float32) for k, v in result.items()}

    def reset(self) -> None:
        self._updates.clear()


class MaskGenerator:
    """
    Client-side mask generation.

    Masks are constructed via pairwise-cancelling PRG seeds:
      mask_i = Σ_{j>i} PRG(seed_ij) - Σ_{j<i} PRG(seed_ji)
    → Σ_i mask_i = 0  (telescope cancellation)
    """

    def __init__(self, client_id: int, n_clients: int, round_id: int = 0):
        self.client_id = client_id
        self.n_clients = n_clients
        self.round_id  = round_id

    def generate_masks(self, model_shapes: Dict[str, tuple]) -> Dict[str, np.ndarray]:
        i = self.client_id
        N = self.n_clients
        r = self.round_id
        masks: Dict[str, np.ndarray] = {}

        for key, shape in model_shapes.items():
            mask = np.zeros(shape, dtype=np.float32)
            for j in range(N):
                if j == i:
                    continue
                seed = int(abs(hash((min(i, j), max(i, j), r, key))) % (2 ** 31))
                rng  = np.random.default_rng(seed)
                pairwise = rng.standard_normal(shape).astype(np.float32)
                if j > i:
                    mask += pairwise
                else:
                    mask -= pairwise
            masks[key] = mask

        return masks

    def mask_update(
        self,
        update:        StateDict,
        model_shapes:  Dict[str, tuple],
        weight:        float = 1.0,          # n_i / N — apply before masking
    ) -> StateDict:
        """
        Returns: weight * update + mask
        Server sums these across clients → FedAvg result (masks cancel).
        """
        masks = self.generate_masks(model_shapes)
        return {k: (weight * update[k]).astype(np.float32) + masks[k]
                for k in update}


def verify_mask_cancellation(
    n_clients: int,
    shapes:    Dict[str, tuple],
    round_id:  int = 0,
) -> bool:
    """Confirm Σ_i mask_i = 0 (unweighted)."""
    total: Dict[str, np.ndarray] = {k: np.zeros(s, dtype=np.float64)
                                     for k, s in shapes.items()}
    for i in range(n_clients):
        gen   = MaskGenerator(i, n_clients, round_id)
        masks = gen.generate_masks(shapes)
        for k in total:
            total[k] += masks[k].astype(np.float64)

    max_abs = max(np.max(np.abs(v)) for v in total.values())
    return max_abs < 1e-3


__all__ = ["SecureAggregator", "MaskGenerator", "verify_mask_cancellation"]
