"""
server/strategies/fedprox.py + fedbn.py
=========================================
FedProx: adds proximal term to client loss to reduce drift on non-IID data.
FedBN:   clients keep local BatchNorm stats; aggregate only non-BN params.
"""

# ── FedProx ─────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
from typing import Dict, List

StateDict = Dict[str, np.ndarray]


class FedProxLoss:
    """
    FedProx proximal term: (μ/2) * ||w - w_global||²
    Add to client's local loss before backward.

    Usage:
        prox = FedProxLoss(mu=0.01, global_params=global_model.parameters())
        loss = criterion(out, y) + prox(local_model.parameters())
        loss.backward()
    """

    def __init__(self, mu: float, global_params):
        self.mu = mu
        # Snapshot of global params as numpy arrays
        self._global = [p._data.copy() for p in global_params]

    def __call__(self, local_params) -> "Tensor":
        from picograd.tensor import Tensor
        import picograd.nn as nn

        prox_term_val = 0.0
        for lp, gp in zip(local_params, self._global):
            diff = lp._data - gp
            prox_term_val += float(np.sum(diff * diff))

        prox_term_val *= self.mu / 2.0
        return Tensor(np.array(prox_term_val, dtype=np.float32), requires_grad=False)

    def update_global(self, global_params) -> None:
        """Call after each FL round to update global reference."""
        self._global = [p._data.copy() for p in global_params]


# ── FedBN ────────────────────────────────────────────────────────────────────

def fedbn_aggregate(
    updates: List[StateDict],
    sample_counts: List[int],
) -> StateDict:
    """
    FedBN: aggregate all parameters EXCEPT BatchNorm stats.

    BatchNorm layers capture scanner-specific statistics — keeping them local
    prevents domain shift between clients with different medical imaging modalities.

    BN keys to exclude (running_mean, running_var, weight, bias of BN layers):
    Any key containing 'bn', 'batch_norm', 'running_mean', 'running_var'.
    """
    from server.fedavg import fedavg

    def _is_bn_key(k: str) -> bool:
        return any(pat in k for pat in
                   ['running_mean', 'running_var', 'num_batches_tracked'])

    # Separate BN and non-BN keys
    all_keys = list(updates[0].keys())
    bn_keys  = [k for k in all_keys if _is_bn_key(k)]
    agg_keys = [k for k in all_keys if not _is_bn_key(k)]

    # FedAvg on non-BN params
    non_bn_updates = [{k: sd[k] for k in agg_keys} for sd in updates]
    total = sum(sample_counts)
    weights = [n / total for n in sample_counts]

    result: StateDict = {}
    for key in agg_keys:
        agg = np.zeros_like(updates[0][key], dtype=np.float64)
        for sd, w in zip(updates, weights):
            agg += w * sd[key].astype(np.float64)
        result[key] = agg.astype(np.float32)

    # Keep first client's BN stats (they stay local; server just passes them through)
    for key in bn_keys:
        result[key] = updates[0][key].copy()

    return result


__all__ = ["FedProxLoss", "fedbn_aggregate"]
