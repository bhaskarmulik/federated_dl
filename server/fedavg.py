"""
server/fedavg.py
=================
Federated Averaging (McMahan et al. 2017).

Works on picograd state_dicts (dict[str, np.ndarray]) -- no torch dependency.
Replaces the original torch-based implementation.

fedavg(updates, sample_counts) -> aggregated state_dict
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

StateDict = Dict[str, np.ndarray]


def fedavg(
    updates:       List[StateDict],
    sample_counts: List[int],
) -> StateDict:
    """
    Federated Averaging.

    Parameters
    ----------
    updates       : list of client state_dicts (dict[str, np.ndarray])
    sample_counts : number of training samples per client (for weighting)

    Returns
    -------
    aggregated state_dict -- weighted average of all client updates
    """
    if not updates:
        raise ValueError("fedavg: no updates provided")
    if len(updates) != len(sample_counts):
        raise ValueError("fedavg: len(updates) != len(sample_counts)")

    total = sum(sample_counts)
    weights = [n / total for n in sample_counts]

    # Weighted sum
    result: StateDict = {}
    for key in updates[0].keys():
        agg = np.zeros_like(updates[0][key], dtype=np.float64)
        for sd, w in zip(updates, weights):
            agg += w * sd[key].astype(np.float64)
        result[key] = agg.astype(np.float32)

    return result


def fedavg_delta(
    global_sd:     StateDict,
    updates:       List[StateDict],
    sample_counts: List[int],
    lr:            float = 1.0,
) -> StateDict:
    """
    FedAvg on *deltas* (update = client_weights - global_weights).
    Returns updated global state_dict.
    """
    # Compute deltas
    deltas = []
    for sd in updates:
        delta = {k: sd[k].astype(np.float64) - global_sd[k].astype(np.float64)
                 for k in sd}
        deltas.append(delta)

    # Average delta
    avg_delta = fedavg(deltas, sample_counts)

    # Apply to global
    return {k: (global_sd[k].astype(np.float64) + lr * avg_delta[k]).astype(np.float32)
            for k in global_sd}


__all__ = ["fedavg", "fedavg_delta"]
