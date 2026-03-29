"""
falcon/data_partition.py
=========================
Non-IID data distribution for federated learning simulation.

Strategies:
  - Dirichlet(alpha): label distribution proportional to Dir(alpha) per client.
    alpha -> infinity : IID
    alpha = 0.5: moderate non-IID
    alpha = 0.1: severe non-IID (some clients see only 1-2 classes)
  - Pathological: each client gets exactly K out of C classes.
  - Quantity imbalance: clients receive different dataset sizes.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional


def dirichlet_partition(
    labels:      np.ndarray,
    n_clients:   int,
    alpha:       float = 0.5,
    seed:        int   = 42,
    min_samples: int   = 1,
) -> List[np.ndarray]:
    """
    Partition dataset indices across clients using Dirichlet distribution.

    Parameters
    ----------
    labels      : (N,) integer class labels
    n_clients   : number of FL clients
    alpha       : Dirichlet concentration (higher = more IID)
    seed        : random seed
    min_samples : minimum samples per client per class

    Returns
    -------
    List of length n_clients, each element is an array of indices.
    """
    rng = np.random.default_rng(seed)
    n_classes   = int(labels.max()) + 1
    class_idx   = [np.where(labels == c)[0] for c in range(n_classes)]
    client_idx  = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx_c = class_idx[c]
        rng.shuffle(idx_c)

        # Dirichlet proportions for this class across clients
        proportions = rng.dirichlet(np.full(n_clients, alpha))

        # Convert proportions to integer counts
        counts = (proportions * len(idx_c)).astype(int)
        counts[-1] = len(idx_c) - counts[:-1].sum()   # fix rounding

        # Assign
        start = 0
        for k, cnt in enumerate(counts):
            end = start + max(cnt, 0)
            client_idx[k].extend(idx_c[start:end].tolist())
            start = end

    return [np.array(ci, dtype=np.int64) for ci in client_idx]


def pathological_partition(
    labels:     np.ndarray,
    n_clients:  int,
    classes_per_client: int = 2,
    seed:       int = 42,
) -> List[np.ndarray]:
    """
    Each client gets exactly `classes_per_client` classes.
    Replicates the partitioning from "Communication-Efficient Learning" (McMahan 2017).
    """
    rng        = np.random.default_rng(seed)
    n_classes  = int(labels.max()) + 1
    class_idx  = [np.where(labels == c)[0] for c in range(n_classes)]

    # Assign class assignments to clients
    class_list = list(range(n_classes)) * (
        (n_clients * classes_per_client) // n_classes + 1)
    rng.shuffle(class_list)

    client_idx = []
    for k in range(n_clients):
        assigned = class_list[k*classes_per_client:(k+1)*classes_per_client]
        idx = np.concatenate([class_idx[c] for c in assigned if c < n_classes])
        client_idx.append(idx)

    return client_idx


def quantity_imbalanced_partition(
    n_total:    int,
    n_clients:  int,
    alpha:      float = 1.0,
    seed:       int   = 42,
) -> List[np.ndarray]:
    """
    Partition n_total samples with quantity imbalance (Dirichlet over sizes).
    Returns index lists of varying lengths.
    """
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    props   = rng.dirichlet(np.full(n_clients, alpha))
    counts  = (props * n_total).astype(int)
    counts[-1] = n_total - counts[:-1].sum()

    result, start = [], 0
    for cnt in counts:
        result.append(indices[start:start + max(cnt, 1)])
        start += max(cnt, 1)
    return result


def partition_stats(client_idx: List[np.ndarray],
                    labels: np.ndarray,
                    n_classes: int) -> np.ndarray:
    """
    Return class distribution matrix (n_clients x n_classes).
    Useful for visualising non-IID-ness.
    """
    dist = np.zeros((len(client_idx), n_classes), dtype=np.int32)
    for k, idx in enumerate(client_idx):
        if len(idx) == 0:
            continue
        cl = labels[idx]
        for c in range(n_classes):
            dist[k, c] = int((cl == c).sum())
    return dist


__all__ = [
    "dirichlet_partition",
    "pathological_partition",
    "quantity_imbalanced_partition",
    "partition_stats",
]
