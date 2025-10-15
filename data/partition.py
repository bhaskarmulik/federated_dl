from __future__ import annotations
from typing import List, Dict
import numpy as np

def dirichlet_partition(labels: np.ndarray, K: int, alpha: float = 0.5) -> List[np.ndarray]:
    """Return index lists for K clients with Dirichlet non-IID splits."""
    n = labels.shape[0]
    classes = np.unique(labels)
    idx_by_class = {c: np.where(labels==c)[0] for c in classes}
    parts = [[] for _ in range(K)]
    for c in classes:
        idx = idx_by_class[c]
        np.random.shuffle(idx)
        # draw proportions for this class across clients
        p = np.random.dirichlet(alpha*np.ones(K))
        # split indices approximately by p
        splits = (np.cumsum(p)*len(idx)).astype(int)[:-1]
        shards = np.split(idx, splits)
        for k, shard in enumerate(shards):
            parts[k].extend(shard.tolist())
    # shuffle each part
    return [np.array(p, dtype=int) for p in [np.random.permutation(p) for p in parts]]
