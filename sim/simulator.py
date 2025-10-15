from __future__ import annotations
from typing import Dict, Any, List
import numpy as np

def simulate_client_indices(labels: np.ndarray, K: int, alpha: float = 0.5):
    from flkit.data.partition import dirichlet_partition
    return dirichlet_partition(labels, K, alpha)
