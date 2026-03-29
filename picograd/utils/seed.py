"""picograd/utils/seed.py"""
import numpy as np
import random


def manual_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    from picograd.backend import get_backend
    get_backend().seed(seed)
