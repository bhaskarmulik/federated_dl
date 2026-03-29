"""
picograd/serialization.py
==========================
save() / load() for picograd state dicts.

A state_dict is a dict[str, np.ndarray] -- exactly what the existing
FL pipeline's sd_to_bytes / bytes_to_sd pattern expects.
"""

from __future__ import annotations
import pickle
import io
import numpy as np
from typing import Dict, Any


def save(obj: Any, path: str) -> None:
    """Serialize obj (typically a state_dict) to disk via pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str) -> Any:
    """Deserialize from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def state_dict_to_bytes(sd: Dict[str, np.ndarray]) -> bytes:
    """Serialize state_dict -> bytes (for gRPC transmission)."""
    buf = io.BytesIO()
    pickle.dump(sd, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()


def bytes_to_state_dict(data: bytes) -> Dict[str, np.ndarray]:
    """Deserialize bytes -> state_dict."""
    return pickle.loads(data)


__all__ = ["save", "load", "state_dict_to_bytes", "bytes_to_state_dict"]
