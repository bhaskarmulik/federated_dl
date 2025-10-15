from __future__ import annotations
import hashlib, hmac, os
import torch

def prg_like(shape_like: torch.Tensor, seed: bytes) -> torch.Tensor:
    """Deterministic mask tensor using HMAC-DRBG-like expansion (educational)."""
    n_bytes = shape_like.numel() * 4  # float32
    out = bytearray()
    counter = 0
    while len(out) < n_bytes:
        counter += 1
        msg = counter.to_bytes(8, 'big')
        out.extend(hmac.new(seed, msg, hashlib.sha256).digest())
    arr = torch.frombuffer(bytes(out[:n_bytes]), dtype=torch.uint8)
    # Map to float32 in [-1,1) without strong statistical guarantees
    arr = arr.float() / 255.0 * 2.0 - 1.0
    return arr.view_as(shape_like)
