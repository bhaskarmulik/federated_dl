"""
picograd/backend/numpy_backend.py
==================================
Concrete NumPy implementation of the Backend ABC.
Every method is a thin, zero-logic wrapper around np.* so future
backends can swap implementations without touching any higher-level code.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence
from picograd.backend._base import Backend, ArrayLike, Shape


class NumpyBackend(Backend):
    """NumPy-based backend.  All operations use numpy internally."""

    # ------------------------------------------------------------------ creation
    def empty(self, shape: Shape, dtype=None) -> np.ndarray:
        return np.empty(shape, dtype=dtype or np.float32)

    def zeros(self, shape: Shape, dtype=None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype or np.float32)

    def ones(self, shape: Shape, dtype=None) -> np.ndarray:
        return np.ones(shape, dtype=dtype or np.float32)

    def full(self, shape: Shape, fill_value, dtype=None) -> np.ndarray:
        return np.full(shape, fill_value, dtype=dtype or np.float32)

    def arange(self, start, stop=None, step=1, dtype=None) -> np.ndarray:
        if stop is None:
            return np.arange(start, dtype=dtype or np.float32)
        return np.arange(start, stop, step, dtype=dtype or np.float32)

    def from_numpy(self, arr) -> np.ndarray:
        return np.asarray(arr, dtype=np.float32) if not isinstance(arr, np.ndarray) else arr

    def to_numpy(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data)

    # ------------------------------------------------------------------ elementwise binary
    def add(self, a, b): return np.add(a, b)
    def sub(self, a, b): return np.subtract(a, b)
    def mul(self, a, b): return np.multiply(a, b)
    def div(self, a, b): return np.true_divide(a, b)
    def pow(self, a, exp): return np.power(a, exp)

    # ------------------------------------------------------------------ elementwise unary
    def exp(self, a): return np.exp(a)
    def log(self, a): return np.log(a)
    def neg(self, a): return np.negative(a)
    def abs(self, a): return np.abs(a)
    def clip(self, a, a_min, a_max): return np.clip(a, a_min, a_max)
    def sqrt(self, a): return np.sqrt(a)

    # ------------------------------------------------------------------ comparison
    def eq(self, a, b): return np.equal(a, b)
    def gt(self, a, b): return np.greater(a, b)
    def lt(self, a, b): return np.less(a, b)
    def where(self, condition, x, y): return np.where(condition, x, y)

    # ------------------------------------------------------------------ reductions
    def sum(self, a, axis=None, keepdims=False):
        return np.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a, axis=None, keepdims=False):
        return np.mean(a, axis=axis, keepdims=keepdims)

    def max(self, a, axis=None, keepdims=False):
        return np.max(a, axis=axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        return np.min(a, axis=axis, keepdims=keepdims)

    def argmax(self, a, axis=None):
        return np.argmax(a, axis=axis)

    def argmin(self, a, axis=None):
        return np.argmin(a, axis=axis)

    # ------------------------------------------------------------------ linear algebra
    def matmul(self, a, b): return np.matmul(a, b)

    def transpose(self, a, axes=None):
        return np.transpose(a, axes=axes)

    # ------------------------------------------------------------------ shape manipulation
    def reshape(self, a, shape): return np.reshape(a, shape)

    def expand(self, a, shape): return np.broadcast_to(a, shape)

    def concatenate(self, tensors, axis=0): return np.concatenate(tensors, axis=axis)

    def stack(self, tensors, axis=0): return np.stack(tensors, axis=axis)

    def split(self, a, indices_or_sections, axis=0):
        return np.split(a, indices_or_sections, axis=axis)

    def squeeze(self, a, axis=None): return np.squeeze(a, axis=axis)

    def unsqueeze(self, a, axis): return np.expand_dims(a, axis=axis)

    def flip(self, a, axis=None): return np.flip(a, axis=axis)

    def pad(self, a, pad_width, mode="constant", constant_values=0):
        if mode == "constant":
            return np.pad(a, pad_width, mode=mode, constant_values=constant_values)
        return np.pad(a, pad_width, mode=mode)

    # ------------------------------------------------------------------ indexing
    def gather(self, a, indices, axis):
        return np.take(a, indices, axis=axis)

    def scatter_add(self, target, indices, src, axis):
        out = target.copy()
        np.add.at(out, [slice(None)] * axis + [indices], src)
        return out

    # ------------------------------------------------------------------ random
    def randn(self, shape, dtype=None):
        arr = np.random.randn(*shape)
        return arr.astype(dtype or np.float32)

    def rand(self, shape, dtype=None):
        arr = np.random.rand(*shape)
        return arr.astype(dtype or np.float32)

    def randint(self, low, high, shape):
        return np.random.randint(low, high, size=shape)

    def seed(self, n):
        np.random.seed(n)

    def dropout_mask(self, shape, p):
        """Boolean keep-mask: True with probability (1-p)."""
        return np.random.rand(*shape) > p

    # ------------------------------------------------------------------ misc
    def copy(self, a): return a.copy()

    def contiguous(self, a): return np.ascontiguousarray(a)

    def dtype_of(self, a): return a.dtype

    def shape_of(self, a): return a.shape

    def numel(self, a): return a.size

    def item(self, a):
        if isinstance(a, np.ndarray):
            return a.item()
        return float(a)

    def cast(self, a, dtype): return a.astype(dtype)
