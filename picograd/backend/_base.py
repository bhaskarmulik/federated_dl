"""
picograd/backend/_base.py
=========================
Abstract base class (Backend) defining the ~40-method dispatch table that
all numerical operations must go through. Concrete backends (NumpyBackend,
future TritonBackend) implement every method here.

Design principle: Every picograd Tensor operation delegates to
  get_backend().<method>(...)
so swapping the backend is a single call: picograd.set_backend(new_backend).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Type aliases used in signatures
# ---------------------------------------------------------------------------
ArrayLike = object          # np.ndarray  or  whatever the backend uses
Shape = Tuple[int, ...]
DType = object              # np.dtype or string like "float32"


class Backend(ABC):
    """Dispatch table for all numerical operations in picograd.

    A Backend is a *stateless* strategy object.  Every method takes raw
    backend arrays (e.g. np.ndarray) and returns raw backend arrays.
    The Tensor class wraps these arrays and calls get_backend().method().
    """

    # ------------------------------------------------------------------
    # 1.  Creation
    # ------------------------------------------------------------------

    @abstractmethod
    def empty(self, shape: Shape, dtype=None) -> ArrayLike:
        """Return uninitialized array of given shape."""

    @abstractmethod
    def zeros(self, shape: Shape, dtype=None) -> ArrayLike:
        """Return array of zeros."""

    @abstractmethod
    def ones(self, shape: Shape, dtype=None) -> ArrayLike:
        """Return array of ones."""

    @abstractmethod
    def full(self, shape: Shape, fill_value, dtype=None) -> ArrayLike:
        """Return array filled with fill_value."""

    @abstractmethod
    def arange(self, start, stop=None, step=1, dtype=None) -> ArrayLike:
        """Return evenly spaced values within a given interval."""

    @abstractmethod
    def from_numpy(self, arr) -> ArrayLike:
        """Convert numpy array → backend array (may be a copy or view)."""

    @abstractmethod
    def to_numpy(self, data: ArrayLike):
        """Convert backend array → numpy ndarray."""

    # ------------------------------------------------------------------
    # 2.  Elementwise binary ops (broadcast semantics)
    # ------------------------------------------------------------------

    @abstractmethod
    def add(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def sub(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def mul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def div(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def pow(self, a: ArrayLike, exp) -> ArrayLike: ...

    # ------------------------------------------------------------------
    # 3.  Elementwise unary ops
    # ------------------------------------------------------------------

    @abstractmethod
    def exp(self, a: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def log(self, a: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def neg(self, a: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def abs(self, a: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def clip(self, a: ArrayLike, a_min, a_max) -> ArrayLike: ...

    @abstractmethod
    def sqrt(self, a: ArrayLike) -> ArrayLike: ...

    # ------------------------------------------------------------------
    # 4.  Comparison
    # ------------------------------------------------------------------

    @abstractmethod
    def eq(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def gt(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def lt(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def where(self, condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike: ...

    # ------------------------------------------------------------------
    # 5.  Reductions
    # ------------------------------------------------------------------

    @abstractmethod
    def sum(self, a: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike: ...

    @abstractmethod
    def mean(self, a: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike: ...

    @abstractmethod
    def max(self, a: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike: ...

    @abstractmethod
    def min(self, a: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike: ...

    @abstractmethod
    def argmax(self, a: ArrayLike, axis=None) -> ArrayLike: ...

    @abstractmethod
    def argmin(self, a: ArrayLike, axis=None) -> ArrayLike: ...

    # ------------------------------------------------------------------
    # 6.  Linear algebra
    # ------------------------------------------------------------------

    @abstractmethod
    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def transpose(self, a: ArrayLike, axes=None) -> ArrayLike:
        """Permute dimensions.  axes=None reverses all dims."""

    # ------------------------------------------------------------------
    # 7.  Shape manipulation
    # ------------------------------------------------------------------

    @abstractmethod
    def reshape(self, a: ArrayLike, shape: Shape) -> ArrayLike: ...

    @abstractmethod
    def expand(self, a: ArrayLike, shape: Shape) -> ArrayLike:
        """Broadcast a to shape (no data copy, read-only view)."""

    @abstractmethod
    def concatenate(self, tensors: Sequence[ArrayLike], axis: int = 0) -> ArrayLike: ...

    @abstractmethod
    def stack(self, tensors: Sequence[ArrayLike], axis: int = 0) -> ArrayLike: ...

    @abstractmethod
    def split(self, a: ArrayLike, indices_or_sections, axis: int = 0): ...

    @abstractmethod
    def squeeze(self, a: ArrayLike, axis=None) -> ArrayLike: ...

    @abstractmethod
    def unsqueeze(self, a: ArrayLike, axis: int) -> ArrayLike: ...

    @abstractmethod
    def flip(self, a: ArrayLike, axis=None) -> ArrayLike: ...

    @abstractmethod
    def pad(self, a: ArrayLike, pad_width, mode: str = "constant", constant_values=0) -> ArrayLike: ...

    # ------------------------------------------------------------------
    # 8.  Indexing / scatter-gather
    # ------------------------------------------------------------------

    @abstractmethod
    def gather(self, a: ArrayLike, indices: ArrayLike, axis: int) -> ArrayLike:
        """Gather elements from a along axis using indices."""

    @abstractmethod
    def scatter_add(self, target: ArrayLike, indices: ArrayLike, src: ArrayLike, axis: int) -> ArrayLike:
        """Return a new array with src values scattered-added into target at indices."""

    # ------------------------------------------------------------------
    # 9.  Random
    # ------------------------------------------------------------------

    @abstractmethod
    def randn(self, shape: Shape, dtype=None) -> ArrayLike:
        """Sample from standard normal N(0,1)."""

    @abstractmethod
    def rand(self, shape: Shape, dtype=None) -> ArrayLike:
        """Sample from uniform U(0,1)."""

    @abstractmethod
    def randint(self, low: int, high: int, shape: Shape) -> ArrayLike: ...

    @abstractmethod
    def seed(self, n: int) -> None: ...

    @abstractmethod
    def dropout_mask(self, shape: Shape, p: float) -> ArrayLike:
        """Return boolean keep-mask with P(keep) = 1-p."""

    # ------------------------------------------------------------------
    # 10.  Misc
    # ------------------------------------------------------------------

    @abstractmethod
    def copy(self, a: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def contiguous(self, a: ArrayLike) -> ArrayLike:
        """Return a C-contiguous copy of a."""

    @abstractmethod
    def dtype_of(self, a: ArrayLike): ...

    @abstractmethod
    def shape_of(self, a: ArrayLike) -> Shape: ...

    @abstractmethod
    def numel(self, a: ArrayLike) -> int: ...

    @abstractmethod
    def item(self, a: ArrayLike):
        """Return Python scalar from a 0-d or 1-element array."""

    @abstractmethod
    def cast(self, a: ArrayLike, dtype) -> ArrayLike:
        """Cast a to given dtype."""

    # ------------------------------------------------------------------
    # 11.  Convenience helpers (non-abstract; use the abstract methods)
    # ------------------------------------------------------------------

    def zeros_like(self, a: ArrayLike) -> ArrayLike:
        return self.zeros(self.shape_of(a), dtype=self.dtype_of(a))

    def ones_like(self, a: ArrayLike) -> ArrayLike:
        return self.ones(self.shape_of(a), dtype=self.dtype_of(a))

    def full_like(self, a: ArrayLike, fill_value) -> ArrayLike:
        return self.full(self.shape_of(a), fill_value, dtype=self.dtype_of(a))
