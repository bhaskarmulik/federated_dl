"""
picograd/tensor.py
==================
The Tensor class: the user-facing object in picograd.

Every Tensor wraps a raw backend array (`_data`) and participates in the
autograd DAG via `_grad_fn`.  Leaf tensors (created by the user) have
`_grad_fn = None` and are the targets of gradient accumulation.

Design matches PyTorch's user-facing API closely so existing code is
easy to port:  torch.Tensor -> picograd.Tensor.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from picograd.autograd.function import Node


class Tensor:
    """
    N-dimensional array with optional gradient tracking.

    Parameters
    ----------
    data         : array-like or np.ndarray
    requires_grad: whether to track ops for autograd
    dtype        : numpy dtype (default float32)
    """

    def __init__(self, data, requires_grad: bool = False, dtype=None):
        from picograd.backend import get_backend
        b = get_backend()

        if isinstance(data, Tensor):
            data = data._data

        if isinstance(data, np.ndarray):
            if dtype is not None:
                self._data = data.astype(dtype)
            elif np.issubdtype(data.dtype, np.floating):
                self._data = data.copy()
            else:
                self._data = data.astype(np.float32)
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data, dtype=dtype or np.float32)
        elif isinstance(data, (int, float)):
            self._data = np.array(data, dtype=dtype or np.float32)
        else:
            self._data = data

        self.requires_grad: bool = requires_grad
        self.grad: Optional[Tensor] = None
        self._grad_fn: Optional["Node"] = None
        self._is_leaf: bool = True

    # ------------------------------------------------------------------ properties
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def numel(self) -> int:
        return self._data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    # ------------------------------------------------------------------ static constructors
    @staticmethod
    def zeros(*shape, dtype=None, requires_grad: bool = False) -> "Tensor":
        import numpy as np
        return Tensor(np.zeros(shape, dtype=dtype or np.float32), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, dtype=None, requires_grad: bool = False) -> "Tensor":
        import numpy as np
        return Tensor(np.ones(shape, dtype=dtype or np.float32), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, dtype=None, requires_grad: bool = False) -> "Tensor":
        import numpy as np
        return Tensor(np.random.randn(*shape).astype(dtype or np.float32), requires_grad=requires_grad)

    @staticmethod
    def rand(*shape, dtype=None, requires_grad: bool = False) -> "Tensor":
        import numpy as np
        return Tensor(np.random.rand(*shape).astype(dtype or np.float32), requires_grad=requires_grad)

    @staticmethod
    def zeros_like(t: "Tensor", requires_grad: bool = False) -> "Tensor":
        import numpy as np
        return Tensor(np.zeros_like(t._data), requires_grad=requires_grad)

    @staticmethod
    def ones_like(t: "Tensor", requires_grad: bool = False) -> "Tensor":
        import numpy as np
        return Tensor(np.ones_like(t._data), requires_grad=requires_grad)

    @staticmethod
    def from_numpy(arr, requires_grad: bool = False) -> "Tensor":
        return Tensor(arr, requires_grad=requires_grad)

    @staticmethod
    def full(shape, fill_value, dtype=None, requires_grad: bool = False) -> "Tensor":
        import numpy as np
        return Tensor(np.full(shape, fill_value, dtype=dtype or np.float32), requires_grad=requires_grad)

    @staticmethod
    def eye(n, dtype=None, requires_grad: bool = False) -> "Tensor":
        import numpy as np
        return Tensor(np.eye(n, dtype=dtype or np.float32), requires_grad=requires_grad)

    @staticmethod
    def arange(start, stop=None, step=1, dtype=None) -> "Tensor":
        import numpy as np
        if stop is None:
            return Tensor(np.arange(start, dtype=dtype or np.float32))
        return Tensor(np.arange(start, stop, step, dtype=dtype or np.float32))

    # ------------------------------------------------------------------ data access
    def numpy(self) -> np.ndarray:
        return self._data

    def item(self):
        return self._data.item()

    def detach(self) -> "Tensor":
        """Return a new Tensor sharing data but with no grad tracking."""
        t = Tensor(self._data, requires_grad=False)
        return t

    def clone(self) -> "Tensor":
        return Tensor(self._data.copy(), requires_grad=self.requires_grad)

    # ------------------------------------------------------------------ backward
    def backward(self, grad: Optional["Tensor"] = None) -> None:
        from picograd.autograd.engine import backward
        backward(self, grad)

    def zero_grad(self) -> None:
        """Zero out accumulated gradient."""
        self.grad = None

    # ------------------------------------------------------------------ gradient utilities
    def retain_grad(self) -> None:
        """Mark non-leaf tensor to retain gradient after backward."""
        self._retain_grad = True

    # ------------------------------------------------------------------ arithmetic operators
    def __add__(self, other) -> "Tensor":
        from picograd.ops.elemwise import Add
        other = _ensure_tensor(other)
        return Add.apply(self, other)

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other) -> "Tensor":
        from picograd.ops.elemwise import Sub
        other = _ensure_tensor(other)
        return Sub.apply(self, other)

    def __rsub__(self, other) -> "Tensor":
        from picograd.ops.elemwise import Sub
        other = _ensure_tensor(other)
        return Sub.apply(other, self)

    def __mul__(self, other) -> "Tensor":
        from picograd.ops.elemwise import Mul
        other = _ensure_tensor(other)
        return Mul.apply(self, other)

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Tensor":
        from picograd.ops.elemwise import Div
        other = _ensure_tensor(other)
        return Div.apply(self, other)

    def __rtruediv__(self, other) -> "Tensor":
        from picograd.ops.elemwise import Div
        other = _ensure_tensor(other)
        return Div.apply(other, self)

    def __neg__(self) -> "Tensor":
        from picograd.ops.elemwise import Neg
        return Neg.apply(self)

    def __pow__(self, exp) -> "Tensor":
        from picograd.ops.elemwise import Pow
        return Pow.apply(self, exp)

    def __matmul__(self, other) -> "Tensor":
        from picograd.ops.matmul import MatMul
        return MatMul.apply(self, other)

    def __rmatmul__(self, other) -> "Tensor":
        from picograd.ops.matmul import MatMul
        other = _ensure_tensor(other)
        return MatMul.apply(other, self)

    # ------------------------------------------------------------------ comparison (no grad)
    def __eq__(self, other) -> "Tensor":
        from picograd.backend import get_backend
        other = _ensure_tensor(other)
        return Tensor(get_backend().eq(self._data, other._data), requires_grad=False)

    def __gt__(self, other) -> "Tensor":
        from picograd.backend import get_backend
        other = _ensure_tensor(other)
        return Tensor(get_backend().gt(self._data, other._data), requires_grad=False)

    def __lt__(self, other) -> "Tensor":
        from picograd.backend import get_backend
        other = _ensure_tensor(other)
        return Tensor(get_backend().lt(self._data, other._data), requires_grad=False)

    def __ge__(self, other) -> "Tensor":
        return (self > other) + (self == other)

    def __le__(self, other) -> "Tensor":
        return (self < other) + (self == other)

    # ------------------------------------------------------------------ shape ops (delegate to Function subclasses)
    def reshape(self, *shape) -> "Tensor":
        from picograd.ops.shape import Reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape.apply(self, shape)

    def view(self, *shape) -> "Tensor":
        return self.reshape(*shape)

    def flatten(self, start_dim: int = 0) -> "Tensor":
        shape = self.shape
        pre = shape[:start_dim]
        flat = 1
        for s in shape[start_dim:]:
            flat *= s
        return self.reshape(*pre, flat)

    def transpose(self, axes=None) -> "Tensor":
        from picograd.ops.shape import Transpose
        return Transpose.apply(self, axes)

    def permute(self, *axes) -> "Tensor":
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return self.transpose(axes=axes)

    def squeeze(self, axis=None) -> "Tensor":
        from picograd.ops.shape import Squeeze
        return Squeeze.apply(self, axis)

    def unsqueeze(self, axis: int) -> "Tensor":
        from picograd.ops.shape import Unsqueeze
        return Unsqueeze.apply(self, axis)

    def expand(self, *shape) -> "Tensor":
        from picograd.ops.shape import Expand
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Expand.apply(self, shape)

    def contiguous(self) -> "Tensor":
        from picograd.backend import get_backend
        return Tensor(get_backend().contiguous(self._data), requires_grad=self.requires_grad)

    # ------------------------------------------------------------------ reduction ops
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        from picograd.ops.reduce import Sum
        return Sum.apply(self, axis, keepdims)

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        from picograd.ops.reduce import Mean
        return Mean.apply(self, axis, keepdims)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        from picograd.ops.reduce import Max
        return Max.apply(self, axis, keepdims)

    def min(self, axis=None, keepdims: bool = False) -> "Tensor":
        from picograd.ops.reduce import Min_op
        return Min_op.apply(self, axis, keepdims)

    # ------------------------------------------------------------------ indexing
    def __getitem__(self, idx) -> "Tensor":
        from picograd.ops.shape import Slice
        return Slice.apply(self, idx)

    def __setitem__(self, idx, value) -> None:
        if isinstance(value, Tensor):
            self._data[idx] = value._data
        else:
            self._data[idx] = value

    # ------------------------------------------------------------------ misc
    def abs(self) -> "Tensor":
        from picograd.ops.elemwise import Abs
        return Abs.apply(self)

    def exp(self) -> "Tensor":
        from picograd.ops.elemwise import Exp
        return Exp.apply(self)

    def log(self) -> "Tensor":
        from picograd.ops.elemwise import Log
        return Log.apply(self)

    def sqrt(self) -> "Tensor":
        from picograd.ops.elemwise import Sqrt
        return Sqrt.apply(self)

    def clip(self, a_min, a_max) -> "Tensor":
        from picograd.ops.elemwise import Clip
        return Clip.apply(self, a_min, a_max)

    def __repr__(self) -> str:
        grad_fn_str = f", grad_fn=<{type(self._grad_fn.function).__name__}>" if self._grad_fn else ""
        requires_grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"tensor({self._data}{requires_grad_str}{grad_fn_str})"

    def __len__(self) -> int:
        return self.shape[0]

    def __bool__(self) -> bool:
        return bool(self._data)

    def __float__(self) -> float:
        return float(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def size(self, dim: int = None):
        if dim is None:
            return self.shape
        return self.shape[dim]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_tensor(val) -> "Tensor":
    """Coerce scalars / numpy arrays to Tensor."""
    if isinstance(val, Tensor):
        return val
    return Tensor(val)


# ---------------------------------------------------------------------------
# Module-level functional shortcuts
# ---------------------------------------------------------------------------

def cat(tensors, axis: int = 0) -> "Tensor":
    from picograd.ops.shape import Cat
    return Cat.apply(*tensors, axis=axis)


def stack(tensors, axis: int = 0) -> "Tensor":
    from picograd.ops.shape import Stack
    return Stack.apply(*tensors, axis=axis)


__all__ = ["Tensor", "cat", "stack"]
