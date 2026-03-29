"""``picograd.Tensor`` — the fundamental data structure.

Wraps a backend array and optionally participates in the autograd graph.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

from .backend import get_backend
from .backend._base import ArrayLike, Backend, DType, Shape

__all__ = ["Tensor"]

#Yeh model ne likha hai cz Ima bored bhai, kitna likhu
class Tensor:
    """Multi-dimensional array with automatic differentiation support.

    Parameters
    ----------
    data : array-like
        Raw backend array **or** a Python list / scalar (auto-converted).
    requires_grad : bool
        If ``True``, operations on this tensor build an autograd graph.
    dtype : optional
        Force a dtype on construction.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype: Optional[DType] = None,
        _backend: Optional[Backend] = None,
    ) -> None:
        self._backend: Backend = _backend or get_backend()

        # Normalise *data* into a backend array.
        if isinstance(data, Tensor):
            data = data._data
        import numpy as np
        if not isinstance(data, np.ndarray):
            data = self._backend.array(data, dtype=dtype)
        elif dtype is not None:
            data = self._backend.astype(data, dtype)
        self._data: ArrayLike = data

        self.requires_grad: bool = requires_grad
        self.grad: Optional[Tensor] = None
        self._grad_fn: Optional[Any] = None  # autograd.Node | None
        self.is_leaf: bool = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def data(self) -> "Tensor":
        """View of the underlying data without autograd attachment."""
        return Tensor(self._data, requires_grad=False, _backend=self._backend)

    @data.setter
    def data(self, value: "Tensor") -> None:
        self._data = value._data if isinstance(value, Tensor) else value

    @property
    def shape(self) -> Shape:
        return self._backend.shape_of(self._data)

    @property
    def dtype(self) -> DType:
        return self._backend.dtype_of(self._data)

    @property
    def ndim(self) -> int:
        return self._backend.ndim(self._data)

    @property
    def numel(self) -> int:
        return self._backend.numel(self._data)

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def numpy(self):
        """Return a plain ``np.ndarray`` (detached)."""
        return self._backend.to_numpy(self._data)

    def item(self) -> float:
        return float(self._data.flat[0])

    def detach(self) -> "Tensor":
        t = Tensor(self._data, requires_grad=False, _backend=self._backend)
        t.is_leaf = True
        return t

    def clone(self) -> "Tensor":
        t = Tensor(self._backend.copy(self._data),
                    requires_grad=self.requires_grad,
                    _backend=self._backend)
        return t

    def contiguous(self) -> "Tensor":
        return Tensor(self._backend.contiguous(self._data),
                      requires_grad=self.requires_grad,
                      _backend=self._backend)

    def float(self) -> "Tensor":
        return Tensor(self._backend.astype(self._data,
                                            self._backend.default_float_dtype()),
                      requires_grad=self.requires_grad,
                      _backend=self._backend)

    def long(self) -> "Tensor":
        import numpy as np
        return Tensor(self._backend.astype(self._data, np.int64),
                      requires_grad=False,
                      _backend=self._backend)

    def zero_(self) -> "Tensor":
        self._data = self._backend.zeros(self.shape, dtype=self.dtype)
        return self

    def fill_(self, value: float) -> "Tensor":
        self._data = self._backend.full(self.shape, value, dtype=self.dtype)
        return self

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        from .autograd.engine import backward
        backward(self, gradient)

    def zero_grad(self) -> None:
        self.grad = None

    # ------------------------------------------------------------------
    # Operator overloads → dispatch to ops.* Functions
    # ------------------------------------------------------------------
    def __add__(self, other: Any) -> "Tensor":
        from .ops.elemwise import Add
        return Add.apply(self, _ensure_tensor(other, self._backend))

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        from .ops.elemwise import Sub
        return Sub.apply(self, _ensure_tensor(other, self._backend))

    def __rsub__(self, other: Any) -> "Tensor":
        from .ops.elemwise import Sub
        return Sub.apply(_ensure_tensor(other, self._backend), self)

    def __mul__(self, other: Any) -> "Tensor":
        from .ops.elemwise import Mul
        return Mul.apply(self, _ensure_tensor(other, self._backend))

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Tensor":
        from .ops.elemwise import Div
        return Div.apply(self, _ensure_tensor(other, self._backend))

    def __rtruediv__(self, other: Any) -> "Tensor":
        from .ops.elemwise import Div
        return Div.apply(_ensure_tensor(other, self._backend), self)

    def __neg__(self) -> "Tensor":
        from .ops.elemwise import Neg
        return Neg.apply(self)

    def __pow__(self, exp: Union[float, int, "Tensor"]) -> "Tensor":
        from .ops.elemwise import Pow
        return Pow.apply(self, _ensure_tensor(exp, self._backend))

    def __matmul__(self, other: "Tensor") -> "Tensor":
        from .ops.matmul import MatMul
        return MatMul.apply(self, other)

    def __getitem__(self, key) -> "Tensor":
        from .ops.shape import Slice
        return Slice.apply(self, key=key)

    # ------------------------------------------------------------------
    # Comparison (no grad)
    # ------------------------------------------------------------------
    def __eq__(self, other: Any) -> "Tensor":  # type: ignore[override]
        o = _ensure_tensor(other, self._backend)
        return Tensor(self._backend.eq(self._data, o._data), _backend=self._backend)

    def __ne__(self, other: Any) -> "Tensor":  # type: ignore[override]
        o = _ensure_tensor(other, self._backend)
        return Tensor(self._backend.ne(self._data, o._data), _backend=self._backend)

    def __gt__(self, other: Any) -> "Tensor":
        o = _ensure_tensor(other, self._backend)
        return Tensor(self._backend.gt(self._data, o._data), _backend=self._backend)

    def __ge__(self, other: Any) -> "Tensor":
        o = _ensure_tensor(other, self._backend)
        return Tensor(self._backend.ge(self._data, o._data), _backend=self._backend)

    def __lt__(self, other: Any) -> "Tensor":
        o = _ensure_tensor(other, self._backend)
        return Tensor(self._backend.lt(self._data, o._data), _backend=self._backend)

    def __le__(self, other: Any) -> "Tensor":
        o = _ensure_tensor(other, self._backend)
        return Tensor(self._backend.le(self._data, o._data), _backend=self._backend)

    # ------------------------------------------------------------------
    # Reduction / shape / unary methods
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        from .ops.reduce import Sum
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        from .ops.reduce import Mean
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        from .ops.reduce import Max
        return Max.apply(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims: bool = False) -> "Tensor":
        from .ops.reduce import Min
        return Min.apply(self, axis=axis, keepdims=keepdims)

    def argmax(self, axis=None) -> "Tensor":
        return Tensor(self._backend.argmax(self._data, axis=axis),
                      _backend=self._backend)

    def argmin(self, axis=None) -> "Tensor":
        return Tensor(self._backend.argmin(self._data, axis=axis),
                      _backend=self._backend)

    def reshape(self, *shape: int) -> "Tensor":
        from .ops.shape import Reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape.apply(self, shape=shape)

    def view(self, *shape: int) -> "Tensor":
        return self.reshape(*shape)

    def transpose(self, *axes) -> "Tensor":
        from .ops.shape import Transpose
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return Transpose.apply(self, axes=axes)

    def permute(self, *dims: int) -> "Tensor":
        return self.transpose(*dims)

    def unsqueeze(self, dim: int) -> "Tensor":
        from .ops.shape import Unsqueeze
        return Unsqueeze.apply(self, dim=dim)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        from .ops.shape import Squeeze
        return Squeeze.apply(self, dim=dim)

    def expand(self, *shape: int) -> "Tensor":
        from .ops.shape import Expand
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Expand.apply(self, shape=shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        s = list(self.shape)
        if end_dim < 0:
            end_dim = len(s) + end_dim
        new_shape = s[:start_dim] + [-1] + s[end_dim + 1:]
        return self.reshape(*new_shape)

    def exp(self) -> "Tensor":
        from .ops.elemwise import Exp
        return Exp.apply(self)

    def log(self) -> "Tensor":
        from .ops.elemwise import Log
        return Log.apply(self)

    def abs(self) -> "Tensor":
        from .ops.elemwise import Abs
        return Abs.apply(self)

    def sqrt(self) -> "Tensor":
        return self ** 0.5

    def tanh(self) -> "Tensor":
        from .ops.activations import Tanh
        return Tanh.apply(self)

    def relu(self) -> "Tensor":
        from .ops.activations import ReLU
        return ReLU.apply(self)

    def sigmoid(self) -> "Tensor":
        from .ops.activations import Sigmoid
        return Sigmoid.apply(self)

    # ------------------------------------------------------------------
    # Static constructors (mirror torch.zeros etc.)
    # ------------------------------------------------------------------
    @staticmethod
    def zeros(*shape: int, requires_grad: bool = False, dtype=None) -> "Tensor":
        B = get_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(B.zeros(shape, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape: int, requires_grad: bool = False, dtype=None) -> "Tensor":
        B = get_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(B.ones(shape, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape: int, requires_grad: bool = False, dtype=None) -> "Tensor":
        B = get_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(B.randn(shape, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def rand(*shape: int, requires_grad: bool = False, dtype=None) -> "Tensor":
        B = get_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(B.rand(shape, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def from_numpy(arr, requires_grad: bool = False) -> "Tensor":
        B = get_backend()
        return Tensor(B.from_numpy(arr), requires_grad=requires_grad)

    @staticmethod
    def eye(n: int, requires_grad: bool = False, dtype=None) -> "Tensor":
        B = get_backend()
        return Tensor(B.eye(n, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def arange(start, stop=None, step=1, dtype=None) -> "Tensor":
        B = get_backend()
        if stop is None:
            start, stop = 0, start
        return Tensor(B.arange(start, stop, step, dtype=dtype))

    @staticmethod
    def empty(*shape: int, dtype=None) -> "Tensor":
        B = get_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(B.empty(shape, dtype=dtype))

    @staticmethod
    def full(shape, fill_value: float, requires_grad: bool = False, dtype=None) -> "Tensor":
        B = get_backend()
        return Tensor(B.full(shape, fill_value, dtype=dtype), requires_grad=requires_grad)

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        data_str = repr(self._data)
        grad_fn_str = ""
        if self._grad_fn is not None:
            grad_fn_str = f", grad_fn=<{self._grad_fn.function_cls.__name__}>"
        elif self.requires_grad:
            grad_fn_str = ", requires_grad=True"
        return f"picograd.Tensor({data_str}{grad_fn_str})"

    def __len__(self) -> int:
        return self.shape[0] if self.ndim > 0 else 1

    def __bool__(self) -> bool:
        if self.numel != 1:
            raise RuntimeError("Boolean value of multi-element tensor is ambiguous")
        return bool(self._data.flat[0])

    def __float__(self) -> float:
        return self.item()

    def __int__(self) -> int:
        return int(self.item())

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _ensure_tensor(x: Any, backend: Backend) -> Tensor:
    """Convert scalars / lists to Tensor."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, _backend=backend)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    from .ops.shape import Cat
    return Cat.apply(*tensors, dim=dim)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    from .ops.shape import Stack
    return Stack.apply(*tensors, dim=dim)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    B = get_backend()
    return Tensor(B.where(condition._data, x._data, y._data),
                  requires_grad=(x.requires_grad or y.requires_grad),
                  _backend=B)
