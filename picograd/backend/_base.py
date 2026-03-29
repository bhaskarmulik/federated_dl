"""Abstract backend interface — the single point of dispatch for all numerical operations.

Every method here operates on *raw backend arrays* (e.g. ``np.ndarray`` for the
NumPy backend).  Higher-level code (Tensor, autograd, nn) never imports a
concrete numerics library directly; it always goes through
``get_backend().<method>(...)``.

To add a new backend (Triton, C++, CuPy, …), subclass ``Backend`` and
implement every abstract method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union

# Type aliases (backend-agnostic)
Shape = Tuple[int, ...]
DType = Any          # concrete type depends on backend (np.float32, etc.)
ArrayLike = Any      # raw backend array (np.ndarray, cupy.ndarray, …)


class Backend(ABC):
    """
    Pure-virtual backend bois.  Dekho bhai, subclasses will provide them implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. ``'numpy'``, ``'triton'``."""
        ...

    @abstractmethod
    def empty(self, shape: Shape, dtype: DType | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def zeros(self, shape: Shape, dtype: DType | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def ones(self, shape: Shape, dtype: DType | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def full(self, shape: Shape, fill_value: float, dtype: DType | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def arange(self, start: float, stop: float, step: float = 1.0, dtype: DType | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def eye(self, n: int, dtype: DType | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def from_numpy(self, arr: Any) -> ArrayLike:
        """Import an ``np.ndarray`` into this backend's array type."""
        ...

    @abstractmethod
    def to_numpy(self, a: ArrayLike) -> Any:
        """Export a backend array to a plain ``np.ndarray``."""
        ...

    @abstractmethod
    def array(self, data: Any, dtype: DType | None = None) -> ArrayLike:
        """Create a backend array from a Python list / nested list / scalar."""
        ...

    @abstractmethod
    def copy(self, a: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def shape_of(self, a: ArrayLike) -> Shape:
        ...

    @abstractmethod
    def dtype_of(self, a: ArrayLike) -> DType:
        ...

    @abstractmethod
    def numel(self, a: ArrayLike) -> int:
        ...

    @abstractmethod
    def ndim(self, a: ArrayLike) -> int:
        ...

    @abstractmethod
    def is_floating_point(self, a: ArrayLike) -> bool:
        ...

    @abstractmethod
    def astype(self, a: ArrayLike, dtype: DType) -> ArrayLike:
        ...

    @abstractmethod
    def default_float_dtype(self) -> DType:
        ...

    @abstractmethod
    def add(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def sub(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def mul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def div(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def neg(self, a: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def pow(self, a: ArrayLike, exp: Union[ArrayLike, float]) -> ArrayLike:
        ...

    @abstractmethod
    def exp(self, a: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def log(self, a: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def abs(self, a: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def sqrt(self, a: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def clip(self, a: ArrayLike, a_min: Optional[float], a_max: Optional[float]) -> ArrayLike:
        ...

    @abstractmethod
    def maximum(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def minimum(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def sign(self, a: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def tanh(self, a: ArrayLike) -> ArrayLike:
        ...


    @abstractmethod
    def eq(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def ne(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def gt(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def ge(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def lt(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def le(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def where(self, condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def sum(self, a: ArrayLike, axis: Optional[Union[int, Tuple[int, ...]]] = None,
            keepdims: bool = False) -> ArrayLike:
        ...

    @abstractmethod
    def mean(self, a: ArrayLike, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> ArrayLike:
        ...

    @abstractmethod
    def max(self, a: ArrayLike, axis: Optional[int] = None,
            keepdims: bool = False) -> ArrayLike:
        ...

    @abstractmethod
    def min(self, a: ArrayLike, axis: Optional[int] = None,
            keepdims: bool = False) -> ArrayLike:
        ...

    @abstractmethod
    def argmax(self, a: ArrayLike, axis: Optional[int] = None) -> ArrayLike:
        ...

    @abstractmethod
    def argmin(self, a: ArrayLike, axis: Optional[int] = None) -> ArrayLike:
        ...

    @abstractmethod
    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def reshape(self, a: ArrayLike, shape: Shape) -> ArrayLike:
        ...

    @abstractmethod
    def transpose(self, a: ArrayLike, axes: Optional[Sequence[int]] = None) -> ArrayLike:
        ...

    @abstractmethod
    def swapaxes(self, a: ArrayLike, axis1: int, axis2: int) -> ArrayLike:
        ...

    @abstractmethod
    def expand_dims(self, a: ArrayLike, axis: int) -> ArrayLike:
        ...

    @abstractmethod
    def squeeze(self, a: ArrayLike, axis: Optional[int] = None) -> ArrayLike:
        ...

    @abstractmethod
    def broadcast_to(self, a: ArrayLike, shape: Shape) -> ArrayLike:
        ...

    @abstractmethod
    def concatenate(self, tensors: Sequence[ArrayLike], axis: int = 0) -> ArrayLike:
        ...

    @abstractmethod
    def stack(self, tensors: Sequence[ArrayLike], axis: int = 0) -> ArrayLike:
        ...

    @abstractmethod
    def split(self, a: ArrayLike, indices_or_sections: Union[int, Sequence[int]],
              axis: int = 0) -> List[ArrayLike]:
        ...

    @abstractmethod
    def flip(self, a: ArrayLike, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> ArrayLike:
        ...


    #Dekho, padding and slcing model ne implemnt kiya hai, so be carefuol
    @abstractmethod
    def pad(self, a: ArrayLike, pad_width: Sequence[Tuple[int, int]],
            mode: str = "constant", constant_values: float = 0.0) -> ArrayLike:
        ...

    @abstractmethod
    def slice_along(self, a: ArrayLike, slices: Tuple) -> ArrayLike:
        """Fancy / basic indexing — equivalent to ``a[slices]``."""
        ...

    @abstractmethod
    def set_slice(self, a: ArrayLike, slices: Tuple, value: ArrayLike) -> ArrayLike:
        """Return a copy slice with val applied.
        """
        ...

    @abstractmethod
    def scatter_add(self, a: ArrayLike, axis: int, index: ArrayLike,
                    src: ArrayLike) -> ArrayLike:
        """``a[index] += src`` along *axis*.  Returns the updated array."""
        ...

    @abstractmethod
    def seed(self, n: int) -> None:
        ...

    @abstractmethod
    def randn(self, shape: Shape, dtype: DType | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def rand(self, shape: Shape, dtype: DType | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def randint(self, low: int, high: int, shape: Shape) -> ArrayLike:
        ...

    @abstractmethod
    def dropout_mask(self, shape: Shape, p: float, dtype: DType | None = None) -> ArrayLike:
        """Return a mask of 0s and 1s where each element is 0 with probability *p*."""
        ...

    @abstractmethod
    def randperm(self, n: int) -> ArrayLike:
        ...


    @abstractmethod
    def contiguous(self, a: ArrayLike) -> ArrayLike:
        ...


    # @abstractmethod
    # def im2col(self, a: ArrayLike, kernel_h: int, kernel_w: int,
    #            stride: int = 1, padding: int = 0, dilation: int = 1) -> ArrayLike:
    #     """Extract sliding local blocks from a batched 4-D image tensor.

    #     Parameters
    #     ----------
    #     a : (N, C, H, W)
    #     Returns col : (N, C*kH*kW, out_H*out_W)
    #     """
    #     ...

    # @abstractmethod
    # def col2im(self, col: ArrayLike, input_shape: Shape,
    #            kernel_h: int, kernel_w: int,
    #            stride: int = 1, padding: int = 0, dilation: int = 1) -> ArrayLike:
    #     """Inverse of ``im2col``."""
    #     ...
