"""NumPy implementation of the ``Backend`` abstract interface.

Every method is a thin, stateless wrapper around NumPy / SciPy operations.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._base import ArrayLike, Backend, DType, Shape


class NumpyBackend(Backend):
    """Backend for NUMPI, later for TRITON if time allows"""

    @property
    def name(self) -> str:
        return "numpy"

    def empty(self, shape: Shape, dtype: DType | None = None) -> np.ndarray:
        return np.empty(shape, dtype=dtype or np.float32)

    def zeros(self, shape: Shape, dtype: DType | None = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype or np.float32)

    def ones(self, shape: Shape, dtype: DType | None = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype or np.float32)

    def full(self, shape: Shape, fill_value: float, dtype: DType | None = None) -> np.ndarray:
        return np.full(shape, fill_value, dtype=dtype or np.float32)

    def arange(self, start: float, stop: float, step: float = 1.0,
               dtype: DType | None = None) -> np.ndarray:
        return np.arange(start, stop, step, dtype=dtype or np.float32)

    def eye(self, n: int, dtype: DType | None = None) -> np.ndarray:
        return np.eye(n, dtype=dtype or np.float32)

    def from_numpy(self, arr: np.ndarray) -> np.ndarray:
        return arr  # already numpy

    def to_numpy(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(a)

    def array(self, data: Any, dtype: DType | None = None) -> np.ndarray:
        return np.array(data, dtype=dtype or np.float32)

    def copy(self, a: np.ndarray) -> np.ndarray:
        return a.copy()

    def shape_of(self, a: np.ndarray) -> Shape:
        return a.shape  # type: ignore[return-value]

    def dtype_of(self, a: np.ndarray) -> DType:
        return a.dtype

    def numel(self, a: np.ndarray) -> int:
        return int(a.size)

    def ndim(self, a: np.ndarray) -> int:
        return int(a.ndim)

    def is_floating_point(self, a: np.ndarray) -> bool:
        return np.issubdtype(a.dtype, np.floating)

    def astype(self, a: np.ndarray, dtype: DType) -> np.ndarray:
        return a.astype(dtype)

    def default_float_dtype(self) -> DType:
        return np.float32

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.add(a, b)

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.subtract(a, b)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.multiply(a, b)

    def div(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.divide(a, b)

    def neg(self, a: np.ndarray) -> np.ndarray:
        return np.negative(a)

    def pow(self, a: np.ndarray, exp: Union[np.ndarray, float]) -> np.ndarray:
        return np.power(a, exp)

    def exp(self, a: np.ndarray) -> np.ndarray:
        return np.exp(a)

    def log(self, a: np.ndarray) -> np.ndarray:
        return np.log(a)

    def abs(self, a: np.ndarray) -> np.ndarray:
        return np.abs(a)

    def sqrt(self, a: np.ndarray) -> np.ndarray:
        return np.sqrt(a)

    def clip(self, a: np.ndarray, a_min: Optional[float],
             a_max: Optional[float]) -> np.ndarray:
        return np.clip(a, a_min, a_max)

    def maximum(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(a, b)

    def minimum(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b)

    def sign(self, a: np.ndarray) -> np.ndarray:
        return np.sign(a)

    def tanh(self, a: np.ndarray) -> np.ndarray:
        return np.tanh(a)

    def eq(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.equal(a, b)

    def ne(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.not_equal(a, b)

    def gt(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.greater(a, b)

    def ge(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.greater_equal(a, b)

    def lt(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.less(a, b)

    def le(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.less_equal(a, b)

    def where(self, condition: np.ndarray, x: np.ndarray,
              y: np.ndarray) -> np.ndarray:
        return np.where(condition, x, y)

    def sum(self, a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.mean(a, axis=axis, keepdims=keepdims)

    def max(self, a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.max(a, axis=axis, keepdims=keepdims)

    def min(self, a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.min(a, axis=axis, keepdims=keepdims)

    def argmax(self, a: np.ndarray, axis=None) -> np.ndarray:
        return np.argmax(a, axis=axis)

    def argmin(self, a: np.ndarray, axis=None) -> np.ndarray:
        return np.argmin(a, axis=axis)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)


    def reshape(self, a: np.ndarray, shape: Shape) -> np.ndarray:
        return np.reshape(a, shape)

    def transpose(self, a: np.ndarray, axes=None) -> np.ndarray:
        return np.transpose(a, axes)

    def swapaxes(self, a: np.ndarray, axis1: int, axis2: int) -> np.ndarray:
        return np.swapaxes(a, axis1, axis2)

    def expand_dims(self, a: np.ndarray, axis: int) -> np.ndarray:
        return np.expand_dims(a, axis)

    def squeeze(self, a: np.ndarray, axis=None) -> np.ndarray:
        return np.squeeze(a, axis=axis)

    def broadcast_to(self, a: np.ndarray, shape: Shape) -> np.ndarray:
        return np.broadcast_to(a, shape)

    def concatenate(self, tensors: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.concatenate(tensors, axis=axis)

    def stack(self, tensors: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.stack(tensors, axis=axis)

    def split(self, a: np.ndarray, indices_or_sections, axis: int = 0) -> List[np.ndarray]:
        return list(np.split(a, indices_or_sections, axis=axis))

    def flip(self, a: np.ndarray, axis=None) -> np.ndarray:
        return np.flip(a, axis=axis)

    def pad(self, a: np.ndarray, pad_width: Sequence[Tuple[int, int]],
            mode: str = "constant", constant_values: float = 0.0) -> np.ndarray:
        return np.pad(a, pad_width, mode=mode, constant_values=(constant_values, constant_values))

    def slice_along(self, a: np.ndarray, slices: Tuple) -> np.ndarray:
        return a[slices]

    def set_slice(self, a: np.ndarray, slices: Tuple, value: np.ndarray) -> np.ndarray:
        out = a.copy()
        out[slices] = value
        return out

    def scatter_add(self, a: np.ndarray, axis: int, index: np.ndarray,
                    src: np.ndarray) -> np.ndarray:
        out = a.copy()
        np.add.at(out, index, src)
        return out


    def seed(self, n: int) -> None:
        np.random.seed(n)

    def randn(self, shape: Shape, dtype: DType | None = None) -> np.ndarray:
        return np.random.randn(*shape).astype(dtype or np.float32)

    def rand(self, shape: Shape, dtype: DType | None = None) -> np.ndarray:
        return np.random.rand(*shape).astype(dtype or np.float32)

    def randint(self, low: int, high: int, shape: Shape) -> np.ndarray:
        return np.random.randint(low, high, size=shape)

    def dropout_mask(self, shape: Shape, p: float,
                     dtype: DType | None = None) -> np.ndarray:
        return (np.random.rand(*shape) >= p).astype(dtype or np.float32)

    def randperm(self, n: int) -> np.ndarray:
        return np.random.permutation(n)


    def contiguous(self, a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(a)


    def im2col(self, a: np.ndarray, kernel_h: int, kernel_w: int,
               stride: int = 1, padding: int = 0, dilation: int = 1) -> np.ndarray:
        N, C, H, W = a.shape
        # effective kernel size with dilation
        kh_eff = kernel_h + (kernel_h - 1) * (dilation - 1)
        kw_eff = kernel_w + (kernel_w - 1) * (dilation - 1)

        if padding > 0:
            a = np.pad(a, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                       mode='constant')

        _, _, H_pad, W_pad = a.shape
        out_h = (H_pad - kh_eff) // stride + 1
        out_w = (W_pad - kw_eff) // stride + 1

        # Build strides for efficient extraction
        col = np.empty((N, C, kernel_h, kernel_w, out_h, out_w), dtype=a.dtype)
        for y in range(kernel_h):
            y_max = y * dilation + stride * out_h
            for x in range(kernel_w):
                x_max = x * dilation + stride * out_w
                col[:, :, y, x, :, :] = a[:, :,
                                           y * dilation:y_max:stride,
                                           x * dilation:x_max:stride]

        # (N, C*kH*kW, out_h*out_w)
        col = col.reshape(N, C * kernel_h * kernel_w, out_h * out_w)
        return col

    def col2im(self, col: np.ndarray, input_shape: Shape,
               kernel_h: int, kernel_w: int,
               stride: int = 1, padding: int = 0, dilation: int = 1) -> np.ndarray:
        N, C, H, W = input_shape
        H_pad = H + 2 * padding
        W_pad = W + 2 * padding

        kh_eff = kernel_h + (kernel_h - 1) * (dilation - 1)
        kw_eff = kernel_w + (kernel_w - 1) * (dilation - 1)
        out_h = (H_pad - kh_eff) // stride + 1
        out_w = (W_pad - kw_eff) // stride + 1

        col_reshaped = col.reshape(N, C, kernel_h, kernel_w, out_h, out_w)
        img = np.zeros((N, C, H_pad, W_pad), dtype=col.dtype)

        for y in range(kernel_h):
            y_max = y * dilation + stride * out_h
            for x in range(kernel_w):
                x_max = x * dilation + stride * out_w
                img[:, :,
                    y * dilation:y_max:stride,
                    x * dilation:x_max:stride] += col_reshaped[:, :, y, x, :, :]

        if padding > 0:
            img = img[:, :, padding:-padding, padding:-padding]
        return img
