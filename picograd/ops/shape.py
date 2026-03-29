"""Shape-manipulation ops: Reshape, Transpose, Squeeze, Unsqueeze, Expand, Slice, Cat, Stack."""

from __future__ import annotations

from ..autograd.function import Context, Function
from ..backend import get_backend

__all__ = [
    "Reshape", "Transpose", "Squeeze", "Unsqueeze",
    "Expand", "Slice", "Cat", "Stack",
]

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, *, shape):
        from ..tensor import Tensor
        B = get_backend()
        ctx.original_shape = B.shape_of(a._data)
        return Tensor(B.reshape(a._data, shape), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        return (Tensor(B.reshape(grad_output._data, ctx.original_shape), _backend=B),)

class Transpose(Function):
    @staticmethod
    def forward(ctx, a, *, axes=None):
        from ..tensor import Tensor
        B = get_backend()
        ctx.axes = axes
        ctx.a_ndim = B.ndim(a._data)
        return Tensor(B.transpose(a._data, axes), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        if ctx.axes is None:
            inv = None  # transpose reversal of full reverse is the same
        else:
            # invert the permutation
            inv = [0] * len(ctx.axes)
            for i, ax in enumerate(ctx.axes):
                inv[ax] = i
            inv = tuple(inv)
        return (Tensor(B.transpose(grad_output._data, inv), _backend=B),)

class Squeeze(Function):
    @staticmethod
    def forward(ctx, a, *, dim=None):
        from ..tensor import Tensor
        B = get_backend()
        ctx.original_shape = B.shape_of(a._data)
        return Tensor(B.squeeze(a._data, axis=dim), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        return (Tensor(B.reshape(grad_output._data, ctx.original_shape), _backend=B),)

class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, a, *, dim):
        from ..tensor import Tensor
        B = get_backend()
        ctx.original_shape = B.shape_of(a._data)
        return Tensor(B.expand_dims(a._data, axis=dim), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        return (Tensor(B.reshape(grad_output._data, ctx.original_shape), _backend=B),)


class Expand(Function):
    @staticmethod
    def forward(ctx, a, *, shape):
        from ..tensor import Tensor
        B = get_backend()
        ctx.original_shape = B.shape_of(a._data)
        return Tensor(B.broadcast_to(a._data, tuple(shape)).copy(), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        from .elemwise import _unbroadcast
        g = _unbroadcast(grad_output._data, ctx.original_shape)
        return (Tensor(g, _backend=B),)


class Slice(Function):
    @staticmethod
    def forward(ctx, a, *, key):
        from ..tensor import Tensor
        B = get_backend()
        ctx.original_shape = B.shape_of(a._data)
        ctx.key = key
        return Tensor(B.slice_along(a._data, key).copy(), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        g = B.zeros(ctx.original_shape, dtype=B.dtype_of(grad_output._data))
        g = B.set_slice(g, ctx.key, grad_output._data)
        return (Tensor(g, _backend=B),)


class Cat(Function):
    @staticmethod
    def forward(ctx, *tensors, dim=0):
        from ..tensor import Tensor
        B = get_backend()
        ctx.dim = dim
        ctx.split_sizes = [B.shape_of(t._data)[dim] for t in tensors]
        arrays = [t._data for t in tensors]
        return Tensor(B.concatenate(arrays, axis=dim), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        import numpy as np
        indices = list(np.cumsum(ctx.split_sizes)[:-1])
        parts = B.split(grad_output._data, indices, axis=ctx.dim)
        return tuple(Tensor(p, _backend=B) for p in parts)


class Stack(Function):
    @staticmethod
    def forward(ctx, *tensors, dim=0):
        from ..tensor import Tensor
        B = get_backend()
        ctx.dim = dim
        ctx.n = len(tensors)
        arrays = [t._data for t in tensors]
        return Tensor(B.stack(arrays, axis=dim), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        # Unstack: split along the dim, then squeeze that dim out
        parts = B.split(grad_output._data, ctx.n, axis=ctx.dim)
        return tuple(Tensor(B.squeeze(p, axis=ctx.dim), _backend=B) for p in parts)
