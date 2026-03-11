"""Reduction operations: Sum, Mean, Max, Min."""

from __future__ import annotations

from ..autograd.function import Context, Function
from ..backend import get_backend

__all__ = ["Sum", "Mean", "Max", "Min"]


class Sum(Function):
    @staticmethod
    def forward(ctx, a, *, axis=None, keepdims=False):
        from ..tensor import Tensor
        B = get_backend()
        ctx.a_shape = B.shape_of(a._data)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return Tensor(B.sum(a._data, axis=axis, keepdims=keepdims), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        g = grad_output._data
        # Restore reduced dimensions so broadcast works
        if ctx.axis is not None and not ctx.keepdims:
            axes = (ctx.axis,) if isinstance(ctx.axis, int) else tuple(ctx.axis)
            for ax in sorted(axes):
                g = B.expand_dims(g, ax)
        return (Tensor(B.broadcast_to(g, ctx.a_shape).copy(), _backend=B),)


class Mean(Function):
    @staticmethod
    def forward(ctx, a, *, axis=None, keepdims=False):
        from ..tensor import Tensor
        B = get_backend()
        ctx.a_shape = B.shape_of(a._data)
        ctx.axis = axis
        ctx.keepdims = keepdims
        # Compute the count of elements in the reduced dims
        shape = ctx.a_shape
        if axis is None:
            ctx.count = B.numel(a._data)
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            ctx.count = 1
            for ax in axes:
                ctx.count *= shape[ax]
        return Tensor(B.mean(a._data, axis=axis, keepdims=keepdims), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        g = grad_output._data
        if ctx.axis is not None and not ctx.keepdims:
            axes = (ctx.axis,) if isinstance(ctx.axis, int) else tuple(ctx.axis)
            for ax in sorted(axes):
                g = B.expand_dims(g, ax)
        g = B.div(B.broadcast_to(g, ctx.a_shape), B.array(ctx.count))
        return (Tensor(g.copy(), _backend=B),)


class Max(Function):
    @staticmethod
    def forward(ctx, a, *, axis=None, keepdims=False):
        from ..tensor import Tensor
        B = get_backend()
        out = B.max(a._data, axis=axis, keepdims=keepdims)
        # Mask of positions equal to the max (for gradient routing)
        if axis is not None and not keepdims:
            max_expanded = B.expand_dims(out, axis)
        elif axis is None:
            max_expanded = out
        else:
            max_expanded = out
        ctx.mask = B.eq(a._data, B.broadcast_to(max_expanded, B.shape_of(a._data)))
        ctx.a_shape = B.shape_of(a._data)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return Tensor(out, _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        g = grad_output._data
        if ctx.axis is not None and not ctx.keepdims:
            g = B.expand_dims(g, ctx.axis)
        g = B.broadcast_to(g, ctx.a_shape)
        # Distribute gradient to max positions; divide to handle ties
        mask = B.astype(ctx.mask, B.default_float_dtype())
        denom = B.sum(mask, axis=ctx.axis, keepdims=True)
        denom = B.maximum(denom, B.ones((), dtype=B.default_float_dtype()))
        result = B.div(B.mul(g, mask), denom)
        return (Tensor(result, _backend=B),)


class Min(Function):
    @staticmethod
    def forward(ctx, a, *, axis=None, keepdims=False):
        from ..tensor import Tensor
        B = get_backend()
        out = B.min(a._data, axis=axis, keepdims=keepdims)
        if axis is not None and not keepdims:
            min_expanded = B.expand_dims(out, axis)
        elif axis is None:
            min_expanded = out
        else:
            min_expanded = out
        ctx.mask = B.eq(a._data, B.broadcast_to(min_expanded, B.shape_of(a._data)))
        ctx.a_shape = B.shape_of(a._data)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return Tensor(out, _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        g = grad_output._data
        if ctx.axis is not None and not ctx.keepdims:
            g = B.expand_dims(g, ctx.axis)
        g = B.broadcast_to(g, ctx.a_shape)
        mask = B.astype(ctx.mask, B.default_float_dtype())
        denom = B.sum(mask, axis=ctx.axis, keepdims=True)
        denom = B.maximum(denom, B.ones((), dtype=B.default_float_dtype()))
        result = B.div(B.mul(g, mask), denom)
        return (Tensor(result, _backend=B),)
