"""
picograd/ops/reduce.py
=======================
Reduction operations: Sum, Mean, Max, Min.

Backward for reductions requires expanding/broadcasting the
incoming gradient back to the input shape.
"""

from __future__ import annotations
import numpy as np
from picograd.autograd.function import Function, Context
from picograd.backend import get_backend


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a, axis, keepdims):
        b = get_backend()
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return b.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad):
        a, = ctx.saved_tensors
        b = get_backend()
        # Expand grad back to input shape
        g = grad
        if not ctx.keepdims and ctx.axis is not None:
            axes = (ctx.axis,) if isinstance(ctx.axis, int) else ctx.axis
            for ax in sorted(axes):
                g = b.unsqueeze(g, ax)
        return (b.expand(g, b.shape_of(a)),)


class Mean(Function):
    @staticmethod
    def forward(ctx: Context, a, axis, keepdims):
        b = get_backend()
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        out = b.mean(a, axis=axis, keepdims=keepdims)
        # Compute the number of elements averaged over
        if axis is None:
            ctx.n = a.size
        elif isinstance(axis, int):
            ctx.n = a.shape[axis]
        else:
            ctx.n = int(np.prod([a.shape[i] for i in axis]))
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        a, = ctx.saved_tensors
        b = get_backend()
        g = grad
        if not ctx.keepdims and ctx.axis is not None:
            axes = (ctx.axis,) if isinstance(ctx.axis, int) else ctx.axis
            for ax in sorted(axes):
                g = b.unsqueeze(g, ax)
        g = b.expand(g, b.shape_of(a))
        # Divide by n (mean divides by n in forward)
        return (b.div(g, b.full(b.shape_of(g), float(ctx.n))),)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a, axis, keepdims):
        b = get_backend()
        out = b.max(a, axis=axis, keepdims=keepdims)
        ctx.save_for_backward(a)
        ctx.out = out
        ctx.axis = axis
        ctx.keepdims = keepdims
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        a, = ctx.saved_tensors
        b = get_backend()
        out = ctx.out
        g = grad

        # Expand out back for comparison
        if not ctx.keepdims and ctx.axis is not None:
            axes = (ctx.axis,) if isinstance(ctx.axis, int) else ctx.axis
            for ax in sorted(axes):
                out = b.unsqueeze(out, ax)
                g = b.unsqueeze(g, ax)

        out_expanded = b.expand(out, b.shape_of(a))
        g_expanded = b.expand(g, b.shape_of(a))

        # Mask: 1 where a == max, else 0
        mask = b.cast(b.eq(a, out_expanded), np.float32)
        # Normalize over ties
        count = b.sum(mask, axis=ctx.axis, keepdims=True)
        count = b.expand(count, b.shape_of(a))
        mask = b.div(mask, count)

        return (b.mul(g_expanded, mask),)


class Min_op(Function):
    @staticmethod
    def forward(ctx: Context, a, axis, keepdims):
        b = get_backend()
        out = b.min(a, axis=axis, keepdims=keepdims)
        ctx.save_for_backward(a)
        ctx.out = out
        ctx.axis = axis
        ctx.keepdims = keepdims
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        a, = ctx.saved_tensors
        b = get_backend()
        out = ctx.out
        g = grad

        if not ctx.keepdims and ctx.axis is not None:
            axes = (ctx.axis,) if isinstance(ctx.axis, int) else ctx.axis
            for ax in sorted(axes):
                out = b.unsqueeze(out, ax)
                g = b.unsqueeze(g, ax)

        out_expanded = b.expand(out, b.shape_of(a))
        g_expanded = b.expand(g, b.shape_of(a))

        mask = b.cast(b.eq(a, out_expanded), np.float32)
        count = b.sum(mask, axis=ctx.axis, keepdims=True)
        count = b.expand(count, b.shape_of(a))
        mask = b.div(mask, count)

        return (b.mul(g_expanded, mask),)


__all__ = ["Sum", "Mean", "Max", "Min_op"]
