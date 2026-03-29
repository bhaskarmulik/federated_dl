"""
picograd/ops/matmul.py
=======================
Matrix multiplication (2D and batched ND).

Gradients:
  grad_A = grad @ B^T
  grad_B = A^T @ grad

For batched (3D+): same rule applied to the last two dims, with
summing over leading batch dimensions if shapes differ (broadcasting).
"""

from __future__ import annotations
import numpy as np
from picograd.autograd.function import Function, Context
from picograd.backend import get_backend


def _batched_matmul(a, b):
    return get_backend().matmul(a, b)


def _t(x):
    """Swap last two dimensions of x (works for any nd >= 2)."""
    b = get_backend()
    ndim = len(b.shape_of(x))
    axes = list(range(ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return b.transpose(x, axes=axes)


def _sum_to(grad, target_shape):
    """
    Sum grad to target_shape (handles broadcast un-reduction for batched matmul).
    Works like _unbroadcast from elemwise but for any rank.
    """
    b = get_backend()
    g = grad
    g_shape = b.shape_of(g)
    t_shape = target_shape

    # Sum over extra leading dims
    while len(b.shape_of(g)) > len(t_shape):
        g = b.sum(g, axis=0)

    # Sum over size-1 dims that were broadcast-expanded
    for i, (ts, gs) in enumerate(zip(t_shape, b.shape_of(g))):
        if ts == 1 and gs != 1:
            g = b.sum(g, axis=i, keepdims=True)

    return b.reshape(g, t_shape)


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a, b_arr):
        b = get_backend()
        ctx.save_for_backward(a, b_arr)
        return b.matmul(a, b_arr)

    @staticmethod
    def backward(ctx: Context, grad):
        a, b_arr = ctx.saved_tensors
        b = get_backend()
        # grad_a = grad @ b^T,   grad_b = a^T @ grad
        grad_a = b.matmul(grad, _t(b_arr))
        grad_b = b.matmul(_t(a), grad)

        # Reduce to original shapes (batch broadcasting)
        grad_a = _sum_to(grad_a, b.shape_of(a))
        grad_b = _sum_to(grad_b, b.shape_of(b_arr))

        return grad_a, grad_b


__all__ = ["MatMul"]
