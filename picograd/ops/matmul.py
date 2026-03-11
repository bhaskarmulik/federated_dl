"""Matrix multiplication (2-D and batched)."""

from __future__ import annotations

from ..autograd.function import Context, Function
from ..backend import get_backend

__all__ = ["MatMul"]


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        from ..tensor import Tensor
        B = get_backend()
        ctx.save_for_backward(a, b)
        return Tensor(B.matmul(a._data, b._data), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        """grad_A = grad @ B^T,  grad_B = A^T @ grad.

        For batched matmul (>=3-D), the batch dims are just broadcast.
        """
        from ..tensor import Tensor
        a, b = ctx.saved_tensors
        B = get_backend()
        g = grad_output._data


        b_t = _swap_last_two(B, b._data)
        ga = B.matmul(g, b_t)

        a_t = _swap_last_two(B, a._data)
        gb = B.matmul(a_t, g)

        ga = _unbroadcast_batch(B, ga, B.shape_of(a._data))
        gb = _unbroadcast_batch(B, gb, B.shape_of(b._data))

        return Tensor(ga, _backend=B), Tensor(gb, _backend=B)


def _swap_last_two(B, a):
    ndim = B.ndim(a)
    if ndim < 2:
        return a
    axes = list(range(ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return B.transpose(a, axes)

#Broadcast but over batch
def _unbroadcast_batch(B, grad, target_shape):
    """Sum over batch dimensions that expanded during matmul broadcast."""
    g_shape = B.shape_of(grad)
    t_ndim = len(target_shape)
    g_ndim = len(g_shape)

    # Sum over extra leading dims
    while len(B.shape_of(grad)) > t_ndim:
        grad = B.sum(grad, axis=0)

    # Sum over size-1 batch dims
    g_shape = B.shape_of(grad)
    axes = []
    for i in range(len(g_shape) - 2):  # skip the last two (matrix dims)
        if i < len(target_shape) - 2 and target_shape[i] == 1 and g_shape[i] != 1:
            axes.append(i)
    if axes:
        grad = B.sum(grad, axis=tuple(axes), keepdims=True)
    return grad
