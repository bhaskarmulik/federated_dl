"""Element-wise differentiable operations: Add, Sub, Mul, Div, Neg, Pow, Exp, Log, Abs."""

from __future__ import annotations

from typing import Optional, Tuple

from ..autograd.function import Context, Function
from ..backend import get_backend

__all__ = ["Add", "Sub", "Mul", "Div", "Neg", "Pow", "Exp", "Log", "Abs"]


#Unboardcasting logic model ne likha hai cz I am incompetant
def _unbroadcast(grad_data, target_shape):
    """Sum-reduce *grad_data* so its shape matches *target_shape*.

    This reverses NumPy-style broadcasting that happened during the forward
    pass (e.g. adding a (3,1) tensor to a (3,4) tensor broadcasts the first
    to (3,4); the gradient must be summed back along axis 1).
    """
    B = get_backend()
    g_shape = B.shape_of(grad_data)
    t_ndim = len(target_shape)
    g_ndim = len(g_shape)

    # 1) sum over leading dims that were prepended by broadcasting
    if g_ndim > t_ndim:
        for _ in range(g_ndim - t_ndim):
            grad_data = B.sum(grad_data, axis=0)
        g_shape = B.shape_of(grad_data)

    # 2) sum over dims that were size-1 in target but expanded
    axes_to_reduce = []
    for i, (gs, ts) in enumerate(zip(g_shape, target_shape)):
        if ts == 1 and gs != 1:
            axes_to_reduce.append(i)
    if axes_to_reduce:
        grad_data = B.sum(grad_data, axis=tuple(axes_to_reduce), keepdims=True)

    return grad_data


class Add(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        from ..tensor import Tensor
        B = get_backend()
        ctx.a_shape = B.shape_of(a._data)
        ctx.b_shape = B.shape_of(b._data)
        return Tensor(B.add(a._data, b._data), _backend=B)

    @staticmethod
    def backward(ctx: Context, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        ga = _unbroadcast(grad_output._data, ctx.a_shape)
        gb = _unbroadcast(grad_output._data, ctx.b_shape)
        return Tensor(ga, _backend=B), Tensor(gb, _backend=B)


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        from ..tensor import Tensor
        B = get_backend()
        ctx.a_shape = B.shape_of(a._data)
        ctx.b_shape = B.shape_of(b._data)
        return Tensor(B.sub(a._data, b._data), _backend=B)

    @staticmethod
    def backward(ctx: Context, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        ga = _unbroadcast(grad_output._data, ctx.a_shape)
        gb = _unbroadcast(B.neg(grad_output._data), ctx.b_shape)
        return Tensor(ga, _backend=B), Tensor(gb, _backend=B)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        from ..tensor import Tensor
        B = get_backend()
        ctx.save_for_backward(a, b)
        return Tensor(B.mul(a._data, b._data), _backend=B)

    @staticmethod
    def backward(ctx: Context, grad_output):
        from ..tensor import Tensor
        a, b = ctx.saved_tensors
        B = get_backend()
        ga = _unbroadcast(B.mul(grad_output._data, b._data), B.shape_of(a._data))
        gb = _unbroadcast(B.mul(grad_output._data, a._data), B.shape_of(b._data))
        return Tensor(ga, _backend=B), Tensor(gb, _backend=B)


class Div(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        from ..tensor import Tensor
        B = get_backend()
        ctx.save_for_backward(a, b)
        return Tensor(B.div(a._data, b._data), _backend=B)

    @staticmethod
    def backward(ctx: Context, grad_output):
        from ..tensor import Tensor
        a, b = ctx.saved_tensors
        B = get_backend()
        ga = _unbroadcast(B.div(grad_output._data, b._data), B.shape_of(a._data))
        gb = _unbroadcast(
            B.neg(B.div(B.mul(grad_output._data, a._data),
                        B.pow(b._data, 2.0))),
            B.shape_of(b._data),
        )
        return Tensor(ga, _backend=B), Tensor(gb, _backend=B)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a):
        from ..tensor import Tensor
        B = get_backend()
        return Tensor(B.neg(a._data), _backend=B)

    @staticmethod
    def backward(ctx: Context, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        return (Tensor(B.neg(grad_output._data), _backend=B),)


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, a, exp):
        from ..tensor import Tensor
        B = get_backend()
        ctx.save_for_backward(a, exp)
        return Tensor(B.pow(a._data, exp._data), _backend=B)

    @staticmethod
    def backward(ctx: Context, grad_output):
        from ..tensor import Tensor
        a, exp = ctx.saved_tensors
        B = get_backend()
        # d/da(a^n) = n * a^(n-1)
        ga = B.mul(B.mul(exp._data, B.pow(a._data, B.sub(exp._data, B.ones((), dtype=B.dtype_of(exp._data))))),
                   grad_output._data)
        ga = _unbroadcast(ga, B.shape_of(a._data))
        # d/dn(a^n) = a^n * ln(a)  (only if exp requires grad)
        gn = B.mul(B.mul(B.pow(a._data, exp._data), B.log(B.clip(B.abs(a._data), 1e-12, None))),
                   grad_output._data)
        gn = _unbroadcast(gn, B.shape_of(exp._data))
        return Tensor(ga, _backend=B), Tensor(gn, _backend=B)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a):
        from ..tensor import Tensor
        B = get_backend()
        out = B.exp(a._data)
        ctx.out = out  # save output — exp's derivative is itself
        return Tensor(out, _backend=B)

    @staticmethod
    def backward(ctx: Context, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        return (Tensor(B.mul(ctx.out, grad_output._data), _backend=B),)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a):
        from ..tensor import Tensor
        B = get_backend()
        ctx.save_for_backward(a)
        return Tensor(B.log(a._data), _backend=B)

    @staticmethod
    def backward(ctx: Context, grad_output):
        from ..tensor import Tensor
        (a,) = ctx.saved_tensors
        B = get_backend()
        return (Tensor(B.div(grad_output._data, a._data), _backend=B),)


class Abs(Function):
    @staticmethod
    def forward(ctx: Context, a):
        from ..tensor import Tensor
        B = get_backend()
        ctx.save_for_backward(a)
        return Tensor(B.abs(a._data), _backend=B)

    @staticmethod
    def backward(ctx: Context, grad_output):
        from ..tensor import Tensor
        (a,) = ctx.saved_tensors
        B = get_backend()
        return (Tensor(B.mul(B.sign(a._data), grad_output._data), _backend=B),)
