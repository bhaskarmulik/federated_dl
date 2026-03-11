"""Activation functions: ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Softmax, LogSoftmax."""

from __future__ import annotations

import math

from ..autograd.function import Context, Function
from ..backend import get_backend

__all__ = [
    "ReLU", "Sigmoid", "Tanh", "GELU", "LeakyReLU",
    "Softmax", "LogSoftmax",
]


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        from ..tensor import Tensor
        B = get_backend()
        mask = B.gt(a._data, B.zeros((), dtype=B.dtype_of(a._data)))
        ctx.mask = mask
        return Tensor(B.mul(a._data, B.astype(mask, B.dtype_of(a._data))), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        return (Tensor(B.mul(grad_output._data,
                             B.astype(ctx.mask, B.dtype_of(grad_output._data))),
                       _backend=B),)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        from ..tensor import Tensor
        B = get_backend()
        sig = B.div(B.ones(B.shape_of(a._data), dtype=B.dtype_of(a._data)),
                    B.add(B.ones(B.shape_of(a._data), dtype=B.dtype_of(a._data)),
                          B.exp(B.neg(a._data))))
        ctx.sig = sig
        return Tensor(sig, _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        s = ctx.sig
        ds = B.mul(s, B.sub(B.ones(B.shape_of(s), dtype=B.dtype_of(s)), s))
        return (Tensor(B.mul(grad_output._data, ds), _backend=B),)


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        from ..tensor import Tensor
        B = get_backend()
        t = B.tanh(a._data)
        ctx.tanh_val = t
        return Tensor(t, _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        # d/dx tanh(x) = 1 - tanh^2(x)
        dt = B.sub(B.ones(B.shape_of(ctx.tanh_val), dtype=B.dtype_of(ctx.tanh_val)),
                   B.pow(ctx.tanh_val, 2.0))
        return (Tensor(B.mul(grad_output._data, dt), _backend=B),)


_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


class GELU(Function):
    @staticmethod
    def forward(ctx, a):
        from ..tensor import Tensor
        B = get_backend()
        x = a._data
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        inner = B.mul(B.array(_SQRT_2_OVER_PI),
                      B.add(x, B.mul(B.array(0.044715), B.pow(x, 3.0))))
        t = B.tanh(inner)
        out = B.mul(B.mul(B.array(0.5), x), B.add(B.ones(B.shape_of(x), dtype=B.dtype_of(x)), t))
        ctx.save_for_backward(a)
        ctx.t = t
        ctx.inner = inner
        return Tensor(out, _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        (a,) = ctx.saved_tensors
        B = get_backend()
        x = a._data
        t = ctx.t
        sech2 = B.sub(B.ones(B.shape_of(t), dtype=B.dtype_of(t)), B.pow(t, 2.0))
        d_inner = B.mul(B.array(_SQRT_2_OVER_PI),
                        B.add(B.ones(B.shape_of(x), dtype=B.dtype_of(x)),
                              B.mul(B.array(3 * 0.044715), B.pow(x, 2.0))))
        # d/dx = 0.5*(1+t) + 0.5*x*sech2*d_inner
        dg = B.add(
            B.mul(B.array(0.5), B.add(B.ones(B.shape_of(t), dtype=B.dtype_of(t)), t)),
            B.mul(B.mul(B.array(0.5), x), B.mul(sech2, d_inner)),
        )
        return (Tensor(B.mul(grad_output._data, dg), _backend=B),)

class LeakyReLU(Function):
    @staticmethod
    def forward(ctx, a, *, negative_slope=0.01):
        from ..tensor import Tensor
        B = get_backend()
        ctx.negative_slope = negative_slope
        mask = B.gt(a._data, B.zeros((), dtype=B.dtype_of(a._data)))
        ctx.mask = mask
        pos = B.mul(a._data, B.astype(mask, B.dtype_of(a._data)))
        neg_mask = B.le(a._data, B.zeros((), dtype=B.dtype_of(a._data)))
        neg = B.mul(B.mul(a._data, B.astype(neg_mask, B.dtype_of(a._data))),
                    B.array(negative_slope))
        return Tensor(B.add(pos, neg), _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        mask_f = B.astype(ctx.mask, B.dtype_of(grad_output._data))
        neg_mask_f = B.sub(B.ones(B.shape_of(mask_f), dtype=B.dtype_of(mask_f)), mask_f)
        slope = B.add(mask_f, B.mul(neg_mask_f, B.array(ctx.negative_slope)))
        return (Tensor(B.mul(grad_output._data, slope), _backend=B),)


class Softmax(Function):
    @staticmethod
    def forward(ctx, a, *, dim=-1):
        from ..tensor import Tensor
        B = get_backend()
        x = a._data
        x_max = B.max(x, axis=dim, keepdims=True)
        e = B.exp(B.sub(x, x_max))
        s = B.div(e, B.sum(e, axis=dim, keepdims=True))
        ctx.s = s
        ctx.dim = dim
        return Tensor(s, _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        s = ctx.s
        g = grad_output._data
        # ds_i/dx_j = s_i*(delta_ij - s_j)
        # Efficient: g_out = s * (g - sum(g*s, dim, keepdims))
        sg = B.mul(s, g)
        result = B.mul(s, B.sub(g, B.sum(sg, axis=ctx.dim, keepdims=True)))
        return (Tensor(result, _backend=B),)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, a, *, dim=-1):
        from ..tensor import Tensor
        B = get_backend()
        x = a._data
        x_max = B.max(x, axis=dim, keepdims=True)
        shifted = B.sub(x, x_max)
        log_sum_exp = B.log(B.sum(B.exp(shifted), axis=dim, keepdims=True))
        out = B.sub(shifted, log_sum_exp)
        ctx.dim = dim
        ctx.softmax = B.exp(out)  # softmax values (for backward)
        return Tensor(out, _backend=B)

    @staticmethod
    def backward(ctx, grad_output):
        from ..tensor import Tensor
        B = get_backend()
        g = grad_output._data
        s = ctx.softmax
        # d/dx log_softmax = g - softmax * sum(g, dim, keepdims)
        result = B.sub(g, B.mul(s, B.sum(g, axis=ctx.dim, keepdims=True)))
        return (Tensor(result, _backend=B),)
