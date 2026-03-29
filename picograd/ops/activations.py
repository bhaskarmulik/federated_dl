"""
picograd/ops/activations.py
============================
Differentiable activation functions.
Each is a Function subclass with correct forward + backward.
"""

from __future__ import annotations
import numpy as np
from picograd.autograd.function import Function, Context
from picograd.backend import get_backend


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a):
        b = get_backend()
        mask = b.gt(a, b.zeros(b.shape_of(a)))
        ctx.save_for_backward(mask)
        return b.where(mask, a, b.zeros(b.shape_of(a)))

    @staticmethod
    def backward(ctx: Context, grad):
        mask, = ctx.saved_tensors
        b = get_backend()
        return (b.mul(grad, b.cast(mask, np.float32)),)


class LeakyReLU(Function):
    @staticmethod
    def forward(ctx: Context, a, negative_slope):
        b = get_backend()
        mask = b.gt(a, b.zeros(b.shape_of(a)))
        ctx.save_for_backward(mask)
        ctx.negative_slope = negative_slope
        return b.where(mask, a, b.mul(b.full(b.shape_of(a), negative_slope), a))

    @staticmethod
    def backward(ctx: Context, grad):
        mask, = ctx.saved_tensors
        b = get_backend()
        slope = b.where(mask,
                        b.ones(b.shape_of(mask)),
                        b.full(b.shape_of(mask), ctx.negative_slope))
        return (b.mul(grad, b.cast(slope, np.float32)),)

    @classmethod
    def apply(cls, tensor, negative_slope=0.01):
        from picograd.tensor import Tensor, _ensure_tensor
        from picograd.autograd.context import _grad_enabled
        from picograd.autograd.function import Node, Context
        needs_grad = _grad_enabled() and tensor.requires_grad
        ctx = Context()
        raw = cls.forward(ctx, tensor._data, negative_slope)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            node = Node(cls, ctx, [(tensor, tensor._grad_fn)])
            out._grad_fn = node
            out._is_leaf = False
        return out


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a):
        b = get_backend()
        out = b.div(b.ones(b.shape_of(a)),
                    b.add(b.ones(b.shape_of(a)), b.exp(b.neg(a))))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        out, = ctx.saved_tensors
        b = get_backend()
        # σ(1 - σ)
        return (b.mul(grad, b.mul(out, b.sub(b.ones(b.shape_of(out)), out))),)


class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, a):
        b = get_backend()
        # tanh via exp: (e^a - e^-a)/(e^a + e^-a)
        ea  = b.exp(a)
        ena = b.exp(b.neg(a))
        out = b.div(b.sub(ea, ena), b.add(ea, ena))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        out, = ctx.saved_tensors
        b = get_backend()
        # 1 - tanh^2
        return (b.mul(grad, b.sub(b.ones(b.shape_of(out)), b.mul(out, out))),)


class GELU(Function):
    """Approximate GELU: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))"""
    _SQRT2OVERPI = np.sqrt(2.0 / np.pi).astype(np.float32)
    _COEFF = np.float32(0.044715)

    @staticmethod
    def forward(ctx: Context, a):
        b = get_backend()
        c = GELU._SQRT2OVERPI
        k = GELU._COEFF
        # inner = c * (a + k * a^3)
        inner = b.mul(b.full(b.shape_of(a), float(c)),
                      b.add(a, b.mul(b.full(b.shape_of(a), float(k)), b.pow(a, 3))))
        # tanh(inner)
        t = b.div(b.sub(b.exp(inner), b.exp(b.neg(inner))),
                  b.add(b.exp(inner), b.exp(b.neg(inner))))
        # out = 0.5 * a * (1 + t)
        out = b.mul(b.full(b.shape_of(a), 0.5),
                    b.mul(a, b.add(b.ones(b.shape_of(a)), t)))
        ctx.save_for_backward(a)
        ctx.t = t
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        a, = ctx.saved_tensors
        b = get_backend()
        c = GELU._SQRT2OVERPI
        k = GELU._COEFF
        t = ctx.t

        # d(tanh)/dx = c*(1 + 3k*x^2)*(1 - tanh^2)
        dtanh = b.mul(
            b.mul(b.full(b.shape_of(a), float(c)),
                  b.add(b.ones(b.shape_of(a)),
                        b.mul(b.full(b.shape_of(a), float(3*k)), b.pow(a, 2)))),
            b.sub(b.ones(b.shape_of(a)), b.mul(t, t))
        )
        # d(gelu)/dx = 0.5*(1 + t) + 0.5*a*dtanh
        dgelu = b.add(
            b.mul(b.full(b.shape_of(a), 0.5), b.add(b.ones(b.shape_of(a)), t)),
            b.mul(b.full(b.shape_of(a), 0.5), b.mul(a, dtanh))
        )
        return (b.mul(grad, dgelu),)


class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, a, axis):
        b = get_backend()
        # Numerically stable: subtract max
        mx = b.max(a, axis=axis, keepdims=True)
        shifted = b.sub(a, b.expand(mx, b.shape_of(a)))
        e = b.exp(shifted)
        s = b.sum(e, axis=axis, keepdims=True)
        out = b.div(e, b.expand(s, b.shape_of(e)))
        ctx.save_for_backward(out)
        ctx.axis = axis
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        out, = ctx.saved_tensors
        b = get_backend()
        axis = ctx.axis
        # Jacobian-vector product: (grad - sum(grad*out, axis)) * out
        dot = b.sum(b.mul(grad, out), axis=axis, keepdims=True)
        dot = b.expand(dot, b.shape_of(out))
        return (b.mul(out, b.sub(grad, dot)),)

    @classmethod
    def apply(cls, tensor, axis=-1):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled
        from picograd.autograd.function import Node, Context
        needs_grad = _grad_enabled() and tensor.requires_grad
        ctx = Context()
        raw = cls.forward(ctx, tensor._data, axis)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            node = Node(cls, ctx, [(tensor, tensor._grad_fn)])
            out._grad_fn = node
            out._is_leaf = False
        return out


class LogSoftmax(Function):
    """Numerically stable log-softmax. Used in CrossEntropyLoss."""
    @staticmethod
    def forward(ctx: Context, a, axis):
        b = get_backend()
        mx = b.max(a, axis=axis, keepdims=True)
        shifted = b.sub(a, b.expand(mx, b.shape_of(a)))
        log_sum_exp = b.log(b.sum(b.exp(shifted), axis=axis, keepdims=True))
        out = b.sub(shifted, b.expand(log_sum_exp, b.shape_of(shifted)))
        ctx.save_for_backward(out)
        ctx.axis = axis
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        out, = ctx.saved_tensors
        b = get_backend()
        axis = ctx.axis
        # grad - softmax * sum(grad, axis)
        softmax = b.exp(out)
        grad_sum = b.sum(grad, axis=axis, keepdims=True)
        grad_sum = b.expand(grad_sum, b.shape_of(grad))
        return (b.sub(grad, b.mul(softmax, grad_sum)),)

    @classmethod
    def apply(cls, tensor, axis=-1):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled
        from picograd.autograd.function import Node, Context
        needs_grad = _grad_enabled() and tensor.requires_grad
        ctx = Context()
        raw = cls.forward(ctx, tensor._data, axis)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            node = Node(cls, ctx, [(tensor, tensor._grad_fn)])
            out._grad_fn = node
            out._is_leaf = False
        return out


__all__ = ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU", "Softmax", "LogSoftmax"]
