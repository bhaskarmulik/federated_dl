"""
picograd/ops/elemwise.py
=========================
All elementwise differentiable operations.

Every class is a Function subclass with:
  - forward(ctx, *raw_arrays) → raw_array
  - backward(ctx, grad_out)   → tuple of grad arrays

Broadcasting: when two tensors broadcast, the backward must sum the
gradient over the added/expanded dimensions.  `_unbroadcast` handles this.
"""

from __future__ import annotations
import numpy as np
from picograd.autograd.function import Function, Context
from picograd.backend import get_backend


# ---------------------------------------------------------------------------
# Broadcast helper
# ---------------------------------------------------------------------------

def _unbroadcast(grad: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Sum grad back to original_shape, reversing NumPy broadcasting rules.

    Broadcasting can:
      a) Prepend dimensions:  (3,) + (2,3) → grad has shape (2,3), need (3,)
      b) Expand size-1 dims:  (3,1) + (3,4) → grad has shape (3,4), need (3,1)

    Strategy:
      1. Sum over leading dims that were prepended.
      2. Sum over size-1 dims that were expanded, keeping dims.
      3. Reshape to original_shape.
    """
    b = get_backend()
    out = grad

    # (1) Sum over extra leading dimensions
    while out.ndim > len(original_shape):
        out = b.sum(out, axis=0)

    # (2) Sum over size-1 dims that were broadcast-expanded
    for i, (o_s, g_s) in enumerate(zip(original_shape, out.shape)):
        if o_s == 1 and g_s != 1:
            out = b.sum(out, axis=i, keepdims=True)

    return b.reshape(out, original_shape)


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------

class Add(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return get_backend().add(a, b)

    @staticmethod
    def backward(ctx: Context, grad):
        a, b = ctx.saved_tensors
        b_obj = get_backend()
        ga = _unbroadcast(grad, b_obj.shape_of(a))
        gb = _unbroadcast(grad, b_obj.shape_of(b))
        return ga, gb


# ---------------------------------------------------------------------------
# Sub
# ---------------------------------------------------------------------------

class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return get_backend().sub(a, b)

    @staticmethod
    def backward(ctx: Context, grad):
        a, b = ctx.saved_tensors
        b_obj = get_backend()
        ga = _unbroadcast(grad, b_obj.shape_of(a))
        gb = _unbroadcast(b_obj.neg(grad), b_obj.shape_of(b))
        return ga, gb


# ---------------------------------------------------------------------------
# Mul  (element-wise product)
# ---------------------------------------------------------------------------

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return get_backend().mul(a, b)

    @staticmethod
    def backward(ctx: Context, grad):
        a, b = ctx.saved_tensors
        b_obj = get_backend()
        ga = _unbroadcast(b_obj.mul(grad, b), b_obj.shape_of(a))
        gb = _unbroadcast(b_obj.mul(grad, a), b_obj.shape_of(b))
        return ga, gb


# ---------------------------------------------------------------------------
# Div  (a / b)
# ---------------------------------------------------------------------------

class Div(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return get_backend().div(a, b)

    @staticmethod
    def backward(ctx: Context, grad):
        a, b = ctx.saved_tensors
        b_obj = get_backend()
        # da = grad / b
        # db = -grad * a / b^2
        ga = _unbroadcast(b_obj.div(grad, b), b_obj.shape_of(a))
        gb_raw = b_obj.neg(b_obj.div(b_obj.mul(grad, a), b_obj.mul(b, b)))
        gb = _unbroadcast(gb_raw, b_obj.shape_of(b))
        return ga, gb


# ---------------------------------------------------------------------------
# Neg  (-a)
# ---------------------------------------------------------------------------

class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a):
        return get_backend().neg(a)

    @staticmethod
    def backward(ctx: Context, grad):
        return (get_backend().neg(grad),)


# ---------------------------------------------------------------------------
# Pow  (a ** exp)
# ---------------------------------------------------------------------------

class Pow(Function):
    @staticmethod
    def forward(ctx: Context, a, exp):
        b = get_backend()
        out = b.pow(a, exp)
        ctx.save_for_backward(a)
        ctx.exp = exp
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        a, = ctx.saved_tensors
        exp = ctx.exp
        b = get_backend()
        # d/da [a^exp] = exp * a^(exp-1)
        ga = b.mul(grad, b.mul(b.full(b.shape_of(a), float(exp)), b.pow(a, exp - 1)))
        return (ga,)


# ---------------------------------------------------------------------------
# Exp  (e^a)
# ---------------------------------------------------------------------------

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a):
        b = get_backend()
        out = b.exp(a)
        ctx.save_for_backward(out)   # save output, not input
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        out, = ctx.saved_tensors
        b = get_backend()
        return (b.mul(grad, out),)


# ---------------------------------------------------------------------------
# Log  (ln a)
# ---------------------------------------------------------------------------

class Log(Function):
    @staticmethod
    def forward(ctx: Context, a):
        b = get_backend()
        ctx.save_for_backward(a)
        return b.log(a)

    @staticmethod
    def backward(ctx: Context, grad):
        a, = ctx.saved_tensors
        b = get_backend()
        return (b.div(grad, a),)


# ---------------------------------------------------------------------------
# Abs  (|a|)
# ---------------------------------------------------------------------------

class Abs(Function):
    @staticmethod
    def forward(ctx: Context, a):
        b = get_backend()
        ctx.save_for_backward(a)
        return b.abs(a)

    @staticmethod
    def backward(ctx: Context, grad):
        a, = ctx.saved_tensors
        b = get_backend()
        # d|a|/da = sign(a)  (0 at 0 by convention)
        sign = b.where(b.gt(a, b.zeros(b.shape_of(a))),
                       b.ones(b.shape_of(a)),
                       b.neg(b.ones(b.shape_of(a))))
        sign = b.where(b.eq(a, b.zeros(b.shape_of(a))),
                       b.zeros(b.shape_of(a)),
                       sign)
        return (b.mul(grad, sign),)


# ---------------------------------------------------------------------------
# Sqrt  (√a)
# ---------------------------------------------------------------------------

class Sqrt(Function):
    @staticmethod
    def forward(ctx: Context, a):
        b = get_backend()
        out = b.sqrt(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        out, = ctx.saved_tensors
        b = get_backend()
        # d√a/da = 1/(2√a)
        return (b.div(grad, b.mul(b.full(b.shape_of(out), 2.0), out)),)


# ---------------------------------------------------------------------------
# Clip  (clamp)
# ---------------------------------------------------------------------------

class Clip(Function):
    @staticmethod
    def forward(ctx: Context, a, a_min, a_max):
        b = get_backend()
        ctx.save_for_backward(a)
        ctx.a_min = a_min
        ctx.a_max = a_max
        return b.clip(a, a_min, a_max)

    @staticmethod
    def backward(ctx: Context, grad):
        a, = ctx.saved_tensors
        b = get_backend()
        # Gradient is 0 where input was clipped, 1 otherwise
        mask = b.mul(b.cast(b.gt(a, b.full(b.shape_of(a), float(ctx.a_min))), np.float32),
                     b.cast(b.lt(a, b.full(b.shape_of(a), float(ctx.a_max))), np.float32))
        return (b.mul(grad, mask),)


__all__ = ["Add", "Sub", "Mul", "Div", "Neg", "Pow", "Exp", "Log", "Abs", "Sqrt", "Clip",
           "_unbroadcast"]
