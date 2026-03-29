"""
picograd/ops/dropout.py
picograd/ops/embedding.py
"""

from __future__ import annotations
import numpy as np
from picograd.autograd.function import Function, Context, Node
from picograd.backend import get_backend


class Dropout(Function):
    """Inverted dropout: scale kept neurons by 1/(1-p) so eval is identity."""

    @staticmethod
    def forward(ctx: Context, x, p, training):
        b = get_backend()
        if not training or p == 0.0:
            ctx.mask = None
            return x
        mask = b.dropout_mask(b.shape_of(x), p)
        ctx.mask = b.cast(mask, np.float32)
        ctx.p = p
        scale = 1.0 / (1.0 - p)
        return b.mul(b.mul(x, ctx.mask), b.full(b.shape_of(x), scale))

    @staticmethod
    def backward(ctx: Context, grad):
        b = get_backend()
        if ctx.mask is None:
            return (grad,)
        scale = 1.0 / (1.0 - ctx.p)
        return (b.mul(b.mul(grad, ctx.mask), b.full(b.shape_of(grad), scale)),)

    @classmethod
    def apply(cls, tensor, p=0.5, training=True):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled

        needs_grad = _grad_enabled() and tensor.requires_grad
        ctx = Context()
        raw = cls.forward(ctx, tensor._data, p, training)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            node = Node(cls, ctx, [(tensor, tensor._grad_fn)])
            out._grad_fn = node
            out._is_leaf = False
        return out


class Embedding(Function):
    """Learnable embedding table lookup. Backward: scatter-add."""

    @staticmethod
    def forward(ctx: Context, weight, indices):
        # weight: (vocab_size, embed_dim), indices: integer array (any shape)
        ctx.save_for_backward(weight)
        ctx.indices = indices
        ctx.vocab_size = weight.shape[0]
        return weight[indices]   # numpy fancy indexing

    @staticmethod
    def backward(ctx: Context, grad):
        weight, = ctx.saved_tensors
        b = get_backend()
        grad_weight = np.zeros_like(weight)
        np.add.at(grad_weight, ctx.indices, grad)
        return (grad_weight, None)   # None for indices (not differentiable)

    @classmethod
    def apply(cls, weight_t, indices_t):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled

        needs_grad = _grad_enabled() and weight_t.requires_grad
        ctx = Context()
        raw = cls.forward(ctx, weight_t._data, indices_t._data.astype(np.int32))
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            node = Node(cls, ctx, [
                (weight_t, weight_t._grad_fn),
                (indices_t, None),
            ])
            out._grad_fn = node
            out._is_leaf = False
        return out


__all__ = ["Dropout", "Embedding"]
