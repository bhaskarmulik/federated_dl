"""
picograd/ops/shape.py
======================
Shape manipulation operations (all differentiable).
"""

from __future__ import annotations
import numpy as np
from picograd.autograd.function import Function, Context
from picograd.backend import get_backend


class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a, shape):
        b = get_backend()
        ctx.input_shape = b.shape_of(a)
        ctx.shape = shape
        return b.reshape(a, shape)

    @staticmethod
    def backward(ctx: Context, grad):
        return (get_backend().reshape(grad, ctx.input_shape),)


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a, axes):
        b = get_backend()
        ctx.axes = axes
        ctx.input_shape = b.shape_of(a)
        return b.transpose(a, axes=axes)

    @staticmethod
    def backward(ctx: Context, grad):
        b = get_backend()
        axes = ctx.axes
        if axes is None:
            return (b.transpose(grad, axes=None),)
        # Inverse permutation
        inv = [0] * len(axes)
        for i, ax in enumerate(axes):
            inv[ax] = i
        return (b.transpose(grad, axes=inv),)


class Squeeze(Function):
    @staticmethod
    def forward(ctx: Context, a, axis):
        b = get_backend()
        ctx.input_shape = b.shape_of(a)
        ctx.axis = axis
        return b.squeeze(a, axis=axis)

    @staticmethod
    def backward(ctx: Context, grad):
        b = get_backend()
        return (b.reshape(grad, ctx.input_shape),)


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx: Context, a, axis):
        b = get_backend()
        ctx.axis = axis
        ctx.input_shape = b.shape_of(a)
        return b.unsqueeze(a, axis)

    @staticmethod
    def backward(ctx: Context, grad):
        b = get_backend()
        return (b.reshape(grad, ctx.input_shape),)


class Expand(Function):
    @staticmethod
    def forward(ctx: Context, a, shape):
        b = get_backend()
        ctx.input_shape = b.shape_of(a)
        ctx.output_shape = shape
        return b.expand(a, shape)

    @staticmethod
    def backward(ctx: Context, grad):
        b = get_backend()
        from picograd.ops.elemwise import _unbroadcast
        return (_unbroadcast(grad, ctx.input_shape),)


class Cat(Function):
    @staticmethod
    def forward(ctx: Context, *args_and_axis):
        # Convention: last element is axis (int), rest are tensors
        axis = args_and_axis[-1]
        tensors = args_and_axis[:-1]
        b = get_backend()
        ctx.axis = axis
        ctx.shapes = [b.shape_of(t) for t in tensors]
        return b.concatenate(tensors, axis=axis)

    @staticmethod
    def backward(ctx: Context, grad):
        b = get_backend()
        axis = ctx.axis
        sizes = [s[axis] for s in ctx.shapes]
        # Split grad back
        split_indices = []
        running = 0
        for s in sizes[:-1]:
            running += s
            split_indices.append(running)
        grads = b.split(grad, split_indices, axis=axis)
        return tuple(grads) + (None,)   # None for the axis arg

    @classmethod
    def apply(cls, *tensors, axis: int = 0):
        """Override apply to handle axis keyword."""
        from picograd.tensor import Tensor, _ensure_tensor
        raw = [_ensure_tensor(t)._data for t in tensors]
        # Pass axis as last positional arg to forward
        inputs_list = list(tensors) + [axis]
        return super().apply(*inputs_list)


class Stack(Function):
    @staticmethod
    def forward(ctx: Context, *args_and_axis):
        axis = args_and_axis[-1]
        tensors = args_and_axis[:-1]
        b = get_backend()
        ctx.axis = axis
        ctx.n = len(tensors)
        return b.stack(tensors, axis=axis)

    @staticmethod
    def backward(ctx: Context, grad):
        b = get_backend()
        # Split along the stacked axis
        grads = b.split(grad, ctx.n, axis=ctx.axis)
        # squeeze the stacked dim out
        result = tuple(b.squeeze(g, axis=ctx.axis) for g in grads)
        return result + (None,)

    @classmethod
    def apply(cls, *tensors, axis: int = 0):
        from picograd.tensor import _ensure_tensor
        inputs_list = list(tensors) + [axis]
        return super().apply(*inputs_list)


class Slice(Function):
    @staticmethod
    def forward(ctx: Context, a, idx):
        b = get_backend()
        ctx.save_for_backward(a)
        ctx.idx = idx
        ctx.input_shape = b.shape_of(a)
        return a[idx]      # numpy fancy indexing

    @staticmethod
    def backward(ctx: Context, grad):
        b = get_backend()
        out = b.zeros(ctx.input_shape)
        np.add.at(out, ctx.idx, grad)   # scatter-add
        return (out,)

    @classmethod
    def apply(cls, tensor, idx):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled
        from picograd.autograd.function import Node

        needs_grad = _grad_enabled() and tensor.requires_grad
        ctx = Context()
        raw_out = cls.forward(ctx, tensor._data, idx)
        out = Tensor(raw_out, requires_grad=needs_grad)
        if needs_grad:
            input_nodes = [(tensor, tensor._grad_fn)]
            node = Node(cls, ctx, input_nodes)
            out._grad_fn = node
            out._is_leaf = False
        return out


__all__ = ["Reshape", "Transpose", "Squeeze", "Unsqueeze", "Expand", "Cat", "Stack", "Slice"]
