"""
picograd/ops/pooling.py
========================
2D pooling operations.

MaxPool2d  -- stores argmax indices for backward (gradient routing)
AvgPool2d  -- distributes gradient equally over pooling window
AdaptiveAvgPool2d -- computes dynamic kernel_size from target output size
"""

from __future__ import annotations
import numpy as np
from picograd.autograd.function import Function, Context, Node
from picograd.backend import get_backend


def _pool2d_out_size(H, W, kernel_size, stride, padding):
    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sH, sW = (stride, stride) if isinstance(stride, int) else stride
    H_out = (H + 2 * padding - kH) // sH + 1
    W_out = (W + 2 * padding - kW) // sW + 1
    return H_out, W_out, kH, kW, sH, sW


class MaxPool2d(Function):
    @staticmethod
    def forward(ctx: Context, x, kernel_size, stride, padding):
        N, C, H, W = x.shape
        H_out, W_out, kH, kW, sH, sW = _pool2d_out_size(H, W, kernel_size, stride, padding)

        if padding > 0:
            x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)),
                           mode='constant', constant_values=-np.inf)
        else:
            x_pad = x

        out = np.full((N, C, H_out, W_out), -np.inf, dtype=x.dtype)
        # Store max indices for backward
        idx_h = np.zeros((N, C, H_out, W_out), dtype=np.int32)
        idx_w = np.zeros((N, C, H_out, W_out), dtype=np.int32)

        for i in range(H_out):
            for j in range(W_out):
                patch = x_pad[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW]  # (N,C,kH,kW)
                patch_flat = patch.reshape(N, C, -1)
                max_idx = patch_flat.argmax(axis=2)
                out[:, :, i, j] = patch_flat.max(axis=2)
                idx_h[:, :, i, j] = max_idx // kW + i * sH - padding
                idx_w[:, :, i, j] = max_idx %  kW + j * sW - padding

        ctx.save_for_backward(x)
        ctx.idx_h = idx_h
        ctx.idx_w = idx_w
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        x, = ctx.saved_tensors
        N, C, H, W = x.shape
        idx_h = ctx.idx_h
        idx_w = ctx.idx_w
        grad_x = np.zeros_like(x)

        N_, C_, H_out, W_out = grad.shape
        for i in range(H_out):
            for j in range(W_out):
                ih = idx_h[:, :, i, j]   # (N, C) -- valid x indices
                iw = idx_w[:, :, i, j]
                for n in range(N):
                    for c in range(C):
                        h_idx = ih[n, c]
                        w_idx = iw[n, c]
                        if 0 <= h_idx < H and 0 <= w_idx < W:
                            grad_x[n, c, h_idx, w_idx] += grad[n, c, i, j]

        return (grad_x,)

    @classmethod
    def apply(cls, tensor, kernel_size, stride=None, padding=0):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled

        if stride is None:
            stride = kernel_size

        needs_grad = _grad_enabled() and tensor.requires_grad
        ctx = Context()
        raw_out = cls.forward(ctx, tensor._data, kernel_size, stride, padding)
        out = Tensor(raw_out, requires_grad=needs_grad)
        if needs_grad:
            node = Node(cls, ctx, [(tensor, tensor._grad_fn)])
            out._grad_fn = node
            out._is_leaf = False
        return out


class AvgPool2d(Function):
    @staticmethod
    def forward(ctx: Context, x, kernel_size, stride, padding):
        N, C, H, W = x.shape
        H_out, W_out, kH, kW, sH, sW = _pool2d_out_size(H, W, kernel_size, stride, padding)

        if padding > 0:
            x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
        else:
            x_pad = x

        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        for i in range(H_out):
            for j in range(W_out):
                patch = x_pad[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW]
                out[:, :, i, j] = patch.mean(axis=(2, 3))

        ctx.save_for_backward(x)
        ctx.kernel_size = (kH, kW)
        ctx.stride = (sH, sW)
        ctx.padding = padding
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        x, = ctx.saved_tensors
        N, C, H, W = x.shape
        kH, kW = ctx.kernel_size
        sH, sW = ctx.stride
        padding = ctx.padding
        H_out, W_out, _, _, _, _ = _pool2d_out_size(H, W, ctx.kernel_size, ctx.stride, padding)

        grad_x = np.zeros_like(x)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * sH - padding
                w_start = j * sW - padding
                h_end = h_start + kH
                w_end = w_start + kW
                h_start_c = max(0, h_start)
                w_start_c = max(0, w_start)
                h_end_c = min(H, h_end)
                w_end_c = min(W, w_end)
                g = grad[:, :, i, j] / (kH * kW)
                grad_x[:, :, h_start_c:h_end_c, w_start_c:w_end_c] += \
                    g[:, :, np.newaxis, np.newaxis]

        return (grad_x,)

    @classmethod
    def apply(cls, tensor, kernel_size, stride=None, padding=0):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled

        if stride is None:
            stride = kernel_size
        needs_grad = _grad_enabled() and tensor.requires_grad
        ctx = Context()
        raw_out = cls.forward(ctx, tensor._data, kernel_size, stride, padding)
        out = Tensor(raw_out, requires_grad=needs_grad)
        if needs_grad:
            node = Node(cls, ctx, [(tensor, tensor._grad_fn)])
            out._grad_fn = node
            out._is_leaf = False
        return out


class AdaptiveAvgPool2d(Function):
    """Computes kernel_size from input/output size then delegates to AvgPool2d."""

    @staticmethod
    def _compute_kernel(in_size, out_size):
        if out_size == 1:
            return in_size, in_size, 0
        stride = in_size // out_size
        kernel = in_size - (out_size - 1) * stride
        return kernel, stride, 0

    @classmethod
    def apply(cls, tensor, output_size):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled

        H, W = tensor.shape[2], tensor.shape[3]
        oH, oW = (output_size, output_size) if isinstance(output_size, int) else output_size
        kH, sH, pH = cls._compute_kernel(H, oH)
        kW, sW, pW = cls._compute_kernel(W, oW)

        needs_grad = _grad_enabled() and tensor.requires_grad
        ctx = Context()
        raw_out = AvgPool2d.forward(ctx, tensor._data, (kH, kW), (sH, sW), 0)
        out = Tensor(raw_out, requires_grad=needs_grad)
        if needs_grad:
            node = Node(AvgPool2d, ctx, [(tensor, tensor._grad_fn)])
            out._grad_fn = node
            out._is_leaf = False
        return out


__all__ = ["MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d"]
