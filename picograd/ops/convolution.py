"""
picograd/ops/convolution.py
============================
Conv2d  -- im2col based (fast, BLAS-accelerated)
ConvTranspose2d -- direct scatter-based (correct)
"""
from __future__ import annotations
import numpy as np
from picograd.autograd.function import Function, Context, Node
from picograd.backend import get_backend


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _conv2d_output_size(H, W, kH, kW, stride, padding, dilation=1):
    H_out = (H + 2*padding - dilation*(kH-1) - 1)//stride + 1
    W_out = (W + 2*padding - dilation*(kW-1) - 1)//stride + 1
    return H_out, W_out


def im2col(x, kH, kW, stride=1, padding=0, dilation=1):
    """(N,C,H,W) -> (N, C*kH*kW, H_out*W_out)"""
    N, C, H, W = x.shape
    H_out, W_out = _conv2d_output_size(H, W, kH, kW, stride, padding, dilation)
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    col = np.zeros((N, C, kH, kW, H_out, W_out), dtype=x.dtype)
    for i in range(kH):
        for j in range(kW):
            ri = i*dilation
            rj = j*dilation
            col[:,: ,i, j, :, :] = x[:, :,
                ri:ri+stride*H_out:stride,
                rj:rj+stride*W_out:stride]
    return col.reshape(N, C*kH*kW, H_out*W_out)


def col2im(col, input_shape, kH, kW, stride=1, padding=0, dilation=1):
    """(N, C*kH*kW, H_out*W_out) -> (N,C,H,W)"""
    N, C, H, W = input_shape
    H_out, W_out = _conv2d_output_size(H, W, kH, kW, stride, padding, dilation)
    H_pad = H + 2*padding
    W_pad = W + 2*padding
    col_r = col.reshape(N, C, kH, kW, H_out, W_out)
    x_pad = np.zeros((N, C, H_pad, W_pad), dtype=col.dtype)
    for i in range(kH):
        for j in range(kW):
            ri = i*dilation
            rj = j*dilation
            x_pad[:, :, ri:ri+stride*H_out:stride, rj:rj+stride*W_out:stride] += col_r[:,:,i,j,:,:]
    if padding > 0:
        return x_pad[:,:,padding:-padding,padding:-padding]
    return x_pad


# -----------------------------------------------------------------------------
# Conv2d
# -----------------------------------------------------------------------------

class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape
        H_out, W_out = _conv2d_output_size(H, W, kH, kW, stride, padding, dilation)

        col   = im2col(x, kH, kW, stride, padding, dilation)  # (N, Ci*kH*kW, Ho*Wo)
        w_col = weight.reshape(C_out, -1)                      # (Co, Ci*kH*kW)

        out = np.tensordot(w_col, col, axes=([1],[1]))  # (Co, N, Ho*Wo)
        out = out.transpose(1, 0, 2).reshape(N, C_out, H_out, W_out)

        if bias is not None:
            out += bias.reshape(1,-1,1,1)

        ctx.save_for_backward(x, weight, bias if bias is not None else np.array([]))
        ctx.col = col; ctx.stride=stride; ctx.padding=padding
        ctx.dilation=dilation; ctx.has_bias=(bias is not None)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight, bias = ctx.saved_tensors
        col=ctx.col; stride=ctx.stride; padding=ctx.padding; dilation=ctx.dilation
        N,C_out,H_out,W_out = grad.shape
        kH,kW = weight.shape[2], weight.shape[3]
        w_col = weight.reshape(C_out,-1)

        grad_2d = grad.reshape(N, C_out, -1)          # (N, Co, Ho*Wo)
        grad_w = np.zeros_like(w_col)
        for n in range(N):
            grad_w += grad_2d[n] @ col[n].T

        grad_col = np.zeros_like(col)
        for n in range(N):
            grad_col[n] = w_col.T @ grad_2d[n]

        grad_x = col2im(grad_col, x.shape, kH, kW, stride, padding, dilation)
        grad_b = grad.sum(axis=(0,2,3)) if ctx.has_bias else np.array([])

        return grad_x, grad_w.reshape(weight.shape), grad_b

    @classmethod
    def apply(cls, x_t, weight_t, bias_t=None, stride=1, padding=0, dilation=1, groups=1):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled
        inputs = [x_t, weight_t] + ([bias_t] if bias_t is not None else [])
        needs_grad = _grad_enabled() and any(t.requires_grad for t in inputs)
        ctx = Context()
        raw = cls.forward(ctx, x_t._data, weight_t._data,
                          bias_t._data if bias_t is not None else None,
                          stride, padding, dilation, groups)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            inp_nodes = [(x_t,x_t._grad_fn),(weight_t,weight_t._grad_fn)]
            if bias_t is not None: inp_nodes.append((bias_t,bias_t._grad_fn))
            node = Node(cls, ctx, inp_nodes); out._grad_fn=node; out._is_leaf=False
        return out


# -----------------------------------------------------------------------------
# ConvTranspose2d -- clean scatter-based implementation
# -----------------------------------------------------------------------------

class ConvTranspose2d(Function):
    """
    Transposed convolution via zero-insertion + regular convolution.
    H_out = (H-1)*stride - 2*padding + kernel + output_padding
    """

    @staticmethod
    def _conv_transpose_forward(x, weight, bias, stride, padding, output_padding):
        """
        x:      (N, C_in,  H,   W)
        weight: (C_in, C_out, kH, kW)
        """
        N, C_in, H, W = x.shape
        _, C_out, kH, kW = weight.shape

        H_out = (H-1)*stride - 2*padding + kH + output_padding
        W_out = (W-1)*stride - 2*padding + kW + output_padding

        out = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

        # Scatter: for each input position (ih,iw) and kernel (ki,kj),
        # add contribution to output at (ih*s-p+ki, iw*s-p+kj)
        for ki in range(kH):
            for kj in range(kW):
                # Output positions this kernel tap writes to
                oh_start = ki - padding
                ow_start = kj - padding
                for ih in range(H):
                    oh = ih * stride + oh_start
                    if oh < 0 or oh >= H_out:
                        continue
                    for iw in range(W):
                        ow = iw * stride + ow_start
                        if ow < 0 or ow >= W_out:
                            continue
                        # x[:,ic,ih,iw] @ weight[ic,:,ki,kj] -> out[:,oc,oh,ow]
                        # out: (N,C_out) += (N,C_in) @ (C_in, C_out)
                        out[:, :, oh, ow] += x[:, :, ih, iw] @ weight[:, :, ki, kj]

        if bias is not None:
            out += bias.reshape(1,-1,1,1)
        return out

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, output_padding):
        out = ConvTranspose2d._conv_transpose_forward(x, weight, bias, stride, padding, output_padding)
        ctx.save_for_backward(x, weight, bias if bias is not None else np.array([]))
        ctx.stride=stride; ctx.padding=padding; ctx.output_padding=output_padding
        ctx.has_bias=(bias is not None)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight, bias = ctx.saved_tensors
        stride=ctx.stride; padding=ctx.padding
        N, C_in, H, W = x.shape
        _, C_out, kH, kW = weight.shape
        H_out, W_out = grad.shape[2], grad.shape[3]

        grad_x = np.zeros_like(x)
        grad_w = np.zeros_like(weight)

        for ki in range(kH):
            for kj in range(kW):
                oh_start = ki - padding
                ow_start = kj - padding
                for ih in range(H):
                    oh = ih * stride + oh_start
                    if oh < 0 or oh >= H_out: continue
                    for iw in range(W):
                        ow = iw * stride + ow_start
                        if ow < 0 or ow >= W_out: continue
                        # grad[:,:,oh,ow]: (N,C_out)
                        # weight[:,:,ki,kj]: (C_in,C_out)
                        grad_x[:,:,ih,iw] += grad[:,:,oh,ow] @ weight[:,:,ki,kj].T
                        grad_w[:,:,ki,kj] += x[:,:,ih,iw].T @ grad[:,:,oh,ow]

        grad_b = grad.sum(axis=(0,2,3)) if ctx.has_bias else np.array([])
        return grad_x, grad_w, grad_b

    @classmethod
    def apply(cls, x_t, weight_t, bias_t=None, stride=1, padding=0, dilation=1, output_padding=0):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled
        inputs = [x_t, weight_t] + ([bias_t] if bias_t is not None else [])
        needs_grad = _grad_enabled() and any(t.requires_grad for t in inputs)
        ctx = Context()
        raw = cls.forward(ctx, x_t._data, weight_t._data,
                          bias_t._data if bias_t is not None else None,
                          stride, padding, dilation, output_padding)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            inp_nodes = [(x_t,x_t._grad_fn),(weight_t,weight_t._grad_fn)]
            if bias_t is not None: inp_nodes.append((bias_t,bias_t._grad_fn))
            node = Node(cls, ctx, inp_nodes); out._grad_fn=node; out._is_leaf=False
        return out


__all__ = ["Conv2d", "ConvTranspose2d", "im2col", "col2im"]
