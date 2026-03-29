"""
picograd/ops/normalization.py
==============================
BatchNorm and LayerNorm as Function subclasses.

BatchNorm:
  - Train mode: normalize over (N, H, W), update running mean/var, return gamma*x_hat + beta
  - Eval  mode: use running mean/var
  - Backward: full 4-step batch-norm gradient formula

LayerNorm:
  - Normalize over the last `normalized_shape` dimensions
  - No running stats (always uses batch statistics)
"""

from __future__ import annotations
import numpy as np
from picograd.autograd.function import Function, Context, Node
from picograd.backend import get_backend


class BatchNorm(Function):
    """
    Differentiable BatchNorm2d.
    running_mean/running_var are updated in-place during training.
    """

    @staticmethod
    def forward(ctx: Context, x, gamma, beta,
                running_mean, running_var,
                training, momentum, eps):
        """
        x: (N, C, H, W) or (N, C)
        gamma, beta: (C,)
        """
        b = get_backend()
        N = x.shape[0]
        C = x.shape[1]
        # axes to reduce over: everything except the channel dim
        if x.ndim == 4:
            axes = (0, 2, 3)
            keepdims_shape = (1, C, 1, 1)
        else:
            axes = (0,)
            keepdims_shape = (1, C)

        if training:
            mean = x.mean(axis=axes)
            var  = x.var(axis=axes)
            # Normalize
            x_hat = (x - mean.reshape(keepdims_shape)) / \
                    np.sqrt(var.reshape(keepdims_shape) + eps)
            # Update running stats
            if running_mean is not None:
                running_mean[:] = (1 - momentum) * running_mean + momentum * mean
                running_var[:]  = (1 - momentum) * running_var  + momentum * var
        else:
            mean = running_mean if running_mean is not None else x.mean(axis=axes)
            var  = running_var  if running_var  is not None else x.var(axis=axes)
            x_hat = (x - mean.reshape(keepdims_shape)) / \
                    np.sqrt(var.reshape(keepdims_shape) + eps)

        out = gamma.reshape(keepdims_shape) * x_hat + beta.reshape(keepdims_shape)

        ctx.save_for_backward(x_hat, gamma)
        ctx.mean = mean
        ctx.var  = var
        ctx.axes = axes
        ctx.keepdims_shape = keepdims_shape
        ctx.eps = eps
        ctx.training = training
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        x_hat, gamma = ctx.saved_tensors
        b = get_backend()
        axes = ctx.axes
        keepdims_shape = ctx.keepdims_shape
        eps = ctx.eps

        # Number of elements per channel
        N_elem = 1
        for ax in axes:
            N_elem *= x_hat.shape[ax]

        # 1. grad w.r.t. gamma:  sum(grad * x_hat, axes)
        grad_gamma = (grad * x_hat).sum(axis=axes)
        # 2. grad w.r.t. beta:   sum(grad, axes)
        grad_beta  = grad.sum(axis=axes)

        # 3. grad w.r.t. x_hat:  grad * gamma
        grad_xhat = grad * gamma.reshape(keepdims_shape)

        # 4. grad w.r.t. x via batch-norm backward formula:
        #   dx = (1/N) * (1/sqrt(var+eps)) * (N*dxhat - sum(dxhat) - x_hat*sum(dxhat*x_hat))
        var_plus_eps = ctx.var.reshape(keepdims_shape)
        inv_std = 1.0 / np.sqrt(var_plus_eps + eps)

        dx = (1.0 / N_elem) * inv_std * (
            N_elem * grad_xhat
            - grad_xhat.sum(axis=axes, keepdims=True)
            - x_hat * (grad_xhat * x_hat).sum(axis=axes, keepdims=True)
        )

        return dx, grad_gamma, grad_beta

    @classmethod
    def apply(cls, x_t, gamma_t, beta_t,
              running_mean, running_var,
              training=True, momentum=0.1, eps=1e-5):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled

        needs_grad = _grad_enabled() and any(
            t.requires_grad for t in [x_t, gamma_t, beta_t]
        )
        ctx = Context()
        raw = cls.forward(ctx, x_t._data, gamma_t._data, beta_t._data,
                          running_mean, running_var,
                          training, momentum, eps)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            inp_nodes = [
                (x_t, x_t._grad_fn),
                (gamma_t, gamma_t._grad_fn),
                (beta_t, beta_t._grad_fn),
            ]
            node = Node(cls, ctx, inp_nodes)
            out._grad_fn = node
            out._is_leaf = False
        return out


class LayerNorm(Function):
    """
    Layer normalization over last len(normalized_shape) dimensions.
    """

    @staticmethod
    def forward(ctx: Context, x, gamma, beta, normalized_shape, eps):
        ndim = len(normalized_shape)
        axes = tuple(range(-ndim, 0))
        mean = x.mean(axis=axes, keepdims=True)
        var  = x.var(axis=axes, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_hat + beta

        ctx.save_for_backward(x_hat, gamma)
        ctx.axes = axes
        ctx.eps = eps
        ctx.var = var
        return out

    @staticmethod
    def backward(ctx: Context, grad):
        x_hat, gamma = ctx.saved_tensors
        axes = ctx.axes
        eps = ctx.eps
        var = ctx.var

        # N_elem: number of elements in the normalized dims
        N_elem = 1
        for s in x_hat.shape[-len(axes):]:
            N_elem *= s

        grad_gamma = (grad * x_hat).sum(axis=tuple(range(x_hat.ndim - len(axes))), keepdims=False)
        if grad_gamma.shape != gamma.shape:
            grad_gamma = grad_gamma.reshape(gamma.shape)
        grad_beta  = grad.sum(axis=tuple(range(x_hat.ndim - len(axes))), keepdims=False)
        if grad_beta.shape != gamma.shape:
            grad_beta = grad_beta.reshape(gamma.shape)

        grad_xhat = grad * gamma
        inv_std = 1.0 / np.sqrt(var + eps)
        dx = (1.0 / N_elem) * inv_std * (
            N_elem * grad_xhat
            - grad_xhat.sum(axis=axes, keepdims=True)
            - x_hat * (grad_xhat * x_hat).sum(axis=axes, keepdims=True)
        )
        return dx, grad_gamma, grad_beta

    @classmethod
    def apply(cls, x_t, gamma_t, beta_t, normalized_shape, eps=1e-5):
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled

        needs_grad = _grad_enabled() and any(
            t.requires_grad for t in [x_t, gamma_t, beta_t]
        )
        ctx = Context()
        raw = cls.forward(ctx, x_t._data, gamma_t._data, beta_t._data,
                          normalized_shape, eps)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            inp_nodes = [
                (x_t, x_t._grad_fn),
                (gamma_t, gamma_t._grad_fn),
                (beta_t, beta_t._grad_fn),
            ]
            node = Node(cls, ctx, inp_nodes)
            out._grad_fn = node
            out._is_leaf = False
        return out


__all__ = ["BatchNorm", "LayerNorm"]
