"""
picograd/nn/loss.py
====================
Loss functions as nn.Module subclasses.

CrossEntropyLoss: Fused LogSoftmax + NLLLoss for numerical stability.
MSELoss:          Mean squared error.
BCELoss:          Binary cross-entropy.
"""

from __future__ import annotations
import numpy as np
from picograd.nn.module import Module
from picograd.tensor import Tensor
from picograd.autograd.function import Function, Context, Node
from picograd.backend import get_backend


# ---------------------------------------------------------------------------
# Fused CrossEntropyLoss Function
# ---------------------------------------------------------------------------

class _CrossEntropyFn(Function):
    """
    Fused log-softmax + NLL loss.
    Inputs: logits (N, C), targets (N,) as integers
    Output: scalar mean loss
    """
    @staticmethod
    def forward(ctx: Context, logits, targets):
        b = get_backend()
        N, C = logits.shape
        # Stable log-softmax
        mx = b.max(logits, axis=1, keepdims=True)
        shifted = b.sub(logits, b.expand(mx, logits.shape))
        log_sum_exp = b.log(b.sum(b.exp(shifted), axis=1, keepdims=True))
        log_probs = b.sub(shifted, b.expand(log_sum_exp, shifted.shape))

        # NLL: -log_probs[n, targets[n]]
        n_idx = np.arange(N)
        loss_per_sample = -log_probs[n_idx, targets.astype(np.int32)]
        loss = b.mean(loss_per_sample)

        # Save softmax (not log-softmax) for backward
        softmax = b.exp(log_probs)
        ctx.save_for_backward(softmax)
        ctx.targets = targets.astype(np.int32)
        ctx.N = N
        return loss

    @staticmethod
    def backward(ctx: Context, grad):
        softmax, = ctx.saved_tensors
        b = get_backend()
        N = ctx.N
        targets = ctx.targets

        # d(CE)/d(logits) = (softmax - one_hot) / N
        d_logits = softmax.copy()
        d_logits[np.arange(N), targets] -= 1.0
        d_logits = b.div(d_logits, b.full(d_logits.shape, float(N)))
        d_logits = b.mul(d_logits, b.full(d_logits.shape, float(b.item(grad))))
        return (d_logits, None)   # None for targets (not differentiable)

    @classmethod
    def apply(cls, logits_t, targets_t):
        from picograd.autograd.context import _grad_enabled
        needs_grad = _grad_enabled() and logits_t.requires_grad
        ctx = Context()
        raw = cls.forward(ctx, logits_t._data, targets_t._data)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            node = Node(cls, ctx, [
                (logits_t, logits_t._grad_fn),
                (targets_t, None),
            ])
            out._grad_fn = node
            out._is_leaf = False
        return out


# ---------------------------------------------------------------------------
# MSE Loss Function
# ---------------------------------------------------------------------------

class _MSEFn(Function):
    @staticmethod
    def forward(ctx: Context, pred, target, reduction):
        b = get_backend()
        diff = b.sub(pred, target)
        sq   = b.mul(diff, diff)
        ctx.save_for_backward(diff)
        ctx.reduction = reduction
        ctx.n = pred.size
        if reduction == 'mean':
            return b.mean(sq)
        elif reduction == 'sum':
            return b.sum(sq)
        else:
            return sq

    @staticmethod
    def backward(ctx: Context, grad):
        diff, = ctx.saved_tensors
        b = get_backend()
        if ctx.reduction == 'mean':
            scale = 2.0 / ctx.n
        elif ctx.reduction == 'sum':
            scale = 2.0
        else:
            scale = 2.0
        return (b.mul(b.mul(diff, b.full(diff.shape, scale)),
                      b.full(diff.shape, b.item(grad))),
                None)   # None for target

    @classmethod
    def apply(cls, pred_t, target_t, reduction='mean'):
        from picograd.autograd.context import _grad_enabled
        needs_grad = _grad_enabled() and pred_t.requires_grad
        ctx = Context()
        raw = cls.forward(ctx, pred_t._data, target_t._data, reduction)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            node = Node(cls, ctx, [
                (pred_t, pred_t._grad_fn),
                (target_t, None),
            ])
            out._grad_fn = node
            out._is_leaf = False
        return out


# ---------------------------------------------------------------------------
# BCE Loss Function
# ---------------------------------------------------------------------------

class _BCEFn(Function):
    @staticmethod
    def forward(ctx: Context, pred, target, reduction):
        b = get_backend()
        eps = 1e-8
        pred_c = b.clip(pred, eps, 1 - eps)
        loss = b.neg(b.add(
            b.mul(target, b.log(pred_c)),
            b.mul(b.sub(b.ones(pred_c.shape), target),
                  b.log(b.sub(b.ones(pred_c.shape), pred_c)))
        ))
        ctx.save_for_backward(pred_c, target)
        ctx.reduction = reduction
        ctx.n = pred.size
        if reduction == 'mean':
            return b.mean(loss)
        elif reduction == 'sum':
            return b.sum(loss)
        return loss

    @staticmethod
    def backward(ctx: Context, grad):
        pred_c, target = ctx.saved_tensors
        b = get_backend()
        # d(BCE)/d(pred) = -(target/pred - (1-target)/(1-pred)) / n
        denom = ctx.n if ctx.reduction == 'mean' else 1.0
        eps = 1e-8
        d = b.neg(b.sub(
            b.div(target, b.clip(pred_c, eps, None)),
            b.div(b.sub(b.ones(pred_c.shape), target),
                  b.clip(b.sub(b.ones(pred_c.shape), pred_c), eps, None))
        ))
        d = b.div(d, b.full(d.shape, float(denom)))
        d = b.mul(d, b.full(d.shape, float(b.item(grad))))
        return (d, None)

    @classmethod
    def apply(cls, pred_t, target_t, reduction='mean'):
        from picograd.autograd.context import _grad_enabled
        needs_grad = _grad_enabled() and pred_t.requires_grad
        ctx = Context()
        raw = cls.forward(ctx, pred_t._data, target_t._data, reduction)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            node = Node(cls, ctx, [
                (pred_t, pred_t._grad_fn),
                (target_t, None),
            ])
            out._grad_fn = node
            out._is_leaf = False
        return out


# ---------------------------------------------------------------------------
# Public Module wrappers
# ---------------------------------------------------------------------------

class CrossEntropyLoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return _CrossEntropyFn.apply(logits, targets)


class MSELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return _MSEFn.apply(pred, target, self.reduction)


class BCELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return _BCEFn.apply(pred, target, self.reduction)


class NLLLoss(Module):
    """Negative log-likelihood loss. Expects log-probabilities as input."""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        from picograd.autograd.context import _grad_enabled
        b = get_backend()
        N = log_probs.shape[0]
        n_idx = np.arange(N)
        # Index into log_probs: log_probs[n, targets[n]]
        gathered = Tensor(log_probs._data[n_idx, targets._data.astype(np.int32)],
                          requires_grad=log_probs.requires_grad)
        return -(gathered.sum() / Tensor(np.array(float(N))))


__all__ = ["CrossEntropyLoss", "MSELoss", "BCELoss", "NLLLoss"]
