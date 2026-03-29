"""
picograd/optim/
===============
Optimizer base + SGD, Adam, AdamW, RMSprop implementations.
All update parameters in-place using raw numpy (no autograd on optimizer step).
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Base Optimizer
# ---------------------------------------------------------------------------

class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params, defaults: Dict[str, Any]):
        # params can be a list of Tensors/Parameters or param_groups
        if isinstance(params, (list, tuple)) and len(params) > 0:
            # Check if it's already a list of param_groups
            if isinstance(params[0], dict):
                self.param_groups = list(params)
            else:
                self.param_groups = [{'params': list(params), **defaults}]
        else:
            self.param_groups = [{'params': list(params), **defaults}]

        self.defaults = defaults
        self.state: Dict[int, Dict] = {}   # id(param) -> optimizer state

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None

    def step(self) -> None:
        raise NotImplementedError

    def _get_state(self, p) -> Dict:
        pid = id(p)
        if pid not in self.state:
            self.state[pid] = {}
        return self.state[pid]

    def add_param_group(self, group: Dict) -> None:
        for k, v in self.defaults.items():
            group.setdefault(k, v)
        self.param_groups.append(group)


# ---------------------------------------------------------------------------
# SGD (with optional momentum + weight decay)
# ---------------------------------------------------------------------------

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and weight decay.
    Matches torch.optim.SGD exactly when same init and data are used.
    """

    def __init__(self, params, lr: float = 1e-2,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0,
                 dampening: float = 0.0,
                 nesterov: bool = False):
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        dampening=dampening, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group['lr']
            momentum = group.get('momentum', 0.0)
            weight_decay = group.get('weight_decay', 0.0)
            dampening = group.get('dampening', 0.0)
            nesterov = group.get('nesterov', False)

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data.copy()

                # L2 regularization
                if weight_decay != 0.0:
                    g = g + weight_decay * p._data

                if momentum != 0.0:
                    state = self._get_state(p)
                    if 'buf' not in state:
                        state['buf'] = g.copy()
                    else:
                        state['buf'] = momentum * state['buf'] + (1 - dampening) * g

                    if nesterov:
                        g = g + momentum * state['buf']
                    else:
                        g = state['buf']

                p._data -= lr * g


# ---------------------------------------------------------------------------
# Adam / AdamW
# ---------------------------------------------------------------------------

class Adam(Optimizer):
    """
    Adam optimizer.  When weight_decay > 0 and decoupled=True -> AdamW.
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 decoupled: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, decoupled=decoupled)
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group.get('weight_decay', 0.0)
            decoupled = group.get('decoupled', False)

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data.copy()

                state = self._get_state(p)
                if 'm' not in state:
                    state['m']  = np.zeros_like(p._data)
                    state['v']  = np.zeros_like(p._data)
                    state['t']  = 0

                state['t'] += 1
                t = state['t']

                # Standard Adam L2 (not decoupled)
                if wd != 0.0 and not decoupled:
                    g = g + wd * p._data

                state['m'] = b1 * state['m'] + (1 - b1) * g
                state['v'] = b2 * state['v'] + (1 - b2) * (g * g)

                # Bias correction
                m_hat = state['m'] / (1 - b1 ** t)
                v_hat = state['v'] / (1 - b2 ** t)

                update = lr * m_hat / (np.sqrt(v_hat) + eps)

                if wd != 0.0 and decoupled:
                    # AdamW: decouple weight decay
                    p._data -= lr * wd * p._data

                p._data -= update


class AdamW(Adam):
    """Adam with decoupled weight decay (AdamW)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, decoupled=True)


# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------

class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99,
                 eps=1e-8, weight_decay=0.0, momentum=0.0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps,
                        weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            wd = group.get('weight_decay', 0.0)
            mom = group.get('momentum', 0.0)

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data.copy()
                if wd != 0.0:
                    g = g + wd * p._data

                state = self._get_state(p)
                if 'sq' not in state:
                    state['sq']  = np.zeros_like(p._data)
                    state['buf'] = np.zeros_like(p._data)

                state['sq'] = alpha * state['sq'] + (1 - alpha) * g * g
                rms = np.sqrt(state['sq']) + eps

                if mom > 0:
                    state['buf'] = mom * state['buf'] + g / rms
                    p._data -= lr * state['buf']
                else:
                    p._data -= lr * g / rms


__all__ = ["Optimizer", "SGD", "Adam", "AdamW", "RMSprop"]
