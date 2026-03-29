"""
picograd/privacy/dp.py
=======================
Differential Privacy engine — DPOptimizer + PrivacyConfig.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PrivacyConfig:
    noise_multiplier: float = 1.0
    max_grad_norm:    float = 1.0
    target_epsilon:   float = 8.0
    target_delta:     float = 1e-5
    enabled:          bool  = True


class DPOptimizer:
    """DP-SGD wrapper: clip gradients + add Gaussian noise per step."""

    def __init__(
        self,
        optimizer,
        noise_multiplier: float = 1.0,
        max_grad_norm:    float = 1.0,
        batch_size:       int   = 32,
        dataset_size:     int   = 1000,
        accountant=None,
    ):
        self.optimizer        = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm    = max_grad_norm
        self.batch_size       = batch_size
        self.dataset_size     = dataset_size
        self.accountant       = accountant
        self._steps           = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self) -> None:
        from picograd.tensor import Tensor
        C  = self.max_grad_norm
        σ  = self.noise_multiplier
        bs = self.batch_size

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data.copy()

                # 1. L2 clip
                l2 = float(np.linalg.norm(g))
                if l2 > C:
                    g = g * (C / l2)

                # 2. Gaussian noise
                noise = np.random.normal(0.0, σ * C, size=g.shape).astype(g.dtype)
                g_noisy = g + noise

                # 3. Average
                p.grad = Tensor(g_noisy / bs)

        self.optimizer.step()
        self._steps += 1

        if self.accountant is not None:
            self.accountant.step(self.noise_multiplier,
                                 self.batch_size / self.dataset_size)

    def get_privacy_spent(self, delta: float = 1e-5):
        if self.accountant is None:
            raise RuntimeError("No accountant attached.")
        return self.accountant.get_epsilon(delta), delta

    @property
    def steps(self):
        return self._steps


__all__ = ["DPOptimizer", "PrivacyConfig"]
