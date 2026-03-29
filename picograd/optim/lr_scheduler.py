"""picograd/optim/lr_scheduler.py"""
from __future__ import annotations
import math
from picograd.optim.optimizer import Optimizer


class _LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step()

    def get_lr(self): raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        lrs = self.get_lr()
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g['lr'] = lr

    @property
    def last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        if self.last_epoch % self.step_size != 0:
            return [g['lr'] for g in self.optimizer.param_groups]
        return [lr * self.gamma for lr in self.base_lrs
                if True] or [g['lr'] * self.gamma
                              for g in self.optimizer.param_groups]

    def get_lr(self):
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base * factor for base in self.base_lrs]


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        T = self.T_max
        return [
            self.eta_min + (base - self.eta_min) * (1 + math.cos(math.pi * t / T)) / 2
            for base in self.base_lrs
        ]


class ReduceLROnPlateau:
    """Reduce LR when a metric stops improving."""

    def __init__(self, optimizer, mode='min', factor=0.1,
                 patience=10, min_lr=0.0, threshold=1e-4):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0

    def step(self, metric):
        if self.mode == 'min':
            improved = metric < self.best - self.threshold
        else:
            improved = metric > self.best + self.threshold

        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            for g in self.optimizer.param_groups:
                new_lr = max(g['lr'] * self.factor, self.min_lr)
                g['lr'] = new_lr
            self.num_bad_epochs = 0


__all__ = ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
