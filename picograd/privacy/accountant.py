"""
picograd/privacy/accountant.py
================================
Renyi Differential Privacy (RDP) accountant.

Tracks cumulative privacy budget across training rounds and converts
to (eps, delta)-DP using the standard conversion from Balle et al. (2020).

Reference:
  Mironov, "Renyi Differential Privacy" (2017).
  Mironov et al., "Renyi Differential Privacy of the Sampled Gaussian Mechanism" (2019).

Usage:
    acc = RDPAccountant()
    for _ in range(100):
        acc.step(noise_multiplier=1.0, sample_rate=0.01)
    eps = acc.get_epsilon(delta=1e-5)
"""

from __future__ import annotations
import numpy as np
from math import lgamma
from typing import List, Tuple

# RDP orders to track -- standard set covers tight bounds for typical sigma
_DEFAULT_ORDERS = list(range(2, 64)) + [128, 256, 512]


# --- Core RDP computation ----------------------------------------------------

def _log_binom(n: int, k: int) -> float:
    """log C(n,k) via log-gamma -- numerically stable."""
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _subsampled_gaussian_rdp(alpha: int, q: float, sigma: float) -> float:
    """
    Exact RDP for the Poisson-subsampled Gaussian mechanism at integer order alpha.

    Formula (Theorem 9, Mironov et al. 2019):
        RDP(alpha) = (1/(alpha-1)) * log Sum_{j=0}^{alpha} C(alpha,j)*q^j*(1-q)^{alpha-j}*exp((j^2-j)/(2sigma^2))

    This is the moment-generating function approach -- exact for integer alpha.
    """
    if sigma <= 0:
        return float('inf')
    if q <= 0:
        return 0.0
    if q > 1:
        q = 1.0

    alpha_int = int(alpha)
    if alpha_int < 2:
        # alpha=1: use KL bound (q^2/2sigma^2)
        return float(q * q / (2.0 * sigma * sigma))

    log_sum = -np.inf
    for j in range(alpha_int + 1):
        log_term = (
            _log_binom(alpha_int, j)
            + j * np.log(q + 1e-300)
            + (alpha_int - j) * np.log(1.0 - q + 1e-300)
            + (j * j - j) / (2.0 * sigma * sigma)
        )
        log_sum = np.logaddexp(log_sum, log_term)

    return float(log_sum / (alpha - 1))


# --- RDP -> (eps, delta) conversion ------------------------------------------------

def _rdp_to_epsilon(orders: List[float],
                    rdp_values: List[float],
                    delta: float) -> float:
    """
    Convert RDP guarantee to (eps, delta)-DP.

    Standard conversion (Proposition 3, Balle et al. 2020):
        eps = rdp(alpha) + log(alpha-1)/alpha - [log(delta) + log(alpha-1)/alpha] / (alpha-1)

    We take the minimum over all candidate orders.
    """
    best_eps = float('inf')
    for alpha, rdp in zip(orders, rdp_values):
        if rdp == 0 or not np.isfinite(rdp):
            continue
        if alpha <= 1:
            continue
        # Convert RDP(alpha) -> (eps, delta)-DP
        # This is the tight formula from Canonne et al. / Balle et al.:
        #   eps(alpha) = rdp(alpha) + log(1 - 1/alpha) - [log(delta) + log(1 - 1/alpha)] / (alpha - 1)
        try:
            log_a = np.log(1.0 - 1.0 / alpha)
            eps = rdp + log_a - (np.log(delta) + log_a) / (alpha - 1)
            if np.isfinite(eps) and eps < best_eps:
                best_eps = eps
        except Exception:
            pass
    return best_eps


# --- RDPAccountant class ------------------------------------------------------

class RDPAccountant:
    """
    Tracks privacy expenditure via Renyi DP composition.

    Each step() call records one round of the subsampled Gaussian mechanism.
    get_epsilon() converts the accumulated RDP to the tightest (eps, delta)-DP.
    """

    def __init__(self, orders: List[float] = None):
        self.orders = list(orders or _DEFAULT_ORDERS)
        self.rdp = np.zeros(len(self.orders), dtype=np.float64)
        self._history: List[Tuple[float, float]] = []

    def step(self, noise_multiplier: float, sample_rate: float) -> None:
        """Record one mechanism application (one FL round / training step)."""
        self._history.append((noise_multiplier, sample_rate))
        for i, alpha in enumerate(self.orders):
            self.rdp[i] += _subsampled_gaussian_rdp(alpha, sample_rate, noise_multiplier)

    def get_epsilon(self, delta: float = 1e-5) -> float:
        """Return current (eps, delta)-DP guarantee -- tight over all tracked orders."""
        return _rdp_to_epsilon(self.orders, self.rdp.tolist(), delta)

    def get_privacy_spent(self, delta: float = 1e-5) -> dict:
        """Full privacy report for the dashboard."""
        eps = self.get_epsilon(delta)
        return {
            "epsilon": eps,
            "delta": delta,
            "steps": len(self._history),
            "rdp_peak": float(np.max(self.rdp)),
        }

    def reset(self) -> None:
        self.rdp[:] = 0.0
        self._history.clear()

    @property
    def num_steps(self) -> int:
        return len(self._history)

    def summary(self, delta: float = 1e-5) -> str:
        eps = self.get_epsilon(delta)
        return (f"RDPAccountant | rounds={len(self._history)} | "
                f"eps={eps:.3f} @ delta={delta:.0e}")


# --- PrivacyConfig ------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class PrivacyConfig:
    """Hyperparameters for differential privacy in FALCON."""
    noise_multiplier: float = 1.0
    max_grad_norm:    float = 1.0
    target_epsilon:   float = 8.0
    target_delta:     float = 1e-5
    enabled:          bool  = True


__all__ = ["RDPAccountant", "PrivacyConfig", "_rdp_to_epsilon", "_subsampled_gaussian_rdp"]
