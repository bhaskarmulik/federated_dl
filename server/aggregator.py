"""
server/aggregator.py
=====================
Main FL aggregation server.

Supports:
  - Plain FedAvg
  - FedAvg + Secure Aggregation (--secure-agg)
  - FedAvg + Differential Privacy clipping signal (--dp)
  - FedBN (--fedbn)
  - FedProx loss helper (client-side)

This replaces the original torch-based aggregator with picograd state dicts.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

from server.fedavg import fedavg, fedavg_delta
from server.secure_agg import SecureAggregator, MaskGenerator

StateDict = Dict[str, np.ndarray]


class Aggregator:
    """
    Central FL aggregation server.

    Collects client updates, applies chosen aggregation strategy,
    and returns the updated global model.

    Parameters
    ----------
    strategy     : 'fedavg' | 'fedbn' | 'fedprox'
    secure_agg   : enable masking-based secure aggregation
    dp_config    : PrivacyConfig if server tracks global DP budget
    min_clients  : minimum clients before aggregation runs
    """

    def __init__(
        self,
        strategy:    str = 'fedavg',
        secure_agg:  bool = False,
        dp_config    = None,
        min_clients: int = 1,
    ):
        self.strategy    = strategy
        self.secure_agg  = secure_agg
        self.dp_config   = dp_config
        self.min_clients = min_clients

        self._pending: List[Tuple[str, StateDict, int]] = []  # (client_id, sd, n)
        self._round   = 0

        if secure_agg:
            # SecureAggregator initialized lazily when first client count is known
            self._sec_agg: Optional[SecureAggregator] = None

        # Privacy accountant (optional)
        if dp_config is not None:
            from picograd.privacy import RDPAccountant
            self.accountant = RDPAccountant()
        else:
            self.accountant = None

    # ------------------------------------------------------------------ ingestion

    def receive_update(
        self,
        client_id:   str,
        state_dict:  StateDict,
        n_samples:   int,
    ) -> None:
        """Accept one client update."""
        self._pending.append((client_id, state_dict, n_samples))

    @property
    def n_pending(self) -> int:
        return len(self._pending)

    @property
    def ready(self) -> bool:
        return self.n_pending >= self.min_clients

    # ------------------------------------------------------------------ aggregation

    def aggregate(self, global_sd: Optional[StateDict] = None) -> StateDict:
        """
        Run aggregation over all pending updates.
        Returns new global state dict.
        Clears pending buffer after aggregation.
        """
        if not self._pending:
            raise RuntimeError("No pending updates to aggregate.")

        client_ids  = [t[0] for t in self._pending]
        updates     = [t[1] for t in self._pending]
        counts      = [t[2] for t in self._pending]

        if self.secure_agg:
            result = self._aggregate_secure(updates, counts, len(self._pending))
        elif self.strategy == 'fedbn':
            from server.strategies.fedprox_fedbn import fedbn_aggregate
            result = fedbn_aggregate(updates, counts)
        else:
            # Plain FedAvg (also covers fedprox -- client handles proximal term)
            result = fedavg(updates, counts)

        # Privacy accounting
        if self.accountant is not None and self.dp_config is not None:
            sample_rate = sum(counts) / max(sum(counts) * 10, 1)  # approx
            self.accountant.step(self.dp_config.noise_multiplier, sample_rate)

        self._pending.clear()
        self._round += 1
        return result

    def _aggregate_secure(
        self,
        updates: List[StateDict],
        counts:  List[int],
        n:       int,
    ) -> StateDict:
        """Secure aggregation: clients pre-weight, server sums (masks cancel)."""
        total  = sum(counts)
        shapes = {k: v.shape for k, v in updates[0].items()}

        # Build masked updates
        sec_agg = SecureAggregator(n)
        for i, (sd, cnt) in enumerate(zip(updates, counts)):
            w_i = cnt / total
            gen = MaskGenerator(i, n, self._round)
            masked = gen.mask_update(sd, shapes, weight=w_i)
            sec_agg.receive(str(i), masked, cnt)

        return sec_agg.aggregate()

    # ------------------------------------------------------------------ privacy reporting

    def get_privacy_budget(self, delta: float = 1e-5) -> Optional[Tuple[float, float]]:
        """Return (eps, delta) spent so far, or None if no accountant."""
        if self.accountant is None:
            return None
        eps = self.accountant.get_epsilon(delta)
        return eps, delta

    @property
    def round(self) -> int:
        return self._round


__all__ = ["Aggregator"]
