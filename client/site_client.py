"""
client/site_client.py
======================
Federated Learning client.

Replaces the original torch-based site_client.py with picograd.
Supports:
  - Local training with any picograd model/optimizer
  - Optional DPOptimizer wrapper
  - FedProx proximal term
  - Grad-CAM explainability reporting
  - Heartbeat metrics
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

import picograd
import picograd.nn as nn
import picograd.optim as optim
from picograd import Tensor
from picograd.privacy.dp import DPOptimizer, PrivacyConfig

StateDict = Dict[str, np.ndarray]


class SiteClient:
    """
    A single FL client (hospital/data silo).

    Usage:
        client = SiteClient('hospital_a', model, train_loader, privacy_config)
        client.set_global_weights(global_sd)
        local_sd, metrics = client.train_round(epochs=1, lr=1e-3)
    """

    def __init__(
        self,
        client_id:      str,
        model:          nn.Module,
        train_loader,
        privacy_config: Optional[PrivacyConfig] = None,
        val_loader      = None,
        use_fedprox:    bool = False,
        fedprox_mu:     float = 0.01,
    ):
        self.client_id      = client_id
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.privacy_config = privacy_config or PrivacyConfig(enabled=False)
        self.use_fedprox    = use_fedprox
        self.fedprox_mu     = fedprox_mu

        self._global_sd: Optional[StateDict] = None
        self._round = 0

    # ------------------------------------------------------------------ weights

    def set_global_weights(self, global_sd: StateDict) -> None:
        """Load global model weights before local training."""
        self.model.load_state_dict(global_sd)
        self._global_sd = {k: v.copy() for k, v in global_sd.items()}

    def get_local_weights(self) -> StateDict:
        return self.model.state_dict()

    # ------------------------------------------------------------------ training

    def train_round(
        self,
        epochs: int = 1,
        lr:     float = 1e-3,
        criterion = None,
    ) -> Tuple[StateDict, Dict]:
        """
        Run local training for one FL round.

        Returns (local_state_dict, metrics_dict).
        """
        if criterion is None:
            criterion = nn.MSELoss() if hasattr(self.model, 'encoder') \
                        else nn.CrossEntropyLoss()

        # Build optimizer
        base_opt = optim.Adam(list(self.model.parameters()), lr=lr)

        if self.privacy_config.enabled:
            n_samples = len(self.train_loader.dataset) \
                        if hasattr(self.train_loader, 'dataset') else 1000
            from picograd.privacy import RDPAccountant
            accountant = RDPAccountant()
            optimizer  = DPOptimizer(
                base_opt,
                noise_multiplier = self.privacy_config.noise_multiplier,
                max_grad_norm    = self.privacy_config.max_grad_norm,
                batch_size       = self.train_loader.batch_size \
                                   if hasattr(self.train_loader,'batch_size') else 32,
                dataset_size     = n_samples,
                accountant       = accountant,
            )
        else:
            optimizer  = base_opt
            accountant = None

        # FedProx helper
        prox = None
        if self.use_fedprox and self._global_sd is not None:
            from server.strategies.fedprox_fedbn import FedProxLoss
            prox = FedProxLoss(self.fedprox_mu, self.model.parameters())

        self.model.train()
        total_loss  = 0.0
        total_steps = 0

        for epoch in range(epochs):
            for batch in self.train_loader:
                x, y = batch if isinstance(batch, (list, tuple)) else (batch, batch)
                self.model.zero_grad()
                out  = self.model(x)

                # Primary loss
                if isinstance(criterion, nn.MSELoss):
                    loss = criterion(out, x)   # autoencoder: reconstruct input
                else:
                    loss = criterion(out, y)

                # FedProx proximal term
                if prox is not None:
                    loss = loss + prox(self.model.parameters())

                loss.backward()
                optimizer.step()
                total_loss  += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)

        # Privacy budget
        epsilon = None
        if accountant is not None:
            epsilon = accountant.get_epsilon(self.privacy_config.target_delta)

        metrics = {
            'client_id':  self.client_id,
            'round':      self._round,
            'train_loss': avg_loss,
            'epsilon':    epsilon,
            'n_samples':  total_steps,
        }

        # Validation loss
        if self.val_loader is not None:
            metrics['val_loss'] = self._eval(criterion)

        self._round += 1
        return self.get_local_weights(), metrics

    def _eval(self, criterion) -> float:
        self.model.eval()
        total = 0.0
        n     = 0
        with picograd.no_grad():
            for batch in self.val_loader:
                x, y = batch if isinstance(batch, (list, tuple)) else (batch, batch)
                out  = self.model(x)
                loss = criterion(out, x if isinstance(criterion, nn.MSELoss) else y)
                total += loss.item()
                n += 1
        self.model.train()
        return total / max(n, 1)

    # ------------------------------------------------------------------ anomaly detection

    def compute_anomaly_threshold(self, percentile: float = 95.0) -> float:
        """Compute local anomaly threshold from validation data."""
        if self.val_loader is None:
            raise RuntimeError("val_loader required for threshold computation")
        errors = []
        self.model.eval()
        with picograd.no_grad():
            for batch in self.val_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                rec  = self.model(x)
                err  = ((x._data - rec._data)**2).mean(axis=(1,2,3))
                errors.extend(err.tolist())
        return float(np.percentile(errors, percentile))


__all__ = ["SiteClient"]
