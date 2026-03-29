"""
picograd/models/anomaly_ae.py
==============================
CNN-Autoencoder for reconstruction-error-based anomaly detection.

Architecture (for 28x28 grayscale input):
  Encoder:  Conv->BN->ReLU (x3, strided) -> Flatten -> Linear -> latent
  Decoder:  Linear -> Reshape -> ConvTranspose->BN->ReLU (x2) -> ConvTranspose -> Sigmoid

Training:  On "normal" samples only.  Loss = MSELoss(input, reconstruction).
Inference: anomaly_score = mean(reconstruction_error_per_pixel).
           Classified as anomalous if score > threshold.
"""

from __future__ import annotations
import numpy as np
import picograd.nn as nn
from picograd.nn.module import Module
from picograd.tensor import Tensor
from typing import Optional, Tuple


class AnomalyEncoder(Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))   # (N,32,14,14)
        x = self.relu2(self.bn2(self.conv2(x)))   # (N,64,7,7)
        x = self.relu3(self.bn3(self.conv3(x)))   # (N,128,4,4)
        x = self.flatten(x)                        # (N,2048)
        x = self.fc(x)                             # (N,latent_dim)
        return x


class AnomalyDecoder(Module):
    def __init__(self, latent_dim: int = 64, out_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc      = nn.Linear(latent_dim, 128 * 4 * 4)
        self.relu_fc = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu1   = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(32)
        self.relu2   = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(32, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: Tensor) -> Tensor:
        x = self.relu_fc(self.fc(z))               # (N,2048)
        x = x.reshape(x.shape[0], 128, 4, 4)      # (N,128,4,4)
        x = self.relu1(self.bn1(self.deconv1(x)))  # (N,64,7,7)
        x = self.relu2(self.bn2(self.deconv2(x)))  # (N,32,14,14)
        x = self.sigmoid(self.deconv3(x))          # (N,1,28,28)
        return x


class AnomalyAE(Module):
    """
    Full CNN-Autoencoder for anomaly detection.

    Input:  (N, C, H, W)  -- normalised to [0,1]
    Output: reconstruction (N, C, H, W)
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 64):
        super().__init__()
        self.encoder = AnomalyEncoder(in_channels, latent_dim)
        self.decoder = AnomalyDecoder(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def forward(self, x: Tensor) -> Tensor:
        z   = self.encoder(x)
        rec = self.decoder(z)
        return rec

    def encode(self, x: Tensor) -> Tensor:
        with picograd.no_grad():
            return self.encoder(x)

    def reconstruct(self, x: Tensor) -> Tensor:
        with picograd.no_grad():
            return self.forward(x)


import picograd


class AnomalyDetector:
    """
    High-level wrapper around AnomalyAE for clinical deployment.

    Usage:
        detector = AnomalyDetector()
        detector.fit(normal_loader, epochs=20)
        detector.set_threshold(val_loader)
        score, rec, err_map = detector.predict(image)
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 64, device=None):
        self.model     = AnomalyAE(in_channels, latent_dim)
        self.threshold: Optional[float] = None
        self._trained  = False

    def fit(self, normal_loader, epochs: int = 20, lr: float = 1e-3,
            verbose: bool = True) -> list:
        """Train autoencoder on normal samples only."""
        import picograd.optim as optim

        optimizer = optim.Adam(list(self.model.parameters()), lr=lr)
        criterion = nn.MSELoss()
        self.model.train()

        all_losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches  = 0
            for batch in normal_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                self.model.zero_grad()
                rec  = self.model(x)
                loss = criterion(rec, x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1
            avg = epoch_loss / max(n_batches, 1)
            all_losses.append(avg)
            if verbose and (epoch % max(1, epochs//5) == 0 or epoch == epochs-1):
                print(f"  [AnomalyDetector] epoch {epoch+1}/{epochs}  loss={avg:.6f}")

        self._trained = True
        return all_losses

    def set_threshold(self, val_loader, percentile: float = 95.0) -> float:
        """
        Compute anomaly threshold from a validation loader of normal samples.
        Threshold = percentile of reconstruction errors.
        """
        self.model.eval()
        errors = []
        with picograd.no_grad():
            for batch in val_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                rec = self.model(x)
                err = ((x._data - rec._data) ** 2).mean(axis=(1,2,3))
                errors.extend(err.tolist())

        self.threshold = float(np.percentile(errors, percentile))
        return self.threshold

    def predict(self, x: Tensor) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Returns:
          anomaly_score : float -- mean reconstruction MSE
          reconstruction: np.ndarray -- reconstructed image
          error_map     : np.ndarray -- per-pixel squared error
        """
        self.model.eval()
        with picograd.no_grad():
            rec = self.model(x)

        x_np   = x._data
        rec_np = rec._data
        err_map = (x_np - rec_np) ** 2            # (N,C,H,W)
        score   = float(err_map.mean())

        return score, rec_np, err_map

    def is_anomalous(self, x: Tensor) -> bool:
        if self.threshold is None:
            raise RuntimeError("Call set_threshold() first.")
        score, _, _ = self.predict(x)
        return score > self.threshold


__all__ = ["AnomalyAE", "AnomalyEncoder", "AnomalyDecoder", "AnomalyDetector"]
