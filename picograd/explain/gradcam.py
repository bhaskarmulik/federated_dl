"""
picograd/explain/gradcam.py
============================
Grad-CAM adapted for CNN-Autoencoders.

Standard Grad-CAM: gradients of class logits w.r.t. last conv feature map.
AE adaptation:     gradients of *reconstruction error* w.r.t. last encoder
                   conv feature map.  Highlights regions the model struggles
                   to reconstruct — i.e., anomalous regions.

Reference:  Selvaraju et al., "Grad-CAM" (2017).
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any

import picograd
import picograd.nn as nn
from picograd.tensor import Tensor


class GradCAM:
    """
    Grad-CAM for CNN-Autoencoders.

    Hooks into a target convolutional layer to capture:
      - forward activations  (feature maps)
      - backward gradients   (grad of loss w.r.t. feature maps)

    Usage:
        gcam = GradCAM(model.encoder, target_layer=model.encoder.conv3)
        heatmap = gcam.compute(image, reconstruction)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations: Optional[np.ndarray] = None
        self._gradients:   Optional[np.ndarray] = None

    # ------------------------------------------------------------------ manual hook approach
    # picograd does not have PyTorch hooks; we implement via subclassing the
    # forward pass and capturing intermediate activations directly.

    def compute(self, x: Tensor, recon: Optional[Tensor] = None,
                input_size: Optional[Tuple[int,int]] = None) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap.

        Parameters
        ----------
        x       : input image tensor (1,C,H,W) — normalised [0,1]
        recon   : reconstruction (1,C,H,W).  If None, forward is re-run.
        input_size : (H,W) for upsampling.  Defaults to x.shape[-2:]

        Returns
        -------
        heatmap : np.ndarray (H,W) — in [0,1], higher = more anomalous
        """
        H, W = input_size or (x.shape[-2], x.shape[-1])
        model = self.model

        # ── 1. Forward through encoder, capturing last conv activations ──
        model.train()   # ensure BN works

        # Run encoder up to conv3 manually, capturing activations
        enc = model.encoder if hasattr(model, 'encoder') else model

        # Forward with gradient tracking
        h = enc.relu1(enc.bn1(enc.conv1(x)))
        h = enc.relu2(enc.bn2(enc.conv2(h)))
        h_conv3_in = enc.relu3(enc.bn3(enc.conv3(h)))  # (1,128,4,4) — target activation
        h_conv3_in._data = h_conv3_in._data.copy()     # ensure own copy

        # Continue through full model
        h_flat = enc.flatten(h_conv3_in)
        z      = enc.fc(h_flat)

        dec    = model.decoder if hasattr(model, 'decoder') else model
        h_d    = dec.relu_fc(dec.fc(z))
        h_d    = h_d.reshape(h_d.shape[0], 128, 4, 4)
        h_d    = dec.relu1(dec.bn1(dec.deconv1(h_d)))
        h_d    = dec.relu2(dec.bn2(dec.deconv2(h_d)))
        recon_computed = dec.sigmoid(dec.deconv3(h_d))

        # ── 2. Compute reconstruction error as scalar target ──
        err = ((x - recon_computed) ** 2).mean()

        # ── 3. Backward to get gradients at conv3 output ──
        # We need dL/d(h_conv3_in).  Track h_conv3_in as a leaf.
        # Simplification: directly compute gradient of error w.r.t. feature map
        # via the autograd engine.
        h_conv3_in.requires_grad = True
        err.backward()

        if h_conv3_in.grad is None:
            # Fallback: use activation magnitudes as importance proxy
            activations = h_conv3_in._data[0]   # (128,4,4)
            weights = activations.mean(axis=(1,2))
        else:
            grads       = h_conv3_in.grad._data[0]  # (128,4,4)
            activations = h_conv3_in._data[0]        # (128,4,4)
            # Global Average Pool of gradients → channel weights
            weights = grads.mean(axis=(1,2))          # (128,)

        # ── 4. Weighted combination of feature maps ──
        cam = np.zeros(activations.shape[1:], dtype=np.float32)   # (4,4)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ── 5. ReLU + normalise + upsample ──
        cam = np.maximum(cam, 0)
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)

        # Bilinear upsample to (H,W)
        heatmap = self._bilinear_upsample(cam, H, W)

        model.eval()
        return heatmap

    @staticmethod
    def _bilinear_upsample(cam: np.ndarray, H: int, W: int) -> np.ndarray:
        """Simple bilinear upsampling of (h,w) → (H,W)."""
        h, w = cam.shape
        # Use numpy zoom-like approach
        row_idx = np.linspace(0, h - 1, H)
        col_idx = np.linspace(0, w - 1, W)
        # Floor indices
        r0 = np.floor(row_idx).astype(int).clip(0, h - 1)
        c0 = np.floor(col_idx).astype(int).clip(0, w - 1)
        r1 = (r0 + 1).clip(0, h - 1)
        c1 = (c0 + 1).clip(0, w - 1)
        # Fractional parts
        dr = row_idx - r0
        dc = col_idx - c0
        # Bilinear interpolation
        dr = dr[:, np.newaxis]   # (H,1)
        dc = dc[np.newaxis, :]   # (1,W)
        result = ((1 - dr) * (1 - dc) * cam[r0[:, None], c0[None, :]] +
                  (    dr) * (1 - dc) * cam[r1[:, None], c0[None, :]] +
                  (1 - dr) * (    dc) * cam[r0[:, None], c1[None, :]] +
                  (    dr) * (    dc) * cam[r1[:, None], c1[None, :]])
        return result.astype(np.float32)


class GradCAMOverlay:
    """Produces colour-mapped Grad-CAM overlay on the original image."""

    @staticmethod
    def apply_colormap_jet(heatmap: np.ndarray) -> np.ndarray:
        """Convert [0,1] grayscale heatmap to (H,W,3) jet colormap RGB."""
        h = np.clip(heatmap, 0, 1)
        r = np.clip(1.5 - np.abs(h * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(h * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(h * 4 - 1), 0, 1)
        return np.stack([r, g, b], axis=-1)

    @staticmethod
    def overlay(image: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.5) -> np.ndarray:
        """
        Blend original image with jet-colormap heatmap.

        Parameters
        ----------
        image   : (H,W) or (H,W,3) in [0,1]
        heatmap : (H,W) in [0,1]
        alpha   : blend weight for heatmap (0=image only, 1=heatmap only)

        Returns
        -------
        blended : (H,W,3) in [0,1]
        """
        H, W = heatmap.shape
        if image.ndim == 2:
            img_rgb = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:
            img_rgb = np.concatenate([image]*3, axis=-1)
        else:
            img_rgb = image

        img_rgb = np.clip(img_rgb, 0, 1)
        jet     = GradCAMOverlay.apply_colormap_jet(heatmap)
        blended = (1 - alpha) * img_rgb + alpha * jet
        return np.clip(blended, 0, 1)


class ExplainabilityReport:
    """
    Generates a structured explainability report for one AE prediction.

    Fields:
      original      : np.ndarray — original input image
      reconstruction: np.ndarray — model reconstruction
      error_map     : np.ndarray — per-pixel squared error
      heatmap       : np.ndarray — Grad-CAM heatmap [0,1]
      overlay       : np.ndarray — image + heatmap blend (RGB)
      anomaly_score : float
      threshold     : float
      is_anomalous  : bool
    """

    def __init__(self, detector, gradcam: GradCAM):
        self.detector = detector
        self.gradcam  = gradcam

    def generate(self, x: Tensor) -> Dict[str, Any]:
        """Run full explainability pipeline on a single image."""
        # 1. Get prediction
        score, rec_np, err_map = self.detector.predict(x)

        # 2. Grad-CAM heatmap
        heatmap = self.gradcam.compute(x)

        # 3. Overlay
        img_np  = x._data[0, 0] if x._data.ndim == 4 else x._data
        overlay = GradCAMOverlay.overlay(img_np, heatmap, alpha=0.5)

        return {
            "original":       x._data[0, 0] if x._data.ndim == 4 else x._data,
            "reconstruction": rec_np[0, 0]  if rec_np.ndim == 4  else rec_np,
            "error_map":      err_map[0, 0] if err_map.ndim == 4 else err_map,
            "heatmap":        heatmap,
            "overlay":        overlay,
            "anomaly_score":  score,
            "threshold":      self.detector.threshold,
            "is_anomalous":   score > (self.detector.threshold or float('inf')),
        }


__all__ = ["GradCAM", "GradCAMOverlay", "ExplainabilityReport"]
