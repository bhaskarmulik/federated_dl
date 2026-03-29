"""
dashboard/backend/services/fl_bridge.py
=========================================
Bridges the FL training loop (server/aggregator.py) to the
dashboard backend (metrics_store + WebSocket broadcasts).

Usage in FL training loop:
    bridge = FLDashboardBridge(app)
    # ... FL round ...
    bridge.on_round_complete(round_num, global_model, client_metrics)
    bridge.on_privacy_update(round_num, epsilon)
"""

from __future__ import annotations
import numpy as np
import base64
import io
from typing import Any, Dict, List, Optional

from dashboard.backend.services.metrics_store import get_store


class FLDashboardBridge:
    """
    Connects FL training events to dashboard metrics store + WebSocket.

    Call methods from within the FL training loop to push live updates
    to the dashboard frontend.
    """

    def __init__(self, app=None):
        self._app   = app
        self._store = get_store()

    # ─── FL lifecycle ────────────────────────────────────────────────────────

    def on_training_start(self, config: Dict) -> None:
        self._store.set_server_status("running",
                                      strategy=config.get("strategy", "fedavg"),
                                      total_rounds=config.get("total_rounds", 10))
        self._emit("server_status", self._store.server)

    def on_training_done(self) -> None:
        self._store.set_server_status("done")
        self._emit("server_status", self._store.server)

    # ─── Round events ────────────────────────────────────────────────────────

    def on_round_start(self, round_num: int) -> None:
        self._store.set_server_status("running", round=round_num)

    def on_round_complete(
        self,
        round_num:      int,
        global_model,
        client_metrics: List[Dict],
        prev_global_sd: Optional[Dict] = None,
    ) -> None:
        """
        Called after each FL aggregation round.

        Parameters
        ----------
        round_num      : current FL round number
        global_model   : picograd nn.Module — the updated global model
        client_metrics : list of per-client metric dicts from SiteClient.train_round()
        prev_global_sd : previous global state_dict (for weight delta computation)
        """
        # Aggregate client losses
        losses = [m.get("train_loss", 0) for m in client_metrics if "train_loss" in m]
        avg_loss = float(np.mean(losses)) if losses else 0.0

        metrics = {
            "global_loss":   avg_loss,
            "n_clients":     len(client_metrics),
            "client_losses": {m.get("client_id", str(i)): m.get("train_loss")
                              for i, m in enumerate(client_metrics)},
        }

        # Weight deltas vs previous round
        if prev_global_sd is not None:
            curr_sd   = global_model.state_dict()
            layer_deltas = {}
            for k in curr_sd:
                if k in prev_global_sd:
                    delta = np.mean(np.abs(
                        curr_sd[k].astype(np.float64) -
                        prev_global_sd[k].astype(np.float64)
                    ))
                    layer_deltas[k] = float(delta)
            self._store.record_weight_delta(round_num, layer_deltas)
            self._emit("weight_update", {"round": round_num, "deltas": layer_deltas})

        self._store.record_round(round_num, metrics)
        self._emit("round_complete", {"round": round_num, **metrics})

    # ─── Client events ───────────────────────────────────────────────────────

    def on_client_connect(self, client_id: str, **meta) -> None:
        self._store.register_client(client_id, **meta)
        self._emit("client_update", self._store.get_clients())

    def on_client_update(self, client_id: str, metrics: Dict = None, **kwargs) -> None:
        combined = {**(metrics or {}), **kwargs}
        self._store.update_client(client_id, **combined)
        self._emit("client_update", self._store.get_clients())

    def on_client_disconnect(self, client_id: str) -> None:
        self._store.update_client(client_id, status="disconnected")
        self._emit("client_update", self._store.get_clients())

    # ─── Privacy ─────────────────────────────────────────────────────────────

    def on_privacy_update(self, round_num: int, epsilon: float,
                          delta: float = 1e-5) -> None:
        self._store.record_privacy(round_num, epsilon, delta)
        self._emit("privacy_update", {
            "round":   round_num,
            "epsilon": epsilon,
            "delta":   delta,
        })

    # ─── Grad-CAM ────────────────────────────────────────────────────────────

    def on_gradcam(self, client_id: str,
                   heatmap: np.ndarray,
                   anomaly_score: float,
                   original: Optional[np.ndarray] = None) -> None:
        """
        Push a Grad-CAM heatmap to the dashboard.

        Parameters
        ----------
        heatmap       : (H,W) float32 array in [0,1]
        anomaly_score : scalar reconstruction error
        original      : (H,W) original input image (optional)
        """
        payload = {
            "heatmap_b64":   self._ndarray_to_b64(heatmap),
            "anomaly_score": anomaly_score,
        }
        if original is not None:
            payload["original_b64"] = self._ndarray_to_b64(original)

        self._store.update_gradcam(client_id, payload)
        self._emit("gradcam_update", {"client_id": client_id, **payload})

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ndarray_to_b64(arr: np.ndarray) -> str:
        """Encode a float [0,1] numpy array as a base64 PNG string."""
        try:
            import struct, zlib

            # Normalize to uint8
            img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            if img.ndim == 2:
                # Convert grayscale to RGB
                img = np.stack([img]*3, axis=-1)

            H, W, C = img.shape

            def png_chunk(name: bytes, data: bytes) -> bytes:
                crc = zlib.crc32(name + data) & 0xFFFFFFFF
                return struct.pack('>I', len(data)) + name + data + struct.pack('>I', crc)

            ihdr_data = struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)
            ihdr = png_chunk(b'IHDR', ihdr_data)

            raw_rows = b''
            for row in img:
                raw_rows += b'\x00' + row.tobytes()
            idat = png_chunk(b'IDAT', zlib.compress(raw_rows))
            iend = png_chunk(b'IEND', b'')

            png_bytes = b'\x89PNG\r\n\x1a\n' + ihdr + idat + iend
            return base64.b64encode(png_bytes).decode('ascii')

        except Exception:
            # Fallback: plain base64 of raw float32 bytes
            return base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')

    def _emit(self, event: str, data: Any) -> None:
        """Emit a WebSocket event if socketio is available."""
        if self._app is not None and hasattr(self._app, 'broadcast_' + event.replace('_', '_')):
            pass  # app has dedicated broadcast methods
        # Direct socketio emit (if attached)
        try:
            socketio = getattr(self._app, '_socketio', None)
            if socketio:
                socketio.emit(event, data)
        except Exception:
            pass


__all__ = ["FLDashboardBridge"]
