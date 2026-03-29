"""
dashboard/backend/services/metrics_store.py
=============================================
Thread-safe in-memory store for all FL training metrics.
Acts as the single source of truth for the dashboard backend.
Designed for real-time updates via WebSocket.
"""

from __future__ import annotations
import threading
import time
from typing import Any, Dict, List, Optional
from collections import deque


class MetricsStore:
    """
    Central in-memory state store for the FALCON dashboard.

    Stores:
      - Server status (running, round, strategy)
      - Connected clients (id, status, last_seen, metrics)
      - Per-round training metrics (loss, accuracy, epsilon)
      - Weight delta snapshots (per-layer per-round)
      - Grad-CAM images (base64 encoded per client)
      - Privacy budget timeline
      - Docker container states
    """

    def __init__(self, max_rounds: int = 500):
        self._lock      = threading.RLock()
        self._max       = max_rounds

        # -- Server state ----------------------------------------------
        self.server: Dict[str, Any] = {
            "status":     "idle",       # idle | running | aggregating | done
            "strategy":   "fedavg",
            "round":      0,
            "total_rounds": 0,
            "start_time": None,
            "secure_agg": False,
            "dp_enabled": False,
        }

        # -- Clients ---------------------------------------------------
        # client_id -> {status, last_seen, round, train_loss, epsilon, n_samples}
        self.clients: Dict[str, Dict[str, Any]] = {}

        # -- Round-level metrics ---------------------------------------
        self.rounds: List[Dict[str, Any]] = []   # list of round dicts

        # -- Weight deltas per layer per round -------------------------
        # layer_name -> deque of (round, mean_abs_delta)
        self.weight_deltas: Dict[str, deque] = {}

        # -- Grad-CAM images -------------------------------------------
        # client_id -> {"heatmap_b64": str, "anomaly_score": float, ...}
        self.gradcam: Dict[str, Dict[str, Any]] = {}

        # -- Privacy timeline ------------------------------------------
        # list of {round, epsilon, delta}
        self.privacy_timeline: List[Dict[str, Any]] = []

        # -- Docker containers -----------------------------------------
        # container_id -> {name, status, client_id, image}
        self.containers: Dict[str, Dict[str, Any]] = {}

        # -- Event log (last 200 entries) ------------------------------
        self.event_log: deque = deque(maxlen=200)

    # -----------------------------------------------------------------
    # Server state
    # -----------------------------------------------------------------

    def set_server_status(self, status: str, **kwargs) -> None:
        with self._lock:
            self.server["status"] = status
            self.server.update(kwargs)
            if status == "running" and self.server["start_time"] is None:
                self.server["start_time"] = time.time()
            self._log(f"Server status -> {status}")

    def set_config(self, strategy: str = "fedavg",
                   secure_agg: bool = False,
                   dp_enabled: bool = False,
                   total_rounds: int = 0) -> None:
        with self._lock:
            self.server.update({
                "strategy":     strategy,
                "secure_agg":   secure_agg,
                "dp_enabled":   dp_enabled,
                "total_rounds": total_rounds,
            })

    # -----------------------------------------------------------------
    # Client management
    # -----------------------------------------------------------------

    def register_client(self, client_id: str, **meta) -> None:
        with self._lock:
            self.clients[client_id] = {
                "status":     "connected",
                "last_seen":  time.time(),
                "round":      0,
                "train_loss": None,
                "epsilon":    None,
                "n_samples":  0,
                **meta,
            }
            self._log(f"Client registered: {client_id}")

    def update_client(self, client_id: str, **updates) -> None:
        with self._lock:
            if client_id not in self.clients:
                self.register_client(client_id)
            self.clients[client_id].update({
                "last_seen": time.time(),
                **updates,
            })

    def remove_client(self, client_id: str) -> None:
        with self._lock:
            self.clients.pop(client_id, None)
            self._log(f"Client disconnected: {client_id}")

    def get_clients(self) -> List[Dict]:
        with self._lock:
            now = time.time()
            result = []
            for cid, c in self.clients.items():
                d = dict(c)
                d["client_id"] = cid
                d["online"]    = (now - c["last_seen"]) < 30  # 30s timeout
                result.append(d)
            return result

    # -----------------------------------------------------------------
    # Round metrics
    # -----------------------------------------------------------------

    def record_round(self, round_num: int, metrics: Dict[str, Any]) -> None:
        with self._lock:
            entry = {
                "round":       round_num,
                "timestamp":   time.time(),
                "global_loss": metrics.get("global_loss"),
                "global_acc":  metrics.get("global_acc"),
                "n_clients":   metrics.get("n_clients", 0),
                "epsilon":     metrics.get("epsilon"),
                "client_losses": metrics.get("client_losses", {}),
            }
            self.rounds.append(entry)
            if len(self.rounds) > self._max:
                self.rounds = self.rounds[-self._max:]
            self.server["round"] = round_num
            self._log(f"Round {round_num} recorded | loss={entry['global_loss']}")

    def get_rounds(self, last_n: Optional[int] = None) -> List[Dict]:
        with self._lock:
            data = self.rounds[-last_n:] if last_n else self.rounds
            return list(data)

    # -----------------------------------------------------------------
    # Weight deltas (for heatmap visualisation)
    # -----------------------------------------------------------------

    def record_weight_delta(self, round_num: int,
                            layer_deltas: Dict[str, float]) -> None:
        with self._lock:
            for layer, delta in layer_deltas.items():
                if layer not in self.weight_deltas:
                    self.weight_deltas[layer] = deque(maxlen=self._max)
                self.weight_deltas[layer].append((round_num, delta))

    def get_weight_heatmap(self) -> Dict[str, List]:
        """Returns {layer_name: [(round, mean_abs_delta), ...]}"""
        with self._lock:
            return {k: list(v) for k, v in self.weight_deltas.items()}

    # -----------------------------------------------------------------
    # Grad-CAM
    # -----------------------------------------------------------------

    def update_gradcam(self, client_id: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self.gradcam[client_id] = {
                "timestamp": time.time(),
                **data,
            }

    def get_gradcam(self, client_id: Optional[str] = None) -> Any:
        with self._lock:
            if client_id:
                return self.gradcam.get(client_id)
            return dict(self.gradcam)

    # -----------------------------------------------------------------
    # Privacy
    # -----------------------------------------------------------------

    def record_privacy(self, round_num: int, epsilon: float,
                       delta: float = 1e-5) -> None:
        with self._lock:
            self.privacy_timeline.append({
                "round":   round_num,
                "epsilon": epsilon,
                "delta":   delta,
                "ts":      time.time(),
            })
            if len(self.privacy_timeline) > self._max:
                self.privacy_timeline = self.privacy_timeline[-self._max:]

    def get_privacy(self) -> Dict[str, Any]:
        with self._lock:
            latest_eps = self.privacy_timeline[-1]["epsilon"] \
                         if self.privacy_timeline else 0.0
            return {
                "timeline":    list(self.privacy_timeline),
                "current_eps": latest_eps,
            }

    # -----------------------------------------------------------------
    # Docker
    # -----------------------------------------------------------------

    def update_container(self, container_id: str, **info) -> None:
        with self._lock:
            self.containers[container_id] = {
                "container_id": container_id,
                "last_seen":    time.time(),
                **info,
            }

    def remove_container(self, container_id: str) -> None:
        with self._lock:
            self.containers.pop(container_id, None)

    def get_containers(self) -> List[Dict]:
        with self._lock:
            return list(self.containers.values())

    # -----------------------------------------------------------------
    # Event log
    # -----------------------------------------------------------------

    def _log(self, message: str, level: str = "info") -> None:
        self.event_log.append({
            "ts":      time.time(),
            "level":   level,
            "message": message,
        })

    def get_log(self, last_n: int = 50) -> List[Dict]:
        with self._lock:
            entries = list(self.event_log)
            return entries[-last_n:]

    # -----------------------------------------------------------------
    # Full snapshot (for initial page load)
    # -----------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "server":    dict(self.server),
                "clients":   self.get_clients(),
                "rounds":    self.get_rounds(last_n=100),
                "privacy":   self.get_privacy(),
                "gradcam":   dict(self.gradcam),
                "containers":self.get_containers(),
                "log":       self.get_log(20),
            }


# Singleton instance shared across the application
_store = MetricsStore()


def get_store() -> MetricsStore:
    return _store


__all__ = ["MetricsStore", "get_store"]
