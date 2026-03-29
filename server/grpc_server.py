"""
server/grpc_server.py
======================
gRPC-based Federated Learning aggregation server.

Implements the FederatedLearning service defined in proto/fl.proto:
  - Register: client enrollment + initial model distribution
  - PullGlobalModel: chunked state_dict streaming to clients
  - PushUpdate: chunked state_dict upload from clients
  - Heartbeat: live training metrics ingestion
  - TriggerAggregation: manual aggregation trigger
  - GetStatus: server health/status

The server manages the full FL round lifecycle:
  wait-for-clients -> aggregate -> broadcast -> repeat

Works without gRPC installed (falls back to in-process simulation).
"""

from __future__ import annotations
import io, pickle, hashlib, threading, time, logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from server.fedavg import fedavg
from server.aggregator import Aggregator
from dashboard.backend.services.metrics_store import get_store

logger = logging.getLogger("falcon.server")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

StateDict = Dict[str, np.ndarray]

# --- Chunk size for model streaming ----------------------------------------
CHUNK_SIZE = 256 * 1024   # 256 KB per chunk


def _serialize_sd(sd: StateDict) -> bytes:
    return pickle.dumps(sd, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_sd(data: bytes) -> StateDict:
    return pickle.loads(data)


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _split_chunks(data: bytes) -> List[bytes]:
    return [data[i:i+CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]


# --- In-process FL server (no gRPC dependency) ----------------------------

class InProcessFLServer:
    """
    Federated Learning server that runs in-process (no gRPC overhead).
    Used for testing, demos, and Docker single-node simulation.

    For true distributed deployment, this is replaced by the gRPC server
    wrapping the same Aggregator logic.
    """

    def __init__(
        self,
        global_model,
        strategy:      str  = "fedavg",
        min_clients:   int  = 1,
        total_rounds:  int  = 10,
        secure_agg:    bool = False,
        dp_enabled:    bool = False,
    ):
        self.global_model  = global_model
        self.min_clients   = min_clients
        self.total_rounds  = total_rounds
        self.secure_agg    = secure_agg
        self.dp_enabled    = dp_enabled
        self.current_round = 0
        self.status        = "idle"

        self._aggregator   = Aggregator(strategy=strategy, secure_agg=secure_agg)
        self._clients: Dict[str, Dict] = {}  # client_id -> {session, n_samples, ...}
        self._lock         = threading.RLock()
        self._store        = get_store()
        self._round_event  = threading.Event()

        # Dashboard
        self._store.set_config(strategy=strategy, secure_agg=secure_agg,
                               dp_enabled=dp_enabled, total_rounds=total_rounds)

    # --- Client lifecycle ------------------------------------------------

    def register(self, client_id: str, n_samples: int = 100,
                 dataset_name: str = "", **kwargs) -> Dict:
        """Register a client. Returns session token + server config."""
        with self._lock:
            token = hashlib.sha256(
                f"{client_id}{time.time()}".encode()
            ).hexdigest()[:16]

            self._clients[client_id] = {
                "session_token": token,
                "n_samples":     n_samples,
                "dataset":       dataset_name,
                "status":        "connected",
                "last_seen":     time.time(),
                "round":         0,
            }
            self._store.register_client(client_id, n_samples=n_samples,
                                         dataset=dataset_name)
            logger.info(f"Client registered: {client_id} ({n_samples} samples)")

        return {
            "accepted":      True,
            "session_token": token,
            "current_round": self.current_round,
            "strategy":      self._aggregator.strategy,
            "secure_agg":    self.secure_agg,
        }

    def pull_global_model(self, client_id: str, round_num: int = -1) -> StateDict:
        """Return current global model state dict."""
        return self.global_model.state_dict()

    def push_update(self, client_id: str, state_dict: StateDict,
                    n_samples: int) -> Dict:
        """Accept a model update from a client."""
        with self._lock:
            if client_id not in self._clients:
                return {"accepted": False, "message": "Not registered"}

            self._aggregator.receive_update(client_id, state_dict, n_samples)
            self._clients[client_id]["status"] = "update_submitted"
            self._clients[client_id]["round"]  = self.current_round
            self._store.update_client(client_id, status="update_submitted",
                                       round=self.current_round)
            logger.info(f"Update received from {client_id} (round {self.current_round})")

            # Auto-aggregate when min_clients reached
            if self._aggregator.n_pending >= self.min_clients:
                self._trigger_aggregation()

        return {"accepted": True, "queue_position": self._aggregator.n_pending}

    def record_heartbeat(self, client_id: str, metrics: Dict) -> None:
        """Ingest live training metrics from a client."""
        with self._lock:
            if client_id in self._clients:
                self._clients[client_id]["last_seen"] = time.time()
                self._store.update_client(client_id, **{
                    k: v for k, v in metrics.items()
                    if k in ("train_loss", "epsilon", "step")
                })

    # --- Aggregation -----------------------------------------------------

    def _trigger_aggregation(self) -> StateDict:
        """Run FedAvg aggregation and update global model."""
        prev_sd = self.global_model.state_dict()

        self.status = "aggregating"
        self._store.set_server_status("aggregating")
        logger.info(f"Aggregating round {self.current_round + 1} "
                    f"({self._aggregator.n_pending} clients)...")

        new_sd = self._aggregator.aggregate(prev_sd)
        self.global_model.load_state_dict(new_sd)
        self.current_round += 1

        # Weight deltas
        layer_deltas = {
            k: float(np.mean(np.abs(new_sd[k].astype(np.float64) -
                                    prev_sd[k].astype(np.float64))))
            for k in new_sd if k in prev_sd
        }
        self._store.record_weight_delta(self.current_round, layer_deltas)

        # Privacy budget
        budget = self._aggregator.get_privacy_budget()
        if budget:
            eps, delta = budget
            self._store.record_privacy(self.current_round, eps, delta)
            logger.info(f"Privacy budget: eps={eps:.3f} @ delta={delta:.0e}")

        # Round metrics
        client_losses = {
            cid: self._clients[cid].get("train_loss")
            for cid in self._clients
        }
        self._store.record_round(self.current_round, {
            "global_loss": float(np.mean(
                [v for v in client_losses.values() if v is not None] or [0.0]
            )),
            "n_clients":   self._aggregator.round - 1,
        })

        self.status = "running" if self.current_round < self.total_rounds else "done"
        self._store.set_server_status(self.status, round=self.current_round)
        self._round_event.set()
        self._round_event.clear()

        logger.info(f"Round {self.current_round} complete. "
                    f"Status: {self.status}")
        return new_sd

    # --- Blocking training driver -----------------------------------------

    def wait_for_round(self, timeout: float = 300) -> bool:
        """Block until the current round's aggregation completes."""
        return self._round_event.wait(timeout=timeout)

    def get_status(self) -> Dict:
        return {
            "status":       self.status,
            "round":        self.current_round,
            "n_clients":    len(self._clients),
            "min_clients":  self.min_clients,
            "strategy":     self._aggregator.strategy,
            "secure_agg":   self.secure_agg,
            "dp_enabled":   self.dp_enabled,
        }

    def is_done(self) -> bool:
        return self.status == "done" or self.current_round >= self.total_rounds


# --- gRPC server stub (real gRPC wired up when proto compiled) -------------

def start_grpc_server(fl_server: InProcessFLServer,
                       host: str = "0.0.0.0",
                       port: int = 50051) -> None:
    """
    Start the gRPC server wrapping InProcessFLServer.
    
    Requires compiled proto stubs (proto/fl_pb2.py, fl_pb2_grpc.py).
    Run: python3 -m grpc_tools.protoc -I proto --python_out=proto 
             --grpc_python_out=proto proto/fl.proto

    Falls back gracefully if grpc is not installed.
    """
    try:
        import grpc
        from concurrent import futures
        # Import compiled proto stubs
        sys.path.insert(0, "proto")
        import fl_pb2, fl_pb2_grpc   # noqa

        class FLServicer(fl_pb2_grpc.FederatedLearningServicer):
            def __init__(self, srv): self._srv = srv

            def Register(self, request, context):
                result = self._srv.register(
                    request.client_id, request.n_samples,
                    request.dataset_name
                )
                return fl_pb2.RegisterResponse(
                    accepted=result["accepted"],
                    session_token=result["session_token"],
                    current_round=result["current_round"],
                    strategy=result["strategy"],
                )

            def GetStatus(self, request, context):
                s = self._srv.get_status()
                return fl_pb2.ServerStatus(**s)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
        fl_pb2_grpc.add_FederatedLearningServicer_to_server(
            FLServicer(fl_server), server
        )
        server.add_insecure_port(f"{host}:{port}")
        server.start()
        logger.info(f"gRPC server listening on {host}:{port}")
        server.wait_for_termination()

    except ImportError:
        logger.warning("grpc or proto stubs not available. "
                       "Run without gRPC using InProcessFLServer directly.")
    except Exception as e:
        logger.error(f"gRPC server failed: {e}")


import sys
__all__ = ["InProcessFLServer", "start_grpc_server"]
