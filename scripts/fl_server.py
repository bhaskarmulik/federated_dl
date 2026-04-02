#!/usr/bin/env python3
"""
scripts/fl_server.py
=====================
FALCON Federated Learning Server - runs on Laptop 1 (the server machine).

Uses Python's built-in socket + threading - NO gRPC, NO extra dependencies.
Everything is serialized with pickle over TCP.

Protocol (per connection):
  Client sends: pickle({ "cmd": "register"|"push"|"pull"|"status", ...payload... })
  Server sends: pickle({ "ok": True, ...response... })

Usage (on Laptop 1):
    python scripts/fl_server.py --rounds 5 --min-clients 1 --port 9999

Then on Laptop 2:
    python scripts/fl_client.py --server-ip 192.168.x.x --port 9999
"""

import sys
import os
import socket
import threading
import pickle
import time
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from picograd.models.anomaly_ae import AnomalyAE
from server.fedavg import fedavg
from picograd import manual_seed
import picograd.nn as nn
import picograd.optim as optim
from picograd import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [SERVER]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fl_server")

# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

HEADER_SIZE = 8   # 8 bytes encode the payload length


def send_msg(sock, obj):
    """Pickle obj and send with a length-prefixed header."""
    data   = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    length = len(data).to_bytes(HEADER_SIZE, "big")
    sock.sendall(length + data)


def recv_msg(sock):
    """Receive a length-prefixed message and unpickle it."""
    # Read header
    header = b""
    while len(header) < HEADER_SIZE:
        chunk = sock.recv(HEADER_SIZE - len(header))
        if not chunk:
            raise ConnectionError("Socket closed while reading header")
        header += chunk

    payload_len = int.from_bytes(header, "big")

    # Read payload
    data = b""
    while len(data) < payload_len:
        chunk = sock.recv(min(65536, payload_len - len(data)))
        if not chunk:
            raise ConnectionError("Socket closed while reading payload")
        data += chunk

    return pickle.loads(data)


# ---------------------------------------------------------------------------
# FL Server state
# ---------------------------------------------------------------------------

class FLServer:
    def __init__(self, rounds, min_clients, latent_dim, lr, local_epochs):
        self.rounds        = rounds
        self.min_clients   = min_clients
        self.latent_dim    = latent_dim
        self.lr            = lr
        self.local_epochs  = local_epochs

        self.global_model  = AnomalyAE(in_channels=1, latent_dim=latent_dim)
        self.current_round = 0
        self.status        = "waiting"

        self._lock              = threading.Lock()
        self._clients           = {}              # client_id -> {n_samples, ...}
        self._pending_updates   = []              # list of (state_dict, n_samples)
        self._round_done_event  = threading.Event()

        n_params = sum(p.numel for p in self.global_model.parameters())
        log.info("Global model: AnomalyAE  latent_dim=%d  params=%s",
                 latent_dim, f"{n_params:,}")
        log.info("FL config: rounds=%d  min_clients=%d  lr=%s",
                 rounds, min_clients, lr)

    # ------------------------------------------------------------------ client handlers

    def handle_register(self, req):
        cid  = req["client_id"]
        nsam = req.get("n_samples", 100)
        with self._lock:
            self._clients[cid] = {"n_samples": nsam, "joined": time.time()}
        log.info("Client registered: %s  (n_samples=%d)  total=%d/%d",
                 cid, nsam, len(self._clients), self.min_clients)
        return {
            "ok":            True,
            "current_round": self.current_round,
            "total_rounds":  self.rounds,
            "latent_dim":    self.latent_dim,
            "lr":            self.lr,
            "local_epochs":  self.local_epochs,
        }

    def handle_pull(self, req):
        with self._lock:
            sd = self.global_model.state_dict()
        return {"ok": True, "state_dict": sd, "round": self.current_round}

    def handle_push(self, req):
        cid  = req["client_id"]
        sd   = req["state_dict"]
        nsam = req.get("n_samples", 100)
        loss = req.get("train_loss", None)

        with self._lock:
            self._pending_updates.append((sd, nsam))
            n_pending = len(self._pending_updates)
            if loss is not None and cid in self._clients:
                self._clients[cid]["train_loss"] = loss

        log.info("Update received from %s  loss=%.5f  pending=%d/%d",
                 cid, loss or 0, n_pending, self.min_clients)

        # Trigger aggregation when enough clients have submitted
        if n_pending >= self.min_clients:
            self._aggregate()

        return {"ok": True, "queue": n_pending}

    def handle_status(self, req):
        with self._lock:
            return {
                "ok":            True,
                "status":        self.status,
                "round":         self.current_round,
                "total_rounds":  self.rounds,
                "n_clients":     len(self._clients),
                "min_clients":   self.min_clients,
            }

    # ------------------------------------------------------------------ aggregation

    def _aggregate(self):
        with self._lock:
            updates = list(self._pending_updates)
            self._pending_updates.clear()

        sds     = [u[0] for u in updates]
        counts  = [u[1] for u in updates]
        new_sd  = fedavg(sds, counts)

        with self._lock:
            self.global_model.load_state_dict(new_sd)
            self.current_round += 1
            rnd = self.current_round
            done = rnd >= self.rounds
            self.status = "done" if done else "waiting"

        # Eval global loss on a small random batch
        x_eval = Tensor(np.random.rand(4, 1, 28, 28).astype(np.float32))
        with __import__("picograd").no_grad():
            rec  = self.global_model(x_eval)
            loss = nn.MSELoss()(rec, x_eval).item()

        log.info("=" * 52)
        log.info("Round %d/%d complete  |  global_loss=%.6f", rnd, self.rounds, loss)
        log.info("=" * 52)

        self._round_done_event.set()
        self._round_done_event.clear()

        if done:
            log.info("All rounds complete. Server finished.")


# ---------------------------------------------------------------------------
# Connection handler (one thread per client connection)
# ---------------------------------------------------------------------------

def handle_connection(conn, addr, fl_server):
    log.info("Connection from %s:%d", addr[0], addr[1])
    try:
        while True:
            try:
                req = recv_msg(conn)
            except (ConnectionError, EOFError):
                break

            cmd = req.get("cmd", "")
            if cmd == "register":
                resp = fl_server.handle_register(req)
            elif cmd == "pull":
                resp = fl_server.handle_pull(req)
            elif cmd == "push":
                resp = fl_server.handle_push(req)
            elif cmd == "status":
                resp = fl_server.handle_status(req)
            elif cmd == "bye":
                break
            else:
                resp = {"ok": False, "error": f"Unknown command: {cmd}"}

            send_msg(conn, resp)

    except Exception as e:
        log.error("Error handling %s: %s", addr, e)
    finally:
        conn.close()
        log.info("Connection closed: %s:%d", addr[0], addr[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FALCON FL Server")
    parser.add_argument("--port",        type=int,   default=9999)
    parser.add_argument("--host",        type=str,   default="0.0.0.0")
    parser.add_argument("--rounds",      type=int,   default=5)
    parser.add_argument("--min-clients", type=int,   default=1,
                        help="Wait for this many clients before aggregating each round")
    parser.add_argument("--latent-dim",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--local-epochs",type=int,   default=1)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    manual_seed(args.seed)
    fl_server = FLServer(
        rounds       = args.rounds,
        min_clients  = args.min_clients,
        latent_dim   = args.latent_dim,
        lr           = args.lr,
        local_epochs = args.local_epochs,
    )

    # Get local IP to display
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "unknown"

    print()
    print("=" * 54)
    print("  FALCON FL Server")
    print("=" * 54)
    print(f"  Host IP:      {local_ip}  (share with clients)")
    print(f"  Port:         {args.port}")
    print(f"  Rounds:       {args.rounds}")
    print(f"  Min clients:  {args.min_clients}")
    print(f"  Latent dim:   {args.latent_dim}")
    print()
    print("  On client machine run:")
    print(f"  python scripts/fl_client.py --server-ip {local_ip} --port {args.port}")
    print("=" * 54)
    print()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(32)
    log.info("Listening on %s:%d ...", args.host, args.port)

    try:
        while fl_server.status != "done":
            conn, addr = server_sock.accept()
            t = threading.Thread(
                target=handle_connection,
                args=(conn, addr, fl_server),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        log.info("Server interrupted by user.")
    finally:
        server_sock.close()
        log.info("Server shut down.")


if __name__ == "__main__":
    main()
