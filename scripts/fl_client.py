#!/usr/bin/env python3
"""
scripts/fl_client.py
=====================
FALCON Federated Learning Client - runs on Laptop 2 (or any remote machine).

Connects to fl_server.py running on Laptop 1 via TCP socket.
No gRPC, no extra dependencies beyond numpy.

Usage (on Laptop 2, after server is running on Laptop 1):
    python scripts/fl_client.py --server-ip 192.168.1.100 --port 9999

Optional flags:
    --client-id  myname        (default: hostname)
    --n-samples  500           (number of local training samples)
    --dataset    BreastMNIST   (auto-downloaded or synthetic fallback)
    --alpha      0.5           (Dirichlet non-IID concentration)
    --dp                       (enable Differential Privacy)
    --noise-mult 1.0           (DP noise multiplier sigma)
"""

import sys
import os
import socket
import pickle
import time
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from picograd.models.anomaly_ae import AnomalyAE
from picograd.privacy import DPOptimizer, RDPAccountant
from picograd import Tensor, manual_seed
import picograd.nn as nn
import picograd.optim as optim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [CLIENT]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fl_client")

HEADER_SIZE = 8


# ---------------------------------------------------------------------------
# Socket message helpers (same as server)
# ---------------------------------------------------------------------------

def send_msg(sock, obj):
    data   = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    length = len(data).to_bytes(HEADER_SIZE, "big")
    sock.sendall(length + data)


def recv_msg(sock):
    header = b""
    while len(header) < HEADER_SIZE:
        chunk = sock.recv(HEADER_SIZE - len(header))
        if not chunk:
            raise ConnectionError("Socket closed while reading header")
        header += chunk
    payload_len = int.from_bytes(header, "big")
    data = b""
    while len(data) < payload_len:
        chunk = sock.recv(min(65536, payload_len - len(data)))
        if not chunk:
            raise ConnectionError("Socket closed while reading payload")
        data += chunk
    return pickle.loads(data)


def send_cmd(sock, payload):
    """Send a command and wait for the server's response."""
    send_msg(sock, payload)
    return recv_msg(sock)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_local_data(dataset_name, n_samples, alpha, seed):
    """
    Load local training data.
    Tries MedMNIST first; falls back to synthetic Gaussian data.
    """
    try:
        import medmnist
        DataClass = getattr(medmnist, dataset_name)
        ds   = DataClass(split="train", download=True)
        imgs = ds.imgs.astype(np.float32) / 255.0
        if imgs.ndim == 3:
            imgs = imgs[:, np.newaxis, :, :]        # (N,1,H,W)
        elif imgs.ndim == 4 and imgs.shape[-1] in (1, 3):
            imgs = imgs.transpose(0, 3, 1, 2)       # (N,C,H,W)
        # Simulate non-IID: take a Dirichlet-weighted subset
        rng = np.random.default_rng(seed)
        n   = min(n_samples, len(imgs))
        idx = rng.choice(len(imgs), n, replace=False)
        X   = imgs[idx]
        log.info("Loaded %s: %d samples  shape=%s", dataset_name, len(X), X.shape)
        return X
    except Exception as e:
        log.warning("MedMNIST not available (%s) -- using synthetic data", e)
        rng = np.random.default_rng(seed)
        X   = rng.random((n_samples, 1, 28, 28)).astype(np.float32)
        log.info("Synthetic data: %d samples  shape=%s", n_samples, X.shape)
        return X


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def local_train(model, X, lr, local_epochs, batch_size,
                use_dp, noise_mult, max_grad_norm, accountant):
    """Run local training for one FL round."""
    crit = nn.MSELoss()
    base_opt = optim.Adam(list(model.parameters()), lr=lr)

    if use_dp:
        optimizer = DPOptimizer(
            base_opt,
            noise_multiplier = noise_mult,
            max_grad_norm    = max_grad_norm,
            batch_size       = batch_size,
            dataset_size     = len(X),
            accountant       = accountant,
        )
    else:
        optimizer = base_opt

    model.train()
    total_loss = 0.0
    n_steps    = 0

    for epoch in range(local_epochs):
        # Shuffle
        idx = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch_idx = idx[start:start + batch_size]
            if len(batch_idx) < 2:
                continue
            x    = Tensor(X[batch_idx])
            model.zero_grad()
            recon = model(x)
            loss  = crit(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_steps    += 1

    avg_loss = total_loss / max(n_steps, 1)
    return avg_loss


# ---------------------------------------------------------------------------
# Main client loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FALCON FL Client")
    parser.add_argument("--server-ip",    required=True,
                        help="IP address of the FL server (Laptop 1)")
    parser.add_argument("--port",         type=int,   default=9999)
    parser.add_argument("--client-id",    type=str,   default=None,
                        help="Unique client identifier (default: hostname)")
    parser.add_argument("--n-samples",    type=int,   default=300)
    parser.add_argument("--dataset",      type=str,   default="BreastMNIST")
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--dp",           action="store_true")
    parser.add_argument("--noise-mult",   type=float, default=1.0)
    parser.add_argument("--max-grad-norm",type=float, default=1.0)
    parser.add_argument("--seed",         type=int,   default=None)
    args = parser.parse_args()

    if args.client_id is None:
        import platform
        args.client_id = platform.node()

    seed = args.seed or int(time.time()) % 10000
    manual_seed(seed)

    print()
    print("=" * 54)
    print("  FALCON FL Client")
    print("=" * 54)
    print(f"  Client ID:   {args.client_id}")
    print(f"  Server:      {args.server_ip}:{args.port}")
    print(f"  Dataset:     {args.dataset}  (n={args.n_samples})")
    print(f"  Non-IID a:   {args.alpha}")
    print(f"  DP:          {'ON (noise=%.1f)' % args.noise_mult if args.dp else 'OFF'}")
    print("=" * 54)
    print()

    # ── Connect to server ──────────────────────────────────────────────
    log.info("Connecting to %s:%d ...", args.server_ip, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((args.server_ip, args.port))
    sock.settimeout(None)
    log.info("Connected.")

    # ── Register ───────────────────────────────────────────────────────
    resp = send_cmd(sock, {
        "cmd":       "register",
        "client_id": args.client_id,
        "n_samples": args.n_samples,
        "dataset":   args.dataset,
    })
    assert resp["ok"], f"Registration failed: {resp}"

    total_rounds = resp["total_rounds"]
    latent_dim   = resp.get("latent_dim",   16)
    lr           = resp.get("lr",           args.__dict__.get("lr", 1e-3))
    local_epochs = resp.get("local_epochs", 1)

    log.info("Registered. Total rounds: %d  latent_dim: %d", total_rounds, latent_dim)

    # ── Load local data ────────────────────────────────────────────────
    X_local = load_local_data(args.dataset, args.n_samples, args.alpha, seed)

    # ── Privacy accountant ─────────────────────────────────────────────
    accountant = RDPAccountant() if args.dp else None

    # ── FL rounds ──────────────────────────────────────────────────────
    log.info("Starting FL rounds ...")
    print()
    print(f"  {'Round':>6}  {'Loss':>10}  {'Epsilon':>8}")
    print("  " + "-" * 30)

    for rnd in range(1, total_rounds + 1):

        # 1. Pull global model
        pull_resp = send_cmd(sock, {"cmd": "pull", "client_id": args.client_id})
        assert pull_resp["ok"]

        # Load weights into local model
        local_model = AnomalyAE(in_channels=1, latent_dim=latent_dim)
        local_model.load_state_dict(pull_resp["state_dict"])

        # 2. Local training
        t0 = time.time()
        avg_loss = local_train(
            local_model, X_local,
            lr           = lr,
            local_epochs = local_epochs,
            batch_size   = args.batch_size,
            use_dp       = args.dp,
            noise_mult   = args.noise_mult,
            max_grad_norm= args.max_grad_norm,
            accountant   = accountant,
        )
        elapsed = time.time() - t0

        eps = accountant.get_epsilon(1e-5) if accountant else 0.0
        eps_str = f"{eps:.3f}" if args.dp else "N/A"
        print(f"  {rnd:>6}  {avg_loss:>10.5f}  {eps_str:>8}   ({elapsed:.1f}s)")

        # 3. Push local update
        push_resp = send_cmd(sock, {
            "cmd":        "push",
            "client_id":  args.client_id,
            "state_dict": local_model.state_dict(),
            "n_samples":  len(X_local),
            "train_loss": avg_loss,
            "epsilon":    eps,
        })
        assert push_resp["ok"], f"Push failed: {push_resp}"

    # ── Done ───────────────────────────────────────────────────────────
    try:
        send_cmd(sock, {"cmd": "bye"})
    except Exception:
        pass   # server may have already closed the connection
    try:
        sock.close()
    except Exception:
        pass

    print()
    print("=" * 54)
    print("  Training complete!")
    if args.dp and accountant:
        final_eps = accountant.get_epsilon(1e-5)
        print(f"  Final privacy budget: epsilon = {final_eps:.3f}  (delta=1e-5)")
    print("=" * 54)
    print()


if __name__ == "__main__":
    main()
