#!/usr/bin/env bash
# scripts/launch_demo.sh
# =======================
# Launch a full FALCON federated learning demo without Docker.
# Runs FL simulation in-process with 3 clients + dashboard.
#
# Usage:
#   chmod +x scripts/launch_demo.sh
#   ./scripts/launch_demo.sh
#   ./scripts/launch_demo.sh --rounds 5 --clients 3 --dp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# ── Parse args ────────────────────────────────────────────────────────────
ROUNDS=10
N_CLIENTS=3
DP=false
SECURE_AGG=false
STRATEGY="fedavg"
ALPHA=0.5

while [[ $# -gt 0 ]]; do
  case $1 in
    --rounds)      ROUNDS=$2;        shift 2;;
    --clients)     N_CLIENTS=$2;     shift 2;;
    --dp)          DP=true;          shift;;
    --secure-agg)  SECURE_AGG=true;  shift;;
    --strategy)    STRATEGY=$2;      shift 2;;
    --alpha)       ALPHA=$2;         shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo ""
echo "═══════════════════════════════════════════════"
echo "  FALCON FL Demo"
echo "═══════════════════════════════════════════════"
echo "  Rounds:      $ROUNDS"
echo "  Clients:     $N_CLIENTS"
echo "  Strategy:    $STRATEGY"
echo "  DP:          $DP"
echo "  SecureAgg:   $SECURE_AGG"
echo "  Non-IID α:   $ALPHA"
echo "═══════════════════════════════════════════════"
echo ""

python3 - <<PYEOF
import sys, os
sys.path.insert(0, '$ROOT')
import numpy as np

import picograd
import picograd.nn as nn
import picograd.optim as optim
from picograd import Tensor, manual_seed
from picograd.models.anomaly_ae import AnomalyAE
from picograd.privacy import DPOptimizer, RDPAccountant, PrivacyConfig
from server.fedavg import fedavg
from server.aggregator import Aggregator
from falcon.data_partition import dirichlet_partition

manual_seed(42)

# ── Hyperparameters ───────────────────────────────────────────────────────
N_ROUNDS   = $ROUNDS
N_CLIENTS  = $N_CLIENTS
USE_DP     = $([[ "$DP" == "true" ]] && echo "True" || echo "False")
USE_SECAGG = $([[ "$SECURE_AGG" == "true" ]] && echo "True" || echo "False")
STRATEGY   = '$STRATEGY'
ALPHA      = $ALPHA
LATENT_DIM = 16    # small for fast demo
IMG_SIZE   = 28
BATCH      = 8
LOCAL_STEPS= 5
LR         = 1e-3

print(f"  Generating synthetic 28×28 grayscale data ...")
# Synthetic "medical" data: Gaussian blobs for normal/anomalous classes
N_TOTAL = N_CLIENTS * 200
X_all   = np.random.randn(N_TOTAL, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
X_all   = (X_all - X_all.min()) / (X_all.max() - X_all.min())  # → [0,1]
Y_all   = np.zeros(N_TOTAL, dtype=np.int32)

# Non-IID partition
parts = dirichlet_partition(Y_all, n_clients=N_CLIENTS, alpha=ALPHA)

# ── Initialize global model ───────────────────────────────────────────────
global_model = AnomalyAE(in_channels=1, latent_dim=LATENT_DIM)
aggregator   = Aggregator(strategy=STRATEGY, secure_agg=USE_SECAGG)

print(f"  Global model: {sum(p.numel for p in global_model.parameters())} parameters")
print()

# ── Privacy setup ─────────────────────────────────────────────────────────
privacy_cfg = PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, enabled=USE_DP)
accountant  = RDPAccountant() if USE_DP else None

# ── Training loop ─────────────────────────────────────────────────────────
print(f"{'Round':>6}  {'Avg Loss':>10}  {'ε':>8}  {'Clients':>8}")
print("─" * 42)

for rnd in range(1, N_ROUNDS + 1):
    global_sd      = global_model.state_dict()
    client_updates = []
    client_counts  = []
    round_losses   = []

    for c_id in range(N_CLIENTS):
        idx   = parts[c_id]
        X_c   = X_all[idx]

        # Local model = copy of global
        local_m = AnomalyAE(in_channels=1, latent_dim=LATENT_DIM)
        local_m.load_state_dict(global_sd)
        local_m.train()

        # Build optimizer (with or without DP)
        base_opt = optim.Adam(list(local_m.parameters()), lr=LR)
        if USE_DP:
            optimizer = DPOptimizer(base_opt, noise_multiplier=1.0,
                                    max_grad_norm=1.0, batch_size=BATCH,
                                    dataset_size=len(idx), accountant=accountant)
        else:
            optimizer = base_opt

        crit = nn.MSELoss()

        # Local training
        step_loss = 0.0
        for step in range(LOCAL_STEPS):
            b_idx   = np.random.choice(len(X_c), BATCH, replace=True)
            x_batch = Tensor(X_c[b_idx])
            local_m.zero_grad()
            recon   = local_m(x_batch)
            loss    = crit(recon, x_batch)
            loss.backward()
            optimizer.step()
            step_loss += loss.item()

        round_losses.append(step_loss / LOCAL_STEPS)
        client_updates.append(local_m.state_dict())
        client_counts.append(len(idx))

    # Aggregate
    agg_sd = fedavg(client_updates, client_counts)
    global_model.load_state_dict(agg_sd)

    avg_loss = float(np.mean(round_losses))
    eps      = accountant.get_epsilon(1e-5) if accountant else 0.0
    eps_str  = f"{eps:.3f}" if USE_DP else "N/A"

    print(f"{rnd:>6}  {avg_loss:>10.5f}  {eps_str:>8}  {N_CLIENTS:>8}")

print()
print("═══════════════════════════════════════════════")
print("  Demo complete!")
if USE_DP:
    final_eps = accountant.get_epsilon(1e-5)
    print(f"  Final privacy budget: ε = {final_eps:.3f} (δ = 1e-5)")
print("═══════════════════════════════════════════════")
PYEOF
