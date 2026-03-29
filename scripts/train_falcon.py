#!/usr/bin/env python3
"""
scripts/train_falcon.py
========================
Main FALCON federated learning training script.

Orchestrates the full pipeline:
  1. Data partitioning (Dirichlet non-IID)
  2. Global model initialization (AnomalyAE)
  3. FL server initialization
  4. Per-round: client local training -> FedAvg -> dashboard broadcast
  5. Anomaly detection evaluation (AUROC)
  6. Grad-CAM explainability
  7. Privacy budget reporting

Usage:
  python3 scripts/train_falcon.py \\
      --rounds 20 --clients 3 --dataset BreastMNIST \\
      --strategy fedavg --dp --secure-agg \\
      --latent-dim 64 --epochs 2 --lr 1e-3 \\
      --alpha 0.5 --dashboard

  # Quick smoke test:
  python3 scripts/train_falcon.py --rounds 3 --clients 2 --quick
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import picograd
import picograd.nn as nn
import picograd.optim as optim
from picograd import Tensor, manual_seed
from picograd.models.anomaly_ae import AnomalyAE, AnomalyDetector
from picograd.privacy import DPOptimizer, RDPAccountant, PrivacyConfig
from picograd.explain.gradcam import GradCAM
from picograd.data.medmnist_dataset import MedMNISTDataset, build_medmnist_loaders
from picograd.data.dataloader import DataLoader, NumpyDataset
from server.grpc_server import InProcessFLServer
from server.fedavg import fedavg
from falcon.data_partition import dirichlet_partition, partition_stats


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FALCON FL Training")
    p.add_argument("--rounds",      type=int,   default=10)
    p.add_argument("--clients",     type=int,   default=3)
    p.add_argument("--dataset",     type=str,   default="BreastMNIST")
    p.add_argument("--strategy",    type=str,   default="fedavg",
                   choices=["fedavg","fedprox","fedbn"])
    p.add_argument("--latent-dim",  type=int,   default=32)
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch",       type=int,   default=16)
    p.add_argument("--alpha",       type=float, default=0.5,
                   help="Dirichlet concentration (lower=more non-IID)")
    p.add_argument("--dp",          action="store_true", help="Enable DP")
    p.add_argument("--noise-mult",  type=float, default=1.0)
    p.add_argument("--max-grad-norm",type=float,default=1.0)
    p.add_argument("--secure-agg",  action="store_true", help="Secure aggregation")
    p.add_argument("--dashboard",   action="store_true", help="Enable dashboard")
    p.add_argument("--quick",       action="store_true",
                   help="Fast smoke test: 2 rounds, 2 clients, tiny model")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--save-model",  type=str,   default=None)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Helper: build synthetic data if MedMNIST unavailable
# -----------------------------------------------------------------------------

def build_data(args):
    """Build train/val/test data. Falls back to synthetic if MedMNIST unavailable."""
    try:
        train_loader, val_loader, test_loader = build_medmnist_loaders(
            args.dataset, batch_size=args.batch, download=True
        )
        # Get all training images as numpy for partitioning
        X_train = np.concatenate([b[0]._data for b in train_loader], axis=0)
        print(f"  Loaded MedMNIST/{args.dataset}: {len(X_train)} training samples")
    except Exception:
        # Synthetic fallback
        n = args.clients * 150
        X_train = np.random.rand(n, 1, 28, 28).astype(np.float32)
        Y_dummy = np.zeros(n, dtype=np.int32)
        ds_train = NumpyDataset(X_train, Y_dummy)
        ds_val   = NumpyDataset(X_train[:50],  Y_dummy[:50])
        ds_test  = NumpyDataset(X_train[-50:], Y_dummy[-50:])
        train_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True)
        val_loader   = DataLoader(ds_val,   batch_size=args.batch)
        test_loader  = DataLoader(ds_test,  batch_size=args.batch)
        print(f"  Synthetic data: {n} samples x 1x28x28")

    return X_train, train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.quick:
        args.rounds    = 3
        args.clients   = 2
        args.latent_dim= 8
        args.epochs    = 1
        args.batch     = 8

    manual_seed(args.seed)

    print("\n" + "="*58)
    print("  FALCON -- Federated Anomaly Learning for Clinical Omni-detection")
    print("="*58)
    print(f"  Rounds:      {args.rounds}   Strategy: {args.strategy}")
    print(f"  Clients:     {args.clients}  Non-IID alpha: {args.alpha}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  DP:          {'ON (sigma={:.1f}, C={:.1f})'.format(args.noise_mult, args.max_grad_norm) if args.dp else 'OFF'}")
    print(f"  SecureAgg:   {'ON' if args.secure_agg else 'OFF'}")
    print(f"  Latent dim:  {args.latent_dim}")
    print("="*58 + "\n")

    # -- Dashboard bridge -------------------------------------------------
    bridge = None
    if args.dashboard:
        from dashboard.backend.services.fl_bridge import FLDashboardBridge
        bridge = FLDashboardBridge()
        bridge.on_training_start({
            "strategy":     args.strategy,
            "total_rounds": args.rounds,
        })

    # -- Data -------------------------------------------------------------
    X_train, train_loader, val_loader, test_loader = build_data(args)

    # -- Non-IID partition -------------------------------------------------
    Y_dummy = np.zeros(len(X_train), dtype=np.int32)
    partitions = dirichlet_partition(Y_dummy, n_clients=args.clients,
                                      alpha=args.alpha, seed=args.seed)
    print(f"  Data partitioned: {[len(p) for p in partitions]} samples per client\n")

    # -- Global model ------------------------------------------------------
    global_model = AnomalyAE(in_channels=1, latent_dim=args.latent_dim)
    n_params = sum(p.numel for p in global_model.parameters())
    print(f"  Model: AnomalyAE ({n_params:,} parameters)\n")

    # -- FL Server ---------------------------------------------------------
    fl_server = InProcessFLServer(
        global_model,
        strategy     = args.strategy,
        min_clients  = args.clients,
        total_rounds = args.rounds,
        secure_agg   = args.secure_agg,
        dp_enabled   = args.dp,
    )

    # Register all clients
    for c in range(args.clients):
        fl_server.register(f"client_{c}", n_samples=len(partitions[c]),
                           dataset_name=args.dataset)

    # -- Privacy -----------------------------------------------------------
    privacy_cfg = PrivacyConfig(
        noise_multiplier = args.noise_mult,
        max_grad_norm    = args.max_grad_norm,
        enabled          = args.dp,
    )
    accountants = [RDPAccountant() for _ in range(args.clients)]

    # -- Training loop -----------------------------------------------------
    print(f"  {'Round':>5}  {'Loss':>10}  {'eps':>8}  {'Time':>6}")
    print("  " + "-"*36)

    for rnd in range(1, args.rounds + 1):
        t0            = time.time()
        global_sd     = global_model.state_dict()
        client_updates= []
        client_counts = []
        client_metrics= []

        for c_id in range(args.clients):
            idx    = partitions[c_id]
            X_c    = X_train[idx]

            local_model = AnomalyAE(in_channels=1, latent_dim=args.latent_dim)
            local_model.load_state_dict(global_sd)
            local_model.train()

            base_opt = optim.Adam(list(local_model.parameters()), lr=args.lr)

            if args.dp:
                optimizer = DPOptimizer(
                    base_opt,
                    noise_multiplier = args.noise_mult,
                    max_grad_norm    = args.max_grad_norm,
                    batch_size       = args.batch,
                    dataset_size     = len(idx),
                    accountant       = accountants[c_id],
                )
            else:
                optimizer = base_opt

            criterion = nn.MSELoss()

            # Local training
            epoch_loss = 0.0
            n_steps    = 0
            for epoch in range(args.epochs):
                for step_start in range(0, len(X_c), args.batch):
                    x_np  = X_c[step_start:step_start + args.batch]
                    if len(x_np) < 2:
                        continue
                    x     = Tensor(x_np)
                    local_model.zero_grad()
                    recon = local_model(x)
                    loss  = criterion(recon, x)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_steps    += 1

            avg_loss = epoch_loss / max(n_steps, 1)
            eps      = accountants[c_id].get_epsilon(1e-5) if args.dp else 0.0

            client_updates.append(local_model.state_dict())
            client_counts.append(len(idx))
            client_metrics.append({
                "client_id":  f"client_{c_id}",
                "train_loss": avg_loss,
                "epsilon":    eps,
            })

            # Heartbeat
            fl_server.record_heartbeat(f"client_{c_id}", {
                "train_loss": avg_loss, "epsilon": eps, "round": rnd
            })

        # FedAvg aggregation
        prev_sd = global_model.state_dict()
        agg_sd  = fedavg(client_updates, client_counts)
        global_model.load_state_dict(agg_sd)

        # Metrics
        avg_loss_round = float(np.mean([m["train_loss"] for m in client_metrics]))
        avg_eps_round  = float(np.mean([m["epsilon"] for m in client_metrics]))
        elapsed        = time.time() - t0

        print(f"  {rnd:>5}  {avg_loss_round:>10.5f}  "
              f"{'N/A':>8}" if not args.dp else
              f"  {rnd:>5}  {avg_loss_round:>10.5f}  {avg_eps_round:>8.3f}  {elapsed:>5.1f}s")

        # Dashboard
        if bridge:
            bridge.on_round_complete(rnd, global_model, client_metrics, prev_sd)
            if args.dp:
                bridge.on_privacy_update(rnd, avg_eps_round)

    print("\n" + "="*58)
    print("  Training complete!")

    # -- Anomaly threshold calibration -------------------------------------
    print("\n  Calibrating anomaly threshold on validation set...")
    detector = AnomalyDetector()
    detector.model = global_model
    threshold = detector.set_threshold(val_loader, percentile=95.0)
    print(f"  Threshold (95th percentile): {threshold:.6f}")

    # -- Grad-CAM on a sample ----------------------------------------------
    print("\n  Computing Grad-CAM on sample image...")
    sample_batch = next(iter(test_loader))
    x_sample     = Tensor(sample_batch[0]._data[:1])   # first image

    gcam    = GradCAM(global_model, target_layer=global_model.encoder.conv3)
    heatmap = gcam.compute(x_sample)
    print(f"  Heatmap: shape={heatmap.shape}, max={heatmap.max():.3f}")

    if bridge:
        bridge.on_gradcam("client_0", heatmap, anomaly_score=threshold,
                          original=x_sample._data[0,0])

    # -- Privacy summary ---------------------------------------------------
    if args.dp:
        final_eps = max(acc.get_epsilon(1e-5) for acc in accountants)
        print(f"\n  Privacy budget: eps = {final_eps:.3f} (delta = 1e-5)")

    # -- Save model --------------------------------------------------------
    if args.save_model:
        picograd.save(global_model.state_dict(), args.save_model)
        print(f"\n  Model saved -> {args.save_model}")

    print("="*58 + "\n")

    if bridge:
        bridge.on_training_done()


if __name__ == "__main__":
    main()
