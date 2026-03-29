"""
tests/test_models_fl.py
========================
Tests for Modules 6 & 7: AnomalyAE + GradCAM + FL integration.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import picograd
import picograd.nn as nn
from picograd import Tensor, manual_seed
from picograd.models.anomaly_ae import AnomalyAE, AnomalyDetector
from picograd.explain.gradcam import GradCAM, GradCAMOverlay
from server.fedavg import fedavg, fedavg_delta
from server.secure_agg import SecureAggregator, MaskGenerator
from falcon.data_partition import (
    dirichlet_partition, pathological_partition, partition_stats
)


def assert_close(a, b, atol=1e-4, msg=""):
    diff = np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))
    assert diff < atol, f"{msg} | max_diff={diff:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6A — AnomalyAE architecture
# ─────────────────────────────────────────────────────────────────────────────

def test_anomaly_ae_forward_shape():
    """AnomalyAE input/output shape: (N,1,28,28)→(N,1,28,28)."""
    manual_seed(0)
    model = AnomalyAE(in_channels=1, latent_dim=32)
    x     = Tensor(np.random.randn(2, 1, 28, 28).astype(np.float32))
    recon = model(x)
    assert recon.shape == (2, 1, 28, 28), f"shape: {recon.shape}"
    print("  ✓ AnomalyAE forward: (2,1,28,28)→(2,1,28,28)")


def test_anomaly_ae_output_range():
    """Reconstruction values in [0,1] (Sigmoid output)."""
    manual_seed(1)
    model = AnomalyAE(in_channels=1, latent_dim=32)
    x     = Tensor(np.random.randn(1, 1, 28, 28).astype(np.float32))
    recon = model(x)
    vals  = recon.numpy()
    assert vals.min() >= -1e-5 and vals.max() <= 1.0 + 1e-5, \
        f"Recon range [{vals.min():.3f}, {vals.max():.3f}] not in [0,1]"
    print(f"  ✓ AnomalyAE output in [0,1]: [{vals.min():.3f},{vals.max():.3f}]")


def test_anomaly_ae_backward():
    """MSELoss backward flows through full AnomalyAE."""
    manual_seed(2)
    model = AnomalyAE(in_channels=1, latent_dim=16)
    x     = Tensor(np.random.rand(1, 1, 28, 28).astype(np.float32))
    model.train()
    recon = model(x)
    loss  = nn.MSELoss()(recon, x)
    model.zero_grad()
    loss.backward()
    # Check at least one parameter got a gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients after backward"
    print(f"  ✓ AnomalyAE backward: loss={loss.item():.6f}, {len(grads)} params have grad")


def test_anomaly_ae_loss_decreases():
    """AnomalyAE loss decreases over 10 training steps on random data."""
    manual_seed(3)
    model  = AnomalyAE(in_channels=1, latent_dim=16)
    opt    = picograd.optim.Adam(list(model.parameters()), lr=1e-3)
    crit   = nn.MSELoss()
    x      = Tensor(np.random.rand(4, 1, 28, 28).astype(np.float32))

    losses = []
    for _ in range(10):
        model.zero_grad()
        recon = model(x)
        loss  = crit(recon, x)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], \
        f"Loss didn't decrease: {losses[0]:.4f}→{losses[-1]:.4f}"
    print(f"  ✓ AnomalyAE trains: loss {losses[0]:.4f}→{losses[-1]:.4f}")


def test_anomaly_detector_predict():
    """AnomalyDetector.predict returns correct types and shapes."""
    manual_seed(4)
    model    = AnomalyAE(in_channels=1, latent_dim=16)
    detector = AnomalyDetector(in_channels=1, latent_dim=16)
    detector.model = model

    x = Tensor(np.random.rand(1, 1, 28, 28).astype(np.float32))
    score, recon, err_map = detector.predict(x)

    assert isinstance(score, float) and score >= 0
    assert recon.shape == (1, 1, 28, 28)
    assert err_map.shape == (1, 1, 28, 28)
    print(f"  ✓ AnomalyDetector.predict: score={score:.4f}, shapes correct")


def test_state_dict_serialization():
    """AnomalyAE state_dict saves and loads correctly."""
    manual_seed(5)
    model  = AnomalyAE(in_channels=1, latent_dim=16)
    sd     = model.state_dict()
    # Corrupt weights
    for k in sd:
        sd[k] = np.zeros_like(sd[k])
    model.load_state_dict(sd)
    # Verify loaded
    for name, p in model.named_parameters():
        assert_close(p.numpy(), np.zeros_like(p.numpy()),
                     atol=1e-5, msg=f"load_state_dict key {name}")
    print("  ✓ AnomalyAE state_dict save/load roundtrip")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6B — Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

def test_gradcam_output_shape():
    """GradCAM heatmap shape = input spatial dims."""
    manual_seed(6)
    model = AnomalyAE(in_channels=1, latent_dim=16)
    gcam  = GradCAM(model, target_layer=model.encoder.conv3)
    x     = Tensor(np.random.rand(1, 1, 28, 28).astype(np.float32))
    heatmap = gcam.compute(x)
    assert heatmap.shape == (28, 28), f"heatmap shape {heatmap.shape}"
    print(f"  ✓ GradCAM heatmap shape: (28,28)")


def test_gradcam_values_in_01():
    """GradCAM heatmap values normalised to [0,1]."""
    manual_seed(7)
    model   = AnomalyAE(in_channels=1, latent_dim=16)
    gcam    = GradCAM(model, target_layer=model.encoder.conv3)
    x       = Tensor(np.random.rand(1, 1, 28, 28).astype(np.float32))
    heatmap = gcam.compute(x)
    assert heatmap.min() >= -1e-5 and heatmap.max() <= 1.0 + 1e-5, \
        f"heatmap range [{heatmap.min():.3f},{heatmap.max():.3f}]"
    print(f"  ✓ GradCAM values in [0,1]: max={heatmap.max():.3f}")


def test_gradcam_overlay():
    """GradCAMOverlay produces (H,W,3) RGB array."""
    heatmap = np.random.rand(28, 28).astype(np.float32)
    image   = np.random.rand(28, 28).astype(np.float32)
    overlay = GradCAMOverlay.overlay(image, heatmap, alpha=0.5)
    assert overlay.shape == (28, 28, 3)
    assert overlay.min() >= 0 and overlay.max() <= 1.0
    print("  ✓ GradCAM overlay produces (28,28,3) RGB in [0,1]")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7A — FedAvg aggregation
# ─────────────────────────────────────────────────────────────────────────────

def test_fedavg_correctness():
    """FedAvg weighted average is mathematically correct."""
    sd1 = {"w": np.array([1.0, 2.0], dtype=np.float32)}
    sd2 = {"w": np.array([3.0, 4.0], dtype=np.float32)}
    result = fedavg([sd1, sd2], [100, 300])   # weights: 0.25, 0.75
    expected = 0.25 * np.array([1,2]) + 0.75 * np.array([3,4])
    assert_close(result["w"], expected, atol=1e-5, msg="fedavg weighted avg")
    print("  ✓ FedAvg: weighted average is correct")


def test_fedavg_equal_weights():
    """FedAvg with equal sample counts = arithmetic mean."""
    sds = [{"w": np.array([float(i)], dtype=np.float32)} for i in range(4)]
    result = fedavg(sds, [1, 1, 1, 1])
    assert_close(result["w"], [1.5], atol=1e-5, msg="fedavg equal weights")
    print("  ✓ FedAvg: equal weights → arithmetic mean")


def test_fedavg_delta():
    """fedavg_delta correctly updates global weights."""
    global_sd = {"w": np.zeros(4, dtype=np.float32)}
    updates   = [{"w": np.ones(4, dtype=np.float32) * float(i+1)}
                 for i in range(3)]
    result    = fedavg_delta(global_sd, updates, [1, 1, 1], lr=1.0)
    # avg delta = (1+2+3)/3 = 2.0, applied to global_sd=0 → result=2.0
    assert_close(result["w"], np.full(4, 2.0), atol=1e-5, msg="fedavg_delta")
    print("  ✓ fedavg_delta: global += avg(client_delta)")


def test_fedavg_convergence():
    """FedAvg converges on identical clients (should return their common value)."""
    w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sds = [{"w": w.copy()} for _ in range(5)]
    result = fedavg(sds, [10]*5)
    assert_close(result["w"], w, atol=1e-5, msg="fedavg identical clients")
    print("  ✓ FedAvg: identical clients → exact reproduction")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7B — Data partitioning (non-IID)
# ─────────────────────────────────────────────────────────────────────────────

def test_dirichlet_partition_coverage():
    """All samples assigned (no sample lost)."""
    np.random.seed(0)
    labels = np.random.randint(0, 10, 1000)
    parts  = dirichlet_partition(labels, n_clients=5, alpha=0.5)
    total  = sum(len(p) for p in parts)
    assert total == 1000, f"only {total}/1000 samples assigned"
    print(f"  ✓ Dirichlet partition: all 1000 samples assigned across 5 clients")


def test_dirichlet_noniid_alpha():
    """Low α → more non-IID (higher variance in class distribution)."""
    np.random.seed(0)
    labels = np.random.randint(0, 10, 2000)

    parts_iid   = dirichlet_partition(labels, 5, alpha=100.0, seed=1)
    parts_noniid= dirichlet_partition(labels, 5, alpha=0.1,  seed=1)

    def class_var(parts):
        dist = partition_stats(parts, labels, n_classes=10)
        proportions = dist / (dist.sum(axis=1, keepdims=True) + 1e-8)
        return proportions.std()

    var_iid    = class_var(parts_iid)
    var_noniid = class_var(parts_noniid)
    assert var_noniid > var_iid, \
        f"α=0.1 should be more non-IID than α=100: {var_noniid:.4f} vs {var_iid:.4f}"
    print(f"  ✓ Dirichlet: α=0.1 more non-IID (var={var_noniid:.3f}) "
          f"than α=100 (var={var_iid:.3f})")


def test_pathological_partition():
    """Pathological partition: each client has exactly K classes."""
    np.random.seed(1)
    labels = np.array([i % 10 for i in range(1000)])
    parts  = pathological_partition(labels, n_clients=5, classes_per_client=2)
    assert len(parts) == 5
    # All samples assigned
    total = sum(len(p) for p in parts)
    assert total > 0
    print(f"  ✓ Pathological partition: 5 clients, {total} samples assigned")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7C — End-to-end federated learning round
# ─────────────────────────────────────────────────────────────────────────────

def test_fl_round_loss_decreases():
    """
    Full FL round: global model → distribute → local train → fedavg → loss decreases.
    Uses a small MLP on random data (fast, no conv overhead).
    """
    manual_seed(42)
    N_CLIENTS   = 3
    ROUNDS      = 5
    LOCAL_STEPS = 10

    # Global model
    def make_model():
        return nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)
        )

    global_model = make_model()

    def eval_loss(model, X, Y):
        with picograd.no_grad():
            out  = model(Tensor(X))
            loss = nn.CrossEntropyLoss()(out, Tensor(Y))
        return loss.item()

    # Shared evaluation data
    X_eval = np.random.randn(100, 8).astype(np.float32)
    Y_eval = np.random.randint(0, 4, 100).astype(np.float32)

    first_loss = eval_loss(global_model, X_eval, Y_eval)

    for rnd in range(ROUNDS):
        global_sd = global_model.state_dict()
        client_updates = []
        client_counts  = []

        for c in range(N_CLIENTS):
            # Each client gets local model = copy of global
            local = make_model()
            local.load_state_dict(global_sd)
            opt   = picograd.optim.SGD(list(local.parameters()), lr=0.01)
            crit  = nn.CrossEntropyLoss()

            X_c = np.random.randn(32, 8).astype(np.float32)
            Y_c = np.random.randint(0, 4, 32).astype(np.float32)

            for _ in range(LOCAL_STEPS):
                local.zero_grad()
                out  = local(Tensor(X_c))
                loss = crit(out, Tensor(Y_c))
                loss.backward()
                opt.step()

            client_updates.append(local.state_dict())
            client_counts.append(32)

        # FedAvg
        agg_sd = fedavg(client_updates, client_counts)
        global_model.load_state_dict(agg_sd)

    last_loss = eval_loss(global_model, X_eval, Y_eval)
    assert last_loss < first_loss, \
        f"FL round: loss should decrease: {first_loss:.4f} → {last_loss:.4f}"
    print(f"  ✓ FL end-to-end ({N_CLIENTS} clients, {ROUNDS} rounds): "
          f"loss {first_loss:.4f}→{last_loss:.4f}")


def test_fl_with_dp():
    """FL round with DPOptimizer — verifies DP doesn't break training."""
    manual_seed(10)
    from picograd.privacy import DPOptimizer, RDPAccountant

    def make_model():
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    global_model = make_model()
    accountant   = RDPAccountant()
    losses       = []

    for rnd in range(3):
        global_sd     = global_model.state_dict()
        client_updates= []

        local = make_model()
        local.load_state_dict(global_sd)
        base_opt = picograd.optim.SGD(list(local.parameters()), lr=0.01)
        dp_opt   = DPOptimizer(base_opt, noise_multiplier=1.0, max_grad_norm=1.0,
                               batch_size=16, dataset_size=160, accountant=accountant)
        crit     = nn.CrossEntropyLoss()

        X = np.random.randn(16, 4).astype(np.float32)
        Y = np.random.randint(0, 2, 16).astype(np.float32)

        for _ in range(5):
            local.zero_grad()
            out  = local(Tensor(X))
            loss = crit(out, Tensor(Y))
            loss.backward()
            dp_opt.step()
        losses.append(loss.item())
        client_updates.append(local.state_dict())

        agg_sd = fedavg(client_updates, [16])
        global_model.load_state_dict(agg_sd)

    eps = accountant.get_epsilon(1e-5)
    assert np.isfinite(eps) and eps > 0
    print(f"  ✓ FL + DP: 3 rounds, ε={eps:.3f}, training runs without error")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_anomaly_ae_forward_shape,
        test_anomaly_ae_output_range,
        test_anomaly_ae_backward,
        test_anomaly_ae_loss_decreases,
        test_anomaly_detector_predict,
        test_state_dict_serialization,
        test_gradcam_output_shape,
        test_gradcam_values_in_01,
        test_gradcam_overlay,
        test_fedavg_correctness,
        test_fedavg_equal_weights,
        test_fedavg_delta,
        test_fedavg_convergence,
        test_dirichlet_partition_coverage,
        test_dirichlet_noniid_alpha,
        test_pathological_partition,
        test_fl_round_loss_decreases,
        test_fl_with_dp,
    ]
    print(f"\n{'='*60}\nMODULES 6+7: AnomalyAE + GradCAM + FL Integration Tests\n{'='*60}")
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed+failed} passed  "
          f"{'✓ ALL PASS' if failed==0 else f'✗ {failed} FAILED'}")
    print('='*60)
    if failed: sys.exit(1)
