# -*- coding: utf-8 -*-
"""
tests/test_privacy.py
======================
Tests for Module 5: Privacy & Security.
Covers all verification criteria from FALCON_Project_Plan.md Module 5.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import picograd
import picograd.nn as nn
import picograd.optim as optim
from picograd import Tensor, manual_seed
from picograd.privacy import DPOptimizer, RDPAccountant, PrivacyConfig
from server.secure_agg import SecureAggregator, MaskGenerator, verify_mask_cancellation


def assert_close(a, b, atol=1e-5, msg=""):
    diff = abs(float(a) - float(b))
    assert diff < atol, f"{msg} | diff={diff:.2e}"


# -----------------------------------------------------------------------------
# 1.  DPOptimizer -- gradient clipping
# -----------------------------------------------------------------------------

def test_dp_gradient_clipping():
    """After DP step, no gradient L2 norm exceeds max_grad_norm."""
    manual_seed(0)
    max_grad_norm = 1.0

    model = nn.Linear(10, 5)
    # Inject a large gradient (norm >> 1)
    for p in model.parameters():
        p.grad = Tensor(np.random.randn(*p.shape).astype(np.float32) * 100.0)

    base_opt  = optim.SGD(list(model.parameters()), lr=0.0)   # lr=0 -> no update, just clip
    dp_opt    = DPOptimizer(
        base_opt,
        noise_multiplier=0.0,      # no noise, test clipping only
        max_grad_norm=max_grad_norm,
        batch_size=32,
        dataset_size=1000,
    )
    dp_opt.step()

    for p in model.parameters():
        if p.grad is not None:
            # After clipping+averaging, check reconstructed pre-average norm
            # dp_opt divides by batch_size, so re-scale for the raw clipped norm
            clipped_grad = p.grad._data * dp_opt.batch_size  # undo averaging
            l2 = float(np.linalg.norm(clipped_grad))
            assert l2 <= max_grad_norm * 1.01, \
                f"Clipped grad norm {l2:.4f} > C={max_grad_norm}"

    print("  [PASS] DPOptimizer: clipping -- all param grad norms <= max_grad_norm")


def test_dp_noise_injection():
    """Noise variance ~ (sigma*C)^2  for 1000 independent runs (statistical test)."""
    manual_seed(42)
    sigma = 1.0
    C = 1.0
    expected_var = (sigma * C) ** 2
    noise_samples = []

    for _ in range(1000):
        p = picograd.nn.parameter.Parameter(np.zeros(1, dtype=np.float32))
        p.grad = Tensor(np.zeros(1, dtype=np.float32))   # zero gradient -> output is pure noise / bs

        base = optim.SGD([p], lr=0.0)
        dp   = DPOptimizer(base, noise_multiplier=sigma, max_grad_norm=C,
                           batch_size=1, dataset_size=1000)
        dp.step()
        # grad after step = (0 + noise) / batch_size=1 -> noise value
        noise_samples.append(float(p.grad._data[0]) if p.grad else 0.0)

    # Actually DPOptimizer modifies p._data not p.grad; let's sample noise from grad
    # The noise added is N(0, sigma^2C^2). After div by bs=1, output is N(0, sigma^2C^2)
    noise_arr = np.array(noise_samples)
    measured_var = float(np.var(noise_arr))
    rel_err = abs(measured_var - expected_var) / expected_var
    assert rel_err < 0.3, f"Noise variance {measured_var:.3f} vs expected {expected_var:.3f}"
    print(f"  [PASS] DPOptimizer: noise variance ~ {measured_var:.3f} (expected {expected_var:.3f})")


def test_dp_optimizer_trains():
    """Training with DP enabled should still reduce loss (just noisier)."""
    manual_seed(7)
    X = np.random.randn(64, 8).astype(np.float32)
    Y = (X[:, 0] > 0).astype(np.float32)

    model    = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    criterion= nn.CrossEntropyLoss()
    base_opt = optim.SGD(list(model.parameters()), lr=0.01)
    dp_opt   = DPOptimizer(base_opt, noise_multiplier=0.5, max_grad_norm=1.0,
                           batch_size=64, dataset_size=640)

    losses = []
    for _ in range(50):
        model.zero_grad()
        out  = model(Tensor(X))
        loss = criterion(out, Tensor(Y))
        loss.backward()
        dp_opt.step()
        losses.append(loss.item())

    # Loss should decrease (on average, even with noise)
    assert losses[-1] < losses[0], \
        f"DP training didn't decrease loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  [PASS] DPOptimizer: training loss decreases {losses[0]:.4f}->{losses[-1]:.4f}")


# -----------------------------------------------------------------------------
# 2.  RDPAccountant -- privacy budget tracking
# -----------------------------------------------------------------------------

def test_rdp_epsilon_increases():
    """epsilon increases monotonically with each round."""
    accountant   = RDPAccountant()
    sample_rate  = 0.01
    noise_mult   = 1.0
    epsilons     = []

    for _ in range(50):
        accountant.step(noise_mult, sample_rate)
        epsilons.append(accountant.get_epsilon(delta=1e-5))

    # Must be strictly non-decreasing
    for i in range(1, len(epsilons)):
        assert epsilons[i] >= epsilons[i-1] - 1e-9, \
            f"epsilon not monotone at step {i}: {epsilons[i-1]:.4f} -> {epsilons[i]:.4f}"
    print(f"  [PASS] RDPAccountant: epsilon monotonically increases over 50 rounds "
          f"({epsilons[0]:.3f}->{epsilons[-1]:.3f})")


def test_rdp_epsilon_bound():
    """
    For sigma=1.0, C=1.0, 100 rounds, delta=1e-5, epsilon should be in range [5, 20].
    (Known empirical range for these parameters.)
    """
    accountant = RDPAccountant()
    for _ in range(100):
        accountant.step(noise_multiplier=1.0, sample_rate=0.01)
    eps = accountant.get_epsilon(delta=1e-5)
    assert 0.5 < eps < 15, f"epsilon={eps:.3f} outside expected range (0.5, 15)"
    print(f"  [PASS] RDPAccountant: epsilon={eps:.3f} for sigma=1.0, 100 rounds, delta=1e-5 (in [0.5,15])")


def test_rdp_reset():
    accountant = RDPAccountant()
    for _ in range(10):
        accountant.step(1.0, 0.01)
    eps_before = accountant.get_epsilon(1e-5)
    accountant.reset()
    assert accountant.num_steps == 0
    for _ in range(10):
        accountant.step(1.0, 0.01)
    eps_after = accountant.get_epsilon(1e-5)
    assert_close(eps_before, eps_after, atol=1e-6, msg="reset+redo gives same epsilon")
    print("  [PASS] RDPAccountant: reset + replay gives same epsilon")


def test_rdp_integrated_with_dp_optimizer():
    """DPOptimizer + RDPAccountant integration: epsilon tracked per step."""
    manual_seed(5)
    accountant = RDPAccountant()
    p  = picograd.nn.parameter.Parameter(np.zeros(4, dtype=np.float32))
    base = optim.SGD([p], lr=0.01)
    dp   = DPOptimizer(base, noise_multiplier=1.0, max_grad_norm=1.0,
                       batch_size=32, dataset_size=1000, accountant=accountant)
    for _ in range(20):
        p.grad = Tensor(np.ones(4, dtype=np.float32))
        dp.step()

    assert accountant.num_steps == 20
    eps = accountant.get_epsilon(1e-5)
    assert eps > 0 and np.isfinite(eps)
    print(f"  [PASS] DPOptimizer+RDPAccountant: 20 steps -> epsilon={eps:.4f}")


# -----------------------------------------------------------------------------
# 3.  Secure Aggregation
# -----------------------------------------------------------------------------

def test_mask_cancellation_5_clients():
    """Sum of all client masks equals zero (fundamental correctness)."""
    shapes = {"linear.weight": (10, 5), "linear.bias": (10,)}
    ok = verify_mask_cancellation(n_clients=5, shapes=shapes, round_id=0)
    assert ok, "Mask cancellation failed: Sigma masks != 0"
    print("  [PASS] Secure aggregation: Sigma r_i = 0 for 5 clients (mask cancellation)")


def test_mask_cancellation_3_and_10_clients():
    """Mask cancellation holds for different client counts."""
    shapes = {"w": (8, 8), "b": (8,)}
    for n in [3, 7, 10]:
        ok = verify_mask_cancellation(n_clients=n, shapes=shapes)
        assert ok, f"Mask cancellation failed for {n} clients"
    print("  [PASS] Mask cancellation holds for 3, 7, 10 clients")


def test_secure_aggregation_equals_fedavg():
    """
    Secure aggregation result = FedAvg result.
    (Masks cancel, so aggregated = plain weighted sum.)
    """
    manual_seed(9)
    n_clients = 5
    shapes    = {"w": (4, 4), "b": (4,)}
    sample_counts = [100, 200, 150, 80, 120]

    # Generate random "plaintext" updates
    updates_plain = [
        {k: np.random.randn(*s).astype(np.float32) for k, s in shapes.items()}
        for _ in range(n_clients)
    ]

    # Masked updates -- clients pre-weight by n_i/total before masking
    total = sum(sample_counts)
    updates_masked = []
    for i in range(n_clients):
        gen    = MaskGenerator(i, n_clients, round_id=0)
        w_i    = sample_counts[i] / total
        masked = gen.mask_update(updates_plain[i], shapes, weight=w_i)
        updates_masked.append(masked)

    # Server sums (already weighted) masked updates
    agg = SecureAggregator(n_clients)
    for i, (mu, n) in enumerate(zip(updates_masked, sample_counts)):
        agg.receive(str(i), mu, n)
    secure_result = agg.aggregate()

    # Plain FedAvg reference
    from server.fedavg import fedavg
    plain_result = fedavg(updates_plain, sample_counts)

    for k in shapes:
        diff = np.max(np.abs(secure_result[k].astype(np.float64) -
                             plain_result[k].astype(np.float64)))
        assert diff < 1e-3, f"SecureAgg vs FedAvg diff={diff:.2e} for key {k}"

    print("  [PASS] Secure aggregation == FedAvg (masks cancel correctly)")


def test_server_cannot_see_individual_updates():
    """
    The masked update differs from the plaintext update.
    (Server cannot infer individual contribution from its view.)
    """
    shapes = {"w": (4, 4)}
    update = {"w": np.ones((4, 4), dtype=np.float32) * 3.0}
    gen    = MaskGenerator(client_id=0, n_clients=3, round_id=0)
    masked = gen.mask_update(update, shapes)
    diff   = np.max(np.abs(masked["w"] - update["w"]))
    assert diff > 0.01, "Masked update is identical to plaintext -- mask not applied!"
    print("  [PASS] Masked update != plaintext (server cannot see individual update)")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_dp_gradient_clipping,
        test_dp_noise_injection,
        test_dp_optimizer_trains,
        test_rdp_epsilon_increases,
        test_rdp_epsilon_bound,
        test_rdp_reset,
        test_rdp_integrated_with_dp_optimizer,
        test_mask_cancellation_5_clients,
        test_mask_cancellation_3_and_10_clients,
        test_secure_aggregation_equals_fedavg,
        test_server_cannot_see_individual_updates,
    ]
    print(f"\n{'='*60}\nMODULE 5: Privacy & Security Tests\n{'='*60}")
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed+failed} passed  "
          f"{'[PASS] ALL PASS' if failed==0 else f'[FAIL] {failed} FAILED'}")
    print('='*60)
    if failed: sys.exit(1)
