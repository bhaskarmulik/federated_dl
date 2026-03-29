"""
tests/test_nn_training.py
==========================
Tests for Modules 3 & 4: nn.Module system + Training infrastructure.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import picograd
import picograd.nn as nn
import picograd.optim as optim
from picograd import Tensor, manual_seed
from picograd.data.dataloader import DataLoader, NumpyDataset

def assert_close(a, b, atol=1e-4, msg=""):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    diff = np.max(np.abs(a - b))
    assert diff < atol, f"{msg} | max_diff={diff:.2e}"

# ── 1. Module system ─────────────────────────────────────────────────────────

def test_linear_forward():
    manual_seed(0)
    lin = nn.Linear(4, 3)
    x = Tensor(np.ones((2, 4), dtype=np.float32))
    out = lin(x)
    assert out.shape == (2, 3), f"shape {out.shape}"
    print("  ✓ Linear(4,3) forward: (2,4)→(2,3)")

def test_linear_backward():
    manual_seed(1)
    lin = nn.Linear(3, 2)
    x = Tensor(np.random.randn(4, 3).astype(np.float32))
    out = lin(x)
    out.sum().backward()
    assert lin.weight.grad is not None
    assert lin.bias.grad is not None
    assert lin.weight.grad.shape == lin.weight.shape
    print("  ✓ Linear backward: weight.grad and bias.grad populated")

def test_sequential():
    manual_seed(2)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = Tensor(np.random.randn(3, 4).astype(np.float32))
    out = model(x)
    assert out.shape == (3, 2)
    out.sum().backward()
    params = list(model.parameters())
    assert len(params) == 4  # w1, b1, w2, b2
    assert all(p.grad is not None for p in params)
    print("  ✓ Sequential: 3-layer MLP forward + backward")

def test_state_dict_roundtrip():
    manual_seed(3)
    model = nn.Linear(4, 2)
    sd1 = model.state_dict()
    # Modify weights
    model.weight._data[:] = 99.0
    # Restore
    model.load_state_dict(sd1)
    assert_close(model.weight.numpy(), sd1["weight"], msg="state_dict restore")
    print("  ✓ state_dict save/load roundtrip")

def test_train_eval_mode():
    model = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(4, 2))
    model.eval()
    assert not model.training
    # Dropout in eval = identity
    x = Tensor(np.ones((10, 4), dtype=np.float32))
    out1 = model(x)
    out2 = model(x)
    assert_close(out1.numpy(), out2.numpy(), msg="eval mode deterministic")
    model.train()
    assert model.training
    print("  ✓ train/eval mode toggle + Dropout deterministic in eval")

def test_batchnorm2d_module():
    manual_seed(4)
    bn = nn.BatchNorm2d(8)
    x = Tensor(np.random.randn(4, 8, 4, 4).astype(np.float32), requires_grad=True)
    out = bn(x)
    assert out.shape == (4, 8, 4, 4)
    out.sum().backward()
    assert bn.weight.grad is not None
    print("  ✓ BatchNorm2d module forward + backward")

def test_conv2d_module():
    manual_seed(5)
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 3, 16, 16).astype(np.float32), requires_grad=True)
    out = conv(x)
    assert out.shape == (2, 8, 16, 16), f"shape {out.shape}"
    out.sum().backward()
    assert conv.weight.grad is not None
    print("  ✓ Conv2d module (with padding): (2,3,16,16)→(2,8,16,16)")

def test_zero_grad():
    manual_seed(6)
    model = nn.Linear(3, 2)
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    model(x).sum().backward()
    assert model.weight.grad is not None
    model.zero_grad()
    assert model.weight.grad is None
    print("  ✓ zero_grad clears all parameter gradients")

# ── 2. Loss functions ────────────────────────────────────────────────────────

def test_crossentropy_loss():
    logits = Tensor(np.array([[2.0, 1.0, 0.1],
                               [0.1, 1.0, 2.0]], dtype=np.float32), requires_grad=True)
    targets = Tensor(np.array([0, 2], dtype=np.float32))
    loss = nn.CrossEntropyLoss()(logits, targets)
    assert loss.shape == (), f"CE loss shape {loss.shape}"
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == (2, 3)
    print(f"  ✓ CrossEntropyLoss forward={loss.item():.4f} + backward")

def test_mse_loss():
    pred   = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    target = Tensor(np.array([1.5, 2.5, 2.5]))
    loss = nn.MSELoss()(pred, target)
    expected = ((0.5**2 + 0.5**2 + 0.5**2) / 3)
    assert_close(loss.item(), expected, atol=1e-5, msg="MSE value")
    loss.backward()
    assert pred.grad is not None
    print(f"  ✓ MSELoss value={loss.item():.4f} + backward")

def test_bce_loss():
    pred   = Tensor(np.array([0.9, 0.1, 0.8]), requires_grad=True)
    target = Tensor(np.array([1.0, 0.0, 1.0]))
    loss = nn.BCELoss()(pred, target)
    assert loss.item() > 0
    loss.backward()
    assert pred.grad is not None
    print(f"  ✓ BCELoss forward={loss.item():.4f} + backward")

# ── 3. Optimizers ────────────────────────────────────────────────────────────

def test_sgd_step():
    """SGD: param -= lr * grad."""
    p = picograd.nn.parameter.Parameter(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    p.grad = Tensor(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    opt = optim.SGD([p], lr=0.1)
    opt.step()
    assert_close(p.numpy(), [0.99, 1.98, 2.97], atol=1e-5, msg="SGD step")
    print("  ✓ SGD step: p -= lr*grad")

def test_adam_step():
    """Adam: first step reduces param in gradient direction."""
    manual_seed(0)
    p = picograd.nn.parameter.Parameter(np.zeros(4, dtype=np.float32))
    p.grad = Tensor(np.ones(4, dtype=np.float32))
    opt = optim.Adam([p], lr=1e-3)
    opt.step()
    # All params should decrease by ~lr (Adam step 1 ≈ lr * sign)
    assert np.all(p.numpy() < 0), "Adam should move params in neg-grad direction"
    print("  ✓ Adam step: params moved in correct direction")

def test_sgd_momentum():
    """SGD with momentum accumulates velocity buffer."""
    p = picograd.nn.parameter.Parameter(np.array([1.0], dtype=np.float32))
    p.grad = Tensor(np.array([1.0], dtype=np.float32))
    opt = optim.SGD([p], lr=0.1, momentum=0.9)
    opt.step()  # buf=1.0, p = 1.0 - 0.1*1.0 = 0.9
    p.grad = Tensor(np.array([1.0], dtype=np.float32))
    opt.step()  # buf=0.9+1=1.9, p = 0.9 - 0.1*1.9 = 0.71
    assert_close(p.numpy(), [0.71], atol=1e-5, msg="SGD momentum 2 steps")
    print("  ✓ SGD with momentum accumulates velocity correctly")

def test_lr_scheduler_step():
    from picograd.optim.lr_scheduler import StepLR
    p = picograd.nn.parameter.Parameter(np.zeros(1, dtype=np.float32))
    opt = optim.SGD([p], lr=0.1)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    # After step 0 (called in __init__): lr=0.1
    # After step 2: lr=0.1*0.5^1=0.05
    # After step 4: lr=0.1*0.5^2=0.025
    sched.step(); sched.step()  # steps 1,2
    assert_close(opt.param_groups[0]['lr'], 0.05, atol=1e-6, msg="StepLR after 2 steps")
    sched.step(); sched.step()  # steps 3,4
    assert_close(opt.param_groups[0]['lr'], 0.025, atol=1e-6, msg="StepLR after 4 steps")
    print("  ✓ StepLR halves lr every 2 steps")

# ── 4. DataLoader ────────────────────────────────────────────────────────────

def test_dataloader_batching():
    x = np.random.randn(100, 4).astype(np.float32)
    y = np.random.randint(0, 3, 100).astype(np.float32)
    ds = NumpyDataset(x, y)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    batches = list(dl)
    assert len(batches) == 7  # ceil(100/16)
    assert batches[0][0].shape == (16, 4)
    assert batches[-1][0].shape == (4, 4)   # last batch 100%16=4
    print("  ✓ DataLoader: correct batch sizes and count")

def test_dataloader_shuffle():
    x = np.arange(20, dtype=np.float32).reshape(20,1)
    y = np.zeros(20, dtype=np.float32)
    ds = NumpyDataset(x, y)
    manual_seed(42)
    dl1 = DataLoader(ds, batch_size=20, shuffle=True)
    b1 = next(iter(dl1))[0].numpy().flatten()
    manual_seed(42)
    dl2 = DataLoader(ds, batch_size=20, shuffle=True)
    b2 = next(iter(dl2))[0].numpy().flatten()
    assert not np.all(b1 == np.arange(20)), "shuffled should not be sorted"
    assert_close(b1, b2, msg="same seed → same shuffle")
    print("  ✓ DataLoader shuffle: reproducible with same seed")

# ── 5. End-to-end: train MLP on XOR ─────────────────────────────────────────

def test_e2e_mlp_xor():
    """Train a 2-layer MLP to solve XOR. Loss must decrease significantly."""
    manual_seed(42)
    # XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    Y = np.array([0, 1, 1, 0], dtype=np.float32)  # targets as class indices

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()), lr=1e-2)

    losses = []
    for _ in range(300):
        x_t = Tensor(X)
        y_t = Tensor(Y)
        model.zero_grad()
        out = model(x_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.3, \
        f"Loss did not decrease enough: {losses[0]:.4f}→{losses[-1]:.4f}"
    print(f"  ✓ MLP XOR: loss {losses[0]:.4f}→{losses[-1]:.4f} (>70% reduction)")

def test_e2e_cnn_mnist_like():
    """Tiny CNN on random 28×28 data — verify loss decreases."""
    manual_seed(7)
    N = 32
    X = np.random.randn(N, 1, 14, 14).astype(np.float32)
    Y = np.random.randint(0, 5, N).astype(np.float32)

    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),        # (N,4,7,7)
        nn.Flatten(),           # (N, 196)
        nn.Linear(4*7*7, 5),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.parameters()), lr=1e-2)

    first_loss = None
    for i in range(20):
        model.zero_grad()
        out = model(Tensor(X))
        loss = criterion(out, Tensor(Y))
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()

    # Just verify it runs without error and produces a scalar loss
    assert isinstance(loss.item(), float)
    print(f"  ✓ CNN end-to-end: {first_loss:.4f}→{loss.item():.4f} (runs without error)")

# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_linear_forward, test_linear_backward, test_sequential,
        test_state_dict_roundtrip, test_train_eval_mode,
        test_batchnorm2d_module, test_conv2d_module, test_zero_grad,
        test_crossentropy_loss, test_mse_loss, test_bce_loss,
        test_sgd_step, test_adam_step, test_sgd_momentum,
        test_lr_scheduler_step,
        test_dataloader_batching, test_dataloader_shuffle,
        test_e2e_mlp_xor, test_e2e_cnn_mnist_like,
    ]
    print(f"\n{'='*60}\nMODULES 3+4: nn.Module + Training Tests\n{'='*60}")
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed+failed} passed  {'✓ ALL PASS' if failed==0 else f'✗ {failed} FAILED'}")
    print('='*60)
    if failed: sys.exit(1)
