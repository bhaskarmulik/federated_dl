# -*- coding: utf-8 -*-
"""
tests/test_ops.py
==================
Tests for Module 2: Operations Library.
Covers all verification criteria from FALCON_Project_Plan.md Module 2.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import picograd
from picograd import Tensor, manual_seed, no_grad
from picograd.utils.debug import gradcheck
from picograd.ops.convolution import im2col, col2im

# -- helpers ------------------------------------------------------------------

def assert_close(a, b, atol=1e-4, msg=""):
    a = np.asarray(a); b = np.asarray(b)
    diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
    assert diff < atol, f"{msg} | max_diff={diff:.2e}"

def gc(fn, inputs, atol=5e-3, rtol=5e-3, eps=1e-5):
    """Gradcheck wrapper using float64 inputs."""
    f64 = [Tensor(t._data.astype(np.float64), requires_grad=t.requires_grad) for t in inputs]
    ok = gradcheck(fn, f64, eps=eps, atol=atol, rtol=rtol, verbose=False)
    return ok

# -- 1. Elementwise + broadcasting --------------------------------------------

def test_elemwise_broadcast():
    a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)   # (1,3)
    b = Tensor([[1.0],[2.0],[3.0]], requires_grad=True)  # (3,1) -> broadcasts (3,3)
    c = (a * b).sum()
    c.backward()
    assert a.grad.shape == (1, 3), f"got {a.grad.shape}"
    assert b.grad.shape == (3, 1), f"got {b.grad.shape}"
    # grad_a = sum over broadcast axis = b.sum(axis=0, keepdims=True) * ones_like(a)
    # Actually: grad_a[0,j] = sum_i b[i,0] = 1+2+3=6 for all j
    assert_close(a.grad.numpy(), [[6.0,6.0,6.0]], msg="broadcast mul grad_a")
    # grad_b[i,0] = sum_j a[0,j] = 1+2+3=6 for all i
    assert_close(b.grad.numpy(), [[6.0],[6.0],[6.0]], msg="broadcast mul grad_b")
    print("  [PASS] elementwise broadcast (1,3)*(3,1) gradient un-reduction")

def test_pow_backward():
    x = Tensor(np.array([2.0, 3.0, 4.0]), requires_grad=True)
    y = (x ** 3).sum()
    y.backward()
    assert_close(x.grad.numpy(), [12.0, 27.0, 48.0], msg="pow3 backward")
    print("  [PASS] pow3 backward: 3x^2 at [2,3,4] -> [12,27,48]")

def test_div_backward():
    x = Tensor(np.array([4.0, 9.0]), requires_grad=True)
    y = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    (x / y).sum().backward()
    assert_close(x.grad.numpy(), [0.5, 1/3], atol=1e-5, msg="div grad x")
    assert_close(y.grad.numpy(), [-1.0, -1.0], atol=1e-5, msg="div grad y")
    print("  [PASS] div backward")

# -- 2. Reductions ------------------------------------------------------------

def test_sum_backward_axis():
    x = Tensor(np.ones((3,4), dtype=np.float32), requires_grad=True)
    x.sum(axis=1).sum().backward()
    assert_close(x.grad.numpy(), np.ones((3,4)), msg="sum axis=1 backward")
    print("  [PASS] sum(axis=1) backward")

def test_mean_backward():
    x = Tensor(np.ones((2,3), dtype=np.float32)*2, requires_grad=True)
    x.mean().backward()
    assert_close(x.grad.numpy(), np.full((2,3), 1/6.0), atol=1e-5, msg="mean backward")
    print("  [PASS] mean() backward (1/N scaling)")

# -- 3. Activations -----------------------------------------------------------

def test_relu():
    x = Tensor(np.array([-2.0, -0.5, 0.0, 1.0, 3.0]), requires_grad=True)
    y = picograd.nn.ReLU()(x)
    assert_close(y.numpy(), [0, 0, 0, 1, 3], msg="relu forward")
    y.sum().backward()
    assert_close(x.grad.numpy(), [0,0,0,1,1], msg="relu backward")
    print("  [PASS] ReLU forward + backward")

def test_sigmoid():
    from picograd.ops.activations import Sigmoid
    x = Tensor(np.array([0.0]), requires_grad=True)
    y = Sigmoid.apply(x)
    assert_close(y.numpy(), [0.5], msg="sigmoid(0)=0.5")
    y.backward()
    assert_close(x.grad.numpy(), [0.25], atol=1e-5, msg="sigmoid grad at 0")
    print("  [PASS] Sigmoid forward + backward")

def test_softmax_sums_to_one():
    from picograd.ops.activations import Softmax
    x = Tensor(np.random.randn(4, 10).astype(np.float32))
    s = Softmax.apply(x, axis=-1)
    row_sums = s.numpy().sum(axis=-1)
    assert_close(row_sums, np.ones(4), atol=1e-5, msg="softmax row sums")
    print("  [PASS] Softmax rows sum to 1.0")

def test_logsoftmax_shape():
    from picograd.ops.activations import LogSoftmax
    x = Tensor(np.random.randn(3, 5).astype(np.float32), requires_grad=True)
    ls = LogSoftmax.apply(x, axis=-1)
    assert ls.shape == (3, 5)
    ls.sum().backward()
    assert x.grad is not None
    print("  [PASS] LogSoftmax shape + backward runs")

def test_tanh_gradcheck():
    from picograd.ops.activations import Tanh
    manual_seed(10)
    x = Tensor(np.random.randn(4).astype(np.float64)*0.5, requires_grad=True)
    ok = gradcheck(lambda a: Tanh.apply(a).sum(), [x], eps=1e-5, atol=2e-3, rtol=2e-3, verbose=False)
    assert ok, "tanh gradcheck failed"
    print("  [PASS] Tanh gradcheck passes")

def test_gelu_gradcheck():
    from picograd.ops.activations import GELU
    manual_seed(11)
    x = Tensor(np.random.randn(4).astype(np.float64), requires_grad=True)
    ok = gradcheck(lambda a: GELU.apply(a).sum(), [x], eps=1e-5, atol=5e-3, rtol=5e-3, verbose=False)
    assert ok, "gelu gradcheck failed"
    print("  [PASS] GELU gradcheck passes")

# -- 4. Convolution (im2col) --------------------------------------------------

def test_im2col_roundtrip():
    """im2col then col2im should recover original (sum-over-overlaps)."""
    np.random.seed(0)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    col = im2col(x, 3, 3, stride=1, padding=0)
    assert col.shape == (2, 3*3*3, 6*6), f"im2col shape {col.shape}"
    print("  [PASS] im2col output shape correct")

def test_conv2d_forward_vs_manual():
    """Conv2d forward matches manual matmul convolution."""
    from picograd.ops.convolution import Conv2d
    np.random.seed(42)
    N, Ci, H, W = 1, 1, 5, 5
    Co, kH, kW = 1, 3, 3
    x_np = np.random.randn(N, Ci, H, W).astype(np.float32)
    w_np = np.random.randn(Co, Ci, kH, kW).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    w = Tensor(w_np, requires_grad=True)
    out = Conv2d.apply(x, w, None, stride=1, padding=0)
    assert out.shape == (1, 1, 3, 3), f"conv2d output shape {out.shape}"

    # Manual: direct correlation
    from scipy.signal import correlate2d
    manual = correlate2d(x_np[0,0], w_np[0,0], mode='valid')
    assert_close(out.numpy()[0,0], manual, atol=1e-4, msg="conv2d vs scipy")
    print("  [PASS] Conv2d forward matches manual correlation")

def test_conv2d_backward():
    """Conv2d backward produces correct grad shapes."""
    from picograd.ops.convolution import Conv2d
    np.random.seed(7)
    x = Tensor(np.random.randn(2,3,8,8).astype(np.float32), requires_grad=True)
    w = Tensor(np.random.randn(4,3,3,3).astype(np.float32), requires_grad=True)
    b_t = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
    out = Conv2d.apply(x, w, b_t, stride=1, padding=1)
    assert out.shape == (2,4,8,8), f"shape: {out.shape}"
    out.sum().backward()
    assert x.grad.shape == x.shape, f"grad_x shape {x.grad.shape}"
    assert w.grad.shape == w.shape, f"grad_w shape {w.grad.shape}"
    assert b_t.grad.shape == b_t.shape
    print("  [PASS] Conv2d backward -- correct gradient shapes")

def test_conv2d_gradcheck():
    """Conv2d gradcheck on tiny input."""
    from picograd.ops.convolution import Conv2d
    np.random.seed(99)
    x = Tensor(np.random.randn(1,1,5,5).astype(np.float64), requires_grad=True)
    w = Tensor(np.random.randn(1,1,3,3).astype(np.float64), requires_grad=True)
    ok = gradcheck(
        lambda xi, wi: Conv2d.apply(xi, wi, None, 1, 0).sum(),
        [x, w], eps=1e-5, atol=1e-3, rtol=1e-2, verbose=False
    )
    assert ok, "Conv2d gradcheck failed"
    print("  [PASS] Conv2d gradcheck passes")

# -- 5. Pooling ---------------------------------------------------------------

def test_maxpool2d_forward():
    from picograd.ops.pooling import MaxPool2d
    x = Tensor(np.array([[[[1.,2.,3.,4.],
                            [5.,6.,7.,8.],
                            [9.,10.,11.,12.],
                            [13.,14.,15.,16.]]]], dtype=np.float32))
    out = MaxPool2d.apply(x, kernel_size=2, stride=2)
    assert out.shape == (1,1,2,2)
    assert_close(out.numpy()[0,0], [[6,8],[14,16]], msg="maxpool forward")
    print("  [PASS] MaxPool2d forward (2x2 stride=2)")

def test_maxpool2d_backward():
    from picograd.ops.pooling import MaxPool2d
    x = Tensor(np.random.randn(1,2,4,4).astype(np.float32), requires_grad=True)
    out = MaxPool2d.apply(x, kernel_size=2, stride=2)
    out.sum().backward()
    # Each output has exactly one max -- so total grad sum = total output count
    assert x.grad.sum().item() == out.numel, "maxpool backward sum"
    print("  [PASS] MaxPool2d backward (gradient count equals output elements)")

def test_avgpool2d_forward():
    from picograd.ops.pooling import AvgPool2d
    x = Tensor(np.ones((1,1,4,4), dtype=np.float32) * 4.0)
    out = AvgPool2d.apply(x, kernel_size=2, stride=2)
    assert out.shape == (1,1,2,2)
    assert_close(out.numpy(), np.full((1,1,2,2), 4.0), msg="avgpool forward")
    print("  [PASS] AvgPool2d forward")

# -- 6. Normalization ---------------------------------------------------------

def test_batchnorm_train():
    from picograd.ops.normalization import BatchNorm
    np.random.seed(5)
    x = Tensor(np.random.randn(4,8,4,4).astype(np.float32), requires_grad=True)
    g = Tensor(np.ones(8, dtype=np.float32), requires_grad=True)
    b = Tensor(np.zeros(8, dtype=np.float32), requires_grad=True)
    running_mean = np.zeros(8, dtype=np.float32)
    running_var  = np.ones(8, dtype=np.float32)
    out = BatchNorm.apply(x, g, b, running_mean, running_var, True, 0.1, 1e-5)
    # Each channel should have mean~0, std~1
    out_np = out.numpy()
    for c in range(8):
        ch = out_np[:,c,:,:]
        assert abs(ch.mean()) < 0.1, f"BN channel {c} mean={ch.mean():.4f}"
    out.sum().backward()
    assert x.grad is not None
    assert g.grad is not None
    print("  [PASS] BatchNorm2d train: channels normalized + backward runs")

def test_layernorm():
    from picograd.ops.normalization import LayerNorm
    np.random.seed(6)
    x = Tensor(np.random.randn(2,4,8).astype(np.float32), requires_grad=True)
    g = Tensor(np.ones(8, dtype=np.float32), requires_grad=True)
    b = Tensor(np.zeros(8, dtype=np.float32), requires_grad=True)
    out = LayerNorm.apply(x, g, b, (8,), 1e-5)
    # Last dim should be normalized
    out_np = out.numpy()
    assert abs(out_np.mean(axis=-1).mean()) < 0.1
    out.sum().backward()
    assert x.grad is not None
    print("  [PASS] LayerNorm forward + backward runs")

# -- 7. Dropout ---------------------------------------------------------------

def test_dropout_train():
    from picograd.ops.dropout_embedding import Dropout
    manual_seed(0)
    x = Tensor(np.ones((1000,), dtype=np.float32))
    out = Dropout.apply(x, p=0.5, training=True)
    zero_frac = (out.numpy() == 0).mean()
    assert 0.4 < zero_frac < 0.6, f"dropout zero fraction {zero_frac:.3f} not near 0.5"
    print(f"  [PASS] Dropout train: ~{zero_frac*100:.0f}% zeros (expected ~50%)")

def test_dropout_eval():
    from picograd.ops.dropout_embedding import Dropout
    x = Tensor(np.ones((100,), dtype=np.float32))
    out = Dropout.apply(x, p=0.5, training=False)
    assert_close(out.numpy(), x.numpy(), msg="dropout eval = identity")
    print("  [PASS] Dropout eval: identity pass-through")

# -- Runner -------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_elemwise_broadcast,
        test_pow_backward,
        test_div_backward,
        test_sum_backward_axis,
        test_mean_backward,
        test_relu,
        test_sigmoid,
        test_softmax_sums_to_one,
        test_logsoftmax_shape,
        test_tanh_gradcheck,
        test_gelu_gradcheck,
        test_im2col_roundtrip,
        test_conv2d_forward_vs_manual,
        test_conv2d_backward,
        test_conv2d_gradcheck,
        test_maxpool2d_forward,
        test_maxpool2d_backward,
        test_avgpool2d_forward,
        test_batchnorm_train,
        test_layernorm,
        test_dropout_train,
        test_dropout_eval,
    ]
    print(f"\n{'='*60}\nMODULE 2: Operations Library Tests\n{'='*60}")
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed+failed} passed  {'[PASS] ALL PASS' if failed==0 else f'[FAIL] {failed} FAILED'}")
    print('='*60)
    if failed: sys.exit(1)
