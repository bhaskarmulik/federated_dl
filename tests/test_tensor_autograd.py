# -*- coding: utf-8 -*-
"""
tests/test_tensor_autograd.py
==============================
Tests for Module 1: Tensor + Autograd Engine.
Covers all verification criteria from FALCON_Project_Plan.md Module 1.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import picograd
from picograd import Tensor, no_grad, manual_seed
from picograd.utils.debug import gradcheck


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def assert_close(a, b, atol=1e-5, msg=""):
    a = np.array(a) if not isinstance(a, np.ndarray) else a
    b = np.array(b) if not isinstance(b, np.ndarray) else b
    diff = np.max(np.abs(a - b))
    assert diff < atol, f"{msg} | max_diff={diff:.2e} expected < {atol}"


# -------------------------------------------------------------
# 1. Tensor creation and properties
# -------------------------------------------------------------

def test_tensor_creation():
    t = Tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.ndim == 1
    assert t.numel == 3
    assert t.dtype == np.float32
    print("  [PASS] tensor creation & properties")


def test_tensor_from_numpy_roundtrip():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = Tensor.from_numpy(arr)
    assert_close(t.numpy(), arr, msg="from_numpy roundtrip")
    print("  [PASS] numpy roundtrip")


def test_tensor_static_constructors():
    z = Tensor.zeros(3, 4)
    assert z.shape == (3, 4)
    assert_close(z.numpy(), np.zeros((3, 4)))

    o = Tensor.ones(2, 3)
    assert_close(o.numpy(), np.ones((2, 3)))

    manual_seed(42)
    r = Tensor.randn(5, 5)
    assert r.shape == (5, 5)
    print("  [PASS] static constructors (zeros, ones, randn)")


# -------------------------------------------------------------
# 2. Basic autograd -- scalar operations
# -------------------------------------------------------------

def test_grad_square():
    """x = 3.0 -> y = x*x -> dy/dx = 2x = 6.0"""
    x = Tensor([3.0], requires_grad=True)
    y = x * x
    y.backward()
    assert_close(x.grad.numpy(), [6.0], msg="d(x^2)/dx at x=3")
    print("  [PASS] grad of x*x at x=3 -> 6.0")


def test_grad_chain_rule():
    """f(x) = (x+2)*(x+3) -> df/dx = 2x+5 -> at x=1 -> 7.0"""
    x = Tensor([1.0], requires_grad=True)
    y = (x + Tensor([2.0])) * (x + Tensor([3.0]))
    y.backward()
    assert_close(x.grad.numpy(), [7.0], msg="chain rule (x+2)(x+3)")
    print("  [PASS] chain rule: (x+2)(x+3) at x=1 -> 7.0")


def test_grad_fanout():
    """y = x + x -> dy/dx = 2.0 (fan-out gradient accumulation)"""
    x = Tensor([4.0], requires_grad=True)
    y = x + x
    y.backward()
    assert_close(x.grad.numpy(), [2.0], msg="fan-out accumulation")
    print("  [PASS] fan-out: x+x -> grad=2.0")


def test_grad_polynomial():
    """y = x^3 + 2*x^2 - x -> dy/dx = 3x^2 + 4x - 1 -> at x=2 -> 19"""
    x = Tensor([2.0], requires_grad=True)
    y = x**3 + Tensor([2.0]) * x**2 - x
    y.backward()
    assert_close(x.grad.numpy(), [19.0], atol=1e-4, msg="polynomial grad")
    print("  [PASS] polynomial: 3x^2+4x-1 at x=2 -> 19.0")


def test_grad_accumulation():
    """Multiple backward calls accumulate gradients."""
    x = Tensor([1.0], requires_grad=True)
    y1 = x * x
    y1.backward()
    y2 = x * x
    y2.backward()
    # grad should be 2 + 2 = 4.0
    assert_close(x.grad.numpy(), [4.0], msg="gradient accumulation over 2 backward passes")
    print("  [PASS] gradient accumulation across multiple backward passes")


# -------------------------------------------------------------
# 3. Vector / matrix gradients
# -------------------------------------------------------------

def test_grad_matmul():
    """C = A @ B -> dL/dA = dL/dC @ B^T"""
    np.random.seed(0)
    A = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    B = Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
    C = A @ B
    # L = sum(C)
    C.sum().backward()

    # Analytic: dL/dA = ones(3,5) @ B^T
    expected_A = np.ones((3, 5)) @ B.numpy().T
    expected_B = A.numpy().T @ np.ones((3, 5))
    assert_close(A.grad.numpy(), expected_A, atol=1e-4, msg="matmul grad A")
    assert_close(B.grad.numpy(), expected_B, atol=1e-4, msg="matmul grad B")
    print("  [PASS] matmul gradients (A and B)")


def test_grad_broadcast_add():
    """Test broadcasting: (3,1) + (3,4) and correct gradient un-broadcasting."""
    a = Tensor(np.ones((3, 1), dtype=np.float32), requires_grad=True)
    b = Tensor(np.ones((3, 4), dtype=np.float32), requires_grad=True)
    c = a + b
    c.sum().backward()
    assert a.grad.shape == (3, 1), f"a.grad shape {a.grad.shape} != (3,1)"
    assert_close(a.grad.numpy(), np.full((3, 1), 4.0), msg="broadcast add grad a")
    assert_close(b.grad.numpy(), np.ones((3, 4)), msg="broadcast add grad b")
    print("  [PASS] broadcast add gradient un-reduction")


# -------------------------------------------------------------
# 4. no_grad context manager
# -------------------------------------------------------------

def test_no_grad():
    x = Tensor([5.0], requires_grad=True)
    with no_grad():
        y = x * x
    assert y._grad_fn is None, "no_grad should suppress grad_fn"
    assert not y.requires_grad
    print("  [PASS] no_grad suppresses graph construction")


def test_no_grad_decorator():
    @no_grad()
    def inference(t):
        return t * t * t

    x = Tensor([2.0], requires_grad=True)
    out = inference(x)
    assert out._grad_fn is None
    print("  [PASS] no_grad as decorator")


# -------------------------------------------------------------
# 5. Serialization
# -------------------------------------------------------------

def test_save_load():
    import picograd
    import tempfile
    
    sd = {"weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
          "bias":   np.array([0.5, -0.5], dtype=np.float32)}
    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, "falcon_test_sd.pkl")
    picograd.save(sd, path)
    loaded = picograd.load(path)
    for k in sd:
        assert_close(loaded[k], sd[k], msg=f"save/load key {k}")
    print("  [PASS] save/load roundtrip preserves state_dict")


def test_state_dict_bytes():
    from picograd.serialization import state_dict_to_bytes, bytes_to_state_dict
    sd = {"w": np.random.randn(4, 4).astype(np.float32)}
    b = state_dict_to_bytes(sd)
    sd2 = bytes_to_state_dict(b)
    assert_close(sd["w"], sd2["w"])
    print("  [PASS] state_dict_to_bytes / bytes_to_state_dict roundtrip")


# -------------------------------------------------------------
# 6. gradcheck (finite differences)
# -------------------------------------------------------------

def test_gradcheck_mul():
    """Gradcheck for element-wise multiplication."""
    manual_seed(1)
    x = Tensor(np.random.randn(3).astype(np.float64) + 1.0, requires_grad=True)
    y = Tensor(np.random.randn(3).astype(np.float64) + 1.0, requires_grad=True)
    ok = gradcheck(lambda a, b: (a * b).sum(), [x, y], eps=1e-5, atol=1e-4, verbose=False)
    assert ok, "gradcheck failed for Mul"
    print("  [PASS] gradcheck: Mul")


def test_gradcheck_exp_log():
    """exp(x).log() = identity, grad should be all-ones.
    Uses rtol=5e-3: composed transcendentals have O(1e-3) FD truncation (expected)."""
    manual_seed(2)
    x = Tensor(np.abs(np.random.randn(4)) + 0.5, requires_grad=True)
    ok = gradcheck(lambda a: a.exp().log().sum(), [x], eps=1e-5, atol=5e-3, rtol=5e-3, verbose=False)
    assert ok, "gradcheck failed for exp+log"
    print("  [PASS] gradcheck: exp -> log composition")


def test_gradcheck_matmul():
    """Matrix multiply gradcheck."""
    manual_seed(3)
    A = Tensor(np.random.randn(3, 4), requires_grad=True)
    B = Tensor(np.random.randn(4, 2), requires_grad=True)
    ok = gradcheck(lambda a, b: (a @ b).sum(), [A, B], eps=1e-5, atol=1e-2, rtol=1e-2, verbose=False)
    assert ok, "gradcheck failed for matmul"
    print("  [PASS] gradcheck: MatMul")


# -------------------------------------------------------------
# Run all tests
# -------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_tensor_creation,
        test_tensor_from_numpy_roundtrip,
        test_tensor_static_constructors,
        test_grad_square,
        test_grad_chain_rule,
        test_grad_fanout,
        test_grad_polynomial,
        test_grad_accumulation,
        test_grad_matmul,
        test_grad_broadcast_add,
        test_no_grad,
        test_no_grad_decorator,
        test_save_load,
        test_state_dict_bytes,
        test_gradcheck_mul,
        test_gradcheck_exp_log,
        test_gradcheck_matmul,
    ]
    print(f"\n{'='*60}")
    print("MODULE 1: Tensor + Autograd Tests")
    print('='*60)
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed+failed} passed  {'[PASS] ALL PASS' if failed==0 else f'[FAIL] {failed} FAILED'}")
    print('='*60)
    if failed: sys.exit(1)
