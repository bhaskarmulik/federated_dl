"""Comprehensive test for picograd Phases 1 through 2.4."""
import numpy as np
from picograd import Tensor, no_grad, cat, stack


def ok(msg):
    print(f'[OK] {msg}')


# ===== PHASE 1: Core =====
print('=== Phase 1: Tensor + Backend + Autograd ===')

x = Tensor([1.0, 2.0, 3.0])
assert x.shape == (3,)
ok('Tensor creation')

z = Tensor.zeros(2, 3); o = Tensor.ones(2, 3); r = Tensor.randn(2, 3)
ok('Static constructors')

x = Tensor([3.0], requires_grad=True)
y = x * x; y.backward()
assert np.allclose(x.grad.numpy(), [6.0])
ok('x^2 backward')

x = Tensor([1.0], requires_grad=True)
y = (x + Tensor([2.0])) * (x + Tensor([3.0]))
y.backward()
assert np.allclose(x.grad.numpy(), [7.0])
ok('Chain rule')

x = Tensor([5.0], requires_grad=True)
y = x + x; y.backward()
assert np.allclose(x.grad.numpy(), [2.0])
ok('Fan-out')

x = Tensor([1.0], requires_grad=True)
with no_grad():
    y = x * x
assert y._grad_fn is None
ok('no_grad')

# ===== PHASE 2.1: Elementwise =====
print('\n=== Phase 2.1: Elementwise ===')

a = Tensor([1.0, 2.0], requires_grad=True)
b = Tensor([3.0, 4.0], requires_grad=True)
(a + b).sum().backward()
assert np.allclose(a.grad.numpy(), [1, 1])
ok('Add')

a = Tensor([5.0, 6.0], requires_grad=True)
b = Tensor([1.0, 2.0], requires_grad=True)
(a - b).sum().backward()
assert np.allclose(b.grad.numpy(), [-1, -1])
ok('Sub')

a = Tensor([2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0], requires_grad=True)
(a * b).sum().backward()
assert np.allclose(a.grad.numpy(), [4, 5])
ok('Mul')

a = Tensor([6.0, 8.0], requires_grad=True)
b = Tensor([2.0, 4.0], requires_grad=True)
(a / b).sum().backward()
assert np.allclose(a.grad.numpy(), [0.5, 0.25])
ok('Div')

a = Tensor([1.0, -2.0], requires_grad=True)
(-a).sum().backward()
assert np.allclose(a.grad.numpy(), [-1, -1])
ok('Neg')

a = Tensor([0.0, 1.0], requires_grad=True)
a.exp().sum().backward()
assert np.allclose(a.grad.numpy(), [1.0, np.e], atol=1e-5)
ok('Exp')

a = Tensor([1.0, float(np.e)], requires_grad=True)
a.log().sum().backward()
assert np.allclose(a.grad.numpy(), [1.0, 1.0 / np.e], atol=1e-5)
ok('Log')

a = Tensor([2.0, 3.0], requires_grad=True)
(a ** 3).sum().backward()
assert np.allclose(a.grad.numpy(), [12.0, 27.0])
ok('Pow')

a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
b = Tensor([[10.0], [20.0]], requires_grad=True)
(a + b).sum().backward()
assert a.grad.shape == (1, 3) and b.grad.shape == (2, 1)
ok('Broadcasting')

# ===== PHASE 2.2: Reduce + MatMul =====
print('\n=== Phase 2.2: Reduce + MatMul ===')

a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
a.sum().backward()
assert np.allclose(a.grad.numpy(), np.ones((2, 2)))
ok('Sum')

a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
a.mean().backward()
assert np.allclose(a.grad.numpy(), 0.25 * np.ones((2, 2)))
ok('Mean')

a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
a.sum(axis=1).sum().backward()
assert np.allclose(a.grad.numpy(), np.ones((2, 2)))
ok('Sum(axis)')

A = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
B = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
(A @ B).sum().backward()
assert A.grad.shape == (3, 4) and B.grad.shape == (4, 2)
ok('MatMul shapes')

A_np = np.random.randn(2, 3).astype(np.float32)
B_np = np.random.randn(3, 2).astype(np.float32)
A = Tensor(A_np.copy(), requires_grad=True)
B = Tensor(B_np.copy(), requires_grad=True)
(A @ B).sum().backward()
eps = 1e-4
grad_A_num = np.zeros_like(A_np)
for i in range(A_np.shape[0]):
    for j in range(A_np.shape[1]):
        Ap = A_np.copy(); Ap[i, j] += eps
        Am = A_np.copy(); Am[i, j] -= eps
        grad_A_num[i, j] = (np.sum(Ap @ B_np) - np.sum(Am @ B_np)) / (2 * eps)
assert np.allclose(A.grad.numpy(), grad_A_num, atol=1e-3)
ok('MatMul grad check')

# ===== PHASE 2.3: Shape ops =====
print('\n=== Phase 2.3: Shape ops ===')

a = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
a.reshape(3, 2).sum().backward()
assert a.grad.shape == (2, 3)
ok('Reshape')

a = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
a.T.sum().backward()
assert a.grad.shape == (2, 3)
ok('Transpose')

a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
a.unsqueeze(0).squeeze(0).sum().backward()
assert a.grad.shape == (3, 4)
ok('Unsqueeze/Squeeze')

a = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)
a.expand(3, 4).sum().backward()
assert a.grad.shape == (3, 1) and np.allclose(a.grad.numpy(), [[4], [4], [4]])
ok('Expand')

a = Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
a[1:3, 2:4].sum().backward()
exp = np.zeros((4, 5)); exp[1:3, 2:4] = 1.0
assert np.allclose(a.grad.numpy(), exp)
ok('Slice')

a = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
b = Tensor(np.ones((2, 3), dtype=np.float32) * 2, requires_grad=True)
cat([a, b], dim=0).sum().backward()
assert a.grad.shape == (2, 3) and b.grad.shape == (2, 3)
ok('Cat')

a = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
b = Tensor(np.ones(3, dtype=np.float32) * 2, requires_grad=True)
stack([a, b], dim=0).sum().backward()
assert a.grad.shape == (3,)
ok('Stack')

a = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
a.flatten().sum().backward()
assert a.grad.shape == (2, 3, 4)
ok('Flatten')

# ===== PHASE 2.4: Activations =====
print('\n=== Phase 2.4: Activations ===')

a = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
a.relu().sum().backward()
assert np.allclose(a.grad.numpy(), [0, 0, 1, 1])
ok('ReLU')

a = Tensor([0.0], requires_grad=True)
a.sigmoid().sum().backward()
assert np.allclose(a.grad.numpy(), [0.25], atol=1e-5)
ok('Sigmoid')

a = Tensor([0.0], requires_grad=True)
a.tanh().sum().backward()
assert np.allclose(a.grad.numpy(), [1.0], atol=1e-5)
ok('Tanh')

from picograd.ops.activations import GELU, LeakyReLU, Softmax, LogSoftmax

a = Tensor([0.0, 1.0, -1.0], requires_grad=True)
GELU.apply(a).sum().backward()
assert a.grad.shape == (3,)
ok('GELU')

a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
LeakyReLU.apply(a, negative_slope=0.1).sum().backward()
assert np.allclose(a.grad.numpy(), [0.1, 0.1, 0.1, 1.0, 1.0], atol=1e-5)
ok('LeakyReLU')

a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
s = Softmax.apply(a, dim=-1)
assert np.allclose(s.numpy().sum(), 1.0, atol=1e-5)
ok('Softmax forward')

a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
LogSoftmax.apply(a, dim=-1).sum().backward()
assert np.allclose(a.grad.numpy().sum(), 0.0, atol=1e-5)
ok('LogSoftmax')

# Softmax numerical gradient check
a_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
a = Tensor(a_np.copy(), requires_grad=True)
Softmax.apply(a, dim=-1).sum().backward()
analytic = a.grad.numpy()
eps = 1e-4
num = np.zeros_like(a_np)
for i in range(3):
    ap = a_np.copy(); ap[0, i] += eps
    am = a_np.copy(); am[0, i] -= eps
    def sm(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    num[0, i] = (sm(ap).sum() - sm(am).sum()) / (2 * eps)
assert np.allclose(analytic, num, atol=1e-3)
ok('Softmax grad check')

print('\n========== ALL TESTS PASSED (Phases 1-2.4) ==========')
