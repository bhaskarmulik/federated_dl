"""picograd — a from-scratch deep learning framework with autograd.

Usage::

    import picograd
    x = picograd.Tensor([2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    print(x.grad)  # [4.0, 6.0]
"""

from .tensor import Tensor, cat, stack, where
from .autograd import no_grad, enable_grad, is_grad_enabled, Function
from .backend import get_backend, set_backend

# Convenience re-exports matching torch.* surface
manual_seed = lambda n: get_backend().seed(n)

__all__ = [
    "Tensor",
    "cat",
    "stack",
    "where",
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "Function",
    "get_backend",
    "set_backend",
    "manual_seed",
]

__version__ = "0.1.0"
