"""
picograd — A from-scratch deep learning framework with DAG-based autograd.

Public API surface mirrors PyTorch where possible for easy porting.

Usage:
    import picograd
    import picograd.nn as nn
    import picograd.optim as optim
    from picograd import Tensor
"""

# Core tensor
from picograd.tensor import Tensor, cat, stack

# Autograd
from picograd.autograd.context import no_grad, enable_grad
from picograd.autograd.engine import backward

# Backend management
from picograd.backend import get_backend, set_backend, NumpyBackend

# Serialization
from picograd.serialization import save, load, state_dict_to_bytes, bytes_to_state_dict

# Utilities
from picograd.utils.seed import manual_seed
from picograd.utils.debug import gradcheck, detect_anomaly

# Sub-packages (imported as modules, matching PyTorch style)
from . import nn
from . import optim
from . import data


# Functional helpers — commonly used ops accessible at top level
import numpy as np


def tensor(data, requires_grad: bool = False, dtype=None) -> Tensor:
    """Create a Tensor from data (array, list, scalar)."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor.zeros(*shape, requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor.ones(*shape, requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor.randn(*shape, requires_grad=requires_grad)


def rand(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor.rand(*shape, requires_grad=requires_grad)


def zeros_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor.zeros_like(t, requires_grad=requires_grad)


def ones_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor.ones_like(t, requires_grad=requires_grad)


def from_numpy(arr, requires_grad: bool = False) -> Tensor:
    return Tensor.from_numpy(arr, requires_grad=requires_grad)


def arange(start, stop=None, step=1, dtype=None) -> Tensor:
    return Tensor.arange(start, stop, step, dtype=dtype)


def eye(n, dtype=None) -> Tensor:
    return Tensor.eye(n, dtype=dtype)


def full(shape, fill_value, requires_grad: bool = False) -> Tensor:
    return Tensor.full(shape, fill_value, requires_grad=requires_grad)


__version__ = "0.1.0"

__all__ = [
    # Core
    "Tensor", "tensor", "cat", "stack",
    # Constructors
    "zeros", "ones", "randn", "rand", "zeros_like", "ones_like",
    "from_numpy", "arange", "eye", "full",
    # Autograd
    "no_grad", "enable_grad", "backward",
    # Backend
    "get_backend", "set_backend", "NumpyBackend",
    # Serialization
    "save", "load", "state_dict_to_bytes", "bytes_to_state_dict",
    # Utils
    "manual_seed", "gradcheck", "detect_anomaly",
    # Sub-packages
    "nn", "optim", "data",
]
