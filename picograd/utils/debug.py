"""
picograd/utils/debug.py
========================
Gradient checking (finite differences) and anomaly detection.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Sequence, Optional


def _zero_all(inputs):
    """Zero gradients on all Tensor inputs."""
    for inp in inputs:
        inp.grad = None


def gradcheck(
    func: Callable,
    inputs: Sequence,
    eps: float = 1e-4,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    verbose: bool = True,
) -> bool:
    """
    Verify analytical gradients against finite differences.

    Parameters
    ----------
    func   : function taking picograd Tensors -> scalar Tensor
    inputs : list of Tensors with requires_grad=True
    eps    : finite difference step
    atol   : absolute tolerance
    rtol   : relative tolerance

    Returns True if all gradients match within tolerance.
    """
    from picograd.tensor import Tensor

    # Forward + backward pass to get analytic gradients
    _zero_all(inputs)
    out = func(*inputs)
    out.backward()

    # Capture analytic grads BEFORE running FD (which modifies data in-place)
    analytic_grads = []
    for inp in inputs:
        if inp.requires_grad:
            analytic_grads.append(
                inp.grad._data.copy() if inp.grad is not None else np.zeros_like(inp._data)
            )
        else:
            analytic_grads.append(None)

    # Reset all grads
    _zero_all(inputs)

    passed = True
    for i, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue

        analytic_grad = analytic_grads[i]
        numeric_grad = np.zeros_like(inp._data)

        for j in range(inp._data.size):
            orig = float(inp._data.flat[j])

            inp._data.flat[j] = orig + eps
            _zero_all(inputs)
            val_plus = func(*inputs)._data.sum()

            inp._data.flat[j] = orig - eps
            _zero_all(inputs)
            val_minus = func(*inputs)._data.sum()

            numeric_grad.flat[j] = (float(val_plus) - float(val_minus)) / (2.0 * eps)
            inp._data.flat[j] = orig  # restore

        _zero_all(inputs)

        max_err = float(np.max(np.abs(analytic_grad - numeric_grad)))
        rel_err = max_err / (float(np.max(np.abs(numeric_grad))) + 1e-8)

        ok = max_err < atol or rel_err < rtol
        if verbose:
            status = "[PASS]" if ok else "[FAIL]"
            print(f"  input[{i}]: max_abs_err={max_err:.2e}  rel_err={rel_err:.2e}  {status}")
        if not ok:
            passed = False

    return passed


class detect_anomaly:
    """
    Context manager that raises on NaN/Inf in gradients.
    Usage:
        with picograd.detect_anomaly():
            out = model(x)
            out.backward()
    """
    def __enter__(self):
        self._orig = _anomaly_mode[0]
        _anomaly_mode[0] = True
        return self

    def __exit__(self, *args):
        _anomaly_mode[0] = self._orig


_anomaly_mode = [False]


def is_anomaly_mode() -> bool:
    return _anomaly_mode[0]
