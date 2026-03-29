"""
picograd/autograd/context.py
=============================
Thread-local flag controlling graph construction.

Usage:
    with picograd.no_grad():
        out = model(x)   # no DAG nodes created

    @picograd.no_grad()
    def inference(x):
        return model(x)
"""

import threading
from typing import Optional


_LOCAL = threading.local()


def _grad_enabled() -> bool:
    """Return True if gradient tracking is currently active."""
    return getattr(_LOCAL, "grad_enabled", True)


def _set_grad_enabled(enabled: bool) -> None:
    _LOCAL.grad_enabled = enabled


class no_grad:
    """Context manager / decorator that disables gradient computation."""

    def __enter__(self):
        self._prev = _grad_enabled()
        _set_grad_enabled(False)
        return self

    def __exit__(self, *args):
        _set_grad_enabled(self._prev)

    def __call__(self, func):
        """Allow use as a decorator."""
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with no_grad():
                return func(*args, **kwargs)
        return wrapper


class enable_grad:
    """Context manager that (re-)enables gradient computation."""

    def __enter__(self):
        self._prev = _grad_enabled()
        _set_grad_enabled(True)
        return self

    def __exit__(self, *args):
        _set_grad_enabled(self._prev)


__all__ = ["no_grad", "enable_grad", "_grad_enabled"]
