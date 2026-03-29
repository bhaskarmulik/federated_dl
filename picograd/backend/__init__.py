"""
picograd/backend/__init__.py
============================
Module-level singleton for the active backend.
Usage:
    from picograd.backend import get_backend, set_backend
    b = get_backend()
    b.matmul(a, b)

    # swap backends:
    picograd.set_backend(TritonBackend())
"""

from picograd.backend._base import Backend
from picograd.backend.numpy_backend import NumpyBackend

_BACKEND: Backend = NumpyBackend()   # default backend


def get_backend() -> Backend:
    """Return the currently active backend."""
    return _BACKEND


def set_backend(backend: Backend) -> None:
    """Replace the active backend with a new one."""
    global _BACKEND
    if not isinstance(backend, Backend):
        raise TypeError(f"Expected a Backend instance, got {type(backend)}")
    _BACKEND = backend


__all__ = ["Backend", "NumpyBackend", "get_backend", "set_backend"]
