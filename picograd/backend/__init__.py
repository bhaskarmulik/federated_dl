"""Backend registry — module-level singleton for the active backend.

Usage::

    from picograd.backend import get_backend, set_backend
    B = get_backend()          # returns the current Backend instance
    B.matmul(a, b)
"""

from __future__ import annotations

from typing import Optional

from ._base import Backend

__all__ = ["get_backend", "set_backend", "Backend"]

_CURRENT_BACKEND: Optional[Backend] = None


def get_backend() -> Backend:
    """Return the active backend, lazily initialising to ``NumpyBackend``."""
    global _CURRENT_BACKEND
    if _CURRENT_BACKEND is None:
        from .numpy_backend import NumpyBackend
        _CURRENT_BACKEND = NumpyBackend()
    return _CURRENT_BACKEND


def set_backend(backend: Backend) -> None:
    """Replace the active backend globally."""
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = backend
