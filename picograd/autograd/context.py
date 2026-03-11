"""Autograd context managers: ``no_grad`` and ``enable_grad``."""

from __future__ import annotations

import threading

__all__ = ["no_grad", "enable_grad", "is_grad_enabled"]

# Thread-local flag, (every thread - own op)
_tls = threading.local()


def is_grad_enabled() -> bool:
    return getattr(_tls, "grad_enabled", True)


def _set_grad_enabled(val: bool) -> None:
    _tls.grad_enabled = val


class no_grad:
    """Cwantext manager for disabling them gradient tracking stuff."""

    def __enter__(self) -> "no_grad":
        self._prev = is_grad_enabled()
        _set_grad_enabled(False)
        return self

    def __exit__(self, *args: object) -> None:
        _set_grad_enabled(self._prev)

    def __call__(self, func):  # decorator usage
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class enable_grad:

    def __enter__(self) -> "enable_grad":
        self._prev = is_grad_enabled()
        _set_grad_enabled(True)
        return self

    def __exit__(self, *args: object) -> None:
        _set_grad_enabled(self._prev)
