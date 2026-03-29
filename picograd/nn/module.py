"""
picograd/nn/module.py
======================
Base Module class — the foundation of every neural network layer.

Mirrors PyTorch's nn.Module:
  - Parameters registered via __setattr__
  - state_dict() / load_state_dict() for serialization
  - train() / eval() mode toggling
  - children() / named_parameters() iterators
  - __call__ dispatches to forward()
"""

from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Tuple, Any
import numpy as np


class Module:
    """Base class for all picograd neural network modules."""

    def __init__(self):
        # Private storage (bypass __setattr__ to avoid infinite recursion)
        object.__setattr__(self, '_parameters', {})   # name → Parameter
        object.__setattr__(self, '_modules', {})       # name → Module
        object.__setattr__(self, '_buffers', {})       # name → ndarray (non-grad)
        object.__setattr__(self, '_training', True)

    # ------------------------------------------------------------------ attribute dispatch
    def __setattr__(self, name: str, value):
        from picograd.nn.parameter import Parameter
        # Route to the appropriate private dict
        params    = object.__getattribute__(self, '_parameters')
        modules   = object.__getattribute__(self, '_modules')
        buffers   = object.__getattribute__(self, '_buffers')

        if isinstance(value, Parameter):
            params[name] = value
        elif isinstance(value, Module):
            modules[name] = value
        else:
            # Clear from special dicts if overwriting
            params.pop(name, None)
            modules.pop(name, None)
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        _params  = object.__getattribute__(self, '_parameters')
        _modules = object.__getattribute__(self, '_modules')
        _buffers = object.__getattribute__(self, '_buffers')

        if name in _params:
            return _params[name]
        if name in _modules:
            return _modules[name]
        if name in _buffers:
            return _buffers[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ------------------------------------------------------------------ training mode
    @property
    def training(self) -> bool:
        return object.__getattribute__(self, '_training')

    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, '_training', mode)
        for m in self.children():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # ------------------------------------------------------------------ iterators
    def children(self) -> Iterator["Module"]:
        for m in object.__getattribute__(self, '_modules').values():
            yield m

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        for k, v in object.__getattribute__(self, '_modules').items():
            yield k, v

    def modules(self) -> Iterator["Module"]:
        yield self
        for m in self.children():
            yield from m.modules()

    def parameters(self, recurse: bool = True) -> Iterator:
        """Yield all Parameters (optionally recursively)."""
        yield from self._parameters.values()
        if recurse:
            for m in self.children():
                yield from m.parameters(recurse=True)

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Any]]:
        for name, p in object.__getattribute__(self, '_parameters').items():
            full = f"{prefix}.{name}" if prefix else name
            yield full, p
        if recurse:
            for child_name, child in object.__getattribute__(self, '_modules').items():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                yield from child.named_parameters(prefix=child_prefix, recurse=True)

    # ------------------------------------------------------------------ state dict
    def state_dict(self, prefix: str = "") -> Dict[str, np.ndarray]:
        """Return ordered dict of all parameter tensors as numpy arrays."""
        sd = {}
        for name, p in object.__getattribute__(self, '_parameters').items():
            key = f"{prefix}{name}"
            sd[key] = p._data.copy()
        for child_name, child in object.__getattribute__(self, '_modules').items():
            child_prefix = f"{prefix}{child_name}."
            sd.update(child.state_dict(prefix=child_prefix))
        # Also include buffers
        for name, buf in object.__getattribute__(self, '_buffers').items():
            key = f"{prefix}{name}"
            if buf is not None:
                sd[key] = buf.copy() if isinstance(buf, np.ndarray) else buf
        return sd

    def load_state_dict(self, sd: Dict[str, np.ndarray],
                        strict: bool = True, prefix: str = "") -> None:
        """Load parameters from a state dict."""
        for name, p in object.__getattribute__(self, '_parameters').items():
            key = f"{prefix}{name}"
            if key in sd:
                p._data = np.array(sd[key], dtype=np.float32)
            elif strict:
                raise KeyError(f"Missing key '{key}' in state_dict")

        for child_name, child in object.__getattribute__(self, '_modules').items():
            child_prefix = f"{prefix}{child_name}."
            child.load_state_dict(sd, strict=strict, prefix=child_prefix)

        for name in object.__getattribute__(self, '_buffers'):
            key = f"{prefix}{name}"
            if key in sd:
                object.__getattribute__(self, '_buffers')[name] = np.array(sd[key], dtype=np.float32)

    def register_buffer(self, name: str, tensor) -> None:
        """Register a non-parameter buffer (e.g. running_mean in BatchNorm)."""
        bufs = object.__getattribute__(self, '_buffers')
        bufs[name] = tensor

    # ------------------------------------------------------------------ forward
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.forward() not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------ grad utilities
    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for p in self.parameters():
            p.grad = None

    # ------------------------------------------------------------------ repr
    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        if self.extra_repr():
            lines.append(f"  {self.extra_repr()}")
        for name, m in object.__getattribute__(self, '_modules').items():
            m_repr = repr(m).replace('\n', '\n  ')
            lines.append(f"  ({name}): {m_repr}")
        lines.append(")")
        return "\n".join(lines)


__all__ = ["Module"]
