"""picograd/nn/parameter.py"""
from __future__ import annotations
from picograd.tensor import Tensor
import numpy as np


class Parameter(Tensor):
    """
    A Tensor that is automatically registered as a model parameter
    when assigned as a Module attribute.

    Parameters always have requires_grad=True.
    """

    def __init__(self, data, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self):
        return f"Parameter(shape={self.shape}, dtype={self.dtype})"


__all__ = ["Parameter"]
