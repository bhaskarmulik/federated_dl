"""
picograd/nn/layers.py
======================
All built-in nn.Module layer implementations.
Each wraps the corresponding ops.* Function with learnable Parameters.
"""

from __future__ import annotations
import numpy as np
import math
from picograd.nn.module import Module
from picograd.nn.parameter import Parameter
from picograd.tensor import Tensor


# ---------------------------------------------------------------------------
# Linear (fully connected)
# ---------------------------------------------------------------------------

class Linear(Module):
    """y = x @ W^T + b   (matches torch.nn.Linear)"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming uniform init (PyTorch default)
        k = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(
            np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        )
        if bias:
            self.bias = Parameter(
                np.random.uniform(-k, k, (out_features,)).astype(np.float32)
            )
        else:
            self._no_bias = True

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features) → (..., out_features)
        out = x @ self.weight.T
        if hasattr(self, 'bias'):
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        kH, kW = self.kernel_size
        # Kaiming uniform
        n = in_channels * kH * kW
        k = 1.0 / math.sqrt(n)
        self.weight = Parameter(
            np.random.uniform(-k, k, (out_channels, in_channels, kH, kW)).astype(np.float32)
        )
        if bias:
            self.bias = Parameter(np.random.uniform(-k, k, (out_channels,)).astype(np.float32))
        else:
            self._no_bias = True

    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.convolution import Conv2d as Conv2dOp
        b = getattr(self, 'bias', None)
        return Conv2dOp.apply(x, self.weight, b,
                              self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}")


# ---------------------------------------------------------------------------
# ConvTranspose2d
# ---------------------------------------------------------------------------

class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

        kH, kW = self.kernel_size
        n = in_channels * kH * kW
        k = 1.0 / math.sqrt(n)
        # weight shape: (in_channels, out_channels, kH, kW)
        self.weight = Parameter(
            np.random.uniform(-k, k, (in_channels, out_channels, kH, kW)).astype(np.float32)
        )
        if bias:
            self.bias = Parameter(np.random.uniform(-k, k, (out_channels,)).astype(np.float32))
        else:
            self._no_bias = True

    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.convolution import ConvTranspose2d as CTOp
        b = getattr(self, 'bias', None)
        return CTOp.apply(x, self.weight, b,
                          self.stride, self.padding, self.dilation, self.output_padding)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}")


# ---------------------------------------------------------------------------
# BatchNorm2d / BatchNorm1d
# ---------------------------------------------------------------------------

class _BatchNormBase(Module):
    def __init__(self, num_features: int, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = Parameter(np.ones(num_features, dtype=np.float32))
            self.bias   = Parameter(np.zeros(num_features, dtype=np.float32))

        self.register_buffer('running_mean', np.zeros(num_features, dtype=np.float32))
        self.register_buffer('running_var',  np.ones(num_features, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.normalization import BatchNorm as BNOp
        # access buffers
        rm = object.__getattribute__(self, '_buffers')['running_mean']
        rv = object.__getattribute__(self, '_buffers')['running_var']
        w = self.weight if self.affine else Tensor(np.ones(self.num_features, dtype=np.float32))
        b = self.bias   if self.affine else Tensor(np.zeros(self.num_features, dtype=np.float32))
        return BNOp.apply(x, w, b, rm, rv, self.training, self.momentum, self.eps)

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"


class BatchNorm2d(_BatchNormBase):
    pass


class BatchNorm1d(_BatchNormBase):
    pass


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(np.ones(normalized_shape, dtype=np.float32))
            self.bias   = Parameter(np.zeros(normalized_shape, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.normalization import LayerNorm as LNOp
        w = self.weight if self.elementwise_affine else Tensor(np.ones(self.normalized_shape, dtype=np.float32))
        b = self.bias   if self.elementwise_affine else Tensor(np.zeros(self.normalized_shape, dtype=np.float32))
        return LNOp.apply(x, w, b, self.normalized_shape, self.eps)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}"


# ---------------------------------------------------------------------------
# Activation layers
# ---------------------------------------------------------------------------

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.activations import ReLU as ReLUOp
        return ReLUOp.apply(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.activations import LeakyReLU as LReLU
        return LReLU.apply(x, self.negative_slope)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.activations import Sigmoid as SigOp
        return SigOp.apply(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.activations import Tanh as TanhOp
        return TanhOp.apply(x)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.activations import GELU as GELUOp
        return GELUOp.apply(x)


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.activations import Softmax as SoftmaxOp
        return SoftmaxOp.apply(x, self.dim)


# ---------------------------------------------------------------------------
# Pooling layers
# ---------------------------------------------------------------------------

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.pooling import MaxPool2d as MPOp
        return MPOp.apply(x, self.kernel_size, self.stride, self.padding)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.pooling import AvgPool2d as APOp
        return APOp.apply(x, self.kernel_size, self.stride, self.padding)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.pooling import AdaptiveAvgPool2d as AAPOp
        return AAPOp.apply(x, self.output_size)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        from picograd.ops.dropout_embedding import Dropout as DOp
        return DOp.apply(x, self.p, self.training)

    def extra_repr(self):
        return f"p={self.p}"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        )

    def forward(self, indices: Tensor) -> Tensor:
        from picograd.ops.dropout_embedding import Embedding as EmbOp
        return EmbOp.apply(self.weight, indices)

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.embedding_dim}"


# ---------------------------------------------------------------------------
# Container modules
# ---------------------------------------------------------------------------

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, str(i), m)
        self._order = list(range(len(modules)))

    def forward(self, x: Tensor) -> Tensor:
        mods = object.__getattribute__(self, '_modules')
        for i in self._order:
            x = mods[str(i)](x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return object.__getattribute__(self, '_modules')[str(idx)]

    def __len__(self) -> int:
        return len(self._order)


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                setattr(self, str(i), m)
            self._len = len(modules)
        else:
            self._len = 0

    def append(self, module: Module) -> "ModuleList":
        setattr(self, str(self._len), module)
        self._len += 1
        return self

    def __getitem__(self, idx: int) -> Module:
        return object.__getattribute__(self, '_modules')[str(idx)]

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        for i in range(self._len):
            yield self[i]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList is not a sequential model; iterate manually.")


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

class Flatten(Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim)

    def extra_repr(self):
        return f"start_dim={self.start_dim}"


__all__ = [
    "Linear", "Conv2d", "ConvTranspose2d",
    "BatchNorm2d", "BatchNorm1d", "LayerNorm",
    "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU", "Softmax",
    "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
    "Dropout", "Embedding",
    "Sequential", "ModuleList", "Flatten",
]
