import torch
from flkit.core.layers import SimpleDense
from flkit.core.vectorize import flatten_params, load_flat_params

def test_flatten_roundtrip():
    m = SimpleDense(10, 2)
    v = flatten_params(m)
    v2 = v.clone()
    v2 += 0.1
    load_flat_params(m, v2)
    assert torch.allclose(flatten_params(m), v2)
