"""
picograd/autograd/engine.py
============================
Reverse-mode automatic differentiation engine.

`backward(root_tensor, grad)` performs:
  1. Topological sort of the DAG (reverse post-order DFS).
  2. Walk nodes in reverse order, calling Function.backward().
  3. Route and accumulate gradients back to leaf Tensors.

Fan-out (a tensor used in multiple ops) is handled by *accumulating*
gradients: leaf.grad += incoming_grad  (not just assignment).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from picograd.tensor import Tensor
    from picograd.autograd.function import Node


def _build_topo(root: "Tensor") -> List["Node"]:
    """
    Return a list of Nodes in *reverse* topological order (output first).
    Uses iterative DFS with a visited set to handle DAGs (shared subgraphs).
    """
    order: List["Node"] = []
    visited = set()

    def dfs(node: "Node") -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for (tensor, parent_node) in node.input_nodes:
            if parent_node is not None:
                dfs(parent_node)
        order.append(node)

    if root._grad_fn is not None:
        dfs(root._grad_fn)

    # order is currently: leaves → root (forward order).
    # We need root → leaves (reverse order) for backprop.
    order.reverse()
    return order


def backward(root: "Tensor", grad=None) -> None:
    """
    Compute gradients for all leaf tensors that contributed to root.

    Parameters
    ----------
    root  : Tensor — scalar (or any shape) tensor to differentiate from.
    grad  : initial gradient (same shape as root).  Defaults to ones.
    """
    from picograd.tensor import Tensor
    from picograd.backend import get_backend

    b = get_backend()

    # Seed gradient
    if grad is None:
        grad = b.ones(b.shape_of(root._data))
    elif isinstance(grad, Tensor):
        grad = grad._data

    # Map from Node-id to accumulated gradient (raw backend array)
    grad_map: Dict[int, object] = {}

    if root._grad_fn is None:
        # root is a leaf — accumulate directly
        if root.requires_grad:
            if root.grad is None:
                root.grad = Tensor(b.copy(grad), requires_grad=False)
            else:
                root.grad._data = b.add(root.grad._data, grad)
        return

    # Seed the root node
    grad_map[id(root._grad_fn)] = grad

    topo_order = _build_topo(root)

    for node in topo_order:
        if id(node) not in grad_map:
            continue
        g_out = grad_map[id(node)]

        # Call the operation's backward
        grads = node.function.backward(node.ctx, g_out)
        if not isinstance(grads, (list, tuple)):
            grads = (grads,)

        # Route gradients to parent tensors / nodes
        for (tensor, parent_node), g in zip(node.input_nodes, grads):
            if g is None or tensor is None:
                continue
            if not tensor.requires_grad:
                continue

            if parent_node is None:
                # tensor is a leaf — accumulate grad
                if tensor.grad is None:
                    tensor.grad = Tensor(b.copy(g), requires_grad=False)
                else:
                    tensor.grad._data = b.add(tensor.grad._data, g)
            else:
                # tensor is a non-leaf — accumulate in grad_map for the node
                pid = id(parent_node)
                if pid in grad_map:
                    grad_map[pid] = b.add(grad_map[pid], g)
                else:
                    grad_map[pid] = g


__all__ = ["backward"]
