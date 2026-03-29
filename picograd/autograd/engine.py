"""Backward engine — topological-sort-based reverse-mode autodiff."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from ..tensor import Tensor
    from .function import Node

__all__ = ["backward"]


def _topo_sort(root: "Node") -> List["Node"]:
    """Return nodes in reverse-topological order (root first)."""
    visited: Set[int] = set()
    order: List["Node"] = []

    def _dfs(node: Optional["Node"]) -> None:
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        for parent in node.inputs:
            _dfs(parent)
        order.append(node)

    _dfs(root)
    order.reverse()  # root first → we process from output toward leaves
    return order


def backward(root_tensor: "Tensor", grad: Optional["Tensor"] = None) -> None:
    """Compute gradients for all leaves reachable from *root_tensor*.

    Parameters
    ----------
    root_tensor:
        Must be a scalar (``numel == 1``) unless *grad* is provided.
    grad:
        Seed gradient (same shape as *root_tensor*).  Defaults to ones.
    """
    from ..tensor import Tensor

    if root_tensor._grad_fn is None:
        return  # leaf — nothing to differentiate

    if grad is None:
        if root_tensor.numel != 1:
            raise RuntimeError(
                "backward() requires grad for non-scalar tensors"
            )
        grad = Tensor(root_tensor._backend.ones(root_tensor.shape,
                                                  dtype=root_tensor.dtype),
                       requires_grad=False)

    # ----- topological ordering -------------------------------------------
    nodes = _topo_sort(root_tensor._grad_fn)

    # Map  node-id  →  accumulated gradient *of that node's output*
    grad_map: Dict[int, "Tensor"] = {id(root_tensor._grad_fn): grad}

    for node in nodes:
        out_grad = grad_map.get(id(node))
        if out_grad is None:
            continue

        # Call the backward of the function that produced this node.
        grads = node.function_cls.backward(node.ctx, out_grad)

        # ``grads`` is a tuple with one entry per original input.
        input_tensors = node._input_tensors

        for inp_tensor, inp_node, g in zip(input_tensors, node.inputs, grads):
            if g is None:
                continue
            if not isinstance(inp_tensor, Tensor):
                continue
            if not inp_tensor.requires_grad:
                continue

            # --- leaf tensor: accumulate into .grad ---------------------------
            if inp_tensor.is_leaf:
                if inp_tensor.grad is None:
                    inp_tensor.grad = Tensor(g._data.copy(), requires_grad=False)
                else:
                    inp_tensor.grad._data = inp_tensor._backend.add(
                        inp_tensor.grad._data, g._data
                    )
            # --- intermediate: accumulate for its grad_fn --------------------
            if inp_tensor._grad_fn is not None:
                nid = id(inp_tensor._grad_fn)
                if nid in grad_map:
                    grad_map[nid] = Tensor(
                        inp_tensor._backend.add(grad_map[nid]._data, g._data),
                        requires_grad=False,
                    )
                else:
                    grad_map[nid] = g
