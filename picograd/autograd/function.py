"""
picograd/autograd/function.py
==============================
Core autograd primitives:

  Function   — abstract base for all differentiable operations
  Context    — holds saved tensors / metadata for the backward pass
  Node       — a vertex in the DAG; wraps a Function + its Context

Every differentiable op in picograd/ops/ subclasses Function and
implements forward() + backward().  apply() is the only public entry
point; it orchestrates forward execution and DAG construction.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Tuple, List

if TYPE_CHECKING:
    from picograd.tensor import Tensor


class Context:
    """
    Scratch-pad passed to forward() and backward().

    Forward saves tensors (or plain values) needed for backward:
        ctx.save_for_backward(input, weight)
        ctx.stride = stride               # arbitrary metadata

    Backward retrieves them:
        inp, wgt = ctx.saved_tensors
        s = ctx.stride
    """

    def __init__(self):
        self._saved_tensors: Tuple["Tensor", ...] = ()
        self._needs_input_grad: Tuple[bool, ...] = ()

    def save_for_backward(self, *tensors: "Tensor") -> None:
        """Save tensors for retrieval in backward()."""
        self._saved_tensors = tensors

    @property
    def saved_tensors(self) -> Tuple["Tensor", ...]:
        return self._saved_tensors


class Node:
    """
    A vertex in the computational DAG.

    Attributes
    ----------
    function    : type[Function]  — the op class (not instance)
    ctx         : Context         — state saved during forward
    input_nodes : list[Node|None] — parent nodes (None for leaves)
    """

    __slots__ = ("function", "ctx", "input_nodes")

    def __init__(self, function, ctx: Context, input_nodes: List):
        self.function = function
        self.ctx = ctx
        self.input_nodes = input_nodes   # list[Optional[Node]]


class Function:
    """
    Abstract base for all differentiable operations.

    Subclasses must implement:
      forward(ctx, *inputs) → raw backend array(s)
      backward(ctx, *grad_outputs) → tuple of raw backend arrays (one per input)

    The `apply` classmethod is the single entry point:
      out = Add.apply(a, b)
    """

    @staticmethod
    def forward(ctx: Context, *inputs) -> Any:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, *grad_outputs) -> Tuple:
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: "Tensor") -> "Tensor":
        """
        1. Run forward() with raw backend data.
        2. If any input requires grad and grad is enabled, build a Node.
        3. Return output Tensor (with _grad_fn set if appropriate).
        """
        from picograd.tensor import Tensor
        from picograd.autograd.context import _grad_enabled

        # Determine if we need to track this op
        needs_grad = _grad_enabled() and any(
            isinstance(t, Tensor) and t.requires_grad for t in inputs
        )

        ctx = Context()
        # Pass raw data to forward (avoids circular imports in ops)
        raw_inputs = [t._data if isinstance(t, Tensor) else t for t in inputs]
        raw_out = cls.forward(ctx, *raw_inputs)

        # Wrap output
        if isinstance(raw_out, (list, tuple)):
            # Multi-output ops (e.g. split) return a tuple of Tensors
            out_tensors = []
            for r in raw_out:
                t = Tensor(r, requires_grad=needs_grad)
                out_tensors.append(t)
            if needs_grad:
                node = Node(cls, ctx, [t._grad_fn for t in inputs if isinstance(t, Tensor)])
                for t in out_tensors:
                    t._grad_fn = node
            return tuple(out_tensors)

        out = Tensor(raw_out, requires_grad=needs_grad)
        if needs_grad:
            input_nodes = []
            for t in inputs:
                if isinstance(t, Tensor):
                    # leaf: no grad_fn; non-leaf: has grad_fn
                    input_nodes.append((t, t._grad_fn))
                else:
                    input_nodes.append((None, None))
            node = Node(cls, ctx, input_nodes)
            out._grad_fn = node
            out._is_leaf = False
        return out


__all__ = ["Function", "Context", "Node"]
