
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..tensor import Tensor

__all__ = ["Function", "Context", "Node"]

#yeh kyuki model ne abtaya
class Context:

    def __init__(self) -> None:
        self._saved: Tuple["Tensor", ...] = ()
        self._non_tensor: dict[str, Any] = {}

    def save_for_backward(self, *tensors: "Tensor") -> None:
        self._saved = tensors

    @property
    def saved_tensors(self) -> Tuple["Tensor", ...]:
        return self._saved

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self._non_tensor[key] = value

    def __getattr__(self, key: str) -> Any:
        try:
            return self._non_tensor[key]
        except KeyError:
            raise AttributeError(key)


#Computing DAG node banaya
class Node:

    __slots__ = ("function_cls", "ctx", "inputs", "_input_tensors")

    def __init__(
        self,
        function_cls: type["Function"],
        ctx: Context,
        inputs: Tuple[Optional["Node"], ...],
    ) -> None:
        self.function_cls = function_cls
        self.ctx = ctx
        self.inputs = inputs
        self._input_tensors: Tuple = ()


class Function:

    @staticmethod
    def forward(ctx: Context, *args: Any, **kwargs: Any) -> "Tensor":
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: "Tensor") -> Tuple[Optional["Tensor"], ...]:
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: "Tensor", **kwargs: Any) -> "Tensor":
        from ..tensor import Tensor
        from .context import is_grad_enabled

        ctx = Context()

        needs_grad = is_grad_enabled() and any(
            isinstance(t, Tensor) and t.requires_grad for t in inputs
        )

        result = cls.forward(ctx, *inputs, **kwargs)

        if needs_grad:
            parent_nodes: List[Optional[Node]] = []
            for inp in inputs:
                if isinstance(inp, Tensor) and inp.requires_grad:
                    parent_nodes.append(inp._grad_fn if inp._grad_fn is not None else None)
                else:
                    parent_nodes.append(None)

            node = Node(cls, ctx, tuple(parent_nodes))
            node._input_tensors = inputs
            result._grad_fn = node
            result.requires_grad = True
            result.is_leaf = False

        return result
