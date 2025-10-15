# NOTE: This is a functional skeleton; fill in the generated proto imports after compiling fl.proto:
# from flkit.proto import fl_pb2, fl_pb2_grpc
from __future__ import annotations
import asyncio
from typing import Dict, List, Tuple
import torch

class CoordinatorServer:  # (fl_pb2_grpc.CoordinatorServicer)
    def __init__(self):
        self.round = 0
        self.buffer: List[Tuple[torch.Tensor, int]] = []
        self.B, self.T_ms, self.Smax = 6, 3000, 2

    async def aggregate_if_ready(self):
        if len(self.buffer) >= self.B:
            agg = sum((n/sum(n for _,n in self.buffer))*d for d,n in self.buffer)
            self.buffer.clear()
            # TODO: apply agg to global params
            return agg

# TODO: Implement gRPC server bootstrap using grpc.aio.server()
