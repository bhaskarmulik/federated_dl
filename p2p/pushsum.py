from __future__ import annotations
from typing import List, Tuple
import random
import torch

class PushSum:
    def __init__(self, init_vec: torch.Tensor, neighbors: List[str]):
        self.x = init_vec.clone().detach()
        self.w = torch.tensor(1.0, device=self.x.device)
        self.neighbors = neighbors

    async def step(self, inbox: List[Tuple[torch.Tensor, float]]):
        # Split mass
        send_x, send_w = self.x/2, self.w/2
        self.x, self.w = self.x/2, self.w/2
        # In a real impl, send to a random neighbor over TLS
        # await send(neighbor, send_x, send_w)
        # Process inbox
        for (rx, rw) in inbox:
            self.x += rx
            self.w += torch.tensor(rw, device=self.x.device)
        # Local model = x / w
        return self.x / self.w
