from __future__ import annotations
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from flkit.core.train_loop import train_one_epoch
from flkit.core.vectorize import flatten_params

class ClientAgent:
    def __init__(self, client_id: str, model, dataset, idxs, cfg: Dict[str, Any], device: str = "cpu"):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.loader = DataLoader(Subset(dataset, idxs), batch_size=cfg.get("batch_size", 64), shuffle=True)

    def local_train(self) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.get("lr", 1e-3))
        loss, acc = train_one_epoch(self.model, self.loader, opt, device=self.device)
        delta = flatten_params(self.model)  # for MVP we treat this as delta vs a known base externally
        stats = {"loss": loss, "acc": acc}
        n = len(self.loader.dataset)
        return delta, n, stats
