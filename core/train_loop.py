from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader

def train_one_epoch(model, loader: DataLoader, optimizer, device: str = "cpu") -> Tuple[float, float]:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        total += x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum())
    return loss_sum/total, correct/max(total,1)

@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str = "cpu") -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        total += x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum())
    return loss_sum/total, correct/max(total,1)
