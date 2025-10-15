from __future__ import annotations
from typing import Tuple
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class FolderPatches(Dataset):
    """Assumes structure:
    root/
      tumor/*.png|jpg
      normal/*.png|jpg
    """
    def __init__(self, root: str, transform=None):
        self.samples = []
        self.transform = transform
        for label, cls in enumerate(["normal","tumor"]):
            d = os.path.join(root, cls)
            if not os.path.isdir(d): 
                continue
            for name in os.listdir(d):
                if name.lower().endswith(('.png','.jpg','.jpeg')):
                    self.samples.append((os.path.join(d, name), label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        else:
            import torchvision.transforms as T
            tf = T.Compose([T.Resize((32,32)), T.ToTensor()])
            img = tf(img)
        return img, label
