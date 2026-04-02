"""Minimal dataset and dataloader utilities for picograd."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from picograd.tensor import Tensor


class Dataset:
    """Base dataset interface."""

    def __len__(self) -> int:
        raise NotImplementedError
    
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError


class NumpyDataset(Dataset):
    """Dataset backed by one or more NumPy-compatible arrays."""

    def __init__(self, *arrays: Any):
        if not arrays:
            raise ValueError("NumpyDataset requires at least one array")

        self.arrays = tuple(np.asarray(arr) for arr in arrays)
        n_samples = len(self.arrays[0])
        if any(len(arr) != n_samples for arr in self.arrays[1:]):
            raise ValueError("All arrays in NumpyDataset must have the same length")

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, index: int):
        items = tuple(Tensor(arr[index]) for arr in self.arrays)
        if len(items) == 1:
            return items[0]
        return items


class DataLoader:
    """Simple batch loader with optional shuffling."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        size = len(self.dataset)
        if self.drop_last:
            return size // self.batch_size
        return (size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterable[Any]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            batch = [self.dataset[int(i)] for i in batch_indices]
            yield self._collate(batch)

    @staticmethod
    def _collate(batch: Sequence[Any]) -> Any:
        first = batch[0]
        if isinstance(first, (tuple, list)):
            columns = zip(*batch)
            return tuple(DataLoader._stack(values) for values in columns)
        return DataLoader._stack(batch)

    @staticmethod
    def _stack(values: Sequence[Any]) -> Tensor:
        first = values[0]
        if isinstance(first, Tensor):
            data = np.stack([value._data for value in values], axis=0)
            requires_grad = any(value.requires_grad for value in values)
            return Tensor(data, requires_grad=requires_grad)
        return Tensor(np.stack([np.asarray(value) for value in values], axis=0))
