from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from hsd.common.logging import get_logger

logger = get_logger(__name__)

NEITHER_LABEL: int = 0


@dataclass
class EpochResampler:
    labels: Sequence[int]
    has_targets: Sequence[bool]
    max_neither_ratio: float
    seed: int
    neither_indices: np.ndarray = field(init=False)
    harmful_indices: np.ndarray = field(init=False)
    max_harmful_class_size: int = field(init=False)

    def __post_init__(self) -> None:
        labels_array = np.asarray(self.labels)
        self.neither_indices = np.where(labels_array == NEITHER_LABEL)[0]
        self.harmful_indices = np.where(labels_array != NEITHER_LABEL)[0]
        harmful_labels = labels_array[self.harmful_indices]
        if harmful_labels.size == 0:
            self.max_harmful_class_size = 0
        else:
            _, counts = np.unique(harmful_labels, return_counts=True)
            self.max_harmful_class_size = int(counts.max())

    def resample(self, epoch: int) -> list[int]:
        rng = np.random.default_rng(self.seed + epoch)
        target_neither_size = min(
            len(self.neither_indices),
            int(self.max_harmful_class_size * self.max_neither_ratio),
        )
        if target_neither_size <= 0 or len(self.neither_indices) == 0:
            sampled_neither = np.array([], dtype=int)
        else:
            sampled_neither = rng.choice(self.neither_indices, size=target_neither_size, replace=False)
        combined = np.concatenate([self.harmful_indices, sampled_neither])
        rng.shuffle(combined)
        has_targets_array = np.asarray(self.has_targets)
        share = float(has_targets_array[combined].mean()) if combined.size > 0 else 0.0
        logger.info(
            "epoch resampled",
            extra={"epoch": epoch, "num_examples": int(combined.size), "has_targets_share": share},
        )
        return combined.tolist()


class LengthGroupedSampler(Sampler[int]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        resampler: Optional[EpochResampler] = None,
        seed: int = 42,
        mega_batch_multiplier: int = 50,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.resampler = resampler
        self.seed = seed
        self.mega_batch_multiplier = mega_batch_multiplier
        self.epoch = 0
        self._cached_epoch: Optional[int] = None
        self._cached_indices: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _get_epoch_indices(self) -> list[int]:
        if self.resampler is None:
            return list(range(len(self.lengths)))
        if self._cached_epoch != self.epoch:
            self._cached_indices = self.resampler.resample(self.epoch)
            self._cached_epoch = self.epoch
        return self._cached_indices

    def __iter__(self) -> Iterator[int]:
        indices = self._get_epoch_indices()
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in permutation]
        mega_batch_size = self.batch_size * self.mega_batch_multiplier
        result: list[int] = []
        for start in range(0, len(shuffled), mega_batch_size):
            mega_batch = shuffled[start : start + mega_batch_size]
            mega_batch.sort(key=lambda index: self.lengths[index], reverse=True)
            result.extend(mega_batch)
        return iter(result)

    def __len__(self) -> int:
        return len(self._get_epoch_indices())
