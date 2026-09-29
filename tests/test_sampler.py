from __future__ import annotations

import numpy as np

from hsd.training.sampler import EpochResampler, LengthGroupedSampler


def test_epoch_resampler_caps_neither_ratio() -> None:
    labels = [0] * 100 + [1] * 10 + [2] * 5 + [3] * 3
    has_targets = [False] * 100 + [True] * 18
    resampler = EpochResampler(
        labels=labels, has_targets=has_targets, max_neither_ratio=3.0, seed=42
    )
    indices = resampler.resample(epoch=0)
    labels_array = np.asarray(labels)
    resampled_labels = labels_array[indices]
    neither_count = int((resampled_labels == 0).sum())
    harmful_count = int((resampled_labels != 0).sum())
    assert harmful_count == 18
    assert neither_count <= 10 * 3.0


def test_epoch_resampler_uses_all_harmful_rows() -> None:
    labels = [0] * 50 + [1] * 7 + [2] * 4
    has_targets = [False] * 50 + [True] * 11
    resampler = EpochResampler(
        labels=labels, has_targets=has_targets, max_neither_ratio=2.0, seed=1
    )
    indices = resampler.resample(epoch=0)
    labels_array = np.asarray(labels)
    resampled_labels = labels_array[indices]
    assert int((resampled_labels == 1).sum()) == 7
    assert int((resampled_labels == 2).sum()) == 4


def test_epoch_resampler_varies_by_epoch() -> None:
    labels = [0] * 200 + [1] * 20
    has_targets = [False] * 220
    resampler = EpochResampler(
        labels=labels, has_targets=has_targets, max_neither_ratio=3.0, seed=42
    )
    first_epoch = resampler.resample(epoch=0)
    second_epoch = resampler.resample(epoch=1)
    assert first_epoch != second_epoch


def test_length_grouped_sampler_sorts_within_mega_batches() -> None:
    lengths = list(range(500, 0, -1))
    lengths = lengths[::-1]
    sampler = LengthGroupedSampler(lengths=lengths, batch_size=4, seed=0)
    indices = list(iter(sampler))
    assert len(indices) == len(lengths)
    mega_batch_size = 4 * 50
    for start in range(0, len(indices), mega_batch_size):
        chunk = indices[start : start + mega_batch_size]
        chunk_lengths = [lengths[index] for index in chunk]
        assert chunk_lengths == sorted(chunk_lengths, reverse=True)


def test_length_grouped_sampler_with_resampler_respects_epoch() -> None:
    labels = [0] * 50 + [1] * 10
    has_targets = [False] * 60
    lengths = list(range(1, 61))
    resampler = EpochResampler(
        labels=labels, has_targets=has_targets, max_neither_ratio=2.0, seed=5
    )
    sampler = LengthGroupedSampler(lengths=lengths, batch_size=4, resampler=resampler, seed=5)
    sampler.set_epoch(0)
    epoch_zero_indices = list(iter(sampler))
    sampler.set_epoch(1)
    epoch_one_indices = list(iter(sampler))
    assert (
        set(epoch_zero_indices) != set(epoch_one_indices)
        or epoch_zero_indices != epoch_one_indices
    )
