from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from hsd.common.labels import TARGET_NAMES
from hsd.training.sampler import EpochResampler, LengthGroupedSampler

TARGET_COLUMNS: list[str] = [f"t_{name}" for name in TARGET_NAMES]


def round_up_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


@dataclass
class DynamicPaddingCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 8

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        padded_length = round_up_to_multiple(max_length, self.pad_to_multiple_of)
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = torch.full((len(examples), padded_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), padded_length), dtype=torch.long)
        labels = torch.empty(len(examples), dtype=torch.long)
        severities = torch.empty(len(examples), dtype=torch.long)
        has_targets = torch.empty(len(examples), dtype=torch.bool)
        targets = torch.empty((len(examples), len(TARGET_COLUMNS)), dtype=torch.float)

        for row_index, example in enumerate(examples):
            sequence_length = len(example["input_ids"])
            input_ids[row_index, :sequence_length] = torch.tensor(
                example["input_ids"], dtype=torch.long
            )
            attention_mask[row_index, :sequence_length] = torch.tensor(
                example["attention_mask"], dtype=torch.long
            )
            labels[row_index] = int(example["label"])
            severities[row_index] = int(example["severity"])
            has_targets[row_index] = bool(example["has_targets"])
            for column_index, column in enumerate(TARGET_COLUMNS):
                targets[row_index, column_index] = float(example[column])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
            "severity": severities,
            "has_targets": has_targets,
            "targets": targets,
        }


def build_train_loader(
    dataset: Any,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_neither_ratio: float,
    seed: int,
    num_workers: int,
) -> DataLoader:
    resampler = EpochResampler(
        labels=dataset["label"],
        has_targets=dataset["has_targets"],
        max_neither_ratio=max_neither_ratio,
        seed=seed,
    )
    sampler = LengthGroupedSampler(
        lengths=dataset["length"],
        batch_size=batch_size,
        resampler=resampler,
        seed=seed,
    )
    collator = DynamicPaddingCollator(tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_eval_loader(
    dataset: Any,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    collator = DynamicPaddingCollator(tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
