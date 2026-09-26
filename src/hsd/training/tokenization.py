
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from hsd.common.labels import TARGET_NAMES
from hsd.common.logging import get_logger

logger = get_logger(__name__)

TARGET_COLUMNS: list[str] = [f"t_{name}" for name in TARGET_NAMES]

KEEP_COLUMNS: list[str] = [
    "input_ids",
    "attention_mask",
    "label",
    "severity",
    "source",
    "uid",
    "split",
    "has_targets",
    "length",
] + TARGET_COLUMNS


@dataclass(frozen=True)
class TokenizationConfig:
    parquet_path: Path
    cache_dir: Path
    tokenizer_name: str
    max_length: int
    num_proc: int


def compute_cache_key(config: TokenizationConfig) -> str:
    stat = config.parquet_path.stat()
    payload = f"{config.tokenizer_name}|{config.max_length}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_raw_dataframe(parquet_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(parquet_path)
    frame["has_targets"] = frame[TARGET_COLUMNS[0]].notna()
    for column in TARGET_COLUMNS:
        frame[column] = frame[column].fillna(False).astype(int)
    return frame


def build_tokenize_function(tokenizer: PreTrainedTokenizerBase, max_length: int):
    def tokenize_batch(examples: dict[str, list]) -> dict[str, list]:
        encoded = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["length"] = [len(ids) for ids in encoded["input_ids"]]
        return encoded

    return tokenize_batch


def tokenize_dataset(config: TokenizationConfig) -> DatasetDict:
    cache_key = compute_cache_key(config)
    cache_path = config.cache_dir / cache_key
    if cache_path.exists():
        logger.info("loading tokenized cache", extra={"cache_path": str(cache_path)})
        return load_from_disk(str(cache_path))

    logger.info("tokenizing dataset from scratch", extra={"cache_path": str(cache_path)})
    frame = load_raw_dataframe(config.parquet_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenize_batch = build_tokenize_function(tokenizer, config.max_length)

    splits: dict[str, Dataset] = {}
    for split_name in sorted(frame["split"].unique()):
        split_frame = frame[frame["split"] == split_name].reset_index(drop=True)
        dataset = Dataset.from_pandas(split_frame, preserve_index=False)
        dataset = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=2000,
            num_proc=config.num_proc,
        )
        dataset = dataset.remove_columns(
            [column for column in dataset.column_names if column not in KEEP_COLUMNS]
        )
        splits[str(split_name)] = dataset

    dataset_dict = DatasetDict(splits)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(cache_path))
    return dataset_dict


def parse_args() -> TokenizationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", type=Path, default=Path("data/processed/dataset.parquet"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/tokenized"))
    parser.add_argument("--tokenizer-name", type=str, default="xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-proc", type=int, default=2)
    args = parser.parse_args()
    return TokenizationConfig(
        parquet_path=args.parquet_path,
        cache_dir=args.cache_dir,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )


def main() -> None:
    config = parse_args()
    dataset_dict = tokenize_dataset(config)
    for split_name, dataset in dataset_dict.items():
        logger.info("tokenized split ready", extra={"split": split_name, "num_rows": len(dataset)})


if __name__ == "__main__":
    main()
