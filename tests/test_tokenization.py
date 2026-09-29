
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hsd.common.labels import TARGET_NAMES
from hsd.training.tokenization import TARGET_COLUMNS, TokenizationConfig, tokenize_dataset


def build_sample_frame() -> pd.DataFrame:
    rows = []
    for index in range(6):
        source = "mhs" if index % 2 == 0 else "jigsaw"
        row = {
            "uid": f"u{index}",
            "source": source,
            "text": f"sample text number {index}",
            "label": index % 4,
            "severity": -100 if index % 4 == 0 else index % 4,
            "split": "train" if index < 4 else "val",
        }
        for name in TARGET_NAMES:
            row[f"t_{name}"] = bool(index % 2 == 0) if source == "mhs" else None
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_parquet(tmp_path: Path) -> Path:
    frame = build_sample_frame()
    path = tmp_path / "dataset.parquet"
    frame.to_parquet(path)
    return path


def test_tokenize_dataset_produces_expected_columns(tmp_path: Path, sample_parquet: Path) -> None:
    config = TokenizationConfig(
        parquet_path=sample_parquet,
        cache_dir=tmp_path / "cache",
        tokenizer_name="xlm-roberta-base",
        max_length=16,
        num_proc=1,
    )
    dataset_dict = tokenize_dataset(config)
    assert set(dataset_dict.keys()) == {"train", "val"}
    train_dataset = dataset_dict["train"]
    expected_columns = [
        "input_ids", "attention_mask", "label", "severity", "has_targets", "length"
    ] + TARGET_COLUMNS
    for column in expected_columns:
        assert column in train_dataset.column_names


def test_has_targets_matches_source(tmp_path: Path, sample_parquet: Path) -> None:
    config = TokenizationConfig(
        parquet_path=sample_parquet,
        cache_dir=tmp_path / "cache",
        tokenizer_name="xlm-roberta-base",
        max_length=16,
        num_proc=1,
    )
    dataset_dict = tokenize_dataset(config)
    combined = pd.concat([dataset_dict[split].to_pandas() for split in dataset_dict.keys()])
    assert (combined["has_targets"] == (combined["source"] == "mhs")).all()


def test_cache_is_reused(tmp_path: Path, sample_parquet: Path) -> None:
    config = TokenizationConfig(
        parquet_path=sample_parquet,
        cache_dir=tmp_path / "cache",
        tokenizer_name="xlm-roberta-base",
        max_length=16,
        num_proc=1,
    )
    tokenize_dataset(config)
    cache_files_first = sorted((config.cache_dir).rglob("*"))
    first_mtimes = [path.stat().st_mtime for path in cache_files_first if path.is_file()]
    tokenize_dataset(config)
    cache_files_second = sorted((config.cache_dir).rglob("*"))
    second_mtimes = [path.stat().st_mtime for path in cache_files_second if path.is_file()]
    assert first_mtimes == second_mtimes
