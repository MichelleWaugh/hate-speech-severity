from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from hsd.common.config import SplitConfig
from hsd.common.logging import get_logger

logger = get_logger(__name__)

TRAIN = "train"
VAL = "val"
TEST = "test"


@dataclass(frozen=True)
class SplitAssignment:
    fold: np.ndarray
    split: pd.Series


def stratify_key(frame: pd.DataFrame) -> pd.Series:
    return frame["source"].astype(str) + "|" + frame["label_name"].astype(str)


def assign_folds(frame: pd.DataFrame, config: SplitConfig, seed: int) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=config.folds, shuffle=True, random_state=seed)
    folds = np.full(len(frame), -1, dtype="int16")
    strata = stratify_key(frame)
    groups = frame["text_hash"]
    for fold_index, (_, holdout_index) in enumerate(
        splitter.split(frame, strata, groups=groups)
    ):
        folds[holdout_index] = fold_index
    if (folds < 0).any():
        raise RuntimeError("Not every row received a fold assignment")
    return folds


def folds_to_split(folds: np.ndarray, config: SplitConfig) -> pd.Series:
    labels = np.where(
        folds == config.test_fold, TEST, np.where(folds == config.val_fold, VAL, TRAIN)
    )
    return pd.Series(labels, dtype="object")


def add_split_column(frame: pd.DataFrame, config: SplitConfig, seed: int) -> pd.DataFrame:
    if config.test_fold == config.val_fold:
        raise ValueError("test_fold and val_fold must differ")
    folds = assign_folds(frame, config, seed)
    return frame.assign(split=folds_to_split(folds, config).to_numpy())


def assert_no_leakage(frame: pd.DataFrame) -> None:
    hashes_per_split = frame.groupby("text_hash")["split"].nunique()
    leaking = hashes_per_split[hashes_per_split > 1]
    if len(leaking):
        raise ValueError(f"{len(leaking)} text_hash groups span more than one split")


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab([frame["source"], frame["label_name"]], frame["split"])


def apply_split(frame: pd.DataFrame, config: SplitConfig, seed: int) -> pd.DataFrame:
    result = add_split_column(frame, config, seed)
    assert_no_leakage(result)
    proportions = result["split"].value_counts(normalize=True).round(3).to_dict()
    logger.info("Split proportions: %s", proportions)
    return result