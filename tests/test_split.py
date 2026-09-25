import numpy as np
import pandas as pd
import pytest

from hsd.common.config import SplitConfig
from hsd.data.dedup import add_conflict_flags, add_duplicate_flags, annotate_duplicates
from hsd.data.split import add_split_column, apply_split, assert_no_leakage, split_summary

RNG = np.random.default_rng(0)


def make_frame(
    n_hashes: int,
    per_hash: int,
    sources=("jigsaw", "mhs"),
    labels=("neither", "offensive", "hate", "threat"),
) -> pd.DataFrame:
    rows = []
    for hash_index in range(n_hashes):
        for copy_index in range(per_hash):
            rows.append(
                {
                    "uid": f"u{hash_index}-{copy_index}",
                    "text_hash": f"h{hash_index}",
                    "source": sources[hash_index % len(sources)],
                    "label": labels.index(labels[hash_index % len(labels)]),
                    "label_name": labels[hash_index % len(labels)],
                }
            )
    return pd.DataFrame(rows)


def test_add_duplicate_flags_marks_repeated_hashes_only() -> None:
    frame = pd.DataFrame(
        {
            "uid": ["a", "b", "c", "d"],
            "text_hash": ["h1", "h1", "h2", "h3"],
        }
    )
    result = add_duplicate_flags(frame)
    assert result["is_duplicate"].tolist() == [True, True, False, False]
    assert result["dup_group_size"].tolist() == [2, 2, 1, 1]
    assert len(result) == len(frame)


def test_add_conflict_flags_only_within_duplicate_groups() -> None:
    frame = pd.DataFrame(
        {
            "uid": ["a", "b", "c", "d", "e"],
            "text_hash": ["h1", "h1", "h2", "h2", "h3"],
            "label": [0, 1, 2, 2, 3],
            "is_duplicate": [True, True, True, True, False],
        }
    )
    result = add_conflict_flags(frame)
    assert result["label_conflict"].tolist() == [True, True, False, False, False]


def test_annotate_duplicates_preserves_row_count_and_individual_labels() -> None:
    frame = pd.DataFrame(
        {
            "uid": ["a", "b", "c"],
            "text_hash": ["h1", "h1", "h2"],
            "source": ["jigsaw", "mhs", "jigsaw"],
            "label": [0, 3, 1],
            "label_name": ["neither", "threat", "offensive"],
        }
    )
    result = annotate_duplicates(frame)
    assert len(result) == 3
    assert result["label"].tolist() == [0, 3, 1]
    assert result.loc[result["uid"] == "a", "label_conflict"].item()
    assert not result.loc[result["uid"] == "c", "label_conflict"].item()


def test_split_keeps_duplicate_groups_together() -> None:
    frame = make_frame(n_hashes=40, per_hash=3)
    config = SplitConfig(folds=10, val_fold=1, test_fold=0)
    result = add_split_column(frame, config, seed=42)
    assert_no_leakage(result)
    for _text_hash, group in result.groupby("text_hash"):
        assert group["split"].nunique() == 1


def test_split_is_deterministic_for_same_seed() -> None:
    frame = make_frame(n_hashes=50, per_hash=1)
    config = SplitConfig(folds=10, val_fold=1, test_fold=0)
    first = add_split_column(frame, config, seed=42)["split"].tolist()
    second = add_split_column(frame, config, seed=42)["split"].tolist()
    assert first == second


def test_split_rejects_equal_val_and_test_fold() -> None:
    frame = make_frame(n_hashes=20, per_hash=1)
    config = SplitConfig(folds=10, val_fold=0, test_fold=0)
    with pytest.raises(ValueError):
        add_split_column(frame, config, seed=42)


def test_split_produces_roughly_80_10_10() -> None:
    frame = make_frame(n_hashes=500, per_hash=1)
    config = SplitConfig(folds=10, val_fold=1, test_fold=0)
    result = apply_split(frame, config, seed=42)
    proportions = result["split"].value_counts(normalize=True)
    assert proportions[["train", "val", "test"]].round(1).tolist() == pytest.approx(
        [0.8, 0.1, 0.1], abs=0.05
    )


def test_split_summary_reports_all_combinations() -> None:
    frame = make_frame(n_hashes=80, per_hash=1)
    config = SplitConfig(folds=10, val_fold=1, test_fold=0)
    result = apply_split(frame, config, seed=42)
    summary = split_summary(result)
    assert set(summary.columns) <= {"train", "val", "test"}
    assert summary.to_numpy().sum() == len(frame)


def test_assert_no_leakage_detects_a_broken_split() -> None:
    frame = pd.DataFrame(
        {"text_hash": ["h1", "h1"], "split": ["train", "test"]}
    )
    with pytest.raises(ValueError):
        assert_no_leakage(frame)