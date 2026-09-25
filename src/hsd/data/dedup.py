import pandas as pd

from hsd.common.labels import CLASS_TO_ID
from hsd.common.logging import get_logger

logger = get_logger(__name__)


def add_duplicate_flags(frame: pd.DataFrame) -> pd.DataFrame:
    group_sizes = frame.groupby("text_hash")["uid"].transform("size")
    return frame.assign(
        is_duplicate=group_sizes.gt(1),
        dup_group_size=group_sizes.astype("int32"),
    )


def add_conflict_flags(frame: pd.DataFrame) -> pd.DataFrame:
    label_counts = frame.groupby("text_hash")["label"].transform("nunique")
    return frame.assign(label_conflict=frame["is_duplicate"] & label_counts.gt(1))


def annotate_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    with_duplicates = add_duplicate_flags(frame)
    annotated = add_conflict_flags(with_duplicates)
    duplicate_rows = int(annotated["is_duplicate"].sum())
    duplicate_groups = int(annotated.loc[annotated["is_duplicate"], "text_hash"].nunique())
    conflicting_groups = int(annotated.loc[annotated["label_conflict"], "text_hash"].nunique())
    logger.info(
        "Duplicates: %d rows across %d groups, %d groups with conflicting labels",
        duplicate_rows,
        duplicate_groups,
        conflicting_groups,
    )
    return annotated


def dominant_label(frame: pd.DataFrame) -> pd.Series:
    ranks = {label: rank for rank, label in enumerate(sorted(CLASS_TO_ID.values(), reverse=True))}
    return frame["label"].map(ranks)


def duplicate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    duplicates = frame.loc[frame["is_duplicate"]]
    if duplicates.empty:
        empty_columns = ["text_hash", "dup_group_size", "sources", "labels", "label_conflict"]
        return pd.DataFrame(columns=empty_columns)
    grouped = duplicates.groupby("text_hash")
    return pd.DataFrame(
        {
            "dup_group_size": grouped["uid"].size(),
            "sources": grouped["source"].agg(lambda values: sorted(set(values))),
            "labels": grouped["label_name"].agg(lambda values: sorted(set(values))),
            "label_conflict": grouped["label_conflict"].any(),
        }
    ).reset_index()