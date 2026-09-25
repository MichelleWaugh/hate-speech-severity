from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from hsd.common.labels import CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS

FALLBACK_CLASS = "neither"


def validate_masks(masks: Mapping[str, pd.Series], length: int) -> None:
    unknown = [name for name in masks if name not in CLASS_TO_ID]
    if unknown:
        raise ValueError(f"Unknown class names in label rules: {unknown}")
    if FALLBACK_CLASS in masks:
        raise ValueError(f"{FALLBACK_CLASS!r} is the fallback class and cannot have a rule")
    wrong_length = [name for name, mask in masks.items() if len(mask) != length]
    if wrong_length:
        raise ValueError(f"Masks for {wrong_length} do not match the frame length {length}")


def assign_labels(
    masks: Mapping[str, pd.Series], priority: Sequence[str], index: pd.Index
) -> pd.Series:
    validate_masks(masks, len(index))
    values = np.full(len(index), CLASS_TO_ID[FALLBACK_CLASS], dtype="int8")
    for name in reversed(list(priority)):
        mask = masks.get(name)
        if mask is not None:
            values[mask.to_numpy(dtype=bool)] = CLASS_TO_ID[name]
    return pd.Series(values, index=index, dtype="int8")


def add_label_columns(
    frame: pd.DataFrame, masks: Mapping[str, pd.Series], priority: Sequence[str]
) -> pd.DataFrame:
    labels = assign_labels(masks, priority, frame.index)
    return frame.assign(label=labels, label_name=labels.map(ID_TO_CLASS))


def class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(frame["source"], frame["label_name"])
    return counts.reindex(columns=CLASS_NAMES, fill_value=0)


def class_shares(frame: pd.DataFrame) -> pd.Series:
    counts = frame["label_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    return counts / max(len(frame), 1)


def underrepresented_classes(frame: pd.DataFrame, min_share: float) -> list[str]:
    shares = class_shares(frame)
    return [name for name in CLASS_NAMES if shares[name] < min_share]