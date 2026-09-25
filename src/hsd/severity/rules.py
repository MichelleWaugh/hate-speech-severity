from dataclasses import dataclass

import numpy as np
import pandas as pd

from hsd.common.config import read_yaml, resolve
from hsd.common.labels import IGNORE_INDEX, SEVERITY_NAMES, SEVERITY_TO_ID

NONE_SEVERITY_NAME = "none"


@dataclass(frozen=True)
class JigsawSeverityConfig:
    weights: dict[str, int]
    bins: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class MhsThreshold:
    hate_speech_score_exclusive_min: float | None
    dehumanize_min: float | None
    insult_min: float | None
    genocide_min: float | None
    violence_min: float | None


@dataclass(frozen=True)
class SeverityConfig:
    jigsaw: JigsawSeverityConfig
    mhs_critical: MhsThreshold
    mhs_high: MhsThreshold
    mhs_medium: MhsThreshold
    min_level_share: float


def parse_threshold(raw: dict) -> MhsThreshold:
    return MhsThreshold(
        hate_speech_score_exclusive_min=raw.get("hate_speech_score_exclusive_min"),
        dehumanize_min=raw.get("dehumanize_min"),
        insult_min=raw.get("insult_min"),
        genocide_min=raw.get("genocide_min"),
        violence_min=raw.get("violence_min"),
    )


def load_severity_config(path: str | None = None) -> SeverityConfig:
    raw = read_yaml(resolve(path) if path else resolve("configs/severity.yaml"))
    jigsaw_raw = raw["jigsaw"]
    bins = {
        name: (int(bounds[0]), int(bounds[1])) for name, bounds in jigsaw_raw["bins"].items()
    }
    if set(bins) != set(SEVERITY_NAMES):
        raise ValueError(f"jigsaw.bins must define exactly {SEVERITY_NAMES}, got {list(bins)}")
    return SeverityConfig(
        jigsaw=JigsawSeverityConfig(
            weights={name: int(value) for name, value in jigsaw_raw["weights"].items()},
            bins=bins,
        ),
        mhs_critical=parse_threshold(raw["mhs"]["critical"]),
        mhs_high=parse_threshold(raw["mhs"]["high"]),
        mhs_medium=parse_threshold(raw["mhs"]["medium"]),
        min_level_share=float(raw["min_level_share"]),
    )


def jigsaw_severity_score(frame: pd.DataFrame, config: JigsawSeverityConfig) -> pd.Series:
    missing = [f"j_{name}" for name in config.weights if f"j_{name}" not in frame.columns]
    if missing:
        raise KeyError(f"Frame is missing Jigsaw signal columns {missing}")
    score = pd.Series(0, index=frame.index, dtype="int32")
    for name, weight in config.weights.items():
        column = frame[f"j_{name}"].fillna(0).astype("int32")
        score = score + column.clip(0, 1) * weight
    return score


def bin_jigsaw_score(score: pd.Series, config: JigsawSeverityConfig) -> pd.Series:
    result = pd.Series(NONE_SEVERITY_NAME, index=score.index, dtype="object")
    for name, (low, high) in config.bins.items():
        in_range = score.between(low, high)
        result = result.mask(in_range, name)
    unresolved = (result == NONE_SEVERITY_NAME) & score.gt(0)
    if unresolved.any():
        raise ValueError(f"{int(unresolved.sum())} Jigsaw scores fell outside every bin")
    return result


def mhs_required_columns() -> list[str]:
    return [
        "m_hate_speech_score",
        "m_insult",
        "m_dehumanize",
        "m_genocide",
        "m_violence",
    ]


def threshold_mask(frame: pd.DataFrame, threshold: MhsThreshold) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    if threshold.genocide_min is not None:
        mask = mask | frame["m_genocide"].ge(threshold.genocide_min).fillna(False)
    if threshold.violence_min is not None:
        mask = mask | frame["m_violence"].ge(threshold.violence_min).fillna(False)
    if threshold.hate_speech_score_exclusive_min is not None:
        mask = mask | frame["m_hate_speech_score"].gt(
            threshold.hate_speech_score_exclusive_min
        ).fillna(False)
    if threshold.dehumanize_min is not None:
        mask = mask | frame["m_dehumanize"].ge(threshold.dehumanize_min).fillna(False)
    if threshold.insult_min is not None:
        mask = mask | frame["m_insult"].ge(threshold.insult_min).fillna(False)
    return mask


def classify_mhs_severity(frame: pd.DataFrame, config: SeverityConfig) -> pd.Series:
    missing = [column for column in mhs_required_columns() if column not in frame.columns]
    if missing:
        raise KeyError(f"Frame is missing MHS signal columns {missing}")
    result = pd.Series("low", index=frame.index, dtype="object")
    result = result.mask(threshold_mask(frame, config.mhs_medium), "medium")
    result = result.mask(threshold_mask(frame, config.mhs_high), "high")
    result = result.mask(threshold_mask(frame, config.mhs_critical), "critical")
    return result


def mhs_severity_score(frame: pd.DataFrame) -> pd.Series:
    missing = [column for column in mhs_required_columns() if column not in frame.columns]
    if missing:
        raise KeyError(f"Frame is missing MHS signal columns {missing}")
    return (
        frame["m_hate_speech_score"].fillna(0)
        + frame["m_violence"].fillna(0)
        + frame["m_dehumanize"].fillna(0)
        + frame["m_genocide"].fillna(0)
    ).astype("float32")


def severity_name_to_id(names: pd.Series) -> pd.Series:
    ids = names.map({**SEVERITY_TO_ID, NONE_SEVERITY_NAME: IGNORE_INDEX})
    if ids.isna().any():
        unknown = sorted(set(names[ids.isna()]))
        raise ValueError(f"Unknown severity names {unknown}")
    return ids.astype("int16")


def harmful_mask(labels: pd.Series) -> pd.Series:
    return labels.astype("string") != "neither"


def apply_severity(
    names: pd.Series, scores: pd.Series, is_harmful: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    final_names = names.where(is_harmful, NONE_SEVERITY_NAME)
    final_scores = scores.where(is_harmful, np.float32(0.0)).astype("float32")
    final_ids = severity_name_to_id(final_names)
    return final_ids, final_scores, final_names