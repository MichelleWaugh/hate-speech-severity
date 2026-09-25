from pathlib import Path

import pandas as pd

from hsd.common.config import LabelMapConfig, MhsConfig, MhsRule, TextConfig
from hsd.common.logging import get_logger
from hsd.data.download import ensure_mhs, mhs_required_columns
from hsd.data.io import is_cached, write_parquet_atomic
from hsd.data.labeling import add_label_columns
from hsd.data.text import add_text_columns, uid_series

logger = get_logger(__name__)

SOURCE = "mhs"
ORIGIN = "mhs"
COUNT_SIGNAL = "m_n_annotators"
BASE_COLUMNS = [
    "uid",
    "source",
    "origin",
    "source_id",
    "text_raw",
    "text",
    "text_hash",
    "is_short",
]


def score_signal(config: MhsConfig) -> str:
    return f"m_{config.score_column}"


def rating_signals(config: MhsConfig) -> list[str]:
    return [f"m_{name}" for name in config.rating_columns]


def mean_signal_columns(config: MhsConfig) -> list[str]:
    return [score_signal(config), *rating_signals(config)]


def target_signal_name(name: str) -> str:
    return f"t_{name.removeprefix('target_')}"


def target_signal_columns(config: MhsConfig) -> list[str]:
    return [target_signal_name(name) for name in config.target_columns]


def frame_columns(config: MhsConfig) -> list[str]:
    return [
        *BASE_COLUMNS,
        *mean_signal_columns(config),
        COUNT_SIGNAL,
        *target_signal_columns(config),
    ]


def read_annotator_rows(config: MhsConfig, annotators_path: Path) -> pd.DataFrame:
    columns = mhs_required_columns(config)
    if annotators_path.exists():
        return pd.read_parquet(annotators_path, columns=columns)
    return ensure_mhs(config, annotators_path, False).to_pandas()


def aggregate_annotators(rows: pd.DataFrame, config: MhsConfig) -> pd.DataFrame:
    grouped = rows.groupby(config.id_column, sort=True)
    conflicting = int((grouped[config.text_column].nunique() > 1).sum())
    if conflicting:
        logger.warning("%d comments carry more than one text; keeping the first", conflicting)
    numeric = [config.score_column, *config.rating_columns]
    means = grouped[numeric].mean().astype("float32")
    means.columns = mean_signal_columns(config)
    targets = grouped[config.target_columns].max().astype("boolean")
    targets.columns = target_signal_columns(config)
    aggregated = pd.concat(
        [
            grouped[config.text_column].first().rename("text_raw"),
            means,
            grouped.size().astype("Int16").rename(COUNT_SIGNAL),
            targets,
        ],
        axis=1,
    )
    aggregated.index.name = "source_id"
    aggregated = aggregated.reset_index()
    aggregated["source_id"] = aggregated["source_id"].astype(str)
    return aggregated


def to_signal_frame(
    aggregated: pd.DataFrame, config: MhsConfig, text_config: TextConfig
) -> pd.DataFrame:
    frame = aggregated.assign(source=SOURCE, origin=ORIGIN)
    frame["uid"] = uid_series(SOURCE, frame["source_id"])
    return add_text_columns(frame, text_config)[frame_columns(config)]


def load_mhs(
    config: MhsConfig,
    text_config: TextConfig,
    annotators_path: Path,
    cache_path: Path,
    force: bool,
) -> pd.DataFrame:
    if is_cached(cache_path, force):
        logger.info("Reusing cached MHS table")
        return pd.read_parquet(cache_path)
    rows = read_annotator_rows(config, annotators_path)
    aggregated = aggregate_annotators(rows, config)
    comments = int(rows[config.id_column].nunique())
    if len(aggregated) != comments:
        raise ValueError(f"Aggregation produced {len(aggregated)} rows for {comments} comments")
    write_parquet_atomic(to_signal_frame(aggregated, config, text_config), cache_path)
    logger.info("MHS: aggregated %d annotator rows into %d comments", len(rows), comments)
    return pd.read_parquet(cache_path)


def rule_mask(frame: pd.DataFrame, rule: MhsRule) -> pd.Series:
    if rule.column not in frame.columns:
        raise KeyError(f"MHS frame is missing signal column {rule.column!r}")
    values = frame[rule.column]
    mask = values.ge(rule.minimum) if rule.inclusive else values.gt(rule.minimum)
    return mask.fillna(False).astype(bool)


def label_mhs(frame: pd.DataFrame, label_map: LabelMapConfig) -> pd.DataFrame:
    masks = {name: rule_mask(frame, rule) for name, rule in label_map.mhs.items()}
    return add_label_columns(frame, masks, label_map.priority)