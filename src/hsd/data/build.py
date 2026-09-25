import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from hsd.common.config import DataConfig, LabelMapConfig, load_data_config, load_label_map
from hsd.common.logging import get_logger
from hsd.common.paths import resolve
from hsd.data.dedup import annotate_duplicates
from hsd.data.harmonize_jigsaw import label_jigsaw, load_jigsaw
from hsd.data.harmonize_mhs import label_mhs, load_mhs
from hsd.data.io import write_parquet_atomic
from hsd.data.labeling import underrepresented_classes
from hsd.data.split import apply_split

logger = get_logger(__name__)

JIGSAW_SIGNALS = [
    "j_toxic",
    "j_severe_toxic",
    "j_obscene",
    "j_threat",
    "j_insult",
    "j_identity_hate",
]
MHS_SIGNALS = [
    "m_hate_speech_score",
    "m_insult",
    "m_humiliate",
    "m_dehumanize",
    "m_violence",
    "m_genocide",
    "m_attack_defend",
    "m_hatespeech",
]
MHS_COUNT_SIGNAL = "m_n_annotators"
TARGET_COLUMNS = [
    "t_race",
    "t_religion",
    "t_origin",
    "t_gender",
    "t_sexuality",
    "t_age",
    "t_disability",
    "t_politics",
]
BASE_COLUMNS = [
    "uid",
    "source",
    "origin",
    "source_id",
    "text_raw",
    "text",
    "text_hash",
    "is_short",
    "label",
    "label_name",
    "split",
    "is_duplicate",
    "dup_group_size",
    "label_conflict",
]
UNIFIED_COLUMNS = [
    *BASE_COLUMNS,
    *JIGSAW_SIGNALS,
    *MHS_SIGNALS,
    MHS_COUNT_SIGNAL,
    *TARGET_COLUMNS,
]
REQUIRED_COLUMNS = ["uid", "source", "text", "label", "label_name", "split"]
STRING_COLUMNS = [
    "uid",
    "source",
    "origin",
    "source_id",
    "text_raw",
    "text",
    "text_hash",
    "label_name",
    "split",
]
BOOL_COLUMNS = ["is_short", "is_duplicate", "label_conflict"]


def load_labeled_sources(
    config: DataConfig, label_map: LabelMapConfig, force: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jigsaw_labeled, _ = load_jigsaw(
        config.jigsaw,
        config.text,
        config.outputs.jigsaw,
        config.outputs.jigsaw_unlabeled,
        force,
    )
    mhs_raw = load_mhs(
        config.mhs,
        config.text,
        config.outputs.mhs_annotators,
        config.outputs.mhs,
        force,
    )
    jigsaw = label_jigsaw(jigsaw_labeled, label_map)
    mhs = label_mhs(mhs_raw, label_map)
    return jigsaw, mhs


def combine_sources(jigsaw: pd.DataFrame, mhs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([jigsaw, mhs], ignore_index=True, sort=False)


def enforce_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=UNIFIED_COLUMNS).copy()
    for column in STRING_COLUMNS:
        result[column] = result[column].astype("string")
    for column in JIGSAW_SIGNALS:
        result[column] = result[column].astype("Int8")
    for column in MHS_SIGNALS:
        result[column] = result[column].astype("float32")
    result[MHS_COUNT_SIGNAL] = result[MHS_COUNT_SIGNAL].astype("Int16")
    for column in TARGET_COLUMNS:
        result[column] = result[column].astype("boolean")
    for column in BOOL_COLUMNS:
        result[column] = result[column].astype("bool")
    result["dup_group_size"] = result["dup_group_size"].astype("int32")
    result["label"] = result["label"].astype("int8")
    return result


def validate_schema(frame: pd.DataFrame) -> None:
    if list(frame.columns) != UNIFIED_COLUMNS:
        raise ValueError("Unified frame columns do not match the expected schema")
    for column in REQUIRED_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"Required column {column!r} contains missing values")
    if not frame["uid"].is_unique:
        raise ValueError("uid column contains duplicate values")


def report_underrepresented(frame: pd.DataFrame, label_map: LabelMapConfig) -> None:
    weak = underrepresented_classes(frame, label_map.min_class_share)
    if weak:
        logger.warning(
            "Classes below min_class_share %.3f: %s", label_map.min_class_share, weak
        )


def build_unified(
    config: DataConfig, label_map: LabelMapConfig, force: bool
) -> pd.DataFrame:
    jigsaw, mhs = load_labeled_sources(config, label_map, force)
    combined = combine_sources(jigsaw, mhs)
    duplicated = annotate_duplicates(combined)
    split_frame = apply_split(duplicated, config.split, config.seed)
    schema_ready = enforce_schema(split_frame)
    validate_schema(schema_ready)
    report_underrepresented(schema_ready, label_map)
    logger.info("Unified dataset assembled with %d rows", len(schema_ready))
    return schema_ready


def sample_rows(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows >= len(frame):
        return frame
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unified hate speech dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def resolve_output_path(args: argparse.Namespace, config: DataConfig) -> Path:
    return resolve(args.output) if args.output else config.outputs.unified


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_data_config(args.config)
    label_map = load_label_map(args.label_map)
    try:
        frame = build_unified(config, label_map, args.force)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
        logger.error("%s", error)
        return 1
    if args.max_rows is not None:
        frame = sample_rows(frame, args.max_rows, config.seed)
    output_path = resolve_output_path(args, config)
    write_parquet_atomic(frame, output_path)
    logger.info("Wrote %d rows to %s", len(frame), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())