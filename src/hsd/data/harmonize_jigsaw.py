from pathlib import Path

import pandas as pd

from hsd.common.config import JigsawConfig, LabelMapConfig, TextConfig
from hsd.common.logging import get_logger
from hsd.data.download import JigsawFiles, ensure_jigsaw
from hsd.data.io import is_cached, write_parquet_atomic
from hsd.data.labeling import add_label_columns
from hsd.data.text import add_text_columns, uid_series

logger = get_logger(__name__)

SOURCE = "jigsaw"
SIGNAL_PREFIX = "j_"
ORIGIN_TRAIN = "jigsaw_train"
ORIGIN_TEST = "jigsaw_test"
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


def signal_name(label: str) -> str:
    return f"{SIGNAL_PREFIX}{label}"


def signal_columns(config: JigsawConfig) -> list[str]:
    return [signal_name(label) for label in config.label_columns]


def labeled_columns(config: JigsawConfig) -> list[str]:
    return [*BASE_COLUMNS, *signal_columns(config)]


def read_table(path: Path, columns: list[str], config: JigsawConfig) -> pd.DataFrame:
    dtypes = {
        column: "string" for column in columns if column in (config.id_column, config.text_column)
    }
    dtypes.update({column: "int8" for column in columns if column in config.label_columns})
    return pd.read_csv(
        path, usecols=columns, dtype=dtypes, keep_default_na=False, encoding="utf-8"
    )


def read_train(files: JigsawFiles, config: JigsawConfig) -> pd.DataFrame:
    columns = [config.id_column, config.text_column, *config.label_columns]
    return read_table(files.train, columns, config)


def read_test(files: JigsawFiles, config: JigsawConfig) -> pd.DataFrame:
    texts = read_table(files.test, [config.id_column, config.text_column], config)
    labels = read_table(files.test_labels, [config.id_column, *config.label_columns], config)
    merged = texts.merge(labels, on=config.id_column, how="left", validate="one_to_one")
    if merged[config.label_columns].isna().any().any():
        raise ValueError("test.csv contains ids that are missing from test_labels.csv")
    return merged.astype({label: "int8" for label in config.label_columns})


def split_labeled(
    frame: pd.DataFrame, config: JigsawConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unlabeled = frame[config.label_columns].lt(0).any(axis=1)
    return frame.loc[~unlabeled].copy(), frame.loc[unlabeled].copy()


def validate_binary(frame: pd.DataFrame, config: JigsawConfig) -> None:
    invalid = ~frame[config.label_columns].isin([0, 1]).all(axis=1)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} Jigsaw rows have label values outside 0 and 1")


def to_signal_frame(
    frame: pd.DataFrame, config: JigsawConfig, text_config: TextConfig, origin: str
) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            config.id_column: "source_id",
            config.text_column: "text_raw",
            **{label: signal_name(label) for label in config.label_columns},
        }
    )
    renamed["uid"] = uid_series(SOURCE, renamed["source_id"])
    renamed["source"] = SOURCE
    renamed["origin"] = origin
    with_text = add_text_columns(renamed, text_config)
    signals = signal_columns(config)
    return with_text[labeled_columns(config)].astype({column: "Int8" for column in signals})


def assemble(
    parts: list[tuple[pd.DataFrame, str]], config: JigsawConfig, text_config: TextConfig
) -> pd.DataFrame:
    frames = [
        to_signal_frame(part, config, text_config, origin) for part, origin in parts if len(part)
    ]
    if not frames:
        return pd.DataFrame(columns=labeled_columns(config))
    return pd.concat(frames, ignore_index=True)


def build_frames(
    files: JigsawFiles, config: JigsawConfig, text_config: TextConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_labeled, train_unlabeled = split_labeled(read_train(files, config), config)
    test_labeled, test_unlabeled = split_labeled(read_test(files, config), config)
    validate_binary(train_labeled, config)
    validate_binary(test_labeled, config)
    labeled = assemble(
        [(train_labeled, ORIGIN_TRAIN), (test_labeled, ORIGIN_TEST)], config, text_config
    )
    unlabeled = assemble(
        [(train_unlabeled, ORIGIN_TRAIN), (test_unlabeled, ORIGIN_TEST)], config, text_config
    )[BASE_COLUMNS]
    logger.info(
        "Jigsaw: %d labeled rows (%d train, %d test), %d unlabeled test rows",
        len(labeled),
        len(train_labeled),
        len(test_labeled),
        len(unlabeled),
    )
    return labeled, unlabeled


def load_jigsaw(
    config: JigsawConfig,
    text_config: TextConfig,
    labeled_path: Path,
    unlabeled_path: Path,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if is_cached(labeled_path, force) and is_cached(unlabeled_path, force):
        logger.info("Reusing cached Jigsaw tables")
    else:
        files = ensure_jigsaw(config)
        labeled, unlabeled = build_frames(files, config, text_config)
        write_parquet_atomic(labeled, labeled_path)
        write_parquet_atomic(unlabeled, unlabeled_path)
    return pd.read_parquet(labeled_path), pd.read_parquet(unlabeled_path)


def any_flag(frame: pd.DataFrame, labels: list[str]) -> pd.Series:
    names = [signal_name(label) for label in labels]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"Jigsaw frame is missing signal columns {missing}")
    return frame[names].eq(1).fillna(False).any(axis=1).astype(bool)


def label_jigsaw(frame: pd.DataFrame, label_map: LabelMapConfig) -> pd.DataFrame:
    masks = {name: any_flag(frame, labels) for name, labels in label_map.jigsaw.items()}
    return add_label_columns(frame, masks, label_map.priority)