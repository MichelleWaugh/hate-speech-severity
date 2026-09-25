import argparse
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import Dataset, get_dataset_config_names, load_dataset

from hsd.common.config import DataConfig, JigsawConfig, MhsConfig, load_data_config
from hsd.common.logging import get_logger

logger = get_logger(__name__)

JIGSAW_TRAIN = "train.csv"
JIGSAW_TEST = "test.csv"
JIGSAW_TEST_LABELS = "test_labels.csv"
JIGSAW_FILES = (JIGSAW_TRAIN, JIGSAW_TEST, JIGSAW_TEST_LABELS)
MAX_ARCHIVE_DEPTH = 3
SOURCES = ("jigsaw", "mhs")


@dataclass(frozen=True)
class JigsawFiles:
    train: Path
    test: Path
    test_labels: Path


def find_file(directory: Path, name: str) -> Path | None:
    matches = sorted(directory.rglob(name))
    return matches[0] if matches else None


def require_file(directory: Path, name: str) -> Path:
    path = find_file(directory, name)
    if path is None:
        raise FileNotFoundError(f"{name} not found under {directory}")
    return path


def missing_jigsaw_files(directory: Path) -> list[str]:
    return [name for name in JIGSAW_FILES if find_file(directory, name) is None]


def extract_archive(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError(
                    f"Archive {archive.name} contains an unsafe path: {member.filename}"
                )
        handle.extractall(root)


def unpack_archives(directory: Path) -> None:
    extracted: set[Path] = set()
    for _ in range(MAX_ARCHIVE_DEPTH):
        pending = [path for path in sorted(directory.rglob("*.zip")) if path not in extracted]
        if not pending:
            return
        for archive in pending:
            logger.info("Extracting %s", archive.name)
            extract_archive(archive, archive.parent)
            extracted.add(archive)


def ensure_jigsaw(config: JigsawConfig) -> JigsawFiles:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    if missing_jigsaw_files(config.raw_dir):
        unpack_archives(config.raw_dir)
    missing = missing_jigsaw_files(config.raw_dir)
    if missing:
        raise FileNotFoundError(
            f"Missing Jigsaw files {missing} in {config.raw_dir}. Accept the competition rules at "
            f"https://www.kaggle.com/c/{config.competition}/data, download the archive and place "
            "the zip in that folder"
        )
    return JigsawFiles(
        train=require_file(config.raw_dir, JIGSAW_TRAIN),
        test=require_file(config.raw_dir, JIGSAW_TEST),
        test_labels=require_file(config.raw_dir, JIGSAW_TEST_LABELS),
    )


def count_rows(path: Path, id_column: str) -> int:
    return len(pd.read_csv(path, usecols=[id_column]))


def validate_jigsaw_files(files: JigsawFiles, config: JigsawConfig) -> None:
    expected = (
        (files.train, [config.id_column, config.text_column, *config.label_columns]),
        (files.test, [config.id_column, config.text_column]),
        (files.test_labels, [config.id_column, *config.label_columns]),
    )
    for path, required in expected:
        header = set(pd.read_csv(path, nrows=0).columns)
        absent = [column for column in required if column not in header]
        if absent:
            raise ValueError(f"{path.name} is missing columns {absent}")
        logger.info("%s: %d rows", path.name, count_rows(path, config.id_column))


def mhs_required_columns(config: MhsConfig) -> list[str]:
    return [
        config.id_column,
        config.text_column,
        config.score_column,
        *config.rating_columns,
        *config.target_columns,
    ]


def iter_config_names(config: MhsConfig) -> Iterator[str]:
    yield config.config_name
    try:
        discovered = get_dataset_config_names(config.dataset_id)
    except Exception as error:
        logger.warning("Could not list configs for %s: %s", config.dataset_id, error)
        return
    for name in discovered:
        if name != config.config_name:
            yield name


def load_mhs_split(config: MhsConfig, name: str, force: bool) -> Dataset:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    return load_dataset(
        config.dataset_id,
        name,
        split="train",
        cache_dir=str(config.cache_dir),
        download_mode="force_redownload" if force else None,
    )


def fetch_mhs_dataset(config: MhsConfig, force: bool) -> Dataset:
    required = mhs_required_columns(config)
    problems: list[str] = []
    last_error: Exception | None = None
    for name in iter_config_names(config):
        try:
            dataset = load_mhs_split(config, name, force)
        except Exception as error:
            last_error = error
            problems.append(f"config {name!r} failed to load: {error}")
            logger.warning("%s", problems[-1])
            continue
        absent = [column for column in required if column not in dataset.column_names]
        if absent:
            problems.append(f"config {name!r} lacks columns {absent}")
            logger.warning("%s", problems[-1])
            continue
        logger.info("Loaded MHS config %r with %d rows", name, dataset.num_rows)
        return dataset
    raise RuntimeError(
        f"Could not load {config.dataset_id}: " + "; ".join(problems)
    ) from last_error


def export_annotator_table(dataset: Dataset, path: Path, force: bool) -> None:
    if path.exists() and not force:
        logger.info("Annotator-level table already exists at %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    dataset.to_parquet(str(partial))
    partial.replace(path)
    logger.info("Saved annotator-level table with %d rows to %s", dataset.num_rows, path)


def ensure_mhs(config: MhsConfig, annotators_path: Path, force: bool) -> Dataset:
    full = fetch_mhs_dataset(config, force)
    export_annotator_table(full, annotators_path, force)
    return full.select_columns(mhs_required_columns(config))


def download_sources(config: DataConfig, sources: Sequence[str], force: bool) -> None:
    if "jigsaw" in sources:
        files = ensure_jigsaw(config.jigsaw)
        validate_jigsaw_files(files, config.jigsaw)
    if "mhs" in sources:
        dataset = ensure_mhs(config.mhs, config.outputs.mhs_annotators, force)
        comments = len(dataset.unique(config.mhs.id_column))
        logger.info(
            "MHS: %d annotator-level rows across %d unique comments", dataset.num_rows, comments
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire the Jigsaw and MHS datasets")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_data_config(args.config)
    try:
        download_sources(config, args.sources, args.force)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())