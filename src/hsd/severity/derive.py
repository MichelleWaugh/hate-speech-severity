import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from hsd.common.config import load_data_config
from hsd.common.labels import IGNORE_INDEX, SEVERITY_NAMES
from hsd.common.logging import get_logger
from hsd.common.paths import resolve
from hsd.data.io import write_parquet_atomic
from hsd.severity.rules import (
    NONE_SEVERITY_NAME,
    SeverityConfig,
    apply_severity,
    bin_jigsaw_score,
    classify_mhs_severity,
    harmful_mask,
    jigsaw_severity_score,
    load_severity_config,
    mhs_severity_score,
)

logger = get_logger(__name__)


def derive_jigsaw_severity(
    frame: pd.DataFrame, config: SeverityConfig
) -> tuple[pd.Series, pd.Series]:
    scores = jigsaw_severity_score(frame, config.jigsaw).astype("float32")
    names = bin_jigsaw_score(scores, config.jigsaw)
    return names, scores


def derive_mhs_severity(
    frame: pd.DataFrame, config: SeverityConfig
) -> tuple[pd.Series, pd.Series]:
    names = classify_mhs_severity(frame, config)
    scores = mhs_severity_score(frame)
    return names, scores


def derive_severity_by_source(frame: pd.DataFrame, config: SeverityConfig) -> pd.DataFrame:
    names = pd.Series(NONE_SEVERITY_NAME, index=frame.index, dtype="object")
    scores = pd.Series(0.0, index=frame.index, dtype="float32")
    jigsaw_mask = frame["source"] == "jigsaw"
    mhs_mask = frame["source"] == "mhs"
    if jigsaw_mask.any():
        jigsaw_names, jigsaw_scores = derive_jigsaw_severity(frame.loc[jigsaw_mask], config)
        names.loc[jigsaw_mask] = jigsaw_names
        scores.loc[jigsaw_mask] = jigsaw_scores
    if mhs_mask.any():
        mhs_names, mhs_scores = derive_mhs_severity(frame.loc[mhs_mask], config)
        names.loc[mhs_mask] = mhs_names
        scores.loc[mhs_mask] = mhs_scores
    unknown_sources = ~(jigsaw_mask | mhs_mask)
    if unknown_sources.any():
        raise ValueError(f"{int(unknown_sources.sum())} rows have an unrecognized source value")
    is_harmful = harmful_mask(frame["label_name"])
    severity_ids, severity_scores, severity_names = apply_severity(names, scores, is_harmful)
    return frame.assign(
        severity=severity_ids,
        severity_score=severity_scores,
        severity_name=severity_names,
    )


def validate_derived(frame: pd.DataFrame, config: SeverityConfig) -> None:
    neither_rows = frame.loc[frame["label_name"] == "neither"]
    if not (neither_rows["severity"] == IGNORE_INDEX).all():
        raise ValueError("Some neither rows have a severity other than the ignore index")
    if not (neither_rows["severity_name"] == NONE_SEVERITY_NAME).all():
        raise ValueError("Some neither rows have a severity_name other than none")
    harmful_rows = frame.loc[frame["label_name"] != "neither"]
    if harmful_rows.empty:
        return
    shares = harmful_rows["severity_name"].value_counts(normalize=True)
    under_represented = [
        level for level in SEVERITY_NAMES if shares.get(level, 0.0) < config.min_level_share
    ]
    if under_represented:
        raise ValueError(
            f"Severity levels below min_level_share {config.min_level_share:.3f}: "
            f"{under_represented}"
        )


def build_dataset(
    unified_path: Path, severity_config_path: str | None
) -> tuple[pd.DataFrame, SeverityConfig]:
    if not unified_path.exists():
        raise FileNotFoundError(f"{unified_path} does not exist, run hsd.data.build first")
    frame = pd.read_parquet(unified_path)
    config = load_severity_config(severity_config_path)
    derived = derive_severity_by_source(frame, config)
    return derived, config


def log_distribution(frame: pd.DataFrame) -> None:
    by_source = pd.crosstab(frame["source"], frame["severity_name"])
    by_class = pd.crosstab(frame["label_name"], frame["severity_name"])
    logger.info("Severity by source:\n%s", by_source)
    logger.info("Severity by class:\n%s", by_class)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive severity labels for the unified dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--severity-config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_config = load_data_config(args.config)
    input_path = resolve(args.input) if args.input else data_config.outputs.unified
    default_output = input_path.with_name("dataset.parquet")
    output_path = resolve(args.output) if args.output else default_output
    return input_path, output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path, output_path = resolve_paths(args)
    try:
        frame, config = build_dataset(input_path, args.severity_config)
        validate_derived(frame, config)
    except (FileNotFoundError, ValueError, KeyError) as error:
        logger.error("%s", error)
        return 1
    log_distribution(frame)
    if args.check:
        logger.info("Validation passed for %s", output_path)
        return 0
    write_parquet_atomic(frame, output_path)
    logger.info("Wrote %d rows to %s", len(frame), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())