import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hsd.common.config import load_data_config
from hsd.common.logging import get_logger
from hsd.common.paths import resolve
from hsd.data.build import TARGET_COLUMNS
from hsd.data.labeling import class_distribution

plt.switch_backend("Agg")

logger = get_logger(__name__)

LENGTH_QUANTILES = [0.5, 0.9, 0.95, 0.99]


def to_int_dict(table: pd.DataFrame) -> dict:
    return {
        str(index): {str(column): int(value) for column, value in row.items()}
        for index, row in table.to_dict(orient="index").items()
    }


def text_length_percentiles(frame: pd.DataFrame) -> dict:
    lengths = frame["text"].fillna("").str.len()
    return {str(int(q * 100)): float(lengths.quantile(q)) for q in LENGTH_QUANTILES}


def duplicate_stats(frame: pd.DataFrame) -> dict:
    duplicated = frame.loc[frame["is_duplicate"]]
    conflicting = frame.loc[frame["label_conflict"]]
    return {
        "duplicate_rows": int(frame["is_duplicate"].sum()),
        "duplicate_groups": int(duplicated["text_hash"].nunique()),
        "conflicting_groups": int(conflicting["text_hash"].nunique()),
    }


def target_group_counts(frame: pd.DataFrame) -> dict:
    return {column: int(frame[column].fillna(False).sum()) for column in TARGET_COLUMNS}


def compute_stats(frame: pd.DataFrame) -> dict:
    return {
        "total_rows": int(len(frame)),
        "class_counts_by_source": to_int_dict(class_distribution(frame)),
        "split_counts_by_source": to_int_dict(pd.crosstab(frame["source"], frame["split"])),
        "text_length_percentiles": text_length_percentiles(frame),
        **duplicate_stats(frame),
        "target_group_counts": target_group_counts(frame),
    }


def write_stats_json(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)


def plot_class_distribution(frame: pd.DataFrame, path: Path) -> None:
    table = class_distribution(frame)
    axis = table.plot(kind="bar", figsize=(8, 5))
    axis.set_xlabel("source")
    axis.set_ylabel("rows")
    axis.set_title("Class distribution by source")
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_split_distribution(frame: pd.DataFrame, path: Path) -> None:
    table = pd.crosstab(frame["source"], frame["split"])
    axis = table.plot(kind="bar", figsize=(8, 5))
    axis.set_xlabel("source")
    axis.set_ylabel("rows")
    axis.set_title("Split distribution by source")
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_target_groups(frame: pd.DataFrame, path: Path) -> None:
    counts = pd.Series(target_group_counts(frame)).sort_values(ascending=False)
    axis = counts.plot(kind="bar", figsize=(8, 5), color="steelblue")
    axis.set_xlabel("target group")
    axis.set_ylabel("rows")
    axis.set_title("MHS target group counts")
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute unified dataset statistics")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_data_config(args.config)
    input_path = resolve(args.input) if args.input else config.outputs.unified
    if not input_path.exists():
        logger.error("%s does not exist, run hsd.data.build first", input_path)
        return 1
    frame = pd.read_parquet(input_path)
    output_dir = resolve(args.output_dir) if args.output_dir else config.outputs.reports
    stats = compute_stats(frame)
    write_stats_json(stats, output_dir / "stats.json")
    plot_class_distribution(frame, output_dir / "class_distribution.png")
    plot_split_distribution(frame, output_dir / "source_split_distribution.png")
    plot_target_groups(frame, output_dir / "target_groups.png")
    logger.info("Wrote statistics and figures to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())