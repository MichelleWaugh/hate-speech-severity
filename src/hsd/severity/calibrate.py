import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from hsd.common.config import load_data_config
from hsd.common.logging import get_logger
from hsd.common.paths import resolve
from hsd.severity.derive import derive_severity_by_source
from hsd.severity.rules import load_severity_config

plt.switch_backend("Agg")

logger = get_logger(__name__)

SCORE_QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99]


def score_quantiles(frame: pd.DataFrame, column: str) -> dict:
    values = frame[column].dropna()
    return {str(int(q * 100)): float(values.quantile(q)) for q in SCORE_QUANTILES}


def cross_tab_counts(frame: pd.DataFrame, row: str, column: str) -> dict:
    table = pd.crosstab(frame[row], frame[column])
    return {
        str(index): {str(col): int(value) for col, value in cols.items()}
        for index, cols in table.to_dict(orient="index").items()
    }


def harmful_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["label_name"] != "neither"]


def build_calibration_report(frame: pd.DataFrame) -> dict:
    derived_harmful = harmful_rows(frame)
    jigsaw_scores = derived_harmful.loc[derived_harmful["source"] == "jigsaw", "severity_score"]
    mhs_scores = derived_harmful.loc[derived_harmful["source"] == "mhs", "severity_score"]
    return {
        "jigsaw_score_quantiles": score_quantiles(
            derived_harmful.loc[derived_harmful["source"] == "jigsaw"], "severity_score"
        )
        if len(jigsaw_scores)
        else {},
        "mhs_score_quantiles": score_quantiles(
            derived_harmful.loc[derived_harmful["source"] == "mhs"], "severity_score"
        )
        if len(mhs_scores)
        else {},
        "severity_by_source": cross_tab_counts(derived_harmful, "source", "severity_name"),
        "severity_by_class": cross_tab_counts(derived_harmful, "label_name", "severity_name"),
        "severity_share_overall": {
            str(level): float(share)
            for level, share in derived_harmful["severity_name"]
            .value_counts(normalize=True)
            .items()
        },
    }


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_rubric_copy(severity_config_path: Path, path: Path) -> None:
    with severity_config_path.open("r", encoding="utf-8") as handle:
        rubric = yaml.safe_load(handle)
    write_json(rubric, path)


def plot_severity_by_source(frame: pd.DataFrame, path: Path) -> None:
    table = pd.crosstab(harmful_rows(frame)["source"], harmful_rows(frame)["severity_name"])
    axis = table.plot(kind="bar", figsize=(8, 5))
    axis.set_xlabel("source")
    axis.set_ylabel("harmful rows")
    axis.set_title("Derived severity by source")
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the severity rubric thresholds")
    parser.add_argument("--config", default=None)
    parser.add_argument("--severity-config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_config = load_data_config(args.config)
    input_path = resolve(args.input) if args.input else data_config.outputs.unified
    if not input_path.exists():
        logger.error("%s does not exist, run hsd.data.build first", input_path)
        return 1
    severity_config_path = resolve(args.severity_config) if args.severity_config else resolve(
        "configs/severity.yaml"
    )
    frame = pd.read_parquet(input_path)
    config = load_severity_config(str(severity_config_path))
    derived = derive_severity_by_source(frame, config)
    report = build_calibration_report(derived)
    output_dir = resolve(args.output_dir) if args.output_dir else resolve("reports/severity")
    write_json(report, output_dir / "calibration.json")
    write_rubric_copy(severity_config_path, output_dir / "rubric.json")
    plot_severity_by_source(derived, output_dir / "severity_by_source.png")
    logger.info("Wrote calibration report and figure to %s", output_dir)
    logger.info("Overall severity shares: %s", report["severity_share_overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())