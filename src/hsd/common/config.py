from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hsd.common.labels import CLASS_NAMES
from hsd.common.paths import CONFIGS, resolve


@dataclass(frozen=True)
class JigsawConfig:
    competition: str
    raw_dir: Path
    id_column: str
    text_column: str
    label_columns: list[str]


@dataclass(frozen=True)
class MhsConfig:
    dataset_id: str
    config_name: str
    cache_dir: Path
    id_column: str
    text_column: str
    score_column: str
    rating_columns: list[str]
    target_columns: list[str]


@dataclass(frozen=True)
class TextConfig:
    min_chars: int
    max_repeat: int


@dataclass(frozen=True)
class SplitConfig:
    folds: int
    val_fold: int
    test_fold: int


@dataclass(frozen=True)
class OutputPaths:
    jigsaw: Path
    jigsaw_unlabeled: Path
    mhs: Path
    mhs_annotators: Path
    unified: Path
    reports: Path


@dataclass(frozen=True)
class DataConfig:
    seed: int
    jigsaw: JigsawConfig
    mhs: MhsConfig
    text: TextConfig
    split: SplitConfig
    outputs: OutputPaths


@dataclass(frozen=True)
class MhsRule:
    column: str
    minimum: float
    inclusive: bool


@dataclass(frozen=True)
class LabelMapConfig:
    priority: list[str]
    jigsaw: dict[str, list[str]]
    mhs: dict[str, MhsRule]
    min_class_share: float


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping")
    return content


def load_data_config(path: str | Path | None = None) -> DataConfig:
    raw = read_yaml(resolve(path) if path else CONFIGS / "data.yaml")
    jigsaw = raw["jigsaw"]
    mhs = raw["mhs"]
    text = raw["text"]
    split = raw["split"]
    outputs = raw["outputs"]
    return DataConfig(
        seed=int(raw["seed"]),
        jigsaw=JigsawConfig(
            competition=jigsaw["competition"],
            raw_dir=resolve(jigsaw["raw_dir"]),
            id_column=jigsaw["id_column"],
            text_column=jigsaw["text_column"],
            label_columns=list(jigsaw["label_columns"]),
        ),
        mhs=MhsConfig(
            dataset_id=mhs["dataset_id"],
            config_name=mhs["config_name"],
            cache_dir=resolve(mhs["cache_dir"]),
            id_column=mhs["id_column"],
            text_column=mhs["text_column"],
            score_column=mhs["score_column"],
            rating_columns=list(mhs["rating_columns"]),
            target_columns=list(mhs["target_columns"]),
        ),
        text=TextConfig(min_chars=int(text["min_chars"]), max_repeat=int(text["max_repeat"])),
        split=SplitConfig(
            folds=int(split["folds"]),
            val_fold=int(split["val_fold"]),
            test_fold=int(split["test_fold"]),
        ),
        outputs=OutputPaths(
            jigsaw=resolve(outputs["jigsaw"]),
            jigsaw_unlabeled=resolve(outputs["jigsaw_unlabeled"]),
            mhs=resolve(outputs["mhs"]),
            mhs_annotators=resolve(outputs["mhs_annotators"]),
            unified=resolve(outputs["unified"]),
            reports=resolve(outputs["reports"]),
        ),
    )


def parse_mhs_rule(rule: dict[str, Any]) -> MhsRule:
    if "min" in rule:
        return MhsRule(column=rule["column"], minimum=float(rule["min"]), inclusive=True)
    return MhsRule(column=rule["column"], minimum=float(rule["min_exclusive"]), inclusive=False)


def load_label_map(path: str | Path | None = None) -> LabelMapConfig:
    raw = read_yaml(resolve(path) if path else CONFIGS / "label_map.yaml")
    priority = list(raw["priority"])
    if set(priority) != set(CLASS_NAMES) or len(priority) != len(CLASS_NAMES):
        raise ValueError(f"priority must be a permutation of {CLASS_NAMES}, got {priority}")
    return LabelMapConfig(
        priority=priority,
        jigsaw={name: list(columns) for name, columns in raw["jigsaw"].items()},
        mhs={name: parse_mhs_rule(rule) for name, rule in raw["mhs"].items()},
        min_class_share=float(raw["min_class_share"]),
    )