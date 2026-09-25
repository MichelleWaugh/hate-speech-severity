import numpy as np
import pandas as pd
import pytest

from hsd.common.config import (
    JigsawConfig,
    LabelMapConfig,
    MhsConfig,
    MhsRule,
    TextConfig,
    load_data_config,
    load_label_map,
)
from hsd.data import harmonize_jigsaw, harmonize_mhs
from hsd.data.download import JigsawFiles
from hsd.data.labeling import assign_labels, class_distribution, underrepresented_classes

JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
PRIORITY = ["threat", "hate", "offensive", "neither"]
TEXT = TextConfig(min_chars=3, max_repeat=3)


@pytest.fixture
def label_map() -> LabelMapConfig:
    return LabelMapConfig(
        priority=PRIORITY,
        jigsaw={
            "threat": ["threat"],
            "hate": ["identity_hate"],
            "offensive": ["toxic", "severe_toxic", "obscene", "insult"],
        },
        mhs={
            "threat": MhsRule("m_violence", 2.5, True),
            "hate": MhsRule("m_hate_speech_score", 0.5, False),
            "offensive": MhsRule("m_insult", 2.5, True),
        },
        min_class_share=0.02,
    )


@pytest.fixture
def jigsaw_config(tmp_path) -> JigsawConfig:
    return JigsawConfig(
        competition="test",
        raw_dir=tmp_path,
        id_column="id",
        text_column="comment_text",
        label_columns=JIGSAW_LABELS,
    )


@pytest.fixture
def mhs_config(tmp_path) -> MhsConfig:
    return MhsConfig(
        dataset_id="test/test",
        config_name="default",
        cache_dir=tmp_path,
        id_column="comment_id",
        text_column="text",
        score_column="hate_speech_score",
        rating_columns=["insult", "violence"],
        target_columns=["target_race", "target_gender"],
    )


def write_jigsaw_files(directory) -> None:
    train = pd.DataFrame(
        {
            "id": ["a1", "a2", "a3"],
            "comment_text": ["Hello world", "NA", "line one\nline two"],
            "toxic": [0, 1, 1],
            "severe_toxic": [0, 0, 0],
            "obscene": [0, 1, 0],
            "threat": [0, 0, 0],
            "insult": [0, 1, 0],
            "identity_hate": [0, 0, 1],
        }
    )
    test = pd.DataFrame(
        {"id": ["t1", "t2", "t3"], "comment_text": ["fine text", "i will find you", "unscored"]}
    )
    labels = pd.DataFrame(
        {
            "id": ["t1", "t2", "t3"],
            "toxic": [0, 1, -1],
            "severe_toxic": [0, 0, -1],
            "obscene": [0, 0, -1],
            "threat": [0, 1, -1],
            "insult": [0, 0, -1],
            "identity_hate": [0, 0, -1],
        }
    )
    train.to_csv(directory / "train.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)
    labels.to_csv(directory / "test_labels.csv", index=False)


def jigsaw_files(directory) -> JigsawFiles:
    return JigsawFiles(
        train=directory / "train.csv",
        test=directory / "test.csv",
        test_labels=directory / "test_labels.csv",
    )


def test_assign_labels_applies_priority() -> None:
    masks = {
        "threat": pd.Series([True, False, False, False, False]),
        "hate": pd.Series([True, True, False, False, False]),
        "offensive": pd.Series([True, True, True, False, False]),
    }
    labels = assign_labels(masks, PRIORITY, pd.RangeIndex(5))
    assert labels.tolist() == [3, 2, 1, 0, 0]
    assert labels.dtype == "int8"


def test_assign_labels_rejects_fallback_rule() -> None:
    with pytest.raises(ValueError):
        assign_labels({"neither": pd.Series([True])}, PRIORITY, pd.RangeIndex(1))


def test_assign_labels_rejects_unknown_class() -> None:
    with pytest.raises(ValueError):
        assign_labels({"spam": pd.Series([True])}, PRIORITY, pd.RangeIndex(1))


def test_assign_labels_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        assign_labels({"hate": pd.Series([True, False])}, PRIORITY, pd.RangeIndex(3))


def test_label_jigsaw_follows_priority(label_map: LabelMapConfig) -> None:
    columns = harmonize_jigsaw.signal_columns(
        JigsawConfig("test", ".", "id", "comment_text", JIGSAW_LABELS)
    )
    rows = [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0],
    ]
    frame = pd.DataFrame(rows, columns=columns).astype("Int8")
    labeled = harmonize_jigsaw.label_jigsaw(frame, label_map)
    assert labeled["label_name"].tolist() == [
        "neither",
        "offensive",
        "offensive",
        "offensive",
        "hate",
        "threat",
        "threat",
    ]


def test_label_jigsaw_reports_missing_columns(label_map: LabelMapConfig) -> None:
    with pytest.raises(KeyError):
        harmonize_jigsaw.label_jigsaw(pd.DataFrame({"j_toxic": [1]}), label_map)


def test_build_frames_keeps_every_row(tmp_path, jigsaw_config: JigsawConfig) -> None:
    write_jigsaw_files(tmp_path)
    labeled, unlabeled = harmonize_jigsaw.build_frames(
        jigsaw_files(tmp_path), jigsaw_config, TEXT
    )
    assert len(labeled) == 5
    assert len(unlabeled) == 1
    assert unlabeled["text_raw"].tolist() == ["unscored"]
    assert "NA" in labeled["text_raw"].tolist()
    assert "line one\nline two" in labeled["text_raw"].tolist()
    assert "line one line two" in labeled["text"].tolist()
    assert labeled["origin"].tolist() == ["jigsaw_train"] * 3 + ["jigsaw_test"] * 2
    assert str(labeled["j_threat"].dtype) == "Int8"
    all_uids = pd.concat([labeled["uid"], unlabeled["uid"]])
    assert all_uids.is_unique


def test_build_frames_labels_end_to_end(
    tmp_path, jigsaw_config: JigsawConfig, label_map: LabelMapConfig
) -> None:
    write_jigsaw_files(tmp_path)
    labeled, _ = harmonize_jigsaw.build_frames(jigsaw_files(tmp_path), jigsaw_config, TEXT)
    result = harmonize_jigsaw.label_jigsaw(labeled, label_map)
    assert result["label_name"].tolist() == ["neither", "offensive", "hate", "neither", "threat"]


def test_build_frames_rejects_ids_missing_from_test_labels(
    tmp_path, jigsaw_config: JigsawConfig
) -> None:
    write_jigsaw_files(tmp_path)
    labels = pd.read_csv(tmp_path / "test_labels.csv").iloc[:2]
    labels.to_csv(tmp_path / "test_labels.csv", index=False)
    with pytest.raises(ValueError):
        harmonize_jigsaw.build_frames(jigsaw_files(tmp_path), jigsaw_config, TEXT)


def test_load_jigsaw_reuses_cache(tmp_path, jigsaw_config: JigsawConfig) -> None:
    write_jigsaw_files(tmp_path)
    labeled_path = tmp_path / "out" / "labeled.parquet"
    unlabeled_path = tmp_path / "out" / "unlabeled.parquet"
    first, _ = harmonize_jigsaw.load_jigsaw(
        jigsaw_config, TEXT, labeled_path, unlabeled_path, False
    )
    for name in ("train.csv", "test.csv", "test_labels.csv"):
        (tmp_path / name).unlink()
    second, _ = harmonize_jigsaw.load_jigsaw(
        jigsaw_config, TEXT, labeled_path, unlabeled_path, False
    )
    assert first.shape == second.shape
    assert first["uid"].tolist() == second["uid"].tolist()
    with pytest.raises(FileNotFoundError):
        harmonize_jigsaw.load_jigsaw(jigsaw_config, TEXT, labeled_path, unlabeled_path, True)


def annotator_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "comment_id": [1, 1, 1, 2],
            "text": ["First comment here", "First comment here", "First comment here", "ok"],
            "hate_speech_score": [1.0, 1.0, 1.0, -2.0],
            "insult": [4, 2, 3, 0],
            "violence": [0, 1, 2, 4],
            "target_race": [True, False, False, False],
            "target_gender": [False, False, False, False],
        }
    )


def test_aggregate_annotators_returns_one_row_per_comment(mhs_config: MhsConfig) -> None:
    result = harmonize_mhs.aggregate_annotators(annotator_rows(), mhs_config)
    assert result["source_id"].tolist() == ["1", "2"]
    assert result["m_insult"].tolist() == [3.0, 0.0]
    assert result["m_violence"].tolist() == [1.0, 4.0]
    assert result["m_hate_speech_score"].tolist() == [1.0, -2.0]
    assert result["m_n_annotators"].tolist() == [3, 1]
    assert result["t_race"].tolist() == [True, False]
    assert result["t_gender"].tolist() == [False, False]
    assert result["text_raw"].tolist() == ["First comment here", "ok"]
    assert result["m_insult"].dtype == "float32"
    assert str(result["m_n_annotators"].dtype) == "Int16"
    assert str(result["t_race"].dtype) == "boolean"


def test_mhs_signal_frame_keeps_short_texts(mhs_config: MhsConfig) -> None:
    aggregated = harmonize_mhs.aggregate_annotators(annotator_rows(), mhs_config)
    frame = harmonize_mhs.to_signal_frame(aggregated, mhs_config, TEXT)
    assert len(frame) == 2
    assert frame["is_short"].tolist() == [False, True]
    assert frame["uid"].is_unique
    assert (frame["source"] == "mhs").all()
    assert list(frame.columns) == harmonize_mhs.frame_columns(mhs_config)


def test_label_mhs_boundaries(label_map: LabelMapConfig) -> None:
    frame = pd.DataFrame(
        {
            "m_violence": [2.5, 2.49, 0, 0, 0, np.nan, 0, 3.0],
            "m_hate_speech_score": [0, 0, 0.5, 0.51, 0, 0.0, 1.0, 1.0],
            "m_insult": [0, 0, 0, 0, 2.5, np.nan, 3.0, 0],
        }
    ).astype("float32")
    labeled = harmonize_mhs.label_mhs(frame, label_map)
    assert labeled["label"].tolist() == [3, 0, 0, 2, 1, 0, 2, 3]
    assert labeled["label_name"].tolist() == [
        "threat",
        "neither",
        "neither",
        "hate",
        "offensive",
        "neither",
        "hate",
        "threat",
    ]


def test_label_mhs_reports_missing_columns(label_map: LabelMapConfig) -> None:
    with pytest.raises(KeyError):
        harmonize_mhs.label_mhs(pd.DataFrame({"m_violence": [1.0]}), label_map)


def test_shipped_label_rules_reference_existing_signals() -> None:
    data = load_data_config()
    labels = load_label_map()
    assert {rule.column for rule in labels.mhs.values()} <= set(
        harmonize_mhs.mean_signal_columns(data.mhs)
    )
    jigsaw_labels = {label for group in labels.jigsaw.values() for label in group}
    assert jigsaw_labels <= set(data.jigsaw.label_columns)


def test_class_distribution_and_shares() -> None:
    frame = pd.DataFrame(
        {
            "source": ["jigsaw"] * 4 + ["mhs"] * 2,
            "label_name": ["neither", "neither", "neither", "hate", "offensive", "neither"],
        }
    )
    table = class_distribution(frame)
    assert list(table.columns) == ["neither", "offensive", "hate", "threat"]
    assert table.loc["jigsaw", "neither"] == 3
    assert table.loc["mhs", "offensive"] == 1
    assert table["threat"].sum() == 0
    assert underrepresented_classes(frame, 0.05) == ["threat"]
    assert underrepresented_classes(frame, 0.2) == ["offensive", "hate", "threat"]