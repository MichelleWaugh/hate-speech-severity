import pandas as pd
import pytest

from hsd.common.labels import IGNORE_INDEX
from hsd.severity.rules import (
    NONE_SEVERITY_NAME,
    JigsawSeverityConfig,
    MhsThreshold,
    SeverityConfig,
    apply_severity,
    bin_jigsaw_score,
    classify_mhs_severity,
    harmful_mask,
    jigsaw_severity_score,
    load_severity_config,
    mhs_severity_score,
    severity_name_to_id,
    threshold_mask,
)

JIGSAW_WEIGHTS = {
    "toxic": 1,
    "obscene": 1,
    "insult": 1,
    "identity_hate": 2,
    "threat": 2,
    "severe_toxic": 3,
}
JIGSAW_BINS = {
    "low": (1, 2),
    "medium": (3, 4),
    "high": (5, 6),
    "critical": (7, 999),
}


@pytest.fixture
def jigsaw_config() -> JigsawSeverityConfig:
    return JigsawSeverityConfig(weights=JIGSAW_WEIGHTS, bins=JIGSAW_BINS)


@pytest.fixture
def severity_config() -> SeverityConfig:
    return SeverityConfig(
        jigsaw=JigsawSeverityConfig(weights=JIGSAW_WEIGHTS, bins=JIGSAW_BINS),
        mhs_critical=MhsThreshold(None, None, None, 1.0, 3.0),
        mhs_high=MhsThreshold(2.0, 2.5, None, None, None),
        mhs_medium=MhsThreshold(0.5, None, 3.0, None, None),
        min_level_share=0.03,
    )


def test_jigsaw_severity_score_sums_weighted_signals(jigsaw_config: JigsawSeverityConfig) -> None:
    frame = pd.DataFrame(
        {
            "j_toxic": [1, 0, 1],
            "j_obscene": [0, 0, 1],
            "j_insult": [0, 0, 1],
            "j_identity_hate": [0, 0, 0],
            "j_threat": [0, 0, 0],
            "j_severe_toxic": [0, 1, 0],
        }
    ).astype("Int8")
    score = jigsaw_severity_score(frame, jigsaw_config)
    assert score.tolist() == [1, 3, 3]


def test_jigsaw_severity_score_treats_missing_signals_as_zero(
    jigsaw_config: JigsawSeverityConfig,
) -> None:
    frame = pd.DataFrame(
        {
            "j_toxic": pd.array([1, None], dtype="Int8"),
            "j_obscene": pd.array([None, None], dtype="Int8"),
            "j_insult": pd.array([0, 0], dtype="Int8"),
            "j_identity_hate": pd.array([0, 0], dtype="Int8"),
            "j_threat": pd.array([0, 0], dtype="Int8"),
            "j_severe_toxic": pd.array([0, 0], dtype="Int8"),
        }
    )
    score = jigsaw_severity_score(frame, jigsaw_config)
    assert score.tolist() == [1, 0]


def test_jigsaw_severity_score_missing_columns_raises(
    jigsaw_config: JigsawSeverityConfig,
) -> None:
    with pytest.raises(KeyError):
        jigsaw_severity_score(pd.DataFrame({"j_toxic": [1]}), jigsaw_config)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "none"),
        (1, "low"),
        (2, "low"),
        (3, "medium"),
        (4, "medium"),
        (5, "high"),
        (6, "high"),
        (7, "critical"),
        (50, "critical"),
    ],
)
def test_bin_jigsaw_score_edges(
    jigsaw_config: JigsawSeverityConfig, score: int, expected: str
) -> None:
    result = bin_jigsaw_score(pd.Series([score]), jigsaw_config)
    assert result.iloc[0] == expected


def mhs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "m_hate_speech_score": [0.0, 0.6, 2.1, 0.0, -3.0],
            "m_insult": [0.0, 3.0, 0.0, 0.0, 0.0],
            "m_dehumanize": [0.0, 0.0, 0.0, 2.6, 0.0],
            "m_genocide": [0.0, 0.0, 0.0, 0.0, 0.0],
            "m_violence": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).astype("float32")


def test_classify_mhs_severity_cascade_order(severity_config: SeverityConfig) -> None:
    result = classify_mhs_severity(mhs_frame(), severity_config)
    assert result.tolist() == ["low", "medium", "high", "high", "low"]


def test_classify_mhs_severity_critical_overrides_lower_levels(
    severity_config: SeverityConfig,
) -> None:
    frame = mhs_frame()
    frame.loc[1, "m_genocide"] = 1.0
    result = classify_mhs_severity(frame, severity_config)
    assert result.iloc[1] == "critical"


def test_classify_mhs_severity_missing_columns_raises(severity_config: SeverityConfig) -> None:
    with pytest.raises(KeyError):
        classify_mhs_severity(pd.DataFrame({"m_violence": [1.0]}), severity_config)


def test_threshold_mask_treats_missing_values_as_false() -> None:
    frame = pd.DataFrame({"m_violence": pd.array([None, 3.0], dtype="float32")})
    mask = threshold_mask(frame, MhsThreshold(None, None, None, None, 3.0))
    assert mask.tolist() == [False, True]


def test_mhs_severity_score_sums_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "m_hate_speech_score": [1.0],
            "m_insult": [10.0],
            "m_dehumanize": [2.0],
            "m_genocide": [0.5],
            "m_violence": [0.25],
        }
    ).astype("float32")
    score = mhs_severity_score(frame)
    assert score.iloc[0] == pytest.approx(3.75)


def test_harmful_mask_excludes_only_neither() -> None:
    labels = pd.Series(["neither", "offensive", "hate", "threat"])
    assert harmful_mask(labels).tolist() == [False, True, True, True]


def test_severity_name_to_id_maps_known_names() -> None:
    names = pd.Series(["low", "medium", "high", "critical", NONE_SEVERITY_NAME])
    ids = severity_name_to_id(names)
    assert ids.tolist() == [0, 1, 2, 3, IGNORE_INDEX]


def test_severity_name_to_id_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        severity_name_to_id(pd.Series(["extreme"]))


def test_apply_severity_forces_none_for_non_harmful_rows() -> None:
    names = pd.Series(["high", "low"])
    scores = pd.Series([5.0, 1.0], dtype="float32")
    is_harmful = pd.Series([True, False])
    ids, final_scores, final_names = apply_severity(names, scores, is_harmful)
    assert ids.tolist() == [2, IGNORE_INDEX]
    assert final_scores.tolist() == [5.0, 0.0]
    assert final_names.tolist() == ["high", NONE_SEVERITY_NAME]


def test_shipped_severity_config_loads_and_matches_expected_shape() -> None:
    config = load_severity_config()
    assert set(config.jigsaw.bins) == {"low", "medium", "high", "critical"}
    assert config.min_level_share == pytest.approx(0.03)