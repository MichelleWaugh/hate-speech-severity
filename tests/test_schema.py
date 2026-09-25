import pandas as pd
import pytest

from hsd.data.build import (
    JIGSAW_SIGNALS,
    MHS_COUNT_SIGNAL,
    MHS_SIGNALS,
    TARGET_COLUMNS,
    UNIFIED_COLUMNS,
    combine_sources,
    enforce_schema,
    sample_rows,
    validate_schema,
)


def jigsaw_like_frame() -> pd.DataFrame:
    data = {
        "uid": ["j1", "j2"],
        "source": ["jigsaw", "jigsaw"],
        "origin": ["jigsaw_train", "jigsaw_train"],
        "source_id": ["1", "2"],
        "text_raw": ["hello", "world"],
        "text": ["hello", "world"],
        "text_hash": ["h1", "h2"],
        "is_short": [False, False],
        "label": pd.array([0, 1], dtype="int8"),
        "label_name": ["neither", "offensive"],
    }
    for column in JIGSAW_SIGNALS:
        data[column] = pd.array([0, 1], dtype="Int8")
    return pd.DataFrame(data)


def mhs_like_frame() -> pd.DataFrame:
    data = {
        "uid": ["m1", "m2"],
        "source": ["mhs", "mhs"],
        "origin": ["mhs", "mhs"],
        "source_id": ["10", "11"],
        "text_raw": ["ok", "not ok"],
        "text": ["ok", "not ok"],
        "text_hash": ["h3", "h4"],
        "is_short": [True, False],
        "label": pd.array([0, 2], dtype="int8"),
        "label_name": ["neither", "hate"],
    }
    for column in MHS_SIGNALS:
        data[column] = pd.array([0.0, 1.5], dtype="float32")
    data[MHS_COUNT_SIGNAL] = pd.array([3, 5], dtype="Int16")
    for column in TARGET_COLUMNS:
        data[column] = pd.array([False, True], dtype="boolean")
    return pd.DataFrame(data)


def finalize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        split="train",
        is_duplicate=False,
        dup_group_size=1,
        label_conflict=False,
    )


def test_combine_sources_keeps_every_row_and_fills_missing_signals() -> None:
    combined = combine_sources(jigsaw_like_frame(), mhs_like_frame())
    assert len(combined) == 4
    assert combined.loc[combined["source"] == "jigsaw", MHS_SIGNALS[0]].isna().all()
    assert combined.loc[combined["source"] == "mhs", JIGSAW_SIGNALS[0]].isna().all()


def test_enforce_schema_orders_and_types_columns() -> None:
    combined = finalize(combine_sources(jigsaw_like_frame(), mhs_like_frame()))
    result = enforce_schema(combined)
    assert list(result.columns) == UNIFIED_COLUMNS
    assert str(result["label"].dtype) == "int8"
    assert str(result["is_duplicate"].dtype) == "bool"
    assert str(result[JIGSAW_SIGNALS[0]].dtype) == "Int8"
    assert str(result[MHS_SIGNALS[0]].dtype) == "float32"
    assert str(result[MHS_COUNT_SIGNAL].dtype) == "Int16"
    assert str(result[TARGET_COLUMNS[0]].dtype) == "boolean"


def test_enforce_schema_preserves_row_count() -> None:
    combined = finalize(combine_sources(jigsaw_like_frame(), mhs_like_frame()))
    result = enforce_schema(combined)
    assert len(result) == len(combined)


def test_validate_schema_accepts_well_formed_frame() -> None:
    combined = finalize(combine_sources(jigsaw_like_frame(), mhs_like_frame()))
    validate_schema(enforce_schema(combined))


def test_validate_schema_rejects_duplicate_uid() -> None:
    combined = finalize(combine_sources(jigsaw_like_frame(), jigsaw_like_frame()))
    with pytest.raises(ValueError):
        validate_schema(enforce_schema(combined))


def test_validate_schema_rejects_missing_required_value() -> None:
    combined = finalize(combine_sources(jigsaw_like_frame(), mhs_like_frame()))
    result = enforce_schema(combined)
    result.loc[0, "label_name"] = pd.NA
    with pytest.raises(ValueError):
        validate_schema(result)


def test_validate_schema_rejects_wrong_column_order() -> None:
    combined = finalize(combine_sources(jigsaw_like_frame(), mhs_like_frame()))
    result = enforce_schema(combined)
    shuffled = result[list(reversed(result.columns))]
    with pytest.raises(ValueError):
        validate_schema(shuffled)


def test_sample_rows_respects_requested_count() -> None:
    frame = pd.DataFrame({"uid": [f"u{i}" for i in range(20)]})
    sampled = sample_rows(frame, 5, seed=42)
    assert len(sampled) == 5


def test_sample_rows_is_deterministic() -> None:
    frame = pd.DataFrame({"uid": [f"u{i}" for i in range(20)]})
    first = sample_rows(frame, 5, seed=42)["uid"].tolist()
    second = sample_rows(frame, 5, seed=42)["uid"].tolist()
    assert first == second


def test_sample_rows_returns_full_frame_when_max_rows_exceeds_length() -> None:
    frame = pd.DataFrame({"uid": ["a", "b", "c"]})
    sampled = sample_rows(frame, 10, seed=42)
    assert len(sampled) == 3


def test_unified_columns_end_with_target_columns() -> None:
    assert UNIFIED_COLUMNS[-len(TARGET_COLUMNS):] == TARGET_COLUMNS
    assert "label" in UNIFIED_COLUMNS
    assert "split" in UNIFIED_COLUMNS