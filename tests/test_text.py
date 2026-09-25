import pandas as pd
import pytest

from hsd.common.config import TextConfig
from hsd.data.text import add_text_columns, make_uid, normalize_text, text_hash, uid_series


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("see https://example.com/a?b=1 now", "see <url> now"),
        ("visit www.example.com today", "visit <url> today"),
        ("@john_doe hi there", "@user hi there"),
        ("mail me at name@example.com", "mail me at name@example.com"),
        ("sooooo good", "sooo good"),
        ("what!!!!!!", "what!!!"),
        ("aaa stays", "aaa stays"),
        ("  spaced \n\t out  ", "spaced out"),
        ("he\u200bllo", "hello"),
        ("Tom &amp; Jerry", "Tom & Jerry"),
        ("\uff26\uff35\uff2c\uff2c", "FULL"),
        ("You ARE Wrong", "You ARE Wrong"),
        ("&#8220;quoted&#8221;", "\u201cquoted\u201d"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_normalize_text_respects_max_repeat() -> None:
    assert normalize_text("heyyyyy", max_repeat=2) == "heyy"


def test_normalize_text_rejects_invalid_max_repeat() -> None:
    with pytest.raises(ValueError):
        normalize_text("text", max_repeat=0)


@pytest.mark.parametrize(
    "raw",
    [
        "see https://example.com now @someone",
        "sooooo   good!!!!!!",
        "Tom &amp; Jerry \u200b",
        "plain text",
    ],
)
def test_normalize_text_is_idempotent(raw: str) -> None:
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_text_hash_ignores_case_and_punctuation() -> None:
    assert text_hash("Hello,  World!") == text_hash("hello world")


def test_text_hash_separates_different_texts() -> None:
    assert text_hash("hello there") != text_hash("hello world")


def test_text_hash_separates_punctuation_only_texts() -> None:
    assert text_hash("!!!") != text_hash("???")


def test_make_uid_is_deterministic_and_source_specific() -> None:
    assert make_uid("jigsaw", "abc") == make_uid("jigsaw", "abc")
    assert make_uid("jigsaw", "abc") != make_uid("mhs", "abc")
    assert make_uid("mhs", 7) == make_uid("mhs", "7")


def test_uid_series_keeps_index() -> None:
    ids = pd.Series(["a", "b"], index=[10, 20])
    result = uid_series("mhs", ids)
    assert result.index.tolist() == [10, 20]
    assert result.iloc[0] == make_uid("mhs", "a")


def test_add_text_columns_keeps_every_row_and_original_text() -> None:
    frame = pd.DataFrame({"text_raw": ["Hello   world", "", None, "ok", "abc"]})
    result = add_text_columns(frame, TextConfig(min_chars=3, max_repeat=3))
    assert len(result) == len(frame)
    assert result["text_raw"].tolist() == ["Hello   world", "", "", "ok", "abc"]
    assert result["text"].tolist() == ["Hello world", "", "", "ok", "abc"]
    assert result["is_short"].tolist() == [False, True, True, True, False]


def test_add_text_columns_hashes_equivalent_texts_identically() -> None:
    frame = pd.DataFrame({"text_raw": ["Hello, World!", "hello world", "different"]})
    result = add_text_columns(frame, TextConfig(min_chars=3, max_repeat=3))
    assert result["text_hash"].iloc[0] == result["text_hash"].iloc[1]
    assert result["text_hash"].iloc[0] != result["text_hash"].iloc[2]