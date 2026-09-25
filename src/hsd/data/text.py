import hashlib
import html
import re
import unicodedata
from functools import lru_cache

import pandas as pd

from hsd.common.config import TextConfig

URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
ZERO_WIDTH_PATTERN = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALNUM_PATTERN = re.compile(r"[\W_]+")

URL_TOKEN = "<url>"
MENTION_TOKEN = "@user"


@lru_cache(maxsize=8)
def repeat_pattern(max_repeat: int) -> re.Pattern[str]:
    if max_repeat < 1:
        raise ValueError("max_repeat must be at least 1")
    return re.compile(r"(.)\1{" + str(max_repeat) + r",}", re.DOTALL)


def collapse_repeats(text: str, max_repeat: int) -> str:
    return repeat_pattern(max_repeat).sub(lambda match: match.group(1) * max_repeat, text)


def normalize_text(text: str, max_repeat: int = 3) -> str:
    cleaned = html.unescape(text)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = ZERO_WIDTH_PATTERN.sub("", cleaned)
    cleaned = CONTROL_PATTERN.sub("", cleaned)
    cleaned = URL_PATTERN.sub(URL_TOKEN, cleaned)
    cleaned = MENTION_PATTERN.sub(MENTION_TOKEN, cleaned)
    cleaned = collapse_repeats(cleaned, max_repeat)
    return WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def text_hash(text: str) -> str:
    key = NON_ALNUM_PATTERN.sub(" ", text.lower()).strip()
    if not key:
        key = text.lower().strip()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def make_uid(source: str, source_id: str | int) -> str:
    return hashlib.sha1(f"{source}:{source_id}".encode()).hexdigest()


def uid_series(source: str, source_ids: pd.Series) -> pd.Series:
    return source_ids.map(lambda value: make_uid(source, value))


def clean_series(texts: pd.Series, max_repeat: int) -> pd.Series:
    return texts.map(lambda value: normalize_text(value, max_repeat))


def hash_series(texts: pd.Series) -> pd.Series:
    return texts.map(text_hash)


def add_text_columns(frame: pd.DataFrame, config: TextConfig) -> pd.DataFrame:
    raw = frame["text_raw"].fillna("").astype(str)
    cleaned = clean_series(raw, config.max_repeat)
    return frame.assign(
        text_raw=raw,
        text=cleaned,
        text_hash=hash_series(cleaned),
        is_short=cleaned.str.len() < config.min_chars,
    )