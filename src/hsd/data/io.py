from pathlib import Path

import pandas as pd


def is_cached(path: Path, force: bool) -> bool:
    return path.exists() and not force


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    frame.to_parquet(partial, engine="pyarrow", compression="zstd", index=False)
    partial.replace(path)