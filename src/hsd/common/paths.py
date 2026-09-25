from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
DATA_CACHE = DATA / "cache"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def resolve(relative: str | Path) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else ROOT / path