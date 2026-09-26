from typing import Final

CLASS_NAMES: Final[list[str]] = ["neither", "offensive", "hate", "threat"]
SEVERITY_NAMES: Final[list[str]] = ["low", "medium", "high", "critical"]
TARGET_NAMES: Final[list[str]] = [
    "race",
    "religion",
    "origin",
    "gender",
    "sexuality",
    "age",
    "disability",
    "politics",
]
IGNORE_INDEX: Final[int] = -100
