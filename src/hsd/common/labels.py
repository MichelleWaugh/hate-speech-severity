from typing import Final

CLASS_NAMES: Final[list[str]] = ["neither", "offensive", "hate", "threat"]
SEVERITY_NAMES: Final[list[str]] = ["low", "medium", "high", "critical"]
IGNORE_INDEX: Final[int] = -100

NUM_CLASSES: Final[int] = len(CLASS_NAMES)
NUM_SEVERITY_LEVELS: Final[int] = len(SEVERITY_NAMES)

CLASS_TO_ID: Final[dict[str, int]] = {name: index for index, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS: Final[dict[int, str]] = {index: name for name, index in CLASS_TO_ID.items()}
SEVERITY_TO_ID: Final[dict[str, int]] = {name: index for index, name in enumerate(SEVERITY_NAMES)}
ID_TO_SEVERITY: Final[dict[int, str]] = {index: name for name, index in SEVERITY_TO_ID.items()}